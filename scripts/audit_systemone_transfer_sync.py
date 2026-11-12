#!/usr/bin/env python3
"""Read-only selected transfer/synchronization audit; run under nice -n 10."""

import argparse
import ast
from collections import defaultdict
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import json
from pathlib import Path


HOST_NAMES = {
    "aten::item", "aten::_local_scalar_dense", "aten::_to_copy",
    "acl_memcpy_host_to_device", "acl_memcpy_device_to_host", "Event::synchronize",
    "AscendCL@aclrtMemcpyAsync", "AscendCL@aclrtMemcpyAsyncWithCondition",
    "AscendCL@aclrtStreamWaitEvent", "AscendCL@aclrtSynchronizeStreamWithTimeout",
    "AscendCL@aclrtSynchronizeEvent", "AscendCL@aclrtSynchronizeDeviceWithTimeout",
}
DEVICE_NAMES = {"EVENT WAIT", "MEMCPY_ASYNC"}
SCOPES = {"clef::backbone", "clef::joint_pooler", "clef::joint_head", "clef::schema_indices_h2d"}
CALLS = {"cpu", "item", "tolist", "synchronize", "wait_stream", "async_tensor_h2d"}


def ns(value):
    return int(Decimal(str(value)) * 1000)


def union(intervals):
    end = None
    total = 0
    for start, finish in sorted(intervals):
        if end is None or start > end:
            total += finish - start
            end = finish
        elif finish > end:
            total += finish - end
            end = finish
    return total


def digest(data):
    return hashlib.sha256(data).hexdigest()


def source_audit():
    roots = {"vllm": Path("/workspace/yangchen/work/src/vllm"),
             "ascend": Path("/workspace/yangchen/work/src/vllm-ascend")}
    targets = [
        ("vllm", "vllm/model_executor/models/clef_joint_schema.py", {"JointSchemaHead", "JointSchemaPooler"}),
        ("vllm", "vllm/entrypoints/openai/decisions/strategies.py", {"JointSchemaStrategy"}),
        ("vllm", "vllm/entrypoints/generate/structured_decisions/serving.py", {"ServingStructuredDecisions"}),
        ("vllm", "vllm/v1/worker/gpu/pool/pooling_runner.py", {"PoolingRunner"}),
        ("vllm", "vllm/v1/worker/gpu/async_utils.py", {"AsyncPoolingOutput"}),
        ("vllm", "vllm/v1/pool/metadata.py", {"PoolingMetadata", "PoolingCursor"}),
        ("ascend", "vllm_ascend/worker/v2/model_runner.py", {"NPUModelRunner"}),
    ]
    reports = []
    for root, relative, classes in targets:
        path = roots[root] / relative
        data = path.read_bytes()
        text = data.decode()
        lines = text.splitlines()
        findings = []
        for cls in ast.parse(text).body:
            if not isinstance(cls, ast.ClassDef) or cls.name not in classes:
                continue
            for function in cls.body:
                if not isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                for node in ast.walk(function):
                    if isinstance(node, ast.Call):
                        name = node.func.attr if isinstance(node.func, ast.Attribute) else (
                            node.func.id if isinstance(node.func, ast.Name) else "")
                        cpu_copy = name == "to" and node.args and isinstance(node.args[0], ast.Constant) and node.args[0].value == "cpu"
                        dtype_cast = name == "float" and cls.name in {"JointSchemaHead", "JointSchemaPooler"}
                        if name in CALLS or cpu_copy or dtype_cast:
                            findings.append({"class": cls.name, "function": function.name,
                                "kind": "call", "name": name, "line": node.lineno,
                                "expression": ast.get_source_segment(text, node),
                                "context": [{"line": i + 1, "text": lines[i]} for i in
                                            range(max(function.lineno - 1, node.lineno - 4),
                                                  min(function.end_lineno, node.end_lineno + 2))]})
                    if isinstance(node, ast.For) and cls.name in {"JointSchemaHead", "JointSchemaPooler", "JointSchemaStrategy", "ServingStructuredDecisions"}:
                        if (cls.name == "JointSchemaStrategy" and function.name not in {"encode", "read"}):
                            continue
                        findings.append({"class": cls.name, "function": function.name, "kind": "python_loop",
                                         "line": node.lineno, "header": lines[node.lineno - 1].strip(),
                                         "iterator": ast.get_source_segment(text, node.iter)})
        reports.append({"path": str(path), "sha256_at_audit": digest(data), "findings": findings})
    return reports


def trace_audit(path, window_name):
    data = path.read_bytes()
    document = json.loads(data)
    events = document.get("traceEvents", []) if isinstance(document, dict) else document
    scopes = []
    for index, event in enumerate(events):
        if event.get("ph") == "X" and event.get("name") in SCOPES:
            start = ns(event["ts"])
            scopes.append((event["name"], event.get("pid"), event.get("tid"), start,
                           start + ns(event.get("dur", 0)), index))
    grouped = {}
    host_intervals = defaultdict(list)
    for index, event in enumerate(events):
        name = event.get("name")
        if event.get("ph") != "X" or name not in HOST_NAMES | DEVICE_NAMES:
            continue
        start = ns(event["ts"])
        end = start + ns(event.get("dur", 0))
        domain = "device_task" if name in DEVICE_NAMES else "host_cpu_or_runtime_api"
        key = (domain, name)
        entry = grouped.setdefault(key, {"domain": domain, "name": name, "count": 0,
            "cumulative_us": 0.0, "evidence": [], "within_host_scopes": {}, "intervals": []})
        entry["count"] += 1
        entry["cumulative_us"] += (end - start) / 1000
        entry["intervals"].append((start, end))
        enclosing = [scope[0] for scope in scopes if scope[1] == event.get("pid")
                     and scope[2] == event.get("tid") and scope[3] <= start and end <= scope[4]]
        for scope in enclosing:
            scoped = entry["within_host_scopes"].setdefault(scope, {"count": 0, "cumulative_us": 0.0})
            scoped["count"] += 1
            scoped["cumulative_us"] += (end - start) / 1000
        if len(entry["evidence"]) < 3:
            entry["evidence"].append({"event_index": index, "pid": event.get("pid"), "tid": event.get("tid"),
                "start_ns": start, "end_ns": end, "arguments": event.get("args", {})})
        # item encloses _local_scalar_dense. The leaf is retained for a host
        # interval union; all other selected API/copy nesting is unioned too.
        if domain != "device_task" and name != "aten::item":
            host_intervals[(event.get("pid"), event.get("tid"))].append((start, end))
    for entry in grouped.values():
        entry["mean_us"] = entry["cumulative_us"] / entry["count"]
        entry["interval_union_us_same_clock"] = union(entry.pop("intervals")) / 1000
    return {"window": window_name, "trace": str(path), "trace_sha256": digest(data),
        "scope": "Entire short profiler capture, including stop tail; counts are not independently additive",
        "selected_operations": list(grouped.values()),
        "selected_host_interval_union_us_by_pid_tid": [
            {"pid": pid, "tid": tid, "union_us": union(intervals) / 1000}
            for (pid, tid), intervals in host_intervals.items()],
        "method": "Exact selected names; per-operation counts/inclusive CPU time; same-thread interval union removes nesting. Cross-thread times and device waits remain separate; no CPU-compute or critical-path total is inferred."}


def overlap_evidence(run):
    path = run / "tp1-4096-observed/analysis-long3868-agent/summary.json"
    document = json.loads(path.read_text())
    correlation = document["ranks"][0]["timeline_metadata"][0]["module_flow_correlation"]
    head = next(row for row in correlation["modules"] if row["module"] == "clef::joint_head")
    wait = next(row for row in head["operations"] if row["name"] == "EVENT WAIT")["evidence"][0]
    start, end = wait["start_ns"], wait["end_ns"]
    instances = [row for row in correlation["module_instances"] if row["module"] == "clef::backbone"
                 and row.get("device_start_ns", end + 1) <= start < row.get("device_end_ns", start - 1)]
    backbone = instances[0]
    overlap = max(0, min(end, backbone["device_end_ns"]) - max(start, backbone["device_start_ns"]))
    return {"analysis_source": str(path), "analysis_source_sha256": digest(path.read_bytes()),
        "head_wait": wait, "backbone": {key: backbone.get(key) for key in (
            "scope_event_index", "cpu_start_ns", "cpu_end_ns", "device_start_ns", "device_end_ns", "request_ids", "observer_evidence")},
        "overlap_ns": overlap, "overlap_ms": overlap / 1e6,
        "interpretation": "The head-labelled device wait overlaps the still-running backbone. Wait duration cannot be added to backbone compute or attributed as independent head arithmetic cost."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", default="-")
    args = parser.parse_args()
    run = args.run_dir.resolve()
    traces = []
    for configuration in ("tp1-4096-observed", "tp2-4096-observed"):
        for window in ("window-fixed307", "window-long3868", "window-controlled-length-4032", "window-fields8", "window-fields-8"):
            manifest = run / configuration / window / "manifest.json"
            if not manifest.is_file():
                continue
            document = json.loads(manifest.read_text())
            if not document.get("passed"):
                continue
            for directory in document["new_rank_directories"]:
                path = Path(directory) / "ASCEND_PROFILER_OUTPUT/trace_view.json"
                if path.is_file():
                    audited = trace_audit(path, f"{configuration}/{window}")
                    audited["window_manifest"] = {"path": str(manifest), "sha256": digest(manifest.read_bytes()),
                                                  "passed": document.get("passed")}
                    requests = manifest.parent / "profile-requests.json"
                    if requests.is_file():
                        saved = json.loads(requests.read_text())
                        audited["requests"] = {"path": str(requests), "sha256": digest(requests.read_bytes()),
                            "successful_requests": saved["summary"].get("successful_requests"),
                            "actual_input_tokens": saved["summary"].get("actual_input_tokens")}
                    traces.append(audited)
    report = {"created_at": datetime.now(timezone.utc).isoformat(), "run_dir": str(run),
        "source_findings": source_audit(), "trace_findings": traces,
        "head_wait_backbone_overlap": overlap_evidence(run),
        "interpretation": [
            "Clef JointSchemaHead/JointSchemaPooler contain no explicit cpu/item/tolist calls. The pooler loops over prompts; head loops over fixed module layers.",
            "Schema encoding and response construction contain CPU Python field/option loops. Head computation handles fields/options as tensors.",
            "JointSchemaStrategy.read calls result.outputs.data.tolist after AsyncPoolingOutput has copied outputs to CPU and synchronized its copy event; this tolist is CPU postprocessing.",
            "PoolingRunner constructs prompt lengths from NumPy and token IDs on CPU. int(prompt_lens.max()) and CPU tensor indexing can generate item/local_scalar_dense without NPU scalar synchronization.",
            "NPUModelRunner cpu/item and speculative count-event synchronization sites are guarded by adaptive/speculative/LM-head sampling conditions. Source presence alone does not establish execution on the baseline pooling path.",
            "_to_copy includes dtype conversions and device copies; only explicitly directional memcpy names establish H2D/D2H direction. Runtime event synchronization includes blocked waiting, not CPU arithmetic.",
            "Trace item/local_scalar_dense arguments do not identify source tensor device. CPU metadata source paths establish CPU scalar conversions, while individual backend scalar provenance is left unspecified; names alone do not establish NPU synchronization.",
            "Host/API nesting, item/local_scalar nesting, and head device waits overlapping backbone must not be summed as independent latency contributions."]}
    payload = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if args.output == "-":
        print(payload, end="")
    else:
        Path(args.output).write_text(payload)


if __name__ == "__main__":
    main()
