#!/usr/bin/env python3
"""Summarize profiling startup memory and verified environment evidence.

Run with the target venv Python. --supplement reads current cgroup v1/v2 and
package metadata when the archived before snapshot lacks those fields. It
performs no NPU operation, allocation, synchronization or service mutation.
"""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import re
import sys


PACKAGES = {
    "torch": "2.10.0+cpu", "torch-npu": "2.10.0.post7",
    "torchvision": "0.25.0+cpu", "torchaudio": "2.10.0+cpu",
    "transformers": "5.19.0", "triton": "3.6.0",
    "triton-ascend": "3.6.0.post2", "flash-linear-attention-npu-a3": "26.9.1",
    "flash-attn-npu": "0.5.0",
    "vllm": None, "vllm-ascend": None,
}

MEMORY_LINE = re.compile(
    r"Free memory on device \((?P<free>[\d.]+)/(?P<total>[\d.]+) GiB\).*?"
    r"Desired GPU memory utilization is \((?P<utilization>[\d.]+), (?P<budget>[\d.]+) GiB\).*?"
    r"Actual usage: (?P<weights>[\d.]+) GiB for weights, (?P<activation>[\d.]+) GiB for peak "
    r"activation, (?P<non_torch>[\d.]+) GiB for non-torch memory, (?P<graph>[\d.]+) GiB for NPU graph memory.*?"
    r"Current KV cache memory: (?P<cache>[\d.]+) GiB\. After warmup: torch reserved memory "
    r"(?P<reserved>[\d.]+) GiB, torch allocated memory (?P<allocated>[\d.]+) GiB")


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read_evidence(path):
    data = path.read_bytes()
    return {"path": str(path), "bytes": len(data), "sha256": sha(data),
            "content": data.decode(errors="replace")}


def snapshot_environment(run):
    before = run / "before"
    manifest = json.loads((before / "manifest.json").read_text())
    profiler = json.loads((before / "torch-profiler.txt").read_text())
    command_status = {key: {"returncode": value.get("returncode"),
                           "path": str(before / value["path"]), "sha256": value.get("sha256")}
                      for key, value in manifest["commands"].items()}
    before_files = manifest.get("files", {})
    freeze_lines = (before / "pip-freeze.txt").read_text().splitlines()
    normalize = lambda value: value.lower().replace("_", "-")
    wanted = tuple(normalize(name) for name in PACKAGES)
    versions = [line for line in freeze_lines
                if any(normalize(line).startswith(name + separator)
                       for name in wanted for separator in ("==", " @ "))
                or line.startswith("-e ") and "vllm" in line]
    return {
        "before_created_at": manifest.get("created_at"),
        "manifest": {"path": str(before / "manifest.json"),
                     "sha256": sha((before / "manifest.json").read_bytes())},
        "project_head": (before / "project-head.txt").read_text().strip(),
        "original_project_head": (before / "original-project-head.txt").read_text().strip(),
        "vllm_head": (before / "vllm-head.txt").read_text().strip(),
        "vllm_ascend_head": (before / "vllm_ascend-head.txt").read_text().strip(),
        "command_status": command_status,
        "pip_check_passed": manifest["commands"]["pip-check"].get("returncode") == 0
                            and (before / "pip-check.txt").read_text().strip() == "No broken requirements found.",
        "pip_check_output": (before / "pip-check.txt").read_text().strip(),
        "selected_pip_freeze_lines": versions,
        "torch_profiler_actual": profiler,
        "performance_environment": {key: manifest["performance_environment"].get(key) for key in (
            "OMP_NUM_THREADS", "TASK_QUEUE_ENABLE", "ASCEND_HOME_PATH", "ATB_HOME_PATH",
            "VIRTUAL_ENV", "MODEL_DIR", "ASCEND_RT_VISIBLE_DEVICES", "ASCEND_VISIBLE_DEVICES")},
        "before_cgroup_v2": {path: before_files.get(path) for path in (
            "/sys/fs/cgroup/cpu.max", "/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/cpuset.cpus.effective")},
        "operating_system": before_files.get("/etc/os-release"),
        "driver": before_files.get("/usr/local/Ascend/driver/version.info"),
        "model_manifest": {path: {key: value.get(key) for key in ("sha256", "bytes")}
                           for path, value in before_files.items() if path.endswith("/model_manifest.json")},
        "application_patch_hashes": {path: {key: value.get(key) for key in ("sha256", "bytes")}
                                     for path, value in before_files.items() if "/scripts/patches/" in path},
        "npu_mapping": read_evidence(before / "npu-smi-mapping.txt"),
        "npu_smi_before": read_evidence(before / "npu-smi.txt"),
        "msprof_version_status": "--version succeeded" if manifest["commands"]["msprof-version"].get("returncode") == 0
                                 else "--version unsupported/failed; consult archived help and CANN package metadata",
    }


def current_supplement():
    raw = {}
    for name in (
            "/sys/fs/cgroup/cpu.max", "/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/cpuset.cpus.effective",
            "/sys/fs/cgroup/cpu,cpuacct/cpu.cfs_quota_us", "/sys/fs/cgroup/cpu,cpuacct/cpu.cfs_period_us",
            "/sys/fs/cgroup/memory/memory.limit_in_bytes", "/sys/fs/cgroup/cpuset/cpuset.cpus",
            "/proc/self/cgroup"):
        path = Path(name)
        if path.is_file():
            raw[name] = read_evidence(path)
    cgroup = {key: value["content"].strip() for key, value in raw.items()}
    quota = cgroup.get("/sys/fs/cgroup/cpu,cpuacct/cpu.cfs_quota_us")
    period = cgroup.get("/sys/fs/cgroup/cpu,cpuacct/cpu.cfs_period_us")
    limit = cgroup.get("/sys/fs/cgroup/memory/memory.limit_in_bytes")
    metadata = {}
    for name, expected in PACKAGES.items():
        try:
            distribution = importlib.metadata.distribution(name)
            version = distribution.version
            metadata[name] = {"version": version, "expected": expected,
                              "matches_expected": version == expected if expected else None,
                              "distribution_path": str(distribution.locate_file(""))}
        except importlib.metadata.PackageNotFoundError as error:
            metadata[name] = {"error": str(error)}
    cann = []
    for path in sorted(Path("/usr/local/Ascend/cann/share/info").glob("*/version.info")):
        cann.append(read_evidence(path))
    atb = []
    for base in (Path("/usr/local/Ascend/nnal/atb/latest"), Path("/usr/local/Ascend/nnal/atb")):
        for path in sorted(base.glob("**/version.info"))[:10]:
            if str(path.resolve()) not in {item["resolved_path"] for item in atb}:
                item = read_evidence(path)
                item["resolved_path"] = str(path.resolve())
                atb.append(item)
    return {"collected_at": datetime.now(timezone.utc).isoformat(),
            "scope": "Supplemental read-only metadata gathered after before snapshot; no NPU operation executed",
            "python": {"version": sys.version, "executable": sys.executable},
            "packages": metadata, "cgroup_files": raw,
            "cpu_quota_equivalent_cores": int(quota) / int(period) if quota and period and int(quota) > 0 else None,
            "memory_limit_bytes": int(limit) if limit else None,
            "memory_limit_GiB": int(limit) / 2**30 if limit else None,
            "cpuset_visible_cpus": cgroup.get("/sys/fs/cgroup/cpuset/cpuset.cpus"),
            "cann_package_versions": cann, "atb_package_versions": atb}


def service_memory(path):
    data = path.read_bytes()
    rows = []
    for number, line in enumerate(data.decode(errors="replace").splitlines(), 1):
        match = MEMORY_LINE.search(line)
        if not match:
            continue
        values = {key: float(value) for key, value in match.groupdict().items()}
        pid = re.search(r"(?:Worker(?:_TP(?P<rank>\d+))? pid=)(?P<pid>\d+)", line)
        rows.append({"rank": int(pid["rank"]) if pid and pid["rank"] else 0 if pid else None,
                     "pid": int(pid["pid"]) if pid else None,
                     "free_on_startup_GiB": values["free"], "device_total_GiB": values["total"],
                     "memory_utilization_fraction": values["utilization"],
                     "requested_budget_GiB": values["budget"], "weights_GiB": values["weights"],
                     "dummy_profile_peak_activation_GiB": values["activation"],
                     "non_torch_GiB": values["non_torch"], "npu_graph_GiB": values["graph"],
                     "kv_mamba_cache_pool_GiB": values["cache"],
                     "warmup_torch_reserved_GiB": values["reserved"],
                     "warmup_torch_allocated_GiB": values["allocated"],
                     "evidence": {"path": str(path), "line": number, "text": line}})
    return {"configuration": path.parent.name, "service_log": str(path),
            "service_log_sha256": sha(data), "service_log_bytes": len(data), "ranks": rows,
            "complete_memory_summary_found": bool(rows)}


def actual_cache_storage(run):
    records = []
    errors = []
    for path in sorted(run.glob("*/cpu-timings/cpu-*.jsonl")):
        data = path.read_bytes()
        for number, line in enumerate(data.decode(errors="replace").splitlines(), 1):
            if "clef::cache_initialize" not in line:
                continue
            try:
                row = json.loads(line)
            except ValueError:
                continue
            evidence = {"path": str(path), "line": number, "file_sha256_as_read": sha(data)}
            if row.get("cache_metadata_error"):
                errors.append({"error": row["cache_metadata_error"], "evidence": evidence})
            cache = row.get("cache_memory")
            if not cache:
                continue
            specs = {}
            for spec in cache.get("spec_by_layer", {}).values():
                kind = spec["kind"]
                entry = specs.setdefault(kind, {"layer_count": 0, "page_size_bytes": [],
                                               "state_content_size_bytes": []})
                entry["layer_count"] += 1
                for key in ("page_size_bytes", "state_content_size_bytes"):
                    if spec.get(key) is not None and spec[key] not in entry[key]:
                        entry[key].append(spec[key])
            logical = {}
            for view in cache.get("actual_tensor_views", []):
                if view.get("layer") is not None:
                    kind = view.get("spec_kind") or "unclassified"
                    logical[kind] = logical.get(kind, 0) + view["logical_view_bytes"]
            records.append({"configuration": path.parents[1].name, "pid": row.get("pid"),
                "rank": row.get("rank"), "rank_source": row.get("rank_source"),
                "num_blocks": cache.get("num_blocks"), "layout": cache.get("layout"),
                "group_count": len(cache.get("groups", [])), "spec_summary": specs,
                "actual_tensor_view_count": len(cache.get("actual_tensor_views", [])),
                "unique_storage_count": len(cache.get("unique_storages", [])),
                "unique_storage_bytes": cache["unique_storage_bytes"],
                "unique_storage_GiB": cache["unique_storage_bytes"] / 2**30,
                "unique_storage_bytes_by_kind": cache.get("unique_storage_bytes_by_kind"),
                "logical_named_tensor_view_bytes_sum_by_kind": logical,
                "logical_view_scope": "Per-layer tensor view capacity; shared/overlapping views are not additive physical allocations",
                "deduplication": cache.get("deduplication"), "coverage": cache.get("coverage"),
                "evidence": evidence})
    return {"status": "actual_storage_metadata_available" if records else
                       "pending_observed_cache_initialize_or_memory_record",
            "records": records, "errors": errors,
            "classification": "Shared KV/Mamba backing allocations are one shared physical category; startup available budget and realized storage are separate values"}


def memory_trace_summaries(run):
    unit_source = None
    workspace_export_source = None
    windows = {}
    for manifest_path in run.glob("*/window-*/manifest.json"):
        manifest_data = manifest_path.read_bytes()
        manifest = json.loads(manifest_data)
        window = {"name": manifest_path.parent.name, "manifest": str(manifest_path),
                  "manifest_sha256": sha(manifest_data), "passed": manifest.get("passed"),
                  "request_window_started_at": manifest.get("request_window_started_at"),
                  "request_window_finished_at": manifest.get("request_window_finished_at")}
        for directory in manifest.get("new_rank_directories", []):
            windows[str(Path(directory).resolve())] = window
    try:
        source = Path(importlib.metadata.distribution("torch-npu").locate_file(
            "torch_npu/profiler/analysis/prof_common_func/_constant.py"))
        data = source.read_bytes()
        for number, line in enumerate(data.decode().splitlines(), 1):
            if re.search(r"B_TO_MB\s*=\s*1024(?:\.0)?\s*\*\*\s*2", line):
                unit_source = {"path": str(source), "line": number, "text": line.strip(), "sha256": sha(data)}
                break
        source = Path(importlib.metadata.distribution("torch-npu").locate_file(
            "torch_npu/profiler/analysis/prof_view/_memory_view_parser.py"))
        data = source.read_bytes()
        lines = data.decode().splitlines()
        for number, line in enumerate(lines, 1):
            if "cur_record.component_type == Constant.WORKSPACE_TYPE" in line:
                workspace_export_source = {"path": str(source), "line": number,
                    "text": "\n".join(lines[number - 1:number + 4]), "sha256": sha(data)}
                break
    except (OSError, importlib.metadata.PackageNotFoundError):
        pass
    reports = []
    for path in sorted(run.glob("*/traces/**/memory_record.csv")):
        components = {}
        count = 0
        for number, row in enumerate(csv.DictReader(path.open()), 2):
            count += 1
            name = row.get("Component", "unknown")
            device = row.get("Device Type", "unknown")
            entry = components.setdefault((name, device), {"component": name, "device": device,
                "record_count": 0, "metrics": {}})
            entry["record_count"] += 1
            for column in ("Total Allocated(MB)", "Total Reserved(MB)", "Total Active(MB)"):
                if not row.get(column):
                    continue
                value = float(row[column])
                point = {"value": value, "csv_row": number,
                         "timestamp_us": row.get("Timestamp(us)", "").strip()}
                metric = entry["metrics"].setdefault(column, {"first": point, "min": point, "max": point})
                metric["last"] = point
                if value < metric["min"]["value"]:
                    metric["min"] = point
                if value > metric["max"]["value"]:
                    metric["max"] = point
        for component in components.values():
            for metric in component["metrics"].values():
                metric["window_max_minus_min"] = metric["max"]["value"] - metric["min"]["value"]
                metric["window_max_minus_first"] = metric["max"]["value"] - metric["first"]["value"]
        operator = path.parent / "operator_memory.csv"
        allocations = {}
        named_workspace_count = 0
        allocation_count = 0
        if operator.is_file():
            for number, row in enumerate(csv.DictReader(operator.open()), 2):
                allocation_count += 1
                if "workspace" in row.get("Name", "").lower():
                    named_workspace_count += 1
                key = (row.get("Name"), float(row.get("Size(KB)") or 0))
                item = allocations.setdefault(key, {"name": key[0], "size_source_KB": key[1],
                    "count": 0, "example_csv_row": number,
                    "allocation_time_us": row.get("Allocation Time(us)", "").strip(),
                    "release_time_us": row.get("Release Time(us)", "").strip()})
                item["count"] += 1
        reports.append({"path": str(path), "sha256": sha(path.read_bytes()), "record_count": count,
                        "window": windows.get(str(path.parents[1].resolve())),
                        "unit": "CSV label MB is MiB (2^20 bytes)" if unit_source else "CSV source label MB; conversion unverified",
                        "components": list(components.values()),
                        "workspace_component_present": any(key[0] == "WORKSPACE" for key in components),
                        "operator_memory": {"path": str(operator), "sha256": sha(operator.read_bytes()) if operator.is_file() else None,
                            "allocation_count": allocation_count, "named_workspace_allocation_count": named_workspace_count,
                            "largest_allocation_groups": sorted(allocations.values(), key=lambda row: row["size_source_KB"], reverse=True)[:5]}})
    return {"unit_source": unit_source, "workspace_export_source": workspace_export_source, "records": reports,
            "scope": "Allocator fluctuation during short profiled requests; activation and temporary workspace remain combined without allocation attribution",
            "workspace_limit": "No separately typed WORKSPACE records or named allocations means independent workspace peak is unknown. The window max-minus-min PTA envelope only bounds any incremental workspace within those recorded PTA allocations; persistent baseline or untracked allocations are outside this bound.",
            "component_relation": "PTA+GE is composite; APP/PTA/GE/PTA+GE totals must not be added together"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", default="-")
    parser.add_argument("--supplement", action="store_true")
    args = parser.parse_args()
    run = args.run_dir.resolve()
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(), "run_dir": str(run),
        "units": {"startup_memory": "GiB = 2^30 bytes; log values rounded to 2 decimal places",
                  "npu_smi_hbm": "tool-labelled MB field; preserve raw output and do not equate device-wide HBM with per-worker allocated"},
        "scope": {"weights": "vLLM measured model memory usage per worker",
                  "cache": "vLLM startup KV/Mamba cache pool combined budget",
                  "activation": "dummy profile_run peak allocator increase used for startup budgeting; includes temporary torch allocations",
                  "non_torch": "device used minus torch reservation relative to startup baseline; includes runtime/HCCL/driver-related allocations",
                  "allocated_reserved": "PyTorch NPU allocator statistics after service warmup",
                  "workspace": "Pending allocation-attribution evidence; startup log does not separate runtime activations and temporary workspaces"},
        "environment": snapshot_environment(run),
        "services": [service_memory(path) for path in sorted(run.glob("*/service.log"))],
        "actual_cache_storage": actual_cache_storage(run),
        "profiled_request_memory": memory_trace_summaries(run),
    }
    if args.supplement:
        report["environment"]["supplemental_metadata"] = current_supplement()
    payload = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if args.output == "-":
        sys.stdout.write(payload)
    else:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(payload)


if __name__ == "__main__":
    main()
