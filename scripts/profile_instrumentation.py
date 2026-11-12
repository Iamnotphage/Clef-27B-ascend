#!/usr/bin/env python3
"""Prepare/apply reversible observation-only patches to the pinned source trees.

Set CLEF_PROFILE_MARKERS=1 before starting the service to record CPU enqueue
scopes. Set CLEF_PROFILE_OBSERVE_DIR to write per-process timing JSONL. Keeping
both unset returns original callables at import time. No device synchronization
is inserted. Async scopes report wall time, including await, without a CPU
record_function spanning concurrent coroutines.
Set CLEF_PROFILE_SCOPE_FILTER to comma-separated exact marker names to observe
only those scopes. An unset filter includes all scopes; an empty filter includes
none. Filtering happens when the service imports the patched modules.
Nested scopes include inner observation overhead; the outermost JSONL write is
outside its own recorded interval. Use observation-disabled runs for throughput.
"""

import argparse
import ast
from datetime import datetime, timezone
import difflib
import hashlib
import json
from pathlib import Path
import subprocess


OBSERVER = '''# SPDX-License-Identifier: Apache-2.0
"""Optional Clef profiling scopes; CPU timing is distinct from device time."""
from contextlib import nullcontext
from contextvars import ContextVar
from functools import wraps
import inspect
import json
import os
from pathlib import Path
import sys
import threading
import time

_MARKERS = os.environ.get("CLEF_PROFILE_MARKERS") == "1"
_DIRECTORY = os.environ.get("CLEF_PROFILE_OBSERVE_DIR")
_FILTER_RAW = os.environ.get("CLEF_PROFILE_SCOPE_FILTER")
_SCOPE_FILTER = (None if _FILTER_RAW is None else
                 {name.strip() for name in _FILTER_RAW.split(",") if name.strip()})
_REQUEST_ID = ContextVar("clef_profile_request_id", default=None)
_BATCH_REQUEST_IDS = ContextVar("clef_profile_batch_request_ids", default=())


def cache_metadata(worker, supplied_config):
    """Read bounded cache metadata without tensor transfer or device work."""
    runner = getattr(worker, "model_runner", None)
    config = getattr(runner, "kv_cache_config", supplied_config)
    groups = getattr(config, "kv_cache_groups", ())
    spec_by_layer = {}
    group_rows = []
    for index, group in enumerate(groups):
        spec = group.kv_cache_spec
        per_layer = getattr(spec, "kv_cache_specs", {})
        names = list(group.layer_names)
        group_rows.append({"index": index, "spec_kind": type(spec).__name__, "layers": names})
        for name in names:
            item = per_layer.get(name, spec)
            spec_by_layer[name] = {
                "kind": type(item).__name__,
                "page_size_bytes": getattr(item, "page_size_bytes", None),
                "state_content_size_bytes": getattr(item, "state_content_size_bytes", None),
                "shapes": getattr(item, "shapes", None),
                "dtypes": [str(dtype) for dtype in getattr(item, "dtypes", ())],
            }
    tensors, storages = [], {}

    def visit(value, path, layer=None, depth=0):
        if len(tensors) >= 512 or depth > 4:
            return
        if callable(getattr(value, "untyped_storage", None)):
            storage = value.untyped_storage()
            pointer, size = storage.data_ptr(), storage.nbytes()
            device = str(value.device)
            key = f"{device}:{pointer}:{size}"
            kind = spec_by_layer.get(layer, {}).get("kind")
            tensors.append({"path": path, "layer": layer, "spec_kind": kind,
                "shape": list(value.shape), "stride": list(value.stride()),
                "dtype": str(value.dtype), "device": device,
                "storage_offset_elements": value.storage_offset(),
                "logical_view_bytes": value.numel() * value.element_size(),
                "storage_key": key})
            entry = storages.setdefault(key, {"device": device, "data_ptr": pointer,
                "nbytes": size, "layers": [], "spec_kinds": []})
            if layer is not None and layer not in entry["layers"]:
                entry["layers"].append(layer)
            if kind is not None and kind not in entry["spec_kinds"]:
                entry["spec_kinds"].append(kind)
        elif isinstance(value, dict):
            for key, child in list(value.items())[:256]:
                visit(child, f"{path}.{key}", layer, depth + 1)
        elif isinstance(value, (list, tuple)):
            for index, child in enumerate(value[:256]):
                visit(child, f"{path}[{index}]", layer, depth + 1)

    # runner.kv_caches supplies actual backing allocations. Per-layer bindings
    # supply the spec association; both are deduplicated by storage pointer.
    visit(getattr(runner, "kv_caches", ()), "runner.kv_caches")
    compilation = getattr(runner, "compilation_config", None)
    context = getattr(compilation, "static_forward_context", {})
    for layer in list(spec_by_layer)[:256]:
        module = context.get(layer) if isinstance(context, dict) else None
        visit(getattr(module, "kv_cache", ()), f"layers.{layer}.kv_cache", layer)
    by_kind = {}
    for storage in storages.values():
        kinds = sorted(storage["spec_kinds"])
        category = kinds[0] if len(kinds) == 1 else (
            "shared:" + "+".join(kinds) if kinds else "unclassified")
        by_kind[category] = by_kind.get(category, 0) + storage["nbytes"]
    descriptors = [{key: getattr(item, key, None) for key in
                    ("size", "layers", "layer_stride", "block_stride", "offset", "host_resident")}
                   for item in list(getattr(config, "kv_cache_tensors", ()))[:256]]
    return {"num_blocks": getattr(config, "num_blocks", None),
        "layout": getattr(config, "kv_cache_layout", None), "groups": group_rows,
        "spec_by_layer": spec_by_layer, "configured_descriptors": descriptors,
        "actual_tensor_views": tensors, "unique_storages": list(storages.values()),
        "unique_storage_bytes": sum(item["nbytes"] for item in storages.values()),
        "unique_storage_bytes_by_kind": by_kind,
        "coverage": "runner cache list plus configured layer bindings; bounded metadata read only",
        "deduplication": "device, untyped_storage.data_ptr and storage.nbytes; mixed-spec shared allocations remain shared"}


def observe(name):
    def decorate(function):
        if not (_MARKERS or _DIRECTORY) or (_SCOPE_FILTER is not None and name not in _SCOPE_FILTER):
            return function
        signature = inspect.signature(function)

        def begin(args, kwargs):
            bound = signature.bind_partial(*args, **kwargs).arguments
            request_id = bound.get("request_id")
            request_id_source = "explicit_argument" if request_id is not None else None
            if request_id is None:
                raw_request = bound.get("raw_request")
                headers = getattr(raw_request, "headers", {})
                header_id = headers.get("X-Request-Id")
                if header_id is not None:
                    # Match BaseServing._base_request_id + create_decision.
                    request_id = f"decision-{header_id}"
                    request_id_source = "http_header"
            if request_id is None:
                request = bound.get("request")
                request_id = getattr(request, "request_id", None)
                if request_id is not None:
                    if name in {"clef::api_handler_including_json", "clef::frontend_request"}:
                        request_id = f"decision-{request_id}"
                    request_id_source = "request_object"
            if request_id is None:
                request_id = _REQUEST_ID.get()
                request_id_source = "context" if request_id is not None else None
            batch_ids = _BATCH_REQUEST_IDS.get()
            if "scheduler_output" in bound:
                output = bound.get("scheduler_output")
                scheduled = getattr(output, "num_scheduled_tokens", None)
                batch_ids = tuple(scheduled) if isinstance(scheduled, dict) else ()
                if request_id is None and len(batch_ids) == 1:
                    request_id = batch_ids[0]
                    request_id_source = "scheduler_output_single_request"
            token = _REQUEST_ID.set(request_id)
            batch_token = _BATCH_REQUEST_IDS.set(batch_ids)
            return token, batch_token, request_id, time.time_ns(), time.perf_counter_ns(), request_id_source, bound, batch_ids

        def finish(state):
            token, batch_token, request_id, wall_ns, perf_ns, request_id_source, bound, batch_ids = state
            end_perf_ns = time.perf_counter_ns()
            _REQUEST_ID.reset(token)
            _BATCH_REQUEST_IDS.reset(batch_token)
            if _DIRECTORY:
                rank = os.environ.get("RANK")
                rank_source = "environment" if rank is not None else None
                # Inspect an already loaded/initialized process group only.
                # Observation never initializes distributed or imports torch.
                torch = sys.modules.get("torch")
                distributed = getattr(torch, "distributed", None)
                if (rank is None and distributed is not None
                        and distributed.is_available() and distributed.is_initialized()):
                    rank = distributed.get_rank()
                    rank_source = "initialized_torch_distributed"
                row = {
                    "name": name, "request_id": request_id,
                    "batch_request_ids": list(batch_ids),
                    "request_id_source": request_id_source,
                    "pid": os.getpid(), "thread_id": threading.get_ident(),
                    "rank": rank, "rank_source": rank_source,
                    "local_rank": os.environ.get("LOCAL_RANK"),
                    "start_wall_ns": wall_ns, "start_perf_ns": perf_ns,
                    "end_perf_ns": end_perf_ns,
                    "duration_ns": end_perf_ns - perf_ns,
                    "timing_scope": "host_wall_inclusive",
                }
                if name == "clef::request_finished_metrics":
                    stats = bound.get("req_stats")
                    fields = ("queued_ts", "scheduled_ts", "first_token_ts", "last_token_ts")
                    native = {key: getattr(stats, key, None) for key in fields}
                    queued, scheduled, first, last = (native[key] for key in fields)
                    native["queue_seconds"] = (
                        scheduled - queued if queued and scheduled else None)
                    native["scheduled_to_output_seconds"] = (
                        last - scheduled if last and scheduled else None)
                    native["scheduled_to_first_output_seconds"] = (
                        first - scheduled if first and scheduled else None)
                    native["num_preemptions"] = getattr(stats, "num_preemptions", None)
                    native["num_prompt_tokens"] = bound.get("num_prompt_tokens")
                    native["timestamp_scope"] = "engine_process_monotonic_seconds; compare differences within that process only"
                    row["engine_metrics"] = native
                if name == "clef::cache_initialize":
                    try:
                        row["cache_memory"] = cache_metadata(bound.get("self"), bound.get("kv_cache_config"))
                    except (AttributeError, TypeError, RuntimeError, ValueError) as error:
                        row["cache_metadata_error"] = f"{type(error).__name__}: {error}"
                directory = Path(_DIRECTORY)
                directory.mkdir(parents=True, exist_ok=True)
                # Each process has its own append-only file. One write preserves
                # rows when asynchronous requests finish on different threads.
                fd = os.open(directory / f"cpu-{os.getpid()}.jsonl",
                             os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
                try:
                    os.write(fd, (json.dumps(row) + "\\n").encode())
                finally:
                    os.close(fd)

        if inspect.iscoroutinefunction(function):
            @wraps(function)
            async def asynchronous(*args, **kwargs):
                state = begin(args, kwargs)
                try:
                    return await function(*args, **kwargs)
                finally:
                    finish(state)
            return asynchronous

        @wraps(function)
        def synchronous(*args, **kwargs):
            state = begin(args, kwargs)
            if _MARKERS:
                from torch.profiler import record_function
                scope = record_function(name)
            else:
                scope = nullcontext()
            try:
                with scope:
                    return function(*args, **kwargs)
            finally:
                finish(state)
        return synchronous
    return decorate
'''


TARGETS = {
    "vllm/model_executor/models/clef_joint_schema.py": {
        ("JointSchemaPooler", "forward"): "clef::joint_pooler",
        ("JointSchemaPooler", "_indices"): "clef::schema_indices_h2d",
        ("JointSchemaHead", "forward"): "clef::joint_head",
        ("EvidenceRoutingLayer", "forward"): "clef::joint_evidence_routing",
    },
    "vllm/model_executor/models/clef_flash.py": {
        ("ClefFlashForJointSchema", "_embed_lexical"): "clef::candidate_embedding_tp",
    },
    "vllm/model_executor/models/qwen3_5.py": {
        ("Qwen3_5ForConditionalGeneration", "forward"): "clef::backbone",
    },
    "vllm/model_executor/models/qwen3_next.py": {
        ("Qwen3NextDecoderLayer", "forward"): "clef::backbone_decoder_layer",
        ("Qwen3NextModel", "forward"): "clef::language_backbone",
    },
    "vllm/model_executor/models/qwen3_vl.py": {
        ("Qwen3_VisionTransformer", "forward"): "clef::vision_encoder",
    },
    "vllm/entrypoints/generate/structured_decisions/serving.py": {
        (None, "parse_questions"): "clef::request_schema_parse",
        ("ServingStructuredDecisions", "create_decision"): "clef::frontend_request",
    },
    "vllm/entrypoints/generate/structured_decisions/api_router.py": {
        (None, "create_structured_decision"): "clef::api_handler_including_json",
    },
    "vllm/entrypoints/openai/decisions/strategies.py": {
        ("JointSchemaStrategy", "render_state"): "clef::state_render",
        ("JointSchemaStrategy", "_tokens"): "clef::tokenizer",
        ("JointSchemaStrategy", "encode"): "clef::schema_tokenize_encode",
        ("JointSchemaStrategy", "read"): "clef::frontend_read_including_engine",
        ("ClefFlashStrategy", "_render_media"): "clef::media_decode_process",
    },
    "vllm/v1/core/sched/scheduler.py": {
        ("Scheduler", "schedule"): "clef::scheduler",
        ("Scheduler", "add_request"): "clef::scheduler_add_request",
    },
    "vllm/v1/metrics/stats.py": {
        ("IterationStats", "update_from_finished_request"): "clef::request_finished_metrics",
    },
}

ASCEND_TARGETS = {
    "vllm_ascend/worker/worker.py": {
        ("NPUWorker", "execute_model"): "clef::worker_execute_model",
        ("NPUWorker", "initialize_from_config"): "clef::cache_initialize",
    },
    "vllm_ascend/models/qwen3_5_vl.py": {
        ("_AscendVLPreprocessMixin", "_process_image_input"): "clef::image_device_process",
        ("_AscendVLPreprocessMixin", "_process_video_input"): "clef::video_device_process",
        ("_AscendVLPreprocessMixin", "_resize_and_patchify"): "clef::image_resize_patchify",
    },
}


def sha(content):
    return hashlib.sha256(content).hexdigest()


def annotate(source, targets):
    tree = ast.parse(source)
    rows = source.splitlines(keepends=True)
    edits = []
    found = set()
    for outer in tree.body:
        candidates = outer.body if isinstance(outer, ast.ClassDef) else [outer]
        for node in candidates:
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            key = (outer.name if isinstance(outer, ast.ClassDef) else None, node.name)
            if key not in targets:
                continue
            # Insert beneath existing decorators (e.g. @staticmethod), keeping
            # the binding and HTTP decorator contracts intact.
            indent = rows[node.lineno - 1][:node.col_offset]
            edits.append((node.lineno - 1, f'{indent}@_clef_observe({targets[key]!r})\n'))
            found.add(key)
    if found != set(targets):
        raise ValueError(f"Expected functions missing: {set(targets) - found}")
    imports = [node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))]
    if not imports:
        raise ValueError("Expected import anchor missing")
    edits.append((imports[0].lineno - 1,
                  "from vllm.profiler.clef_observe import observe as _clef_observe\n"))
    for line, content in sorted(edits, reverse=True):
        rows.insert(line, content)
    result = "".join(rows)
    ast.parse(result)
    return result


def prepare(vllm_root, ascend_root):
    expected_heads = {
        vllm_root: "24420b7a341af37a9bc3186e6fc6062512e0aa58",
        ascend_root: "89eb8a5e7a76d085b453a1aeadb6883e12d5ea2d",
    }
    for root, expected in expected_heads.items():
        actual = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
        if actual != expected:
            raise ValueError(f"Pinned source HEAD differs at {root}: {actual}")
    changes = []
    helper = "vllm/profiler/clef_observe.py"
    if (vllm_root / helper).exists():
        raise ValueError("Observation helper already exists; restore prior patch first")
    changes.append(("vllm", helper, b"", OBSERVER.encode()))
    for relative, targets in TARGETS.items():
        before = (vllm_root / relative).read_bytes()
        if b"_clef_observe" in before:
            raise ValueError(f"Already instrumented: {relative}")
        after = annotate(before.decode(), targets).encode()
        changes.append(("vllm", relative, before, after))
    for relative, targets in ASCEND_TARGETS.items():
        before = (ascend_root / relative).read_bytes()
        if b"_clef_observe" in before:
            raise ValueError(f"Already instrumented: {relative}")
        after = annotate(before.decode(), targets).encode()
        changes.append(("ascend", relative, before, after))
    relative = "vllm_ascend/profiler/torch_npu_profiler.py"
    before = (ascend_root / relative).read_bytes()
    anchor = "            profile_memory=profiler_config.torch_profiler_with_memory,\n"
    if before.decode().count(anchor) != 1:
        raise ValueError("Pinned Ascend profiler anchor differs")
    after = before.decode().replace(anchor, anchor +
        "            record_shapes=profiler_config.torch_profiler_record_shapes,\n"
        "            with_flops=profiler_config.torch_profiler_with_flops,\n").encode()
    ast.parse(after)
    changes.append(("ascend", relative, before, after))
    return changes


def restore(manifest_path):
    manifest = json.loads(manifest_path.read_text())
    if not manifest["applied"]:
        raise ValueError("Dry-run manifest has no applied source changes")
    for row in manifest["files"]:
        path = Path(row["path"])
        current = path.read_bytes() if path.exists() else b""
        if sha(current) != row["after_sha256"]:
            raise ValueError(f"Concurrent source edit; restore manually: {path}")
        if sha((manifest_path.parent / row["backup"]).read_bytes()) != row["before_sha256"]:
            raise ValueError(f"Backup SHA-256 differs: {row['backup']}")
    for row in manifest["files"]:
        path = Path(row["path"])
        if row["existed"]:
            path.write_bytes((manifest_path.parent / row["backup"]).read_bytes())
        else:
            path.unlink()
    print(json.dumps({"restored": len(manifest["files"]), "manifest": str(manifest_path)}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vllm-root", type=Path)
    parser.add_argument("--ascend-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--restore", type=Path, metavar="MANIFEST")
    args = parser.parse_args()
    if args.restore:
        restore(args.restore.resolve())
        return
    if not all((args.vllm_root, args.ascend_root, args.output_dir)):
        parser.error("--vllm-root, --ascend-root and --output-dir are required")
    roots = {"vllm": args.vllm_root.resolve(), "ascend": args.ascend_root.resolve()}
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "instrumentation-manifest.json"
    if manifest_path.exists():
        parser.error("Use a new output directory for each instrumentation run")
    changes = prepare(roots["vllm"], roots["ascend"])
    patches = {"vllm": [], "ascend": []}
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "applied": args.apply, "roots": {key: str(value) for key, value in roots.items()},
        "scope": "CPU inclusive wall/enqueue scopes; device time requires trace linkage",
        "synchronization_inserted": False,
        "observation_overhead": "Inclusive parent scopes contain child logger overhead. Each own logger write is outside that scope's interval. Formal throughput must disable both observation environment variables.",
        "scope_filter": "CLEF_PROFILE_SCOPE_FILTER is an optional comma-separated exact marker allowlist, read at module import; unset includes all scopes, empty includes none.",
        "profiler_forwarded": ["record_shapes", "with_flops"],
        "schedule_forwarded": False,
        "encoding_audit_boundary": "Source SHA audit requires restoring observation patches; saved encoding evidence can be used during numerical comparison.",
        "marker_coverage": {
            "status": "static_targets_prepared_pending_runtime_trace_acceptance",
            "acceptance": "Check each expected marker in actual worker trace; Ascend runtime class overrides can replace source methods.",
            "targets": [{"source_tree": root, "file": path, "class": cls,
                         "function": function, "marker": marker}
                        for root, source_targets in (("vllm", TARGETS), ("ascend", ASCEND_TARGETS))
                        for path, targets in source_targets.items()
                        for (cls, function), marker in targets.items()],
            "rank_mapping": "Prefer RANK env, then an already initialized torch.distributed rank; null requires PID-to-worker metadata mapping.",
        },
        "files": [],
    }
    for root, relative, before, after in changes:
        path = roots[root] / relative
        backup = Path("source-before") / root / relative
        (output / backup).parent.mkdir(parents=True, exist_ok=True)
        (output / backup).write_bytes(before)
        manifest["files"].append({"path": str(path), "relative": relative,
            "root": root, "backup": str(backup), "existed": path.exists(),
            "before_sha256": sha(before), "after_sha256": sha(after)})
        patches[root].extend(difflib.unified_diff(
            before.decode().splitlines(keepends=True), after.decode().splitlines(keepends=True),
            fromfile=f"a/{relative}" if before else "/dev/null", tofile=f"b/{relative}"))
    for root, patch in patches.items():
        (output / f"{root}-profiling.patch").write_text("".join(patch))
    # Write recovery metadata before mutating source files.
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    if args.apply:
        for root, relative, before, after in changes:
            (roots[root] / relative).write_bytes(after)
    print(json.dumps({"applied": args.apply, "files": len(changes), "manifest": str(manifest_path)}))


if __name__ == "__main__":
    main()
