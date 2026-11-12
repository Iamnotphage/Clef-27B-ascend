#!/usr/bin/env python3
"""Prepare/apply reversible observation-only patches to the pinned source trees.

Set CLEF_PROFILE_MARKERS=1 before starting the service to record CPU enqueue
scopes. Set CLEF_PROFILE_OBSERVE_DIR to write per-process timing JSONL. Keeping
both unset returns original callables at import time. No device synchronization
is inserted. Async scopes report wall time, including await, without a CPU
record_function spanning concurrent coroutines.
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
_REQUEST_ID = ContextVar("clef_profile_request_id", default=None)


def observe(name):
    def decorate(function):
        if not (_MARKERS or _DIRECTORY):
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
            token = _REQUEST_ID.set(request_id)
            return token, request_id, time.time_ns(), time.perf_counter_ns(), request_id_source

        def finish(state):
            token, request_id, wall_ns, perf_ns, request_id_source = state
            end_perf_ns = time.perf_counter_ns()
            _REQUEST_ID.reset(token)
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
                    "request_id_source": request_id_source,
                    "pid": os.getpid(), "thread_id": threading.get_ident(),
                    "rank": rank, "rank_source": rank_source,
                    "local_rank": os.environ.get("LOCAL_RANK"),
                    "start_wall_ns": wall_ns, "start_perf_ns": perf_ns,
                    "end_perf_ns": end_perf_ns,
                    "duration_ns": end_perf_ns - perf_ns,
                    "timing_scope": "host_wall_inclusive",
                }
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
        "profiler_forwarded": ["record_shapes", "with_flops"],
        "schedule_forwarded": False,
        "encoding_audit_boundary": "Source SHA audit requires restoring observation patches; saved encoding evidence can be used during numerical comparison.",
        "marker_coverage": {
            "status": "static_targets_prepared_pending_runtime_trace_acceptance",
            "acceptance": "Check each expected marker in actual worker trace; Ascend runtime class overrides can replace source methods.",
            "targets": [{"file": path, "class": cls, "function": function, "marker": marker}
                        for path, targets in TARGETS.items() for (cls, function), marker in targets.items()],
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
