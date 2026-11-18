#!/usr/bin/env python3
"""Standalone NPU dispatch probe for the E numeric reference (worker 22).

WHY THIS EXISTS
---------------
The E comparison is candidate-vs-reference.  Several questions about the
*reference* side are still open: which native op/kernel the reference GEMM
actually dispatches to for an ND weight vs an NZ weight, what math options are
actually in effect at runtime, and which host libraries are actually mapped.
A script that only compares output values cannot answer any of that.

This script therefore does ONE controlled window and records dispatch evidence:

  * a real CPU+NPU profiler window (``torch_npu.profiler``),
  * the two reference calls under **separately labelled** ``record_function``
    ranges (``REF_ND`` uses the ND weight, ``REF_NZ`` uses the product-format
    NZ weight), so every captured op can be attributed to one of them,
  * the runtime logical BF16 byte SHA-256 of x and of the **real** per-rank
    down_proj weight slice (sources are named; these hashes identify THIS run
    only and must not be used to retro-attribute earlier runs),
  * the actual dtype/shape/stride/format/storage of every operand,
  * the raw, uninterpreted values of the math-related device options,
  * the host library paths from /proc/self/maps plus their file SHA-256.

WEIGHT SOURCE (same as the frozen reference comparison)
-------------------------------------------------------
The weight is the model's real ``...mlp.down_proj.weight`` for ``--layer``,
column-sliced per rank exactly as the TP2 reference comparison does, then
transposed to the operator's ``w[K,N]`` orientation.  A random weight is never
substituted.

WHAT IT DELIBERATELY DOES NOT DO
--------------------------------
  * It does not load or call the custom E operator.
  * It does not change the reference: ``REF_ND`` is exactly the call the
    current runner already makes (``matmul(x, w)`` with the ND weight).
    ``REF_NZ`` is an *additional* diagnostic label for the same mathematical
    product with the product physical weight format.
  * It does not sweep shapes, block dims or baseK.
  * It does not claim matching/unmatching of anything.

A ``record_function`` label is NOT dispatch evidence by itself -- it only
correlates timeline entries.  The dispatch evidence is whatever op/kernel
records the profiler actually exports; this script reports those records as it
finds them, and reports an absent field as absent.

DEVICE MAPPING
--------------
Each rank must use its own logical device, otherwise two ranks point at the same
one.  The device is taken from ``LOCAL_RANK`` (falling back to ``RANK``) and
applied with ``torch.npu.set_device``; ``--device`` overrides it explicitly.
Two equivalent invocations:

    torchrun --nproc_per_node=2 wave3_dispatch_probe_22.py --out-dir <BASE>
    RANK=1 LOCAL_RANK=1 python3 wave3_dispatch_probe_22.py --out-dir <BASE>

Each invocation writes ``<BASE>/rank<N>/``; N comes from ``RANK``/``--rank``.

Outputs (per rank):

    meta.json             all recorded facts + script SHA
    numeric.json          REF_ND vs REF_NZ full-tensor compare (background only)
    exported_fields.json  what the profiler actually exported, incl. absences
    maps_filtered.txt     relevant /proc/self/maps lines
    trace/                raw profiler output (unmodified)
"""

import argparse
import csv
import glob
import hashlib
import json
import os
import platform
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

EXIT_OK = 0
EXIT_PRECONDITION = 2
EXIT_MATMUL_NZ_FAILED = 4

DEFAULT_MODEL_DIR = "/workspace/yangchen/models/Clef-27B"
WEIGHT_KEY_TEMPLATE = "model.language_model.layers.{layer}.mlp.down_proj.weight"

# Columns that plausibly carry an op / kernel / task identity in the profiler's
# exported CSV files.  Used only to *select* which columns to summarise; values
# are copied verbatim and never truncated.
IDENTITY_COLUMN_HINTS = ("op", "kernel", "name", "type", "task", "id", "node")
# Field families we are explicitly looking for and may have to report as absent.
WANTED_FIELD_HINTS = ("tiling", "block_dim", "core_type", "op_type", "kernel_type")
MAX_VALUES_PER_COLUMN = 40
MAX_ROWS_PER_CSV = 5000


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def logical_bf16_sha(t) -> str:
    """SHA-256 over the *logical* BF16 bytes of a tensor, in row-major order.

    A tensor produced by ``npu_format_cast`` carries an internal physical layout;
    hashing the logical content yields a digest comparable with a plain ND tensor
    holding the same values.
    """
    cpu = t.detach().to("cpu").contiguous()
    if cpu.dtype != torch.bfloat16:
        raise TypeError(f"expected bfloat16, got {cpu.dtype}")
    raw = cpu.view(torch.uint8).numpy().tobytes()
    return sha256_bytes(raw)


def tensor_facts(name: str, t) -> dict:
    """Everything we can state about a tensor's real storage, no interpretation."""
    facts = {
        "name": name,
        "dtype": str(t.dtype),
        "shape": list(t.shape),
        "stride": list(t.stride()),
        "storage_offset": int(t.storage_offset()),
        "numel": int(t.numel()),
        "element_size": int(t.element_size()),
        "is_contiguous": bool(t.is_contiguous()),
        "untyped_storage_nbytes": int(t.untyped_storage().nbytes()),
        "is_npu": bool(t.device.type == "npu"),
        "device": str(t.device),
    }
    try:
        facts["npu_format"] = int(torch_npu.get_npu_format(t))
    except Exception as exc:  # pragma: no cover - depends on device state
        facts["npu_format_error"] = f"{type(exc).__name__}: {exc}"
    return facts


def raw_option(name: str) -> dict:
    """Read a device option and report the raw value, with no interpretation.

    An empty result (``b''``) is a real observation: the option was not
    explicitly set in this process.  It is reported verbatim rather than being
    mapped onto a default.
    """
    try:
        value = torch_npu._C._npu_getOption(name)
    except Exception as exc:
        return {"name": name, "error": f"{type(exc).__name__}: {exc}"}
    out = {"name": name, "repr": repr(value)}
    if isinstance(value, (bytes, bytearray)):
        out["hex"] = bytes(value).hex()
        out["is_empty"] = len(value) == 0
    return out


def enable_internal_format() -> dict:
    """Turn on internal-format tensor creation and report how it was done."""
    record = {"requested": True, "set_via": None, "error": None}
    try:
        torch.npu.config.allow_internal_format = True
        record["set_via"] = "torch.npu.config"
        return record
    except Exception as exc:
        record["error"] = f"torch.npu.config: {type(exc).__name__}: {exc}"
    try:  # fallback path, same underlying option
        torch_npu.npu.config.allow_internal_format = True
        record["set_via"] = "torch_npu.npu.config"
    except Exception as exc:
        record["error"] += f" | torch_npu.npu.config: {type(exc).__name__}: {exc}"
    return record


def profiler_level_value(level: int):
    """Map a CLI integer onto the fixed ProfilerLevel enum member.

    ``ProfilerLevel.Level0`` is the *string* "Level0" (prof_common_func
    ``_constant.py``); passing a bare integer is an invalid parameter and makes
    the profiler fall back to its default, so the mapping must be explicit.
    """
    mapping = {
        0: torch_npu.profiler.ProfilerLevel.Level0,
        1: torch_npu.profiler.ProfilerLevel.Level1,
        2: torch_npu.profiler.ProfilerLevel.Level2,
    }
    if level not in mapping:
        raise ValueError(f"unsupported profiler level {level!r}")
    return mapping[level]


def collect_maps(out_dir: str) -> dict:
    """Filter /proc/self/maps for relevant libraries and hash the host files.

    Host mapping identity is deliberately kept separate from device kernel binary
    identity: this only proves which *host* files were mapped.
    """
    patterns = (
        "torch_npu", "/torch/", "cann", "ascend", "op_api", "libopapi",
        "libascendcl", "libhccl", "libacl", "libprof", "libmsprof",
    )
    selected, total = [], 0
    try:
        with open("/proc/self/maps", "r", errors="replace") as f:
            for line in f:
                total += 1
                parts = line.rstrip("\n").split(None, 5)
                if len(parts) < 6:
                    continue
                path = parts[5].strip()
                if not path.startswith("/"):
                    continue
                low = path.lower()
                if any(p in low for p in patterns):
                    selected.append(line.rstrip("\n"))
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}

    filtered_path = os.path.join(out_dir, "maps_filtered.txt")
    with open(filtered_path, "w") as f:
        f.write("\n".join(selected) + ("\n" if selected else ""))

    files, seen = [], set()
    for line in selected:
        parts = line.split(None, 5)
        if len(parts) < 6:
            continue
        raw_path = parts[5].strip()
        deleted = raw_path.endswith("(deleted)")
        path = raw_path[: -len("(deleted)")].strip() if deleted else raw_path
        if path in seen:
            continue
        seen.add(path)
        entry = {"path": path, "deleted_at_map_time": deleted}
        try:
            entry["size"] = os.path.getsize(path)
            entry["sha256"] = sha256_file(path)
        except Exception as exc:
            entry["hash_error"] = f"{type(exc).__name__}: {exc}"
        files.append(entry)

    return {
        "maps_total_lines": total,
        "filtered_lines": len(selected),
        "filtered_file": os.path.relpath(filtered_path, out_dir),
        "unique_files": len(files),
        "files": files,
        "note": "host mapping identity only; says nothing about the device kernel binary",
    }


def summarise_exported(trace_dir: str) -> dict:
    """Report what the profiler actually exported, without inventing fields.

    CSV parsing uses ``csv.reader`` so quoted fields containing commas survive;
    values are stored verbatim (no truncation) because they carry the target
    dispatch identity.
    """
    files, csv_summaries = [], []
    identity_fields, wanted_present = {}, set()

    for path in sorted(glob.glob(os.path.join(trace_dir, "**", "*"), recursive=True)):
        if not os.path.isfile(path):
            continue
        rel = os.path.relpath(path, trace_dir)
        entry = {"path": rel, "size": os.path.getsize(path)}
        try:
            entry["sha256"] = sha256_file(path)
        except Exception as exc:
            entry["hash_error"] = f"{type(exc).__name__}: {exc}"
        files.append(entry)

        if not path.lower().endswith(".csv"):
            continue
        try:
            with open(path, "r", newline="", errors="replace") as f:
                reader = csv.reader(f)
                header = next(reader, None)
                if header is None:
                    csv_summaries.append({"path": rel, "error": "empty file"})
                    continue
                header = [h.lstrip("﻿") for h in header]
                rows = []
                for row in reader:
                    rows.append(row)
                    if len(rows) >= MAX_ROWS_PER_CSV:
                        break
        except Exception as exc:
            csv_summaries.append({"path": rel, "error": f"{type(exc).__name__}: {exc}"})
            continue

        identity_cols, csv_wanted = {}, []
        for idx, col in enumerate(header):
            low = col.lower()
            if any(hint in low for hint in IDENTITY_COLUMN_HINTS):
                values, seen = [], set()
                for row in rows:
                    if idx >= len(row):
                        continue
                    value = row[idx]
                    if value and value not in seen:
                        seen.add(value)
                        values.append(value)
                    if len(values) >= MAX_VALUES_PER_COLUMN:
                        break
                if values:
                    identity_cols[col] = values
            if any(hint in low for hint in WANTED_FIELD_HINTS):
                csv_wanted.append(col)

        csv_summaries.append({
            "path": rel,
            "header": header,
            "rows_read": len(rows),
            "identity_columns": identity_cols,
            "wanted_fields_present": csv_wanted,
        })
        for col, values in identity_cols.items():
            identity_fields.setdefault(col, [])
            for value in values:
                if value not in identity_fields[col]:
                    identity_fields[col].append(value)
        wanted_present.update(csv_wanted)

    tiling_cols = [c for c in wanted_present if "tiling" in c.lower()]
    absent = [hint for hint in WANTED_FIELD_HINTS
              if not any(hint in c.lower() for c in wanted_present)]

    return {
        "trace_files": files,
        "csv_summaries": csv_summaries,
        # Dispatch identity is what these fields contain -- not "a CSV existed".
        "identity_fields": identity_fields,
        "identity_fields_found": sorted(identity_fields),
        "identity_fields_absent": not bool(identity_fields),
        "wanted_field_hints": list(WANTED_FIELD_HINTS),
        "wanted_fields_found": sorted(wanted_present),
        "wanted_fields_absent": absent,
        "tiling_columns": tiling_cols,
        "tiling_key_present": bool(tiling_cols),
    }


def compare_outputs(y_nd, y_nz) -> dict:
    """Full-tensor compare of the two reference labels (background evidence).

    This is NOT dispatch evidence; it only records whether the two physical
    weight formats produced the same output in this window.
    """
    a = y_nd.detach().float()
    b = y_nz.detach().float()
    if a.shape != b.shape:
        return {"error": f"shape mismatch {tuple(a.shape)} vs {tuple(b.shape)}"}
    d = (a - b).abs()
    tol = 1e-5 + 0.016 * b.abs()
    cpu_d = d.to("cpu")
    return {
        "note": "background only: output equality does not identify dispatch",
        "shape": list(a.shape),
        "elements": int(d.numel()),
        "max_abs_diff": float(cpu_d.max()) if d.numel() else 0.0,
        "nonzero_count": int((cpu_d > 0).sum()),
        "gt_gate_tolerance_count": int((d > tol).to("cpu").sum()),
        "tol_formula": "1e-5 + 0.016*|REF_NZ|",
    }


def load_weight(model_dir: str, layer: int, rank: int, tp_size: int) -> dict:
    """Load the model's real down_proj weight and slice it exactly as the
    reference comparison does (column slice per rank, then transpose)."""
    from safetensors import safe_open

    index_path = Path(model_dir) / "model.safetensors.index.json"
    weight_map = json.loads(index_path.read_text())["weight_map"]
    key = WEIGHT_KEY_TEMPLATE.format(layer=layer)
    shard = Path(model_dir) / weight_map[key]

    with safe_open(shard, framework="pt", device="cpu") as handle:
        full = handle.get_tensor(key)

    per = full.shape[1] // tp_size
    start = rank * per
    stop = start + per
    sliced = full[:, start:stop]
    w_cpu = sliced.t().contiguous()

    return {
        "w_cpu": w_cpu,
        "source": {
            "model_dir": str(model_dir),
            "index_file": str(index_path),
            "index_file_size": index_path.stat().st_size,
            "model_key": key,
            "shard_file": str(shard),
            "shard_size": shard.stat().st_size,
            "full_shape": list(full.shape),
            "tp_size": tp_size,
            "rank_slice": [start, stop],
            "sliced_shape": list(sliced.shape),
            "transposed_shape": list(w_cpu.shape),
            "transform": "full[:, rank*per:(rank+1)*per].t().contiguous()",
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", required=True, help="base directory; this script creates <out-dir>/rank<N>/")
    ap.add_argument("--rank", type=int, default=int(os.environ.get("RANK", "0")))
    ap.add_argument("--local-rank", type=int, default=int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0"))))
    ap.add_argument("--device", default=None, help="explicit device, e.g. npu:1; default follows --local-rank")
    ap.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--tp-size", type=int, default=2)
    ap.add_argument("--rows", type=int, default=967)
    ap.add_argument("--seed", type=int, default=20261118)
    ap.add_argument("--profiler-level", type=int, default=1, choices=(0, 1, 2))
    ap.add_argument("--record-op-args", action="store_true", default=False)
    ap.add_argument("--no-op-attr", action="store_true", default=False)
    args = ap.parse_args()

    rank_dir = os.path.join(os.path.abspath(args.out_dir), f"rank{args.rank}")
    trace_dir = os.path.join(rank_dir, "trace")
    os.makedirs(trace_dir, exist_ok=True)

    global torch, torch_npu
    import torch
    import torch_npu

    device = args.device if args.device else f"npu:{args.local_rank}"
    # torch_npu.set_device accepts an int index; keep the string form for logging.
    device_index = int(device.split(":")[1]) if device.startswith("npu:") else int(args.local_rank)
    torch.npu.set_device(device_index)

    meta = {
        "probe": "wave3_dispatch_probe_22",
        "identity_scope": (
            "this run only: x is generated from --seed and the weight is the real per-rank "
            "model slice; both are identified by the runtime hashes below. Do not use these "
            "hashes to retro-attribute earlier runs."
        ),
        "started_utc": utcnow(),
        "rank": args.rank,
        "argv": sys.argv,
        "device_mapping": {
            "device": device,
            "rank": args.rank,
            "local_rank": args.local_rank,
            "source": "explicit --device" if args.device else "LOCAL_RANK (fallback RANK)",
            "note": "each rank must use its own logical device; two ranks on one device collide",
        },
        "host": {
            "platform": platform.platform(),
            "python": sys.version.split()[0],
            "pid": os.getpid(),
        },
        "versions": {
            "torch": getattr(torch, "__version__", None),
            "torch_npu": getattr(torch_npu, "__version__", None),
            "torch_npu_file": getattr(torch_npu, "__file__", None),
        },
        "script": {"path": os.path.abspath(__file__), "sha256": sha256_file(os.path.abspath(__file__))},
        "shape_request": {"rows": args.rows, "seed": args.seed, "layer": args.layer, "tp_size": args.tp_size},
    }

    # ---- real weight, real slice (never a random substitute)
    loaded = load_weight(args.model_dir, args.layer, args.rank, args.tp_size)
    w_cpu = loaded["w_cpu"]
    k, n = w_cpu.shape
    meta["weight"] = loaded["source"]
    meta["weight"]["logical_bf16_sha256"] = logical_bf16_sha(w_cpu)
    meta["shape"] = {"m": args.rows, "k": int(k), "n": int(n)}

    # ---- x from the seed, hashed at runtime
    gen = torch.Generator().manual_seed(args.seed)
    x_cpu = torch.randn(args.rows, k, generator=gen).to(torch.bfloat16)
    meta["inputs"] = {
        "x_logical_bf16_sha256": logical_bf16_sha(x_cpu),
        "generator": "torch.randn(float32, Generator(seed)) -> bfloat16, [rows,K]",
    }

    x = x_cpu.to(device)
    w_nd = w_cpu.to(device)
    meta["tensors"] = {"x": tensor_facts("x", x), "w_nd": tensor_facts("w_nd", w_nd)}

    # ---- options BEFORE touching internal format (raw values, no interpretation).
    # Read after the device is initialised so the getter reports real process state.
    meta["options_before"] = [
        raw_option("ALLOW_MATMUL_HF32"), raw_option("ALLOW_INTERNAL_FORMAT"), raw_option("CUBE_MATH_TYPE")
    ]

    meta["allow_internal_format"] = enable_internal_format()
    meta["options_after_set"] = [
        raw_option("ALLOW_MATMUL_HF32"), raw_option("ALLOW_INTERNAL_FORMAT"), raw_option("CUBE_MATH_TYPE")
    ]

    w_nz = None
    try:
        w_nz = torch_npu.npu_format_cast(w_nd, 29)
        meta["tensors"]["w_nz"] = tensor_facts("w_nz", w_nz)
        meta["tensors"]["w_nz"]["logical_bf16_sha256"] = logical_bf16_sha(w_nz)
        meta["tensors"]["w_nz"]["logical_matches_w_nd"] = (
            meta["tensors"]["w_nz"]["logical_bf16_sha256"] == meta["weight"]["logical_bf16_sha256"]
        )
    except Exception as exc:
        meta["npu_format_cast_error"] = f"{type(exc).__name__}: {exc}"
        meta["npu_format_cast_traceback"] = traceback.format_exc()

    # ---- PRECONDITION: no NZ descriptor, no measurement. Stop before any matmul.
    nz_format_ok = w_nz is not None and meta["tensors"].get("w_nz", {}).get("npu_format") == 29
    meta["nz_format_ok"] = bool(nz_format_ok)
    if not nz_format_ok:
        meta["stopped_before_matmul"] = True
        meta["stop_reason"] = "npu_format_cast did not yield npu_format==29"
        meta["finished_utc"] = utcnow()
        with open(os.path.join(rank_dir, "meta.json"), "w") as f:
            json.dump(meta, f, indent=1, sort_keys=True)
        print("PROBE_SUMMARY " + json.dumps({
            "rank": args.rank, "out_dir": rank_dir, "exit": "PRECONDITION_NOT_MET",
            "nz_format_ok": False, "device": device,
            "npu_formats": {key: val.get("npu_format") for key, val in meta["tensors"].items()},
        }))
        return EXIT_PRECONDITION

    # ---- warmup OUTSIDE the profiled window (keeps the labels clean)
    warmup = {}
    try:
        torch.matmul(x, w_nd)
        torch.npu.synchronize()
        warmup["REF_ND"] = "ok"
    except Exception as exc:
        warmup["REF_ND"] = f"{type(exc).__name__}: {exc}"
    try:
        torch.matmul(x, w_nz)
        torch.npu.synchronize()
        warmup["REF_NZ"] = "ok"
    except Exception as exc:
        warmup["REF_NZ"] = f"{type(exc).__name__}: {exc}"
    meta["warmup"] = warmup

    if warmup.get("REF_NZ") != "ok":
        meta["stopped_before_profiling"] = True
        meta["finished_utc"] = utcnow()
        with open(os.path.join(rank_dir, "meta.json"), "w") as f:
            json.dump(meta, f, indent=1, sort_keys=True)
        print("PROBE_SUMMARY " + json.dumps({
            "rank": args.rank, "out_dir": rank_dir, "exit": "REF_NZ_UNSUPPORTED", "warmup": warmup,
        }))
        return EXIT_MATMUL_NZ_FAILED

    # ---- the single profiled window
    prof_mod = torch_npu.profiler
    level_value = profiler_level_value(args.profiler_level)
    exp_kwargs = dict(
        profiler_level=level_value,
        export_type=[prof_mod.ExportType.Text],
        record_op_args=bool(args.record_op_args),
    )
    exp_record = {"requested": {**exp_kwargs, "profiler_level": str(level_value)},
                  "op_attr": not args.no_op_attr}
    if args.no_op_attr:
        exp_cfg = prof_mod._ExperimentalConfig(**exp_kwargs)
        exp_record["built_with"] = "without_op_attr"
    else:
        try:
            exp_cfg = prof_mod._ExperimentalConfig(op_attr=True, **exp_kwargs)
            exp_record["built_with"] = "op_attr"
        except Exception as exc:
            exp_record["op_attr_error"] = f"{type(exc).__name__}: {exc}"
            exp_cfg = prof_mod._ExperimentalConfig(**exp_kwargs)
            exp_record["built_with"] = "without_op_attr"
    meta["experimental_config"] = exp_record

    y_nd = y_nz = None
    with prof_mod.profile(
        activities=[prof_mod.ProfilerActivity.CPU, prof_mod.ProfilerActivity.NPU],
        schedule=prof_mod.schedule(wait=0, warmup=0, active=2, repeat=1, skip_first=0),
        on_trace_ready=prof_mod.tensorboard_trace_handler(trace_dir, worker_name=f"rank{args.rank}"),
        record_shapes=True,
        profile_memory=False,
        with_stack=False,
        with_flops=False,
        with_modules=False,
        experimental_config=exp_cfg,
    ) as prof:
        with torch.profiler.record_function("REF_ND"):
            y_nd = torch.matmul(x, w_nd)
            torch.npu.synchronize()
        prof.step()
        with torch.profiler.record_function("REF_NZ"):
            y_nz = torch.matmul(x, w_nz)
            torch.npu.synchronize()
        prof.step()

    meta["labels"] = {
        "REF_ND": "torch.matmul(x, w_nd) -- the current reference call (ND weight)",
        "REF_NZ": "torch.matmul(x, w_nz) -- same product with the product physical weight format (NZ)",
        "note": "labels only correlate the timeline; the dispatch evidence is the exported op/kernel records",
    }

    if y_nd is not None and y_nz is not None:
        with open(os.path.join(rank_dir, "numeric.json"), "w") as f:
            json.dump(compare_outputs(y_nd, y_nz), f, indent=1, sort_keys=True)

    exported = summarise_exported(trace_dir)
    with open(os.path.join(rank_dir, "exported_fields.json"), "w") as f:
        json.dump(exported, f, indent=1, sort_keys=True)

    meta["host_maps"] = collect_maps(rank_dir)
    meta["device_identity"] = {
        "source": "profiler-exported records only (see exported_fields.json)",
        "identity_fields_found": exported["identity_fields_found"],
        "identity_fields_absent": exported["identity_fields_absent"],
        "wanted_fields_found": exported["wanted_fields_found"],
        "wanted_fields_absent": exported["wanted_fields_absent"],
        "tiling_key_present": exported["tiling_key_present"],
        "note": ("host maps cannot identify the device kernel binary; a field that was "
                 "not exported is reported as absent rather than inferred"),
    }
    meta["finished_utc"] = utcnow()

    with open(os.path.join(rank_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1, sort_keys=True)

    print("PROBE_SUMMARY " + json.dumps({
        "rank": args.rank,
        "out_dir": rank_dir,
        "device": device,
        "nz_format_ok": True,
        "npu_formats": {key: val.get("npu_format") for key, val in meta["tensors"].items()},
        "weight_sha256": meta["weight"]["logical_bf16_sha256"],
        "x_sha256": meta["inputs"]["x_logical_bf16_sha256"],
        "trace_files": len(exported["trace_files"]),
        "csv_files": len(exported["csv_summaries"]),
        "identity_fields_found": exported["identity_fields_found"],
        "tiling_key_present": exported["tiling_key_present"],
        "wanted_fields_absent": exported["wanted_fields_absent"],
        "host_mapped_files": meta["host_maps"].get("unique_files"),
    }))
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
