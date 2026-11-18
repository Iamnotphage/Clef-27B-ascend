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
    ranges (``REF_ND`` uses the ND weight, ``REF_NZ`` uses the production-format
    NZ weight), so every captured op can be attributed to one of them,
  * the runtime logical BF16 byte SHA-256 of x and w (new inputs generated from
    the seed; these hashes identify THIS run only and must not be used to
    retro-attribute earlier runs),
  * the actual dtype/shape/stride/format/storage of every operand,
  * the raw, uninterpreted values of the math-related device options,
  * the host library paths from /proc/self/maps plus their file SHA-256.

WHAT IT DELIBERATELY DOES NOT DO
--------------------------------
  * It does not load or call the custom E operator.
  * It does not change the reference in any way: ``REF_ND`` is exactly the
    call shape the current runner already uses (``matmul(x, w)`` with the ND
    weight).  ``REF_NZ`` is an *additional* diagnostic label for the same
    mathematical product with the production physical weight format.
  * It does not sweep shapes, block dims or baseK.
  * It does not claim matching/unmatching of anything.

A ``record_function`` label is NOT dispatch evidence by itself -- it only
correlates timeline entries.  The dispatch evidence is whatever op/kernel
records the profiler actually exports; this script reports those records as it
finds them and reports the absence of a field as an absence.

Usage (one rank per invocation; each rank writes its own directory):

    python3 wave3_dispatch_probe_22.py --out-dir <BASE> [--rank N]

Outputs:

    <BASE>/rank<N>/meta.json             all recorded facts + script SHA
    <BASE>/rank<N>/numeric.json          REF_ND vs REF_NZ full-tensor compare
    <BASE>/rank<N>/exported_fields.json  what the profiler actually exported
    <BASE>/rank<N>/maps_filtered.txt     relevant /proc/self/maps lines
    <BASE>/rank<N>/trace/                raw profiler output (unmodified)
"""

import argparse
import glob
import hashlib
import json
import os
import platform
import sys
import traceback
from datetime import datetime, timezone

EXIT_OK = 0
EXIT_NO_NZ_FORMAT = 3
EXIT_MATMUL_NZ_FAILED = 4

# Columns that plausibly carry an op / kernel / task identity in the profiler's
# exported CSV files.  Used only to *select* which columns to summarise; the
# values are copied verbatim.
IDENTITY_COLUMN_HINTS = ("op", "kernel", "name", "type", "task", "id", "node")
# Field families we are explicitly looking for and may have to report as absent.
WANTED_FIELD_HINTS = ("tiling", "block_dim", "core_type", "op_type", "kernel_type")


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

    A tensor produced by ``npu_format_cast`` has an internal physical layout; we
    hash the logical content so the digest can be compared against a plain ND
    tensor holding the same values.
    """
    cpu = t.detach().to("cpu").contiguous()
    if cpu.dtype != torch.bfloat16:
        raise TypeError(f"expected bfloat16, got {cpu.dtype}")
    # bfloat16 -> uint8 view keeps the exact 2-byte little-endian encoding.
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

    An empty result (``b''``) is a real observation: it means the option has not
    been explicitly set in this process.  We report it verbatim rather than
    mapping it onto a default.
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


def collect_maps(out_dir: str) -> dict:
    """Filter /proc/self/maps for relevant libraries and hash the host files.

    Host mapping identity is deliberately kept separate from device kernel
    binary identity: this only proves which *host* files were mapped.
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
    """Report what the profiler actually exported, without inventing fields."""
    files, csv_summaries = [], []
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
            with open(path, "r", errors="replace") as f:
                header = f.readline().lstrip("﻿").strip().split(",")
                rows = [f.readline() for _ in range(400)]
        except Exception as exc:
            csv_summaries.append({"path": rel, "error": f"{type(exc).__name__}: {exc}"})
            continue

        lowered = [h.strip().strip('"').lower() for h in header]
        identity_cols, wanted_cols = {}, {}
        for idx, col in enumerate(lowered):
            if any(h in col for h in IDENTITY_COLUMN_HINTS):
                vals, seen = [], set()
                for row in rows:
                    if not row:
                        continue
                    cells = row.rstrip("\n").split(",")
                    if idx >= len(cells):
                        continue
                    v = cells[idx].strip().strip('"')
                    if v and v not in seen:
                        seen.add(v)
                        vals.append(v[:80])
                    if len(vals) >= 40:
                        break
                identity_cols[header[idx].strip()] = vals
            if any(h in col for h in WANTED_FIELD_HINTS):
                wanted_cols[header[idx].strip()] = True

        csv_summaries.append({
            "path": rel,
            "header": [h.strip() for h in header],
            "rows_sampled": len([r for r in rows if r]),
            "identity_columns": identity_cols,
            "wanted_fields_present": sorted(wanted_cols),
        })

    present = set()
    for summary in csv_summaries:
        present.update(summary.get("wanted_fields_present", []))
    return {
        "trace_files": files,
        "csv_summaries": csv_summaries,
        # Explicit gap reporting: these are the fields we went looking for.
        "wanted_field_hints": list(WANTED_FIELD_HINTS),
        "wanted_fields_found": sorted(present),
        "wanted_fields_absent": [h for h in WANTED_FIELD_HINTS if not any(h in p.lower() for p in present)],
        "tiling_key_present": any("tiling" in p.lower() for p in present),
    }


def compare_outputs(y_nd, y_nz, out_dir: str) -> dict:
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
    bad = d > tol
    cpu_d = d.to("cpu")
    return {
        "note": "background only: output equality does not identify dispatch",
        "shape": list(a.shape),
        "elements": int(d.numel()),
        "max_abs_diff": float(cpu_d.max()) if d.numel() else 0.0,
        "nonzero_count": int((cpu_d > 0).sum()),
        "gt_gate_tolerance_count": int(bad.to("cpu").sum()),
        "tol_formula": "1e-5 + 0.016*|REF_NZ|",
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", required=True, help="base directory; this script creates <out-dir>/rank<N>/")
    ap.add_argument("--rank", type=int, default=int(os.environ.get("RANK", "0")))
    ap.add_argument("--m", type=int, default=967)
    ap.add_argument("--k", type=int, default=8704)
    ap.add_argument("--n", type=int, default=5120)
    ap.add_argument("--seed", type=int, default=20261118)
    ap.add_argument("--device", default="npu:0")
    ap.add_argument("--profiler-level", type=int, default=1, choices=(0, 1, 2))
    ap.add_argument("--record-op-args", action="store_true", default=False)
    ap.add_argument("--op-attr", action="store_true", default=True)
    args = ap.parse_args()

    rank_dir = os.path.join(os.path.abspath(args.out_dir), f"rank{args.rank}")
    trace_dir = os.path.join(rank_dir, "trace")
    os.makedirs(trace_dir, exist_ok=True)

    global torch, torch_npu
    import torch
    import torch_npu

    meta = {
        "probe": "wave3_dispatch_probe_22",
        "identity_scope": (
            "this run only: inputs are newly generated from --seed and identified by the "
            "runtime hashes below; do not use these hashes to retro-attribute earlier runs"
        ),
        "started_utc": utcnow(),
        "rank": args.rank,
        "argv": sys.argv,
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
        "shape": {"m": args.m, "k": args.k, "n": args.n, "seed": args.seed},
    }

    # ---- deterministic inputs (generated here; never reused from an older run)
    gen = torch.Generator(device="cpu").manual_seed(args.seed)
    x_cpu = torch.randn(args.m, args.k, generator=gen, dtype=torch.float32).to(torch.bfloat16)
    w_cpu = torch.randn(args.k, args.n, generator=gen, dtype=torch.float32).to(torch.bfloat16)

    meta["inputs"] = {
        "x_logical_bf16_sha256": logical_bf16_sha(x_cpu),
        "w_logical_bf16_sha256": logical_bf16_sha(w_cpu),
        "generator": "torch.randn(float32, Generator(seed)) -> bfloat16, [M,K] then [K,N]",
    }

    x = x_cpu.to(args.device)
    w_nd = w_cpu.to(args.device)
    meta["tensors"] = {"x": tensor_facts("x", x), "w_nd": tensor_facts("w_nd", w_nd)}

    # ---- options BEFORE touching internal format (raw values, no interpretation).
    # Read after the device is initialised so the getter reports real process state.
    opt_before = [raw_option("ALLOW_MATMUL_HF32"), raw_option("ALLOW_INTERNAL_FORMAT"), raw_option("CUBE_MATH_TYPE")]

    set_record = enable_internal_format()
    meta["allow_internal_format"] = set_record
    meta["options_before"] = opt_before
    meta["options_after_set"] = [
        raw_option("ALLOW_MATMUL_HF32"), raw_option("ALLOW_INTERNAL_FORMAT"), raw_option("CUBE_MATH_TYPE")
    ]

    w_nz = None
    try:
        w_nz = torch_npu.npu_format_cast(w_nd, 29)
        meta["tensors"]["w_nz"] = tensor_facts("w_nz", w_nz)
    except Exception as exc:
        meta["npu_format_cast_error"] = f"{type(exc).__name__}: {exc}"
        meta["npu_format_cast_traceback"] = traceback.format_exc()

    if w_nz is not None:
        try:
            meta["tensors"]["w_nz"]["logical_bf16_sha256"] = logical_bf16_sha(w_nz)
            meta["tensors"]["w_nz"]["logical_matches_w_nd"] = (
                meta["tensors"]["w_nz"]["logical_bf16_sha256"] == meta["inputs"]["w_logical_bf16_sha256"]
            )
        except Exception as exc:
            meta["tensors"]["w_nz"]["logical_sha_error"] = f"{type(exc).__name__}: {exc}"
    meta["tensors"]["w_nd"]["logical_bf16_sha256"] = logical_bf16_sha(w_nd)
    meta["tensors"]["x"]["logical_bf16_sha256"] = logical_bf16_sha(x)

    nz_format_ok = w_nz is not None and meta["tensors"].get("w_nz", {}).get("npu_format") == 29
    meta["nz_format_ok"] = bool(nz_format_ok)

    # ---- warmup OUTSIDE the profiled window (keeps the labels clean)
    warmup = {}
    try:
        torch.matmul(x, w_nd)
        torch.npu.synchronize()
        warmup["REF_ND"] = "ok"
    except Exception as exc:
        warmup["REF_ND"] = f"{type(exc).__name__}: {exc}"
    if w_nz is not None:
        try:
            torch.matmul(x, w_nz)
            torch.npu.synchronize()
            warmup["REF_NZ"] = "ok"
        except Exception as exc:
            warmup["REF_NZ"] = f"{type(exc).__name__}: {exc}"
    meta["warmup"] = warmup

    if warmup.get("REF_NZ", "").startswith(("RuntimeError", "TypeError", "NotImplementedError")):
        meta["finished_utc"] = utcnow()
        with open(os.path.join(rank_dir, "meta.json"), "w") as f:
            json.dump(meta, f, indent=1, sort_keys=True)
        print("PROBE_SUMMARY " + json.dumps({"rank": args.rank, "exit": "REF_NZ_UNSUPPORTED",
                                             "warmup": warmup, "nz_format_ok": meta["nz_format_ok"]}))
        return EXIT_MATMUL_NZ_FAILED

    # ---- the single profiled window
    prof_mod = torch_npu.profiler
    exp_kwargs = dict(
        profiler_level=args.profiler_level,
        export_type=[prof_mod.ExportType.Text],
        record_op_args=bool(args.record_op_args),
    )
    exp_record = {"requested": exp_kwargs, "op_attr": bool(args.op_attr)}
    try:
        exp_cfg = prof_mod._ExperimentalConfig(op_attr=bool(args.op_attr), **exp_kwargs)
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
        "REF_ND": "torch.matmul(x, w_nd) -- the current reference call shape (ND weight)",
        "REF_NZ": "torch.matmul(x, w_nz) -- same product with the production physical weight format (NZ)",
        "note": "labels only correlate the timeline; the dispatch evidence is the exported op/kernel records",
    }

    if y_nd is not None and y_nz is not None:
        with open(os.path.join(rank_dir, "numeric.json"), "w") as f:
            json.dump(compare_outputs(y_nd, y_nz, rank_dir), f, indent=1, sort_keys=True)

    exported = summarise_exported(trace_dir)
    with open(os.path.join(rank_dir, "exported_fields.json"), "w") as f:
        json.dump(exported, f, indent=1, sort_keys=True)

    meta["host_maps"] = collect_maps(rank_dir)
    meta["device_identity"] = {
        "source": "profiler-exported records only (see exported_fields.json)",
        "obtained": bool(exported["csv_summaries"]),
        "tiling_key_present": exported["tiling_key_present"],
        "wanted_fields_found": exported["wanted_fields_found"],
        "wanted_fields_absent": exported["wanted_fields_absent"],
        "note": "device kernel binary identity is not derivable from host maps; absent fields are reported as absent",
    }
    meta["finished_utc"] = utcnow()

    with open(os.path.join(rank_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1, sort_keys=True)

    print("PROBE_SUMMARY " + json.dumps({
        "rank": args.rank,
        "out_dir": rank_dir,
        "nz_format_ok": meta["nz_format_ok"],
        "npu_formats": {k: v.get("npu_format") for k, v in meta["tensors"].items()},
        "warmup": warmup,
        "trace_files": len(exported["trace_files"]),
        "csv_files": len(exported["csv_summaries"]),
        "tiling_key_present": exported["tiling_key_present"],
        "wanted_fields_absent": exported["wanted_fields_absent"],
        "host_mapped_files": meta["host_maps"].get("unique_files"),
    }))
    return EXIT_OK if meta["nz_format_ok"] else EXIT_NO_NZ_FORMAT


if __name__ == "__main__":
    sys.exit(main())
