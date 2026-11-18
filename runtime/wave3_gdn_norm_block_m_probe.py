#!/usr/bin/env python3
"""Four-arm ``BLOCK_M`` probe for the GDN output norm (48-call microbenchmark).

Fact boundary: this probe times a **48-call microbenchmark batch** at the real
production shape.  It does not reproduce a per-request layer chain, and a batch
difference must not be converted into a service-latency claim.  Adoption still
requires the service A/B and the official 2PP quality gate.

Frozen contract (``docs/gdn-norm-block-m-2026-11-18.md``):

* shapes: ``x``/``z`` BF16 ``[23208, 128]``, ``weight`` BF16 ``[128]``;
  ``group_size=None`` -> N=128, ``ngroups=1``, RMS, ``norm_before_gate=True``,
  activation ``swish``, ``eps=1e-6``.
* only ``BLOCK_M`` changes, through ``CLEF_GDN_NORM_BLOCK_M`` in
  ``vllm_ascend/ops/triton/layernorm_gated.py``.  The wrapper signature,
  ``out``/``mean``/``rstd`` allocation, ``BLOCK_N`` and the grid derivation are
  untouched, and no ``num_warps``/``num_stages`` argument is passed.

Protocol highlights:

* one ``x``/``z``/``weight`` set, allocated once and shared by every arm; the
  bytes actually used are hashed.
* per arm: one warmup batch, then ``--rounds`` timed batches of
  ``--calls-per-batch`` consecutive calls in a precomputed rotating order.
* one device event pair around each batch; synchronize only before and after
  the batch.  ``batch_device_ms`` and ``batch_wall_ms`` are separate, and
  ``batch_mean_*`` is the batch value divided by the call count.
* a per-call maximum is reported only with ``--per-call``; that extra batch is
  event-instrumented and excluded from the primary comparison.
* compiled ``num_warps``/``num_stages``/coalesce fields and the ``BLOCK_M``
  constexpr are read from the live Triton cache entry of each specialization.
  Missing fields stay ``null``; defaults are never substituted.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path

ROWS = 23208
COLS = 128
ENV_VAR = "CLEF_GDN_NORM_BLOCK_M"
ALLOWED_BLOCK_M = (16, 32, 64, 128)
DEFAULT_TREE = Path("/workspace/yangchen/work/src/vllm-ascend")
TARGET = "vllm_ascend/ops/triton/layernorm_gated.py"


def sha256_tensor(torch, tensor) -> str:
    cpu = tensor.detach().to("cpu").contiguous()
    return hashlib.sha256(cpu.view(torch.uint8).numpy().tobytes()).hexdigest()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stats(values):
    return {"n": len(values), "median": statistics.median(values),
            "mean": statistics.fmean(values), "min": min(values), "max": max(values),
            "samples": [round(value, 6) for value in values]}


def grid_row(arm: int) -> dict:
    blocks = (ROWS + arm - 1) // arm
    return {"blocks": blocks, "ngroups": 1, "requested_grid": [blocks, 1],
            "padding_rows": blocks * arm - ROWS,
            "provenance": "ceil(rows/BLOCK_M), ngroups=1 as derived by the wrapper"}


def unwrap_jit_function(kernel):
    """Walk Triton's decorator wrappers down to the JITFunction.

    ``layer_norm_gated.py`` stacks two ``@triton.heuristics`` decorators over
    ``@triton.jit``.  ``Heuristics`` exposes ``.fn`` but not ``device_caches``;
    only the innermost JITFunction owns the compiled-kernel cache and
    ``arg_names``.
    """
    chain = []
    seen = set()
    current = kernel
    while not hasattr(current, "device_caches"):
        chain.append(type(current).__name__)
        if id(current) in seen or not hasattr(current, "fn"):
            return None, chain
        seen.add(id(current))
        current = current.fn
    chain.append(type(current).__name__)
    return current, chain


def kernel_info(jit_fn, compiled) -> dict:
    metadata = compiled.metadata
    fields, missing = {}, []
    for name in ("num_warps", "num_stages", "coalesce_factor", "coalesce_axis",
                 "coalesce_grid_ceil_div", "program_grid_mapping_applied", "shared"):
        value = getattr(metadata, name, None)
        fields[name] = value
        if value is None:
            missing.append(name)
    constants = {}
    try:
        arg_names = list(jit_fn.arg_names)
        for key, value in compiled.src.constants.items():
            if isinstance(key, tuple) and len(key) == 1 and isinstance(key[0], int):
                constants[arg_names[key[0]]] = value
            else:
                constants[str(key)] = value
    except Exception as exc:  # introspection must not hide the timing result
        constants = {"error": repr(exc)}
    return {"name": compiled.name, "hash": compiled.hash, "metadata": fields,
            "missing_metadata": missing, "constants": constants}


def compare(torch, actual, expected) -> dict:
    a = actual.detach().to("cpu")
    b = expected.detach().to("cpu")
    if a.dtype in (torch.bfloat16, torch.float16):
        ai, bi = a.view(torch.int16), b.view(torch.int16)
    elif a.dtype == torch.float32:
        ai, bi = a.view(torch.int32), b.view(torch.int32)
    else:
        ai, bi = a, b
    af, bf = a.float(), b.float()
    finite_a = bool(torch.isfinite(af).all())
    finite_b = bool(torch.isfinite(bf).all())
    delta = (af - bf).abs()
    return {
        "dtype": str(a.dtype).replace("torch.", ""),
        "total_elements": int(ai.numel()),
        "equal_elements": int((ai == bi).sum()),
        "bit_equal": bool(torch.equal(ai, bi)),
        "finite_actual": finite_a,
        "finite_expected": finite_b,
        "nan_actual": int(torch.isnan(af).sum()),
        "nan_expected": int(torch.isnan(bf).sum()),
        "inf_actual": int(torch.isinf(af).sum()),
        "inf_expected": int(torch.isinf(bf).sum()),
        "max_abs": float(delta.max()) if finite_a and finite_b else None,
        "mean_abs": float(delta.mean()) if finite_a and finite_b else None,
        "rmse": float(delta.square().mean().sqrt()) if finite_a and finite_b else None,
    }


def timed_batch(torch, call, calls: int) -> dict:
    begin = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)
    torch.npu.synchronize()
    wall_start = time.perf_counter()
    begin.record()
    for _ in range(calls):
        call()
    end.record()
    torch.npu.synchronize()
    wall_ms = (time.perf_counter() - wall_start) * 1000.0
    device_ms = begin.elapsed_time(end)
    return {"batch_device_ms": device_ms, "batch_wall_ms": wall_ms,
            "batch_mean_device_ms": device_ms / calls,
            "batch_mean_wall_ms": wall_ms / calls}


def per_call_batch(torch, call, calls: int) -> dict:
    starts = [torch.npu.Event(enable_timing=True) for _ in range(calls)]
    ends = [torch.npu.Event(enable_timing=True) for _ in range(calls)]
    torch.npu.synchronize()
    wall_start = time.perf_counter()
    for index in range(calls):
        starts[index].record()
        call()
        ends[index].record()
    torch.npu.synchronize()
    wall_ms = (time.perf_counter() - wall_start) * 1000.0
    samples = [starts[index].elapsed_time(ends[index]) for index in range(calls)]
    return {"instrumented_batch_wall_ms": wall_ms,
            "per_call_device_ms": stats(samples),
            "note": "each call carries an event pair; overhead is included and this "
                    "batch is not used for the arm comparison"}


def load_blob(torch, path: Path):
    blob = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(blob, dict):
        raise ValueError("--inputs must be a dict with x, z and weight tensors")
    for key in ("x", "z", "weight"):
        if key not in blob:
            raise ValueError(f"--inputs is missing {key!r}")
    return blob


def build_inputs(torch, args):
    if args.inputs is not None:
        blob = load_blob(torch, args.inputs)
        x, z, weight = blob["x"], blob["z"], blob["weight"]
        eps = float(blob.get("eps", args.eps))
        source = "file"
    else:
        generator = torch.Generator().manual_seed(args.seed)
        x = torch.randn(ROWS, COLS, generator=generator, dtype=torch.float32).to(torch.bfloat16)
        z = torch.randn(ROWS, COLS, generator=generator, dtype=torch.float32).to(torch.bfloat16)
        weight = torch.ones(COLS, dtype=torch.bfloat16)
        eps = args.eps
        source = "synthetic"
    if x.dtype != torch.bfloat16 or z.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16:
        raise ValueError("x, z and weight must all be BF16")
    if tuple(x.shape) != (ROWS, COLS) or tuple(z.shape) != (ROWS, COLS):
        raise ValueError(f"x/z must be [{ROWS}, {COLS}], got {tuple(x.shape)} and {tuple(z.shape)}")
    if tuple(weight.shape) != (COLS,):
        raise ValueError(f"weight must be [{COLS}], got {tuple(weight.shape)}")
    for name, tensor in (("x", x), ("z", z), ("weight", weight)):
        if tensor.stride(-1) != 1 or not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous with stride(-1)==1")
    return {"x": x, "z": z, "weight": weight, "eps": eps, "source": source,
            "path": str(args.inputs) if args.inputs is not None else None,
            "shape": {"x": list(x.shape), "z": list(z.shape), "weight": list(weight.shape)},
            "strides": {"x": list(x.stride()), "z": list(z.stride()), "weight": list(weight.stride())},
            "sha256": {"x": sha256_tensor(torch, x), "z": sha256_tensor(torch, z),
                       "weight": sha256_tensor(torch, weight)}}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--tree", type=Path, default=DEFAULT_TREE)
    parser.add_argument("--arms", default="16,32,64,128")
    parser.add_argument("--reference-arm", type=int, default=64)
    parser.add_argument("--calls-per-batch", type=int, default=48)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--warmup-calls", type=int, default=48)
    parser.add_argument("--per-call", action="store_true")
    parser.add_argument("--seed", type=int, default=20261118)
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument("--inputs", type=Path, default=None)
    parser.add_argument("--expect-source-sha", default=None)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    tree = args.tree.expanduser().resolve()
    arms = [int(value) for value in args.arms.split(",") if value.strip()]
    if len(set(arms)) != len(arms) or any(arm not in ALLOWED_BLOCK_M for arm in arms):
        parser.error(f"arms must be distinct values from {ALLOWED_BLOCK_M}")
    if args.reference_arm not in arms:
        parser.error("--reference-arm must be one of the probed arms")
    if args.calls_per_batch < 1 or args.warmup_calls < 1 or args.rounds < 3:
        parser.error("calls/warmup must be positive and rounds must be at least 3")
    if args.eps <= 0 or not math.isfinite(args.eps):
        parser.error("eps must be a positive finite value")

    report = {"schema": "gdn-norm-block-m-probe/1",
              "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "tree": str(tree), "target": TARGET, "rows": ROWS, "cols": COLS,
              "arms": arms, "reference_arm": args.reference_arm,
              "environment": {"argv": sys.argv[1:] if argv is None else argv,
                              "env": os.environ.get(ENV_VAR)},
              "protocol": {
                  "batch": "48-call microbenchmark batch; not a per-request layer chain",
                  "calls_per_batch": args.calls_per_batch,
                  "warmup_calls_per_arm": args.warmup_calls,
                  "rounds": args.rounds,
                  "round_orders": [[arms[(index + shift) % len(arms)] for index in range(len(arms))]
                                   for shift in range(args.rounds)],
                  "buffers": "x/z/weight allocated once on device and reused; the wrapper allocates "
                             "out and rstd per call (production behavior unchanged)",
                  "allocator": "returned output tuples are dropped after each call; the caching "
                               "allocator reuses the blocks on the same stream, so reuse keeps "
                               "the ordered execution safe",
                  "cache_state": "each arm's Triton specialization is compiled during warmup; no "
                                 "other NPU work runs in this process",
                  "sync": "torch.npu.synchronize() before and after each batch; none inside a batch",
                  "device_metric": "one event pair around the batch; batch_mean_device_ms = "
                                   "batch_device_ms / calls",
                  "wall_metric": "batch_wall_ms includes host dispatch and the final synchronize",
                  "pairing": "per-round paired delta/ratio against the reference arm",
                  "max_metric": "batch max across rounds; per-call max only with --per-call",
                  "env_switching": f"{ENV_VAR} is set between arms, never inside a timed batch",
              }}
    failures = []

    if not tree.is_dir():
        report["error"] = f"tree not found: {tree}"
        report["ready"] = False
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=1) + "\n")
        print(json.dumps(report, indent=1))
        return 1

    sys.path.insert(0, str(tree))
    import torch  # noqa: E402  (deliberately after --help and path selection)
    import torch_npu  # noqa: E402,F401
    import triton  # noqa: E402
    from vllm_ascend.ops.triton.layernorm_gated import (  # noqa: E402
        _layer_norm_fwd_1pass_kernel_npu,
        layer_norm_fwd_npu,
    )

    source_path = Path(inspect.getsourcefile(layer_norm_fwd_npu)).resolve()
    expected_path = (tree / TARGET).resolve()
    source_sha = sha256_file(source_path) if source_path.is_file() else None
    source_text = source_path.read_text() if source_path.is_file() else ""
    report["source"] = {"path": str(source_path), "expected_path": str(expected_path),
                        "path_matches_tree": source_path == expected_path,
                        "sha256": source_sha,
                        "expected_sha256": args.expect_source_sha,
                        "expected_sha256_matches": (None if args.expect_source_sha is None
                                                    else source_sha == args.expect_source_sha),
                        "knob_present": ENV_VAR in source_text}
    if source_path != expected_path:
        failures.append("source path is not the pinned tree target")
    if ENV_VAR not in source_text:
        failures.append(f"{ENV_VAR} is absent; run the installer first")
    if args.expect_source_sha is not None and source_sha != args.expect_source_sha:
        failures.append("source SHA-256 does not match --expect-source-sha")
    jit_fn, unwrap_chain = unwrap_jit_function(_layer_norm_fwd_1pass_kernel_npu)
    report["kernel_object"] = {
        "module_attribute": type(_layer_norm_fwd_1pass_kernel_npu).__name__,
        "unwrap_chain": unwrap_chain,
        "unwrapped": type(jit_fn).__name__ if jit_fn is not None else None,
        "has_device_caches": jit_fn is not None,
        "has_arg_names": bool(jit_fn is not None and hasattr(jit_fn, "arg_names")),
    }
    if jit_fn is None or not hasattr(jit_fn, "arg_names"):
        failures.append("could not unwrap the Triton decorators to a JITFunction")
    if failures:
        report["failures"] = failures
        report["ready"] = False
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=1) + "\n")
        print(json.dumps(report, indent=1))
        return 1

    torch.npu.set_device(args.device)
    device = torch.npu.current_device()
    inputs = build_inputs(torch, args)
    x = inputs["x"].npu()
    z = inputs["z"].npu()
    weight = inputs["weight"].npu()
    eps = inputs["eps"]
    inputs.pop("x")
    inputs.pop("z")
    inputs.pop("weight")
    report["inputs"] = inputs
    report["versions"] = {"torch": torch.__version__, "torch_npu": torch_npu.__version__,
                          "triton": triton.__version__}
    try:
        from importlib.metadata import version as package_version
        report["versions"]["vllm_ascend"] = package_version("vllm-ascend")
    except Exception:
        report["versions"]["vllm_ascend"] = None
    try:
        report["device"] = {"id": device, "name": torch.npu.get_device_name(args.device)}
    except Exception as exc:
        report["device"] = {"id": device, "name": None, "name_error": repr(exc)}

    def call():
        return layer_norm_fwd_npu(x, weight, None, eps, z=z, group_size=None,
                                  norm_before_gate=True, is_rms_norm=True,
                                  activation="swish")

    report["grid"] = {str(arm): grid_row(arm) for arm in arms}
    specializations, correctness = {}, {}
    reference_out = reference_rstd = None
    from triton.runtime import driver as triton_driver
    cache_device = triton_driver.active.get_current_device()
    report["device"]["triton_cache_device"] = cache_device
    cache = jit_fn.device_caches[cache_device][0]

    def flush(phase: str) -> None:
        report["phase"] = phase
        report["failures"] = failures
        report["ready"] = not failures
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n")

    for arm in [args.reference_arm] + [value for value in arms if value != args.reference_arm]:
        os.environ[ENV_VAR] = str(arm)
        before_keys = set(cache.keys())
        for _ in range(args.warmup_calls):
            call()
        torch.npu.synchronize()
        new_keys = [key for key in cache.keys() if key not in before_keys]
        entries = [kernel_info(jit_fn, cache[key]) for key in new_keys]
        block_m_values = [entry["constants"].get("BLOCK_M") for entry in entries
                          if isinstance(entry.get("constants"), dict)]
        specialization_ok = arm in block_m_values
        specializations[str(arm)] = {
            "new_cache_entries": len(entries),
            "kernels": entries,
            "requested_block_m": arm,
            "block_m_values": block_m_values,
            "specialization_ok": specialization_ok,
        }
        if not specialization_ok:
            failures.append(f"BLOCK_M={arm} did not select a matching kernel specialization")

        out, mean, rstd = call()
        torch.npu.synchronize()
        if arm == args.reference_arm:
            reference_out, reference_rstd = out.clone(), rstd.clone()
        correctness[str(arm)] = {
            "mean_is_none": mean is None,
            "out_vs_reference": compare(torch, out, reference_out),
            "rstd_vs_reference": compare(torch, rstd, reference_rstd),
        }
        if not all(correctness[str(arm)][name]["finite_actual"] for name in ("out_vs_reference",
                                                                              "rstd_vs_reference")):
            failures.append(f"BLOCK_M={arm} produced non-finite values")

    report["specializations"] = specializations
    report["correctness"] = correctness
    flush("specializations")

    rounds = []
    for round_index, order in enumerate(report["protocol"]["round_orders"]):
        row = {"round": round_index, "order": order, "arms": {}}
        for arm in order:
            os.environ[ENV_VAR] = str(arm)
            row["arms"][str(arm)] = timed_batch(torch, call, args.calls_per_batch)
        reference = row["arms"][str(args.reference_arm)]
        for arm in order:
            timing = row["arms"][str(arm)]
            timing["paired_device_delta_ms"] = timing["batch_device_ms"] - reference["batch_device_ms"]
            timing["paired_device_ratio"] = (timing["batch_device_ms"] / reference["batch_device_ms"]
                                             if reference["batch_device_ms"] else None)
        rounds.append(row)
        report["timing"] = {"rounds": rounds}
        flush(f"round-{round_index}")

    per_call = None
    if args.per_call:
        per_call = {}
        for arm in arms:
            os.environ[ENV_VAR] = str(arm)
            per_call[str(arm)] = per_call_batch(torch, call, args.calls_per_batch)
    os.environ.pop(ENV_VAR, None)

    summary = {}
    for arm in arms:
        device_values = [row["arms"][str(arm)]["batch_device_ms"] for row in rounds]
        wall_values = [row["arms"][str(arm)]["batch_wall_ms"] for row in rounds]
        deltas = [row["arms"][str(arm)]["paired_device_delta_ms"] for row in rounds]
        ratios = [row["arms"][str(arm)]["paired_device_ratio"] for row in rounds]
        summary[str(arm)] = {
            "batch_device_ms": stats(device_values),
            "batch_wall_ms": stats(wall_values),
            "batch_max_device_ms": max(device_values),
            "batch_mean_device_ms": stats([value / args.calls_per_batch for value in device_values]),
            "paired_device_delta_ms": stats(deltas),
            "paired_device_ratio": (stats(ratios) if all(value is not None for value in ratios) else None),
            "per_call": per_call[str(arm)] if per_call else None,
        }

    report["specializations"] = specializations
    report["correctness"] = correctness
    report["timing"] = {"rounds": rounds, "summary": summary}
    report["failures"] = failures
    report["ready"] = not failures
    report["notes"] = [
        "batch timings are microbenchmark evidence only; service p95 and quality are not implied",
        "bit equality against the reference arm is a target, not a guarantee",
        "no elementwise tolerance gate is introduced here; adoption uses the service A/B and "
        "the official 2PP quality gate",
        "grid is the wrapper-derived request; backend coalesce metadata is reported separately",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n")
    print(json.dumps({key: report[key] for key in ("schema", "at", "source", "device", "inputs",
                                                   "timing", "failures", "ready")},
                     ensure_ascii=False, indent=1))
    print(f"written {args.output}")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
