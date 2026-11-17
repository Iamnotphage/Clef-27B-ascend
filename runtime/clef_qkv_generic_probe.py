#!/usr/bin/env python3
"""Validate/time the independent TP1/TP2 ClefGdnQkvGeneric candidate.

Reference: deployed QKV v2 for HK=8 when --reference-vendor is supplied;
FLA l2norm plus BSND copies for HK=16. Both shapes also get a CPU FP32
normalization oracle. This script only loads libraries and runs local NPU ops.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time

TORCH = None
ACLNN = None
LIBRARIES = []
DIM = 128


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--vendor", type=Path, required=True)
    p.add_argument("--reference-vendor", default=str(
        Path(__file__).resolve().parents[1] / "Clef-27B/src/ascendc/vendors/customize"),
        help="Deployed TP2 QKV v2; empty string uses FLA for both TP sizes")
    p.add_argument("--key-heads", default="8,16")
    p.add_argument("--lengths", default="1,15,16,17,31,32,33,63,64,65,307,967,1595")
    p.add_argument("--tiles", default="16,32,64")
    p.add_argument("--seeds", default="20261117,101")
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--eps", type=float, default=1e-6)
    p.add_argument("--max-abs", type=float, default=0.00390625,
                   help="Local Q/K screening tolerance; full service quality is evaluated separately")
    p.add_argument("--output", type=Path, required=True)
    return p


def setup(vendor, reference_vendor, device):
    global TORCH, ACLNN
    paths = [str(vendor)]
    if reference_vendor:
        paths.append(str(Path(reference_vendor).resolve()))
    for prefix in sys.path:
        fla = Path(prefix) / "fla_npu/opp/vendors/fla_npu_transformer"
        if fla.is_dir():
            paths.append(str(fla))
    if os.environ.get("ASCEND_CUSTOM_OPP_PATH"):
        paths.append(os.environ["ASCEND_CUSTOM_OPP_PATH"])
    os.environ["ASCEND_CUSTOM_OPP_PATH"] = os.pathsep.join(dict.fromkeys(paths))
    import torch
    import torch_npu  # noqa: F401
    from fla_npu.ops.ascendc import _runtime
    TORCH, ACLNN = torch, _runtime
    torch.set_num_threads(1)
    torch.npu.config.allow_internal_format = True
    torch.npu.set_device(device)
    runtime = _runtime.runtime()
    for directory in [vendor] + ([Path(reference_vendor)] if reference_vendor else []):
        lib = ctypes.CDLL(str(directory / "op_api/lib/libcust_opapi.so"),
                          mode=getattr(os, "RTLD_LOCAL", 0)
                          | getattr(os, "RTLD_NOW", 0)
                          | getattr(os, "RTLD_NODELETE", 0))
        LIBRARIES.append(lib)
        runtime._libraries.insert(0, lib)


def allocate(tokens, hk, device):
    torch = TORCH
    q = torch.empty((1, tokens, hk, DIM), dtype=torch.bfloat16, device=device)
    return q, torch.empty_like(q), torch.empty(
        (1, tokens, 3 * hk, DIM), dtype=torch.bfloat16, device=device)


def fused(x, outputs, eps, tile):
    q, k, v = outputs
    ACLNN.call_aclnn(
        "aclnnClefGdnQkvGeneric",
        lambda ctx: [ctx.tensor(x, "x"), ctypes.c_double(eps), ctypes.c_int64(tile),
                     ctx.tensor(q, "qOut"), ctx.tensor(k, "kOut"), ctx.tensor(v, "vOut")],
        outputs,
        get_workspace_argtypes=[ctypes.c_void_p, ctypes.c_double, ctypes.c_int64,
                                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                                ctypes.POINTER(ctypes.c_uint64),
                                ctypes.POINTER(ctypes.c_void_p)],
    )
    return outputs


def reference(x, hk, outputs, eps, use_vendor):
    if use_vendor:
        q, k, v = outputs
        ACLNN.call_aclnn(
            "aclnnClefGdnQkvNorm",
            lambda ctx: [ctx.tensor(x, "x"), ctypes.c_double(eps),
                         ctx.tensor(q, "qOut"), ctx.tensor(k, "kOut"), ctx.tensor(v, "vOut")],
            outputs,
            get_workspace_argtypes=[ctypes.c_void_p, ctypes.c_double,
                                    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                                    ctypes.POINTER(ctypes.c_uint64),
                                    ctypes.POINTER(ctypes.c_void_p)],
        )
        return outputs
    from vllm.third_party.flash_linear_attention.ops.l2norm import l2norm_fwd
    q = l2norm_fwd(x[:hk].transpose(0, 1).unsqueeze(0).contiguous(), eps=eps)
    k = l2norm_fwd(x[hk:2 * hk].transpose(0, 1).unsqueeze(0).contiguous(), eps=eps)
    v = x[2 * hk:].transpose(0, 1).unsqueeze(0).contiguous()
    return q, k, v


def cpu_reference(x, hk, eps):
    def norm(y):
        y = y.transpose(0, 1).unsqueeze(0).float()
        return (y * (y.square().sum(-1, keepdim=True) + eps).rsqrt()).to(TORCH.bfloat16)
    return (norm(x[:hk]), norm(x[hk:2 * hk]),
            x[2 * hk:].transpose(0, 1).unsqueeze(0).contiguous())


def difference(actual, expected):
    torch = TORCH
    actual, expected = actual.detach().float().cpu(), expected.detach().float().cpu()
    finite_a, finite_b = bool(torch.isfinite(actual).all()), bool(torch.isfinite(expected).all())
    delta = (actual - expected).abs()
    return {"finite_actual": finite_a, "finite_reference": finite_b,
            "nan_actual": int(torch.isnan(actual).sum()), "inf_actual": int(torch.isinf(actual).sum()),
            "nan_reference": int(torch.isnan(expected).sum()), "inf_reference": int(torch.isinf(expected).sum()),
            "bit_equal": bool(torch.equal(actual, expected)),
            "max_abs": float(delta.max()) if finite_a and finite_b else None,
            "mean_abs": float(delta.mean()) if finite_a and finite_b else None,
            "rmse": float(delta.square().mean().sqrt()) if finite_a and finite_b else None}


def benchmark(calls, warmup, iters):
    torch = TORCH
    names = list(calls)
    for _ in range(warmup):
        for call in calls.values():
            call()
    torch.npu.synchronize()
    events, host = {name: [] for name in names}, {name: [] for name in names}
    for i in range(iters):
        order = names if i % 2 == 0 else list(reversed(names))
        for name in order:
            begin = torch.npu.Event(enable_timing=True)
            finish = torch.npu.Event(enable_timing=True)
            started = time.perf_counter()
            begin.record(); calls[name](); finish.record()
            host[name].append((time.perf_counter() - started) * 1000)
            events[name].append((begin, finish))
    torch.npu.synchronize()
    result = {}
    for name in names:
        durations = [a.elapsed_time(b) for a, b in events[name]]
        result[name] = {"device_event_ms_median": statistics.median(durations),
                        "device_event_ms_mean": statistics.mean(durations),
                        "device_event_ms_min": min(durations),
                        "host_enqueue_ms_median": statistics.median(host[name]),
                        "device_event_ms": durations}
    return result


def ints(value):
    return [int(item) for item in value.split(",") if item.strip()]


def main(argv=None):
    args = parser().parse_args(argv)
    key_heads, lengths, tiles, seeds = map(ints, (args.key_heads, args.lengths, args.tiles, args.seeds))
    if (not key_heads or not set(key_heads) <= {8, 16} or not lengths or min(lengths) < 1
            or not tiles or not set(tiles) <= {16, 32, 64} or not seeds
            or args.warmup < 0 or args.iters < 1 or not math.isfinite(args.eps) or args.eps <= 0
            or not math.isfinite(args.max_abs) or args.max_abs < 0):
        raise SystemExit("Invalid probe dimensions, timing count, eps or tolerance")
    vendor = args.vendor.expanduser().resolve()
    reference_vendor = str(Path(args.reference_vendor).expanduser().resolve()) if args.reference_vendor else ""
    library = vendor / "op_api/lib/libcust_opapi.so"
    report = {"created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "vendor": str(vendor), "reference_vendor": reference_vendor,
              "library_sha256": hashlib.sha256(library.read_bytes()).hexdigest(),
              "device": args.device, "eps": args.eps,
              "precision": "bfloat16 inputs/outputs, float32 normalization",
              "screening_max_abs": args.max_abs, "warmup": args.warmup,
              "iterations": args.iters, "cases": []}
    setup(vendor, reference_vendor, args.device)
    torch = TORCH
    device = f"npu:{args.device}"
    for hk in key_heads:
        for tokens in lengths:
            for seed in seeds:
                generator = torch.Generator(device="cpu").manual_seed(seed)
                x_cpu = torch.randn(5 * hk, tokens, DIM, generator=generator).to(torch.bfloat16)
                # Zero, unit and sparse rows expose epsilon/tail/row-stride errors.
                x_cpu[0, 0] = 0
                x_cpu[hk, 0] = 1
                if tokens > 1:
                    x_cpu[0, 1] = 0
                    x_cpu[0, 1, 73] = 1
                x = x_cpu.to(device)
                ref_outputs = allocate(tokens, hk, device)
                use_vendor = hk == 8 and bool(reference_vendor)
                ref = reference(x, hk, ref_outputs, args.eps, use_vendor)
                cpu = cpu_reference(x_cpu, hk, args.eps)
                outputs = {tile: allocate(tokens, hk, device) for tile in tiles}
                for tile in tiles:
                    # Sentinel fill also checks that every logical output is written.
                    for tensor in outputs[tile]:
                        tensor.fill_(float("nan"))
                    fused(x, outputs[tile], args.eps, tile)
                torch.npu.synchronize()
                calls = {"reference": lambda: reference(x, hk, ref_outputs, args.eps, use_vendor)}
                calls.update({f"tile{tile}": lambda tile=tile: fused(x, outputs[tile], args.eps, tile) for tile in tiles})
                timings = benchmark(calls, args.warmup, args.iters)
                # Check the last tightly queued invocation too, exercising
                # buffer reuse and MTE ordering rather than a single launch.
                checks = {}
                for tile in tiles:
                    checks[tile] = {"vs_reference": {}, "vs_cpu_fp32": {}}
                    for part, actual, expected, oracle in zip(("q", "k", "v"), outputs[tile], ref, cpu):
                        checks[tile]["vs_reference"][part] = difference(actual, expected)
                        checks[tile]["vs_cpu_fp32"][part] = difference(actual, oracle)
                for tile in tiles:
                    comparison = checks[tile]
                    passed = all(d["finite_actual"] and d["finite_reference"]
                                 and d["max_abs"] <= args.max_abs
                                 for group in comparison.values() for d in group.values())
                    passed = passed and comparison["vs_reference"]["v"]["bit_equal"]
                    record = {"key_heads": hk, "value_heads": 3 * hk, "tokens": tokens,
                              "tile_tokens": tile, "seed": seed,
                              "reference": "deployed_qkv_v2" if use_vendor else "fla_l2norm_bs_nd",
                              "comparison": comparison, "passed_local_screen": passed,
                              "timing": {"reference": timings["reference"], "candidate": timings[f"tile{tile}"]}}
                    record["speedup_median"] = (timings["reference"]["device_event_ms_median"]
                                                 / timings[f"tile{tile}"]["device_event_ms_median"])
                    report["cases"].append(record)
                    print(json.dumps({k: v for k, v in record.items() if k not in ("comparison", "timing")}), flush=True)
    report["failed_cases"] = sum(not c["passed_local_screen"] for c in report["cases"])
    report["passed_local_screen"] = report["failed_cases"] == 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n")
    print(json.dumps({"cases": len(report["cases"]), "failed_cases": report["failed_cases"],
                      "output": str(args.output)}))
    return 0 if report["passed_local_screen"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
