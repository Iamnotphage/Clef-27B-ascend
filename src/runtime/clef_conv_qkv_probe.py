#!/usr/bin/env python3
"""Validate and time the single-sequence BF16 Conv+QKV candidate on A3.

The candidate consumes contiguous [T,5120] and packed [4,5120] weights.
Its reference is causal_conv1d_fn followed by the deployed QKV norm op
(or FLA l2norm when --reference-vendor is empty).  CPU FP32 accumulation
and a float64 oracle distinguish convolution ordering from L2 differences.
No service file, installed package, or model weight is modified.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import statistics
import time

HEADS, HK, HV, DIM, CHANNELS = 40, 8, 24, 128, 5120
TORCH = None
ACLNN = None
LIBRARIES = []


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--vendor", required=True, type=Path,
                   help="Extracted candidate vendors/clef_conv_qkv directory")
    p.add_argument("--reference-vendor", default=str(
        Path(__file__).resolve().parents[2] / "src/ascendc/vendors/customize"),
        help="Deployed QKV norm vendor; empty string selects FLA l2norm")
    p.add_argument("--lengths", default="1,2,3,4,31,32,33,307,967,1595")
    p.add_argument("--seeds", default="20261117,101,2027")
    p.add_argument("--tiles", default="32", help="16,32,64 or a comma-separated sweep")
    p.add_argument("--input-channels", default="5120,8192",
                   help="Contiguous QKV or packed QKVZ input rows")
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--eps", type=float, default=1e-6)
    p.add_argument("--max-abs", type=float, default=0.03125,
                   help="Local screening threshold, independent of competition quality gates")
    p.add_argument("--cpu-reference", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--save-outputs", type=Path, default=None,
                   help="Save each fused q/k/v (CPU tensors) for a cross-vendor bitwise diff")
    p.add_argument("--output", required=True, type=Path)
    return p


def setup(vendor: Path, reference_vendor: str, device: int):
    global TORCH, ACLNN
    # Custom op paths must be visible before torch_npu's first import.
    paths = [str(vendor)]
    if reference_vendor:
        paths.append(str(Path(reference_vendor).resolve()))
    import sys
    for prefix in sys.path:
        fla_vendor = Path(prefix) / "fla_npu/opp/vendors/fla_npu_transformer"
        if fla_vendor.is_dir():
            paths.append(str(fla_vendor))
    if os.environ.get("ASCEND_CUSTOM_OPP_PATH"):
        paths.append(os.environ["ASCEND_CUSTOM_OPP_PATH"])
    os.environ["ASCEND_CUSTOM_OPP_PATH"] = os.pathsep.join(dict.fromkeys(paths))
    import torch
    import torch_npu  # noqa: F401
    from fla_npu.ops.ascendc import _runtime
    TORCH, ACLNN = torch, _runtime
    torch.set_num_threads(1)
    torch.npu.set_device(device)
    runtime = _runtime.runtime()
    for directory in [vendor] + ([Path(reference_vendor)] if reference_vendor else []):
        library = ctypes.CDLL(str(directory / "op_api/lib/libcust_opapi.so"),
                              mode=getattr(os, "RTLD_LOCAL", 0)
                              | getattr(os, "RTLD_NOW", 0)
                              | getattr(os, "RTLD_NODELETE", 0))
        LIBRARIES.append(library)
        runtime._libraries.insert(0, library)
    return runtime


def candidate(x, weight, outputs, eps, tile):
    q, k, v = outputs
    ACLNN.call_aclnn(
        "aclnnClefGdnConvQkv",
        lambda ctx: [ctx.tensor(x, "x"), ctx.tensor(weight, "weight"),
                     ctypes.c_double(eps), ctypes.c_int64(tile),
                     ctx.tensor(q, "qOut"), ctx.tensor(k, "kOut"), ctx.tensor(v, "vOut")],
        outputs,
        get_workspace_argtypes=[ctypes.c_void_p, ctypes.c_void_p,
                                ctypes.c_double, ctypes.c_int64,
                                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                                ctypes.POINTER(ctypes.c_uint64),
                                ctypes.POINTER(ctypes.c_void_p)],
    )
    return outputs


def norm_reference(conv, outputs, eps, use_vendor):
    if use_vendor:
        q, k, v = outputs
        ACLNN.call_aclnn(
            "aclnnClefGdnQkvNorm",
            lambda ctx: [ctx.tensor(conv, "x"), ctypes.c_double(eps),
                         ctx.tensor(q, "qOut"), ctx.tensor(k, "kOut"), ctx.tensor(v, "vOut")],
            outputs,
            get_workspace_argtypes=[ctypes.c_void_p, ctypes.c_double,
                                    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                                    ctypes.POINTER(ctypes.c_uint64),
                                    ctypes.POINTER(ctypes.c_void_p)],
        )
        return outputs
    from vllm.third_party.flash_linear_attention.ops.l2norm import l2norm_fwd
    q = l2norm_fwd(conv[:HK].unsqueeze(0)).transpose(1, 2).contiguous()
    k = l2norm_fwd(conv[HK:2 * HK].unsqueeze(0)).transpose(1, 2).contiguous()
    v = conv[2 * HK:].transpose(0, 1).unsqueeze(0).contiguous()
    return q, k, v


def allocate(tokens, device):
    torch = TORCH
    return (torch.empty((1, tokens, HK, DIM), dtype=torch.bfloat16, device=device),
            torch.empty((1, tokens, HK, DIM), dtype=torch.bfloat16, device=device),
            torch.empty((1, tokens, HV, DIM), dtype=torch.bfloat16, device=device))


def diff(a, b):
    torch = TORCH
    a, b = a.detach().float(), b.detach().float()
    finite_a, finite_b = bool(torch.isfinite(a).all()), bool(torch.isfinite(b).all())
    delta = (a - b).abs()
    return {"finite_a": finite_a, "finite_b": finite_b,
            "nan_a": int(torch.isnan(a).sum()), "inf_a": int(torch.isinf(a).sum()),
            "bit_equal": bool(torch.equal(a, b)),
            "max_abs": float(delta.max()) if finite_a and finite_b else None,
            "mean_abs": float(delta.mean()) if finite_a and finite_b else None,
            "rmse": float(delta.square().mean().sqrt()) if finite_a and finite_b else None}


def cpu_convolution(x, weight, dtype):
    torch = TORCH
    x, weight = x.to(dtype), weight.to(dtype)
    # The four taps are accumulated oldest to newest.  This explicitly
    # exposes separate multiply/add rounding; the NPU uses MulAddDst.
    padded = torch.nn.functional.pad(x, (0, 0, 3, 0))
    acc = padded[:x.shape[0]] * weight[0]
    for j in range(1, 4):
        acc = acc + padded[j:j + x.shape[0]] * weight[j]
    return torch.nn.functional.silu(acc).to(torch.bfloat16)


def cpu_qkv(conv, eps):
    torch = TORCH
    values = conv.reshape(conv.shape[0], HEADS, DIM).float()
    def norm(a):
        return (a * (a.square().sum(-1, keepdim=True) + eps).rsqrt()).to(torch.bfloat16)
    return (norm(values[:, :HK]).unsqueeze(0),
            norm(values[:, HK:2 * HK]).unsqueeze(0),
            values[:, 2 * HK:].to(torch.bfloat16).unsqueeze(0))


def benchmark(reference, fused, warmup, iters):
    torch = TORCH
    for _ in range(warmup):
        reference(); fused()
    torch.npu.synchronize()
    events = {"reference": [], "fused": []}
    host = {"reference": [], "fused": []}
    for iteration in range(iters):
        order = [("reference", reference), ("fused", fused)]
        if iteration % 2:
            order.reverse()
        for name, call in order:
            start = torch.npu.Event(enable_timing=True)
            end = torch.npu.Event(enable_timing=True)
            t0 = time.perf_counter()
            start.record(); call(); end.record()
            host[name].append((time.perf_counter() - t0) * 1000)
            events[name].append((start, end))
    torch.npu.synchronize()
    result = {}
    for name, pairs in events.items():
        values = [float(a.elapsed_time(b)) for a, b in pairs]
        ordered = sorted(values)
        result[name] = {"event_ms_median": statistics.median(values),
                        "event_ms_min": min(values),
                        "event_ms_p95": ordered[min(len(ordered) - 1, math.ceil(.95 * len(ordered)) - 1)],
                        "host_enqueue_ms_median": statistics.median(host[name]),
                        "event_ms_samples": values}
    result["speedup"] = result["reference"]["event_ms_median"] / result["fused"]["event_ms_median"]
    return result


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    args = parser().parse_args()
    lengths = [int(x) for x in args.lengths.split(",")]
    seeds = [int(x) for x in args.seeds.split(",")]
    tiles = [int(x) for x in args.tiles.split(",")]
    if min(lengths) <= 0 or any(x not in (16, 32, 64) for x in tiles) or args.iters <= 0:
        raise SystemExit("Positive lengths/iters and tile sizes 16/32/64 are required")
    if args.warmup < 0 or args.eps <= 0:
        raise SystemExit("Warmup must be nonnegative and eps positive")
    vendor = args.vendor.resolve()
    setup(vendor, args.reference_vendor, args.device)
    torch = TORCH
    from fla_npu.ops.ascendc import causal_conv1d_fn
    device = f"npu:{args.device}"
    report = {"created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "device": device, "vendor": str(vendor), "reference_vendor": args.reference_vendor,
              "candidate_library_sha256": digest(vendor / "op_api/lib/libcust_opapi.so"),
              "config": {"lengths": lengths, "seeds": seeds, "tiles": tiles,
                         "warmup": args.warmup, "iters": args.iters, "eps": args.eps,
                         "max_abs_screening": args.max_abs},
              "torch_version": torch.__version__, "rows": []}
    failures = 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        for tokens in lengths:
            for seed, input_channels in itertools.product(
                    seeds, [int(v) for v in args.input_channels.split(",")]):
                if input_channels not in (5120, 8192):
                    raise ValueError("input channels must be 5120 or 8192")
                generator = torch.Generator().manual_seed(seed + tokens)
                packed_cpu = (torch.randn((tokens, input_channels), generator=generator) * .5).to(torch.bfloat16)
                x_cpu = packed_cpu[:, :CHANNELS].contiguous()
                w_cpu = (torch.randn((4, CHANNELS), generator=generator) * .2).to(torch.bfloat16)
                packed, weight = packed_cpu.to(device), w_cpu.to(device)
                x = packed[:, :CHANNELS]
                cu = torch.tensor([0, tokens], dtype=torch.int32, device=device)
                refs = allocate(tokens, device)
                def convolution():
                    return causal_conv1d_fn(x, weight, None, conv_states=None,
                                            query_start_loc=cu, cache_indices=None,
                                            has_initial_state=None, activation="silu",
                                            pad_slot_id=-1, null_block_id=0, head_num=HEADS)
                def reference():
                    return norm_reference(convolution(), refs, args.eps, bool(args.reference_vendor))
                baseline = tuple(value.clone() for value in reference())
                cpu_info = None
                if args.cpu_reference:
                    conv_npu = convolution().transpose(0, 1).reshape(tokens, CHANNELS).cpu()
                    conv32 = cpu_convolution(x_cpu, w_cpu, torch.float32)
                    conv64 = cpu_convolution(x_cpu, w_cpu, torch.float64)
                    cpu_outputs = cpu_qkv(conv32, args.eps)
                    cpu_info = {"npu_conv_vs_serial_fp32": diff(conv_npu, conv32),
                                "npu_conv_vs_serial_fp64": diff(conv_npu, conv64),
                                "serial_fp32_vs_fp64": diff(conv32, conv64)}
                for tile in tiles:
                    outputs = allocate(tokens, device)
                    def fused():
                        return candidate(packed, weight, outputs, args.eps, tile)
                    for tensor in outputs:
                        tensor.fill_(float("nan"))
                    fused(); torch.npu.synchronize()
                    comparison = {name: diff(a, b) for name, a, b in zip(("q", "k", "v"), outputs, baseline)}
                    valid = all(d["finite_a"] and d["finite_b"] and d["max_abs"] <= args.max_abs
                                for d in comparison.values())
                    row = {"tokens": tokens, "seed": seed, "tile_tokens": tile,
                           "input_channels": input_channels,
                           "candidate_vs_reference": comparison, "local_screen_pass": valid}
                    if cpu_info is not None:
                        row["cpu_conv_reference"] = cpu_info
                        row["candidate_vs_cpu_fp32"] = {
                            name: diff(a.cpu(), b) for name, a, b in zip(("q", "k", "v"), outputs, cpu_outputs)}
                    if args.save_outputs is not None:
                        args.save_outputs.mkdir(parents=True, exist_ok=True)
                        torch.save(
                            {name: value.cpu() for name, value in zip(("q", "k", "v"), outputs)},
                            args.save_outputs / f"{tokens}-{seed}-{tile}-{input_channels}.pt")
                    row["timing"] = benchmark(reference, fused, args.warmup, args.iters)
                    failures += int(not valid)
                    report["rows"].append(row)
                    report["local_screen_failures"] = failures
                    # Preserve completed cases if a later candidate faults.
                    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
                    print(json.dumps({"tokens": tokens, "seed": seed, "tile": tile,
                                      "screen_pass": valid,
                                      "max_abs": {k: d["max_abs"] for k, d in comparison.items()},
                                      "speedup": row["timing"]["speedup"]}), flush=True)
    print(json.dumps({"output": str(args.output), "rows": len(report["rows"]), "failures": failures}))
    return int(failures != 0)


if __name__ == "__main__":
    raise SystemExit(main())
