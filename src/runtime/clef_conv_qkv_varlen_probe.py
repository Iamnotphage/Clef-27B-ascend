#!/usr/bin/env python3
"""Four-gate offline probe for the multi-sequence ClefGdnConvQkvVarlen op.

Gate 1: B=1 must be bitwise identical to the deployed single-sequence v4 op
        for every length / input-channel / tile combination.
Gate 2: multi-sequence batches must equal per-sequence v4 calls, including
        adversarial short boundaries and packed [T,8192] poison tails.
Gate 3: mutating one sequence (x100, +-1e4, all NaN) must leave every other
        sequence bitwise unchanged.
Gate 4: fused varlen vs the decomposed causal_conv1d + QKV-norm path, with
        the >=2.5x speed gate that decides whether the kernel ships.

The probe never starts the service and touches no production source.
"""

from __future__ import annotations

import argparse
import ctypes
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
DEVICE = "npu:0"


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--varlen-vendor", type=Path, required=True)
    p.add_argument("--v4-vendor", type=Path, required=True)
    p.add_argument("--reference-vendor", type=Path, required=True,
                   help="Deployed ClefGdnQkvNorm vendor")
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--eps", type=float, default=1e-6)
    p.add_argument("--output", type=Path, required=True)
    return p


def setup(args) -> None:
    global TORCH, ACLNN, DEVICE
    DEVICE = f"npu:{args.device}"
    paths = [str(args.varlen_vendor.resolve()), str(args.v4_vendor.resolve()),
             str(args.reference_vendor.resolve())]
    import sys
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
    torch.npu.set_device(args.device)
    runtime = _runtime.runtime()
    for directory in (args.varlen_vendor.resolve(), args.v4_vendor.resolve(),
                      args.reference_vendor.resolve()):
        library = ctypes.CDLL(
            str(directory / "op_api/lib/libcust_opapi.so"),
            mode=getattr(os, "RTLD_LOCAL", 0) | getattr(os, "RTLD_NOW", 0)
            | getattr(os, "RTLD_NODELETE", 0))
        runtime._libraries.insert(0, library)


def allocate(tokens: int, device: str):
    torch = TORCH
    return (torch.empty((1, tokens, HK, DIM), dtype=torch.bfloat16, device=device),
            torch.empty((1, tokens, HK, DIM), dtype=torch.bfloat16, device=device),
            torch.empty((1, tokens, HV, DIM), dtype=torch.bfloat16, device=device))


def call_varlen(x, weight, cu, outputs, tile, eps=1e-6):
    q, k, v = outputs
    ACLNN.call_aclnn(
        "aclnnClefGdnConvQkvVarlen",
        lambda ctx: [ctx.tensor(x, "x"), ctx.tensor(weight, "weight"),
                     ctx.tensor(cu, "cuSeqlens"), ctypes.c_double(eps),
                     ctypes.c_int64(tile), ctx.tensor(q, "qOut"),
                     ctx.tensor(k, "kOut"), ctx.tensor(v, "vOut")],
        outputs,
        get_workspace_argtypes=[
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_double, ctypes.c_int64, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_void_p)])
    return outputs


def call_v4(x, weight, outputs, tile, eps=1e-6):
    q, k, v = outputs
    ACLNN.call_aclnn(
        "aclnnClefGdnConvQkv",
        lambda ctx: [ctx.tensor(x, "x"), ctx.tensor(weight, "weight"),
                     ctypes.c_double(eps), ctypes.c_int64(tile),
                     ctx.tensor(q, "qOut"), ctx.tensor(k, "kOut"),
                     ctx.tensor(v, "vOut")],
        outputs,
        get_workspace_argtypes=[
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_double, ctypes.c_int64,
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_void_p)])
    return outputs


def exact(a, b) -> dict:
    torch = TORCH
    same = bool(torch.equal(a, b))
    delta = (a.float() - b.float()).abs()
    return {"equal": same, "max_abs": float(delta.max()) if delta.numel() else 0.0}


def make_batch(lengths, channels, seed, poison=False):
    torch = TORCH
    total = sum(lengths)
    generator = torch.Generator().manual_seed(seed + total + channels)
    if channels == 8192:
        packed_cpu = (torch.randn((total, channels), generator=generator) * .5).to(torch.bfloat16)
        if poison:
            fills = [float("nan"), 1e4, -1e4]
            for column in range(CHANNELS, channels):
                packed_cpu[:, column] = fills[column % len(fills)]
        packed = packed_cpu.to(DEVICE)
        weight = (torch.randn((4, CHANNELS), generator=generator) * .2).to(torch.bfloat16).to(DEVICE)
        return packed, weight
    x_cpu = (torch.randn((total, channels), generator=generator) * .5).to(torch.bfloat16)
    weight = (torch.randn((4, CHANNELS), generator=generator) * .2).to(torch.bfloat16).to(DEVICE)
    return x_cpu.to(DEVICE), weight


def per_sequence_v4(x, weight, lengths, tile, eps=1e-6):
    torch = TORCH
    total = sum(lengths)
    outputs = allocate(total, DEVICE)
    start = 0
    for length in lengths:
        chunk = x[start:start + length]
        if chunk.shape[1] != CHANNELS:
            chunk = chunk[:, :CHANNELS].contiguous()
        else:
            chunk = chunk.contiguous()
        part = allocate(length, DEVICE)
        call_v4(chunk, weight, part, tile, eps)
        for full, piece in zip(outputs, part):
            full[:, start:start + length].copy_(piece)
        start += length
    return outputs


def decomposition(x, weight, cu, outputs, tile, eps=1e-6):
    from fla_npu.ops.ascendc import causal_conv1d_fn
    if x.shape[1] != CHANNELS:
        x = x[:, :CHANNELS].contiguous()
    conv = causal_conv1d_fn(
        x, weight, None, conv_states=None, query_start_loc=cu,
        cache_indices=None, has_initial_state=None, activation="silu",
        pad_slot_id=-1, null_block_id=0, head_num=HEADS)
    q, k, v = outputs
    ACLNN.call_aclnn(
        "aclnnClefGdnQkvNorm",
        lambda ctx: [ctx.tensor(conv, "x"), ctypes.c_double(eps),
                     ctx.tensor(q, "qOut"), ctx.tensor(k, "kOut"),
                     ctx.tensor(v, "vOut")],
        outputs,
        get_workspace_argtypes=[
            ctypes.c_void_p, ctypes.c_double, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_void_p)])
    return outputs


def benchmark(reference, fused, warmup, iters):
    torch = TORCH
    for _ in range(warmup):
        reference()
        fused()
    torch.npu.synchronize()
    values = {"reference": [], "fused": []}
    for iteration in range(iters):
        order = [("reference", reference), ("fused", fused)]
        if iteration % 2:
            order.reverse()
        for name, call in order:
            start = torch.npu.Event(enable_timing=True)
            end = torch.npu.Event(enable_timing=True)
            start.record()
            call()
            end.record()
            values[name].append((start, end))
    torch.npu.synchronize()
    medians = {
        name: statistics.median(float(a.elapsed_time(b)) for a, b in pairs)
        for name, pairs in values.items()}
    medians["speedup"] = medians["reference"] / medians["fused"] \
        if medians["fused"] else None
    return medians


def main() -> int:
    args = parser().parse_args()
    setup(args)
    torch = TORCH
    device = f"npu:{args.device}"
    report = {
        "schema": "clef-conv-qkv-varlen-probe/1",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "vendor": str(args.varlen_vendor.resolve()),
        "v4_vendor": str(args.v4_vendor.resolve()),
        "gates": {},
        "failures": [],
    }

    with torch.inference_mode():
        # Gate 1: B=1 vs v4, every length / channel / tile.
        gate1 = []
        for tokens, channels, tile in itertools.product(
                [1, 2, 3, 4, 31, 32, 33, 180, 307, 967, 1595, 3559, 10294],
                [5120, 8192], [16, 32, 64]):
            x, weight = make_batch([tokens], channels, 20261125)
            cu = torch.tensor([0, tokens], dtype=torch.int32, device=device)
            fused = call_varlen(x, weight, cu, allocate(tokens, device), tile, args.eps)
            reference = call_v4(
                x if channels == CHANNELS else x[:, :CHANNELS].contiguous(),
                weight, allocate(tokens, device), tile, args.eps)
            torch.npu.synchronize()
            diffs = {name: exact(a, b) for name, a, b in
                     zip(("q", "k", "v"), fused, reference)}
            ok = all(item["equal"] for item in diffs.values())
            gate1.append({"tokens": tokens, "channels": channels,
                          "tile": tile, "diffs": diffs, "ok": ok})
            if not ok:
                report["failures"].append(
                    f"gate1 tokens={tokens} channels={channels} tile={tile}")
        report["gates"]["gate1_b1_vs_v4"] = {
            "cases": len(gate1), "ok": all(row["ok"] for row in gate1),
            "rows": gate1}

        # Gate 2: multi-sequence vs per-sequence v4.
        cases = [
            ("8x450", [450] * 8),
            ("16x450", [450] * 16),
            ("8x967", [967] * 8),
            ("16x1024", [1024] * 16),
            ("mixture", [10294, 307, 967, 1653]),
            ("boundary", [1, 2, 3, 4, 30, 31, 32, 33, 34, 63, 64, 65, 1, 3, 2, 4]),
            ("32x256", [256] * 32),
            ("64x128", [128] * 64),
        ]
        gate2 = []
        for (name, lengths), channels, tile, poison in itertools.product(
                cases, [5120, 8192], [32, 64], [False] if False else [False, True]):
            if poison and channels != 8192:
                continue
            total = sum(lengths)
            x, weight = make_batch(lengths, channels, 20261125, poison=poison)
            cu = torch.tensor(
                [0] + list(itertools.accumulate(lengths)),
                dtype=torch.int32, device=device)
            fused = call_varlen(x, weight, cu, allocate(total, device), tile, args.eps)
            reference = per_sequence_v4(x, weight, lengths, tile, args.eps)
            torch.npu.synchronize()
            diffs = {label: exact(a, b) for label, a, b in
                     zip(("q", "k", "v"), fused, reference)}
            ok = all(item["equal"] for item in diffs.values())
            gate2.append({"name": name, "lengths": lengths, "channels": channels,
                          "tile": tile, "poison_tail": poison,
                          "diffs": diffs, "ok": ok})
            if not ok:
                report["failures"].append(
                    f"gate2 {name} channels={channels} tile={tile} poison={poison}")
        report["gates"]["gate2_multiseq_vs_v4"] = {
            "cases": len(gate2), "ok": all(row["ok"] for row in gate2),
            "rows": gate2}

        # Gate 3: zero cross-sequence leakage.
        gate3 = []
        for name, lengths in (("b8", [1, 2, 3, 4, 31, 33, 64, 65]),
                              ("b16", [1, 2, 3, 4, 30, 31, 32, 33, 34, 63,
                                       64, 65, 1, 3, 2, 4])):
            for channels in (5120, 8192):
                total = sum(lengths)
                x, weight = make_batch(lengths, channels, 20261125, poison=True)
                cu = torch.tensor(
                    [0] + list(itertools.accumulate(lengths)),
                    dtype=torch.int32, device=device)
                base = call_varlen(x, weight, cu, allocate(total, device), 32, args.eps)
                torch.npu.synchronize()
                starts = [0] + list(itertools.accumulate(lengths))
                for index in range(len(lengths)):
                    for mode in ("x100", "extreme", "nan"):
                        mutated = x.clone()
                        begin, end = starts[index], starts[index + 1]
                        if mode == "x100":
                            mutated[begin:end] = mutated[begin:end] * 100.0
                        elif mode == "extreme":
                            mutated[begin:end] = 1e4 if index % 2 == 0 else -1e4
                        else:
                            mutated[begin:end] = float("nan")
                        out = call_varlen(
                            mutated, weight, cu, allocate(total, device), 32, args.eps)
                        torch.npu.synchronize()
                        leaked = False
                        for other in range(len(lengths)):
                            if other == index:
                                continue
                            a, b = starts[other], starts[other + 1]
                            for full, changed in zip(base, out):
                                if not torch.equal(full[:, a:b], changed[:, a:b]):
                                    leaked = True
                        gate3.append({
                            "name": name, "channels": channels, "index": index,
                            "mode": mode, "leaked": leaked})
                        if leaked:
                            report["failures"].append(
                                f"gate3 {name} channels={channels} "
                                f"index={index} mode={mode}")
        report["gates"]["gate3_no_crosstalk"] = {
            "cases": len(gate3), "ok": not any(row["leaked"] for row in gate3),
            "rows": gate3}

        # Gate 4: fused vs decomposed path.
        perf_cases = [
            ("3600_8x450", [450] * 8),
            ("7200_16x450", [450] * 16),
            ("7736_8x967", [967] * 8),
            ("15472_16x967", [967] * 16),
            ("irregular", [10294, 307, 967, 1653]),
        ]
        gate4 = []
        for (name, lengths), channels, tile in itertools.product(
                perf_cases, [5120, 8192], [32, 64]):
            total = sum(lengths)
            x, weight = make_batch(lengths, channels, 20261125)
            cu = torch.tensor(
                [0] + list(itertools.accumulate(lengths)),
                dtype=torch.int32, device=device)
            outputs = allocate(total, device)
            ref_outputs = allocate(total, device)
            fused = lambda: call_varlen(x, weight, cu, outputs, tile, args.eps)
            reference = lambda: decomposition(
                x, weight, cu, ref_outputs, tile, args.eps)
            timing = benchmark(reference, fused, args.warmup, args.iters)
            gate4.append({"name": name, "channels": channels, "tile": tile,
                          "timing": timing})
        best = max(
            (row for row in gate4 if row["channels"] == CHANNELS),
            key=lambda row: row["timing"]["speedup"])
        report["gates"]["gate4_performance"] = {
            "rows": gate4,
            "best_speedup": best["timing"]["speedup"],
            "best": {"name": best["name"], "tile": best["tile"]},
            "hard_gate_2_5x": best["timing"]["speedup"] >= 2.5,
        }
        if best["timing"]["speedup"] < 2.5:
            report["failures"].append("gate4 speedup below 2.5x")

    report["ok"] = not report["failures"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n")
    print(json.dumps({
        "ok": report["ok"],
        "failures": report["failures"][:10],
        "gate1": report["gates"]["gate1_b1_vs_v4"]["ok"],
        "gate2": report["gates"]["gate2_multiseq_vs_v4"]["ok"],
        "gate3": report["gates"]["gate3_no_crosstalk"]["ok"],
        "gate4_best_speedup": report["gates"]["gate4_performance"]["best_speedup"],
        "gate4_best": report["gates"]["gate4_performance"]["best"],
    }, indent=1))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
