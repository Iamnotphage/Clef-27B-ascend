#!/usr/bin/env python3
"""Sweep GDN chunk size on 800+ token lengths for the decision-only path."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import itertools
import json
import os
from pathlib import Path
import statistics
import sys
import time

import torch

sys.path.insert(0, "/workspace/yangchen/work/src/vllm-ascend")
sys.path.insert(0, "/workspace/yangchen/work/src/vllm")

from fla_npu.ops.ascendc import chunk_gated_delta_rule_fwd  # noqa: E402
from vllm_ascend.ops.triton.fla.utils import prepare_chunk_indices  # noqa: E402

HK, HV, D = 8, 24, 128
SCALE = D ** -0.5


def bench(fn, warmup: int, iters: int) -> dict:
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    samples, device_samples = [], []
    for _ in range(iters):
        begin = torch.npu.Event(enable_timing=True)
        end = torch.npu.Event(enable_timing=True)
        started = time.perf_counter()
        begin.record()
        fn()
        end.record()
        torch.npu.synchronize()
        samples.append((time.perf_counter() - started) * 1000.0)
        device_samples.append(begin.elapsed_time(end))
    return {"ms": statistics.median(samples),
            "device_ms": statistics.median(device_samples),
            "wall_samples_ms": samples, "device_samples_ms": device_samples}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--tokens", default="145,307,967,1084,1200,1258,1595,10294")
    parser.add_argument("--seq-lens", default="",
                        help="Semicolon-separated varlen layouts, e.g. "
                             "'300x8;450x16;967,967,967,967' or "
                             "'200,250,300,350,400,500,800,1100'. "
                             "Empty keeps the single-sequence T sweep.")
    parser.add_argument("--chunks", default="64,128")
    parser.add_argument("--sequence-modes", default="varlen,dense",
                        help="Single-sequence metadata paths to compare")
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.warmup < 0 or args.iters < 1:
        parser.error("warmup must be non-negative and iters must be positive")

    torch.npu.set_device(args.device)
    chunks = [int(value) for value in args.chunks.split(",")]
    modes = args.sequence_modes.split(",")
    if not modes or any(mode not in ("varlen", "dense") for mode in modes):
        parser.error("sequence modes must be varlen and/or dense")
    if args.seq_lens.strip() and "dense" in modes:
        # Packed multi-sequence inputs only have the varlen metadata path.
        modes = ["varlen"] if "varlen" in modes else parser.error(
            "dense mode is not defined for --seq-lens")
    rows = []
    if args.seq_lens.strip():
        layouts = []
        for group in args.seq_lens.split(";"):
            group = group.strip()
            if not group:
                continue
            if "x" in group:
                count, length = group.split("x")
                lens = [int(length)] * int(count)
            else:
                lens = [int(value) for value in group.split(",") if value.strip()]
            label = f"{len(lens)}x{lens[0]}" if len(set(lens)) == 1 else "mixed"
            layouts.append((label, lens))
        if not layouts:
            parser.error("--seq-lens did not contain a layout")
    else:
        layouts = [(str(T), [T]) for T in [int(value) for value in args.tokens.split(",")]]
    for label, lens in layouts:
        T = sum(lens)
        cu_list = [0]
        for length in lens:
            cu_list.append(cu_list[-1] + length)
        generator = torch.Generator().manual_seed(T)
        # Match the service's L2-normalized Q/K contract. Unnormalized random
        # K destabilizes the triangular solve and makes chunk128 a false NaN.
        q, k = [torch.randn(1, T, HK, D, generator=generator) for _ in range(2)]
        q, k = [(x * torch.rsqrt(x.square().sum(-1, keepdim=True) + 1e-6))
                .to("npu", torch.bfloat16) for x in (q, k)]
        v = torch.randn(1, T, HV, D, generator=generator).to("npu", torch.bfloat16)
        g = (-0.5 * torch.rand(1, T, HV, generator=generator)).to("npu", torch.float32) - 1e-4
        beta = torch.rand(1, T, HV, generator=generator).to("npu", torch.bfloat16)
        cu = tuple(cu_list)
        def make_call(chunk, mode="varlen"):
            indices = prepare_chunk_indices(torch.tensor(cu_list, dtype=torch.int32), chunk)
            ci = tuple(indices.to(torch.int64).reshape(-1).tolist())
            def call():
                return chunk_gated_delta_rule_fwd(
                    q, k, v, g, beta,
                    initial_state=None,
                    output_final_state=False,
                    chunk_size=chunk,
                    cu_seqlens=cu if mode == "varlen" else None,
                    chunk_indices=ci if mode == "varlen" else None,
                    scale=SCALE,
                    layout="BSND",
                    use_exp2=False,
                    use_qk_l2norm_in_kernel=False,
                    allow_neg_eigval=False,
                    # Decision-prefill production omits backward-only A/g.
                    disable_recompute=False,
                    state_v_first=False,
                )
            return call
        with torch.inference_mode():
            reference = make_call(64)()[0]
            if not bool(torch.isfinite(reference).all()):
                raise RuntimeError(f"chunk64 reference is nonfinite at T={T}")
        for mode, chunk in itertools.product(modes, chunks):
            try:
                call = make_call(chunk, mode)
                with torch.no_grad():
                    output = call()
                out = output[0]
                finite = bool(torch.isfinite(out).all())
                equality = {"reference_chunk": 64, "finite": finite,
                            "bit_equal": bool(torch.equal(reference, out))}
                if finite:
                    delta = (reference.float() - out.float()).abs()
                    equality.update(max_abs=float(delta.max()),
                                    rmse=float(delta.square().mean().sqrt()))
                row = {"tokens": T, "layout": label, "sequences": len(lens),
                       "seq_lens": lens, "chunk": chunk, "sequence_mode": mode,
                       "equality": equality}
                if finite:
                    row.update(bench(call, args.warmup, args.iters))
                rows.append(row)
                print(json.dumps({k: v for k, v in row.items()
                                  if not k.endswith("samples_ms")}, allow_nan=False), flush=True)
            except Exception as error:  # noqa: BLE001 - report unsupported chunk sizes
                row = {"tokens": T, "layout": label, "sequences": len(lens),
                       "seq_lens": lens, "chunk": chunk, "sequence_mode": mode,
                       "error": f"{type(error).__name__}: {error}"}
                rows.append(row)
                print(json.dumps(row), flush=True)
    with open(args.output, "w") as handle:
        json.dump({"schema_version": 3, "rows": rows,
                   "created_at": datetime.now(timezone.utc).isoformat(),
                   "device": args.device, "warmup": args.warmup, "iters": args.iters,
                   "qk_normalized": True, "reference_chunk": 64,
                   "varlen_layouts": args.seq_lens or "single-sequence",
                   "disable_recompute": False,
                   "input_kind": "synthetic service-contract; validate full-model quality separately",
                   "host_load": os.getloadavg(),
                   "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
                  handle, indent=1, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"output": args.output}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
