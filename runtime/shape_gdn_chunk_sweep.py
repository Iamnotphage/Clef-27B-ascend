#!/usr/bin/env python3
"""Sweep GDN chunk size on 800+ token lengths for the decision-only path."""

from __future__ import annotations

import argparse
import json
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


def bench(fn, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    samples = []
    for _ in range(iters):
        started = time.perf_counter()
        fn()
        torch.npu.synchronize()
        samples.append((time.perf_counter() - started) * 1000.0)
    return statistics.median(samples)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--tokens", default="967,1084,1200,1258,1595")
    parser.add_argument("--chunks", default="32,64,128")
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    torch.npu.set_device(args.device)
    chunks = [int(value) for value in args.chunks.split(",")]
    rows = []
    for T in [int(value) for value in args.tokens.split(",")]:
        generator = torch.Generator().manual_seed(T)
        q = torch.randn(1, T, HK, D, generator=generator).to("npu", torch.bfloat16)
        k = torch.randn(1, T, HK, D, generator=generator).to("npu", torch.bfloat16)
        v = torch.randn(1, T, HV, D, generator=generator).to("npu", torch.bfloat16)
        g = (-0.5 * torch.rand(1, T, HV, generator=generator)).to("npu", torch.float32) - 1e-4
        beta = torch.rand(1, T, HV, generator=generator).to("npu", torch.bfloat16)
        cu = tuple([0, T])
        reference = None
        for chunk in chunks:
            try:
                indices = prepare_chunk_indices(torch.tensor([0, T], dtype=torch.int32), chunk)
                ci = tuple(indices.to(torch.int64).reshape(-1).tolist())

                def call(chunk=chunk, ci=ci):
                    return chunk_gated_delta_rule_fwd(
                        q, k, v, g, beta,
                        initial_state=None,
                        output_final_state=False,
                        chunk_size=chunk,
                        cu_seqlens=cu,
                        chunk_indices=ci,
                        scale=SCALE,
                        layout="BSND",
                        use_exp2=False,
                        use_qk_l2norm_in_kernel=False,
                        allow_neg_eigval=False,
                        disable_recompute=True,
                        state_v_first=False,
                    )

                with torch.no_grad():
                    output = call()
                ms = bench(call, args.warmup, args.iters)
                out = output[0]
                if reference is None:
                    reference = out
                    equality = {"reference": True}
                else:
                    delta = (reference.float() - out.float()).abs()
                    equality = {
                        "bit_equal": bool(torch.equal(reference, out)),
                        "max_abs": float(delta.max()),
                    }
                row = {"tokens": T, "chunk": chunk, "ms": ms, "equality": equality}
                rows.append(row)
                print(json.dumps({**row, "ms": round(ms, 4)}), flush=True)
            except Exception as error:  # noqa: BLE001 - report unsupported chunk sizes
                row = {"tokens": T, "chunk": chunk, "error": f"{type(error).__name__}: {error}"}
                rows.append(row)
                print(json.dumps(row), flush=True)
    with open(args.output, "w") as handle:
        json.dump({"rows": rows}, handle, indent=1)
        handle.write("\n")
    print(json.dumps({"output": args.output}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
