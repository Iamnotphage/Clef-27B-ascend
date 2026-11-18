#!/usr/bin/env python3
"""Time GDN prefill ops across token lengths to locate shape-dependent paths.

The final service shows a sharp latency step at 308 tokens (307 fast, 308+
slow on a freshly started service).  This probe isolates the conv1d, fused
chunk-GDN, and gating ops at the same lengths with the real TP2 head layout and
the decision-only call shapes, so a kernel-level cause can be separated from
service state.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time

import torch

sys.path.insert(0, "/workspace/yangchen/work/src/vllm-ascend")
sys.path.insert(0, "/workspace/yangchen/work/src/vllm")

from fla_npu.ops.ascendc import causal_conv1d_fn, chunk_gated_delta_rule_fwd  # noqa: E402
from vllm_ascend.ops.triton.fla.utils import prepare_chunk_indices  # noqa: E402

HK, HV, D = 8, 24, 128
DIM, WIDTH, CONV_HEADS = 5120, 4, 40
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
    parser.add_argument("--tokens", default="203,204,230,306,307,308,310,320,400,600,967")
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    torch.npu.set_device(args.device)
    torch.manual_seed(20261117)
    lengths = [int(value) for value in args.tokens.split(",")]
    rows = []
    for T in lengths:
        generator = torch.Generator().manual_seed(T)
        q = torch.randn(1, T, HK, D, generator=generator).to("npu", torch.bfloat16)
        k = torch.randn(1, T, HK, D, generator=generator).to("npu", torch.bfloat16)
        v = torch.randn(1, T, HV, D, generator=generator).to("npu", torch.bfloat16)
        g = (-0.5 * torch.rand(1, T, HV, generator=generator)).to("npu", torch.float32) - 1e-4
        beta = torch.rand(1, T, HV, generator=generator).to("npu", torch.bfloat16)
        cu = torch.tensor([0, T], dtype=torch.int32)
        chunk_indices = prepare_chunk_indices(cu, 64)
        cu_tuple = tuple(cu.to(torch.int64).reshape(-1).tolist())
        ci_tuple = tuple(chunk_indices.to(torch.int64).reshape(-1).tolist())

        def chunk_call() -> None:
            chunk_gated_delta_rule_fwd(
                q, k, v, g, beta,
                initial_state=None,
                output_final_state=False,
                chunk_size=64,
                cu_seqlens=cu_tuple,
                chunk_indices=ci_tuple,
                scale=SCALE,
                layout="BSND",
                use_exp2=False,
                use_qk_l2norm_in_kernel=False,
                allow_neg_eigval=False,
                disable_recompute=True,
                state_v_first=False,
            )

        x = torch.randn(T, DIM, generator=generator).to("npu", torch.bfloat16)
        weight = torch.randn(WIDTH, DIM, generator=generator).to("npu", torch.bfloat16)
        query_start_loc = torch.tensor([0, T], dtype=torch.int32, device="npu")

        def conv_call() -> None:
            causal_conv1d_fn(
                x, weight, None,
                conv_states=None,
                query_start_loc=query_start_loc,
                cache_indices=None,
                has_initial_state=None,
                activation="silu",
                pad_slot_id=-1,
                null_block_id=0,
                head_num=CONV_HEADS,
            )

        row = {
            "tokens": T,
            "chunk_ms": bench(chunk_call, args.warmup, args.iters),
            "conv_ms": bench(conv_call, args.warmup, args.iters),
        }
        rows.append(row)
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}), flush=True)

    summary = {
        "device": args.device,
        "tokens": lengths,
        "rows": rows,
        "chunk_48x_ms": {row["tokens"]: row["chunk_ms"] * 48 for row in rows},
        "conv_48x_ms": {row["tokens"]: row["conv_ms"] * 48 for row in rows},
    }
    with open(args.output, "w") as handle:
        json.dump(summary, handle, indent=1)
        handle.write("\n")
    print(json.dumps({"output": args.output}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
