#!/usr/bin/env python3
"""Wave-3 TP2 all-reduce probe: latency vs payload and the K-split question.

Two questions for the T1/T2 work:

1. Is the exposed per-layer all-reduce launch-dominated or bandwidth-dominated?
   Fit ``latency = fixed + payload / bandwidth`` over a payload sweep at the
   real hidden size (5120) and the S2 batch shapes.
2. Does halving the payload per collective (two ``[T, H/2]`` calls instead of
   one ``[T, H]``) return earlier?  This measures the **allreduce payload
   split**, which is what the next matmul consumes as its K dimension.  It is
   NOT a K-reduction split of the down_proj GEMM (K=8704): that variant keeps
   a full ``[M, 5120]`` partial output per block and needs one full-payload
   allreduce per block, so its communication doubles.  Record the dimension
   explicitly in every conclusion.

Run:
  torchrun --standalone --nproc_per_node=2 runtime/wave3_allreduce_payload_probe.py \
      --output /workspace/yangchen/logs/perf-wave3-20261118/allreduce-payload.json

Only timing; the tensors are random BF16 and the values are never used.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time

import torch
import torch.distributed as dist
import torch_npu

DTYPE = torch.bfloat16


def bench_one(tensor, warmup, iters):
    for _ in range(warmup):
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    torch.npu.synchronize()
    wall, dev = [], []
    for _ in range(iters):
        start = torch.npu.Event(enable_timing=True)
        end = torch.npu.Event(enable_timing=True)
        begin = time.perf_counter()
        start.record()
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        end.record()
        torch.npu.synchronize()
        wall.append((time.perf_counter() - begin) * 1000.0)
        dev.append(start.elapsed_time(end))
    return {
        "ms": statistics.median(wall),
        "device_ms": statistics.median(dev),
        "min_ms": min(wall),
        "samples": len(wall),
    }


def bench_chain(tensor, calls, repeats, warmup=2):
    for _ in range(warmup):
        for _ in range(calls):
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    torch.npu.synchronize()
    per_call = []
    for _ in range(repeats):
        begin = time.perf_counter()
        for _ in range(calls):
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        torch.npu.synchronize()
        per_call.append((time.perf_counter() - begin) * 1000.0 / calls)
    return {"ms_per_call": statistics.median(per_call), "calls": calls, "repeats": repeats}


def pair_latency(a, b, warmup, iters):
    """Two collectives issued back to back, one sync at the end."""
    for _ in range(warmup):
        dist.all_reduce(a, op=dist.ReduceOp.SUM)
        dist.all_reduce(b, op=dist.ReduceOp.SUM)
    torch.npu.synchronize()
    wall = []
    for _ in range(iters):
        begin = time.perf_counter()
        dist.all_reduce(a, op=dist.ReduceOp.SUM)
        dist.all_reduce(b, op=dist.ReduceOp.SUM)
        torch.npu.synchronize()
        wall.append((time.perf_counter() - begin) * 1000.0)
    return {"pair_ms": statistics.median(wall), "samples": len(wall)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", default="967,1595,4800,9600")
    parser.add_argument("--hidden", type=int, default=5120)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--chain", type=int, default=32)
    parser.add_argument("--split-tokens", default="967,1595")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.npu.set_device(local_rank)
    dist.init_process_group("hccl", init_method="env://", world_size=world, rank=rank)
    torch.manual_seed(20261118 + rank)

    device = f"npu:{local_rank}"
    rows = []
    for tokens in [int(v) for v in args.tokens.split(",")]:
        tensor = torch.randn(tokens, args.hidden, device=device, dtype=DTYPE)
        payload_mb = tensor.numel() * tensor.element_size() / 1024**2
        one = bench_one(tensor, args.warmup, args.iters)
        chain = bench_chain(tensor, args.chain, 5)
        row = {
            "kind": "sweep",
            "rank": rank,
            "tokens": tokens,
            "hidden": args.hidden,
            "payload_mb": payload_mb,
            "one": one,
            "chain": chain,
            "gb_per_s": payload_mb / 1024 / (chain["ms_per_call"] / 1000.0),
        }
        rows.append(row)
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v)
                          for k, v in {**row, "one": {kk: round(vv, 4) for kk, vv in one.items()},
                                       "chain": {kk: round(vv, 4) if isinstance(vv, float) else vv
                                                 for kk, vv in chain.items()}}.items()}), flush=True)
        del tensor
        torch.npu.empty_cache()

    # Allreduce payload split: one [T,H] vs two [T,H/2]. This is the
    # "next-matmul K split" design, not the down_proj K-reduction split.
    for tokens in [int(v) for v in args.split_tokens.split(",")]:
        whole = torch.randn(tokens, args.hidden, device=device, dtype=DTYPE)
        half_a = torch.randn(tokens, args.hidden // 2, device=device, dtype=DTYPE)
        half_b = torch.randn(tokens, args.hidden // 2, device=device, dtype=DTYPE)
        one = bench_one(whole, args.warmup, args.iters)
        pair = pair_latency(half_a, half_b, args.warmup, args.iters)
        row = {
            "kind": "payload_split",
            "dimension": "allreduce_payload_eq_next_matmul_k",
            "rank": rank,
            "tokens": tokens,
            "hidden": args.hidden,
            "whole_one_ms": one["ms"],
            "half_pair_ms": pair["pair_ms"],
            "ratio_half_over_whole": pair["pair_ms"] / one["ms"],
            "payload_mb": tokens * args.hidden * 2 / 1024**2,
        }
        rows.append(row)
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}),
              flush=True)
        del whole, half_a, half_b
        torch.npu.empty_cache()

    if rank == 0:
        sweep = [r for r in rows if r["kind"] == "sweep"]
        # Least squares fit latency(ms) = fixed_ms + slope * payload_mb.
        xs = [r["payload_mb"] for r in sweep]
        ys = [r["chain"]["ms_per_call"] for r in sweep]
        n = len(xs)
        mean_x, mean_y = sum(xs) / n, sum(ys) / n
        denom = sum((x - mean_x) ** 2 for x in xs) or 1.0
        slope = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / denom
        fixed = mean_y - slope * mean_x
        analysis = {
            "fit": {"fixed_ms": fixed, "slope_ms_per_mb": slope,
                    # slope is ms per MB; 1 MB/ms == 1 GB/s.
                    "implied_bw_gbs": (1.0 / slope) if slope > 0 else None,
                    "points": n},
            "ratio_at": {str(r["tokens"]): r["ratio_half_over_whole"]
                         for r in rows if r["kind"] == "payload_split"},
            "dimension_note": "ratio is the allreduce-payload split (next matmul K); a down_proj K-reduction split keeps full [M,5120] partials and doubles the full-payload allreduce count, so it needs its own cost model and is not implied by this ratio",
            "t2_gate": "ratio_half_over_whole < 0.90 -> payload split returns earlier; still only necessary, not sufficient (needs the overlap bound)",
        }
        with open(args.output, "w") as handle:
            json.dump({"rows": rows, "analysis": analysis, "world_size": world,
                       "hidden": args.hidden, "host_load": os.getloadavg(),
                       "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")},
                      handle, indent=1)
            handle.write("\n")
        print(json.dumps(analysis, indent=1), flush=True)
    dist.barrier()
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
