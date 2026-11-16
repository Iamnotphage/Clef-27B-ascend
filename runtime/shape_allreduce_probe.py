#!/usr/bin/env python3
"""Time TP2 all-reduce across token lengths that bracket the service step.

Run with: torchrun --standalone --nproc_per_node=2 runtime/shape_allreduce_probe.py

The service performs ~130 all-reduces per request; the MLP down projection's
``[T, 5120]`` BF16 activation is the dominant shape.  A sharp cost change at
308 tokens would explain the service-level step without a matmul/kernel jump.
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


def bench(tensor, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    torch.npu.synchronize()
    samples = []
    for _ in range(iters):
        started = time.perf_counter()
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        torch.npu.synchronize()
        samples.append((time.perf_counter() - started) * 1000.0)
    return statistics.median(samples)


def bench_chain(tensor, calls: int, repeats: int) -> float:
    for _ in range(2):
        for _ in range(calls):
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    torch.npu.synchronize()
    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        for _ in range(calls):
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        torch.npu.synchronize()
        samples.append((time.perf_counter() - started) * 1000.0)
    return statistics.median(samples)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", default="203,306,307,308,310,320,400,967")
    parser.add_argument("--hidden", type=int, default=5120)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--chain", type=int, default=64)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.npu.set_device(local_rank)
    dist.init_process_group("hccl", init_method="env://", world_size=world, rank=rank)
    torch.manual_seed(20261117 + rank)
    rows = []
    for tokens in [int(value) for value in args.tokens.split(",")]:
        tensor = torch.randn(tokens, args.hidden, device=f"npu:{local_rank}", dtype=torch.bfloat16)
        row = {
            "rank": rank,
            "tokens": tokens,
            "one_ms": bench(tensor, args.warmup, args.iters),
            "chain_ms": bench_chain(tensor, args.chain, 5),
        }
        rows.append(row)
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}), flush=True)
        del tensor
        torch.npu.empty_cache()
    if rank == 0 and args.output:
        with open(args.output, "w") as handle:
            json.dump({"rows": rows, "world_size": world, "hidden": args.hidden, "chain": args.chain}, handle, indent=1)
            handle.write("\n")
    dist.barrier()
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
