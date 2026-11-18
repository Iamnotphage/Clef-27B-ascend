#!/usr/bin/env python3
"""Wave-3 dual-stream N-split probe (TP2, A3).

The only remaining condition for an N/column split is whether two half-payload
collectives issued on two streams hide the per-call fixed cost better than one
full-payload collective.  Measures, at 967/1595 tokens:

  whole        one [T,5120] all-reduce, default stream, one sync
  pair_seq     two [T,2560] all-reduces back to back on the default stream
  pair_dual    the same two, issued on two NPU streams, wait both, one sync

Writes JSON; prints the ratios.  Timing only, random BF16 data.
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


def timeit(fn, warmup, iters):
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    samples = []
    for _ in range(iters):
        begin = time.perf_counter()
        fn()
        torch.npu.synchronize()
        samples.append((time.perf_counter() - begin) * 1000.0)
    return {"ms": statistics.median(samples), "min_ms": min(samples), "samples": len(samples)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", default="967,1595")
    parser.add_argument("--hidden", type=int, default=5120)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.npu.set_device(local_rank)
    dist.init_process_group("hccl", init_method="env://")
    device = f"npu:{local_rank}"
    torch.manual_seed(20261118 + rank)

    stream_a = torch.npu.Stream(device=device)
    stream_b = torch.npu.Stream(device=device)
    rows = []
    for tokens in [int(v) for v in args.tokens.split(",")]:
        whole = torch.randn(tokens, args.hidden, device=device, dtype=DTYPE)
        half_a = torch.randn(tokens, args.hidden // 2, device=device, dtype=DTYPE)
        half_b = torch.randn(tokens, args.hidden // 2, device=device, dtype=DTYPE)

        def run_whole():
            dist.all_reduce(whole, op=dist.ReduceOp.SUM)

        def run_pair_seq():
            dist.all_reduce(half_a, op=dist.ReduceOp.SUM)
            dist.all_reduce(half_b, op=dist.ReduceOp.SUM)

        def run_pair_dual():
            current = torch.npu.current_stream()
            with torch.npu.stream(stream_a):
                stream_a.wait_stream(current)
                dist.all_reduce(half_a, op=dist.ReduceOp.SUM)
            with torch.npu.stream(stream_b):
                stream_b.wait_stream(current)
                dist.all_reduce(half_b, op=dist.ReduceOp.SUM)
            current.wait_stream(stream_a)
            current.wait_stream(stream_b)

        whole_ms = timeit(run_whole, args.warmup, args.iters)
        seq_ms = timeit(run_pair_seq, args.warmup, args.iters)
        dual_ms = timeit(run_pair_dual, args.warmup, args.iters)
        row = {
            "rank": rank,
            "tokens": tokens,
            "whole_ms": whole_ms["ms"],
            "pair_seq_ms": seq_ms["ms"],
            "pair_dual_ms": dual_ms["ms"],
            "ratio_seq_over_whole": seq_ms["ms"] / whole_ms["ms"],
            "ratio_dual_over_whole": dual_ms["ms"] / whole_ms["ms"],
            "payload_mb": tokens * args.hidden * 2 / 1024**2,
        }
        rows.append(row)
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}),
              flush=True)
        del whole, half_a, half_b
        torch.npu.empty_cache()

    if rank == 0:
        analysis = {
            "dimension": "N/output-column split (two [T,5120/2] collectives), not K-reduction",
            "gate": "ratio_dual_over_whole < 0.90 at 967 -> dual-stream N split may help; >= 0.90 -> close",
            "at_967": next((r["ratio_dual_over_whole"] for r in rows if r["tokens"] == 967), None),
        }
        with open(args.output, "w") as handle:
            json.dump({"rows": rows, "analysis": analysis, "host_load": os.getloadavg(),
                       "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}, handle, indent=1)
            handle.write("\n")
        print(json.dumps(analysis, indent=1), flush=True)
    dist.barrier()
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
