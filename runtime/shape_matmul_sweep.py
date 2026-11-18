#!/usr/bin/env python3
"""Time every real model matmul shape at the service's token-range boundary.

The service has a sharp device-side latency step at 308 tokens.  The MLP
gate/up and down shapes were already flat across the boundary, so this probe
covers the remaining projections with the exact (M, K, N) triples observed in
the device trace at 307 tokens.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time

import torch
import torch.nn.functional as functional

# (name, K, N) from the profiled aclnnMatmul shapes at 307 tokens.
SHAPES = [
    ("gate_up", 5120, 17408),
    ("down", 17408, 5120),
    ("gdn_in", 5120, 8192),
    ("out_or_attn", 5120, 3072),
    ("proj_7168", 5120, 7168),
    ("small_48", 5120, 48),
    ("k8192", 8192, 5120),
    ("k3072", 3072, 5120),
    ("k7168", 7168, 5120),
]


def bench(call, warmup: int, iters: int, flush, flush_mib: int) -> float:
    for _ in range(warmup):
        call()
    torch.npu.synchronize()
    samples = []
    for _ in range(iters):
        if flush is not None:
            flush.zero_()
            torch.npu.synchronize()
        started = time.perf_counter()
        call()
        torch.npu.synchronize()
        samples.append((time.perf_counter() - started) * 1000.0)
    return statistics.median(samples)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--tokens", default="203,306,307,308,310,320,400")
    parser.add_argument("--iters", type=int, default=15)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--flush-mib", type=int, default=256)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    torch.npu.set_device(args.device)
    torch.manual_seed(20261117)
    flush = None
    if args.flush_mib:
        flush = torch.zeros(args.flush_mib * 1024 * 1024 // 2, dtype=torch.bfloat16, device="npu")
    tokens = [int(value) for value in args.tokens.split(",")]
    rows = []
    for name, k, n in SHAPES:
        weight = torch.randn(n, k, device="npu", dtype=torch.bfloat16)
        for m in tokens:
            x = torch.randn(m, k, device="npu", dtype=torch.bfloat16)

            def call(x=x, weight=weight):
                functional.linear(x, weight)

            ms = bench(call, args.warmup, args.iters, flush, args.flush_mib)
            row = {"name": name, "m": m, "k": k, "n": n, "cold_ms": ms}
            rows.append(row)
            print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}), flush=True)
        del weight
        torch.npu.empty_cache()
    with open(args.output, "w") as handle:
        json.dump({"tokens": tokens, "shapes": SHAPES, "rows": rows, "flush_mib": args.flush_mib}, handle, indent=1)
        handle.write("\n")
    print(json.dumps({"output": args.output}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
