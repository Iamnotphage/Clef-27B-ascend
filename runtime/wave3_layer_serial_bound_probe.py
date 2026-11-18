#!/usr/bin/env python3
"""Wave-3 single-layer serial baseline and ideal-overlap bound (TP2, A3).

Real layer shape: MLP down_proj ``Y[M,8704] @ W[8704,5120]`` followed by the
row-parallel all-reduce and the fused AddRmsNorm.  Measured per M:

  gemm, allreduce, addrmsnorm, serial (all three, one sync at the end),
  ideal_bound = max(gemm + addrmsnorm, allreduce),
  recoverable = serial - ideal_bound.

``recoverable`` is the ceiling that ANY overlap or fused kernel could win per
layer; it is not a predicted gain.  Only if it clears the decision threshold
do we spend a window on a dual-stream split.

Run:
  torchrun --standalone --nproc_per_node=2 runtime/wave3_layer_serial_bound_probe.py \
      --output /workspace/yangchen/logs/perf-wave3-20261118/layer-serial-bound.json
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
K_MLP = 8704
N_HIDDEN = 5120


def timed(fn, warmup, iters, sync_each=True):
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    wall, dev = [], []
    for _ in range(iters):
        start = torch.npu.Event(enable_timing=True)
        end = torch.npu.Event(enable_timing=True)
        begin = time.perf_counter()
        start.record()
        out = fn()
        end.record()
        if sync_each:
            torch.npu.synchronize()
        wall.append((time.perf_counter() - begin) * 1000.0)
        dev.append(start.elapsed_time(end))
    return {"ms": statistics.median(wall), "device_ms": statistics.median(dev)}, out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", default="600,967,1595")
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.npu.set_device(local_rank)
    dist.init_process_group("hccl", init_method="env://")
    device = f"npu:{local_rank}"
    torch.manual_seed(20261118 + rank)

    weight = torch.randn(N_HIDDEN, K_MLP, device=device, dtype=DTYPE)
    residual = torch.randn(1, N_HIDDEN, device=device, dtype=DTYPE)
    gamma = torch.ones(N_HIDDEN, device=device, dtype=DTYPE)
    eps = 1e-6

    def addrmsnorm(y):
        try:
            out = torch_npu.npu_add_rms_norm(y, residual, gamma, eps)
            return out[0] if isinstance(out, (tuple, list)) else out
        except Exception:  # noqa: BLE001 - fall back to the unfused pair
            summed = y + residual
            return torch_npu.npu_rms_norm(summed, gamma, eps)[0]

    rows = []
    for tokens in [int(v) for v in args.tokens.split(",")]:
        x = torch.randn(tokens, K_MLP, device=device, dtype=DTYPE)
        y = torch.matmul(x, weight.t())

        # A/B/C are measured with exactly one host sync per iteration so their
        # launch+sync overhead is comparable; comparing sync-per-op timings
        # against a single-sync chain is invalid.
        def chain_all():
            out = torch.matmul(x, weight.t())
            dist.all_reduce(out, op=dist.ReduceOp.SUM)
            return addrmsnorm(out)

        def chain_compute_only():
            out = torch.matmul(x, weight.t())
            return addrmsnorm(out)

        def chain_comm_only():
            dist.all_reduce(y, op=dist.ReduceOp.SUM)

        a, _ = timed(chain_all, args.warmup, args.iters)
        b, _ = timed(chain_compute_only, args.warmup, args.iters)
        c, _ = timed(chain_comm_only, args.warmup, args.iters)
        bound = max(b["ms"], c["ms"])
        row = {
            "rank": rank,
            "tokens": tokens,
            "gemm_plus_norm_ms": b["ms"],
            "allreduce_only_ms": c["ms"],
            "serial_all_ms": a["ms"],
            "ideal_bound_ms": bound,
            "recoverable_ms": max(0.0, a["ms"] - bound),
            "serial_minus_compute_ms": a["ms"] - b["ms"],
            "serial_minus_comm_ms": a["ms"] - c["ms"],
            "payload_mb": tokens * N_HIDDEN * 2 / 1024**2,
            "serial_device_ms": a["device_ms"],
            "compute_device_ms": b["device_ms"],
            "comm_device_ms": c["device_ms"],
        }
        rows.append(row)
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}),
              flush=True)
        del x, y
        torch.npu.empty_cache()

    if rank == 0:
        analysis = {
            "dimension": "per-layer serial baseline; ideal overlap bound = max(compute, allreduce)",
            "per_layer_recoverable_ms": {str(r["tokens"]): r["recoverable_ms"] for r in rows},
            "per_request_recoverable_ms_x64": {str(r["tokens"]): r["recoverable_ms"] * 64 for r in rows},
            "note": "recoverable is a ceiling, not a gain; double counting across layers is avoided by using the measured serial chain",
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
