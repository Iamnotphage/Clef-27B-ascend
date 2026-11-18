#!/usr/bin/env python3
"""B-group probe: real NZ GEMM cost at the service shapes.

Loads one layer's real BF16 weights from the checkpoint, slices them to the
TP2 per-rank shapes, converts to FRACTAL_NZ, and times the GEMMs that E-group
fusion would consume:

  gate/up : X[967,5120]  @ W[8704,5120]^T
  down    : X[967,8704]  @ W[5120,8704]^T
  tiles   : N split into 2 (2560) and 4 (1280) blocks, cumulative and per block

Records pack time, per-call wall/event medians, and keeps the NZ weights in
the same warm state across arms.  No communication, no service.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch
import torch_npu
from safetensors import safe_open

NZ_FORMAT = 29


def pct(values, q):
    ordered = sorted(values)
    idx = (len(ordered) - 1) * q
    lo, hi = int(idx), min(int(idx) + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (idx - lo)


def stats(values):
    return {"n": len(values), "median": statistics.median(values),
            "mean": statistics.fmean(values), "min": min(values), "max": max(values),
            "p90": pct(values, 0.90), "stdev": statistics.pstdev(values) if len(values) > 1 else 0.0}


def timed(fn, warmup, iters):
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    wall, dev = [], []
    for _ in range(iters):
        start = torch.npu.Event(enable_timing=True)
        end = torch.npu.Event(enable_timing=True)
        begin = time.perf_counter()
        start.record()
        fn()
        end.record()
        torch.npu.synchronize()
        wall.append((time.perf_counter() - begin) * 1000.0)
        dev.append(start.elapsed_time(end))
    return wall, dev


def load_weight(model_dir: Path, key: str) -> torch.Tensor:
    index = json.loads((model_dir / "model.safetensors.index.json").read_text())["weight_map"]
    shard = index[key]
    with safe_open(model_dir / shard, framework="pt", device="cpu") as handle:
        return handle.get_tensor(key)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, default=Path("/workspace/yangchen/models/Clef-27B"))
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--tokens", type=int, default=967)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--tp", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    torch.npu.set_device(0)
    prefix = f"model.language_model.layers.{args.layer}.mlp"
    gate_up = load_weight(args.model_dir, f"{prefix}.gate_proj.weight")   # [inter, hidden]
    down = load_weight(args.model_dir, f"{prefix}.down_proj.weight")      # [hidden, inter]
    inter_per_rank = gate_up.shape[0] // args.tp
    gate_up_r = gate_up[args.rank * inter_per_rank:(args.rank + 1) * inter_per_rank, :]
    down_r = down[:, args.rank * inter_per_rank:(args.rank + 1) * inter_per_rank]
    hidden = gate_up.shape[1]

    m = args.tokens
    gen = torch.Generator().manual_seed(20261118)
    x_gu = torch.randn(m, hidden, generator=gen).to(torch.bfloat16).npu()
    x_down = torch.randn(m, inter_per_rank, generator=gen).to(torch.bfloat16).npu()

    rows = {}

    def run(name, weight_cpu, x, n_out):
        pack_begin = time.perf_counter()
        w_npu = weight_cpu.npu()
        w_nz = torch_npu.npu_format_cast(w_npu, NZ_FORMAT)
        torch.npu.synchronize()
        pack_ms = (time.perf_counter() - pack_begin) * 1000.0
        wt = w_nz.t()
        wall, dev = timed(lambda: torch.matmul(x, wt), args.warmup, args.iters)
        rows[name] = {"n_out": n_out, "pack_ms": pack_ms,
                      "wall_ms": stats(wall), "device_ms": stats(dev)}
        print(json.dumps({"arm": name, "n_out": n_out, "pack_ms": round(pack_ms, 3),
                          "wall_median": round(statistics.median(wall), 4),
                          "dev_median": round(statistics.median(dev), 4)}), flush=True)
        del w_npu, w_nz, wt
        torch.npu.empty_cache()

    # gate/up full and N splits
    run("gate_up_full", gate_up_r, x_gu, inter_per_rank)
    for tiles, label in ((2, "gate_up_n2"), (4, "gate_up_n4")):
        step = inter_per_rank // tiles
        for t in range(tiles):
            run(f"{label}_tile{t}", gate_up_r[t * step:(t + 1) * step, :], x_gu, step)

    # down full and N splits
    run("down_full", down_r, x_down, hidden)
    for tiles, label in ((2, "down_n2"), (4, "down_n4")):
        step = hidden // tiles
        for t in range(tiles):
            run(f"{label}_tile{t}", down_r[t * step:(t + 1) * step, :], x_down, step)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "shape": {"tokens": m, "hidden": hidden, "inter_per_rank": inter_per_rank,
                  "tp": args.tp, "rank": args.rank, "layer": args.layer},
        "source": {"gate_proj": f"{prefix}.gate_proj.weight", "down_proj": f"{prefix}.down_proj.weight"},
        "rows": rows}, ensure_ascii=False, indent=1) + "\n")
    print(f"written {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
