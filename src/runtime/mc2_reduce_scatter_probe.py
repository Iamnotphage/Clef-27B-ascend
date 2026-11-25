#!/usr/bin/env python3
"""Probe BF16 MatMulReduceScatter on Ascend A3 TP2.

For each TP rank this loads one shard of layer L's down_proj:

    x: [M, K]           K = intermediate_size / TP
    W: [N, K]           N = hidden_size (row-parallel shard)
    y = x @ W.T         [M, N] partial, then all_reduce sums the TP ranks

The fused candidate is ``npu_mm_reduce_scatter_base(x, W.T, hcom, 2,
comm_mode="aiv")``.  It returns the local [M/2, N] reduce-scattered shard;
``all_gather_into_tensor`` reconstructs the full [M, N] output for an
equivalence check.  Times are rank-0 device event means.

Run with two ranks:
  source scripts/activate_env.sh
  torchrun --standalone --nproc_per_node=2 runtime/mc2_reduce_scatter_probe.py \
      --tokens 308,970,1596 --iters 10 --warmup 3 --output <json>
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch_npu
import torch.nn.functional as functional


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


HERE = Path(__file__).resolve().parent
probe = load("probe_mlp_ffn_for_mc2_rs", HERE / "probe_mlp_ffn.py")


def timed(fn, warmup: int, iterations: int):
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    dist.barrier()
    start = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        fn()
    end.record()
    torch.npu.synchronize()
    dist.barrier()
    return start.elapsed_time(end) / iterations


def diff(actual, reference):
    delta = (actual.detach().float() - reference.detach().float()).abs()
    return {
        "shape": list(actual.shape),
        "bit_equal": bool(torch.equal(actual, reference)),
        "max_abs": float(delta.max().item()),
        "mean_abs": float(delta.mean().item()),
        "finite": bool(torch.isfinite(actual).all().item()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=Path("/workspace/yangchen/models/Clef-27B"))
    parser.add_argument("--layer", type=int, default=31)
    parser.add_argument("--tokens", default="308,970,1596")
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.set_num_threads(1)
    probe.initialize_npu(torch, torch_npu, local_rank)
    dist.init_process_group("hccl", init_method="env://", world_size=world, rank=rank)
    backend = dist.group.WORLD._get_backend(torch.device("npu"))
    hcom = backend.get_hccl_comm_name(rank)
    device = f"npu:{local_rank}"

    _packed, down_cpu, identity = probe.load_weights(args.model_dir, torch, args.layer, rank)
    weight = down_cpu.to(device)  # [N, K]
    weight_t = weight.t().contiguous()  # [K, N]
    report = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "rank": rank,
        "world_size": world,
        "layer": args.layer,
        "hcom": hcom,
        "weight_shape": list(weight.shape),
        "weight_t_shape": list(weight_t.shape),
        "weight_t_contiguous": weight_t.is_contiguous(),
        "cases": [],
    }

    for requested in [int(item) for item in args.tokens.split(",") if item.strip()]:
        tokens = requested + ((world - requested % world) % world)
        if tokens % world:
            raise ValueError("tokens must be divisible by world_size")
        generator = torch.Generator(device="cpu").manual_seed(20261116 + rank)
        x = (torch.randn(tokens, identity["intermediate_size"] // world,
                         generator=generator).to(device, torch.bfloat16))
        reference = functional.linear(x, weight)
        dist.all_reduce(reference, op=dist.ReduceOp.SUM)
        baseline_ms = timed(
            lambda: dist.all_reduce(functional.linear(x, weight), op=dist.ReduceOp.SUM),
            args.warmup, args.iters)

        for mode in ("aiv", None):
            record = {
                "requested_tokens": requested,
                "tokens": tokens,
                "comm_mode": mode,
                "baseline_ms": baseline_ms,
            }
            try:
                local = torch_npu.npu_mm_reduce_scatter_base(
                    x, weight_t, hcom, world, reduce_op="sum", comm_mode=mode
                )
                torch.npu.synchronize()
                record["local_output_shape"] = list(local.shape)
                record["local_vs_reference_slice"] = diff(
                    local, reference[rank * (tokens // world):(rank + 1) * (tokens // world)]
                )
                gathered = torch.empty(
                    (tokens, identity["hidden_size"]), dtype=torch.bfloat16, device=device
                )
                dist.all_gather_into_tensor(gathered, local)
                torch.npu.synchronize()
                record["full_vs_reference"] = diff(gathered, reference)
                record["fused_plus_gather_ms"] = timed(
                    lambda: dist.all_gather_into_tensor(
                        gathered,
                        torch_npu.npu_mm_reduce_scatter_base(
                            x, weight_t, hcom, world, reduce_op="sum", comm_mode=mode
                        ),
                    ),
                    args.warmup, args.iters,
                )
                record["fused_local_ms"] = timed(
                    lambda: torch_npu.npu_mm_reduce_scatter_base(
                        x, weight_t, hcom, world, reduce_op="sum", comm_mode=mode
                    ),
                    args.warmup, args.iters,
                )
                record["speedup_full"] = baseline_ms / record["fused_plus_gather_ms"]
                record["speedup_local"] = baseline_ms / record["fused_local_ms"]
                record["status"] = "ok"
            except Exception as exc:  # noqa: BLE001
                record["status"] = "error"
                record["error"] = repr(exc)
            report["cases"].append(record)
            print(json.dumps({"rank": rank, **record}, ensure_ascii=False), flush=True)
            if record["status"] != "ok":
                torch.npu.empty_cache()
        del x, reference
        torch.npu.empty_cache()

    if rank == 0 and args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n")
    dist.barrier()
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
