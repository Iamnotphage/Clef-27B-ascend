#!/usr/bin/env python3
"""Compare down_proj + TP all_reduce with the fused MC2 op.

Run with two ranks (torchrun --nproc_per_node=2).  Each rank owns one TP shard
of layer L's down_proj, so:

    baseline = F.linear(x_rank, W_rank) + all_reduce(sum)
    mc2      = npu_mm_all_reduce_base(x_rank, W_rank.T, hcom, reduce_op="sum")

Both should produce the same full [M, hidden] output.  The probe reports
correctness and rank-0 device time at M=307/969/1595 (optionally 10294).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import statistics
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
probe = load("probe_mlp_ffn_for_mc2", HERE / "probe_mlp_ffn.py")


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
    actual_f = actual.detach().float()
    reference_f = reference.detach().float()
    delta = (actual_f - reference_f).abs()
    return {
        "bit_equal": bool(torch.equal(actual, reference)),
        "max_abs": float(delta.max().item()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=Path("/workspace/yangchen/models/Clef-27B"))
    parser.add_argument("--layer", type=int, default=31)
    parser.add_argument("--tokens", default="307,969,1595")
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.set_num_threads(1)
    probe.initialize_npu(torch, torch_npu, local_rank)
    dist.init_process_group("hccl", init_method="env://", world_size=world, rank=rank)
    group = dist.group.WORLD
    backend = group._get_backend(torch.device("npu"))
    hcom = backend.get_hccl_comm_name(rank)
    device = f"npu:{local_rank}"

    packed_cpu, down_cpu, identity = probe.load_weights(
        args.model_dir, torch, args.layer, rank
    )
    weight = down_cpu.to(device)  # [hidden, shard]
    weight_t = weight.t().contiguous()  # [shard, hidden]
    report = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "rank": rank,
        "world_size": world,
        "layer": args.layer,
        "hcom": hcom,
        "weight_shape": list(weight.shape),
        "weight_t_shape": list(weight_t.shape),
        "weight_t_contiguous": weight_t.is_contiguous(),
        "weight_format": str(torch_npu.get_npu_format(weight)),
        "weight_t_format": str(torch_npu.get_npu_format(weight_t)),
        "cases": [],
    }

    for tokens in [int(item) for item in args.tokens.split(",") if item.strip()]:
        generator = torch.Generator(device="cpu").manual_seed(20261115 + rank)
        x = (torch.randn(tokens, identity["intermediate_size"] // 2, generator=generator)
             .to(device, torch.bfloat16))
        reference = functional.linear(x, weight)
        dist.all_reduce(reference, op=dist.ReduceOp.SUM)
        fused = torch_npu.npu_mm_all_reduce_base(
            x, weight_t, hcom, reduce_op="sum"
        )
        torch.npu.synchronize()
        record = {
            "tokens": tokens,
            "mc2_vs_baseline": diff(fused, reference),
            "baseline_ms": timed(
                lambda: dist.all_reduce(functional.linear(x, weight), op=dist.ReduceOp.SUM),
                args.warmup, args.iters),
            "mc2_ms": timed(
                lambda: torch_npu.npu_mm_all_reduce_base(x, weight_t, hcom, reduce_op="sum"),
                args.warmup, args.iters),
        }
        record["speedup"] = record["baseline_ms"] / record["mc2_ms"]
        report["cases"].append(record)
        print(json.dumps(record, ensure_ascii=False), flush=True)
        del x, reference, fused
        torch.npu.empty_cache()

    if rank == 0 and args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n")
    dist.barrier()
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
