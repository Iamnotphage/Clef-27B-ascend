#!/usr/bin/env python3
"""Probe an MLP-only Dense SP path on Ascend A3 TP2.

Baseline (current TP):
    x [M, H] -> gate/up [M, 2I/TP] -> SwiGLU -> down [M, H] partial
    -> all_reduce -> [M, H]

Sequence-parallel candidate:
    x_local [M/TP, H]
    -> npu_all_gather_base_mm(x_local, gate_up.T) -> [M, 2I/TP]
    -> SwiGLU -> [M, I/TP]
    -> npu_mm_reduce_scatter_base(act, down.T) -> [M/TP, H]

The comparison slices the baseline output to the rank's local token shard.
Run with torchrun --standalone --nproc_per_node=2.
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
probe = load("probe_mlp_ffn_for_sp", HERE / "probe_mlp_ffn.py")


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

    packed, down, identity = probe.load_weights(args.model_dir, torch, args.layer, rank)
    packed = packed.to(device)
    down = down.to(device)
    packed_t = packed.t().contiguous()
    down_t = down.t().contiguous()
    report = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "rank": rank,
        "world_size": world,
        "layer": args.layer,
        "hcom": hcom,
        "packed_shape": list(packed.shape),
        "packed_t_shape": list(packed_t.shape),
        "down_shape": list(down.shape),
        "down_t_shape": list(down_t.shape),
        "cases": [],
    }

    for requested in [int(item) for item in args.tokens.split(",") if item.strip()]:
        tokens = requested + ((world - requested % world) % world)
        local_tokens = tokens // world
        generator = torch.Generator(device="cpu").manual_seed(20261116 + rank)
        x = (torch.randn(tokens, identity["hidden_size"], generator=generator)
             .to(device, torch.bfloat16))
        x_local = x[rank * local_tokens:(rank + 1) * local_tokens].contiguous()

        def baseline():
            act = torch_npu.npu_swiglu(functional.linear(x, packed))
            out = functional.linear(act, down)
            dist.all_reduce(out, op=dist.ReduceOp.SUM)
            return out

        def sp():
            gate_up, _gathered = torch_npu.npu_all_gather_base_mm(
                x_local, packed_t, hcom, world, comm_mode="aiv"
            )
            act = torch_npu.npu_swiglu(gate_up)
            return torch_npu.npu_mm_reduce_scatter_base(
                act, down_t, hcom, world, reduce_op="sum", comm_mode="aiv"
            )

        reference = baseline()
        torch.npu.synchronize()
        record = {"requested_tokens": requested, "tokens": tokens}
        try:
            actual = sp()
            torch.npu.synchronize()
            record["sp_vs_baseline_slice"] = diff(
                actual, reference[rank * local_tokens:(rank + 1) * local_tokens]
            )
            record["baseline_ms"] = timed(baseline, args.warmup, args.iters)
            record["sp_ms"] = timed(sp, args.warmup, args.iters)
            record["speedup"] = record["baseline_ms"] / record["sp_ms"]
            record["status"] = "ok"
        except Exception as exc:  # noqa: BLE001
            record["status"] = "error"
            record["error"] = repr(exc)
        report["cases"].append(record)
        print(json.dumps({"rank": rank, **record}, ensure_ascii=False), flush=True)
        del x, x_local, reference
        torch.npu.empty_cache()

    if rank == 0 and args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n")
    dist.barrier()
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
