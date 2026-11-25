#!/usr/bin/env python3
"""Probe GEMM/AllReduce overlap for the TP2 down_proj.

Baseline:
    partial = F.linear(x, down_weight)          # [M, H]
    dist.all_reduce(partial)                    # full output

Chunked overlap:
    split x by rows into C chunks
    compute chunk i on the compute stream
    all_reduce chunk i on a separate HCCL stream while computing chunk i+1
    join the communication stream

The full output is compared with the baseline; no layout change is needed.
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
probe = load("probe_mlp_ffn_for_overlap", HERE / "probe_mlp_ffn.py")


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
    parser.add_argument("--chunks", default="2,4")
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
    device = f"npu:{local_rank}"
    weight = probe.load_weights(args.model_dir, torch, args.layer, rank)[1].to(device)
    report = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "rank": rank, "world_size": world, "layer": args.layer,
        "weight_shape": list(weight.shape), "cases": [],
    }

    for requested in [int(item) for item in args.tokens.split(",") if item.strip()]:
        tokens = requested + ((world - requested % world) % world)
        generator = torch.Generator(device="cpu").manual_seed(20261116)
        x = (torch.randn(tokens, weight.shape[1], generator=generator)
             .to(device, torch.bfloat16))

        def baseline():
            out = functional.linear(x, weight)
            dist.all_reduce(out, op=dist.ReduceOp.SUM)
            return out

        reference = baseline()
        torch.npu.synchronize()
        record = {"requested_tokens": requested, "tokens": tokens, "baseline_ms": timed(
            baseline, args.warmup, args.iters)}
        for chunks in [int(item) for item in args.chunks.split(",") if item.strip()]:
            bounds = [(tokens * i) // chunks for i in range(chunks + 1)]
            compute_stream = torch.npu.current_stream()
            comm_stream = torch.npu.Stream()
            outputs = [None] * chunks

            def chunked():
                events = []
                works = []
                for i in range(chunks):
                    lo, hi = bounds[i], bounds[i + 1]
                    with torch.npu.stream(compute_stream):
                        part = functional.linear(x[lo:hi], weight)
                        outputs[i] = part
                        event = torch.npu.Event()
                        event.record(compute_stream)
                    events.append(event)
                for i in range(chunks):
                    with torch.npu.stream(comm_stream):
                        comm_stream.wait_event(events[i])
                        works.append(dist.all_reduce(outputs[i], op=dist.ReduceOp.SUM, async_op=True))
                for work in works:
                    work.wait()
                torch.npu.current_stream().wait_stream(comm_stream)

            chunked()
            torch.npu.synchronize()
            actual = torch.cat(outputs, dim=0)
            record[f"chunks_{chunks}"] = {
                "vs_baseline": diff(actual, reference),
                "ms": timed(chunked, args.warmup, args.iters),
            }
            record[f"chunks_{chunks}"]["speedup"] = (
                record["baseline_ms"] / record[f"chunks_{chunks}"]["ms"]
            )
        report["cases"].append(record)
        print(json.dumps({"rank": rank, **record}, ensure_ascii=False), flush=True)
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
