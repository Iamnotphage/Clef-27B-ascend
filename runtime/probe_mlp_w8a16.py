#!/usr/bin/env python3
"""Probe the MLP chain with W8A16 (weight-only int8) against BF16 on one NPU.

Reuses the pinned W8A8 probe's weight loading and per-channel quantization.
The W8A16 arm keeps BF16 activations and calls
``torch_npu.npu_weight_quant_batchmatmul``; the down projection stays BF16 in
both arms, matching the full-model candidate.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
import time
from pathlib import Path

import torch
import torch_npu
import torch.nn.functional as functional

BASE_PATH = Path(__file__).with_name("probe_mlp_w8a8.py")


def load_base():
    spec = importlib.util.spec_from_file_location("probe_mlp_w8a8_base", BASE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def w8a16_mlp(x, weights):
    gate_up = torch_npu.npu_weight_quant_batchmatmul(
        x=x,
        weight=weights["gate_up_int8"],
        antiquant_scale=weights["gate_up_scale"],
        antiquant_offset=weights["gate_up_offset"],
    )
    hidden = torch_npu.npu_swiglu(gate_up)
    return functional.linear(hidden, weights["down_bf16"])


def measure_arm(function, x, weights, iterations, device):
    start = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)
    device_ms, wall_ms = [], []
    start.record()
    end.record()
    torch.npu.synchronize(device)
    for _ in range(iterations):
        torch.npu.synchronize(device)
        began = time.perf_counter()
        start.record()
        output = function(x, weights)
        end.record()
        torch.npu.synchronize(device)
        wall_ms.append((time.perf_counter() - began) * 1000.0)
        device_ms.append(float(start.elapsed_time(end)))
        del output
    return {"device_ms": device_ms, "synchronized_wall_ms": wall_ms}


def summarize(values):
    return {
        "count": len(values),
        "median": statistics.median(values),
        "mean": statistics.mean(values),
        "min": min(values),
        "max": max(values),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=Path("/workspace/yangchen/models/Clef-27B"))
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--rank", type=int, choices=(0, 1), required=True)
    parser.add_argument("--rounds", type=int, default=6)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    base = load_base()
    device = torch.device("npu:0")
    document = torch.load(args.input, map_location="cpu", weights_only=True)
    x = document["input"] if isinstance(document, dict) else document
    if x.dim() != 2 or x.shape[-1] != 5120:
        raise SystemExit(f"unexpected capture shape {tuple(x.shape)}")
    x = x.to(torch.bfloat16).to(device).contiguous()

    packed, metadata = base.load_weights(args.model_dir, args.layer, args.rank)
    weights = base.to_device_weights(packed, device, "nz")
    weights["gate_up_offset"] = torch.zeros(
        weights["gate_up_scale"].numel(), dtype=torch.bfloat16, device=device
    )
    del packed

    functions = {"bf16": base.bf16_mlp, "w8a16": w8a16_mlp}
    for _ in range(args.warmup):
        for function in functions.values():
            function(x, weights)
    torch.npu.synchronize(device)

    reference = base.bf16_mlp(x, weights)
    candidate = w8a16_mlp(x, weights)
    torch.npu.synchronize(device)
    accuracy = base.compare_outputs(reference, candidate)
    del reference, candidate

    rounds = []
    for index in range(args.rounds):
        order = ["bf16", "w8a16"] if index % 2 == 0 else ["w8a16", "bf16"]
        arms = {
            name: measure_arm(functions[name], x, weights, args.iterations, device)
            for name in order
        }
        rounds.append({"round": index + 1, "order": order, "arms": arms})

    paired = {}
    for metric in ("device_ms", "synchronized_wall_ms"):
        ratios = []
        for pair in rounds:
            bf16 = statistics.median(pair["arms"]["bf16"][metric])
            w8a16 = statistics.median(pair["arms"]["w8a16"][metric])
            ratios.append(bf16 / w8a16)
        paired[metric] = {
            "per_round_speedup_bf16_over_w8a16": ratios,
            "median_speedup": statistics.median(ratios),
            "mean_speedup": statistics.mean(ratios),
            "w8a16_faster_rounds": sum(1 for ratio in ratios if ratio > 1.0),
        }

    report = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "input": str(args.input),
        "tokens": int(x.shape[0]),
        "layer": args.layer,
        "rank": args.rank,
        "rounds": args.rounds,
        "iterations": args.iterations,
        "metadata": metadata,
        "accuracy": accuracy,
        "summaries": {
            metric: {
                name: summarize(
                    [sample for pair in rounds for sample in pair["arms"][name][metric]]
                )
                for name in functions
            }
            for metric in ("device_ms", "synchronized_wall_ms")
        },
        "paired": paired,
        "rounds_detail": rounds,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({
        "device_speedup_median": paired["device_ms"]["median_speedup"],
        "wall_speedup_median": paired["synchronized_wall_ms"]["median_speedup"],
        "w8a16_faster_rounds": paired["device_ms"]["w8a16_faster_rounds"],
        "relative_rmse": accuracy.get("relative_rmse"),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
