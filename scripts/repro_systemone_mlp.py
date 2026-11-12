#!/usr/bin/env python3
"""Synthetic BF16/ND MLP matmul signature replay for an exclusive NPU window.

The baseline exports shape/dtype/format, and does not export physical strides.
This replay explicitly uses contiguous x/weight and verifies its kernel signature
after msprof export. Synthetic single-op timing is independent of model latency.
"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=16)
    parser.add_argument("--rotating-weights", type=int, default=3)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--describe", action="store_true")
    args = parser.parse_args()
    if min(args.warmup, args.iterations, args.rotating_weights) < 1:
        parser.error("warmup, iterations and rotating weights must be positive")
    spec = {"input_shape": [307, 5120], "weight_shape": [34816, 5120],
            "output_shape": [307, 34816], "dtype": "bfloat16", "format": "ND",
            "operation": "torch.nn.functional.linear(x, weight)",
            "input_stride_assumption": [5120, 1], "weight_stride_assumption": [5120, 1],
            "stride_evidence_boundary": "Baseline trace lacks physical strides; replay assumes contiguous tensors",
            "weights_bytes_each": 34816 * 5120 * 2,
            "rotating_weights": args.rotating_weights,
            "cache_boundary": "Weight rotation reduces same-matrix reuse; baseline full-model cache state is independent"}
    if args.describe:
        print(json.dumps(spec, indent=2))
        return
    import torch
    import torch_npu
    import torch.nn.functional as F

    started = time.perf_counter()
    torch.set_num_threads(1)
    torch.npu.set_device(args.device)
    torch.manual_seed(0)
    torch.npu.manual_seed_all(0)
    device = f"npu:{args.device}"
    with torch.inference_mode():
        x = torch.randn((307, 5120), dtype=torch.bfloat16, device=device) * 0.1
        weights = [torch.randn((34816, 5120), dtype=torch.bfloat16, device=device) * 0.1
                   for _ in range(args.rotating_weights)]
        x = torch_npu.npu_format_cast(x.contiguous(), 2)
        weights = [torch_npu.npu_format_cast(weight.contiguous(), 2) for weight in weights]
        phases = []
        cursor = 0
        for phase in ("steady_single_weight", "rotating_weights"):
            for index in range(args.warmup):
                weight = weights[0] if phase == "steady_single_weight" else weights[index % len(weights)]
                output = F.linear(x, weight)
            torch.npu.synchronize()
            start = torch.npu.Event(enable_timing=True)
            end = torch.npu.Event(enable_timing=True)
            start.record()
            for index in range(args.iterations):
                weight = weights[0] if phase == "steady_single_weight" else weights[index % len(weights)]
                output = F.linear(x, weight)
            end.record()
            torch.npu.synchronize()
            elapsed_ms = start.elapsed_time(end)
            phases.append({"name": phase, "warmup": args.warmup, "iterations": args.iterations,
                "matching_matmul_formal_index_range_0based_inclusive": [cursor + args.warmup,
                    cursor + args.warmup + args.iterations - 1],
                "device_event_total_ms": elapsed_ms,
                "device_event_mean_ms": elapsed_ms / args.iterations})
            cursor += args.warmup + args.iterations
        final_weight = weights[(args.iterations - 1) % len(weights)]
        expected = F.linear(x[:2].cpu().float(), final_weight[:64].cpu().float())
        actual = output[:2, :64].cpu().float()
        difference = (expected - actual).abs()
        passed = bool(torch.allclose(expected, actual, atol=0.02, rtol=0.03))
        report = {"created_at": datetime.now(timezone.utc).isoformat(), "specification": spec,
            "passed": passed, "device": device, "warmup": args.warmup,
            "iterations": args.iterations, "input_stride": list(x.stride()),
            "weight_stride": list(weights[0].stride()), "input_npu_format": torch_npu.get_npu_format(x),
            "weight_npu_format": torch_npu.get_npu_format(weights[0]),
            "phases": phases,
            "torch_allocated_bytes": torch.npu.memory_allocated(),
            "torch_reserved_bytes": torch.npu.memory_reserved(),
            "probe_cpu_fp32_max_absolute_difference": difference.max().item(),
            "wall_seconds": time.perf_counter() - started,
            "scope": "Synthetic one-op replay; use matching_matmul_formal_index_range for each phase in chronological msprof CSV matching kernels; exclude warmup and probe matmul"}
    payload = json.dumps(report, indent=2) + "\n"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(payload)
    print(payload, end="")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
