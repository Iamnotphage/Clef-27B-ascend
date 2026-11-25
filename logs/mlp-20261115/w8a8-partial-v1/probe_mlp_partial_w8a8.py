#!/usr/bin/env python3
"""Compare BF16, both-projection W8A8, gate-up-only W8A8 and down-only W8A8.

Reuse the real TP2 weight shards, captured-input validation, NZ initialization,
measurement and reporting from probe_mlp_w8a8. The default eight rounds rotate
the four arms and reverse every paired round. Each arm occupies each position
twice, and every pair has equal precedence in both directions.
"""

from __future__ import annotations

import statistics

import probe_mlp_w8a8 as base


ARM_NAMES = ("bf16", "both", "gate_up_only", "down_only")


def gate_up_only_mlp(x, weights):
    quantized, scale = base.torch_npu.npu_dynamic_quant(x, dst_type=base.torch.int8)
    gate_up = base.torch_npu.npu_quant_matmul(
        quantized, weights["gate_up_int8"], weights["gate_up_scale"],
        pertoken_scale=scale, bias=None, output_dtype=base.torch.bfloat16,
    )
    hidden = base.torch_npu.npu_swiglu(gate_up)
    return base.functional.linear(hidden, weights["down_bf16"])


def down_only_mlp(x, weights):
    gate_up = base.functional.linear(x, weights["gate_up_bf16"])
    hidden = base.torch_npu.npu_swiglu(gate_up)
    quantized, scale = base.torch_npu.npu_dynamic_quant(hidden, dst_type=base.torch.int8)
    return base.torch_npu.npu_quant_matmul(
        quantized, weights["down_int8"], weights["down_scale"],
        pertoken_scale=scale, bias=None, output_dtype=base.torch.bfloat16,
    )


def round_order(index):
    rotation = (index // 2) % len(ARM_NAMES)
    order = list(ARM_NAMES[rotation:] + ARM_NAMES[:rotation])
    if index % 2:
        order.reverse()
    return order


def benchmark_shape(x, weights, args, device):
    """Measure four complete MLP chains with selectively quantized projections.

    Defaults to eight rotated/reversed rounds. Captured input defaults to its
    full token count. Numerical differences remain diagnostics for the later
    end-to-end quality evaluation. Outputs are local TP2 partial sums.
    """
    functions = {
        "bf16": base.bf16_mlp,
        "both": base.w8a8_mlp,
        "gate_up_only": gate_up_only_mlp,
        "down_only": down_only_mlp,
    }
    for _ in range(args.warmup):
        for function in functions.values():
            function(x, weights)
    base.torch.npu.synchronize(device)
    reference = functions["bf16"](x, weights)
    output_info = {"bf16": base.tensor_info(reference)}
    accuracy = {}
    for name in ARM_NAMES[1:]:
        candidate = functions[name](x, weights)
        base.torch.npu.synchronize(device)
        output_info[name] = base.tensor_info(candidate)
        accuracy[name] = base.compare_outputs(reference, candidate)
        del candidate
    del reference
    base.torch.npu.reset_peak_memory_stats(device)
    allocated_before = base.torch.npu.memory_allocated(device)
    rounds = []
    for index in range(args.rounds):
        order = round_order(index)
        arms = {
            name: base.measure_arm(functions[name], x, weights, args.iterations, device)
            for name in order
        }
        rounds.append({"round": index + 1, "order": order, "arms": arms})
    summaries = {}
    paired = {}
    for metric in ("device_ms", "synchronized_wall_ms"):
        summaries[metric] = {
            name: base.timing_summary([
                sample for row in rounds for sample in row["arms"][name][metric]
            ])
            for name in ARM_NAMES
        }
        paired[metric] = {}
        for name in ARM_NAMES[1:]:
            ratios = [
                statistics.median(row["arms"]["bf16"][metric])
                / statistics.median(row["arms"][name][metric])
                for row in rounds
            ]
            paired[metric][name] = {
                "speedup_bf16_over_candidate_by_round": ratios,
                "median_speedup": statistics.median(ratios),
                "candidate_faster_rounds": sum(value > 1 for value in ratios),
            }
    return {
        "tokens": x.shape[0],
        "input": base.tensor_info(x),
        "output": output_info,
        "accuracy": accuracy,
        "all_outputs_finite": all(
            row["reference_finite"] and row["candidate_finite"] for row in accuracy.values()
        ),
        "quantized_projections": {
            "bf16": [], "both": ["gate_up", "down"],
            "gate_up_only": ["gate_up"], "down_only": ["down"],
        },
        "rounds": rounds,
        "position_counts": {
            name: [sum(row["order"][position] == name for row in rounds) for position in range(4)]
            for name in ARM_NAMES
        },
        "summary": summaries,
        "paired_summary": paired,
        "memory": {
            "allocated_before_bytes": allocated_before,
            "peak_allocated_bytes": base.torch.npu.max_memory_allocated(device),
            "peak_delta_bytes": base.torch.npu.max_memory_allocated(device) - allocated_before,
            "reserved_bytes": base.torch.npu.memory_reserved(device),
            "scope": "process allocator during warmed timing, includes resident weights and input",
        },
    }


if __name__ == "__main__":
    raise SystemExit(base.main(benchmark_function=benchmark_shape, default_rounds=8))
