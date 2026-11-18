#!/usr/bin/env python3
"""Probe SwiGLU npu_ffn support, then replay real TP2 MLP weights and inputs.

The default interface phase isolates each dtype in a subprocess. The benchmark
phase runs those probes first, then compares complete local MLP chains. Weight
conversion is performed once during preparation. Input/output casts are included
in the explicitly named BF16-to-FP16-to-BF16 variants. TP all-reduce is outside
this single-device experiment.
Layer/rank default to 0/0. Captured inputs use their full token count by default.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time
import traceback


PROBE_MARKER = "CLEF_FFN_PROBE_JSON="
GATE_ORDER_ERROR_RATIO = 10.0


def initialize_npu(torch, torch_npu, device):
    before = torch_npu._C._npu_getOption("ALLOW_INTERNAL_FORMAT")
    # Match vllm_ascend.worker.worker before device setup and weight creation.
    torch.npu.config.allow_internal_format = True
    after = torch_npu._C._npu_getOption("ALLOW_INTERNAL_FORMAT")
    before = before.decode() if isinstance(before, bytes) else before
    after = after.decode() if isinstance(after, bytes) else after
    if after != "enable":
        raise RuntimeError(f"ALLOW_INTERNAL_FORMAT remained {after!r} after enabling it")
    torch.npu.set_device(device)
    return {"requested": True, "option_before": before, "option_after": after}


def compare_outputs(actual, reference, torch, atol, rtol):
    actual = actual.detach().float().cpu()
    reference = reference.detach().float().cpu()
    actual_finite = bool(torch.isfinite(actual).all())
    reference_finite = bool(torch.isfinite(reference).all())
    result = {
        "actual_finite": actual_finite,
        "reference_finite": reference_finite,
        "atol": atol,
        "rtol": rtol,
        "within_tolerance": False,
        "max_absolute_difference": None,
        "rmse": None,
        "relative_rmse": None,
        "reference_max_absolute": None,
        "normalized_max_error": None,
    }
    if actual_finite and reference_finite:
        difference = actual - reference
        rmse = difference.square().mean().sqrt().item()
        reference_rms = reference.square().mean().sqrt().item()
        maximum_error = difference.abs().max().item()
        reference_maximum = reference.abs().max().item()
        result.update(
            within_tolerance=bool(torch.allclose(actual, reference, atol=atol, rtol=rtol)),
            max_absolute_difference=maximum_error,
            rmse=rmse,
            relative_rmse=rmse / max(reference_rms, 1e-30),
            reference_max_absolute=reference_maximum,
            normalized_max_error=maximum_error / max(reference_maximum, 1e-30),
        )
    return result


def numerical_check_passes(comparison, args):
    return bool(
        comparison["actual_finite"] and comparison["reference_finite"]
        and comparison["relative_rmse"] <= args.max_relative_rmse
        and comparison["normalized_max_error"] <= args.max_normalized_error
    )


def identify_gate_order(first_half, second_half, args):
    if first_half["rmse"] is None or second_half["rmse"] is None:
        return "inconclusive"
    if numerical_check_passes(first_half, args) and (
        first_half["rmse"] * GATE_ORDER_ERROR_RATIO < second_half["rmse"]
    ):
        return "first_half"
    if numerical_check_passes(second_half, args) and (
        second_half["rmse"] * GATE_ORDER_ERROR_RATIO < first_half["rmse"]
    ):
        return "second_half"
    return "inconclusive"


def interface_worker(args):
    """Keep failed ACLNN calls in a disposable process."""
    result = {"dtype": args._probe_dtype, "status": "error", "shape": [4, 64, 128, 64]}
    try:
        import torch
        import torch_npu
        import torch.nn.functional as functional

        torch.set_num_threads(1)
        torch.manual_seed(args.seed)
        result["allow_internal_format"] = initialize_npu(torch, torch_npu, args.device)
        dtype = getattr(torch, args._probe_dtype)
        device = f"npu:{args.device}"
        # A preactivation std near 2 separates the two SwiGLU gate orders.
        x = (torch.randn(4, 64) * 0.5).to(dtype)
        packed = (torch.randn(128, 64) * 0.5).to(dtype)
        down = (torch.randn(64, 64) * 0.5).to(dtype)
        gate, up = functional.linear(x.float(), packed.float()).chunk(2, dim=-1)
        expected = functional.linear(functional.silu(gate) * up, down.float())
        reversed_gate = functional.linear(gate * functional.silu(up), down.float())
        with torch.inference_mode():
            x_npu = x.to(device)
            weight1_npu = packed.t().contiguous().to(device)
            weight2_npu = down.t().contiguous().to(device)
            result["npu_formats"] = {
                "input": torch_npu.get_npu_format(x_npu),
                "weight1": torch_npu.get_npu_format(weight1_npu),
                "weight2": torch_npu.get_npu_format(weight2_npu),
            }
            actual = torch_npu.npu_ffn(
                x_npu, weight1_npu, weight2_npu, "swiglu",
                inner_precise=0 if dtype == torch.bfloat16 else 1,
            )
            torch.npu.synchronize()
            result["silu_first_half_times_second_half"] = compare_outputs(
                actual, expected, torch, args.atol, args.rtol,
            )
            result["first_half_times_silu_second_half"] = compare_outputs(
                actual, reversed_gate, torch, args.atol, args.rtol,
            )
            result["gate_order"] = identify_gate_order(
                result["silu_first_half_times_second_half"],
                result["first_half_times_silu_second_half"], args,
            )
            result["gate_order_criteria"] = {
                "max_relative_rmse": args.max_relative_rmse,
                "max_normalized_error": args.max_normalized_error,
                "minimum_alternative_rmse_ratio": GATE_ORDER_ERROR_RATIO,
                "probe_input_and_weight_std": 0.5,
            }
            result["status"] = "supported"
            result["output_shape"] = list(actual.shape)
            result["output_dtype"] = str(actual.dtype)
            result["torch"] = torch.__version__
            result["torch_npu"] = torch_npu.__version__
    except Exception as error:
        result["error_type"] = type(error).__name__
        result["error"] = str(error)
        result["traceback"] = traceback.format_exc()
    print(PROBE_MARKER + json.dumps(result, allow_nan=False), flush=True)


def run_interface_probes(args):
    results = {}
    for dtype in ("bfloat16", "float16"):
        command = [
            sys.executable, str(Path(__file__).resolve()),
            "--_probe-dtype", dtype, "--device", str(args.device),
            "--seed", str(args.seed), "--atol", str(args.atol), "--rtol", str(args.rtol),
            "--max-relative-rmse", str(args.max_relative_rmse),
            "--max-normalized-error", str(args.max_normalized_error),
        ]
        try:
            process = subprocess.run(
                command, capture_output=True, text=True, timeout=args.probe_timeout,
            )
        except subprocess.TimeoutExpired as error:
            results[dtype] = {"status": "timeout", "timeout_seconds": error.timeout}
            continue
        records = [line[len(PROBE_MARKER):] for line in process.stdout.splitlines()
                   if line.startswith(PROBE_MARKER)]
        result = json.loads(records[-1]) if records else {"status": "process_failed"}
        result["exit_code"] = process.returncode
        if process.returncode:
            result["status"] = "process_failed"
        result["stderr_tail"] = process.stderr[-6000:]
        if not records:
            result["stdout_tail"] = process.stdout[-6000:]
        results[dtype] = result
    return results


def load_weights(model_dir, torch, layer, rank):
    from safetensors import safe_open

    config = json.loads((model_dir / "config.json").read_text())
    text_config = config.get("text_config", config)
    hidden = text_config["hidden_size"]
    intermediate = text_config["intermediate_size"]
    if layer >= text_config["num_hidden_layers"]:
        raise ValueError(f"Layer {layer} exceeds the configured layer count")
    if text_config.get("hidden_act") != "silu" or intermediate % 2:
        raise ValueError("The probe requires a SiLU-gated MLP with an even intermediate size")
    index_path = model_dir / "model.safetensors.index.json"
    weight_map = json.loads(index_path.read_text())["weight_map"]
    shard_width = intermediate // 2
    shard_start = rank * shard_width
    shard_end = shard_start + shard_width
    tensors = {}
    evidence = {}
    for projection in ("gate_proj", "up_proj", "down_proj"):
        suffix = f".layers.{layer}.mlp.{projection}.weight"
        names = [name for name in weight_map if name.endswith(suffix)]
        if len(names) != 1:
            raise ValueError(f"Expected one layer-{layer} {projection} tensor, found {names}")
        name = names[0]
        source = model_dir / weight_map[name]
        with safe_open(str(source), framework="pt", device="cpu") as handle:
            view = handle.get_slice(name)
            expected = [hidden, intermediate] if projection == "down_proj" else [intermediate, hidden]
            if view.get_shape() != expected:
                raise ValueError(f"Unexpected shape for {name}: {view.get_shape()}, expected {expected}")
            tensor = view[:, shard_start:shard_end] if projection == "down_proj" else view[shard_start:shard_end, :]
            tensor = tensor.contiguous()
        if tensor.dtype != torch.bfloat16:
            raise ValueError(f"Expected BF16 source weights for {name}, found {tensor.dtype}")
        tensors[projection] = tensor
        evidence[name] = {
            "source": str(source), "full_shape": expected, "rank_shape": list(tensor.shape),
            "tp_size": 2, "tp_rank": rank,
            "sha256_rank_tensor_bytes": hashlib.sha256(
                memoryview(tensor.view(torch.uint8).numpy())
            ).hexdigest(),
        }
    packed = torch.cat((tensors["gate_proj"], tensors["up_proj"]), dim=0)
    return packed, tensors["down_proj"], {
        "hidden_size": hidden, "intermediate_size": intermediate,
        "layer": layer, "tp_size": 2, "tp_rank": rank, "weights": evidence,
        "index_sha256": hashlib.sha256(index_path.read_bytes()).hexdigest(),
    }


def load_captured_input(args, hidden_size, torch):
    saved = torch.load(args.input_file, map_location="cpu", weights_only=True)
    if not isinstance(saved, dict) or not isinstance(saved.get("metadata"), dict):
        raise ValueError("Captured input must contain an 'input' Tensor and a metadata dict")
    input_cpu = saved.get("input")
    metadata = saved["metadata"]
    if not isinstance(input_cpu, torch.Tensor) or input_cpu.ndim != 2:
        raise ValueError("Captured input must be a 2D Tensor")
    if input_cpu.shape[1] != hidden_size or input_cpu.dtype != torch.bfloat16:
        raise ValueError("Captured input must be BF16 with the model hidden dimension")
    available_tokens = input_cpu.shape[0]
    if available_tokens < 1 or not bool(torch.isfinite(input_cpu).all()):
        raise ValueError("Captured input must contain finite values and at least one token")
    expected_metadata = {
        "layer": args.layer, "rank": args.rank,
        "shape": list(input_cpu.shape), "dtype": str(input_cpu.dtype),
        "input_token_count": available_tokens,
    }
    for field, expected in expected_metadata.items():
        if metadata.get(field) != expected:
            raise ValueError(f"Capture metadata {field}={metadata.get(field)!r}, expected {expected!r}")
    if not isinstance(metadata.get("request_name"), str) or not metadata["request_name"].strip():
        raise ValueError("Capture metadata requires a nonempty request_name")
    if "prefix" in metadata and not str(metadata["prefix"]).endswith(f".layers.{args.layer}.mlp"):
        raise ValueError("Capture metadata prefix disagrees with the selected layer")
    tokens = args.tokens if args.tokens is not None else [available_tokens]
    if max(tokens) > available_tokens:
        raise ValueError(f"Capture has {available_tokens} tokens, requested {max(tokens)}")
    return input_cpu, metadata, tokens


def measure(function, torch, warmup, iterations):
    for _ in range(warmup):
        output = function()
    torch.npu.synchronize()
    start = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)
    wall_start = time.perf_counter()
    start.record()
    for _ in range(iterations):
        output = function()
    end.record()
    torch.npu.synchronize()
    wall_seconds = time.perf_counter() - wall_start
    return {
        "device_event_mean_ms": start.elapsed_time(end) / iterations,
        "synchronized_wall_mean_ms": wall_seconds * 1000 / iterations,
        "warmup": warmup, "iterations": iterations,
    }


def benchmark(args):
    import torch
    import torch_npu
    import torch.nn.functional as functional

    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    device = f"npu:{args.device}"
    packed_cpu, down_cpu, identity = load_weights(args.model_dir, torch, args.layer, args.rank)
    input_cpu = None
    input_metadata = None
    tokens_to_run = args.tokens if args.tokens is not None else [307, 1024]
    if args.input_file:
        input_cpu, input_metadata, tokens_to_run = load_captured_input(args, identity["hidden_size"], torch)
    format_initialization = initialize_npu(torch, torch_npu, args.device)
    preparation_start = time.perf_counter()
    packed_bf16 = packed_cpu.to(device)
    down_bf16 = down_cpu.to(device)
    packed_fp16 = packed_bf16.to(torch.float16)
    down_fp16 = down_bf16.to(torch.float16)
    gate_fp16, up_fp16 = packed_fp16.chunk(2, dim=0)
    # npu_ffn SwiGLU applies SiLU to the second half of its first matmul.
    packed_kn_fp16 = torch.cat((up_fp16, gate_fp16), dim=0).t().contiguous()
    down_kn_fp16 = down_fp16.t().contiguous()
    torch.npu.synchronize()
    preparation_seconds = time.perf_counter() - preparation_start
    result = {
        "model": identity,
        "allow_internal_format": format_initialization,
        "input_source": str(args.input_file) if args.input_file else "seeded_synthetic_normal",
        "input_metadata": input_metadata,
        "captured_tokens": input_cpu.shape[0] if input_cpu is not None else None,
        "tokens_to_run": tokens_to_run,
        "synthetic_input_std": args.input_std if input_cpu is None else None,
        "preparation_seconds_including_weight_copies_and_casts": preparation_seconds,
        "resident_weight_tensor_bytes": sum(
            value.numel() * value.element_size() for value in
            (packed_bf16, down_bf16, packed_fp16, down_fp16, packed_kn_fp16, down_kn_fp16)
        ),
        "weight_layout": {"decomposed": "contiguous [out,in] ND", "fused": "contiguous [in,out] ND"},
        "packed_weight_order": {"decomposed": ["gate", "up"], "fused": ["up", "gate"]},
        "numerical_diagnostics": {
            "max_relative_rmse": args.max_relative_rmse,
            "max_normalized_error": args.max_normalized_error,
            "require_all_outputs_finite": True,
            "timing_policy": "Finite outputs are timed; end-to-end evaluation determines adoption",
        },
        "weight_npu_formats": {
            name: torch_npu.get_npu_format(value) for name, value in {
                "packed_bf16": packed_bf16, "down_bf16": down_bf16,
                "packed_fp16": packed_fp16, "down_fp16": down_fp16,
                "packed_kn_fp16": packed_kn_fp16, "down_kn_fp16": down_kn_fp16,
            }.items()
        },
        "cases": [],
        "scope": f"Local TP2 rank{args.rank} output before all-reduce; warmed repeated layer{args.layer} weights",
    }
    for tokens in tokens_to_run:
        source = input_cpu[:tokens] if input_cpu is not None else (
            torch.randn(tokens, identity["hidden_size"]) * args.input_std
        ).to(torch.bfloat16)
        x_bf16 = source.contiguous().to(device)
        x_fp16 = x_bf16.to(torch.float16)

        def decomposed(x, packed, down):
            return functional.linear(torch_npu.npu_swiglu(functional.linear(x, packed)), down)

        def fused(x):
            return torch_npu.npu_ffn(x, packed_kn_fp16, down_kn_fp16, "swiglu", inner_precise=1)

        functions = {
            "bf16_decomposed": lambda: decomposed(x_bf16, packed_bf16, down_bf16),
            "fp16_decomposed": lambda: decomposed(x_fp16, packed_fp16, down_fp16),
            "fp16_fused": lambda: fused(x_fp16),
            "bf16_fp16_bf16_decomposed": lambda: decomposed(
                x_bf16.to(torch.float16), packed_fp16, down_fp16,
            ).to(torch.bfloat16),
            "bf16_fp16_bf16_fused": lambda: fused(x_bf16.to(torch.float16)).to(torch.bfloat16),
        }
        torch.npu.synchronize()
        torch.npu.reset_peak_memory_stats()
        outputs = {name: function().detach().float().cpu() for name, function in functions.items()}
        quality = {
            "fp16_fused_vs_fp16_decomposed": compare_outputs(
                outputs["fp16_fused"], outputs["fp16_decomposed"], torch, args.atol, args.rtol,
            ),
            "converted_decomposed_vs_bf16": compare_outputs(
                outputs["bf16_fp16_bf16_decomposed"], outputs["bf16_decomposed"], torch, args.atol, args.rtol,
            ),
            "converted_fused_vs_bf16": compare_outputs(
                outputs["bf16_fp16_bf16_fused"], outputs["bf16_decomposed"], torch, args.atol, args.rtol,
            ),
        }
        all_finite = all(bool(torch.isfinite(value).all()) for value in outputs.values())
        fp16_equivalent = numerical_check_passes(quality["fp16_fused_vs_fp16_decomposed"], args)
        case = {
            "tokens": tokens, "status": "numerical_check_failed", "input_shape": list(x_bf16.shape),
            "input_selection": "synthetic" if input_cpu is None else (
                "full_capture" if tokens == input_cpu.shape[0] else "explicit_prefix"
            ),
            "input_sha256": hashlib.sha256(
                memoryview(source.contiguous().view(torch.uint8).numpy())
            ).hexdigest(),
            "input_stride": list(x_bf16.stride()), "input_npu_format": torch_npu.get_npu_format(x_bf16),
            "quality": quality, "all_outputs_finite": all_finite,
            "performance_interpretation_valid": all_finite,
            "fp16_fused_numerical_diagnostic_passed": fp16_equivalent,
            "bf16_replacement_quality_passed": numerical_check_passes(quality["converted_fused_vs_bf16"], args),
            "torch_peak_allocated_bytes": torch.npu.max_memory_allocated(),
            "torch_peak_reserved_bytes": torch.npu.max_memory_reserved(),
        }
        result["cases"].append(case)
        if not case["performance_interpretation_valid"]:
            case["reason"] = "Nonfinite output detected; numerical diagnostics retained and timing skipped"
            continue
        timings = {name: [] for name in functions}
        for repeat in range(args.repeats):
            order = list(functions)
            if repeat % 2:
                order.reverse()
            for name in order:
                record = measure(functions[name], torch, args.warmup, args.iterations)
                record.update(repeat=repeat, order=order)
                timings[name].append(record)
        medians = {name: statistics.median(row["device_event_mean_ms"] for row in rows)
                   for name, rows in timings.items()}
        case.update({
            "status": "completed", "timings": timings, "device_median_ms": medians,
            "fp16_fused_speedup": medians["fp16_decomposed"] / medians["fp16_fused"],
            "converted_fused_vs_bf16_speedup": medians["bf16_decomposed"] / medians["bf16_fp16_bf16_fused"],
            "fused_io_cast_overhead_ms": medians["bf16_fp16_bf16_fused"] - medians["fp16_fused"],
            "torch_peak_allocated_bytes": torch.npu.max_memory_allocated(),
            "torch_peak_reserved_bytes": torch.npu.max_memory_reserved(),
        })
    result["completed_cases"] = sum(case["status"] == "completed" for case in result["cases"])
    result["numerical_checks_passed"] = all(case["status"] != "numerical_check_failed" for case in result["cases"])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--model-dir", type=Path, default=Path("/workspace/yangchen/models/Clef-27B"))
    parser.add_argument("--phase", choices=("interface", "benchmark"), default="interface")
    parser.add_argument("--input-file", type=Path)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--rank", type=int, choices=(0, 1), default=0)
    parser.add_argument("--tokens", nargs="+", type=int, help="Defaults to full captured input, or 307 1024 for synthetic input")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--input-std", type=float, default=0.1)
    parser.add_argument("--atol", type=float, default=0.001)
    parser.add_argument("--rtol", type=float, default=0.01)
    parser.add_argument("--max-relative-rmse", type=float, default=0.01)
    parser.add_argument("--max-normalized-error", type=float, default=0.02)
    parser.add_argument("--probe-timeout", type=int, default=120)
    parser.add_argument("--_probe-dtype", choices=("bfloat16", "float16"), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if min((args.tokens or []) + [args.warmup, args.iterations, args.repeats, args.probe_timeout]) < 1:
        parser.error("tokens, warmup, iterations, repeats and timeout must be positive")
    if args.layer < 0:
        parser.error("layer must be nonnegative")
    if args.input_std <= 0 or min(args.atol, args.rtol, args.max_relative_rmse, args.max_normalized_error) < 0:
        parser.error("input-std must be positive and tolerances must be nonnegative")
    if args._probe_dtype:
        interface_worker(args)
        return 0
    started = time.perf_counter()
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(), "phase": args.phase,
        "device": args.device, "completed": False,
        "documented_constraint": "SwiGLU npu_ffn supports ungrouped FP16 with inner_precise=1; BF16 support is probed",
        "numerical_tolerance_scope": "Local tensor diagnostic; end-to-end decision quality requires separate evaluation",
        "interface": run_interface_probes(args),
    }
    exit_code = 0
    try:
        fp16 = report["interface"]["float16"]
        if fp16["status"] != "supported" or fp16.get("gate_order") != "second_half":
            raise RuntimeError("FP16 SwiGLU must identify the second-half gate with decisive relative error separation")
        if args.phase == "benchmark":
            import torch
            with torch.inference_mode():
                report["benchmark"] = benchmark(args)
            if not report["benchmark"]["numerical_checks_passed"]:
                raise RuntimeError("Nonfinite output detected; diagnostics retained and timing skipped for those cases")
        report["completed"] = True
    except Exception as error:
        exit_code = 1
        report["error_type"] = type(error).__name__
        report["error"] = str(error)
        report["traceback"] = traceback.format_exc()
    report["wall_seconds"] = time.perf_counter() - started
    payload = json.dumps(report, indent=2, allow_nan=False, default=str) + "\n"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.report.with_name(args.report.name + ".tmp")
        temporary.write_text(payload)
        temporary.replace(args.report)
    print(payload, end="")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
