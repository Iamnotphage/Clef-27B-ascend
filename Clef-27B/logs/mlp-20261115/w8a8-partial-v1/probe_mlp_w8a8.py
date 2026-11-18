#!/usr/bin/env python3
"""Measure one TP2 MLP shard with real Clef weights on a single NPU.

The reference is BF16 linear -> BF16 SwiGLU -> BF16 linear. The candidate
uses the pinned Ascend W8A8_DYNAMIC linear contract twice around BF16 SwiGLU.
Its output is the rank-local down-projection partial sum, before TP reduction.

Weights use one-time symmetric per-output-channel absmax quantization. Scales
are stored in BF16, and those stored values are promoted to FP32 for quantizing
the weights, matching the service converter. Down
scales are computed over the full checkpoint input dimension before TP slicing.
The timed region contains both activation quantizations and both quant matmuls.
Weight preparation, transfers, accuracy calculations and TP reduction have
separate scope. A synthetic-input result describes this local microbenchmark.

An optional input file contains a CPU tensor, or a torch.save dictionary with
``input`` and optional ``metadata`` entries. Leading dimensions are flattened
into tokens. Captured inputs default to their full token count. Explicit token
counts use the first T rows without repetition. Captured layer/rank metadata
must agree with the selected weight shard.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
import time
import traceback
from pathlib import Path

import torch
import torch.nn.functional as functional
import torch_npu
from safetensors import safe_open


TP_SIZE = 2
ACL_FORMAT_FRACTAL_NZ = 29
ASCEND_SOURCE = Path("/workspace/yangchen/work/src/vllm-ascend")
PINNED_ASCEND_COMMIT = "89eb8a5e7a76d085b453a1aeadb6883e12d5ea2d"
SOURCE_FILES = (
    "vllm_ascend/quantization/methods/w8a8/w8a8_dynamic.py",
    "vllm_ascend/quantization/method_adapters.py",
    "vllm_ascend/ops/linear.py",
    "vllm_ascend/ops/activation.py",
    "vllm_ascend/utils.py",
    "vllm_ascend/worker/worker.py",
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_sha256(tensor: torch.Tensor) -> str:
    data = tensor.detach().cpu().contiguous().view(torch.uint8).numpy()
    return hashlib.sha256(memoryview(data)).hexdigest()


def tensor_info(tensor: torch.Tensor) -> dict:
    return {
        "shape": list(tensor.shape),
        "stride": list(tensor.stride()),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "contiguous": tensor.is_contiguous(),
        "npu_format": (
            int(torch_npu.get_npu_format(tensor)) if tensor.device.type == "npu" else None
        ),
        "logical_bytes": tensor.numel() * tensor.element_size(),
        "storage_bytes": tensor.untyped_storage().nbytes(),
    }


def source_provenance() -> dict:
    result = subprocess.run(
        ["git", "-C", str(ASCEND_SOURCE), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=False,
    )
    actual = result.stdout.strip() if result.returncode == 0 else None
    return {
        "expected_commit": PINNED_ASCEND_COMMIT,
        "actual_commit": actual,
        "commit_matches": actual == PINNED_ASCEND_COMMIT,
        "files": {
            relative: {
                "path": str(ASCEND_SOURCE / relative),
                "sha256": file_sha256(ASCEND_SOURCE / relative),
            }
            for relative in SOURCE_FILES
        },
        "linear_contract": "npu_dynamic_quant(int8) + npu_quant_matmul(output_dtype=BF16)",
        "weight_scale_dtype": "torch.bfloat16, matching BF16 params_dtype",
        "weight_layout": "W[N,K] -> W[K,N].contiguous() -> FRACTAL_NZ(29)",
        "weight_nz_mode": 1,
        "bf16_layout": "ND, matching weight_nz_mode=1 on A3",
        "activation": "torch_npu.npu_swiglu, gate followed by up along the last axis",
    }


def per_channel_scales(weight: torch.Tensor) -> torch.Tensor:
    """Bound CPU temporary storage while reducing the complete input axis."""
    scales = torch.empty(weight.shape[0], dtype=torch.float32)
    for start in range(0, weight.shape[0], 256):
        block = weight[start:start + 256].float()
        if not torch.isfinite(block).all():
            raise ValueError("checkpoint weight contains non-finite values")
        scales[start:start + 256] = block.abs().amax(dim=1) / 127.0
    scales.masked_fill_(scales == 0, 1.0)
    return scales


def quantize_weight(weight: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    quantized = torch.empty_like(weight, dtype=torch.int8)
    for start in range(0, weight.shape[0], 256):
        block = weight[start:start + 256].float()
        block.div_(scales[start:start + 256, None]).round_().clamp_(-127, 127)
        quantized[start:start + 256] = block.to(torch.int8)
    return quantized


def load_weights(model_dir: Path, layer: int, rank: int) -> tuple[dict, dict]:
    config_path = model_dir / "config.json"
    index_path = model_dir / "model.safetensors.index.json"
    config = json.loads(config_path.read_text())
    text_config = config["text_config"]
    hidden = text_config["hidden_size"]
    intermediate = text_config["intermediate_size"]
    if not 0 <= layer < text_config["num_hidden_layers"]:
        raise ValueError(f"layer {layer} is outside the model layer range")
    if text_config["hidden_act"] != "silu" or intermediate % TP_SIZE:
        raise ValueError("this probe requires a dense SiLU MLP divisible across TP2")
    weight_map = json.loads(index_path.read_text())["weight_map"]
    local_intermediate = intermediate // TP_SIZE
    start = rank * local_intermediate
    stop = start + local_intermediate
    projections = {}
    identities = {}
    for name in ("gate", "up", "down"):
        key = f"model.language_model.layers.{layer}.mlp.{name}_proj.weight"
        shard = model_dir / weight_map[key]
        expected_shape = (hidden, intermediate) if name == "down" else (intermediate, hidden)
        with safe_open(shard, framework="pt", device="cpu") as handle:
            full = handle.get_tensor(key)
            if tuple(full.shape) != expected_shape or full.dtype != torch.bfloat16:
                raise ValueError(f"unexpected checkpoint tensor {key}: {full.shape}, {full.dtype}")
            if name == "down":
                scales_fp32 = per_channel_scales(full)
                weight = full[:, start:stop].contiguous()
                shard_axis = 1
            else:
                weight = full[start:stop, :].clone().contiguous()
                scales_fp32 = per_channel_scales(weight)
                shard_axis = 0
            scales_bf16 = scales_fp32.to(torch.bfloat16).contiguous()
            if not (torch.isfinite(scales_bf16).all() and (scales_bf16 > 0).all()):
                raise ValueError(f"unrepresentable BF16 weight scale in {key}")
            quantized = quantize_weight(weight, scales_bf16.float())
            identities[name] = {
                "key": key,
                "shard_file": str(shard),
                "shard_file_bytes": shard.stat().st_size,
                "checkpoint_shape": list(full.shape),
                "checkpoint_dtype": str(full.dtype),
                "tp_slice_axis": shard_axis,
                "tp_slice_start": start,
                "tp_slice_stop": stop,
                "bf16_slice": tensor_info(weight),
                "bf16_slice_sha256": tensor_sha256(weight),
                "int8_slice_sha256": tensor_sha256(quantized),
                "bf16_scale_sha256": tensor_sha256(scales_bf16),
                "scale_shape": list(scales_bf16.shape),
                "scale_reduction_extent": expected_shape[1],
                "scale_min": float(scales_bf16.min()),
                "scale_max": float(scales_bf16.max()),
            }
            projections[name] = (weight, quantized, scales_bf16)
        del full
    gate, up, down = (projections[name] for name in ("gate", "up", "down"))
    packed = {
        "gate_up_bf16": torch.cat((gate[0], up[0]), dim=0),
        "down_bf16": down[0],
        "gate_up_int8": torch.cat((gate[1], up[1]), dim=0),
        "down_int8": down[1],
        "gate_up_scale": torch.cat((gate[2], up[2]), dim=0),
        "down_scale": down[2],
    }
    return packed, {
        "model_dir": str(model_dir),
        "config_sha256": file_sha256(config_path),
        "index_sha256": file_sha256(index_path),
        "layer": layer,
        "tp_size": TP_SIZE,
        "rank": rank,
        "hidden_size": hidden,
        "intermediate_size": intermediate,
        "local_intermediate_size": local_intermediate,
        "projections": identities,
        "quantization": {
            "method": "per-output-channel symmetric absmax, stored scale = BF16(max(abs(W))/127)",
            "weight_quantization_scale": "stored BF16 scale promoted to FP32 before division and rounding",
            "rounding": "torch.round, nearest with ties to even, clamp [-127,127]",
            "zero_weight_row_scale": 1.0,
            "stored_scale_dtype": "torch.bfloat16",
            "weight_offset": "zero; the pinned dynamic linear apply consumes weight_scale",
            "down_scale_scope": "full checkpoint K before TP input-column slicing",
        },
    }


def load_input(args, hidden_size: int) -> tuple[torch.Tensor, dict]:
    if args.input_file:
        path = args.input_file.expanduser().resolve()
        document = torch.load(path, map_location="cpu", weights_only=True)
        tensor = document["input"] if isinstance(document, dict) else document
        metadata = document.get("metadata", {}) if isinstance(document, dict) else {}
        if isinstance(metadata, dict):
            for field in ("layer", "rank"):
                requested = getattr(args, field)
                if field in metadata and metadata[field] != requested:
                    raise ValueError(
                        f"input metadata {field}={metadata[field]!r} disagrees with "
                        f"--{field} {requested}: {path}"
                    )
        identity = {
            "kind": "captured_activation",
            "file": str(path),
            "file_sha256": file_sha256(path),
            "metadata": json.loads(json.dumps(metadata, default=str)),
        }
    else:
        generator = torch.Generator(device="cpu").manual_seed(args.seed)
        tensor = torch.randn(max(args.tokens), hidden_size, generator=generator)
        identity = {
            "kind": "synthetic",
            "distribution": "standard normal FP32 converted to BF16",
            "seed": args.seed,
        }
    if not isinstance(tensor, torch.Tensor) or tensor.ndim < 2:
        raise ValueError("input must be a tensor with token dimensions and a hidden dimension")
    if tensor.shape[-1] != hidden_size or not tensor.is_floating_point():
        raise ValueError(f"input must have floating dtype and final dimension {hidden_size}")
    identity["source_tensor"] = tensor_info(tensor)
    tensor = tensor.reshape(-1, hidden_size).to(torch.bfloat16).contiguous()
    if tensor.shape[0] == 0:
        raise ValueError("input must contain at least one token")
    if args.tokens is None:
        args.tokens = [tensor.shape[0]]
    if tensor.shape[0] < max(args.tokens):
        raise ValueError(f"input has {tensor.shape[0]} rows, requested {max(args.tokens)} tokens")
    if not torch.isfinite(tensor).all():
        raise ValueError("input contains non-finite values after BF16 conversion")
    identity["selection"] = "first T rows of the flattened input for each token count"
    identity["available_tokens"] = tensor.shape[0]
    identity["token_selections"] = [
        {
            "tokens": tokens,
            "kind": "full" if tokens == tensor.shape[0] else "prefix",
            "row_range": [0, tokens],
        }
        for tokens in args.tokens
    ]
    return tensor, identity


def to_device_weights(weights: dict, device: torch.device) -> dict:
    prepared = {}
    for name, value in weights.items():
        if name.endswith("_int8"):
            value = value.transpose(0, 1).contiguous().to(device)
            prepared[name] = torch_npu.npu_format_cast(value, ACL_FORMAT_FRACTAL_NZ)
        else:
            prepared[name] = value.to(device)
    return prepared


def bf16_mlp(x: torch.Tensor, weights: dict) -> torch.Tensor:
    gate_up = functional.linear(x, weights["gate_up_bf16"])
    hidden = torch_npu.npu_swiglu(gate_up)
    return functional.linear(hidden, weights["down_bf16"])


def w8a8_mlp(x: torch.Tensor, weights: dict) -> torch.Tensor:
    quantized_x, input_scale = torch_npu.npu_dynamic_quant(x, dst_type=torch.int8)
    gate_up = torch_npu.npu_quant_matmul(
        quantized_x, weights["gate_up_int8"], weights["gate_up_scale"],
        pertoken_scale=input_scale, bias=None, output_dtype=torch.bfloat16,
    )
    hidden = torch_npu.npu_swiglu(gate_up)
    quantized_hidden, hidden_scale = torch_npu.npu_dynamic_quant(hidden, dst_type=torch.int8)
    return torch_npu.npu_quant_matmul(
        quantized_hidden, weights["down_int8"], weights["down_scale"],
        pertoken_scale=hidden_scale, bias=None, output_dtype=torch.bfloat16,
    )


def compare_outputs(reference: torch.Tensor, candidate: torch.Tensor) -> dict:
    reference = reference.detach().float().cpu().double().reshape(-1)
    candidate = candidate.detach().float().cpu().double().reshape(-1)
    finite_reference = torch.isfinite(reference)
    finite_candidate = torch.isfinite(candidate)
    result = {
        "elements": reference.numel(),
        "reference_finite": bool(finite_reference.all()),
        "candidate_finite": bool(finite_candidate.all()),
        "reference_nonfinite": int((~finite_reference).sum()),
        "candidate_nonfinite": int((~finite_candidate).sum()),
        "max_abs_error": None,
        "rmse": None,
        "relative_rmse": None,
        "cosine_similarity": None,
    }
    if result["reference_finite"] and result["candidate_finite"]:
        difference = candidate - reference
        rmse = float(difference.square().mean().sqrt())
        reference_rms = float(reference.square().mean().sqrt())
        reference_norm = float(reference.norm())
        candidate_norm = float(candidate.norm())
        result.update(
            max_abs_error=float(difference.abs().max()),
            rmse=rmse,
            reference_rms=reference_rms,
            candidate_rms=float(candidate.square().mean().sqrt()),
            relative_rmse=rmse / reference_rms if reference_rms else None,
            cosine_similarity=(
                float(torch.dot(reference, candidate)) / (reference_norm * candidate_norm)
                if reference_norm and candidate_norm else None
            ),
        )
    return result


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower, upper = math.floor(position), math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def timing_summary(values: list[float]) -> dict:
    return {
        "samples": len(values),
        "mean_ms": statistics.mean(values),
        "median_ms": statistics.median(values),
        "p95_ms": percentile(values, 0.95),
        "min_ms": min(values),
        "max_ms": max(values),
    }


def measure_arm(function, x, weights, iterations: int, device) -> dict:
    start = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)
    device_ms = []
    wall_ms = []
    # Create the device events before the measured calls.
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


def benchmark_shape(x, weights, args, device) -> dict:
    functions = {"bf16": bf16_mlp, "w8a8_dynamic": w8a8_mlp}
    for _ in range(args.warmup):
        for function in functions.values():
            function(x, weights)
    torch.npu.synchronize(device)
    reference = bf16_mlp(x, weights)
    candidate = w8a8_mlp(x, weights)
    torch.npu.synchronize(device)
    output_info = {"reference": tensor_info(reference), "candidate": tensor_info(candidate)}
    accuracy = compare_outputs(reference, candidate)
    del reference, candidate
    torch.npu.reset_peak_memory_stats(device)
    allocated_before = torch.npu.memory_allocated(device)
    rounds = []
    for index in range(args.rounds):
        order = ["bf16", "w8a8_dynamic"] if index % 2 == 0 else ["w8a8_dynamic", "bf16"]
        arms = {
            name: measure_arm(functions[name], x, weights, args.iterations, device)
            for name in order
        }
        rounds.append({"round": index + 1, "order": order, "arms": arms})
    summaries = {}
    paired = {}
    for metric in ("device_ms", "synchronized_wall_ms"):
        summaries[metric] = {
            name: timing_summary([
                sample for pair in rounds for sample in pair["arms"][name][metric]
            ])
            for name in functions
        }
        ratios = [
            statistics.median(pair["arms"]["bf16"][metric])
            / statistics.median(pair["arms"]["w8a8_dynamic"][metric])
            for pair in rounds
        ]
        paired[metric] = {
            "speedup_bf16_over_w8a8_by_round": ratios,
            "median_speedup": statistics.median(ratios),
            "w8a8_faster_rounds": sum(value > 1 for value in ratios),
        }
    return {
        "tokens": x.shape[0],
        "input": tensor_info(x),
        "output": output_info,
        "accuracy": accuracy,
        "rounds": rounds,
        "summary": summaries,
        "paired_summary": paired,
        "memory": {
            "allocated_before_bytes": allocated_before,
            "peak_allocated_bytes": torch.npu.max_memory_allocated(device),
            "peak_delta_bytes": torch.npu.max_memory_allocated(device) - allocated_before,
            "reserved_bytes": torch.npu.memory_reserved(device),
            "scope": "process allocator during warmed timing, includes resident weights and input",
        },
    }


def write_report(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def main(benchmark_function=benchmark_shape, default_rounds=6) -> int:
    parser = argparse.ArgumentParser(description=benchmark_function.__doc__ or __doc__)
    parser.add_argument("--device", default="0", help="logical NPU index or npu:<index>")
    parser.add_argument("--model-dir", type=Path, default=Path("/workspace/yangchen/models/Clef-27B"))
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--rank", type=int, choices=range(TP_SIZE), default=0)
    parser.add_argument(
        "--tokens", type=int, nargs="+",
        help="defaults to all captured rows with --input-file, or synthetic 307 and 1024",
    )
    parser.add_argument("--input-file", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--rounds", type=int, default=default_rounds)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    if args.tokens is None and args.input_file is None:
        args.tokens = [307, 1024]
    if ((args.tokens is not None and min(args.tokens) < 1)
            or args.warmup < 1 or args.rounds < 2 or args.iterations < 1):
        parser.error("tokens, warmup and iterations must be positive; rounds must be at least two")
    if args.tokens is not None and len(set(args.tokens)) != len(args.tokens):
        parser.error("token counts must be unique")
    device_name = args.device if args.device.startswith("npu:") else f"npu:{args.device}"
    report = {
        "schema_version": 1,
        "status": "running",
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "device": device_name,
        "torch": torch.__version__,
        "torch_npu": torch_npu.__version__,
        "scope": "single-layer TP2 rank-local MLP partial output before all-reduce",
        "quality_scope": "local numerical comparison on the recorded input, full-model quality requires evaluation",
        "timing_scope": "both linears, BF16 SwiGLU and candidate activation quantization, preloaded weights/input",
        "timing": {
            "warmup_per_arm_per_shape": args.warmup,
            "rounds": args.rounds,
            "iterations_per_arm_per_round": args.iterations,
            "order": "Exact arm order is recorded in every shape's rounds",
            "wall": "perf_counter enclosing event records, full chain and device synchronize",
            "device": "NPU event elapsed time around the full chain on the current stream",
        },
        "shapes": [],
    }
    try:
        torch.set_num_threads(4)
        report["plugin"] = source_provenance()
        with torch.inference_mode():
            weights_cpu, report["weights"] = load_weights(args.model_dir.resolve(), args.layer, args.rank)
            input_cpu, report["input_source"] = load_input(args, report["weights"]["hidden_size"])
            device = torch.device(device_name)
            torch.npu.config.allow_internal_format = True
            report["allow_internal_format"] = str(
                torch_npu._C._npu_getOption("ALLOW_INTERNAL_FORMAT"))
            torch.npu.set_device(device)
            report["device_name"] = torch.npu.get_device_name(device)
            weights = to_device_weights(weights_cpu, device)
            del weights_cpu
            torch.npu.synchronize(device)
            report["device_weights"] = {name: tensor_info(value) for name, value in weights.items()}
            for name in ("gate_up_int8", "down_int8"):
                if report["device_weights"][name]["npu_format"] != ACL_FORMAT_FRACTAL_NZ:
                    raise RuntimeError(f"{name} did not convert to FRACTAL_NZ")
            report["resident_weight_logical_bytes"] = sum(
                value.numel() * value.element_size() for value in weights.values()
            )
            for tokens in args.tokens:
                selected = input_cpu[:tokens].contiguous()
                x = selected.to(device)
                result = benchmark_function(x, weights, args, device)
                result["input_sha256"] = tensor_sha256(selected)
                report["shapes"].append(result)
                write_report(args.report, report)
                print(json.dumps({"tokens": tokens, "accuracy": result["accuracy"],
                                  "paired_summary": result["paired_summary"]}), flush=True)
                del x
        report["status"] = "complete"
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        report["traceback"] = traceback.format_exc()
        print(report["error"], file=sys.stderr, flush=True)
    report["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    write_report(args.report, report)
    return 0 if report["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
