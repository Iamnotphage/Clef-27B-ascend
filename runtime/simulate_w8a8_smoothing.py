#!/usr/bin/env python3
"""Offline CPU simulation: W8A8 dynamic quantization vs activation smoothing.

The target is the gate/up projection of one real captured MLP input.  All
matmuls run in float32 over BF16 checkpoint values, so the measured error is
the quantization error itself, not CPU BF16 rounding.

Schemes:
  bf16            reference (BF16 values, float32 matmul)
  w8a8            per-token activation absmax/127 + per-output-channel weight
  w8a16           weight-only int8, BF16 activations
  w8a8_smooth_A   SmoothQuant-style per-input-channel scaling before W8A8
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as functional
from safetensors import safe_open

TP_SIZE = 2
ALPHAS = (0.3, 0.5, 0.7)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_capture(path: Path) -> tuple[torch.Tensor, dict]:
    document = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(document, dict):
        return document["input"], document.get("metadata", {})
    return document, {}


def load_projection(model_dir: Path, layer: int, rank: int, name: str) -> torch.Tensor:
    index = json.loads((model_dir / "model.safetensors.index.json").read_text())
    key = f"model.language_model.layers.{layer}.mlp.{name}_proj.weight"
    with safe_open(model_dir / index["weight_map"][key], framework="pt", device="cpu") as handle:
        full = handle.get_tensor(key)
    if name == "down":
        local = full.shape[1] // TP_SIZE
        return full[:, rank * local:(rank + 1) * local].float()
    local = full.shape[0] // TP_SIZE
    return full[rank * local:(rank + 1) * local, :].float()


def quant_per_token(activation: torch.Tensor) -> torch.Tensor:
    scale = activation.abs().amax(dim=1, keepdim=True).clamp_min(1e-8) / 127.0
    return (activation / scale).round().clamp_(-127, 127) * scale


def quant_per_channel(weight: torch.Tensor) -> torch.Tensor:
    scale = weight.abs().amax(dim=1, keepdim=True).clamp_min(1e-8) / 127.0
    return (weight / scale).round().clamp_(-127, 127) * scale


def smoothing_scales(activation: torch.Tensor, weight: torch.Tensor, alpha: float) -> torch.Tensor:
    x_max = activation.abs().amax(dim=0).clamp_min(1e-5)
    w_max = weight.abs().amax(dim=0).clamp_min(1e-5)
    return (x_max.pow(alpha) / w_max.pow(1.0 - alpha)).clamp_min(1e-5)


def relative_rmse(candidate: torch.Tensor, reference: torch.Tensor) -> float:
    return float((candidate - reference).norm() / reference.norm())


def cosine(candidate: torch.Tensor, reference: torch.Tensor) -> float:
    # float64 accumulation: a 5M-element float32 dot loses ~1e-3 of cosine.
    a = candidate.flatten().double()
    b = reference.flatten().double()
    return float(torch.dot(a, b) / (a.norm() * b.norm()))


def evaluate(activation, gate, up, mode, alpha=None):
    if mode == "bf16":
        gate_q, up_q = gate, up
    elif mode == "w8a8":
        x_q = quant_per_token(activation)
        gate_q, up_q = quant_per_channel(gate), quant_per_channel(up)
        activation = x_q
    elif mode == "w8a16":
        gate_q, up_q = quant_per_channel(gate), quant_per_channel(up)
    elif mode == "w8a8_smooth":
        scale_gate = smoothing_scales(activation, gate, alpha)
        scale_up = smoothing_scales(activation, up, alpha)
        # Upper bound: each projection gets its own smoothing vector.
        out_gate = quant_per_token(activation / scale_gate) @ quant_per_channel(gate * scale_gate).T
        out_up = quant_per_token(activation / scale_up) @ quant_per_channel(up * scale_up).T
        return out_gate, out_up
    elif mode == "w8a8_smooth_shared":
        # Implementable with one vector folded into the preceding norm: both
        # projections share the smoothing scale.
        w_max = torch.maximum(gate.abs().amax(dim=0), up.abs().amax(dim=0)).clamp_min(1e-5)
        x_max = activation.abs().amax(dim=0).clamp_min(1e-5)
        scale = (x_max.pow(alpha) / w_max.pow(1.0 - alpha)).clamp_min(1e-5)
        return (
            quant_per_token(activation / scale) @ quant_per_channel(gate * scale).T,
            quant_per_token(activation / scale) @ quant_per_channel(up * scale).T,
        )
    else:
        raise ValueError(mode)
    return activation @ gate_q.T, activation @ up_q.T


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path,
                        default=Path("/workspace/yangchen/models/Clef-27B"))
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--rank", type=int, choices=(0, 1), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    activation, metadata = load_capture(args.input)
    if activation.dtype != torch.bfloat16:
        raise SystemExit(f"unexpected capture dtype {activation.dtype}")
    activation = activation.float()
    if activation.shape[-1] != 5120:
        raise SystemExit(f"unexpected hidden size {activation.shape}")
    gate = load_projection(args.model_dir, args.layer, args.rank, "gate")
    up = load_projection(args.model_dir, args.layer, args.rank, "up")

    results = {}
    reference_gate, reference_up = evaluate(activation, gate, up, "bf16")
    reference_hidden = functional.silu(reference_gate) * reference_up
    for mode, alpha in (
        [("bf16", None), ("w8a16", None), ("w8a8", None)]
        + [("w8a8_smooth", value) for value in ALPHAS]
        + [("w8a8_smooth_shared", value) for value in ALPHAS]
    ):
        gate_out, up_out = evaluate(activation, gate, up, mode, alpha)
        hidden = functional.silu(gate_out) * up_out
        label = mode if alpha is None else f"{mode}_{alpha}"
        results[label] = {
            "gate_rel_rmse": relative_rmse(gate_out, reference_gate),
            "up_rel_rmse": relative_rmse(up_out, reference_up),
            "hidden_rel_rmse": relative_rmse(hidden, reference_hidden),
            "hidden_cosine": cosine(hidden, reference_hidden),
        }

    channel_absmax = activation.abs().amax(dim=0)
    top_channels = torch.topk(channel_absmax, 5)
    scales_05 = smoothing_scales(activation, gate, 0.5)
    report = {
        "input": str(args.input),
        "input_sha256": sha256_file(args.input),
        "model_dir": str(args.model_dir),
        "layer": args.layer,
        "rank": args.rank,
        "tokens": int(activation.shape[0]),
        "metadata": metadata,
        "activation_channel_top5": [
            {"channel": int(index), "absmax": float(value)}
            for value, index in zip(top_channels.values, top_channels.indices)
        ],
        "smooth_scale_alpha_0_5_channel_3994": float(scales_05[3994]),
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
