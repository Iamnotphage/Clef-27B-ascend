#!/usr/bin/env python3
"""Compare BF16 ND and FRACTAL_NZ weights for the local TP2 MLP chain.

Reuses the captured inputs and weight slicing from ``probe_mlp_ffn.py``.
Every case runs the real chain ``linear -> npu_swiglu -> linear``.  Two timing
modes are reported:

* ``warm``: repeated calls on the same resident weights (what the earlier
  local probes measured);
* ``cold``: a 256 MiB read flushes the 192 MiB L2 before each call, which
  approximates the service's layer-by-layer weight streaming.

Nominal bandwidth counts unique weight bytes divided by the MLP time; it is
not a byte-level HBM counter.
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


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


HERE = Path(__file__).resolve().parent
probe = load("probe_mlp_ffn_for_nz", HERE / "probe_mlp_ffn.py")
NZ_FORMAT = 29


def diff(actual, reference):
    actual_f = actual.detach().float()
    reference_f = reference.detach().float()
    difference = actual_f - reference_f
    rmse = difference.square().mean().sqrt().item()
    reference_rms = reference_f.square().mean().sqrt().item()
    return {
        "bit_equal": bool(torch.equal(actual, reference)),
        "max_abs": float(difference.abs().max().item()),
        "relative_rmse": rmse / max(reference_rms, 1e-30),
    }


def measured(function, warmup, iterations, flush=None):
    """Return (warm_ms, cold_median_ms) for one function."""
    for _ in range(warmup):
        function()
    torch.npu.synchronize()
    start = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        function()
    end.record()
    torch.npu.synchronize()
    warm_ms = start.elapsed_time(end) / iterations
    if flush is None:
        return warm_ms, None
    cold = []
    for _ in range(max(3, iterations // 2)):
        flush()
        torch.npu.synchronize()
        begin = torch.npu.Event(enable_timing=True)
        finish = torch.npu.Event(enable_timing=True)
        begin.record()
        function()
        finish.record()
        torch.npu.synchronize()
        cold.append(begin.elapsed_time(finish))
    return warm_ms, statistics.median(cold)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=Path("/workspace/yangchen/models/Clef-27B"))
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--input-file", type=Path, default=None)
    parser.add_argument("--tokens", default="307,401,512,1595,10294")
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--flush-mib", type=int, default=256)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    torch.set_num_threads(1)
    torch.manual_seed(20261115)
    initialization = probe.initialize_npu(torch, torch_npu, args.device)
    device = f"npu:{args.device}"
    packed_cpu, down_cpu, identity = probe.load_weights(args.model_dir, torch, args.layer, args.rank)
    packed = packed_cpu.to(device)
    down = down_cpu.to(device)
    packed_nz = torch_npu.npu_format_cast(packed, NZ_FORMAT)
    down_nz = torch_npu.npu_format_cast(down, NZ_FORMAT)
    formats = {
        "packed_nd": torch_npu.get_npu_format(packed),
        "down_nd": torch_npu.get_npu_format(down),
        "packed_nz": torch_npu.get_npu_format(packed_nz),
        "down_nz": torch_npu.get_npu_format(down_nz),
    }
    flush = None
    if args.flush_mib:
        flush_buffer = torch.empty(args.flush_mib * 1024 * 1024 // 2,
                                   dtype=torch.bfloat16, device=device)
        flush_buffer.uniform_(-1.0, 1.0)
        torch.npu.synchronize()
        flush = lambda: torch.sum(flush_buffer)

    weights = {
        "gate_up": packed.numel() * packed.element_size(),
        "down": down.numel() * down.element_size(),
    }
    report = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "device": f"npu:{args.device}",
        "layer": args.layer,
        "rank": args.rank,
        "identity": identity,
        "allow_internal_format": initialization,
        "formats": formats,
        "weight_bytes": weights,
        "flush_mib": args.flush_mib,
        "cases": [],
    }

    token_values = [int(item) for item in args.tokens.split(",") if item.strip()]
    args.tokens = token_values
    if args.input_file:
        input_cpu, metadata, _ = probe.load_captured_input(args, identity["hidden_size"], torch)
        report["input_file"] = str(args.input_file)
        report["input_metadata"] = metadata
    else:
        input_cpu = None
        report["input_file"] = None

    def chain(x, gate_up, down_weight):
        return functional.linear(
            torch_npu.npu_swiglu(functional.linear(x, gate_up)), down_weight
        )

    for tokens in token_values:
        if input_cpu is not None:
            if tokens > input_cpu.shape[0]:
                continue
            x = input_cpu[:tokens].contiguous().to(device)
        else:
            x = (torch.randn(tokens, identity["hidden_size"]) * 0.5).to(
                device, torch.bfloat16
            )
        cases = {
            "nd": lambda: chain(x, packed, down),
            "nz_both": lambda: chain(x, packed_nz, down_nz),
            "nz_gate_up": lambda: chain(x, packed_nz, down),
            "nz_down": lambda: chain(x, packed, down_nz),
        }
        outputs = {}
        for name, fn in cases.items():
            try:
                outputs[name] = fn()
            except Exception as exc:  # noqa: BLE001
                report["cases"].append({
                    "tokens": tokens, "case": name,
                    "error": f"{type(exc).__name__}: {exc}",
                })
                continue
            warm_ms, cold_ms = measured(fn, args.warmup, args.iters, flush)
            weight_bytes = (
                weights["gate_up"] + weights["down"] if name == "nz_both"
                else weights["gate_up"] if name == "nz_gate_up"
                else weights["down"] if name == "nz_down"
                else weights["gate_up"] + weights["down"]
            )
            record = {
                "tokens": tokens,
                "case": name,
                "warm_ms": warm_ms,
                "cold_ms": cold_ms,
                "nominal_warm_gb_s": weight_bytes / warm_ms / 1e6,
                "nominal_cold_gb_s": (weight_bytes / cold_ms / 1e6) if cold_ms else None,
            }
            if "nd" in outputs:
                record.update(diff(outputs[name], outputs["nd"]))
            report["cases"].append(record)
            print(json.dumps(record, ensure_ascii=False), flush=True)
        del x
        torch.npu.empty_cache()

    report["summary"] = []
    for tokens in sorted({case["tokens"] for case in report["cases"]}):
        row = {"tokens": tokens}
        for case in report["cases"]:
            if case["tokens"] != tokens or "error" in case:
                continue
            row[case["case"]] = {
                "warm_ms": case["warm_ms"],
                "cold_ms": case["cold_ms"],
                "nominal_cold_gb_s": case["nominal_cold_gb_s"],
            }
        report["summary"].append(row)
    text = json.dumps(report, ensure_ascii=False, indent=1)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(json.dumps({"cases": len(report["cases"]), "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
