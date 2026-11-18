#!/usr/bin/env python3
"""NPU head microbenchmark for the fused cross-attention candidate.

Execution owner: 21.  This imports torch/torch_npu and runs the real
JointSchemaHead from the installed vLLM tree; it is never run by the CPU
reviewer.  The two arms only differ by CLEF_JOINT_FUSED_ATTN; weights, inputs
and the single-field flag are shared.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import sys
import time
from pathlib import Path


def sha256_tensor(torch, tensor) -> str:
    host = tensor.detach().to("cpu").contiguous()
    return hashlib.sha256(host.view(torch.uint16).numpy().tobytes()).hexdigest()


def stats(values):
    ordered = sorted(values)
    idx = (len(ordered) - 1) * 0.95
    lo, hi = int(idx), min(int(idx) + 1, len(ordered) - 1)
    return {"n": len(values), "mean": statistics.fmean(values),
            "median": statistics.median(values), "min": min(values),
            "max": max(values),
            "p95": ordered[lo] + (ordered[hi] - ordered[lo]) * (idx - lo)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree", default="/workspace/yangchen/work/src/vllm")
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--length", type=int, default=967)
    ap.add_argument("--options", type=int, default=4)
    ap.add_argument("--hidden-size", type=int, default=5120)
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--routing-layers", type=int, default=2)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--heads", type=int, default=16)
    ap.add_argument("--feedforward", type=int, default=4096)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--seed", type=int, default=20261119)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    sys.path.insert(0, str(Path(args.tree).resolve()))
    import torch
    import torch_npu
    from vllm.model_executor.models.clef_joint_schema import JointSchemaHead, SchemaIndices

    torch.npu.set_device(args.device)
    torch.manual_seed(args.seed)
    head = JointSchemaHead(args.hidden_size, args.width, args.routing_layers,
                           args.layers, args.heads, args.feedforward)
    with torch.no_grad():
        for parameter in head.parameters():
            parameter.normal_(0.0, 0.02)
    head = head.to("npu").to(torch.bfloat16).eval()

    length, options = args.length, args.options
    hidden_cpu = torch.randn(length, args.hidden_size, dtype=torch.float32).to(torch.bfloat16)
    lexical_cpu = torch.randn(options, args.hidden_size, dtype=torch.float32).to(torch.bfloat16)
    hidden = hidden_cpu.npu()
    lexical = lexical_cpu.npu()
    schema = SchemaIndices(
        positions=torch.arange(0, 4, device="npu"),
        span_ids=torch.tensor([0, 0, 1, 1], device="npu"),
        span_lengths=torch.tensor([2, 2], device="npu"),
        question_types=torch.tensor([1], device="npu"),
        option_question=torch.zeros(options, dtype=torch.long, device="npu"),
        option_slots=torch.arange(options, device="npu"),
        option_mask=torch.ones(1, options, dtype=torch.bool, device="npu"),
    )
    os.environ["CLEF_JOINT_SINGLE_FIELD"] = "1"

    def arm(value: int):
        os.environ["CLEF_JOINT_FUSED_ATTN"] = str(value)
        for _ in range(args.warmup):
            head(hidden, lexical, schema)
        torch.npu.synchronize()
        outputs, devices, walls = [], [], []
        for _ in range(args.iters):
            start = torch.npu.Event(enable_timing=True)
            end = torch.npu.Event(enable_timing=True)
            begin = time.perf_counter()
            start.record()
            out = head(hidden, lexical, schema)
            end.record()
            torch.npu.synchronize()
            devices.append(start.elapsed_time(end))
            walls.append((time.perf_counter() - begin) * 1000.0)
            outputs.append(out.detach().to("cpu"))
        return outputs, devices, walls

    baseline, base_dev, base_wall = arm(0)
    try:
        candidate, cand_dev, cand_wall = arm(1)
        error = None
    except Exception as exc:
        candidate, cand_dev, cand_wall = None, [], []
        error = f"{type(exc).__name__}: {exc}"

    report = {
        "schema": "joint-fused-attention-probe/1",
        "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "args": vars(args) | {"output": str(args.output)},
        "versions": {"torch": torch.__version__, "torch_npu": torch_npu.__version__},
        "input": {"length": length, "options": options,
                  "hidden_sha256": sha256_tensor(torch, hidden_cpu),
                  "lexical_sha256": sha256_tensor(torch, lexical_cpu)},
        "baseline": {"device_ms": stats(base_dev), "wall_ms": stats(base_wall)},
        "candidate": ({"device_ms": stats(cand_dev), "wall_ms": stats(cand_wall)}
                      if candidate is not None else None),
        "candidate_error": error,
    }
    if candidate is not None:
        base = baseline[-1].float()
        cand = candidate[-1].float()
        report["correctness"] = {
            "bit_equal": bool(torch.equal(baseline[-1], candidate[-1])),
            "finite": bool(torch.isfinite(cand).all()),
            "max_abs": float((base - cand).abs().max()),
            "mean_abs": float((base - cand).abs().mean()),
        }
        report["delta"] = {
            "device_median_ms": report["candidate"]["device_ms"]["median"] - report["baseline"]["device_ms"]["median"],
            "device_p95_ms": report["candidate"]["device_ms"]["p95"] - report["baseline"]["device_ms"]["p95"],
            "wall_median_ms": report["candidate"]["wall_ms"]["median"] - report["baseline"]["wall_ms"]["median"],
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=1))
    return 0 if candidate is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())
