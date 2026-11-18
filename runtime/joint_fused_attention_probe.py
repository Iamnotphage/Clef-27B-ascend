#!/usr/bin/env python3
"""NPU head microbenchmark for the fused cross-attention candidate.

Execution owner: 21.  This imports torch/torch_npu and runs the real
JointSchemaHead from the installed vLLM tree.  The arms share weights, inputs,
the single-field flag and an inference-mode context; only
CLEF_JOINT_FUSED_ATTN differs.  Four interleaved rounds (A/B, B/A, B/A, A/B)
keep per-call raw device and wall samples.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
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


def summarize_arms(devices: dict, walls: dict) -> dict:
    report = {}
    for value in (0, 1):
        dev = devices[str(value)]
        arm_wall = walls[str(value)]
        report[str(value)] = {
            "device_ms": stats(dev),
            "wall_ms": stats(arm_wall),
            "device_samples": [round(item, 6) for item in dev],
            "wall_samples": [round(item, 6) for item in arm_wall],
            "samples": [{"index": sample, "device_ms": round(dev[sample], 6),
                         "wall_ms": round(arm_wall[sample], 6)}
                        for sample in range(len(dev))],
        }
    return report


def schema_for(options: int):
    # One question span and one span per option, each two tokens.
    spans = [(2 * index, 2 * index + 2) for index in range(1 + options)]
    positions = []
    span_ids = []
    for index, (start, end) in enumerate(spans):
        positions.extend(range(start, end))
        span_ids.extend([index] * (end - start))
    return positions, span_ids, [end - start for start, end in spans]


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
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--rounds", type=int, default=4)
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
    positions, span_ids, lengths = schema_for(options)
    hidden_cpu = torch.randn(length, args.hidden_size, dtype=torch.float32).to(torch.bfloat16)
    lexical_cpu = torch.randn(options, args.hidden_size, dtype=torch.float32).to(torch.bfloat16)
    hidden = hidden_cpu.npu()
    lexical = lexical_cpu.npu()
    schema = SchemaIndices(
        positions=torch.tensor(positions, device="npu"),
        span_ids=torch.tensor(span_ids, device="npu"),
        span_lengths=torch.tensor(lengths, device="npu"),
        question_types=torch.tensor([1], device="npu"),
        option_question=torch.zeros(options, dtype=torch.long, device="npu"),
        option_slots=torch.arange(options, device="npu"),
        option_mask=torch.ones(1, options, dtype=torch.bool, device="npu"),
    )
    os.environ["CLEF_JOINT_SINGLE_FIELD"] = "1"
    head_dim = args.width // args.heads
    scale = 1.0 / math.sqrt(head_dim)
    interface = {"input_layout": "BSH", "scale": scale, "keep_prob": 1.0,
                 "dtype": "bfloat16", "head_num": args.heads,
                 "head_dim": head_dim, "checks": []}

    def interface_call(q_len: int):
        query = torch.randn(1, q_len, args.width, dtype=torch.bfloat16, device="npu")
        key = torch.randn(1, 8, args.width, dtype=torch.bfloat16, device="npu")
        out = torch_npu.npu_fusion_attention(
            query, key, key, args.heads, "BSH", scale=scale,
            keep_prob=1.0, sparse_mode=0)[0]
        torch.npu.synchronize()
        interface["checks"].append({"q_len": q_len, "out_shape": list(out.shape),
                                    "finite": bool(torch.isfinite(out).all())})

    try:
        interface_call(1)
        interface_call(options)
        interface["ok"] = all(item["finite"] for item in interface["checks"])
    except Exception as exc:
        interface["ok"] = False
        interface["error"] = f"{type(exc).__name__}: {exc}"

    def arm(value: int, outputs: list, devices: list, walls: list):
        os.environ["CLEF_JOINT_FUSED_ATTN"] = str(value)
        for _ in range(args.warmup):
            head(hidden, lexical, schema)
        torch.npu.synchronize()
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
            outputs.append(out)

    orders = [[0, 1], [1, 0], [1, 0], [0, 1]]
    rounds = []
    raw = {0: [], 1: []}
    wall_all = {0: [], 1: []}
    last = {}
    error = None
    try:
        with torch.inference_mode():
            for index in range(min(args.rounds, len(orders))):
                devices = {"0": [], "1": []}
                walls = {"0": [], "1": []}
                for value in orders[index]:
                    outs = []
                    arm(value, outs, devices[str(value)], walls[str(value)])
                    raw[value].extend(devices[str(value)])
                    wall_all[value].extend(walls[str(value)])
                    last[value] = outs[-1].detach().to("cpu")
                arms_report = summarize_arms(devices, walls)
                rounds.append({"round": index, "order": orders[index],
                               "raw_units": "ms", "arms": arms_report})
                partial = {
                    "schema": "joint-fused-attention-probe/2",
                    "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "args": vars(args) | {"output": str(args.output)},
                    "rounds": rounds,
                    "raw_device": {str(key): raw[key] for key in (0, 1)},
                    "raw_wall": {str(key): wall_all[key] for key in (0, 1)},
                    "partial": True,
                }
                args.output.parent.mkdir(parents=True, exist_ok=True)
                (args.output.with_name("partial.json")).write_text(
                    json.dumps(partial, ensure_ascii=False, indent=1) + "\n")
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"

    sdpa = {}
    for name in ("flash_sdp_enabled", "math_sdp_enabled", "mem_efficient_sdp_enabled"):
        try:
            sdpa[name] = bool(getattr(torch.backends.cuda, name)())
        except Exception:
            sdpa[name] = None
    torch_npu_git_version = None
    try:
        version_mod = getattr(torch_npu, "version", None)
        torch_npu_git_version = getattr(version_mod, "git_version", None) or str(version_mod)
    except Exception:
        pass
    compatible_impl = {
        "COMPATIBLE_IMPL": os.environ.get("COMPATIBLE_IMPL"),
        "NPU_COMPATIBLE_IMPL": os.environ.get("NPU_COMPATIBLE_IMPL"),
    }
    report = {
        "schema": "joint-fused-attention-probe/2",
        "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "sdpa_flags": sdpa,
        "torch_npu_git_version": torch_npu_git_version,
        "compatible_impl": compatible_impl,
        "args": vars(args) | {"output": str(args.output)},
        "versions": {"torch": torch.__version__, "torch_npu": torch_npu.__version__},
        "input": {"length": length, "options": options,
                  "hidden_sha256": sha256_tensor(torch, hidden_cpu),
                  "lexical_sha256": sha256_tensor(torch, lexical_cpu),
                  "schema_spans": 1 + options},
        "interface": interface,
        "rounds": rounds,
        "baseline": {"device_ms": stats(raw[0]), "wall_ms": stats(wall_all[0])} if raw[0] else None,
        "candidate": {"device_ms": stats(raw[1]), "wall_ms": stats(wall_all[1])} if raw[1] else None,
        "candidate_error": error,
    }
    if not error and raw[0] and raw[1]:
        base = last[0].float()
        cand = last[1].float()
        report["correctness"] = {
            "bit_equal": bool(torch.equal(last[0], last[1])),
            "finite": bool(torch.isfinite(cand).all()),
            "max_abs": float((base - cand).abs().max()),
            "mean_abs": float((base - cand).abs().mean()),
        }
        report["delta"] = {
            "device_median_ms": report["candidate"]["device_ms"]["median"] - report["baseline"]["device_ms"]["median"],
            "device_p95_ms": report["candidate"]["device_ms"]["p95"] - report["baseline"]["device_ms"]["p95"],
            "wall_median_ms": report["candidate"]["wall_ms"]["median"] - report["baseline"]["wall_ms"]["median"],
            "paired_round_device_median_ms": [
                row["arms"]["1"]["device_ms"]["median"] - row["arms"]["0"]["device_ms"]["median"]
                for row in rounds],
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=1))
    return 0 if (not error and interface.get("ok")) else 1


if __name__ == "__main__":
    raise SystemExit(main())
