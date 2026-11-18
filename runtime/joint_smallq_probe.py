#!/usr/bin/env python3
"""Real NPU gate then ABBA timing. Worker 21 alone executes this probe.

Uses installed postimage and actual checkpoint weights. The kernel core and
whole head compare against the installed original MHA/SDPA path on the same
BF16 tensors. Raw first calls, gate records, every wall/device timing sample,
weights/input/source hashes and true hit shapes are saved. No service control.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time
import traceback


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(4 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stats(values):
    ordered = sorted(values)
    p = (len(ordered) - 1) * 0.95
    lo = int(p)
    hi = min(lo + 1, len(ordered) - 1)
    return {"n": len(values), "mean": statistics.fmean(values),
            "median": statistics.median(values), "min": min(values),
            "max": max(values), "p95": ordered[lo] + (ordered[hi] - ordered[lo]) * (p - lo)}


def run(args, report):
    sys.path.insert(0, str(args.tree.resolve()))
    import torch
    import torch_npu
    import triton
    import torch.nn.functional as F
    from safetensors.torch import load_file
    from vllm.model_executor.models import clef_joint_schema as source
    from vllm.model_executor.models import clef_smallq_kernel as kernel

    report["versions"] = {"torch": torch.__version__, "torch_npu": torch_npu.__version__,
                          "triton": triton.__version__}
    report["source"] = {"head": file_hash(source.__file__), "kernel": file_hash(kernel.__file__),
                        "weights": file_hash(args.weights)}
    report["module_paths"] = {"head": source.__file__, "kernel": kernel.__file__,
                              "torch": torch.__file__, "torch_npu": torch_npu.__file__,
                              "triton": triton.__file__}
    report["sdpa_configuration"] = {
        "COMPATIBLE_IMPL": os.environ.get("COMPATIBLE_IMPL"),
        "NPU_COMPATIBLE_IMPL": os.environ.get("NPU_COMPATIBLE_IMPL"),
        "flash_sdp_enabled": torch.backends.cuda.flash_sdp_enabled(),
        "math_sdp_enabled": torch.backends.cuda.math_sdp_enabled(),
        "mem_efficient_sdp_enabled": torch.backends.cuda.mem_efficient_sdp_enabled(),
    }
    torch.npu.set_device(args.device)
    device = f"npu:{args.device}"
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    head = source.JointSchemaHead(5120, 1024, 2, 4, 16, 4096)
    head.load_state_dict(load_file(str(args.weights), device="cpu"), strict=True)
    head = head.to(device=device, dtype=torch.bfloat16).eval()
    os.environ["CLEF_JOINT_SINGLE_FIELD"] = "1"
    os.environ["CLEF_JOINT_FUSED_ATTN"] = "0"

    def digest(tensor):
        raw = tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
        return hashlib.sha256(raw).hexdigest()

    def compare(a, b):
        af, bf = a.detach().float().cpu(), b.detach().float().cpu()
        delta = (af - bf).abs()
        return {"finite": bool(torch.isfinite(af).all() and torch.isfinite(bf).all()),
                "bit_equal": bool(torch.equal(a, b)), "elements": a.numel(),
                "different": int(torch.count_nonzero(af != bf)),
                "max_abs": float(delta.max()), "mean_abs": float(delta.mean()),
                "rms_abs": float(delta.square().mean().sqrt()),
                "max_rel_floor_1e-3": float((delta / af.abs().clamp_min(1e-3)).max()),
                "reference_sha256": digest(a), "candidate_sha256": digest(b)}

    def random(shape):
        return torch.randn(shape, generator=generator).to(torch.bfloat16).to(device)

    def measure(fn):
        start, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
        torch.npu.synchronize()
        began = time.perf_counter_ns()
        start.record()
        out = fn()
        end.record()
        torch.npu.synchronize()
        return out, {"wall_ms": (time.perf_counter_ns() - began) / 1e6,
                     "device_ms": start.elapsed_time(end)}

    def reference_core(q, k, v):
        def view(t):
            return t.view(1, t.shape[1], 16, 64).transpose(1, 2)
        out = F.scaled_dot_product_attention(view(q), view(k), view(v),
                                             dropout_p=0.0, is_causal=False)
        return out.transpose(1, 2).contiguous().view(q.shape)

    def schema(length, options, questions=1):
        spans = questions + options
        assert 2 * spans <= length and options % questions == 0
        each = options // questions
        return source.SchemaIndices(
            positions=torch.arange(2 * spans, device=device),
            span_ids=torch.arange(spans, device=device).repeat_interleave(2),
            span_lengths=torch.full((spans,), 2, device=device, dtype=torch.long),
            question_types=torch.ones(questions, device=device, dtype=torch.long),
            option_question=torch.arange(questions, device=device).repeat_interleave(each),
            option_slots=torch.arange(options, device=device),
            option_mask=torch.ones(questions, each, device=device, dtype=torch.bool))

    def whole(value, hidden, lexical, indices):
        os.environ["CLEF_JOINT_SMALLQ"] = str(value)
        return head(hidden, lexical, indices)

    def time_pair(label, off, on):
        row = {"label": label, "rounds": []}
        for fn in (off, on):
            for _ in range(args.warmup):
                fn()
        for index, order in enumerate(((0, 1), (1, 0), (1, 0), (0, 1))):
            arms = {}
            for value in order:
                samples = [measure((off, on)[value])[1] | {"index": n}
                           for n in range(args.iters)]
                arms[str(value)] = {"samples": samples,
                                   "wall_ms": stats([s["wall_ms"] for s in samples]),
                                   "device_ms": stats([s["device_ms"] for s in samples])}
            row["rounds"].append({"round": index, "order": list(order), "arms": arms})
        row["paired_wall_median_delta_ms"] = [r["arms"]["1"]["wall_ms"]["median"]
                                                - r["arms"]["0"]["wall_ms"]["median"]
                                                for r in row["rounds"]]
        return row

    cores, heads = [], []
    report["gates"] = {"core": [], "head": [], "fallback": []}
    report["first_calls"] = []
    report["timing"] = []
    with torch.inference_mode():
        for length in args.lengths:
            for queries in (1, args.options):
                q, k, v = random((1, queries, 1024)), random((1, length, 1024)), random((1, length, 1024))
                baseline, first_a = measure(lambda: reference_core(q, k, v))
                candidate, first_b = measure(lambda: kernel.smallq_attention(q, k, v))
                diff = compare(baseline, candidate)
                record = {"length": length, "queries": queries, "q_sha256": digest(q),
                          "k_sha256": digest(k), "v_sha256": digest(v), "difference": diff}
                report["gates"]["core"].append(record)
                report["first_calls"].append({"scope": "core", "length": length, "queries": queries,
                                              "baseline": first_a, "candidate_includes_first_jit": first_b})
                if not diff["finite"] or diff["max_abs"] > args.core_atol:
                    raise RuntimeError(f"core gate failed: {record}")
                cores.append((length, queries, q, k, v))

            hidden, lexical = random((length, 5120)), random((args.options, 5120))
            indices = schema(length, args.options)
            baseline, first_a = measure(lambda: whole(0, hidden, lexical, indices))
            calls = []
            record = {"length": length, "hidden_sha256": digest(hidden), "lexical_sha256": digest(lexical),
                      "schema_spans": 1 + args.options, "calls": calls}
            report["gates"]["head"].append(record)
            report["first_calls"].append({"scope": "head", "length": length,
                                          "baseline": first_a,
                                          "candidate": "instrumented correctness only; excluded from timing"})
            original_kernel = kernel.smallq_attention

            def observed(q, k, v):
                out = original_kernel(q, k, v)
                ref = reference_core(q, k, v)
                diff = compare(ref, out)
                calls.append({"q_shape": list(q.shape), "k_shape": list(k.shape),
                              "dtype": str(q.dtype), "q_stride": list(q.stride()),
                              "k_stride": list(k.stride()), "v_stride": list(v.stride()),
                              "difference": diff})
                if not diff["finite"] or diff["max_abs"] > args.core_atol:
                    raise RuntimeError("projected attention gate failed")
                return out

            kernel.smallq_attention = observed
            try:
                candidate = whole(1, hidden, lexical, indices)
            finally:
                kernel.smallq_attention = original_kernel
            diff = compare(baseline, candidate)
            probs = compare(baseline.float().softmax(-1), candidate.float().softmax(-1))
            record.update(difference=diff, probability_difference=probs)
            if (len(calls) != 6 or [c["q_shape"][1] for c in calls] != [args.options] * 2 + [1] * 4
                    or not diff["finite"] or diff["max_abs"] > args.head_atol
                    or probs["max_abs"] > args.prob_atol):
                raise RuntimeError("whole-head gate failed")
            heads.append((length, hidden, lexical, indices))

        # Real-device fallback: multi-field routes and L above supported domain.
        for length, options, questions in ((307, 8, 2), (2049, args.options, 1)):
            hidden, lexical = random((length, 5120)), random((options, 5120))
            indices = schema(length, options, questions)
            off = whole(0, hidden, lexical, indices)

            def unexpected(*_):
                raise RuntimeError("unsupported domain reached kernel")

            original_kernel = kernel.smallq_attention
            kernel.smallq_attention = unexpected
            try:
                on = whole(1, hidden, lexical, indices)
            finally:
                kernel.smallq_attention = original_kernel
            diff = compare(off, on)
            report["gates"]["fallback"].append({"length": length, "questions": questions, "difference": diff})
            if not diff["finite"] or not diff["bit_equal"]:
                raise RuntimeError("fallback gate failed")

        report["gate_passed"] = True
        # Persist gate results before timing; no timing is attempted after a failed gate.
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        if not args.gate_only:
            for length, queries, q, k, v in cores:
                report["timing"].append(time_pair(
                    {"scope": "core", "length": length, "queries": queries},
                    lambda: reference_core(q, k, v), lambda: kernel.smallq_attention(q, k, v)))
            for length, hidden, lexical, indices in heads:
                report["timing"].append(time_pair(
                    {"scope": "head", "length": length, "options": args.options},
                    lambda: whole(0, hidden, lexical, indices),
                    lambda: whole(1, hidden, lexical, indices)))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tree", type=Path, default=Path("/workspace/yangchen/work/src/vllm"))
    ap.add_argument("--weights", type=Path, default=Path("/workspace/yangchen/models/Clef-27B/joint_head.safetensors"))
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--lengths", type=lambda s: [int(x) for x in s.split(",")], default=[307, 967, 1595])
    ap.add_argument("--options", type=int, default=4)
    ap.add_argument("--seed", type=int, default=20261119)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--core-atol", type=float, default=0.0078125)
    ap.add_argument("--head-atol", type=float, default=0.03125)
    ap.add_argument("--prob-atol", type=float, default=0.005)
    ap.add_argument("--gate-only", action="store_true")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if not args.lengths or any(length < 2 * (1 + args.options) or length > 2048 for length in args.lengths):
        ap.error("lengths must cover all schema spans and be <=2048")
    if not 1 <= args.options <= 8 or args.iters < 1 or args.warmup < 0:
        ap.error("options must be 1..8, iters >=1, warmup >=0")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {"schema": "joint-smallq-probe/1", "args": {k: str(v) if isinstance(v, Path) else v
                                                         for k, v in vars(args).items()},
              "gate_passed": False, "inference_mode": True,
              "quality_scope": "synthetic inputs, real weights; full OD/ERQA gate still required"}
    try:
        run(args, report)
    except Exception:
        report["error"] = traceback.format_exc()
    finally:
        os.environ["CLEF_JOINT_SMALLQ"] = "0"
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "gate_passed": report["gate_passed"],
                      "error": report.get("error")}, indent=2))
    return int("error" in report)


if __name__ == "__main__":
    raise SystemExit(main())
