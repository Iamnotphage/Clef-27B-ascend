#!/usr/bin/env python3
"""Owner 27's direct f810 A/D diagnostic; no service control.

A and D bind complete methods on one shared head/storage. A uses exact f810
methods; D is the frozen dispatch candidate. B remains a gate-only reference
for metadata-only bitwise validation. Production files stay intact.
"""
import argparse
import ast
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time
import traceback
from types import MethodType

PRE = "f8101b98228bbd472a63a1efe6827583e35c6cc9dede4379b925f2b016670863"
B_HEAD = "cf02419978ce62f18ca908b8d4c9646531fb00d3b07871b310576f963e5ec612"
POST = "086357bddbe4fbfbda1630e5ac70503c44f06113d582089c5c54669720d3260a"
KERNEL = "08ad88f6cebfb148aff8d973097d5bcf234e25d144a824385b18b8b078e82dff"
WEIGHTS = "a010ac04f078e699988e4049cbea5e62c962393f59fec366640b64e8d69a4953"
CORE_ATOL, HEAD_ATOL, PROB_ATOL = 0.0078125, 0.03125, 0.005
ORDERS = (("A", "D"), ("D", "A"), ("D", "A"), ("A", "D"))
HERE = Path(__file__).resolve().parent


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(4 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_hash(torch, tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def cpu_cases(torch):
    """Exactly reproduce first-probe random draw order, independent of weights."""
    generator = torch.Generator(device="cpu").manual_seed(20261119)

    def rand(shape):
        return torch.randn(shape, generator=generator).to(torch.bfloat16)

    for length in (307, 967, 1595):
        for queries in (1, 4):
            yield {"scope": "core", "length": length, "queries": queries}, {
                "q": rand((1, queries, 1024)), "k": rand((1, length, 1024)), "v": rand((1, length, 1024))}
        yield {"scope": "head", "length": length, "options": 4}, {
            "hidden": rand((length, 5120)), "lexical": rand((4, 5120))}
    for length, options, questions in ((307, 8, 2), (2049, 4, 1)):
        yield {"scope": "fallback", "length": length, "options": options, "questions": questions}, {
            "hidden": rand((length, 5120)), "lexical": rand((options, 5120))}


class ArmController:
    """Only swaps Python method/backend references; parameters remain identical."""
    def __init__(self, head, original, frozen_b, candidate, kernel):
        self.head, self.original, self.candidate = head, original, candidate
        self.kernel, self.frozen_b = kernel, frozen_b
        self.frozen_backend = kernel.smallq_attention
        self.bindings = [(head, "forward"), (head, "_single_field_decoder")]
        self.bindings += [(layer, "forward") for layer in head.evidence_layers]
        self.saved = [(obj, name, name in obj.__dict__, obj.__dict__.get(name)) for obj, name in self.bindings]
        self.saved_env = os.environ.get("CLEF_JOINT_SMALLQ")

    def select(self, arm):
        if arm not in ("A", "B", "D"):
            raise ValueError("unknown arm: " + arm)
        module = self.original if arm == "A" else self.frozen_b if arm == "B" else self.candidate
        self.head.forward = MethodType(module.JointSchemaHead.forward, self.head)
        self.head._single_field_decoder = MethodType(module.JointSchemaHead._single_field_decoder, self.head)
        for layer in self.head.evidence_layers:
            layer.forward = MethodType(module.EvidenceRoutingLayer.forward, layer)
        os.environ["CLEF_JOINT_SMALLQ"] = "0" if arm == "A" else "1"
        self.kernel.smallq_attention = self.frozen_backend

    def close(self):
        for obj, name, existed, value in self.saved:
            if existed:
                setattr(obj, name, value)
            else:
                delattr(obj, name)
        self.kernel.smallq_attention = self.frozen_backend
        if self.saved_env is None:
            os.environ.pop("CLEF_JOINT_SMALLQ", None)
        else:
            os.environ["CLEF_JOINT_SMALLQ"] = self.saved_env

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def load_reference(path, suffix):
    name = "vllm.model_executor.models.clef_smallq_ad_" + suffix
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def stats(values):
    ordered = sorted(values)
    point = (len(ordered) - 1) * .95
    lo, hi = int(point), min(int(point) + 1, len(ordered) - 1)
    return {"n": len(values), "median": statistics.median(values), "mean": statistics.fmean(values),
            "min": min(values), "max": max(values),
            "p95": ordered[lo] + (ordered[hi] - ordered[lo]) * (point - lo)}


def verify_single_variable(b_path, d_path):
    """Reject differences outside the guard body or after projection begins."""
    b, d = ast.parse(b_path.read_text()), ast.parse(d_path.read_text())
    def split(tree):
        helper = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                      and node.name == "_smallq_cross_attention")
        tail_index = next(index for index, node in enumerate(helper.body)
                          if isinstance(node, ast.If) and ast.unparse(node.test) == "key is value")
        rest = ast.Module(body=[node for node in tree.body if node is not helper], type_ignores=[])
        return (ast.dump(rest), ast.dump(helper.args),
                [ast.dump(node) for node in helper.body[:4]],
                [ast.dump(node) for node in helper.body[tail_index:]])
    # First four statements are docstring, os import, env and module guards.
    if split(b) != split(d):
        raise ValueError("D changes code outside per-call tensor/parameter metadata checks")
    return True


def run(args, report):
    def flush():
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    fixture = json.loads(args.input_manifest.read_text())
    if fixture["original_probe_sha256"] != "25686c9de1f1f5eb20c9a5be5b16eb1b2688a1aa15519be6ef19870f63dc6cad":
        raise ValueError("input provenance mismatch")
    if file_hash(args.reference) != PRE or file_hash(args.weights) != WEIGHTS:
        raise ValueError("reference source/checkpoint hash mismatch")
    sys.path.insert(0, str(args.tree.resolve()))
    import torch
    import torch_npu
    import triton
    from importlib.metadata import version
    import torch.nn.functional as F
    from safetensors.torch import load_file
    from vllm.model_executor.models import clef_joint_schema as candidate
    from vllm.model_executor.models import clef_smallq_kernel as kernel
    if file_hash(candidate.__file__) != POST or file_hash(kernel.__file__) != KERNEL:
        raise ValueError("installed D frozen source/kernel mismatch")
    if file_hash(args.frozen_b) != B_HEAD:
        raise ValueError("frozen B source mismatch")
    report["single_variable_verified"] = verify_single_variable(args.frozen_b, Path(candidate.__file__))
    original = load_reference(args.reference, "f810_reference")
    frozen_b = load_reference(args.frozen_b, "b_reference")
    report["source"] = {"reference": PRE, "head": POST, "frozen_B": B_HEAD, "kernel": KERNEL, "weights": WEIGHTS,
                        "probe": file_hash(__file__), "input_manifest": file_hash(args.input_manifest)}
    report["module_paths"] = {"reference": str(args.reference), "frozen_B": str(args.frozen_b), "head": candidate.__file__,
                              "kernel": kernel.__file__, "torch": torch.__file__, "torch_npu": torch_npu.__file__}
    report["versions"] = {"torch": torch.__version__, "torch_npu": torch_npu.__version__,
                          "triton_module": triton.__version__, "triton_ascend_distribution": version("triton-ascend")}
    report["sdpa_configuration"] = {"COMPATIBLE_IMPL": os.environ.get("COMPATIBLE_IMPL"),
                                  "NPU_COMPATIBLE_IMPL": os.environ.get("NPU_COMPATIBLE_IMPL"),
                                  "flash": torch.backends.cuda.flash_sdp_enabled(),
                                  "math": torch.backends.cuda.math_sdp_enabled(),
                                  "mem_efficient": torch.backends.cuda.mem_efficient_sdp_enabled()}
    torch.npu.set_device(args.device)
    device = f"npu:{args.device}"
    head = candidate.JointSchemaHead(5120, 1024, 2, 4, 16, 4096)
    head.load_state_dict(load_file(str(args.weights), device="cpu"), strict=True)
    head = head.to(device=device, dtype=torch.bfloat16).eval()
    parameter_addresses = [parameter.data_ptr() for parameter in head.parameters()]
    os.environ["CLEF_JOINT_SINGLE_FIELD"] = "1"
    os.environ["CLEF_JOINT_FUSED_ATTN"] = "0"

    def sdpa(q, k, v):
        def view(t):
            return t.view(1, t.shape[1], 16, 64).transpose(1, 2)
        out = F.scaled_dot_product_attention(view(q), view(k), view(v), dropout_p=0.0, is_causal=False)
        return out.transpose(1, 2).contiguous().view(q.shape)

    def compare(a, b):
        af, bf = a.detach().float().cpu(), b.detach().float().cpu()
        delta = (af - bf).abs()
        return {"finite": bool(torch.isfinite(af).all() and torch.isfinite(bf).all()),
                "bit_equal": bool(torch.equal(af, bf)), "different": int(torch.count_nonzero(af != bf)),
                "max_abs": float(delta.max()), "mean_abs": float(delta.mean()),
                "rms_abs": float(delta.square().mean().sqrt()),
                "max_rel_floor_1e-3": float((delta / af.abs().clamp_min(1e-3)).max()),
                "reference_sha256": tensor_hash(torch, a), "candidate_sha256": tensor_hash(torch, b)}

    def enforce(diff, threshold):
        if not diff["finite"] or diff["max_abs"] > threshold:
            raise RuntimeError(f"frozen numerical gate failed: {diff}")

    def measure(fn):
        start, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
        torch.npu.synchronize()
        t0 = time.perf_counter_ns()
        start.record()
        t1 = time.perf_counter_ns()
        out = fn()
        t2 = time.perf_counter_ns()
        end.record()
        t3 = time.perf_counter_ns()
        torch.npu.synchronize()
        t4 = time.perf_counter_ns()
        sample = {"wall_ms": (t4 - t0) / 1e6, "device_ms": start.elapsed_time(end),
                  "host_submit_ms": (t2 - t1) / 1e6, "sync_wait_ms": (t4 - t3) / 1e6,
                  "event_record_host_ms": (t1 - t0 + t3 - t2) / 1e6}
        if any(not math.isfinite(value) or value < 0 for value in sample.values()) or sample["device_ms"] <= 0:
            raise RuntimeError("invalid timing sample")
        return out, sample

    def schema(label):
        options, questions = label["options"], label.get("questions", 1)
        spans, each = questions + options, options // questions
        return candidate.SchemaIndices(
            torch.arange(2 * spans, device=device),
            torch.arange(spans, device=device).repeat_interleave(2),
            torch.full((spans,), 2, device=device, dtype=torch.long),
            torch.ones(questions, device=device, dtype=torch.long),
            torch.arange(questions, device=device).repeat_interleave(each),
            torch.arange(options, device=device),
            torch.ones(questions, each, device=device, dtype=torch.bool))

    report.update(gates=[], first_calls=[], timing=[])
    cases = []
    with torch.inference_mode(), ArmController(head, original, frozen_b, candidate, kernel) as control:
        for label, tensors in cpu_cases(torch):
            hashes = {name + "_sha256": tensor_hash(torch, tensor) for name, tensor in tensors.items()}
            if label["scope"] != "fallback":
                key = f"{label['scope']}/{label['length']}/{label.get('queries', 4)}"
                if hashes != fixture["inputs"][key]:
                    raise ValueError("original input hash mismatch: " + key)
            tensors = {name: tensor.to(device) for name, tensor in tensors.items()}
            indices = schema(label) if label["scope"] != "core" else None
            row = {"label": label, "input_hashes": hashes, "arms": {}}
            report["gates"].append(row)
            outputs = {}
            for arm in ("A", "B", "D"):
                control.select(arm)
                calls, mha_calls, hooks = [], [], []
                row["arms"][arm] = {"backend_calls": calls, "mha_calls": mha_calls}
                selected_backend = kernel.smallq_attention

                def observed(q, k, v):
                    if label["scope"] == "fallback":
                        raise RuntimeError("fallback reached replacement backend")
                    out = selected_backend(q, k, v)
                    diff = compare(sdpa(q, k, v), out)
                    calls.append({"q_shape": list(q.shape), "k_shape": list(k.shape), "dtype": str(q.dtype),
                                  "q_stride": list(q.stride()), "k_stride": list(k.stride()),
                                  "v_stride": list(v.stride()), "difference": diff})
                    enforce(diff, CORE_ATOL)
                    return out

                def mha_hook(module, inputs):
                    mha_calls.append({"q_shape": list(inputs[0].shape), "kv_same_object": inputs[1] is inputs[2]})

                try:
                    if label["scope"] == "core":
                        fn = sdpa if arm == "A" else selected_backend
                        outputs[arm], first = measure(lambda: fn(**tensors))
                        report["first_calls"].append({"label": label, "arm": arm, "samples": first,
                                                      "includes_first_compile_or_cache_lookup": arm in ("B", "D")})
                    else:
                        kernel.smallq_attention = observed
                        for attention in [layer.attention for layer in head.evidence_layers] + [layer.multihead_attn for layer in head.layers]:
                            hooks.append(attention.register_forward_pre_hook(mha_hook))
                        outputs[arm] = head(tensors["hidden"], tensors["lexical"], indices)
                finally:
                    for hook in hooks:
                        hook.remove()
                    kernel.smallq_attention = selected_backend
                diff = compare(outputs["A"], outputs[arm])
                row["arms"][arm]["difference_from_A"] = diff
                enforce(diff, CORE_ATOL if label["scope"] == "core" else HEAD_ATOL)
                if arm == "D":
                    bd_diff = compare(outputs["B"], outputs["D"])
                    row["arms"][arm]["difference_from_B"] = bd_diff
                    if not bd_diff["finite"] or not bd_diff["bit_equal"]:
                        raise RuntimeError("metadata-only D must equal frozen B bitwise")
                if label["scope"] != "core":
                    prob = compare(outputs["A"].float().softmax(-1), outputs[arm].float().softmax(-1))
                    row["arms"][arm]["probability_difference_from_A"] = prob
                    enforce(prob, PROB_ATOL)
                    if label["scope"] == "fallback":
                        if calls or not diff["bit_equal"]:
                            raise RuntimeError("fallback must be bitwise, with zero replacement calls")
                    elif arm == "A":
                        if (calls or [c["q_shape"][1] for c in mha_calls] != [4, 4, 1, 1, 1, 1]
                                or [c["kv_same_object"] for c in mha_calls] != [False, False, True, True, True, True]):
                            raise RuntimeError("original MHA path/count mismatch")
                    elif mha_calls or [c["q_shape"][1] for c in calls] != [4, 4, 1, 1, 1, 1]:
                        raise RuntimeError("replacement backend path/count mismatch")
                flush()
            if label == {"scope": "head", "length": 307, "options": 4}:
                control.select("D")
                os.environ["CLEF_JOINT_SMALLQ"] = "0"
                def forbidden_backend(*_):
                    raise RuntimeError("default-off reached backend")
                kernel.smallq_attention = forbidden_backend
                try:
                    off = head(tensors["hidden"], tensors["lexical"], indices)
                finally:
                    kernel.smallq_attention = control.frozen_backend
                report["default_off"] = compare(outputs["A"], off)
                if not report["default_off"]["finite"] or not report["default_off"]["bit_equal"]:
                    raise RuntimeError("default-off must equal original A")
            if label["scope"] != "fallback":
                cases.append((label, tensors, indices))
        if parameter_addresses != [parameter.data_ptr() for parameter in head.parameters()]:
            raise RuntimeError("arm selection changed weight storage")
        report["same_weight_storage_verified"] = True
        report["gate_passed"] = True
        flush()
        if args.gate_only:
            return
        for label, tensors, indices in cases:
            row = {"label": label, "rounds": []}
            report["timing"].append(row)
            for number, order in enumerate(ORDERS):
                round_ = {"round": number, "order": list(order), "arms": {}}
                row["rounds"].append(round_)
                for arm in order:
                    control.select(arm)
                    if label["scope"] == "core":
                        backend = sdpa if arm == "A" else kernel.smallq_attention
                        fn = lambda: backend(**tensors)
                    else:
                        fn = lambda: head(tensors["hidden"], tensors["lexical"], indices)
                    for _ in range(5):
                        fn()
                    samples = []
                    round_["arms"][arm] = {"samples": samples}
                    for index in range(20):
                        _, sample = measure(fn)
                        samples.append(sample | {"index": index})
                    for metric in ("wall_ms", "device_ms", "host_submit_ms", "sync_wait_ms", "event_record_host_ms"):
                        round_["arms"][arm][metric] = stats([sample[metric] for sample in samples])
                    flush()
            row["paired_deltas"] = {}
            for left, right in (("A", "D"),):
                row["paired_deltas"][f"{right}_minus_{left}"] = {
                    metric: [r["arms"][right][metric]["median"] - r["arms"][left][metric]["median"]
                             for r in row["rounds"]]
                    for metric in ("wall_ms", "device_ms", "host_submit_ms", "sync_wait_ms")}
            flush()
    if (file_hash(candidate.__file__) != POST or file_hash(kernel.__file__) != KERNEL
            or file_hash(args.frozen_b) != B_HEAD):
        raise RuntimeError("frozen production/reference files changed")
    report["frozen_files_unchanged"] = True


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tree", type=Path, default=Path("/workspace/yangchen/work/src/vllm"))
    ap.add_argument("--device", type=int, choices=(0, 1), default=0)
    ap.add_argument("--weights", type=Path, default=Path("/workspace/yangchen/models/Clef-27B/joint_head.safetensors"))
    ap.add_argument("--reference", type=Path, default=HERE / "joint_smallq_ad/f810_reference.py")
    ap.add_argument("--frozen-b", type=Path, default=HERE / "joint_smallq_ad/b_reference.py")
    ap.add_argument("--input-manifest", type=Path, default=HERE / "joint_smallq_ad/inputs.json")
    ap.add_argument("--gate-only", action="store_true")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {"schema": "joint-smallq-probe/1", "experiment": "joint-smallq-ad/1", "gate_passed": False,
              "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              "arm_paths": {"A": "same head/storage; exact production f810 methods; original MHA",
                            "B": "gate-only frozen 2e8 methods; D/B bitwise validation",
                            "D": "same head/storage; per-call metadata dedup guard; identical projection/kernel"},
              "experimental_variable": "exact f810 methods versus frozen D methods and smallQ; no candidate changes",
              "core_control": "A projected SDPA versus D frozen smallQ; report separately from full head",
              "timing_notes": {"selection": "method/backend/env changes outside measurement; warmup5 per arm block",
                               "device_ms": "event interval includes stream gaps; not kernel-only time",
                               "host_submit_ms": "time in Python/framework call; can include internal synchronization",
                               "sync_wait_ms": "host waiting after end event submission",
                               "event_record_host_ms": "host time in start/end event record",
                               "gap": "stream gaps cannot be independently separated from these counters"},
              "thresholds": {"core_atol": CORE_ATOL, "head_atol": HEAD_ATOL, "prob_atol": PROB_ATOL},
              "inference_mode": True, "seed": 20261119, "warmup_per_block": 5, "iters_per_arm_round": 20,
              "orders": ORDERS, "quality_scope": "synthetic inputs, real shared weights; OD/ERQA and S1/S2 unmeasured"}
    try:
        run(args, report)
    except Exception:
        report["error"] = traceback.format_exc()
    finally:
        os.environ["CLEF_JOINT_SMALLQ"] = "0"
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "experiment": report["experiment"],
                      "gate_passed": report["gate_passed"], "error": report.get("error")}))
    return int("error" in report)


if __name__ == "__main__":
    raise SystemExit(main())
