#!/usr/bin/env python3
"""Compare a private BF16 gate/up column-pair operator with the original MLP."""
import argparse
import ctypes as C
import hashlib
import json
import os
from pathlib import Path
import random
import statistics
import time

from probe_mlp_ffn import initialize_npu, load_weights

HIDDEN = 5120
INTERMEDIATE = 8704
LAYERS = (31, 0, 63)
SHAPES = (967, 307, 1595)
PAIRS = 7
API = "aclnnClefMlpGateUpPair"
ABI = [C.c_void_p, C.c_void_p, C.c_int64, C.c_void_p,
       C.POINTER(C.c_uint64), C.POINTER(C.c_void_p)]


def tensor_sha(tensor, torch):
    return hashlib.sha256(memoryview(tensor.view(torch.uint8).numpy())).hexdigest()


def pair_columns(torch, tensor, half):
    gate, up = tensor.split(INTERMEDIATE, dim=0)
    return torch.cat((gate.reshape(-1, half, HIDDEN), up.reshape(-1, half, HIDDEN)),
                     dim=1).reshape(2 * INTERMEDIATE, HIDDEN).contiguous()


def unpack_columns(tensor, half):
    pairs = tensor.reshape(len(tensor), -1, 2, half)
    return tuple(pairs[:, :, i, :].reshape(len(tensor), INTERMEDIATE) for i in (0, 1))


def compare(torch, actual, expected, rel_limit, max_limit):
    if not bool(torch.isfinite(actual).all() & torch.isfinite(expected).all()):
        return {"finite": False, "passed": False}
    a, b = actual.float(), expected.float()
    error = a - b
    rel = float(error.square().mean().sqrt() / b.square().mean().sqrt().clamp_min(1e-30))
    max_abs = float(error.abs().max())
    normalized = max_abs / max(float(b.abs().max()), 1e-30)
    return {"finite": True, "exact": bool(torch.equal(actual, expected)), "rel_rmse": rel,
            "max_abs": max_abs, "normalized_max": normalized,
            "passed": rel <= rel_limit and normalized <= max_limit}


def dispatch(runtime_api, runtime, torch, x, w, mode, guarded=False):
    width = INTERMEDIATE if mode & 1 else 2 * INTERMEDIATE
    guard = 256
    kwargs = {"dtype": torch.bfloat16, "device": x.device}
    if guarded:
        backing = torch.full((x.shape[0] * width + 2 * guard,), float("nan"), **kwargs)
        out = backing[guard:-guard].view(x.shape[0], width)
    else:
        out = torch.empty((x.shape[0], width), **kwargs)

    def descriptors(ctx):
        return (ctx.tensor(x, "x"),
                ctx.tensor(w, "pairedWeight", acl_format_override=29,
                           storage_shape_override=(HIDDEN // 16, 2 * INTERMEDIATE // 16, 16, 16)),
                mode, ctx.tensor(out, "y"))

    if not guarded:
        runtime_api.call_aclnn(API, descriptors, (out,), get_workspace_argtypes=ABI)
        return out

    ctx = runtime_api._CallContext(runtime, x.device)
    getter = runtime.symbol(API + "GetWorkspaceSize")
    getter.restype, getter.argtypes = C.c_int, ABI
    launch = runtime.symbol(API)
    launch.restype = C.c_int
    launch.argtypes = [C.c_void_p, C.c_uint64, C.c_void_p, C.c_void_p]
    size, executor = C.c_uint64(), C.c_void_p()
    pad = 512
    try:
        rc = getter(*descriptors(ctx), C.byref(size), C.byref(executor))
        if rc or not executor.value or size.value > 64 << 20:
            raise RuntimeError(f"GetWorkspaceSize failed or exceeded bounds: rc={rc}, bytes={size.value}")
        scratch = torch.full((size.value + 2 * pad,), 0xA5, dtype=torch.uint8, device=x.device)
        stream = runtime_api.current_stream_ptr(x.device)
        rc = launch(C.c_void_p(scratch.data_ptr() + pad), size, executor, C.c_void_p(stream))
        if rc:
            raise RuntimeError(f"experimental ACLNN launch returned {rc}")
        torch.npu.synchronize()
        guards = {
            "workspace": bool((scratch[:pad] == 0xA5).all() & (scratch[-pad:] == 0xA5).all()),
            "output": bool(torch.isnan(backing[:guard]).all() & torch.isnan(backing[-guard:]).all()),
        }
    finally:
        ctx.destroy()
    return out, {"workspace_bytes": size.value, "guards": guards}


def run(args, report, save):
    os.environ["ACLNN_CACHE_LIMIT"] = "1000000"
    os.environ["ASCEND_CUSTOM_OPP_PATH"] = str(args.vendor) + (
        ":" + os.environ["ASCEND_CUSTOM_OPP_PATH"] if os.environ.get("ASCEND_CUSTOM_OPP_PATH") else "")
    import torch
    import torch.nn.functional as F
    import torch_npu
    from fla_npu.ops.ascendc import _runtime

    torch.set_num_threads(1)
    runtime = _runtime.runtime()
    lib = args.vendor / "op_api/lib/libcust_opapi.so"
    runtime._libraries.insert(0, C.CDLL(str(lib), mode=os.RTLD_LOCAL | os.RTLD_NOW))
    captures = args.captures.read_bytes()
    case = next(c for c in json.loads(captures)["cases"] if c["name"] == "od-0677")
    report.update(capture_sha256=hashlib.sha256(captures).hexdigest(),
                  opapi_sha256=hashlib.sha256(lib.read_bytes()).hexdigest(),
                  order={"layers": LAYERS, "pairs": PAIRS, "ranks": args.ranks},
                  input="od-0677 on aa3e, 1595-token parents, explicit prefixes for 307 and 967",
                  ranks={})
    begin = time.monotonic()

    for rank in args.ranks:
        row = report["ranks"][rank] = {"inputs": [], "shapes": {}}
        initialize_npu(torch, torch_npu, rank)
        device = f"npu:{rank}"
        layers = []
        for layer in LAYERS:
            gate, down, hashes = load_weights(args.model_dir, torch, layer, rank)
            entry = next(t for t in case["tensors"] if t["metadata"]["layer"] == layer
                         and t["metadata"]["rank"] == rank)
            matrix = torch.load(entry["path"], map_location="cpu", weights_only=True)["input"].contiguous()
            if matrix.dtype != torch.bfloat16 or tuple(matrix.shape) != (1595, HIDDEN):
                raise ValueError("expected canonical BF16 capture")
            parent_sha = tensor_sha(matrix, torch)
            prefix_sha = tensor_sha(matrix[:967], torch)
            if parent_sha != entry["input_sha256"] or prefix_sha != entry["m967_prefix_sha256"]:
                raise ValueError("capture digest mismatch")
            row["inputs"].append({"layer": layer, "parent_sha256": parent_sha,
                                  "307_prefix_sha256": tensor_sha(matrix[:307], torch),
                                  "967_prefix_sha256": prefix_sha, "weights": hashes})
            cpu_weights = (gate, down, pair_columns(torch, gate, 128), pair_columns(torch, gate, 64))
            nz_weights = tuple(torch_npu.npu_format_cast(w.to(device), 29) for w in cpu_weights)
            if any(int(torch_npu.get_npu_format(w)) != 29 for w in nz_weights):
                raise ValueError("expected NZ29 weight storage")
            layers.append((matrix.to(device), *nz_weights))
        torch.npu.synchronize()

        def call(x, w, mode, guarded=False):
            return dispatch(_runtime, runtime, torch, x, w, mode, guarded)

        for tokens in args.tokens:
            if time.monotonic() - begin > args.limit_s:
                raise TimeoutError("experiment deadline")
            shapes = row["shapes"][tokens] = {"numeric": [], "timing": []}
            inputs = [(x[:tokens], g, d, p, s) for x, g, d, p, s in layers]
            candidates = ((0, 128, 3), (2, 64, 4)) if tokens == 307 else ((0, 128, 3),)
            for layer, data in zip(LAYERS, inputs):
                x, g, d, _, _ = data
                ref = F.linear(x, g)
                gate_ref, up_ref = ref.chunk(2, dim=-1)
                activation_ref = torch_npu.npu_swiglu(ref)
                down_ref = F.linear(activation_ref, d)
                for mode, half, index in candidates:
                    w = data[index]
                    raw, raw_info = call(x, w, mode, True)
                    act, act_info = call(x, w, mode + 1, True)
                    gate, up = unpack_columns(raw, half)
                    result = {"layer": layer, "mode": mode, "raw": raw_info, "fused": act_info,
                              "gate": compare(torch, gate, gate_ref, 3e-4, .006),
                              "up": compare(torch, up, up_ref, 3e-4, .006),
                              "activation": compare(torch, act, activation_ref, .002, .01),
                              "activation_same_operands": compare(
                                  torch, act, torch_npu.npu_swiglu(torch.cat((gate, up), dim=-1)), 8e-4, .006),
                              "chain": compare(torch, F.linear(act, d), down_ref, .003, .015),
                              "raw_repeat_exact": bool(torch.equal(raw, call(x, w, mode))),
                              "fused_repeat_exact": bool(torch.equal(act, call(x, w, mode + 1)))}
                    result["passed"] = (
                        all(raw_info["guards"].values()) and all(act_info["guards"].values())
                        and result["raw_repeat_exact"] and result["fused_repeat_exact"]
                        and all(result[k]["passed"] for k in ("gate", "up", "activation",
                                                              "activation_same_operands", "chain")))
                    shapes["numeric"].append(result)
                    save()
                    if not result["passed"]:
                        raise RuntimeError(f"BF16 gate failed: rank {rank}, M{tokens}, layer {layer}, mode {mode}")

            def execute(stage, arm):
                for data in inputs:
                    x, g, d, _, _ = data
                    if arm.startswith("vendor"):
                        out = F.linear(x, g)
                        if stage != "raw":
                            out = torch_npu.npu_swiglu(out)
                    else:
                        index, mode = (4, 2) if arm == "short" else (3, 0)
                        out = call(x, data[index], mode + (stage != "raw"))
                    if stage == "chain":
                        out = F.linear(out, d)
                return out

            for stage in ("raw", "activation", "chain"):
                arms = ["vendor", "candidate", "vendor_repeat"] + (["short"] if tokens == 307 else [])
                for arm in arms:
                    for _ in range(4):
                        execute(stage, arm)
                    torch.npu.synchronize()
                orders = random.Random(5529 + tokens)
                for pair in range(PAIRS):
                    order = list(arms)
                    orders.shuffle(order)
                    for arm in order:
                        start, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
                        begin_ns = time.perf_counter_ns()
                        start.record()
                        execute(stage, arm)
                        end.record()
                        end.synchronize()
                        wall = (time.perf_counter_ns() - begin_ns) / len(LAYERS) / 1e3
                        event = start.elapsed_time(end) * 1e3 / len(LAYERS)
                        shapes["timing"].append({"stage": stage, "pair": pair, "arm": arm,
                                                 "wall_us": wall, "event_us": event})
                save()
        del layers, inputs, nz_weights, cpu_weights, x, g, d, w, data
        torch.npu.empty_cache()
    report["seconds"] = time.monotonic() - begin


def summarize(report):
    rows = report["ranks"]
    output = {}
    for tokens in rows[next(iter(rows))]["shapes"]:
        samples = {}
        for row in rows.values():
            for item in row["shapes"][tokens]["timing"]:
                key = (item["stage"], item["arm"], item["pair"])
                value = samples.setdefault(key, {})
                for metric in ("wall_us", "event_us"):
                    value[metric] = max(value.get(metric, 0), item[metric])
        stages = output[tokens] = {}
        for stage in ("raw", "activation", "chain"):
            arms = stages[stage] = {}
            for arm in ("vendor", "candidate", "short", "vendor_repeat"):
                if (stage, arm, 0) not in samples:
                    continue
                arms[arm] = {}
                for metric in ("wall_us", "event_us"):
                    times = [samples[stage, arm, i][metric] for i in range(PAIRS)]
                    base = [samples[stage, "vendor", i][metric] for i in range(PAIRS)]
                    delta = [a - b for a, b in zip(times, base)]
                    arms[arm][metric] = {"median": statistics.median(times),
                                         "paired_delta_to_vendor": statistics.median(delta),
                                         "paired_abs_max": max(map(abs, delta)),
                                         "wins": sum(x < 0 for x in delta)}
    return output


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--vendor", type=Path, required=True)
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--captures", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--tokens", type=int, nargs="+", default=SHAPES, choices=SHAPES)
    p.add_argument("--ranks", type=int, nargs="+", default=(0, 1), choices=(0, 1))
    p.add_argument("--limit-s", type=int, default=140)
    args = p.parse_args()
    if args.output.exists() or len(set(args.tokens)) != len(args.tokens) or len(set(args.ranks)) != len(args.ranks):
        p.error("use unique ranks/shapes and a new output")
    report = {"status": "running"}

    def save():
        target = args.output.with_suffix(".tmp")
        target.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        target.replace(args.output)

    try:
        run(args, report, save)
        report["summary"] = summarize(report)
        report["status"] = "passed"
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
