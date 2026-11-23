#!/usr/bin/env python3
"""Profile BF16 gate/up memory and validate the M307 padding alternative."""
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
from probe_mlp_gate_up_pair import (
    HIDDEN, LAYERS, compare, dispatch, pair_columns, tensor_sha, unpack_columns,
)

TIMING_PAIRS = 9
PROFILE_PASSES = 4
METRICS = ("Memory", "L2Cache", "PipeUtilization")


def prepare(args, torch, torch_npu, rank):
    initialize_npu(torch, torch_npu, rank)
    device = f"npu:{rank}"
    manifest = args.captures.read_bytes()
    case = next(c for c in json.loads(manifest)["cases"] if c["name"] == "od-0677")
    inputs, evidence = [], []
    for layer in LAYERS:
        gate, down, weight_sha = load_weights(args.model_dir, torch, layer, rank)
        item = next(t for t in case["tensors"] if t["metadata"]["layer"] == layer
                    and t["metadata"]["rank"] == rank)
        parent = torch.load(item["path"], map_location="cpu", weights_only=True)["input"].contiguous()
        if parent.dtype != torch.bfloat16 or tuple(parent.shape) != (1595, HIDDEN):
            raise ValueError("unexpected captured parent shape or dtype")
        source_sha = tensor_sha(parent, torch)
        if source_sha != item["input_sha256"] or tensor_sha(parent[:967], torch) != item["m967_prefix_sha256"]:
            raise ValueError("captured parent or M967 prefix digest differs")
        evidence.append({"layer": layer, "parent_sha256": source_sha,
                         "m307_prefix_sha256": tensor_sha(parent[:307], torch), "weights": weight_sha})
        cpu_weights = (gate, pair_columns(torch, gate, 128), down)
        weights = tuple(torch_npu.npu_format_cast(w.to(device), 29) for w in cpu_weights)
        if any(int(torch_npu.get_npu_format(w)) != 29 for w in weights):
            raise ValueError("expected NZ29 weight storage")
        inputs.append((parent.to(device), *weights))
    torch.npu.synchronize()
    return inputs, {"manifest_sha256": hashlib.sha256(manifest).hexdigest(), "inputs": evidence}


def raw(F, custom, row, m, arm):
    parent, gate, paired = row[:3]
    x = parent[:m]
    if arm == "vendor_paired":
        return F.linear(x, paired)
    if arm == "custom":
        return custom(x, paired)
    if arm == "pad320":
        return F.linear(F.pad(x, (0, 0, 0, 320 - m)), gate)[:m]
    if arm == "prepad320":
        padded_parent = row[4]
        return F.linear(padded_parent, gate)[:m]
    return F.linear(x, gate)


def measure(torch, F, torch_npu, custom, layers, result):
    result.update(numeric=[], samples=[])
    for layer, row in zip(LAYERS, layers):
        parent, gate, _, down = row
        base = F.linear(parent[:307], gate)
        padded = raw(F, custom, row, 307, "pad320")
        gate_comp = compare(torch, padded, base, 3e-4, .006)
        chain_comp = compare(torch, F.linear(torch_npu.npu_swiglu(padded), down),
                            F.linear(torch_npu.npu_swiglu(base), down), .003, .015)
        packed = raw(F, custom, row, 307, "vendor_paired")
        gate_part, up_part = unpack_columns(packed, 128)
        permuted = compare(torch, torch.cat((gate_part, up_part), dim=-1), base, 3e-4, .006)
        result["numeric"].append({"layer": layer, "pad307_raw": gate_comp,
                                  "pad307_chain": chain_comp, "vendor_column_pair": permuted})
        if not (gate_comp["passed"] and chain_comp["passed"] and permuted["passed"]):
            raise RuntimeError(f"numerical gate failed on layer {layer}")

    padded_layers = [(*row, F.pad(row[0][:307], (0, 0, 0, 13))) for row in layers]
    torch.npu.synchronize()

    def execute(m, stage, arm):
        for row in padded_layers:
            out = raw(F, custom, row, m, arm)
            if stage == "chain":
                out = F.linear(torch_npu.npu_swiglu(out), row[3])
        return out

    configs = (
        (307, "raw", ("vendor", "vendor_repeat", "vendor_paired", "custom", "pad320", "prepad320")),
        (307, "chain", ("vendor", "vendor_repeat", "pad320")),
        (967, "raw", ("vendor", "vendor_repeat", "vendor_paired", "custom")),
    )
    record_timings(torch, execute, configs, result["samples"])


def record_timings(torch, execute, configs, samples):
    for m, stage, arms in configs:
        for arm in arms:
            for _ in range(4):
                execute(m, stage, arm)
            torch.npu.synchronize()
        order_gen = random.Random(3277 + m)
        for pair in range(TIMING_PAIRS):
            order = list(arms)
            order_gen.shuffle(order)
            for arm in order:
                start, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
                stamp = time.perf_counter_ns()
                start.record()
                execute(m, stage, arm)
                end.record()
                end.synchronize()
                wall = (time.perf_counter_ns() - stamp) / len(LAYERS) / 1000
                event = start.elapsed_time(end) * 1000 / len(LAYERS)
                samples.append({"m": m, "stage": stage, "arm": arm, "pair": pair,
                                "wall_us": wall, "event_us": event})



def measure_native(torch, F, torch_npu, python_call, native_call, layers, result, hybrid):
    result.update(numeric=[], samples=[])
    shapes = (307, 967, 1536, 1568, 1595) if hybrid else (307, 967, 1595)
    for m in shapes:
        for layer, row in zip(LAYERS, layers):
            parent, gate, paired, down = row
            x = parent[:m]
            vendor = F.linear(torch_npu.npu_swiglu(F.linear(x, gate)), down)
            raw = native_call(x, paired, 0)
            fused = native_call(x, paired, 1)
            entry = {"m": m, "layer": layer,
                     "raw_bitwise": bool(torch.equal(raw, python_call(x, paired, 0))),
                     "fused_bitwise": bool(torch.equal(fused, python_call(x, paired, 1))),
                     "chain": compare(torch, F.linear(fused, down), vendor, .003, .015)}
            if hybrid:
                natural = native_call(x, gate, 0)
                entry["hybrid_chain"] = compare(
                    torch, F.linear(torch_npu.npu_swiglu(natural), down), vendor, .003, .015)
            result["numeric"].append(entry)
            if not (entry["raw_bitwise"] and entry["fused_bitwise"] and entry["chain"]["passed"]
                    and (not hybrid or entry["hybrid_chain"]["passed"])):
                raise RuntimeError(f"native bridge output mismatch for M{m}, layer {layer}")

    def execute(m, stage, arm):
        for parent, gate, paired, down in layers:
            x = parent[:m]
            if arm.startswith("vendor"):
                out = F.linear(x, gate)
                if stage == "chain":
                    out = torch_npu.npu_swiglu(out)
            elif arm == "hybrid":
                out = native_call(x, gate, 0)
                if stage == "chain":
                    out = torch_npu.npu_swiglu(out)
            else:
                call = native_call if arm == "native" else python_call
                out = call(x, paired, int(stage == "chain"))
            if stage == "chain":
                out = F.linear(out, down)
        return out

    arms = ("vendor", "vendor_repeat", "python", "native")
    if hybrid:
        arms += ("hybrid",)
    configs = [(m, stage, arms) for m in shapes for stage in ("raw", "chain")]
    record_timings(torch, execute, configs, result["samples"])

def profile(args, torch, F, torch_npu, custom, layers, report, native=None):
    prof = torch_npu.profiler
    workload = [(m, arm) for m in (307, 967) for arm in ("vendor", "vendor_paired", "custom")]
    if native is None:
        workload.append((307, "vendor_hot"))
    else:
        workload = [(m, arm) for m in (307, 967) for arm in ("vendor", "custom", "native")]

    def execute(m, arm, count):
        for _ in range(count):
            for row in (layers[:1] * len(layers) if arm == "vendor_hot" else layers):
                if arm == "native":
                    result = native(row[0][:m], row[2], 0)
                else:
                    result = raw(F, custom, row, m, arm)
        return result

    for m, arm in workload:
        execute(m, arm, 3)
    torch.npu.synchronize()
    report.update(arm_order=[{"m": m, "arm": arm, "calls": PROFILE_PASSES * len(LAYERS)}
                             for m, arm in workload])
    trace_dir = args.output.parent / (args.metric + "-trace")
    trace_dir.mkdir()
    config = prof._ExperimentalConfig(
        profiler_level=prof.ProfilerLevel.Level1,
        aic_metrics=getattr(prof.AiCMetrics, args.metric),
        data_simplification=False,
    )
    with prof.profile(activities=[prof.ProfilerActivity.CPU, prof.ProfilerActivity.NPU],
                      record_shapes=True, with_stack=False, experimental_config=config,
                      on_trace_ready=prof.tensorboard_trace_handler(str(trace_dir))):
        for m, arm in workload:
            with torch.profiler.record_function(f"clef_gateup_{m}_{arm}"):
                execute(m, arm, PROFILE_PASSES)
                torch.npu.synchronize()
    report["trace_dir"] = str(trace_dir)


def summarize(ranks):
    pairs = {}
    for rank in ranks.values():
        for row in rank["samples"]:
            key = (row["m"], row["stage"], row["arm"], row["pair"])
            worst = pairs.setdefault(key, {})
            for metric in ("wall_us", "event_us"):
                worst[metric] = max(worst.get(metric, 0), row[metric])
    result = {}
    for m, stage, arm, pair in sorted(pairs):
        if pair:
            continue
        record = result.setdefault(str(m), {}).setdefault(stage, {}).setdefault(arm, {})
        for metric in ("wall_us", "event_us"):
            times = [pairs[m, stage, arm, p][metric] for p in range(TIMING_PAIRS)]
            deltas = [v - pairs[m, stage, "vendor", p][metric] for p, v in enumerate(times)]
            record[metric] = {"median": statistics.median(times),
                              "paired_delta": statistics.median(deltas),
                              "paired_abs_max": max(map(abs, deltas)),
                              "wins": sum(v < 0 for v in deltas)}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metric", choices=METRICS)
    parser.add_argument("--native-library", type=Path)
    parser.add_argument("--hybrid", action="store_true")
    parser.add_argument("--vendor", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--captures", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.hybrid and (not args.native_library or args.metric):
        parser.error("--hybrid requires native timing")
    if args.output.exists():
        parser.error("use a fresh output path")
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

    def python_call(x, weights, mode):
        return dispatch(_runtime, runtime, torch, x, weights, mode)

    def custom(x, weights):
        return python_call(x, weights, 0)

    report = {"status": "running", "opapi_sha256": hashlib.sha256(lib.read_bytes()).hexdigest(),
              "metric": args.metric, "ranks": {}}

    native = None
    if args.native_library:
        os.environ["CLEF_GATE_UP_OPAPI"] = str(lib.resolve())
        torch.ops.load_library(str(args.native_library))
        native = torch.ops.clef_gateup_experiment.run
        report["native_sha256"] = hashlib.sha256(args.native_library.read_bytes()).hexdigest()
    start = time.monotonic()
    try:
        for rank in ((0,) if args.metric else (0, 1)):
            layers, evidence = prepare(args, torch, torch_npu, rank)
            report["ranks"][rank] = evidence
            if args.metric:
                profile(args, torch, F, torch_npu, custom, layers, evidence, native)
            elif args.native_library:
                measure_native(torch, F, torch_npu, python_call, native, layers, evidence, args.hybrid)
            else:
                measure(torch, F, torch_npu, custom, layers, evidence)
            del layers
            torch.npu.empty_cache()
        if not args.metric:
            report["summary"] = summarize(report["ranks"])
        report["status"] = "passed"
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        report["seconds"] = time.monotonic() - start
        with args.output.open("x") as f:
            json.dump(report, f, indent=2, allow_nan=False)
            f.write("\n")


if __name__ == "__main__":
    main()
