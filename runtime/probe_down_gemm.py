#!/usr/bin/env python3
"""Bounded BF16 TP2 down-GEMM diagnostic; default is a CPU-only plan.

Only the device owner may use --execute. No service calls, source patches,
subprocesses, communication, padding or background work are performed.
Timing and external profiling are separate invocations. The ring conditions
the target with three other real down weights (255 MiB); it is a cache proxy,
not a replay of the 64-layer service. All inputs are derived with the vendor
BF16 gate/SwiGLU from historical captured MLP inputs.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import time
from pathlib import Path
from types import SimpleNamespace

M_VALUES = (307, 967, 1595)
MODES = ("warm", "read_flush_proxy", "streamed_down_ring")
CAPTURES = {
    307: "od-0105-7b4ab816610eaeda",
    967: "od-0677-a3153de1d6509904",
    1595: "od-0677-a3153de1d6509904",
}
K, N = 8704, 5120


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("plan", "timing", "once"), default="plan")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--ranks", default="0,1")
    parser.add_argument("--tokens", default="307,967,1595")
    parser.add_argument("--pairs", type=int, default=5)
    parser.add_argument("--model-dir", type=Path, default=Path("/workspace/yangchen/models/Clef-27B"))
    parser.add_argument("--capture-dir", type=Path,
                        default=Path("/workspace/yangchen/logs/mlp-20261115-1027/capture"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        args.ranks = [int(x) for x in args.ranks.split(",")]
        args.tokens = [int(x) for x in args.tokens.split(",")]
    except ValueError:
        parser.error("ranks and tokens must be comma-separated integers")
    if not args.ranks or len(set(args.ranks)) != len(args.ranks) or set(args.ranks) - {0, 1}:
        parser.error("ranks must be unique TP2 ranks 0/1; device equals rank")
    if not args.tokens or len(set(args.tokens)) != len(args.tokens) or set(args.tokens) - set(M_VALUES):
        parser.error("only unique M=307,967,1595 are allowed; no tail sweep")
    if not 3 <= args.pairs <= 5:
        parser.error("pairs must be in [3,5]")
    if args.phase != "plan" and (not args.execute or args.output is None):
        parser.error("device phases require --execute and --output")
    if args.phase == "once" and (len(args.ranks) != 1 or args.tokens != [967]):
        parser.error("external-profile replay requires one rank and --tokens 967")
    if args.output and args.output.exists():
        parser.error("output already exists; use a new empty output path")
    return args


def plan(args):
    return {
        "schema": "clef-down-gemm-v1", "phase": args.phase,
        "ranks_serial": args.ranks, "device_equals_rank": True,
        "m": args.tokens, "k": K, "n": N, "layer": 31,
        "dtype": "BF16", "weight_format": "FRACTAL_NZ29",
        "pairs": args.pairs, "modes": list(MODES),
        "target_calls_timing_max": len(args.ranks) * len(args.tokens) * (3 + 8 * args.pairs),
        "decoy_calls_timing_max": len(args.ranks) * len(args.tokens) * 6 * args.pairs,
        "extra_live_tensor_mib_estimate": 1000,
        "allocated_tensor_cap_mib": 1536,
        "expected_owner_timeout_seconds": 240,
        "timing_scope": "single down F.linear; event, enqueue and synchronized wall separately",
        "excluded_setup": "CPU load, NZ weight conversion, gate/SwiGLU activation derivation, cache conditioning",
        "cache_scope": "warm; 256 MiB read-flush proxy; target+3 real-down-weight ring (340 MiB)",
        "service_scope": "isolated per-rank proxy; no allreduce or full-service latency claim",
        "input_scope": "historical layer31 capture; M967 is explicitly a prefix of M1595",
        "missing": ["actual vendor tiling", "per-core duration", "L1/L0 bytes and waits", "MTE/Cube/Fixpipe dependency waits"],
        "stop": ["missing capture/shape/BF16", "nonfinite or non-repeatable output", "allocation cap/error/timeout",
                 "no concrete tiling/pipe evidence: stop without custom tile sweep",
                 "candidate gain <= paired A/A noise: stop; service claims require owner end-to-end pairing"],
    }


def write_report(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


def tensor_sha(tensor, torch):
    return hashlib.sha256(memoryview(tensor.contiguous().view(torch.uint8).numpy())).hexdigest()


def load_decoy(args, layer, rank, torch):
    from safetensors import safe_open

    index = json.loads((args.model_dir / "model.safetensors.index.json").read_text())["weight_map"]
    names = [name for name in index if name.endswith(f".layers.{layer}.mlp.down_proj.weight")]
    if len(names) != 1:
        raise ValueError(f"Expected one layer{layer} down weight")
    name = names[0]
    with safe_open(str(args.model_dir / index[name]), framework="pt", device="cpu") as handle:
        view = handle.get_slice(name)
        if view.get_shape() != [N, 2 * K]:
            raise ValueError(f"Unexpected full shape for {name}")
        tensor = view[:, rank * K:(rank + 1) * K].contiguous()
    if tensor.dtype != torch.bfloat16:
        raise ValueError(f"Expected BF16 for {name}")
    return tensor, {"layer": layer, "rank": rank, "name": name, "sha256": tensor_sha(tensor, torch)}


def run_rank(args, rank, torch, torch_npu, helper, report, save_progress):
    import torch.nn.functional as functional

    helper.initialize_npu(torch, torch_npu, rank)
    device = f"npu:{rank}"
    packed_cpu, down_cpu, identity = helper.load_weights(args.model_dir, torch, 31, rank)
    if list(packed_cpu.shape) != [2 * K, N] or list(down_cpu.shape) != [N, K]:
        raise ValueError("Expected real TP2 gate [17408,5120] and down [5120,8704]")
    packed = torch_npu.npu_format_cast(packed_cpu.to(device), 29)
    down = torch_npu.npu_format_cast(down_cpu.to(device), 29)
    del packed_cpu, down_cpu
    decoys, decoy_metadata = [], []
    if args.phase == "timing":
        for layer in (0, 32, 63):
            cpu, metadata = load_decoy(args, layer, rank, torch)
            decoys.append(torch_npu.npu_format_cast(cpu.to(device), 29))
            decoy_metadata.append(metadata)
            del cpu
    flush_buffer = torch.ones(256 * 1024 * 1024 // 2, dtype=torch.bfloat16, device=device)
    report.update(rank=rank, device=device, identity=identity,
                  decoy_weights=decoy_metadata, cases=[])
    save_progress()

    def check_memory():
        allocated = int(torch.npu.memory_allocated())
        if allocated > 1536 * 1024 * 1024:
            raise RuntimeError(f"Allocated tensor cap exceeded: {allocated} bytes")
        return allocated

    for m in args.tokens:
        capture = args.capture_dir / f"{CAPTURES[m]}.layer31.rank{rank}.pt"
        cpu, metadata, _ = helper.load_captured_input(
            SimpleNamespace(input_file=capture, layer=31, rank=rank, tokens=[m]), N, torch)
        input_sha = tensor_sha(cpu, torch)
        x = cpu[:m].contiguous().to(device)
        y = torch_npu.npu_swiglu(functional.linear(x, packed))
        if list(y.shape) != [m, K] or y.dtype != torch.bfloat16:
            raise ValueError("Derived down input must be BF16 [M,8704]")
        if not bool(torch.isfinite(y).all().item()):
            raise ValueError("Nonfinite derived down input")
        y_cpu = y.cpu()
        case = {"m": m, "k": K, "n": N, "capture": str(capture),
                "capture_metadata": metadata, "capture_tensor_sha256": input_sha,
                "capture_prefix": m != cpu.shape[0],
                "down_input_sha256": tensor_sha(y_cpu, torch),
                "input_stride": list(y.stride()), "input_format": int(torch_npu.get_npu_format(y)),
                "weight_format": int(torch_npu.get_npu_format(down)),
                "allocated_tensor_bytes": check_memory(), "samples": []}
        del cpu, x, y_cpu
        reference = functional.linear(y, down)
        for _ in range(2):
            functional.linear(y, down)
        torch.npu.synchronize()
        if not bool(torch.isfinite(reference).all().item()):
            raise ValueError("Nonfinite BF16 reference output")

        def condition(mode):
            start = time.perf_counter()
            if mode == "warm":
                functional.linear(y, down)
            elif mode == "read_flush_proxy":
                torch.sum(flush_buffer)
            else:
                # Each decoy has distinct storage and real BF16 TP2 weights.
                # Together they read 255 MiB before the target 85 MiB weight.
                for weight in decoys:
                    functional.linear(y, weight)
            torch.npu.synchronize()
            return (time.perf_counter() - start) * 1000

        def sample(mode, pair, repeat):
            conditioning = condition(mode)
            begin = torch.npu.Event(enable_timing=True)
            end = torch.npu.Event(enable_timing=True)
            wall_start = time.perf_counter()
            begin.record()
            enqueue_start = time.perf_counter()
            actual = functional.linear(y, down)
            enqueue_ms = (time.perf_counter() - enqueue_start) * 1000
            end.record()
            torch.npu.synchronize()
            wall_ms = (time.perf_counter() - wall_start) * 1000
            device_ms = float(begin.elapsed_time(end))
            # Validation is outside both timing intervals.
            if not torch.equal(actual, reference):
                raise ValueError("Identical vendor down call produced non-repeatable output")
            return {"mode": mode, "pair": pair, "repeat": repeat,
                    "device_ms": device_ms, "wall_ms": wall_ms, "enqueue_ms": enqueue_ms,
                    "conditioning_wall_ms": conditioning,
                    "effective_tflops": 2 * m * K * N / (device_ms * 1e9), "bit_equal": True}

        if args.phase == "once":
            # External profiler owns collection. Exactly one cold target after
            # setup/warmup; only this shape is the diagnostic target.
            condition("read_flush_proxy")
            actual = functional.linear(y, down)
            torch.npu.synchronize()
            if not torch.equal(actual, reference):
                raise ValueError("External-profile replay output changed")
            case["external_profile_target_calls"] = 1
        else:
            for pair in range(args.pairs):
                order = MODES[pair % 3:] + MODES[:pair % 3]
                for mode in order:
                    for repeat in (0, 1):
                        case["samples"].append(sample(mode, pair, repeat))
        report["cases"].append(case)
        save_progress()
        print(json.dumps({"rank": rank, "m": m, "samples": len(case["samples"])}), flush=True)
        del y, reference
    return report


def main(argv=None):
    args = arguments(argv)
    report = plan(args)
    if args.phase == "plan":
        print(json.dumps(report, indent=2))
        return 0
    # Imports below this boundary initialize the owner's device process only.
    import torch
    import torch_npu

    spec = importlib.util.spec_from_file_location("down_probe_helpers", Path(__file__).with_name("probe_mlp_ffn.py"))
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    torch.set_num_threads(1)
    report.update(torch=torch.__version__, torch_npu=torch_npu.__version__, ranks=[], status="running")
    write_report(args.output, report)
    try:
        with torch.inference_mode():
            for rank in args.ranks:
                rank_report = {"rank": rank, "cases": []}
                report["ranks"].append(rank_report)
                run_rank(args, rank, torch, torch_npu, helper, rank_report,
                         lambda: write_report(args.output, report))
                write_report(args.output, report)
                torch.npu.synchronize()
                torch.npu.empty_cache()
        report["status"] = "complete"
    except Exception as error:
        report.update(status="stopped", error=f"{type(error).__name__}: {error}")
        write_report(args.output, report)
        raise
    write_report(args.output, report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
