#!/usr/bin/env python3
"""Replay original TP2 MLP weights and archived real inputs. Default is CPU plan."""
from __future__ import annotations

import argparse
import ctypes as C
import hashlib
import json
import os
from pathlib import Path
import time

H = 5120
I = 8704
LAYERS = (31, 0, 63)
PAIRS = 5
ARMS = ("vendor", "fused", "vendor_repeat")
API = "aclnnClefMlpFullFusion"
ABI = [C.c_void_p] * 3 + [C.c_int64, C.c_void_p, C.POINTER(C.c_uint64), C.POINTER(C.c_void_p)]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--phase", choices=("numeric", "timing", "profile"), default="numeric")
    parser.add_argument("--tokens", type=int, nargs="+", choices=(307, 967, 1595), default=[967])
    parser.add_argument("--ranks", type=int, nargs="+", choices=(0, 1), default=[0, 1])
    parser.add_argument("--vendor", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--limit-s", type=float, default=55)
    parser.add_argument("--model-dir", type=Path, default=Path("/workspace/yangchen/models/Clef-27B"))
    parser.add_argument("--captures", type=Path, default=Path(
        "/workspace/yangchen/logs/mlp-20261115-1027/capture-requests.json"))
    args = parser.parse_args()
    if args.execute and (not args.vendor or not args.output or args.output.exists()):
        parser.error("execution requires a private vendor and a new output")
    return args


def execute(args, report):
    os.environ["ACLNN_CACHE_LIMIT"] = "1000000"
    os.environ["ASCEND_CUSTOM_OPP_PATH"] = str(args.vendor) + ":" + os.environ.get("ASCEND_CUSTOM_OPP_PATH", "")
    import torch
    import torch.nn.functional as F
    import torch_npu
    from fla_npu.ops.ascendc import _runtime
    from probe_mlp_ffn import initialize_npu, load_weights

    torch.set_num_threads(1)
    runtime = _runtime.runtime()
    libpath = args.vendor / "op_api/lib/libcust_opapi.so"
    library = C.CDLL(str(libpath), mode=os.RTLD_LOCAL | os.RTLD_NOW)
    runtime._libraries.insert(0, library)
    begin = time.monotonic()
    manifest_data = args.captures.read_bytes()
    manifest = json.loads(manifest_data)
    report.update(status="running", cases=[], capture_manifest_sha256=hashlib.sha256(manifest_data).hexdigest(),
                  opapi_sha256=hashlib.sha256(libpath.read_bytes()).hexdigest())

    def save():
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    def check_deadline():
        if time.monotonic() - begin > args.limit_s:
            raise TimeoutError("probe deadline")

    def descriptors(ctx, x, gate, down, output, diagnostic):
        return [
            ctx.tensor(x, "x"),
            ctx.tensor(gate, "gate", acl_format_override=29,
                       storage_shape_override=(H // 16, 2 * I // 16, 16, 16)),
            ctx.tensor(down, "down", acl_format_override=29,
                       storage_shape_override=(I // 16, H // 16, 16, 16)),
            diagnostic, ctx.tensor(output, "output")
        ]

    def fused(x, gate, down):
        output = torch.empty((len(x), H), device=x.device, dtype=torch.bfloat16)
        _runtime.call_aclnn(API, lambda ctx: descriptors(ctx, x, gate, down, output, 0), (output,),
                            get_workspace_argtypes=ABI)
        return output

    def diagnosed(x, gate, down):
        # Same generated entry, inspected requested workspace and guard regions.
        # The ordinary benchmark uses the runtime's allocation and dispatch path.
        guard = 512
        out_guard = guard // 2
        width = H + 3 * I
        output_store = torch.full((len(x) * width + 2 * out_guard,), float("nan"),
                                  dtype=torch.bfloat16, device=x.device)
        output = output_store[out_guard:-out_guard].view(len(x), width)
        ctx = _runtime._CallContext(runtime, x.device)
        get = runtime.symbol(API + "GetWorkspaceSize")
        get.restype, get.argtypes = C.c_int, ABI
        launch = runtime.symbol(API)
        launch.restype = C.c_int
        launch.argtypes = [C.c_void_p, C.c_uint64, C.c_void_p, C.c_void_p]
        size, executor = C.c_uint64(), C.c_void_p()
        try:
            rc = get(*descriptors(ctx, x, gate, down, output, 1), C.byref(size), C.byref(executor))
            if rc or not executor.value:
                raise RuntimeError(f"diagnostic GetWorkspace failed: {rc}")
            user_size = (24 * 2 * 128 * 512 + len(x) * I) * 2
            if not user_size <= size.value <= user_size + 64 * 1024**2:
                raise RuntimeError(f"unexpected workspace: {size.value}, user region {user_size}")
            scratch = torch.full((size.value + 2 * guard,), 0xA5, dtype=torch.uint8, device=x.device)
            stream = _runtime.current_stream_ptr(x.device)
            rc = launch(C.c_void_p(scratch.data_ptr() + guard), size, executor, C.c_void_p(stream))
            if rc:
                raise RuntimeError(f"diagnostic launch failed: {rc}")
            torch.npu.synchronize()
            guards = {
                "output": bool(torch.isnan(output_store[:out_guard]).all() &
                               torch.isnan(output_store[-out_guard:]).all()),
                "workspace": bool((scratch[:guard] == 0xA5).all() & (scratch[-guard:] == 0xA5).all()),
            }
        finally:
            ctx.destroy()
        sections = output.view(-1).split([len(x) * H] + [len(x) * I] * 3)
        return [part.view(len(x), -1) for part in sections], size.value, guards

    def compared(candidate, ref, rel_limit, max_limit):
        finite = bool(torch.isfinite(candidate).all() & torch.isfinite(ref).all())
        if not finite:
            return dict(finite=False, passed=False)
        a, b = candidate.float(), ref.float()
        error = a - b
        relative = float(error.square().mean().sqrt() / b.square().mean().sqrt().clamp_min(1e-30))
        max_abs = float(error.abs().max())
        normalized = max_abs / max(float(b.abs().max()), 1e-30)
        return dict(finite=True, rel_rmse=relative, max_abs=max_abs, normalized_max=normalized,
                    exact=bool(torch.equal(candidate, ref)),
                    passed=relative <= rel_limit and normalized <= max_limit)

    def prepare(rank, tokens, row):
        initialize_npu(torch, torch_npu, rank)
        device = f"npu:{rank}"
        case = next(c for c in manifest["cases"] if c["name"] == ("od-0105" if tokens == 307 else "od-0677"))
        result = []
        for layer in LAYERS:
            check_deadline()
            start = time.monotonic()
            packed, down, weights = load_weights(args.model_dir, torch, layer, rank)
            item = next(t for t in case["tensors"] if t["metadata"]["layer"] == layer
                        and t["metadata"]["rank"] == rank)
            full = torch.load(item["path"], map_location="cpu", weights_only=True)["input"].contiguous()
            digest = hashlib.sha256(memoryview(full.view(torch.uint8).numpy())).hexdigest()
            if (digest != item["input_sha256"] or full.dtype != torch.bfloat16 or
                    full.shape[1] != H or len(full) < tokens):
                raise ValueError("real input capture mismatch")
            x = full[:tokens].contiguous().to(device)
            g = torch_npu.npu_format_cast(packed.to(device), 29)
            d = torch_npu.npu_format_cast(down.to(device), 29)
            torch.npu.synchronize()
            if (int(torch_npu.get_npu_format(x)), int(torch_npu.get_npu_format(g)),
                int(torch_npu.get_npu_format(d))) != (2, 29, 29):
                raise ValueError("expected ND input and original NZ29 weights")
            row["preparation"].append(dict(layer=layer, input_sha256=digest, weights=weights,
                input_kind="explicit_1595_prefix" if tokens == 967 else "full_capture",
                seconds=time.monotonic() - start))
            result.append((x, g, d))
        return result

    def numerical(inputs, row):
        for layer, (x, g, d) in zip(LAYERS, inputs):
            check_deadline()
            entry = dict(layer=layer)
            row["numeric"].append(entry)
            save()
            vendor_gate_up = F.linear(x, g)
            gate_ref, up_ref = vendor_gate_up.chunk(2, dim=-1)
            act_ref = torch_npu.npu_swiglu(vendor_gate_up)
            out_ref = F.linear(act_ref, d)
            (y, gate, up, activation), workspace, guards = diagnosed(x, g, d)
            entry.update(workspace_bytes=workspace, guards=guards)
            entry["gate"] = compared(gate, gate_ref, 3e-4, .006)
            entry["up"] = compared(up, up_ref, 3e-4, .006)
            entry["activation_same_operands"] = compared(
                activation, torch_npu.npu_swiglu(torch.cat((gate, up), dim=-1)), 8e-4, .006)
            entry["activation"] = compared(activation, act_ref, .002, .01)
            entry["down_same_operands"] = compared(y, F.linear(activation, d), .001, .008)
            entry["chain"] = compared(y, out_ref, .003, .015)
            out = fused(x, g, d)
            entry["ordinary_matches_diagnostic"] = bool(torch.equal(out, y))
            entry["ordinary_repeat_exact"] = bool(torch.equal(out, fused(x, g, d)))
            entry["canonical_nd"] = (out.dtype == torch.bfloat16 and out.shape == out_ref.shape
                                     and out.stride() == (H, 1) and torch_npu.get_npu_format(out) == 2)
            passed = (all(guards.values()) and entry["ordinary_matches_diagnostic"]
                      and entry["ordinary_repeat_exact"] and entry["canonical_nd"]
                      and all(entry[k]["passed"] for k in ("gate", "up", "activation_same_operands",
                          "activation", "down_same_operands", "chain")))
            entry["passed"] = passed
            save()
            if not passed:
                raise RuntimeError(f"stage check failed for layer {layer}")

    def sample(inputs, row, mode, pairs, profile=None):
        count = len(inputs)

        def call(arm):
            for x, w, d in inputs:
                if arm == "fused":
                    out = fused(x, w, d)
                else:
                    out = F.linear(torch_npu.npu_swiglu(F.linear(x, w)), d)
            return out

        warmup = {}
        for arm in ARMS[:2]:
            start = time.monotonic()
            for _ in range(5):
                call(arm)
            torch.npu.synchronize()
            warmup[arm] = dict(calls=count * 5, seconds=time.monotonic() - start)
        row["warmup"].append(dict(mode=mode, phase=profile or "unprofiled", arms=warmup))
        # Bracket each three-layer pass once. Include allocation and both dispatch paths.
        for pair in range(pairs):
            for arm in ARMS if pair % 2 == 0 else reversed(ARMS):
                check_deadline()
                first, last = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
                label = f"fullmlp/{profile}/{pair}/{arm}"
                from contextlib import nullcontext
                marker = torch.profiler.record_function(label) if profile else nullcontext()
                with marker:
                    start = time.perf_counter()
                    first.record()
                    result = call(arm)
                    last.record()
                    torch.npu.synchronize()
                    wall = time.perf_counter() - start
                row["samples"].append(dict(phase=profile or "unprofiled", mode=mode, pair=pair, arm=arm,
                    event_ms=first.elapsed_time(last) / count, wall_ms=wall * 1000 / count))
                del result
            save()

    try:
        with torch.inference_mode():
            for tokens in args.tokens:
                for rank in args.ranks:
                    row = dict(rank=rank, m=tokens, preparation=[], numeric=[], warmup=[], samples=[])
                    report["cases"].append(row)
                    save()
                    inputs = prepare(rank, tokens, row)
                    if args.phase == "numeric":
                        numerical(inputs, row)
                    elif args.phase == "timing":
                        sample(inputs, row, "streaming_31_0_63", PAIRS)
                        sample(inputs[:1], row, "warm_31", PAIRS)
                    else:
                        assert len(args.ranks) == len(args.tokens) == 1
                        sample(inputs, row, "streaming_31_0_63", 2)
                        cfg = torch_npu.profiler._ExperimentalConfig(
                            profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
                            aic_metrics=torch_npu.profiler.AiCMetrics.AiCoreNone,
                            data_simplification=False, record_op_args=False,
                            export_type=[torch_npu.profiler.ExportType.Text])
                        trace_dir = args.output.parent / f"traces-r{rank}"
                        with torch_npu.profiler.profile(
                            activities=[torch_npu.profiler.ProfilerActivity.CPU,
                                        torch_npu.profiler.ProfilerActivity.NPU],
                            experimental_config=cfg, record_shapes=False, profile_memory=False,
                            with_stack=False, with_modules=False,
                            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(
                                str(trace_dir), worker_name=f"fullmlp-r{rank}", analyse_flag=False)):
                            sample(inputs, row, "streaming_31_0_63", 2, "profile")
                        sample(inputs, row, "streaming_31_0_63", 2)
                    del inputs
                    torch.npu.synchronize()
                    torch.npu.empty_cache()
        report["status"] = "complete"
    except BaseException as error:
        report.update(status="failed", error=repr(error))
        raise
    finally:
        report["device_process_s"] = time.monotonic() - begin
        save()


def main():
    args = parse_args()
    report = dict(status="plan_only", phase=args.phase, tokens=args.tokens, ranks=args.ranks, layers=LAYERS,
                  arms=ARMS, pairs=PAIRS, output="BF16 ND", inputs="same archived tokens and original NZ29 weights",
                  candidate="one MIX launch, workspace per call, both CUBE segments accumulate FP32 over full K")
    if args.execute:
        execute(args, report)
    else:
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
