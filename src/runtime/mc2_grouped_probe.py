#!/usr/bin/env python3
"""Prototype down_proj + AllReduce through aclnnGroupedMatMulAllReduce.

``npu_mm_all_reduce_base`` selects ``MatmulAllReduce``, which has no
ascend910_93 kernel in this CANN package (161001).  CANN 9.2.1 does ship
``aclnnGroupedMatMulAllReduce`` for ascend910_93, so this probe calls it with a
single-element tensor list on each TP rank and compares it with
``F.linear + all_reduce``.

Run with two ranks:
  torchrun --standalone --nproc_per_node=2 runtime/mc2_grouped_probe.py ...
"""

from __future__ import annotations

import argparse
import ctypes
import importlib.util
import json
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch_npu
import torch.nn.functional as functional

from fla_npu.ops.ascendc import _runtime as aclnn_runtime


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


HERE = Path(__file__).resolve().parent
probe = load("probe_mlp_ffn_for_mc2_grouped", HERE / "probe_mlp_ffn.py")

_OPAPI = ctypes.CDLL("libopapi.so")
_OPAPI.aclCreateTensorList.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint64]
_OPAPI.aclCreateTensorList.restype = ctypes.c_void_p
_OPAPI.aclDestroyTensorList.argtypes = [ctypes.c_void_p]
_OPAPI.aclDestroyTensorList.restype = ctypes.c_int


class _TensorList:
    def __init__(self, ctx, tensors):
        pointers = [ctx.tensor(tensor, "list") for tensor in tensors]
        array = (ctypes.c_void_p * len(pointers))(*[ptr.value for ptr in pointers])
        self.ptr = _OPAPI.aclCreateTensorList(array, len(pointers))
        self.array = array
        if not self.ptr:
            raise RuntimeError("aclCreateTensorList returned null")

    def destroy(self):
        _OPAPI.aclDestroyTensorList(self.ptr)


def tensor_list(ctx, tensors):
    resource = _TensorList(ctx, tensors)
    ctx.resources.append(resource)
    return ctypes.c_void_p(resource.ptr)


def timed(fn, warmup, iterations):
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    dist.barrier()
    start = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        fn()
    end.record()
    torch.npu.synchronize()
    dist.barrier()
    return start.elapsed_time(end) / iterations


def diff(actual, reference):
    delta = (actual.detach().float() - reference.detach().float()).abs()
    return {"bit_equal": bool(torch.equal(actual, reference)),
            "max_abs": float(delta.max().item())}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=Path("/workspace/yangchen/models/Clef-27B"))
    parser.add_argument("--layer", type=int, default=31)
    parser.add_argument("--tokens", default="307,969")
    parser.add_argument("--split-items", default="0")
    parser.add_argument("--bias-mode", choices=("none","zero"), default="none")
    parser.add_argument("--group-list", choices=("none","m"), default="none")
    parser.add_argument("--x-3d", action="store_true")
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.set_num_threads(1)
    probe.initialize_npu(torch, torch_npu, local_rank)
    dist.init_process_group("hccl", init_method="env://", world_size=world, rank=rank)
    backend = dist.group.WORLD._get_backend(torch.device("npu"))
    hcom = backend.get_hccl_comm_name(rank)
    device = f"npu:{local_rank}"

    _packed, down_cpu, identity = probe.load_weights(args.model_dir, torch, args.layer, rank)
    weight = down_cpu.to(device)
    weight_t = weight.t().contiguous()
    report = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "rank": rank, "world_size": world, "layer": args.layer,
        "hcom": hcom,
        "weight_shape": list(weight.shape),
        "weight_t_shape": list(weight_t.shape),
        "weight_t_contiguous": weight_t.is_contiguous(),
        "cases": [],
    }

    for tokens in [int(item) for item in args.tokens.split(",") if item.strip()]:
        generator = torch.Generator(device="cpu").manual_seed(20261115 + rank)
        x = (torch.randn(tokens, identity["intermediate_size"] // 2, generator=generator)
             .to(device, torch.bfloat16))
        reference = functional.linear(x, weight)
        dist.all_reduce(reference, op=dist.ReduceOp.SUM)
        for split in [int(item) for item in args.split_items.split(",") if item.strip()]:
            y = torch.empty(tokens, identity["hidden_size"], dtype=torch.bfloat16, device=device)
            bias = torch.zeros(identity["hidden_size"], dtype=torch.bfloat16, device=device)
            x_arg = x.unsqueeze(0) if args.x_3d else x
            y_arg = y.unsqueeze(0) if args.x_3d else y
            def call(split=split, y=y, x_arg=x_arg, y_arg=y_arg):
                return aclnn_runtime.call_aclnn(
                    "aclnnGroupedMatMulAllReduce",
                    lambda ctx: [
                        tensor_list(ctx, [x_arg]),
                        tensor_list(ctx, [weight_t]),
                        tensor_list(ctx, [bias]) if args.bias_mode == "zero" else None,
                        ctx.int_array([tokens]) if args.group_list == "m" else None,
                        ctypes.c_int64(split),
                        ctypes.c_char_p(hcom.encode("utf-8")),
                        ctypes.c_char_p(b"sum"),
                        ctypes.c_int64(0),
                        ctypes.c_int64(0),
                        tensor_list(ctx, [y_arg]),
                    ],
                    (y,),
                    get_workspace_argtypes=[
                        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                        ctypes.c_void_p, ctypes.c_int64, ctypes.c_char_p,
                        ctypes.c_char_p, ctypes.c_int64, ctypes.c_int64,
                        ctypes.c_void_p,
                        ctypes.POINTER(ctypes.c_uint64),
                        ctypes.POINTER(ctypes.c_void_p),
                    ],
                )
            try:
                call()
                torch.npu.synchronize()
                record = {
                    "tokens": tokens,
                    "split_item": split,
                    "vs_baseline": diff(y, reference),
                    "baseline_ms": timed(
                        lambda: dist.all_reduce(functional.linear(x, weight), op=dist.ReduceOp.SUM),
                        args.warmup, args.iters),
                    "grouped_ms": timed(call, args.warmup, args.iters),
                }
                record["speedup"] = record["baseline_ms"] / record["grouped_ms"]
                report["cases"].append(record)
                print(json.dumps(record, ensure_ascii=False), flush=True)
            except Exception as exc:  # noqa: BLE001
                report["cases"].append({
                    "tokens": tokens, "split_item": split,
                    "error": f"{type(exc).__name__}: {exc}",
                })
                print(json.dumps(report["cases"][-1], ensure_ascii=False), flush=True)
            del y
        del x, reference
        torch.npu.empty_cache()

    if rank == 0 and args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n")
    dist.barrier()
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
