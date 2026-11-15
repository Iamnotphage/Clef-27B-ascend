#!/usr/bin/env python3
"""Probe the fused Ascend C GDN preprocessing op against the accepted path.

The operator takes the head-first convolution output ``x`` ``[40, T, 128]``
and writes contiguous BSND ``q``/``k`` ``[1, T, 8, 128]`` (L2-normalized) and
``v`` ``[1, T, 24, 128]``.  This probe:

1. checks every Q/K/V element against the accepted one-copy BSND path,
2. measures the fused kernel against that path with NPU events,
3. covers the real token lengths (307 probe, 401/512/1595 text, 10294 image).

Set ``CLEF_QKV_NORM_VENDOR`` to the extracted ``vendors/customize`` directory.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import sys
import time

VENDOR = os.environ.get(
    "CLEF_QKV_NORM_VENDOR",
    "/workspace/yangchen/work/ascendc/ClefGdnQkvNorm/build_out/install/packages/vendors/customize",
)
FLA_VENDOR = (
    "/workspace/yangchen/venvs/clef-py312/lib/python3.12/site-packages/"
    "fla_npu/opp/vendors/fla_npu_transformer"
)
# CANN discovers custom kernels during torch_npu import, so the vendor path
# must be complete before the first import.
_opp = [
    VENDOR,
    f"{VENDOR}/op_api/lib",
    os.path.dirname(os.path.dirname(VENDOR)),
    FLA_VENDOR,
    f"{FLA_VENDOR}/op_api/lib",
]
os.environ["ASCEND_CUSTOM_OPP_PATH"] = os.pathsep.join(
    _opp + ([os.environ["ASCEND_CUSTOM_OPP_PATH"]]
            if os.environ.get("ASCEND_CUSTOM_OPP_PATH") else [])
)
os.environ.setdefault("LD_LIBRARY_PATH", f"{VENDOR}/op_api/lib")

import torch  # noqa: E402
import torch_npu  # noqa: E402
import fla_npu  # noqa: E402
from fla_npu.ops.ascendc import _runtime as aclnn_runtime  # noqa: E402
from vllm.third_party.flash_linear_attention.ops.l2norm import l2norm_fwd  # noqa: E402

HK, HV, D = 8, 24, 128
HEADS = 40


def install_library():
    lib_path = f"{VENDOR}/op_api/lib/libcust_opapi.so"
    library = ctypes.CDLL(
        lib_path,
        mode=getattr(os, "RTLD_LOCAL", 0)
        | getattr(os, "RTLD_NOW", 0)
        | getattr(os, "RTLD_NODELETE", 0),
    )
    runtime = aclnn_runtime.runtime()
    runtime._libraries.insert(0, library)
    return runtime, lib_path


def run_fused(runtime, x, q, k, v, eps=1e-6):
    argtypes = [
        ctypes.c_void_p,
        ctypes.c_double,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_void_p),
    ]
    aclnn_runtime.call_aclnn(
        "aclnnClefGdnQkvNorm",
        lambda ctx: [
            ctx.tensor(x, "x"),
            ctypes.c_double(eps),
            ctx.tensor(q, "qOut"),
            ctx.tensor(k, "kOut"),
            ctx.tensor(v, "vOut"),
        ],
        (q, k, v),
        get_workspace_argtypes=argtypes,
    )


def baseline(x):
    q = l2norm_fwd(x[:HK].transpose(0, 1).unsqueeze(0).contiguous())
    k = l2norm_fwd(x[HK:2 * HK].transpose(0, 1).unsqueeze(0).contiguous())
    v = x[2 * HK:].transpose(0, 1).unsqueeze(0).contiguous()
    return q, k, v


def diff(a, b):
    return {
        "max_abs": float((a.float() - b.float()).abs().max()),
        "bit_equal": bool(torch.equal(a, b)),
    }


def bench(fn, warmup=5, iters=50):
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    started = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.npu.synchronize()
    return (time.perf_counter() - started) * 1000.0 / iters


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--output", default=None)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--lengths", default="307,401,512,1595,10294")
    args = parser.parse_args()
    torch.npu.set_device(args.device)
    runtime, lib_path = install_library()
    generator = torch.Generator(device="cpu").manual_seed(20261115)
    report = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "device": args.device,
        "vendor": VENDOR,
        "library": lib_path,
        "cases": [],
    }
    for tokens in [int(item) for item in args.lengths.split(",") if item.strip()]:
        x = torch.randn(HEADS, tokens, D, generator=generator).to(
            args.device, torch.bfloat16
        )
        q_ref, k_ref, v_ref = baseline(x)
        q = torch.empty_like(q_ref)
        k = torch.empty_like(k_ref)
        v = torch.empty_like(v_ref)
        run_fused(runtime, x, q, k, v)
        torch.npu.synchronize()
        record = {
            "tokens": tokens,
            "q": diff(q, q_ref),
            "k": diff(k, k_ref),
            "v": diff(v, v_ref),
            "fused_ms": bench(lambda: run_fused(runtime, x, q, k, v),
                              iters=args.iters),
            "baseline_ms": bench(lambda: baseline(x), iters=args.iters),
        }
        record["speedup"] = record["baseline_ms"] / record["fused_ms"]
        report["cases"].append(record)
        print(json.dumps(record, ensure_ascii=False), flush=True)
    report["fails"] = sum(
        1 for case in report["cases"]
        if not (case["q"]["bit_equal"] and case["k"]["bit_equal"]
                and case["v"]["bit_equal"])
    )
    report["max_abs"] = max(
        max(case[part]["max_abs"] for part in ("q", "k", "v"))
        for case in report["cases"]
    )
    text = json.dumps(report, ensure_ascii=False, indent=1)
    print(json.dumps({"fails": report["fails"], "max_abs": report["max_abs"]}))
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
