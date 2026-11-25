#!/usr/bin/env python3
"""Measure the effective HBM bandwidth of one logical A3 NPU.

The MLP profile reports ``aic_mte2_ratio≈0.994`` at about 0.53 TB/s, which
leaves open whether that is the HBM ceiling or an MTE2 efficiency limit.  This
probe measures two independent data points on the same device:

* ``copy``: a device-to-device copy (reads N bytes, writes N bytes);
* ``sum``: an AI Core reduction (reads N bytes, writes a scalar);
* ``add_out``: an elementwise add into a preallocated output (reads N, writes N).

Work sets from 64 MiB to 1 GiB cross the 192 MiB L2, and the median of several
launches is reported.  The script only allocates 2 GiB at the largest size, so
it can run beside an idle service that holds most of HBM.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time

import torch
import torch_npu


def timed(fn, iters: int, warmup: int) -> float:
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    started = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.npu.synchronize()
    return (time.perf_counter() - started) / iters


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--devices", default="npu:0,npu:1")
    parser.add_argument("--sizes-mib", default="64,256,512,1024")
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    report = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "torch": torch.__version__,
        "torch_npu": torch_npu.__version__,
        "cases": [],
    }
    for device in [item.strip() for item in args.devices.split(",") if item.strip()]:
        torch.npu.set_device(device)
        for size_mib in [int(item) for item in args.sizes_mib.split(",") if item.strip()]:
            elements = size_mib * 1024 * 1024 // 2  # bf16
            nbytes = elements * 2
            x = torch.empty(elements, dtype=torch.bfloat16, device=device)
            y = torch.empty_like(x)
            x.uniform_(-1.0, 1.0)
            torch.npu.synchronize()
            cases = {
                "copy": (lambda: y.copy_(x), 2 * nbytes),
                "sum": (lambda: torch.sum(x), nbytes),
                "add_out": (lambda: torch.add(x, 1.0, out=y), 2 * nbytes),
            }
            for name, (fn, traffic) in cases.items():
                elapsed = timed(fn, args.iters, args.warmup)
                record = {
                    "device": device,
                    "size_mib": size_mib,
                    "case": name,
                    "seconds": elapsed,
                    "traffic_bytes": traffic,
                    "gb_per_s": traffic / elapsed / 1e9,
                }
                report["cases"].append(record)
                print(json.dumps(record, ensure_ascii=False), flush=True)
            del x, y
            torch.npu.empty_cache()
    text = json.dumps(report, ensure_ascii=False, indent=1)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
