#!/usr/bin/env python3
"""Stage 0.2 probe A: pinned H2D/D2H bandwidth and copy-stream overlap.

The Tier 2 cache restores tens of MiB of prefix state from host DRAM before a
suffix prefill and writes captured state back after it.  This probe measures,
for the package sizes that matter (one rank of a 350/967/10,294-token prefix):

- pinned-host and pageable H2D/D2H latency and effective bandwidth;
- the host time that enqueueing an asynchronous copy adds to the compute
  stream (the rest must be hidden behind the suffix prefill);
- the copy stream's own completion time, which bounds how much of the restore
  can be pipelined.

The tensors are ``uint8`` so the numbers bound any BF16/FP32 payload of the
same byte size.  Nothing here touches the model or the serving tree.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import statistics
import time

import torch


def percentile(values, fraction):
    ordered = sorted(values)
    index = (len(ordered) - 1) * fraction
    lower, upper = math.floor(index), math.ceil(index)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)


def summarize(values):
    return {
        "count": len(values),
        "min_ms": min(values),
        "p50_ms": percentile(values, 0.50),
        "p95_ms": percentile(values, 0.95),
        "mean_ms": statistics.fmean(values),
    }


def bytes_to_gib(value: int) -> float:
    return value / float(1 << 30)


def timed_copy(stream, copy, iterations: int) -> list[float]:
    """Event-timed copies, both events on the copy stream itself."""
    elapsed = []
    for _ in range(iterations):
        start = torch.npu.Event(enable_timing=True)
        end = torch.npu.Event(enable_timing=True)
        with torch.npu.stream(stream):
            start.record()
            copy()
            end.record()
        end.synchronize()
        elapsed.append(start.elapsed_time(end))
    return elapsed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes-mib", default="40,70,360")
    parser.add_argument("--devices", default="npu:0,npu:1")
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    sizes = [int(value) for value in args.sizes_mib.split(",") if value.strip()]
    devices = [value.strip() for value in args.devices.split(",") if value.strip()]
    report = {
        "schema": "tiered-prefix-cache-pinned-bandwidth/1",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "sizes_bytes": {str(size): size * (1 << 20) for size in sizes},
        "iterations": args.iterations,
        "devices": {},
    }

    for device in devices:
        torch.npu.set_device(device)
        device_report = {}
        for size_mib in sizes:
            nbytes = size_mib * (1 << 20)
            device_tensor = torch.empty(nbytes, dtype=torch.uint8, device=device)
            try:
                pinned = torch.empty(nbytes, dtype=torch.uint8, pin_memory=True)
                pinned_ok = True
            except Exception as error:  # noqa: BLE001 - report the fallback
                pinned = torch.empty(nbytes, dtype=torch.uint8)
                pinned_ok = False
                entry_pin_error = f"{type(error).__name__}: {error}"
            streams = {"pinned": pinned, "pageable": torch.empty(
                nbytes, dtype=torch.uint8)}
            entry = {"pinned_allocated": pinned_ok}
            if not pinned_ok:
                entry["pinned_error"] = entry_pin_error
            copy_stream = torch.npu.Stream()
            for kind, host in streams.items():
                for direction in ("h2d", "d2h"):
                    for _ in range(args.warmup):
                        if direction == "h2d":
                            device_tensor.copy_(host, non_blocking=True)
                        else:
                            host.copy_(device_tensor, non_blocking=True)
                    torch.npu.synchronize()

                    def copy(host=host, direction=direction):
                        if direction == "h2d":
                            device_tensor.copy_(host, non_blocking=True)
                        else:
                            host.copy_(device_tensor, non_blocking=True)

                    torch.npu.synchronize()
                    elapsed = timed_copy(copy_stream, copy, args.iterations)
                    torch.npu.synchronize()
                    summary = summarize(elapsed)
                    summary["bandwidth_gib_s"] = (
                        bytes_to_gib(nbytes) / (summary["p50_ms"] / 1000.0)
                        if summary["p50_ms"] else None)
                    entry[f"{kind}_{direction}"] = summary

                    # Host-side enqueue cost, with the copy still outstanding.
                    enqueue = []
                    for _ in range(args.iterations):
                        begin = time.perf_counter()
                        with torch.npu.stream(copy_stream):
                            copy()
                        enqueue.append((time.perf_counter() - begin) * 1000.0)
                        torch.npu.synchronize()
                    entry[f"{kind}_{direction}_enqueue"] = summarize(enqueue)

            device_report[str(size_mib)] = entry
            del streams["pinned"]
            del streams["pageable"]
            del device_tensor
            torch.npu.empty_cache()
        report["devices"][device] = device_report

    report["environment"] = {
        "torch": torch.__version__,
        "torch_npu": getattr(__import__("torch_npu"), "__version__", None)
        if os.environ.get("CLEF_SKIP_TORCH_NPU_VERSION") != "1" else None,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp")
    temporary.write_text(json.dumps(report, indent=1) + "\n")
    temporary.replace(args.output)
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
