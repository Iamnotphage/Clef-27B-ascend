#!/usr/bin/env python3
"""A3 real-NPU verification for the Tier 2 async transfer path.

Three hard gates:

1. Adversarial source-overwrite: after ``AsyncCapture.submit`` the compute
   stream immediately overwrites the source tensor.  Twenty rounds per size
   must still publish the pre-overwrite bytes (``torch.equal``).
2. End-to-end ``AsyncCapture -> HostStore(SHM) -> Prefetcher -> wait`` round
   trip must restore bitwise, with measured ``submit`` host enqueue, H2D
   effective bandwidth and zero-blocking ``wait`` time.
3. The 4 GiB host-store stress (run separately) stays clean.

The script uses one logical NPU and ~1-2 GiB HBM, never starts vLLM, and
writes a lightweight JSON report.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import threading
import time

import torch
import torch_npu  # noqa: F401 - registers the NPU backend

from runtime.tiered_prefix_cache.async_transfer import (
    AsyncCapture,
    PinnedRing,
    Prefetcher,
)
from runtime.tiered_prefix_cache.host_store import HostStore

SIZES_MIB = (70, 360)
OVERWRITE_ROUNDS = 20


def percentile(values, fraction):
    ordered = sorted(values)
    index = (len(ordered) - 1) * fraction
    lower, upper = __import__("math").floor(index), __import__("math").ceil(index)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)


def make_tensor(mib: int, device: str = "npu:0") -> torch.Tensor:
    elements = (mib * (1 << 20)) // 2  # bfloat16
    return torch.randn(elements, dtype=torch.bfloat16, device=device)


def element_count(mib: int) -> int:
    return (mib * (1 << 20)) // 2


def overwrite_round(capture: AsyncCapture, mib: int,
                    round_index: int) -> dict:
    source = make_tensor(mib)
    reference = source.cpu().clone()
    event = threading.Event()
    published = {}
    begin = time.perf_counter()
    ticket = capture.submit(
        {"k": source},
        lambda host: (published.update(host), event.set()))
    submit_ms = (time.perf_counter() - begin) * 1000.0
    if ticket is None:
        return {"round": round_index, "dropped": True}
    # Adversarial overwrite on the compute stream, immediately after submit.
    source.zero_()
    if not capture.drain(timeout=60):
        raise RuntimeError("capture drain timed out")
    event.wait(5)
    equal = torch.equal(published["k"], reference)
    return {
        "round": round_index,
        "dropped": False,
        "equal": equal,
        "submit_ms": submit_ms,
        "host_dtype": str(published["k"].dtype),
    }


def round_trip(store: HostStore, prefetcher: Prefetcher, capture: AsyncCapture,
               mib: int, device: str = "npu:0") -> dict:
    source = make_tensor(mib, device)
    reference = source.cpu().clone()
    published = {}
    event = threading.Event()
    ticket = capture.submit({"k": source},
                            lambda host: (published.update(host), event.set()))
    if ticket is None:
        raise RuntimeError("capture dropped a round-trip package")
    capture.drain(timeout=60)
    event.wait(5)
    key = bytes([mib % 256]) * 32
    if not store.publish(key, {0: {"k": published["k"]}},
                         prefix_memory=torch.zeros(1024, dtype=torch.float32)):
        raise RuntimeError("host store rejected the round-trip package")
    lease = store.acquire(key)
    if lease is None:
        raise RuntimeError("round-trip lease missing")
    entry = lease.entry
    assert entry is not None
    begin = time.perf_counter()
    restored = prefetcher.load(entry.shards[0], device)
    enqueue_ms = (time.perf_counter() - begin) * 1000.0
    groups = list(restored.tensors)
    begin = time.perf_counter()
    restored.wait(groups)  # stream-to-stream, no host block
    wait_ms = (time.perf_counter() - begin) * 1000.0
    # Completion is measured on the compute stream, not the host.
    begin = time.perf_counter()
    torch.npu.current_stream().synchronize()
    sync_ms = (time.perf_counter() - begin) * 1000.0
    ok = torch.equal(restored.tensors["k"].cpu(), published["k"])
    restored.close()
    lease.release()
    store.drop(key)
    target = entry.shards[0]["k"]
    nbytes = int(target.numel() * target.element_size())
    return {
        "mib": mib,
        "equal": ok,
        "enqueue_ms": enqueue_ms,
        "wait_cpu_ms": wait_ms,
        "compute_sync_ms": sync_ms,
        "h2d_bandwidth_gib_s": (
            (nbytes / float(1 << 30))
            / max((enqueue_ms + sync_ms) / 1000.0, 1e-9)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--rounds", type=int, default=OVERWRITE_ROUNDS)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    torch.npu.set_device(args.device)
    ring = PinnedRing(slots=3, slot_bytes=512 << 20)
    capture = AsyncCapture(ring)
    store = HostStore(budget_bytes=2 << 30, admission="always", use_shm=True)
    prefetcher = Prefetcher(staging_slots=2, staging_bytes=384 << 20)
    report = {
        "schema": "tiered-prefix-cache-async-npu/1",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "device": args.device,
        "overwrite": {},
        "round_trip": {},
        "capture_errors": [],
    }
    try:
        for mib in SIZES_MIB:
            # One unmeasured round warms the allocator, snapshot pool and
            # pinned ring; steady-state enqueue is what the online path pays.
            overwrite_round(capture, mib, -1)
            rounds = [overwrite_round(capture, mib, index)
                      for index in range(args.rounds)]
            successes = [item for item in rounds if not item.get("dropped")]
            equal = sum(1 for item in successes if item["equal"])
            times = [item["submit_ms"] for item in successes]
            report["overwrite"][str(mib)] = {
                "rounds": len(rounds),
                "not_dropped": len(successes),
                "equal": equal,
                "all_equal": equal == len(successes) and bool(successes),
                "submit_ms": {
                    "mean": statistics.fmean(times) if times else None,
                    "p50": percentile(times, 0.50) if times else None,
                    "p95": percentile(times, 0.95) if times else None,
                },
            }
        for mib in SIZES_MIB:
            report["round_trip"][str(mib)] = round_trip(
                store, prefetcher, capture, mib, args.device)
        report["host_store"] = store.snapshot()
        report["capture_dropped"] = capture.dropped
        report["ring_dropped"] = ring.dropped
    finally:
        report["capture_errors"] = [str(error) for error in capture.errors]
        report["prefetch_errors"] = [str(error) for error in prefetcher.errors]
        capture.close()
        prefetcher.close()
        store.close()
        torch.npu.empty_cache()

    report["ok"] = (
        all(value["all_equal"] for value in report["overwrite"].values())
        and all(value["equal"] for value in report["round_trip"].values())
        and not report["capture_errors"]
        and not report["prefetch_errors"]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps({
        key: report[key] for key in
        ("overwrite", "round_trip", "capture_errors", "ok")}, indent=1))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
