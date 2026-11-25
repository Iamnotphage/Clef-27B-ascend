#!/usr/bin/env python3
"""Tight-budget, high-concurrency eviction stress for the Tier 2 host store.

Mirrors the Stage 4 failure mode without the service: a 4 GiB budget, many
threads, entries larger than the working set, leases held across lookups, and
a fraction of POSIX-SHM prefix-memory items.  The run fails loudly if any
invariant breaks; the JSON report keeps the counters for the acceptance
record.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import threading
import time

from runtime.tiered_prefix_cache.host_store import HostStore


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budget-gib", type=float, default=4.0)
    parser.add_argument("--entry-mib", type=float, default=1.0)
    parser.add_argument("--entries", type=int, default=20_000)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--shm-every", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20261124)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    budget = int(args.budget_gib * (1 << 30))
    entry_bytes = int(args.entry_mib * (1 << 20))
    store = HostStore(budget_bytes=budget, admission="second_hit",
                      use_shm=True, eviction_watermark=0.95)
    errors: list[str] = []
    lock = threading.Lock()
    published = 0

    def worker(worker_id: int) -> None:
        nonlocal published
        rng = random.Random(args.seed + worker_id)
        key_count = max(1, args.entries // 4)
        for index in range(args.entries // args.threads):
            key = bytes(rng.randrange(key_count).to_bytes(4, "little")) * 8
            key = key[:32].ljust(32, b"\0")
            try:
                store.observe(key)
                lease = store.acquire(key)
                if lease is not None:
                    time.sleep(0)
                    lease.release()
                else:
                    payload = bytes([worker_id % 256]) * entry_bytes
                    prefix_memory = None
                    with lock:
                        should_shm = (
                            args.shm_every > 0
                            and published % args.shm_every == 0)
                        published += 1
                    if should_shm:
                        prefix_memory = bytes([worker_id % 256]) * 2048
                    store.publish(
                        key, {0: {"kv": payload},
                              1: {"kv": payload}},
                        prefix_memory=prefix_memory)
                if rng.random() < 0.05:
                    store.drop(key)
            except BaseException as error:  # noqa: BLE001
                with lock:
                    errors.append(f"{type(error).__name__}: {error}")
                return

    started = time.perf_counter()
    threads = [threading.Thread(target=worker, args=(index,))
               for index in range(args.threads)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=300)
        if thread.is_alive():
            errors.append("worker timeout")
    wall = time.perf_counter() - started
    snapshot = store.snapshot()
    owned = store.shm_names()
    residual = {
        path.name for path in
        Path("/dev/shm").glob(f"clef_pfx_{store.session}_*")
    }
    unexpected = residual - owned
    if snapshot["bytes"] > budget:
        errors.append("budget exceeded")
    if snapshot["leases"] != 0:
        errors.append("lease leak")
    if unexpected:
        errors.append(f"{len(unexpected)} unowned SHM objects remain")
    store.close()
    residual_after = {
        path.name for path in
        Path("/dev/shm").glob(f"clef_pfx_{store.session}_*")
    }
    if residual_after:
        errors.append(f"{len(residual_after)} SHM objects remain after close")

    report = {
        "schema": "tiered-prefix-cache-stress/1",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "budget_bytes": budget,
        "entry_bytes": entry_bytes,
        "entries_per_thread": args.entries // args.threads,
        "threads": args.threads,
        "wall_seconds": wall,
        "snapshot": snapshot,
        "shm_live_before_close": len(residual & owned),
        "shm_unowned_before_close": len(unexpected),
        "shm_residual_after_close": len(residual_after),
        "errors": errors,
        "ok": not errors,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps({"ok": report["ok"], "errors": errors,
                      "entries": snapshot["entries"],
                      "bytes": snapshot["bytes"],
                      "counters": snapshot["counters"],
                      "wall_seconds": round(wall, 2)}, indent=1))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
