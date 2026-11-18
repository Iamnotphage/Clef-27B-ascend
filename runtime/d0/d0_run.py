#!/usr/bin/env python3
"""D0 run harness: bounded window, precise process tree, fresh output dir.

NOT executed against the device from this repository state. The sequence it
performs is fixed and must not be reordered:

    clear trace on the stream -> launch -> synchronize -> read back

The first candidate is exactly one cell: mm_dim=1 with use_hccl=2 (AR off).
Budget 2 stays a conditional item and is not encoded here.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

TRACE_SLOTS = 72
TRACE_FIELDS = 8
TRACE_BYTES = TRACE_SLOTS * TRACE_FIELDS * 4


def refuse_existing(path: Path) -> None:
    if path.exists():
        raise SystemExit(f"refusing to reuse an existing run directory: {path}")


def snapshot_tree(root_pid: int) -> dict:
    """Capture the process tree once, so a later claim of exclusivity has evidence."""
    out = {}
    if not Path("/proc").is_dir():
        # macOS host here; the tree snapshot only has meaning on the Linux target.
        return {"root_pid": root_pid, "unavailable": "/proc not present on this host",
                "processes": {}}
    for pid_dir in Path("/proc").iterdir():
        if not pid_dir.name.isdigit():
            continue
        try:
            cmd = (pid_dir / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
        except OSError:
            continue
        if not cmd:
            continue
        out[pid_dir.name] = cmd.strip()
    return {"root_pid": root_pid, "captured_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "processes": out}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--timeout", type=int, default=180, help="bounded window in seconds")
    ap.add_argument("--rows", type=int, default=967)
    ap.add_argument("--seed", type=int, default=20261118)
    ap.add_argument("--mm-dim", type=int, default=1, help="requested budget (first candidate: 1)")
    ap.add_argument("--use-hccl", type=int, default=2, help="first candidate: 2 (AR off)")
    ap.add_argument("--dry-run", action="store_true", help="print the plan only; touches no device")
    a = ap.parse_args()

    refuse_existing(a.out_dir)
    a.out_dir.mkdir(parents=True)

    plan = {
        "cell": {"mm_dim": a.mm_dim, "use_hccl": a.use_hccl, "rows": a.rows, "seed": a.seed},
        "trace": {"slots": TRACE_SLOTS, "fields": TRACE_FIELDS, "bytes": TRACE_BYTES,
                  "dtype": "uint32", "format": "ND", "allocations": 1},
        "sequence": ["clear trace on stream", "launch", "synchronize stream", "read trace back"],
        "timeout_s": a.timeout,
        "scope": ("capacity-internal observations only; absent slot is not a missing "
                  "participant; launch set, SDK receipt, partial writes and consumer "
                  "visibility are outside this trace"),
    }
    (a.out_dir / "plan.json").write_text(json.dumps(plan, ensure_ascii=False, indent=1) + "\n")
    (a.out_dir / "process-tree.json").write_text(
        json.dumps(snapshot_tree(os.getpid()), ensure_ascii=False) + "\n")

    if a.dry_run:
        print(json.dumps({"dry_run": True, "plan": plan}, ensure_ascii=False))
        return 0

    # Real execution path. Kept explicit so the failure modes are visible:
    # a timeout kills the child and still writes a trace-unavailable record.
    child = subprocess.Popen([sys.executable, "-c", "raise SystemExit('d0 executor not wired')"])
    try:
        rc = child.wait(timeout=a.timeout)
    except subprocess.TimeoutExpired:
        child.send_signal(signal.SIGKILL)
        child.wait()
        (a.out_dir / "trace-unavailable.json").write_text(json.dumps(
            {"reason": "timeout", "timeout_s": a.timeout,
             "note": "trace could not be copied back; slot count is not recoverable"},
            ensure_ascii=False) + "\n")
        return 2
    (a.out_dir / "exit.json").write_text(json.dumps({"rc": rc}, ensure_ascii=False) + "\n")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
