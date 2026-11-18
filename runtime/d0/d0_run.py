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


def _read_stat(pid: str):
    try:
        raw = Path(f"/proc/{pid}/stat").read_text()
    except OSError:
        return None
    head, rest = raw.split(") ", 1)
    fields = rest.split()
    return {"comm": head.split("(")[-1], "state": fields[0], "ppid": int(fields[1]),
            "pgrp": int(fields[2]), "session": int(fields[3])}


def snapshot_tree(root_pid: int) -> dict:
    """Capture root identity plus its full descendant set.

    Each entry keeps PID, PPID, PGID and session so exclusivity claims rest on
    the tree rather than on a flat command list.
    """
    if not Path("/proc").is_dir():
        return {"root_pid": root_pid, "unavailable": "/proc not present on this host",
                "processes": {}, "descendants": []}
    procs = {}
    for pid_dir in Path("/proc").iterdir():
        if not pid_dir.name.isdigit():
            continue
        st = _read_stat(pid_dir.name)
        if st is None:
            continue
        try:
            cmd = (pid_dir / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace").strip()
        except OSError:
            cmd = ""
        st["cmdline"] = cmd
        procs[pid_dir.name] = st
    kids = {}
    for pid, st in procs.items():
        kids.setdefault(str(st["ppid"]), []).append(pid)
    desc, stack = [], [str(root_pid)]
    while stack:
        cur = stack.pop()
        for child in kids.get(cur, []):
            desc.append({"pid": int(child), **procs[child]})
            stack.append(child)
    return {"root_pid": root_pid,
            "root": procs.get(str(root_pid)),
            "captured_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "processes": procs, "descendants": desc}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--timeout", type=int, default=180, help="bounded window in seconds")
    ap.add_argument("--rows", type=int, default=967)
    ap.add_argument("--seed", type=int, default=20261118)
    # First candidate is fixed; the CLI cannot widen it.
    ap.add_argument("--mm-dim", type=int, default=1, choices=(1,),
                    help="first candidate is fixed at 1")
    ap.add_argument("--use-hccl", type=int, default=2, choices=(2,),
                    help="first candidate is fixed at 2 (AR off)")
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
    # Real executor lives in its own module; this process only supervises it.
    child = subprocess.Popen(
        [sys.executable, str(Path(__file__).with_name("d0_launch.py")),
         "--out-dir", str(a.out_dir), "--rows", str(a.rows), "--seed", str(a.seed),
         "--mm-dim", str(a.mm_dim), "--use-hccl", str(a.use_hccl)],
        start_new_session=True,                       # own session and process group
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    (a.out_dir / "process-tree.json").write_text(
        json.dumps(snapshot_tree(os.getpid()), ensure_ascii=False) + "\n")
    try:
        out, err = child.communicate(timeout=a.timeout)
        rc = child.returncode
    except subprocess.TimeoutExpired:
        # Kill the whole group, not just the direct child, then drain pipes.
        try:
            os.killpg(os.getpgid(child.pid), signal.SIGKILL)
        except OSError:
            child.kill()
        out, err = child.communicate()
        (a.out_dir / "trace-unavailable.json").write_text(json.dumps(
            {"reason": "timeout", "timeout_s": a.timeout,
             "note": "trace could not be copied back; slot count is not recoverable"},
            ensure_ascii=False) + "\n")
        rc = 2
    (a.out_dir / "raw-rc.json").write_text(json.dumps({"raw_rc": rc}, ensure_ascii=False) + "\n")
    (a.out_dir / "stdout.log").write_text(out or "")
    (a.out_dir / "stderr.log").write_text(err or "")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
