#!/usr/bin/env python3
"""Restore a Clef service from a phase-0 snapshot and record the new PID.

Uses the exact argv, cwd and whitelisted environment captured before a run.
The new process is detached in its own session, stdout/stderr go to the
requested log, and the PID is written next to it.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

PREFIXES = ("CLEF_", "ASCEND_", "ATB_", "HCCL_", "FLA_")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--log-name", default="service.log")
    args = parser.parse_args()

    snapshot = json.loads(args.snapshot.read_text())
    process = snapshot["process"]
    argv = process["argv"]
    cwd = process["cwd"]
    recorded = process.get("environment") or {}
    environment = {
        key: value for key, value in os.environ.items()
        if not (key in recorded or key.startswith(PREFIXES))
    }
    environment.update(recorded)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.output_dir / args.log_name
    handle = log_path.open("ab")
    started = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    child = subprocess.Popen(
        argv, cwd=cwd, env=environment, stdout=handle, stderr=handle,
        start_new_session=True, close_fds=True)
    pid_path = args.output_dir / "restored.pid"
    pid_path.write_text(str(child.pid) + "\n")
    record = {
        "pid": child.pid, "started_at": started, "argv": argv, "cwd": cwd,
        "environment": recorded, "log": str(log_path),
        "source_snapshot": str(args.snapshot.resolve()),
    }
    (args.output_dir / "restore-launch.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=1) + "\n")
    print(json.dumps({"pid": child.pid, "pid_file": str(pid_path), "log": str(log_path)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
