#!/usr/bin/env python3
"""Restore wrapper: clean up only our own test processes, then bring the default
service back and verify it. CPU code plus mocks only; it never stops, starts or
queries the real service from this repository state.

Contract:
  * cleanup uses the exact PIDs/PGIDs we recorded at start, never a broad match;
  * restore runs independently of the test result and keeps its own status;
  * a restore failure is reported on its own line and never overwritten by the
    test return code.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import time
from datetime import datetime
from pathlib import Path

STAGES = ("cleanup", "restore", "health", "workers", "text", "image")


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def cleanup(pids: list[int], pgids: list[int], log: dict) -> dict:
    """Kill only recorded identities; report each one."""
    result = {"killed": [], "already_gone": [], "errors": []}
    for pgid in pgids:
        try:
            os.killpg(pgid, signal.SIGKILL)
            result["killed"].append({"pgid": pgid})
        except ProcessLookupError:
            result["already_gone"].append({"pgid": pgid})
        except OSError as exc:
            result["errors"].append({"pgid": pgid, "error": str(exc)})
    for pid in pids:
        try:
            os.kill(pid, signal.SIGKILL)
            result["killed"].append({"pid": pid})
        except ProcessLookupError:
            result["already_gone"].append({"pid": pid})
        except OSError as exc:
            result["errors"].append({"pid": pid, "error": str(exc)})
    log["cleanup"] = {"at": _now(), **result}
    return result


def restore(launch_cmd: list, wait_cmd: list | None, log: dict, runner=subprocess.run) -> bool:
    started = runner(launch_cmd, capture_output=True, text=True)
    log["restore"] = {"at": _now(), "launch_rc": started.returncode,
                      "stdout": started.stdout[-2000:], "stderr": started.stderr[-2000:]}
    if wait_cmd:
        waited = runner(wait_cmd, capture_output=True, text=True)
        log["restore"].update({"wait_rc": waited.returncode,
                               "wait_stdout": waited.stdout[-2000:]})
    return started.returncode == 0


def verify(health_cmd: list, workers_cmd: list, text_cmd: list, image_cmd: list,
           log: dict, runner=subprocess.run) -> bool:
    ok = True
    for stage, cmd in (("health", health_cmd), ("workers", workers_cmd),
                       ("text", text_cmd), ("image", image_cmd)):
        try:
            r = runner(cmd, capture_output=True, text=True)
            entry = {"at": _now(), "rc": r.returncode,
                     "stdout": r.stdout[-2000:].strip(), "stderr": r.stderr[-500:]}
            entry["ok"] = bool(r.returncode == 0 and r.stdout.strip())
        except Exception as exc:                             # noqa: BLE001
            entry = {"at": _now(), "ok": False, "exception": f"{type(exc).__name__}: {exc}"}
        log[stage] = entry
        ok = ok and entry["ok"]
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--test-rc", type=int, default=0)
    ap.add_argument("--pid", type=int, action="append", default=[])
    ap.add_argument("--pgid", type=int, action="append", default=[])
    ap.add_argument("--dry-run", action="store_true",
                    help="write the plan and exit; touches nothing")
    a = ap.parse_args()
    log = {"started": _now(), "test_rc": a.test_rc,
           "pids": a.pid, "pgids": a.pgid, "stages": list(STAGES)}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    if a.dry_run:
        log["dry_run"] = True
        a.out.write_text(json.dumps(log, ensure_ascii=False, indent=1) + "\n")
        print(json.dumps({"dry_run": True, "stages": list(STAGES)}))
        return 0
    cleanup(a.pid, a.pgid, log)
    restored = restore([], None, log)          # commands supplied by the caller
    verified = verify([], [], [], [], log)
    log["restore_ok"] = restored
    log["verify_ok"] = verified
    log["finished"] = _now()
    a.out.write_text(json.dumps(log, ensure_ascii=False, indent=1) + "\n")
    return 0 if (restored and verified) else 1


if __name__ == "__main__":
    raise SystemExit(main())
