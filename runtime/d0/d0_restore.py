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


def _identity(pid: int) -> dict | None:
    """Start time and boot id, so a recycled pid is never mistaken for ours."""
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
        starttime = stat.split(") ", 1)[1].split()[19]
    except (OSError, IndexError):
        return None
    boot = ""
    try:
        boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    except OSError:
        pass
    return {"pid": pid, "starttime": starttime, "boot_id": boot}


def _verify_identity(target: dict) -> tuple[bool, str]:
    """A target is only killable when the recorded identity still matches."""
    pid = target.get("pid")
    if not isinstance(pid, int) or pid <= 1:
        return False, "refused: pid missing, non-integer or <= 1"
    if pid == os.getpid():
        return False, "refused: target is the current process"
    now = _identity(pid)
    if now is None:
        return False, "unavailable: process not present"
    if target.get("starttime") and now["starttime"] != target["starttime"]:
        return False, "refused: starttime mismatch (pid likely reused)"
    if target.get("boot_id") and now["boot_id"] != target["boot_id"]:
        return False, "refused: boot_id mismatch"
    return True, "ok"


def cleanup(targets: list[dict], log: dict, killer=None) -> dict:
    """Kill only targets whose recorded identity still matches.

    A target without a usable identity is never signalled; the outcome is
    recorded as refused or unavailable rather than as cleaned.
    """
    killer = killer or (lambda kind, value: os.killpg(value, signal.SIGKILL)
                        if kind == "pgid" else os.kill(value, signal.SIGKILL))
    result = {"killed": [], "already_gone": [], "refused": [], "unavailable": [],
              "errors": []}
    for t in targets:
        ok, why = _verify_identity(t)
        if not ok:
            bucket = "unavailable" if why.startswith("unavailable") else "refused"
            result[bucket].append({**t, "reason": why})
            continue
        kind = "pgid" if t.get("pgid") else "pid"
        value = t.get("pgid") or t["pid"]
        try:
            killer(kind, value)
            result["killed"].append({**t, "signalled": kind})
        except ProcessLookupError:
            result["already_gone"].append({**t})
        except OSError as exc:
            result["errors"].append({**t, "error": str(exc)})
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


def _parse_health(text: str) -> bool:
    try:
        return json.loads(text).get("status") == "ready"
    except Exception:                                        # noqa: BLE001
        return False


def _count_workers(text: str) -> int:
    return sum(1 for line in text.splitlines() if "VLLMWorker_TP" in line)


def _parse_answer(text: str, expect: str) -> bool:
    try:
        return json.loads(text)["answers"]["answer"].get("type") == expect
    except Exception:                                        # noqa: BLE001
        return False


def verify(cmds: dict, log: dict, runner=subprocess.run, timeout: int = 120) -> bool:
    """Each stage keeps its own rc, payload and timing; non-empty is not enough."""
    ok = True
    for stage in ("health", "workers", "text", "image"):
        cmd = cmds.get(stage)
        started = _now()
        try:
            r = runner(cmd, capture_output=True, text=True, timeout=timeout)
            entry = {"started": started, "finished": _now(), "rc": r.returncode,
                     "stdout": r.stdout, "stderr": r.stderr}
            if stage == "health":
                entry["ok"] = r.returncode == 0 and _parse_health(r.stdout)
            elif stage == "workers":
                entry["workers"] = _count_workers(r.stdout)
                entry["ok"] = r.returncode == 0 and entry["workers"] == 2
            elif stage == "text":
                entry["ok"] = r.returncode == 0 and _parse_answer(r.stdout, "score")
            else:
                entry["ok"] = r.returncode == 0 and _parse_answer(r.stdout, "choice")
        except Exception as exc:                             # noqa: BLE001
            entry = {"started": started, "finished": _now(), "ok": False,
                     "exception": f"{type(exc).__name__}: {exc}"}
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
