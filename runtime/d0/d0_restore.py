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

# Sources taken read-only from the recorded successful restore (seq2809/2821).
# They are the actual paths used then; this module never runs them here.
# The config is pinned rather than inherited: an inherited CLEF_LAUNCH_CONFIG
# is not a verified default.
SERVICE_LABEL = "final-default-s1"
SERVICE_CONFIG = "s1"
SERVICE_LAUNCH = ["bash", "/workspace/yangchen/logs/perf-wave2-20261118/launch.sh",
                  SERVICE_LABEL]
SERVICE_WAIT = ["bash", "/workspace/yangchen/logs/perf-wave2-20261118/wait.sh"]
HEALTH_URL = "http://127.0.0.1:18143/health"
TEXT_SOURCE = ("/workspace/yangchen/logs/tuning/20261114-013000-fix4/"
               "opendecisions/all2352.json")
TEXT_SAMPLE = "od-1534"
IMAGE_SOURCE = ("/workspace/yangchen/logs/perf-wave2-20261118/erqa/samples/"
                "erqa-0000/request.json")


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
    if not target.get("starttime") or not target.get("boot_id"):
        return False, "refused: starttime and boot_id are required"
    now = _identity(pid)
    if now is None:
        return False, "unavailable: process not present"
    if now["starttime"] != target["starttime"]:
        return False, "refused: starttime mismatch (pid likely reused)"
    if not now["boot_id"] or now["boot_id"] != target["boot_id"]:
        return False, "refused: boot_id mismatch"
    want_pgid = target.get("pgid")
    if want_pgid is not None and os.getpgid(pid) != want_pgid:
        return False, "refused: pgid does not match the recorded group"
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


def restore(launch_cmd: list, wait_cmd: list | None, log: dict, runner=subprocess.run,
            timeout: int = 300) -> bool:
    started = runner(launch_cmd, capture_output=True, text=True, timeout=timeout,
                     env={**os.environ, "CLEF_LAUNCH_CONFIG": SERVICE_CONFIG})
    log["restore"] = {"at": _now(), "launch_rc": started.returncode,
                      "stdout": started.stdout[-2000:], "stderr": started.stderr[-2000:]}
    if wait_cmd:
        w_started = _now()
        try:
            waited = runner(wait_cmd, capture_output=True, text=True, timeout=timeout)
            log["restore"].update({"wait_started": w_started, "wait_finished": _now(),
                                   "wait_rc": waited.returncode,
                                   "wait_stdout": waited.stdout, "wait_stderr": waited.stderr})
            wait_ok = waited.returncode == 0
        except Exception as exc:                             # noqa: BLE001
            log["restore"].update({"wait_started": w_started, "wait_finished": _now(),
                                   "wait_exception": f"{type(exc).__name__}: {exc}"})
            wait_ok = False
    else:
        wait_ok = True
    return started.returncode == 0 and wait_ok


def _parse_health(text: str) -> bool:
    try:
        return json.loads(text).get("status") == "ready"
    except Exception:                                        # noqa: BLE001
        return False


def _count_workers(text: str, service_pids: set[int] | None = None) -> int:
    """Count worker rows whose PID field is exactly one of the service pids.

    Substring matching would confuse 111 with 1111, and an empty service set
    would silently degrade to a whole-machine count, so attribution is exact
    and a missing set simply matches nothing.
    """
    want = service_pids or set()
    n = 0
    for line in text.splitlines():
        if "VLLMWorker_TP" not in line:
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        for c in cells:
            if c.isdigit() and int(c) in want:
                n += 1
                break
    return n


def _parse_answer(text: str, expect: str) -> bool:
    try:
        return json.loads(text)["answers"]["answer"].get("type") == expect
    except Exception:                                        # noqa: BLE001
        return False


def verify(cmds: dict, log: dict, runner=subprocess.run, timeout: int = 120,
           service_pids: set[int] | None = None) -> bool:
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
                entry["workers"] = _count_workers(r.stdout, service_pids)
                entry["service_pids"] = sorted(service_pids or [])
                entry["ok"] = (r.returncode == 0 and bool(service_pids)
                               and entry["workers"] == 2)
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
    cleanup([{"pid": p} for p in a.pid], log)
    restored = restore(SERVICE_LAUNCH, SERVICE_WAIT, log)
    verified = verify({"health": [], "workers": [], "text": [], "image": []}, log)
    log["restore_ok"] = restored
    log["verify_ok"] = verified
    log["finished"] = _now()
    a.out.write_text(json.dumps(log, ensure_ascii=False, indent=1) + "\n")
    return 0 if (restored and verified) else 1


if __name__ == "__main__":
    raise SystemExit(main())
