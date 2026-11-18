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
SERVICE_LOG_ROOT = "/workspace/yangchen/logs/perf-wave2-20261118"
SERVICE_CONFIG = "s1"
SERVICE_LAUNCH = ["bash", "/workspace/yangchen/logs/perf-wave2-20261118/launch.sh",
                  SERVICE_LABEL]
SERVICE_WAIT = ["bash", "/workspace/yangchen/logs/perf-wave2-20261118/wait.sh"]
HEALTH_URL = "http://127.0.0.1:18143/health"
TEXT_SOURCE = ("/workspace/yangchen/logs/tuning/20261114-013000-fix4/"
               "opendecisions/all2352.json")
TEXT_SAMPLE = "od-1534"
API_BASE = "http://127.0.0.1:18143"
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
    l_started = _now()
    try:
        started = runner(launch_cmd, capture_output=True, text=True, timeout=timeout,
                         env={**os.environ, "CLEF_LAUNCH_CONFIG": SERVICE_CONFIG})
        log["restore"] = {"launch_started": l_started, "launch_finished": _now(),
                          "launch_rc": started.returncode,
                          "stdout": started.stdout, "stderr": started.stderr}
        launch_ok = started.returncode == 0
    except Exception as exc:                                 # noqa: BLE001
        log["restore"] = {"launch_started": l_started, "launch_finished": _now(),
                          "launch_exception": f"{type(exc).__name__}: {exc}"}
        launch_ok = False
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
    return launch_ok and wait_ok


def _parse_health(text: str) -> bool:
    try:
        return json.loads(text).get("status") == "ready"
    except Exception:                                        # noqa: BLE001
        return False


def _count_workers(text: str, service_pids: set[int] | None = None) -> int:
    """Count worker rows whose PID cell is exactly one of the service pids."""
    return len(_worker_pids(text, service_pids))


def _worker_pids(text: str, service_pids: set[int] | None = None) -> set[int]:
    """Distinct PIDs from the PID column of worker rows, filtered to the tree."""
    want = service_pids or set()
    found: set[int] = set()
    for line in text.splitlines():
        if "VLLMWorker_TP" not in line:
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        # npu-smi rows read | chip | pid | name | hbm | pid |, so the PID is the
        # second cell; memory figures must never be read as PIDs.
        if len(cells) < 2:
            continue
        cell = cells[1]
        if cell.isdigit() and int(cell) in want:
            found.add(int(cell))
    return found


def _unused_count_workers(text: str, service_pids: set[int] | None = None) -> int:
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


def discover_service_pids(label: str, before: set[int] | None = None) -> dict:
    """Read the api pid the launcher wrote and collect its live descendants.

    The launcher records `api.pid` inside its label directory; workers are
    children of that process. Pids already present before the restore are
    excluded so an unrelated tree cannot be attributed to the new service.
    """
    pid_file = Path(SERVICE_LOG_ROOT) / label / "api.pid"
    info = {"pid_file": str(pid_file), "exists": pid_file.is_file()}
    if not info["exists"]:
        info["error"] = "api.pid missing"
        return info
    try:
        root = int(pid_file.read_text().strip())
    except (OSError, ValueError) as exc:
        info["error"] = f"api.pid unreadable: {exc}"
        return info
    info["api_pid"] = root
    ident = _identity(root)
    info["api_identity"] = ident
    if ident is None:
        info["error"] = "api pid not alive"
        return info
    boot_before = ident.get("boot_id")
    if before is not None and root in before:
        # The pid existed before this restore: it cannot be the new service.
        info["error"] = "api pid predates the restore; not attributing it"
        return info
    descendants: set[int] = set()
    tree_rows: list[dict] = []
    if Path("/proc").is_dir():
        children: dict[int, list[int]] = {}
        for d in Path("/proc").iterdir():
            if not d.name.isdigit():
                continue
            st = _identity(int(d.name))
            if st is None:
                continue
            try:
                ppid = int((d / "stat").read_text().split(") ", 1)[1].split()[1])
            except (OSError, IndexError):
                continue
            children.setdefault(ppid, []).append(int(d.name))
            st["ppid"] = ppid
            tree_rows.append(st)
        stack = [root]
        while stack:
            cur = stack.pop()
            for c in children.get(cur, []):
                if c not in descendants:
                    descendants.add(c)
                    stack.append(c)
    if before:
        descendants -= set(before)
    # The root identity must be unchanged after the walk, otherwise the tree
    # may have been rebuilt underneath us.
    after = _identity(root)
    boot_after = (after or {}).get("boot_id")
    # A missing boot id can never establish a stable identity.
    info["root_identity_stable"] = bool(
        after is not None and boot_before and boot_after
        and after["starttime"] == ident["starttime"]
        and boot_after == boot_before)
    if not info["root_identity_stable"]:
        info["error"] = "root identity changed or unprovable during collection"
    # A failed identity check yields no attributable pids at all, so a caller
    # cannot accidentally verify against a tree we cannot vouch for.
    info["pids"] = (sorted({root} | descendants)
                    if info["root_identity_stable"] else [])
    info["descendant_count"] = len(descendants)
    info["tree"] = [r for r in tree_rows if r["pid"] == root or r["pid"] in descendants]
    return info


def build_commands(render) -> dict:
    """Real verification commands: health URL, npu-smi table, and the two POSTs
    with the recorded request bodies. `render` posts JSON and returns a string."""
    text_body = render("text")
    image_body = render("image")
    return {
        "health": ["curl", "-s", "-m", "5", f"{API_BASE}/health"],
        "workers": ["npu-smi", "info"],
        "text": ["curl", "-s", "-m", "120", "-X", "POST",
                 f"{API_BASE}/v1/systemone", "-H", "Content-Type: application/json",
                 "-d", text_body],
        "image": ["curl", "-s", "-m", "120", "-X", "POST",
                  f"{API_BASE}/v1/decide", "-H", "Content-Type: application/json",
                  "-d", image_body],
    }


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
                pids = _worker_pids(r.stdout, service_pids)
                entry["workers"] = len(pids)
                entry["worker_pids"] = sorted(pids)
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
    targets = [{"pid": p, "starttime": None, "boot_id": None} for p in a.pid]
    cleaned = cleanup(targets, log)
    if cleaned["refused"] or cleaned["unavailable"]:
        # Incomplete identity means we cannot prove what we are cleaning, so the
        # restore refuses rather than launching over an unknown process.
        log["restore_blocked"] = "cleanup targets lack a provable identity"
        log["restore_ok"] = log["verify_ok"] = False
        a.out.write_text(json.dumps(log, ensure_ascii=False, indent=1) + "\n")
        return 1
    restored = restore(SERVICE_LAUNCH, SERVICE_WAIT, log)
    verified = False
    if restored:
        disc = discover_service_pids(SERVICE_LABEL)
        log["service_discovery"] = disc
        if not disc.get("error") and disc.get("pids"):
            verified = verify(build_commands(lambda k: Path(
                IMAGE_SOURCE if k == "image" else TEXT_SOURCE).read_text() if k == "image"
                else ""), log, service_pids=set(disc["pids"]))
    log["restore_ok"], log["verify_ok"] = restored, verified
    log["restore_ok"] = restored
    log["verify_ok"] = verified
    log["finished"] = _now()
    a.out.write_text(json.dumps(log, ensure_ascii=False, indent=1) + "\n")
    return 0 if (restored and verified) else 1


if __name__ == "__main__":
    raise SystemExit(main())
