#!/usr/bin/env python3
"""Precise stop for the default service. CPU only; never runs from here.

Replaces a broad `pgrep -f` loop with an identity-checked stop:

  * targets are the recorded api pid and its live descendants;
  * every target must still match its start time and boot id, otherwise it is
    refused rather than signalled;
  * SIGTERM first, then a bounded wait, then an identity recheck before any
    SIGKILL, so a recycled pid is never touched;
  * unknown pids are refused and recorded, never killed.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import time
from datetime import datetime
from pathlib import Path


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def identity(pid: int) -> dict | None:
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
        fields = stat.split(") ", 1)[1].split()
        ppid, pgrp = int(fields[1]), int(fields[2])
        starttime = fields[19]
    except (OSError, IndexError):
        return None
    try:
        boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    except OSError:
        boot = ""
    try:
        cmdline = (Path(f"/proc/{pid}/cmdline").read_bytes()
                   .replace(b"\0", b" ").decode(errors="replace").strip())
    except OSError:
        cmdline = ""
    return {"pid": pid, "ppid": ppid, "pgid": pgrp,
            "starttime": starttime, "boot_id": boot, "cmdline": cmdline}


def descendants(root: int) -> list[int]:
    if not Path("/proc").is_dir():
        return []
    children: dict[int, list[int]] = {}
    for d in Path("/proc").iterdir():
        if not d.name.isdigit():
            continue
        ident = identity(int(d.name))
        if ident is None:
            continue
        children.setdefault(ident["ppid"], []).append(int(d.name))
    out, stack = [], [root]
    while stack:
        cur = stack.pop()
        for c in children.get(cur, []):
            if c not in out:
                out.append(c)
                stack.append(c)
    return out


def can_signal(target: dict) -> tuple[bool, str]:
    pid = target.get("pid")
    if not isinstance(pid, int) or pid <= 1:
        return False, "refused: pid missing, non-integer or <= 1"
    if pid == os.getpid():
        return False, "refused: target is the current process"
    if not target.get("starttime") or not target.get("boot_id"):
        return False, "refused: starttime and boot_id are required"
    live = identity(pid)
    if live is None:
        return False, "unavailable: already gone"
    if live["starttime"] != target["starttime"] or live["boot_id"] != target["boot_id"]:
        return False, "refused: identity changed (pid likely reused)"
    return True, "ok"


def stop(api_pid: int, log: dict, *, term_wait: float = 20.0,
         kill_wait: float = 10.0, signaller=None) -> dict:
    signaller = signaller or (lambda pid, sig: os.kill(pid, sig))
    api = identity(api_pid)
    if api is None:
        log["stop"] = {"error": "api pid not alive", "api_pid": api_pid}
        return log["stop"]
    targets = [api] + [identity(p) for p in descendants(api_pid)]
    targets = [t for t in targets if t]
    record = {"api": api, "targets": [t["pid"] for t in targets],
              "refused": [], "unavailable": [], "termed": [], "killed": [],
              "started": now()}

    for t in targets:
        ok, why = can_signal(t)
        if not ok:
            (record["unavailable"] if why.startswith("unavailable")
             else record["refused"]).append({"pid": t["pid"], "reason": why})
            continue
        try:
            signaller(t["pid"], signal.SIGTERM)
            record["termed"].append(t["pid"])
        except OSError as exc:
            record["unavailable"].append({"pid": t["pid"], "error": str(exc)})

    deadline = time.time() + term_wait
    while time.time() < deadline:
        if all(identity(p) is None for p in record["termed"]):
            break
        time.sleep(0.5)

    for pid in record["termed"]:
        live = identity(pid)
        if live is None:
            continue
        # Identity is rechecked before escalating; a reused pid is refused.
        if live["starttime"] != next(t["starttime"] for t in targets if t["pid"] == pid):
            record["refused"].append({"pid": pid, "reason": "refused: identity changed before kill"})
            continue
        try:
            signaller(pid, signal.SIGKILL)
            record["killed"].append(pid)
        except OSError as exc:
            record["unavailable"].append({"pid": pid, "error": str(exc)})

    deadline = time.time() + kill_wait
    while time.time() < deadline:
        if all(identity(p) is None for p in targets and [t["pid"] for t in targets]):
            break
        time.sleep(0.5)

    record["finished"] = now()
    record["survivors"] = [t["pid"] for t in targets if identity(t["pid"]) is not None]
    log["stop"] = record
    return record


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--api-pid", type=int, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--dry-run", action="store_true", help="record the plan only")
    a = ap.parse_args()
    log: dict = {}
    if a.dry_run:
        api = identity(a.api_pid)
        log["stop"] = {"dry_run": True, "api": api,
                       "descendants": descendants(a.api_pid)}
    else:
        stop(a.api_pid, log)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(log, ensure_ascii=False, indent=1) + "\n")
    return 0 if not log["stop"].get("survivors") else 1


if __name__ == "__main__":
    raise SystemExit(main())
