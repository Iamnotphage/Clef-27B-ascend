#!/usr/bin/env python3
"""Launch the Clef entry and pin its vLLM processes to fixed CPU sets.

Opt-in: ``CLEF_BIND_CPUS`` must contain a JSON object with the keys
``api``/``engine``/``tp0``/``tp1`` (``taskset`` CPU lists).  The supervisor
passes the child's exit status through and forwards SIGTERM/SIGINT to the
whole process group, so stopping the service stays a single signal.

The map is machine-specific; on the 2026-11-24 A3 host the measured map is
``{"api":"160-175","engine":"320-335","tp0":"240-255","tp1":"400-415"}``.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

POLL_SECONDS = 1.0
REPORT_ENV = "CLEF_BIND_REPORT"


def classify(args: str, cpus: dict[str, str]) -> str | None:
    if "Worker_TP0" in args:
        return cpus.get("tp0")
    if "Worker_TP1" in args:
        return cpus.get("tp1")
    if "EngineCore" in args:
        return cpus.get("engine")
    if "vllm.entrypoints.cli.main serve" in args:
        return cpus.get("api")
    return None


def processes(session: int) -> list[tuple[int, str]]:
    """Only the child's own session; never match unrelated host commands."""
    out = subprocess.run(["ps", "-eo", "pid,sess,args"], capture_output=True, text=True).stdout
    rows = []
    for line in out.splitlines()[1:]:
        parts = line.split(None, 2)
        if len(parts) == 3 and parts[0].isdigit() and parts[1].isdigit():
            if int(parts[1]) == session:
                rows.append((int(parts[0]), parts[2]))
    return rows


def allowed(pid: int) -> str | None:
    try:
        text = Path(f"/proc/{pid}/status").read_text()
    except OSError:
        return None
    for line in text.splitlines():
        if line.startswith("Cpus_allowed_list"):
            return line.split(":", 1)[1].strip()
    return None


def auto_map() -> dict[str, str] | None:
    """Reproduce the measured map: skip nodes 0-1 (OS/IRQ), then 2/4/3/5."""
    out = subprocess.run(["numactl", "--hardware"], capture_output=True, text=True)
    if out.returncode:
        return None
    nodes: dict[int, list[int]] = {}
    for line in out.stdout.splitlines():
        name, sep, value = line.partition(":")
        if not sep or "cpus" not in name:
            continue
        parts = name.split()
        if len(parts) == 3 and parts[0] == "node" and parts[1].isdigit():
            nodes[int(parts[1])] = [int(cpu) for cpu in value.split()]
    if min(nodes, default=0) < 0 or any(idx not in nodes for idx in (2, 3, 4, 5)):
        return None

    def span(idx: int) -> str | None:
        cpus = sorted(nodes[idx])[:16]
        if len(cpus) != 16:
            return None
        if cpus[-1] - cpus[0] == 15:
            return f"{cpus[0]}-{cpus[-1]}"
        return ",".join(str(cpu) for cpu in cpus)

    selected = {"api": span(2), "engine": span(4), "tp0": span(3), "tp1": span(5)}
    if any(not value for value in selected.values()):
        return None
    return selected


def main() -> int:
    raw = os.environ.get("CLEF_BIND_CPUS", "").strip()
    if not raw:
        raise SystemExit("CLEF_BIND_CPUS is required")
    if raw == "auto":
        cpus = auto_map()
        if cpus is None:
            raise SystemExit("CLEF_BIND_CPUS=auto is unsupported on this topology")
    else:
        cpus = json.loads(raw)
    missing = [key for key in ("api", "engine", "tp0", "tp1") if key not in cpus]
    if missing:
        raise SystemExit(f"CLEF_BIND_CPUS is missing {missing}")
    if len(sys.argv) < 2:
        raise SystemExit("usage: clef_bind_supervisor.py <command> [args...]")

    child = subprocess.Popen(sys.argv[1:], start_new_session=True)
    group = child.pid

    def forward(signum, _frame):
        try:
            os.killpg(group, signum)
        except ProcessLookupError:
            pass

    signal.signal(signal.SIGTERM, forward)
    signal.signal(signal.SIGINT, forward)

    bound: dict[str, dict] = {}
    started = time.monotonic()
    while child.poll() is None:
        for pid, args in processes(group):
            if str(pid) in bound:
                continue
            target = classify(args, cpus)
            if not target:
                continue
            result = subprocess.run(["taskset", "-acp", target, str(pid)],
                                    capture_output=True, text=True)
            if result.returncode == 0:
                bound[str(pid)] = {"cpus": target, "args": args[:120],
                                   "allowed": allowed(pid),
                                   "after_seconds": round(time.monotonic() - started, 2)}
                if REPORT_ENV and len(bound) >= 4:
                    Path(os.environ[REPORT_ENV]).write_text(
                        json.dumps(bound, indent=1) + "\n")
        time.sleep(POLL_SECONDS)

    code = child.wait()
    if REPORT_ENV:
        Path(os.environ[REPORT_ENV]).write_text(json.dumps(
            {"exit": code, "bound": bound}, indent=1) + "\n")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
