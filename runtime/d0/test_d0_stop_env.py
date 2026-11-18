#!/usr/bin/env python3
"""CPU counterexamples for the precise stop and the child-env isolation."""
import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import d0_run  # noqa: E402
import d0_stop  # noqa: E402

fails = []


def check(name, cond, extra=""):
    print(f"{name}: {'OK' if cond else 'FAIL ' + str(extra)}")
    if not cond:
        fails.append(name)


def main() -> int:
    # 1 child env carries the vendor, the parent's environment is untouched
    before = dict(os.environ)
    env = d0_run.child_env("/some/vendor")
    check("parent env unchanged after child_env", os.environ == before)

    with tempfile.TemporaryDirectory() as td:
        v = Path(td) / "vendors" / "clef_d0_trace_22"
        (v / "op_api" / "lib").mkdir(parents=True)
        env = d0_run.child_env(str(v), base={"LD_LIBRARY_PATH": "/orig"})
        check("child env prepends vendor",
              env["LD_LIBRARY_PATH"].startswith(str(v / "op_api" / "lib"))
              and env["ASCEND_CUSTOM_OPP_PATH"].startswith(str(v))
              and env["LD_LIBRARY_PATH"].endswith("/orig"), env)
        check("child env keeps world size", env["WORLD_SIZE"] == "2")

    # 2 stop refuses a target without identity and never signals it
    calls = []
    with tempfile.TemporaryDirectory() as td:
        log = {}
        rec = d0_stop.stop(os.getpid(), log,
                           signaller=lambda p, s: calls.append((p, s)))
        check("stop refuses self/unknown", not calls
              and ("refused" in rec or "error" in rec), rec)

    # 3 identity gate rejects a missing start time
    ok, why = d0_stop.can_signal({"pid": 424242})
    check("stop requires identity fields", not ok and "starttime" in why, why)

    ok, why = d0_stop.can_signal({"pid": 1, "starttime": "1", "boot_id": "b"})
    check("stop refuses pid 1", not ok and "<= 1" in why, why)

    # 4 a target whose identity changed is refused before any signal
    orig = d0_stop.identity
    d0_stop.identity = lambda pid: {"pid": pid, "ppid": 1, "pgid": pid,
                                    "starttime": "999", "boot_id": "b", "cmdline": "x"}
    try:
        ok, why = d0_stop.can_signal({"pid": 424242, "starttime": "111", "boot_id": "b"})
    finally:
        d0_stop.identity = orig
    check("stop refuses reused pid", not ok and "identity changed" in why, why)

    # 5 a saved identity that disagrees with the live process stops the stop
    orig = d0_stop.identity
    d0_stop.identity = lambda pid: {"pid": pid, "ppid": 1, "pgid": 5,
                                    "starttime": "111", "boot_id": "b", "cmdline": "x"}
    calls = []
    try:
        log = {}
        rec = d0_stop.stop(424242, log, saved={"pid": 424242, "starttime": "222",
                                              "boot_id": "b", "pgid": 5},
                           signaller=lambda p, s: calls.append((p, s)))
        check("saved identity mismatch refuses", not calls
              and "does not match" in (rec.get("error") or ""), rec)
        log = {}
        rec = d0_stop.stop(424242, log, saved={"pid": 424242, "starttime": "111",
                                              "boot_id": "b", "pgid": 6},
                           signaller=lambda p, s: calls.append((p, s)))
        check("saved pgid mismatch refuses", not calls
              and "pgid" in (rec.get("error") or ""), rec)
        log = {}
        rec = d0_stop.stop(424242, log, saved={"pid": 424242, "starttime": "111",
                                              "boot_id": "b", "pgid": 5},
                           signaller=lambda p, s: calls.append((p, s)))
        check("matching identity proceeds", len(calls) >= 1, rec)
    finally:
        d0_stop.identity = orig

    # 6 real fixture: TERM is enough, the kill loop must not report it
    import subprocess
    child = subprocess.Popen(["sleep", "60"])
    ident = d0_stop.identity(child.pid)
    log = {}
    rec = d0_stop.stop(child.pid, log, saved=ident, term_wait=3, kill_wait=2)
    child.wait(timeout=5)
    check("clean term exits with rc0 semantics",
          rec["termed"] and not rec["killed"] and not rec["refused"]
          and not rec["survivors"] and child.returncode == -15, rec)

    print("RESULT:", "ALL PASS" if not fails else f"FAILURES: {fails}")
    return 0 if not fails else 1


if __name__ == "__main__":
    raise SystemExit(main())
