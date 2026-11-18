#!/usr/bin/env python3
"""CPU fixture for the D0 child supervision. Uses shell sleepers, no NPU.

Covers: timeout kills the whole session group, a leader that exits first while
a grandchild holds the pipe, and nonzero child rc kept apart from wrapper rc.
"""
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from d0_run import run_child  # noqa: E402

fails = []


def check(name, cond, extra=""):
    print(f"{name}: {'OK' if cond else 'FAIL ' + str(extra)}")
    if not cond:
        fails.append(name)


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def main() -> int:
    # 1 timeout: the whole group, including a grandchild, must be gone
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        marker = d / "grandchild.pid"
        script = (f"sh -c 'sleep 60 & echo $! > {marker}; wait' ")
        t0 = time.time()
        rc = run_child(["sh", "-c", script], d, timeout=2)
        dt = time.time() - t0
        time.sleep(0.5)
        gpid = int(marker.read_text()) if marker.exists() else 0
        meta = json.loads((d / "raw-rc.json").read_text())
        check("timeout kills group", rc == 2 and meta["timed_out"] is True
              and meta["wrapper_rc"] == 2, meta)
        check("grandchild gone after timeout", gpid and not alive(gpid), f"pid={gpid}")
        check("timeout bounded", dt < 20, f"{dt:.1f}s")
        check("trace unavailable recorded", (d / "trace-unavailable.json").exists())

    # 2 leader exits first while a grandchild keeps the pipe open
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        marker = d / "gc.pid"
        script = f"sleep 60 & echo $! > {marker}; exit 0"
        t0 = time.time()
        rc = run_child(["sh", "-c", script], d, timeout=2)
        dt = time.time() - t0
        time.sleep(0.5)
        gpid = int(marker.read_text()) if marker.exists() else 0
        meta = json.loads((d / "raw-rc.json").read_text())
        check("leader-first bounded", dt < 40, f"{dt:.1f}s")
        check("leader-first timed out with wrapper 2",
              meta["timed_out"] is True and meta["wrapper_rc"] == 2
              and meta["child_raw_rc"] == 0, meta)
        check("leader-first grandchild exited", gpid and not alive(gpid), f"pid={gpid}")
        after = json.loads((d / "tree-after.json").read_text())
        check("survivors judged by captured pgid", "surviving_members" in after, after.keys())

    # 3 nonzero child rc is kept separate from the wrapper rc
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        rc = run_child(["sh", "-c", "exit 7"], d, timeout=10)
        meta = json.loads((d / "raw-rc.json").read_text())
        check("nonzero rc preserved", rc == 7 and meta["child_raw_rc"] == 7
              and meta["wrapper_rc"] == 7 and meta["timed_out"] is False, meta)

    # 4 clean exit
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        rc = run_child(["sh", "-c", "echo hi"], d, timeout=10)
        check("clean exit", rc == 0 and (d / "stdout.log").read_text().strip() == "hi")

    print("RESULT:", "ALL PASS" if not fails else f"FAILURES: {fails}")
    return 0 if not fails else 1


if __name__ == "__main__":
    raise SystemExit(main())
