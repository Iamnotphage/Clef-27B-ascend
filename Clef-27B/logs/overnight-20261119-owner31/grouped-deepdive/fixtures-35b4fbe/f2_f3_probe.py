#!/usr/bin/env python3
"""F2/F3 real-process probes for supervised_probe_window 34f77ea (CPU only).

F2: parent spawns a setsid worker and exits before the first periodic registry
    record; cleanup runs immediately after start(). Check whether the worker is
    killed and whether survivors/gate see it.
F3: parent spawns an intermediate that holds a setsid grandchild, then exits.
    The first and only record happens in stop(); check whether the grandchild is
    registered and survives.
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, sys.argv[1])
import supervised_probe_window as W  # noqa: E402


def alive(pid):
    return Path(f"/proc/{pid}").is_dir()


def run_f2(tmp):
    tmp = Path(tmp)
    worker_file = tmp / "f2-worker.pid"
    script = (
        "setsid sleep 300 >/dev/null 2>&1 & echo $! > %s; exit 0" % worker_file
    )
    subprocess.run(["bash", "-c", script], check=False, timeout=30)
    worker = int(worker_file.read_text().strip())
    child = subprocess.Popen(["true"])
    child.wait()
    reg = W.DescendantRegistry(child.pid, tmp / "f2-descendants.json", protected=set())
    reg.start()
    res = W.cleanup_attributed(child.pid, None, reg, set(), wait=0.2)
    surv_now = W.attributed_survivors(reg, child.pid, None)
    worker_alive_after_cleanup = alive(worker)
    snap_after_cleanup = reg.snapshot()
    reg.stop()
    surv_stop = W.attributed_survivors(reg, child.pid, None)
    snap_stop = reg.snapshot()
    # cleanup the orphan ourselves so the probe leaves nothing behind
    try:
        os.kill(worker, 9)
    except ProcessLookupError:
        pass
    return {
        "case": "F2_cleanup_before_first_record",
        "worker_pid": worker,
        "cleanup": res,
        "survivors_immediately_after_cleanup": surv_now,
        "worker_alive_after_cleanup": worker_alive_after_cleanup,
        "registry_entries_after_cleanup": len(snap_after_cleanup),
        "survivors_after_stop": surv_stop,
        "registry_entries_after_stop": len(snap_stop),
        "worker_in_stop_registry": any(e["pid"] == worker for e in snap_stop),
    }


def run_f3(tmp):
    tmp = Path(tmp)
    worker_file = tmp / "f3-worker.pid"
    # intermediate stays alive holding the setsid grandchild
    script = (
        "bash -c 'setsid sleep 300 >/dev/null 2>&1 & echo $! > %s; sleep 300' & "
        "echo $! > %s; exit 0" % (worker_file, tmp / "f3-intermediate.pid")
    )
    subprocess.run(["bash", "-c", script], check=False, timeout=30)
    worker = int(worker_file.read_text().strip())
    intermediate = int((tmp / "f3-intermediate.pid").read_text().strip())
    child = subprocess.Popen(["true"])
    child.wait()
    reg = W.DescendantRegistry(child.pid, tmp / "f3-descendants.json", protected=set())
    reg.stop()  # first and only record happens here
    snap = reg.snapshot()
    registered = {e["pid"] for e in snap}
    worker_registered = worker in registered
    intermediate_registered = intermediate in registered
    surv = W.attributed_survivors(reg, child.pid, None)
    for pid in (worker, intermediate):
        try:
            os.kill(pid, 9)
        except ProcessLookupError:
            pass
    return {
        "case": "F3_intermediate_orphan_with_setsid_grandchild",
        "intermediate_pid": intermediate,
        "grandchild_pid": worker,
        "registered_pids": sorted(registered),
        "intermediate_registered": intermediate_registered,
        "grandchild_registered": worker_registered,
        "survivors": surv,
        "grandchild_alive_after_stop": alive(worker),
    }


def main():
    if not sys.platform.startswith("linux"):
        print(json.dumps({"error": "linux only"}))
        return 2
    subreaper = W.enable_subreaper()
    base = Path(sys.argv[2])
    base.mkdir(parents=True, exist_ok=True)
    out = {"subreaper": subreaper, "module": W.__file__}
    out["F2"] = run_f2(base / "f2")
    out["F3"] = run_f3(base / "f3")
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
