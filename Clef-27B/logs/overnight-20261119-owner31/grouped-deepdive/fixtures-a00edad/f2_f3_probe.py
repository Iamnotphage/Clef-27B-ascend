#!/usr/bin/env python3
"""F2/F3 real-process probes for supervised_probe_window a00edad (CPU only).

F2: parent (env token) spawns a setsid worker and exits before the first
    periodic record; cleanup runs immediately after start(). Verify worker is
    registered, killed, reaped, and an unrelated untokened process survives.
F3: parent spawns an intermediate that spawns a setsid grandchild; verify both
    are registered on stop() and cleaned.
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, sys.argv[1])
import supervised_probe_window as W  # noqa: E402

TOKEN = "probe-owner-%d" % os.getpid()
ENV = {**os.environ, W.OWNER_ENV: TOKEN}


def alive(pid):
    return Path(f"/proc/{pid}").is_dir()


def state(pid):
    try:
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0]
    except OSError:
        return None


def run_f2(tmp):
    tmp = Path(tmp)
    tmp.mkdir(parents=True, exist_ok=True)
    worker_file = tmp / "f2-worker.pid"
    script = "setsid sleep 60 >/dev/null 2>&1 & echo $! > %s; exit 0" % worker_file
    subprocess.run(["bash", "-c", script], check=False, timeout=30, env=ENV)
    worker = int(worker_file.read_text().strip())
    unrelated = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=True
    )
    child = subprocess.Popen(["true"], env=ENV)
    child.wait()
    reg = W.DescendantRegistry(
        child.pid, tmp / "f2-descendants.json", protected=set(), token=TOKEN
    ).start()
    res = W.cleanup_attributed(child.pid, None, reg, set(), wait=0.2, reap=True)
    worker_after = alive(worker)
    worker_state = state(worker)
    unrelated_alive = alive(unrelated.pid)
    snap = reg.snapshot()
    reg.stop()
    survivors = W.attributed_survivors(reg, child.pid, None)
    try:
        os.kill(worker, 9)
    except ProcessLookupError:
        pass
    unrelated.kill()
    unrelated.wait(timeout=5)
    return {
        "case": "F2_cleanup_before_first_record_with_token",
        "worker_pid": worker,
        "cleanup": res,
        "worker_exists_after_cleanup": worker_after,
        "worker_state_after_cleanup": worker_state,
        "worker_registered": any(e["pid"] == worker for e in snap),
        "unrelated_pid": unrelated.pid,
        "unrelated_alive_after_cleanup": unrelated_alive,
        "registry_pids": [e["pid"] for e in snap],
        "survivors_after_stop": survivors,
    }


def run_f3(tmp):
    tmp = Path(tmp)
    tmp.mkdir(parents=True, exist_ok=True)
    worker_file = tmp / "f3-worker.pid"
    intermediate_py = (
        "import subprocess,sys,time\n"
        "w=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)'],"
        "start_new_session=True)\n"
        "open(%r,'w').write(str(w.pid))\n"
        "time.sleep(60)\n" % str(worker_file)
    )
    parent_py = (
        "import subprocess,sys\n"
        "subprocess.Popen([sys.executable,'-c',%r], start_new_session=True)\n"
        "sys.exit(0)\n" % intermediate_py
    )
    parent = subprocess.Popen([sys.executable, "-c", parent_py], env=ENV)
    parent.wait(timeout=10)
    deadline = time.time() + 10
    worker = None
    while time.time() < deadline:
        if worker_file.exists() and worker_file.read_text().strip():
            worker = int(worker_file.read_text().strip())
            break
        time.sleep(0.05)
    reg = W.DescendantRegistry(
        parent.pid, tmp / "f3-descendants.json", protected=set(), token=TOKEN
    ).start()
    reg.stop()
    snap = reg.snapshot()
    registered = {e["pid"] for e in snap}
    intermediate = [
        e["pid"]
        for e in snap
        if e["pid"] != worker and e.get("cmdline") and "time.sleep(60)" in " ".join(e["cmdline"])
    ]
    res = W.cleanup_attributed(parent.pid, None, reg, set(), wait=0.2, reap=True)
    workers_alive = {pid: alive(pid) for pid in registered if pid != os.getpid()}
    survivors = W.attributed_survivors(reg, parent.pid, None)
    for e in snap:
        try:
            os.kill(e["pid"], 9)
        except ProcessLookupError:
            pass
    return {
        "case": "F3_intermediate_and_grandchild_with_token",
        "grandchild_pid": worker,
        "registered_pids": sorted(registered),
        "grandchild_registered": worker in registered,
        "registered_count": len(registered),
        "cleanup": res,
        "alive_after_cleanup": workers_alive,
        "survivors": survivors,
    }


def main():
    if not sys.platform.startswith("linux"):
        print(json.dumps({"error": "linux only"}))
        return 2
    subreaper = W.enable_subreaper()
    base = Path(sys.argv[2])
    base.mkdir(parents=True, exist_ok=True)
    out = {"subreaper": subreaper, "module": W.__file__, "token": TOKEN}
    out["F2"] = run_f2(base / "f2")
    out["F3"] = run_f3(base / "f3")
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
