#!/usr/bin/env python3
"""Start two TP1 Clef instances bound to the two logical NPUs of one physical card.

Instance A -> ASCEND_RT_VISIBLE_DEVICES=0, port 8000
Instance B -> ASCEND_RT_VISIBLE_DEVICES=1, port 8001
Starts are staggered to avoid overlapping host-side weight-load peaks.
Writes $RUN/dual-tp1.json so stop_dual_tp1.py can manage exactly these pids.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from urllib.request import urlopen

parser = argparse.ArgumentParser()
parser.add_argument("--run-dir", required=True)
parser.add_argument("--label", default="p3-dual-tp1")
args = parser.parse_args()
run = Path(args.run_dir)
original = json.loads((run / "restore-service.json").read_text())
out = run / args.label
if out.exists():
    print(json.dumps({"error": f"label dir exists: {out}"}))
    sys.exit(1)

base_argv = list(original["argv"])
# normalize to TP1/4096/seqs per unified conditions; seqs set per instance below


def build_instance(port, seqs):
    argv = list(base_argv)
    for flag, value in [("--tensor-parallel-size", "1"), ("--max-model-len", "4096"),
                        ("--max-num-batched-tokens", "4096"), ("--max-num-seqs", str(seqs)),
                        ("--port", str(port))]:
        if flag in argv:
            i = argv.index(flag)
            if i + 1 < len(argv) and not argv[i + 1].startswith("-"):
                argv[i + 1] = value
                continue
        argv += [flag, value]
    return argv


def launch(argv, device, name):
    env = dict(os.environ)
    env.update(original["environment"])
    env["ASCEND_RT_VISIBLE_DEVICES"] = device
    for key in ["CLEF_PROFILE_MARKERS", "CLEF_PROFILE_OBSERVE_DIR", "CLEF_PROFILE_SCOPE_FILTER"]:
        env.pop(key, None)
    inst_dir = out / name
    inst_dir.mkdir(parents=True)
    log = (inst_dir / "service.log").open("wb")
    proc = subprocess.Popen(argv, cwd=original["cwd"], env=env, stdin=subprocess.DEVNULL,
                            stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    (inst_dir / "launch.json").write_text(json.dumps(
        {"pid": proc.pid, "argv": argv, "device": device, "port": port_of(argv),
         "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "log": str(inst_dir / "service.log")},
        ensure_ascii=False, indent=1))
    return proc


def port_of(argv):
    return int(argv[argv.index("--port") + 1])


def wait_health(pid, port, timeout_seconds):
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        try:
            with urlopen(f"http://127.0.0.1:{port}/health", timeout=3) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            pass
        if pid.poll() is not None:
            return False
        time.sleep(5)
    return False


procs = {}
instances = [("A", "0", 8000), ("B", "1", 8001)]
record = {"label": args.label, "instances": {}}
for name, device, port in instances:
    seqs = json.loads(os.environ.get("DUAL_TP1_SEQS", "1"))
    proc = launch(build_instance(port, seqs), device, name)
    procs[name] = proc
    ok = wait_health(proc, port, 600)
    record["instances"][name] = {"pid": proc.pid, "device": device, "port": port,
                                 "healthy": ok,
                                 "log": str(out / name / "service.log")}
    print(json.dumps({"instance": name, "pid": proc.pid, "port": port, "healthy": ok}), flush=True)
    if not ok:
        record["status"] = f"instance {name} failed"
        break
    time.sleep(5)  # stagger: let A finish allocator setup before B starts loading

record["started_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
record["status"] = "running" if all(i.get("healthy") for i in record["instances"].values()) else "degraded"
run.joinpath("dual-tp1.json").write_text(json.dumps(record, ensure_ascii=False, indent=1))
print(json.dumps({"status": record["status"]}))
sys.exit(0 if record["status"] == "running" else 1)
