#!/usr/bin/env python3
"""Stop exactly the dual-TP1 instances recorded in $RUN/dual-tp1.json."""
import argparse
import json
import os
import signal
import sys
import time
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--run-dir", required=True)
args = parser.parse_args()
run = Path(args.run_dir)
record_path = run / "dual-tp1.json"
if not record_path.exists():
    print(json.dumps({"error": "no dual-tp1.json"}))
    sys.exit(1)
record = json.loads(record_path.read_text())


def alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def descendants(pid):
    result = {}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            stat = (entry / "stat").read_text().rsplit(")", 1)[-1].split()
            ppid = int(stat[1])
            if ppid == pid:
                result[int(entry.name)] = descendants(int(entry.name))
        except (OSError, IndexError, ValueError):
            continue
    return result


def flatten(d):
    out = []
    for k, v in d.items():
        out.append(k)
        out.extend(flatten(v))
    return out


killed = []
for name, info in record.get("instances", {}).items():
    pid = info.get("pid")
    if pid and alive(pid):
        tree = flatten(descendants(pid))
        os.kill(pid, signal.SIGTERM)
        killed.append({"instance": name, "pid": pid, "children": tree})

deadline = time.monotonic() + 45
all_pids = [k["pid"] for k in killed] + [c for k in killed for c in k["children"]]
while time.monotonic() < deadline and any(alive(p) for p in all_pids):
    time.sleep(1)
remaining = [p for p in all_pids if alive(p)]
for p in remaining:
    try:
        os.kill(p, signal.SIGKILL)
    except OSError:
        pass
(run / "dual-tp1-stopped.json").write_text(json.dumps(
    {"stopped": killed, "forced": remaining,
     "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}, indent=1))
print(json.dumps({"stopped": [k["pid"] for k in killed], "forced": remaining}))
sys.exit(0)
