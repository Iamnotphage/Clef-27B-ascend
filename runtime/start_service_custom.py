#!/usr/bin/env python3
"""Start the snapshotted Clef service with custom argv overrides.

Keeps the manage_systemone_profile_service.py-compatible stop flow: the real
argv is written to current-service.json so the identity check still passes.
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
parser.add_argument("--label", required=True)
parser.add_argument("--overrides", required=False,
                    help='JSON: {"set":{"flag":"value"},"remove":["flag"],"append":["arg"],'
                         '"env":{"KEY":"value"},"wait_health_seconds":600}')
parser.add_argument("--overrides-file", required=False, help="Read the overrides JSON from this file")
args = parser.parse_args()
if bool(args.overrides) == bool(args.overrides_file):
    parser.error("provide exactly one of --overrides or --overrides-file")
spec = json.loads(open(args.overrides_file).read() if args.overrides_file else args.overrides)
run = Path(args.run_dir)
original = json.loads((run / "restore-service.json").read_text())
out = run / args.label
launch_marker = out / "launch.json"
if launch_marker.exists():
    print(json.dumps({"error": f"label already launched: {out}"}))
    sys.exit(1)
out.mkdir(parents=True, exist_ok=True)

argv = list(original["argv"])
for flag, value in spec.get("set", {}).items():
    value = str(value)
    if flag in argv:
        i = argv.index(flag)
        nxt = argv[i + 1] if i + 1 < len(argv) else None
        if nxt is not None and not nxt.startswith("-"):
            argv[i + 1] = value
        else:
            argv.insert(i + 1, value)
    else:
        argv += [flag, value]
for flag in spec.get("remove", []):
    while flag in argv:
        i = argv.index(flag)
        del argv[i]
        if i < len(argv) and not argv[i].startswith("-"):
            del argv[i]
argv += [str(a) for a in spec.get("append", [])]

environment = dict(os.environ)
environment.update(original["environment"])
environment.update({k: str(v) for k, v in spec.get("env", {}).items()})
for key in ["CLEF_PROFILE_MARKERS", "CLEF_PROFILE_OBSERVE_DIR", "CLEF_PROFILE_SCOPE_FILTER"]:
    environment.pop(key, None)

cwd = original["cwd"]
log_path = out / "service.log"
with log_path.open("wb") as log:
    process = subprocess.Popen(argv, cwd=cwd, env=environment, stdin=subprocess.DEVNULL,
                               stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
ENV_PREFIXES = ("CLEF_", "ASCEND_", "ATB_", "HCCL_")
ENV_KEYS = ("PATH", "LD_LIBRARY_PATH", "PYTHONPATH", "PYTHONUNBUFFERED", "VIRTUAL_ENV",
            "OMP_NUM_THREADS", "TASK_QUEUE_ENABLE", "MODEL_DIR", "HF_HOME",
            "TORCH_HOME", "TRITON_CACHE_DIR", "XDG_CACHE_HOME")
effective_env = {k: environment[k] for k in sorted(environment)
                 if k in ENV_KEYS or k.startswith(ENV_PREFIXES)}
record = {"pid": process.pid, "argv": argv, "cwd": cwd,
          "environment": effective_env,
          "environment_source_snapshot": original["environment"],
          "environment_override_keys": sorted(spec.get("env", {}).keys()),
          "label": args.label, "custom_overrides": spec,
          "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "log": str(log_path)}
(run / "current-service.json").write_text(json.dumps(record, ensure_ascii=False, indent=1))
(out / "launch.json").write_text(json.dumps(record, ensure_ascii=False, indent=1))
print(json.dumps({"pid": process.pid, "label": args.label}, ensure_ascii=False))

wait_seconds = spec.get("wait_health_seconds", 0)
port = "8000"
if "--port" in argv:
    i = argv.index("--port")
    if i + 1 < len(argv):
        port = argv[i + 1]
deadline = time.monotonic() + wait_seconds
ok = False
while time.monotonic() < deadline:
    try:
        with urlopen(f"http://127.0.0.1:{port}/health", timeout=3) as resp:
            if resp.status == 200:
                ok = True
                break
    except Exception:
        pass
    if process.poll() is not None:
        print(json.dumps({"error": "process exited early", "returncode": process.returncode}))
        sys.exit(1)
    time.sleep(5)
print(json.dumps({"healthy": ok}))
sys.exit(0 if ok or not wait_seconds else 1)
