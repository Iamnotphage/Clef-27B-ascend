#!/usr/bin/env python3
"""Start the snapshotted Clef service with custom argv overrides.

Keeps the manage_systemone_profile_service.py-compatible stop flow: the real
argv is written to current-service.json so the identity check still passes.

Every override removes all existing occurrences of the flag (including
``--flag=value``) before appending one canonical pair, so snapshots that
contain historical duplicates are normalized. The launch record stores the
normalized argv, the dropped duplicates and the effective override list.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from urllib.request import urlopen

BOOLEAN_FLAGS = {
    "--enforce-eager",
    "--no-enable-prefix-caching",
    "--no-enable-chunked-prefill",
    "--skip-mm-profiling",
    "--disable-log-stats",
    "--trust-remote-code",
}

parser = argparse.ArgumentParser()
parser.add_argument("--run-dir", required=True)
parser.add_argument("--label", required=True)
parser.add_argument("--overrides", required=False,
                    help='JSON: {"set":{"flag":"value"},"remove":["flag"],"append":["arg"],'
                         '"env":{"KEY":"value"},"wait_health_seconds":600}')
parser.add_argument("--overrides-file", required=False, help="Read the overrides JSON from this file")
parser.add_argument("--dry-run", action="store_true",
                    help="Print the normalized launch record without starting a process.")
args = parser.parse_args()
if bool(args.overrides) == bool(args.overrides_file):
    parser.error("provide exactly one of --overrides or --overrides_file")
spec = json.loads(open(args.overrides_file).read() if args.overrides_file else args.overrides)
run = Path(args.run_dir)
original = json.loads((run / "restore-service.json").read_text())
out = run / args.label
launch_marker = out / "launch.json"
if launch_marker.exists():
    print(json.dumps({"error": f"label already launched: {out}"}))
    sys.exit(1)


def parse_items(argv):
    items, index = [], 0
    while index < len(argv):
        token = argv[index]
        if token.startswith("--"):
            if "=" in token:
                flag, value = token.split("=", 1)
                items.append((flag, value, [token]))
                index += 1
            elif index + 1 < len(argv) and not argv[index + 1].startswith("--"):
                items.append((token, argv[index + 1], [token, argv[index + 1]]))
                index += 2
            else:
                items.append((token, None, [token]))
                index += 1
        else:
            items.append((None, token, [token]))
            index += 1
    return items


def rebuild(items):
    tokens = []
    for _, _, original_tokens in items:
        tokens.extend(original_tokens)
    return tokens


def normalize(argv):
    """Resolve duplicate flags with argparse's last-wins semantics."""
    items = parse_items(argv)
    last = {}
    for position, (flag, _, _) in enumerate(items):
        if flag is not None:
            last[flag] = position
    kept, duplicates = [], []
    for position, (flag, value, tokens) in enumerate(items):
        if flag is not None and last[flag] != position:
            duplicates.append({"flag": flag, "dropped_value": value,
                               "kept_value": items[last[flag]][1]})
            continue
        kept.append((flag, value, tokens))
    return rebuild(kept), duplicates


def strip_flag(argv, flag):
    return rebuild([item for item in parse_items(argv) if item[0] != flag])


argv, normalized_duplicates = normalize(list(original["argv"]))
for flag, value in spec.get("set", {}).items():
    argv = strip_flag(argv, flag)
    value = str(value)
    if flag in BOOLEAN_FLAGS:
        if value.lower() in ("0", "false", "no", "off"):
            continue
        argv += [flag]
    else:
        argv += [flag, value]
for flag in spec.get("remove", []):
    argv = strip_flag(argv, flag)
argv += [str(item) for item in spec.get("append", [])]
argv, post_duplicates = normalize(argv)
if post_duplicates:
    raise SystemExit(
        "duplicate flags remain after overrides: "
        + json.dumps(post_duplicates, ensure_ascii=False))

environment = dict(os.environ)
environment.update(original["environment"])
environment.update({k: str(v) for k, v in spec.get("env", {}).items()})
for key in ["CLEF_PROFILE_MARKERS", "CLEF_PROFILE_OBSERVE_DIR", "CLEF_PROFILE_SCOPE_FILTER"]:
    environment.pop(key, None)

cwd = original["cwd"]
log_path = out / "service.log"
ENV_PREFIXES = ("CLEF_", "ASCEND_", "ATB_", "HCCL_")
ENV_KEYS = ("PATH", "LD_LIBRARY_PATH", "PYTHONPATH", "PYTHONUNBUFFERED", "VIRTUAL_ENV",
            "OMP_NUM_THREADS", "TASK_QUEUE_ENABLE", "MODEL_DIR", "HF_HOME",
            "TORCH_HOME", "TRITON_CACHE_DIR", "XDG_CACHE_HOME")
effective_env = {k: environment[k] for k in sorted(environment)
                 if k in ENV_KEYS or k.startswith(ENV_PREFIXES)}
record = {"pid": None, "argv": argv, "cwd": cwd,
          "argv_sha256": hashlib.sha256(
              "\0".join(argv).encode()).hexdigest(),
          "normalized_duplicates": normalized_duplicates,
          "post_override_duplicates": post_duplicates,
          "environment": effective_env,
          "environment_source_snapshot": original["environment"],
          "environment_override_keys": sorted(spec.get("env", {}).keys()),
          "label": args.label, "custom_overrides": spec,
          "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "log": str(log_path)}
if args.dry_run:
    print(json.dumps(record, ensure_ascii=False, indent=1))
    sys.exit(0)

out.mkdir(parents=True, exist_ok=True)
with log_path.open("wb") as log:
    process = subprocess.Popen(argv, cwd=cwd, env=environment, stdin=subprocess.DEVNULL,
                               stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
record["pid"] = process.pid
(run / "current-service.json").write_text(json.dumps(record, ensure_ascii=False, indent=1))
(out / "launch.json").write_text(json.dumps(record, ensure_ascii=False, indent=1))
print(json.dumps({"pid": process.pid, "label": args.label}, ensure_ascii=False))

wait_seconds = spec.get("wait_health_seconds", 0)
port = "8000"
if "--port" in argv:
    index = argv.index("--port")
    if index + 1 < len(argv):
        port = argv[index + 1]
deadline = time.monotonic() + wait_seconds
ok = False
while time.monotonic() < deadline:
    try:
        with urlopen(f"http://127.0.0.1:{port}/health", timeout=3) as response:
            if response.status == 200:
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
