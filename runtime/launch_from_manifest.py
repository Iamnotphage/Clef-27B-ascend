#!/usr/bin/env python3
"""Launch the Clef API service from a recorded launch manifest.

The manifest supplies the exact served argparse and a credential-free
environment captured from a verified run.  This launcher only changes the
requested environment, port, profiler configuration and observation switches,
so an experiment cannot silently drift from the recorded service.

Examples:
  python runtime/launch_from_manifest.py \
    --manifest /workspace/yangchen/logs/gdn-data-path-20261115/head-entry/launch-manifest.json \
    --run-dir /workspace/yangchen/logs/tuning5-20261117/shape-trace/run \
    --set CLEF_GDN_QKV_NORM_MAX_TOKENS=0
"""

from __future__ import annotations

import argparse
import json
import socket
import subprocess
import time
from pathlib import Path

DEFAULT_CWD = "/workspace/yangchen/Clef-27B-ascend"


def replace_flag(argv: list[str], flag: str, value: str | None) -> list[str]:
    result: list[str] = []
    index = 0
    while index < len(argv):
        item = argv[index]
        if item == flag:
            index += 2
            continue
        if item.startswith(flag + "="):
            index += 1
            continue
        result.append(item)
        index += 1
    if value is not None:
        result += [flag, value]
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--cwd", default=DEFAULT_CWD)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--profile", action="store_true", help="enable the torch profiler and /start_profile")
    parser.add_argument("--observations", action="store_true", help="enable CLEF CPU observation scopes")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                        help="environment override; repeatable")
    parser.add_argument("--unset", action="append", default=[], help="environment variable to remove")
    parser.add_argument("--argv-set", action="append", default=[], metavar="FLAG[=VALUE]",
                        help="replace or drop a served CLI flag; repeatable")
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text())
    argv = list(manifest["argv"])
    environment = dict(manifest.get("environment") or {})
    environment.update(manifest.get("service_environment") or {})

    for item in args.set:
        key, _, value = item.partition("=")
        if not key or not _:
            parser.error(f"--set expects KEY=VALUE, got {item!r}")
        environment[key] = value
    for key in args.unset:
        environment.pop(key, None)

    if args.port is not None:
        argv = replace_flag(argv, "--port", str(args.port))
    port = None
    for index, item in enumerate(argv):
        if item == "--port" and index + 1 < len(argv):
            port = int(argv[index + 1])
    if port is not None:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.settimeout(0.5)
            if probe.connect_ex(("127.0.0.1", port)) == 0:
                raise SystemExit(f"port {port} is already serving; stop the recorded service first")
    for item in args.argv_set:
        flag, separator, value = item.partition("=")
        argv = replace_flag(argv, flag, value if separator else None)

    if args.profile:
        config = {
            "profiler": "torch",
            "torch_profiler_dir": str(args.run_dir / "traces"),
            "torch_profiler_with_stack": False,
            "torch_profiler_with_memory": False,
            "torch_profiler_record_shapes": False,
            "ignore_frontend": True,
        }
        argv = replace_flag(argv, "--profiler-config", json.dumps(config, separators=(",", ":")))
    else:
        argv = replace_flag(argv, "--profiler-config", None)

    if args.observations:
        environment["CLEF_PROFILE_MARKERS"] = "1"
        environment["CLEF_PROFILE_OBSERVE_DIR"] = str(args.run_dir / "cpu-timings")
    else:
        environment.pop("CLEF_PROFILE_MARKERS", None)
        environment.pop("CLEF_PROFILE_OBSERVE_DIR", None)

    args.run_dir.mkdir(parents=True, exist_ok=False)
    log_path = args.run_dir / "service.log"
    with log_path.open("wb") as log:
        process = subprocess.Popen(
            argv,
            cwd=args.cwd,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

    record = {
        "pid": process.pid,
        "argv": argv,
        "cwd": args.cwd,
        "environment": environment,
        "profile": args.profile,
        "observations": args.observations,
        "manifest": str(args.manifest),
        "log": str(log_path),
    }
    time.sleep(2)
    if process.poll() is not None:
        tail = log_path.read_text(errors="replace")[-2000:]
        raise SystemExit(f"service exited immediately (rc={process.returncode}):\n{tail}")
    (args.run_dir / "launch.json").write_text(json.dumps(record, ensure_ascii=False, indent=1) + "\n")
    print(json.dumps({"pid": process.pid, "run_dir": str(args.run_dir), "log": str(log_path)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
