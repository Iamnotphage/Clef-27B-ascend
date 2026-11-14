#!/usr/bin/env python3
"""Capture the running Clef service and replay the frozen baseline.

The snapshot records the process identity (PID, start time, parent, cwd, full
argv, whitelisted environment), the NPU mapping, source/patch hashes, the
health payload and the frozen 8-case /v1/systemone replay.  Everything is
written as one JSON document so the same tool can verify the restored service
at the end of a run.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import urllib.request

ENV_KEYS = {
    "PATH", "LD_LIBRARY_PATH", "PYTHONPATH", "PYTHONUNBUFFERED", "VIRTUAL_ENV",
    "OMP_NUM_THREADS", "TASK_QUEUE_ENABLE", "MODEL_DIR", "HF_HOME",
    "TORCH_HOME", "TRITON_CACHE_DIR", "XDG_CACHE_HOME", "CLEF_LOG_DIR",
    "CLEF_VENV", "CLEF_WORKSPACE", "CLEF_CANN_ENV", "CLEF_ATB_ENV",
    "CLEF_OD_DATA", "CLEF_OD_SCORER",
}
ENV_PREFIXES = ("CLEF_", "ASCEND_", "ATB_", "HCCL_", "FLA_")
SECRET_MARKERS = ("TOKEN", "PASSWORD", "SECRET", "KEY", "CREDENTIAL")
IGNORED_RESPONSE_FIELDS = {"id", "created"}


def now():
    return datetime.now(timezone.utc).isoformat()


def run(command, timeout=30):
    try:
        result = subprocess.run(
            command, capture_output=True, text=True, timeout=timeout, check=False)
        return {
            "command": command,
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
    except (OSError, subprocess.SubprocessError) as exc:
        return {"command": command, "error": f"{type(exc).__name__}: {exc}"}


def sha256_file(path: Path) -> str | None:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def process_info(pid: int):
    base = Path("/proc") / str(pid)
    info = {"pid": pid}
    try:
        stat = (base / "stat").read_text().split()
        info["ppid"] = int(stat[3])
        info["state"] = stat[2]
    except (OSError, IndexError, ValueError) as exc:
        info["stat_error"] = repr(exc)
    for name, field in (("cmdline", "argv"), ("environ", "environment")):
        try:
            raw = (base / name).read_bytes()
        except OSError:
            continue
        if name == "cmdline":
            info[field] = [item.decode("utf-8", "replace") for item in raw.split(b"\0") if item]
        else:
            environment = {}
            for item in raw.split(b"\0"):
                if not item or b"=" not in item:
                    continue
                key, value = item.split(b"=", 1)
                key = key.decode("utf-8", "replace")
                if key in ENV_KEYS or key.startswith(ENV_PREFIXES):
                    value = value.decode("utf-8", "replace")
                    if any(marker in key.upper() for marker in SECRET_MARKERS):
                        value = "<redacted>"
                    environment[key] = value
            info[field] = dict(sorted(environment.items()))
    try:
        info["cwd"] = os.readlink(base / "cwd")
    except OSError as exc:
        info["cwd_error"] = repr(exc)
    info["ps"] = run(["ps", "-o", "pid,ppid,lstart,etime,args", "-p", str(pid)])
    return info


def process_tree(pid: int):
    output = run(["ps", "-eo", "pid,ppid,stat,lstart,etime,args", "--sort=pid"])
    lines = output.get("stdout", "").splitlines()
    return output, [
        line for line in lines
        if str(pid) in line.split()[:2]
    ]


def source_state(paths):
    state = {}
    for path in paths:
        path = Path(path)
        state[str(path)] = {
            "exists": path.exists(),
            "git_head": run(["git", "-C", str(path), "rev-parse", "HEAD"])["stdout"].strip(),
            "git_branch": run(["git", "-C", str(path), "rev-parse", "--abbrev-ref", "HEAD"])["stdout"].strip(),
            "git_status": run(["git", "-C", str(path), "status", "--porcelain"])["stdout"].splitlines(),
        }
    return state


def normalize(value):
    if isinstance(value, dict):
        return {k: normalize(v) for k, v in value.items() if k not in IGNORED_RESPONSE_FIELDS}
    if isinstance(value, list):
        return [normalize(item) for item in value]
    return value


def post_json(url, payload, timeout):
    body = json.dumps(payload, ensure_ascii=False).encode()
    request = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.status, json.loads(response.read())


def replay_frozen(baseline_path, url, timeout):
    baseline = json.loads(Path(baseline_path).read_text())
    report = {"url": url, "baseline": str(baseline_path), "cases": {}, "passed": True}
    for name, case in sorted(baseline.items()):
        try:
            status, response = post_json(url, case["request"], timeout)
            equal = (
                status == case["status"]
                and normalize(case["response"]) == normalize(response)
            )
            report["cases"][name] = {
                "status": status, "expected_status": case["status"],
                "equal": equal, "actual": None if equal else normalize(response),
            }
            report["passed"] = report["passed"] and equal
        except Exception as exc:  # noqa: BLE001 - snapshot records any failure
            report["cases"][name] = {"error": f"{type(exc).__name__}: {exc}"}
            report["passed"] = False
    report["summary"] = {
        "cases": len(report["cases"]),
        "passed": sum(1 for item in report["cases"].values() if item.get("equal")),
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--baseline", type=Path, default=None,
                        help="冻结 8 用例基线 JSON；提供时执行回放")
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--hash", action="append", default=[], help="要计算 SHA256 的文件")
    parser.add_argument("--source", action="append", default=[], help="要记录 git 状态的目录")
    args = parser.parse_args()

    document = {
        "captured_at": now(),
        "host": run(["hostname"])["stdout"].strip(),
        "pid": args.pid,
        "process": process_info(args.pid),
        "npu_smi": run(["npu-smi", "info"])["stdout"],
        "source": source_state(args.source),
        "file_sha256": {
            path: sha256_file(Path(path)) for path in args.hash
        },
    }
    tree, selected = process_tree(args.pid)
    document["process_tree"] = selected
    document["process_tree_command"] = tree.get("command")
    try:
        with urllib.request.urlopen(f"{args.url.rstrip('/')}/health", timeout=30) as response:
            document["health"] = {
                "status_code": response.status,
                "body": response.read().decode("utf-8", "replace"),
                "headers": dict(response.headers.items()),
            }
    except Exception as exc:  # noqa: BLE001
        document["health"] = {"error": f"{type(exc).__name__}: {exc}"}
    if args.baseline:
        document["frozen_replay"] = replay_frozen(
            args.baseline, f"{args.url.rstrip('/')}/systemone", args.timeout)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, ensure_ascii=False, indent=1) + "\n")
    print(json.dumps({
        "output": str(args.output),
        "pid": args.pid,
        "health": document["health"].get("status_code"),
        "frozen_passed": document.get("frozen_replay", {}).get("passed"),
    }, ensure_ascii=False))
    return 0 if document.get("frozen_replay", {"passed": True})["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
