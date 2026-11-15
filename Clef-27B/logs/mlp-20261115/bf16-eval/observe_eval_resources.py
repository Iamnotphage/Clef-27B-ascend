#!/usr/bin/env python3
"""Sample raw resources beside an evaluation driver until it finishes.

Writes a new JSONL file. Each sample has UTC and monotonic boundaries, driver
snapshots before and after collection, raw npu-smi output, and raw cgroup CPU
statistics/quota. Missing values remain null with explicit error information.
The observer reads resources and never manages service or device lifecycle.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path, PurePosixPath
import subprocess
import time


NPU_TIMEOUT_SECONDS = 10
TERMINAL_STATUSES = {"complete", "completed", "failed"}


def timestamp():
    return {"utc": datetime.now(timezone.utc).isoformat(), "monotonic_seconds": time.monotonic()}


def read_text(path):
    try:
        return {"path": str(path), "status": "ok", "raw": Path(path).read_text(), "error": None}
    except OSError as error:
        return {"path": str(path), "status": "error", "raw": None,
                "error": f"{type(error).__name__}: {error}"}


def read_driver(path):
    record = {**timestamp(), "path": str(path), "status": "unknown", "current_step": None,
              "step_status": "unknown", "step_started_at": None, "step_finished_at": None,
              "driver_started_at": None, "driver_finished_at": None, "error": None}
    try:
        driver = json.loads(Path(path).read_text())
        record["status"] = driver.get("status", "unknown")
        record["driver_started_at"] = driver.get("started_at")
        record["driver_finished_at"] = driver.get("finished_at")
        record["error"] = driver.get("error")
        steps = driver.get("steps", [])
        active = [step for step in steps if step.get("status") == "running"]
        step = active[-1] if active else steps[-1] if steps else {}
        record.update(current_step=step.get("name"), step_status=step.get("status", "unknown"),
                      step_started_at=step.get("started_at"), step_finished_at=step.get("finished_at"))
    except (OSError, ValueError, TypeError, AttributeError) as error:
        record["status"] = "unknown"
        record["error"] = f"{type(error).__name__}: {error}"
    return record


def npu_snapshot():
    record = {"command": ["npu-smi", "info"], "timeout_seconds": NPU_TIMEOUT_SECONDS,
              "started": timestamp(), "status": "unknown", "returncode": None,
              "timed_out": False, "stdout": None, "stderr": None, "error": None}
    try:
        result = subprocess.run(record["command"], capture_output=True, text=True,
                                check=False, timeout=NPU_TIMEOUT_SECONDS)
        record.update(status="ok" if result.returncode == 0 else "error",
                      returncode=result.returncode, stdout=result.stdout, stderr=result.stderr)
    except subprocess.TimeoutExpired as error:
        def decoded(value):
            return value.decode(errors="replace") if isinstance(value, bytes) else value
        record.update(status="timeout", timed_out=True, stdout=decoded(error.stdout),
                      stderr=decoded(error.stderr), error=str(error))
    except OSError as error:
        record.update(status="error", error=f"{type(error).__name__}: {error}")
    record["finished"] = timestamp()
    return record


def cgroup_snapshot():
    membership = read_text("/proc/self/cgroup")
    mounts = read_text("/proc/self/mountinfo")
    result = {"membership": membership, "v1": {"status": "unknown", "directory": None},
              "v2": {"status": "unknown", "directory": None}, "discovery_error": None}
    try:
        if membership["raw"] is None or mounts["raw"] is None:
            raise ValueError(membership["error"] or mounts["error"])
        groups = [line.split(":", 2) for line in membership["raw"].splitlines()]
        for line in mounts["raw"].splitlines():
            before, after = line.split(" - ", 1)
            fields, details = before.split(), after.split()
            filesystem = details[0]
            if filesystem not in ("cgroup", "cgroup2"):
                continue
            version = "v2" if filesystem == "cgroup2" else "v1"
            matching = [group for group in groups if
                        (version == "v2" and group[1] == "") or
                        (version == "v1" and "cpu" in group[1].split(",") and "cpu" in details[2].split(","))]
            if not matching:
                continue
            root, mountpoint = PurePosixPath(fields[3]), Path(fields[4])
            group_path = PurePosixPath(matching[0][2])
            if group_path.is_relative_to(root):
                directory = mountpoint / str(group_path.relative_to(root))
            else:
                # A cgroup namespace can expose '/' relative to the mounted root.
                directory = mountpoint / str(group_path).lstrip("/")
            names = ("cpu.stat", "cpu.max") if version == "v2" else (
                "cpu.stat", "cpu.cfs_quota_us", "cpu.cfs_period_us")
            files = {name: read_text(directory / name) for name in names}
            result[version] = {"status": "ok" if all(item["status"] == "ok" for item in files.values()) else "error",
                               "directory": str(directory), "mountinfo_line": line, "files": files}
    except (ValueError, IndexError, OSError) as error:
        result["discovery_error"] = f"{type(error).__name__}: {error}"
    return result


def sample(driver_path):
    started = timestamp()
    before = read_driver(driver_path)
    npu = npu_snapshot()
    cgroup = cgroup_snapshot()
    meminfo = read_text("/proc/meminfo")
    keys = ("MemTotal", "MemAvailable", "MemFree", "Buffers", "Cached", "SwapTotal", "SwapFree")
    fields = dict(line.split(":", 1) for line in (meminfo["raw"] or "").splitlines() if ":" in line)
    memory = {"path": meminfo["path"], "status": meminfo["status"], "error": meminfo["error"],
              "fields": {key: fields[key].strip() if key in fields else None for key in keys}}
    after = read_driver(driver_path)
    return {"schema_version": 1, "started": started, "finished": timestamp(),
            "current_step": before["current_step"], "driver_before": before, "driver_after": after,
            "npu_smi": npu, "cgroup": cgroup, "meminfo": memory,
            "terminal_driver": after["status"] in TERMINAL_STATUSES}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--driver", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="new JSONL path")
    parser.add_argument("--interval", type=float, default=15.0)
    args = parser.parse_args(argv)
    if not math.isfinite(args.interval) or args.interval <= 0:
        parser.error("--interval must be finite and positive")
    args.driver = args.driver.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        while True:
            record = sample(args.driver)
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            stream.flush()
            if record["terminal_driver"]:
                break
            deadline = record["started"]["monotonic_seconds"] + args.interval
            time.sleep(max(0.0, deadline - time.monotonic()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
