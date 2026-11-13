#!/usr/bin/env python3
"""Run one fresh, full OpenDecisions evaluation and retain its provenance."""

import argparse
import datetime
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys


DATA_REVISION = "ff0b640be2b72c7d06d9d656a34ad6eed79ec7fe"
DATA_SHA256 = "78614972006c5e5efbbfc121ce31e15cf576ef6c0cdf24a229b33d34d2bfcda6"


def stamp():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path,
                        default=Path("/workspace/yangchen/datasets/clef-full-20261112/full-data"))
    parser.add_argument("--api-url", default="http://127.0.0.1:8000/v1")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    data_dir = args.data_dir.resolve()
    data_file = data_dir / "open-decisions2352.jsonl"
    blob = data_file.read_bytes()
    rows = [json.loads(line) for line in blob.splitlines() if line.strip()]
    digest = hashlib.sha256(blob).hexdigest()
    if digest != DATA_SHA256 or len(rows) != 2352:
        parser.error("OpenDecisions data must match the pinned 2,352-record snapshot")
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        parser.error("--output-dir must be empty; use a fresh directory for every run")
    project = Path(__file__).resolve().parents[2]
    runner = project / "scripts/run_evalscope_offline.py"
    command = [sys.executable, str(runner), "--data-dir", str(data_dir),
               "--work-dir", str(output / "evalscope"), "--api-url", args.api_url,
               "--benchmark", "open_decisions", "--scope", "full"]
    environment = dict(os.environ, HF_HUB_OFFLINE="1", PYTHONUNBUFFERED="1")
    launch = {"started_utc": stamp(), "supervisor_pid": os.getpid(),
              "command": command, "working_directory": str(project),
              "dataset": {"version": "1.0.1", "revision": DATA_REVISION,
                          "path": str(data_file), "sha256": digest,
                          "bytes": len(blob), "rows": len(rows), "split": "test"},
              "model": "clef", "eval_batch_size": 1,
              "additional_temperature_calibration": False,
              "evalscope_version": importlib.metadata.version("evalscope"),
              "runner_sha256": hashlib.sha256(runner.read_bytes()).hexdigest(),
              "environment_overrides": {"HF_HUB_OFFLINE": "1", "PYTHONUNBUFFERED": "1"}}
    write_json(output / "launch.json", launch)
    with (output / "native.log").open("w") as log:
        child = subprocess.Popen(command, cwd=project, env=environment,
                                 stdin=subprocess.DEVNULL, stdout=log,
                                 stderr=subprocess.STDOUT)
        launch["pid"] = child.pid
        write_json(output / "launch.json", launch)
        print(json.dumps(launch, ensure_ascii=False), flush=True)
        returncode = child.wait()
    exit_record = {"finished_utc": stamp(), "pid": child.pid,
                   "exit_code": returncode, "output_directory": str(output)}
    write_json(output / "exit.json", exit_record)
    print(json.dumps(exit_record), flush=True)
    return returncode


if __name__ == "__main__":
    raise SystemExit(main())
