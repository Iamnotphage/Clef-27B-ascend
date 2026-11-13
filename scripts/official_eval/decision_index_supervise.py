#!/usr/bin/env python3
"""Gate the native DI run on WorkflowEvals completion and record its exit."""

import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


ROOT = Path("/workspace/yangchen/logs/official-baseline-20261113/decision-index")
WORKFLOW_STATUS = Path("/workspace/yangchen/logs/official-baseline-20261113/workflowevals/clef-tp2-4096-one-20261113-status.json")
PROJECT = Path("/workspace/yangchen/Clef-27B-ascend")
WRAPPER = PROJECT / "scripts/official_eval/decision_index_baseline.sh"
EXPECTED_WORKFLOWS = {
    "invoice_processing", "customer_service", "security_incidents",
    "agent_trace_observability",
}


def stamp():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    with (ROOT / "supervisor.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while True:
            workflow = json.loads(WORKFLOW_STATUS.read_text())
            statuses = {item["workflow"]: item["status"] for item in workflow["workflows"]}
            ready = (workflow["status"] == "complete"
                     and set(statuses) == EXPECTED_WORKFLOWS
                     and all(value == "complete" for value in statuses.values()))
            gate = {"checked_at_utc": stamp(), "supervisor_pid": os.getpid(),
                    "workflow_status_file": str(WORKFLOW_STATUS),
                    "workflow_status": workflow["status"], "workflows": statuses,
                    "ready": ready}
            write_json(ROOT / "workflow-gate.json", gate)
            print(json.dumps({"event": "workflow_gate", **gate}), flush=True)
            if ready:
                break
            time.sleep(45)
        environment = dict(os.environ, HF_HUB_OFFLINE="1", PYTHONUNBUFFERED="1")
        command = ["bash", str(WRAPPER)]
        with (ROOT / "full-native.log").open("a") as log:
            child = subprocess.Popen(command, cwd=PROJECT, env=environment,
                                     stdin=subprocess.DEVNULL, stdout=log,
                                     stderr=subprocess.STDOUT)
            launch = {"started_at_utc": stamp(), "pid": child.pid,
                      "supervisor_pid": os.getpid(), "command": command,
                      "working_directory": str(PROJECT),
                      "wrapper_sha256": hashlib.sha256(WRAPPER.read_bytes()).hexdigest(),
                      "setup_report": str(ROOT / "setup.json"),
                      "suite_verification": str(ROOT / "suite-verification.json"),
                      "environment_overrides": {"HF_HUB_OFFLINE": "1", "PYTHONUNBUFFERED": "1"},
                      "workflow_gate": gate}
            write_json(ROOT / "launch.json", launch)
            (ROOT / "full.pid").write_text(str(child.pid) + "\n")
            print(json.dumps({"event": "decision_index_started", **launch}), flush=True)
            returncode = child.wait()
        exit_record = {"finished_at_utc": stamp(), "pid": child.pid,
                       "exit_code": returncode, "native_output": str(ROOT / "full"),
                       "native_log": str(ROOT / "full-native.log")}
        write_json(ROOT / "exit.json", exit_record)
        print(json.dumps({"event": "decision_index_exited", **exit_record}), flush=True)
        return returncode


if __name__ == "__main__":
    raise SystemExit(main())
