#!/usr/bin/env python3
"""Run pinned native WorkflowEvals workflows against the existing Clef HTTP service."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

WORKFLOWS = {
    "invoice_processing": "evalsafe-invoice-processing",
    "customer_service": "evalsafe-customer-service",
    "security_incidents": "evalsafe-security-incidents",
    "agent_trace_observability": "evalsafe-agent-trace-observability",
}
DEFAULT_SHA = "0ac3b8ad845429f0d8e064ecfb2430a47c5a25cb"


def now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkout", type=Path, default=Path("/workspace/yangchen/work/src/workflowevals-" + DEFAULT_SHA))
    parser.add_argument("--python", default="/workspace/yangchen/venvs/workflowevals-py313/bin/python")
    parser.add_argument("--data-root", type=Path, default=Path("/workspace/yangchen/datasets/clef-official-20261113/workflowevals"))
    parser.add_argument("--source-manifest", type=Path, default=Path("/workspace/yangchen/logs/official-datasets-download-20261113.json"))
    parser.add_argument("--output-root", type=Path, default=Path("/workspace/yangchen/logs/official-baseline-20261113/workflowevals"))
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--name", default="clef-tp2-4096-20261113")
    parser.add_argument("--workflows", nargs="+", choices=WORKFLOWS, default=list(WORKFLOWS))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--questions", choices=("one", "all"), default="one")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "http").mkdir(exist_ok=True)
    run_root = args.output_root / "runs"
    run_root.mkdir(exist_ok=True)
    native_runs = args.checkout / "runs"
    if not native_runs.exists():
        native_runs.symlink_to(run_root, target_is_directory=True)
    if native_runs.resolve() != run_root.resolve():
        raise SystemExit("Checkout runs path must point to the selected output root")
    source = json.loads(args.source_manifest.read_text())
    revisions = {item["repo_id"].rsplit("/", 1)[-1]: item["revision"] for item in source["workflowevals"]["source_revisions"]}
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=args.checkout, text=True).strip()
    if sha != DEFAULT_SHA:
        raise SystemExit(f"Unexpected upstream checkout: {sha}")
    diff = subprocess.check_output(["git", "diff", "--", "core"], cwd=args.checkout)
    diff_path = args.output_root / "workflowevals-clef-http-offline.patch"
    diff_path.write_bytes(diff)
    status_path = args.output_root / (args.name + "-status.json")
    status = {
        "status": "running", "started_at_utc": now(), "workflowevals_commit": sha,
        "adapter_patch_sha256": hashlib.sha256(diff).hexdigest(),
        "base_url": args.base_url, "model": "clef", "question_mode": args.questions, "workers": 1, "question_workers": 1,
        "source_manifest_sha256": hashlib.sha256(args.source_manifest.read_bytes()).hexdigest(),
        "data_root": str(args.data_root), "output_root": str(args.output_root),
        "limit": args.limit, "dry_run": args.dry_run, "workflows": [],
    }
    write_json(status_path, status)
    failed = False
    for workflow in args.workflows:
        trace = args.output_root / "http" / f"{workflow}-{args.name}.jsonl"
        command = [args.python, str(args.checkout / "run.py"), workflow, "--model", "local:clef", "--base-url", args.base_url,
                   "--questions", args.questions, "--workers", "1", "--question-workers", "1", "--dataset-revision", revisions[WORKFLOWS[workflow]], "--name", args.name]
        if args.limit:
            command += ["--limit", str(args.limit)]
        if args.dry_run:
            command.append("--dry-run")
        env = os.environ.copy()
        env.update({"WORKFLOWEVALS_DATA_ROOT": str(args.data_root), "WORKFLOWEVALS_SOURCE_MANIFEST": str(args.source_manifest),
                    "WORKFLOWEVALS_HTTP_TRACE": str(trace), "HF_HUB_OFFLINE": "1", "PYTHONUNBUFFERED": "1"})
        log = args.output_root / f"{workflow}-{args.name}.log"
        item = {"workflow": workflow, "dataset_revision": revisions[WORKFLOWS[workflow]], "started_at_utc": now(),
                "command": command, "trace": str(trace), "log": str(log), "status": "running"}
        status["workflows"].append(item)
        write_json(status_path, status)
        print(json.dumps({"starting": workflow, "log": str(log)}), flush=True)
        with log.open("a") as output:
            result = subprocess.run(command, cwd=args.checkout, env=env, stdout=output, stderr=subprocess.STDOUT)
        item.update({"finished_at_utc": now(), "exit_code": result.returncode, "status": "complete" if result.returncode == 0 else "failed"})
        failed |= result.returncode != 0
        write_json(status_path, status)
        print(json.dumps({"finished": workflow, "exit_code": result.returncode}), flush=True)
    status.update({"finished_at_utc": now(), "status": "completed_with_errors" if failed else "complete"})
    write_json(status_path, status)
    return int(failed)


if __name__ == "__main__":
    sys.exit(main())
