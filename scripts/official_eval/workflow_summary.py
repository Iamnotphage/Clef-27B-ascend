#!/usr/bin/env python3
"""Summarize completed native WorkflowEvals results and the matching HTTP traces."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

EXPECTED_CASES = {"invoice_processing": 150, "customer_service": 204, "security_incidents": 240, "agent_trace_observability": 111}


def percentile(values, fraction):
    if not values:
        return None
    values = sorted(values)
    position = (len(values) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def sha256(path):
    result = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            result.update(chunk)
    return result.hexdigest()


def summarize(root, name):
    report = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "run_name": name,
              "question_mode": "one", "workers": 1, "question_workers": 1,
              "reference": "consensus", "workflows": [],
              "input_token_saturation_note": "input_tokens == 4096 is saturation evidence, not an exact truncation count"}
    for workflow, expected_count in EXPECTED_CASES.items():
        folder = root / "runs" / workflow / name
        result_path, score_path = folder / "results.json", folder / "scores.json"
        bundle, scores = json.loads(result_path.read_text()), json.loads(score_path.read_text())
        run = bundle["runs"][0]
        cases = run["cases"]
        case_ids = [case["case_id"] for case in cases]
        expected_ids = {case["case_id"] for case in bundle["cases"]}
        assert len(case_ids) == expected_count and set(case_ids) == expected_ids and len(set(case_ids)) == len(case_ids), workflow
        assert run["configuration"]["model"]["question_mode"] == "one", workflow
        execution_status = Counter(case["execution"]["status"] for case in cases)
        decision_status = Counter(decision["status"] for case in cases for decision in case["decisions"])
        calls = [call for case in cases for call in case["calls"]]
        trace_path = root / "http" / f"{workflow}-{name}.jsonl"
        http_status, errors = Counter(), []
        latency, input_tokens = [], []
        http_count = questions = output_tokens = saturated = 0
        with trace_path.open() as source:
            for line in source:
                row = json.loads(line)
                http_count += 1
                questions += len(row["request"]["questions"])
                http_status[str(row.get("http_status", "connection_error"))] += 1
                latency.append(row["wall_time_s"])
                usage = row.get("response", {}).get("usage", {})
                value = usage.get("input_tokens_total", usage.get("input_tokens"))
                if value is not None:
                    input_tokens.append(value)
                    saturated += value == 4096
                output_tokens += usage.get("output_tokens_total", usage.get("output_tokens", 0)) or 0
                if row.get("error"):
                    errors.append({"request_index": http_count, "error": row["error"], "http_status": row.get("http_status")})
        assert http_count == len(calls), (workflow, http_count, len(calls))
        assert questions == http_count, (workflow, questions, http_count)
        consensus = [row for row in scores["series"] if row["run_id"] == name and row["label_set_id"] == "labels-consensus"]
        assert len(consensus) == 1, workflow
        consensus = consensus[0]
        native_rows = [row for row in scores["scores"] if row["run_id"] == name and row["label_set_id"] == "labels-consensus"]
        expected_decisions = expected_count * len(scores["policy_ids"])
        assert len(native_rows) == expected_decisions, workflow
        assert consensus["cases"] == expected_count, workflow
        for metric, denominator in consensus["metric_denominators"].items():
            eligible = [row["metrics"][metric] for row in native_rows if metric in row["metrics"]]
            assert len(eligible) == denominator, (workflow, metric)
            assert abs(sum(eligible) / denominator - consensus["metrics"][metric]) < 1e-12, (workflow, metric)
        report["workflows"].append({
            "workflow": workflow, "case_count": len(cases), "expected_cases": expected_count,
            "all_cases_present": True, "policy_ids": scores["policy_ids"], "expected_policy_decisions": expected_decisions,
            "execution_status_counts": dict(execution_status), "decision_status_counts": dict(decision_status),
            "native_consensus_series": consensus, "native_consensus_score_status_counts": dict(Counter(row["status"] for row in native_rows)),
            "source_dataset": bundle["dataset_source"], "dataset_revision_hash": bundle["dataset_revision"],
            "http_requests": http_count, "questions": questions, "http_status_counts": dict(http_status), "errors": errors,
            "input_tokens": {"count": len(input_tokens), "minimum": min(input_tokens) if input_tokens else None,
                             "maximum": max(input_tokens) if input_tokens else None, "sum": sum(input_tokens), "saturated_at_4096": saturated},
            "output_tokens": output_tokens,
            "http_latency_seconds": {"mean": sum(latency) / len(latency), "p50": percentile(latency, .5), "p95": percentile(latency, .95), "sum": sum(latency)},
            "files": {"results": str(result_path), "scores": str(score_path), "http_trace": str(trace_path)},
            "file_sha256": {"results": sha256(result_path), "scores": sha256(score_path), "http_trace": sha256(trace_path)},
        })
    report["evaluation_completed"] = True
    report["case_count"] = sum(row["case_count"] for row in report["workflows"])
    report["http_requests"] = sum(row["http_requests"] for row in report["workflows"])
    report["questions"] = sum(row["questions"] for row in report["workflows"])
    report["saturated_at_4096"] = sum(row["input_tokens"]["saturated_at_4096"] for row in report["workflows"])
    report["all_case_executions_ok"] = all(set(row["execution_status_counts"]) == {"ok"} for row in report["workflows"])
    report["all_http_requests_ok"] = all(row["http_status_counts"] == {"200": row["http_requests"]} and not row["errors"] for row in report["workflows"])
    report["total_policy_decisions"] = sum(row["expected_policy_decisions"] for row in report["workflows"])
    status_path = root / (name + "-status.json")
    status = json.loads(status_path.read_text())
    assert status["status"] in ("complete", "completed_with_errors"), status["status"]
    preflight_path = root / "preflight.json"
    preflight = json.loads(preflight_path.read_text())
    checkout = Path(status["workflows"][0]["command"][1]).parent
    source_checks = []
    for item in preflight["business_and_scoring_source_files"]:
        actual = sha256(checkout / item["path"])
        assert actual == item["upstream_sha256"], item["path"]
        source_checks.append({"path": item["path"], "sha256": actual, "matches_upstream": True})
    environment_path = root / "environment.json"
    environment = json.loads(environment_path.read_text())
    report.update({
        "started_at_utc": status["started_at_utc"], "finished_at_utc": status["finished_at_utc"],
        "runner_elapsed_seconds": (datetime.fromisoformat(status["finished_at_utc"]) - datetime.fromisoformat(status["started_at_utc"])).total_seconds(),
        "upstream_commit": status["workflowevals_commit"], "adapter_patch_sha256": status["adapter_patch_sha256"],
        "source_manifest_sha256": status["source_manifest_sha256"], "service_base_url": status["base_url"],
        "official_workflow_and_scoring_unchanged": True, "postrun_upstream_source_verification": source_checks,
        "client_environment": {"python_version": environment["python"]["version"], "uv_lock_sha256": environment["uv_lock_sha256"],
                               "wheel_count": environment["wheel_count"], "all_wheel_hashes_match_upstream_uv_lock": environment["all_wheel_hashes_match_upstream_uv_lock"]},
        "environment_manifest": str(environment_path), "preflight_report": str(preflight_path), "status_report": str(status_path),
        "native_commands": {item["workflow"]: item["command"] for item in status["workflows"]},
    })
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/workspace/yangchen/logs/official-baseline-20261113/workflowevals"))
    parser.add_argument("--name", default="clef-tp2-4096-one-20261113")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = summarize(args.root, args.name)
    output = args.output or args.root / "summary.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ("evaluation_completed", "case_count", "http_requests", "questions", "total_policy_decisions", "all_case_executions_ok", "all_http_requests_ok", "saturated_at_4096")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
