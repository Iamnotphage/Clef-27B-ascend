#!/usr/bin/env python3
"""Summarize saved SystemOne benchmark reports without running inference.

Example:
  python scripts/summarize_systemone_benchmarks.py \
      --report tp1-1024=/absolute/round1.json \
      --report tp1-1024=/absolute/round2.json \
      --report tp1-1024=/absolute/round3.json --output-dir /absolute/summary \
      --require-stable-baseline

Use --evidence-kind synthetic for generated integration-test reports. Sampling
eligibility checks cannot establish machine identity, profiler state, source or
patch equality; those checks additionally require the run manifest and logs.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import statistics


FIXED_REQUEST_SHA256 = "ab2724637a36029759ea673964b59f0c30fc2e27af4b7d7fa3bdcd784de33aef"
SYNTHETIC_WORD = re.compile(r"(?:^|[-_.:/ ])(?:synthetic|fixture|test)(?:$|[-_.:/ ])", re.IGNORECASE)


def numeric(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def percentile(values, quantile):
    if not values:
        return None
    values = sorted(values)
    position = (len(values) - 1) * quantile
    lower, upper = math.floor(position), math.ceil(position)
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def distribution(values):
    mean = statistics.mean(values) if values else None
    stdev = statistics.stdev(values) if len(values) > 1 else (0.0 if values else None)
    return {
        "sample_count": len(values), "mean": mean,
        "min": min(values) if values else None, "max": max(values) if values else None,
        "range": max(values) - min(values) if values else None,
        "relative_range": (max(values) - min(values)) / mean if mean else None,
        "p50": percentile(values, 0.5), "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99), "sample_standard_deviation": stdev,
        "coefficient_of_variation": stdev / mean if mean else None,
        "percentile_method": "linear interpolation at (n-1)*quantile",
    }


def digest_json(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def synthetic_reasons(report, label, path, requested_kind):
    reasons = []
    if requested_kind == "synthetic":
        reasons.append("CLI evidence kind is synthetic")
    if SYNTHETIC_WORD.search(label) or SYNTHETIC_WORD.search(str(path)):
        reasons.append("Label or source path identifies synthetic/test/fixture evidence")
    for source in (report, report.get("provenance", {}), report.get("metadata", {}), report.get("sample_source", {})):
        if not isinstance(source, dict):
            continue
        if any(source.get(key) is True for key in ("synthetic", "is_synthetic", "fixture", "test_only")):
            reasons.append("Report explicitly marks synthetic/test/fixture evidence")
        if any(isinstance(source.get(key), str) and SYNTHETIC_WORD.search(source[key]) for key in ("kind", "evidence_kind", "source", "mode")):
            reasons.append("Report provenance identifies synthetic/test/fixture evidence")
    return sorted(set(reasons))


def parse_report_argument(value):
    label, separator, filename = value.partition("=")
    if not separator or not label.strip() or not filename:
        raise ValueError("--report needs CONFIG_LABEL=/absolute/report.json")
    path = Path(filename).expanduser()
    if not path.is_absolute():
        raise ValueError("--report paths must be absolute")
    return label.strip(), path


def read_round(label, path, evidence_kind):
    raw = path.read_bytes()

    def reject_nonfinite(value):
        raise ValueError(f"Nonfinite JSON value: {value}")

    def finite_float(value):
        result = float(value)
        if not math.isfinite(result):
            reject_nonfinite(value)
        return result

    report = json.loads(raw, parse_constant=reject_nonfinite, parse_float=finite_float)
    if not isinstance(report, dict):
        raise ValueError(f"Report must be an object: {path}")
    attempts = report.get("requests")
    configuration = report.get("configuration", {})
    warmups = report.get("warmup", [])
    if not isinstance(attempts, list) or not all(isinstance(row, dict) for row in attempts):
        raise ValueError(f"requests must be an array of objects: {path}")
    if not isinstance(configuration, dict) or not isinstance(warmups, list) or not all(isinstance(row, dict) for row in warmups):
        raise ValueError(f"Invalid configuration/warmup structure: {path}")
    saved_summary = report.get("summary", {})
    if not isinstance(saved_summary, dict):
        raise ValueError(f"summary must be an object: {path}")
    wall_seconds = saved_summary.get("measurement_wall_seconds")
    wall_seconds = wall_seconds if numeric(wall_seconds) and wall_seconds > 0 else None
    successful = [row for row in attempts if row.get("passed") is True]
    latencies = [row["elapsed_ms"] for row in successful if numeric(row.get("elapsed_ms")) and row["elapsed_ms"] >= 0]
    tokens, fields, candidates, request_hashes = [], [], [], []
    for row in successful:
        count = row.get("actual_input_tokens")
        if count is None and isinstance(row.get("response"), dict):
            usage = row["response"].get("usage", {})
            count = usage.get("input_tokens") if isinstance(usage, dict) else None
        if isinstance(count, int) and not isinstance(count, bool) and count > 0:
            tokens.append(count)
        request = row.get("request")
        questions = request.get("questions", {}) if isinstance(request, dict) else {}
        field_count = row.get("field_count", len(questions) if questions else report.get("fields_per_request"))
        if isinstance(field_count, int) and not isinstance(field_count, bool) and field_count > 0:
            fields.append(field_count)
        counts = row.get("candidate_counts")
        if counts is None and isinstance(questions, dict) and questions:
            try:
                counts = {qid: 2 if question["type"] == "noul" else len(question["criteria"]) for qid, question in questions.items()}
            except (KeyError, TypeError):
                counts = None
        if isinstance(counts, dict):
            candidates.append(counts)
        if isinstance(request, dict):
            request_hashes.append(digest_json(request))
        elif isinstance(row.get("request_sha256"), str):
            request_hashes.append(row["request_sha256"])
    stages = {
        key: distribution([row[key] for row in successful if numeric(row.get(key)) and row[key] >= 0])
        for key in ("http_ms", "json_decode_ms", "protocol_validation_ms", "client_queue_ms", "client_total_ms")
    }
    peak_concurrency = saved_summary.get("observed_peak_http_concurrency")
    peak_concurrency = peak_concurrency if isinstance(peak_concurrency, int) and not isinstance(peak_concurrency, bool) else None
    issues = []
    if len(attempts) < 300:
        issues.append("Fewer than 300 measured requests")
    if not numeric(configuration.get("warmup")) or configuration["warmup"] < 10:
        issues.append("Configured per-shape warmup is below 10 or missing")
    if len(warmups) < 10 or any(row.get("passed") is not True for row in warmups):
        issues.append("Warmup evidence needs at least 10 successful requests")
    warmup_hashes = [digest_json(row["request"]) if isinstance(row.get("request"), dict) else row.get("request_sha256") for row in warmups]
    if not warmup_hashes or set(warmup_hashes) != {FIXED_REQUEST_SHA256}:
        issues.append("Warmup requests must match the fixed measured request shape")
    if wall_seconds is None or wall_seconds < 30:
        issues.append("Measured window is below 30 seconds or missing")
    if report.get("passed") is not True or not attempts or len(successful) != len(attempts):
        issues.append("Report and all measured requests must pass")
    if any(row.get("http_status") != 200 for row in attempts):
        issues.append("Every measured request needs HTTP 200 evidence")
    if len(latencies) != len(successful) or len(tokens) != len(successful) or len(fields) != len(successful):
        issues.append("Successful request latency/token/field evidence is incomplete")
    if configuration.get("concurrency") != 1:
        issues.append("Fixed baseline requires client concurrency 1")
    if set(request_hashes) != {FIXED_REQUEST_SHA256} or len(request_hashes) != len(attempts):
        issues.append("Fixed baseline requires the exact historical mixed POST body")
    if set(tokens) != {307} or set(fields) != {3}:
        issues.append("Fixed baseline requires 307 input tokens and three fields")
    expected_candidates = {"department": 2, "refund_requested": 2, "severity": 3}
    if len(candidates) != len(attempts) or any(value != expected_candidates for value in candidates):
        issues.append("Fixed baseline requires candidate counts 2/2/3")
    synthetic = synthetic_reasons(report, label, path, evidence_kind)
    issues.extend(synthetic)
    count = len(successful)
    row = {
        "configuration_label": label, "source_report": str(path.resolve()),
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "run_id": report.get("run_id"), "model": report.get("model"),
        "started_at": report.get("started_at"), "finished_at": report.get("finished_at"),
        "evidence_kind": "synthetic" if synthetic else "unverified_benchmark_report",
        "configuration": configuration,
        "service_configuration": report.get("service_configuration"),
        "sample_source": report.get("sample_source"),
        "request_template_sha256": report.get("request_template_sha256"),
        "actual_request_sha256": sorted(set(request_hashes)),
        "requested_requests": configuration.get("requests"),
        "actual_samples": len(attempts), "successful_requests": count,
        "warmup_samples": len(warmups), "measurement_wall_seconds": wall_seconds,
        "success_rate": count / len(attempts) if attempts else None,
        "report_passed": report.get("passed") is True,
        "requests_per_second": count / wall_seconds if wall_seconds else None,
        "decisions_per_second": sum(fields) / wall_seconds if wall_seconds and len(fields) == count else None,
        "input_tokens_per_second": sum(tokens) / wall_seconds if wall_seconds and len(tokens) == count else None,
        "successful_decisions": sum(fields), "successful_input_tokens": sum(tokens),
        "input_token_counts": sorted(set(tokens)), "field_counts": sorted(set(fields)),
        "successful_request_latency_ms": distribution(latencies),
        "successful_stage_ms": stages, "observed_peak_http_concurrency": peak_concurrency,
        "measurement_protocol_passed": not issues, "ineligibility_reasons": issues,
        "saved_summary": saved_summary,
    }
    private = {"latencies": latencies, "stages": {key: [request[key] for request in successful if numeric(request.get(key)) and request[key] >= 0] for key in stages}}
    return row, private


def summarize_groups(rounds, observations):
    labels = dict.fromkeys(row["configuration_label"] for row in rounds)
    groups = {}
    for label in labels:
        indexed = [(row, observed) for row, observed in zip(rounds, observations) if row["configuration_label"] == label]
        rows = [row for row, _ in indexed]
        hashes = [row["source_sha256"] for row in rows]
        issues = []
        if len(rows) < 3:
            issues.append("At least three rounds are required")
        if len(set(hashes)) != len(hashes):
            issues.append("Repeated source SHA-256 values cannot establish independent rounds")
        run_ids = [row["run_id"] for row in rows if isinstance(row["run_id"], str) and row["run_id"]]
        if len(set(run_ids)) != len(run_ids):
            issues.append("Repeated run_id values cannot establish independent rounds")
        legacy_intervals = []
        for row in rows:
            if isinstance(row["run_id"], str) and row["run_id"]:
                continue
            try:
                start = datetime.fromisoformat(row["started_at"])
                end = datetime.fromisoformat(row["finished_at"])
                if start.tzinfo is None or end.tzinfo is None or end <= start:
                    raise ValueError("Invalid interval")
                legacy_intervals.append((start, end))
            except (ValueError, TypeError):
                issues.append("Legacy reports without run_id require valid start/end timestamps to establish independent windows")
        legacy_intervals.sort()
        if any(current[0] < previous[1] for previous, current in zip(legacy_intervals, legacy_intervals[1:])):
            issues.append("Legacy report time windows overlap; independent rounds are unverified")
        if any(not row["measurement_protocol_passed"] for row in rows):
            issues.append("At least one round fails fixed-baseline requirements")
        signatures = {
            digest_json({
                "model": row.get("model"),
                "request_hashes": row["actual_request_sha256"],
                "concurrency": row["configuration"].get("concurrency"),
                "validation_atol": row["configuration"].get("validation_atol"),
                "latency_scope": row["configuration"].get("latency_scope"),
                "service_configuration": row["service_configuration"],
            }) for row in rows
        }
        if len(signatures) > 1:
            issues.append("Input, client measurement settings or supplied service configuration differ between rounds")
        rps = [row["requests_per_second"] for row in rows if row["requests_per_second"] is not None]
        total_wall = sum(row["measurement_wall_seconds"] or 0 for row in rows)
        latencies = [value for _, observed in indexed for value in observed["latencies"]]
        groups[label] = {
            "round_count": len(rows), "distinct_source_report_count": len(set(hashes)),
            "source_sha256": hashes, "run_ids": run_ids, "measurement_protocol_passed": not issues,
            "ineligibility_reasons": issues,
            "round_requests_per_second": distribution(rps),
            "total_measured_samples": sum(row["actual_samples"] for row in rows),
            "total_successful_requests": sum(row["successful_requests"] for row in rows),
            "total_measurement_wall_seconds": total_wall,
            "pooled_successful_requests_per_second": sum(row["successful_requests"] for row in rows) / total_wall if total_wall else None,
            "pooled_request_latency_ms": distribution(latencies),
            "pooled_stage_ms": {key: distribution([value for _, observed in indexed for value in observed["stages"][key]]) for key in observations[0]["stages"]},
            "aggregation_scope": "Latency percentiles pool successful individual requests, rather than averaging per-round percentiles; round throughput statistics weight rounds equally",
        }
    return groups


CSV_FIELDS = [
    "configuration_label", "round_number", "source_report", "source_sha256", "run_id", "started_at", "finished_at", "evidence_kind",
    "configuration", "service_configuration", "actual_samples", "successful_requests", "warmup_samples",
    "measurement_wall_seconds", "success_rate", "requests_per_second", "decisions_per_second", "input_tokens_per_second",
    "p50_ms", "p95_ms", "p99_ms", "latency_sample_count", "latency_mean_ms", "latency_sample_stdev_ms", "latency_cv",
    "observed_peak_http_concurrency", "client_queue_p50_ms", "client_queue_p95_ms", "client_queue_mean_ms",
    "http_mean_ms", "json_decode_mean_ms", "protocol_validation_mean_ms",
    "measurement_protocol_passed", "ineligibility_reasons",
]


def csv_round(row):
    value = {key: row.get(key) for key in CSV_FIELDS}
    latency = row["successful_request_latency_ms"]
    stages = row["successful_stage_ms"]
    value.update({
        "p50_ms": latency["p50"], "p95_ms": latency["p95"], "p99_ms": latency["p99"],
        "latency_sample_count": latency["sample_count"], "latency_mean_ms": latency["mean"],
        "latency_sample_stdev_ms": latency["sample_standard_deviation"], "latency_cv": latency["coefficient_of_variation"],
        "client_queue_p50_ms": stages["client_queue_ms"]["p50"], "client_queue_p95_ms": stages["client_queue_ms"]["p95"],
        "client_queue_mean_ms": stages["client_queue_ms"]["mean"], "http_mean_ms": stages["http_ms"]["mean"],
        "json_decode_mean_ms": stages["json_decode_ms"]["mean"], "protocol_validation_mean_ms": stages["protocol_validation_ms"]["mean"],
    })
    return {key: json.dumps(item, ensure_ascii=False, separators=(",", ":"), allow_nan=False) if isinstance(item, (dict, list)) else item for key, item in value.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--report", action="append", required=True, help="Repeat CONFIG_LABEL=/absolute/report.json for each round")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--evidence-kind", choices=("measured", "synthetic"), default="measured")
    parser.add_argument("--require-stable-baseline", action="store_true")
    args = parser.parse_args()
    rounds, observations, counts = [], [], {}
    try:
        for argument in args.report:
            label, path = parse_report_argument(argument)
            row, observed = read_round(label, path, args.evidence_kind)
            counts[label] = counts.get(label, 0) + 1
            row["round_number"] = counts[label]
            rounds.append(row)
            observations.append(observed)
        groups = summarize_groups(rounds, observations)
    except (OSError, ValueError, TypeError, KeyError) as exc:
        parser.error(str(exc))
    summary = {
        "schema_version": 1, "created_at": datetime.now(timezone.utc).isoformat(),
        "scope": "Saved benchmark file aggregation; no inference is executed",
        "evidence_scope": "Statistics and sampling-protocol checks derived solely from supplied benchmark JSON files",
        "hardware_execution_verified": None,
        "hardware_execution_verification": "Independent run manifest and raw machine logs are required to verify hardware execution, source/patch equality, server configuration and profiler-off state; the CLI evidence kind does not verify these facts",
        "evidence_kind_requested": args.evidence_kind,
        "fixed_baseline_criteria": {
            "minimum_distinct_rounds": 3, "minimum_measured_requests_per_round": 300,
            "minimum_warmup_requests_per_round": 10, "minimum_window_seconds_per_round": 30,
            "all_warmup_and_measured_requests_pass": True,
            "request_sha256": FIXED_REQUEST_SHA256, "input_tokens": 307,
            "fields": 3, "candidate_counts": [2, 2, 3], "client_concurrency": 1,
            "synthetic_reports_eligible": False,
            "stability_cv_acceptance_limit": None,
            "stability_note": "Round CV and range are reported; no unstated CV cutoff is applied",
        },
        "limits": [
            "Sampling eligibility requires separate run-manifest verification of sources, patches, server configuration, hardware and profiler-off state",
            "Client-only HTTP timing does not isolate server queueing",
            "Synthetic classification relies on explicit evidence markers, labels, paths and the CLI evidence kind",
        ],
        "all_groups_measurement_protocol_passed": bool(groups) and all(group["measurement_protocol_passed"] for group in groups.values()),
        "rounds": rounds, "groups": groups,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path, csv_path = args.output_dir / "summary.json", args.output_dir / "performance-matrix.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(csv_round(row) for row in rounds)
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"summary": str(json_path.resolve()), "performance_matrix": str(csv_path.resolve()), "all_groups_measurement_protocol_passed": summary["all_groups_measurement_protocol_passed"], "groups": {label: {"round_count": group["round_count"], "measurement_protocol_passed": group["measurement_protocol_passed"], "ineligibility_reasons": group["ineligibility_reasons"]} for label, group in groups.items()}}, ensure_ascii=False))
    return 1 if args.require_stable_baseline and not summary["all_groups_measurement_protocol_passed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
