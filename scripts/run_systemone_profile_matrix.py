#!/usr/bin/env python3
"""Run a sequential extension matrix from prepared, actually encoded requests.

Example:
  python scripts/run_systemone_profile_matrix.py --plan /absolute/workloads/matrix-plan.json \
      --output-dir /absolute/matrix-tp1 --max-model-len 4096 --groups controlled_length field_count

Each sample gets its own benchmark subprocess, log, original JSON report and
before/after Prometheus snapshots. These extension measurements have one round
per shape. They retain their actual sample count/window and do not establish the
three-round fixed-307-token stable baseline. The caller supplies server settings;
hardware execution and configuration additionally require the run manifest.
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
import subprocess
import time
from urllib.error import HTTPError, URLError
from urllib.request import urlopen


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def save(path, document):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(document, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def percentile(values, quantile):
    if not values:
        return None
    values = sorted(values)
    position = (len(values) - 1) * quantile
    low, high = math.floor(position), math.ceil(position)
    return values[low] + (values[high] - values[low]) * (position - low)


def distribution(values):
    mean = statistics.mean(values) if values else None
    stdev = statistics.stdev(values) if len(values) > 1 else (0.0 if values else None)
    return {"sample_count": len(values), "p50": percentile(values, .5), "p95": percentile(values, .95), "p99": percentile(values, .99), "mean": mean, "min": min(values) if values else None, "max": max(values) if values else None, "sample_standard_deviation": stdev, "coefficient_of_variation": stdev / mean if mean else None, "percentile_method": "linear interpolation at (n-1)*quantile"}


def metrics_snapshot(base_url, directory, phase, timeout):
    directory.mkdir(parents=True, exist_ok=True)
    result = {"started_at": timestamp(), "url": base_url.rstrip("/") + "/metrics", "passed": False}
    started = time.perf_counter()
    raw = b""
    try:
        with urlopen(result["url"], timeout=timeout) as response:
            result["http_status"] = response.status
            raw = response.read()
        result["passed"] = result["http_status"] == 200 and bool(raw)
    except HTTPError as exc:
        raw = exc.read()
        result.update(http_status=exc.code, error=f"HTTPError: {exc}")
    except (OSError, URLError) as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    path = directory / ("metrics-" + phase + ".prom")
    path.write_bytes(raw)
    result.update(path=str(path.resolve()), sha256=sha256(raw), bytes=len(raw), elapsed_seconds=time.perf_counter() - started, finished_at=timestamp())
    save(directory / ("metrics-" + phase + ".json"), result)
    return result


def select_samples(plan_path, groups=None, names=None, max_model_len=4096):
    raw = plan_path.read_bytes()
    plan = json.loads(raw)
    samples = plan.get("samples") if isinstance(plan, dict) else None
    if not isinstance(samples, list) or not samples:
        raise ValueError("Matrix plan needs a nonempty samples array")
    known_names, known_groups = set(), set()
    for sample in samples:
        if not isinstance(sample, dict) or not isinstance(sample.get("name"), str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", sample["name"]):
            raise ValueError("Each sample needs a safe, nonempty name")
        if sample["name"] in known_names:
            raise ValueError("Sample names must be unique")
        known_names.add(sample["name"])
        known_groups.add(sample.get("group"))
    if names and set(names) - known_names:
        raise ValueError(f"Unknown sample names: {sorted(set(names) - known_names)}")
    if groups and set(groups) - known_groups:
        raise ValueError(f"Unknown sample groups: {sorted(set(groups) - known_groups)}")
    selected = []
    for sample in samples:
        if groups and sample.get("group") not in groups or names and sample["name"] not in names:
            continue
        expected = sample.get("actual_input_tokens")
        if not isinstance(expected, int) or isinstance(expected, bool) or expected <= 0:
            raise ValueError(f"{sample['name']}: actual encoded input token count is required")
        if expected > max_model_len:
            raise ValueError(f"{sample['name']}: {expected} encoded tokens exceed --max-model-len={max_model_len}; select an explicit compatible subset")
        filename = sample.get("sample_file")
        if not isinstance(filename, str) or not filename:
            raise ValueError(f"{sample['name']}: missing sample_file")
        path = (plan_path.parent / filename).resolve()
        sample_bytes = path.read_bytes()
        document = json.loads(sample_bytes)
        rows = document.get("samples") if isinstance(document, dict) else None
        if not isinstance(rows, list) or len(rows) != 1 or not isinstance(rows[0].get("request"), dict):
            raise ValueError(f"{sample['name']}: sample file must contain exactly one complete request")
        request_hash = sha256(json.dumps(rows[0]["request"], ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode())
        if sample.get("request_sha256") != request_hash:
            raise ValueError(f"{sample['name']}: request SHA-256 differs from matrix plan")
        selected.append(dict(sample, sample_path=str(path), sample_file_sha256=sha256(sample_bytes)))
    if not selected:
        raise ValueError("The explicit group/name intersection selects no samples")
    return selected, {"path": str(plan_path.resolve()), "sha256": sha256(raw), "scope": plan.get("scope"), "recommended_measurement": plan.get("recommended_measurement"), "total_plan_samples": len(samples)}


def inspect_benchmark(report, expected_tokens, expected_request_sha256, requested):
    measured = report.get("requests", [])
    warmup = report.get("warmup", [])
    if not isinstance(measured, list) or not isinstance(warmup, list) or not all(isinstance(row, dict) for row in measured + warmup):
        raise ValueError("Benchmark warmup/requests must be arrays of objects")
    failures, token_counts = [], []
    for phase, attempts in (("warmup", warmup), ("measured", measured)):
        for index, row in enumerate(attempts):
            response = row.get("response")
            usage = response.get("usage") if isinstance(response, dict) else None
            actual = usage.get("input_tokens") if isinstance(usage, dict) else None
            if phase == "measured" and isinstance(actual, int) and not isinstance(actual, bool):
                token_counts.append(actual)
            issues = []
            if row.get("passed") is not True or row.get("http_status") != 200:
                issues.append("HTTP/protocol validation failed")
            if not isinstance(actual, int) or isinstance(actual, bool) or actual != expected_tokens:
                issues.append(f"API usage.input_tokens={actual!r}, expected encoded plan={expected_tokens}")
            if row.get("actual_input_tokens") != actual:
                issues.append("Saved client actual_input_tokens differs from API usage")
            if row.get("request_sha256") != expected_request_sha256:
                issues.append("Sent request SHA-256 differs from matrix plan")
            if issues:
                failures.append({"phase": phase, "index": row.get("index", index), "issues": issues, "error": row.get("error"), "validation_errors": row.get("errors")})
    if len(measured) != requested:
        failures.append({"phase": "measured", "issues": [f"Expected {requested} attempts; received {len(measured)}"]})
    successful = [row for row in measured if row.get("passed") is True]
    saved_summary = report.get("summary", {})
    wall = saved_summary.get("measurement_wall_seconds") if isinstance(saved_summary, dict) else None
    wall = wall if finite(wall) and wall > 0 else None
    if wall is None:
        failures.append({"phase": "measured", "issues": ["A positive actual measurement window is required"]})
    latency = distribution([row["elapsed_ms"] for row in successful if finite(row.get("elapsed_ms")) and row["elapsed_ms"] >= 0])
    stages = {key: distribution([row[key] for row in successful if finite(row.get(key)) and row[key] >= 0]) for key in ("http_ms", "json_decode_ms", "protocol_validation_ms", "client_queue_ms")}
    fields = sum(row.get("field_count", 0) for row in successful)
    successful_tokens = sum(row["response"]["usage"]["input_tokens"] for row in successful if isinstance(row.get("response"), dict) and isinstance(row["response"].get("usage"), dict) and isinstance(row["response"]["usage"].get("input_tokens"), int))
    return {
        "passed": report.get("passed") is True and not failures,
        "actual_samples": len(measured), "successful_requests": len(successful), "warmup_samples": len(warmup),
        "success_rate": len(successful) / len(measured) if measured else None,
        "measurement_wall_seconds": wall,
        "requests_per_second": len(successful) / wall if wall else None,
        "decisions_per_second": fields / wall if wall else None,
        "input_tokens_per_second": successful_tokens / wall if wall else None,
        "latency_ms": latency, "stages_ms": stages,
        "observed_peak_http_concurrency": saved_summary.get("observed_peak_http_concurrency") if isinstance(saved_summary, dict) else None,
        "api_input_token_counts": sorted(set(token_counts)),
        "expected_input_tokens": expected_tokens, "token_and_request_consistency_passed": not failures,
        "failures": failures, "original_summary": saved_summary,
        "below_300_requests": len(measured) < 300, "below_30_second_window": wall is None or wall < 30,
        "throughput_scope": "Protocol-passed measured responses; token/request consistency independently gates case.passed. Failed cases retain observed metrics for diagnosis",
    }


CSV_FIELDS = ["name", "group", "passed", "expected_input_tokens", "api_input_token_counts", "actual_samples", "warmup_samples", "measurement_wall_seconds", "success_rate", "requests_per_second", "decisions_per_second", "input_tokens_per_second", "p50_ms", "p95_ms", "p99_ms", "latency_mean_ms", "latency_stdev_ms", "latency_cv", "http_mean_ms", "json_decode_mean_ms", "protocol_validation_mean_ms", "client_queue_mean_ms", "observed_peak_http_concurrency", "token_and_request_consistency_passed", "below_300_requests", "below_30_second_window", "report_path", "report_sha256", "sample_file_sha256"]


def write_outputs(output, summary):
    summary["finished_at"] = timestamp()
    save(output / "matrix-summary.json", summary)
    with (output / "performance-matrix.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, lineterminator="\n")
        writer.writeheader()
        for case in summary["cases"]:
            value = {key: case.get(key) for key in CSV_FIELDS}
            latency, stages = case.get("latency_ms", {}), case.get("stages_ms", {})
            value.update({"p50_ms": latency.get("p50"), "p95_ms": latency.get("p95"), "p99_ms": latency.get("p99"), "latency_mean_ms": latency.get("mean"), "latency_stdev_ms": latency.get("sample_standard_deviation"), "latency_cv": latency.get("coefficient_of_variation"), **{key + "_mean_ms": stages.get(key + "_ms", {}).get("mean") for key in ("http", "json_decode", "protocol_validation", "client_queue")}})
            writer.writerow({key: json.dumps(item, ensure_ascii=False) if isinstance(item, (dict, list)) else item for key, item in value.items()})


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--plan", "--matrix-plan", dest="plan", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--groups", action="append", nargs="+")
    parser.add_argument("--names", action="append", nargs="+")
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--requests", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--metrics-timeout", type=float, default=30)
    parser.add_argument("--benchmark-script", type=Path, default=Path(__file__).with_name("benchmark_systemone.py"))
    parser.add_argument("--python", default=__import__("sys").executable)
    parser.add_argument("--run-manifest", type=Path, help="Independent source/config/hardware manifest, indexed by SHA only")
    args = parser.parse_args()
    if min(args.max_model_len, args.requests, args.concurrency) <= 0 or args.warmup < 0 or not all(finite(value) and value > 0 for value in (args.timeout, args.metrics_timeout)):
        parser.error("Positive counts/timeouts and nonnegative warmup are required")
    if (args.output_dir / "matrix-summary.json").exists():
        parser.error("Output already contains matrix-summary.json; use a new directory to preserve prior evidence")
    try:
        samples, provenance = select_samples(args.plan.resolve(), [value for group in args.groups or [] for value in group], [value for group in args.names or [] for value in group], args.max_model_len)
        benchmark_source = {"path": str(args.benchmark_script.resolve()), "sha256": sha256(args.benchmark_script.read_bytes())}
        run_manifest = {"path": str(args.run_manifest.resolve()), "sha256": sha256(args.run_manifest.read_bytes())} if args.run_manifest else None
    except (OSError, ValueError, TypeError, KeyError) as exc:
        parser.error(str(exc))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "schema_version": 1, "created_at": timestamp(), "passed": False, "status": "running", "cases": [],
        "evidence_scope": "Sequential HTTP extension workload matrix with original benchmark reports and adjacent metrics; hardware/source/server configuration verified separately by run manifest",
        "hardware_execution_verified": None, "run_manifest": run_manifest,
        "measurement_classification": "extension_matrix", "rounds_per_sample": 1, "stable_fixed_baseline_evidence": False,
        "scope_note": "One round per shape; per-case flags identify fewer than 300 requests or a window below 30 seconds. Three independent fixed-request rounds are required for the stable baseline",
        "plan": provenance, "benchmark_script": benchmark_source,
        "runner_script": {"path": str(Path(__file__).resolve()), "sha256": sha256(Path(__file__).read_bytes())},
        "configuration": {"base_url": args.base_url.rstrip("/"), "max_model_len_supplied_by_caller": args.max_model_len, "requests": args.requests, "warmup": args.warmup, "concurrency": args.concurrency, "timeout_seconds": args.timeout, "selected_groups": args.groups, "selected_names": args.names, "sample_hbm": True},
        "selected_sample_count": len(samples), "ordering": [sample["name"] for sample in samples],
        "latency_scope": "Original benchmark elapsed_ms: HTTP transfer, UTF-8/JSON decode and protocol validation, with client executor queue separately reported",
        "metrics_scope": "Before/after snapshots occur outside each benchmark measurement window; metrics can contain counters from earlier requests",
    }
    write_outputs(args.output_dir, summary)
    for sample in samples:
        directory = args.output_dir / sample["name"]
        directory.mkdir(parents=True, exist_ok=True)
        report_path, log_path = directory / "benchmark.json", directory / "benchmark.log"
        case = {"name": sample["name"], "group": sample.get("group"), "variable": sample.get("variable"), "controlled_variables": sample.get("controlled_variables"), "plan_sample": sample, "sample_file_sha256": sample["sample_file_sha256"], "passed": False, "started_at": timestamp(), "report_path": str(report_path.resolve()), "log_path": str(log_path.resolve())}
        print(json.dumps({"name": sample["name"], "status": "starting", "expected_input_tokens": sample["actual_input_tokens"]}), flush=True)
        case["metrics_before"] = metrics_snapshot(args.base_url, directory, "before", args.metrics_timeout)
        command = [args.python, str(args.benchmark_script.resolve()), "--base-url", args.base_url.rstrip("/"), "--sample-file", sample["sample_path"], "--warmup", str(args.warmup), "--requests", str(args.requests), "--concurrency", str(args.concurrency), "--timeout", str(args.timeout), "--sample-hbm", "--report", str(report_path.resolve())]
        case["command"] = command
        try:
            with log_path.open("w", encoding="utf-8") as log:
                completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
            case["returncode"] = completed.returncode
            report_bytes = report_path.read_bytes()
            case["report_sha256"] = sha256(report_bytes)
            case.update(inspect_benchmark(json.loads(report_bytes), sample["actual_input_tokens"], sample["request_sha256"], args.requests))
            case["passed"] = case["passed"] and completed.returncode == 0
        except (OSError, ValueError, TypeError, KeyError) as exc:
            case.update(passed=False, error=f"{type(exc).__name__}: {exc}")
        finally:
            case["metrics_after"] = metrics_snapshot(args.base_url, directory, "after", args.metrics_timeout)
            case["passed"] = case["passed"] and case["metrics_before"]["passed"] and case["metrics_after"]["passed"]
            if log_path.is_file():
                case["log_sha256"] = sha256(log_path.read_bytes())
            case["finished_at"] = timestamp()
            summary["cases"].append(case)
            write_outputs(args.output_dir, summary)
        print(json.dumps({"name": case["name"], "passed": case["passed"], "actual_samples": case.get("actual_samples"), "api_input_token_counts": case.get("api_input_token_counts"), "requests_per_second": case.get("requests_per_second"), "failures": case.get("failures"), "error": case.get("error")}), flush=True)
    summary["passed"] = len(summary["cases"]) == len(samples) and all(case["passed"] for case in summary["cases"])
    summary["status"] = "complete" if summary["passed"] else "failed"
    write_outputs(args.output_dir, summary)
    print(json.dumps({"passed": summary["passed"], "samples": len(summary["cases"]), "summary": str((args.output_dir / "matrix-summary.json").resolve())}), flush=True)
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
