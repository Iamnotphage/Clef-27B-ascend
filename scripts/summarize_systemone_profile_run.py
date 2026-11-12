#!/usr/bin/env python3
"""Aggregate compact performance evidence from an entire saved profiling run.

Reads JSON files only, without importing model code or contacting a service.
Profiler configuration and active collection are separate facts. Only the
unobserved 1024 stable-round reports enter the formal 307-token comparison.
Original requests/responses remain at their indexed source paths.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import statistics


FIXED_REQUEST_SHA256 = "ab2724637a36029759ea673964b59f0c30fc2e27af4b7d7fa3bdcd784de33aef"
STAGE_KEYS = ("http_ms", "json_decode_ms", "protocol_validation_ms", "client_queue_ms")


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


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


def epoch(value):
    try:
        parsed = datetime.fromisoformat(value)
        return parsed.timestamp() if parsed.tzinfo is not None else None
    except (ValueError, TypeError):
        return None


def option(argv, *names):
    for index, value in enumerate(argv):
        for name in names:
            if value == name and index + 1 < len(argv):
                return argv[index + 1]
            if value.startswith(name + "="):
                return value[len(name) + 1:]
    return None


def integer(value):
    try:
        return int(value) if not isinstance(value, bool) else None
    except (ValueError, TypeError):
        return None


def parse_launch(document, path, raw):
    argv = document.get("argv", document.get("command", []))
    argv = shlex.split(argv) if isinstance(argv, str) else argv
    argv = argv if isinstance(argv, list) and all(isinstance(value, str) for value in argv) else []
    environment = document.get("environment", {})
    environment = environment if isinstance(environment, dict) else {}
    profiler = option(argv, "--profiler-config")
    try:
        profiler_config = json.loads(profiler) if profiler is not None else None
    except ValueError:
        profiler_config = {"unparsed_value": profiler}
    observations = document.get("observations") is True or bool(environment.get("CLEF_PROFILE_OBSERVE_DIR")) or environment.get("CLEF_PROFILE_MARKERS") == "1"
    cpu_only = document.get("cpu_timing_only") is True or (bool(environment.get("CLEF_PROFILE_OBSERVE_DIR")) and environment.get("CLEF_PROFILE_MARKERS") != "1")
    return {
        "source": str(path), "sha256": digest(raw), "label": document.get("label", path.parent.name),
        "pid": document.get("pid"), "started_at": document.get("started_at"), "cwd": document.get("cwd"),
        "tp": integer(option(argv, "--tensor-parallel-size", "-tp")),
        "max_model_len": integer(option(argv, "--max-model-len")),
        "max_num_seqs": integer(option(argv, "--max-num-seqs")),
        "max_num_batched_tokens": integer(option(argv, "--max-num-batched-tokens")),
        "profile_configured": profiler is not None or document.get("profile") is True,
        "profiler_config": profiler_config, "observations": observations, "cpu_timing_only": cpu_only,
        "argv": argv, "performance_environment": {key: environment.get(key) for key in ("OMP_NUM_THREADS", "TASK_QUEUE_ENABLE", "ASCEND_RT_VISIBLE_DEVICES", "CLEF_PROFILE_MARKERS", "CLEF_PROFILE_OBSERVE_DIR", "CLEF_PROFILE_SCOPE_FILTER")},
    }


def nearest_launch(path, root, launches):
    for parent in (path.parent, *path.parents):
        if parent == root.parent:
            break
        candidate = str(parent / "launch.json")
        if candidate in launches:
            return candidate
    return None


def discover(root, output, max_json_bytes):
    reports, launches, windows, excluded, errors = [], {}, [], [], []
    for directory, names, files in os.walk(root):
        parent = Path(directory)
        retained = []
        for name in sorted(names):
            path = parent / name
            if path.resolve() == output.resolve() or "snapshot" in name.lower() or name == "traces" or name == "ASCEND_PROFILER_OUTPUT" or name.endswith("_ascend_pt"):
                excluded.append({"path": str(path), "reason": "output, snapshot copy, or raw trace directory"})
            else:
                retained.append(name)
        names[:] = retained
        for filename in sorted(files):
            if not filename.endswith(".json"):
                continue
            path = parent / filename
            if re.search(r"(?:^|[-_])warmup(?:[-_.]|$)", filename, re.IGNORECASE):
                excluded.append({"path": str(path), "reason": "warmup-only report"})
                continue
            if "snapshot" in filename.lower():
                excluded.append({"path": str(path), "reason": "snapshot copy"})
                continue
            try:
                if path.stat().st_size > max_json_bytes:
                    excluded.append({"path": str(path), "reason": "JSON exceeds read-size limit"})
                    continue
                raw = path.read_bytes()
                document = json.loads(raw)
            except (OSError, ValueError) as exc:
                errors.append({"path": str(path), "error": f"{type(exc).__name__}: {exc}"})
                continue
            if not isinstance(document, dict):
                continue
            if filename == "launch.json":
                launches[str(path)] = parse_launch(document, path, raw)
            if filename == "manifest.json" and "trace_root" in document and ("start" in document or "request_window_started_at" in document):
                windows.append({"source": str(path), "sha256": digest(raw), "start": document.get("start"), "stop": document.get("stop"), "request_window_started_at": document.get("request_window_started_at"), "request_window_finished_at": document.get("request_window_finished_at"), "request_report": str(parent / "profile-requests.json")})
            summary = document.get("summary")
            if isinstance(document.get("requests"), list) and isinstance(summary, dict) and "successful_requests_per_second" in summary:
                reports.append((path, digest(raw), document))
    for window in windows:
        window["launch_source"] = nearest_launch(Path(window["source"]), root, launches)
    # Exact file copies and reserialized copies with the same run/time identity
    # supply one measurement. Every skipped copy remains indexed.
    unique, hashes, identities = [], {}, {}
    for path, sha, report in sorted(reports, key=lambda row: str(row[0])):
        identity = (report.get("run_id"), report.get("started_at"), report.get("finished_at"))
        identity_valid = all(isinstance(value, str) and value for value in identity)
        original = hashes.get(sha) or (identities.get(identity) if identity_valid else None)
        if original:
            excluded.append({"path": str(path), "sha256": sha, "reason": "duplicate source measurement", "original": original})
            continue
        hashes[sha] = str(path)
        if identity_valid:
            identities[identity] = str(path)
        unique.append((path, sha, report))
    return unique, launches, windows, excluded, errors


def active_profile(path, report, launch_source, launch, windows):
    related = [window for window in windows if window["launch_source"] == launch_source and launch_source is not None]
    direct = next((window for window in related if window["request_report"] == str(path)), None)
    if path.name == "profile-requests.json":
        return True, "profile-requests.json inside an explicit collection window", direct["source"] if direct else None
    if launch is None:
        return None, "No ancestor launch.json; collection state is unverified", None
    if not launch["profile_configured"]:
        return False, "Ancestor launch has no configured profiler", None
    started = epoch(report.get("started_at"))
    finished = epoch(report.get("finished_at"))
    if started is None or finished is None:
        return None, "Profiler is configured and report timestamps are unavailable", None
    # The fixed API wrapper starts inactive. Successful start/stop control calls
    # determine state; an unsuccessful stop leaves subsequent state unknown.
    events = []
    for window in related:
        start = window.get("start")
        stop = window.get("stop")
        if isinstance(start, dict) and start.get("passed") is True:
            when = epoch(start.get("finished_at")) or epoch(window.get("request_window_started_at"))
            if when is not None:
                events.append((when, True, window["source"]))
        if isinstance(stop, dict):
            when = epoch(stop.get("finished_at")) or epoch(window.get("request_window_finished_at"))
            if when is not None:
                events.append((when, False if stop.get("passed") is True else None, window["source"]))
    events.sort(key=lambda item: item[0])
    state, evidence = False, None
    for when, value, source in events:
        if when <= started:
            state, evidence = value, source
        elif when < finished and value is True:
            return True, "Report overlaps a successful start_profile interval", source
    if state is True:
        return True, "Successful start_profile remains active at report start", evidence
    if state is None:
        return None, "A prior stop_profile failed; collection state is unverified", evidence
    return False, "Configured profiler is inactive outside the recorded start/stop windows", evidence


def candidate_counts(attempt, report):
    counts = attempt.get("candidate_counts")
    if isinstance(counts, dict):
        return counts
    request = attempt.get("request", report.get("request_template", {}))
    try:
        return {qid: 2 if question["type"] == "noul" else len(question["criteria"]) for qid, question in request["questions"].items()}
    except (KeyError, TypeError, AttributeError):
        return {}


def compact_report(path, sha, report, root, launches, windows):
    launch_source = nearest_launch(path, root, launches)
    launch = launches.get(launch_source)
    active, active_basis, window_source = active_profile(path, report, launch_source, launch, windows)
    attempts = report["requests"]
    if not all(isinstance(row, dict) for row in attempts):
        raise ValueError("requests must be an array of objects")
    successful = [row for row in attempts if row.get("passed") is True]
    sample_names, sample_groups, fields, candidates, tokens, hashes = [], [], [], [], [], []
    for row in attempts:
        if isinstance(row.get("sample_name"), str):
            sample_names.append(row["sample_name"])
        metadata = row.get("sample_metadata", {})
        if isinstance(metadata, dict) and isinstance(metadata.get("group"), str):
            sample_groups.append(metadata["group"])
        counts = candidate_counts(row, report)
        if counts:
            candidates.append(counts)
            fields.append(len(counts))
        response = row.get("response", {})
        usage = response.get("usage", {}) if isinstance(response, dict) else {}
        count = usage.get("input_tokens") if isinstance(usage, dict) else None
        count = row.get("actual_input_tokens") if count is None else count
        if isinstance(count, int) and not isinstance(count, bool) and count > 0:
            tokens.append(count)
        value = row.get("request_sha256", report.get("request_template_sha256"))
        if isinstance(value, str):
            hashes.append(value)
    latency_values = [row["elapsed_ms"] for row in successful if finite(row.get("elapsed_ms")) and row["elapsed_ms"] >= 0]
    stages = {key: distribution([row[key] for row in successful if finite(row.get(key)) and row[key] >= 0]) for key in STAGE_KEYS}
    saved = report["summary"]
    wall = saved.get("measurement_wall_seconds")
    wall = wall if finite(wall) and wall > 0 else None
    configuration = report.get("configuration", {})
    configuration = configuration if isinstance(configuration, dict) else {}
    profile_configured = launch["profile_configured"] if launch else None
    observed = launch["observations"] if launch else None
    cpu_only = launch["cpu_timing_only"] if launch else None
    stable_path = any(re.fullmatch(r"tp[12]-1024-stable", part) for part in path.relative_to(root).parts) and bool(re.fullmatch(r"round-\d+\.json", path.name))
    if active is True:
        classification = "profiled_window"
    elif active is None:
        classification = "profiling_state_unverified"
    elif "heterogeneous-precision" in path.name:
        classification = "precision_regression"
    elif observed or cpu_only:
        classification = "observer_enabled_unprofiled"
    elif stable_path and launch and launch["max_model_len"] == 1024:
        classification = "stable_307"
    else:
        classification = "extension_unprofiled"
    warmup = report.get("warmup", [])
    stable_protocol = classification == "stable_307" and report.get("passed") is True and len(successful) == len(attempts) >= 300 and isinstance(warmup, list) and len(warmup) >= 10 and all(isinstance(row, dict) and row.get("passed") is True for row in warmup) and wall is not None and wall >= 30 and configuration.get("concurrency") == 1 and launch["max_num_seqs"] == 1 and set(tokens) == {307} and len(tokens) == len(attempts) and set(fields) == {3} and set(hashes) == {FIXED_REQUEST_SHA256}
    candidate_variants = {json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True) for value in candidates}
    candidate_variants = [json.loads(value) for value in sorted(candidate_variants)]
    successful_token_total = 0
    successful_field_total = 0
    for row in successful:
        usage = row.get("response", {}).get("usage", {}) if isinstance(row.get("response"), dict) else {}
        actual = usage.get("input_tokens", row.get("actual_input_tokens")) if isinstance(usage, dict) else row.get("actual_input_tokens")
        if isinstance(actual, int) and not isinstance(actual, bool) and actual > 0:
            successful_token_total += actual
        successful_field_total += len(candidate_counts(row, report))
    row = {
        "source_report": str(path), "source_report_relative_path": str(path.relative_to(root)), "source_report_sha256": sha,
        "run_id": report.get("run_id"), "started_at": report.get("started_at"), "finished_at": report.get("finished_at"),
        "launch_source": launch_source, "launch_sha256": launch["sha256"] if launch else None,
        **{key: launch.get(key) if launch else None for key in ("tp", "max_model_len", "max_num_seqs", "max_num_batched_tokens", "observations", "cpu_timing_only")},
        "profile_configured": profile_configured, "profile_active": active, "profile_active_basis": active_basis, "profile_window_manifest": window_source,
        "measurement_classification": classification, "stable_round_protocol_passed": bool(stable_protocol), "eligible_for_formal_stable_comparison": False,
        "sample_names": sorted(set(sample_names)) or ["mixed"], "sample_groups": sorted(set(sample_groups)),
        "input_token_histogram": dict(sorted(Counter(tokens).items())), "input_token_counts": sorted(set(tokens)),
        "field_counts": sorted(set(fields)), "candidate_counts": candidate_variants,
        "candidate_total_counts": sorted({sum(value.values()) for value in candidate_variants if all(isinstance(count, int) for count in value.values())}),
        "request_sha256": sorted(set(hashes)), "concurrency": configuration.get("concurrency"),
        "configured_warmup": configuration.get("warmup"), "warmup_samples": len(warmup) if isinstance(warmup, list) else None,
        "requested_requests": configuration.get("requests"), "actual_samples": len(attempts), "successful_requests": len(successful),
        "report_passed": report.get("passed") is True, "all_measured_requests_passed": bool(attempts) and len(successful) == len(attempts),
        "success_rate": len(successful) / len(attempts) if attempts else None, "measurement_wall_seconds": wall,
        "requests_per_second": len(successful) / wall if wall else None,
        "decisions_per_second": successful_field_total / wall if wall else None,
        "input_tokens_per_second": successful_token_total / wall if wall else None,
        "latency_ms": distribution(latency_values), "stages_ms": stages,
        "observed_peak_http_concurrency": saved.get("observed_peak_http_concurrency"),
        "below_300_requests": len(attempts) < 300, "below_30_second_window": wall is None or wall < 30,
        "saved_requests_per_second": saved.get("successful_requests_per_second"),
    }
    return row, latency_values


CSV_FIELDS = ["source_report", "source_report_sha256", "launch_source", "launch_sha256", "measurement_classification", "tp", "max_model_len", "max_num_seqs", "max_num_batched_tokens", "profile_configured", "profile_active", "observations", "cpu_timing_only", "concurrency", "sample_names", "sample_groups", "input_token_counts", "field_counts", "candidate_counts", "candidate_total_counts", "actual_samples", "warmup_samples", "measurement_wall_seconds", "success_rate", "report_passed", "requests_per_second", "decisions_per_second", "input_tokens_per_second", "p50_ms", "p95_ms", "p99_ms", "latency_sample_count", "latency_mean_ms", "latency_stdev_ms", "latency_cv", "http_mean_ms", "http_p95_ms", "json_decode_mean_ms", "protocol_validation_mean_ms", "client_queue_mean_ms", "client_queue_p95_ms", "observed_peak_http_concurrency", "stable_round_protocol_passed", "eligible_for_formal_stable_comparison", "below_300_requests", "below_30_second_window"]


def aggregate(root, output, max_json_bytes=128 * 1024 * 1024):
    reports, launches, windows, excluded, errors = discover(root, output, max_json_bytes)
    rows, latency = [], {}
    for path, sha, report in reports:
        try:
            row, values = compact_report(path, sha, report, root, launches, windows)
            rows.append(row)
            latency[row["source_report"]] = values
        except (ValueError, TypeError, KeyError) as exc:
            errors.append({"path": str(path), "error": f"{type(exc).__name__}: {exc}"})
    stable_groups = {}
    for tp in sorted({row["tp"] for row in rows if row["measurement_classification"] == "stable_307" and row["tp"] is not None}):
        group = [row for row in rows if row["measurement_classification"] == "stable_307" and row["tp"] == tp]
        run_ids = [row["run_id"] for row in group]
        valid = len(group) >= 3 and all(row["stable_round_protocol_passed"] for row in group) and all(isinstance(value, str) and value for value in run_ids) and len(set(run_ids)) == len(run_ids)
        for row in group:
            row["eligible_for_formal_stable_comparison"] = valid
        stable_groups[f"tp{tp}-1024"] = {"measurement_protocol_passed": valid, "round_count": len(group), "profiled_throughput_included": False, "source_reports": [row["source_report"] for row in group], "round_requests_per_second": distribution([row["requests_per_second"] for row in group if row["requests_per_second"] is not None]), "pooled_request_latency_ms": distribution([value for row in group for value in latency[row["source_report"]]])}
    classifications = {}
    for classification in sorted({row["measurement_classification"] for row in rows}):
        group = [row for row in rows if row["measurement_classification"] == classification]
        classifications[classification] = {"report_count": len(group), "actual_samples": sum(row["actual_samples"] for row in group), "failed_reports": sum(not row["report_passed"] for row in group), "source_reports": [row["source_report"] for row in group]}
    return {
        "schema_version": 1, "created_at": datetime.now(timezone.utc).isoformat(), "run_dir": str(root),
        "evidence_scope": "Compact statistics recomputed from saved benchmark JSON and nearest ancestor launch records; no inference, model import, NPU or HTTP operation",
        "hardware_execution_verified": None, "configuration_verification_scope": "Launch JSON and control manifests are indexed by SHA; hardware and source-patch equivalence require the independent run manifest",
        "formal_throughput_policy": "Only unobserved, profiler-inactive 1024 stable_307 rounds with independent run IDs and full sampling protocol enter formal stable_groups; profiled and observer-enabled throughput stays in separate classes",
        "latency_scope": "Successful original elapsed_ms covers HTTP, UTF-8/JSON decode and protocol validation; client executor queue is separate; percentiles pool original successful per-request measurements",
        "report_count": len(rows), "classification_counts": {key: value["report_count"] for key, value in classifications.items()},
        "rows": rows, "stable_groups": stable_groups, "classifications": classifications,
        "launches": launches, "profile_windows": windows, "excluded": excluded, "read_errors": errors,
        "raw_request_response_bodies_copied": False,
    }


def write_summary(output, summary):
    output.mkdir(parents=True, exist_ok=True)
    with (output / "all-performance-matrix.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in summary["rows"]:
            value = {key: row.get(key) for key in CSV_FIELDS}
            latency, stages = row["latency_ms"], row["stages_ms"]
            value.update({"p50_ms": latency["p50"], "p95_ms": latency["p95"], "p99_ms": latency["p99"], "latency_sample_count": latency["sample_count"], "latency_mean_ms": latency["mean"], "latency_stdev_ms": latency["sample_standard_deviation"], "latency_cv": latency["coefficient_of_variation"], "http_p95_ms": stages["http_ms"]["p95"], "client_queue_p95_ms": stages["client_queue_ms"]["p95"], **{key.removesuffix("_ms") + "_mean_ms": stages[key]["mean"] for key in STAGE_KEYS}})
            writer.writerow({key: json.dumps(item, ensure_ascii=False, separators=(",", ":")) if isinstance(item, (list, dict)) else item for key, item in value.items()})
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-json-mib", type=int, default=128)
    args = parser.parse_args()
    if not args.run_dir.is_dir() or args.max_json_mib <= 0:
        parser.error("--run-dir must exist and --max-json-mib must be positive")
    summary = aggregate(args.run_dir.resolve(), args.output_dir.resolve(), args.max_json_mib * 1024 * 1024)
    write_summary(args.output_dir, summary)
    print(json.dumps({"report_count": summary["report_count"], "classification_counts": summary["classification_counts"], "read_errors": len(summary["read_errors"]), "summary": str((args.output_dir / "summary.json").resolve()), "matrix": str((args.output_dir / "all-performance-matrix.csv").resolve())}), flush=True)
    return 0 if summary["report_count"] > 0 and not summary["read_errors"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
