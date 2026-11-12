#!/usr/bin/env python3
"""Benchmark /v1/systemone requests with standard libraries.

Example:
  python scripts/benchmark_systemone.py --warmup 3 --requests 30 --concurrency 1 \
      --sample-hbm --report /workspace/yangchen/logs/systemone-benchmark.json

The default remains the historical fixed three-field mixed request. --sample-file
accepts a complete POST JSON object, or {"samples": [{"name": "short",
"request": {...}, "metadata": {...}}]} for a deterministic round-robin workload.
Throughput counts validated successful requests and their actual decisions.
Latency includes HTTP transfer, response decoding and protocol validation.
"""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import statistics
import threading
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

try:
    from .verify_systemone import cases, finite_float, get_health, reject_constant, validate
except ImportError:
    from verify_systemone import cases, finite_float, get_health, reject_constant, validate


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def percentile(values, fraction):
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * fraction
    lower, upper = math.floor(index), math.ceil(index)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)


def distribution(values):
    """Keep the existing interpolated percentiles and make their sample size explicit."""
    mean = statistics.mean(values) if values else None
    stdev = statistics.stdev(values) if len(values) > 1 else (0.0 if values else None)
    return {
        "sample_count": len(values),
        "p50": percentile(values, 0.5), "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "min": min(values) if values else None, "max": max(values) if values else None,
        "mean": mean, "sample_standard_deviation": stdev,
        "coefficient_of_variation": stdev / mean if mean else None,
        "percentile_method": "linear interpolation at (n-1)*quantile",
    }


def load_samples(path, model):
    """Load explicit request samples without changing candidate or field order."""
    source = {"kind": "historical_mixed"}
    if path is None:
        rows = [{"name": "mixed", "request": next(r for n, r in cases(model) if n == "mixed")}]
    else:
        raw = path.read_bytes()
        document = json.loads(raw, parse_float=finite_float, parse_constant=reject_constant)
        source = {"kind": "sample_file", "path": str(path.resolve()), "sha256": hashlib.sha256(raw).hexdigest()}
        if isinstance(document, dict) and "samples" in document:
            rows = document["samples"]
        else:
            rows = [{"name": path.stem, "request": document}]
    if not isinstance(rows, list) or not rows:
        raise ValueError("samples must be a nonempty JSON array")
    samples = []
    names = set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"Sample {index} must be an object")
        request = row.get("request")
        if not isinstance(request, dict) or "state" not in request:
            raise ValueError(f"Sample {index} needs a complete request object with state")
        request = dict(request)
        request.setdefault("model", model)
        if not isinstance(request["model"], str) or not request["model"]:
            raise ValueError(f"Sample {index} needs a model name")
        questions = request.get("questions")
        if not isinstance(questions, dict) or not questions:
            raise ValueError(f"Sample {index} needs nonempty questions")
        counts = {}
        for qid, question in questions.items():
            if not isinstance(question, dict) or question.get("type") not in ("choice", "noul", "score"):
                raise ValueError(f"Sample {index}, {qid}: invalid question type")
            kind = question["type"]
            criteria = question.get("criteria")
            if kind == "choice" and (not isinstance(criteria, dict) or not criteria):
                raise ValueError(f"Sample {index}, {qid}: choice criteria must be a nonempty object")
            if kind == "score" and (not isinstance(criteria, list) or not criteria):
                raise ValueError(f"Sample {index}, {qid}: score criteria must be a nonempty array")
            counts[qid] = 2 if kind == "noul" else len(criteria)
        name = row.get("name", f"sample_{index}")
        if not isinstance(name, str) or not name or name in names:
            raise ValueError("Sample names must be nonempty and unique")
        metadata = row.get("metadata", {})
        if not isinstance(metadata, dict):
            raise ValueError(f"Sample {index}: metadata must be an object")
        names.add(name)
        body = json.dumps(request, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")
        samples.append({
            "name": name, "request": request, "body": body,
            "request_sha256": hashlib.sha256(body).hexdigest(),
            "field_count": len(questions), "candidate_counts": counts,
            "candidate_count": sum(counts.values()), "metadata": metadata,
        })
    return samples, source


class SendingConcurrency:
    """Observe requests currently inside urlopen/read, rather than queued futures."""
    def __init__(self):
        self.lock = threading.Lock()
        self.active = 0
        self.peak = 0

    def enter(self):
        with self.lock:
            self.active += 1
            self.peak = max(self.peak, self.active)
            return self.active

    def leave(self):
        with self.lock:
            self.active -= 1


def hbm_devices(output):
    """Parse the final used/total pair of chip rows in npu-smi 26.1.1."""
    devices = []
    card = None
    for line in output.splitlines():
        if "Process id" in line:
            break
        cells = line.split("|")
        if len(cells) < 5:
            continue
        name_row = re.fullmatch(r"\s*(\d+)\s+[A-Za-z][A-Za-z0-9_]*\s*", cells[1])
        if name_row:
            card = int(name_row[1])
            continue
        chip_row = re.fullmatch(r"\s*(\d+)\s+(\d+)\s*", cells[1])
        pairs = re.findall(r"(\d+)\s*/\s*(\d+)", cells[3])
        if card is not None and chip_row and pairs and int(pairs[-1][1]) > 0:
            used, total = map(int, pairs[-1])
            devices.append({
                "card": card, "chip": int(chip_row[1]),
                "physical_device": int(chip_row[2]),
                "used_mib": used, "total_mib": total,
            })
    return devices


class HbmSampler:
    def __init__(self, interval):
        self.interval = interval
        self.samples = []
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _sample(self):
        sample = {"started_at": timestamp(), "monotonic_seconds": time.monotonic()}
        try:
            result = subprocess.run(
                ["npu-smi", "info"], capture_output=True, text=True, timeout=5,
                check=False,
            )
            sample.update(
                returncode=result.returncode, stdout=result.stdout,
                stderr=result.stderr,
                devices=hbm_devices(result.stdout) if result.returncode == 0 else [],
            )
            if not sample["devices"]:
                sample["error"] = "No HBM chip rows parsed; inspect the raw npu-smi output."
        except (OSError, subprocess.SubprocessError) as exc:
            sample["error"] = f"{type(exc).__name__}: {exc}"
        sample["finished_at"] = timestamp()
        self.samples.append(sample)

    def _run(self):
        deadline = time.monotonic()
        while not self.stop_event.is_set():
            self._sample()
            deadline += self.interval
            # Slow npu-smi commands determine the actual interval in that case.
            if self.stop_event.wait(max(0, deadline - time.monotonic())):
                return

    def start(self):
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        self.thread.join(timeout=7)
        self._sample()
        peaks = {}
        totals = []
        valid_samples = [s for s in self.samples if s.get("devices")]
        for sample in valid_samples:
            totals.append(sum(device["used_mib"] for device in sample["devices"]))
            for device in sample["devices"]:
                key = (device["card"], device["chip"], device["physical_device"])
                if key not in peaks or device["used_mib"] > peaks[key]["used_mib"]:
                    peaks[key] = dict(device, observed_at=sample["started_at"])
        return {
            "enabled": True,
            "requested_interval_seconds": self.interval,
            "scope": "Device HBM usage during warmup and measured requests, with a final adjacent sample; includes other processes.",
            "method": "npu-smi info polling; peaks refer to observed samples",
            "sampled_peak_hbm": {
                "per_device": list(peaks.values()),
                "max_total_used_mib": max(totals) if totals else None,
            },
            "valid_sample_count": len(valid_samples),
            "samples": self.samples,
        }


def send_request(base_url, body, request, index, phase, timeout, atol,
                 submitted_at=None, sample=None, concurrency=None, run_id="benchmark"):
    worker_started = time.perf_counter()
    request_id = f"clef-bench-{run_id}-{phase}-{index}"
    result = {
        "index": index, "phase": phase, "started_at": timestamp(), "request": request,
        "passed": False, "client_request_id": request_id,
        "request_sha256": hashlib.sha256(body).hexdigest(),
        "client_queue_ms": max(0, (worker_started - submitted_at) * 1000) if submitted_at is not None else 0.0,
        "worker_started_monotonic_seconds": worker_started,
        "submitted_monotonic_seconds": submitted_at,
        "server_queue_ms": None,
        "server_queue_measurement": "Requires correlated server-side timing; HTTP latency includes server queueing",
        "actual_input_tokens": None,
    }
    questions = request["questions"]
    result["field_count"] = len(questions)
    result["candidate_counts"] = {qid: 2 if q["type"] == "noul" else len(q["criteria"]) for qid, q in questions.items()}
    result["candidate_count"] = sum(result["candidate_counts"].values())
    if sample:
        result.update(sample_name=sample["name"], sample_metadata=sample["metadata"])
    started = time.perf_counter()
    result["request_started_monotonic_seconds"] = started
    try:
        message = Request(
            base_url + "/v1/systemone", data=body,
            headers={"Content-Type": "application/json", "X-Request-ID": request_id}, method="POST",
        )
        http_started = time.perf_counter()
        result["http_started_monotonic_seconds"] = http_started
        if concurrency:
            result["http_inflight_at_start"] = concurrency.enter()
        try:
            with urlopen(message, timeout=timeout) as response:
                result["http_status"] = response.status
                result["response_request_id"] = response.headers.get("X-Request-ID")
                response_bytes = response.read()
        finally:
            result["http_finished_monotonic_seconds"] = time.perf_counter()
            result["http_ms"] = (result["http_finished_monotonic_seconds"] - http_started) * 1000
            if concurrency:
                concurrency.leave()
        decode_started = time.perf_counter()
        result["response_text"] = response_bytes.decode("utf-8")
        result["response"] = json.loads(
            result["response_text"], parse_float=finite_float,
            parse_constant=reject_constant,
        )
        result["json_decode_ms"] = (time.perf_counter() - decode_started) * 1000
        validation_started = time.perf_counter()
        if not isinstance(result["response"], dict):
            raise ValueError("Response JSON must be an object")
        result["errors"] = validate(request, result["response"], atol)
        result["protocol_validation_ms"] = (time.perf_counter() - validation_started) * 1000
        result["passed"] = result["http_status"] == 200 and not result["errors"]
        usage = result["response"].get("usage")
        if isinstance(usage, dict):
            result["actual_input_tokens"] = usage.get("input_tokens")
    except HTTPError as exc:
        result.update(http_status=exc.code, response_text=exc.read().decode("utf-8", errors="replace"), error=str(exc))
    except (OSError, URLError, ValueError, TypeError, KeyError) as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    finished = time.perf_counter()
    result["elapsed_ms"] = (finished - started) * 1000
    result["client_total_ms"] = (finished - (submitted_at if submitted_at is not None else worker_started)) * 1000
    result["finished_at"] = timestamp()
    return result


def summarize(attempts, wall_seconds, fields_per_request, requested):
    successful = [r for r in attempts if r["passed"]]
    latency = [r["elapsed_ms"] for r in successful]
    tokens = [r["actual_input_tokens"] for r in successful if isinstance(r.get("actual_input_tokens"), int)]
    decisions = sum(r.get("field_count", fields_per_request) for r in successful)
    count = len(attempts)
    return {
        "requested_requests": requested,
        "completed_requests": count,
        "successful_requests": len(successful),
        "failed_requests": count - len(successful),
        "success_rate": len(successful) / count if count else None,
        "measurement_wall_seconds": wall_seconds,
        "successful_requests_per_second": len(successful) / wall_seconds if wall_seconds else None,
        "successful_decisions": decisions,
        "decisions_per_second": decisions / wall_seconds if wall_seconds else None,
        "input_tokens_per_second": sum(tokens) / wall_seconds if wall_seconds else None,
        "successful_request_latency_ms": distribution(latency),
        "successful_stage_ms": {
            stage: distribution([r[stage] for r in successful if stage in r])
            for stage in ("client_queue_ms", "http_ms", "json_decode_ms", "protocol_validation_ms", "client_total_ms")
        },
        "actual_input_tokens": {
            "successful_total": sum(tokens),
            "per_request_histogram": dict(sorted(Counter(tokens).items())),
            "min": min(tokens) if tokens else None,
            "max": max(tokens) if tokens else None,
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="clef")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--requests", type=int, default=30)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--sample-file", type=Path, help="Complete request JSON or an object containing a samples array")
    parser.add_argument("--min-duration-seconds", type=float, default=0, help="Continue measured requests until both the count and this window are satisfied")
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--sample-hbm", action="store_true")
    parser.add_argument("--hbm-interval", type=float, default=1.0)
    parser.add_argument("--report", "--output", dest="report", type=Path, required=True)
    args = parser.parse_args()
    if args.warmup < 0 or min(args.requests, args.concurrency) < 1:
        parser.error("--warmup must be nonnegative; --requests and --concurrency must be positive")
    if not all(math.isfinite(v) for v in (args.timeout, args.atol, args.hbm_interval, args.min_duration_seconds)) or min(args.timeout, args.hbm_interval) <= 0 or min(args.atol, args.min_duration_seconds) < 0:
        parser.error("Timeout and HBM interval must be positive finite values; atol must be nonnegative")
    base_url = args.base_url.rstrip("/")
    try:
        samples, sample_source = load_samples(args.sample_file, args.model)
    except (OSError, ValueError, TypeError) as exc:
        parser.error(str(exc))
    template, body = samples[0]["request"], samples[0]["body"]
    fields = len(template["questions"])
    run_id = str(time.time_ns())
    sending_concurrency = SendingConcurrency()
    report = {
        "schema_version": 2, "started_at": timestamp(), "passed": False,
        "run_id": run_id,
        "base_url": base_url, "model": args.model,
        "configuration": {
            "warmup": args.warmup, "requests": args.requests,
            "concurrency": args.concurrency, "timeout_seconds": args.timeout,
            "minimum_measurement_seconds": args.min_duration_seconds,
            "validation_atol": args.atol,
            "warmup_policy": "Sequential per sample; each sample receives the requested warmup count and all must pass",
            "latency_scope": "HTTP transfer, decoding and protocol validation; excludes client queue time",
            "http_scope": "Client urlopen through full response-body read; includes network, server queue, processing and serialization",
            "json_decode_scope": "UTF-8 response text conversion and JSON parsing",
            "client_queue_scope": "executor.submit timestamp through worker function entry; separately reported from elapsed_ms",
            "server_queue_scope": "Unavailable in client-only timing; correlate client_request_id with server telemetry",
            "concurrency_scope": "ThreadPoolExecutor max workers bounds active HTTP requests; observed peak excludes queued futures",
            "throughput_window": "First submission through completion of all measured requests; excludes warmup",
            "sample_selection": "Measured requests round-robin in sample-file order; each sample prewarmed separately",
        },
        "request_template": template,
        "request_template_sha256": hashlib.sha256(body).hexdigest(),
        "template_hash_scope": "Exact compact UTF-8 JSON POST body; preserves question and candidate order",
        "fields_per_request": fields,
        "candidate_counts": {qid: 2 if q["type"] == "noul" else len(q["criteria"]) for qid, q in template["questions"].items()},
        "sample_source": sample_source,
        "samples": [{k: v for k, v in sample.items() if k != "body"} for sample in samples],
        "health": get_health(base_url, 0, args.timeout),
        "warmup": [], "requests": [],
        "hbm": {"enabled": False},
    }
    sampler = HbmSampler(args.hbm_interval) if args.sample_hbm else None
    wall_seconds = None
    if report["health"]["passed"]:
        if sampler:
            sampler.start()
        try:
            for sample in samples:
                for _ in range(args.warmup):
                    result = send_request(base_url, sample["body"], sample["request"], len(report["warmup"]), "warmup", args.timeout, args.atol, sample=sample, run_id=run_id)
                    report["warmup"].append(result)
                    if not result["passed"]:
                        report["abort_reason"] = "Warmup request failed validation; inspect the saved response."
                        break
                if "abort_reason" in report:
                    break
            if len(report["warmup"]) == args.warmup * len(samples) and all(r["passed"] for r in report["warmup"]):
                started = time.perf_counter()
                with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
                    def submit(index):
                        sample = samples[index % len(samples)]
                        return executor.submit(send_request, base_url, sample["body"], sample["request"], index, "measured", args.timeout, args.atol, time.perf_counter(), sample, sending_concurrency, run_id)

                    pending = {submit(index) for index in range(args.requests)}
                    submitted_count = args.requests
                    while pending:
                        done, pending = wait(pending, return_when=FIRST_COMPLETED)
                        for future in done:
                            result = future.result()
                            report["requests"].append(result)
                            print(json.dumps({k: result.get(k) for k in ("index", "passed", "elapsed_ms", "actual_input_tokens", "error")}), flush=True)
                        while len(pending) < args.concurrency and time.perf_counter() - started < args.min_duration_seconds:
                            pending.add(submit(submitted_count))
                            submitted_count += 1
                wall_seconds = time.perf_counter() - started
                report["requests"].sort(key=lambda r: r["index"])
        finally:
            if sampler:
                report["hbm"] = sampler.stop()
    else:
        report["abort_reason"] = "Health check failed before warmup."
    report["summary"] = summarize(report["requests"], wall_seconds, fields, args.requests)
    report["summary"]["observed_peak_http_concurrency"] = sending_concurrency.peak
    report["summary"]["minimum_measurement_duration_satisfied"] = wall_seconds is not None and wall_seconds >= args.min_duration_seconds
    report["passed"] = len(report["requests"]) >= args.requests and report["summary"]["minimum_measurement_duration_satisfied"] and all(r["passed"] for r in report["requests"])
    report["finished_at"] = timestamp()
    args.report.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.report.with_name(args.report.name + ".tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(args.report)
    print(json.dumps({"passed": report["passed"], "report": str(args.report.resolve()), "summary": report["summary"]}, ensure_ascii=False), flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
