#!/usr/bin/env python3
"""Benchmark the fixed three-field /v1/systemone request with standard libraries.

Example:
  python scripts/benchmark_systemone.py --warmup 3 --requests 30 --concurrency 1 \
      --sample-hbm --report /workspace/yangchen/logs/systemone-benchmark.json

Throughput counts validated successful requests and their three decisions.
Latency includes HTTP transfer, response decoding and protocol validation.
"""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
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


def send_request(base_url, body, request, index, phase, timeout, atol):
    result = {"index": index, "phase": phase, "started_at": timestamp(), "request": request, "passed": False}
    started = time.perf_counter()
    try:
        message = Request(
            base_url + "/v1/systemone", data=body,
            headers={"Content-Type": "application/json"}, method="POST",
        )
        with urlopen(message, timeout=timeout) as response:
            result["http_status"] = response.status
            result["response_text"] = response.read().decode("utf-8")
        result["response"] = json.loads(
            result["response_text"], parse_float=finite_float,
            parse_constant=reject_constant,
        )
        if not isinstance(result["response"], dict):
            raise ValueError("Response JSON must be an object")
        result["errors"] = validate(request, result["response"], atol)
        result["passed"] = result["http_status"] == 200 and not result["errors"]
        usage = result["response"].get("usage")
        if isinstance(usage, dict):
            result["actual_input_tokens"] = usage.get("input_tokens")
    except HTTPError as exc:
        result.update(http_status=exc.code, response_text=exc.read().decode("utf-8", errors="replace"), error=str(exc))
    except (OSError, URLError, ValueError, TypeError, KeyError) as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    result["elapsed_ms"] = (time.perf_counter() - started) * 1000
    result["finished_at"] = timestamp()
    return result


def summarize(attempts, wall_seconds, fields_per_request, requested):
    successful = [r for r in attempts if r["passed"]]
    latency = [r["elapsed_ms"] for r in successful]
    tokens = [r["actual_input_tokens"] for r in successful]
    count = len(attempts)
    return {
        "requested_requests": requested,
        "completed_requests": count,
        "successful_requests": len(successful),
        "failed_requests": count - len(successful),
        "success_rate": len(successful) / count if count else None,
        "measurement_wall_seconds": wall_seconds,
        "successful_requests_per_second": len(successful) / wall_seconds if wall_seconds else None,
        "successful_decisions": len(successful) * fields_per_request,
        "decisions_per_second": len(successful) * fields_per_request / wall_seconds if wall_seconds else None,
        "successful_request_latency_ms": {
            "p50": percentile(latency, 0.5), "p95": percentile(latency, 0.95),
            "min": min(latency) if latency else None,
            "max": max(latency) if latency else None,
            "mean": sum(latency) / len(latency) if latency else None,
            "percentile_method": "linear interpolation at (n-1)*quantile",
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
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--sample-hbm", action="store_true")
    parser.add_argument("--hbm-interval", type=float, default=1.0)
    parser.add_argument("--report", "--output", dest="report", type=Path, required=True)
    args = parser.parse_args()
    if args.warmup < 0 or min(args.requests, args.concurrency) < 1:
        parser.error("--warmup must be nonnegative; --requests and --concurrency must be positive")
    if not all(math.isfinite(v) for v in (args.timeout, args.atol, args.hbm_interval)) or min(args.timeout, args.hbm_interval) <= 0 or args.atol < 0:
        parser.error("Timeout and HBM interval must be positive finite values; atol must be nonnegative")
    base_url = args.base_url.rstrip("/")
    template = next(request for name, request in cases(args.model) if name == "mixed")
    body = json.dumps(template, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    fields = len(template["questions"])
    report = {
        "schema_version": 1, "started_at": timestamp(), "passed": False,
        "base_url": base_url, "model": args.model,
        "configuration": {
            "warmup": args.warmup, "requests": args.requests,
            "concurrency": args.concurrency, "timeout_seconds": args.timeout,
            "validation_atol": args.atol,
            "warmup_policy": "Sequential; all requested warmups must pass before measurement",
            "latency_scope": "HTTP transfer, decoding and protocol validation; excludes client queue time",
            "throughput_window": "First submission through completion of all measured requests; excludes warmup",
        },
        "request_template": template,
        "request_template_sha256": hashlib.sha256(body).hexdigest(),
        "template_hash_scope": "Exact compact UTF-8 JSON POST body; preserves question and candidate order",
        "fields_per_request": fields,
        "candidate_counts": {qid: 2 if q["type"] == "noul" else len(q["criteria"]) for qid, q in template["questions"].items()},
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
            for index in range(args.warmup):
                result = send_request(base_url, body, template, index, "warmup", args.timeout, args.atol)
                report["warmup"].append(result)
                if not result["passed"]:
                    report["abort_reason"] = "Warmup request failed validation; inspect the saved response."
                    break
            if len(report["warmup"]) == args.warmup and all(r["passed"] for r in report["warmup"]):
                started = time.perf_counter()
                with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
                    futures = [executor.submit(send_request, base_url, body, template, index, "measured", args.timeout, args.atol) for index in range(args.requests)]
                    for future in as_completed(futures):
                        result = future.result()
                        report["requests"].append(result)
                        print(json.dumps({k: result.get(k) for k in ("index", "passed", "elapsed_ms", "actual_input_tokens", "error")}), flush=True)
                wall_seconds = time.perf_counter() - started
                report["requests"].sort(key=lambda r: r["index"])
        finally:
            if sampler:
                report["hbm"] = sampler.stop()
    else:
        report["abort_reason"] = "Health check failed before warmup."
    report["summary"] = summarize(report["requests"], wall_seconds, fields, args.requests)
    report["passed"] = len(report["requests"]) == args.requests and all(r["passed"] for r in report["requests"])
    report["finished_at"] = timestamp()
    args.report.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.report.with_name(args.report.name + ".tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(args.report)
    print(json.dumps({"passed": report["passed"], "report": str(args.report.resolve()), "summary": report["summary"]}, ensure_ascii=False), flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
