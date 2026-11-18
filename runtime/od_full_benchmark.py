#!/usr/bin/env python3
"""Time the frozen OpenDecisions 2,352 requests at a chosen client concurrency.

The report keeps every request payload, raw response, per-request latency and
the client-side in-flight count, so ``runtime/od_raw_acceptance.py`` can score
the run and diff it against the frozen reference without touching the service.
Concurrency above 1 is what actually fills the server's batch (``max_num_seqs``).
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import math
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path


def percentile(values, fraction):
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * fraction
    lower, upper = math.floor(index), math.ceil(index)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)


def now():
    return datetime.now(timezone.utc).isoformat()


def post(url, payload, timeout):
    body = json.dumps(
        payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    request = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.status, response.read().decode("utf-8")


def probabilities(answer):
    if "probabilities" in answer:
        return dict(answer["probabilities"])
    value = answer["noul"]
    return {"yes": value, "no": 1.0 - value}


def finite_float(value):
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"non-finite JSON number: {value}")
    return parsed


def reject_constant(value):
    raise ValueError(f"non-standard JSON constant: {value}")


class Inflight:
    def __init__(self):
        self.lock = threading.Lock()
        self.current = 0

    def enter(self):
        with self.lock:
            self.current += 1
            return self.current

    def leave(self):
        with self.lock:
            self.current -= 1


def run_one(index, sample, url, timeout, inflight):
    record = {
        "index": index,
        "name": sample.get("name", f"od-{index:04d}"),
        "request": sample["request"],
        "started_at": now(),
        "inflight_at_start": inflight.enter(),
    }
    begin = time.perf_counter()
    try:
        status, raw = post(url, sample["request"], timeout)
        record["http_status"] = status
        record["response_text"] = raw
        response = json.loads(raw, parse_float=finite_float, parse_constant=reject_constant)
        record["response"] = response
        record["input_tokens"] = response.get("usage", {}).get("input_tokens")
        answer = response["answers"]["answer"]
        record["probabilities"] = probabilities(answer)
        record["passed"] = True
        record["errors"] = []
        record["status"] = "ok"
    except Exception as exc:  # noqa: BLE001 - keep every failure
        if isinstance(exc, urllib.error.HTTPError):
            record["http_status"] = exc.code
            record["response_text"] = exc.read().decode("utf-8", "replace")
        record["status"] = "error"
        record["passed"] = False
        record["errors"] = [f"{type(exc).__name__}: {exc}"]
        record["error"] = record["errors"][0]
    finally:
        inflight.leave()
    record["elapsed_ms"] = (time.perf_counter() - begin) * 1000.0
    record["finished_at"] = now()
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-file", type=Path, required=True)
    parser.add_argument("--url", default="http://127.0.0.1:18143/v1/systemone")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--journal", type=Path,
                        help="Append completed requests immediately; refuses an existing file")
    args = parser.parse_args()
    if args.concurrency < 1:
        parser.error("--concurrency must be positive")

    document = json.loads(args.sample_file.read_text(encoding="utf-8"))
    samples = document["samples"] if isinstance(document, dict) else document
    if args.limit is not None:
        samples = samples[: args.limit]

    report = {
        "created_at": now(),
        "url": args.url,
        "sample_file": str(args.sample_file),
        "concurrency": args.concurrency,
        "samples": len(samples),
        "configuration": {"requests": len(samples), "concurrency": args.concurrency,
                          "warmup": args.warmup, "timeout": args.timeout},
        "requests": [],
        "summary": {},
    }

    journal = None
    if args.journal:
        args.journal.parent.mkdir(parents=True, exist_ok=True)
        journal = args.journal.open("x", encoding="utf-8", buffering=1)

    if args.warmup:
        warmup = samples[: min(args.warmup, len(samples))]
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            list(pool.map(lambda item: run_one(0, item, args.url, args.timeout, Inflight()),
                          warmup))

    inflight = Inflight()
    started = time.perf_counter()
    results = [None] * len(samples)
    done = 0
    progress_lock = threading.Lock()
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = {
            pool.submit(run_one, index, sample, args.url, args.timeout, inflight): index
            for index, sample in enumerate(samples)
        }
        for future in as_completed(futures):
            index = futures[future]
            results[index] = future.result()
            if journal:
                journal.write(json.dumps(results[index], ensure_ascii=False, allow_nan=False) + "\n")
            with progress_lock:
                done += 1
                if done % 200 == 0:
                    print(json.dumps({"done": done, "of": len(samples)}), flush=True)

    report["requests"] = results
    wall = time.perf_counter() - started
    successful = [item for item in results if item and item["status"] == "ok"]
    latencies = [item["elapsed_ms"] for item in successful]
    report["summary"] = {
        "requested": len(samples),
        "successful": len(successful),
        "failed": len(samples) - len(successful),
        "wall_seconds": wall,
        "requests_per_second": len(successful) / wall if wall else None,
        "decisions_per_second": len(successful) / wall if wall else None,
        "latency_ms": {
            "sample_count": len(latencies),
            "mean": sum(latencies) / len(latencies) if latencies else None,
            "p50": percentile(latencies, 0.50),
            "p95": percentile(latencies, 0.95),
            "p99": percentile(latencies, 0.99),
            "min": min(latencies) if latencies else None,
            "max": max(latencies) if latencies else None,
        },
        "client_inflight_max": max(
            (item["inflight_at_start"] for item in successful), default=None),
        "percentile_method": "linear interpolation at (n-1)*quantile",
    }
    report["finished_at"] = now()
    if journal:
        journal.close()
    args.report.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.report.with_name(f".{args.report.name}.tmp")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(args.report)
    print(json.dumps(report["summary"], ensure_ascii=False))


if __name__ == "__main__":
    raise SystemExit(main())
