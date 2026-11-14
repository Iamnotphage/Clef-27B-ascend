#!/usr/bin/env python3
"""Time the frozen OpenDecisions 2,352 requests and keep every probability.

The existing benchmark keeps latency but not the per-sample answers, and this
round needs both: the A/B must prove the switch is bitwise neutral on the full
set and measure S1 on the same ordering.

Usage:
  python runtime/od_full_benchmark.py \
    --sample-file .../all2352.json \
    --url http://127.0.0.1:18143/v1/systemone \
    --report .../quality/od-off.json
"""

from __future__ import annotations

import argparse
import json
import math
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
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def post(url, payload, timeout):
    body = json.dumps(
        payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    request = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.status, json.loads(response.read())


def probabilities(answer):
    if "probabilities" in answer:
        return dict(answer["probabilities"])
    value = answer["noul"]
    return {"yes": value, "no": 1.0 - value}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-file", type=Path, required=True)
    parser.add_argument("--url", default="http://127.0.0.1:18143/v1/systemone")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    document = json.loads(args.sample_file.read_text(encoding="utf-8"))
    samples = document["samples"] if isinstance(document, dict) else document
    if args.limit is not None:
        samples = samples[: args.limit]

    report = {
        "created_at": now(),
        "url": args.url,
        "sample_file": str(args.sample_file),
        "samples": len(samples),
        "requests": [],
        "summary": {},
    }

    for index in range(min(args.warmup, len(samples))):
        post(args.url, samples[index]["request"], args.timeout)

    started = time.perf_counter()
    for index, sample in enumerate(samples):
        request = sample["request"]
        record = {
            "index": index,
            "name": sample.get("name", f"od-{index:04d}"),
            "started_at": now(),
        }
        begin = time.perf_counter()
        try:
            status, response = post(args.url, request, args.timeout)
            record["http_status"] = status
            record["response"] = response
            record["input_tokens"] = response.get("usage", {}).get("input_tokens")
            answer = response["answers"]["answer"]
            record["probabilities"] = probabilities(answer)
            record["status"] = "ok"
        except Exception as exc:  # noqa: BLE001 - keep every failure
            record["status"] = "error"
            record["error"] = f"{type(exc).__name__}: {exc}"
        record["elapsed_ms"] = (time.perf_counter() - begin) * 1000.0
        report["requests"].append(record)
        if (index + 1) % 200 == 0:
            print(json.dumps({"done": index + 1, "of": len(samples)}), flush=True)

    wall = time.perf_counter() - started
    successful = [item for item in report["requests"] if item["status"] == "ok"]
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
        "percentile_method": "linear interpolation at (n-1)*quantile",
    }
    report["finished_at"] = now()
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
