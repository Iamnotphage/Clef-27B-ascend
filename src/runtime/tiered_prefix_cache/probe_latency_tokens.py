#!/usr/bin/env python3
"""Stage 0.2 probe C: end-to-end suffix-length vs full-prompt latency.

The Tier 2 hit runs the whole 64-layer backbone and the joint head on the
suffix plus the cached prefix memory, so its lower bound is the service
latency of a prompt with the suffix's token count, plus the state restore and
the head's longer memory.  This probe measures the service's sequential
latency for controlled token lengths and for the real 967 / 10,294-token
requests, so the net gain can be computed from measured numbers instead of
from per-layer kernels alone.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics
import sys
import time
import urllib.request


def post(url: str, payload: dict, timeout: float) -> dict:
    body = json.dumps(
        payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    request = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def percentile(values, fraction):
    ordered = sorted(values)
    index = (len(ordered) - 1) * fraction
    lower, upper = math.floor(index), math.ceil(index)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)


def synthetic_request(repeats: int) -> dict:
    state = (
        "Context: " +
        ("The quick brown fox jumps over the lazy dog. " * repeats).strip() +
        "\nQuestion: classify the context.")
    return {
        "model": "clef",
        "state": state,
        "questions": {"answer": {
            "type": "choice",
            "instructions": "Classify the text.",
            "criteria": {"alpha": "first", "beta": "second",
                         "gamma": "third", "delta": "fourth"},
        }},
    }


def measure(url: str, payload: dict, repeats: int, timeout: float) -> dict:
    latencies = []
    tokens = None
    for _ in range(repeats):
        begin = time.perf_counter()
        response = post(url, payload, timeout)
        latencies.append((time.perf_counter() - begin) * 1000.0)
        tokens = response.get("usage", {}).get("input_tokens")
    return {
        "input_tokens": tokens,
        "repeats": repeats,
        "mean_ms": statistics.fmean(latencies),
        "p50_ms": percentile(latencies, 0.50),
        "p95_ms": percentile(latencies, 0.95),
        "min_ms": min(latencies),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:18143/v1/systemone")
    parser.add_argument("--od-sample-file", type=Path, default=Path(
        "/workspace/yangchen/work/aa3e-conv-qkv-formal-20261123.Q3ozHxgd/"
        "all2352.json"))
    parser.add_argument("--erqa-data-dir", type=Path, default=Path(
        "/workspace/yangchen/work/aa3e-conv-qkv-formal-20261123.Q3ozHxgd/"
        "erqa-fixed"))
    parser.add_argument("--targets", default="180,200,260")
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"
                            / "official_eval"))
    from erqa_protocol import build_request as build_erqa, load_samples  # noqa

    report = {
        "schema": "tiered-prefix-cache-latency/1",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "url": args.url,
        "repeats": args.repeats,
        "synthetic": {},
        "real": {},
    }

    for target in [int(value) for value in args.targets.split(",") if value.strip()]:
        low, high = 0, 512
        best = None
        while low <= high:
            middle = (low + high) // 2
            summary = measure(args.url, synthetic_request(middle), 1, args.timeout)
            tokens = summary["input_tokens"] or 0
            if best is None or abs(tokens - target) < abs(best[1] - target):
                best = (middle, tokens)
            if tokens < target:
                low = middle + 1
            elif tokens > target:
                high = middle - 1
            else:
                break
        repeats, tokens = best
        report["synthetic"][str(tokens)] = measure(
            args.url, synthetic_request(repeats), args.repeats, args.timeout)
        print(json.dumps({"synthetic": tokens, **report["synthetic"][str(tokens)]}),
              flush=True)

    document = json.loads(args.od_sample_file.read_text(encoding="utf-8"))
    samples = document["samples"] if isinstance(document, dict) else document
    od = next(sample for sample in samples if sample["name"] == "od-1534")
    report["real"]["od-1534"] = measure(
        args.url, od["request"], args.repeats, args.timeout)

    samples, _ = load_samples(args.erqa_data_dir)
    erqa_sample = next(sample for sample in samples if sample["id"] == "erqa-0167")
    erqa = build_erqa(erqa_sample, args.erqa_data_dir, wire_format="state-list")
    measure(args.url, erqa, 1, args.timeout)  # warm the Tier 1 encoder cache
    report["real"]["erqa-0167"] = measure(args.url, erqa, 5, args.timeout)
    print(json.dumps({"real": report["real"]}, ensure_ascii=False), flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
