#!/usr/bin/env python3
"""Probe how service history changes per-token-length latency modes.

The final S1 run shows a stable per-exact-token-length split: some lengths
answer in ~110 ms, others in ~170 ms.  A fresh service answers every tested
length in the fast mode; running the OpenDecisions prefix in file order moves
specific lengths into the slow mode.  This driver replays the prefix in
controllable chunks and re-measures fixed probe samples after every chunk, so
the first poisoning request can be bisected.

It uses only the stdlib HTTP client and the same complete request objects as
the benchmark, so it does not depend on a particular benchmark revision.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
import urllib.request
from collections import Counter
from pathlib import Path

PROBES = {
    "od-0383": 203,
    "od-0182": 204,
    "od-0773": 310,
}


def post(url: str, payload: dict, timeout: float) -> tuple[float, dict]:
    body = json.dumps(payload).encode()
    request = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}, method="POST"
    )
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=timeout) as response:
        data = json.loads(response.read())
    return (time.perf_counter() - started) * 1000.0, data


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:18143")
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--probe-file", type=Path, default=None,
                        help="sample file supplying the fixed probe requests; defaults to --samples")
    parser.add_argument("--chunk", type=int, default=25)
    parser.add_argument("--limit", type=int, default=400)
    parser.add_argument("--probe-repeats", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    all_samples = json.loads(args.samples.read_text())["samples"]
    probe_source = args.probe_file or args.samples
    probe_samples = json.loads(probe_source.read_text())["samples"]
    by_name = {sample["name"]: sample for sample in probe_samples}
    url = args.base_url.rstrip("/") + "/v1/systemone"
    stages: list[dict] = []
    chunk_rows: list[dict] = []

    def probe(stage: int) -> dict:
        result = {}
        for name, expected in PROBES.items():
            sample = by_name[name]
            latencies, tokens = [], []
            for _ in range(args.probe_repeats):
                elapsed, response = post(url, sample["request"], args.timeout)
                latencies.append(elapsed)
                tokens.append(response.get("usage", {}).get("input_tokens"))
            result[name] = {
                "expected_tokens": expected,
                "actual_tokens": tokens,
                "median_ms": statistics.median(latencies),
                "latencies_ms": latencies,
            }
        return result

    stages.append({"stage": 0, "requests_consumed": 0, "probes": probe(0)})
    print(json.dumps({"stage": 0, "probes": {k: round(v["median_ms"], 1) for k, v in stages[0]["probes"].items()}}), flush=True)

    consumed = 0
    while consumed < min(args.limit, len(all_samples)):
        batch = all_samples[consumed: consumed + args.chunk]
        for sample in batch:
            elapsed, response = post(url, sample["request"], args.timeout)
            chunk_rows.append({
                "index": consumed,
                "name": sample["name"],
                "elapsed_ms": elapsed,
                "input_tokens": response.get("usage", {}).get("input_tokens"),
            })
            consumed += 1
        probes = probe(consumed)
        stages.append({"stage": consumed, "requests_consumed": consumed, "probes": probes})
        print(json.dumps({"stage": consumed, "chunk_tokens": dict(sorted(Counter(
            row["input_tokens"] for row in chunk_rows[-len(batch):]).items())),
            "probes": {k: round(v["median_ms"], 1) for k, v in probes.items()}}), flush=True)

    report = {
        "base_url": args.base_url,
        "chunk": args.chunk,
        "limit": args.limit,
        "probe_names": PROBES,
        "stages": stages,
        "chunk_requests": chunk_rows,
    }
    args.output.write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps({"output": str(args.output), "requests": consumed}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
