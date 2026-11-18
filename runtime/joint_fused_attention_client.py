#!/usr/bin/env python3
"""Fixed-input client for the joint-head service A/B.

Extracts one frozen OpenDecisions sample, hashes the exact JSON body, and
sends it repeatedly to the already-running service.  It has no NPU import and
does not start or stop the service.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import time
import urllib.request
from pathlib import Path


def percentile(values, fraction):
    ordered = sorted(values)
    if not ordered:
        return None
    idx = (len(ordered) - 1) * fraction
    lo, hi = math.floor(idx), math.ceil(idx)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (idx - lo)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=("/workspace/yangchen/logs/tuning/"
                                       "20261114-013000-fix4/opendecisions/all2352.json"))
    ap.add_argument("--name", default="od-1534")
    ap.add_argument("--url", default="http://127.0.0.1:18143/v1/systemone")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--timeout", type=float, default=180.0)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    doc = json.loads(Path(args.data).read_text())
    sample = next(s for s in doc["samples"] if s.get("name") == args.name)
    payload = sample["request"]
    body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()
    digest = hashlib.sha256(body).hexdigest()
    rows = []
    for index in range(args.warmup + args.iters):
        request = urllib.request.Request(
            args.url, data=body, headers={"Content-Type": "application/json"})
        begin = time.perf_counter()
        with urllib.request.urlopen(request, timeout=args.timeout) as response:
            response.read()
            status = response.status
        elapsed = (time.perf_counter() - begin) * 1000.0
        if index >= args.warmup:
            rows.append({"ms": elapsed, "status": status})
    latencies = [row["ms"] for row in rows]
    report = {
        "schema": "joint-fused-attention-client/1",
        "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "data": args.data, "name": args.name, "url": args.url,
        "body_sha256": digest, "body_bytes": len(body),
        "warmup": args.warmup, "iters": args.iters,
        "n": len(latencies),
        "mean_ms": statistics.fmean(latencies),
        "median_ms": statistics.median(latencies),
        "p95_ms": percentile(latencies, 0.95),
        "min_ms": min(latencies), "max_ms": max(latencies),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n")
    print(json.dumps({k: report[k] for k in ("name", "body_sha256", "n", "mean_ms",
                                             "median_ms", "p95_ms")}, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
