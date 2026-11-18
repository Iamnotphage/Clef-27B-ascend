#!/usr/bin/env python3
"""Fixed-input client for the joint-head service A/B.

Extracts one frozen OpenDecisions sample, validates the canonical body hash,
and sends it repeatedly to the already-running service.  It has no NPU import
and does not start or stop the service.  Every response is kept and checked for
the expected answer type; only valid responses enter the latency sample.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import time
import urllib.error
import urllib.request
from pathlib import Path

EXPECTED_SHA = "568ea1583377dcc6044d3fcd4a099c3beb4cd0e118155f95b595bda72b1c76bf"


def percentile(values, fraction):
    ordered = sorted(values)
    if not ordered:
        return None
    idx = (len(ordered) - 1) * fraction
    lo, hi = math.floor(idx), math.ceil(idx)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (idx - lo)


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=("/workspace/yangchen/logs/tuning/"
                                       "20261114-013000-fix4/opendecisions/all2352.json"))
    ap.add_argument("--name", default="od-1534")
    ap.add_argument("--url", default="http://127.0.0.1:18143/v1/systemone")
    ap.add_argument("--expect-sha256", default=EXPECTED_SHA)
    ap.add_argument("--expect-type", default="score")
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
    if digest != args.expect_sha256:
        raise SystemExit(f"input body hash mismatch: {digest} != {args.expect_sha256}")

    rows = []
    for index in range(args.warmup + args.iters):
        started_at = now()
        begin = time.perf_counter()
        status, response, error = None, None, None
        try:
            request = urllib.request.Request(
                args.url, data=body, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(request, timeout=args.timeout) as handle:
                status = handle.status
                response = json.loads(handle.read())
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        elapsed = (time.perf_counter() - begin) * 1000.0
        answer = ((response or {}).get("answers") or {}).get("answer") or {}
        answer_type = answer.get("type")
        valid = (status == 200 and error is None
                 and answer_type == args.expect_type)
        if index >= args.warmup:
            rows.append({"index": index - args.warmup, "started_at": started_at,
                         "finished_at": now(), "ms": elapsed, "status": status,
                         "answer_type": answer_type, "valid": valid,
                         "error": error, "response": response})

    valid_rows = [row for row in rows if row["valid"]]
    latencies = [row["ms"] for row in valid_rows]
    report = {
        "schema": "joint-fused-attention-client/2",
        "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "data": args.data, "name": args.name, "url": args.url,
        "body_sha256": digest, "expected_sha256": args.expect_sha256,
        "body_bytes": len(body), "expect_type": args.expect_type,
        "warmup": args.warmup, "iters": args.iters,
        "n": len(latencies), "failures": len(rows) - len(valid_rows),
        "mean_ms": statistics.fmean(latencies) if latencies else None,
        "median_ms": statistics.median(latencies) if latencies else None,
        "p95_ms": percentile(latencies, 0.95) if latencies else None,
        "min_ms": min(latencies) if latencies else None,
        "max_ms": max(latencies) if latencies else None,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n")
    print(json.dumps({k: report[k] for k in ("name", "body_sha256", "n", "failures",
                                             "mean_ms", "median_ms", "p95_ms")},
                     ensure_ascii=False, indent=1))
    return 0 if latencies and not report["failures"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
