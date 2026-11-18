#!/usr/bin/env python3
"""Replay frozen requests against the live service and compare with the saved run.

Used to establish that the frozen OpenDecisions B1 baseline is repeatable on the
service process that is still running, without touching model weights or the
request template.  Each request is sent unmodified; probabilities are compared
with the frozen response of the same sample.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.request
from pathlib import Path


def post(url, payload, timeout):
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def probabilities(answer):
    if "probabilities" in answer:
        return answer["probabilities"]
    p_yes = answer["noul"]
    return {"yes": p_yes, "no": 1.0 - p_yes}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npu-run", type=Path, required=True,
                        help="frozen benchmark JSON with samples and responses")
    parser.add_argument("--sample-ids", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True,
                        help="OpenDecisions JSONL; maps data ids to od-NNNN sample names")
    parser.add_argument("--url", default="http://127.0.0.1:8000/v1/systemone")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    wanted = json.loads(Path(args.sample_ids).read_text(encoding="utf-8"))
    run = json.loads(Path(args.npu_run).read_text(encoding="utf-8"))
    samples = {item["name"]: item for item in run["samples"]}
    measured = {item["sample_name"]: item for item in run["requests"]}
    ordered_ids = []
    with open(args.data, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                ordered_ids.append(json.loads(line)["id"])
    by_id = {data_id: f"od-{index:04d}" for index, data_id in enumerate(ordered_ids)}
    report = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "url": args.url,
        "npu_run": str(args.npu_run),
        "npu_run_sha256": hashlib.sha256(Path(args.npu_run).read_bytes()).hexdigest(),
        "repeats": args.repeats,
        "cases": {},
        "passed": True,
    }
    for sample_id in wanted:
        # The benchmark names samples od-NNNN; accept either name or data id.
        name = sample_id if sample_id in samples else by_id.get(sample_id)
        if name is None:
            report["cases"][sample_id] = {"error": "unknown sample"}
            report["passed"] = False
            continue
        request = samples[name]["request"]
        frozen = probabilities(measured[name]["response"]["answers"]["answer"])
        entry = {"name": name, "frozen": frozen, "attempts": []}
        for attempt in range(args.repeats):
            started = time.perf_counter()
            try:
                response = post(args.url, request, args.timeout)
            except Exception as exc:  # noqa: BLE001 - record every failure
                entry["attempts"].append({"error": f"{type(exc).__name__}: {exc}"})
                report["passed"] = False
                continue
            elapsed = time.perf_counter() - started
            actual = probabilities(response["answers"]["answer"])
            keys = sorted(set(actual) | set(frozen))
            differences = {
                key: abs(actual.get(key, 0.0) - frozen.get(key, 0.0)) for key in keys
            }
            record = {
                "elapsed_ms": elapsed * 1000.0,
                "max_prob_diff": max(differences.values()),
                "exact": all(value == 0.0 for value in differences.values()),
                "input_tokens": response["usage"]["input_tokens"],
                "frozen_input_tokens": measured[name]["actual_input_tokens"],
            }
            entry["attempts"].append(record)
            report["passed"] = report["passed"] and record["exact"]
        report["cases"][sample_id] = entry
    report["summary"] = {
        "cases": len(report["cases"]),
        "attempts": sum(len(entry["attempts"]) for entry in report["cases"].values()
                        if "attempts" in entry),
        "exact_attempts": sum(1 for entry in report["cases"].values()
                              for attempt in entry.get("attempts", [])
                              if attempt.get("exact")),
        "max_prob_diff": max(
            (attempt.get("max_prob_diff", 0.0)
             for entry in report["cases"].values()
             for attempt in entry.get("attempts", [])),
            default=None,
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    print(json.dumps({"passed": report["passed"], **report["summary"]}, ensure_ascii=False))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
