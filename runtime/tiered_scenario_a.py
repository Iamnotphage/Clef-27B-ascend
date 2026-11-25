#!/usr/bin/env python3
"""Scenario A: one STATE, four SCHEMA variants (20 x 4).

The Tier 2 key covers only the tokens before the schema delimiter, so the
first request for a state populates the prefix and the three other schemas
must resume from the same captured state.  Three rounds are sent:

* R1 sequential, cold store (each state populates once, then its three other
  schemas hit);
* R2 sequential, warm store (pure hits);
* R3 sequential, warm store (repeat).

Serial sends keep every step a B1 request, so the three rounds must be
bitwise equal per (state, variant); a mismatch is cross-talk, not batch
composition.  Mixed-batch behaviour is covered by the OD/ERQA S2 runs.

Every response is kept so the driver can require R1 == R2 == R3 bitwise per
(state, variant) and zero cross-talk between states.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import math
import time
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
    body = json.dumps(payload, ensure_ascii=False, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")
    request = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.status, json.loads(response.read())


def build_requests(sample_file: Path, states: int, variants: int,
                   min_state_chars: int):
    document = json.loads(sample_file.read_text(encoding="utf-8"))
    samples = document["samples"] if isinstance(document, dict) else document
    usable = [sample for sample in samples
              if sample.get("request", {}).get("state")]
    # The engine's text minimum prefix is 512 tokens before the schema
    # delimiter; pick long states so every variant can hit.
    long_states = sorted(
        (sample for sample in usable
         if len(sample["request"]["state"]) >= min_state_chars),
        key=lambda sample: -len(sample["request"]["state"]))
    if len(long_states) < states:
        raise SystemExit(
            f"only {len(long_states)} states >= {min_state_chars} chars, "
            f"need {states}")
    bases = long_states[:states]
    requests = []
    for state_index, base_sample in enumerate(bases):
        base = base_sample["request"]
        for variant in range(variants):
            donor = usable[(state_index * variants + variant) % len(usable)]["request"]
            payload = {
                "model": base.get("model", "clef"),
                "state": base["state"],
                "questions": donor["questions"],
            }
            requests.append({
                "name": f"s{state_index:02d}-q{variant}",
                "state_index": state_index,
                "variant": variant,
                "request": payload,
            })
    return requests


def run_round(requests, url, concurrency, timeout):
    results = [None] * len(requests)

    def one(index):
        entry = {"name": requests[index]["name"], "index": index}
        begin = time.perf_counter()
        try:
            status, response = post(url, requests[index]["request"], timeout)
            entry.update(http_status=status, response=response, status="ok")
        except Exception as exc:  # noqa: BLE001 - keep every failure
            entry.update(status="error", errors=[f"{type(exc).__name__}: {exc}"])
        entry["elapsed_ms"] = (time.perf_counter() - begin) * 1000.0
        return entry

    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        for index, entry in zip(range(len(requests)),
                                pool.map(one, range(len(requests)))):
            results[index] = entry
    wall = time.perf_counter() - started
    latencies = [entry["elapsed_ms"] for entry in results if entry["status"] == "ok"]
    return {
        "wall_seconds": wall,
        "concurrency": concurrency,
        "requests": results,
        "summary": {
            "requested": len(requests),
            "successful": sum(1 for entry in results if entry["status"] == "ok"),
            "failed": sum(1 for entry in results if entry["status"] != "ok"),
            "decisions_per_second": (
                sum(1 for entry in results if entry["status"] == "ok") / wall
                if wall else None),
            "latency_ms": {
                "mean": sum(latencies) / len(latencies) if latencies else None,
                "p50": percentile(latencies, 0.50),
                "p95": percentile(latencies, 0.95),
            },
        },
    }


def answer_probabilities(entry):
    answer = entry["response"]["answers"]["answer"]
    if "probabilities" in answer:
        return answer["probabilities"]
    value = answer["noul"]
    return {"yes": value, "no": 1.0 - value}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-file", type=Path, required=True)
    parser.add_argument("--url", default="http://127.0.0.1:18143/v1/systemone")
    parser.add_argument("--states", type=int, default=20)
    parser.add_argument("--variants", type=int, default=4)
    parser.add_argument("--min-state-chars", type=int, default=2600,
                        help="text proxy for the engine's 512-token prefix floor")
    parser.add_argument("--concurrency", type=int, default=1,
                        help="client workers per round; 1 keeps every step B1")
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    requests = build_requests(args.sample_file, args.states, args.variants,
                              args.min_state_chars)
    report = {
        "created_at": now(),
        "url": args.url,
        "states": args.states,
        "variants": args.variants,
        "rounds": {},
        "self_check": {},
    }
    report["rounds"]["r1"] = run_round(requests, args.url, 1, args.timeout)
    report["rounds"]["r2"] = run_round(requests, args.url, args.concurrency,
                                       args.timeout)
    report["rounds"]["r3"] = run_round(requests, args.url, args.concurrency,
                                       args.timeout)
    for left, right in (("r1", "r2"), ("r1", "r3"), ("r2", "r3")):
        first = {entry["name"]: entry for entry in report["rounds"][left]["requests"]}
        current = {entry["name"]: entry for entry in report["rounds"][right]["requests"]}
        maximum = 0.0
        changes = []
        for name, entry in first.items():
            peer = current.get(name)
            if entry.get("status") != "ok" or not peer or peer.get("status") != "ok":
                changes.append(name)
                continue
            left_probs = answer_probabilities(entry)
            right_probs = answer_probabilities(peer)
            for key in set(left_probs) | set(right_probs):
                delta = abs(float(left_probs.get(key, 0.0))
                            - float(right_probs.get(key, 0.0)))
                maximum = max(maximum, delta)
                if delta != 0.0:
                    changes.append(name)
                    break
        report["self_check"][f"{left}_vs_{right}"] = {
            "changed": sorted(set(changes)), "max_prob_diff": maximum}
    report["finished_at"] = now()
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(report, ensure_ascii=False, allow_nan=False) + "\n")
    summary = {name: round_["summary"] for name, round_ in report["rounds"].items()}
    summary["self_check"] = report["self_check"]
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
