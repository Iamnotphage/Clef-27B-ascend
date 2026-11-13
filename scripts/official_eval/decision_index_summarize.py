#!/usr/bin/env python3
"""Summarize native DI results without changing its scorers or aggregation."""

import argparse
import collections
import datetime
import json
from pathlib import Path


def summarize(directory, context_limit):
    latest = {}
    result_rows = 0
    with (directory / "results.jsonl").open() as source:
        for line in source:
            if not line.endswith("\n"):
                continue
            row = json.loads(line)
            result_rows += 1
            raw = row.get("raw_output") or row.get("response") or {}
            usage = raw.get("usage") or {}
            latest[row["run_id"]] = {
                "status": row["status"],
                "catalog_id": row["catalog_id"],
                "dataset": row.get("dataset"),
                "input_tokens": usage.get("input_tokens"),
                "wall_ms": row.get("total_wall_ms"),
                "error": row.get("error"),
            }
    by_catalog = {}
    tokens = []
    for row in latest.values():
        key = str(row["catalog_id"])
        bucket = by_catalog.setdefault(key, {
            "dataset": row["dataset"], "statuses": collections.Counter(),
            "input_tokens_at_context_limit": 0,
        })
        bucket["statuses"][row["status"]] += 1
        value = row["input_tokens"]
        if isinstance(value, int):
            tokens.append(value)
            bucket["input_tokens_at_context_limit"] += value == context_limit
    tokens.sort()
    scores_path = directory / "scores.json"
    native = json.loads(scores_path.read_text()) if scores_path.exists() else None
    return {
        "created_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "source_directory": str(directory),
        "result_rows_including_retries": result_rows,
        "unique_requests": len(latest),
        "latest_status_counts": dict(collections.Counter(row["status"] for row in latest.values())),
        "context_limit": context_limit,
        "input_tokens": {
            "observed": len(tokens),
            "min": tokens[0] if tokens else None,
            "p50": tokens[(len(tokens) - 1) // 2] if tokens else None,
            "max": tokens[-1] if tokens else None,
            "at_context_limit": sum(value == context_limit for value in tokens),
            "interpretation": "A saturated input length is an observation; exact state truncation requires comparing the original encoded input.",
        },
        "by_catalog": by_catalog,
        "native_scores": native,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--context-limit", type=int, default=4096)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = summarize(args.directory, args.context_limit)
    if args.output:
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({
        "unique_requests": report["unique_requests"],
        "latest_status_counts": report["latest_status_counts"],
        "input_tokens": report["input_tokens"],
        "native_complete": (report["native_scores"] or {}).get("complete"),
        "decision_index": (report["native_scores"] or {}).get("decision_index"),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
