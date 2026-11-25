#!/usr/bin/env python3
"""Summarize one ``runtime/tiered_bare_acceptance.sh`` output directory.

Prints a markdown table for the OD S1 / C16 / C64 runs and the ERQA S1 / C16
runs, plus the final store counters, so the report numbers come straight from
the artifacts instead of being transcribed by hand.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def od_row(path: Path):
    data = json.loads(path.read_text())
    summary = data["summary"]
    return {
        "run": path.stem,
        "decisions_per_second": summary["decisions_per_second"],
        "p50": summary["latency_ms"]["p50"],
        "p95": summary["latency_ms"]["p95"],
        "failed": summary["failed"],
    }


def erqa_row(directory: Path):
    summary = json.loads((directory / "summary.json").read_text())
    latency = summary.get("latency_ms") or {}
    throughput = summary.get("throughput") or {}
    wall = None
    predictions = directory / "predictions.jsonl"
    if predictions.is_file():
        started, finished = [], []
        for line in predictions.read_text().splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("started_at") and record.get("finished_at"):
                started.append(record["started_at"])
                finished.append(record["finished_at"])
        if started and finished:
            from datetime import datetime

            parse = datetime.fromisoformat
            wall = (parse(max(finished)) - parse(min(started))).total_seconds()
    return {
        "run": directory.name,
        "correct": summary["accuracy"]["correct"],
        "total": summary["accuracy"]["total"],
        "accuracy": summary["accuracy"]["value"],
        "mean": latency.get("mean"),
        "p50": latency.get("p50"),
        "p95": latency.get("p95"),
        "decisions_per_second": throughput.get("decisions_per_second"),
        "wall_seconds": wall,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    root = args.directory

    print("| run | dec/s | p50 (ms) | p95 (ms) | failed |")
    print("| --- | ---: | ---: | ---: | ---: |")
    for name in ("od-s1-r1", "od-s1-r2", "od-c16-r1", "od-c16-r2",
                 "od-c64-r1", "od-c64-r2"):
        path = root / f"{name}.json"
        if not path.is_file():
            continue
        row = od_row(path)
        print(f"| {row['run']} | {row['decisions_per_second']:.3f} | "
              f"{row['p50']:.1f} | {row['p95']:.1f} | {row['failed']} |")

    print()
    print("| run | acc | correct | mean (ms) | p50 (ms) | p95 (ms) | dec/s |")
    print("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for name in ("erqa-s1-r1", "erqa-s1-r2", "erqa-c16-r1", "erqa-c16-r2"):
        directory = root / name
        if not (directory / "summary.json").is_file():
            continue
        row = erqa_row(directory)
        dec = row["decisions_per_second"]
        dec_text = f"{dec:.3f}" if dec is not None else "-"

        def number(value):
            return f"{value:.2f}" if value is not None else "-"

        print(f"| {row['run']} | {row['accuracy']:.4f} | "
              f"{row['correct']}/{row['total']} | {number(row['mean'])} | "
              f"{number(row['p50'])} | {number(row['p95'])} | {dec_text} |")

    counters = root / "counters.json"
    if counters.is_file():
        print()
        print("counters:", json.dumps(json.loads(counters.read_text()),
                                      sort_keys=True))


if __name__ == "__main__":
    main()
