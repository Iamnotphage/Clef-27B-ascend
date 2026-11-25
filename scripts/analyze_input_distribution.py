#!/usr/bin/env python3
"""Summarize input features and latency by length bucket.

Inputs are the native benchmark reports, not a re-tokenization:
- OpenDecisions: od_full_benchmark JSON (or .json.gz), current fusion run.
- ERQA: predictions.jsonl from the official 400-question run.

The output is JSON so the report can quote exact numbers and the measurement
can be reproduced without the model service.
"""

from __future__ import annotations

import argparse
import gzip
import json
import statistics
from collections import Counter
from pathlib import Path


def percentile(values, fraction):
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * fraction
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)


def distribution(values):
    values = [v for v in values if v is not None]
    if not values:
        return {"count": 0}
    return {
        "count": len(values),
        "min": min(values),
        "p10": percentile(values, 0.10),
        "p25": percentile(values, 0.25),
        "p50": percentile(values, 0.50),
        "p75": percentile(values, 0.75),
        "p90": percentile(values, 0.90),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "max": max(values),
        "mean": statistics.mean(values),
    }


def bins(values, edges):
    """Bucket rows by a lower-inclusive, upper-exclusive edge list."""
    rows = []
    for i, lower in enumerate(edges):
        upper = edges[i + 1] if i + 1 < len(edges) else None
        selected = [
            row for row in values
            if row >= lower and (upper is None or row < upper)
        ]
        rows.append({"lower": lower, "upper": upper, "count": len(selected)})
    return rows


def summarize_rows(rows, token_key="tokens", latency_key="latency_ms", edges=None):
    latencies = [row[latency_key] for row in rows if row.get(latency_key) is not None]
    summary = {
        "requests": len(rows),
        "tokens": distribution([row[token_key] for row in rows]),
        "latency_ms": distribution(latencies),
        "time_share_by_token_bucket": [],
        "latency_by_token_bucket": [],
    }
    if edges:
        total_time = sum(latencies) or 1.0
        for bucket in bins([row[token_key] for row in rows], edges):
            selected = [
                row for row in rows
                if row[token_key] >= bucket["lower"]
                and (bucket["upper"] is None or row[token_key] < bucket["upper"])
            ]
            bucket_latency = [
                row[latency_key] for row in selected
                if row.get(latency_key) is not None
            ]
            summary["time_share_by_token_bucket"].append({
                **bucket,
                "share": bucket["count"] / len(rows) if rows else 0.0,
                "time_share": sum(bucket_latency) / total_time,
            })
            summary["latency_by_token_bucket"].append({
                **bucket,
                "latency_ms": distribution(bucket_latency),
            })
    return summary


def load_json(path: Path):
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            return json.load(handle)
    return json.loads(path.read_text(encoding="utf-8"))


def open_decisions(path: Path):
    report = load_json(path)
    rows = []
    field_counts = Counter()
    candidate_counts = Counter()
    question_types = Counter()
    for item in report.get("requests", []):
        if item.get("status") != "ok" and item.get("http_status") != 200:
            continue
        request = item.get("request") or {}
        questions = request.get("questions") or {}
        fields = len(questions)
        candidates = 0
        qtypes = Counter()
        for question in questions.values():
            kind = question.get("type")
            qtypes[kind] += 1
            question_types[kind] += 1
            if kind == "noul":
                candidates += 2
            elif kind == "choice":
                candidates += len(question.get("criteria") or {})
            elif kind == "score":
                candidates += len(question.get("criteria") or [])
        field_counts[fields] += 1
        candidate_counts[candidates] += 1
        rows.append({
            "tokens": item.get("input_tokens"),
            "latency_ms": item.get("elapsed_ms"),
            "fields": fields,
            "candidates": candidates,
            "question_types": dict(qtypes),
        })
    return rows, field_counts, candidate_counts, question_types


def erqa(path: Path):
    rows = []
    image_counts = Counter()
    option_counts = Counter()
    question_types = Counter()
    visual_patterns = Counter()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        if item.get("status") != "ok":
            continue
        rows.append({
            "tokens": item.get("input_tokens"),
            "latency_ms": item.get("latency_ms"),
            "images": item.get("image_count"),
            "options": item.get("option_count"),
            "question_type": item.get("question_type"),
            "visual_index_pattern": item.get("visual_index_pattern"),
        })
        image_counts[item.get("image_count")] += 1
        option_counts[item.get("option_count")] += 1
        question_types[item.get("question_type")] += 1
        visual_patterns[item.get("visual_index_pattern")] += 1
    return rows, image_counts, option_counts, question_types, visual_patterns


def image_bucket_summary(rows):
    edges = [1, 2, 3, 5, 9, 17]
    out = []
    for i, lower in enumerate(edges[:-1]):
        upper = edges[i + 1]
        selected = [row for row in rows if lower <= (row["images"] or 0) < upper]
        out.append({
            "lower": lower,
            "upper_exclusive": upper,
            "count": len(selected),
            "latency_ms": distribution([row["latency_ms"] for row in selected]),
        })
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--od-report", type=Path,
        default=Path("logs/gdn-data-path-20261115/qkv-norm-entry5/od-full.json.gz"),
    )
    parser.add_argument(
        "--erqa-predictions", type=Path,
        default=Path("logs/gdn-data-path-20261115/qkv-norm-erqa/full/predictions.jsonl"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("logs/input-distribution-2026-11-16.json"),
    )
    args = parser.parse_args()

    od_rows, od_fields, od_candidates, od_types = open_decisions(args.od_report)
    erqa_rows, erqa_images, erqa_options, erqa_types, erqa_visual = erqa(args.erqa_predictions)
    token_edges = [0, 256, 384, 512, 768, 969, 1024, 1200, 1595, 2048, 10294]

    result = {
        "open_decisions": {
            "path": str(args.od_report),
            **summarize_rows(od_rows, edges=token_edges),
            "field_counts": dict(sorted(od_fields.items())),
            "candidate_counts": dict(sorted(od_candidates.items())),
            "question_types": dict(sorted(od_types.items())),
            "cap_1200": {
                "covered": sum(1 for row in od_rows if row["tokens"] <= 1200),
                "total": len(od_rows),
                "share": sum(1 for row in od_rows if row["tokens"] <= 1200) / len(od_rows) if od_rows else 0,
            },
        },
        "erqa": {
            "path": str(args.erqa_predictions),
            **summarize_rows(erqa_rows, edges=token_edges),
            "image_counts": dict(sorted(erqa_images.items())),
            "option_counts": dict(sorted(erqa_options.items())),
            "question_types": dict(sorted(erqa_types.items())),
            "visual_index_patterns": dict(sorted(erqa_visual.items())),
            "image_buckets": image_bucket_summary(erqa_rows),
            "cap_1200": {
                "covered": sum(1 for row in erqa_rows if row["tokens"] <= 1200),
                "total": len(erqa_rows),
                "share": sum(1 for row in erqa_rows if row["tokens"] <= 1200) / len(erqa_rows) if erqa_rows else 0,
            },
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({
        "od_requests": len(od_rows),
        "od_token_p95": result["open_decisions"]["tokens"]["p95"],
        "erqa_requests": len(erqa_rows),
        "erqa_token_p95": result["erqa"]["tokens"]["p95"],
        "erqa_cap_1200_share": result["erqa"]["cap_1200"]["share"],
        "output": str(args.output),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
