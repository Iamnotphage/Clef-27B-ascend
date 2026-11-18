#!/usr/bin/env python3
"""Rescore ERQA predictions offline and verify the frozen request protocol.

Independent of the runner's summary code: it reads ``predictions.jsonl``,
applies the official exact-match rule, recomputes every aggregate, and (with
``--verify-requests``) rebuilds each request from the fixed dataset to confirm
the saved structural hash matches the protocol that produced the prediction.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys

from erqa_protocol import (
    OFFICIAL_RECORDS, build_request, load_samples, parse_options, request_summary,
)


def official_correct(prediction, gold):
    if prediction is None:
        return False
    return str(prediction).replace(".", "").strip().lower() == str(gold).strip().lower()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--verify-requests", action="store_true")
    parser.add_argument("--allow-subset", action="store_true",
                        help="smoke 子集允许缺少固定快照中的其他样本")
    args = parser.parse_args()

    run_dir = args.run_dir.expanduser().resolve()
    predictions = [
        json.loads(line) for line in
        (run_dir / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    samples, manifest = load_samples(args.data_dir)
    by_id = {sample["id"]: sample for sample in samples}
    ids = [row["id"] for row in predictions]
    duplicates = [key for key, count in Counter(ids).items() if count > 1]
    unknown = sorted(set(ids) - set(by_id))
    missing = sorted(set(by_id) - set(ids))
    if duplicates or unknown or (missing and not args.allow_subset):
        raise SystemExit(json.dumps({
            "duplicates": duplicates[:10], "unknown": unknown[:10],
            "missing": missing[:10]}, ensure_ascii=False))
    denominator = len(predictions) if args.allow_subset else OFFICIAL_RECORDS

    correct = 0
    groups = defaultdict(lambda: {"total": 0, "correct": 0, "failed": 0})
    request_mismatch = []
    for row in predictions:
        sample = by_id[row["id"]]
        gold = sample["answer"]
        is_correct = official_correct(row.get("prediction"), gold)
        correct += int(is_correct)
        if row.get("status") != "ok":
            groups["status:failed"]["total"] += 1
            groups["status:failed"]["failed"] += 1
            groups["status:failed"]["correct"] += 0
        for kind, key in (
            ("type", sample["question_type"]),
            ("images", "single" if sample["image_count"] == 1 else "multi"),
            ("options", str(len(parse_options(sample["question"])))),
            ("visual", sample["visual_index_pattern"]),
        ):
            bucket = groups[f"{kind}:{key}"]
            bucket["total"] += 1
            bucket["correct"] += int(is_correct)
            bucket["failed"] += int(row.get("status") != "ok")
        if args.verify_requests:
            request = build_request(sample, args.data_dir, model=row.get("model", "clef"))
            summary = request_summary(request, sample)
            structural = hashlib.sha256(json.dumps(
                {**summary, "images": summary["image_sha256"]},
                ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
            if structural != row.get("request_structural_sha256"):
                request_mismatch.append(row["id"])

    report = {
        "run_dir": str(run_dir),
        "data": {"tfrecord_sha256": manifest["source"]["sha256"]},
        "coverage": {
            "fixed_total": denominator, "predictions": len(predictions),
            "ok": sum(1 for row in predictions if row.get("status") == "ok"),
            "failed": sum(1 for row in predictions if row.get("status") != "ok"),
        },
        "official_rule": "prediction.replace('.', '').strip().lower() == gold.strip().lower()",
        "accuracy": {"correct": correct, "total": denominator,
                     "value": correct / denominator if denominator else None},
        "breakdown": {
            key: {**value, "accuracy": value["correct"] / value["total"] if value["total"] else None}
            for key, value in sorted(groups.items())
        },
        "request_verification": {
            "checked": bool(args.verify_requests),
            "mismatches": request_mismatch,
            "passed": not request_mismatch,
        },
    }
    summary_path = run_dir / "summary.json"
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text())
        report["summary_comparison"] = {
            "summary_accuracy": summary["accuracy"]["value"],
            "offline_accuracy": report["accuracy"]["value"],
            "equal": summary["accuracy"]["correct"] == correct,
        }
    text = json.dumps(report, ensure_ascii=False, indent=1) + "\n"
    if args.output:
        args.output.write_text(text)
    print(json.dumps({
        "accuracy": report["accuracy"], "coverage": report["coverage"],
        "request_verification": report["request_verification"],
        "summary_comparison": report.get("summary_comparison"),
    }, ensure_ascii=False))
    return 0 if not request_mismatch else 1


if __name__ == "__main__":
    sys.exit(main())
