#!/usr/bin/env python3
"""Compare two ERQA runs by sample ID using the official exact-match rule.

Both inputs are the predictions.jsonl files written by erqa_baseline.py.
The full official dataset is required by default. Probability differences are
diagnostics, and every failed request remains in the accuracy denominator.
Run erqa_score.py --verify-requests separately to verify each saved request
against the image files and frozen dataset protocol.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import sys

from erqa_protocol import load_samples, parse_options
from erqa_score import official_correct


def file_identity(path):
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def load_predictions(path, samples, allow_subset):
    with path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    ids = [row["id"] for row in rows]
    duplicates = sorted(key for key, count in Counter(ids).items() if count > 1)
    unknown = sorted(set(ids) - set(samples))
    missing = sorted(set(samples) - set(ids))
    if not rows or duplicates or unknown or (missing and not allow_subset):
        raise ValueError(f"{path}: invalid coverage, duplicates={duplicates}, "
                         f"unknown={unknown}, missing={missing}")
    for row in rows:
        if row.get("status") != "ok":
            continue
        sample = samples[row["id"]]
        labels = {label for label, _ in parse_options(sample["question"])}
        probabilities = row.get("probabilities")
        if not isinstance(probabilities, dict) or set(probabilities) != labels:
            raise ValueError(f"{path}: {row['id']} probability labels disagree with dataset")
        values = list(probabilities.values())
        if not all(isinstance(value, (int, float)) and not isinstance(value, bool)
                   and math.isfinite(value) and value >= 0 for value in values):
            raise ValueError(f"{path}: {row['id']} contains invalid probabilities")
        if abs(sum(values) - 1.0) > 0.01:
            raise ValueError(f"{path}: {row['id']} probabilities sum to {sum(values)}")
        if row.get("prediction") not in labels:
            raise ValueError(f"{path}: {row['id']} prediction is outside dataset labels")
    return {row["id"]: row for row in rows}


def distribution(values):
    if not values:
        return {"count": 0}
    ordered = sorted(values)
    result = {"count": len(values), "max": max(values), "mean": sum(values) / len(values)}
    for name, fraction in (("p50", 0.5), ("p90", 0.9), ("p95", 0.95), ("p99", 0.99)):
        position = (len(ordered) - 1) * fraction
        lower, upper = math.floor(position), math.ceil(position)
        result[name] = ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)
    return result


def summarize(items):
    reference_correct = sum(item["reference_correct"] for item in items)
    candidate_correct = sum(item["candidate_correct"] for item in items)
    paired = [item for item in items if item["max_prob_diff"] is not None]
    return {
        "items": len(items),
        "reference_correct": reference_correct,
        "candidate_correct": candidate_correct,
        "reference_accuracy": reference_correct / len(items),
        "candidate_accuracy": candidate_correct / len(items),
        "accuracy_difference": (candidate_correct - reference_correct) / len(items),
        "reference_failed": sum(item["reference_status"] != "ok" for item in items),
        "candidate_failed": sum(item["candidate_status"] != "ok" for item in items),
        "prediction_changes": sum(item["prediction_changed"] for item in items),
        "both_correct": sum(item["reference_correct"] and item["candidate_correct"] for item in items),
        "correct_to_incorrect": sum(item["reference_correct"] and not item["candidate_correct"] for item in items),
        "incorrect_to_correct": sum(not item["reference_correct"] and item["candidate_correct"] for item in items),
        "both_incorrect": sum(not item["reference_correct"] and not item["candidate_correct"] for item in items),
        "paired_successful": len(paired),
        "exact_equal_probability_items": sum(item["max_prob_diff"] == 0 for item in paired),
        "max_prob_diff": distribution([item["max_prob_diff"] for item in paired]),
        "tvd": distribution([item["tvd"] for item in paired]),
        "prob_diff_over": {
            str(threshold): sum(item["max_prob_diff"] > threshold for item in paired)
            for threshold in (0.001, 0.01, 0.03, 0.05, 0.1)
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-subset", action="store_true",
                        help="Compare matching smoke subsets and use their size as denominator")
    args = parser.parse_args()

    samples, manifest = load_samples(args.data_dir)
    by_id = {sample["id"]: sample for sample in samples}
    reference = load_predictions(args.reference, by_id, args.allow_subset)
    candidate = load_predictions(args.candidate, by_id, args.allow_subset)
    if set(reference) != set(candidate):
        raise ValueError("reference and candidate sample ID sets differ")

    items = []
    for sample in samples:
        sample_id = sample["id"]
        if sample_id not in reference:
            continue
        left, right = reference[sample_id], candidate[sample_id]
        paired_ok = left.get("status") == "ok" and right.get("status") == "ok"
        probabilities_left = left.get("probabilities")
        probabilities_right = right.get("probabilities")
        differences = []
        if paired_ok:
            differences = [abs(probabilities_left[label] - probabilities_right[label])
                           for label in probabilities_left]
        left_request = left.get("request_structural_sha256")
        right_request = right.get("request_structural_sha256")
        items.append({
            "id": sample_id,
            "record_index": sample["record_index"],
            "question_type": sample["question_type"],
            "image_count": sample["image_count"],
            "option_count": len(parse_options(sample["question"])),
            "visual_index_pattern": sample["visual_index_pattern"],
            "gold": sample["answer"],
            "reference_status": left.get("status"),
            "candidate_status": right.get("status"),
            "reference_prediction": left.get("prediction"),
            "candidate_prediction": right.get("prediction"),
            "reference_correct": left.get("status") == "ok" and official_correct(left.get("prediction"), sample["answer"]),
            "candidate_correct": right.get("status") == "ok" and official_correct(right.get("prediction"), sample["answer"]),
            "prediction_changed": left.get("prediction") != right.get("prediction"),
            "reference_probabilities": probabilities_left,
            "candidate_probabilities": probabilities_right,
            "max_prob_diff": max(differences) if differences else None,
            "tvd": sum(differences) / 2 if differences else None,
            "reference_input_tokens": left.get("input_tokens"),
            "candidate_input_tokens": right.get("input_tokens"),
            "input_tokens_equal": paired_ok and left.get("input_tokens") is not None
                and left.get("input_tokens") == right.get("input_tokens"),
            "request_hash_equal": bool(left_request and right_request and left_request == right_request),
        })

    breakdown = {}
    for field in ("question_type", "image_count", "option_count", "visual_index_pattern"):
        groups = defaultdict(list)
        for item in items:
            groups[str(item[field])].append(item)
        breakdown[field] = {key: summarize(group) for key, group in sorted(groups.items())}
    request_mismatches = [item["id"] for item in items if not item["request_hash_equal"]]
    token_mismatches = [item["id"] for item in items
                        if item["max_prob_diff"] is not None and not item["input_tokens_equal"]]
    report = {
        "reference": file_identity(args.reference),
        "candidate": file_identity(args.candidate),
        "dataset": {
            "path": str(args.data_dir.resolve()),
            "tfrecord_sha256": manifest["source"]["sha256"],
            "scope": "subset" if args.allow_subset else "full",
        },
        "official_rule": "prediction.replace('.', '').strip().lower() == gold.strip().lower()",
        "probability_threshold_role": "diagnostic statistics",
        "summary": summarize(items),
        "alignment": {
            "request_hash_mismatches": request_mismatches,
            "input_token_mismatches": token_mismatches,
            "paired_input_token_matches": sum(item["input_tokens_equal"] for item in items),
            "passed": not request_mismatches and not token_mismatches,
        },
        "breakdown": breakdown,
        "items": items,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"summary": report["summary"], "alignment": report["alignment"],
                      "output": str(args.output)}, ensure_ascii=False))
    return 0 if report["alignment"]["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
