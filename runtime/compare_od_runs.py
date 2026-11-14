#!/usr/bin/env python3
"""Compare two full OpenDecisions benchmark reports bit for bit.

The switch must be numerically invisible: every probability of the OFF run
must equal the ON run, and both must equal the frozen service baseline used by
the submission.  Any difference is reported with its sample and magnitude.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def answer_probabilities(record):
    answer = record["response"]["answers"]["answer"]
    if "probabilities" in answer:
        return dict(answer["probabilities"]), answer.get("choice")
    value = answer["noul"]
    return {"yes": value, "no": 1.0 - value}, None


def frozen_probabilities(record):
    answer = record["response"]["answers"]["answer"]
    if "probabilities" in answer:
        return dict(answer["probabilities"]), answer.get("choice")
    value = answer["noul"]
    return {"yes": value, "no": 1.0 - value}, None


def compare_pair(left, right):
    keys = sorted(set(left) | set(right))
    differences = {key: abs(left.get(key, 0.0) - right.get(key, 0.0)) for key in keys}
    return max(differences.values()) if differences else 0.0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--off", type=Path, required=True)
    parser.add_argument("--on", type=Path, required=True)
    parser.add_argument("--frozen", type=Path, default=None,
                        help="s1-final-full.json from the frozen baseline")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    off = load(args.off)["requests"]
    on = load(args.on)["requests"]
    frozen = None
    if args.frozen:
        document = load(args.frozen)
        frozen = {
            item["sample_name"]: item for item in document["requests"]
            if item.get("phase") == "measured"
        }

    if len(off) != len(on):
        raise SystemExit(f"request count differs: off={len(off)} on={len(on)}")

    items = []
    for left, right in zip(off, on):
        if left["status"] != "ok" or right["status"] != "ok":
            items.append({"index": left["index"], "name": left["name"],
                          "status": "error"})
            continue
        off_probs, off_choice = answer_probabilities(left)
        on_probs, on_choice = answer_probabilities(right)
        difference = compare_pair(off_probs, on_probs)
        item = {
            "index": left["index"],
            "name": left["name"],
            "status": "ok",
            "max_prob_diff": difference,
            "same": difference == 0.0,
            "input_tokens_off": left.get("input_tokens"),
            "input_tokens_on": right.get("input_tokens"),
            "latency_ms_off": left["elapsed_ms"],
            "latency_ms_on": right["elapsed_ms"],
        }
        if frozen is not None:
            base = frozen.get(left["name"])
            if base is not None:
                frozen_probs, frozen_choice = frozen_probabilities(base)
                item["max_prob_diff_vs_frozen"] = compare_pair(off_probs, frozen_probs)
                item["max_prob_diff_on_vs_frozen"] = compare_pair(on_probs, frozen_probs)
                item["choice_off_vs_frozen"] = off_choice != frozen_choice
                item["choice_on_vs_frozen"] = on_choice != frozen_choice
        items.append(item)

    valid = [item for item in items if item["status"] == "ok"]
    summary = {
        "requests": len(items),
        "compared": len(valid),
        "errors": len(items) - len(valid),
        "bitwise_identical": sum(1 for item in valid if item["same"]),
        "differing": sum(1 for item in valid if not item["same"]),
        "max_prob_diff_off_vs_on": max((item["max_prob_diff"] for item in valid), default=None),
        "token_counts_equal": sum(
            1 for item in valid if item["input_tokens_off"] == item["input_tokens_on"]
        ),
        "largest": sorted(
            (
                {
                    "name": item["name"],
                    "max_prob_diff": item["max_prob_diff"],
                    "max_prob_diff_vs_frozen": item.get("max_prob_diff_vs_frozen"),
                }
                for item in valid
            ),
            key=lambda row: -(row["max_prob_diff"] or 0.0),
        )[:10],
    }
    if frozen is not None:
        summary["frozen_samples"] = sum(1 for item in valid if "max_prob_diff_vs_frozen" in item)
        summary["off_vs_frozen_max"] = max(
            (item["max_prob_diff_vs_frozen"] for item in valid
             if "max_prob_diff_vs_frozen" in item), default=None)
        summary["on_vs_frozen_max"] = max(
            (item["max_prob_diff_on_vs_frozen"] for item in valid
             if "max_prob_diff_on_vs_frozen" in item), default=None)
        summary["choice_changes_off_vs_frozen"] = sum(
            1 for item in valid if item.get("choice_off_vs_frozen"))
        summary["choice_changes_on_vs_frozen"] = sum(
            1 for item in valid if item.get("choice_on_vs_frozen"))
    report = {"off": str(args.off), "on": str(args.on),
              "frozen": str(args.frozen) if args.frozen else None,
              "summary": summary, "items": items}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    raise SystemExit(main())
