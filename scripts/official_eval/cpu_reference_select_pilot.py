#!/usr/bin/env python3
"""Freeze the CPU-reference pilot subsets with a recorded, deterministic rule.

Writes two files per selection: ``<stem>.ids.json`` (the plain id list the
runner consumes) and ``<stem>.report.json`` (the rule, per-sample attributes and
the coverage counters used to defend the choice).
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

OD_TYPES = ("choice", "noul", "score")
ERQA_CLASSES = (
    "Action Reasoning",
    "Multi-view Reasoning",
    "Other",
    "Pointing",
    "Spatial Reasoning",
    "State Estimation",
    "Task Reasoning",
    "Trajectory Reasoning",
)


def read_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def margin(probabilities):
    ordered = sorted(probabilities.values(), reverse=True)
    if len(ordered) < 2:
        return ordered[0] if ordered else 0.0
    return ordered[0] - ordered[1]


def argmax(probabilities):
    return max(probabilities, key=probabilities.get)


def select_opendecisions(args):
    tasks = read_jsonl(Path(args.data_dir) / "open-decisions2352.jsonl")
    run = json.loads(Path(args.npu_run).read_text(encoding="utf-8"))
    saved = run["samples"]
    measured = {item["sample_name"]: item for item in run["requests"]}
    if len(tasks) != len(saved):
        raise SystemExit("data and benchmark sample counts differ")
    known = set()
    if args.batching_diff:
        diff = json.loads(Path(args.batching_diff).read_text(encoding="utf-8"))
        for name, comparison in (diff.get("comparisons") or {}).items():
            for item in comparison.get("top_prob_diffs") or []:
                if item.get("choice_changed") or (item.get("max_prob_diff") or 0) >= 0.03:
                    known.add(item["id"])
    rows = []
    for index, (task, item) in enumerate(zip(tasks, saved)):
        response = measured[item["name"]]["response"]["answers"]["answer"]
        probabilities = response.get("probabilities")
        if probabilities is None:
            p_yes = response["noul"]
            probabilities = {"yes": p_yes, "no": 1.0 - p_yes}
        kind = task["question"]["type"]
        correct = None
        if kind == "choice":
            correct = argmax(probabilities) == task["expected"]
        elif kind == "noul":
            expected = task["expected"]
            if isinstance(expected, str):
                expected = expected.strip().lower() in ("true", "yes", "1")
            correct = (probabilities.get("yes", probabilities.get("true", 0.0)) >= 0.5) == bool(expected)
        else:
            levels = [str(i) for i in range(len(task["question"]["criteria"]))]
            expected = task["expected"]
            correct = argmax({k: probabilities[k] for k in levels}) == str(expected)
        rows.append(
            {
                "id": task["id"],
                "index": index,
                "type": kind,
                "tokens": measured[item["name"]]["actual_input_tokens"],
                "margin": margin(probabilities),
                "correct": bool(correct),
                "candidates": len(probabilities),
                "known_batching_diff": task["id"] in known,
            }
        )
    per_type = int(args.per_type)
    selected, reasons = [], {}

    def take(pool, label, rows_in, count):
        if len(pool) >= per_type or count <= 0:
            return
        for row in rows_in:
            if len(pool) >= per_type or count <= 0:
                return
            if row["id"] in reasons:
                continue
            reasons[row["id"]] = label
            pool.append(row)
            count -= 1

    for kind in OD_TYPES:
        pool = [row for row in rows if row["type"] == kind]
        chosen = []
        take(chosen, "known-batching-diff",
             [row for row in pool if row["known_batching_diff"]], 8)
        take(chosen, "npu-error-low-margin",
             sorted([r for r in pool if not r["correct"]], key=lambda r: (r["margin"], r["index"])), 10)
        take(chosen, "npu-correct-low-margin",
             sorted([r for r in pool if r["correct"]], key=lambda r: (r["margin"], r["index"])), 10)
        take(chosen, "longest-input",
             sorted(pool, key=lambda r: (-r["tokens"], r["index"])), 6)
        take(chosen, "shortest-input",
             sorted(pool, key=lambda r: (r["tokens"], r["index"])), 4)
        take(chosen, "high-margin",
             sorted(pool, key=lambda r: (-r["margin"], r["index"])), 4)
        take(chosen, "lowest-margin-fill",
             sorted(pool, key=lambda r: (r["margin"], r["index"])), per_type - len(chosen))
        if len(chosen) < per_type:
            take(chosen, "order-fill", sorted(pool, key=lambda r: r["index"]),
                 per_type - len(chosen))
        selected.extend(chosen)
        if len(chosen) != per_type:
            raise SystemExit(f"{kind}: selected {len(chosen)} of {per_type}")
    selected.sort(key=lambda row: row["index"])
    return selected, reasons, {
        "records": len(rows),
        "per_type": per_type,
        "selected": len(selected),
        "known_batching_diff_ids": sorted(known),
    }


def select_erqa(args):
    data_dir = Path(args.data_dir)
    records = read_jsonl(data_dir / "samples.jsonl")
    predictions = {row["id"]: row for row in read_jsonl(Path(args.npu_run) / "predictions.jsonl")}
    rows = []
    for record in records:
        prediction = predictions[record["id"]]
        rows.append(
            {
                "id": record["id"],
                "index": record["record_index"],
                "type": record["question_type"],
                "pattern": record["visual_index_pattern"],
                "images": record["image_count"],
                "options": prediction["option_count"],
                "tokens": prediction["input_tokens"],
                "margin": margin(prediction["probabilities"]),
                "correct": prediction["correct"],
            }
        )
    selected, reasons = [], {}
    target = int(args.total)

    def take(rows_in, label, count=1):
        for row in rows_in:
            if len(selected) >= target or count <= 0:
                return
            if row["id"] in reasons:
                continue
            reasons[row["id"]] = label
            selected.append(row)
            count -= 1

    interleaved = [row for row in rows if row["pattern"] == "interleaved"]
    take(interleaved, "interleaved-all", len(interleaved))
    take(sorted(rows, key=lambda r: (-r["images"], r["index"])), "most-images", 2)
    take(sorted(rows, key=lambda r: (-r["tokens"], r["index"])), "longest-input", 2)
    take(sorted(rows, key=lambda r: (r["tokens"], r["index"])), "shortest-input", 1)
    take([row for row in sorted(rows, key=lambda r: r["index"]) if row["options"] == 2],
         "two-options", 2)
    take([row for row in sorted(rows, key=lambda r: r["index"]) if row["options"] == 3],
         "three-options", 1)
    for name in ERQA_CLASSES:
        take([row for row in sorted(rows, key=lambda r: (-r["images"], r["margin"]))
              if row["type"] == name], f"class:{name}", 1)
    take(sorted([row for row in rows if not row["correct"]],
                key=lambda r: (r["margin"], r["index"])), "lowest-margin-error", 4)
    for name in ERQA_CLASSES:
        take([row for row in sorted(rows, key=lambda r: (r["margin"], r["index"]))
              if row["type"] == name], f"class-second:{name}", 1)
    take(sorted(rows, key=lambda r: (-r["images"], r["index"])), "multi-image-fill", 6)
    take(sorted(rows, key=lambda r: (r["margin"], r["index"])), "lowest-margin-fill", 8)
    stride = max(len(rows) // max(target, 1), 1)
    for position in range(0, len(rows), stride):
        take([rows[position]], "stride-fill", 1)
    for row in sorted(rows, key=lambda r: r["index"]):
        take([row], "order-fill", 1)
    if len(selected) != target:
        raise SystemExit(f"selected {len(selected)} of {target}")
    selected.sort(key=lambda row: row["index"])
    return selected, reasons, {
        "records": len(rows),
        "selected": len(selected),
        "target": target,
        "interleaved_available": len(interleaved),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("opendecisions", "erqa"), required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--npu-run", type=Path, required=True)
    parser.add_argument("--output-stem", type=Path, required=True)
    parser.add_argument("--per-type", type=int, default=32)
    parser.add_argument("--total", type=int, default=64)
    parser.add_argument("--batching-diff", type=Path, default=None)
    args = parser.parse_args()
    if args.dataset == "opendecisions":
        selected, reasons, coverage = select_opendecisions(args)
    else:
        selected, reasons, coverage = select_erqa(args)
    ids = [row["id"] for row in selected]
    if len(ids) != len(set(ids)):
        raise SystemExit("duplicate ids in the frozen selection")
    stem = args.output_stem.expanduser().resolve()
    stem.parent.mkdir(parents=True, exist_ok=True)
    stem.with_suffix(".ids.json").write_text(
        json.dumps(ids, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    report = {
        "dataset": args.dataset,
        "rule": {
            "opendecisions": "per type: known batching diff, lowest margin, highest margin, "
                             "longest input, NPU errors, shortest input, then lowest-margin fill",
            "erqa": "all interleaved, most images, longest/shortest input, 2/3 options, "
                    "lowest-margin errors, two per class, margin/pattern/image fills, stride",
        }[args.dataset],
        "coverage": coverage,
        "counts": {
            "types": dict(Counter(row.get("type") for row in selected)),
            "patterns": dict(Counter(row.get("pattern") for row in selected if "pattern" in row)),
            "correct": dict(Counter(str(row["correct"]) for row in selected)),
            "image_counts": dict(Counter(str(row.get("images", 0)) for row in selected)),
            "option_counts": dict(Counter(str(row.get("options", 0)) for row in selected)),
        },
        "samples": [{**row, "reason": reasons[row["id"]]} for row in selected],
    }
    stem.with_suffix(".report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"ids": len(ids), "stem": str(stem), "counts": report["counts"]},
                     ensure_ascii=False))


if __name__ == "__main__":
    raise SystemExit(main())
