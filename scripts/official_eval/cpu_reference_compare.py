#!/usr/bin/env python3
"""Compare the CPU BF16 reference against a frozen NPU baseline run.

Reads the per-sample records written by ``cpu_reference_baseline.py`` and the
frozen NPU artifacts (the /v1/systemone benchmark for OpenDecisions, the
/v1/decide predictions for ERQA), then reports per-item probability distances,
decision flips, correctness changes, the input-equivalence checks and the
full-sample score under the fixed protocol for both implementations.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import importlib.util
import json
import math
from pathlib import Path

REFERENCE_SHA256 = "0e304cf7c6500e8bb59bef7e2afd2c6373f82596dfb3b57d1aa93c175e2dc3a3"
OPENDECISIONS_DATA_SHA256 = (
    "78614972006c5e5efbbfc121ce31e15cf576ef6c0cdf24a229b33d34d2bfcda6"
)
SCORER_SHA256 = "478d14ae7a779c72e0dcd6c3225a3462e24faa8d7e8092afb0d6422ff84e4377"
ERQA_TFRECORD_SHA256 = (
    "1974f5e587eac144cc3ba325477477b8ad6a56c22ce4e419c34229520b4c93e2"
)


def read_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8")


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def percentile(values, fraction):
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * fraction
    lower, upper = math.floor(index), math.ceil(index)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)


def quantiles(values):
    values = [value for value in values if value is not None]
    if not values:
        return {"count": 0}
    return {
        "count": len(values),
        "mean": sum(values) / len(values),
        "p50": percentile(values, 0.50),
        "p90": percentile(values, 0.90),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "max": max(values),
        "min": min(values),
    }


def expected_value(probabilities, levels):
    return sum(level * probabilities[str(level)] for level in levels)


def cpu_probabilities(kind, probabilities):
    """Map the reference's noul option ids (true/false) to the scorer's yes/no."""
    if kind == "noul":
        p_true = probabilities["true"]
        return {"yes": p_true, "no": probabilities.get("false", 1.0 - p_true)}
    return dict(probabilities)


def top_margin(probabilities):
    ordered = sorted(probabilities.values(), reverse=True)
    if len(ordered) < 2:
        return ordered[0] if ordered else None
    return ordered[0] - ordered[1]


def load_cpu_records(run_dir):
    records = {}
    for path in sorted((run_dir / "samples").glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("status") == "ok":
            records[record["id"]] = record
    return records


def load_scorer(path):
    if sha256_file(path) != SCORER_SHA256:
        raise SystemExit("scorer SHA256 does not match the fixed revision")
    spec = importlib.util.spec_from_file_location("opendecisions_author_score", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def score_opendecisions(tasks, predictions, scorer):
    rows = [predictions[task["id"]] for task in tasks if task["id"] in predictions]
    result = scorer.score(tasks, rows)
    outcomes = [scorer.item_outcome(task, predictions.get(task["id"], {}).get("probs"))
                for task in tasks]
    valid = [outcome for outcome in outcomes if outcome["valid"]]
    score_valid = [outcome for outcome in valid if outcome["type"] == "score"]
    derived = {
        "accuracy": sum(outcome["correct"] for outcome in valid) / len(valid) if valid else None,
        "correct": sum(outcome["correct"] for outcome in valid),
        "accuracy_denominator": len(valid),
        "score_mae": (sum(outcome["nmae"] * (outcome["k"] - 1) for outcome in score_valid)
                      / len(score_valid)) if score_valid else None,
        "score_mae_denominator": len(score_valid),
        "missing": len(tasks) - len(rows),
    }
    return result, derived, outcomes


def compare_opendecisions(args):
    data_path = Path(args.data_dir) / "open-decisions2352.jsonl"
    if sha256_file(data_path) != OPENDECISIONS_DATA_SHA256:
        raise SystemExit("OpenDecisions data SHA256 mismatch")
    tasks = read_jsonl(data_path)
    npu_run = json.loads(Path(args.npu_run).read_text(encoding="utf-8"))
    measured = {item["sample_name"]: item for item in npu_run["requests"]}
    cpu = load_cpu_records(args.cpu_run)
    scorer = load_scorer(args.scorer)

    npu_predictions = {}
    cpu_predictions = {}
    items = []
    for index, (task, saved) in enumerate(zip(tasks, npu_run["samples"])):
        sample_id = task["id"]
        measured_item = measured[saved["name"]]
        response = measured_item["response"]["answers"]["answer"]
        npu_probs = response.get("probabilities")
        if npu_probs is None:
            p_yes = response["noul"]
            npu_probs = {"yes": p_yes, "no": 1.0 - p_yes}
        npu_predictions[sample_id] = {"id": sample_id, "probs": npu_probs}
        record = cpu.get(sample_id)
        if record is None:
            items.append({"id": sample_id, "index": index, "status": "missing-cpu"})
            continue
        kind = task["question"]["type"]
        cpu_probs = cpu_probabilities(kind, record["probabilities"]["answer"])
        cpu_predictions[sample_id] = {"id": sample_id, "probs": cpu_probs}
        keys = sorted(set(cpu_probs) | set(npu_probs))
        differences = {key: abs(cpu_probs.get(key, 0.0) - npu_probs.get(key, 0.0)) for key in keys}
        item = {
            "id": sample_id,
            "index": index,
            "name": saved["name"],
            "type": kind,
            "tier": task["tier"],
            "candidates": len(npu_probs),
            "tokens_cpu": record["input"]["input_tokens"],
            "tokens_npu": measured_item["actual_input_tokens"],
            "max_prob_diff": max(differences.values()),
            "mean_prob_diff": sum(differences.values()) / len(differences),
            "tvd": 0.5 * sum(differences.values()),
            "cpu_margin": top_margin(cpu_probs),
            "npu_margin": top_margin(npu_probs),
            "cpu_probs": cpu_probs,
            "npu_probs": npu_probs,
        }
        if kind == "choice":
            cpu_best = max(sorted(npu_probs), key=lambda key: cpu_probs[key])
            npu_best = max(sorted(npu_probs), key=lambda key: npu_probs[key])
            item.update(
                cpu_argmax=cpu_best,
                npu_argmax=npu_best,
                argmax_same=cpu_best == npu_best,
                cpu_correct=cpu_best == task["expected"],
                npu_correct=npu_best == task["expected"],
            )
        elif kind == "noul":
            item.update(
                cpu_yes=cpu_probs["yes"],
                npu_yes=npu_probs["yes"],
                cpu_decision="yes" if cpu_probs["yes"] >= 0.5 else "no",
                npu_decision="yes" if npu_probs["yes"] >= 0.5 else "no",
                argmax_same=(cpu_probs["yes"] >= 0.5) == (npu_probs["yes"] >= 0.5),
                cpu_correct=(cpu_probs["yes"] >= 0.5) == (task["expected"] == "yes"),
                npu_correct=(npu_probs["yes"] >= 0.5) == (task["expected"] == "yes"),
            )
        else:
            levels = list(range(len(task["labels"])))
            cpu_ev = expected_value(cpu_probs, levels)
            npu_ev = expected_value(npu_probs, levels)
            gold = int(task["expected"])
            cpu_best = max(levels, key=lambda level: (cpu_probs[str(level)], -level))
            npu_best = max(levels, key=lambda level: (npu_probs[str(level)], -level))
            item.update(
                cpu_expected_value=cpu_ev,
                npu_expected_value=npu_ev,
                expected_value_diff=abs(cpu_ev - npu_ev),
                cpu_argmax=cpu_best,
                npu_argmax=npu_best,
                argmax_same=cpu_best == npu_best,
                cpu_correct=cpu_best == gold,
                npu_correct=npu_best == gold,
            )
        items.append(item)

    complete = [item for item in items if item.get("status") != "missing-cpu"]
    valid_items = [item for item in complete if "cpu_correct" in item]
    max_diffs = [item["max_prob_diff"] for item in valid_items]
    summary = {
        "dataset": "opendecisions",
        "tasks": len(tasks),
        "cpu_records": len(cpu),
        "compared": len(complete),
        "missing_cpu": len(items) - len(complete),
        "token_count_matches": sum(1 for item in complete if item["tokens_cpu"] == item["tokens_npu"]),
        "token_count_mismatches": [
            {"id": item["id"], "cpu": item["tokens_cpu"], "npu": item["tokens_npu"]}
            for item in complete if item["tokens_cpu"] != item["tokens_npu"]
        ],
        "max_prob_diff_quantiles": quantiles(max_diffs),
        "mean_prob_diff_quantiles": quantiles([item["mean_prob_diff"] for item in valid_items]),
        "tvd_quantiles": quantiles([item["tvd"] for item in valid_items]),
        "max_prob_diff_over": {
            str(threshold): sum(1 for value in max_diffs if value > threshold)
            for threshold in (0.001, 0.01, 0.03, 0.05, 0.1)
        },
        "flips": {
            "choice_changed": sum(1 for item in valid_items
                                  if item["type"] == "choice" and not item["argmax_same"]),
            "noul_crossed_0_5": sum(1 for item in valid_items
                                    if item["type"] == "noul" and not item["argmax_same"]),
            "score_argmax_changed": sum(1 for item in valid_items
                                        if item["type"] == "score" and not item["argmax_same"]),
            "score_expected_diff_over_0_05": sum(
                1 for item in valid_items
                if item["type"] == "score" and item["expected_value_diff"] > 0.05),
            "max_score_expected_diff": max(
                (item["expected_value_diff"] for item in valid_items
                 if item["type"] == "score"), default=None),
        },
        "correctness": {
            "both_correct": sum(1 for item in valid_items if item["cpu_correct"] and item["npu_correct"]),
            "cpu_correct_npu_wrong": sum(1 for item in valid_items if item["cpu_correct"] and not item["npu_correct"]),
            "cpu_wrong_npu_correct": sum(1 for item in valid_items if not item["cpu_correct"] and item["npu_correct"]),
            "both_wrong": sum(1 for item in valid_items if not item["cpu_correct"] and not item["npu_correct"]),
        },
        "by_type": {},
        "by_tier": {},
        "by_token_bucket": {},
        "by_candidates": {},
        "largest_differences": sorted(
            (
                {
                    "id": item["id"],
                    "type": item["type"],
                    "tier": item["tier"],
                    "tokens": item["tokens_npu"],
                    "max_prob_diff": item["max_prob_diff"],
                    "choice_changed": item["type"] == "choice" and not item["argmax_same"],
                    "cpu_correct": item["cpu_correct"],
                    "npu_correct": item["npu_correct"],
                    "cpu_margin": item["cpu_margin"],
                    "npu_margin": item["npu_margin"],
                }
                for item in valid_items
            ),
            key=lambda row: -row["max_prob_diff"],
        )[:25],
    }
    for key in ("type", "tier"):
        grouped = defaultdict(list)
        for item in valid_items:
            grouped[item[key]].append(item)
        summary[f"by_{key}"] = {
            name: {
                "items": len(group),
                "choice_changed": sum(1 for item in group if item["type"] == "choice"
                                      and not item["argmax_same"]),
                "noul_crossed": sum(1 for item in group if item["type"] == "noul"
                                    and not item["argmax_same"]),
                "score_argmax_changed": sum(1 for item in group if item["type"] == "score"
                                            and not item["argmax_same"]),
                "max_prob_diff_max": max(item["max_prob_diff"] for item in group),
                "max_prob_diff_p95": percentile([item["max_prob_diff"] for item in group], 0.95),
                "cpu_accuracy": sum(item["cpu_correct"] for item in group) / len(group),
                "npu_accuracy": sum(item["npu_correct"] for item in group) / len(group),
            }
            for name, group in sorted(grouped.items())
        }
    buckets = [(0, 256), (256, 512), (512, 1024), (1024, 4096)]
    for low, high in buckets:
        group = [item for item in valid_items if low < item["tokens_npu"] <= high]
        if not group:
            continue
        summary["by_token_bucket"][f"{low}-{high}"] = {
            "items": len(group),
            "max_prob_diff_max": max(item["max_prob_diff"] for item in group),
            "max_prob_diff_p95": percentile([item["max_prob_diff"] for item in group], 0.95),
            "choice_changed": sum(1 for item in group if item["type"] == "choice"
                                  and not item["argmax_same"]),
        }
    grouped = defaultdict(list)
    for item in valid_items:
        grouped[item["candidates"]].append(item)
    summary["by_candidates"] = {
        str(name): {
            "items": len(group),
            "max_prob_diff_max": max(item["max_prob_diff"] for item in group),
            "choice_changed": sum(1 for item in group if item["type"] == "choice"
                                  and not item["argmax_same"]),
        }
        for name, group in sorted(grouped.items())
    }

    cpu_scores, cpu_derived, cpu_outcomes = score_opendecisions(tasks, cpu_predictions, scorer)
    npu_scores, npu_derived, npu_outcomes = score_opendecisions(tasks, npu_predictions, scorer)
    comparison = {
        "cpu": {**{key: cpu_scores[key] for key in cpu_scores}, "derived": cpu_derived},
        "npu": {**{key: npu_scores[key] for key in npu_scores}, "derived": npu_derived},
        "difference": {
            key: (None if cpu_derived[key] is None or npu_derived[key] is None
                  else cpu_derived[key] - npu_derived[key])
            for key in ("accuracy", "score_mae")
        },
    }
    for key in ("calibration", "intelligence", "capability", "calibration_items_tvd"):
        if (key in cpu_scores and key in npu_scores
                and cpu_scores[key] is not None and npu_scores[key] is not None):
            comparison["difference"][key] = cpu_scores[key] - npu_scores[key]
    with (args.output / "cpu-predictions.jsonl").open("w", encoding="utf-8") as handle:
        for row in cpu_predictions.values():
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (args.output / "per-item.jsonl").open("w", encoding="utf-8") as handle:
        for item in items:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    return {"summary": summary, "scores": comparison, "outcomes": len(cpu_outcomes)}


def compare_erqa(args):
    data_dir = Path(args.data_dir)
    manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest["source"]["sha256"] != ERQA_TFRECORD_SHA256:
        raise SystemExit("ERQA tfrecord SHA256 mismatch")
    records = read_jsonl(data_dir / "samples.jsonl")
    npu_rows = {row["id"]: row for row in read_jsonl(Path(args.npu_run) / "predictions.jsonl")}
    cpu = load_cpu_records(args.cpu_run)
    items = []
    for record in records:
        sample_id = record["id"]
        npu_row = npu_rows[sample_id]
        cpu_record = cpu.get(sample_id)
        item = {
            "id": sample_id,
            "index": record["record_index"],
            "type": record["question_type"],
            "pattern": record["visual_index_pattern"],
            "images": record["image_count"],
            "options": npu_row["option_count"],
            "gold": record["answer"],
            "tokens_npu": npu_row["input_tokens"],
            "npu_prediction": npu_row["prediction"],
            "npu_correct": npu_row["correct"],
            "npu_probs": npu_row["probabilities"],
        }
        if cpu_record is None:
            item["status"] = "missing-cpu"
            items.append(item)
            continue
        cpu_probs = cpu_record["probabilities"]["answer"]
        npu_probs = npu_row["probabilities"]
        keys = sorted(set(cpu_probs) | set(npu_probs))
        differences = {key: abs(cpu_probs.get(key, 0.0) - npu_probs.get(key, 0.0)) for key in keys}
        cpu_prediction = max(sorted(cpu_probs), key=lambda key: cpu_probs[key])
        item.update(
            status="ok",
            tokens_cpu=cpu_record["input"]["input_tokens"],
            cpu_prediction=cpu_prediction,
            cpu_correct=cpu_prediction.replace(".", "").strip().lower()
            == record["answer"].strip().lower(),
            cpu_probs=cpu_probs,
            max_prob_diff=max(differences.values()),
            mean_prob_diff=sum(differences.values()) / len(differences),
            tvd=0.5 * sum(differences.values()),
            cpu_margin=top_margin(cpu_probs),
            npu_margin=top_margin(npu_probs),
            prediction_same=cpu_prediction == npu_row["prediction"],
        )
        items.append(item)
    complete = [item for item in items if item.get("status") == "ok"]
    summary = {
        "dataset": "erqa",
        "records": len(records),
        "cpu_records": len(cpu),
        "compared": len(complete),
        "missing_cpu": len(items) - len(complete),
        "token_count_matches": sum(1 for item in complete if item["tokens_cpu"] == item["tokens_npu"]),
        "token_count_mismatches": [
            {"id": item["id"], "cpu": item["tokens_cpu"], "npu": item["tokens_npu"],
             "images": item["images"]}
            for item in complete if item["tokens_cpu"] != item["tokens_npu"]
        ],
        "cpu_accuracy": sum(item["cpu_correct"] for item in complete) / len(complete),
        "npu_accuracy": sum(item["npu_correct"] for item in complete) / len(complete),
        "cpu_correct": sum(item["cpu_correct"] for item in complete),
        "npu_correct": sum(item["npu_correct"] for item in complete),
        "prediction_changes": sum(1 for item in complete if not item["prediction_same"]),
        "max_prob_diff_quantiles": quantiles([item["max_prob_diff"] for item in complete]),
        "tvd_quantiles": quantiles([item["tvd"] for item in complete]),
        "max_prob_diff_over": {
            str(threshold): sum(1 for item in complete if item["max_prob_diff"] > threshold)
            for threshold in (0.001, 0.01, 0.03, 0.05, 0.1)
        },
        "correctness": {
            "both_correct": sum(1 for item in complete if item["cpu_correct"] and item["npu_correct"]),
            "cpu_correct_npu_wrong": sum(1 for item in complete if item["cpu_correct"] and not item["npu_correct"]),
            "cpu_wrong_npu_correct": sum(1 for item in complete if not item["cpu_correct"] and item["npu_correct"]),
            "both_wrong": sum(1 for item in complete if not item["cpu_correct"] and not item["npu_correct"]),
        },
    }
    for key in ("type", "pattern"):
        grouped = defaultdict(list)
        for item in complete:
            grouped[item[key]].append(item)
        summary[f"by_{key}"] = {
            name: {
                "items": len(group),
                "cpu_correct": sum(item["cpu_correct"] for item in group),
                "npu_correct": sum(item["npu_correct"] for item in group),
                "cpu_accuracy": sum(item["cpu_correct"] for item in group) / len(group),
                "npu_accuracy": sum(item["npu_correct"] for item in group) / len(group),
                "prediction_changes": sum(1 for item in group if not item["prediction_same"]),
                "max_prob_diff_max": max(item["max_prob_diff"] for item in group),
            }
            for name, group in sorted(grouped.items())
        }
    for name, group in (("single-image", [item for item in complete if item["images"] == 1]),
                        ("multi-image", [item for item in complete if item["images"] > 1])):
        summary[f"by_{name}"] = {
            "items": len(group),
            "cpu_correct": sum(item["cpu_correct"] for item in group),
            "npu_correct": sum(item["npu_correct"] for item in group),
            "cpu_accuracy": sum(item["cpu_correct"] for item in group) / len(group) if group else None,
            "npu_accuracy": sum(item["npu_correct"] for item in group) / len(group) if group else None,
            "prediction_changes": sum(1 for item in group if not item["prediction_same"]),
            "max_prob_diff_max": max((item["max_prob_diff"] for item in group), default=None),
        }
    summary["by_option_count"] = {
        str(name): {
            "items": len(group),
            "cpu_correct": sum(item["cpu_correct"] for item in group),
            "npu_correct": sum(item["npu_correct"] for item in group),
            "prediction_changes": sum(1 for item in group if not item["prediction_same"]),
        }
        for name, group in sorted(
            {name: [item for item in complete if item["options"] == name]
             for name in {item["options"] for item in complete}}.items())
    }
    summary["largest_differences"] = sorted(
        (
            {
                "id": item["id"],
                "images": item["images"],
                "tokens": item["tokens_npu"],
                "max_prob_diff": item["max_prob_diff"],
                "prediction_changed": not item["prediction_same"],
                "cpu_correct": item["cpu_correct"],
                "npu_correct": item["npu_correct"],
                "cpu_margin": item["cpu_margin"],
                "npu_margin": item["npu_margin"],
            }
            for item in complete
        ),
        key=lambda row: -row["max_prob_diff"],
    )[:25]
    with (args.output / "per-item.jsonl").open("w", encoding="utf-8") as handle:
        for item in items:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    return {"summary": summary, "scores": None, "outcomes": len(complete)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("opendecisions", "erqa"), required=True)
    parser.add_argument("--cpu-run", type=Path, required=True)
    parser.add_argument("--npu-run", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--scorer", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.dataset == "opendecisions":
        if not args.scorer:
            raise SystemExit("--scorer is required for OpenDecisions")
        result = compare_opendecisions(args)
        comparison = {
            "created_at": __import__("datetime").datetime.now(
                __import__("datetime").timezone.utc).isoformat(),
            "dataset": "opendecisions",
            "cpu_run": str(args.cpu_run),
            "npu_run": str(args.npu_run),
            "cpu_run_identity_sha256": sha256_file(args.cpu_run / "run-identity.json"),
            "reference_sha256": REFERENCE_SHA256,
            "scorer_sha256": SCORER_SHA256,
            "summary": result["summary"],
            "scores": result["scores"],
            "outcomes": result["outcomes"],
        }
    else:
        result = compare_erqa(args)
        comparison = {
            "created_at": __import__("datetime").datetime.now(
                __import__("datetime").timezone.utc).isoformat(),
            "dataset": "erqa",
            "cpu_run": str(args.cpu_run),
            "npu_run": str(args.npu_run),
            "cpu_run_identity_sha256": sha256_file(args.cpu_run / "run-identity.json"),
            "reference_sha256": REFERENCE_SHA256,
            "summary": result["summary"],
        }
    write_json(args.output / "comparison.json", comparison)
    print(json.dumps(
        {
            "dataset": args.dataset,
            "compared": result["summary"]["compared"],
            "token_count_matches": result["summary"]["token_count_matches"],
            "output": str(args.output),
        },
        ensure_ascii=False,
    ))


if __name__ == "__main__":
    raise SystemExit(main())
