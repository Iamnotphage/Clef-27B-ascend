#!/usr/bin/env python3
"""Recompute OpenDecisions quality from raw benchmark JSON and compare runs.

The raw files produced by scripts/benchmark_systemone.py keep every HTTP request
and response, so quality can be re-scored offline without touching the service.
Samples are aligned by the request ``index`` (dataset order), which preserves
the dataset's repeated questions. JSONL is parsed line by line because item
text may contain Unicode line separators.
"""

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path

TYPES = ("choice", "noul", "score")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_tasks(path):
    tasks = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                tasks.append(json.loads(line))
    return tasks


def load_scorer(path):
    spec = importlib.util.spec_from_file_location("opendecisions_author_score", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_template_rows(path):
    rows = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def extract_probabilities(response):
    answer = response["answers"]["answer"]
    probabilities = answer.get("probabilities")
    if probabilities is None and answer.get("type") == "noul":
        p_yes = answer["noul"]
        probabilities = {"yes": p_yes, "no": 1.0 - p_yes}
    return probabilities


def choice_of(task, probabilities):
    labels = task["labels"]
    return max(sorted(labels), key=lambda label: probabilities[label])


def expected_level(probabilities):
    levels = sorted(probabilities, key=int)
    return sum(int(level) * probabilities[level] for level in levels)


def load_run(path, tasks, template_rows):
    data = json.loads(Path(path).read_text())
    requests = data["requests"]
    if len(requests) != len(tasks):
        raise SystemExit(f"{path}: {len(requests)} requests, expected {len(tasks)}")
    indices = [request["index"] for request in requests]
    if (any(type(index) is not int for index in indices)
            or set(indices) != set(range(len(tasks)))):
        raise SystemExit(f"{path}: request indices must cover each dataset index exactly once")
    requests = sorted(requests, key=lambda request: request["index"])
    rows, records, order_mismatch = [], [], 0
    for request in requests:
        index = request["index"]
        task = tasks[index]
        template = template_rows[index]["model_output"]["metadata"]["choice_request"]
        if request["request"] != template:
            order_mismatch += 1
        probabilities = extract_probabilities(request["response"])
        if set(probabilities) != set(task["labels"]):
            raise SystemExit(
                f"{path}: index {index} labels {sorted(probabilities)} != {sorted(task['labels'])}")
        rows.append({"id": task["id"], "probs": probabilities})
        records.append({
            "index": index,
            "id": task["id"],
            "type": task["question"]["type"],
            "tier": task["tier"],
            "probabilities": probabilities,
            "input_tokens": request["response"]["usage"]["input_tokens"],
            "passed": request["passed"],
            "errors": request.get("errors") or [],
        })
    configuration = data.get("configuration") or {}
    summary = data.get("summary") or {}
    source = {
        "path": str(Path(path).resolve()),
        "sha256": sha256(path),
        "requested": configuration.get("requests"),
        "concurrency": configuration.get("concurrency"),
        "warmup": configuration.get("warmup"),
        "successful": summary.get("successful_requests"),
        "failed": summary.get("failed_requests"),
        "wall_seconds": summary.get("measurement_wall_seconds"),
        "decisions_per_second": summary.get("decisions_per_second"),
        "p50_ms": (summary.get("successful_request_latency_ms") or {}).get("p50"),
        "p95_ms": (summary.get("successful_request_latency_ms") or {}).get("p95"),
        "p99_ms": (summary.get("successful_request_latency_ms") or {}).get("p99"),
        "order_mismatch": order_mismatch,
        "index_order_normalized": indices != list(range(len(tasks))),
    }
    return rows, records, source


def quantile(values, q):
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def compare(reference, candidate, tasks, reference_name, candidate_name):
    diffs = []
    for index, (left, right) in enumerate(zip(reference, candidate)):
        task = tasks[index]
        labels = task["labels"]
        max_diff = max(abs(left["probabilities"][label] - right["probabilities"][label])
                       for label in labels)
        left_choice = choice_of(task, left["probabilities"]) if task["question"]["type"] == "choice" else None
        right_choice = choice_of(task, right["probabilities"]) if task["question"]["type"] == "choice" else None
        left_p = left["probabilities"].get("yes")
        right_p = right["probabilities"].get("yes")
        score_ev_diff = None
        if task["question"]["type"] == "score":
            score_ev_diff = abs(expected_level(left["probabilities"]) - expected_level(right["probabilities"]))
        diffs.append({
            "index": index,
            "id": left["id"],
            "type": task["question"]["type"],
            "tier": task["tier"],
            "max_prob_diff": max_diff,
            "choice_reference": left_choice,
            "choice_candidate": right_choice,
            "choice_changed": left_choice != right_choice,
            "p_yes_reference": left_p,
            "p_yes_candidate": right_p,
            "noul_crossed": (left_p is not None and (left_p >= 0.5) != (right_p >= 0.5)),
            "score_ev_diff": score_ev_diff,
        })
    max_values = [item["max_prob_diff"] for item in diffs]
    score_values = [item["score_ev_diff"] for item in diffs if item["score_ev_diff"] is not None]
    return {
        "reference": reference_name,
        "candidate": candidate_name,
        "items": len(diffs),
        "exact_equal_items": sum(1 for item in diffs if item["max_prob_diff"] == 0.0),
        "choice_changes": sum(1 for item in diffs if item["choice_changed"]),
        "noul_crossings": sum(1 for item in diffs if item["noul_crossed"]),
        "prob_diff_over_0_03": sum(1 for value in max_values if value > 0.03),
        "prob_diff_over_0_05": sum(1 for value in max_values if value > 0.05),
        "max_prob_diff": max(max_values),
        "prob_diff_quantiles": {
            "p50": quantile(max_values, 0.50),
            "p90": quantile(max_values, 0.90),
            "p95": quantile(max_values, 0.95),
            "p99": quantile(max_values, 0.99),
        },
        "score_items": len(score_values),
        "score_diff_over_0_05": sum(1 for value in score_values if value > 0.05),
        "max_score_ev_diff": max(score_values) if score_values else None,
        "top_prob_diffs": sorted(diffs, key=lambda item: -item["max_prob_diff"])[:20],
        "changed_items": [item for item in diffs
                          if item["choice_changed"] or item["noul_crossed"]
                          or item["max_prob_diff"] > 0.03],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--scorer", required=True)
    parser.add_argument("--template", required=True,
                        help="baseline EvalScope predictions JSONL carrying the fixed request template")
    parser.add_argument("--run", action="append", default=[], metavar="NAME=PATH")
    parser.add_argument("--reference", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    tasks = load_tasks(args.data)
    template_rows = load_template_rows(args.template)
    if len(tasks) != len(template_rows):
        raise SystemExit("data and template row counts differ")
    for index, (task, row) in enumerate(zip(tasks, template_rows)):
        if row["metadata"]["source_id"] != task["id"]:
            raise SystemExit(f"template order mismatch at index {index}")
    scorer = load_scorer(args.scorer)

    runs = {}
    for spec in args.run:
        name, _, path = spec.partition("=")
        if not path:
            raise SystemExit(f"bad --run {spec!r}")
        rows, records, source = load_run(path, tasks, template_rows)
        result = scorer.score(tasks, rows)
        outcomes = [scorer.item_outcome(task, row["probs"]) for task, row in zip(tasks, rows)]
        valid = [outcome for outcome in outcomes if outcome["valid"]]
        score_outcomes = [outcome for outcome in valid if outcome["type"] == "score"]
        correct = sum(outcome["correct"] for outcome in valid)
        tokens = [record["input_tokens"] for record in records]
        source.update({
            "scoring_result": result,
            "accuracy": correct / len(valid),
            "correct": correct,
            "accuracy_denominator": len(valid),
            "score_mae": (sum(outcome["nmae"] * (outcome["k"] - 1) for outcome in score_outcomes)
                          / len(score_outcomes)) if score_outcomes else None,
            "score_mae_denominator": len(score_outcomes),
            "input_tokens": {
                "mean": sum(tokens) / len(tokens),
                "p50": quantile(tokens, 0.50),
                "p95": quantile(tokens, 0.95),
                "max": max(tokens),
                "total": sum(tokens),
            },
            "protocol_failures": sum(1 for record in records if not record["passed"]),
        })
        runs[name] = {"rows": rows, "records": records, "source": source}

    if args.reference not in runs:
        raise SystemExit("reference run is missing")
    reference = runs[args.reference]["records"]
    output = {
        "data": {"path": str(Path(args.data).resolve()), "sha256": sha256(args.data), "items": len(tasks)},
        "scorer": {"path": str(Path(args.scorer).resolve()), "sha256": sha256(args.scorer)},
        "template": {"path": str(Path(args.template).resolve()), "sha256": sha256(args.template)},
        "runs": {name: run["source"] for name, run in runs.items()},
        "comparisons": {
            name: compare(reference, run["records"], tasks, args.reference, name)
            for name, run in runs.items() if name != args.reference
        },
    }
    Path(args.output).write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({
        "runs": {name: {
            "capability": run["source"]["scoring_result"]["capability"],
            "intelligence": run["source"]["scoring_result"]["intelligence"],
            "calibration": run["source"]["scoring_result"]["calibration"],
            "accuracy": run["source"]["accuracy"],
            "failure": run["source"]["protocol_failures"],
            "sha256": run["source"]["sha256"],
        } for name, run in runs.items()},
        "comparisons": {name: {key: value for key, value in result.items()
                               if key not in ("top_prob_diffs", "changed_items")}
                        for name, result in output["comparisons"].items()},
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
