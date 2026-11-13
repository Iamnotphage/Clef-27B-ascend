#!/usr/bin/env python3
"""Estimate clustered sampling uncertainty from frozen native workflow scores."""

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import random
import statistics


def percentile(values, q):
    position = (len(values) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def bootstrap(groups, *, repeats, confidence, seed):
    """Resample complete clusters and preserve the native micro-mean denominator."""
    rng = random.Random(seed)
    values = []
    for _ in range(repeats):
        correct = count = 0
        for _ in groups:
            selected_correct, selected_count = rng.choice(groups)
            correct += selected_correct
            count += selected_count
        values.append(correct / count)
    values.sort()
    alpha = (1 - confidence) / 2
    return {
        "confidence_level": confidence,
        "lower": percentile(values, alpha),
        "upper": percentile(values, 1 - alpha),
        "bootstrap_standard_error": statistics.stdev(values),
    }


def summarize(source, *, repeats, confidence, seed):
    workflows = []
    for workflow in source["workflows"]:
        metrics = {}
        for metric, expected in workflow["native_metrics"].items():
            grouped = defaultdict(lambda: [0, 0])
            seen = set()
            for row in workflow["rows"]:
                if metric not in row["metrics"]:
                    continue
                identity = (row["case_id"], row["policy_id"])
                if identity in seen:
                    raise ValueError(f"Duplicate case/policy score: {identity}")
                seen.add(identity)
                value = row["metrics"][metric]
                if not isinstance(value, bool):
                    raise ValueError("Expected native binary agreement metrics")
                group = grouped[row["cluster_id"]]
                group[0] += value
                group[1] += 1
            groups = [grouped[key] for key in sorted(grouped)]
            correct = sum(group[0] for group in groups)
            count = sum(group[1] for group in groups)
            point = correct / count
            if count != workflow["native_denominators"][metric] or abs(point - expected) > 1e-12:
                raise ValueError("Native score or denominator mismatch")
            metric_seed = seed + int.from_bytes(hashlib.sha256(
                f'{workflow["workflow"]}/{metric}'.encode()).digest()[:8], "big")
            metrics[metric] = {
                "point_estimate": point, "correct_policy_decisions": correct,
                "policy_decisions": count, "cluster_count": len(groups),
                **bootstrap(groups, repeats=repeats, confidence=confidence, seed=metric_seed),
            }
        workflows.append({
            "workflow": workflow["workflow"], "case_count": workflow["cases"],
            "cluster_field": workflow["cluster_field"],
            "source_score_file": workflow["source_score_file"],
            "source_score_sha256": workflow["source_score_sha256"],
            "metrics": metrics,
        })
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_name": source["run_name"],
        "method": "cluster percentile bootstrap; sample clusters with replacement and recompute native micro mean",
        "bootstrap_repeats": repeats, "seed": seed, "confidence_level": confidence,
        "python_version": platform.python_version(),
        "independent_inference_runs": 1, "runtime_repeatability_measured": False,
        "reference": "frozen consensus labels",
        "interpretation": "Marginal sampling intervals conditional on representative, independent clusters. Runtime variation, label error and protocol/truncation bias require separate validation.",
        "workflows": workflows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=10000)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=20261113)
    args = parser.parse_args()
    if args.repeats < 2 or not 0 < args.confidence < 1:
        parser.error("repeats must be >=2 and confidence must be between 0 and 1")
    data = args.input.read_bytes()
    report = summarize(json.loads(data), repeats=args.repeats,
                       confidence=args.confidence, seed=args.seed)
    report["source_input_sha256"] = hashlib.sha256(data).hexdigest()
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    for workflow in report["workflows"]:
        print(json.dumps({"workflow": workflow["workflow"], "metrics": workflow["metrics"]}))


if __name__ == "__main__":
    main()
