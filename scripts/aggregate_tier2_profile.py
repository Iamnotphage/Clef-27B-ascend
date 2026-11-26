#!/usr/bin/env python3
"""Aggregate the multi-granularity Tier 2 profile windows.

W1/W2 are the OpenDecisions cold-populate and Tier 2B suffix-resume windows;
W3/W4 are the ERQA multi-image equivalents.  W5/W6 add the Tier 2A
boundary-representation resume (continuous hidden/residual at layer 56), which
is the new warm-cache path for an exact repeated prompt.

Each window is produced by ``scripts/profile_systemone.py`` against a profile
service and then parsed by ``scripts/analyze_systemone_profile.py``.  The
output is a compact JSON that ``scripts/plot_paper_figures.py`` consumes; raw
Chrome traces are never copied into the repository.

Run::

    python scripts/aggregate_tier2_profile.py \
        --root /tmp/profile-tier2-out \
        --output logs/profile-tier2-20261126/aggregate.json
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
from pathlib import Path

WINDOWS = {
    "W1_od_cold": ("w1-od-cold", "OpenDecisions",
                   "Populate (full 967 tokens)"),
    "W2_od_hit": ("w2-od-hit", "OpenDecisions",
                  "Tier 2B hit (suffix 199 tokens)"),
    "W3_erqa_cold": ("w3-erqa-cold", "ERQA 4-image",
                     "Populate (full 1260 tokens)"),
    "W4_erqa_hit": ("w4-erqa-hit", "ERQA 4-image",
                    "Tier 2B hit (suffix 236 tokens)"),
    "W5_od_repr": ("w5-od-repr-hit", "OpenDecisions",
                   "Tier 2A hit (boundary 56, full prompt)"),
    "W6_erqa_repr": ("w6-erqa-repr-hit", "ERQA 4-image",
                     "Tier 2A hit (boundary 56, full prompt)"),
}

VIT_DIMS = ("1152", "2152", "1728", "576", "2304", "4608")


def classify(name, communication, shape=""):
    low = name.lower()
    if communication:
        return "hccl"
    if low.startswith("aclnnconvolution") or any(d in shape for d in VIT_DIMS):
        return "vit"
    if "matmul" in low or "addmm" in low:
        return "matmul"
    if "fla" in low or "gdn" in low or "delta" in low or "conv" in low:
        return "gdn"
    if "attention" in low or "fia" in low:
        return "attention"
    if "rms" in low or "norm" in low:
        return "norm"
    if any(k in low for k in ("silu", "mul", "add", "cast", "transpose",
                              "copy", "slice", "concat", "gather", "scatter",
                              "index", "sigmoid", "zero")):
        return "elementwise"
    return "other"


def read_ops(path, rank):
    rows = list(csv.DictReader(path.open()))
    groups = collections.defaultdict(float)
    names = collections.defaultdict(float)
    for row in rows:
        if row["rank"] != str(rank):
            continue
        name = row["name"]
        union = float(row.get("interval_union_us") or 0)
        comm = row.get("communication") == "True"
        groups[classify(name, comm, row.get("shape", ""))] += union
        names[name] += union
    return dict(groups), dict(sorted(names.items(), key=lambda kv: -kv[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--windows", nargs="*", default=None,
                        help="optional subset of window keys")
    args = parser.parse_args()
    selected = (args.windows if args.windows else list(WINDOWS))
    report = {"schema": "profile-tier2-aggregate/1", "windows": {}}
    for key in selected:
        directory, dataset, scenario = WINDOWS[key]
        base = args.root / directory
        manifest = json.loads((base / "manifest.json").read_text())
        requests = json.loads((base / "profile-requests.json").read_text())
        evidence = json.loads(
            (base / "trace-analysis/compact-evidence.json").read_text())
        summary = requests["summary"]
        rows = []
        for index, rank in enumerate(evidence["ranks"]):
            groups, names = read_ops(
                base / "trace-analysis/operations.csv", index)
            rows.append({
                "rank": index,
                "compute_union_ms": rank["compute_kernel_union_us"] / 1000,
                "communication_union_ms": rank["communication_union_us"] / 1000,
                "communication_nonoverlapped_ms":
                    rank["communication_nonoverlapped_us"] / 1000,
                "no_device_kernel_ms":
                    rank["no_device_kernel_within_window_us"] / 1000,
                "kernel_window_span_ms":
                    rank["kernel_window_span_us"] / 1000,
                "category_union_ms": {k: v / 1000
                                      for k, v in groups.items()},
                "top_ops": [{"name": n, "union_ms": v / 1000}
                            for n, v in list(names.items())[:12]],
            })
        per_request = len(requests["requests"])
        report["windows"][key] = {
            "dataset": dataset,
            "scenario": scenario,
            "requests": per_request,
            "input_tokens":
                summary["actual_input_tokens"]["per_request_histogram"],
            "latency_ms": {k: summary["successful_request_latency_ms"][k]
                           for k in ("p50", "p95", "mean", "min", "max")},
            "manifest_passed": manifest.get("passed"),
            "ranks": rows,
            "per_request": [{"index": r["index"],
                             "elapsed_ms": r["elapsed_ms"],
                             "input_tokens": r.get("actual_input_tokens")}
                            for r in requests["requests"]],
        }
        print(key, "ok", per_request)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=1) + "\n")
    print("wrote", args.output)


if __name__ == "__main__":
    main()
