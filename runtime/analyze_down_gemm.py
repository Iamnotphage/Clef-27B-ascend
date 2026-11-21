#!/usr/bin/env python3
"""CPU-only evidence extraction for exact TP2 down shapes and probe timing.

Example: --kernel-csv 0=/path/rank0/kernel_details.csv --kernel-csv 1=/path/rank1/kernel_details.csv
Kernel durations and launch gaps are kept separate. Pipeline activity counters
overlap and do not identify dependency waits or a critical-path contribution.
"""
from __future__ import annotations

import argparse
import csv
from decimal import Decimal
import hashlib
import json
import math
from pathlib import Path
import statistics

METRICS = ("aicore_time(us)", "aic_mac_ratio", "aic_mte1_ratio", "aic_mte2_ratio",
           "aic_fixpipe_ratio", "aic_scalar_ratio", "cube_utilization(%)")


def distribution(values):
    if not values or any(not math.isfinite(value) for value in values):
        raise ValueError("Need finite observations")
    return {"min": min(values), "median": statistics.median(values),
            "mean": statistics.mean(values), "max": max(values)}


def read_kernel_csv(path, rank, m=967, requests=10, layers=64):
    selected, gaps, predecessors, previous = [], [], {}, {}
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames or []
        required = {"Input Shapes", "Input Formats", "Type", "Name", "Duration(us)", "Start Time(us)", "Stream ID"}
        if not required.issubset(fields):
            raise ValueError(f"Missing CSV columns: {sorted(required - set(fields))}")
        for row in reader:
            shape = row["Input Shapes"].strip().strip('"').replace(" ", "")
            key = (row.get("Device_id"), row["Stream ID"])
            start = Decimal(row["Start Time(us)"].strip())
            duration = float(row["Duration(us)"])
            if not start.is_finite() or not math.isfinite(duration) or duration < 0:
                raise ValueError("Invalid kernel timestamp/duration")
            if row["Type"] == "MatMulV2" and shape == f"{m},8704;544,320,16,16":
                if row["Input Formats"] != "ND;FRACTAL_NZ":
                    raise ValueError("Exact down shape has unexpected input formats")
                if row.get("Device_id", str(rank)) != str(rank):
                    raise ValueError("CSV device identity disagrees with supplied rank")
                selected.append(row)
                if key in previous:
                    end, name = previous[key]
                    gaps.append(float(start - end))
                    predecessors[name] = predecessors.get(name, 0) + 1
            previous[key] = (start + Decimal(row["Duration(us)"].strip()), row["Name"])
    if len(selected) != requests * layers:
        raise ValueError(f"Expected {requests}*{layers} down rows, got {len(selected)}")
    durations = [float(row["Duration(us)"]) for row in selected]
    metrics = {}
    for key in METRICS:
        try:
            metrics[key] = sum(float(row[key]) * d for row, d in zip(selected, durations)) / sum(durations)
        except (KeyError, ValueError):
            metrics[key] = None
    return {
        "rank": rank, "csv": str(path), "csv_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "rows": len(selected), "requests": requests, "layers_per_request": layers,
        "m": m, "k": 8704, "n": 5120,
        "duration_us": distribution(durations),
        "per_request_down_sum_ms": [sum(durations[i:i + layers]) / 1000 for i in range(0, len(durations), layers)],
        "mean_per_request_down_sum_ms": sum(durations) / requests / 1000,
        "effective_tflops_from_mean_kernel": 2 * m * 8704 * 5120 / (statistics.mean(durations) * 1e6),
        "block_num_values": sorted({row.get("Block Num", "missing") for row in selected}),
        "duration_weighted_metrics": metrics,
        "same_stream_predecessor_gap_us": distribution(gaps) if gaps else None,
        "same_stream_predecessors": predecessors,
        "scope": "kernel cumulative time; ordered groups assume supplied requests/layers; no cross-rank clock arithmetic",
        "missing": ["actual tiling/baseM/baseN/baseK/singleCoreM/N/K", "per-core duration and work mapping",
                    "L1/L0 traffic/reuse", "MTE/Cube/Fixpipe dependency wait intervals"],
        "interpretation": "Block Num is launch metadata; activity ratios overlap; predecessor gap is outside the kernel",
    }


def summarize_probe(data):
    if data.get("schema") != "clef-down-gemm-v1" or data.get("phase") != "timing" or data.get("status") != "complete":
        raise ValueError("Need a complete timing-phase clef-down-gemm-v1 result")
    if sorted(row["rank"] for row in data["ranks"]) != [0, 1]:
        raise ValueError("Both independent TP2 ranks are required")
    rows = []
    for rank in data["ranks"]:
        for case in rank["cases"]:
            for mode in data["modes"]:
                samples = [x for x in case["samples"] if x["mode"] == mode]
                pairs = []
                for pair in range(data["pairs"]):
                    matched = sorted((x for x in samples if x["pair"] == pair), key=lambda x: x["repeat"])
                    if len(matched) != 2 or [x["repeat"] for x in matched] != [0, 1]:
                        raise ValueError("Missing A/A repeat pair")
                    if any(not x["bit_equal"] or x["device_ms"] <= 0 or x["wall_ms"] <= 0 for x in matched):
                        raise ValueError("Invalid or non-repeatable timed sample")
                    a, b = [x["device_ms"] for x in matched]
                    pairs.append(200 * abs(a - b) / (a + b))
                rows.append({"rank": rank["rank"], "m": case["m"], "mode": mode,
                             "device_ms": distribution([x["device_ms"] for x in samples]),
                             "wall_ms": distribution([x["wall_ms"] for x in samples]),
                             "enqueue_ms": distribution([x["enqueue_ms"] for x in samples]),
                             "conditioning_wall_ms": distribution([x["conditioning_wall_ms"] for x in samples]),
                             "paired_aa_abs_delta_pct": distribution(pairs),
                             "input_is_prefix": case["capture_prefix"]})
    slowest = []
    for m in data["m"]:
        for mode in data["modes"]:
            selected = [row for row in rows if row["m"] == m and row["mode"] == mode]
            if len(selected) != 2:
                raise ValueError("Each M/mode must have both ranks")
            slowest.append({"m": m, "mode": mode,
                            "slower_device_rank": max(selected, key=lambda x: x["device_ms"]["median"])["rank"],
                            "device_ms": max(x["device_ms"]["median"] for x in selected),
                            "wall_ms": max(x["wall_ms"]["median"] for x in selected),
                            "aa_noise_max_pct": max(x["paired_aa_abs_delta_pct"]["max"] for x in selected)})
    return {"rank_modes": rows, "slower_rank_medians": slowest,
            "decision": "Baseline diagnostics only; no custom-kernel gain measured. Stop tile search without concrete tiling/pipe evidence."}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kernel-csv", action="append", default=[], metavar="RANK=PATH")
    parser.add_argument("--requests", type=int, default=10)
    parser.add_argument("--layers", type=int, default=64)
    parser.add_argument("--tokens", type=int, default=967)
    parser.add_argument("--timing-json", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if not args.kernel_csv and not args.timing_json:
        parser.error("Supply kernel CSVs or a timing JSON")
    if args.requests < 1 or args.layers < 1 or args.tokens < 1:
        parser.error("requests/layers/tokens must be positive")
    result = {"schema": "clef-down-analysis-v1", "kernels": []}
    for value in args.kernel_csv:
        rank, path = value.split("=", 1)
        rank = int(rank)
        if rank not in (0, 1) or rank in [x["rank"] for x in result["kernels"]]:
            parser.error("Supply each TP2 rank at most once")
        result["kernels"].append(read_kernel_csv(Path(path), rank, args.tokens, args.requests, args.layers))
    if len(result["kernels"]) == 2:
        rows = result["kernels"]
        result["slower_rank_mean_down_sum_ms"] = max(x["mean_per_request_down_sum_ms"] for x in rows)
        totals = [x["mean_per_request_down_sum_ms"] for x in rows]
        result["rank_mean_spread_pct"] = 200 * abs(totals[0] - totals[1]) / sum(totals)
    if args.timing_json:
        result["probe"] = summarize_probe(json.loads(args.timing_json.read_text()))
    text = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
