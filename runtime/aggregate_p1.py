#!/usr/bin/env python3
"""Aggregate phase-1 point results into one matrix with batch evidence."""
import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--p1-dir", required=True)
parser.add_argument("--output-json", required=True)
parser.add_argument("--output-csv", required=True)
parser.add_argument("--tools", default="/workspace/yangchen/work/profiling-tools-20261113")
args = parser.parse_args()

rows = []
for point_dir in sorted(Path(args.p1_dir).glob("p1-s*-c*")):
    summary_path = point_dir / "point-summary.json"
    if not summary_path.exists():
        continue
    summary = json.loads(summary_path.read_text())
    batch_path = point_dir / "batch-reconstruction.json"
    if batch_path.exists():
        batch = json.loads(batch_path.read_text())
    else:
        proc = subprocess.run(
            ["python3", "/workspace/yangchen/logs/tuning/reconstruct_batch.py",
             "--before", str(point_dir / "metrics-before.txt"),
             "--after", str(point_dir / "metrics-after.txt"),
             "--warmup", "10",
             "--output", str(batch_path)],
            capture_output=True, text=True, timeout=60)
        batch = json.loads(batch_path.read_text()) if batch_path.exists() else {}
    label = summary["label"]
    parts = label.replace("p1-s", "").split("-c")
    row = {
        "label": label,
        "max_num_seqs": int(parts[0]),
        "http_concurrency": int(parts[1]),
        "passed": summary.get("passed"),
        "successful": summary.get("successful"),
        "failed": summary.get("failed"),
        "wall_seconds": summary.get("wall_seconds"),
        "rps": summary.get("rps"),
        "decisions_per_second": summary.get("decisions_per_second"),
        "input_tokens_per_second": summary.get("input_tokens_per_second"),
        "p50_ms": summary.get("p50_ms"),
        "p95_ms": summary.get("p95_ms"),
        "p99_ms": summary.get("p99_ms"),
        "mean_batch": batch.get("mean_batch_per_iteration"),
        "batch_distribution": batch.get("batch_solution"),
        "bucket_check_error": batch.get("bucket_check_error_iterations"),
        "hbm_max_total_used_mib": summary.get("hbm_max_total_used_mib"),
    }
    # cpu throttling differential
    try:
        def parse_stat(p):
            fields = {}
            for line in Path(p).read_text().splitlines():
                parts = line.split()
                if len(parts) == 2:
                    fields[parts[0]] = int(parts[1])
            return fields
        before = parse_stat(point_dir / "cpu-stat-before.txt")
        after = parse_stat(point_dir / "cpu-stat-after.txt")
        row["throttled_periods_delta"] = after["nr_throttled"] - before["nr_throttled"]
        row["throttled_time_delta_ms"] = (after["throttled_time"] - before["throttled_time"]) / 1e6
        row["cpu_periods_delta"] = after["nr_periods"] - before["nr_periods"]
    except (OSError, KeyError):
        row["throttled_periods_delta"] = None
    rows.append(row)

Path(args.output_json).write_text(json.dumps(rows, indent=1))
fields = ["label", "max_num_seqs", "http_concurrency", "passed", "successful", "failed",
          "wall_seconds", "rps", "decisions_per_second", "input_tokens_per_second",
          "p50_ms", "p95_ms", "p99_ms", "mean_batch", "batch_distribution",
          "bucket_check_error", "hbm_max_total_used_mib",
          "throttled_periods_delta", "throttled_time_delta_ms", "cpu_periods_delta"]
with open(args.output_csv, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
print(json.dumps(rows, indent=1))
