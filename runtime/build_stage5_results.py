#!/usr/bin/env python3
"""Rebuild measurements-v2.csv, results.csv and metrics.csv from the final
acceptance summary. Every stage-5 row is regenerated; historical rows stay.

Input:  Clef-27B/logs/tuning3-20261113/stage5-summary.json
Output: Clef-27B/logs/tuning2-20261113/measurements-v2.csv
        Clef-27B/results.csv
        Clef-27B/metrics.csv
"""
import csv
import json
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARCH3 = os.path.join(REPO, "Clef-27B", "logs", "tuning3-20261113")
ARCH2 = os.path.join(REPO, "Clef-27B", "logs", "tuning2-20261113")
SUM = json.load(open(os.path.join(ARCH3, "stage5-summary.json")))
MV2 = os.path.join(ARCH2, "measurements-v2.csv")

with open(MV2, newline="") as handle:
    reader = csv.DictReader(handle)
    fields = reader.fieldnames
    rows = [
        row for row in reader
        if not row["phase"].startswith("stage5-")
    ]

existing = {row["measurement_id"] for row in rows}
for measurement in SUM["measurements"]:
    if measurement["measurement_id"] in existing:
        continue
    row = {key: "" for key in fields}
    row.update({key: str(value) for key, value in measurement.items() if key in fields})
    row["schema_version"] = "2.0"
    rows.append(row)

with open(MV2, "w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
print(f"measurements-v2.csv now {len(rows)} rows")

IMAGE = ("Ubuntu 22.04.5 aarch64;CANN 9.2.1;torch_npu 2.10.0.post7;"
         "vLLM 24420b7a;vLLM-Ascend 89eb8a5e")
RESOURCES = "A3;1物理卡;2逻辑设备"
FIELDS = ["model", "dataset", "image", "resources", "precision",
          "tensor_parallel_size", "max_num_seqs", "max_num_batched_tokens",
          "prefix_cache", "schema_token_cache", "workload_basis",
          "s1_p95_ms", "s2_decisions_per_second", "s2_concurrency"]
s1, s2 = SUM["s1"], SUM["s2"]
candidate = s2["throughput_candidate_points"]
candidate_c8 = next(
    point for point in candidate if point["measurement_id"].endswith("c8"))
summary_rows = [
    {"model": "clef", "dataset": "OpenDecisions", "image": IMAGE, "resources": RESOURCES,
     "precision": "bfloat16", "tensor_parallel_size": "2", "max_num_seqs": "1",
     "max_num_batched_tokens": "4096", "prefix_cache": "false", "schema_token_cache": "false",
     "workload_basis": "opendecisions-1.0.1-full2352", "s1_p95_ms": s1["p95_ms"],
     "s2_decisions_per_second": s2["max_decisions_per_second"],
     "s2_concurrency": "16"},
    {"model": "clef", "dataset": "OpenDecisions", "image": IMAGE, "resources": RESOURCES,
     "precision": "bfloat16", "tensor_parallel_size": "2", "max_num_seqs": "4",
     "max_num_batched_tokens": "4096", "prefix_cache": "false", "schema_token_cache": "true",
     "workload_basis": "opendecisions-1.0.1-full2352-throughput-candidate-quality-fail",
     "s1_p95_ms": "", "s2_decisions_per_second": candidate_c8["decisions_per_second"],
     "s2_concurrency": "8"},
]
res_path = os.path.join(REPO, "Clef-27B", "results.csv")
with open(res_path, "w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(summary_rows)
print(f"results.csv rewritten with accepted + candidate rows ({len(summary_rows)})")

quality = SUM["quality"]
metrics_path = os.path.join(REPO, "Clef-27B", "metrics.csv")
with open(metrics_path, "w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=[
        "model", "dataset", "accuracy", "score_mae",
        "calibration", "intelligence", "capability"], lineterminator="\n")
    writer.writeheader()
    writer.writerow({
        "model": "Cloudflare/clef",
        "dataset": "OpenDecisions",
        "accuracy": quality["accuracy"],
        "score_mae": quality["score_mae"],
        "calibration": quality["calibration"],
        "intelligence": quality["intelligence"],
        "capability": quality["capability"],
    })
print("metrics.csv rewritten with the final accepted quality row")
