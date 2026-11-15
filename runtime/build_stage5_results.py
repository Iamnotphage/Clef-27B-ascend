#!/usr/bin/env python3
"""Rebuild measurements-v2.csv, results.csv and metrics.csv from archived runs.
Every stage-5 row is regenerated; historical rows and other datasets stay.

Input:  Clef-27B/logs/tuning3-20261113/stage5-summary.json
        Clef-27B/logs/gdn-decision-20261115/entry-s1/{od-full.json.gz,launch-manifest.json}
Output: Clef-27B/logs/tuning2-20261113/measurements-v2.csv
        Clef-27B/results.csv
        Clef-27B/metrics.csv
"""
import csv
import gzip
import hashlib
import json
import os
from pathlib import Path

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARCH3 = os.path.join(REPO, "Clef-27B", "logs", "tuning3-20261113")
ARCH2 = os.path.join(REPO, "Clef-27B", "logs", "tuning2-20261113")
SUM = json.load(open(os.path.join(ARCH3, "stage5-summary.json")))
MV2 = os.path.join(ARCH2, "measurements-v2.csv")
metrics_path = os.path.join(REPO, "Clef-27B", "metrics.csv")
metric_fields = [
    "model", "dataset", "accuracy", "score_mae",
    "calibration", "intelligence", "capability",
]
other_metrics = []
if os.path.exists(metrics_path):
    with open(metrics_path, newline="") as handle:
        reader = csv.DictReader(handle)
        metric_fields = reader.fieldnames
        other_metrics = [
            row for row in reader
            if (row["model"], row["dataset"]) != ("Cloudflare/clef", "OpenDecisions")
        ]
ENTRY_S1 = Path(REPO) / "Clef-27B/logs/gdn-decision-20261115/entry-s1"
entry_report_path = ENTRY_S1 / "od-full.json.gz"
entry_manifest_path = ENTRY_S1 / "launch-manifest.json"
with gzip.open(entry_report_path, "rt", encoding="utf-8") as handle:
    entry_report = json.load(handle)
entry_manifest = json.loads(entry_manifest_path.read_text(encoding="utf-8"))
entry_summary = entry_report["summary"]
entry_effective = entry_manifest["effective"]
entry_gdn_enabled = entry_manifest["gdn_decision_prefill"]["enabled"]
entry_source = entry_report_path.relative_to(REPO).as_posix()
entry_manifest_source = entry_manifest_path.relative_to(REPO).as_posix()
entry_manifest_sha256 = hashlib.sha256(entry_manifest_path.read_bytes()).hexdigest()
entry_config_id = (
    f"tp{entry_effective['tensor_parallel_size']}"
    f"-seqs{entry_effective['max_num_seqs']}"
    f"-t{entry_effective['max_num_batched_tokens']}"
    f"-len{entry_effective['max_model_len']}"
    f"-eager-nz1-cache{entry_effective['schema_token_cache']}"
    f"-pooldefault-gdn{int(entry_gdn_enabled)}"
)
entry_s1 = {
    "schema_version": "2.0",
    "measurement_id": "od-s1-gdn-entry-c1",
    "labels": "od-s1-gdn-entry-c1",
    "phase": "stage5-s1",
    "run_id": "gdn-decision-20261115-entry-s1",
    "workload": "opendecisions-1.0.1",
    "source_file": entry_source,
    "source_sha256": hashlib.sha256(entry_report_path.read_bytes()).hexdigest(),
    "config_id": entry_config_id,
    "deployment": "single-tp2",
    "tp_size": entry_effective["tensor_parallel_size"],
    "max_num_seqs": entry_effective["max_num_seqs"],
    "max_num_batched_tokens": entry_effective["max_num_batched_tokens"],
    "max_model_len": entry_effective["max_model_len"],
    "weight_nz_mode": 1,
    "schema_cache": entry_effective["schema_token_cache"],
    "kv_pool": "default",
    "compile_mode": "eager",
    "requests": entry_summary["requested"],
    "wall_seconds": entry_summary["wall_seconds"],
    "rps": entry_summary["requests_per_second"],
    "decisions_per_second": entry_summary["decisions_per_second"],
    "input_tokens_per_second": sum(
        row["input_tokens"] for row in entry_report["requests"]
        if row["status"] == "ok"
    ) / entry_summary["wall_seconds"],
    "p50_ms": entry_summary["latency_ms"]["p50"],
    "p95_ms": entry_summary["latency_ms"]["p95"],
    "p99_ms": entry_summary["latency_ms"]["p99"],
    "percentile_basis": "per_run",
    "batch_evidence": "c1 sequential (B1)",
    "failed": entry_summary["failed"],
    "hbm_total_mib": "",
    "notes": (
        "accepted S1 through start.sh; full 2352; "
        f"gdn_decision_prefill={str(entry_gdn_enabled).lower()}; "
        f"launch_manifest={entry_manifest_source}; "
        f"launch_manifest_sha256={entry_manifest_sha256}"
    ),
}

with open(MV2, newline="") as handle:
    reader = csv.DictReader(handle)
    fields = reader.fieldnames
    rows = [
        row for row in reader
        if not row["phase"].startswith("stage5-")
    ]

existing = {row["measurement_id"] for row in rows}
for measurement in [*SUM["measurements"], entry_s1]:
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
          "s1_p95_ms", "s2_decisions_per_second", "s2_concurrency",
          "s1_gdn_decision_prefill", "s2_gdn_decision_prefill",
          "s1_measurement_id", "s2_measurement_id"]
s2 = SUM["s2"]
accepted_s2 = max(s2["points"], key=lambda point: point["decisions_per_second"])
candidate = s2["throughput_candidate_points"]
candidate_c8 = next(
    point for point in candidate if point["measurement_id"].endswith("c8"))
summary_rows = [
    {"model": "clef", "dataset": "OpenDecisions", "image": IMAGE, "resources": RESOURCES,
     "precision": "bfloat16", "tensor_parallel_size": "2", "max_num_seqs": "1",
     "max_num_batched_tokens": "4096", "prefix_cache": "false", "schema_token_cache": "false",
     "workload_basis": "opendecisions-1.0.1-full2352", "s1_p95_ms": entry_s1["p95_ms"],
     "s2_decisions_per_second": s2["max_decisions_per_second"],
     "s2_concurrency": s2["max_concurrency"],
     "s1_gdn_decision_prefill": str(entry_gdn_enabled).lower(),
     "s2_gdn_decision_prefill": "false",
     "s1_measurement_id": entry_s1["measurement_id"],
     "s2_measurement_id": accepted_s2["measurement_id"]},
    {"model": "clef", "dataset": "OpenDecisions", "image": IMAGE, "resources": RESOURCES,
     "precision": "bfloat16", "tensor_parallel_size": "2", "max_num_seqs": "4",
     "max_num_batched_tokens": "4096", "prefix_cache": "false", "schema_token_cache": "true",
     "workload_basis": "opendecisions-1.0.1-full2352-throughput-candidate-quality-fail",
     "s1_p95_ms": "", "s2_decisions_per_second": candidate_c8["decisions_per_second"],
     "s2_concurrency": "8", "s1_gdn_decision_prefill": "",
     "s2_gdn_decision_prefill": "false", "s1_measurement_id": "",
     "s2_measurement_id": candidate_c8["measurement_id"]},
]
res_path = os.path.join(REPO, "Clef-27B", "results.csv")
with open(res_path, "w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(summary_rows)
print(f"results.csv rewritten with accepted + candidate rows ({len(summary_rows)})")

quality = SUM["quality"]
with open(metrics_path, "w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=metric_fields, lineterminator="\n")
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
    writer.writerows(other_metrics)
print("metrics.csv updated with accepted OpenDecisions quality; other rows preserved")
