#!/usr/bin/env python3
"""Append round-3 stage-5 acceptance measurements to measurements-v2.csv and
rewrite results.csv S1/S2 rows with the OpenDecisions 1.0.1 measured values.

Reads (remote-run evidence fetched into the repo archive):
  Clef-27B/logs/tuning3-20261113/stage5-summary.json
Writes:
  Clef-27B/logs/tuning2-20261113/measurements-v2.csv (schema 2.0, appended)
  Clef-27B/results.csv (12-column S1/S2, workload_basis=opendecisions-1.0.1)
"""
import csv
import hashlib
import json
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARCH3 = os.path.join(REPO, "Clef-27B", "logs", "tuning3-20261113")
ARCH2 = os.path.join(REPO, "Clef-27B", "logs", "tuning2-20261113")
SUM = json.load(open(os.path.join(ARCH3, "stage5-summary.json")))
MV2 = os.path.join(ARCH2, "measurements-v2.csv")

with open(MV2, newline="") as f:
    reader = csv.DictReader(f)
    fields = reader.fieldnames
    rows = list(reader)

existing = {r["measurement_id"] for r in rows}
MERGE_EXTRA = [f for f in fields if f.startswith(("merged_", "per_round_", "pooled"))]

for m in SUM["measurements"]:
    if m["measurement_id"] in existing:
        continue
    r = {k: "" for k in fields}
    r.update({k: str(v) for k, v in m.items() if k in fields})
    r["schema_version"] = "2.0"
    rows.append(r)

with open(MV2, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(rows)
print(f"measurements-v2.csv now {len(rows)} rows")

# results.csv: OpenDecisions S1/S2 measured rows
IMAGE = ("Ubuntu 22.04.5 aarch64;CANN 9.2.1;torch_npu 2.10.0.post7;"
         "vLLM 24420b7a;vLLM-Ascend 89eb8a5e")
RESOURCES = "A3;1物理卡;2逻辑设备"
FIELDS = ["model", "dataset", "image", "resources", "precision",
          "tensor_parallel_size", "max_num_seqs", "max_num_batched_tokens",
          "prefix_cache", "schema_token_cache", "workload_basis",
          "s1_p95_ms", "s2_decisions_per_second", "s2_concurrency"]
s1, s2 = SUM["s1"], SUM["s2"]
summary_rows = [
    {"model": "clef", "dataset": "OpenDecisions", "image": IMAGE, "resources": RESOURCES,
     "precision": "bfloat16", "tensor_parallel_size": "2", "max_num_seqs": "1",
     "max_num_batched_tokens": "4096", "prefix_cache": "false", "schema_token_cache": "false",
     "workload_basis": "opendecisions-1.0.1-full2352", "s1_p95_ms": s1["p95_ms"],
     "s2_decisions_per_second": "", "s2_concurrency": ""},
    {"model": "clef", "dataset": "OpenDecisions", "image": IMAGE, "resources": RESOURCES,
     "precision": "bfloat16", "tensor_parallel_size": "2", "max_num_seqs": "4",
     "max_num_batched_tokens": "4096", "prefix_cache": "false", "schema_token_cache": "true",
     "workload_basis": "opendecisions-1.0.1-full2352", "s1_p95_ms": "",
     "s2_decisions_per_second": s2["decisions_per_second"], "s2_concurrency": s2["concurrency"]},
]
res_path = os.path.join(REPO, "Clef-27B", "results.csv")
with open(res_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=FIELDS)
    w.writeheader()
    w.writerows(summary_rows)
print(f"results.csv rewritten with OpenDecisions measured S1/S2 ({len(summary_rows)} rows)")
