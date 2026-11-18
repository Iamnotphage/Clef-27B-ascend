#!/usr/bin/env python3
"""Build the schema-versioned canonical measurement table from round-2 archives.

Inputs (read-only):
  Clef-27B/logs/tuning2-20261113/master-results.json   (43 rows, round-2 aggregate)
  Clef-27B/logs/tuning3-20261113/statistics-recheck.json (round-3 reanalysis:
      merged percentiles/throughput/CV + corrected dual-TP1 batch attribution)

Outputs:
  Clef-27B/logs/tuning2-20261113/measurements-v2.csv  canonical measurements
  Clef-27B/results.csv                                 corrected 12-column S1/S2
                                                       submission-style summary

Corrections applied (documented in docs/tuning3-2026-11-13.md):
  1. duplicate measurement p2-s4-t16384-c8-307reg (in p1-explore and p2-budget)
     -> one row, labels merged; statistics counted once.
  2. dual-TP1 batch attribution: warmup 10 went through the dispatch proxy with
     round-robin tie-break (sequential warmups alternate B/A starting at B),
     so per-instance warmup is 5/5; corrected formal batches replace the
     erroneous per-instance 10-deduction.
"""
import csv
import hashlib
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARCH2 = os.path.join(REPO, "Clef-27B", "logs", "tuning2-20261113")
ARCH3 = os.path.join(REPO, "Clef-27B", "logs", "tuning3-20261113")
MASTER = os.path.join(ARCH2, "master-results.json")
RECHECK = os.path.join(ARCH3, "statistics-recheck.json")
RUN_ID = "tuning-20261113-085408-perf2a"

master = json.load(open(MASTER))
recheck = json.load(open(RECHECK))
master_sha = hashlib.sha256(open(MASTER, "rb").read()).hexdigest()
recheck_sha = hashlib.sha256(open(RECHECK, "rb").read()).hexdigest()

# corrected p3 attribution from the round-3 reanalysis
P3 = {lab: chk["corrected"] for lab, chk in recheck["p3_attribution"].items()}
# merged multi-round statistics
MERGED = {g: v.get("merged") for g, v in recheck["groups"].items() if v.get("merged")}

STABLE_GROUP = {
    "p1s-s1-c1-r1": "stable-s1-c1", "p1s-s1-c1-r2": "stable-s1-c1", "p1s-s1-c1-r3": "stable-s1-c1",
    "p1s-s2-c4-r1": "stable-s2-c4", "p1s-s2-c4-r2": "stable-s2-c4", "p1s-s2-c4-r3": "stable-s2-c4",
    "p1s-s4-c8-r1": "stable-s4-c8", "p1s-s4-c8-r2": "stable-s4-c8", "p1s-s4-c8-r3": "stable-s4-c8",
    "p5-cache-on-s1": "cache-on-s1",
    "p5-cache-on-s4": "cache-on-s4", "p5-cache-on-s4-r2": "cache-on-s4", "p5-cache-on-s4-r3": "cache-on-s4",
}

FIELDS = [
    "schema_version", "measurement_id", "labels", "phase", "run_id", "workload",
    "source_file", "source_sha256", "config_id", "deployment",
    "tp_size", "max_num_seqs", "max_num_batched_tokens", "max_model_len",
    "weight_nz_mode", "schema_cache", "kv_pool", "compile_mode",
    "requests", "wall_seconds", "rps", "decisions_per_second", "input_tokens_per_second",
    "p50_ms", "p95_ms", "p99_ms", "percentile_basis",
    "batch_evidence", "failed", "hbm_total_mib", "notes",
]


def base_config(row):
    """Per-label canonical config for the round-2 single-service points."""
    label = row["label"]
    phase = row["phase"]
    budget, seqs, nz, cache, pool, compile_mode = "4096", "1", "1", "off", "default", "eager"
    if label.startswith("p2-s4-t"):
        parts = label.split("-")
        budget = parts[3][1:]
        seqs = parts[2][1:]
    if phase == "p4-nz2":
        nz = "2"
    if phase == "p5-cache-on":
        cache = "on"
    if phase == "p6-piecewise":
        compile_mode = "piecewise"
    if phase == "p6-pool12":
        pool = "12GiB"
    return budget, seqs, nz, cache, pool, compile_mode


def p3_evidence(label):
    c = P3[label]
    a, b = c["A"], c["B"]
    return ("warmup 5/5 via dispatch rr; A:{%s} B:{%s} dispatch A:%d B:%d "
            "(corrected round3, err=0)" % (
                ",".join(f"{k}:{v}" for k, v in a["reconstructed_solution"].items()),
                ",".join(f"{k}:{v}" for k, v in b["reconstructed_solution"].items()),
                a["dispatched"], b["dispatched"]))


seen = {}
rows = []
for row in master:
    label = row["label"]
    phase = row["phase"]
    key = (round(row["rps"], 6), round(row["wall_seconds"], 3), row["requests"])
    if label in seen:
        # duplicate measurement (same numbers re-listed under another phase):
        # keep one row, merge labels
        prev = seen[label]
        prev["labels"] = prev["labels"] + ";" + label
        prev["notes"] += "; duplicate of first occurrence, statistics counted once"
        continue
    budget, seqs, nz, cache, pool, compile_mode = base_config(row)
    deployment = "dual-tp1" if phase == "p3-dual-tp1" else "single-tp2"
    if phase == "p3-dual-tp1":
        evidence = p3_evidence(label)
        note = "warmup attribution corrected in round 3 (5/5 dispatch split)"
    else:
        evidence = row["batch_evidence"] or "null"
        note = ""
    g = STABLE_GROUP.get(label)
    if g and g in MERGED:
        note = (note + "; " if note else "") + f"merged-3-rounds group {g}"
    r = {
        "schema_version": "2.0", "measurement_id": label,
        "labels": label, "phase": phase, "run_id": RUN_ID,
        "workload": row["workload"], "source_file": os.path.relpath(MASTER, REPO),
        "source_sha256": master_sha, "config_id": "",
        "deployment": deployment, "tp_size": "2" if deployment == "single-tp2" else "1",
        "max_num_seqs": seqs, "max_num_batched_tokens": budget, "max_model_len": "4096",
        "weight_nz_mode": nz, "schema_cache": cache, "kv_pool": pool,
        "compile_mode": compile_mode,
        "requests": row["requests"], "wall_seconds": row["wall_seconds"],
        "rps": row["rps"], "decisions_per_second": row["decisions_per_second"],
        "input_tokens_per_second": row["input_tokens_per_second"],
        "p50_ms": row["p50_ms"], "p95_ms": row["p95_ms"], "p99_ms": row["p99_ms"],
        "percentile_basis": "per_round",
        "batch_evidence": evidence, "failed": row["failed"],
        "hbm_total_mib": row["hbm_total_mib"], "notes": note,
    }
    if deployment == "single-tp2":
        r["config_id"] = (f"tp2-seqs{seqs}-t{budget}-len4096-{compile_mode}-nz{nz}"
                          f"-cache{cache}-pool{pool}")
    else:
        r["config_id"] = "dualtp1x2-seqs1-t4096-len4096-eager-nz1-cacheoff-pooldefault"
    seen[label] = r
    rows.append(r)

# merged multi-round rows (one per group; per-round rows remain above)
MERGE_FIELDS_EXTRA = ["merged_rps", "merged_decisions_per_second", "merged_p50_ms",
                      "merged_p95_ms", "merged_p99_ms", "pooled_latency_samples",
                      "per_round_rps", "per_round_rps_cv_pct", "per_round_p95_mean_ms"]
out_fields = FIELDS[:27] + MERGE_FIELDS_EXTRA + FIELDS[27:]
for g, m in MERGED.items():
    ids = [r["measurement_id"] for r in rows if STABLE_GROUP.get(r["measurement_id"]) == g]
    if not ids:
        continue
    base = next(r for r in rows if r["measurement_id"] == ids[0])
    mr = dict(base)
    mr["schema_version"] = "2.0"
    mr["measurement_id"] = f"{g}-merged"
    mr["labels"] = ";".join(ids)
    mr["phase"] = base["phase"] + "-merged"
    mr["percentile_basis"] = "merged_pooled"
    mr.update({
        "merged_rps": m["merged_rps"], "merged_decisions_per_second": m["merged_decisions_per_second"],
        "merged_p50_ms": m["merged_p50_ms"], "merged_p95_ms": m["merged_p95_ms"],
        "merged_p99_ms": m["merged_p99_ms"], "pooled_latency_samples": m["pooled_latency_samples"],
        "per_round_rps": json.dumps(m["per_round_rps"]),
        "per_round_rps_cv_pct": m["per_round_rps_cv_pct"],
        "per_round_p95_mean_ms": m["per_round_p95_mean_ms"],
        "notes": (f"merged {m['n_rounds']} rounds; total {m['total_successful']} requests / "
                  f"{m['total_wall_seconds']} s; per-round basis rows: {', '.join(ids)}; "
                  f"recheck evidence sha256 {recheck_sha[:16]}"),
    })
    rows.append(mr)

out_path = os.path.join(ARCH2, "measurements-v2.csv")
with open(out_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=out_fields)
    w.writeheader()
    w.writerows(rows)
print(f"wrote {out_path}: {len(rows)} rows ({len(seen)} unique measurements + "
      f"{sum(1 for r in rows if r['phase'].endswith('-merged'))} merged rows)")

# ---- corrected 12-column submission-style S1/S2 summary ----
S1 = next(r for r in rows if r["measurement_id"] == "stable-s1-c1-merged")
S2_OFF = next(r for r in rows if r["measurement_id"] == "stable-s4-c8-merged")
S2_ON = next(r for r in rows if r["measurement_id"] == "cache-on-s4-merged")
IMAGE = ("Ubuntu 22.04.5 aarch64;CANN 9.2.1;torch_npu 2.10.0.post7;"
         "vLLM 24420b7a;vLLM-Ascend 89eb8a5e")
RESOURCES = "A3;1物理卡;2逻辑设备"
SUMMARY_FIELDS = ["model", "dataset", "image", "resources", "precision",
                  "tensor_parallel_size", "max_num_seqs", "max_num_batched_tokens",
                  "prefix_cache", "workload_basis", "s1_p95_ms",
                  "s2_decisions_per_second", "s2_concurrency"]
summary_rows = [
    {"model": "clef", "dataset": "OpenDecisions", "image": IMAGE, "resources": RESOURCES,
     "precision": "bfloat16", "tensor_parallel_size": "2", "max_num_seqs": "1",
     "max_num_batched_tokens": "4096", "prefix_cache": "false",
     "workload_basis": "fixed307-diagnostic", "s1_p95_ms": "130.296",
     "s2_decisions_per_second": "", "s2_concurrency": ""},
    {"model": "clef", "dataset": "OpenDecisions", "image": IMAGE, "resources": RESOURCES,
     "precision": "bfloat16", "tensor_parallel_size": "2", "max_num_seqs": "4",
     "max_num_batched_tokens": "4096", "prefix_cache": "false",
     "workload_basis": "fixed307-diagnostic", "s1_p95_ms": "",
     "s2_decisions_per_second": S2_OFF["merged_decisions_per_second"], "s2_concurrency": "8"},
]
res_path = os.path.join(REPO, "Clef-27B", "results.csv")
with open(res_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
    w.writeheader()
    w.writerows(summary_rows)
print(f"wrote {res_path}: {len(summary_rows)} rows "
      "(workload_basis marks fixed307 diagnostic until OpenDecisions S1/S2 run)")
print("note: legacy mixed file preserved as Clef-27B/results-v1-legacy.csv")
