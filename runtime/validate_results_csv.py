#!/usr/bin/env python3
"""Validate schema-versioned result CSVs.

Usage:
  python3 validate_results_csv.py <file.csv> [--kind measurements|summary]

Checks:
  - every row has exactly the header's column count
  - schema_version present and uniform (measurements kind)
  - measurement_id unique (measurements kind)
  - numeric columns parse as float/int where present
  - required configuration fields non-empty (measurements kind)
  - batch_evidence is valid JSON or the documented corrected-evidence string
Exit code 0 = pass; 1 = violations found (printed).
"""
import csv
import json
import sys

NUMERIC = {"requests", "wall_seconds", "rps", "decisions_per_second",
           "input_tokens_per_second", "p50_ms", "p95_ms", "p99_ms",
           "failed", "hbm_total_mib", "merged_rps", "merged_decisions_per_second",
           "merged_p50_ms", "merged_p95_ms", "merged_p99_ms",
           "per_round_rps_cv_pct", "per_round_p95_mean_ms",
           "s1_p95_ms", "s2_decisions_per_second", "s2_concurrency"}
REQUIRED_NONEMPTY = ["schema_version", "measurement_id", "run_id", "workload",
                     "source_file", "source_sha256", "config_id", "deployment"]


def main():
    path = sys.argv[1]
    kind = "measurements"
    if "--kind" in sys.argv:
        kind = sys.argv[sys.argv.index("--kind") + 1]
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames
        rows = list(reader)
    problems = []
    width = len(fields)
    with open(path, newline="") as f:
        for i, raw in enumerate(csv.reader(f), start=1):
            if len(raw) != width:
                problems.append(f"line {i}: {len(raw)} columns, header has {width}")
    if kind == "measurements":
        ids = set()
        for i, row in enumerate(rows, start=2):
            for key in REQUIRED_NONEMPTY:
                if not row.get(key):
                    problems.append(f"line {i}: empty required field {key}")
            for key in NUMERIC & set(fields):
                v = row.get(key, "")
                if v not in ("", None):
                    try:
                        float(v)
                    except ValueError:
                        problems.append(f"line {i}: {key}={v!r} not numeric")
            mid = row.get("measurement_id")
            if mid in ids:
                problems.append(f"line {i}: duplicate measurement_id {mid}")
            ids.add(mid)
            ev = row.get("batch_evidence", "")
            if ev and ev != "null":
                if ev.startswith("{") or ev.startswith("["):
                    try:
                        json.loads(ev)
                    except json.JSONDecodeError:
                        problems.append(f"line {i}: batch_evidence not valid JSON")
                elif not (ev.startswith("warmup 5/5") or ev.startswith("A:{")):
                    problems.append(f"line {i}: batch_evidence unrecognized form: {ev[:40]}")
        versions = {row.get("schema_version") for row in rows}
        if len(versions) > 1:
            problems.append(f"mixed schema_version values: {versions}")
    else:  # summary kind: 12/13-column official-style rows
        for i, row in enumerate(rows, start=2):
            if not row.get("model") or not row.get("dataset"):
                problems.append(f"line {i}: model/dataset empty")
            for key in ("tensor_parallel_size", "max_num_seqs", "max_num_batched_tokens"):
                if not str(row.get(key, "")).strip():
                    problems.append(f"line {i}: {key} empty")
            s1, s2 = row.get("s1_p95_ms", ""), row.get("s2_decisions_per_second", "")
            if not s1 and not s2:
                problems.append(f"line {i}: row carries neither S1 nor S2 value")
            for key in ("s1_p95_ms", "s2_decisions_per_second"):
                v = row.get(key, "")
                if v:
                    try:
                        float(v)
                    except ValueError:
                        problems.append(f"line {i}: {key}={v!r} not numeric")
    if problems:
        print(f"FAIL {path}: {len(problems)} problems")
        for p in problems[:40]:
            print(" ", p)
        sys.exit(1)
    print(f"OK {path}: {len(rows)} rows, {width} columns ({kind})")


if __name__ == "__main__":
    main()
