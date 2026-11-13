#!/usr/bin/env python3
"""Aggregate every tuning-round measurement into one master CSV/JSON."""
import csv
import glob
import json
import os
import statistics
import sys

RUN = sys.argv[1]


def point(path):
    try:
        s = json.load(open(os.path.join(path, "point-summary.json")))
    except (OSError, json.JSONDecodeError):
        return None
    b = {}
    bpath = os.path.join(path, "batch-reconstruction.json")
    if os.path.exists(bpath):
        b = json.load(open(bpath))
    return s, b


def round_stats(paths, key="rps"):
    vals = []
    for p in paths:
        d = point(p)
        if d and d[0].get(key):
            vals.append(d[0][key])
    if not vals:
        return None, None
    mean = statistics.mean(vals)
    cv = statistics.stdev(vals) / mean * 100 if len(vals) > 1 else 0.0
    return mean, cv


rows = []

# phase 1: exploration + extension
for p in sorted(glob.glob(os.path.join(RUN, "p1", "p1-*")) + glob.glob(os.path.join(RUN, "p2-s4-t16384-c8-307reg"))):
    d = point(p)
    if not d:
        continue
    s, b = d
    rows.append({
        "phase": "p1-explore" if "/p1/p1-" in p.replace("/p1/p1s-", "/p1/XX") and "p1s" not in s["label"] else "p1-explore",
        "label": s["label"], "workload": "fixed307", "config": s["label"].replace("p1-", "").replace("-", " "),
        "requests": s["successful"], "wall_seconds": round(s["wall_seconds"], 3),
        "rps": round(s["rps"], 4), "decisions_per_second": round(s["decisions_per_second"], 3),
        "input_tokens_per_second": round(s["input_tokens_per_second"], 1),
        "p50_ms": round(s["p50_ms"], 2), "p95_ms": round(s["p95_ms"], 2), "p99_ms": round(s["p99_ms"], 2),
        "batch_evidence": json.dumps(b.get("batch_solution")),
        "failed": s["failed"], "hbm_total_mib": s.get("hbm_max_total_used_mib"),
    })

# phase 1 stable rounds
stable_groups = {}
for p in sorted(glob.glob(os.path.join(RUN, "p1", "p1s-*"))):
    d = point(p)
    if not d:
        continue
    s, b = d
    lab = s["label"].rsplit("-r", 1)[0]
    stable_groups.setdefault(lab, []).append((p, s, b))
    rows.append({
        "phase": "p1-stable", "label": s["label"], "workload": "fixed307",
        "config": lab.replace("p1s-", "").replace("-", " "),
        "requests": s["successful"], "wall_seconds": round(s["wall_seconds"], 3),
        "rps": round(s["rps"], 4), "decisions_per_second": round(s["decisions_per_second"], 3),
        "input_tokens_per_second": round(s["input_tokens_per_second"], 1),
        "p50_ms": round(s["p50_ms"], 2), "p95_ms": round(s["p95_ms"], 2), "p99_ms": round(s["p99_ms"], 2),
        "batch_evidence": json.dumps(b.get("batch_solution")),
        "failed": s["failed"], "hbm_total_mib": s.get("hbm_max_total_used_mib"),
    })
for lab, entries in stable_groups.items():
    mean, cv = round_stats([e[0] for e in entries])
    p95s = [e[1]["p95_ms"] for e in entries]
    print(f"STABLE {lab}: rounds={len(entries)} rps_mean={mean:.3f} cv={cv:.2f}% p95={p95s}")

# phase 2 token budget (long inputs)
for p in sorted(glob.glob(os.path.join(RUN, "p2-s4-t*"))):
    d = point(p)
    if not d:
        continue
    s, b = d
    workload = "long3868" if "307reg" not in s["label"] else "fixed307"
    rows.append({
        "phase": "p2-budget", "label": s["label"], "workload": workload,
        "config": s["label"].replace("p2-", "").replace("-", " "),
        "requests": s["successful"], "wall_seconds": round(s["wall_seconds"] or 0, 3),
        "rps": round(s["rps"] or 0, 4), "decisions_per_second": round(s["decisions_per_second"] or 0, 3),
        "input_tokens_per_second": round(s["input_tokens_per_second"] or 0, 1),
        "p50_ms": round(s["p50_ms"] or 0, 2), "p95_ms": round(s["p95_ms"] or 0, 2), "p99_ms": round(s["p99_ms"] or 0, 2),
        "batch_evidence": json.dumps(b.get("batch_solution")),
        "failed": s["failed"], "hbm_total_mib": s.get("hbm_max_total_used_mib"),
    })

# phase 3 dual TP1
for p in sorted(glob.glob(os.path.join(RUN, "p3-c*"))):
    d = point(p)
    if not d:
        continue
    s, b = d
    bA = json.load(open(os.path.join(p, "batch-A.json"))) if os.path.exists(os.path.join(p, "batch-A.json")) else {}
    bB = json.load(open(os.path.join(p, "batch-B.json"))) if os.path.exists(os.path.join(p, "batch-B.json")) else {}
    proxy = {}
    ppath = os.path.join(p, "proxy-stats.json")
    if os.path.exists(ppath):
        raw = json.load(open(ppath))
        proxy = {k.split("//")[1]: v.get("dispatched") for k, v in raw.items() if k.startswith("http")}
    rows.append({
        "phase": "p3-dual-tp1", "label": s["label"], "workload": "fixed307",
        "config": "dual tp1 seqs1+seqs1 via dispatch entry",
        "requests": s["successful"], "wall_seconds": round(s["wall_seconds"], 3),
        "rps": round(s["rps"], 4), "decisions_per_second": round(s["decisions_per_second"], 3),
        "input_tokens_per_second": round(s["input_tokens_per_second"], 1),
        "p50_ms": round(s["p50_ms"], 2), "p95_ms": round(s["p95_ms"], 2), "p99_ms": round(s["p99_ms"], 2),
        "batch_evidence": "A:%s B:%s dispatch:%s" % (json.dumps(bA.get("batch_solution")), json.dumps(bB.get("batch_solution")), proxy),
        "failed": s["failed"], "hbm_total_mib": s.get("hbm_max_total_used_mib"),
    })

# phase 4 NZ
for p in sorted(glob.glob(os.path.join(RUN, "p4nz2-*"))):
    d = point(p)
    if not d:
        continue
    s, b = d
    rows.append({
        "phase": "p4-nz2", "label": s["label"], "workload": "fixed307",
        "config": s["label"].replace("p4nz2-", "").replace("-", " ") + " weight_nz_mode=2",
        "requests": s["successful"], "wall_seconds": round(s["wall_seconds"], 3),
        "rps": round(s["rps"], 4), "decisions_per_second": round(s["decisions_per_second"], 3),
        "input_tokens_per_second": round(s["input_tokens_per_second"], 1),
        "p50_ms": round(s["p50_ms"], 2), "p95_ms": round(s["p95_ms"], 2), "p99_ms": round(s["p99_ms"], 2),
        "batch_evidence": json.dumps(b.get("batch_solution")),
        "failed": s["failed"], "hbm_total_mib": s.get("hbm_max_total_used_mib"),
    })

# phase 5 cache
for p in sorted(glob.glob(os.path.join(RUN, "p5-*s*"))):
    if not os.path.basename(p).startswith("p5-cache-on"):
        continue
    d = point(p)
    if not d:
        continue
    s, b = d
    rows.append({
        "phase": "p5-cache-on", "label": s["label"], "workload": "fixed307",
        "config": s["label"].replace("p5-cache-on-", "").replace("-", " ") + " schema_cache=on",
        "requests": s["successful"], "wall_seconds": round(s["wall_seconds"] or 0, 3),
        "rps": round(s["rps"] or 0, 4), "decisions_per_second": round(s["decisions_per_second"] or 0, 3),
        "input_tokens_per_second": round(s["input_tokens_per_second"] or 0, 1),
        "p50_ms": round(s["p50_ms"] or 0, 2), "p95_ms": round(s["p95_ms"] or 0, 2), "p99_ms": round(s["p99_ms"] or 0, 2),
        "batch_evidence": json.dumps(b.get("batch_solution")),
        "failed": s["failed"], "hbm_total_mib": s.get("hbm_max_total_used_mib"),
    })

# phase 6
for pat, phase, cfg in [
    ("p6-compile-s4", "p6-piecewise", "s4 t4096 eager-off piecewise"),
    ("p6-pool12-s4", "p6-pool12", "s4 t4096 kv_pool=12GiB"),
]:
    p = os.path.join(RUN, pat)
    d = point(p)
    if d:
        s, b = d
        rows.append({
            "phase": phase, "label": s["label"], "workload": "fixed307",
            "config": cfg,
            "requests": s["successful"], "wall_seconds": round(s["wall_seconds"], 3),
            "rps": round(s["rps"], 4), "decisions_per_second": round(s["decisions_per_second"], 3),
            "input_tokens_per_second": round(s["input_tokens_per_second"], 1),
            "p50_ms": round(s["p50_ms"], 2), "p95_ms": round(s["p95_ms"], 2), "p99_ms": round(s["p99_ms"], 2),
            "batch_evidence": json.dumps(b.get("batch_solution")),
            "failed": s["failed"], "hbm_total_mib": s.get("hbm_max_total_used_mib"),
        })

fields = ["phase", "label", "workload", "config", "requests", "wall_seconds", "rps",
          "decisions_per_second", "input_tokens_per_second", "p50_ms", "p95_ms", "p99_ms",
          "batch_evidence", "failed", "hbm_total_mib"]
with open(os.path.join(RUN, "master-results.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(rows)
json.dump(rows, open(os.path.join(RUN, "master-results.json"), "w"), indent=1)
print("rows:", len(rows), "->", os.path.join(RUN, "master-results.csv"))
