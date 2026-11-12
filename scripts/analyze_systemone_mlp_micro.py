#!/usr/bin/env python3
"""Summarize same-signature msprof matmul replay phases without device work."""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics


def clean(value):
    return str(value).replace('"', "").replace(" ", "").strip()


def evidence(path):
    data = path.read_bytes()
    return {"path": str(path), "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", default="-")
    args = parser.parse_args()
    root = args.root.resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    report = {"created_at": datetime.now(timezone.utc).isoformat(), "root": str(root),
        "manifest": manifest, "passed": True, "runs": [],
        "baseline_signature": {"name": "aclnnMatmul_MatMulCommon_MatMulV2", "shape": "307,5120;34816,5120",
            "dtype": "DT_BF16;DT_BF16", "format": "ND;ND", "output_shape": "307,34816", "block_num": 24},
        "scope": "Synthetic single-op hardware-counter replay with the exported baseline signature; physical baseline strides and full-model cache state remain unverified",
        "bound_interpretation": "Native PipeUtilization establishes high MTE2 busy time. Replay memory/L0/L2 counters add evidence; absolute HBM saturation needs a validated device-wide bandwidth denominator and counter aggregation scope.",
        "evidence_index": [evidence(root / "manifest.json"), evidence(root / "before-npu-smi.txt")]}
    for run in manifest["runs"]:
        if Path(run["log"]).is_file():
            report["evidence_index"].append(evidence(Path(run["log"])))
    for path in sorted(root.glob("launch-*.sh")):
        report["evidence_index"].append(evidence(path))
    replay_source = root.parent / "tools/repro_systemone_mlp.py"
    if replay_source.is_file():
        report["evidence_index"].append(evidence(replay_source))
    for metric in ("Memory", "MemoryL0", "L2Cache"):
        replay_file = root / f"replay-{metric}.json"
        replay = json.loads(replay_file.read_text())
        sources = sorted((root / f"{metric}-retry").glob("**/op_summary_*.csv"))
        matching = []
        for path in sources:
            with path.open() as stream:
                for number, row in enumerate(csv.DictReader(stream), 2):
                    if row.get("OP Type") == "MatMulV2" and clean(row.get("Input Shapes")) == "307,5120;34816,5120":
                        matching.append((float(row["Task Start Time(us)"]), number, path, row))
            report["evidence_index"].append(evidence(path))
        matching.sort(key=lambda row: row[0])
        signature_ok = len(matching) == 2 * (replay["warmup"] + replay["iterations"]) and all(
            row[3].get("Op Name") == report["baseline_signature"]["name"]
            and row[3].get("Input Data Types") == "DT_BF16;DT_BF16"
            and row[3].get("Input Formats") == "ND;ND"
            and clean(row[3].get("Output Shapes")) == "307,34816"
            and int(row[3]["Block Num"]) == 24 for row in matching)
        run = {"metric": metric, "replay": replay, "matching_matmul_count": len(matching),
               "signature_matches_native_exported_fields": signature_ok, "phases": []}
        report["passed"] = report["passed"] and replay["passed"] and signature_ok
        report["evidence_index"].append(evidence(replay_file))
        for phase in replay["phases"]:
            start, end = phase["matching_matmul_formal_index_range_0based_inclusive"]
            selected = matching[start:end + 1]
            if len(selected) != phase["iterations"]:
                run["phases"].append({"name": phase["name"], "sample_count": len(selected),
                                      "error": "Missing expected chronological matching kernels"})
                report["passed"] = False
                continue
            duration = [float(row[3]["Task Duration(us)"]) for row in selected]
            fields = [key for key in selected[0][3] if key.startswith(("aic_", "aiv_")) or key == "aicore_time(us)"]
            counters = {}
            for field in fields:
                values = [float(row[3][field]) for row in selected if row[3].get(field) not in ("", "N/A")]
                if values:
                    counters[field] = {"min": min(values), "max": max(values), "mean": statistics.mean(values)}
            summary = {"name": phase["name"], "sample_count": len(selected),
                "task_duration_us": {"mean": statistics.mean(duration), "p50": statistics.median(duration),
                    "min": min(duration), "max": max(duration)}, "hardware_counters_source_labels": counters,
                "evidence": [{"path": str(row[2]), "csv_row": row[1], "task_start_time_us": row[3]["Task Start Time(us)"].strip()}
                             for row in selected]}
            if metric == "L2Cache":
                hit = sum(float(row[3][f"aic_r{channel}_read_cache_hit"]) for row in selected for channel in (0, 1))
                miss = sum(float(row[3][f"aic_r{channel}_read_cache_miss_allocate"]) for row in selected for channel in (0, 1))
                summary["derived_l2_read_hit_fraction_r0_r1"] = hit / (hit + miss) if hit + miss else None
                summary["l2_fraction_scope"] = "read_hit / (read_hit + read_miss_allocate), summing exported r0/r1 counters; preserves exporter counter definition"
            run["phases"].append(summary)
        report["runs"].append(run)
    payload = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.output == "-":
        print(payload, end="")
    else:
        Path(args.output).write_text(payload)


if __name__ == "__main__":
    main()
