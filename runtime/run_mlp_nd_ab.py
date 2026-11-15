#!/usr/bin/env python3
"""Paired ND versus NZ INT8 weight-format experiment on real MLP captures.

Runs the partial-W8A8 probe with explicit weight formats in balanced order
(round 1 ND->NZ, round 2 NZ->ND), then reports per-shape NZ/ND ratios for the
gate-up-only and both-projection chains. Ratios above one mean ND is faster.
"""

import argparse
import hashlib
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

SHORT_TOKENS = [
    145, 188, 203, 204, 205, 220, 252, 254, 269, 270, 280, 281,
    307, 335, 383, 448, 449, 498, 511, 662, 1595,
]
CASES = [
    {"name": "od-0677-short", "case": "od-0677", "layers": [0, 31, 63],
     "tokens": SHORT_TOKENS, "rounds": 8, "iterations": 20},
    {"name": "erqa-0089-662", "case": "erqa-0089", "layers": [31],
     "tokens": [662], "rounds": 6, "iterations": 10},
    {"name": "erqa-0167-10294", "case": "erqa-0167", "layers": [31],
     "tokens": [10294], "rounds": 4, "iterations": 6},
]
ARMS = ("bf16", "both", "gate_up_only", "down_only")
METRICS = ("device_ms", "synchronized_wall_ms")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_report(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=1) + "\n")
    temporary.replace(path)


def capture_paths(captures, case_name, layer, rank):
    for case in captures["cases"]:
        if case["name"] != case_name:
            continue
        for tensor in case["tensors"]:
            metadata = tensor["metadata"]
            if metadata["layer"] == layer and metadata["rank"] == rank:
                return tensor["path"]
    raise ValueError(f"missing capture {case_name} layer {layer} rank {rank}")


def summarize(driver, output_dir):
    runs = {}
    for job in driver["jobs"]:
        if job["returncode"] != 0:
            continue
        key = (job["case"], job["layer"], job["round"], job["format"])
        runs[key] = json.loads(Path(job["report"]).read_text())

    rows = []
    for case in [item for item in CASES if item["name"] in driver.get("cases", [])]:
        for layer in case["layers"]:
            for tokens in case["tokens"]:
                pair_reports = []
                for round_index in (1, 2):
                    nd = runs.get((case["name"], layer, round_index, "nd"))
                    nz = runs.get((case["name"], layer, round_index, "nz"))
                    if nd and nz:
                        nd_shape = next(
                            (shape for shape in nd["shapes"] if shape["tokens"] == tokens), None)
                        nz_shape = next(
                            (shape for shape in nz["shapes"] if shape["tokens"] == tokens), None)
                        if nd_shape and nz_shape:
                            pair_reports.append((nd, nz, nd_shape, nz_shape))
                if not pair_reports:
                    continue
                first_nd, first_nz, _, _ = pair_reports[0]
                row = {"case": case["name"], "layer": layer, "tokens": tokens,
                       "nd_format": first_nd.get("expected_int8_format"),
                       "nz_format": first_nz.get("expected_int8_format"),
                       "nd_status": first_nd.get("status"), "nz_status": first_nz.get("status"),
                       "rounds": len(pair_reports)}
                for metric in METRICS:
                    row[metric] = {}
                    for arm in ARMS:
                        round_rows = []
                        ratios = []
                        for nd, nz, nd_shape, nz_shape in pair_reports:
                            nd_ms = nd_shape["summary"][metric][arm]["median_ms"]
                            nz_ms = nz_shape["summary"][metric][arm]["median_ms"]
                            ratio = nz_ms / nd_ms
                            ratios.append(ratio)
                            round_rows.append({"round": len(round_rows) + 1,
                                               "nd_ms": nd_ms, "nz_ms": nz_ms,
                                               "nd_speedup_vs_nz": ratio})
                        row[metric][arm] = {
                            "rounds": round_rows,
                            "median_nd_speedup_vs_nz": statistics.median(ratios),
                            "nd_faster_rounds": sum(value > 1 for value in ratios),
                        }
                rows.append(row)

    def bin_name(tokens):
        if 145 <= tokens <= 255:
            return "145-255"
        if 256 <= tokens <= 383:
            return "256-383"
        if 384 <= tokens <= 511:
            return "384-511"
        return ">=512"

    bins = {}
    for row in rows:
        bucket = bins.setdefault(
            bin_name(row["tokens"]),
            {"rows": [], **{metric: {} for metric in METRICS}},
        )
        bucket["rows"].append(row["tokens"])
        for metric in METRICS:
            for arm in ("gate_up_only", "both"):
                bucket[metric].setdefault(arm, []).append(
                    row[metric][arm]["median_nd_speedup_vs_nz"])
    for bucket in bins.values():
        bucket["count"] = len(bucket["rows"])
        for metric in METRICS:
            for arm in ("gate_up_only", "both"):
                values = bucket[metric].get(arm, [])
                bucket[metric][arm] = {
                    "median_nd_speedup_vs_nz": statistics.median(values) if values else None,
                    "mean_nd_speedup_vs_nz": statistics.mean(values) if values else None,
                    "nd_faster_shapes": sum(value > 1 for value in values),
                    "n": len(values),
                }
    return {
        "rows": rows,
        "bins": bins,
        "gate_up_device_nd_faster": sum(
            1 for row in rows
            if row["device_ms"]["gate_up_only"]["median_nd_speedup_vs_nz"] > 1),
        "gate_up_device_total": len(rows),
        "verdict": (
            "nd-faster" if rows and statistics.median(
                row["device_ms"]["gate_up_only"]["median_nd_speedup_vs_nz"] for row in rows) > 1.02
            else "nz-faster-or-neutral"
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--captures", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path,
                        default=Path("/workspace/yangchen/models/Clef-27B"))
    parser.add_argument("--device", type=int, choices=(0, 1), default=0)
    parser.add_argument("--rank", type=int, choices=(0, 1), default=0)
    parser.add_argument("--case", action="append", choices=[case["name"] for case in CASES],
                        help="limit to one or more named cases")
    args = parser.parse_args()
    selected = [case for case in CASES if not args.case or case["name"] in args.case]
    captures = json.loads(args.captures.read_text())
    if not captures.get("completed") or not captures.get("all_answers_equal"):
        parser.error("capture requests must be complete and match the frozen answers")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    driver_path = args.output_dir / "driver.json"
    if driver_path.exists():
        parser.error("use a new output directory for each run")
    probe = Path(__file__).with_name("probe_mlp_partial_w8a8.py")
    driver = {
        "captures": str(args.captures),
        "captures_sha256": sha256(args.captures),
        "probe": str(probe),
        "probe_sha256": sha256(probe),
        "device": args.device,
        "rank": args.rank,
        "model_dir": str(args.model_dir),
        "cases": [case["name"] for case in selected],
        "jobs": [],
        "completed": False,
    }
    write_report(driver_path, driver)
    for case in [item for item in CASES if item["name"] in driver.get("cases", [])]:
        for layer in case["layers"]:
            input_file = capture_paths(captures, case["case"], layer, args.rank)
            for round_index, order in enumerate((("nd", "nz"), ("nz", "nd")), start=1):
                for weight_format in order:
                    name = f"{case['name']}-l{layer:02d}-r{round_index}-{weight_format}"
                    report = args.output_dir / f"{name}.json"
                    argv = [
                        sys.executable, str(probe),
                        "--device", str(args.device),
                        "--model-dir", str(args.model_dir),
                        "--layer", str(layer), "--rank", str(args.rank),
                        "--input-file", input_file,
                        "--tokens", *[str(tokens) for tokens in case["tokens"]],
                        "--weight-format", weight_format,
                        "--warmup", "10",
                        "--rounds", str(case["rounds"]),
                        "--iterations", str(case["iterations"]),
                        "--report", str(report),
                    ]
                    job = {"case": case["name"], "layer": layer, "tokens": case["tokens"],
                           "round": round_index, "format": weight_format,
                           "argv": argv, "report": str(report),
                           "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
                    driver["jobs"].append(job)
                    write_report(driver_path, driver)
                    began = time.monotonic()
                    with (args.output_dir / f"{name}.log").open("wb") as log:
                        result = subprocess.run(argv, stdout=log, stderr=subprocess.STDOUT,
                                                timeout=1800)
                    job.update(returncode=result.returncode,
                               wall_seconds=time.monotonic() - began)
                    write_report(driver_path, driver)
                    print(json.dumps({k: job[k] for k in
                                      ("case", "layer", "round", "format", "returncode",
                                       "wall_seconds")}), flush=True)
                    if result.returncode:
                        return result.returncode
    driver["completed"] = True
    driver["summary"] = summarize(driver, args.output_dir)
    write_report(driver_path, driver)
    print(json.dumps({"completed": True,
                      "verdict": driver["summary"]["verdict"],
                      "gate_up_device_nd_faster":
                          f"{driver['summary']['gate_up_device_nd_faster']}/"
                          f"{driver['summary']['gate_up_device_total']}"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
