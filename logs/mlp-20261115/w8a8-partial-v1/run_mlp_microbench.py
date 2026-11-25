#!/usr/bin/env python3
"""Run serial full-chain MLP probes using verified captured activations."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time


def write_report(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--captures", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--kind", choices=("w8a8", "partial_w8a8", "ffn"), required=True)
    parser.add_argument("--layers", nargs="+", type=int, default=[0, 31, 63])
    parser.add_argument("--rank", type=int, choices=(0, 1), default=0)
    parser.add_argument("--device", type=int, choices=(0, 1), default=0)
    args = parser.parse_args()
    captures = json.loads(args.captures.read_text())
    if not captures.get("completed") or not captures.get("all_answers_equal"):
        parser.error("capture requests must be complete and match the frozen answers")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    driver_report = args.output_dir / "driver.json"
    if driver_report.exists():
        parser.error("use a new output directory for each run")
    script = Path(__file__).with_name(f"probe_mlp_{args.kind}.py")
    report = {
        "kind": args.kind,
        "captures": str(args.captures),
        "captures_sha256": hashlib.sha256(args.captures.read_bytes()).hexdigest(),
        "probe_sha256": hashlib.sha256(script.read_bytes()).hexdigest(),
        "device": args.device,
        "jobs": [],
        "completed": False,
    }
    write_report(driver_report, report)
    for case in captures["cases"]:
        selected = [item for item in case["tensors"]
                    if item["metadata"]["rank"] == args.rank
                    and item["metadata"]["layer"] in args.layers]
        if len(selected) != len(set(args.layers)):
            raise ValueError(f"missing capture layers for {case['name']}")
        for item in selected:
            layer = item["metadata"]["layer"]
            name = f"{case['name']}-l{layer:02d}-r{args.rank}"
            result_path = args.output_dir / f"{name}.json"
            argv = [sys.executable, str(script), "--device", str(args.device),
                    "--model-dir", str(args.model_dir), "--layer", str(layer),
                    "--rank", str(args.rank), "--input-file", item["path"],
                    "--warmup", "10", "--iterations", "30", "--report", str(result_path)]
            if args.kind in ("w8a8", "partial_w8a8"):
                argv += ["--rounds", "8" if args.kind == "partial_w8a8" else "6"]
            else:
                argv += ["--phase", "benchmark", "--repeats", "6"]
            job = {"name": name, "argv": argv, "report": str(result_path),
                   "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
            report["jobs"].append(job)
            write_report(driver_report, report)
            with (args.output_dir / f"{name}.log").open("wb") as log:
                began = time.monotonic()
                result = subprocess.run(argv, stdout=log, stderr=subprocess.STDOUT, timeout=900)
            job.update(returncode=result.returncode, wall_seconds=time.monotonic() - began)
            write_report(driver_report, report)
            print(json.dumps(job), flush=True)
            if result.returncode:
                return result.returncode
    report["completed"] = True
    write_report(driver_report, report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
