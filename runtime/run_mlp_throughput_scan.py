#!/usr/bin/env python3
"""Explore concurrency on the frozen 588-item OpenDecisions subset.

The caller prepares a warm service. Each concurrency has one 588-request
window with zero additional warmup. The summary reports the highest observed
successful-request throughput together with latency and failure counts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

from run_mlp_dataset_eval import REPOSITORY, file_identity, now, verify_report, write_driver


ITEMS = 588
DEFAULT_SAMPLE_FILE = Path(
    "/workspace/yangchen/logs/gdn-decision-20261115/ab/ab-subset-stride4.json"
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-file", type=Path, default=DEFAULT_SAMPLE_FILE)
    parser.add_argument("--base-url", default="http://127.0.0.1:18143")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--concurrencies", nargs="+", type=int, default=[2, 4, 8, 16])
    args = parser.parse_args()
    if min(args.concurrencies) < 1 or len(set(args.concurrencies)) != len(args.concurrencies):
        parser.error("--concurrencies must contain distinct positive integers")
    output = args.output_dir.expanduser().resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        parser.error(f"--output-dir must be a new or empty directory: {output}")
    sample_file = args.sample_file.expanduser().resolve()
    if not sample_file.is_file():
        parser.error(f"missing sample file {sample_file}")
    samples = json.loads(sample_file.read_text())["samples"]
    if len(samples) != ITEMS:
        parser.error(f"expected {ITEMS} samples, found {len(samples)}")
    output.mkdir(parents=True, exist_ok=True)
    driver_path = output / "driver.json"
    summary_path = output / "summary.json"
    driver = {
        "started_at": now(), "status": "running", "steps": [],
        "scope": "subset588 exploratory scan; one window per concurrency; caller-prewarmed service",
        "configuration": {"base_url": args.base_url.rstrip("/"), "python": args.python,
                          "requests_per_point": ITEMS, "warmup": 0,
                          "concurrencies": args.concurrencies},
        "sample_file": file_identity(sample_file),
        "script": file_identity(Path(__file__).resolve()),
        "benchmark_script": file_identity(REPOSITORY / "scripts/benchmark_systemone.py"),
        "summary": str(summary_path),
    }
    summary = {
        "scope": driver["scope"], "sample_file": driver["sample_file"], "points": [],
        "peak_successful_requests_per_second": None,
        "peak_scope": "Highest observed successful requests per second among completed reports in this scan",
    }
    write_driver(driver_path, driver)
    write_driver(summary_path, summary)
    try:
        for concurrency in args.concurrencies:
            name = f"c{concurrency}"
            report_path = output / f"{name}.json"
            log_path = output / f"{name}.log"
            argv = [args.python, str(REPOSITORY / "scripts/benchmark_systemone.py"),
                    "--base-url", args.base_url.rstrip("/"), "--sample-file", str(sample_file),
                    "--requests", str(ITEMS), "--concurrency", str(concurrency),
                    "--warmup", "0", "--report", str(report_path)]
            step = {"name": name, "concurrency": concurrency, "argv": argv,
                    "cwd": str(REPOSITORY), "started_at": now(), "status": "running",
                    "returncode": None, "log": str(log_path), "report": str(report_path)}
            driver["steps"].append(step)
            write_driver(driver_path, driver)
            began = time.monotonic()
            try:
                with log_path.open("wb") as log:
                    result = subprocess.run(argv, cwd=REPOSITORY, stdout=log,
                                            stderr=subprocess.STDOUT, check=False)
                step["returncode"] = result.returncode
                raw = json.loads(report_path.read_text())
                step["verification"] = verify_report("od-full", report_path, ITEMS)
                metrics = raw["summary"]
                latency = metrics.get("successful_request_latency_ms") or {}
                point = {
                    "concurrency": concurrency,
                    "successful_requests_per_second": metrics["successful_requests_per_second"],
                    "p50_ms": latency.get("p50"), "p95_ms": latency.get("p95"),
                    "p99_ms": latency.get("p99"),
                    "successful_requests": metrics["successful_requests"],
                    "failed_requests": metrics["failed_requests"],
                    "completed_requests": metrics["completed_requests"],
                    "measurement_wall_seconds": metrics["measurement_wall_seconds"],
                    "passed": step["verification"]["passed"],
                    "returncode": result.returncode, "report": file_identity(report_path),
                }
                summary["points"].append(point)
                measured = [item for item in summary["points"]
                            if item["successful_requests_per_second"] is not None]
                summary["peak_successful_requests_per_second"] = max(
                    measured, key=lambda item: item["successful_requests_per_second"], default=None)
                step["status"] = "complete" if point["passed"] and result.returncode == 0 else "failed"
                if step["status"] == "failed":
                    raise RuntimeError(f"{name} failed standard benchmark validation; log {log_path}")
            except (Exception, KeyboardInterrupt) as exc:
                step["status"] = "failed"
                step["error"] = f"{type(exc).__name__}: {exc}"
                raise
            finally:
                step["finished_at"] = now()
                step["wall_seconds"] = time.monotonic() - began
                write_driver(driver_path, driver)
                write_driver(summary_path, summary)
                print(json.dumps(step, ensure_ascii=False), flush=True)
        driver["status"] = "complete"
    except (Exception, KeyboardInterrupt) as exc:
        driver["status"] = "failed"
        driver["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        driver["finished_at"] = now()
        summary["status"] = driver["status"]
        summary["failed_requests_total"] = sum(point["failed_requests"] for point in summary["points"])
        write_driver(driver_path, driver)
        write_driver(summary_path, summary)
    print(json.dumps({"driver": str(driver_path), "summary": str(summary_path),
                      "status": driver["status"]}, ensure_ascii=False))
    return 0 if driver["status"] == "complete" else 1


if __name__ == "__main__":
    sys.exit(main())
