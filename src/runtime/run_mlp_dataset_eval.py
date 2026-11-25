#!/usr/bin/env python3
"""Evaluate an already-running Clef service on frozen OD2352 and ERQA400.

OD warmup is a total request count. It runs in a separate window, followed by
one complete concurrency-1 measurement. Each subprocess has a dedicated log
and a durable driver entry. Service lifecycle is managed by the caller.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time


REPOSITORY = Path(__file__).resolve().parents[1]
WORKSPACE = Path("/workspace/yangchen")
OD_REFERENCE = WORKSPACE / "logs/tuning/20261114-013000-fix4/opendecisions"
OD_AUTHOR = WORKSPACE / "logs/opendecisions-baseline-20261113-1600"


def now():
    return datetime.now(timezone.utc).isoformat()


def write_driver(path, driver):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(driver, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def file_identity(path):
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def verify_report(name, path, expected):
    report = json.loads(path.read_text())
    if name in ("od-warmup", "od-full"):
        summary = report["summary"]
        checks = {
            "passed": report["passed"],
            "complete": len(report["requests"]) == expected,
            "successful": summary["successful_requests"] == expected,
            "zero_failures": summary["failed_requests"] == 0,
        }
    elif name == "od-score":
        candidate = report["runs"]["candidate"]
        checks = {
            "protocol": candidate["protocol_failures"] == 0,
            "request_alignment": candidate["order_mismatch"] == 0,
            "complete": candidate["scoring_result"]["items"] == expected,
        }
    elif name in ("erqa-full", "erqa-score"):
        coverage = report["coverage"]
        checks = {
            "complete": coverage["fixed_total"] == expected,
            "successful": coverage["ok"] == expected,
            "zero_failures": coverage["failed"] == 0,
        }
        if name == "erqa-score":
            checks["requests_verified"] = report["request_verification"]["checked"]
            checks["request_alignment"] = report["request_verification"]["passed"]
            checks["summary_equal"] = report["summary_comparison"]["equal"]
    else:
        checks = {
            "complete": report["summary"]["items"] == expected,
            "zero_failures": report["summary"]["candidate_failed"] == 0,
            "alignment": report["alignment"]["passed"],
        }
    return {"report": file_identity(path), "checks": checks, "passed": all(checks.values())}


def run_step(driver, driver_path, name, argv, report, expected):
    log_path = driver_path.parent / f"{name}.log"
    step = {
        "name": name, "argv": [str(value) for value in argv],
        "cwd": str(REPOSITORY), "started_at": now(),
        "log": str(log_path), "report": str(report), "status": "running",
        "returncode": None,
    }
    driver["steps"].append(step)
    write_driver(driver_path, driver)
    began = time.monotonic()
    try:
        with log_path.open("wb") as log:
            result = subprocess.run(step["argv"], cwd=REPOSITORY, stdout=log,
                                    stderr=subprocess.STDOUT, check=False)
        step["returncode"] = result.returncode
        if result.returncode:
            raise RuntimeError(f"{name} exited {result.returncode}; log {log_path}")
        step["verification"] = verify_report(name, report, expected)
        if not step["verification"]["passed"]:
            raise RuntimeError(f"{name} report verification failed; report {report}")
        step["status"] = "complete"
    except BaseException as exc:
        step["status"] = "failed"
        step["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        step["finished_at"] = now()
        step["wall_seconds"] = time.monotonic() - began
        write_driver(driver_path, driver)
        print(json.dumps(step, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:18143")
    parser.add_argument("--warmup", type=int, default=2352,
                        help="Total OD warmup requests in a separate window, default 2352")
    parser.add_argument("--run-erqa", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--od-sample-file", type=Path, default=OD_REFERENCE / "all2352.json")
    parser.add_argument("--od-data", type=Path, default=WORKSPACE / "datasets/clef-full-20261112/full-data/open-decisions2352.jsonl")
    parser.add_argument("--od-reference", type=Path, default=OD_REFERENCE / "s1-final-full.json")
    parser.add_argument("--od-scorer", type=Path, default=OD_AUTHOR / "author-score.py")
    parser.add_argument("--od-template", type=Path, default=OD_AUTHOR / "evalscope/predictions/clef/clef_open_decisions_full2352_default.jsonl")
    parser.add_argument("--erqa-data-dir", type=Path, default=WORKSPACE / "datasets/erqa-official-20261114")
    parser.add_argument("--erqa-reference", type=Path, default=WORKSPACE / "logs/erqa-baseline-20261114/full/predictions.jsonl")
    args = parser.parse_args()
    if args.warmup < 0:
        parser.error("--warmup must be nonnegative")
    output = args.output_dir.expanduser().resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        parser.error(f"--output-dir must be a new or empty directory: {output}")
    sources = {}
    for name in ("od_sample_file", "od_data", "od_reference", "od_scorer", "od_template"):
        path = getattr(args, name).expanduser().resolve()
        setattr(args, name, path)
        if not path.is_file():
            parser.error(f"missing input {path}")
        sources[name] = file_identity(path)
    samples = json.loads(args.od_sample_file.read_text())["samples"]
    if len(samples) != 2352:
        parser.error(f"expected 2352 OD samples, found {len(samples)}")
    args.erqa_data_dir = args.erqa_data_dir.expanduser().resolve()
    args.erqa_reference = args.erqa_reference.expanduser().resolve()
    if args.run_erqa:
        for name, path in (
            ("erqa_manifest", args.erqa_data_dir / "manifest.json"),
            ("erqa_samples", args.erqa_data_dir / "samples.jsonl"),
            ("erqa_reference", args.erqa_reference),
        ):
            if not path.is_file():
                parser.error(f"missing input {path}")
            sources[name] = file_identity(path)
    output.mkdir(parents=True, exist_ok=True)
    driver_path = output / "driver.json"
    base_url = args.base_url.rstrip("/")
    driver = {
        "started_at": now(), "status": "running", "steps": [], "sources": sources,
        "configuration": {"base_url": base_url, "python": args.python,
                          "od_items": 2352, "od_concurrency": 1,
                          "od_warmup_total_requests": args.warmup, "run_erqa": args.run_erqa},
        "performance_scope": "Single full-dataset run at client concurrency 1; OD warmup is a separate window; ERQA runner timings include its recorded retry policy",
        "script": file_identity(Path(__file__).resolve()),
    }
    write_driver(driver_path, driver)
    benchmark = [args.python, REPOSITORY / "scripts/benchmark_systemone.py",
                 "--base-url", base_url, "--sample-file", args.od_sample_file,
                 "--concurrency", "1", "--warmup", "0"]
    try:
        if args.warmup:
            warmup_report = output / "od-warmup.json"
            run_step(driver, driver_path, "od-warmup", benchmark + ["--requests", str(args.warmup), "--report", warmup_report], warmup_report, args.warmup)
        od_report = output / "od-full.json"
        run_step(driver, driver_path, "od-full", benchmark + ["--requests", "2352", "--report", od_report], od_report, 2352)
        od_score = output / "od-quality-and-diff.json"
        run_step(driver, driver_path, "od-score", [
            args.python, REPOSITORY / "runtime/od_raw_acceptance.py",
            "--data", args.od_data, "--scorer", args.od_scorer,
            "--template", args.od_template, "--run", f"baseline={args.od_reference}",
            "--run", f"candidate={od_report}", "--reference", "baseline", "--output", od_score,
        ], od_score, 2352)
        if args.run_erqa:
            erqa = output / "erqa"
            run_step(driver, driver_path, "erqa-full", [
                args.python, REPOSITORY / "scripts/official_eval/erqa_baseline.py",
                "--data-dir", args.erqa_data_dir, "--output-dir", erqa,
                "--api-url", base_url + "/v1", "--scope", "full",
            ], erqa / "summary.json", 400)
            run_step(driver, driver_path, "erqa-score", [
                args.python, REPOSITORY / "scripts/official_eval/erqa_score.py",
                "--run-dir", erqa, "--data-dir", args.erqa_data_dir,
                "--output", erqa / "offline-score.json", "--verify-requests",
            ], erqa / "offline-score.json", 400)
            run_step(driver, driver_path, "erqa-diff", [
                args.python, REPOSITORY / "scripts/official_eval/erqa_compare.py",
                "--reference", args.erqa_reference, "--candidate", erqa / "predictions.jsonl",
                "--data-dir", args.erqa_data_dir, "--output", erqa / "diff.json",
            ], erqa / "diff.json", 400)
        driver["status"] = "complete"
    except (Exception, KeyboardInterrupt) as exc:
        driver["status"] = "failed"
        driver["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        driver["finished_at"] = now()
        write_driver(driver_path, driver)
    print(json.dumps({"driver": str(driver_path), "status": driver["status"]}, ensure_ascii=False))
    return 0 if driver["status"] == "complete" else 1


if __name__ == "__main__":
    sys.exit(main())
