#!/usr/bin/env python3
"""Warm up, bracket a short request window, stop in finally, and analyse ranks.

The server must already have --profiler-config enabled. This command never
launches or terminates a model service. Export analysis runs outside TP daemons.
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
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from verify_systemone import get_health


def now():
    return datetime.now(timezone.utc).isoformat()


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def control(base_url, endpoint, timeout):
    row = {"url": base_url.rstrip("/") + endpoint, "started_at": now()}
    started = time.perf_counter()
    try:
        with urlopen(Request(row["url"], data=b"", method="POST"), timeout=timeout) as response:
            row.update(http_status=response.status, body=response.read().decode(errors="replace"))
        row["passed"] = row["http_status"] == 200
    except HTTPError as exc:
        row.update(passed=False, http_status=exc.code, body=exc.read().decode(errors="replace"), error=str(exc))
    except OSError as exc:
        row.update(passed=False, error=f"{type(exc).__name__}: {exc}")
    row.update(finished_at=now(), elapsed_seconds=time.perf_counter() - started)
    return row


def execute(argv, log):
    row = {"argv": argv, "started_at": now(), "log": str(log)}
    with log.open("w") as handle:
        result = subprocess.run(argv, stdout=handle, stderr=subprocess.STDOUT)
    row.update(returncode=result.returncode, finished_at=now())
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--trace-root", type=Path, required=True)
    parser.add_argument("--expected-ranks", type=int, required=True)
    parser.add_argument("--sample-file", type=Path)
    parser.add_argument("--warmup-sample-file", type=Path, help="采集外使用独立同 shape 样本；用于保留真实媒体 encoder cache miss")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--requests", type=int, default=5)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--stop-timeout", type=float, default=600)
    parser.add_argument("--skip-analysis", action="store_true", help="保留原始数据，稍后逐 rank 独立解析")
    args = parser.parse_args()
    if args.warmup < 10 or args.requests < 1 or args.concurrency < 1 or args.expected_ranks < 1 or args.timeout <= 0 or not 300 <= args.stop_timeout <= 600:
        parser.error("预热至少 10 次，请求与 rank 数为正数，stop 超时为 300–600 秒")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    root = args.trace_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    before = {str(p.resolve()) for p in root.rglob("*_ascend_pt") if p.is_dir()}
    manifest = {"created_at": now(), "base_url": args.base_url, "trace_root": str(root),
                "expected_ranks": args.expected_ranks, "concurrency": args.concurrency, "passed": False,
                "collection_control_passed": False, "trace_completeness_passed": False,
                "acceptance_scope": "HTTP control and exported rank structure; request-window alignment, rank clocks and service logs also require review",
                "throughput_scope": "Profiled requests measure instrumentation overhead; use separate unprofiled rounds for performance."}
    if args.sample_file:
        manifest["sample"] = {"path": str(args.sample_file.resolve()), "sha256": hashlib.sha256(args.sample_file.read_bytes()).hexdigest()}
    if args.warmup_sample_file:
        manifest["warmup_sample"] = {"path": str(args.warmup_sample_file.resolve()), "sha256": hashlib.sha256(args.warmup_sample_file.read_bytes()).hexdigest()}
    save(output / "manifest.json", manifest)
    benchmark = Path(__file__).with_name("benchmark_systemone.py")
    common = [sys.executable, str(benchmark), "--base-url", args.base_url, "--warmup", "0", "--concurrency", str(args.concurrency), "--timeout", str(args.timeout)]
    requests_command = common + (["--sample-file", str(args.sample_file)] if args.sample_file else [])
    warmup_sample = args.warmup_sample_file or args.sample_file
    warmup_command = common + (["--sample-file", str(warmup_sample)] if warmup_sample else [])
    try:
        manifest["health"] = get_health(args.base_url.rstrip("/"), 0, args.timeout)
        if not manifest["health"]["passed"]:
            raise RuntimeError("健康检查失败")
        manifest["warmup"] = execute([*warmup_command, "--requests", str(args.warmup), "--report", str(output / "warmup.json")], output / "warmup.log")
        if manifest["warmup"]["returncode"]:
            raise RuntimeError("采集外预热失败")
        # A timed-out start may have reached workers; always send stop after the
        # attempt, including non-200 responses and interruption during requests.
        try:
            manifest["start"] = control(args.base_url, "/start_profile", args.timeout)
            save(output / "manifest.json", manifest)
            if not manifest["start"]["passed"]:
                raise RuntimeError("start_profile 未返回 HTTP 200")
            manifest["request_window_started_at"] = now()
            manifest["requests"] = execute([*requests_command, "--requests", str(args.requests), "--report", str(output / "profile-requests.json")], output / "profile-requests.log")
            manifest["request_window_finished_at"] = now()
        finally:
            manifest["stop"] = control(args.base_url, "/stop_profile", args.stop_timeout)
            save(output / "manifest.json", manifest)
        ranks = sorted(p.resolve() for p in root.rglob("*_ascend_pt") if p.is_dir() and str(p.resolve()) not in before)
        manifest["new_rank_directories"] = [str(path) for path in ranks]
        if len(ranks) != args.expected_ranks:
            raise RuntimeError(f"本窗口新 rank 目录数量 {len(ranks)}，预期 {args.expected_ranks}")
        manifest["analysis"] = []
        if not args.skip_analysis:
            for index, path in enumerate(ranks):
                # Independent ordinary Python process, not a TP daemon child.
                code = "from torch_npu.profiler.profiler import analyse; import sys; analyse(sys.argv[1], max_process_number=8)"
                manifest["analysis"].append(execute([sys.executable, "-c", code, str(path)], output / f"analyse-rank-{index}.log"))
        manifest["export_files"] = {str(path): [{"path": str(p), "bytes": p.stat().st_size} for p in sorted((path / "ASCEND_PROFILER_OUTPUT").glob("*")) if p.is_file()] for path in ranks}
        manifest["collection_control_passed"] = bool(manifest["stop"]["passed"] and manifest["requests"]["returncode"] == 0 and
                                                    all(row["returncode"] == 0 for row in manifest["analysis"]))
        analysis_command = [sys.executable, str(Path(__file__).with_name("analyze_systemone_profile.py")),
                            str(root), "--expected-ranks", str(args.expected_ranks),
                            "--output-dir", str(output / "trace-analysis"), "--strict"]
        for path in ranks:
            analysis_command += ["--trace-dir", str(path)]
        manifest["trace_completeness"] = execute(analysis_command, output / "trace-completeness.log")
        manifest["trace_completeness_passed"] = manifest["trace_completeness"]["returncode"] == 0
        manifest["passed"] = manifest["collection_control_passed"] and manifest["trace_completeness_passed"]
    except (Exception, KeyboardInterrupt) as exc:
        manifest["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        manifest["finished_at"] = now()
        save(output / "manifest.json", manifest)
    print(json.dumps({"passed": manifest["passed"], "manifest": str(output / "manifest.json"), "error": manifest.get("error")}, ensure_ascii=False))
    return 0 if manifest["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
