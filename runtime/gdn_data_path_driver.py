#!/usr/bin/env python3
"""Drive the GDN data-path switch matrix inside one service process.

The service must be started with ``CLEF_GDN_FLAG_FILE`` pointing at the same
path this driver writes; ``--service-pid`` checks that before the first
request so an A/B cannot silently compare two identical arms.  Every switch is
an atomic JSON rewrite (temp file + ``os.replace``) with a fresh ``epoch``.
After the rewrite the driver sends one warm-up request and reads the service
log until both TP worker processes have printed the matching
``clef-gdn-path epoch=...`` line, which proves both ranks adopted the same
configuration before timing starts.

Two modes share the same machinery:

``--arms R,B,I,C,BI,BC,IC,BIC --rounds N``
    Screening: every arm runs once per round on the same sample list.  The
    order rotates and reverses every round so a warm-up ramp cannot favour one
    arm.  Each arm's report is written as soon as it finishes.

``--arms R,BIC --pairs 6``
    Paired A/B: the two arms alternate order every pair, and every pair keeps
    both reports plus the paired p95/mean/throughput deltas.

The original flag file is restored on every exit path, including exceptions.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ARMS = {
    # decision_prefill and no_recompute are fixed on for every arm: this round
    # measures the data path, not the accepted state-export optimization.
    "R": dict(bsnd=False, skip_initial_state=False, skip_conv_state=False),
    "B": dict(bsnd=True, skip_initial_state=False, skip_conv_state=False),
    "I": dict(bsnd=False, skip_initial_state=True, skip_conv_state=False),
    "C": dict(bsnd=False, skip_initial_state=False, skip_conv_state=True),
    "BI": dict(bsnd=True, skip_initial_state=True, skip_conv_state=False),
    "BC": dict(bsnd=True, skip_initial_state=False, skip_conv_state=True),
    "IC": dict(bsnd=False, skip_initial_state=True, skip_conv_state=True),
    "BIC": dict(bsnd=True, skip_initial_state=True, skip_conv_state=True),
}
SETTLE_SECONDS = 1.2


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def service_flag_file(pid: int) -> str | None:
    try:
        raw = Path(f"/proc/{pid}/environ").read_bytes()
    except OSError:
        return None
    for item in raw.split(b"\0"):
        if item.startswith(b"CLEF_GDN_FLAG_FILE="):
            return item.split(b"=", 1)[1].decode("utf-8", "replace")
    return None


def service_log_path(pid: int) -> Path | None:
    try:
        target = os.readlink(f"/proc/{pid}/fd/1")
    except OSError:
        return None
    if target.startswith("/") and Path(target).is_file():
        return Path(target)
    return None


def next_epoch(path: Path) -> int:
    epoch = int(time.time_ns() // 1_000_000)
    try:
        previous = json.loads(path.read_text(encoding="utf-8")).get("epoch", 0)
    except (OSError, ValueError, AttributeError):
        previous = 0
    if isinstance(previous, int) and not isinstance(previous, bool):
        epoch = max(epoch, previous + 1)
    return epoch


def write_flags(path: Path, arm: str, epoch: int) -> dict:
    payload = {
        "epoch": epoch,
        "decision_prefill": True,
        "no_recompute": True,
        **ARMS[arm],
        "preprocess_fusion": False,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{epoch}")
    temporary.write_text(
        json.dumps(payload, separators=(",", ":")) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)
    return payload


def post(url: str, payload: dict, timeout: float) -> None:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        response.read()


def wait_for_epoch(log_path: Path, epoch: int, pids: int, timeout: float) -> list[int]:
    """Wait until ``pids`` distinct worker processes logged this epoch."""
    marker = f"clef-gdn-path epoch={epoch} "
    deadline = time.monotonic() + timeout
    seen: dict[int, int] = {}
    while time.monotonic() < deadline:
        try:
            with log_path.open("rb") as handle:
                handle.seek(0, os.SEEK_END)
                size = handle.tell()
                handle.seek(max(0, size - 4_000_000))
                text = handle.read().decode("utf-8", "replace")
        except OSError:
            text = ""
        for line in text.splitlines():
            if marker not in line or " pid=" not in line:
                continue
            tail = line.split(" pid=", 1)[1].split()[0]
            if tail.isdigit():
                seen[int(tail)] = 1
        if len(seen) >= pids:
            return sorted(seen)
        time.sleep(0.5)
    raise SystemExit(
        f"epoch {epoch}: only {len(seen)} rank(s) logged the switch, "
        f"expected {pids}: {sorted(seen)}"
    )


def run_benchmark(args, report: Path) -> dict:
    command = [
        args.python,
        "-u",
        str(args.benchmark),
        "--sample-file",
        str(args.sample_file),
        "--url",
        args.url,
        "--concurrency",
        str(args.concurrency),
        "--warmup",
        str(args.warmup),
        "--report",
        str(report),
    ]
    if args.limit is not None:
        command += ["--limit", str(args.limit)]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise SystemExit(f"benchmark failed: {result.stderr[-2000:]}")
    return json.loads(report.read_text(encoding="utf-8"))


def summarize(report: dict, path: Path) -> dict:
    summary = report["summary"]
    latency = summary["latency_ms"]
    return {
        "report": str(path),
        "successful": summary["successful"],
        "failed": summary.get("failed"),
        "decisions_per_second": summary["decisions_per_second"],
        "p50": latency["p50"],
        "p95": latency["p95"],
        "p99": latency["p99"],
        "mean": latency["mean"],
    }


def orders(arms: list[str], rounds: int) -> list[list[str]]:
    """Rotate and reverse every other round to balance position effects."""
    result = []
    for index in range(rounds):
        shift = index % len(arms)
        order = arms[shift:] + arms[:shift]
        if index % 2 == 1:
            order = list(reversed(order))
        result.append(order)
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument(
        "--benchmark",
        type=Path,
        default=Path(__file__).with_name("od_full_benchmark.py"),
    )
    parser.add_argument("--sample-file", type=Path, required=True)
    parser.add_argument("--url", default="http://127.0.0.1:18143/v1/systemone")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--arms", required=True, help="e.g. R,BIC or R,B,I,C")
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--pairs", type=int, default=0, help="paired A/B count")
    parser.add_argument("--flag-file", type=Path, required=True)
    parser.add_argument("--service-pid", type=int, required=True)
    parser.add_argument("--service-log", type=Path, default=None)
    parser.add_argument("--verify-ranks", type=int, default=2)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    arms = [item.strip() for item in args.arms.split(",") if item.strip()]
    if not arms or any(item not in ARMS for item in arms):
        parser.error(f"unknown arm in {args.arms!r}; choose from {sorted(ARMS)}")
    if args.pairs and len(arms) != 2:
        parser.error("--pairs requires exactly two arms")
    if args.pairs and args.rounds != 1:
        parser.error("--pairs and --rounds are separate modes")

    configured = service_flag_file(args.service_pid)
    if configured is None or Path(configured).resolve() != args.flag_file.resolve():
        print(json.dumps({
            "error": "service did not opt into this flag file",
            "service_CLEF_GDN_FLAG_FILE": configured,
            "requested": str(args.flag_file),
        }, ensure_ascii=False))
        return 2
    log_path = args.service_log or service_log_path(args.service_pid)
    if log_path is None:
        parser.error("cannot resolve the service log; pass --service-log")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    original = args.flag_file.read_bytes() if args.flag_file.exists() else None
    report = {
        "created_at": now(),
        "arms": arms,
        "mode": "paired" if args.pairs else "screen",
        "sample_file": str(args.sample_file),
        "limit": args.limit,
        "warmup": args.warmup,
        "concurrency": args.concurrency,
        "flag_file": str(args.flag_file),
        "service_log": str(log_path),
        "runs": [],
    }
    try:
        if args.pairs:
            plan = [arms if index % 2 == 0 else list(reversed(arms))
                    for index in range(args.pairs)]
        else:
            plan = orders(arms, args.rounds)
        sequence = 0
        for round_index, order in enumerate(plan, start=1):
            round_record = {"round": round_index, "order": order, "arms": {}}
            for arm in order:
                sequence += 1
                epoch = next_epoch(args.flag_file)
                payload = write_flags(args.flag_file, arm, epoch)
                time.sleep(SETTLE_SECONDS)
                sample = json.loads(args.sample_file.read_text(encoding="utf-8"))
                samples = sample["samples"] if isinstance(sample, dict) else sample
                post(args.url, samples[0]["request"], 300)
                ranks = wait_for_epoch(log_path, epoch, args.verify_ranks, timeout=30)
                stem = f"r{round_index:02d}-{arm}"
                benchmark_report = args.output_dir / f"{stem}.json"
                report_data = run_benchmark(args, benchmark_report)
                summary = summarize(report_data, benchmark_report)
                summary["epoch"] = epoch
                summary["ranks"] = ranks
                summary["flags"] = payload
                round_record["arms"][arm] = summary
                per_arm = args.output_dir / f"{stem}-summary.json"
                per_arm.write_text(
                    json.dumps(summary, ensure_ascii=False, indent=1) + "\n",
                    encoding="utf-8",
                )
                print(json.dumps({"round": round_index, "arm": arm, **summary},
                                 ensure_ascii=False), flush=True)
            report["runs"].append(round_record)
        if args.pairs:
            pairs = []
            for round_record in report["runs"]:
                first, second = arms
                a = round_record["arms"][first]
                b = round_record["arms"][second]
                pairs.append({
                    "round": round_record["round"],
                    "order": round_record["order"],
                    "p95_delta_ms": a["p95"] - b["p95"],
                    "p95_reduction_pct": (a["p95"] - b["p95"]) / a["p95"] * 100.0,
                    "mean_delta_ms": a["mean"] - b["mean"],
                    "throughput_delta": b["decisions_per_second"] - a["decisions_per_second"],
                    "second_faster": b["p95"] < a["p95"],
                })
            reductions = [pair["p95_reduction_pct"] for pair in pairs]
            report["summary"] = {
                "pairs": len(pairs),
                "second_faster_count": sum(1 for pair in pairs if pair["second_faster"]),
                "median_p95_reduction_pct": statistics.median(reductions),
                "mean_p95_reduction_pct": statistics.mean(reductions),
                "median_p95_delta_ms": statistics.median(
                    pair["p95_delta_ms"] for pair in pairs),
                "median_throughput_delta": statistics.median(
                    pair["throughput_delta"] for pair in pairs),
            }
    finally:
        if original is None:
            args.flag_file.unlink(missing_ok=True)
        else:
            temporary = args.flag_file.with_name(
                f".{args.flag_file.name}.restore-{os.getpid()}"
            )
            temporary.write_bytes(original)
            os.replace(temporary, args.flag_file)
    (args.output_dir / "driver.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )
    if "summary" in report:
        print(json.dumps(report["summary"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
