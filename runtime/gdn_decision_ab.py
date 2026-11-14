#!/usr/bin/env python3
"""Interleaved OFF/ON A/B for the decision-only GDN switches.

Both arms run inside the same service process; only the flag file changes.  The
order alternates every pair (OFF->ON, then ON->OFF, ...) so a service warm-up
drift cannot favour one arm, the mistake that invalidated the round-3 A/B when
each arm needed its own service start.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

FLAG_FILE = Path("/dev/shm/clef-gdn-flags.json")
SETTLE_SECONDS = 2.0


def write_flag(decision_prefill: bool, no_recompute: bool | None = None) -> None:
    values = {"decision_prefill": decision_prefill}
    if no_recompute is not None:
        values["no_recompute"] = no_recompute
    FLAG_FILE.write_text(json.dumps(values), encoding="utf-8")
    time.sleep(SETTLE_SECONDS)


def run_arm(python, benchmark, sample_file, url, limit, report) -> dict:
    command = [
        python, "-u", str(benchmark),
        "--sample-file", str(sample_file),
        "--url", url,
        "--limit", str(limit),
        "--report", str(report),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise SystemExit(f"benchmark failed: {result.stderr[-1500:]}")
    return json.loads(Path(report).read_text(encoding="utf-8"))["summary"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--benchmark", type=Path,
                        default=Path(__file__).with_name("od_full_benchmark.py"))
    parser.add_argument("--sample-file", type=Path, required=True)
    parser.add_argument("--url", default="http://127.0.0.1:18143/v1/systemone")
    parser.add_argument("--limit", type=int, default=600)
    parser.add_argument("--pairs", type=int, default=6)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pairs = []
    for pair in range(args.pairs):
        off_first = pair % 2 == 0
        sequence = ["off", "on"] if off_first else ["on", "off"]
        result = {"pair": pair + 1, "order": sequence, "arms": {}}
        for arm in sequence:
            write_flag(arm == "on")
            report = args.output_dir / f"pair{pair + 1:02d}-{arm}.json"
            summary = run_arm(args.python, args.benchmark, args.sample_file,
                              args.url, args.limit, report)
            result["arms"][arm] = {
                "p50": summary["latency_ms"]["p50"],
                "p95": summary["latency_ms"]["p95"],
                "p99": summary["latency_ms"]["p99"],
                "mean": summary["latency_ms"]["mean"],
                "decisions_per_second": summary["decisions_per_second"],
                "successful": summary["successful"],
                "report": str(report),
            }
        off, on = result["arms"]["off"], result["arms"]["on"]
        result["p95_delta_ms"] = off["p95"] - on["p95"]
        result["p95_reduction_pct"] = (off["p95"] - on["p95"]) / off["p95"] * 100.0
        result["mean_delta_ms"] = off["mean"] - on["mean"]
        result["throughput_delta"] = on["decisions_per_second"] - off["decisions_per_second"]
        result["on_faster"] = on["p95"] < off["p95"]
        pairs.append(result)
        print(json.dumps({
            "pair": result["pair"], "order": sequence,
            "p95_off": round(off["p95"], 3), "p95_on": round(on["p95"], 3),
            "reduction_pct": round(result["p95_reduction_pct"], 3),
            "on_faster": result["on_faster"],
        }), flush=True)

    reductions = [pair["p95_reduction_pct"] for pair in pairs]
    summary = {
        "pairs": len(pairs),
        "on_faster_count": sum(1 for pair in pairs if pair["on_faster"]),
        "median_p95_reduction_pct": statistics.median(reductions),
        "mean_p95_reduction_pct": statistics.mean(reductions),
        "median_p95_delta_ms": statistics.median(pair["p95_delta_ms"] for pair in pairs),
        "median_throughput_delta": statistics.median(
            pair["throughput_delta"] for pair in pairs),
    }
    report = {"created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "limit": args.limit, "url": args.url,
              "sample_file": str(args.sample_file), "summary": summary,
              "pairs": pairs}
    output = args.output_dir / "ab-summary.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                      encoding="utf-8")
    write_flag(False)
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
