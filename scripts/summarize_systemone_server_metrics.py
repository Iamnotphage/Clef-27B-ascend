#!/usr/bin/env python3
"""Compare archived native server metric deltas with saved client benchmarks."""

import argparse
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import re


METRICS = ("request_queue_time_seconds", "request_inference_time_seconds",
           "request_prefill_time_seconds", "e2e_request_latency_seconds")
LINE = re.compile(r"^([^\s{]+)(?:\{([^}]*)\})?\s+([^\s]+)")
LABEL = re.compile(r'(\w+)="((?:\\.|[^"\\])*)"')


def evidence(path):
    data = path.read_bytes()
    return {"path": str(path), "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}


def parse_metrics(path):
    records = {}
    selected = []
    for number, line in enumerate(path.read_text().splitlines(), 1):
        if line.startswith("#"):
            continue
        match = LINE.match(line)
        if not match:
            continue
        name, labels, value = match.groups()
        for metric in METRICS:
            for suffix in ("sum", "count", "created"):
                if name != f"vllm:{metric}_{suffix}":
                    continue
                label_map = {key: json.loads('"' + value + '"') for key, value in LABEL.findall(labels or "")}
                key = (metric, tuple(sorted(label_map.items())))
                if suffix in records.setdefault(key, {}):
                    raise ValueError(f"Duplicate metric series: {path}:{number}")
                records[key][suffix] = Decimal(value)
                selected.append({"line": number, "text": line})
    return records, {**evidence(path), "selected_original_lines": selected}


def summarize_configuration(run, configuration):
    directory = run / configuration
    rounds = []
    sources = []
    created_values = {}
    aggregate = {}
    client_totals = {"formal_requests": 0, "http_ms_sum": 0.0, "elapsed_ms_sum": 0.0,
                     "json_decode_ms_sum": 0.0, "protocol_validation_ms_sum": 0.0}
    for index in (1, 2, 3):
        before, before_source = parse_metrics(directory / f"metrics-before-{index}.txt")
        after, after_source = parse_metrics(directory / f"metrics-after-{index}.txt")
        sources.extend((before_source, after_source))
        benchmark_path = directory / f"round-{index}.json"
        benchmark = json.loads(benchmark_path.read_text())
        summary = benchmark["summary"]
        formal = summary["successful_requests"]
        expected = benchmark["configuration"]["warmup"] + benchmark["configuration"]["requests"]
        row = {"round": index, "expected_count_delta": expected,
               "benchmark": evidence(benchmark_path), "client_formal_requests": formal,
               "client_benchmark_passed": benchmark.get("passed") is True,
               "client_http_formal_mean_ms": summary["successful_stage_ms"]["http_ms"]["mean"],
               "same_label_series_before_after": set(before) == set(after), "metrics": [],
               "passed": benchmark.get("passed") is True and expected == 310 and formal == 300}
        if set(before) != set(after):
            row["passed"] = False
        for key in sorted(set(before) & set(after)):
            metric, label_pairs = key
            first, last = before[key], after[key]
            complete = all(part in first and part in last for part in ("sum", "count", "created"))
            if not complete:
                row["passed"] = False
                row["metrics"].append({"metric": metric, "error": "Missing sum/count/created series"})
                continue
            delta_count = last["count"] - first["count"]
            delta_sum = last["sum"] - first["sum"]
            unchanged = first["created"] == last["created"]
            passed = unchanged and delta_count == expected and delta_sum >= 0
            row["passed"] = row["passed"] and passed
            row["metrics"].append({"metric": metric, "labels": dict(label_pairs),
                "count_delta": int(delta_count), "sum_delta_seconds": float(delta_sum),
                "sum_delta_seconds_exact_decimal_from_export": str(delta_sum),
                "mean_ms_including_warmup": float(delta_sum * 1000 / delta_count) if delta_count else None,
                "created_before": str(first["created"]), "created_after": str(last["created"]),
                "same_created": unchanged, "passed": passed})
            created_values.setdefault(key, set()).update((first["created"], last["created"]))
            total = aggregate.setdefault(key, {"count": Decimal(0), "sum": Decimal(0)})
            total["count"] += delta_count
            total["sum"] += delta_sum
        if len(row["metrics"]) != len(METRICS):
            row["passed"] = False
        client_totals["formal_requests"] += formal
        for source_name, destination in (("http_ms", "http_ms_sum"), ("json_decode_ms", "json_decode_ms_sum"),
                                          ("protocol_validation_ms", "protocol_validation_ms_sum")):
            client_totals[destination] += summary["successful_stage_ms"][source_name]["mean"] * formal
        client_totals["elapsed_ms_sum"] += summary["successful_request_latency_ms"]["mean"] * formal
        rounds.append(row)
    native = [{"metric": metric, "labels": dict(labels), "count": int(total["count"]),
               "sum_seconds": float(total["sum"]),
               "mean_ms_including_warmup": float(total["sum"] * 1000 / total["count"])}
              for (metric, labels), total in aggregate.items() if total["count"]]
    same_created = all(len(values) == 1 for values in created_values.values())
    formal = client_totals["formal_requests"]
    client = {"formal_request_count": formal, "http_mean_ms": client_totals["http_ms_sum"] / formal,
              "http_decode_validation_elapsed_mean_ms": client_totals["elapsed_ms_sum"] / formal,
              "json_decode_mean_ms": client_totals["json_decode_ms_sum"] / formal,
              "protocol_validation_mean_ms": client_totals["protocol_validation_ms_sum"] / formal}
    means = {item["metric"]: item["mean_ms_including_warmup"] for item in native}
    remainder = (means["e2e_request_latency_seconds"] - means["request_queue_time_seconds"]
                 - means["request_inference_time_seconds"])
    return {"configuration": configuration, "passed": all(row["passed"] for row in rounds) and same_created,
        "same_series_created_across_all_rounds": same_created,
        "rounds": rounds, "native_metrics_aggregate": native, "client_benchmark_aggregate": client,
        "native_e2e_minus_queue_minus_running_mean_ms_same_930_requests": remainder,
        "client_http_900_minus_native_e2e_930_ms_descriptive_only": client["http_mean_ms"] - means["e2e_request_latency_seconds"],
        "source_files": sources}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", default="-")
    args = parser.parse_args()
    report = {"created_at": datetime.now(timezone.utc).isoformat(), "run_dir": str(args.run_dir.resolve()),
        "scope": "Archived profiler-disabled stable service histograms; each delta includes 10 warmup + 300 formal requests. Three rounds provide n=930 server samples and n=900 formal client samples.",
        "definitions": {
            "queue": "Native engine QUEUED to first SCHEDULED event interval",
            "running_inference": "Native first SCHEDULED to last engine output event; includes model/decision-head execution and required waits, rather than a sum of kernels",
            "prefill": "Native first SCHEDULED to first output event; pooling performs a single output step",
            "e2e": "Native request arrival to finished request statistics; API work before EngineClient.encode is outside this scope",
            "native_remainder": "Same-population native e2e minus queue minus running contains frontend/IPC/postprocessing timing gaps",
            "client_http": "Benchmark urlopen through complete response-body read; JSON decode and protocol validation are separate",
            "comparison": "Client 900 and server 930 populations differ. Their mean difference mixes timing scope and warmup effects; matched-request telemetry is required for a frontend-only latency estimate."},
        "configurations": [summarize_configuration(args.run_dir.resolve(), config)
                           for config in ("tp1-1024-stable", "tp2-1024-stable")]}
    report["passed"] = all(row["passed"] for row in report["configurations"])
    definitions = Path("/workspace/yangchen/work/src/vllm/vllm/v1/metrics/stats.py")
    if definitions.is_file():
        report["native_definition_source"] = {**evidence(definitions), "lines": [
            {"line": number, "text": line} for number, line in enumerate(definitions.read_text().splitlines(), 1)
            if any(token in line for token in ("e2e_latency = self._time_since(req_stats.arrival_time)",
                "queued_time = req_stats.scheduled_ts", "prefill_time = req_stats.first_token_ts",
                "inference_time = req_stats.last_token_ts"))]}
    payload = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if args.output == "-":
        print(payload, end="")
    else:
        Path(args.output).write_text(payload)
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
