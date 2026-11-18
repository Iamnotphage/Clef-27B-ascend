#!/usr/bin/env python3
"""Run the formal ERQA baseline against Clef's native /v1/decide endpoint.

The runner keeps the official record order, builds one multimodal Choice
request per sample, sends it at client concurrency 1, and stores the complete
wire request, raw response, every attempt, timings and token usage.  Scoring
uses the official exact-match rule from ``eval_harness.py``; failures stay in
the fixed selected-sample denominator (400 samples for the full run).

A new output directory is required for a new run.  ``--resume`` reuses an
existing directory: a sample is skipped only when its saved structural request
hash matches the current protocol and its final status is ``ok``.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone

from erqa_protocol import (
    OFFICIAL_TFRECORD_SHA256, build_request, load_samples, parse_options,
    request_summary, score, wire_body,
)

SMOKE_TYPES = [
    "Trajectory Reasoning", "Action Reasoning", "Pointing", "State Estimation",
    "Spatial Reasoning", "Multi-view Reasoning", "Task Reasoning", "Other",
]


def now():
    return datetime.now(timezone.utc).isoformat()


def sha256(blob):
    return hashlib.sha256(blob).hexdigest()


def percentile(values, fraction):
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * fraction
    lower, upper = math.floor(index), math.ceil(index)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)


def distribution(values):
    if not values:
        return {"count": 0}
    return {
        "count": len(values), "min": min(values), "max": max(values),
        "mean": statistics.mean(values), "p50": percentile(values, 0.5),
        "p95": percentile(values, 0.95), "p99": percentile(values, 0.99),
    }


def smoke_selection(samples):
    by_type = defaultdict(list)
    for sample in samples:
        by_type[sample["question_type"]].append(sample)
    chosen = []
    for name in SMOKE_TYPES:
        if by_type[name]:
            chosen.append(by_type[name][0])
    chosen += [s for s in samples if s["image_count"] >= 7][:3]
    chosen += [s for s in samples if s["visual_index_pattern"] == "interleaved"][:4]
    chosen += [s for s in samples if len(parse_options(s["question"])) == 2][:2]
    chosen += [s for s in samples if len(parse_options(s["question"])) == 3][:1]
    chosen += sorted(samples, key=lambda s: len(s["question"]), reverse=True)[:1]
    unique, seen = [], set()
    for sample in chosen:
        if sample["id"] not in seen:
            seen.add(sample["id"])
            unique.append(sample)
    return unique


def service_info(url):
    try:
        with urllib.request.urlopen(f"{url.rstrip('/')}/decide/info", timeout=30) as response:
            return {"status_code": response.status, "body": json.loads(response.read())}
    except Exception as exc:  # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {exc}"}


def post(url, body, timeout):
    request = urllib.request.Request(
        f"{url.rstrip('/')}/decide", data=body,
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.status, response.read()


def run_one(sample, data_dir, sample_dir, args):
    request = build_request(sample, data_dir, model=args.model)
    body = wire_body(request)
    summary = request_summary(request, sample)
    structural = sha256(json.dumps(
        {**summary, "images": summary["image_sha256"]},
        ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode())
    sample_dir.mkdir(parents=True, exist_ok=True)
    (sample_dir / "request.json").write_bytes(body)
    (sample_dir / "request.meta.json").write_text(json.dumps({
        "structural_sha256": structural,
        "wire_sha256": sha256(body),
        "body_bytes": len(body),
        **summary,
    }, ensure_ascii=False, indent=1) + "\n")

    record = {
        "id": sample["id"],
        "record_index": sample["record_index"],
        "question_type": sample["question_type"],
        "gold": sample["answer"],
        "option_count": len(parse_options(sample["question"])),
        "image_count": sample["image_count"],
        "visual_index_pattern": sample["visual_index_pattern"],
        "image_sha256": [img["sha256"] for img in sample["images"]],
        "image_order": summary["image_order"],
        "request_structural_sha256": structural,
        "request_wire_sha256": sha256(body),
        "body_bytes": len(body),
        "attempts": [],
        "status": "failed",
        "prediction": None,
        "correct": False,
    }
    attempts_path = sample_dir / "attempts.jsonl"
    for attempt in range(1, args.retries + 1):
        started_at, started = now(), time.monotonic()
        attempt_record = {"attempt": attempt, "started_at": started_at}
        try:
            status, raw = post(args.api_url, body, args.timeout)
            latency_ms = (time.monotonic() - started) * 1000
            (sample_dir / f"response-attempt-{attempt}.bin").write_bytes(raw)
            (sample_dir / "response.json").write_bytes(raw)
            response = json.loads(raw)
            answer = response["answers"]["answer"]
            attempt_record.update({
                "status_code": status,
                "finished_at": now(),
                "latency_ms": latency_ms,
                "response_sha256": sha256(raw),
                "input_tokens": response.get("usage", {}).get("input_tokens"),
                "output_tokens": response.get("usage", {}).get("output_tokens"),
            })
            record.update({
                "status": "ok",
                "prediction": answer["choice"],
                "probabilities": answer["probabilities"],
                "confidence": answer.get("confidence"),
                "input_tokens": response["usage"]["input_tokens"],
                "output_tokens": response["usage"]["output_tokens"],
                "correct": score(answer["choice"], sample["answer"]),
                "started_at": started_at,
                "finished_at": attempt_record["finished_at"],
                "latency_ms": latency_ms,
            })
        except Exception as exc:  # noqa: BLE001 - every failure is evidence
            attempt_record.update({
                "finished_at": now(),
                "latency_ms": (time.monotonic() - started) * 1000,
                "error_class": type(exc).__name__,
                "error": str(exc)[:2000],
            })
            if isinstance(exc, urllib.error.HTTPError):
                try:
                    raw = exc.read()
                    (sample_dir / f"response-attempt-{attempt}.bin").write_bytes(raw)
                    attempt_record["error_body"] = raw.decode("utf-8", "replace")[:4000]
                except Exception:
                    pass
        record["attempts"].append(attempt_record)
        with attempts_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(attempt_record, ensure_ascii=False) + "\n")
        if record["status"] == "ok":
            break
        if attempt < args.retries:
            time.sleep(args.retry_interval)
    (sample_dir / "record.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=1) + "\n")
    return record


def categorize(records):
    groups = defaultdict(lambda: {"total": 0, "correct": 0, "failed": 0})
    for record in records:
        keys = [
            ("type", record["question_type"]),
            ("images", "single" if record["image_count"] == 1 else "multi"),
            ("options", str(record["option_count"])),
            ("visual", record["visual_index_pattern"]),
        ]
        for kind, key in keys:
            bucket = groups[f"{kind}:{key}"]
            bucket["total"] += 1
            bucket["correct"] += int(record["correct"])
            bucket["failed"] += int(record["status"] != "ok")
    return {
        key: {**value, "accuracy": value["correct"] / value["total"] if value["total"] else None}
        for key, value in sorted(groups.items())
    }


def summarize(records, elapsed, args, service, manifest):
    ok = [r for r in records if r["status"] == "ok"]
    correct = sum(int(r["correct"]) for r in records)
    errors = Counter()
    retries = 0
    for record in records:
        for attempt in record["attempts"]:
            if "error_class" in attempt:
                errors[attempt["error_class"]] += 1
        retries += max(0, len(record["attempts"]) - 1)
    latencies = [r["latency_ms"] for r in ok]
    tokens = [r["input_tokens"] for r in ok]
    return {
        "run": {
            "scope": args.scope, "api_url": args.api_url, "model": args.model,
            "retries": args.retries, "timeout": args.timeout,
            "started_at": args.started_at, "finished_at": now(),
            "wall_seconds": elapsed,
        },
        "data": {
            "tfrecord_sha256": manifest["source"]["sha256"],
            "records": manifest["records"], "images": manifest["images"]["total"],
            "question_types": manifest["question_types"],
        },
        "service": service,
        "coverage": {
            "fixed_total": len(records), "ok": len(ok), "failed": len(records) - len(ok),
            "success_rate": len(ok) / len(records) if records else None,
            "retry_count": retries, "error_classes": dict(errors),
        },
        "accuracy": {
            "correct": correct, "total": len(records),
            "value": correct / len(records) if records else None,
            "denominator": f"fixed {len(records)}-sample set; failures count as incorrect",
            "ok_only_correct": correct, "ok_only_total": len(ok),
            "ok_only_value": correct / len(ok) if ok else None,
        },
        "breakdown": categorize(records),
        "latency_ms": distribution(latencies),
        "input_tokens": distribution(tokens),
        "images": distribution([r["image_count"] for r in records]),
        "throughput": {
            "decisions_per_second": len(ok) / elapsed if elapsed else None,
            "basis": "successful decisions / wall clock, client concurrency 1",
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--api-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", default="clef")
    parser.add_argument("--scope", choices=("smoke", "full"), default="full")
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--retry-interval", type=float, default=5.0)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    args.started_at = now()
    samples, manifest = load_samples(args.data_dir)
    if manifest["source"]["sha256"] != OFFICIAL_TFRECORD_SHA256:
        parser.error("Unexpected ERQA TFRecord SHA256")
    selected = smoke_selection(samples) if args.scope == "smoke" else samples
    if args.limit:
        selected = selected[: args.limit]

    output = args.output_dir.expanduser().resolve()
    if output.exists() and any(output.iterdir()) and not args.resume:
        parser.error(f"输出目录非空；新 run 使用全新目录，或加 --resume：{output}")
    output.mkdir(parents=True, exist_ok=True)
    (output / "samples").mkdir(exist_ok=True)
    service = service_info(args.api_url)
    if service.get("error"):
        parser.error(f"服务不可用：{service['error']}")

    records, skipped = [], 0
    started = time.monotonic()
    for index, sample in enumerate(selected, start=1):
        sample_dir = output / "samples" / sample["id"]
        record_path = sample_dir / "record.json"
        if args.resume and record_path.is_file():
            saved = json.loads(record_path.read_text())
            meta_path = sample_dir / "request.meta.json"
            meta = json.loads(meta_path.read_text()) if meta_path.is_file() else {}
            if saved.get("status") == "ok" and meta.get("structural_sha256") == saved.get(
                    "request_structural_sha256"):
                records.append(saved)
                skipped += 1
                continue
        record = run_one(sample, args.data_dir, sample_dir, args)
        records.append(record)
        print(json.dumps({
            "progress": f"{index}/{len(selected)}", "id": record["id"],
            "status": record["status"], "prediction": record["prediction"],
            "gold": record["gold"], "correct": record["correct"],
            "latency_ms": round(record.get("latency_ms", 0), 1),
        }, ensure_ascii=False), flush=True)
    elapsed = time.monotonic() - started

    summary = summarize(records, elapsed, args, service, manifest)
    summary["run"]["skipped_resumed"] = skipped
    (output / "predictions.jsonl").write_text("".join(
        json.dumps({key: record.get(key) for key in (
            "id", "record_index", "question_type", "gold", "prediction",
            "probabilities", "confidence", "correct", "status", "image_count",
            "option_count", "visual_index_pattern", "image_sha256", "image_order",
            "request_structural_sha256", "request_wire_sha256", "body_bytes",
            "input_tokens", "output_tokens", "latency_ms",
            "started_at", "finished_at")}, ensure_ascii=False) + "\n"
        for record in records), encoding="utf-8")
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=1) + "\n")
    print(json.dumps({
        "summary": str(output / "summary.json"),
        "accuracy": summary["accuracy"], "coverage": summary["coverage"],
        "latency_ms": summary["latency_ms"],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
