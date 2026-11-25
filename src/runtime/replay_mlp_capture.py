#!/usr/bin/env python3
"""Capture MLP inputs from frozen text and multimodal decision requests."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
import urllib.request


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--url", default="http://127.0.0.1:18143/v1/decide")
    parser.add_argument("--od-samples", type=Path, default=Path(
        "/workspace/yangchen/logs/tuning/20261114-013000-fix4/opendecisions/all2352.json"))
    parser.add_argument("--od-baseline", type=Path, default=Path(
        "/workspace/yangchen/logs/gdn-decision-20261115/entry-s1/od-full.json"))
    parser.add_argument("--erqa-samples", type=Path, default=Path(
        "/workspace/yangchen/logs/erqa-baseline-20261114/full/samples"))
    parser.add_argument("--layers", nargs="+", type=int, default=[0, 31, 63])
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    args.capture_dir.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    arm = args.capture_dir / "arm.json"
    if arm.exists():
        raise SystemExit(f"An existing capture is armed at {arm}")

    samples = {row["name"]: row for row in read_json(args.od_samples)["samples"]}
    baseline = read_json(args.od_baseline)["requests"]
    short = min(baseline, key=lambda row: abs(row["input_tokens"] - 307))
    long = max(baseline, key=lambda row: row["input_tokens"])
    cases = []
    for row in (short, long):
        cases.append({
            "name": row["name"], "request": samples[row["name"]]["request"],
            "reference": row["response"], "source": str(args.od_samples),
        })
    for name in ("erqa-0089", "erqa-0167"):
        directory = args.erqa_samples / name
        cases.append({
            "name": name, "request": read_json(directory / "request.json"),
            "reference": read_json(directory / "response.json"),
            "source": str(directory / "request.json"),
        })

    report = {"url": args.url, "layers": args.layers, "cases": [], "completed": False}
    try:
        for case in cases:
            expected_tokens = case["reference"]["usage"]["input_tokens"]
            digest = hashlib.sha256(case["name"].encode()).hexdigest()[:16]
            pattern = f"{case['name']}-{digest}.layer*.rank*.pt"
            if any(args.capture_dir.glob(pattern)):
                raise RuntimeError(
                    f"Existing captures for {case['name']} require a fresh capture directory")
            write_json(arm, {
                "request_name": case["name"], "token_count": expected_tokens,
                "layers": args.layers,
            })
            body = json.dumps(case["request"], ensure_ascii=False).encode("utf-8")
            request = urllib.request.Request(
                args.url, data=body, headers={"Content-Type": "application/json"})
            started = time.perf_counter()
            with urllib.request.urlopen(request, timeout=300) as response:
                actual = json.loads(response.read())
            arm.unlink()
            record = {
                "name": case["name"], "source": case["source"],
                "request_sha256": hashlib.sha256(body).hexdigest(),
                "input_tokens": actual["usage"]["input_tokens"],
                "expected_input_tokens": expected_tokens,
                "answers_equal": actual["answers"] == case["reference"]["answers"],
                "input_tokens_equal": actual["usage"]["input_tokens"] == expected_tokens,
                "output_tokens_equal": (
                    actual["usage"].get("output_tokens")
                    == case["reference"]["usage"].get("output_tokens")
                ),
                "diagnostics_equal": actual.get("diagnostics") == case["reference"].get("diagnostics"),
                "joint_equal": actual.get("joint") == case["reference"].get("joint"),
                "response": actual,
                "capture_wall_seconds": time.perf_counter() - started,
                "tensors": [],
            }
            import torch

            paths = sorted(args.capture_dir.glob(pattern))
            for path in paths:
                saved = torch.load(path, map_location="cpu", weights_only=True)
                metadata = saved["metadata"]
                tensor = saved["input"]
                if not isinstance(tensor, torch.Tensor) or tensor.ndim != 2:
                    raise RuntimeError(f"Capture must contain a 2D tensor in {path}")
                checks = {
                    "request_name_equal": metadata.get("request_name") == case["name"],
                    "shape_equal": metadata.get("shape") == list(tensor.shape),
                    "dtype_equal": metadata.get("dtype") == str(tensor.dtype),
                    "bf16": tensor.dtype == torch.bfloat16,
                    "hidden_size_5120": tensor.shape[1] == 5120,
                    "nonempty": tensor.shape[0] > 0,
                    "finite": bool(torch.isfinite(tensor).all()),
                    "metadata_input_tokens_equal": metadata.get("input_token_count") == tensor.shape[0],
                    "metadata_request_tokens_equal": metadata.get("token_count") == expected_tokens,
                }
                record["tensors"].append({
                    "path": str(path), "bytes": path.stat().st_size,
                    "metadata": metadata,
                    "shape": list(tensor.shape),
                    "stride": list(tensor.stride()),
                    "dtype": str(tensor.dtype),
                    "input_token_count": tensor.shape[0],
                    "input_tokens_equal_response": tensor.shape[0] == actual["usage"]["input_tokens"],
                    "checks": checks,
                    "valid": all(checks.values()),
                    "input_sha256": hashlib.sha256(
                        memoryview(tensor.contiguous().view(torch.uint8).numpy())
                    ).hexdigest(),
                })
                del saved, tensor
            observed = {(item["metadata"]["layer"], item["metadata"]["rank"])
                        for item in record["tensors"]}
            expected = {(layer, rank) for layer in args.layers for rank in (0, 1)}
            record["capture_complete"] = observed == expected and len(record["tensors"]) == len(expected)
            record["all_tensors_valid"] = all(item["valid"] for item in record["tensors"])
            record["passed"] = all(record[key] for key in (
                "answers_equal", "input_tokens_equal", "output_tokens_equal",
                "diagnostics_equal", "joint_equal", "capture_complete", "all_tensors_valid",
            ))
            report["cases"].append(record)
            write_json(args.report, report)
            print(json.dumps({key: record[key] for key in (
                "name", "input_tokens", "answers_equal", "capture_complete", "passed")}), flush=True)
            if not record["passed"]:
                raise RuntimeError(f"Capture validation failed for {case['name']}; see recorded checks")
        report["completed"] = True
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        arm.unlink(missing_ok=True)
    report["all_answers_equal"] = len(report["cases"]) == len(cases) and all(
        case["answers_equal"] for case in report["cases"])
    report["passed"] = report["completed"] and all(case["passed"] for case in report["cases"])
    write_json(args.report, report)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
