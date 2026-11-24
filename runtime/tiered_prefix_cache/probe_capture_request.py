#!/usr/bin/env python3
"""Send one frozen request with the Stage 0.2 capture arm armed.

The capture hooks in ``probe_capture/sitecustomize.py`` only act while
``<probe-dir>/arm.json`` exists, so this helper writes the arm, posts the exact
frozen request, removes the arm, and records the response.  OD requests use
the frozen ``all2352.json`` payload; ERQA requests are rebuilt in the
``state-list`` / ``/v1/systemone`` form.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
import urllib.request


def post(url: str, payload: dict, timeout: float) -> dict:
    body = json.dumps(
        payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    request = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def od_request(sample_file: Path, name: str) -> tuple[dict, str]:
    document = json.loads(sample_file.read_text(encoding="utf-8"))
    samples = document["samples"] if isinstance(document, dict) else document
    for sample in samples:
        if sample.get("name") == name:
            return sample["request"], name
    raise SystemExit(f"OD sample not found: {name}")


def erqa_request(data_dir: Path, name: str) -> tuple[dict, str]:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"
                            / "official_eval"))
    from erqa_protocol import build_request, load_samples  # noqa: PLC0415

    samples, _ = load_samples(data_dir)
    for sample in samples:
        if sample.get("id") == name:
            return build_request(
                sample, data_dir, wire_format="state-list"), name
    raise SystemExit(f"ERQA sample not found: {name}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("od", "erqa"), required=True)
    parser.add_argument("--od-sample-file", type=Path, default=Path(
        "/workspace/yangchen/work/aa3e-conv-qkv-formal-20261123.Q3ozHxgd/"
        "all2352.json"))
    parser.add_argument("--erqa-data-dir", type=Path, default=Path(
        "/workspace/yangchen/work/aa3e-conv-qkv-formal-20261123.Q3ozHxgd/"
        "erqa-fixed"))
    parser.add_argument("--name", required=True)
    parser.add_argument("--probe-dir", type=Path, required=True)
    parser.add_argument("--arm-id", required=True)
    parser.add_argument("--token-count", type=int, required=True)
    parser.add_argument("--split", type=int, required=True)
    parser.add_argument("--model-layers", default="0,8,16,24,32,40,48,56,62")
    parser.add_argument("--url", default="http://127.0.0.1:18143/v1/systemone")
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.dataset == "od":
        request, name = od_request(args.od_sample_file, args.name)
    else:
        request, name = erqa_request(args.erqa_data_dir, args.name)
    arm = {
        "arm_id": args.arm_id,
        "request_name": name,
        "token_count": args.token_count,
        "split": args.split,
        "model_layers": [int(value) for value in args.model_layers.split(",")
                         if value.strip()],
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    args.probe_dir.mkdir(parents=True, exist_ok=True)
    arm_path = args.probe_dir / "arm.json"
    arm_path.write_text(json.dumps(arm, ensure_ascii=False, indent=1) + "\n")
    try:
        started = time.perf_counter()
        response = post(args.url, request, args.timeout)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
    finally:
        arm_path.unlink(missing_ok=True)
    record = {
        "arm": arm,
        "url": args.url,
        "name": name,
        "input_tokens": response.get("usage", {}).get("input_tokens"),
        "token_count_expected": args.token_count,
        "elapsed_ms": elapsed_ms,
        "response": response,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, ensure_ascii=False) + "\n")
    print(json.dumps({
        "name": name,
        "input_tokens": record["input_tokens"],
        "expected": args.token_count,
        "elapsed_ms": elapsed_ms,
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
