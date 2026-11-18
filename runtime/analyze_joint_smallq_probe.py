#!/usr/bin/env python3
"""Recompute paired small-Q statistics from raw samples, without torch/NPU."""
import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
import statistics


def summary(values):
    return {"median": statistics.median(values), "min": min(values), "max": max(values)}


def analyze(data):
    if data.get("schema") != "joint-smallq-probe/1" or data.get("gate_passed") is not True or data.get("error"):
        raise ValueError("probe schema/gate/error check failed")
    result = {"scope": "synthetic inputs, real checkpoint; microbenchmark only",
              "versions": data["versions"], "source": data["source"],
              "sdpa_configuration": data["sdpa_configuration"], "entries": []}
    expected_labels = {("core", length, q) for length in (307, 967, 1595) for q in (1, 4)}
    expected_labels |= {("head", length, 4) for length in (307, 967, 1595)}
    seen = set()
    for entry in data["timing"]:
        label = entry["label"]
        identity = (label["scope"], label["length"], label.get("queries", label.get("options")))
        if identity not in expected_labels or identity in seen:
            raise ValueError(f"unexpected/duplicate timing entry: {identity}")
        seen.add(identity)
        if len(entry["rounds"]) != 4:
            raise ValueError("expected four interleaved rounds")
        row = {"label": label, "raw_samples_per_arm": 80, "rounds": [], "paired": {}}
        for index, (round_, order) in enumerate(zip(entry["rounds"], ([0, 1], [1, 0], [1, 0], [0, 1]))):
            if round_["order"] != order or round_["round"] != index:
                raise ValueError("round index/order mismatch")
            item = {"round": index, "order": order}
            for metric in ("wall_ms", "device_ms"):
                arms = []
                for arm in ("0", "1"):
                    samples = round_["arms"][arm]["samples"]
                    if len(samples) != 20 or [sample["index"] for sample in samples] != list(range(20)):
                        raise ValueError("missing or reordered raw samples")
                    values = [sample[metric] for sample in samples]
                    if any(not math.isfinite(value) or value <= 0 for value in values):
                        raise ValueError("nonfinite/nonpositive timing")
                    arms.append(statistics.median(values))
                item[metric] = {"baseline_median": arms[0], "candidate_median": arms[1],
                                "delta": arms[1] - arms[0],
                                "delta_percent": (arms[1] / arms[0] - 1) * 100}
            row["rounds"].append(item)
        for metric in ("wall_ms", "device_ms"):
            deltas = [round_[metric]["delta"] for round_ in row["rounds"]]
            row["paired"][metric] = {
                "delta": summary(deltas),
                "delta_percent": summary([round_[metric]["delta_percent"] for round_ in row["rounds"]]),
                "faster_rounds": sum(value < 0 for value in deltas),
                "slower_rounds": sum(value > 0 for value in deltas),
                "baseline_round_medians": summary([round_[metric]["baseline_median"] for round_ in row["rounds"]]),
                "candidate_round_medians": summary([round_[metric]["candidate_median"] for round_ in row["rounds"]]),
            }
        result["entries"].append(row)
    if seen != expected_labels:
        raise ValueError("missing timing entries")
    gates = data["gates"]
    if len(gates["core"]) != 6 or len(gates["head"]) != 3 or len(gates["fallback"]) != 2:
        raise ValueError("incomplete gate coverage")
    for entry in gates["core"]:
        if not entry["difference"]["finite"] or entry["difference"]["max_abs"] > 0.0078125:
            raise ValueError("core gate failed")
    for entry in gates["head"]:
        if [call["q_shape"][1] for call in entry["calls"]] != [4, 4, 1, 1, 1, 1]:
            raise ValueError("head did not reach six expected attention calls")
        if any(not call["difference"]["finite"] or call["difference"]["max_abs"] > 0.0078125
               for call in entry["calls"]):
            raise ValueError("projected attention gate failed")
        for key, threshold in (("difference", 0.03125), ("probability_difference", 0.005)):
            if not entry[key]["finite"] or entry[key]["max_abs"] > threshold:
                raise ValueError("head gate failed")
    if any(not entry["difference"]["bit_equal"] or not entry["difference"]["finite"]
           for entry in gates["fallback"]):
        raise ValueError("fallback gate failed")
    result["gate_summary"] = {
        "core_max_abs": max(entry["difference"]["max_abs"] for entry in gates["core"]),
        "head_max_abs": max(entry["difference"]["max_abs"] for entry in gates["head"]),
        "probability_max_abs": max(entry["probability_difference"]["max_abs"] for entry in gates["head"]),
        "fallback_bit_equal": True,
    }
    result["first_calls"] = data["first_calls"]
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("input", type=Path)
    ap.add_argument("--expect-sha", required=True, help="SHA256 of decompressed original JSON bytes")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    raw = args.input.read_bytes()
    if args.input.suffix == ".gz":
        raw = gzip.decompress(raw)
    digest = hashlib.sha256(raw).hexdigest()
    if digest != args.expect_sha:
        raise ValueError(f"source SHA mismatch: {digest}")
    result = analyze(json.loads(raw))
    result["input_sha256"] = digest
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    for row in result["entries"]:
        print(json.dumps({"label": row["label"], "paired": row["paired"]}))


if __name__ == "__main__":
    main()
