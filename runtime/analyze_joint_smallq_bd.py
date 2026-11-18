#!/usr/bin/env python3
"""Validate and recompute B/D raw samples using only the standard library."""
import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
import statistics

ORDERS = (("B", "D"), ("D", "B"), ("D", "B"), ("B", "D"))
METRICS = ("wall_ms", "device_ms", "host_submit_ms", "sync_wait_ms", "event_record_host_ms")
PAIRS = (("D", "B"),)
LIMITS = {"core_atol": 0.0078125, "head_atol": 0.03125, "prob_atol": 0.005}


def summary(values):
    return {"median": statistics.median(values), "min": min(values), "max": max(values)}


def identity(label):
    return (label["scope"], label["length"], label.get("queries", label.get("options")), label.get("questions", 1))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def difference(diff, threshold):
    require(diff["finite"] is True and math.isfinite(diff["max_abs"])
            and 0 <= diff["max_abs"] <= threshold, "numerical gate failed")


def analyze(data, inputs):
    require(data.get("schema") == "joint-smallq-probe/1" and data.get("experiment") == "joint-smallq-bd/1",
            "expected independent B/D experiment")
    require(data.get("gate_passed") is True and not data.get("error"), "probe gate/error failed")
    require(data.get("same_weight_storage_verified") is True and data.get("frozen_files_unchanged") is True,
            "shared storage or frozen files failed")
    require(data.get("single_variable_verified") is True, "single variable check failed")
    require(data["source"] == {
        "reference": "f8101b98228bbd472a63a1efe6827583e35c6cc9dede4379b925f2b016670863",
        "head": "086357bddbe4fbfbda1630e5ac70503c44f06113d582089c5c54669720d3260a",
        "frozen_B": "cf02419978ce62f18ca908b8d4c9646531fb00d3b07871b310576f963e5ec612",
        "kernel": "08ad88f6cebfb148aff8d973097d5bcf234e25d144a824385b18b8b078e82dff",
        "weights": "a010ac04f078e699988e4049cbea5e62c962393f59fec366640b64e8d69a4953",
        "probe": "94ca8428cd44b8130bdb1915b93069a7f687e2ccc957f8218ec9c10fc59030b7",
        "input_manifest": "189fd36285b762273c0ef7d1dd99cb2a43a7db50f68b5ff7ab0036b320b20429",
    }, "frozen source/checkpoint hashes differ")
    difference(data["default_off"], 0)
    require(data["default_off"]["bit_equal"] is True, "default off differs from f810")
    require(data["thresholds"] == LIMITS and data["inference_mode"] is True, "frozen gates/inference mismatch")
    require(data["seed"] == 20261119 and data["warmup_per_block"] == 5 and data["iters_per_arm_round"] == 20,
            "seed/warmup/count mismatch")
    expected = {("core", length, q, 1) for length in (307, 967, 1595) for q in (1, 4)}
    expected |= {("head", length, 4, 1) for length in (307, 967, 1595)}
    expected_gates = expected | {("fallback", 307, 8, 2), ("fallback", 2049, 4, 1)}
    gates, seen = [], set()
    for row in data["gates"]:
        ident = identity(row["label"])
        require(ident in expected_gates and ident not in seen, "unexpected/duplicate gate")
        seen.add(ident)
        scope, length, queries, _ = ident
        if scope != "fallback":
            require(row["input_hashes"] == inputs["inputs"][f"{scope}/{length}/{queries}"], "input hash mismatch")
        require(set(row["arms"]) == set("ABD"), "missing gate arm")
        for arm, value in row["arms"].items():
            difference(value["difference_from_A"], LIMITS["core_atol" if scope == "core" else "head_atol"])
            if arm == "D":
                difference(value["difference_from_B"], 0)
                require(value["difference_from_B"]["bit_equal"] is True, "D differs from B bitwise")
                require(value["difference_from_A"]["candidate_sha256"] ==
                        row["arms"]["B"]["difference_from_A"]["candidate_sha256"], "B/D output hashes differ")
            if scope != "core":
                difference(value["probability_difference_from_A"], LIMITS["prob_atol"])
            calls, mha = value["backend_calls"], value["mha_calls"]
            if scope == "fallback":
                require(not calls and value["difference_from_A"]["bit_equal"], "fallback path differs")
            elif scope == "head" and arm == "A":
                require(not calls and [c["q_shape"][1] for c in mha] == [4, 4, 1, 1, 1, 1]
                        and [c["kv_same_object"] for c in mha] == [False, False, True, True, True, True], "A path differs")
            elif scope == "head":
                require(not mha and [c["q_shape"][1] for c in calls] == [4, 4, 1, 1, 1, 1], "B/D path differs")
                for call in calls:
                    require(call["dtype"] == "torch.bfloat16", "backend dtype differs")
                    difference(call["difference"], LIMITS["core_atol"])
        gates.append({"label": row["label"], "arms": {
            arm: {key: value[key] for key in ("difference_from_A", "difference_from_B", "probability_difference_from_A") if key in value}
            for arm, value in row["arms"].items()}})
    require(seen == expected_gates, "missing gates")
    result = {"experiment": data["experiment"], "scope": data["quality_scope"], "source": data["source"],
              "versions": data["versions"], "sdpa_configuration": data["sdpa_configuration"],
              "timing_notes": data["timing_notes"], "gates": gates, "entries": [], "raw_count": 0,
              "first_calls": data["first_calls"], "default_off": data["default_off"],
              "single_variable_verified": True, "D_vs_B_all_gates_bitwise": True}
    seen = set()
    for entry in data["timing"]:
        ident = identity(entry["label"])
        require(ident in expected and ident not in seen, "unexpected/duplicate timing entry")
        seen.add(ident)
        require(len(entry["rounds"]) == 4, "missing rounds")
        row = {"label": entry["label"], "rounds": [], "paired": {}}
        for index, (block, order) in enumerate(zip(entry["rounds"], ORDERS)):
            require(block["round"] == index and block["order"] == list(order), "round/order mismatch")
            require(set(block["arms"]) == set("BD"), "missing timing arm")
            medians = {}
            for arm in "BD":
                samples = block["arms"][arm]["samples"]
                require(len(samples) == 20 and [s["index"] for s in samples] == list(range(20)), "sample count/order mismatch")
                result["raw_count"] += len(samples)
                medians[arm] = {}
                for metric in METRICS:
                    values = [sample[metric] for sample in samples]
                    require(all(math.isfinite(v) and v > 0 for v in values), "invalid raw timing")
                    medians[arm][metric] = statistics.median(values)
                    require(medians[arm][metric] == block["arms"][arm][metric]["median"], "stored median mismatch")
                for sample in samples:
                    require(abs(sample["wall_ms"] - sum(sample[m] for m in
                            ("host_submit_ms", "sync_wait_ms", "event_record_host_ms"))) < 1e-9, "wall partition mismatch")
            row["rounds"].append({"round": index, "order": list(order), "medians": medians})
        row["arm_round_medians"] = {
            arm: {metric: summary([r["medians"][arm][metric] for r in row["rounds"]]) for metric in METRICS}
            for arm in "BD"}
        for right, left in PAIRS:
            pair = right + "_minus_" + left
            row["paired"][pair] = {}
            for metric in METRICS:
                deltas = [r["medians"][right][metric] - r["medians"][left][metric] for r in row["rounds"]]
                if metric != "event_record_host_ms":
                    require(deltas == entry["paired_deltas"][pair][metric], "stored pair mismatch")
                row["paired"][pair][metric] = {
                    "deltas": deltas, "delta": summary(deltas),
                    "delta_percent": summary([100 * (r["medians"][right][metric] / r["medians"][left][metric] - 1)
                                              for r in row["rounds"]]),
                    "faster_rounds": sum(v < 0 for v in deltas), "slower_rounds": sum(v > 0 for v in deltas)}
        result["entries"].append(row)
    require(seen == expected and result["raw_count"] == 1440, "incomplete timing coverage")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--expect-sha", required=True)
    parser.add_argument("--inputs", type=Path, default=Path(__file__).parent / "joint_smallq_acb/inputs.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw = args.input.read_bytes()
    if args.input.suffix == ".gz":
        raw = gzip.decompress(raw)
    digest = hashlib.sha256(raw).hexdigest()
    require(digest == args.expect_sha, "original JSON SHA mismatch: " + digest)
    data = json.loads(raw)
    require(hashlib.sha256(args.inputs.read_bytes()).hexdigest() == data["source"]["input_manifest"],
            "input manifest file hash differs")
    result = analyze(data, json.loads(args.inputs.read_text()))
    result["input_sha256"] = digest
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"raw_count": result["raw_count"], "entries": len(result["entries"]), "output": str(args.output)}))


if __name__ == "__main__":
    main()
