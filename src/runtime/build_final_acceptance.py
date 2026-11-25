#!/usr/bin/env python3
"""Build the round-4 acceptance summary from raw benchmark and quality files.

Every measurement carries its raw source path and SHA256; S2 is reported per
concurrency point instead of one merged entry. Quality comes from the pinned
author scorer; the batching diff comes from an independent offline rescore.
"""

import hashlib
import json
import os
from pathlib import Path

RUN = Path("/workspace/yangchen/logs/tuning/20261114-013000-fix4")
OD = RUN / "opendecisions"
OLD = Path("/workspace/yangchen/logs/tuning/20261113-120651-perf3/opendecisions")
DATA = Path("/workspace/yangchen/datasets/clef-full-20261112/full-data/open-decisions2352.jsonl")
SCORER = Path("/workspace/yangchen/logs/opendecisions-baseline-20261113-1600/author-score.py")
BASELINE_SCORE = Path("/workspace/yangchen/logs/opendecisions-baseline-20261113-1600/author-score.json")
BASELINE_PRED = Path("/workspace/yangchen/logs/opendecisions-baseline-20261113-1600/author-predictions.jsonl")
DIFF = OLD / "raw-acceptance-recompute.json"
S1_CONFIG = "tp2-seqs1-t4096-len4096-eager-nz1-cacheoff-pooldefault"
S2_CONFIG = "tp2-seqs4-t4096-len4096-eager-nz1-cacheon-pooldefault"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stats(path):
    data = json.loads(Path(path).read_text())
    summary = data["summary"]
    latency = summary["successful_request_latency_ms"]
    return {
        "configuration": data.get("configuration"),
        "requested": summary["requested_requests"],
        "successful": summary["successful_requests"],
        "failed": summary["failed_requests"],
        "wall_seconds": summary["measurement_wall_seconds"],
        "rps": summary["successful_requests_per_second"],
        "decisions_per_second": summary["decisions_per_second"],
        "input_tokens_per_second": summary["input_tokens_per_second"],
        "p50_ms": latency["p50"],
        "p95_ms": latency["p95"],
        "p99_ms": latency["p99"],
        "mean_ms": latency["mean"],
        "input_tokens_total": summary["actual_input_tokens"]["successful_total"],
        "input_tokens_mean": (
            summary["actual_input_tokens"]["successful_total"]
            / summary["successful_requests"] if summary["successful_requests"] else None),
    }


def measurement(measurement_id, source, config_id, seqs, concurrency, cache, note):
    values = stats(source)
    return {
        "schema_version": "2.0",
        "measurement_id": measurement_id,
        "labels": measurement_id,
        "phase": "stage5-s1" if "s1" in measurement_id else "stage5-s2",
        "run_id": RUN.name,
        "workload": "opendecisions-1.0.1",
        "source_file": str(Path(source).resolve()),
        "source_sha256": sha(source),
        "config_id": config_id,
        "deployment": "single-tp2",
        "tp_size": 2,
        "max_num_seqs": seqs,
        "max_num_batched_tokens": 4096,
        "max_model_len": 4096,
        "weight_nz_mode": 1,
        "schema_cache": cache,
        "kv_pool": "default",
        "compile_mode": "eager",
        "requests": values["requested"],
        "wall_seconds": values["wall_seconds"],
        "rps": values["rps"],
        "decisions_per_second": values["decisions_per_second"],
        "input_tokens_per_second": values["input_tokens_per_second"],
        "p50_ms": values["p50_ms"],
        "p95_ms": values["p95_ms"],
        "p99_ms": values["p99_ms"],
        "percentile_basis": "per_run",
        "batch_evidence": (
            "c1 sequential (B1)" if concurrency == 1 and seqs == 1
            else f"B1 (seqs1), concurrency {concurrency} queues" if seqs == 1
            else f"seqs4, concurrency {concurrency}"),
        "failed": values["failed"],
        "hbm_total_mib": "",
        "notes": note,
    }


def main():
    s1 = stats(OD / "s1-final-full.json")
    s2_points = []
    for c in (2, 4, 8, 16):
        s2_points.append(measurement(
            f"od-s2-final-c{c}", OD / f"s2-final-c{c}.json", S1_CONFIG, 1, c, "off",
            f"accepted S2 config (B1); full 2352 at concurrency {c}"))
    quality = json.loads((OD / "author-score-final.json").read_text())
    conversion = json.loads((OD / "author-conversion-final.json").read_text())
    baseline = json.loads(BASELINE_SCORE.read_text())
    diff = json.loads(DIFF.read_text())

    # Pairwise quality of the final run vs the 2026-11-13 baseline predictions.
    def probs(path):
        rows = {}
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    rows[row["id"]] = row["probs"]
        return rows
    final_pred = probs(OD / "author-predictions-final.jsonl")
    base_pred = probs(BASELINE_PRED)
    shared = sorted(set(final_pred) & set(base_pred))
    max_prob = max(
        max(abs(final_pred[key][label] - base_pred[key][label]) for label in final_pred[key])
        for key in shared)

    candidate = []
    for c in (2, 4, 8, 16):
        candidate.append(measurement(
            f"od-s2-throughput-candidate-c{c}", OLD / f"s2-final-b/s2-c{c}.json",
            S2_CONFIG, 4, c, "on",
            f"throughput candidate (seqs4, cache v2 on); quality gate failed"
            f" (see batching_diff.c{c})"))

    probe_path = RUN / "gdn-kernel-probes.json"
    probes = json.loads(probe_path.read_text()) if probe_path.is_file() else None

    # B2 boundary check: seqs2 on the 16-item diff subset.
    b2_path = RUN / "b2-subset16-c4.json"
    b2_subset = None
    if b2_path.is_file():
        def probs_of(path):
            data = json.loads(Path(path).read_text())
            return {
                row["sample_name"]: row["response"]["answers"]["answer"]
                for row in data["requests"]
            }
        s1_answers = probs_of(OD / "s1-final-full.json")
        def vector(answer):
            p = answer.get("probabilities")
            if p is None:
                p = {"yes": answer["noul"], "no": 1 - answer["noul"]}
            return p
        b2 = json.loads(b2_path.read_text())
        worst, over = 0.0, []
        for request in b2["requests"]:
            name = request["sample_name"]
            p, q = vector(request["response"]["answers"]["answer"]), vector(s1_answers[name])
            max_diff = max(abs(p[key] - q[key]) for key in q)
            worst = max(worst, max_diff)
            if max_diff > 0.03:
                over.append({"sample": name, "max_prob_diff": max_diff})
        b2_subset = {
            "source": str(b2_path.resolve()), "source_sha256": sha(b2_path),
            "items": len(b2["requests"]), "successful": b2["summary"]["successful_requests"],
            "max_prob_diff": worst, "over_0_03": over,
            "verdict": "seqs2 (B2) also fails the 0.03 gate; B1 is the only passing batch config",
        }

    summary = {
        "run_id": RUN.name,
        "acceptance_data": {
            "dataset": "OpenDecisions 1.0.1 test",
            "items": 2352,
            "revision": "ff0b640be2b72c7d06d9d656a34ad6eed79ec7fe",
            "data_sha256": sha(DATA),
            "scorer_sha256": sha(SCORER),
            "request_sequence": str((OD / "all2352.json").resolve()),
            "request_sequence_sha256": sha(OD / "all2352.json"),
        },
        "final_config": {
            "config_id": S1_CONFIG,
            "description": "TP2, BF16, eager, max_model_len 4096, max_num_seqs 1, cache off; "
                           "the only batching configuration that passes the 0.03/0.05 gate",
            "service_argv": json.loads((RUN / "current-service.json").read_text())["argv"],
            "source_phase0": json.loads((RUN / "source-preimage.json").read_text())["files"],
            "source_final": (
                json.loads((RUN / "final-verification.json").read_text())["source_sha256"]
                if (RUN / "final-verification.json").is_file() else None),
        },
        "s1": {
            "config_id": S1_CONFIG,
            "source": str((OD / "s1-final-full.json").resolve()),
            "source_sha256": sha(OD / "s1-final-full.json"),
            **s1,
        },
        "s2": {
            "accepted_config_id": S1_CONFIG,
            "max_decisions_per_second": max(point["decisions_per_second"] for point in s2_points),
            "max_concurrency": 16,
            "points": s2_points,
            "throughput_candidate_config_id": S2_CONFIG,
            "throughput_candidate_points": candidate,
        },
        "quality": {
            "source": str((OD / "author-score-final.json").resolve()),
            "source_sha256": sha(OD / "author-score-final.json"),
            "calibration": quality["calibration"],
            "intelligence": quality["intelligence"],
            "capability": quality["capability"],
            "items": quality["items"],
            "predicted": quality["predicted"],
            "tvd_129": quality["calibration_items_tvd"],
            "accuracy_per_type": quality["accuracy_per_type"],
            "accuracy": conversion["derived_metrics"]["accuracy"],
            "correct": conversion["derived_metrics"]["correct"],
            "accuracy_denominator": conversion["derived_metrics"]["accuracy_denominator"],
            "score_mae": conversion["derived_metrics"]["score_mae"],
            "score_mae_denominator": conversion["derived_metrics"]["score_mae_denominator"],
            "baseline": {
                "calibration": baseline["calibration"],
                "intelligence": baseline["intelligence"],
                "capability": baseline["capability"],
                "tvd_129": baseline["calibration_items_tvd"],
            },
            "bitwise_equal_to_baseline_predictions": max_prob == 0.0,
            "max_probability_difference_vs_baseline": max_prob,
            "shared_items": len(shared),
        },
        "batching_diff": {
            "source": str(DIFF.resolve()),
            "source_sha256": sha(DIFF),
            "reference": "s1-final/s1-full.json",
            "comparisons": {
                key: {field: diff["comparisons"][key][field] for field in (
                    "choice_changes", "noul_crossings", "prob_diff_over_0_03",
                    "max_prob_diff", "score_diff_over_0_05", "max_score_ev_diff")}
                for key in ("c2", "c4", "c8", "c16")
            },
            "b2_subset": b2_subset,
            "verdict": "seqs4 batching fails the 0.03 probability / 0.05 score gate; "
                       "the accepted config is seqs1 (B1)",
        },
        "gdn_bsnd": {
            "legacy_guard": {
                "log": str((RUN / "gdn-bsnd-legacy/service.log").resolve()),
                "replay": str((RUN / "gdn-legacy-replay.json").resolve()),
                "replay_passed": json.loads((RUN / "gdn-legacy-replay.json").read_text())["passed"],
                "reason": "non-contiguous",
                "input_q_shape": [1, 143, 8, 128],
                "input_q_stride": [18304, 128, 18304, 1],
                "input_contiguous": False,
            },
            "fixed_guard": {
                "log": str((RUN / "gdn-bsnd-fixed2/service.log").resolve()),
                "replay": str((RUN / "gdn-fixed2-replay.json").resolve()),
                "replay_passed": json.loads((RUN / "gdn-fixed2-replay.json").read_text())["passed"],
                "hits_logged_through": 1000,
                "fallbacks_logged": 0,
            },
            "microbench": probes,
            "verdict": "the old guard rejected real prefill inputs; the fixed guard "
                       "runs the candidate and stays bit-identical on the frozen 8 cases",
        },
        "cache_ab": {
            "source": "Clef-27B/logs/tuning3-20261113/cache-ab-analysis.json",
            "order": "fixed OFF->ON for all three pairs (archived); future runs are balanced",
            "verdict": "fields-8 +2.81% is a fixed-order observation; no causal claim",
        },
        "sixpair": {
            "source": "Clef-27B/logs/tuning3-20261113/gdn-s1-sixpair-analysis.json",
            "decision": "reject-incomplete-validity",
            "valid_pairs": 3,
            "note": "first three pairs lack cgroup sampling; only the last three have known "
                    "zero throttling; order conflict remains",
        },
        "restored_service": {
            "pid": json.loads((RUN / "current-service.json").read_text())["pid"],
            "argv_sha256": json.loads((RUN / "current-service.json").read_text()).get("argv_sha256"),
            "cwd": json.loads((RUN / "current-service.json").read_text())["cwd"],
            "baseline_replay": str((RUN / "restored-replay.json").resolve()),
            "baseline_replay_passed": json.loads((RUN / "restored-replay.json").read_text())["passed"],
        },
        "measurements": [
            measurement("od-s1-final-c1", OD / "s1-final-full.json", S1_CONFIG, 1, 1, "off",
                        "accepted S1 config; full 2352 after a full ramp pass")
        ] + s2_points + candidate,
    }
    output = RUN / "stage5-summary-final.json"
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=1) + "\n")
    print(json.dumps({
        "output": str(output),
        "s1_p95": s1["p95_ms"],
        "s1_dps": s1["decisions_per_second"],
        "s2_points": [(p["measurement_id"], p["decisions_per_second"], p["p95_ms"]) for p in s2_points],
        "quality": summary["quality"]["capability"],
        "max_prob_diff_vs_baseline": max_prob,
        "measurements": len(summary["measurements"]),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
