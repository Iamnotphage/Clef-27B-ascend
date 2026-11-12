#!/usr/bin/env python3
"""Measure real CPU protocol/schema encoding with observations disabled.

Uses a fresh process, OMP=1 and the pinned ModelConfig/tokenizer/strategy, with
no model, engine, HTTP or NPU operation. A temporary _tokens wrapper caches only
schema fragments; every selected rendered state always bypasses that cache.
All timed outputs must exactly match independently saved IDs/schema/order.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import statistics
import sys
import time
import traceback


DEFAULT_NAMES = ["fields-1", "fields-8", "candidates-2", "candidates-16", "controlled-length-256", "controlled-length-4032"]
TIMINGS = ("protocol_parse_ms", "questions_parse_ms", "parse_ms", "render_ms", "encode_ms", "total_ms")


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def json_digest(value):
    return digest(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def distribution(values):
    ordered = sorted(values)

    def quantile(value):
        if not ordered:
            return None
        position = (len(ordered) - 1) * value
        low, high = math.floor(position), math.ceil(position)
        return ordered[low] + (ordered[high] - ordered[low]) * (position - low)

    return {"sample_count": len(values), "mean": statistics.mean(values) if values else None, "p50": quantile(.5), "p95": quantile(.95), "min": min(values) if values else None, "max": max(values) if values else None, "sample_standard_deviation": statistics.stdev(values) if len(values) > 1 else None, "percentile_method": "linear interpolation at (n-1)*quantile"}


def clean_environment():
    if any(name == "torch" or name.startswith("vllm") for name in sys.modules):
        raise RuntimeError("Use a new Python process; torch/vLLM were already imported")
    removed = sorted(key for key in os.environ if key.startswith("CLEF_PROFILE_"))
    for key in removed:
        os.environ.pop(key)
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[key] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    return {"removed_observer_variable_names": removed, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "TOKENIZERS_PARALLELISM": "false", "observer_variables_remaining": sorted(key for key in os.environ if key.startswith("CLEF_PROFILE_"))}


def normalized_encoding(input_ids, schema, orders):
    return {"input_ids": [int(value) for value in input_ids], "joint_schema": schema, "option_output_orders": orders}


def check_encoding(actual, expected):
    for key in ("input_ids", "joint_schema", "option_output_orders"):
        if actual[key] != expected[key]:
            raise ValueError(f"Exact saved encoding mismatch: {key}")


class SchemaTokenCache:
    """A process-local ablation; immutable entries and fresh lists on return."""
    def __init__(self, original, rendered_states):
        self.original = original
        self.excluded_states = frozenset(rendered_states)
        self.cache = {}
        self.hits = 0
        self.misses = 0
        self.state_bypasses = 0

    def __call__(self, text):
        if text in self.excluded_states:
            self.state_bypasses += 1
            return self.original(text)
        if text in self.cache:
            self.hits += 1
            return list(self.cache[text])
        result = self.original(text)
        self.cache[text] = tuple(result)
        self.misses += 1
        return result

    def summary(self):
        return {"entries": len(self.cache), "hits": self.hits, "misses": self.misses, "state_bypass_calls": self.state_bypasses, "rendered_state_count_excluded": len(self.excluded_states), "state_cache_entries": sum(text in self.excluded_states for text in self.cache), "immutable_cache_entries": True, "fresh_lists_on_cache_hit": True}


def load_inputs(workloads_dir, fixed_report_path, names):
    manifest_path = workloads_dir / "manifest.json"
    manifest_raw = manifest_path.read_bytes()
    manifest = json.loads(manifest_raw)
    evidence_path = Path(manifest["encoding_evidence"]["path"])
    if not evidence_path.is_file():
        evidence_path = workloads_dir / "encoding-evidence.json"
    evidence_raw = evidence_path.read_bytes()
    if digest(evidence_raw) != manifest["encoding_evidence"]["sha256"]:
        raise ValueError("Workload encoding-evidence SHA differs from its manifest")
    expected = {case["name"]: case["service_encoding"] for case in json.loads(evidence_raw)["cases"]}
    samples = {sample["name"]: sample for sample in manifest["samples"]}
    fixed_raw = fixed_report_path.read_bytes()
    fixed = next(case for case in json.loads(fixed_raw)["cases"] if case["name"] == "mixed")
    if fixed.get("passed") is not True:
        raise ValueError("Saved fixed mixed encoding comparison must pass")
    inputs = [{"name": "fixed307", "request": fixed["request"], "expected": {"input_ids": fixed["vllm_encoding"]["input_ids"], "joint_schema": fixed["vllm_joint_schema"], "option_output_orders": fixed["vllm_option_output_orders"]}, "source": {"path": str(fixed_report_path.resolve()), "sha256": digest(fixed_raw), "case": "mixed"}}]
    for name in names:
        entry = samples[name]
        path = Path(entry["sample_file"])
        if not path.is_file():
            path = workloads_dir / (name + ".json")
        raw = path.read_bytes()
        if digest(raw) != entry["sample_file_sha256"]:
            raise ValueError(f"{name}: sample file SHA differs from manifest")
        sample = json.loads(raw)["samples"][0]
        if any(sample["request"].get(key) for key in ("images", "videos", "audio")):
            raise ValueError("CPU schema encoding benchmark accepts pure-text samples")
        encoding = expected[name]
        inputs.append({"name": name, "request": sample["request"], "expected": {key: encoding[key] for key in ("input_ids", "joint_schema", "option_output_orders")}, "source": {"path": str(path.resolve()), "sha256": digest(raw), "encoding_evidence": str(evidence_path.resolve()), "encoding_evidence_sha256": digest(evidence_raw)}})
    return inputs, {"manifest_path": str(manifest_path.resolve()), "manifest_sha256": digest(manifest_raw), "encoding_evidence_path": str(evidence_path.resolve()), "encoding_evidence_sha256": digest(evidence_raw)}


def one_iteration(data, strategy, request_class, parse_questions, expected):
    started = time.perf_counter_ns()
    request = request_class.model_validate(data)
    protocol_done = time.perf_counter_ns()
    questions = parse_questions(request, strategy.limits())
    questions_done = time.perf_counter_ns()
    state = strategy.render_state(request.state)
    render_done = time.perf_counter_ns()
    ids, schema, orders = strategy.encode(questions, state)
    encode_done = time.perf_counter_ns()
    # Equality verification and digesting are outside every measured interval.
    check_encoding(normalized_encoding(ids, schema, orders), expected)
    return {"protocol_parse_ms": (protocol_done - started) / 1e6, "questions_parse_ms": (questions_done - protocol_done) / 1e6, "parse_ms": (questions_done - started) / 1e6, "render_ms": (render_done - questions_done) / 1e6, "encode_ms": (encode_done - render_done) / 1e6, "total_ms": (encode_done - started) / 1e6, "exact_ids_schema_order_match": True}


def measure_variant(inputs, strategy, encoder, warmup, iterations, label):
    results = []
    for item in inputs:
        for _ in range(warmup):
            one_iteration(item["request"], strategy, encoder.request_class, encoder.parse_questions, item["expected"])
        samples = [dict(iteration=index, **one_iteration(item["request"], strategy, encoder.request_class, encoder.parse_questions, item["expected"])) for index in range(iterations)]
        results.append({"name": item["name"], "variant": label, "source": item["source"], "actual_input_tokens": len(item["expected"]["input_ids"]), "expected_encoding_sha256": json_digest(item["expected"]), "field_count": len(item["request"]["questions"]), "all_exact_encodings_match": True, "timing_ms": {key: distribution([row[key] for row in samples]) for key in TIMINGS}, "iterations": samples})
        print(json.dumps({"variant": label, "name": item["name"], "encode_p50_ms": results[-1]["timing_ms"]["encode_ms"]["p50"], "total_p50_ms": results[-1]["timing_ms"]["total_ms"]["p50"]}), flush=True)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workloads-dir", type=Path, required=True)
    parser.add_argument("--fixed-encoding-report", type=Path, default=Path("/workspace/yangchen/logs/systemone-encoding-20261112.json"))
    parser.add_argument("--model-dir", type=Path, default=Path("/workspace/yangchen/models/Clef-27B"))
    parser.add_argument("--vllm-src", type=Path, default=Path("/workspace/yangchen/work/src/vllm"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--names", nargs="+", default=DEFAULT_NAMES)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--baseline-only", action="store_true")
    args = parser.parse_args()
    if args.warmup < 10 or args.iterations < 100 or args.max_model_len <= 0:
        parser.error("Use at least 10 warmups, 100 measured iterations and positive max model length")
    started = time.perf_counter()
    report = {"schema_version": 1, "created_at": datetime.now(timezone.utc).isoformat(), "passed": False, "status": "preparing", "weights_loaded": False, "engine_constructed": False, "http_requests_sent": False, "npu_operations_executed": False, "warmup_per_shape_per_variant": args.warmup, "measured_iterations_per_shape_per_variant": args.iterations, "script_sha256": digest(Path(__file__).read_bytes()), "baseline": [], "schema_cache_ablation": []}
    encoder = None
    original = None
    try:
        report["clean_environment"] = clean_environment()
        inputs, provenance = load_inputs(args.workloads_dir, args.fixed_encoding_report, args.names)
        report["input_provenance"] = provenance
        from prepare_systemone_profile_workloads import ServingEncoder

        encoder = ServingEncoder(args.model_dir.resolve(), args.vllm_src.resolve(), args.max_model_len)
        import torch

        torch.set_num_threads(1)
        strategy = encoder.strategies[args.max_model_len]
        observer = sys.modules.get("vllm.profiler.clef_observe")
        runtime = {"helper_loaded": observer is not None, "markers_enabled": bool(getattr(observer, "_MARKERS", False)), "observe_directory_enabled": bool(getattr(observer, "_DIRECTORY", None)), "scope_filter_enabled": bool(getattr(observer, "_FILTER_RAW", None)), "torch_num_threads": torch.get_num_threads()}
        if runtime["markers_enabled"] or runtime["observe_directory_enabled"] or runtime["scope_filter_enabled"]:
            raise RuntimeError("Observer helper remained active after environment cleanup")
        report["runtime"] = runtime
        report["encoder"] = encoder.provenance
        report["load_and_validation_seconds"] = time.perf_counter() - started
        measured_started = time.perf_counter()
        report["baseline"] = measure_variant(inputs, strategy, encoder, args.warmup, args.iterations, "unobserved_baseline")
        if not args.baseline_only:
            states = [strategy.render_state(item["request"]["state"]) for item in inputs]
            original = strategy._tokens
            cache = SchemaTokenCache(original, states)
            strategy._tokens = cache
            try:
                report["schema_cache_ablation"] = measure_variant(inputs, strategy, encoder, args.warmup, args.iterations, "schema_tokens_cached_state_uncached")
            finally:
                strategy._tokens = original
                original = None
            report["schema_cache"] = cache.summary()
            if report["schema_cache"]["state_cache_entries"] != 0 or cache.state_bypasses < (args.warmup + args.iterations) * len(inputs):
                raise ValueError("State bypass/cache exclusion audit failed")
            report["comparisons"] = [{"name": baseline["name"], "baseline_encode_mean_ms": baseline["timing_ms"]["encode_ms"]["mean"], "cached_encode_mean_ms": cached["timing_ms"]["encode_ms"]["mean"], "encode_mean_ms_saved": baseline["timing_ms"]["encode_ms"]["mean"] - cached["timing_ms"]["encode_ms"]["mean"], "baseline_total_mean_ms": baseline["timing_ms"]["total_ms"]["mean"], "cached_total_mean_ms": cached["timing_ms"]["total_ms"]["mean"], "scope": "Offline CPU encoding ablation only; service E2E benefit requires an unprofiled API A/B experiment"} for baseline, cached in zip(report["baseline"], report["schema_cache_ablation"])]
        report["measurement_and_equality_checks_seconds"] = time.perf_counter() - measured_started
        report["timing_scope"] = "Protocol validation + parse_questions + render_state + strategy.encode; excludes loading, fixture JSON reads, equality checks, hashing and logging. Per-iteration exact equality checks run immediately after the timed region"
        report["ablation_scope"] = "Temporary instance _tokens wrapper only; every selected rendered state bypasses the cache; immutable tuples with fresh lists on hits; no installed-source/service changes"
        report.update(passed=True, status="complete")
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
    finally:
        if encoder is not None:
            if original is not None:
                encoder.strategies[args.max_model_len]._tokens = original
            encoder.close()
    usage = resource.getrusage(resource.RUSAGE_SELF)
    report["resource_usage"] = {"elapsed_seconds": time.perf_counter() - started, "user_cpu_seconds": usage.ru_utime, "system_cpu_seconds": usage.ru_stime, "max_rss_bytes": usage.ru_maxrss if sys.platform == "darwin" else usage.ru_maxrss * 1024}
    report["finished_at"] = datetime.now(timezone.utc).isoformat()
    save(args.output_dir / "report.json", report)
    compact = {key: value for key, value in report.items() if key not in ("baseline", "schema_cache_ablation", "traceback")}
    for key in ("baseline", "schema_cache_ablation"):
        compact[key] = [{name: value for name, value in row.items() if name != "iterations"} for row in report[key]]
    save(args.output_dir / "summary.json", compact)
    print(json.dumps({"passed": report["passed"], "status": report["status"], "report": str((args.output_dir / "report.json").resolve()), "error": report.get("error"), "measurement_seconds": report.get("measurement_and_equality_checks_seconds")}), flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
