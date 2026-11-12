#!/usr/bin/env python3
"""Verify run-wide classification and compact evidence with local JSON files."""

import csv
import json
from pathlib import Path
import tempfile
import unittest

import summarize_systemone_profile_run as summarize


def save(path, document):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document))


def launch(profile=False, observed=False):
    argv = ["python", "-m", "vllm", "--tensor-parallel-size", "1", "--max-model-len=1024", "--max-num-seqs", "1", "--max-num-batched-tokens", "1024"]
    if profile:
        argv += ["--profiler-config", '{"profiler":"torch"}']
    return {"argv": argv, "profile": profile, "observations": observed, "cpu_timing_only": False}


def report(run_id, count=300, wall=30, started="2026-11-12T00:00:30+00:00", finished="2026-11-12T00:00:40+00:00"):
    row = {"passed": True, "http_status": 200, "elapsed_ms": 100, "sample_name": "mixed", "actual_input_tokens": 307, "response": {"usage": {"input_tokens": 307}}, "candidate_counts": {"department": 2, "refund_requested": 2, "severity": 3}, "request_sha256": summarize.FIXED_REQUEST_SHA256, "request": {"state": "RAW-LARGE-BODY-SENTINEL"}, "http_ms": 99, "json_decode_ms": .1, "protocol_validation_ms": .2, "client_queue_ms": 10}
    return {"run_id": run_id, "started_at": started, "finished_at": finished, "passed": True, "configuration": {"requests": count, "warmup": 10, "concurrency": 1}, "requests": [dict(row) for _ in range(count)], "warmup": [dict(row) for _ in range(10)], "summary": {"successful_requests_per_second": count / wall, "measurement_wall_seconds": wall, "observed_peak_http_concurrency": 1}}


def window(parent, stop_passed=True):
    return {"source": str(parent / "manifest.json"), "launch_source": str(parent.parent / "launch.json"), "request_report": str(parent / "profile-requests.json"), "start": {"passed": True, "finished_at": "2026-11-12T00:00:10+00:00"}, "stop": {"passed": stop_passed, "finished_at": "2026-11-12T00:00:20+00:00"}, "request_window_started_at": "2026-11-12T00:00:10+00:00", "request_window_finished_at": "2026-11-12T00:00:19+00:00"}


class RunSummaryTests(unittest.TestCase):
    def test_launch_parser_keeps_profiler_configuration_separate(self):
        path = Path("/run/case/launch.json")
        value = launch(profile=True)
        value["environment"] = {"CLEF_PROFILE_OBSERVE_DIR": "/run/cpu", "CLEF_PROFILE_MARKERS": "0", "OMP_NUM_THREADS": "1"}
        parsed = summarize.parse_launch(value, path, b"source")
        self.assertEqual(parsed["tp"], 1)
        self.assertEqual(parsed["max_model_len"], 1024)
        self.assertTrue(parsed["profile_configured"])
        self.assertTrue(parsed["observations"])
        self.assertTrue(parsed["cpu_timing_only"])

    def test_active_intervals_and_failed_stop(self):
        launch_path = "/run/case/launch.json"
        parsed = summarize.parse_launch(launch(True), Path(launch_path), b"source")
        control = window(Path("/run/case/window"))
        after = report("after")
        active, _, _ = summarize.active_profile(Path("/run/case/unprofiled-fixed307.json"), after, launch_path, parsed, [control])
        self.assertIs(active, False)
        during = report("during", started="2026-11-12T00:00:12+00:00", finished="2026-11-12T00:00:18+00:00")
        active, _, _ = summarize.active_profile(Path("/run/case/batch.json"), during, launch_path, parsed, [control])
        self.assertIs(active, True)
        active, _, _ = summarize.active_profile(Path(control["request_report"]), during, launch_path, parsed, [control])
        self.assertIs(active, True)
        failed = window(Path("/run/case/window"), stop_passed=False)
        active, _, _ = summarize.active_profile(Path("/run/case/after.json"), after, launch_path, parsed, [failed])
        self.assertIsNone(active)

    def test_snapshots_warmups_trace_dirs_and_duplicate_reports_are_excluded(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            save(root / "case/launch.json", launch())
            document = report("same", count=30, wall=3)
            save(root / "case/benchmark.json", document)
            save(root / "case/copy.json", document)
            save(root / "case/warmup.json", report("warmup", count=10, wall=1))
            save(root / "snapshot-before/benchmark.json", report("snapshot"))
            broken = root / "case/traces/rank_ascend_pt/trace.json"
            broken.parent.mkdir(parents=True)
            broken.write_text("invalid large raw trace is never parsed")
            summary = summarize.aggregate(root, root / "output")
            self.assertEqual(summary["report_count"], 1)
            self.assertEqual(summary["read_errors"], [])
            reasons = [entry["reason"] for entry in summary["excluded"]]
            self.assertIn("warmup-only report", reasons)
            self.assertIn("duplicate source measurement", reasons)

    def test_formal_rounds_exclude_observed_and_profiled_windows(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stable = root / "tp1-1024-stable"
            save(stable / "launch.json", launch())
            for index in range(3):
                save(stable / f"round-{index + 1}.json", report(f"stable-{index}"))
            observed = root / "tp1-observed"
            save(observed / "launch.json", launch(True, True))
            save(observed / "cpu-matrix/benchmark.json", report("cpu-matrix", count=30, wall=3))
            active = report("profile", count=5, wall=.01, started="2026-11-12T00:00:12+00:00", finished="2026-11-12T00:00:18+00:00")
            save(observed / "window/profile-requests.json", active)
            control = window(observed / "window")
            control["trace_root"] = str(observed / "traces")
            save(observed / "window/manifest.json", control)
            summary = summarize.aggregate(root, root / "output")
            self.assertEqual(summary["classification_counts"], {"observer_enabled_unprofiled": 1, "profiled_window": 1, "stable_307": 3})
            self.assertTrue(summary["stable_groups"]["tp1-1024"]["measurement_protocol_passed"])
            self.assertEqual(summary["stable_groups"]["tp1-1024"]["round_requests_per_second"]["mean"], 10)
            self.assertEqual(summary["stable_groups"]["tp1-1024"]["pooled_request_latency_ms"]["sample_count"], 900)
            for row in summary["rows"]:
                if row["measurement_classification"] == "observer_enabled_unprofiled":
                    self.assertTrue(row["profile_configured"])
                    self.assertFalse(row["profile_active"])
                    self.assertFalse(row["eligible_for_formal_stable_comparison"])
            summarize.write_summary(root / "output", summary)
            raw_summary = (root / "output/summary.json").read_text()
            self.assertNotIn("RAW-LARGE-BODY-SENTINEL", raw_summary)
            self.assertFalse(summary["raw_request_response_bodies_copied"])
            with (root / "output/all-performance-matrix.csv").open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 5)
            self.assertEqual(rows[0]["http_mean_ms"], "99")

    def test_short_stable_rounds_and_duplicate_run_ids_cannot_pass_protocol(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stable = root / "tp1-1024-stable"
            save(stable / "launch.json", launch())
            for index in range(3):
                save(stable / f"round-{index + 1}.json", report(f"short-{index}", count=30, wall=3))
            summary = summarize.aggregate(root, root / "output")
            self.assertFalse(summary["stable_groups"]["tp1-1024"]["measurement_protocol_passed"])
            self.assertFalse(any(row["eligible_for_formal_stable_comparison"] for row in summary["rows"]))


if __name__ == "__main__":
    unittest.main()
