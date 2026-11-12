#!/usr/bin/env python3
"""Check aggregation, report independence and evidence limits with local fixtures."""

import copy
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import summarize_systemone_benchmarks as summarize
from verify_systemone import cases


def report_fixture(index=0, count=300, window=40, include_run_id=True):
    request = next(request for name, request in cases("clef") if name == "mixed")
    request_row = {"passed": True, "http_status": 200, "request": request, "actual_input_tokens": 307}
    report = {
        "schema_version": 2, "passed": True, "model": "clef",
        "started_at": f"2026-11-12T00:{index:02d}:00+00:00",
        "finished_at": f"2026-11-12T00:{index:02d}:45+00:00",
        "configuration": {"requests": count, "warmup": 10, "concurrency": 1, "validation_atol": 1e-5, "latency_scope": "HTTP, JSON and validation"},
        "warmup": [dict(request_row) for _ in range(10)],
        "requests": [dict(request_row, elapsed_ms=100 + item % 10 + index, client_queue_ms=item * 0.1, http_ms=99.0, json_decode_ms=0.1, protocol_validation_ms=0.1) for item in range(count)],
        "summary": {"measurement_wall_seconds": window, "successful_requests_per_second": 9999, "observed_peak_http_concurrency": 1},
    }
    if include_run_id:
        report["run_id"] = f"round-{index}"
    return report


class SummaryTests(unittest.TestCase):
    def test_recomputes_rates_hashes_and_pooled_percentiles(self):
        with tempfile.TemporaryDirectory(prefix="clef-summary-") as directory:
            rounds, observations = [], []
            for index in range(3):
                path = Path(directory) / f"round{index}.json"
                path.write_text(json.dumps(report_fixture(index, window=40 + index)))
                row, observation = summarize.read_round("tp1-1024", path, "measured")
                self.assertEqual(row["source_sha256"], hashlib.sha256(path.read_bytes()).hexdigest())
                self.assertAlmostEqual(row["requests_per_second"], 300 / (40 + index))
                self.assertAlmostEqual(row["decisions_per_second"], 900 / (40 + index))
                self.assertAlmostEqual(row["input_tokens_per_second"], 300 * 307 / (40 + index))
                self.assertTrue(row["measurement_protocol_passed"])
                rounds.append(row)
                observations.append(observation)
            groups = summarize.summarize_groups(rounds, observations)
            group = groups["tp1-1024"]
            self.assertTrue(group["measurement_protocol_passed"])
            self.assertEqual(group["pooled_request_latency_ms"]["sample_count"], 900)
            all_latencies = sum([observation["latencies"] for observation in observations], [])
            self.assertEqual(group["pooled_request_latency_ms"]["p95"], summarize.percentile(all_latencies, 0.95))
            self.assertGreater(group["round_requests_per_second"]["coefficient_of_variation"], 0)
            self.assertGreater(group["round_requests_per_second"]["range"], 0)

    def test_duplicate_files_and_run_ids_are_rejected(self):
        with tempfile.TemporaryDirectory(prefix="clef-summary-") as directory:
            path = Path(directory) / "round.json"
            path.write_text(json.dumps(report_fixture()))
            row, observation = summarize.read_round("tp1", path, "measured")
            group = summarize.summarize_groups([row] * 3, [observation] * 3)["tp1"]
            self.assertFalse(group["measurement_protocol_passed"])
            self.assertTrue(any("SHA-256" in reason for reason in group["ineligibility_reasons"]))
            self.assertTrue(any("run_id" in reason for reason in group["ineligibility_reasons"]))

    def test_legacy_intervals_and_changed_inputs_are_rejected(self):
        with tempfile.TemporaryDirectory(prefix="clef-summary-") as directory:
            rounds, observations = [], []
            for index in range(3):
                report = report_fixture(index, include_run_id=False)
                if index == 1:
                    report["started_at"] = "2026-11-12T00:00:10+00:00"
                if index == 2:
                    report["requests"][0] = copy.deepcopy(report["requests"][0])
                    report["requests"][0]["request"]["state"] = "changed input"
                path = Path(directory) / f"round{index}.json"
                path.write_text(json.dumps(report))
                row, observation = summarize.read_round("tp1", path, "measured")
                rounds.append(row)
                observations.append(observation)
            group = summarize.summarize_groups(rounds, observations)["tp1"]
            self.assertFalse(group["measurement_protocol_passed"])
            self.assertTrue(any("overlap" in reason for reason in group["ineligibility_reasons"]))
            self.assertTrue(any("differ" in reason for reason in group["ineligibility_reasons"]))
            self.assertFalse(rounds[2]["measurement_protocol_passed"])

    def test_synthetic_and_historical_short_windows_are_ineligible(self):
        with tempfile.TemporaryDirectory(prefix="clef-summary-") as directory:
            path = Path(directory) / "round.json"
            report = report_fixture()
            report["synthetic"] = True
            path.write_text(json.dumps(report))
            row, _ = summarize.read_round("tp1", path, "measured")
            self.assertFalse(row["measurement_protocol_passed"])
            self.assertEqual(row["evidence_kind"], "synthetic")
            report.pop("synthetic")
            path.write_text(json.dumps(report))
            row, _ = summarize.read_round("tp1", path, "synthetic")
            self.assertFalse(row["measurement_protocol_passed"])
            history = Path(__file__).resolve().parents[1] / "docs/evidence/systemone-benchmark-tp1-1024-2026-11-12.json"
            row, _ = summarize.read_round("tp1", history, "measured")
            self.assertFalse(row["measurement_protocol_passed"])
            self.assertEqual(row["actual_samples"], 30)
            self.assertIsNone(row["successful_stage_ms"]["client_queue_ms"]["mean"])
            self.assertEqual(row["successful_request_latency_ms"]["sample_count"], 30)

    def test_cli_outputs_csv_and_cannot_self_verify_hardware(self):
        with tempfile.TemporaryDirectory(prefix="clef-summary-") as directory:
            directory = Path(directory)
            arguments = []
            for index in range(3):
                path = directory / f"round{index}.json"
                path.write_text(json.dumps(report_fixture(index)))
                arguments += ["--report", f"tp1-1024={path}"]
            command = [sys.executable, str(Path(summarize.__file__).resolve()), *arguments, "--output-dir", str(directory / "output"), "--require-stable-baseline"]
            completed = subprocess.run(command, capture_output=True, text=True, timeout=10)
            self.assertEqual(completed.returncode, 0, completed.stderr + completed.stdout)
            summary = json.loads((directory / "output/summary.json").read_text())
            self.assertIsNone(summary["hardware_execution_verified"])
            self.assertIn("solely", summary["evidence_scope"])
            self.assertTrue(summary["all_groups_measurement_protocol_passed"])
            with (directory / "output/performance-matrix.csv").open(newline="") as handle:
                matrix = list(csv.DictReader(handle))
            self.assertEqual(len(matrix), 3)
            self.assertEqual(matrix[0]["actual_samples"], "300")
            self.assertEqual(json.loads(matrix[0]["configuration"])["warmup"], 10)
            completed = subprocess.run(command + ["--evidence-kind", "synthetic"], capture_output=True, text=True, timeout=10)
            self.assertEqual(completed.returncode, 1)
            summary = json.loads((directory / "output/summary.json").read_text())
            self.assertFalse(summary["all_groups_measurement_protocol_passed"])
            self.assertIsNone(summary["hardware_execution_verified"])


if __name__ == "__main__":
    unittest.main()
