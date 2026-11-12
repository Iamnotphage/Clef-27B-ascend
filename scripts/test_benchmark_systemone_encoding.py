#!/usr/bin/env python3
"""Exercise the CPU schema ablation without importing torch/vLLM."""

import copy
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import benchmark_systemone_encoding as benchmark


class EncodingBenchmarkTests(unittest.TestCase):
    def test_schema_cache_never_caches_state_and_returns_fresh_lists(self):
        calls = []

        def tokens(text):
            calls.append(text)
            return [ord(value) for value in text]

        cache = benchmark.SchemaTokenCache(tokens, ["state", "another state"])
        first = cache("schema")
        first.append(999)
        self.assertEqual(cache("schema"), [ord(value) for value in "schema"])
        cache("state")
        cache("state")
        self.assertEqual(calls.count("schema"), 1)
        self.assertEqual(calls.count("state"), 2)
        summary = cache.summary()
        self.assertEqual(summary["state_cache_entries"], 0)
        self.assertEqual(summary["state_bypass_calls"], 2)
        self.assertEqual(summary["hits"], 1)
        self.assertIsInstance(cache.cache["schema"], tuple)

    def test_each_encoding_component_must_match_exactly(self):
        expected = {"input_ids": [1, 2], "joint_schema": {"option_counts": [2]}, "option_output_orders": [[1, 0]]}
        benchmark.check_encoding(copy.deepcopy(expected), expected)
        for key in expected:
            actual = copy.deepcopy(expected)
            actual[key] = []
            with self.assertRaises(ValueError):
                benchmark.check_encoding(actual, expected)

    def test_clean_environment_runs_before_model_import(self):
        with patch.dict(os.environ, {"CLEF_PROFILE_MARKERS": "1", "CLEF_PROFILE_OBSERVE_DIR": "/tmp/cpu", "CLEF_PROFILE_SCOPE_FILTER": "everything", "OMP_NUM_THREADS": "32"}, clear=True):
            audit = benchmark.clean_environment()
            self.assertEqual(audit["observer_variables_remaining"], [])
            self.assertEqual(os.environ["OMP_NUM_THREADS"], "1")
            self.assertNotIn("CLEF_PROFILE_MARKERS", os.environ)
        with patch.dict(sys.modules, {"torch": SimpleNamespace()}):
            with self.assertRaises(RuntimeError):
                benchmark.clean_environment()

    def test_iteration_times_parse_render_encode_and_checks_output(self):
        class Request:
            @staticmethod
            def model_validate(value):
                return SimpleNamespace(state=value["state"])

        class Strategy:
            @staticmethod
            def limits():
                return None

            @staticmethod
            def render_state(state):
                return state

            @staticmethod
            def encode(questions, state):
                return [1, 2], {"option_counts": [2]}, [[0, 1]]

        expected = {"input_ids": [1, 2], "joint_schema": {"option_counts": [2]}, "option_output_orders": [[0, 1]]}
        row = benchmark.one_iteration({"state": "real"}, Strategy(), Request, lambda request, limits: ["q"], expected)
        self.assertTrue(row["exact_ids_schema_order_match"])
        self.assertTrue(all(row[key] >= 0 for key in benchmark.TIMINGS))
        self.assertAlmostEqual(row["total_ms"], row["parse_ms"] + row["render_ms"] + row["encode_ms"])
        expected["input_ids"] = [9]
        with self.assertRaises(ValueError):
            benchmark.one_iteration({"state": "real"}, Strategy(), Request, lambda request, limits: ["q"], expected)

    def test_saved_evidence_and_samples_are_hash_checked(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            encoding = {"input_ids": [1, 2], "joint_schema": {"option_counts": [2]}, "option_output_orders": [[0, 1]]}
            sample_path = directory / "fields-1.json"
            sample_path.write_text(json.dumps({"samples": [{"request": {"model": "clef", "state": "real", "questions": {}}}]}))
            evidence_path = directory / "encoding-evidence.json"
            evidence_path.write_text(json.dumps({"cases": [{"name": "fields-1", "service_encoding": encoding}]}))
            manifest = {"samples": [{"name": "fields-1", "sample_file": str(sample_path), "sample_file_sha256": benchmark.digest(sample_path.read_bytes())}], "encoding_evidence": {"path": str(evidence_path), "sha256": benchmark.digest(evidence_path.read_bytes())}}
            (directory / "manifest.json").write_text(json.dumps(manifest))
            fixed = directory / "fixed.json"
            fixed.write_text(json.dumps({"cases": [{"name": "mixed", "passed": True, "request": {"state": "fixed"}, "vllm_encoding": {"input_ids": [1, 2]}, "vllm_joint_schema": encoding["joint_schema"], "vllm_option_output_orders": [[0, 1]]}]}))
            inputs, _ = benchmark.load_inputs(directory, fixed, ["fields-1"])
            self.assertEqual([row["name"] for row in inputs], ["fixed307", "fields-1"])
            self.assertEqual(inputs[1]["expected"], encoding)
            sample_path.write_text("changed")
            with self.assertRaises(ValueError):
                benchmark.load_inputs(directory, fixed, ["fields-1"])


if __name__ == "__main__":
    unittest.main()
