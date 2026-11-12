#!/usr/bin/env python3
"""Check extension-matrix failure preservation without any HTTP requests."""

import copy
import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import run_systemone_profile_matrix as matrix


def create_plan(directory):
    samples = []
    for name, group in (("first", "length"), ("bad", "fields"), ("last", "fields")):
        request = {"model": "clef", "state": name, "questions": {"field": {"type": "choice", "criteria": {"a": "one", "b": "two"}}}}
        request_sha = matrix.sha256(json.dumps(request, ensure_ascii=False, separators=(",", ":")).encode())
        filename = name + ".json"
        (directory / filename).write_text(json.dumps({"samples": [{"name": name, "request": request, "metadata": {"actual_input_tokens": 50}}]}))
        samples.append({"name": name, "group": group, "actual_input_tokens": 50, "sample_file": filename, "request_sha256": request_sha, "variable": group, "controlled_variables": ["state"]})
    path = directory / "matrix-plan.json"
    path.write_text(json.dumps({"samples": samples}))
    return path


def report_fixture(tokens=50):
    row = {"passed": True, "http_status": 200, "actual_input_tokens": tokens, "response": {"usage": {"input_tokens": tokens}}, "request_sha256": "fixed", "elapsed_ms": 100, "field_count": 1}
    return {"passed": True, "warmup": [copy.deepcopy(row)], "requests": [copy.deepcopy(row), copy.deepcopy(row)], "summary": {"measurement_wall_seconds": .2, "observed_peak_http_concurrency": 1}}


class MatrixTests(unittest.TestCase):
    def test_plan_selection_prechecks_tokens_and_body_sha(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            path = create_plan(directory)
            samples, _ = matrix.select_samples(path, groups=["fields"], names=["last"], max_model_len=4096)
            self.assertEqual([sample["name"] for sample in samples], ["last"])
            for invalid in (None, True, 0):
                plan = json.loads(path.read_text())
                plan["samples"][0]["actual_input_tokens"] = invalid
                path.write_text(json.dumps(plan))
                with self.assertRaises(ValueError):
                    matrix.select_samples(path)
                create_plan(directory)
            with self.assertRaises(ValueError):
                matrix.select_samples(path, max_model_len=49)
            sample_path = directory / "first.json"
            document = json.loads(sample_path.read_text())
            document["samples"][0]["request"]["state"] = "modified"
            sample_path.write_text(json.dumps(document))
            with self.assertRaises(ValueError):
                matrix.select_samples(path)

    def test_api_usage_is_authoritative_and_bool_is_rejected(self):
        report = report_fixture()
        result = matrix.inspect_benchmark(report, 50, "fixed", 2)
        self.assertTrue(result["passed"])
        self.assertEqual(result["actual_samples"], 2)
        self.assertTrue(result["below_300_requests"])
        self.assertTrue(result["below_30_second_window"])
        report = copy.deepcopy(report)
        report["requests"][0]["response"]["usage"]["input_tokens"] = True
        result = matrix.inspect_benchmark(report, 50, "fixed", 2)
        self.assertFalse(result["passed"])
        self.assertEqual(result["api_input_token_counts"], [50])
        self.assertTrue(any("expected encoded plan=50" in issue for failure in result["failures"] for issue in failure["issues"]))
        report = report_fixture(tokens=49)
        result = matrix.inspect_benchmark(report, 50, "fixed", 2)
        self.assertFalse(result["passed"])
        self.assertEqual(result["api_input_token_counts"], [49])

    def test_failed_warmup_is_preserved(self):
        report = report_fixture()
        report["warmup"][0].update(passed=False, http_status=500, error="real failure")
        result = matrix.inspect_benchmark(report, 50, "fixed", 2)
        self.assertFalse(result["passed"])
        self.assertEqual(result["failures"][0]["phase"], "warmup")
        self.assertEqual(result["failures"][0]["error"], "real failure")

    def test_runner_continues_after_mismatch_and_keeps_all_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            plan = create_plan(directory)
            fake = directory / "benchmark.py"
            fake.write_text('''import argparse,hashlib,json
from pathlib import Path
p=argparse.ArgumentParser()
p.add_argument("--sample-file");p.add_argument("--report");p.add_argument("--requests",type=int);p.add_argument("--warmup",type=int)
a,_=p.parse_known_args()
s=json.loads(Path(a.sample_file).read_text())["samples"][0]
r=s["request"];n=s["metadata"]["actual_input_tokens"]-(r["state"]=="bad")
h=hashlib.sha256(json.dumps(r,ensure_ascii=False,separators=(",",":")).encode()).hexdigest()
row={"passed":True,"http_status":200,"actual_input_tokens":n,"response":{"usage":{"input_tokens":n}},"request_sha256":h,"elapsed_ms":100,"field_count":1}
report={"passed":True,"warmup":[dict(row) for _ in range(a.warmup)],"requests":[dict(row) for _ in range(a.requests)],"summary":{"measurement_wall_seconds":.2,"observed_peak_http_concurrency":1}}
Path(a.report).write_text(json.dumps(report))
print("saved original report")
''')
            output = directory / "output"

            def no_http_snapshot(base_url, target, phase, timeout):
                raw = b"metric 1\n"
                path = target / ("metrics-" + phase + ".prom")
                path.write_bytes(raw)
                return {"passed": True, "path": str(path), "sha256": matrix.sha256(raw), "http_status": 200}

            arguments = ["matrix", "--plan", str(plan), "--benchmark-script", str(fake), "--output-dir", str(output), "--warmup", "1", "--requests", "2"]
            with patch("sys.argv", arguments), patch.object(matrix, "metrics_snapshot", side_effect=no_http_snapshot) as metrics:
                code = matrix.main()
            self.assertEqual(code, 1)
            self.assertEqual(metrics.call_count, 6)
            summary = json.loads((output / "matrix-summary.json").read_text())
            self.assertFalse(summary["passed"])
            self.assertEqual([case["passed"] for case in summary["cases"]], [True, False, True])
            self.assertEqual(summary["cases"][1]["api_input_token_counts"], [49])
            self.assertIsNone(summary["hardware_execution_verified"])
            self.assertFalse(summary["stable_fixed_baseline_evidence"])
            with (output / "performance-matrix.csv").open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 3)
            self.assertEqual(rows[1]["passed"], "False")
            self.assertTrue((output / "bad/benchmark.json").is_file())
            self.assertTrue((output / "bad/benchmark.log").is_file())


if __name__ == "__main__":
    unittest.main()
