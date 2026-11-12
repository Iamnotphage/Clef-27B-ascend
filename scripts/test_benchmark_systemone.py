#!/usr/bin/env python3
"""Exercise workload hashes, variable decision counts and real HTTP timing."""

import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import unittest

import benchmark_systemone as benchmark


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def do_GET(self):
        self.send_response(200)
        self.end_headers()

    def do_POST(self):
        request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        time.sleep(0.008)
        if request["state"] == "fail":
            self.send_response(503)
            self.end_headers()
            self.wfile.write(b"{}")
            return
        answers = {}
        for qid, question in request["questions"].items():
            kind = question["type"]
            keys = list(question["criteria"]) if kind == "choice" else ["yes", "no"] if kind == "noul" else [str(i) for i in range(len(question["criteria"]))]
            answer = {"type": kind, "probabilities": {key: 1 / len(keys) for key in keys}, "confidence": 1 / len(keys)}
            if kind == "choice":
                answer["choice"] = keys[0]
            elif kind == "noul":
                answer["noul"] = 0.5
            else:
                answer["score"] = (len(keys) - 1) / 2
                answer["legend"] = dict(zip(keys, question["criteria"]))
            answers[qid] = answer
        response = {
            "model": request["model"], "object": "structured_decision",
            "answers": answers, "usage": {"input_tokens": 100 + len(answers), "output_tokens": 0},
            "diagnostics": {qid: {} for qid in answers},
        }
        self.send_response(200)
        self.send_header("X-Request-ID", self.headers["X-Request-ID"])
        self.end_headers()
        self.wfile.write(json.dumps(response).encode())


class BenchmarkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.url = f"http://127.0.0.1:{cls.server.server_port}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()

    def test_default_request_bytes_and_percentiles(self):
        samples, source = benchmark.load_samples(None, "clef")
        mixed = next(request for name, request in benchmark.cases("clef") if name == "mixed")
        historical_body = json.dumps(mixed, ensure_ascii=False, separators=(",", ":")).encode()
        self.assertEqual(samples[0]["body"], historical_body)
        self.assertEqual(samples[0]["request_sha256"], hashlib.sha256(historical_body).hexdigest())
        self.assertEqual(samples[0]["candidate_count"], 7)
        self.assertEqual(source["kind"], "historical_mixed")
        distribution = benchmark.distribution([10, 20, 30])
        self.assertAlmostEqual(distribution["p99"], 29.8)
        self.assertEqual(distribution["sample_count"], 3)
        self.assertEqual(distribution["sample_standard_deviation"], 10)

    def test_sample_file_rejects_empty_and_preserves_order(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.json"
            path.write_text('{"samples": []}')
            with self.assertRaises(ValueError):
                benchmark.load_samples(path, "clef")
            request = next(request for name, request in benchmark.cases("clef") if name == "choice_reordered")
            path.write_text(json.dumps(request))
            samples, source = benchmark.load_samples(path, "clef")
            self.assertEqual(list(samples[0]["request"]["questions"]["department"]["criteria"]), ["shipping", "billing"])
            self.assertEqual(source["sha256"], hashlib.sha256(path.read_bytes()).hexdigest())

    def test_http_failure_has_timing_and_is_excluded_from_throughput(self):
        samples, _ = benchmark.load_samples(None, "clef")
        request = dict(samples[0]["request"], state="fail")
        body = json.dumps(request).encode()
        tracker = benchmark.SendingConcurrency()
        result = benchmark.send_request(self.url, body, request, 0, "measured", 5, 1e-5, time.perf_counter() - 0.01, concurrency=tracker)
        self.assertFalse(result["passed"])
        self.assertEqual(result["http_status"], 503)
        self.assertGreater(result["client_queue_ms"], 9)
        self.assertGreater(result["http_ms"], 0)
        self.assertEqual(tracker.active, 0)
        summary = benchmark.summarize([result], 1.0, 3, 1)
        self.assertEqual(summary["successful_requests_per_second"], 0)
        self.assertEqual(summary["input_tokens_per_second"], 0)

    def test_cli_round_robin_minimum_window_and_stage_timing(self):
        with tempfile.TemporaryDirectory() as directory:
            sample_path = Path(directory) / "samples.json"
            report_path = Path(directory) / "report.json"
            by_name = dict(benchmark.cases("clef"))
            sample_path.write_text(json.dumps({"samples": [
                {"name": "single", "request": by_name["choice"], "metadata": {"expected_tokens": 101}},
                {"name": "mixed", "request": by_name["mixed"]},
            ]}))
            result = subprocess.run([
                sys.executable, str(Path(benchmark.__file__).resolve()),
                "--base-url", self.url, "--warmup", "1", "--requests", "5", "--concurrency", "2",
                "--min-duration-seconds", "0.12", "--sample-file", str(sample_path), "--report", str(report_path),
            ], capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            report = json.loads(report_path.read_text())
            summary, rows = report["summary"], report["requests"]
            self.assertTrue(report["passed"])
            self.assertEqual(len(report["warmup"]), 2)
            self.assertGreater(len(rows), 5)
            self.assertGreaterEqual(summary["measurement_wall_seconds"], 0.12)
            self.assertEqual(summary["successful_decisions"], sum(row["field_count"] for row in rows))
            self.assertEqual(summary["actual_input_tokens"]["successful_total"], sum(row["actual_input_tokens"] for row in rows))
            self.assertEqual(summary["observed_peak_http_concurrency"], 2)
            self.assertEqual(summary["successful_request_latency_ms"]["sample_count"], len(rows))
            self.assertGreater(max(row["client_queue_ms"] for row in rows), 1)
            self.assertEqual([row["sample_name"] for row in rows[:4]], ["single", "mixed", "single", "mixed"])
            for row in rows:
                self.assertEqual(row["client_request_id"], row["response_request_id"])
                self.assertEqual(row["actual_input_tokens"], 100 + row["field_count"])
                self.assertGreater(row["elapsed_ms"], row["http_ms"])
                self.assertGreaterEqual(row["json_decode_ms"], 0)
                self.assertGreaterEqual(row["protocol_validation_ms"], 0)
                self.assertIsNone(row["server_queue_ms"])


if __name__ == "__main__":
    unittest.main()
