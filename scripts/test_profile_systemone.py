#!/usr/bin/env python3
"""Failure-injection checks for profiling control; no NPU or model is used."""

import contextlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import profile_systemone as profile
import snapshot_systemone_profile as snapshot


class ProfileControlTests(unittest.TestCase):
    def run_case(self, *, start_ok=True, request_error=None, health_ok=True):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "window"
            endpoints = []

            def control(base, endpoint, timeout):
                endpoints.append(endpoint)
                return {"passed": start_ok if endpoint == "/start_profile" else True,
                        "http_status": 200 if start_ok or endpoint == "/stop_profile" else 500}

            def execute(argv, log):
                if log.name == "profile-requests.log" and request_error is not None:
                    raise request_error
                return {"returncode": 0, "argv": argv}

            argv = ["profile_systemone.py", "--output-dir", str(output), "--trace-root", str(root / "traces"),
                    "--expected-ranks", "2", "--skip-analysis"]
            with patch.object(sys, "argv", argv), patch.object(profile, "control", control), \
                 patch.object(profile, "execute", execute), \
                 patch.object(profile, "get_health", return_value={"passed": health_ok}), \
                 contextlib.redirect_stdout(io.StringIO()):
                code = profile.main()
            return code, endpoints, json.loads((output / "manifest.json").read_text())

    def test_stop_after_failed_start(self):
        code, endpoints, report = self.run_case(start_ok=False)
        self.assertEqual(code, 1)
        self.assertEqual(endpoints, ["/start_profile", "/stop_profile"])
        self.assertTrue(report["stop"]["passed"])
        self.assertFalse(report["passed"])

    def test_stop_after_request_interruption(self):
        code, endpoints, report = self.run_case(request_error=KeyboardInterrupt())
        self.assertEqual(code, 1)
        self.assertEqual(endpoints, ["/start_profile", "/stop_profile"])
        self.assertIn("KeyboardInterrupt", report["error"])
        self.assertTrue(report["stop"]["passed"])

    def test_http_success_requires_raw_rank_traces(self):
        code, endpoints, report = self.run_case()
        self.assertEqual(code, 1)
        self.assertEqual(endpoints, ["/start_profile", "/stop_profile"])
        self.assertIn("rank", report["error"])
        self.assertFalse(report["trace_completeness_passed"])

    def test_failed_health_skips_profiler(self):
        code, endpoints, report = self.run_case(health_ok=False)
        self.assertEqual(code, 1)
        self.assertEqual(endpoints, [])
        self.assertFalse(report["passed"])

    def test_snapshot_filters_credentials(self):
        self.assertEqual(snapshot.redact_argv(["serve", "--api-key", "example-secret", "--token=abc"]),
                         ["serve", "--api-key", "<redacted>", "--token=<redacted>"])
        self.assertNotIn("password", snapshot.redact("https://name:password@example.invalid/"))


if __name__ == "__main__":
    unittest.main()
