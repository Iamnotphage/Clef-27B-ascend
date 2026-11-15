#!/usr/bin/env python3
"""Unit tests for the GDN data-path arm driver.

Run with the project interpreter:
  python3 runtime/test_gdn_data_path_driver.py
"""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


driver = load("gdn_data_path_driver_test", HERE / "gdn_data_path_driver.py")


class DriverTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.flag_file = self.root / "flags.json"
        self.log = self.root / "service.log"
        self.sample = self.root / "samples.json"
        self.sample.write_text(json.dumps({"samples": [
            {"name": "one", "request": {"model": "clef", "questions": {}}},
            {"name": "two", "request": {"model": "clef", "questions": {}}},
        ]}))
        self.output = self.root / "out"

    def test_write_flags_is_atomic_and_booleans(self):
        payload = driver.write_flags(self.flag_file, "BIC", 123)
        self.assertEqual(payload["epoch"], 123)
        self.assertTrue(payload["decision_prefill"])
        self.assertTrue(payload["no_recompute"])
        self.assertTrue(payload["bsnd"])
        self.assertTrue(payload["skip_initial_state"])
        self.assertTrue(payload["skip_conv_state"])
        self.assertFalse(payload["preprocess_fusion"])
        on_disk = json.loads(self.flag_file.read_text())
        self.assertEqual(on_disk, payload)
        self.assertEqual(
            [path.name for path in self.root.iterdir() if path.name.startswith(".flags")],
            [],
        )

    def test_next_epoch_is_strictly_increasing(self):
        self.flag_file.write_text(json.dumps({"epoch": 41}))
        self.assertGreaterEqual(driver.next_epoch(self.flag_file), 42)
        self.flag_file.write_text("not json")
        self.assertGreater(driver.next_epoch(self.flag_file), 0)

    def test_orders_rotate_and_reverse(self):
        plan = driver.orders(["R", "B", "I", "C"], 4)
        self.assertEqual(plan[0], ["R", "B", "I", "C"])
        self.assertEqual(plan[1], ["R", "C", "I", "B"])
        self.assertEqual(plan[2], ["I", "C", "R", "B"])
        self.assertEqual(plan[3], ["I", "B", "R", "C"])
        for order in plan:
            self.assertEqual(sorted(order), sorted(["R", "B", "I", "C"]))

    def test_wait_for_epoch_counts_distinct_pids(self):
        self.log.write_text(
            "noise\n"
            "clef-gdn-path epoch=7 pid=101 bsnd=0\n"
            "clef-gdn-path epoch=7 pid=102 bsnd=0\n"
            "clef-gdn-path epoch=6 pid=103 bsnd=0\n"
        )
        self.assertEqual(driver.wait_for_epoch(self.log, 7, 2, 0.5), [101, 102])
        with self.assertRaisesRegex(SystemExit, "only 0 rank"):
            driver.wait_for_epoch(self.log, 8, 2, 0.3)

    def run_driver(self, arms="R", rounds=1, pairs=0, fake=None, opted=True):
        argv = [
            "--sample-file", str(self.sample),
            "--flag-file", str(self.flag_file),
            "--service-pid", "1",
            "--service-log", str(self.log),
            "--output-dir", str(self.output),
            "--arms", arms,
            "--rounds", str(rounds),
            "--warmup", "1",
            "--limit", "2",
        ]
        if pairs:
            argv += ["--pairs", str(pairs)]
        with mock.patch.object(driver, "service_flag_file",
                               return_value=str(self.flag_file) if opted else None), \
                mock.patch.object(driver, "wait_for_epoch", return_value=[101, 102]), \
                mock.patch.object(driver, "SETTLE_SECONDS", 0.0), \
                mock.patch.object(driver, "post"), \
                mock.patch.object(driver, "run_benchmark", side_effect=fake or self.fake_benchmark):
            return driver.main(argv)

    @staticmethod
    def fake_benchmark(args, report: Path) -> dict:
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(json.dumps({"summary": {
            "requested": 2, "successful": 2, "failed": 0,
            "wall_seconds": 1.0, "decisions_per_second": 2.0,
            "latency_ms": {"p50": 100.0, "p95": 120.0, "p99": 130.0, "mean": 105.0},
        }}))
        return json.loads(report.read_text())

    def test_screen_mode_saves_every_arm(self):
        self.flag_file.write_text(json.dumps({"old": True}))
        self.log.write_text("")
        self.assertEqual(self.run_driver("R,B", rounds=2), 0)
        record = json.loads((self.output / "driver.json").read_text())
        self.assertEqual(record["mode"], "screen")
        self.assertEqual(len(record["runs"]), 2)
        for round_record in record["runs"]:
            self.assertEqual(sorted(round_record["arms"]), ["B", "R"])
            for summary in round_record["arms"].values():
                self.assertEqual(summary["ranks"], [101, 102])
        self.assertEqual(
            json.loads(self.flag_file.read_text()), {"old": True})

    def test_paired_mode_summary_and_order(self):
        self.log.write_text("")
        self.assertEqual(self.run_driver("R,BIC", pairs=3), 0)
        record = json.loads((self.output / "driver.json").read_text())
        self.assertEqual([run["order"] for run in record["runs"]],
                         [["R", "BIC"], ["BIC", "R"], ["R", "BIC"]])
        self.assertEqual(record["summary"]["pairs"], 3)

    def test_failure_restores_the_original_flag_file(self):
        original = b'{"epoch": 9, "bsnd": true}\n'
        self.flag_file.write_bytes(original)
        self.log.write_text("")
        with self.assertRaises(SystemExit):
            self.run_driver("R", fake=mock.Mock(side_effect=SystemExit("boom")))
        self.assertEqual(self.flag_file.read_bytes(), original)
        self.assertFalse(any(path.name.startswith(".flags") for path in self.root.iterdir()))

    def test_service_must_opt_into_the_flag_file(self):
        self.log.write_text("")
        self.assertEqual(self.run_driver("R", opted=False), 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
