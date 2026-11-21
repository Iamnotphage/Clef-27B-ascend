#!/usr/bin/env python3
"""CPU-only checks for diagnostic selection, timing units and stop boundaries."""
import contextlib
import copy
import csv
from decimal import Decimal
import importlib.util
import io
from pathlib import Path
import tempfile
import unittest


def load(name):
    path = Path(__file__).with_name(name + ".py")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


analysis = load("analyze_down_gemm")
probe = load("probe_down_gemm")


class DiagnosticTests(unittest.TestCase):
    def csv_fixture(self, directory, wrong_shape=False):
        path = Path(directory) / "kernel_details.csv"
        fields = ["Device_id", "Input Shapes", "Input Formats", "Type", "Name",
                  "Duration(us)", "Start Time(us)", "Stream ID", "Block Num"] + list(analysis.METRICS)
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for i, duration in enumerate((100, 110, 120, 130)):
                start = Decimal("1794934703930000.000") + 1000 * i
                base = {"Device_id": "0", "Stream ID": "45"}
                writer.writerow(dict(base, **{"Start Time(us)": str(start), "Duration(us)": "5.000",
                                               "Name": "SwiGlu", "Type": "SwiGlu", "Input Shapes": "967,17408"}))
                writer.writerow(dict(base, **{"Start Time(us)": str(start + 1), "Duration(us)": "1.000",
                                               "Name": "different_stream", "Type": "Other", "Stream ID": "46"}))
                writer.writerow(dict(base, **{"Start Time(us)": str(start + Decimal("5.020")), "Duration(us)": duration,
                                               "Name": "vendor_down", "Type": "MatMulV2", "Block Num": "24",
                                               "Input Shapes": '"967,8704;544,320,16,16"' if not wrong_shape else '"967,5120;320,1088,16,16"',
                                               "Input Formats": "ND;FRACTAL_NZ"},
                                     **{key: ".8" for key in analysis.METRICS}))
        return path

    def test_exact_shape_counts_units_and_decimal_gaps(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.csv_fixture(directory)
            result = analysis.read_kernel_csv(path, 0, requests=2, layers=2)
            self.assertAlmostEqual(result["mean_per_request_down_sum_ms"], .230)
            self.assertEqual(result["per_request_down_sum_ms"], [.210, .250])
            self.assertEqual(result["same_stream_predecessors"], {"SwiGlu": 4})
            self.assertAlmostEqual(result["same_stream_predecessor_gap_us"]["median"], .020)
            self.assertAlmostEqual(result["effective_tflops_from_mean_kernel"], 2 * 967 * 8704 * 5120 / (115 * 1e6))

    def test_wrong_shape_or_count_or_rank_stops(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.csv_fixture(directory)
            with self.assertRaises(ValueError):
                analysis.read_kernel_csv(path, 0, requests=3, layers=2)
            with self.assertRaises(ValueError):
                analysis.read_kernel_csv(path, 1, requests=2, layers=2)
            path = self.csv_fixture(directory, wrong_shape=True)
            with self.assertRaises(ValueError):
                analysis.read_kernel_csv(path, 0, requests=2, layers=2)

    def timing_fixture(self):
        data = {"schema": "clef-down-gemm-v1", "phase": "timing", "status": "complete",
                "m": [967], "modes": list(probe.MODES), "pairs": 3, "ranks": []}
        for rank in (0, 1):
            samples = []
            for mode in probe.MODES:
                for pair in range(3):
                    for repeat in (0, 1):
                        samples.append({"mode": mode, "pair": pair, "repeat": repeat,
                                        "device_ms": .32 + rank * .01 + repeat * .001,
                                        "wall_ms": .52 - rank * .02 + repeat * .002, "enqueue_ms": .04,
                                        "conditioning_wall_ms": .8, "bit_equal": True})
            data["ranks"].append({"rank": rank, "cases": [{"m": 967, "capture_prefix": True, "samples": samples}]})
        return data

    def test_slower_rank_and_aa_noise(self):
        result = analysis.summarize_probe(self.timing_fixture())
        for row in result["slower_rank_medians"]:
            self.assertEqual(row["slower_device_rank"], 1)
            self.assertEqual(row["slower_wall_rank"], 0)
            self.assertAlmostEqual(row["device_ms"], .3305)
            self.assertAlmostEqual(row["wall_ms"], .521)
            self.assertAlmostEqual(row["aa_noise_max_pct"], 200 * .001 / .641)
            self.assertAlmostEqual(row["aa_wall_noise_max_pct"], 200 * .002 / 1.002)

    def test_incomplete_or_invalid_timing_stops(self):
        source = self.timing_fixture()
        cases = []
        data = copy.deepcopy(source)
        data["ranks"].pop()
        cases.append(data)
        data = copy.deepcopy(source)
        data["ranks"][0]["cases"][0]["samples"].pop()
        cases.append(data)
        data = copy.deepcopy(source)
        data["ranks"][0]["cases"][0]["samples"][0]["bit_equal"] = False
        cases.append(data)
        data = copy.deepcopy(source)
        data["ranks"][0]["cases"][0]["samples"][0]["device_ms"] = float("nan")
        cases.append(data)
        for data in cases:
            with self.assertRaises(ValueError):
                analysis.summarize_probe(data)

    def test_plan_and_device_boundaries(self):
        self.assertEqual(probe.arguments([]).phase, "plan")
        for argv in (["--phase", "timing"], ["--tokens", "1024"], ["--ranks", "0,0"],
                     ["--pairs", "6"], ["--phase", "once", "--execute", "--output", "/tmp/unused-down-probe.json"]):
            with self.subTest(argv=argv), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                probe.arguments(argv)


if __name__ == "__main__":
    unittest.main()
