#!/usr/bin/env python3
"""Synthetic fixtures for the profiler parser; these do not execute an NPU."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

try:
    from . import analyze_systemone_profile as analysis
except ImportError:
    import analyze_systemone_profile as analysis


SCRIPT = Path(__file__).with_name("analyze_systemone_profile.py")


def synthetic_rank(parent, name, rank):
    """Write a tiny synthetic torch_npu-style rank, clearly labelled as such."""
    directory = parent / f"{name}_ascend_pt"
    (directory / "FRAMEWORK").mkdir(parents=True)
    (directory / "FRAMEWORK" / "synthetic-fixture.txt").write_text("Synthetic unit-test fixture; no hardware measurement.\n")
    output = directory / "ASCEND_PROFILER_OUTPUT"
    output.mkdir()
    (directory / f"profiler_info_{rank}.json").write_text(json.dumps({"rank_id": rank, "synthetic": True}))
    rows = [
        ["MatMul", "MatMul", 100, 50, "[307,5120]", "BF16", 0],
        ["hcom_allReduce", "HCCL", 120, 50, "[307,5120]", "BF16", 1],
        ["MatMul", "MatMul", 170, 20, "[307,5120]", "BF16", 0],
    ]
    with (output / "kernel_details.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["Name", "Type", "Start Time(us)", "Duration(us)", "Input Shapes", "Input Data Types", "Stream Id"])
        writer.writerows(rows)
    events = [{"ph": "M", "name": "process_name", "pid": 1, "args": {"name": "Ascend Hardware"}}]
    events += [{"ph": "X", "name": name, "pid": 1, "tid": stream,
                "ts": start, "dur": duration, "cat": "kernel"}
               for name, _, start, duration, _, _, stream in rows]
    # CPU hccl launch events must not add to NPU kernel count or device time.
    events += [{"ph": "X", "name": "hccl_all_reduce", "pid": 2, "tid": 1,
                "ts": 115, "dur": 5, "cat": "cpu_op"},
               {"ph": "X", "name": "clef.joint_head", "pid": 2, "tid": 1,
                "ts": 150, "dur": 40, "cat": "user_annotation"}]
    (output / "trace_view.json").write_text(json.dumps({"synthetic": True, "traceEvents": events}))
    (output / "communication.json").write_text(json.dumps({
        "step0": {"collective": {"hcom_allReduce": {
            "Communication Time Info": {"Elapsed Time(ms)": 0.05, "Wait Time(ms)": 0.01},
            "Communication Bandwidth Info": {"HCCS": {"Transit Size(MB)": 1}},
        }}},
    }))
    return directory


class SyntheticProfileTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="synthetic-clef-profile-")
        self.root = Path(self.temporary.name)
        self.output_index = 0

    def tearDown(self):
        self.temporary.cleanup()

    def run_analysis(self, expected_ranks=2, selected=None):
        self.output_index += 1
        output = self.root / f"analysis-{self.output_index}"
        argv = [sys.executable, str(SCRIPT), str(self.root), "--output-dir", str(output),
                "--expected-ranks", str(expected_ranks), "--strict"]
        for path in selected or []:
            argv += ["--trace-dir", str(path)]
        result = subprocess.run(argv, capture_output=True, text=True, timeout=30)
        self.assertTrue((output / "summary.json").is_file(), result.stderr)
        return result, json.loads((output / "summary.json").read_text()), output

    def test_interval_union_and_overlap(self):
        self.assertEqual(analysis.union_us([(0, 10), (5, 15), (20, 25)]), 20)
        self.assertEqual(analysis.union_us([(0, 0), (4, 2), (2, 5)]), 3)
        self.assertEqual(analysis.intersection_us([(0, 10), (5, 15)], [(7, 12), (30, 40)]), 5)
        self.assertEqual(analysis.intersection_us([], [(1, 2)]), 0)

    def test_empty_raw_directories_fail_strict(self):
        first = synthetic_rank(self.root, "empty-raw-rank0", 0)
        synthetic_rank(self.root, "full-rank1", 1)
        (first / "FRAMEWORK" / "synthetic-fixture.txt").unlink()
        result, report, _ = self.run_analysis()
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(report["passed"])

    def test_inclusive_cpu_observation_overlap_and_request_index(self):
        # Two concurrent same-name CPU spans overlap by 1 ms. The enclosing
        # request is inclusive and must remain separately reported.
        rows = [
            {"name": "clef::tokenizer", "pid": 42, "rank": None, "request_id": "synthetic-a",
             "start_wall_ns": 1000000000000, "duration_ns": 2000000, "timing_scope": "host_wall_inclusive"},
            {"name": "clef::tokenizer", "pid": 42, "rank": None, "request_id": "synthetic-b",
             "start_wall_ns": 1000001000000, "duration_ns": 2000000, "timing_scope": "host_wall_inclusive"},
            {"name": "clef::frontend_request", "pid": 42, "rank": None, "request_id": "synthetic-a",
             "start_wall_ns": 1000000000000, "duration_ns": 5000000, "timing_scope": "host_wall_inclusive"},
        ]
        (self.root / "cpu-42.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n")
        report = analysis.cpu_observations(self.root)
        self.assertTrue(report["passed"])
        self.assertEqual(report["event_count"], 3)
        tokenizer = next(row for row in report["stages"] if row["name"] == "clef::tokenizer")
        self.assertEqual(tokenizer["inclusive_cumulative_ms"], 4)
        self.assertEqual(tokenizer["interval_union_ms"], 3)
        self.assertEqual(tokenizer["overlap_with_same_name_ms"], 1)
        self.assertEqual(tokenizer["mean_ms"], 2)
        request = next(row for row in report["requests"] if row["request_id"] == "synthetic-a")
        self.assertEqual(request["event_count"], 2)
        self.assertEqual(request["wall_range_ns"], [1000000000000, 1000005000000])
        with (self.root / "cpu-42.jsonl").open("a") as stream:
            stream.write("invalid json\n")
        report = analysis.cpu_observations(self.root)
        self.assertFalse(report["passed"])
        self.assertEqual(report["errors"][0]["line"], 4)

    def test_two_rank_evidence_and_no_double_counting(self):
        synthetic_rank(self.root, "fixture-a", 0)
        synthetic_rank(self.root, "fixture-b", 1)
        result, report, output = self.run_analysis()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(report["passed"])
        self.assertEqual(report["identified_rank_ids"], [0, 1])
        self.assertEqual(len(report["ranks"]), 2)
        for rank in report["ranks"]:
            self.assertTrue(rank["completeness"]["valid"])
            self.assertEqual(rank["kernel_source"], "kernel CSV")
            # CSV and timeline contain the same three kernels: count once.
            self.assertEqual(rank["kernel_count"], 3)
            self.assertEqual(rank["timeline_metadata"][0]["device_kernel_event_count"], 3)
            self.assertEqual(rank["kernel_cumulative_us"], 120)
            self.assertEqual(rank["kernel_union_us"], 90)
            self.assertEqual(rank["compute_union_us"], 70)
            self.assertEqual(rank["communication_union_us"], 50)
            self.assertEqual(rank["compute_communication_overlap_us"], 30)
            self.assertEqual(rank["communication_nonoverlapped_us"], 20)
            self.assertEqual(rank["kernel_window_span_us"], 90)
            self.assertEqual(rank["top_operations"][0]["name"], "MatMul")
            self.assertEqual(rank["top_operations"][0]["count"], 2)
            self.assertIsNone(rank["top_operations"][0]["critical_path_contribution_us"])
            self.assertEqual(rank["top_operations"][0]["shape"], "[307,5120]")
            self.assertEqual(rank["top_operations"][0]["dtype"], "BF16")
            self.assertEqual(rank["top_operations"][0]["evidence"][0]["evidence_line"], 2)
            self.assertEqual(rank["cpu_scopes"][0]["name"], "clef.joint_head")
            self.assertEqual(rank["communication_details"][0]["reported_elapsed_us"], 50)
            self.assertEqual(rank["communication_details"][0]["reported_wait_us"], 10)
            self.assertIsNone(rank["communication_details"][0]["reported_synchronization_us"])
            self.assertTrue(all(len(row["sha256"]) == 64 for row in rank["exported_file_index"]))
        for name in ("top20.csv", "operations.csv", "communication.csv", "cpu_scopes.csv"):
            self.assertTrue((output / name).is_file())
        with (output / "communication.csv").open() as stream:
            self.assertEqual(len(list(csv.DictReader(stream))), 2)

    def test_missing_one_rank_timeline_fails_strict(self):
        synthetic_rank(self.root, "fixture-a", 0)
        rank1 = synthetic_rank(self.root, "fixture-b", 1)
        (rank1 / "ASCEND_PROFILER_OUTPUT" / "trace_view.json").unlink()
        result, report, _ = self.run_analysis()
        self.assertEqual(result.returncode, 1)
        self.assertFalse(report["passed"])
        failed = [rank for rank in report["ranks"] if not rank["completeness"]["valid"]]
        self.assertEqual([rank["rank"] for rank in failed], [1])
        self.assertFalse(failed[0]["completeness"]["valid_device_timeline"])
        self.assertEqual(failed[0]["kernel_count"], 3)

    def test_duplicate_rank_metadata_fails_strict(self):
        synthetic_rank(self.root, "fixture-a", 0)
        synthetic_rank(self.root, "fixture-b", 0)
        result, report, _ = self.run_analysis()
        self.assertEqual(result.returncode, 1)
        self.assertFalse(report["passed"])
        self.assertEqual(report["duplicate_rank_ids"], [0])
        self.assertEqual(report["discovered_trace_count"], 2)

    def test_trace_dir_selects_one_window(self):
        window1 = self.root / "window1"
        window2 = self.root / "window2"
        synthetic_rank(window1, "fixture-a", 0)
        synthetic_rank(window1, "fixture-b", 1)
        chosen = [synthetic_rank(window2, "fixture-a", 0), synthetic_rank(window2, "fixture-b", 1)]
        result, all_report, _ = self.run_analysis()
        self.assertEqual(result.returncode, 1)
        self.assertEqual(all_report["discovered_trace_count"], 4)
        result, selected_report, _ = self.run_analysis(selected=chosen)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(selected_report["passed"])
        self.assertEqual(selected_report["discovered_trace_count"], 2)
        self.assertEqual({row["directory"] for row in selected_report["ranks"]}, {str(path.resolve()) for path in chosen})


if __name__ == "__main__":
    unittest.main(verbosity=2)
