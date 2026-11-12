#!/usr/bin/env python3
"""Synthetic fixtures for the profiler parser; these do not execute an NPU."""
from __future__ import annotations

import csv
from datetime import datetime
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
        self.assertFalse(analysis.COMM.search("aclnnDiv_BroadcastToAiCore_BroadcastTo BroadcastTo"))
        self.assertTrue(analysis.COMM.search("hccl_broadcast"))

    def test_epoch_timestamp_precision_and_hardware_metrics(self):
        path = self.root / "kernel_details.csv"
        with path.open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["Name", "Type", "Start Time(us)", "Duration(us)", "aic_mac_ratio", "aic_mac_time(us)", "Accelerator Core"])
            writer.writerows([["same-op", "MatMul", "1794494750394243.468", "1.820", 0.5, 0.9, "AI_CORE"],
                              ["same-op", "MatMul", "1794494750394245.288", "0.004", 0.25, 0.001, "AI_CORE"]])
        rows, _, _ = analysis.kernel_rows(path, self.root)
        self.assertEqual(rows[0]["start_ns"], 1794494750394243468)
        self.assertEqual(rows[0]["duration_us"], 1.820)
        self.assertEqual(analysis.union_us(analysis.intervals_ns(rows)), 1824)
        summary = analysis.summarize_rows(rows)[0]
        self.assertEqual(summary["cumulative_us"], 1.824)
        self.assertEqual(summary["interval_union_us"], 1.824)
        self.assertEqual(summary["hardware_metrics"]["aic_mac_ratio"]["samples"], 2)
        self.assertAlmostEqual(summary["hardware_metrics"]["aic_mac_ratio"]["duration_weighted_mean"], (0.5 * 1820 + 0.25 * 4) / 1824)
        self.assertAlmostEqual(summary["hardware_metrics"]["aic_mac_time(us)"]["sum_us"], 0.901)
        trace = self.root / "trace_view.json"
        trace.write_text('{"traceEvents":[{"ph":"M","name":"process_name","pid":1,"args":{"name":"Ascend Hardware"}},'
                         '{"ph":"X","name":"same-op","pid":1,"tid":1,"cat":"kernel","ts":1794494750394243.468,"dur":1.820}]}')
        kernels, _, _ = analysis.trace_rows(trace, self.root)
        self.assertEqual(kernels[0]["start_ns"], rows[0]["start_ns"])
        self.assertEqual(kernels[0]["duration_ns"], rows[0]["duration_ns"])

    def test_hccl_invocation_names_group_without_losing_evidence(self):
        path = self.root / "kernel_details.csv"
        with path.open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["Name", "Type", "Start Time(us)", "Duration(us)"])
            writer.writerows([["hcom_allReduce__503_0_1", "hcom_allReduce_", 100, 50],
                              ["hcom_allReduce__503_1_1", "hcom_allReduce_", 200, 50]])
        rows, _, _ = analysis.kernel_rows(path, self.root)
        summary = analysis.summarize_rows(rows)
        self.assertEqual(len(summary), 1)
        self.assertEqual(summary[0]["name"], "hcom_allReduce")
        self.assertEqual(summary[0]["count"], 2)
        self.assertEqual(summary[0]["cumulative_us"], 100)
        self.assertEqual(len(summary[0]["observed_names_sample"]), 2)
        self.assertEqual(summary[0]["evidence"][0]["name"], "hcom_allReduce__503_0_1")

    def test_communication_totals_are_excluded_from_operation_count(self):
        payload = {"Communication Time Info": {"Elapse Time(ms)": 0.05},
                   "Communication Bandwidth Info": {"HCCS": {"Transit Size(MB)": 1}}}
        path = self.root / "communication.json"
        path.write_text(json.dumps({"step": {"collective": {"hcom_allReduce_1": payload, "Total Op Info": payload}}}))
        details = analysis.communication_json(path, self.root)
        summary = analysis.communication_summary(details)
        self.assertEqual(summary["aggregate_entries_excluded"], 1)
        self.assertEqual(summary["by_collective"][0]["collective_type"], "all_reduce")
        self.assertEqual(summary["by_collective"][0]["operation_count"], 1)
        self.assertEqual(summary["by_collective"][0]["reported_elapsed_us_sum"], 50)

    def test_memory_components_keep_workspace_separate(self):
        directory = synthetic_rank(self.root, "memory-fixture", 0)
        output = directory / "ASCEND_PROFILER_OUTPUT"
        with (output / "memory_record.csv").open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["Component", "Timestamp(us)", "Total Allocated(MB)", "Total Reserved(MB)", "Total Active(MB)", "Device Type"])
            writer.writerows([["PTA", 100, 1024, 2048, 1024, "NPU:0"], ["PTA", 150, 1100, 2048, 1100, "NPU:0"],
                              ["WORKSPACE", 120, 128, 128, 128, "NPU:0"]])
        with (output / "operator_memory.csv").open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["Name", "Size(KB)", "Allocation Time(us)", "Release Time(us)", "Duration(us)", "Device Type"])
            writer.writerow(["synthetic-op", 65536, 100, "", "", "NPU:0"])
        result, report, _ = self.run_analysis(expected_ranks=1)
        self.assertEqual(result.returncode, 0, result.stderr)
        memory = report["ranks"][0]["memory"]
        self.assertTrue(memory["workspace_component_present"])
        components = {row["component"]: row for row in memory["component_device_peaks"]}
        self.assertEqual(components["PTA"]["peak_allocated_reported_mb"], 1100)
        self.assertEqual(components["WORKSPACE"]["peak_allocated_reported_mb"], 128)
        self.assertEqual(memory["top20_operator_allocations"][0]["size_reported_kb"], 65536)
        self.assertIsNone(memory["top20_operator_allocations"][0]["release_time_us"])

    def test_rank_identity_requires_metadata(self):
        directory = synthetic_rank(self.root, "worker0", 0)
        (directory / "profiler_info_0.json").unlink()
        result, report, _ = self.run_analysis(expected_ranks=1)
        self.assertEqual(result.returncode, 1)
        self.assertFalse(report["passed"])
        self.assertIsNone(report["ranks"][0]["rank"])
        self.assertIn("unidentified", report["ranks"][0]["rank_identity"]["source"])

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

    def test_native_engine_queue_durations_use_same_process_differences(self):
        rows = [{"name": "clef::request_finished_metrics", "pid": 99, "rank": None,
                 "request_id": f"synthetic-{index}", "start_wall_ns": 1000000000 + index * 10000000,
                 "duration_ns": 1000, "engine_metrics": {"queue_seconds": queue, "scheduled_to_output_seconds": 0.1}}
                for index, queue in enumerate((0.001, 0.002, 0.003))]
        (self.root / "cpu-99.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n")
        report = analysis.cpu_observations(self.root)
        queue = report["native_engine_metric_summaries"]["queue_seconds"]
        self.assertEqual(queue["sample_count"], 3)
        self.assertAlmostEqual(queue["sum_ms"], 6)
        self.assertAlmostEqual(queue["mean_ms"], 2)
        self.assertAlmostEqual(queue["p50_ms"], 2)
        self.assertAlmostEqual(queue["p95_ms"], 2.9)

    def test_benchmark_selects_measured_ids_and_keeps_unassigned_window_events(self):
        request_id = "clef-bench-synthetic-measured-0"
        started_at, finished_at = "2026-11-12T14:00:00+00:00", "2026-11-12T14:00:01+00:00"
        start = int(datetime.fromisoformat(started_at).timestamp()) * 1000000000
        rows = [
            {"name": "clef::frontend_request", "pid": 1, "rank": None, "request_id": request_id,
             "start_wall_ns": start, "duration_ns": 1000000000},
            {"name": "clef::request_finished_metrics", "pid": 2, "rank": None, "request_id": "decision-" + request_id + "-0",
             "start_wall_ns": start + 500000000, "duration_ns": 1000, "engine_metrics": {"queue_seconds": 0.001}},
            {"name": "clef::backbone", "pid": 3, "rank": 0, "request_id": None,
             "start_wall_ns": start + 100000000, "duration_ns": 100000000},
            {"name": "clef::frontend_request", "pid": 1, "rank": None, "request_id": "warmup-id",
             "start_wall_ns": start, "duration_ns": 100000},
            {"name": "clef::backbone", "pid": 3, "rank": 0, "request_id": None,
             "start_wall_ns": start - 2000000000, "duration_ns": 100000000},
        ]
        (self.root / "cpu-1.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n")
        benchmark = self.root / "benchmark.json"
        benchmark.write_text(json.dumps({"requests": [{"client_request_id": request_id, "started_at": started_at,
                                                       "finished_at": finished_at, "elapsed_ms": 1000, "passed": True}]}))
        report = analysis.cpu_observations(self.root, [benchmark])
        self.assertTrue(report["passed"])
        self.assertEqual(report["event_count"], 3)
        self.assertEqual(report["excluded_event_count"], 2)
        self.assertEqual(len(report["requests"]), 1)
        request = report["requests"][0]
        self.assertEqual(request["request_id"], request_id)
        self.assertEqual(request["event_count"], 2)
        self.assertEqual(request["client"]["elapsed_ms"], 1000)
        self.assertEqual(len(request["native_engine_metrics"]), 1)
        self.assertEqual(report["native_engine_metric_summaries"]["queue_seconds"]["mean_ms"], 1)

    def test_batch_membership_and_startup_cache_metadata(self):
        request_id = "clef-bench-batch-measured-0"
        started_at, finished_at = "2026-11-12T14:00:00+00:00", "2026-11-12T14:00:01+00:00"
        start = int(datetime.fromisoformat(started_at).timestamp()) * 1000000000
        rows = [
            {"name": "clef::worker_execute_model", "pid": 3, "rank": 0, "request_id": None,
             "batch_request_ids": ["decision-" + request_id + "-0-825ecd8c", "decision-unselected-other"],
             "start_wall_ns": start + 100000000, "duration_ns": 100000000},
            {"name": "clef::cache_initialize", "pid": 3, "rank": 0, "request_id": None,
             "start_wall_ns": start - 2000000000, "duration_ns": 100000000,
             "cache_memory": {"unique_storage_bytes": 1024, "unique_storage_bytes_by_kind": {"shared:FullAttentionSpec+MambaSpec": 1024}}},
        ]
        (self.root / "cpu-3.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n")
        benchmark = self.root / "benchmark.json"
        benchmark.write_text(json.dumps({"requests": [{"client_request_id": request_id, "started_at": started_at,
                                                       "finished_at": finished_at, "elapsed_ms": 1000, "passed": True}]}))
        report = analysis.cpu_observations(self.root, [benchmark])
        self.assertEqual(report["event_count"], 1)
        self.assertEqual(report["requests"][0]["request_id"], request_id)
        self.assertEqual(report["requests"][0]["stages_by_process_rank"][0]["shared_batch_event_count"], 1)
        self.assertEqual(report["cache_pool_observations"][0]["cache_memory"]["unique_storage_bytes"], 1024)

    def test_internal_worker_suffix_resolves_and_empty_steps_are_excluded(self):
        request_id = "clef-bench-synthetic-measured-0"
        started_at, finished_at = "2026-11-12T14:00:00+00:00", "2026-11-12T14:00:01+00:00"
        start = int(datetime.fromisoformat(started_at).timestamp()) * 1000000000
        rows = [{"name": "clef::worker_execute_model", "pid": 3, "rank": 0,
                 "request_id": "decision-" + request_id + "-0-825ecd8c",
                 "batch_request_ids": ["decision-" + request_id + "-0-825ecd8c"],
                 "start_wall_ns": start, "duration_ns": 100000000},
                {"name": "clef::worker_execute_model", "pid": 3, "rank": 0, "request_id": None,
                 "batch_request_ids": [], "start_wall_ns": start + 100000000, "duration_ns": 100000}]
        (self.root / "cpu-3.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n")
        benchmark = self.root / "benchmark.json"
        benchmark.write_text(json.dumps({"requests": [{"client_request_id": request_id, "started_at": started_at,
                                                       "finished_at": finished_at, "elapsed_ms": 1000, "passed": True}]}))
        report = analysis.cpu_observations(self.root, [benchmark])
        self.assertEqual(report["event_count"], 1)
        self.assertEqual(report["excluded_event_count"], 1)
        self.assertEqual(report["stages"][0]["mean_ms"], 100)
        self.assertEqual(report["requests"][0]["request_id"], request_id)

    def test_explicit_flow_correlates_modules_without_summing_nested_scopes(self):
        directory = synthetic_rank(self.root, "flow-fixture", 0)
        trace_path = directory / "ASCEND_PROFILER_OUTPUT/trace_view.json"
        trace = json.loads(trace_path.read_text())
        trace["traceEvents"] += [
            {"ph": "X", "name": "clef::backbone", "pid": 2, "tid": 1, "ts": 80, "dur": 40, "cat": "user_annotation"},
            {"ph": "X", "name": "clef::language_backbone", "pid": 2, "tid": 1, "ts": 70, "dur": 60, "cat": "user_annotation"},
            {"ph": "s", "name": "flow", "cat": "ac2g", "id": 1, "pid": 2, "tid": 1, "ts": 95},
            {"ph": "f", "name": "flow", "cat": "ac2g", "id": 1, "pid": 1, "tid": 0, "ts": 100},
            {"ph": "s", "name": "flow", "cat": "ac2g", "id": 2, "pid": 2, "tid": 1, "ts": 115},
            {"ph": "f", "name": "flow", "cat": "ac2g", "id": 2, "pid": 1, "tid": 1, "ts": 120},
            # Reverse input order verifies timestamp ordering of endpoints.
            {"ph": "f", "name": "flow", "cat": "ac2g", "id": 3, "pid": 1, "tid": 0, "ts": 170},
            {"ph": "s", "name": "flow", "cat": "ac2g", "id": 3, "pid": 2, "tid": 1, "ts": 160},
        ]
        trace_path.write_text(json.dumps(trace))
        result, report, _ = self.run_analysis(expected_ranks=1)
        self.assertEqual(result.returncode, 0, result.stderr)
        rank = report["ranks"][0]
        correlation = rank["timeline_metadata"][0]["module_flow_correlation"]
        self.assertEqual(correlation["flow_pairs_assigned_to_modules"], 3)
        modules = {row["module"]: row for row in correlation["modules"]}
        self.assertEqual(modules["clef::backbone"]["kernel_count"], 2)
        self.assertEqual(modules["clef::backbone"]["kernel_cumulative_us"], 100)
        self.assertEqual(modules["clef::backbone"]["kernel_union_us"], 70)
        self.assertEqual(modules["clef::language_backbone"]["kernel_union_us"], 70)
        self.assertEqual(modules["clef.joint_head"]["kernel_union_us"], 20)
        instances = {row["module"]: row for row in correlation["module_instances"]}
        self.assertEqual(instances["clef::backbone"]["device_span_us"], 70)
        self.assertEqual(instances["clef::backbone"]["cpu_inclusive_duration_us"], 40)
        self.assertEqual(instances["clef::backbone"]["device_start_us"], 100)
        self.assertEqual(instances["clef.joint_head"]["device_start_us"], 170)
        self.assertEqual(rank["kernel_count"], 3)
        self.assertIn("clef::backbone", rank["communication_operations"][0]["flow_observed_modules"])
        self.assertIsNone(modules["clef::backbone"]["critical_path_contribution_us"])

    def test_request_worker_window_and_observer_id_association(self):
        directory = synthetic_rank(self.root, "request-flow", 0)
        trace_path = directory / "ASCEND_PROFILER_OUTPUT/trace_view.json"
        trace = json.loads(trace_path.read_text())
        trace["traceEvents"] += [{"ph": "X", "name": "clef::worker_execute_model", "pid": 2, "tid": 1,
                                  "ts": 90, "dur": 110, "cat": "user_annotation"}]
        for identity, source, target, stream in [(1, 95, 100, 0), (2, 115, 120, 1), (3, 160, 170, 0)]:
            trace["traceEvents"] += [{"ph": "s", "cat": "async_npu", "id": identity, "pid": 2, "tid": 1, "ts": source},
                                      {"ph": "f", "cat": "async_npu", "id": identity, "pid": 1, "tid": stream, "ts": target}]
        trace_path.write_text(json.dumps(trace))
        result, report, _ = self.run_analysis(expected_ranks=1)
        self.assertEqual(result.returncode, 0, result.stderr)
        window = report["ranks"][0]["request_device_windows"][0]
        self.assertEqual(window["mapped_csv_kernel_count"], 3)
        self.assertEqual(window["kernel_union_us"], 90)
        self.assertEqual(window["top_operations"][0]["nonoverlapped_device_coverage_us"], 40)
        observation = {"name": "clef::worker_execute_model", "pid": 2, "rank": 0,
                       "request_id": "synthetic-request", "start_wall_ns": 89000, "duration_ns": 112000,
                       "batch_request_ids": ["synthetic-request"]}
        (self.root / "cpu-2.jsonl").write_text(json.dumps(observation) + "\n")
        report["cpu_observations"] = analysis.cpu_observations(self.root)
        analysis.associate_request_ids(report)
        self.assertEqual(window["request_ids"], ["synthetic-request"])
        self.assertEqual(window["observer_start_minus_trace_start_ns"], -1000)
        self.assertEqual(report["request_association"]["matched_module_instances"], 1)

    def test_task_stream_rounding_match_and_sync_wait_are_separate(self):
        path = self.root / "kernel_details.csv"
        with path.open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["Name", "Type", "Start Time(us)", "Duration(us)", "Task ID", "Stream ID", "Input Shapes", "Input Data Types"])
            writer.writerow(["MatMul", "MatMul", 170, "20.001", 5, 1, "[3,5120]", "BF16"])
        csv_rows, _, _ = analysis.kernel_rows(path, self.root)
        trace = self.root / "trace_view.json"
        trace.write_text(json.dumps({"traceEvents": [
            {"ph": "M", "name": "process_name", "pid": 1, "args": {"name": "Ascend Hardware"}},
            {"ph": "X", "name": "clef::joint_head", "pid": 2, "tid": 1, "ts": 90, "dur": 20, "cat": "user_annotation"},
            {"ph": "X", "name": "EVENT WAIT", "pid": 1, "tid": 1, "ts": 110, "dur": 60},
            {"ph": "X", "name": "MatMul", "pid": 1, "tid": 1, "ts": 170, "dur": "20.0004", "args": {"Task Id": 5, "Physic Stream Id": 1}},
            {"ph": "s", "id": 1, "cat": "async_npu", "pid": 2, "tid": 1, "ts": 95},
            {"ph": "f", "id": 1, "cat": "async_npu", "pid": 1, "tid": 1, "ts": 110},
            {"ph": "s", "id": 2, "cat": "async_npu", "pid": 2, "tid": 1, "ts": 100},
            {"ph": "f", "id": 2, "cat": "async_npu", "pid": 1, "tid": 1, "ts": 170},
        ]}))
        kernels, _, metadata = analysis.trace_rows(trace, self.root, csv_rows)
        matmul = next(row for row in kernels if row["name"] == "MatMul")
        self.assertEqual(matmul["shape"], "[3,5120]")
        self.assertEqual(matmul["csv_evidence"]["max_timestamp_or_duration_error_ns"], 1)
        module = metadata["module_flow_correlation"]["modules"][0]
        self.assertEqual(module["kernel_union_us"], 80)
        self.assertEqual(module["active_kernel_union_us"], 20)
        self.assertEqual(module["compute_kernel_union_us"], 20)
        self.assertEqual(module["sync_wait_union_us"], 60)
        matched, method, error = analysis.match_export_row(csv_rows[0], analysis.export_lookup(kernels))
        self.assertIsNotNone(matched)
        self.assertEqual(error, 1)

    def test_two_rank_evidence_and_no_double_counting(self):
        synthetic_rank(self.root, "fixture-a", 0)
        synthetic_rank(self.root, "fixture-b", 1)
        result, report, output = self.run_analysis()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(report["passed"])
        self.assertEqual(report["identified_rank_ids"], [0, 1])
        self.assertEqual(len(report["ranks"]), 2)
        for rank in report["ranks"]:
            self.assertIsInstance(rank["profiler_metadata"], list)
            self.assertEqual(rank["profiler_metadata"][0]["data"]["rank_id"], rank["rank"])
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
            self.assertEqual(rank["top_operations"][0]["nonoverlapped_device_coverage_us"], 40)
            self.assertEqual(rank["top_operations"][0]["active_compute_unique_coverage_us"], 70)
            self.assertEqual(rank["top_operations"][0]["active_compute_union_denominator_us"], 70)
            self.assertEqual(rank["top_operations"][0]["device_time_removal_upper_bound_us"], 70)
            self.assertIsNone(rank["top_operations"][0]["critical_path_contribution_us"])
            self.assertEqual(rank["top_operations"][0]["shape"], "[307,5120]")
            self.assertEqual(rank["top_operations"][0]["dtype"], "BF16")
            self.assertEqual(rank["top_operations"][0]["evidence"][0]["evidence_line"], 2)
            self.assertEqual(rank["cpu_scopes"][0]["name"], "clef.joint_head")
            self.assertIsNone(rank["cpu_scopes"][0]["nonoverlapped_device_coverage_us"])
            self.assertEqual(rank["communication_details"][0]["reported_elapsed_us"], 50)
            self.assertEqual(rank["communication_details"][0]["reported_wait_us"], 10)
            self.assertIsNone(rank["communication_details"][0]["reported_synchronization_us"])
            self.assertTrue(all(len(row["sha256"]) == 64 for row in rank["exported_file_index"]))
        for name in ("top20.csv", "operations.csv", "communication.csv", "cpu_scopes.csv", "module_instances.csv", "compact-evidence.json"):
            self.assertTrue((output / name).is_file())
        compact = json.loads((output / "compact-evidence.json").read_text())
        self.assertNotIn("all_operations", compact["ranks"][0])
        self.assertNotIn("cpu_scope_events", compact["ranks"][0])
        for shortened, full in zip(compact["ranks"][0]["top_operations"], report["ranks"][0]["top_operations"]):
            for key in ("name", "shape", "dtype", "count", "cumulative_us", "active_compute_unique_coverage_us"):
                self.assertEqual(shortened[key], full[key])
            self.assertLessEqual(len(shortened["evidence"]), 1)
        self.assertEqual(compact["ranks"][0]["communication_size_distribution"][0]["reported_transit_bytes"], 1048576)
        self.assertLessEqual(len(compact["ranks"][0]["communication_examples"]), 3)
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
