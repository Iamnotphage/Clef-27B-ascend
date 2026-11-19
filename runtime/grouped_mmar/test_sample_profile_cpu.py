#!/usr/bin/env python3
"""CPU-only tests for the H11 sample-profile collector preparation.

No torch, torch_npu, msprof, NPU, SSH, service or real collector output is used.
Synthetic records are explicitly labelled ``mock_only``; mock collector scripts
only prove this harness's handling of modelled exit/flush/persistence outcomes.
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sample_profile_collector as spc  # noqa: E402

APP = ["torchrun", "--nproc_per_node=2", "run_probe.py", "--arm", "H11", "--execute"]
MSPROF = "/opt/mock/msprof"
PLAN = Path(__file__).with_name("h11_sample_profile_plan.json")


def target(**over):
    base = dict(
        kernel_name="K_fresh_abc",
        stream_id=7,
        task_id=99,
        raw_submit_ns=900,
        raw_end_ns=1900,
        submit_ns=1000,
        end_ns=2000,
        end_kind="timeout",
        raw_clock="utc_ns",
        clock="monotonic_ns",
        offset_ns=100,
        clock_provenance="host plog UTC to collector monotonic via start/stop offset",
        source="StreamLaunchKernelV2",
        provenance="raw/plog:10",
    )
    base.update(over)
    return spc.ExpectedTarget(**base)


def record(**over):
    base = dict(
        valid=True,
        kernel_name="K_fresh_abc",
        raw_timestamp_ns=1500,
        timestamp_ns=1500,
        stream_id=7,
        task_id=99,
        raw_clock="monotonic_ns",
        clock="monotonic_ns",
        offset_ns=0,
        clock_provenance="collector raw timestamp, identity to common monotonic",
        source="synthetic",
        aic_sample=True,
    )
    base.update(over)
    return spc.SampleRecord(**base)


def adapter(records=(), **over):
    over.setdefault("mock_only", True)
    return spc.AdapterResult(records=tuple(records), **over)


def collector(**over):
    base = dict(
        exit_code=0,
        timed_out=False,
        truncated=False,
        output_missing=False,
        collector_exited_early=False,
        ranks_alive=False,
    )
    base.update(over)
    return base


def checks(**over):
    base = dict(argv=True, fresh_output=True, classification=True)
    base.update(over)
    return base


def classify(target_obj=None, adapter_obj=None, collector_obj=None, harness_checks=None):
    return spc.classify_profile(
        target_obj or target(),
        adapter_obj or adapter((record(),)),
        collector_obj or collector(),
        harness_checks=checks() if harness_checks is None else harness_checks,
    )


class CollectorArgvTest(unittest.TestCase):
    def test_exact_list_and_app_tail(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "profile"
            argv = spc.build_collector_argv(APP, out, msprof=MSPROF)
            prefix = [
                MSPROF,
                "--output",
                str(out),
                "--task-time",
                "l1",
                "--ai-core",
                "on",
                "--aic-mode",
                "sample-based",
                "--aic-freq",
                "100",
                "--aic-metrics",
                "PipeUtilization",
                "--delay",
                "10",
                "--duration",
                "120",
            ]
            self.assertEqual(argv, prefix + APP)
            self.assertEqual(argv[-len(APP):], APP)
            self.assertFalse(out.exists())

    def test_app_option_like_args_are_not_consumed(self):
        app = ["/app", "--aic-freq", "999", "--output", "app-out", "--delay", "1"]
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "profile"
            argv = spc.build_collector_argv(app, out, msprof=MSPROF)
            prefix = argv[: len(argv) - len(app)]
            self.assertEqual(argv[-len(app):], app)
            self.assertNotIn("999", prefix)
            self.assertNotIn("app-out", prefix)
            self.assertEqual(prefix[0], MSPROF)

    def test_rejects_string_relative_and_existing_output(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(spc.ProfileError):
                spc.build_collector_argv("torchrun", Path(td) / "p", msprof=MSPROF)
            with self.assertRaises(spc.ProfileError):
                spc.build_collector_argv(APP, "relative/out", msprof=MSPROF)
            existing = Path(td) / "existing"
            existing.mkdir()
            with self.assertRaises(spc.ProfileError):
                spc.build_collector_argv(APP, existing, msprof=MSPROF)

    def test_no_forbidden_or_msopprof_flags(self):
        with tempfile.TemporaryDirectory() as td:
            argv = spc.build_collector_argv(APP, Path(td) / "p", msprof=MSPROF)
            prefix = argv[: len(argv) - len(APP)]
            for flag in spc.FORBIDDEN_COLLECTOR_FLAGS:
                self.assertNotIn(flag, prefix)
            self.assertFalse(any("msopprof" in token for token in prefix))

    def test_flag_units_defaults_and_no_documented_conflict(self):
        specs = {row["name"]: row for row in spc.flag_specs()}
        self.assertEqual(specs["--aic-freq"]["value"], "100")
        self.assertEqual(specs["--aic-freq"]["unit"], "Hz")
        self.assertEqual(specs["--delay"]["value"], "10")
        self.assertEqual(specs["--delay"]["unit"], "s")
        self.assertEqual(specs["--duration"]["value"], "120")
        self.assertEqual(specs["--duration"]["unit"], "s")
        self.assertIsNone(specs["--task-time"]["unit"])
        for row in specs.values():
            self.assertEqual(row["help_conflict"], "none documented")
            self.assertFalse(row["runtime_proven"])
            self.assertNotEqual(row["unit"], "ms")


class TargetTest(unittest.TestCase):
    def test_arbitrary_fresh_ids_validate(self):
        self.assertEqual(spc.validate_target(target()), [])
        self.assertEqual(
            spc.validate_target(target(kernel_name="K_other_run", stream_id=8, task_id=100)),
            [],
        )

    def test_missing_fields_fail_closed(self):
        for name in spc.REQUIRED_TARGET_FIELDS:
            with self.subTest(field=name):
                reasons = spc.validate_target(target(**{name: None}))
                self.assertIn(f"missing_{name}", reasons)

    def test_ambiguity_clock_end_kind_window(self):
        self.assertIn("ambiguous_target", spc.validate_target(target(ambiguities=("two",))))
        self.assertIn("unknown_clock", spc.validate_target(target(clock="other")))
        self.assertIn("unknown_end_kind", spc.validate_target(target(end_kind="other")))
        self.assertIn("invalid_window", spc.validate_target(target(submit_ns=2000, end_ns=1000)))

    def test_no_hardcoded_old_ids_or_kernel(self):
        source = Path(spc.__file__).read_text()
        self.assertNotIn("stream_id == 47", source)
        self.assertNotIn("task_id == 13", source)
        plan = json.loads(PLAN.read_text())
        self.assertTrue(plan["target"]["no_hardcode"])
        self.assertTrue(all(value is None for value in plan["target"]["values"].values()))


class AdapterTest(unittest.TestCase):
    def _write(self, directory, value):
        path = Path(directory) / "samples.json"
        path.write_text(json.dumps(value))
        return path

    def test_synthetic_labelled_record(self):
        with tempfile.TemporaryDirectory() as td:
            path = self._write(
                td,
                {
                    "mock_only": True,
                    "records": [record().to_dict()],
                    "output_files": [{"path": "raw/out", "size": 1}],
                },
            )
            result = spc.SyntheticAdapter.load(path)
            self.assertTrue(result.mock_only)
            self.assertFalse(result.real)
            self.assertEqual(len(result.records), 1)
            self.assertEqual(result.reasons, ())
            self.assertEqual(result.output_files, ({"path": "raw/out", "size": 1},))

    def test_synthetic_unlabelled_is_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            for value in ({"records": []}, {"mock_only": False, "records": []}):
                with self.subTest(value=value):
                    path = self._write(td, value)
                    result = spc.SyntheticAdapter.load(path)
                    self.assertIn("synthetic_not_labeled", result.reasons)

    def test_synthetic_real_masquerade_and_missing_key_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            bad = record().to_dict()
            bad["source"] = "real"
            path = self._write(td, {"mock_only": True, "records": [bad]})
            result = spc.SyntheticAdapter.load(path)
            self.assertTrue(any(r.startswith("invalid_synthetic_record") for r in result.reasons))
            broken = record().to_dict()
            broken.pop("stream_id")
            path = self._write(td, {"mock_only": True, "records": [broken]})
            result = spc.SyntheticAdapter.load(path)
            self.assertTrue(any(r.startswith("invalid_synthetic_record") for r in result.reasons))

    def test_real_adapter_never_fabricates_schema(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "out"
            out.mkdir()
            (out / "raw.bin").write_bytes(b"raw")
            result = spc.RealOutputAdapter.load(out)
            self.assertTrue(result.real)
            self.assertEqual(result.records, ())
            self.assertIn("unknown_schema", result.reasons)
            self.assertEqual(len(result.output_files), 1)
            missing = spc.RealOutputAdapter.load(Path(td) / "missing")
            self.assertIn("unknown_schema", missing.reasons)
            self.assertIn("output_missing", missing.reasons)


class ClassificationTest(unittest.TestCase):
    def test_candidate_keeps_capability_unproven(self):
        result = classify()
        self.assertEqual(result["status"], spc.STATUS_CANDIDATE)
        self.assertFalse(result["capability_proven"])
        self.assertFalse(result["capability_demonstrated"])
        self.assertFalse(result["manual_device_review_required"])
        self.assertTrue(result["mock_candidate"])
        self.assertTrue(result["harness_pass"])
        self.assertTrue(result["mock_only"])
        self.assertFalse(result["kernel_counter_evidence"])
        self.assertIn("mock_only", result["reasons"])

    def test_kernel_name_only_is_unproven(self):
        result = classify(adapter_obj=adapter((record(stream_id=None, task_id=None),)))
        self.assertEqual(result["status"], spc.STATUS_UNPROVEN)
        self.assertIn("missing_sample_stream", result["reasons"])
        self.assertIn("missing_sample_task", result["reasons"])

    def test_clock_mismatch_is_unproven(self):
        result = classify(adapter_obj=adapter((record(clock="utc_ns"),)))
        self.assertEqual(result["status"], spc.STATUS_UNPROVEN)
        self.assertIn("clock_mismatch", result["reasons"])

    def test_missing_offset_raw_or_provenance_is_unproven(self):
        for over in (
            {"offset_ns": None},
            {"raw_timestamp_ns": None},
            {"clock_provenance": None},
        ):
            with self.subTest(over=over):
                result = classify(adapter_obj=adapter((record(**over),)))
                self.assertEqual(result["status"], spc.STATUS_UNPROVEN)
        self.assertIn("missing_offset_ns", spc.validate_target(target(offset_ns=None)))
        self.assertIn(
            "missing_clock_provenance", spc.validate_target(target(clock_provenance=None))
        )

    def test_conversion_mismatch_is_unproven(self):
        self.assertIn(
            "conversion_mismatch", spc.validate_target(target(submit_ns=1001))
        )
        result = classify(adapter_obj=adapter((record(timestamp_ns=1501),)))
        self.assertEqual(result["status"], spc.STATUS_UNPROVEN)
        self.assertIn("conversion_mismatch", result["reasons"])

    def test_identity_domain_requires_zero_offset(self):
        bad = record(
            raw_clock="monotonic_ns",
            clock="monotonic_ns",
            raw_timestamp_ns=1500,
            timestamp_ns=1501,
            offset_ns=1,
        )
        result = classify(adapter_obj=adapter((bad,)))
        self.assertEqual(result["status"], spc.STATUS_UNPROVEN)
        self.assertIn("identity_offset_nonzero", result["reasons"])

    def test_invalid_sample_is_unproven(self):
        result = classify(adapter_obj=adapter((record(valid=False),)))
        self.assertEqual(result["status"], spc.STATUS_UNPROVEN)
        self.assertIn("sample_invalid", result["reasons"])

    def test_before_and_after_window_are_not_success(self):
        for ts, reason in ((500, "before_submit"), (2500, "after_end")):
            with self.subTest(ts=ts):
                result = classify(
                    adapter_obj=adapter((record(raw_timestamp_ns=ts, timestamp_ns=ts),))
                )
                self.assertEqual(result["status"], spc.STATUS_NOT_SUCCESS)
                self.assertIn(reason, result["reasons"])

    def test_kernel_stream_task_mismatch_is_not_success(self):
        cases = (
            (record(kernel_name="K_other"), "kernel_mismatch"),
            (record(stream_id=8), "stream_mismatch"),
            (record(task_id=100), "task_mismatch"),
        )
        for bad, reason in cases:
            with self.subTest(reason=reason):
                result = classify(adapter_obj=adapter((bad,)))
                self.assertEqual(result["status"], spc.STATUS_NOT_SUCCESS)
                self.assertIn(reason, result["reasons"])

    def test_multiple_matches_are_ambiguous(self):
        result = classify(adapter_obj=adapter((record(), record())))
        self.assertEqual(result["status"], spc.STATUS_UNPROVEN)
        self.assertIn("multiple_candidates", result["reasons"])

    def test_empty_timeline_and_collector_failures_are_unproven(self):
        cases = (
            ("empty", adapter(()), collector(), "sample_empty"),
            ("timeline", adapter((), timeline_only=True), collector(), "timeline_only"),
            ("truncated", adapter((record(),)), collector(truncated=True), "truncated"),
            ("timeout", adapter((record(),)), collector(timed_out=True), "collector_timeout"),
            ("error", adapter((record(),)), collector(exit_code=3), "collector_error"),
            ("output", adapter((record(),)), collector(output_missing=True), "output_missing"),
            ("early", adapter((record(),)), collector(collector_exited_early=True), "collector_exited_early"),
            ("ranks", adapter((record(),)), collector(ranks_alive=True), "ranks_alive"),
        )
        for name, adapter_obj, collector_obj, reason in cases:
            with self.subTest(name=name):
                result = classify(adapter_obj=adapter_obj, collector_obj=collector_obj)
                self.assertEqual(result["status"], spc.STATUS_UNPROVEN)
                self.assertIn(reason, result["reasons"])
                self.assertFalse(result["capability_proven"])
                self.assertFalse(result["kernel_counter_evidence"])

    def test_real_adapter_unknown_schema_unproven(self):
        result = classify(
            adapter_obj=spc.AdapterResult(
                records=(), reasons=("unknown_schema",), mock_only=False, real=True
            )
        )
        self.assertEqual(result["status"], spc.STATUS_UNPROVEN)
        self.assertIn("unknown_schema", result["reasons"])
        self.assertFalse(result["capability_proven"])


class PlanTest(unittest.TestCase):
    def test_plan_matches_helper_and_boundaries(self):
        plan = json.loads(PLAN.read_text())
        self.assertEqual(plan["flags"], spc.flag_specs())
        self.assertEqual(plan["coverage"]["total_s"], 130)
        self.assertEqual(plan["coverage"]["timeout_approx_s"], 180)
        self.assertFalse(plan["coverage"]["covers_timeout"])
        self.assertFalse(plan["coverage"]["covers_late_pending"])
        self.assertTrue(plan["target"]["no_hardcode"])
        self.assertTrue(all(value is None for value in plan["target"]["values"].values()))
        self.assertIn("raw_submit_ns", plan["target"]["fields"])
        self.assertIn("clock_provenance", plan["target"]["fields"])
        self.assertTrue(plan["clock"]["clock_provenance_required"])
        self.assertEqual(plan["clock"]["same_label_without_offset"], "capability_unproven")
        self.assertTrue(plan["mock_only"])
        self.assertFalse(plan["device_go"])
        self.assertFalse(plan["success"]["kernel_name_only"])
        self.assertTrue(plan["success"]["capability_demonstrated_requires_manual_device_review"])
        self.assertEqual(plan["lifecycle"]["restore"], "exactly one begin_restore")
        self.assertEqual(plan["freeze"]["baseline_commit"], "8f4b843")
        self.assertTrue(plan["freeze"]["no_8ec"])
        self.assertTrue(plan["freeze"]["no_minus2_go"])
        self.assertTrue(plan["tests"]["existing_50_not_rerun"])
        self.assertIn("--aic-freq", plan["collector"]["command_prefix"])
        self.assertIn("100", plan["collector"]["command_prefix"])


class LifecycleTest(unittest.TestCase):
    def test_clean(self):
        result = spc.classify_lifecycle(output_present=True)
        self.assertFalse(result["cleanup_required"])
        self.assertFalse(result["restore_required"])
        self.assertEqual(result["reasons"], [])

    def test_early_collector_rank_survives_requires_restore(self):
        result = spc.classify_lifecycle(
            collector_exit="early",
            ranks_alive=True,
            output_present=True,
            restore_count=0,
        )
        self.assertTrue(result["cleanup_required"])
        self.assertTrue(result["restore_required"])
        self.assertIn("collector_exited_early", result["reasons"])
        self.assertIn("ranks_alive", result["reasons"])
        self.assertIn("restore_missing", result["reasons"])
        self.assertFalse(result["one_restore_ok"])

    def test_unique_restore_and_double_restore(self):
        one = spc.classify_lifecycle(
            collector_exit="early", ranks_alive=True, output_present=True, restore_count=1
        )
        self.assertTrue(one["one_restore_ok"])
        two = spc.classify_lifecycle(
            collector_exit="early", ranks_alive=True, output_present=True, restore_count=2
        )
        self.assertTrue(two["gate_fail_closed"])
        self.assertIn("multiple_restores", two["reasons"])
        self.assertFalse(two["one_restore_ok"])

    def test_missing_output_and_raw_artifacts(self):
        result = spc.classify_lifecycle(
            output_present=False, raw_artifacts_retained=False
        )
        self.assertIn("output_missing", result["reasons"])
        self.assertIn("raw_artifacts_missing", result["reasons"])


class CliTest(unittest.TestCase):
    def test_child_argv_cli_only_prints(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "profile"
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                rc = spc.main(
                    [
                        "child-argv",
                        "--app-argv-json",
                        json.dumps(APP),
                        "--output",
                        str(out),
                        "--msprof",
                        MSPROF,
                    ]
                )
            self.assertEqual(rc, 0)
            self.assertEqual(json.loads(buf.getvalue()), spc.build_collector_argv(APP, out, msprof=MSPROF))


class MockCollectorTest(unittest.TestCase):
    MOCK_COLLECTOR = r'''
import subprocess, sys
from pathlib import Path
mode, out = sys.argv[1], Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
app = Path(__file__).with_name("mock_app.py")
if mode == "normal":
    (out / "samples.json").write_text('{"mock_only": true, "records": []}')
    sys.exit(0)
if mode == "no_output":
    sys.exit(0)
if mode == "truncated":
    (out / "partial.txt").write_text("partial")
    sys.exit(3)
if mode == "exit_early":
    child = subprocess.Popen([sys.executable, str(app)], start_new_session=True)
    (out / "child.pid").write_text(str(child.pid))
    sys.exit(0)
if mode == "wait_app":
    child = subprocess.Popen([sys.executable, str(app)], start_new_session=True)
    (out / "child.pid").write_text(str(child.pid))
    child.wait()
    sys.exit(0)
sys.exit(2)
'''
    MOCK_APP = "import time\ntime.sleep(30)\n"

    def _scripts(self, td):
        root = Path(td)
        (root / "mock_app.py").write_text(self.MOCK_APP)
        (root / "mock_collector.py").write_text(self.MOCK_COLLECTOR)
        return root / "mock_collector.py"

    def test_normal_no_output_truncated(self):
        with tempfile.TemporaryDirectory() as td:
            collector_script = self._scripts(td)
            for mode, expected_exit, expected_files in (
                ("normal", 0, 1),
                ("no_output", 0, 0),
                ("truncated", 3, 1),
            ):
                with self.subTest(mode=mode):
                    out = Path(td) / f"out-{mode}"
                    proc = subprocess.run(
                        [sys.executable, str(collector_script), mode, str(out)],
                        timeout=5,
                        check=False,
                    )
                    self.assertEqual(proc.returncode, expected_exit)
                    loaded = spc.RealOutputAdapter.load(out)
                    self.assertTrue(loaded.real)
                    self.assertIn("unknown_schema", loaded.reasons)
                    self.assertEqual(len(loaded.output_files), expected_files)
                    classified = spc.classify_profile(
                        target(),
                        loaded,
                        collector(exit_code=proc.returncode),
                        harness_checks=checks(),
                    )
                    self.assertEqual(classified["status"], spc.STATUS_UNPROVEN)
                    self.assertFalse(classified["capability_proven"])

    def test_collector_exits_early_rank_survives(self):
        with tempfile.TemporaryDirectory() as td:
            collector_script = self._scripts(td)
            out = Path(td) / "out-early"
            proc = subprocess.run(
                [sys.executable, str(collector_script), "exit_early", str(out)],
                timeout=5,
                check=False,
            )
            self.assertEqual(proc.returncode, 0)
            child_pid = int((out / "child.pid").read_text())
            try:
                os.kill(child_pid, 0)
                alive = True
            except OSError:
                alive = False
            try:
                self.assertTrue(alive)
                life = spc.classify_lifecycle(
                    collector_exit="early",
                    ranks_alive=True,
                    output_present=True,
                    restore_count=0,
                )
                self.assertTrue(life["cleanup_required"])
                self.assertTrue(life["restore_required"])
                self.assertIn("restore_missing", life["reasons"])
            finally:
                with contextlib.suppress(OSError):
                    os.killpg(child_pid, signal.SIGKILL)

    def test_collector_waits_for_app(self):
        with tempfile.TemporaryDirectory() as td:
            collector_script = self._scripts(td)
            out = Path(td) / "out-wait"
            proc = subprocess.Popen(
                [sys.executable, str(collector_script), "wait_app", str(out)],
                start_new_session=True,
            )
            child_pid = None
            try:
                deadline = time.time() + 3
                while time.time() < deadline and not (out / "child.pid").exists():
                    time.sleep(0.02)
                self.assertEqual(proc.poll(), None)
                child_pid = int((out / "child.pid").read_text())
                life = spc.classify_lifecycle(
                    collector_exit="timeout",
                    ranks_alive=True,
                    output_present=True,
                    restore_count=0,
                )
                self.assertTrue(life["cleanup_required"])
                self.assertIn("collector_timeout", life["reasons"])
            finally:
                with contextlib.suppress(OSError):
                    os.killpg(proc.pid, signal.SIGKILL)
                proc.wait(timeout=5)
                if child_pid is not None:
                    with contextlib.suppress(OSError):
                        os.killpg(child_pid, signal.SIGKILL)


if __name__ == "__main__":
    unittest.main()
