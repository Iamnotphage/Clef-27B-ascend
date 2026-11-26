#!/usr/bin/env python3
"""Regressions for entrypoint argv boundaries and multimodal overrides."""

import argparse
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock


ENTRY = Path(__file__).resolve().parents[2] / "inference.py"
SPEC = importlib.util.spec_from_file_location("clef_inference_launch_test", ENTRY)
launcher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(launcher)


class LaunchTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        environment = mock.patch.dict(os.environ, {"CLEF_GDN_DECISION_PREFILL": "0"}, clear=True)
        environment.start()
        self.addCleanup(environment.stop)

    def launch(self, *arguments):
        output = io.StringIO()
        argv = [str(ENTRY), "--model-dir", str(self.root / "model"),
                "--output-dir", str(self.root / "output"), "--dry-run", *arguments]
        with mock.patch.object(launcher.sys, "argv", argv), \
                mock.patch.object(launcher, "software_provenance", return_value={}), \
                mock.patch.object(launcher, "resolve_schema_cache", return_value=(False, 0, None)), \
                contextlib.redirect_stdout(output):
            launcher.main()
        self.assertFalse((self.root / "output").exists())
        return json.loads(output.getvalue())

    def test_conv_qkv_installs_the_imported_preimage_once(self):
        target = self.root / launcher.CONV_QKV_TARGET
        target.parent.mkdir(parents=True)
        original = (Path(__file__).resolve().parent / "qkv_norm/qkv_gdn.py").read_bytes()
        target.write_bytes(original)
        package = SimpleNamespace(origin=str(target.parent.parent / "__init__.py"))
        output = self.root / "output"
        with mock.patch.object(launcher.importlib.util, "find_spec", return_value=package):
            first = launcher.conv_qkv_patch(True, "/vendor", output, apply=True)
            before = target.stat()
            second = launcher.conv_qkv_patch(True, "/vendor", output, apply=True)
            after = target.stat()
        self.assertTrue(first["ready"] and first["installed"])
        self.assertEqual(first["source_sha256_before"], launcher.CONV_QKV_PREIMAGE)
        self.assertEqual(first["source_sha256"], launcher.CONV_QKV_POSTIMAGE)
        self.assertEqual((Path(first["backup_dir"]) / "gdn.py.preimage").read_bytes(), original)
        self.assertTrue(second["ready"])
        self.assertFalse(second["installed"])
        self.assertEqual((after.st_ino, after.st_mtime_ns), (before.st_ino, before.st_mtime_ns))

    def test_conv_qkv_unknown_source_and_backup_conflict_fail_closed(self):
        target = self.root / launcher.CONV_QKV_TARGET
        target.parent.mkdir(parents=True)
        output = self.root / "output"
        backup = output / "conv-qkv-source-backup" / "gdn.py.preimage"
        package = SimpleNamespace(origin=str(target.parent.parent / "__init__.py"))
        with mock.patch.object(launcher.importlib.util, "find_spec", return_value=package):
            with self.assertRaisesRegex(SystemExit, "source state could not be verified"):
                launcher.conv_qkv_patch(True, "/vendor", output, apply=True)
            target.write_bytes(b"unknown source")
            with self.assertRaisesRegex(SystemExit, "source state could not be verified"):
                launcher.conv_qkv_patch(True, "/vendor", output, apply=True)
            self.assertEqual(target.read_bytes(), b"unknown source")
            self.assertFalse(output.exists())
            original = (Path(__file__).resolve().parent / "qkv_norm/qkv_gdn.py").read_bytes()
            target.write_bytes(original)
            backup.parent.mkdir(parents=True)
            backup.write_bytes(b"unknown backup")
            with self.assertRaisesRegex(SystemExit, "backup SHA mismatch"):
                launcher.conv_qkv_patch(True, "/vendor", output, apply=True)
        self.assertEqual(target.read_bytes(), original)
        self.assertEqual(backup.read_bytes(), b"unknown backup")

    def test_conv_qkv_dry_run_and_off_do_not_resolve_or_edit_the_tree(self):
        decision = {"enabled": True, "requested": True}
        with mock.patch.object(launcher, "gdn_decision_prefill", return_value=decision), \
                mock.patch.object(launcher, "installer_call") as installer:
            on = self.launch("--config", "s1")
            self.assertTrue(on["gdn_decision_prefill"]["conv_qkv_patch"]["pending_install"])
            self.assertEqual(on["service_environment"]["CLEF_GDN_CONV_QKV"], "0")
            with mock.patch.dict(os.environ, {"CLEF_GDN_CONV_QKV": "off"}):
                off = self.launch("--config", "s1")
            self.assertFalse(off["gdn_decision_prefill"]["conv_qkv_patch"]["requested"])
            self.assertEqual(off["service_environment"]["CLEF_GDN_CONV_QKV"], "0")
            installer.assert_not_called()

    def assert_mm(self, launch, value, source):
        command = launch["argv"]
        self.assertEqual(command.count("--mm-processor-kwargs"), 1)
        self.assertEqual(command[command.index("--mm-processor-kwargs") + 1], value)
        self.assertEqual(launch["effective"]["mm_processor_kwargs"], value)
        self.assertEqual(launch["sources"]["mm_processor_kwargs"], source)

    def without_scheduler(self, launch):
        """Drop the launcher-owned Tier 1 scheduler pair from argv."""
        argv = list(launch["argv"])
        if argv[-2:] == [
            "--scheduler-cls",
            "clef_encoder_cache_scheduler.EncoderCacheBudgetScheduler",
        ]:
            argv = argv[:-2]
        return argv

    def test_s1_data_path_defaults_follow_the_decision_patch(self):
        disabled = self.launch("--config", "s1")
        for key in ("bsnd", "skip_initial_state", "skip_conv_state",
                    "preprocess_fusion"):
            self.assertFalse(
                disabled["gdn_decision_prefill"]["data_path"][key]["effective"])
        enabled_patch = {
            "enabled": True,
            "requested": True,
            "source": "config:s1",
            "patch_version": 3,
            "patch_state": {},
            "files": {},
            "flag_module_sha256": "x",
            "installer_sha256": "x",
        }
        with mock.patch.object(launcher, "gdn_decision_prefill",
                               return_value=dict(enabled_patch)):
            launch = self.launch("--config", "s1")
        record = launch["gdn_decision_prefill"]["data_path"]
        for key in ("bsnd", "skip_initial_state", "skip_conv_state"):
            self.assertTrue(record[key]["effective"])
            self.assertEqual(record[key]["source"], "config:s1")
        self.assertTrue(record["preprocess_fusion"]["effective"])
        self.assertEqual(launch["service_environment"]["CLEF_GDN_BSND"], "1")
        self.assertEqual(
            launch["service_environment"]["CLEF_GDN_SKIP_INITIAL_STATE"], "1")
        self.assertEqual(
            launch["service_environment"]["CLEF_GDN_SKIP_CONV_STATE"], "1")
        self.assertEqual(
            launch["service_environment"]["CLEF_GDN_PREPROCESS_FUSION"], "1")
        custom = self.launch("--config", "custom")
        for key in ("bsnd", "skip_initial_state", "skip_conv_state"):
            self.assertFalse(
                custom["gdn_decision_prefill"]["data_path"][key]["effective"])

    def test_flag_file_values_are_recorded_and_win(self):
        flag = self.root / "flags.json"
        flag.write_text(json.dumps({
            "epoch": 9, "bsnd": False, "skip_conv_state": True,
        }))
        os.environ["CLEF_GDN_FLAG_FILE"] = str(flag)
        self.addCleanup(os.environ.pop, "CLEF_GDN_FLAG_FILE", None)
        enabled_patch = {"enabled": True, "requested": True,
                         "source": "config:s1", "files": {},
                         "patch_state": {}, "flag_module_sha256": "x",
                         "installer_sha256": "x", "patch_version": 3}
        with mock.patch.object(launcher, "gdn_decision_prefill",
                               return_value=dict(enabled_patch)):
            launch = self.launch("--config", "s1")
        record = launch["gdn_decision_prefill"]["data_path"]
        self.assertFalse(record["bsnd"]["effective"])
        self.assertEqual(record["bsnd"]["source"], "flag_file")
        self.assertTrue(record["skip_conv_state"]["effective"])
        self.assertEqual(record["skip_conv_state"]["source"], "flag_file")
        self.assertEqual(record["flag_file"]["epoch"], 9)
        self.assertIsNotNone(record["flag_file"]["sha256"])
        self.assertNotIn("CLEF_GDN_BSND", launch["service_environment"])

    def test_explicit_data_path_env_wins_and_is_recorded(self):
        enabled_patch = {"enabled": True, "requested": True,
                         "source": "config:s1", "files": {},
                         "patch_state": {}, "flag_module_sha256": "x",
                         "installer_sha256": "x", "patch_version": 3}
        os.environ["CLEF_GDN_BSND"] = "0"
        self.addCleanup(os.environ.pop, "CLEF_GDN_BSND", None)
        with mock.patch.object(launcher, "gdn_decision_prefill",
                               return_value=dict(enabled_patch)):
            launch = self.launch("--config", "s1")
        record = launch["gdn_decision_prefill"]["data_path"]["bsnd"]
        self.assertFalse(record["effective"])
        self.assertEqual(record["source"], "env")
        self.assertNotIn("CLEF_GDN_BSND", launch["service_environment"])

    def test_s1_defaults_emit_one_mm_option(self):
        launch = self.launch("--config", "s1")
        self.assert_mm(launch, launcher.MM_PROCESSOR_KWARGS, "config:s1")

    def test_encoder_cache_default_is_tier1_and_explicit_off_disables(self):
        baseline = self.launch("--config", "s1")
        self.assertIn("--scheduler-cls", baseline["argv"])
        self.assertEqual(baseline["argv"][-2:], [
            "--scheduler-cls",
            "clef_encoder_cache_scheduler.EncoderCacheBudgetScheduler"])
        self.assertEqual(baseline["service_environment"]
                         ["CLEF_ENCODER_CACHE_TOKENS"], "262144")
        self.assertEqual(baseline["effective"]["encoder_cache_tokens"], 262144)
        self.assertEqual(baseline["sources"]["encoder_cache_tokens"], "config:s1")
        self.assertEqual(len(baseline["effective"]
                             ["encoder_cache_scheduler_sha256"]), 64)
        disabled = self.launch("--config", "s1", "--encoder-cache-tokens", "0")
        self.assertNotIn("--scheduler-cls", disabled["argv"])
        self.assertNotIn("CLEF_ENCODER_CACHE_TOKENS",
                         disabled["service_environment"])
        self.assertNotIn("encoder_cache_tokens", disabled["effective"])

    def test_s1_defaults_serve_erqa_without_overrides(self):
        launch = self.launch("--config", "s1")
        for key, value in (
            ("max_model_len", 16384),
            ("max_num_seqs", 16),
            ("max_num_batched_tokens", 16384),
            ("gpu_memory_utilization", 0.85),
            ("limit_mm_per_prompt", '{"image":16,"video":0}'),
        ):
            self.assertEqual(launch["effective"][key], value)
            self.assertEqual(launch["sources"][key], "config:s1")
        self.assertIn('"longest_edge":1048576',
                      launch["effective"]["mm_processor_kwargs"])
        self.assertIn('"shortest_edge":65536',
                      launch["effective"]["mm_processor_kwargs"])
        self.assertEqual(launch["effective"]["encoder_cache_tokens"], 262144)
        self.assertEqual(launch["sources"]["encoder_cache_tokens"], "config:s1")
        self.assertEqual(launch["service_environment"]
                         ["CLEF_ENCODER_CACHE_TOKENS"], "262144")
        self.assertEqual(launch["argv"][-2:], [
            "--scheduler-cls",
            "clef_encoder_cache_scheduler.EncoderCacheBudgetScheduler"])
        record = launch["systemone_mm_state"]
        self.assertEqual(record["mode"], "extract_question_dict")
        self.assertTrue(record["pending_install"])
        self.assertFalse(record["enabled"])
        self.assertEqual(launch["service_environment"]["CLEF_STATE_MM_MODE"],
                         "extract_question_dict")

    def test_systemone_mm_mode_is_recorded_and_validated(self):
        os.environ["CLEF_STATE_MM_MODE"] = "extract_question_dict"
        self.addCleanup(os.environ.pop, "CLEF_STATE_MM_MODE", None)
        launch = self.launch("--config", "s1")
        self.assertEqual(launch["systemone_mm_state"]["mode"],
                         "extract_question_dict")
        self.assertEqual(launch["service_environment"]["CLEF_STATE_MM_MODE"],
                         "extract_question_dict")
        os.environ["CLEF_STATE_MM_MODE"] = "bogus"
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                self.launch("--config", "s1")

    def test_systemone_mm_dry_run_never_probes_the_tree(self):
        with mock.patch.object(launcher.subprocess, "run") as run:
            launch = self.launch("--config", "s1")
        run.assert_not_called()
        self.assertTrue(launch["systemone_mm_state"]["pending_install"])
        self.assertIsNotNone(launch["systemone_mm_state"]["installer_sha256"])
        self.assertIsNotNone(launch["systemone_mm_state"]["module_sha256"])

    def test_systemone_mm_apply_verifies_the_pinned_report(self):
        contract = launcher.load_python_contract(
            launcher.SYSTEMONE_MM_INSTALLER, "clef_systemone_mm_contract_test")
        base = {
            "patch_version": contract.PATCH_VERSION,
            "upstream_sha256": contract.UPSTREAM_SHA256,
            "postimage_sha256": contract.POSTIMAGE_SHA256,
            "module_expected_sha256": contract.MODULE_SHA256,
        }
        preimage = {**base, "sha256": contract.UPSTREAM_SHA256,
                    "state": "preimage", "module_state": "missing", "ready": False}
        postimage = {**base, "sha256": contract.POSTIMAGE_SHA256,
                     "state": "postimage", "module_state": "current",
                     "module_sha256": contract.MODULE_SHA256, "ready": True}
        results = [
            SimpleNamespace(returncode=0, stdout=json.dumps(preimage), stderr=""),
            SimpleNamespace(returncode=0,
                            stdout=json.dumps({"result": "applied"}), stderr=""),
            SimpleNamespace(returncode=0, stdout=json.dumps(postimage), stderr=""),
        ]
        with mock.patch.object(launcher.subprocess, "run", side_effect=results) as run:
            record = launcher.systemone_mm_state_patch(apply=True)
        self.assertTrue(record["enabled"])
        self.assertEqual(record["patch_state"], "postimage")
        self.assertEqual(record["installed_module_sha256"], contract.MODULE_SHA256)
        self.assertEqual(run.call_count, 3)

        unknown = {**base, "sha256": "0" * 64, "state": "unknown",
                   "module_state": "missing", "ready": False}
        with mock.patch.object(launcher.subprocess, "run",
                               return_value=SimpleNamespace(
                                   returncode=1, stdout=json.dumps(unknown),
                                   stderr="")):
            with self.assertRaisesRegex(SystemExit, "SHA-256 mismatch"):
                launcher.systemone_mm_state_patch(apply=True)

    def test_encoder_cache_default_and_override_only_change_retention(self):
        baseline = self.launch("--config", "s1")
        with mock.patch.dict(os.environ, {"PYTHONPATH": "/existing/path"}):
            enabled = self.launch("--config", "s1", "--encoder-cache-tokens", "327680")
        self.assertEqual(enabled["argv"], baseline["argv"])
        self.assertEqual(enabled["service_environment"]["CLEF_ENCODER_CACHE_TOKENS"], "327680")
        self.assertEqual(enabled["service_environment"]["PYTHONPATH"],
                         str(Path(__file__).resolve().parent) + os.pathsep + "/existing/path")
        self.assertEqual(enabled["effective"]["max_num_batched_tokens"], 16384)
        self.assertEqual(enabled["effective"]["encoder_cache_tokens"], 327680)
        self.assertEqual(enabled["sources"]["encoder_cache_tokens"], "explicit")
        self.assertEqual(len(enabled["effective"]["encoder_cache_scheduler_sha256"]), 64)

    def test_encoder_cache_keeps_erqa_phase_parameters(self):
        args = ("--max-model-len", "16384", "--max-num-batched-tokens", "16384",
                "--limit-mm-per-prompt", '{"image":16,"video":0}',
                "--mm-processor-kwargs", '{"min_pixels":65536,"max_pixels":1048576}')
        baseline = self.launch(*args)
        enabled = self.launch(*args, "--encoder-cache-tokens", "131072")
        self.assertEqual(enabled["argv"], baseline["argv"])
        for key in ("max_model_len", "max_num_batched_tokens", "limit_mm_per_prompt", "mm_processor_kwargs"):
            self.assertEqual(enabled["effective"][key], baseline["effective"][key])

    def test_encoder_cache_rejects_shrink_and_scheduler_conflicts(self):
        for args in (("--encoder-cache-tokens", "-1"),
                     ("--encoder-cache-tokens", "4095"),
                     ("--encoder-cache-tokens", "327680", "--", "--scheduler-cls", "other.Class"),
                     ("--encoder-cache-tokens", "327680", "--", "--async-scheduling")):
            with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
                self.launch(*args)

    def test_executor_cache_default_and_explicit_override_are_recorded(self):
        self.assertEqual(self.launch("--config", "s1")["service_environment"]
                         ["ACLNN_CACHE_LIMIT"], "1000000")
        os.environ["ACLNN_CACHE_LIMIT"] = "500000"
        self.assertEqual(self.launch("--config", "s1")["service_environment"]
                         ["ACLNN_CACHE_LIMIT"], "500000")

    def test_mlp_nz_follows_enabled_patch_and_respects_ablation(self):
        self.assertNotIn("CLEF_MLP_BF16_NZ", self.launch("--config", "s1")
                         ["service_environment"])
        with mock.patch.object(launcher, "gdn_decision_prefill",
                               return_value={"enabled": True}):
            self.assertEqual(self.launch("--config", "s1")["service_environment"]
                             ["CLEF_MLP_BF16_NZ"], "1")
            os.environ["CLEF_MLP_BF16_NZ"] = "0"
            self.assertEqual(self.launch("--config", "s1")["service_environment"]
                             ["CLEF_MLP_BF16_NZ"], "0")

    def test_chunk128_is_opt_in_and_dry_run_has_no_side_effect(self):
        self.assertFalse(self.launch("--config", "s1")["gdn_decision_prefill"]
                         ["gdn_chunk128_patch"]["requested"])
        with mock.patch.object(launcher, "gdn_decision_prefill",
                               return_value={"enabled": True}):
            os.environ["CLEF_GDN_CHUNK128"] = "0"
            self.assertFalse(self.launch("--config", "s1")["gdn_decision_prefill"]
                             ["gdn_chunk128_patch"]["requested"])
            os.environ["CLEF_GDN_CHUNK128"] = "1"
            with mock.patch.object(launcher.subprocess, "run") as run:
                record = self.launch("--config", "s1")["gdn_decision_prefill"]
                self.assertTrue(record["gdn_chunk128_patch"]["pending_install"])
                run.assert_not_called()
        self.assertFalse(launcher.gdn_chunk128_patch("s1")["requested"])

    def test_chunk128_rejects_incomplete_ready_report(self):
        os.environ["CLEF_GDN_CHUNK128"] = "1"
        incomplete = mock.Mock(returncode=0, stdout=json.dumps({
            "ready": True, "files": {}}))
        with mock.patch.object(launcher.subprocess, "run", return_value=incomplete):
            with self.assertRaisesRegex(SystemExit, "incomplete"):
                launcher.gdn_chunk128_patch("s1", decision_enabled=True)

    def test_custom_default_omits_mm_option(self):
        launch = self.launch("--config", "custom")
        self.assertNotIn("--mm-processor-kwargs", launch["argv"])
        self.assertIsNone(launch["effective"]["mm_processor_kwargs"])

    def test_mm_passthrough_override_in_both_forms(self):
        value = '{"images_kwargs": {"size": {"shortest_edge": 4096, "longest_edge": 65536}}}'
        for extra in (("--mm-processor-kwargs", value), ("--mm-processor-kwargs=" + value,)):
            self.assert_mm(self.launch("--", *extra), value, "passthrough")

    def test_mm_explicit_override_and_passthrough_precedence(self):
        self.assert_mm(self.launch("--mm-processor-kwargs", '{"a":1}'), '{"a":1}', "explicit")
        launch = self.launch("--mm-processor-kwargs", '{"a":1}', "--", "--mm-processor-kwargs", '{"a":2}')
        self.assert_mm(launch, '{"a":2}', "passthrough")

    def test_custom_accepts_mm_passthrough(self):
        self.assert_mm(self.launch("--config", "custom", "--", "--mm-processor-kwargs", "{}"), "{}", "passthrough")

    def test_quantized_launch_keeps_json_and_value_tokens(self):
        mm = '{"images_kwargs": {"size": {"shortest_edge": 4096}}}'
        loader = '{"enable_weights_track": true}'
        launch = self.launch("--", "--mm-processor-kwargs", mm,
                             "--quantization", "ascend", "--model-loader-extra-config", loader)
        self.assertEqual(self.without_scheduler(launch)[-4:], ["--quantization", "ascend", "--model-loader-extra-config", loader])
        self.assert_mm(launch, mm, "passthrough")
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument("--mm-processor-kwargs", type=json.loads)
        parser.add_argument("--quantization")
        parser.add_argument("--model-loader-extra-config", type=json.loads)
        parsed, _ = parser.parse_known_args(launch["argv"][5:])
        self.assertEqual(parsed.quantization, "ascend")
        self.assertEqual(parsed.mm_processor_kwargs, json.loads(mm))
        self.assertEqual(parsed.model_loader_extra_config, json.loads(loader))

    def test_unknown_equals_and_bare_flags_preserve_tokens(self):
        arguments = ["--quantization=ascend", '--model-loader-extra-config={"x": "a b=c"}', "--disable-log-requests"]
        self.assertEqual(self.without_scheduler(self.launch("--", *arguments))[-3:], arguments)

    def test_unknown_separate_values_preserve_spaces_equals_and_empty(self):
        arguments = ["--download-dir", "/tmp/weights a=b", "--revision", ""]
        self.assertEqual(self.without_scheduler(self.launch("--", *arguments))[-4:], arguments)

    def test_existing_owned_override_and_fixed_flags(self):
        launch = self.launch("--", "--port", "19001", "--runner=pooling", "--enforce-eager")
        self.assertEqual(launch["effective"]["port"], 19001)
        self.assertEqual(launch["sources"]["port"], "passthrough")
        self.assertEqual(launch["argv"].count("--runner"), 1)
        self.assertEqual(launch["argv"].count("--enforce-eager"), 1)

    def test_duplicates_missing_values_and_fixed_conflicts_still_fail(self):
        for tokens in (["--mm-processor-kwargs", "{}", "--mm-processor-kwargs={}"],
                       ["--quantization", "ascend", "--quantization=ascend"],
                       ["--mm-processor-kwargs"], ["--runner", "generate"]):
            with self.subTest(tokens=tokens), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    self.launch("--", *tokens)

    def test_tiered_repo_points_to_src_for_runtime_namespace(self):
        launch = self.launch("--config", "s1")
        repo = launch["service_environment"]["CLEF_TIER2_REPO"]
        self.assertEqual(repo, str(ENTRY.parent / "src"))
        self.assertTrue(
            (Path(repo) / "runtime" / "tiered_prefix_cache" / "engine_hooks.py").is_file())

    def test_gdn_decision_prefill_passes_resolved_tree_and_output_backup(self):
        site = self.root / "ascend_site"
        site.mkdir()
        failed = SimpleNamespace(returncode=1, stdout="", stderr="no tree")
        with mock.patch.dict(os.environ, {"CLEF_GDN_DECISION_PREFILL": "1"}), \
                mock.patch.object(launcher, "resolve_package_tree", return_value=site) as tree, \
                mock.patch.object(launcher.subprocess, "run", return_value=failed) as run:
            with self.assertRaises(SystemExit):
                launcher.gdn_decision_prefill(
                    "s1", apply=True, output_dir=self.root / "out")
        tree.assert_called_once_with("vllm_ascend")
        command = run.call_args.args[0]
        self.assertEqual(command[-1], "--check")
        self.assertIn("--tree", command)
        self.assertEqual(command[command.index("--tree") + 1], str(site))
        self.assertIn("--backup-dir", command)
        self.assertEqual(
            command[command.index("--backup-dir") + 1],
            str((self.root / "out").resolve() / "patch-backup" / "gdn-decision"))


class TieredMemoryPolicyTests(unittest.TestCase):
    """Static-RAM clamp for the Tier 2A/2B per-rank pools and ring slots."""

    def test_128_gib_keeps_the_measured_defaults(self):
        policy = launcher.tiered_memory_policy(
            True, ram_limit=128 << 30, rank_env=None, slots_env=None)
        self.assertTrue(policy["enabled"])
        self.assertIsNone(policy["rank_bytes"])
        self.assertIsNone(policy["representation_bytes"])
        self.assertIsNone(policy["ring_slots"])
        self.assertEqual(policy["reason"], "ram>=112GiB")

    def test_96_gib_clamps_to_8_plus_22_gib_and_two_slots(self):
        policy = launcher.tiered_memory_policy(
            True, ram_limit=96 << 30, rank_env=None, slots_env=None)
        self.assertTrue(policy["enabled"])
        self.assertEqual(policy["rank_bytes"], 8 << 30)
        self.assertEqual(policy["representation_bytes"], 22 << 30)
        self.assertEqual(policy["ring_slots"], 2)

    def test_64_gib_clamps_to_4_plus_10_gib_and_two_slots(self):
        policy = launcher.tiered_memory_policy(
            True, ram_limit=64 << 30, rank_env=None, slots_env=None)
        self.assertTrue(policy["enabled"])
        self.assertEqual(policy["rank_bytes"], 4 << 30)
        self.assertEqual(policy["representation_bytes"], 10 << 30)
        self.assertEqual(policy["ring_slots"], 2)

    def test_32_gib_disables_the_cache(self):
        policy = launcher.tiered_memory_policy(
            True, ram_limit=32 << 30, rank_env=None, slots_env=None)
        self.assertFalse(policy["enabled"])
        self.assertIsNone(policy["rank_bytes"])
        self.assertIsNone(policy["representation_bytes"])
        self.assertEqual(policy["reason"], "ram<=48GiB")

    def test_unreadable_limit_keeps_the_defaults(self):
        with mock.patch.object(launcher, "static_ram_limit_bytes",
                               return_value=None):
            policy = launcher.tiered_memory_policy(True)
        self.assertTrue(policy["enabled"])
        self.assertIsNone(policy["rank_bytes"])
        self.assertIsNone(policy["representation_bytes"])
        self.assertEqual(policy["reason"], "ram_limit_unreadable")

    def test_explicit_rank_bytes_skip_the_clamp(self):
        policy = launcher.tiered_memory_policy(
            True, ram_limit=32 << 30, rank_env=str(8 << 30), slots_env=None)
        self.assertTrue(policy["enabled"])
        self.assertEqual(policy["rank_bytes"], 8 << 30)
        self.assertEqual(policy["representation_bytes"], 28 << 30)
        self.assertEqual(policy["reason"], "explicit_rank_bytes")

    def test_explicit_representation_bytes_skip_the_clamp(self):
        policy = launcher.tiered_memory_policy(
            True, ram_limit=32 << 30, rank_env=None,
            slots_env=None, repr_env=str(12 << 30))
        self.assertTrue(policy["enabled"])
        self.assertEqual(policy["rank_bytes"], 10 << 30)
        self.assertEqual(policy["representation_bytes"], 12 << 30)
        self.assertEqual(policy["reason"], "explicit_rank_bytes")

    def test_static_limit_uses_the_tighter_of_memtotal_and_cgroup(self):
        meminfo = io.StringIO("MemTotal:       134217728 kB\n")
        with mock.patch("builtins.open", return_value=meminfo), \
                mock.patch.object(launcher.Path, "read_text",
                                  side_effect=["68719476736", "max"]):
            self.assertEqual(launcher.static_ram_limit_bytes(), 64 << 30)


if __name__ == "__main__":
    unittest.main(verbosity=2)
