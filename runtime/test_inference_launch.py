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
import unittest
from unittest import mock


ENTRY = Path(__file__).resolve().parent.parent / "Clef-27B" / "inference.py"
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

    def assert_mm(self, launch, value, source):
        command = launch["argv"]
        self.assertEqual(command.count("--mm-processor-kwargs"), 1)
        self.assertEqual(command[command.index("--mm-processor-kwargs") + 1], value)
        self.assertEqual(launch["effective"]["mm_processor_kwargs"], value)
        self.assertEqual(launch["sources"]["mm_processor_kwargs"], source)

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

    def test_s1_and_s2_defaults_emit_one_mm_option(self):
        for config in ("s1", "s2"):
            launch = self.launch("--config", config)
            self.assert_mm(launch, launcher.MM_PROCESSOR_KWARGS, f"config:{config}")

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
        self.assertEqual(launch["argv"][-4:], ["--quantization", "ascend", "--model-loader-extra-config", loader])
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
        self.assertEqual(self.launch("--", *arguments)["argv"][-3:], arguments)

    def test_unknown_separate_values_preserve_spaces_equals_and_empty(self):
        arguments = ["--download-dir", "/tmp/weights a=b", "--revision", ""]
        self.assertEqual(self.launch("--", *arguments)["argv"][-4:], arguments)

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


if __name__ == "__main__":
    unittest.main(verbosity=2)
