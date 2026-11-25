"""Real-installer tests for the aa3e official entry (temp tree only).

Every test uses a mkdtemp tree with a fake ``vllm`` package layout; the real
audited installers from ``runtime/`` run against it.  No production tree, NPU,
service or network is touched.  The 197863d5 start needs 32's read-only byte
copy via ``CLEF_ORIGINAL_JOINT_SCHEMA`` (skipped only when not provided).
"""
import argparse
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import tempfile
import types
import unittest
from unittest import mock

REPO = Path(__file__).resolve().parents[2]
ENTRY = REPO / "inference.py"
SPEC = importlib.util.spec_from_file_location("clef_inference_joint_test", ENTRY)
launcher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(launcher)

F810 = REPO / "src" / "runtime" / "joint_single_field" / "clef_joint_schema.py"
AA3E = REPO / "src" / "runtime" / "joint_fused_attention" / "clef_joint_schema.py"
ORIGINAL = os.environ.get("CLEF_ORIGINAL_JOINT_SCHEMA")
TARGET_REL = "vllm/model_executor/models/clef_joint_schema.py"


class JointFusedEntryTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="clef-joint-entry-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.tree = self.root / "site"
        self.target = self.tree / TARGET_REL
        self.target.parent.mkdir(parents=True)
        (self.tree / "vllm" / "__init__.py").write_text("")
        self.output = self.root / "output"
        self.probe = {"tree": str(self.tree), "target": str(self.target), "probe_rc": 0,
                      "probe_stdout": str(self.tree / "vllm/__init__.py"),
                      "probe_stderr": ""}
        environment = mock.patch.dict(os.environ, {}, clear=False)
        environment.start()
        self.addCleanup(environment.stop)
        os.environ.pop(launcher.JOINT_FLAG_ENV, None)
        os.environ.pop(launcher.JOINT_ALLOW_ENV, None)

    def use_probe(self):
        patcher = mock.patch.object(launcher, "resolve_vllm_tree",
                                    side_effect=lambda: dict(self.probe))
        patcher.start()
        self.addCleanup(patcher.stop)

    def start(self, kind):
        if kind == "f810":
            shutil.copy2(F810, self.target)
        elif kind == "aa3e":
            shutil.copy2(AA3E, self.target)
        elif kind == "original":
            if not ORIGINAL:
                self.skipTest("CLEF_ORIGINAL_JOINT_SCHEMA (32 copy) not provided")
            shutil.copy2(ORIGINAL, self.target)
        elif kind == "unknown":
            self.target.write_text("unknown joint schema\n")
        else:
            raise AssertionError(kind)
        return launcher.sha256_file(self.target)

    def read_record(self):
        return json.loads((self.output / "joint-patch-record.json").read_text())

    def backup(self, name):
        return self.output / "patch-backup" / name / "clef_joint_schema.py.preimage"

    def test_f810_start_applies_fused_once(self):
        self.use_probe()
        self.assertEqual(self.start("f810"), launcher.JOINT_SINGLE_FIELD_SHA)
        record = launcher.fused_joint_patch("s1", self.output)
        self.assertTrue(record["ok"])
        self.assertEqual(record["branch"], "f810->fused")
        self.assertEqual(record["guard_action"], "noop")
        self.assertEqual(record["fused_action"], "apply")
        self.assertEqual(launcher.sha256_file(self.target), launcher.JOINT_FUSED_SHA)
        self.assertEqual(launcher.sha256_file(self.backup("joint-fused")),
                         launcher.JOINT_SINGLE_FIELD_SHA)
        self.assertFalse(self.backup("joint-single-field").exists())
        self.assertEqual(os.environ[launcher.JOINT_FLAG_ENV], "1")
        self.assertEqual(os.environ[launcher.JOINT_ALLOW_ENV], launcher.JOINT_FUSED_SHA)
        disk = self.read_record()
        self.assertEqual(disk["branch"], "f810->fused")
        self.assertTrue(disk["ok"])
        self.assertEqual(disk["guard_check"]["report"]["status_before"], "single-field")
        self.assertEqual(disk["fused_check"]["report"]["state_before"], "preimage")

    def test_original_start_runs_guard_then_fused(self):
        self.use_probe()
        self.assertEqual(self.start("original"), launcher.JOINT_ORIGINAL_SHA)
        record = launcher.fused_joint_patch("s1", self.output)
        self.assertEqual(record["branch"], "original->f810->fused")
        self.assertEqual(record["guard_action"], "apply")
        self.assertEqual(record["fused_action"], "apply")
        self.assertEqual(launcher.sha256_file(self.backup("joint-single-field")),
                         launcher.JOINT_ORIGINAL_SHA)
        self.assertEqual(launcher.sha256_file(self.backup("joint-fused")),
                         launcher.JOINT_SINGLE_FIELD_SHA)
        self.assertEqual(launcher.sha256_file(self.target), launcher.JOINT_FUSED_SHA)
        disk = self.read_record()
        self.assertEqual(disk["guard_apply"]["report"]["status_after"], "single-field")
        self.assertEqual(disk["fused_apply"]["report"]["state_after"], "postimage")

    def test_aa3e_start_is_noop_and_repeatable(self):
        self.use_probe()
        self.assertEqual(self.start("aa3e"), launcher.JOINT_FUSED_SHA)
        first = launcher.fused_joint_patch("s1", self.output)
        second = launcher.fused_joint_patch("s1", self.output)
        for record in (first, second):
            self.assertTrue(record["ok"])
            self.assertEqual(record["branch"], "aa3e-noop")
            self.assertEqual(record["guard_action"], "noop")
            self.assertEqual(record["fused_action"], "noop")
        self.assertEqual(launcher.sha256_file(self.target), launcher.JOINT_FUSED_SHA)
        self.assertFalse(self.backup("joint-single-field").exists())
        self.assertFalse(self.backup("joint-fused").exists())
        disk = self.read_record()
        self.assertEqual(disk["target_sha256_before"], launcher.JOINT_FUSED_SHA)
        self.assertEqual(disk["target_sha256_after"], launcher.JOINT_FUSED_SHA)
        self.assertEqual(disk["default_config"], launcher.DEFAULT_CONFIG)

    def test_unknown_tree_fails_closed_with_record(self):
        self.use_probe()
        before = self.start("unknown")
        with self.assertRaises(SystemExit):
            launcher.fused_joint_patch("s1", self.output)
        self.assertEqual(launcher.sha256_file(self.target), before)
        disk = self.read_record()
        self.assertFalse(disk["ok"])
        self.assertIn("unknown joint schema", disk["error"])
        self.assertEqual(disk["target_sha256_before"], before)

    def test_non_fused_config_pops_candidate_env(self):
        os.environ[launcher.JOINT_FLAG_ENV] = "1"
        os.environ[launcher.JOINT_ALLOW_ENV] = launcher.JOINT_FUSED_SHA
        record = launcher.fused_joint_patch("custom", self.output)
        self.assertFalse(record["requested"])
        self.assertFalse(record["enabled"])
        self.assertNotIn(launcher.JOINT_FLAG_ENV, os.environ)
        self.assertNotIn(launcher.JOINT_ALLOW_ENV, os.environ)
        self.assertFalse(self.output.exists())

    def test_dry_run_is_pending_without_tree_access(self):
        with mock.patch.object(launcher, "resolve_vllm_tree",
                               side_effect=AssertionError("must not resolve in dry-run")):
            record = launcher.fused_joint_patch("s1", self.output, apply=False)
        self.assertTrue(record["requested"])
        self.assertTrue(record["pending_install"])
        self.assertFalse(self.output.exists())
        self.assertEqual(os.environ[launcher.JOINT_FLAG_ENV], "1")

    def test_resolve_tree_validates_probe_and_layout(self):
        probe = types.SimpleNamespace(returncode=0,
                                      stdout=str(self.tree / "vllm/__init__.py") + "\n",
                                      stderr="")
        self.target.write_text("x")
        with mock.patch.object(launcher.subprocess, "run", return_value=probe):
            resolved = launcher.resolve_vllm_tree()
        self.assertEqual(resolved["tree"], str(self.tree))
        self.assertEqual(resolved["target"], str(self.target))
        for result in (types.SimpleNamespace(returncode=1, stdout=probe.stdout, stderr=""),
                       types.SimpleNamespace(returncode=0, stdout="", stderr=""),
                       types.SimpleNamespace(returncode=0,
                                             stdout=str(self.root / "x.py"), stderr="")):
            with mock.patch.object(launcher.subprocess, "run", return_value=result):
                with self.assertRaises(ValueError):
                    launcher.resolve_vllm_tree()
        self.target.unlink()
        with mock.patch.object(launcher.subprocess, "run", return_value=probe):
            with self.assertRaises(ValueError):
                launcher.resolve_vllm_tree()
        self.target.write_text("x")
        self.target.chmod(0o444)
        with mock.patch.object(launcher.subprocess, "run", return_value=probe):
            with self.assertRaises(ValueError):
                launcher.resolve_vllm_tree()

    def test_failure_prevents_service_launch(self):
        argv = [str(ENTRY), "--model-dir", str(self.root / "model"),
                "--output-dir", str(self.output)]
        with mock.patch.object(launcher.sys, "argv", argv), \
                mock.patch.object(launcher, "build_launch",
                                  side_effect=SystemExit("patch setup failed")), \
                mock.patch.object(launcher.os, "execv") as execv:
            with self.assertRaises(SystemExit):
                launcher.main()
        execv.assert_not_called()

    def dry_run(self, *arguments):
        output = io.StringIO()
        argv = [str(ENTRY), "--model-dir", str(self.root / "model"),
                "--output-dir", str(self.output), "--dry-run", *arguments]
        with mock.patch.object(launcher.sys, "argv", argv), \
                mock.patch.object(launcher, "software_provenance", return_value={}), \
                mock.patch.object(launcher, "resolve_schema_cache",
                                  return_value=(False, 0, None)), \
                contextlib.redirect_stdout(output):
            launcher.main()
        return json.loads(output.getvalue())

    def test_default_s1_dry_run_selects_fused_without_installing(self):
        with mock.patch.dict(os.environ, {"CLEF_GDN_DECISION_PREFILL": "0"}, clear=True):
            launch = self.dry_run()
        self.assertEqual(launch["config"], launcher.DEFAULT_CONFIG)
        joint = launch["joint_patch"]
        self.assertTrue(joint["requested"])
        self.assertTrue(joint["pending_install"])
        self.assertEqual(joint["mode"], "joint-fused-attention")
        self.assertEqual(launch["service_environment"][launcher.JOINT_FLAG_ENV], "1")
        self.assertEqual(launch["service_environment"][launcher.JOINT_ALLOW_ENV],
                         launcher.JOINT_FUSED_SHA)
        self.assertEqual(launch["service_environment"]["CLEF_JOINT_SINGLE_FIELD"], "1")
        self.assertFalse(self.output.exists())

    def test_non_fused_config_dry_run_pops_candidate_env(self):
        os.environ[launcher.JOINT_FLAG_ENV] = "1"
        os.environ[launcher.JOINT_ALLOW_ENV] = launcher.JOINT_FUSED_SHA
        # The single-field arm has no preset any more; enable it on the
        # conservative config to exercise the non-fused branch.
        with mock.patch.dict(os.environ, {"CLEF_GDN_DECISION_PREFILL": "0"}), \
                mock.patch.dict(launcher.CONFIGS["custom"], {"joint_single_field": True}):
            launch = self.dry_run("--config", "custom")
            self.assertNotIn(launcher.JOINT_FLAG_ENV, launch["service_environment"])
            self.assertNotIn(launcher.JOINT_ALLOW_ENV, launch["service_environment"])
            self.assertNotIn(launcher.JOINT_FLAG_ENV, os.environ)
            self.assertNotIn(launcher.JOINT_ALLOW_ENV, os.environ)
        self.assertEqual(launch["joint_patch"]["requested"], True)
        self.assertTrue(launch["joint_patch"]["pending_install"])

    def test_assets_are_bundled_without_scratch_dependency(self):
        for path in (REPO / "src/runtime/apply_joint_fused_attention.py",
                     REPO / "src/runtime/apply_joint_single_field_patch_fused.py",
                     REPO / "src/runtime/joint_fused_attention/clef_joint_schema.py",
                     REPO / "src/runtime/apply_joint_service_check.py",
                     REPO / "src/runtime/joint_service_check/apply_joint_single_field_patch.py"):
            self.assertTrue(path.is_file(), path)
        self.assertEqual(launcher.sha256_file(REPO / "src/runtime/apply_joint_fused_attention.py"),
                         "2a659d07154768a5e2a4a51001aff05aa966359bdf1deef3b6b1d2f115e7e7c5")
        self.assertEqual(launcher.sha256_file(REPO / "src/runtime/apply_joint_single_field_patch_fused.py"),
                         launcher.JOINT_GUARD_POST_SHA)
        self.assertEqual(launcher.sha256_file(REPO / "src/runtime/joint_fused_attention/clef_joint_schema.py"),
                         launcher.JOINT_FUSED_SHA)
        self.assertEqual(launcher.sha256_file(REPO / "src/runtime/apply_joint_service_check.py"),
                         "1ed35f0a24f8b23d8ac099992d703aca20d7f1460c89ff02c30635dfde6a219c")
        text = ENTRY.read_text()
        self.assertNotIn("scratch-24", text)
        self.assertNotIn("joint-fused-source-control", text)


if __name__ == "__main__":
    unittest.main()
