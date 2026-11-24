#!/usr/bin/env python3
"""CPU-only guard, integration and reversible-install tests for varlen Conv+QKV."""
from __future__ import annotations

import ast
import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest

HERE = Path(__file__).resolve().parent


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


v4 = load("conv_qkv_installer", HERE / "apply_conv_qkv_patch.py")
varlen = load("conv_qkv_varlen_installer", HERE / "apply_conv_qkv_varlen_patch.py")
V4_PREIMAGE = (HERE / "qkv_norm/qkv_gdn.py").read_bytes()
PREIMAGE = v4.patch_gdn(V4_PREIMAGE.decode()).encode()


class InstallTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.tree = Path(self.tmp.name) / "tree"
        self.target = self.tree / varlen.TARGET
        self.target.parent.mkdir(parents=True)
        self.target.write_bytes(PREIMAGE)
        self.target.chmod(0o640)
        self.backup = Path(self.tmp.name) / "backup"

    def tearDown(self):
        self.tmp.cleanup()

    def run_action(self, action):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            code = varlen.main([action, "--tree", str(self.tree),
                                "--backup-dir", str(self.backup)])
        return code, json.loads(output.getvalue())

    def test_preimage_and_postimage_are_pinned(self):
        self.assertEqual(hashlib.sha256(PREIMAGE).hexdigest(), varlen.PREIMAGE_SHA)
        post = varlen.patch_gdn(PREIMAGE.decode()).encode()
        self.assertEqual(hashlib.sha256(post).hexdigest(), varlen.POSTIMAGE_SHA)
        ast.parse(post)

    def test_apply_check_reapply_and_revert_are_exact(self):
        code, report = self.run_action("--check")
        self.assertEqual(code, 0)
        self.assertFalse(report["ready"])
        code, report = self.run_action("--apply")
        self.assertEqual(code, 0)
        self.assertTrue(report["ready"])
        self.assertEqual(report["sha256"], varlen.POSTIMAGE_SHA)
        self.assertEqual(self.target.stat().st_mode & 0o777, 0o640)
        for action in ("--check", "--apply"):
            code, report = self.run_action(action)
            self.assertEqual(code, 0)
            self.assertEqual(report["status_after"], "patched")
        self.assertEqual((self.backup / "gdn.py.preimage").read_bytes(), PREIMAGE)
        for _ in range(2):
            code, report = self.run_action("--revert")
            self.assertEqual(code, 0)
            self.assertEqual(report["status_after"], "preimage")
            self.assertEqual(self.target.read_bytes(), PREIMAGE)

    def test_unknown_source_is_rejected_for_every_action(self):
        unknown = PREIMAGE + b"\n# another local edit\n"
        self.target.write_bytes(unknown)
        for action in ("--check", "--apply", "--revert"):
            code, report = self.run_action(action)
            self.assertEqual(code, 1)
            self.assertIn("refused unknown", report["error"])
            self.assertEqual(self.target.read_bytes(), unknown)
        self.assertFalse(self.backup.exists())

    def test_v4_installer_accepts_the_varlen_postimage(self):
        post = varlen.patch_gdn(PREIMAGE.decode()).encode()
        self.target.write_bytes(post)
        self.assertEqual(v4.source_status(self.target), "patched")
        # The v4 --check path must report ready on the stacked postimage.
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            code = v4.main(["--check", "--tree", str(self.tree),
                            "--backup-dir", str(Path(self.tmp.name) / "v4backup")])
        report = json.loads(output.getvalue())
        self.assertEqual(code, 0)
        self.assertTrue(report["ready"])
        self.assertEqual(report["sha256"], varlen.POSTIMAGE_SHA)


if __name__ == "__main__":
    unittest.main()
