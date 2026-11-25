#!/usr/bin/env python3
"""CPU-only generic QKV guard/ABI, dispatch and exact two-preimage rollback tests."""

from __future__ import annotations

import ast
import contextlib
import ctypes
import fcntl
import importlib.util
import io
import json
import os
from pathlib import Path
import tempfile
import types
import unittest
from unittest import mock

HERE = Path(__file__).resolve().parent


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


installer = load("qkv_generic_installer_test", HERE / "apply_qkv_generic_patch.py")
conv = load("qkv_generic_conv_preimage_test", HERE / "apply_conv_qkv_patch.py")
QKV = (HERE / "qkv_norm/qkv_gdn.py").read_bytes()
PREIMAGES = {"qkv": QKV, "conv-qkv": conv.patch_gdn(QKV.decode()).encode()}


class InstallTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.tree = Path(self.tmp.name) / "tree"
        self.target = self.tree / installer.TARGET
        self.target.parent.mkdir(parents=True)
        self.target.write_bytes(QKV)
        self.target.chmod(0o640)
        self.backup_dir = Path(self.tmp.name) / "backups"

    def action(self, flag, *extra):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            result = installer.main([flag, "--tree", str(self.tree),
                                     "--backup-dir", str(self.backup_dir), *extra])
        return result, json.loads(out.getvalue())

    def test_pinned_base_conv_and_both_generated_postimages(self):
        self.assertEqual(installer.PREIMAGE_SHA["qkv"], conv.PREIMAGE_SHA)
        self.assertEqual(installer.PREIMAGE_SHA["conv-qkv"], conv.POSTIMAGE_SHA)
        for variant, original in PREIMAGES.items():
            with self.subTest(variant=variant):
                self.assertEqual(installer.digest(original), installer.PREIMAGE_SHA[variant])
                post = installer.patch_gdn(original.decode())
                self.assertEqual(installer.digest(post.encode()), installer.POSTIMAGE_SHA[variant])
                ast.parse(post)

    def test_conv_composition_tracks_verified_current_conv_contract(self):
        updated = PREIMAGES["conv-qkv"] + b"\n# updated conv guard contract\n"
        expected = installer.digest(updated)
        with mock.patch.dict(installer.PREIMAGE_SHA, {"conv-qkv": expected}), \
                mock.patch.object(installer._CONV_CONTRACT, "patch_gdn", return_value=updated.decode()):
            composed = installer._conv_postimage_sha()
            self.assertEqual(composed, installer.digest(installer.patch_gdn(updated.decode()).encode()))

    def test_conv_contract_hash_drift_is_rejected(self):
        updated = PREIMAGES["conv-qkv"] + b"\n# unpinned conv edit\n"
        with mock.patch.object(installer._CONV_CONTRACT, "patch_gdn", return_value=updated.decode()):
            with self.assertRaisesRegex(ValueError, "current Conv\\+QKV postimage SHA mismatch"):
                installer._conv_postimage_sha()

    def test_apply_check_reapply_and_exact_revert_from_each_preimage(self):
        for variant, original in PREIMAGES.items():
            with self.subTest(variant=variant):
                self.target.write_bytes(original)
                code, report = self.action("--check")
                self.assertEqual((code, report["ready"], report["variant"]), (0, False, variant))
                code, report = self.action("--apply")
                self.assertEqual((code, report["ready"]), (0, True))
                self.assertEqual(report["sha256"], installer.POSTIMAGE_SHA[variant])
                self.assertEqual(self.target.stat().st_mode & 0o777, 0o640)
                backup = installer.backup_path(self.backup_dir, variant)
                self.assertEqual(backup.read_bytes(), original)
                self.assertEqual(backup.stat().st_mode & 0o777, 0o640)
                for flag in ("--check", "--apply"):
                    code, report = self.action(flag)
                    self.assertEqual((code, report["status_after"]), (0, "patched"))
                for _ in range(2):
                    code, report = self.action("--revert")
                    self.assertEqual((code, report["status_after"], report["variant"]),
                                     (0, "preimage", variant))
                    self.assertEqual(self.target.read_bytes(), original)
        self.assertEqual(len(list(self.backup_dir.glob("*.preimage"))), 2)

    def test_unknown_and_missing_sources_are_rejected(self):
        unknown = QKV + b"\n# another user edit\n"
        self.target.write_bytes(unknown)
        for flag in ("--apply", "--check", "--revert"):
            code, report = self.action(flag)
            self.assertEqual(code, 1)
            self.assertIn("refused unknown", report["error"])
            self.assertEqual(self.target.read_bytes(), unknown)
        self.target.unlink()
        code, report = self.action("--apply")
        self.assertEqual(code, 1)
        self.assertIn("refused missing", report["error"])
        self.assertFalse(self.backup_dir.exists())

    def test_symlink_target_is_rejected(self):
        actual = self.target.with_suffix(".actual")
        self.target.rename(actual)
        self.target.symlink_to(actual)
        code, report = self.action("--apply")
        self.assertEqual(code, 1)
        self.assertIn("refused symlink", report["error"])
        self.assertTrue(self.target.is_symlink())
        self.assertEqual(actual.read_bytes(), QKV)

    def test_bad_backup_prevents_apply(self):
        backup = installer.backup_path(self.backup_dir, "qkv")
        backup.parent.mkdir()
        backup.write_bytes(b"wrong preimage")
        code, report = self.action("--apply")
        self.assertEqual(code, 1)
        self.assertIn("backup SHA mismatch", report["error"])
        self.assertEqual(self.target.read_bytes(), QKV)

    def test_symlink_backup_is_rejected(self):
        actual = Path(self.tmp.name) / "actual-backup"
        actual.write_bytes(QKV)
        backup = installer.backup_path(self.backup_dir, "qkv")
        backup.parent.mkdir()
        backup.symlink_to(actual)
        code, report = self.action("--apply")
        self.assertEqual(code, 1)
        self.assertIn("backup SHA mismatch", report["error"])
        self.assertEqual(self.target.read_bytes(), QKV)

    def test_revert_requires_backup_matching_original_variant(self):
        for variant in PREIMAGES:
            self.target.write_bytes(PREIMAGES[variant])
            self.assertEqual(self.action("--apply")[0], 0)
            post = self.target.read_bytes()
            backup = installer.backup_path(self.backup_dir, variant)
            backup.unlink()
            code, report = self.action("--revert")
            self.assertEqual(code, 1)
            self.assertIn("verified matching preimage", report["error"])
            self.assertEqual(self.target.read_bytes(), post)
            backup.write_bytes(PREIMAGES["conv-qkv" if variant == "qkv" else "qkv"])
            code, report = self.action("--revert")
            self.assertEqual(code, 1)
            self.assertEqual(self.target.read_bytes(), post)

    def test_generated_postimage_drift_is_rejected_before_backup(self):
        with mock.patch.dict(installer.POSTIMAGE_SHA, {"qkv": "0" * 64}):
            code, report = self.action("--apply")
        self.assertEqual(code, 1)
        self.assertIn("postimage SHA mismatch", report["error"])
        self.assertEqual(self.target.read_bytes(), QKV)
        self.assertFalse(self.backup_dir.exists())

    def test_tree_lock_is_shared_for_check_exclusive_for_mutations(self):
        for flag, expected in (("--check", fcntl.LOCK_SH), ("--apply", fcntl.LOCK_EX),
                               ("--revert", fcntl.LOCK_EX)):
            with mock.patch.object(installer.fcntl, "flock", wraps=fcntl.flock) as lock:
                self.assertEqual(self.action(flag)[0], 0)
            self.assertEqual(lock.call_args.args[1], expected)

    def test_manifest_can_record_success_and_cannot_overwrite_sources(self):
        manifest = Path(self.tmp.name) / "manifest.json"
        code, report = self.action("--check", "--manifest", str(manifest))
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(manifest.read_text()), report)
        for protected in (self.target, installer.backup_path(self.backup_dir, "conv-qkv")):
            code, report = self.action("--apply", "--manifest", str(protected))
            self.assertEqual(code, 1)
            self.assertIn("manifest must be separate", report["error"])
            self.assertEqual(self.target.read_bytes(), QKV)
        self.target.write_bytes(b"unrecognized")
        code, report = self.action("--check", "--manifest", str(self.target))
        self.assertEqual(code, 1)
        self.assertEqual(self.target.read_bytes(), b"unrecognized")

    def test_core_attention_and_conv_helpers_are_unchanged(self):
        for original in PREIMAGES.values():
            pre = ast.parse(original)
            post = ast.parse(installer.patch_gdn(original.decode()))
            names = ["_forward_core", "_run_conv_qkv"]
            for name in names:
                before = [n for n in ast.walk(pre) if isinstance(n, ast.FunctionDef) and n.name == name]
                after = [n for n in ast.walk(post) if isinstance(n, ast.FunctionDef) and n.name == name]
                self.assertEqual([ast.dump(n) for n in before], [ast.dump(n) for n in after])


class FakeTensor:
    def __init__(self, shape, dtype="bf16", device=None, contiguous=True):
        self.shape = tuple(shape)
        self.dtype = dtype
        self.device = device or types.SimpleNamespace(type="npu", index=0)
        self.contiguous = contiguous

    def dim(self):
        return len(self.shape)

    def is_contiguous(self):
        return self.contiguous


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        torch = types.SimpleNamespace(
            bfloat16="bf16", empty=lambda shape, dtype, device: FakeTensor(shape, dtype, device),
            empty_like=lambda x: FakeTensor(x.shape, x.dtype, x.device))
        self.namespace = {"os": os, "torch": torch}
        exec(installer.HELPER, self.namespace)
        post = ast.parse(installer.patch_gdn(QKV.decode()))
        norm = next(n for n in post.body if isinstance(n, ast.FunctionDef) and n.name == "_run_qkv_norm")
        exec(compile(ast.Module(body=[norm], type_ignores=[]), "<generic_dispatch>", "exec"), self.namespace)
        self.namespace.update(_QKV_NORM_RUNTIME=None, _QKV_NORM_FAILED=False)
        self.norm = self.namespace["_run_qkv_norm"]
        self.x = FakeTensor((40, 307, 128))
        self.environment = mock.patch.dict(os.environ, {"CLEF_QKV_GENERIC": "1",
                                                        "CLEF_QKV_GENERIC_VENDOR": "/candidate/vendor"}, clear=True)
        self.environment.start()
        self.addCleanup(self.environment.stop)

    def call(self, x=None, hk=8, hv=24, dim=128):
        return self.norm(self.x if x is None else x, hk, hv, dim)

    def fake_runtime(self, failure=None):
        self.calls = []
        runtime = types.SimpleNamespace(_libraries=[])
        def call_aclnn(name, build, outputs, get_workspace_argtypes):
            if failure:
                raise RuntimeError(failure)
            args = build(types.SimpleNamespace(tensor=lambda tensor, name: (tensor, name)))
            self.calls.append((name, args, outputs, get_workspace_argtypes))
        api = types.SimpleNamespace(runtime=lambda: runtime, call_aclnn=call_aclnn)
        ascendc = types.ModuleType("fla_npu.ops.ascendc")
        ascendc._runtime = api
        modules = {"fla_npu": types.ModuleType("fla_npu"),
                   "fla_npu.ops": types.ModuleType("fla_npu.ops"),
                   "fla_npu.ops.ascendc": ascendc}
        return modules, runtime

    def test_disabled_candidate_keeps_legacy_path(self):
        os.environ["CLEF_QKV_GENERIC"] = "0"
        self.namespace["_run_qkv_generic"] = mock.Mock(side_effect=AssertionError("candidate called"))
        self.assertIsNone(self.call())  # Legacy vendor absent, original guard returns None.
        self.namespace["_run_qkv_generic"].assert_not_called()

    def test_generic_dispatch_precedes_legacy_failure_flag(self):
        sentinel = object()
        self.namespace["_QKV_NORM_FAILED"] = True
        self.namespace["_run_qkv_generic"] = mock.Mock(return_value=sentinel)
        self.assertIs(self.call(), sentinel)
        self.namespace["_run_qkv_generic"].assert_called_once_with(self.x, 8, 24, 128)

    def test_unsupported_shapes_return_none_without_poisoning_runtime(self):
        bad_inputs = [FakeTensor((40, 307)), FakeTensor((40, 307, 64)),
                      FakeTensor((80, 307, 128)), FakeTensor((40, 0, 128)),
                      FakeTensor((40, 307, 128), dtype="fp16"),
                      FakeTensor((40, 307, 128), contiguous=False),
                      FakeTensor((40, 307, 128), device=types.SimpleNamespace(type="cpu"))]
        with mock.patch.object(ctypes, "CDLL") as loader:
            for x in bad_inputs:
                self.assertIsNone(self.call(x=x), x.shape)
            for dims in ((7, 21, 128), (8, 32, 128), (8, 24, 64)):
                self.assertIsNone(self.call(hk=dims[0], hv=dims[1], dim=dims[2]))
        loader.assert_not_called()
        self.assertIsNone(self.namespace["_QKV_GENERIC_RUNTIME"])
        self.assertFalse(self.namespace["_QKV_NORM_FAILED"])

    def test_vendor_required_and_invalid_tile_errors_are_visible(self):
        vendor = os.environ.pop("CLEF_QKV_GENERIC_VENDOR")
        with self.assertRaisesRegex(RuntimeError, "requires CLEF_QKV_GENERIC_VENDOR"):
            self.call()
        os.environ["CLEF_QKV_GENERIC_VENDOR"] = vendor
        for value in ("128", "bad"):
            os.environ["CLEF_QKV_GENERIC_TILE"] = value
            with self.assertRaisesRegex(RuntimeError, "16, 32, or 64"):
                self.call()

    def test_tp1_tp2_abi_default_tile_loader_cache_and_first_success_log(self):
        modules, runtime = self.fake_runtime()
        output = io.StringIO()
        with mock.patch.dict("sys.modules", modules), mock.patch.object(ctypes, "CDLL") as loader:
            with contextlib.redirect_stdout(output):
                tp2 = self.call()
                os.environ["CLEF_QKV_GENERIC_TILE"] = "64"
                tp1 = self.call(x=FakeTensor((80, 967, 128)), hk=16, hv=48)
        loader.assert_called_once()
        self.assertEqual(len(runtime._libraries), 1)
        self.assertIs(self.namespace["_QKV_GENERIC_LIBRARY"], runtime._libraries[0])
        self.assertEqual([x.shape for x in tp2], [(1, 307, 8, 128), (1, 307, 8, 128), (1, 307, 24, 128)])
        self.assertEqual([x.shape for x in tp1], [(1, 967, 16, 128), (1, 967, 16, 128), (1, 967, 48, 128)])
        self.assertEqual(output.getvalue().count("clef-qkv-generic"), 1)
        for index, expected_tile in enumerate((32, 64)):
            name, args, outputs, argtypes = self.calls[index]
            self.assertEqual(name, "aclnnClefGdnQkvGeneric")
            self.assertEqual(args[0][1], "x")
            self.assertEqual(args[1].value, 1e-6)
            self.assertEqual(args[2].value, expected_tile)
            self.assertEqual([entry[1] for entry in args[3:]], ["qOut", "kOut", "vOut"])
            self.assertEqual(len(argtypes), 8)

    def test_loader_error_propagates_outside_legacy_except(self):
        modules, runtime = self.fake_runtime()
        with mock.patch.dict("sys.modules", modules), mock.patch.object(ctypes, "CDLL", side_effect=OSError("missing vendor")):
            with self.assertRaisesRegex(OSError, "missing vendor"):
                self.call()
        self.assertFalse(self.namespace["_QKV_NORM_FAILED"])
        self.assertIsNone(self.namespace["_QKV_GENERIC_RUNTIME"])
        self.assertEqual(runtime._libraries, [])

    def test_kernel_error_propagates_and_success_log_is_not_emitted(self):
        modules, _ = self.fake_runtime(failure="candidate kernel failure")
        output = io.StringIO()
        with mock.patch.dict("sys.modules", modules), mock.patch.object(ctypes, "CDLL"):
            with contextlib.redirect_stdout(output):
                with self.assertRaisesRegex(RuntimeError, "candidate kernel failure"):
                    self.call()
        self.assertFalse(self.namespace["_QKV_NORM_FAILED"])
        self.assertFalse(self.namespace["_QKV_GENERIC_LOGGED"])
        self.assertEqual(output.getvalue(), "")

    def test_changing_vendor_in_worker_is_rejected(self):
        modules, _ = self.fake_runtime()
        with mock.patch.dict("sys.modules", modules), mock.patch.object(ctypes, "CDLL"):
            with contextlib.redirect_stdout(io.StringIO()):
                self.call()
            os.environ["CLEF_QKV_GENERIC_VENDOR"] = "/another/vendor"
            with self.assertRaisesRegex(RuntimeError, "changed within a running worker"):
                self.call()


if __name__ == "__main__":
    unittest.main()
