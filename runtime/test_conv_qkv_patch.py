#!/usr/bin/env python3
"""CPU-only guard, integration and reversible-install tests for Conv+QKV."""
from __future__ import annotations

import ast
import contextlib
import ctypes
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
SPEC = importlib.util.spec_from_file_location("conv_qkv_installer_test", HERE / "apply_conv_qkv_patch.py")
installer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(installer)
PREIMAGE = (HERE / "qkv_norm/qkv_gdn.py").read_bytes()


class InstallTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.tree = Path(self.tmp.name) / "tree"
        self.target = self.tree / installer.TARGET
        self.target.parent.mkdir(parents=True)
        self.target.write_bytes(PREIMAGE)
        self.target.chmod(0o640)
        self.backup = Path(self.tmp.name) / "backup"

    def tearDown(self):
        self.tmp.cleanup()

    def run_action(self, action):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            code = installer.main([action, "--tree", str(self.tree),
                                   "--backup-dir", str(self.backup)])
        return code, json.loads(output.getvalue())

    def test_bundled_preimage_and_generated_postimage_are_pinned(self):
        self.assertEqual(installer.digest(PREIMAGE), installer.PREIMAGE_SHA)
        post = installer.patch_gdn(PREIMAGE.decode()).encode()
        self.assertEqual(installer.digest(post), installer.POSTIMAGE_SHA)
        ast.parse(post)

    def test_apply_check_reapply_and_revert_are_exact(self):
        code, report = self.run_action("--check")
        self.assertEqual(code, 0)
        self.assertFalse(report["ready"])
        code, report = self.run_action("--apply")
        self.assertEqual(code, 0)
        self.assertTrue(report["ready"])
        self.assertEqual(report["sha256"], installer.POSTIMAGE_SHA)
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

    def test_missing_source_is_rejected(self):
        self.target.unlink()
        code, report = self.run_action("--apply")
        self.assertEqual(code, 1)
        self.assertIn("refused missing", report["error"])
        self.assertFalse(self.target.exists())

    def test_bad_backup_rejects_apply_without_writing_target(self):
        self.backup.mkdir()
        (self.backup / "gdn.py.preimage").write_bytes(b"unrelated source")
        code, report = self.run_action("--apply")
        self.assertEqual(code, 1)
        self.assertIn("backup SHA mismatch", report["error"])
        self.assertEqual(self.target.read_bytes(), PREIMAGE)

    def test_revert_requires_verified_backup(self):
        self.assertEqual(self.run_action("--apply")[0], 0)
        post = self.target.read_bytes()
        (self.backup / "gdn.py.preimage").unlink()
        code, report = self.run_action("--revert")
        self.assertEqual(code, 1)
        self.assertIn("verified preimage backup", report["error"])
        self.assertEqual(self.target.read_bytes(), post)

    def test_generated_sha_mismatch_rejects_before_backup(self):
        with mock.patch.object(installer, "POSTIMAGE_SHA", "0" * 64):
            code, report = self.run_action("--apply")
        self.assertEqual(code, 1)
        self.assertIn("postimage SHA mismatch", report["error"])
        self.assertEqual(self.target.read_bytes(), PREIMAGE)
        self.assertFalse(self.backup.exists())

    def test_fused_tuple_precedes_head_first_split(self):
        post = installer.patch_gdn(PREIMAGE.decode())
        module = ast.parse(post)
        method = next(node for node in ast.walk(module)
                      if isinstance(node, ast.FunctionDef) and node.name == "_forward_core")
        init = next(node for node in method.body if isinstance(node, ast.Assign)
                    and any(isinstance(target, ast.Name) and target.id == "conv_qkv"
                            for target in node.targets))
        self.assertIsNone(init.value.value)
        branch = next(node for node in method.body if isinstance(node, ast.If)
                      and ast.unparse(node.test) == "conv_qkv is not None")
        self.assertEqual(ast.unparse(branch.body[0]),
                         "query_non_spec, key_non_spec, value_non_spec = conv_qkv")
        self.assertEqual(ast.unparse(branch.body[1]), "qkv_pre_normalized = True")
        self.assertIsInstance(branch.orelse[0], ast.If)
        self.assertIn("_run_qkv_norm", ast.unparse(branch.orelse[0]))
        # The original zero-history convolution remains exactly one fallback
        # call, nested beneath the new helper's None result.
        conv_branch = next(node for node in ast.walk(method) if isinstance(node, ast.If)
                           and ast.unparse(node.test) == "conv_qkv is None")
        self.assertIn("conv_states=None", ast.unparse(conv_branch.body[0]))


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


class GuardTests(unittest.TestCase):
    def setUp(self):
        torch = types.SimpleNamespace(
            bfloat16="bf16", empty=lambda shape, dtype, device: FakeTensor(shape, dtype, device),
            empty_like=lambda x: FakeTensor(x.shape, x.dtype, x.device))
        self.namespace = {"os": os, "torch": torch}
        exec(installer.HELPER, self.namespace)
        self.helper = self.namespace["_run_conv_qkv"]
        self.x = FakeTensor((307, 5120))
        self.weight = FakeTensor((4, 5120), device=self.x.device)
        self.runtime = types.SimpleNamespace(skip_conv_state=True, preprocess_fusion=True, bsnd=True)
        self.metadata = types.SimpleNamespace(num_prefills=1, num_decodes=0,
                                              num_spec_decodes=0, spec_sequence_masks=None)
        self.kwargs = dict(runtime=self.runtime, attn_metadata=self.metadata,
                           conv_head_num=40, tp_size=2, num_k_heads=16, num_v_heads=48,
                           head_k_dim=128, head_v_dim=128, bias=None, activation="silu")
        self.env = mock.patch.dict(os.environ, {"CLEF_GDN_CONV_QKV": "1",
                                                "CLEF_CONV_QKV_VENDOR": "/candidate/vendor"}, clear=True)
        self.env.start()
        self.addCleanup(self.env.stop)

    def call(self, x=None, weight=None, **kwargs):
        return self.helper(x or self.x, weight or self.weight, **(self.kwargs | kwargs))

    def test_disabled_switch_is_noop(self):
        for value in ("0", "false", "", "true"):
            with self.subTest(value=value):
                os.environ["CLEF_GDN_CONV_QKV"] = value
                self.assertIsNone(self.call())

    def test_runtime_switches_and_metadata_are_guarded(self):
        self.assertIsNone(self.call(runtime=None))
        for name in ("skip_conv_state", "preprocess_fusion", "bsnd"):
            with self.subTest(runtime=name):
                state = vars(self.runtime) | {name: False}
                self.assertIsNone(self.call(runtime=types.SimpleNamespace(**state)))
        for name, value in (("num_prefills", 0), ("num_prefills", 2),
                            ("num_decodes", 1), ("num_spec_decodes", 1),
                            ("spec_sequence_masks", object())):
            with self.subTest(metadata=name, value=value):
                state = vars(self.metadata) | {name: value}
                self.assertIsNone(self.call(attn_metadata=types.SimpleNamespace(**state)))

    def test_layer_contract_is_guarded(self):
        for name, value in (("tp_size", 1), ("conv_head_num", 0), ("conv_head_num", 80),
                            ("num_k_heads", 8), ("num_v_heads", 24), ("head_k_dim", 64),
                            ("head_v_dim", 64), ("bias", object()), ("activation", False),
                            ("activation", "relu")):
            with self.subTest(name=name):
                self.assertIsNone(self.call(**{name: value}))

    def test_tensor_shape_dtype_device_and_contiguity_are_guarded(self):
        for tensor in (FakeTensor((307, 5121)), FakeTensor((307, 5120), dtype="fp16"),
                       FakeTensor((1, 307, 5120)), FakeTensor((307, 5120), contiguous=False),
                       FakeTensor((307, 5120), device=types.SimpleNamespace(type="cpu", index=0))):
            with self.subTest(x=vars(tensor)):
                self.assertIsNone(self.call(x=tensor))
        for tensor in (FakeTensor((5120, 4)), FakeTensor((3, 5120)),
                       FakeTensor((4, 5120), dtype="fp16"),
                       FakeTensor((4, 5120), contiguous=False),
                       FakeTensor((4, 5120), device=types.SimpleNamespace(type="npu", index=1))):
            with self.subTest(weight=vars(tensor)):
                self.assertIsNone(self.call(weight=tensor))

    def test_zero_and_over_cap_tokens_fall_back(self):
        for tokens in (0, 1201, 1595):
            self.assertIsNone(self.call(x=FakeTensor((tokens, 5120))))
        os.environ["CLEF_GDN_QKV_NORM_MAX_TOKENS"] = "306"
        self.assertIsNone(self.call())

    def test_eligible_request_requires_explicit_vendor(self):
        os.environ.pop("CLEF_CONV_QKV_VENDOR")
        with self.assertRaisesRegex(RuntimeError, "requires CLEF_CONV_QKV_VENDOR"):
            self.call()

    def test_invalid_tile_is_visible(self):
        os.environ["CLEF_CONV_QKV_TILE"] = "128"
        with self.assertRaisesRegex(RuntimeError, "16, 32, or 64"):
            self.call()
        os.environ["CLEF_CONV_QKV_TILE"] = "invalid"
        with self.assertRaises(ValueError):
            self.call()

    def fake_runtime(self, fail=False):
        self.calls = []
        runtime = types.SimpleNamespace(_libraries=[])
        def call_aclnn(name, build, outputs, get_workspace_argtypes):
            if fail:
                raise RuntimeError("candidate kernel failure")
            args = build(types.SimpleNamespace(tensor=lambda tensor, name: (tensor, name)))
            self.calls.append((name, args, outputs, get_workspace_argtypes))
        api = types.SimpleNamespace(runtime=lambda: runtime, call_aclnn=call_aclnn)
        ascendc = types.ModuleType("fla_npu.ops.ascendc")
        ascendc._runtime = api
        modules = {"fla_npu": types.ModuleType("fla_npu"),
                   "fla_npu.ops": types.ModuleType("fla_npu.ops"),
                   "fla_npu.ops.ascendc": ascendc}
        return modules, runtime

    def test_eligible_call_uses_fused_abi_and_caches_loader(self):
        modules, runtime = self.fake_runtime()
        os.environ["CLEF_CONV_QKV_TILE"] = "64"
        with mock.patch.dict("sys.modules", modules), mock.patch.object(ctypes, "CDLL") as loader:
            outputs = self.call()
            self.call()
        loader.assert_called_once()
        self.assertEqual(len(runtime._libraries), 1)
        self.assertEqual([x.shape for x in outputs], [(1, 307, 8, 128), (1, 307, 8, 128), (1, 307, 24, 128)])
        name, args, _, argtypes = self.calls[0]
        self.assertEqual(name, "aclnnClefGdnConvQkv")
        self.assertEqual(args[0], (self.x, "x"))
        self.assertEqual(args[1], (self.weight, "weight"))
        self.assertEqual(args[2].value, 1e-6)
        self.assertEqual(args[3].value, 64)
        self.assertEqual(len(argtypes), 9)
        self.assertEqual([x[1] for x in args[4:]], ["qOut", "kOut", "vOut"])

    def test_loader_and_kernel_errors_propagate(self):
        modules, _ = self.fake_runtime()
        with mock.patch.dict("sys.modules", modules), mock.patch.object(ctypes, "CDLL", side_effect=OSError("bad vendor")):
            with self.assertRaisesRegex(OSError, "bad vendor"):
                self.call()
        modules, _ = self.fake_runtime(fail=True)
        with mock.patch.dict("sys.modules", modules), mock.patch.object(ctypes, "CDLL"):
            with self.assertRaisesRegex(RuntimeError, "candidate kernel failure"):
                self.call()

    def test_vendor_change_is_visible(self):
        modules, _ = self.fake_runtime()
        with mock.patch.dict("sys.modules", modules), mock.patch.object(ctypes, "CDLL"):
            self.call()
            os.environ["CLEF_CONV_QKV_VENDOR"] = "/another/vendor"
            with self.assertRaisesRegex(RuntimeError, "changed within a running worker"):
                self.call()

    def test_cap_and_activation_true_are_supported(self):
        modules, _ = self.fake_runtime()
        os.environ["CLEF_GDN_QKV_NORM_MAX_TOKENS"] = "1600"
        with mock.patch.dict("sys.modules", modules), mock.patch.object(ctypes, "CDLL"):
            outputs = self.call(x=FakeTensor((1595, 5120)), activation=True)
        self.assertEqual(outputs[0].shape, (1, 1595, 8, 128))


if __name__ == "__main__":
    unittest.main()
