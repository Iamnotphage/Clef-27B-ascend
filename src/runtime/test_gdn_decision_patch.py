#!/usr/bin/env python3
"""Unit tests for the decision-only GDN switches and the patch anchors.

Run with the project interpreter:
  /workspace/yangchen/venvs/clef-py312/bin/python runtime/test_gdn_decision_patch.py
"""

from __future__ import annotations

import importlib.util
import contextlib
import copy
import functools
import io
import json
import os
import sys
import tempfile
import time
import types
import unittest
from pathlib import Path
from unittest import mock


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


HERE = Path(__file__).resolve().parent
flags = load("gdn_decision_flags_test", HERE / "gdn_decision_flags.py")
installer = load("apply_gdn_decision_patch_test", HERE / "apply_gdn_decision_patch.py")
launcher = load("clef_inference_test", HERE.parents[1] / "inference.py")


class FlagTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.flag_file = Path(self.temporary.name) / "flags.json"
        for key in ("CLEF_GDN_DECISION_PREFILL", "CLEF_GDN_NO_RECOMPUTE",
                    "CLEF_GDN_FLAG_FILE", "CLEF_GDN_BSND",
                    "CLEF_GDN_SKIP_INITIAL_STATE", "CLEF_GDN_SKIP_CONV_STATE",
                    "CLEF_GDN_PREPROCESS_FUSION"):
            os.environ.pop(key, None)
        flags.reset_cache()

    def tearDown(self):
        for key in ("CLEF_GDN_DECISION_PREFILL", "CLEF_GDN_NO_RECOMPUTE",
                    "CLEF_GDN_FLAG_FILE", "CLEF_GDN_BSND",
                    "CLEF_GDN_SKIP_INITIAL_STATE", "CLEF_GDN_SKIP_CONV_STATE",
                    "CLEF_GDN_PREPROCESS_FUSION"):
            os.environ.pop(key, None)
        flags.reset_cache()
        self.temporary.cleanup()

    def opt_in(self):
        os.environ["CLEF_GDN_FLAG_FILE"] = str(self.flag_file)
        flags.reset_cache()

    def test_env_defaults(self):
        self.assertFalse(flags.decision_prefill_enabled())
        self.assertFalse(flags.no_recompute_enabled())
        self.assertFalse(flags.runtime_override_enabled())
        os.environ["CLEF_GDN_DECISION_PREFILL"] = "1"
        flags.reset_cache()
        self.assertTrue(flags.decision_prefill_enabled())
        self.assertTrue(flags.no_recompute_enabled())

    def test_env_false_values(self):
        for value in ("0", "false", "False", "no", "off", ""):
            os.environ["CLEF_GDN_DECISION_PREFILL"] = value
            flags.reset_cache()
            self.assertFalse(flags.decision_prefill_enabled(), value)

    def test_file_is_ignored_without_optin(self):
        self.flag_file.write_text(json.dumps({"decision_prefill": True}))
        flags.reset_cache()
        self.assertFalse(flags.decision_prefill_enabled())
        self.assertFalse(flags.no_recompute_enabled())

    def test_file_override_with_optin(self):
        self.opt_in()
        self.flag_file.write_text(json.dumps({"decision_prefill": True}))
        flags.reset_cache()
        self.assertTrue(flags.decision_prefill_enabled())
        self.flag_file.write_text(json.dumps({"decision_prefill": False}))
        os.utime(self.flag_file, (time.time() + 5, time.time() + 5))
        time.sleep(1.1)
        self.assertFalse(flags.decision_prefill_enabled())
        self.assertFalse(flags.no_recompute_enabled())

    def test_no_recompute_independent(self):
        self.opt_in()
        self.flag_file.write_text(
            json.dumps({"decision_prefill": True, "no_recompute": False})
        )
        flags.reset_cache()
        self.assertTrue(flags.decision_prefill_enabled())
        self.assertFalse(flags.no_recompute_enabled())

    def test_file_wins_over_env_when_opted_in(self):
        os.environ["CLEF_GDN_DECISION_PREFILL"] = "0"
        self.opt_in()
        self.flag_file.write_text(json.dumps({"decision_prefill": True}))
        flags.reset_cache()
        self.assertTrue(flags.decision_prefill_enabled())

    def test_mtime_cache(self):
        self.opt_in()
        self.flag_file.write_text(json.dumps({"decision_prefill": True}))
        flags.reset_cache()
        self.assertTrue(flags.decision_prefill_enabled())
        self.flag_file.write_text(json.dumps({"decision_prefill": False}))
        os.utime(self.flag_file, (time.time() + 5, time.time() + 5))
        time.sleep(1.1)
        self.assertFalse(flags.decision_prefill_enabled())

    def test_data_path_keys_and_epoch(self):
        self.opt_in()
        self.flag_file.write_text(json.dumps({
            "epoch": 17,
            "decision_prefill": True,
            "no_recompute": True,
            "bsnd": True,
            "skip_initial_state": True,
            "skip_conv_state": True,
            "preprocess_fusion": True,
        }))
        flags.reset_cache()
        snapshot = flags.read_flags()
        self.assertEqual(snapshot.epoch, 17)
        for key in ("decision_prefill", "no_recompute", "bsnd",
                    "skip_initial_state", "skip_conv_state", "preprocess_fusion"):
            self.assertTrue(getattr(snapshot, key), key)
        self.assertTrue(flags.bsnd_enabled())
        self.assertTrue(flags.skip_initial_state_enabled())
        self.assertTrue(flags.skip_conv_state_enabled())
        self.assertTrue(flags.preprocess_fusion_enabled())

    def test_non_boolean_values_are_ignored(self):
        self.opt_in()
        self.flag_file.write_text(json.dumps({
            "decision_prefill": 1,
            "bsnd": "false",
            "skip_conv_state": True,
        }))
        flags.reset_cache()
        snapshot = flags.read_flags()
        self.assertFalse(snapshot.decision_prefill)
        self.assertFalse(snapshot.no_recompute)
        self.assertFalse(snapshot.bsnd)
        self.assertTrue(snapshot.skip_conv_state)
        self.assertEqual(snapshot.epoch, 0)

    def test_invalid_epoch_falls_back_to_zero(self):
        self.opt_in()
        for value in (-1, True, "3", 1.5):
            self.flag_file.write_text(json.dumps({"epoch": value, "bsnd": True}))
            flags.reset_cache()
            self.assertEqual(flags.read_flags().epoch, 0, value)
            self.assertTrue(flags.bsnd_enabled(), value)

    def test_snapshot_is_per_forward_key(self):
        self.opt_in()
        self.flag_file.write_text(json.dumps({"epoch": 1, "bsnd": True}))
        flags.reset_cache()
        first = object()
        second = object()
        self.assertTrue(flags.snapshot_for(first).bsnd)
        # A rewrite between layers of one forward must not change the values.
        self.flag_file.write_text(json.dumps({"epoch": 2, "bsnd": False}))
        os.utime(self.flag_file, (time.time() + 5, time.time() + 5))
        time.sleep(1.1)
        self.assertTrue(flags.snapshot_for(first).bsnd)
        self.assertEqual(flags.snapshot_for(first).epoch, 1)
        # The next forward adopts the new file.
        self.assertFalse(flags.snapshot_for(second).bsnd)
        self.assertEqual(flags.snapshot_for(second).epoch, 2)

    def test_env_defaults_for_new_keys(self):
        os.environ["CLEF_GDN_BSND"] = "1"
        os.environ["CLEF_GDN_SKIP_INITIAL_STATE"] = "true"
        os.environ["CLEF_GDN_SKIP_CONV_STATE"] = "0"
        flags.reset_cache()
        self.assertTrue(flags.bsnd_enabled())
        self.assertTrue(flags.skip_initial_state_enabled())
        self.assertFalse(flags.skip_conv_state_enabled())
        for key in ("CLEF_GDN_BSND", "CLEF_GDN_SKIP_INITIAL_STATE",
                    "CLEF_GDN_SKIP_CONV_STATE"):
            os.environ.pop(key, None)


SYNTHETIC_DEVICE_OP = '''import os


class DeviceOperator:
    def fla_gdn_prefill(
        q: torch.Tensor,
        prebuilt_meta,
        fused_fwd,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        from vllm.third_party.flash_linear_attention.ops.l2norm import l2norm_fwd
        output, final_state, *_ = fused_fwd(q, layout="BSND")
        output, final_state, *_ = fused_fwd(q, layout="BNSD")
        return output, final_state

    @staticmethod
    def fused_gdn_gating(A_log, a, b, dt_bias):
        return None


class Arch35Operator:
    def fla_gdn_prefill(
        q: torch.Tensor,
        prebuilt_meta,
        fused_fwd,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        output, final_state, *_ = fused_fwd(q)
        return output, final_state
'''

SYNTHETIC_GDN = '''class AscendGatedDeltaNetAttention(GatedDeltaNetAttention):
    @torch.no_grad()
    def _pack_conv_weights(self) -> None:
        pass


def _get_fla_gdn_prefill_op():
    try:
        from fla_npu.ops.ascendc import chunk_gated_delta_rule_fwd
    except ImportError:
        return None
    return chunk_gated_delta_rule_fwd


class Attention:
    def _forward_core(self):
        attn_metadata = get_attn_metadata()
        num_actual_tokens = attn_metadata.num_actual_tokens
        if prefill:
            if num_decodes == 0:
                if split_conv:
                    pass
                else:
                    mixed_qkv_non_spec_output = causal_conv1d_fn(
                        mixed_qkv_non_spec,
                        conv_weights_T,
                        self.conv1d.bias,
                        conv_states=self_kv_cache[0],
                        query_start_loc=query_start_loc_opt,
                        cache_indices=_normalize_causal_cache_indices(cache_indices_opt),
                        has_initial_state=initial_state_mode_opt,
                        activation="silu" if self.activation else None,
                        pad_slot_id=PAD_SLOT_ID,
                        null_block_id=0,
                        head_num=conv_head_num,
                    )
                    mixed_qkv_non_spec = mixed_qkv_non_spec_output
        elif attn_metadata.num_decodes > 0:
            pass
        if attn_metadata.num_prefills > 0:
            if fla_gdn_prefill_op is not None:
                initial_state = gather_ssm_states(
                    ssm_state,
                    prefill_state_indices,
                    prefill_has_initial_state,
                )
                (core_attn_out_non_spec, last_recurrent_state) = DeviceOperator.fla_gdn_prefill(
                    q=query_non_spec,
                    fused_fwd=fla_gdn_prefill_op,
                )
                scatter_ssm_states_(
                    ssm_state,
                    prefill_state_indices,
                    last_recurrent_state.to(ssm_state.dtype).contiguous(),
                )
            elif use_fused_chunk:
                scatter_ssm_states_(
                    ssm_state,
                    prefill_state_indices,
                    last_recurrent_state,
                )
'''


SYNTHETIC_QWEN_INIT = '''    @wraps(_GDN_ORIGINAL_INIT)
    def _gdn_init_with_packed_weight(self, *args, **kwargs):
        _GDN_ORIGINAL_INIT(self, *args, **kwargs)
        initialize_packed_conv_weight(self)
'''


class PatcherTests(unittest.TestCase):
    def test_device_op_patch(self):
        patched = installer.patch_device_op(SYNTHETIC_DEVICE_OP)
        self.assertEqual(patched.count("layout=\"BSND\""), 1)
        self.assertEqual(patched.count("layout=\"BNSD\""), 1)
        self.assertEqual(patched.count("if initial_state is None"), 1)
        self.assertEqual(patched.count("runtime=None,"), 2)
        self.assertIn("from vllm_ascend.ops.gdn_decision_flags import read_flags", patched)
        self.assertIn("runtime.state_free()", patched)
        # The second (A5) definition keeps its body.
        tail = patched.split(installer.DEF_MARKER, 2)[2]
        self.assertIn("fused_fwd(q)", tail)
        self.assertIn("runtime=None,", tail)

    def test_gdn_patch(self):
        patched = installer.patch_gdn(SYNTHETIC_GDN)
        self.assertEqual(patched.count("_gdn_runtime_flags("), 2)
        self.assertEqual(patched.count("_decision_prefill_guard("), 2)
        self.assertIn("skip_conv_state", patched)
        self.assertIn("skip_initial_state", patched)
        self.assertIn("decision_only=runtime is not None,", patched)
        self.assertIn("runtime=runtime,", patched)
        self.assertIn("num_decodes == 0", patched)
        self.assertIn("num_spec_decodes == 0", patched)
        self.assertIn("decision_safe_config", patched)
        self.assertIn("get_pcp_group().world_size == 1", patched)

    def test_qwen_init_patch(self):
        patched = installer.patch_qwen3_5(SYNTHETIC_QWEN_INIT)
        self.assertIn("_decision_safe_config", patched)
        self.assertIn("enable_chunked_prefill", patched)
        self.assertIn("enable_prefix_caching", patched)
        self.assertIn('runner_type == "pooling"', patched)

    def test_runner_guard_behavior(self):
        namespace = {
            "wraps": functools.wraps,
            "_GDN_ORIGINAL_INIT": lambda *args, **kwargs: None,
            "initialize_packed_conv_weight": lambda self: None,
        }
        exec("if True:\n" + installer.INIT_PATCHED, namespace)
        wrapped = namespace["_gdn_init_with_packed_weight"]
        for runner in ("pooling", "generate"):
            for chunked in (False, True):
                for prefix in (False, True):
                    for speculative in (None, object()):
                        config = types.SimpleNamespace(
                            model_config=types.SimpleNamespace(runner_type=runner),
                            scheduler_config=types.SimpleNamespace(enable_chunked_prefill=chunked),
                            cache_config=types.SimpleNamespace(enable_prefix_caching=prefix),
                            speculative_config=speculative,
                        )
                        for positional in (False, True):
                            target = types.SimpleNamespace()
                            if positional:
                                wrapped(target, object(), config)
                            else:
                                wrapped(target, vllm_config=config)
                            self.assertEqual(
                                target._decision_safe_config,
                                runner == "pooling" and not chunked and not prefix and speculative is None,
                                (runner, chunked, prefix, speculative, positional),
                            )
        target = types.SimpleNamespace()
        wrapped(target)
        self.assertFalse(target._decision_safe_config)

    def test_hash_contract_matches_accepted_artifacts(self):
        evidence = HERE.parents[1] / "logs/gdn-decision-20261115"
        accepted = json.loads((evidence / "entry-s1/launch-manifest.json").read_text())
        record = accepted["gdn_decision_prefill"]
        # The historical entry is the v2 contract; v3 must still recognise it
        # as an upgrade, not as unknown content.
        self.assertEqual(installer.V2_POSTIMAGE_SHA, record["files"])
        self.assertEqual(installer.LEGACY_FLAG_SHA256, record["flag_module_sha256"])
        self.assertEqual(installer.sha256_file(installer.FLAG_SOURCE), installer.FLAG_SHA256)
        legacy = json.loads((evidence / "phase0/patch-apply-v5.json").read_text())
        for relative, digest in installer.V1_POSTIMAGE_SHA.items():
            self.assertEqual(digest, legacy["files"][relative]["sha256"])
        self.assertEqual(installer.LEGACY_FLAG_SHA256_V1, legacy["flag_module"]["sha256"])



class InstallerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.tree = self.root / "source tree"
        self.backup = self.root / "backups"
        self.preimages = dict(zip(installer.PREIMAGE_SHA, (
            SYNTHETIC_DEVICE_OP.encode(), SYNTHETIC_GDN.encode(), SYNTHETIC_QWEN_INIT.encode(),
        )))
        self.postimages = dict(zip(installer.PREIMAGE_SHA, (
            installer.patch_device_op(SYNTHETIC_DEVICE_OP).encode(),
            installer.patch_gdn(SYNTHETIC_GDN).encode(),
            installer.patch_qwen3_5(SYNTHETIC_QWEN_INIT).encode(),
        )))
        self.guard_path = "vllm_ascend/patch/worker/patch_qwen3_5.py"
        self.legacy_guard = self.postimages[self.guard_path].replace(
            b'            and vllm_config.model_config.runner_type == "pooling"\n', b'')
        self.legacy_flag = b"# known legacy global flag implementation\n"
        self.v2_images = {
            key: (data.replace(b"runtime=None,", b"")
                  if b"runtime=None," in data
                  else data.replace(b"runtime=runtime,", b""))
            if key != self.guard_path else data
            for key, data in self.postimages.items()
        }
        constants = {
            "PREIMAGE_SHA": {key: installer.sha256_bytes(data) for key, data in self.preimages.items()},
            "POSTIMAGE_SHA": {key: installer.sha256_bytes(data) for key, data in self.postimages.items()},
            "V2_POSTIMAGE_SHA": {key: installer.sha256_bytes(data) for key, data in self.v2_images.items()},
            "V1_POSTIMAGE_SHA": {self.guard_path: installer.sha256_bytes(self.legacy_guard)},
            "LEGACY_POSTIMAGE_SHA": {self.guard_path: installer.sha256_bytes(self.legacy_guard)},
            "LEGACY_FLAG_SHA256": installer.sha256_bytes(self.legacy_flag),
            "LEGACY_FLAG_SHA256_V1": installer.sha256_bytes(self.legacy_flag + b"v1"),
        }
        patcher = mock.patch.multiple(installer, **constants)
        patcher.start()
        self.addCleanup(patcher.stop)
        for relative, data in self.preimages.items():
            path = self.tree / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)

    def call(self, action, *extra):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            code = installer.main([
                "--tree", str(self.tree), "--backup-dir", str(self.backup), action, *extra,
            ])
        return code, json.loads(output.getvalue())

    def snapshot(self, timestamps=False):
        return {
            str(path.relative_to(self.root)): (
                path.read_bytes(), path.stat().st_mtime_ns if timestamps else None,
            )
            for path in self.root.rglob("*") if path.is_file()
        }

    def install(self):
        code, report = self.call("--apply")
        self.assertEqual(code, 0, report)
        self.assertTrue(report["ready"])
        return report

    def set_legacy(self):
        self.install()
        (self.tree / self.guard_path).write_bytes(self.legacy_guard)
        (self.tree / installer.FLAG_TARGET).write_bytes(self.legacy_flag)

    def set_v2(self):
        self.install()
        for relative, data in self.v2_images.items():
            (self.tree / relative).write_bytes(data)

    def test_apply_idempotent_check_and_revert(self):
        self.install()
        before = self.snapshot(timestamps=True)
        code, checked = self.call("--check")
        self.assertEqual(code, 0)
        self.assertTrue(checked["ready"])
        self.install()
        self.assertEqual(before, self.snapshot(timestamps=True))
        code, report = self.call("--revert")
        self.assertEqual(code, 0, report)
        self.assertFalse(report["ready"])
        self.assertFalse((self.tree / installer.FLAG_TARGET).exists())
        for relative, data in self.preimages.items():
            self.assertEqual((self.tree / relative).read_bytes(), data)

    def test_missing_flag_is_repaired(self):
        self.install()
        flag = self.tree / installer.FLAG_TARGET
        flag.unlink()
        _, report = self.call("--check")
        self.assertFalse(report["ready"])
        self.assertEqual(report["flag_module"]["status"], "missing")
        self.install()
        self.assertEqual(flag.read_bytes(), installer.FLAG_SOURCE.read_bytes())

    def test_known_legacy_flag_is_repaired(self):
        self.install()
        (self.tree / installer.FLAG_TARGET).write_bytes(self.legacy_flag)
        self.install()

    def test_v2_tree_upgrades_from_verified_backup(self):
        self.set_v2()
        _, report = self.call("--check")
        self.assertEqual(
            report["files"]["vllm_ascend/device/device_op.py"]["status_before"], "v2")
        report = self.install()
        self.assertEqual(
            report["files"]["vllm_ascend/device/device_op.py"]["action"], "upgraded")
        self.assertEqual(
            report["files"]["vllm_ascend/ops/gdn.py"]["action"], "upgraded")
        for relative, data in self.postimages.items():
            self.assertEqual((self.tree / relative).read_bytes(), data)

    def test_v2_tree_without_backup_refuses_before_writing(self):
        self.set_v2()
        for relative in self.preimages:
            (self.backup / f"{Path(relative).name}.preimage").unlink()
        before = self.snapshot(timestamps=True)
        code, report = self.call("--apply")
        self.assertEqual(code, 1)
        self.assertIn("requires a verified preimage backup", report["error"])
        self.assertEqual(before, self.snapshot(timestamps=True))

    def test_known_legacy_guard_upgrades_from_verified_backup(self):
        self.set_legacy()
        _, report = self.call("--check")
        self.assertEqual(report["files"][self.guard_path]["status_before"], "legacy")
        report = self.install()
        self.assertEqual(report["files"][self.guard_path]["action"], "upgraded")
        self.assertEqual((self.tree / self.guard_path).read_bytes(), self.postimages[self.guard_path])

    def test_known_legacy_guard_without_backup_refuses_before_writing(self):
        self.set_legacy()
        (self.backup / "patch_qwen3_5.py.preimage").unlink()
        before = self.snapshot(timestamps=True)
        code, report = self.call("--apply")
        self.assertEqual(code, 1)
        self.assertIn("legacy upgrade requires a verified preimage backup", report["error"])
        self.assertEqual(before, self.snapshot(timestamps=True))
        self.assertIn("--revert", report["recovery_commands"][0])
        self.assertIn("--apply", report["recovery_commands"][1])
        for command in report["recovery_commands"]:
            self.assertIn("--backup-dir", command)
            self.assertIn("--tree", command)

    def test_known_legacy_install_can_revert(self):
        self.set_legacy()
        code, report = self.call("--revert")
        self.assertEqual(code, 0, report)
        for relative, data in self.preimages.items():
            self.assertEqual((self.tree / relative).read_bytes(), data)

    def test_unknown_late_source_with_marker_leaves_every_file_unchanged(self):
        (self.tree / self.guard_path).write_bytes(self.postimages[self.guard_path] + b"# changed\n")
        before = self.snapshot(timestamps=True)
        code, report = self.call("--apply")
        self.assertEqual(code, 1)
        self.assertIn("refused unknown", report["error"])
        self.assertEqual(before, self.snapshot(timestamps=True))
        self.assertFalse(self.backup.exists())

    def test_wrong_late_backup_leaves_every_file_unchanged(self):
        self.backup.mkdir()
        (self.backup / "patch_qwen3_5.py.preimage").write_bytes(b"wrong backup")
        before = self.snapshot(timestamps=True)
        code, report = self.call("--apply")
        self.assertEqual(code, 1)
        self.assertIn("backup SHA-256 mismatch", report["error"])
        self.assertEqual(before, self.snapshot(timestamps=True))

    def test_revert_checks_all_backups_before_writing(self):
        self.install()
        (self.backup / "patch_qwen3_5.py.preimage").write_bytes(b"wrong backup")
        before = self.snapshot(timestamps=True)
        code, report = self.call("--revert")
        self.assertEqual(code, 1)
        self.assertIn("backup SHA-256 mismatch", report["error"])
        self.assertEqual(before, self.snapshot(timestamps=True))

    def test_unknown_flag_refuses_apply_and_revert_without_writes(self):
        self.install()
        (self.tree / installer.FLAG_TARGET).write_bytes(b"# user modified flag\n")
        before = self.snapshot(timestamps=True)
        for action in ("--apply", "--revert"):
            code, report = self.call(action)
            self.assertEqual(code, 1)
            self.assertIn("flag module has unknown content", report["error"])
            self.assertEqual(before, self.snapshot(timestamps=True))

    def test_generated_hash_mismatch_is_read_only(self):
        before = self.snapshot(timestamps=True)
        with mock.patch.object(installer, "patch_qwen3_5", return_value="changed generated code"):
            code, report = self.call("--apply")
        self.assertEqual(code, 1)
        self.assertIn("generated postimage SHA-256 mismatch", report["error"])
        self.assertEqual(before, self.snapshot(timestamps=True))

    def test_check_with_manifest_is_read_only(self):
        before = self.snapshot(timestamps=True)
        manifest = self.root / "manifest.json"
        code, report = self.call("--check", "--manifest", str(manifest))
        self.assertEqual(code, 0, report)
        self.assertEqual(before, self.snapshot(timestamps=True))
        self.assertFalse(manifest.exists())
        self.assertFalse(self.backup.exists())

    def test_write_failure_rolls_back_completed_changes(self):
        original_write = installer.write_atomic
        calls = 0

        def fail_once(path, data):
            nonlocal calls
            calls += 1
            if calls == 4:
                raise OSError("simulated write failure")
            return original_write(path, data)

        before = self.snapshot()
        with mock.patch.object(installer, "write_atomic", side_effect=fail_once):
            code, report = self.call("--apply")
        self.assertEqual(code, 1)
        self.assertIn("files restored", report["error"])
        self.assertEqual(before, self.snapshot())


class LauncherTests(unittest.TestCase):
    def setUp(self):
        environment = mock.patch.dict(os.environ, {}, clear=True)
        environment.start()
        self.addCleanup(environment.stop)

    def report(self, source_state="patched", flag_state="current"):
        report = {
            "patch_version": installer.PATCH_VERSION,
            "files": {},
            "flag_module": {
                "status": flag_state,
                "present": flag_state != "missing",
                "sha256": {
                    "current": installer.FLAG_SHA256,
                    "legacy": installer.LEGACY_FLAG_SHA256,
                    "legacy-v1": installer.LEGACY_FLAG_SHA256_V1,
                    "missing": None,
                }[flag_state],
                "source_sha256": installer.FLAG_SHA256,
                "expected_sha256": installer.FLAG_SHA256,
            },
            "ready": source_state == "patched" and flag_state == "current",
        }
        for path, digest in installer.POSTIMAGE_SHA.items():
            state = source_state
            if state == "legacy" and path not in installer.LEGACY_POSTIMAGE_SHA:
                state = "patched"
            if state == "v2" and path not in installer.V2_POSTIMAGE_SHA:
                state = "patched"
            report["files"][path] = {
                "status_before": state,
                "sha256": {
                    "patched": digest,
                    "preimage": installer.PREIMAGE_SHA[path],
                    "v2": installer.V2_POSTIMAGE_SHA.get(path),
                    "legacy": installer.LEGACY_POSTIMAGE_SHA.get(path),
                }[state],
                "postimage_sha256": digest,
            }
        return report

    def run_launcher(self, reports, *, apply=True):
        responses = [types.SimpleNamespace(returncode=0, stdout=json.dumps(report), stderr="")
                     for report in reports]
        with mock.patch.object(launcher.subprocess, "run", side_effect=responses) as run:
            record = launcher.gdn_decision_prefill("s1", apply=apply)
        return record, [call.args[0][-1] for call in run.call_args_list]

    def test_current_install_enables_after_check(self):
        record, calls = self.run_launcher([self.report()])
        self.assertTrue(record["enabled"])
        self.assertEqual(calls, ["--check"])
        self.assertEqual(record["files"], installer.POSTIMAGE_SHA)
        self.assertEqual(record["flag_module_sha256"], installer.FLAG_SHA256)

    def test_missing_and_legacy_flag_repair_then_recheck(self):
        for state in ("missing", "legacy"):
            record, calls = self.run_launcher([self.report(flag_state=state), {}, self.report()])
            self.assertTrue(record["enabled"])
            self.assertEqual(calls, ["--check", "--apply", "--check"])

    def test_legacy_guard_upgrades_then_rechecks(self):
        record, calls = self.run_launcher([self.report("legacy", "legacy"), {}, self.report()])
        self.assertTrue(record["enabled"])
        self.assertEqual(calls, ["--check", "--apply", "--check"])

    def test_v2_tree_upgrades_then_rechecks(self):
        record, calls = self.run_launcher([self.report("v2", "legacy"), {}, self.report()])
        self.assertTrue(record["enabled"])
        self.assertEqual(calls, ["--check", "--apply", "--check"])

    def test_dry_run_only_checks_without_install(self):
        for state, flag in (("preimage", "missing"), ("legacy", "legacy"), ("patched", "missing")):
            record, calls = self.run_launcher([self.report(state, flag)], apply=False)
            self.assertFalse(record["enabled"])
            self.assertTrue(record["pending_install"])
            self.assertEqual(calls, ["--check"])

    def test_dry_run_does_not_create_installer_bytecode(self):
        with tempfile.TemporaryDirectory() as directory:
            script = Path(directory) / "installer.py"
            script.write_bytes(launcher.GDN_INSTALLER.read_bytes())
            with mock.patch.object(launcher, "GDN_INSTALLER", script), \
                    mock.patch.object(launcher.sys, "dont_write_bytecode", False):
                record, calls = self.run_launcher([self.report("preimage", "missing")], apply=False)
            self.assertFalse(record["enabled"])
            self.assertEqual(calls, ["--check"])
            self.assertEqual(list(Path(directory).iterdir()), [script])

    def test_explicit_off_does_not_inspect_or_install(self):
        os.environ["CLEF_GDN_DECISION_PREFILL"] = "0"
        record, calls = self.run_launcher([])
        self.assertFalse(record["enabled"])
        self.assertEqual(calls, [])

    def test_partial_unknown_source_does_not_apply(self):
        report = self.report("preimage", "missing")
        item = list(report["files"].values())[-1]
        item.update(status_before="unknown", sha256="modified source")
        response = types.SimpleNamespace(returncode=0, stdout=json.dumps(report), stderr="")
        with mock.patch.object(launcher.subprocess, "run", return_value=response) as run:
            with self.assertRaisesRegex(SystemExit, "source SHA-256 mismatch"):
                launcher.gdn_decision_prefill("s1")
        self.assertEqual([call.args[0][-1] for call in run.call_args_list], ["--check"])

    def test_claimed_current_with_mismatched_hash_is_refused(self):
        valid = self.report()
        changed_source = copy.deepcopy(valid)
        list(changed_source["files"].values())[-1]["sha256"] = "old guard with current marker"
        changed_flag = copy.deepcopy(valid)
        changed_flag["flag_module"]["sha256"] = installer.LEGACY_FLAG_SHA256
        missing_flag = copy.deepcopy(valid)
        missing_flag["flag_module"]["present"] = False
        for report in (changed_source, changed_flag, missing_flag):
            with self.assertRaises(SystemExit):
                self.run_launcher([report])

    def test_apply_success_requires_complete_recheck(self):
        with self.assertRaisesRegex(SystemExit, "patch incomplete"):
            self.run_launcher([self.report(flag_state="missing"), {}, self.report(flag_state="missing")])


if __name__ == "__main__":
    unittest.main(verbosity=2)
