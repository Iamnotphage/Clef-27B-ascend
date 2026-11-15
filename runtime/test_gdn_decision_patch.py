#!/usr/bin/env python3
"""Unit tests for the decision-only GDN switches and the patch anchors.

Run with the project interpreter:
  /workspace/yangchen/venvs/clef-py312/bin/python runtime/test_gdn_decision_patch.py
"""

from __future__ import annotations

import importlib.util
import json
import os
import tempfile
import time
import unittest
from pathlib import Path


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


HERE = Path(__file__).resolve().parent
flags = load("gdn_decision_flags_test", HERE / "gdn_decision_flags.py")
installer = load("apply_gdn_decision_patch_test", HERE / "apply_gdn_decision_patch.py")


class FlagTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.flag_file = Path(self.temporary.name) / "flags.json"
        for key in ("CLEF_GDN_DECISION_PREFILL", "CLEF_GDN_NO_RECOMPUTE",
                    "CLEF_GDN_FLAG_FILE"):
            os.environ.pop(key, None)
        flags.reset_cache()

    def tearDown(self):
        for key in ("CLEF_GDN_DECISION_PREFILL", "CLEF_GDN_NO_RECOMPUTE",
                    "CLEF_GDN_FLAG_FILE"):
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


SYNTHETIC_DEVICE_OP = '''import torch


class DeviceOperator:
    def fla_gdn_prefill(
        q: torch.Tensor,
        prebuilt_meta,
        fused_fwd,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        from vllm.third_party.flash_linear_attention.ops.l2norm import l2norm_fwd
        candidate = True
        if candidate:
            output, final_state, *_ = fused_fwd(
                q,
                initial_state=initial_state_kern,
                output_final_state=True,
                layout="BSND",
                disable_recompute=True,
            )
            if keep_meta is not None:
                # Scatter in the kernel layout so empty segments keep their
                full_final_state = initial_state
                final_state = full_final_state
            return output, final_state
        output, final_state, *_ = fused_fwd(
            q,
            initial_state=initial_state_kern,
            output_final_state=True,
            layout="BNSD",
            disable_recompute=True,
        )
        if keep_meta is not None:
            # Scatter in the kernel layout so empty segments keep their initial state,
            full_final_state = initial_state
            final_state = full_final_state
        return output, final_state


class Arch35Operator:
    def fla_gdn_prefill(
        q: torch.Tensor,
        prebuilt_meta,
        fused_fwd,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        output, final_state, *_ = fused_fwd(
            q, output_final_state=True, disable_recompute=True,
        )
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
    def forward(self):
        if prefill:
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
        self.assertEqual(patched.count("output_final_state=not decision_only,"), 2)
        self.assertEqual(patched.count("disable_recompute=not skip_recompute,"), 2)
        self.assertEqual(patched.count("if decision_only:"), 2)
        self.assertEqual(patched.count("decision_only: bool | None = None,"), 2)
        self.assertEqual(patched.count("if decision_only is None:"), 1)
        self.assertIn("from vllm_ascend.ops.gdn_decision_flags import (", patched)
        # The second definition must be untouched.
        tail = patched.split(installer.DEF_MARKER, 2)[2]
        self.assertIn("output_final_state=True", tail)
        self.assertIn("disable_recompute=True", tail)

    def test_gdn_patch(self):
        patched = installer.patch_gdn(SYNTHETIC_GDN)
        self.assertEqual(patched.count("_decision_prefill_guard("), 2)
        self.assertIn("decision_only=decision_only,", patched)
        self.assertIn("num_decodes == 0", patched)
        self.assertIn("num_spec_decodes == 0", patched)
        self.assertIn("decision_safe_config", patched)

    def test_qwen_init_patch(self):
        patched = installer.patch_qwen3_5(SYNTHETIC_QWEN_INIT)
        self.assertIn("_decision_safe_config", patched)
        self.assertIn("enable_chunked_prefill", patched)
        self.assertIn("enable_prefix_caching", patched)
        self.assertIn('runner_type == "pooling"', patched)

    def test_patch_is_idempotent_signal(self):
        # A patched file must be recognised by the marker check.
        patched = installer.patch_gdn(SYNTHETIC_GDN)
        self.assertIn(installer.PATCH_MARKER["vllm_ascend/ops/gdn.py"], patched)


if __name__ == "__main__":
    unittest.main(verbosity=2)
