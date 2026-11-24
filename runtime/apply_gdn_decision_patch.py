#!/usr/bin/env python3
"""Install or revert the decision-only GDN prefill switches.

The running service uses an editable install of the pinned vLLM-Ascend tree
(``89eb8a5``).  This installer owns four artifacts:

1. ``vllm_ascend/ops/gdn_decision_flags.py`` — new flag module, copied from
   ``runtime/gdn_decision_flags.py``.
2. ``vllm_ascend/device/device_op.py`` — the A3 ``fla_gdn_prefill`` wrapper
   stops requesting the final state and the backward-only intermediates when
   the switches are on.
3. ``vllm_ascend/ops/gdn.py`` — the prefill branch skips the state write-back
   when the wrapper returns no final state.
4. ``vllm_ascend/patch/worker/patch_qwen3_5.py`` — the construction-time
   guard restricts the specialization to a compatible pooling runner.

Every edit is anchored on an exact snippet and gated by the preimage SHA-256,
so a changed upstream tree is refused instead of silently mis-patched.  The
decode path, the Triton fallback branch and the A5 wrapper are untouched.

Usage:
  python runtime/apply_gdn_decision_patch.py --check
  python runtime/apply_gdn_decision_patch.py --apply
  python runtime/apply_gdn_decision_patch.py --revert
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import shlex
import sys
import time
from pathlib import Path

DEFAULT_TREE = Path("/workspace/yangchen/work/src/vllm-ascend")
DEFAULT_BACKUP = Path("/workspace/yangchen/logs/gdn-decision-20261115/phase0")
FLAG_SOURCE = Path(__file__).with_name("gdn_decision_flags.py")
FLAG_TARGET = "vllm_ascend/ops/gdn_decision_flags.py"
PATCH_VERSION = 4
FLAG_SHA256 = "3c9cb53a34a4ef4c2b88ce0e283a1343a9268c578aeab1c506645f3d79147287"
LEGACY_FLAG_SHA256 = "cf1ede3f3e8a894d9f0896a35cd07254ffa756e18e699fa40636cea86043fc70"
LEGACY_FLAG_SHA256_V1 = "02987b37857ffad0ab259fc55fad6bb1f02d39c885664b36353572057b98c731"

PREIMAGE_SHA = {
    "vllm_ascend/device/device_op.py":
        "a08d44b41b337d839548c02b7457510968d5c5057ece01243b022a91348a0823",
    "vllm_ascend/ops/gdn.py":
        "b235e34d8a2427b167d4ce0e9068312904ecd1473fe288cefe1e9b6f95de030d",
    "vllm_ascend/patch/worker/patch_qwen3_5.py":
        "a9a71634ed138f590fbc3ad0de277a5f56a06451522fd091b2657d6daf6084b0",
}
POSTIMAGE_SHA = {
    "vllm_ascend/device/device_op.py":
        "6886fa3fe7855d4dc8c754a9564ed2fe3967efd0cc09c5edd487c57be371d3ce",
    "vllm_ascend/ops/gdn.py":
        "7bc9b8f19e1ab17f7b484121c4c88c7f365c5ddb3a9f7a0380592175c077c893",
    "vllm_ascend/patch/worker/patch_qwen3_5.py":
        "a4de9b4ecc576b83074b50f8c500b8f9419354081b8ce50193240522129155a4",
}
QKV_POSTIMAGE_SHA = {
    "vllm_ascend/device/device_op.py":
        "19cf5d0f34015cc1e4f5f7639a084d3ad64ed24e0a7a1a3b1946b475eec14383",
    "vllm_ascend/ops/gdn.py":
        "c8c0d4d8a693658e075a2232b6938584760672bacf101bf50ca5fdfab2c0be7e",
}
V2_POSTIMAGE_SHA = {
    "vllm_ascend/device/device_op.py":
        "ad92a322a69dbd023cd88603b5882d0c5f7f931dbf20692957efdddd6f7809d2",
    "vllm_ascend/ops/gdn.py":
        "509dafb03b12351ebb17854742681b4aede60958a820d7625af94467c3586f43",
    "vllm_ascend/patch/worker/patch_qwen3_5.py":
        "1b82b52eabe24d8fcdcb55ff16c307a9b04358ba846a4d73ebb4f1443c0b0460",
}
CONV_QKV_POSTIMAGE_SHA = {
    "vllm_ascend/ops/gdn.py":
        "ead25cb2691cb9ab1c322353fc8e168f8bfc9c23b9e4d53de01c762943027e78",
}
CHUNK128_POSTIMAGE_SHA = {
    "vllm_ascend/device/device_op.py":
        "bf027ec99a3d069f913600ddfcb76283f73a4114febe46b151c113c62450c226",
}
ADAPTIVE_POSTIMAGE_SHA = {
    "vllm_ascend/device/device_op.py":
        "2abb3389f8502f17f0f0f0f567d50aed035623dae0740355a66aaa5d5a035b59",
}
V3_POSTIMAGE_SHA = {
    "vllm_ascend/patch/worker/patch_qwen3_5.py":
        "1b82b52eabe24d8fcdcb55ff16c307a9b04358ba846a4d73ebb4f1443c0b0460",
}
MLP_NZ_BLOCK = '''

# --- Clef MLP BF16 NZ patch (enabled by CLEF_MLP_BF16_NZ=1) -----------------
import os as _clef_os

if _clef_os.environ.get("CLEF_MLP_BF16_NZ") == "1":
    import torch_npu as _clef_torch_npu
    from vllm.model_executor.models.qwen2_moe import Qwen2MoeMLP as _ClefMoeMLP

    _clef_mlp_nz_seen = set()
    _clef_mlp_nz_forward = _ClefMoeMLP.forward

    def _clef_mlp_nz_wrapper(self, x):
        if id(self) not in _clef_mlp_nz_seen:
            self.gate_up_proj.weight.data = _clef_torch_npu.npu_format_cast(
                self.gate_up_proj.weight.data, 29)
            self.down_proj.weight.data = _clef_torch_npu.npu_format_cast(
                self.down_proj.weight.data, 29)
            _clef_mlp_nz_seen.add(id(self))
            if len(_clef_mlp_nz_seen) == 1:
                print("clef MLP BF16 NZ applied: gate_up=%s down=%s" % (
                    _clef_torch_npu.get_npu_format(self.gate_up_proj.weight.data),
                    _clef_torch_npu.get_npu_format(self.down_proj.weight.data)),
                    flush=True)
        return _clef_mlp_nz_forward(self, x)

    _ClefMoeMLP.forward = _clef_mlp_nz_wrapper
'''
V1_POSTIMAGE_SHA = {
    "vllm_ascend/patch/worker/patch_qwen3_5.py":
        "d2666387506cb68d9238c6f06baa89253158ee0facc67ad1243cf41ad01edcad",
}
LEGACY_POSTIMAGE_SHA = V1_POSTIMAGE_SHA

DEF_MARKER = "    def fla_gdn_prefill("
IMPORT_ANCHOR = (
    "        from vllm.third_party.flash_linear_attention.ops.l2norm "
    "import l2norm_fwd\n"
)
IMPORT_ADDED = (
    IMPORT_ANCHOR
    + "        from vllm_ascend.ops.gdn_decision_flags import (\n"
    + "            decision_prefill_enabled,\n"
    + "            no_recompute_enabled,\n"
    + "        )\n"
    + "\n"
    + "        decision_only = decision_prefill_enabled()\n"
    + "        skip_recompute = no_recompute_enabled()\n"
)
BSND_ANCHOR = (
    "            if keep_meta is not None:\n"
    "                # Scatter in the kernel layout so empty segments keep their\n"
)
BSND_PATCHED = (
    "            if decision_only:\n"
    "                return output, None\n"
    + BSND_ANCHOR
)
BNSD_ANCHOR = (
    "        if keep_meta is not None:\n"
    "            # Scatter in the kernel layout so empty segments keep their "
    "initial state,\n"
)
BNSD_PATCHED = (
    "        if decision_only:\n"
    "            return output, None\n"
    + BNSD_ANCHOR
)
SIGNATURE_ANCHOR = (
    "        prebuilt_meta,\n"
    "        fused_fwd,\n"
    "    ) -> tuple[torch.Tensor, torch.Tensor]:\n"
)
SIGNATURE_PATCHED = (
    "        prebuilt_meta,\n"
    "        fused_fwd,\n"
    "        *,\n"
    "        decision_only: bool | None = None,\n"
    "    ) -> tuple[torch.Tensor, torch.Tensor]:\n"
)
DECISION_INIT_ANCHOR = "        decision_only = decision_prefill_enabled()\n"
DECISION_INIT_PATCHED = (
    "        if decision_only is None:\n"
    "            decision_only = decision_prefill_enabled()\n"
)
GUARD_HELPER = (
    "\n"
    "def _decision_prefill_guard(decision_safe_config, attn_metadata) -> bool:\n"
    "    \"\"\"True only for a batch that cannot consume a recurrent state.\n"
    "\n"
    "    The engine recorded at construction time that chunked prefill, prefix\n"
    "    caching and speculative decoding are off, so a prefill batch with no\n"
    "    decode token cannot carry a state a later step reads.  The check is\n"
    "    host-side: reading ``prefill_has_initial_state`` would sync the device\n"
    "    once per layer and cost more than the state write it saves.\n"
    "    \"\"\"\n"
    "    from vllm_ascend.ops.gdn_decision_flags import decision_prefill_enabled\n"
    "\n"
    "    return bool(\n"
    "        decision_safe_config\n"
    "        and decision_prefill_enabled()\n"
    "        and attn_metadata.num_prefills > 0\n"
    "        and attn_metadata.num_decodes == 0\n"
    "        and attn_metadata.num_decode_tokens == 0\n"
    "        and attn_metadata.num_spec_decodes == 0\n"
    "        and attn_metadata.num_spec_decode_tokens == 0\n"
    "    )\n"
)
GUARD_ANCHOR = (
    "            if fla_gdn_prefill_op is not None:\n"
    "                initial_state = gather_ssm_states(\n"
)
GUARD_PATCHED = (
    "            decision_only = _decision_prefill_guard(\n"
    "                self._decision_safe_config, attn_metadata\n"
    "            )\n"
    + GUARD_ANCHOR
)
KWARG_ANCHOR = (
    "                    fused_fwd=fla_gdn_prefill_op,\n"
    "                )\n"
)
KWARG_PATCHED = (
    "                    fused_fwd=fla_gdn_prefill_op,\n"
    "                    decision_only=decision_only,\n"
    "                )\n"
)

SCATTER_LATE = (
    "                scatter_ssm_states_(\n"
    "                    ssm_state,\n"
    "                    prefill_state_indices,\n"
    "                    last_recurrent_state.to(ssm_state.dtype).contiguous(),\n"
    "                )\n"
)
SCATTER_LATE_PATCHED = (
    "                if last_recurrent_state is not None:\n"
    "                    scatter_ssm_states_(\n"
    "                        ssm_state,\n"
    "                        prefill_state_indices,\n"
    "                        last_recurrent_state.to(ssm_state.dtype).contiguous(),\n"
    "                    )\n"
)
SCATTER_FUSED = (
    "                scatter_ssm_states_(\n"
    "                    ssm_state,\n"
    "                    prefill_state_indices,\n"
    "                    last_recurrent_state,\n"
    "                )\n"
)
SCATTER_FUSED_PATCHED = (
    "                if last_recurrent_state is not None:\n"
    "                    scatter_ssm_states_(\n"
    "                        ssm_state,\n"
    "                        prefill_state_indices,\n"
    "                        last_recurrent_state,\n"
    "                    )\n"
)


def sha256_bytes(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def status(path: Path, relative: str) -> str:
    if not path.is_file():
        return "missing"
    digest = sha256_file(path)
    if digest == PREIMAGE_SHA[relative]:
        return "preimage"
    if digest == POSTIMAGE_SHA[relative]:
        return "patched"
    if digest == QKV_POSTIMAGE_SHA.get(relative):
        return "qkv"
    if digest == CONV_QKV_POSTIMAGE_SHA.get(relative):
        return "conv"
    if digest == CHUNK128_POSTIMAGE_SHA.get(relative):
        return "chunk128"
    if digest == ADAPTIVE_POSTIMAGE_SHA.get(relative):
        return "adaptive"
    if digest == V3_POSTIMAGE_SHA.get(relative):
        return "v3"
    if digest == V2_POSTIMAGE_SHA.get(relative):
        return "v2"
    if digest == V1_POSTIMAGE_SHA.get(relative):
        return "legacy"
    return "unknown"


A3_FUNCTION = '''
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        initial_state: torch.Tensor | None,
        scale: float,
        prebuilt_meta,
        fused_fwd,
        *,
        decision_only: bool | None = None,
        runtime=None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the A2/A3 Phase6 FLA NPU GDN prefill kernel.

        ``runtime`` is one forward's host-side switch snapshot; the caller
        passes the same object to every GDN layer so the data path cannot
        change between layers.  The decision path skips the final state and
        the backward-only intermediates.  Its data-path switches additionally
        drop the initial-state gather and transposes, the convolution state
        cache, and the Q/K/V copies that only exist to feed the BNSD layout.

        ``initial_state`` is optional.  A fresh full prefill has no history,
        so ``None`` tells the kernel to start from zero and skips the gather,
        the transposes, and the write-back.
        """
        from vllm.third_party.flash_linear_attention.ops.l2norm import l2norm_fwd
        from vllm_ascend.ops.gdn_decision_flags import read_flags

        if runtime is None:
            runtime = read_flags()
        if decision_only is None:
            decision_only = runtime.decision_prefill
        if not decision_only:
            # The extra skips are only defined for a batch that cannot carry a
            # state; keep the accepted v2 behaviour for every other caller.
            runtime = runtime.state_free()
        if initial_state is None and not decision_only:
            raise RuntimeError(
                "fla_gdn_prefill: initial_state=None requires decision_only"
            )
        skip_recompute = runtime.no_recompute

        if runtime.bsnd and q.dim() == 4 and q.dtype == torch.bfloat16 and v.dtype == torch.bfloat16:
            # The prefill convolution returns [H, T, D].  Normalizing that
            # head-first storage removes the pre-norm contiguous copy; only
            # the sequence-major operand of the BSND call needs a layout
            # change.  The l2norm math is unchanged: each row is one head's
            # token vector either way.
            q_head = q.transpose(1, 2)
            k_head = k.transpose(1, 2)
            v_head = v.transpose(1, 2)
            if not (
                q_head.is_contiguous()
                and k_head.is_contiguous()
                and v_head.is_contiguous()
            ):
                q_head = q_head.contiguous()
                k_head = k_head.contiguous()
                v_head = v_head.contiguous()
            q = l2norm_fwd(q_head).transpose(1, 2)
            k = l2norm_fwd(k_head).transpose(1, 2)
            v = v_head.transpose(1, 2)
            if os.environ.get("CLEF_GDN_BSND_STRIDED", "0") != "1":
                # The stable ABI is documented for contiguous operand; the
                # experiment switch measures the strided view for the report.
                q = q.contiguous()
                k = k.contiguous()
                v = v.contiguous()
            g = g.to(torch.float32).contiguous()
            beta = beta.to(v.dtype).contiguous()
            cu_seqlens = prebuilt_meta.cu_seqlens_host
            chunk_indices = prebuilt_meta.chunk_indices_chunk64_host
            keep_meta = prebuilt_meta.keep_meta
            if keep_meta is not None:
                cu_seqlens = prebuilt_meta.cu_seqlens_kern
            initial_state_kern = None
            if initial_state is not None:
                initial_state = initial_state.contiguous()
                if keep_meta is not None:
                    initial_state_kern = (
                        initial_state[keep_meta].transpose(-1, -2).contiguous()
                    )
                else:
                    initial_state_kern = (
                        initial_state.transpose(-1, -2).contiguous()
                    )
            output, final_state, *_ = fused_fwd(
                q,
                k,
                v,
                g,
                beta,
                initial_state=initial_state_kern,
                output_final_state=not decision_only,
                chunk_size=64,
                cu_seqlens=cu_seqlens,
                chunk_indices=chunk_indices,
                scale=scale,
                layout="BSND",
                use_exp2=False,
                use_qk_l2norm_in_kernel=False,
                allow_neg_eigval=False,
                disable_recompute=not skip_recompute,
                state_v_first=False,
            )
            if decision_only:
                return output, None
            if keep_meta is not None:
                # Scatter in the kernel layout so empty segments keep their
                # initial state, then flip back to the caller's [.., Dv, Dk].
                full_final_state = initial_state.transpose(-1, -2).contiguous()
                full_final_state[keep_meta] = final_state
                final_state = full_final_state
            return output, final_state.transpose(-1, -2).contiguous()

        q = l2norm_fwd(q.contiguous()).contiguous()
        k = l2norm_fwd(k.contiguous()).contiguous()
        v = v.contiguous()
        g = g.to(torch.float32).contiguous()
        beta = beta.to(v.dtype).contiguous()
        cu_seqlens = prebuilt_meta.cu_seqlens_host
        chunk_indices = prebuilt_meta.chunk_indices_chunk64_host
        keep_meta = prebuilt_meta.keep_meta
        if keep_meta is not None:
            cu_seqlens = prebuilt_meta.cu_seqlens_kern
        initial_state_kern = None
        if initial_state is not None:
            initial_state = initial_state.contiguous()
            # stateVFirst=false takes [N, Hv, K, V], i.e. the transpose of
            # ssm_state.
            if keep_meta is not None:
                initial_state_kern = (
                    initial_state[keep_meta].transpose(-1, -2).contiguous()
                )
            else:
                initial_state_kern = initial_state.transpose(-1, -2).contiguous()

        # BNSD/NTD take [B, H, T, D]; q/k/v arrive sequence-major. The output
        # is still returned sequence-major, so only the inputs are permuted.
        q = q.transpose(1, 2).contiguous()
        k = k.transpose(1, 2).contiguous()
        v = v.transpose(1, 2).contiguous()

        output, final_state, *_ = fused_fwd(
            q,
            k,
            v,
            g,
            beta,
            initial_state=initial_state_kern,
            output_final_state=not decision_only,
            chunk_size=64,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            scale=scale,
            layout="BNSD",
            use_exp2=False,
            use_qk_l2norm_in_kernel=False,
            allow_neg_eigval=False,
            disable_recompute=not skip_recompute,
            state_v_first=False,
        )
        if decision_only:
            return output, None
        if keep_meta is not None:
            # Scatter in the kernel layout so empty segments keep their initial state,
            # then flip back to the caller's [.., Dv, Dk].
            full_final_state = initial_state.transpose(-1, -2).contiguous()
            full_final_state[keep_meta] = final_state
            final_state = full_final_state
        return output, final_state.transpose(-1, -2).contiguous()
'''


SIGNATURE_ANCHOR_V3 = (
    "        prebuilt_meta,\n"
    "        fused_fwd,\n"
    "    ) -> tuple[torch.Tensor, torch.Tensor]:\n"
)
SIGNATURE_PATCHED_V3 = (
    "        prebuilt_meta,\n"
    "        fused_fwd,\n"
    "        *,\n"
    "        decision_only: bool | None = None,\n"
    "        runtime=None,\n"
    "    ) -> tuple[torch.Tensor, torch.Tensor]:\n"
)


def patch_device_op(text: str) -> str:
    """Replace the A3 wrapper and let the A5 signature accept the new kwargs."""
    parts = text.split(DEF_MARKER, 2)
    if len(parts) != 3:
        raise SystemExit("device_op.py: expected two fla_gdn_prefill definitions")
    head, a3, tail = parts
    end = a3.find("\n    @staticmethod\n", len(DEF_MARKER))
    if end < 0:
        raise SystemExit("device_op.py: A3 function end anchor is missing")
    if a3[:end].count("layout=\"BSND\"") < 1 or a3[:end].count("layout=\"BNSD\"") < 1:
        raise SystemExit("device_op.py: A3 body does not look like the accepted preimage")
    a3 = A3_FUNCTION + a3[end:]
    if tail.count(SIGNATURE_ANCHOR_V3) != 1:
        raise SystemExit("device_op.py: A5 signature anchor is not unique")
    tail = tail.replace(SIGNATURE_ANCHOR_V3, SIGNATURE_PATCHED_V3, 1)
    return head + DEF_MARKER + a3 + DEF_MARKER + tail


INIT_ANCHOR = (
    "    @wraps(_GDN_ORIGINAL_INIT)\n"
    "    def _gdn_init_with_packed_weight(self, *args, **kwargs):\n"
    "        _GDN_ORIGINAL_INIT(self, *args, **kwargs)\n"
    "        initialize_packed_conv_weight(self)\n"
)
INIT_PATCHED = (
    INIT_ANCHOR
    + "        # Decision-only prefill specialization: the engine config decides\n"
    + "        # whether a prefill can carry a state a later step reads.  Record\n"
    + "        # it here because this wrapper is the __init__ the model runs.\n"
    + "        vllm_config = kwargs.get(\"vllm_config\")\n"
    + "        if vllm_config is None and len(args) > 1:\n"
    + "            vllm_config = args[1]\n"
    + "        self._decision_safe_config = bool(\n"
    + "            vllm_config is not None\n"
    + "            and vllm_config.model_config.runner_type == \"pooling\"\n"
    + "            and not vllm_config.scheduler_config.enable_chunked_prefill\n"
    + "            and not vllm_config.cache_config.enable_prefix_caching\n"
    + "            and not vllm_config.speculative_config\n"
    + "        )\n"
)


def patch_qwen3_5(text: str) -> str:
    if text.count(INIT_ANCHOR) != 1:
        raise SystemExit("patch_qwen3_5.py: init anchor is not unique; refusing to patch")
    return text.replace(INIT_ANCHOR, INIT_PATCHED, 1) + MLP_NZ_BLOCK


RUNTIME_HELPER = '''

_PATH_LOG = {"epoch": None}


def _decision_prefill_guard(decision_safe_config, attn_metadata, flags) -> bool:
    """True only for a batch that cannot consume a recurrent state.

    The engine recorded at construction time that chunked prefill, prefix
    caching and speculative decoding are off, so a prefill batch with no
    decode token cannot carry a state a later step reads.  The check is
    host-side: reading ``prefill_has_initial_state`` would sync the device
    once per layer and cost more than the state write it saves.
    """
    return bool(
        decision_safe_config
        and flags.decision_prefill
        and attn_metadata.num_prefills > 0
        and attn_metadata.num_decodes == 0
        and attn_metadata.num_decode_tokens == 0
        and attn_metadata.num_spec_decodes == 0
        and attn_metadata.num_spec_decode_tokens == 0
        and get_pcp_group().world_size == 1
    )


def _gdn_runtime_flags(decision_safe_config, attn_metadata):
    """One forward's host-side switch snapshot, or None when unsafe.

    The snapshot is keyed by the vLLM forward context, so every GDN layer of
    one request reads the same values.  A new experiment epoch is logged once
    per worker process; the driver counts those lines to prove both TP ranks
    adopted the switch before a measurement starts.
    """
    import os

    from vllm_ascend.ops.gdn_decision_flags import snapshot_for

    flags = snapshot_for(get_forward_context())
    if not _decision_prefill_guard(decision_safe_config, attn_metadata, flags):
        return None
    if _PATH_LOG["epoch"] != flags.epoch:
        from vllm.logger import init_logger

        init_logger(__name__).info(
            "clef-gdn-path epoch=%d pid=%d bsnd=%d skip_initial_state=%d "
            "skip_conv_state=%d preprocess_fusion=%d",
            flags.epoch,
            os.getpid(),
            int(flags.bsnd),
            int(flags.skip_initial_state),
            int(flags.skip_conv_state),
            int(flags.preprocess_fusion),
        )
        _PATH_LOG["epoch"] = flags.epoch
    return flags
'''

RUNTIME_INSERT_ANCHOR = "    return chunk_gated_delta_rule_fwd\n"

RUNTIME_CALL_ANCHOR = "        num_actual_tokens = attn_metadata.num_actual_tokens\n"
RUNTIME_CALL_PATCHED = (
    RUNTIME_CALL_ANCHOR
    + "\n"
    + "        runtime = _gdn_runtime_flags(\n"
    + "            getattr(self, \"_decision_safe_config\", False), attn_metadata\n"
    + "        )\n"
)

CONV_CALL_V3 = (
    "                    mixed_qkv_non_spec_output = causal_conv1d_fn(\n"
    "                        mixed_qkv_non_spec,\n"
    "                        conv_weights_T,\n"
    "                        self.conv1d.bias,\n"
    "                        conv_states=self_kv_cache[0],\n"
    "                        query_start_loc=query_start_loc_opt,\n"
    "                        cache_indices=_normalize_causal_cache_indices(cache_indices_opt),\n"
    "                        has_initial_state=initial_state_mode_opt,\n"
    "                        activation=\"silu\" if self.activation else None,\n"
    "                        pad_slot_id=PAD_SLOT_ID,\n"
    "                        null_block_id=0,\n"
    "                        head_num=conv_head_num,\n"
    "                    )\n"
)
CONV_TAIL_V3 = (
    "                    mixed_qkv_non_spec = mixed_qkv_non_spec_output\n"
    "        elif attn_metadata.num_decodes > 0:\n"
)
CONV_ANCHOR_V3 = CONV_CALL_V3 + CONV_TAIL_V3
CONV_PATCHED_V3 = (
    "                    if runtime is not None and runtime.skip_conv_state:\n"
    "                        # No decode step will read the convolution cache,\n"
    "                        # so the prefill starts from left zero padding and\n"
    "                        # skips both the cache read and the tail write.\n"
    "                        mixed_qkv_non_spec_output = causal_conv1d_fn(\n"
    "                            mixed_qkv_non_spec,\n"
    "                            conv_weights_T,\n"
    "                            self.conv1d.bias,\n"
    "                            conv_states=None,\n"
    "                            query_start_loc=query_start_loc_opt,\n"
    "                            cache_indices=None,\n"
    "                            has_initial_state=None,\n"
    "                            activation=\"silu\" if self.activation else None,\n"
    "                            pad_slot_id=PAD_SLOT_ID,\n"
    "                            null_block_id=0,\n"
    "                            head_num=conv_head_num,\n"
    "                        )\n"
    "                    else:\n"
    + "".join("    " + line + "\n" for line in CONV_CALL_V3.splitlines())
    + CONV_TAIL_V3
)

FUSED_ANCHOR_V3 = (
    "            if fla_gdn_prefill_op is not None:\n"
    "                initial_state = gather_ssm_states(\n"
    "                    ssm_state,\n"
    "                    prefill_state_indices,\n"
    "                    prefill_has_initial_state,\n"
    "                )\n"
)
FUSED_PATCHED_V3 = (
    "            if fla_gdn_prefill_op is not None:\n"
    "                if runtime is not None and runtime.skip_initial_state:\n"
    "                    # A fresh full prefill carries no history: the kernel\n"
    "                    # starts from zero, so the gather and its transposes\n"
    "                    # are dead work.\n"
    "                    initial_state = None\n"
    "                else:\n"
    "                    initial_state = gather_ssm_states(\n"
    "                        ssm_state,\n"
    "                        prefill_state_indices,\n"
    "                        prefill_has_initial_state,\n"
    "                    )\n"
)

SCATTER_ANCHOR_V3 = (
    "                    fused_fwd=fla_gdn_prefill_op,\n"
    "                )\n"
    "                scatter_ssm_states_(\n"
    "                    ssm_state,\n"
    "                    prefill_state_indices,\n"
    "                    last_recurrent_state.to(ssm_state.dtype).contiguous(),\n"
    "                )\n"
)
SCATTER_PATCHED_V3 = (
    "                    fused_fwd=fla_gdn_prefill_op,\n"
    "                    decision_only=runtime is not None,\n"
    "                    runtime=runtime,\n"
    "                )\n"
    "                if last_recurrent_state is not None:\n"
    "                    scatter_ssm_states_(\n"
    "                        ssm_state,\n"
    "                        prefill_state_indices,\n"
    "                        last_recurrent_state.to(ssm_state.dtype).contiguous(),\n"
    "                    )\n"
)


def patch_gdn(text: str) -> str:
    if text.count(RUNTIME_INSERT_ANCHOR) != 1:
        raise SystemExit("gdn.py: helper insertion point is not unique")
    if text.count(RUNTIME_CALL_ANCHOR) != 1:
        raise SystemExit("gdn.py: runtime call anchor is not unique")
    if text.count(CONV_ANCHOR_V3) != 1:
        raise SystemExit("gdn.py: prefill convolution anchor is not unique")
    if text.count(FUSED_ANCHOR_V3) != 1 or text.count(SCATTER_ANCHOR_V3) != 1:
        raise SystemExit("gdn.py: fused prefill anchors are not unique")
    if text.count("def _get_fla_gdn_prefill_op():") != 1:
        raise SystemExit("gdn.py: helper anchor is not unique; refusing to patch")
    text = text.replace(RUNTIME_INSERT_ANCHOR, RUNTIME_INSERT_ANCHOR + RUNTIME_HELPER, 1)
    text = text.replace(RUNTIME_CALL_ANCHOR, RUNTIME_CALL_PATCHED, 1)
    text = text.replace(CONV_ANCHOR_V3, CONV_PATCHED_V3, 1)
    text = text.replace(FUSED_ANCHOR_V3, FUSED_PATCHED_V3, 1)
    text = text.replace(SCATTER_ANCHOR_V3, SCATTER_PATCHED_V3, 1)
    return text


def write_atomic(path: Path, data: bytes) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{time.time_ns()}")
    try:
        temporary.write_bytes(data)
        if path.is_file():
            temporary.chmod(path.stat().st_mode & 0o777)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def flag_status(path: Path) -> str:
    if not path.exists():
        return "missing"
    if not path.is_file():
        return "unknown"
    digest = sha256_file(path)
    if digest == FLAG_SHA256:
        return "current"
    if digest == LEGACY_FLAG_SHA256:
        return "legacy"
    if digest == LEGACY_FLAG_SHA256_V1:
        return "legacy-v1"
    return "unknown"


def refresh_report(report: dict, tree: Path, *, after: bool = False) -> None:
    state_key = "status_after" if after else "status_before"
    for relative in PREIMAGE_SHA:
        path = tree / relative
        item = report["files"].setdefault(relative, {})
        item.update({
            state_key: status(path, relative),
            "sha256": sha256_file(path) if path.is_file() else None,
            "preimage_sha256": PREIMAGE_SHA[relative],
            "postimage_sha256": POSTIMAGE_SHA[relative],
            "v2_postimage_sha256": V2_POSTIMAGE_SHA.get(relative),
            "v1_postimage_sha256": V1_POSTIMAGE_SHA.get(relative),
            "adaptive_postimage_sha256": ADAPTIVE_POSTIMAGE_SHA.get(relative),
        })
    flag_target = tree / FLAG_TARGET
    report.setdefault("flag_module", {}).update({
        "path": str(flag_target),
        "present": flag_target.is_file(),
        "status": flag_status(flag_target),
        "sha256": sha256_file(flag_target) if flag_target.is_file() else None,
        "source_sha256": sha256_file(FLAG_SOURCE) if FLAG_SOURCE.is_file() else None,
        "expected_sha256": FLAG_SHA256,
    })
    report["ready"] = (
        all(item[state_key] in ("patched", "qkv", "conv", "chunk128", "adaptive")
            for item in report["files"].values())
        and report["flag_module"]["status"] == "current"
        and report["flag_module"]["source_sha256"] == FLAG_SHA256
    )


def plan_changes(report: dict, tree: Path, backup_dir: Path, *, revert: bool) -> dict:
    """Validate every artifact and produce all bytes before changing any file."""
    writes = {}
    source = report["flag_module"]
    if source["status"] == "unknown":
        raise ValueError("flag module has unknown content; preserve and inspect it before retrying")
    if not revert and source["source_sha256"] != FLAG_SHA256:
        raise ValueError("installer flag source SHA-256 does not match this patch version")
    patchers = {
        "device_op.py": patch_device_op,
        "gdn.py": patch_gdn,
        "patch_qwen3_5.py": patch_qwen3_5,
    }
    for relative, item in report["files"].items():
        path = tree / relative
        current = item["status_before"]
        if current not in ("preimage", "patched", "qkv", "conv", "chunk128",
                           "adaptive", "v2", "v3", "legacy"):
            raise ValueError(f"{relative}: refused {current} source content")
        backup = backup_dir / f"{path.name}.preimage"
        backup_data = None
        if backup.exists():
            if not backup.is_file() or sha256_file(backup) != PREIMAGE_SHA[relative]:
                raise ValueError(f"{relative}: preimage backup SHA-256 mismatch: {backup}")
            backup_data = backup.read_bytes()
        item["backup"] = str(backup)
        if revert:
            if current == "preimage":
                item["action"] = "already-preimage"
                continue
            if backup_data is None:
                raise ValueError(f"{relative}: verified preimage backup required: {backup}")
            writes[path] = backup_data
            item["action"] = "reverted"
        else:
            if current in ("patched", "qkv", "conv", "chunk128", "adaptive"):
                item["action"] = "already-patched"
                continue
            if current == "v3":
                # The v3 postimage already contains the GDN decision patch;
                # the v4 upgrade only appends the MLP NZ hook.
                patched = (path.read_text() + MLP_NZ_BLOCK).encode("utf-8")
                if sha256_bytes(patched) != POSTIMAGE_SHA[relative]:
                    raise ValueError(f"{relative}: v3 upgrade SHA-256 mismatch")
                writes[path] = patched
                item["action"] = "upgraded"
                continue
            if current in ("v2", "legacy"):
                if backup_data is None:
                    raise ValueError(
                        f"{relative}: {current} upgrade requires a verified "
                        f"preimage backup: {backup}")
                preimage = backup_data
                item["action"] = "upgraded"
            else:
                preimage = path.read_bytes()
                if backup_data is None:
                    writes[backup] = preimage
                item["action"] = "patched"
            patched = patchers[path.name](preimage.decode("utf-8")).encode("utf-8")
            if sha256_bytes(patched) != POSTIMAGE_SHA[relative]:
                raise ValueError(f"{relative}: generated postimage SHA-256 mismatch")
            writes[path] = patched
    flag_target = tree / FLAG_TARGET
    if revert:
        if source["status"] != "missing":
            writes[flag_target] = None
            source["action"] = "removed"
        else:
            source["action"] = "already-absent"
    elif source["status"] != "current":
        writes[flag_target] = FLAG_SOURCE.read_bytes()
        source["action"] = "installed"
    else:
        source["action"] = "already-current"
    return writes


def commit_changes(writes: dict) -> None:
    """Roll back completed replacements if a filesystem operation fails."""
    originals = {
        path: path.read_bytes() if path.is_file() else None for path in writes
    }
    completed = []
    try:
        for path, data in writes.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            if data is None:
                path.unlink()
            else:
                write_atomic(path, data)
            completed.append(path)
    except OSError as exc:
        rollback_errors = []
        for path in reversed(completed):
            try:
                if originals[path] is None:
                    path.unlink(missing_ok=True)
                else:
                    write_atomic(path, originals[path])
            except OSError as rollback_exc:
                rollback_errors.append(f"{path}: {rollback_exc}")
        detail = f"; rollback errors: {rollback_errors}" if rollback_errors else "; files restored"
        raise OSError(f"installation write failed: {exc}{detail}") from exc


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tree", type=Path, default=DEFAULT_TREE)
    parser.add_argument("--backup-dir", type=Path, default=DEFAULT_BACKUP)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--revert", action="store_true")
    parser.add_argument("--manifest", type=Path, default=None)
    args = parser.parse_args(argv)
    if sum((args.check, args.apply, args.revert)) != 1:
        parser.error("choose exactly one of --check, --apply, --revert")
    tree = args.tree.expanduser().resolve()
    backup_dir = args.backup_dir.expanduser().resolve()
    recovery = [
        shlex.join([
            sys.executable, str(Path(__file__).resolve()), "--tree", str(tree),
            "--backup-dir", str(backup_dir), action,
        ])
        for action in ("--revert", "--apply")
    ]
    report = {
        "tree": str(tree),
        "patch_version": PATCH_VERSION,
        "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "action": "check" if args.check else "apply" if args.apply else "revert",
        "files": {},
        "recovery_commands": recovery,
    }
    result = 0
    # Lock the existing tree directory: check remains read-only and concurrent
    # installer invocations cannot interleave their preflight and writes.
    try:
        tree_fd = os.open(tree, os.O_RDONLY)
        try:
            fcntl.flock(tree_fd, fcntl.LOCK_SH if args.check else fcntl.LOCK_EX)
            refresh_report(report, tree)
            if not args.check:
                try:
                    writes = plan_changes(report, tree, backup_dir, revert=args.revert)
                    commit_changes(writes)
                except (ValueError, OSError, SystemExit) as exc:
                    report["error"] = str(exc)
                    result = 1
                refresh_report(report, tree, after=True)
        finally:
            os.close(tree_fd)
    except OSError as exc:
        report["error"] = str(exc)
        result = 1
    if args.manifest and not args.check:
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return result


if __name__ == "__main__":
    raise SystemExit(main())
