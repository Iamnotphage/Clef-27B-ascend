#!/usr/bin/env python3
"""Install or revert the decision-only GDN prefill switches.

The running service uses an editable install of the pinned vLLM-Ascend tree
(``89eb8a5``).  This installer owns three artifacts:

1. ``vllm_ascend/ops/gdn_decision_flags.py`` — new flag module, copied from
   ``runtime/gdn_decision_flags.py``.
2. ``vllm_ascend/device/device_op.py`` — the A3 ``fla_gdn_prefill`` wrapper
   stops requesting the final state and the backward-only intermediates when
   the switches are on.
3. ``vllm_ascend/ops/gdn.py`` — the prefill branch skips the state write-back
   when the wrapper returns no final state.

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
import hashlib
import json
import shutil
import time
from pathlib import Path

DEFAULT_TREE = Path("/workspace/yangchen/work/src/vllm-ascend")
DEFAULT_BACKUP = Path("/workspace/yangchen/logs/gdn-decision-20261115/phase0")
FLAG_SOURCE = Path(__file__).with_name("gdn_decision_flags.py")
FLAG_TARGET = "vllm_ascend/ops/gdn_decision_flags.py"

PREIMAGE_SHA = {
    "vllm_ascend/device/device_op.py":
        "a08d44b41b337d839548c02b7457510968d5c5057ece01243b022a91348a0823",
    "vllm_ascend/ops/gdn.py":
        "b235e34d8a2427b167d4ce0e9068312904ecd1473fe288cefe1e9b6f95de030d",
    "vllm_ascend/patch/worker/patch_qwen3_5.py":
        "a9a71634ed138f590fbc3ad0de277a5f56a06451522fd091b2657d6daf6084b0",
}
PATCH_MARKER = {
    "vllm_ascend/device/device_op.py": "decision_prefill_enabled",
    "vllm_ascend/ops/gdn.py": "if last_recurrent_state is not None:",
    "vllm_ascend/patch/worker/patch_qwen3_5.py": "_decision_safe_config",
}

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
    text = path.read_text(encoding="utf-8")
    if sha256_file(path) == PREIMAGE_SHA[relative]:
        return "preimage"
    if PATCH_MARKER[relative] in text:
        return "patched"
    return "unknown"


def patch_device_op(text: str) -> str:
    parts = text.split(DEF_MARKER, 2)
    if len(parts) != 3:
        raise SystemExit("device_op.py: expected two fla_gdn_prefill definitions")
    head, a3, tail = parts
    if a3.count("output_final_state=True,") != 2:
        raise SystemExit("device_op.py: expected two output_final_state=True in the A3 wrapper")
    if a3.count("disable_recompute=True,") != 2:
        raise SystemExit("device_op.py: expected two disable_recompute=True in the A3 wrapper")
    if a3.count(IMPORT_ANCHOR) != 1 or a3.count(BSND_ANCHOR) != 1 or a3.count(BNSD_ANCHOR) != 1:
        raise SystemExit("device_op.py: anchors are not unique; refusing to patch")
    if a3.count(SIGNATURE_ANCHOR) != 1:
        raise SystemExit("device_op.py: A3 signature anchor is not unique")
    a3 = a3.replace(SIGNATURE_ANCHOR, SIGNATURE_PATCHED, 1)
    a3 = a3.replace(IMPORT_ANCHOR, IMPORT_ADDED, 1)
    if a3.count(DECISION_INIT_ANCHOR) != 1:
        raise SystemExit("device_op.py: decision init anchor is not unique")
    a3 = a3.replace(DECISION_INIT_ANCHOR, DECISION_INIT_PATCHED, 1)
    a3 = a3.replace(
        "output_final_state=True,", "output_final_state=not decision_only,", 2
    )
    a3 = a3.replace(
        "disable_recompute=True,", "disable_recompute=not skip_recompute,", 2
    )
    a3 = a3.replace(BSND_ANCHOR, BSND_PATCHED, 1)
    a3 = a3.replace(BNSD_ANCHOR, BNSD_PATCHED, 1)
    if tail.count(SIGNATURE_ANCHOR) != 1:
        raise SystemExit("device_op.py: A5 signature anchor is not unique")
    tail = tail.replace(SIGNATURE_ANCHOR, SIGNATURE_PATCHED, 1)
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
    return text.replace(INIT_ANCHOR, INIT_PATCHED, 1)


def patch_gdn(text: str) -> str:
    if text.count(SCATTER_LATE) != 1 or text.count(SCATTER_FUSED) != 1:
        raise SystemExit("gdn.py: scatter anchors are not unique; refusing to patch")
    if text.count(GUARD_ANCHOR) != 1 or text.count(KWARG_ANCHOR) != 1:
        raise SystemExit("gdn.py: guard anchors are not unique; refusing to patch")
    if text.count("def _get_fla_gdn_prefill_op():") != 1:
        raise SystemExit("gdn.py: helper anchor is not unique; refusing to patch")
    text = text.replace(SCATTER_LATE, SCATTER_LATE_PATCHED, 1)
    text = text.replace(SCATTER_FUSED, SCATTER_FUSED_PATCHED, 1)
    text = text.replace(GUARD_ANCHOR, GUARD_PATCHED, 1)
    text = text.replace(KWARG_ANCHOR, KWARG_PATCHED, 1)
    marker = "    return chunk_gated_delta_rule_fwd\n"
    if text.count(marker) != 1:
        raise SystemExit("gdn.py: helper insertion point is not unique")
    text = text.replace(marker, marker + GUARD_HELPER, 1)
    return text


def write_atomic(path: Path, data: bytes) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{time.time_ns()}")
    temporary.write_bytes(data)
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tree", type=Path, default=DEFAULT_TREE)
    parser.add_argument("--backup-dir", type=Path, default=DEFAULT_BACKUP)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--revert", action="store_true")
    parser.add_argument("--manifest", type=Path, default=None)
    args = parser.parse_args()
    if sum((args.check, args.apply, args.revert)) != 1:
        parser.error("choose exactly one of --check, --apply, --revert")
    tree = args.tree.expanduser().resolve()
    report = {
        "tree": str(tree),
        "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "action": "check" if args.check else "apply" if args.apply else "revert",
        "files": {},
    }
    targets = {
        relative: tree / relative
        for relative in PREIMAGE_SHA
    }
    for relative, path in targets.items():
        report["files"][relative] = {
            "status_before": status(path, relative),
            "sha256": sha256_file(path) if path.is_file() else None,
            "preimage_sha256": PREIMAGE_SHA[relative],
        }
    flag_target = tree / FLAG_TARGET
    report["flag_module"] = {
        "path": str(flag_target),
        "present": flag_target.is_file(),
        "sha256": sha256_file(flag_target) if flag_target.is_file() else None,
        "source_sha256": sha256_file(FLAG_SOURCE),
    }

    if args.check:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0

    if args.revert:
        for relative, path in targets.items():
            if status(path, relative) == "preimage":
                report["files"][relative]["action"] = "already-preimage"
                continue
            if status(path, relative) != "patched":
                report["files"][relative]["action"] = "refused-unknown-state"
                print(json.dumps(report, ensure_ascii=False, indent=2))
                return 1
            backup = args.backup_dir / f"{Path(relative).name}.preimage"
            if not backup.is_file() or sha256_file(backup) != PREIMAGE_SHA[relative]:
                report["files"][relative]["action"] = "refused-missing-or-wrong-backup"
                print(json.dumps(report, ensure_ascii=False, indent=2))
                return 1
            write_atomic(path, backup.read_bytes())
            report["files"][relative]["action"] = "reverted"
            report["files"][relative]["sha256"] = sha256_file(path)
            report["files"][relative]["status_after"] = status(path, relative)
        if flag_target.is_file():
            if sha256_file(flag_target) == sha256_file(FLAG_SOURCE):
                flag_target.unlink()
                report["flag_module"]["action"] = "removed"
            else:
                report["flag_module"]["action"] = "refused-modified"
                print(json.dumps(report, ensure_ascii=False, indent=2))
                return 1
        else:
            report["flag_module"]["action"] = "already-absent"
    else:
        # apply
        for relative, path in targets.items():
            current = status(path, relative)
            if current == "patched":
                report["files"][relative]["action"] = "already-patched"
                continue
            if current != "preimage":
                report["files"][relative]["action"] = "refused-unexpected-content"
                print(json.dumps(report, ensure_ascii=False, indent=2))
                return 1
            backup = args.backup_dir / f"{Path(relative).name}.preimage"
            backup.parent.mkdir(parents=True, exist_ok=True)
            if not backup.is_file():
                backup.write_bytes(path.read_bytes())
            text = path.read_text(encoding="utf-8")
            if relative.endswith("device_op.py"):
                patched = patch_device_op(text)
            elif relative.endswith("patch_qwen3_5.py"):
                patched = patch_qwen3_5(text)
            else:
                patched = patch_gdn(text)
            write_atomic(path, patched.encode("utf-8"))
            report["files"][relative]["action"] = "patched"
            report["files"][relative]["sha256"] = sha256_file(path)
            report["files"][relative]["status_after"] = status(path, relative)
            report["files"][relative]["backup"] = str(backup)
        flag_target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(FLAG_SOURCE, flag_target)
        report["flag_module"]["action"] = "installed"
        report["flag_module"]["sha256"] = sha256_file(flag_target)

    if args.manifest:
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
