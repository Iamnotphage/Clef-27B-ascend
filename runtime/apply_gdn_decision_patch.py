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
PATCH_VERSION = 2
FLAG_SHA256 = "cf1ede3f3e8a894d9f0896a35cd07254ffa756e18e699fa40636cea86043fc70"
LEGACY_FLAG_SHA256 = "02987b37857ffad0ab259fc55fad6bb1f02d39c885664b36353572057b98c731"

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
        "ad92a322a69dbd023cd88603b5882d0c5f7f931dbf20692957efdddd6f7809d2",
    "vllm_ascend/ops/gdn.py":
        "509dafb03b12351ebb17854742681b4aede60958a820d7625af94467c3586f43",
    "vllm_ascend/patch/worker/patch_qwen3_5.py":
        "1b82b52eabe24d8fcdcb55ff16c307a9b04358ba846a4d73ebb4f1443c0b0460",
}
LEGACY_POSTIMAGE_SHA = {
    "vllm_ascend/patch/worker/patch_qwen3_5.py":
        "d2666387506cb68d9238c6f06baa89253158ee0facc67ad1243cf41ad01edcad",
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
    digest = sha256_file(path)
    if digest == PREIMAGE_SHA[relative]:
        return "preimage"
    if digest == POSTIMAGE_SHA[relative]:
        return "patched"
    if digest == LEGACY_POSTIMAGE_SHA.get(relative):
        return "legacy"
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
        all(item[state_key] == "patched" for item in report["files"].values())
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
        if current not in ("preimage", "patched", "legacy"):
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
            if current == "patched":
                item["action"] = "already-patched"
                continue
            if current == "legacy":
                if backup_data is None:
                    raise ValueError(
                        f"{relative}: legacy upgrade requires a verified preimage backup: {backup}")
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
