#!/usr/bin/env python3
"""Install the opt-in single-sequence Conv+QKV experimental service path.

Apply after the verified GDN/QKV installers.  The experimental vendor must be
in ASCEND_CUSTOM_OPP_PATH before torch_npu is imported.  This installer never
changes the vendor, launcher, environment, or model weights.
"""

from __future__ import annotations

import argparse
import ast
import fcntl
import hashlib
import json
import os
from pathlib import Path
import stat
import tempfile

DEFAULT_TREE = Path("/workspace/yangchen/work/src/vllm-ascend")
DEFAULT_BACKUP = Path("/workspace/yangchen/logs/gdn-conv-qkv-20261117/phase0")
TARGET = "vllm_ascend/ops/gdn.py"
PATCH_VERSION = 2
PREIMAGE_SHA = "c8c0d4d8a693658e075a2232b6938584760672bacf101bf50ca5fdfab2c0be7e"
POSTIMAGE_SHA = "ead25cb2691cb9ab1c322353fc8e168f8bfc9c23b9e4d53de01c762943027e78"
# The varlen installer stacks on POSTIMAGE; both count as an installed v4.
VARLEN_POSTIMAGE_SHA = "afe6d4e87c0eb47bfb325fb630620e47f845fb5a893cb1dada78069ad4e3ca07"

HELPER = '''
_CONV_QKV_RUNTIME = None
_CONV_QKV_VENDOR = None


def _run_conv_qkv(x, weight, *, runtime, attn_metadata, conv_head_num,
                  tp_size, num_k_heads, num_v_heads, head_k_dim,
                  head_v_dim, bias, activation):
    """Return normalized BSND q/k/v only for the validated fresh C1 path.

    Eligibility failures return None.  Once an eligible experimental request
    selects the custom op, loader and execution errors propagate to the caller
    so benchmark evidence cannot silently measure the fallback.
    """
    global _CONV_QKV_RUNTIME, _CONV_QKV_VENDOR
    if os.environ.get("CLEF_GDN_CONV_QKV", "0") != "1":
        return None
    if (runtime is None or not runtime.skip_conv_state
            or not runtime.preprocess_fusion or not runtime.bsnd
            or attn_metadata.num_prefills != 1 or attn_metadata.num_decodes != 0
            or getattr(attn_metadata, "num_spec_decodes", 0) != 0
            or getattr(attn_metadata, "spec_sequence_masks", None) is not None
            or tp_size != 2 or conv_head_num != 40
            or num_k_heads != 16 or num_v_heads != 48
            or head_k_dim != 128 or head_v_dim != 128
            or bias is not None or activation not in (True, "silu", "swish")):
        return None
    if (x.dim() != 2 or x.shape[1] != 5120
            or x.dtype != torch.bfloat16 or x.device.type not in ("npu", "privateuseone")
            or weight.dim() != 2 or tuple(weight.shape) != (4, 5120)
            or not weight.is_contiguous() or weight.dtype != torch.bfloat16
            or weight.device != x.device):
        return None
    tokens = x.shape[0]
    if tokens <= 0 or tokens > int(os.environ.get("CLEF_GDN_QKV_NORM_MAX_TOKENS", "1200")):
        return None
    if not x.is_contiguous():
        # Qwen3.5 splits a contiguous [T,8192] QKVZ projection into QKV
        # [T,5120] and Z views. Expose the full packed rows without copying;
        # the fused kernel reads their first 5120 channels and leaves Z alone.
        if x.stride(1) != 1 or x.stride(0) != 8192:
            return None
        storage_end = x.storage_offset() + tokens * 8192
        if storage_end * x.element_size() > x.untyped_storage().nbytes():
            return None
        x = x.as_strided((tokens, 8192), (8192, 1))
    vendor = os.environ.get("CLEF_CONV_QKV_VENDOR")
    if not vendor:
        raise RuntimeError("CLEF_GDN_CONV_QKV=1 requires CLEF_CONV_QKV_VENDOR")
    tile = int(os.environ.get("CLEF_CONV_QKV_TILE", "32"))
    if tile not in (16, 32, 64):
        raise RuntimeError("CLEF_CONV_QKV_TILE must be 16, 32, or 64")
    import ctypes
    from fla_npu.ops.ascendc import _runtime as aclnn_runtime

    if _CONV_QKV_RUNTIME is None:
        library = ctypes.CDLL(
            f"{vendor}/op_api/lib/libcust_opapi.so",
            mode=getattr(os, "RTLD_LOCAL", 0) | getattr(os, "RTLD_NOW", 0)
            | getattr(os, "RTLD_NODELETE", 0),
        )
        _CONV_QKV_RUNTIME = aclnn_runtime.runtime()
        _CONV_QKV_RUNTIME._libraries.insert(0, library)
        _CONV_QKV_VENDOR = vendor
        print("clef-conv-qkv pid=%d tile=%d first_tokens=%d vendor=%s" % (
            os.getpid(), tile, tokens, vendor), flush=True)
    elif vendor != _CONV_QKV_VENDOR:
        raise RuntimeError("CLEF_CONV_QKV_VENDOR changed within a running worker")
    q = torch.empty((1, tokens, 8, 128), dtype=x.dtype, device=x.device)
    k = torch.empty_like(q)
    v = torch.empty((1, tokens, 24, 128), dtype=x.dtype, device=x.device)
    aclnn_runtime.call_aclnn(
        "aclnnClefGdnConvQkv",
        lambda ctx: [ctx.tensor(x, "x"), ctx.tensor(weight, "weight"),
                     ctypes.c_double(1e-6), ctypes.c_int64(tile),
                     ctx.tensor(q, "qOut"), ctx.tensor(k, "kOut"), ctx.tensor(v, "vOut")],
        (q, k, v),
        get_workspace_argtypes=[ctypes.c_void_p, ctypes.c_void_p,
                                ctypes.c_double, ctypes.c_int64,
                                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                                ctypes.POINTER(ctypes.c_uint64),
                                ctypes.POINTER(ctypes.c_void_p)],
    )
    return q, k, v


'''

INIT_ANCHOR = "        # 1. Convolution sequence transformation\n"
HELPER_ANCHOR = "def _rearrange_decode_qkv(\n"
CONV_ANCHOR = '''                        mixed_qkv_non_spec_output = causal_conv1d_fn(
                            mixed_qkv_non_spec,
                            conv_weights_T,
                            self.conv1d.bias,
                            conv_states=None,
                            query_start_loc=query_start_loc_opt,
                            cache_indices=None,
                            has_initial_state=None,
                            activation="silu" if self.activation else None,
                            pad_slot_id=PAD_SLOT_ID,
                            null_block_id=0,
                            head_num=conv_head_num,
                        )
'''
CONV_CALL = '''                        conv_qkv = _run_conv_qkv(
                            mixed_qkv_non_spec, conv_weights_T,
                            runtime=runtime, attn_metadata=attn_metadata,
                            conv_head_num=conv_head_num, tp_size=self.tp_size,
                            num_k_heads=self.num_k_heads, num_v_heads=self.num_v_heads,
                            head_k_dim=self.head_k_dim, head_v_dim=self.head_v_dim,
                            bias=self.conv1d.bias, activation=self.activation,
                        )
                        if conv_qkv is None:
'''
CONV_RESULT = '''                        else:
                            # The fused tuple is consumed below; retain this
                            # token-major input only as the non-None sentinel.
                            mixed_qkv_non_spec_output = mixed_qkv_non_spec
'''
QKV_ANCHOR = '''        qkv_pre_normalized = False
        if attn_metadata.num_prefills > 0 and mixed_qkv_non_spec is not None and conv_head_num:
'''
QKV_RESULT = '''        qkv_pre_normalized = False
        if conv_qkv is not None:
            query_non_spec, key_non_spec, value_non_spec = conv_qkv
            qkv_pre_normalized = True
        elif attn_metadata.num_prefills > 0 and mixed_qkv_non_spec is not None and conv_head_num:
'''


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def patch_gdn(text: str) -> str:
    if digest(text.encode()) != PREIMAGE_SHA:
        raise ValueError("gdn.py: verified QKV preimage required")
    for name, anchor in (("helper", HELPER_ANCHOR), ("init", INIT_ANCHOR),
                         ("convolution", CONV_ANCHOR), ("qkv", QKV_ANCHOR)):
        if text.count(anchor) != 1:
            raise ValueError(f"gdn.py: {name} anchor is not unique")
    text = text.replace(HELPER_ANCHOR, HELPER + HELPER_ANCHOR, 1)
    text = text.replace(INIT_ANCHOR, "        conv_qkv = None\n" + INIT_ANCHOR, 1)
    indented_conv = "".join("    " + line if line.strip() else line
                            for line in CONV_ANCHOR.splitlines(keepends=True))
    text = text.replace(CONV_ANCHOR, CONV_CALL + indented_conv + CONV_RESULT, 1)
    text = text.replace(QKV_ANCHOR, QKV_RESULT, 1)
    ast.parse(text)
    return text


def source_status(path: Path) -> str:
    if not path.is_file():
        return "missing"
    sha = digest(path.read_bytes())
    return "preimage" if sha == PREIMAGE_SHA else (
        "patched" if sha in (POSTIMAGE_SHA, VARLEN_POSTIMAGE_SHA) else "unknown")


def write_atomic(path: Path, data: bytes, mode: int = 0o644) -> None:
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".conv-qkv-", dir=path.parent)
    temporary_path = Path(temporary)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary_path, mode)
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tree", type=Path, default=DEFAULT_TREE)
    parser.add_argument("--backup-dir", type=Path, default=DEFAULT_BACKUP)
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--check", action="store_true")
    actions.add_argument("--apply", action="store_true")
    actions.add_argument("--revert", action="store_true")
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args(argv)
    tree = args.tree.expanduser().resolve()
    target = tree / TARGET
    backup = args.backup_dir.expanduser().resolve() / "gdn.py.preimage"
    report = {"patch_version": PATCH_VERSION, "tree": str(tree), "target": TARGET,
              "action": "check" if args.check else "apply" if args.apply else "revert",
              "preimage_sha256": PREIMAGE_SHA, "postimage_sha256": POSTIMAGE_SHA,
              "ready": False}
    try:
        fd = os.open(tree, os.O_RDONLY)
        try:
            fcntl.flock(fd, fcntl.LOCK_SH if args.check else fcntl.LOCK_EX)
            state = source_status(target)
            report["status_before"] = state
            if state in ("missing", "unknown"):
                raise ValueError(f"{TARGET}: refused {state} source content")
            if args.apply and state == "preimage":
                original = target.read_bytes()
                postimage = patch_gdn(original.decode()).encode()
                if digest(postimage) != POSTIMAGE_SHA:
                    raise ValueError("generated Conv+QKV postimage SHA mismatch")
                if backup.exists():
                    if not backup.is_file() or digest(backup.read_bytes()) != PREIMAGE_SHA:
                        raise ValueError("preimage backup SHA mismatch")
                else:
                    backup.parent.mkdir(parents=True, exist_ok=True)
                    write_atomic(backup, original, stat.S_IMODE(target.stat().st_mode))
                write_atomic(target, postimage, stat.S_IMODE(target.stat().st_mode))
            elif args.revert and state == "patched":
                if not backup.is_file() or digest(backup.read_bytes()) != PREIMAGE_SHA:
                    raise ValueError("verified preimage backup required for revert")
                write_atomic(target, backup.read_bytes(), stat.S_IMODE(target.stat().st_mode))
            after = source_status(target)
            report["status_after"] = after
            report["sha256"] = digest(target.read_bytes())
            report["ready"] = after == "patched"
        finally:
            os.close(fd)
    except (OSError, ValueError, SyntaxError) as exc:
        report["error"] = str(exc)
    if args.manifest:
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return int("error" in report)


if __name__ == "__main__":
    raise SystemExit(main())
