#!/usr/bin/env python3
"""Install/revert the multi-sequence Conv+QKV varlen candidate.

PREIMAGE is the accepted v4 Conv+QKV postimage ``ead25cb2``.  The postimage
adds ``_run_conv_qkv_varlen`` and a clean branch in ``_run_conv_qkv``: one
prefill still executes the unchanged v4 kernel, while 2..64 fresh prefills
use ``aclnnClefGdnConvQkvVarlen`` when ``CLEF_GDN_CONV_QKV_VARLEN=1``.
Any ineligible input returns ``None`` and keeps the decomposed path.

The installer is byte-exact and Fail-Closed on unknown sources.
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
DEFAULT_BACKUP = Path("/workspace/yangchen/logs/conv-qkv-varlen-20261125/phase0")
TARGET = "vllm_ascend/ops/gdn.py"
PATCH_VERSION = 1
PREIMAGE_SHA = "ead25cb2691cb9ab1c322353fc8e168f8bfc9c23b9e4d53de01c762943027e78"
POSTIMAGE_SHA = "afe6d4e87c0eb47bfb325fb630620e47f845fb5a893cb1dada78069ad4e3ca07"

HELPER = '''_CONV_QKV_VARLEN_RUNTIME = None
_CONV_QKV_VARLEN_VENDOR = None


def _run_conv_qkv_varlen(x, weight, *, attn_metadata, tp_size, num_k_heads,
                         num_v_heads, head_k_dim, head_v_dim, bias, activation):
    """Return normalized BSND q/k/v for 2..64 fresh prefills in one call.

    Each sequence is tiled independently inside the Ascend C kernel, so a
    short or NaN neighbour can never leak into another sequence.  Ineligible
    inputs return None and keep the accepted decomposed path.
    """
    global _CONV_QKV_VARLEN_RUNTIME, _CONV_QKV_VARLEN_VENDOR
    if os.environ.get("CLEF_GDN_CONV_QKV_VARLEN", "0") != "1":
        return None
    prefill = getattr(attn_metadata, "non_spec_prefill_metadata", None)
    causal = getattr(prefill, "causal_conv1d", None)
    cu_seqlens = getattr(causal, "query_start_loc", None)
    num_prefills = int(attn_metadata.num_prefills)
    if (cu_seqlens is None or cu_seqlens.dim() != 1
            or cu_seqlens.shape[0] != num_prefills + 1
            or cu_seqlens.dtype != torch.int32
            or not cu_seqlens.is_contiguous()
            or cu_seqlens.device != x.device
            or tp_size != 2 or num_k_heads != 16 or num_v_heads != 48
            or head_k_dim != 128 or head_v_dim != 128
            or bias is not None or activation not in (True, "silu", "swish")
            or x.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16
            or weight.device != x.device):
        return None
    vendor = os.environ.get("CLEF_CONV_QKV_VARLEN_VENDOR")
    if not vendor:
        raise RuntimeError(
            "CLEF_GDN_CONV_QKV_VARLEN=1 requires CLEF_CONV_QKV_VARLEN_VENDOR")
    tile = int(os.environ.get("CLEF_CONV_QKV_VARLEN_TILE", "64"))
    if tile not in (16, 32, 64):
        raise RuntimeError("CLEF_CONV_QKV_VARLEN_TILE must be 16, 32, or 64")
    import ctypes
    from fla_npu.ops.ascendc import _runtime as aclnn_runtime

    if _CONV_QKV_VARLEN_RUNTIME is None:
        library = ctypes.CDLL(
            f"{vendor}/op_api/lib/libcust_opapi.so",
            mode=getattr(os, "RTLD_LOCAL", 0) | getattr(os, "RTLD_NOW", 0)
            | getattr(os, "RTLD_NODELETE", 0),
        )
        _CONV_QKV_VARLEN_RUNTIME = aclnn_runtime.runtime()
        _CONV_QKV_VARLEN_RUNTIME._libraries.insert(0, library)
        _CONV_QKV_VARLEN_VENDOR = vendor
        print("clef-conv-qkv-varlen pid=%d tile=%d first_num_prefills=%d "
              "first_tokens=%d vendor=%s" % (
                  os.getpid(), tile, num_prefills, x.shape[0], vendor), flush=True)
    elif vendor != _CONV_QKV_VARLEN_VENDOR:
        raise RuntimeError(
            "CLEF_CONV_QKV_VARLEN_VENDOR changed within a running worker")
    tokens = x.shape[0]
    q = torch.empty((1, tokens, 8, 128), dtype=x.dtype, device=x.device)
    k = torch.empty_like(q)
    v = torch.empty((1, tokens, 24, 128), dtype=x.dtype, device=x.device)
    aclnn_runtime.call_aclnn(
        "aclnnClefGdnConvQkvVarlen",
        lambda ctx: [ctx.tensor(x, "x"), ctx.tensor(weight, "weight"),
                     ctx.tensor(cu_seqlens, "cuSeqlens"),
                     ctypes.c_double(1e-6), ctypes.c_int64(tile),
                     ctx.tensor(q, "qOut"), ctx.tensor(k, "kOut"),
                     ctx.tensor(v, "vOut")],
        (q, k, v),
        get_workspace_argtypes=[ctypes.c_void_p, ctypes.c_void_p,
                                ctypes.c_void_p, ctypes.c_double,
                                ctypes.c_int64, ctypes.c_void_p,
                                ctypes.c_void_p, ctypes.c_void_p,
                                ctypes.POINTER(ctypes.c_uint64),
                                ctypes.POINTER(ctypes.c_void_p)],
    )
    return q, k, v


'''

ELIGIBILITY_OLD = '''    if (runtime is None or not runtime.skip_conv_state
            or not runtime.preprocess_fusion or not runtime.bsnd
            or attn_metadata.num_prefills != 1 or attn_metadata.num_decodes != 0
'''
ELIGIBILITY_NEW = '''    if (runtime is None or not runtime.skip_conv_state
            or not runtime.preprocess_fusion or not runtime.bsnd
            or attn_metadata.num_prefills < 1
            or attn_metadata.num_prefills > 64
            or attn_metadata.num_decodes != 0
'''
VENDOR_ANCHOR = '    vendor = os.environ.get("CLEF_CONV_QKV_VENDOR")\n'
VARLEN_BRANCH = '''    if attn_metadata.num_prefills > 1:
        return _run_conv_qkv_varlen(
            x, weight, attn_metadata=attn_metadata, tp_size=tp_size,
            num_k_heads=num_k_heads, num_v_heads=num_v_heads,
            head_k_dim=head_k_dim, head_v_dim=head_v_dim,
            bias=bias, activation=activation)
'''


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def patch_gdn(text: str) -> str:
    if digest(text.encode()) != PREIMAGE_SHA:
        raise ValueError("gdn.py: verified Conv+QKV preimage required")
    anchor = "def _run_conv_qkv("
    if text.count(anchor) != 1:
        raise ValueError("gdn.py: _run_conv_qkv anchor is not unique")
    if text.count(ELIGIBILITY_OLD) != 1:
        raise ValueError("gdn.py: v4 eligibility anchor is not unique")
    if text.count(VENDOR_ANCHOR) != 1:
        raise ValueError("gdn.py: v4 vendor anchor is not unique")
    text = text.replace(anchor, HELPER + anchor, 1)
    text = text.replace(ELIGIBILITY_OLD, ELIGIBILITY_NEW, 1)
    text = text.replace(VENDOR_ANCHOR, VARLEN_BRANCH + VENDOR_ANCHOR, 1)
    ast.parse(text)
    if digest(text.encode()) != POSTIMAGE_SHA:
        raise ValueError("gdn.py: generated varlen postimage SHA mismatch")
    return text


def source_status(path: Path) -> str:
    if not path.is_file():
        return "missing"
    sha = digest(path.read_bytes())
    return "preimage" if sha == PREIMAGE_SHA else "patched" if sha == POSTIMAGE_SHA else "unknown"


def write_atomic(path: Path, data: bytes, mode: int = 0o644) -> None:
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".conv-qkv-varlen-", dir=path.parent)
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
                    raise ValueError("generated varlen postimage SHA mismatch")
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
