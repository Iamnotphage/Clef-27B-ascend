#!/usr/bin/env python3
"""Install/revert the opt-in TP1/TP2 generic QKV preprocessing path.

Apply after the verified QKV installer, optionally after Conv+QKV. The vendor
must already be visible to CANN before torch_npu import. This installer changes
only the verified gdn.py source, retaining an exact backup for each preimage.
"""

from __future__ import annotations

import argparse
import ast
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat
import tempfile

DEFAULT_TREE = Path("/workspace/yangchen/work/src/vllm-ascend")
DEFAULT_BACKUP = Path("/workspace/yangchen/logs/gdn-qkv-generic-20261117/phase0")
TARGET = "vllm_ascend/ops/gdn.py"
PATCH_VERSION = 1
HERE = Path(__file__).resolve().parent
_CONV_SPEC = importlib.util.spec_from_file_location(
    "_clef_generic_conv_contract", HERE / "apply_conv_qkv_patch.py")
_CONV_CONTRACT = importlib.util.module_from_spec(_CONV_SPEC)
_CONV_SPEC.loader.exec_module(_CONV_CONTRACT)
PREIMAGE_SHA = {
    "qkv": "c8c0d4d8a693658e075a2232b6938584760672bacf101bf50ca5fdfab2c0be7e",
    "conv-qkv": _CONV_CONTRACT.POSTIMAGE_SHA,
}
POSTIMAGE_SHA = {
    "qkv": "9e18c22a01d646de1935970dda9b9613ef66594c7eaa3028da1be98be2f8e2dd",
}

HELPER = '''_QKV_GENERIC_RUNTIME = None
_QKV_GENERIC_LIBRARY = None
_QKV_GENERIC_VENDOR = None
_QKV_GENERIC_LOGGED = False


def _run_qkv_generic(x, num_k_heads: int, num_v_heads: int, head_dim: int):
    """Return normalized BSND q/k/v for validated TP1/TP2 input shapes.

    Guard failures return None. Selected candidate loader/execution errors
    propagate, so an experiment cannot silently measure the legacy path.
    """
    global _QKV_GENERIC_RUNTIME, _QKV_GENERIC_LIBRARY
    global _QKV_GENERIC_VENDOR, _QKV_GENERIC_LOGGED
    if (num_k_heads not in (8, 16) or num_v_heads != 3 * num_k_heads
            or head_dim != 128 or x.dim() != 3 or not x.is_contiguous()
            or x.dtype != torch.bfloat16 or x.device.type not in ("npu", "privateuseone")
            or x.shape[0] != 5 * num_k_heads or x.shape[2] != 128
            or x.shape[1] <= 0):
        return None
    vendor = os.environ.get("CLEF_QKV_GENERIC_VENDOR")
    if not vendor:
        raise RuntimeError("CLEF_QKV_GENERIC=1 requires CLEF_QKV_GENERIC_VENDOR")
    try:
        tile = int(os.environ.get("CLEF_QKV_GENERIC_TILE", "32"))
    except ValueError as exc:
        raise RuntimeError("CLEF_QKV_GENERIC_TILE must be 16, 32, or 64") from exc
    if tile not in (16, 32, 64):
        raise RuntimeError("CLEF_QKV_GENERIC_TILE must be 16, 32, or 64")
    import ctypes
    from fla_npu.ops.ascendc import _runtime as aclnn_runtime

    if _QKV_GENERIC_RUNTIME is None:
        library = ctypes.CDLL(
            f"{vendor}/op_api/lib/libcust_opapi.so",
            mode=getattr(os, "RTLD_LOCAL", 0) | getattr(os, "RTLD_NOW", 0)
            | getattr(os, "RTLD_NODELETE", 0),
        )
        runtime = aclnn_runtime.runtime()
        runtime._libraries.insert(0, library)
        _QKV_GENERIC_LIBRARY = library
        _QKV_GENERIC_RUNTIME = runtime
        _QKV_GENERIC_VENDOR = vendor
    elif vendor != _QKV_GENERIC_VENDOR:
        raise RuntimeError("CLEF_QKV_GENERIC_VENDOR changed within a running worker")
    tokens = x.shape[1]
    q = torch.empty((1, tokens, num_k_heads, 128), dtype=x.dtype, device=x.device)
    k = torch.empty_like(q)
    v = torch.empty((1, tokens, num_v_heads, 128), dtype=x.dtype, device=x.device)
    aclnn_runtime.call_aclnn(
        "aclnnClefGdnQkvGeneric",
        lambda ctx: [ctx.tensor(x, "x"), ctypes.c_double(1e-6), ctypes.c_int64(tile),
                     ctx.tensor(q, "qOut"), ctx.tensor(k, "kOut"), ctx.tensor(v, "vOut")],
        (q, k, v),
        get_workspace_argtypes=[ctypes.c_void_p, ctypes.c_double, ctypes.c_int64,
                                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                                ctypes.POINTER(ctypes.c_uint64),
                                ctypes.POINTER(ctypes.c_void_p)],
    )
    if not _QKV_GENERIC_LOGGED:
        print("clef-qkv-generic pid=%d tile=%d first_tokens=%d hk=%d hv=%d vendor=%s" % (
            os.getpid(), tile, tokens, num_k_heads, num_v_heads, vendor), flush=True)
        _QKV_GENERIC_LOGGED = True
    return q, k, v


'''

HELPER_ANCHOR = "_QKV_NORM_RUNTIME = None\n"
DISPATCH_ANCHOR = '''    global _QKV_NORM_RUNTIME, _QKV_NORM_FAILED
    if _QKV_NORM_FAILED:
        return None
    import ctypes
    import os

'''
DISPATCH = '''    global _QKV_NORM_RUNTIME, _QKV_NORM_FAILED
    import os
    if os.environ.get("CLEF_QKV_GENERIC", "0") == "1":
        return _run_qkv_generic(x, num_k_heads, num_v_heads, head_dim)
    if _QKV_NORM_FAILED:
        return None
    import ctypes

'''


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def patch_gdn(text: str) -> str:
    if digest(text.encode()) not in PREIMAGE_SHA.values():
        raise ValueError("gdn.py: verified QKV or Conv+QKV preimage required")
    for name, anchor in (("helper", HELPER_ANCHOR), ("dispatch", DISPATCH_ANCHOR)):
        if text.count(anchor) != 1:
            raise ValueError(f"gdn.py: {name} anchor is not unique")
    text = text.replace(HELPER_ANCHOR, HELPER + HELPER_ANCHOR, 1)
    text = text.replace(DISPATCH_ANCHOR, DISPATCH, 1)
    ast.parse(text)
    return text


def _conv_postimage_sha() -> str:
    """Compose the current independently verified Conv+QKV contract.

    Conv guard changes update its own pinned postimage. Rebuild the combined
    generic hash from that source rather than hard-coding an old conv helper.
    Both declared conv hashes are checked before accepting the composition.
    """
    base = (HERE / "qkv_norm/qkv_gdn.py").read_bytes()
    if (digest(base) != PREIMAGE_SHA["qkv"]
            or _CONV_CONTRACT.PREIMAGE_SHA != PREIMAGE_SHA["qkv"]):
        raise ValueError("bundled QKV preimage and Conv+QKV contract disagree")
    conv_image = _CONV_CONTRACT.patch_gdn(base.decode()).encode()
    if digest(conv_image) != PREIMAGE_SHA["conv-qkv"]:
        raise ValueError("generated current Conv+QKV postimage SHA mismatch")
    return digest(patch_gdn(conv_image.decode()).encode())


POSTIMAGE_SHA["conv-qkv"] = _conv_postimage_sha()


def source_state(path: Path) -> tuple[str, str | None]:
    if path.is_symlink():
        return "symlink", None
    if not path.is_file():
        return "missing", None
    sha = digest(path.read_bytes())
    for variant, expected in PREIMAGE_SHA.items():
        if sha == expected:
            return "preimage", variant
    for variant, expected in POSTIMAGE_SHA.items():
        if sha == expected:
            return "patched", variant
    return "unknown", None


def write_atomic(path: Path, data: bytes, mode: int = 0o644) -> None:
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".qkv-generic-", dir=path.parent)
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


def backup_path(directory: Path, variant: str) -> Path:
    return directory / f"gdn.py.{PREIMAGE_SHA[variant]}.preimage"


def verified_backup(path: Path, variant: str) -> bool:
    return (not path.is_symlink() and path.is_file()
            and digest(path.read_bytes()) == PREIMAGE_SHA[variant])


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
    directory = args.backup_dir.expanduser().resolve()
    manifest = args.manifest.expanduser().resolve() if args.manifest else None
    protected = {target.resolve()} | {
        backup_path(directory, variant).resolve() for variant in PREIMAGE_SHA
    }
    report = {"patch_version": PATCH_VERSION, "tree": str(tree), "target": TARGET,
              "action": "check" if args.check else "apply" if args.apply else "revert",
              "preimage_sha256": PREIMAGE_SHA, "postimage_sha256": POSTIMAGE_SHA,
              "ready": False}
    try:
        if manifest in protected:
            raise ValueError("manifest must be separate from target and backups")
        fd = os.open(tree, os.O_RDONLY)
        try:
            fcntl.flock(fd, fcntl.LOCK_SH if args.check else fcntl.LOCK_EX)
            state, variant = source_state(target)
            report.update(status_before=state, variant=variant)
            if state != "preimage" and state != "patched":
                raise ValueError(f"{TARGET}: refused {state} source content")
            backup = backup_path(directory, variant)
            report["backup"] = str(backup)
            mode = stat.S_IMODE(target.stat().st_mode)
            if args.apply and state == "preimage":
                original = target.read_bytes()
                postimage = patch_gdn(original.decode()).encode()
                if digest(postimage) != POSTIMAGE_SHA[variant]:
                    raise ValueError("generated generic QKV postimage SHA mismatch")
                if backup.exists() or backup.is_symlink():
                    if not verified_backup(backup, variant):
                        raise ValueError("preimage backup SHA mismatch")
                else:
                    directory.mkdir(parents=True, exist_ok=True)
                    write_atomic(backup, original, mode)
                write_atomic(target, postimage, mode)
            elif args.revert and state == "patched":
                if not verified_backup(backup, variant):
                    raise ValueError("verified matching preimage backup required for revert")
                write_atomic(target, backup.read_bytes(), mode)
            after, after_variant = source_state(target)
            report.update(status_after=after, variant=after_variant,
                          sha256=digest(target.read_bytes()), ready=after == "patched")
        finally:
            os.close(fd)
    except (OSError, ValueError, SyntaxError) as exc:
        report["error"] = str(exc)
    if manifest and manifest not in protected:
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return int("error" in report)


if __name__ == "__main__":
    raise SystemExit(main())
