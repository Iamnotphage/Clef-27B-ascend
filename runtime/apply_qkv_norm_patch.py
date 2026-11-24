#!/usr/bin/env python3
"""Install/revert the GDN QKV-boundary fusion postimage.

This is a v4 companion to ``apply_gdn_decision_patch.py``.  The GDN v3
installer owns the decision-only prefill changes; this installer replaces the
same two source files with a postimage that adds the optional
``ClefGdnQkvNorm`` preprocess path while retaining every v3 change.

The postimage files are checked in under ``runtime/qkv_norm/`` so installation
is a hash-gated copy, not a second set of fragile text anchors.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

DEFAULT_TREE = Path("/workspace/yangchen/work/src/vllm-ascend")
DEFAULT_BACKUP = Path("/workspace/yangchen/logs/gdn-data-path-20261115/qkv-norm-phase0")
HERE = Path(__file__).resolve().parent
SOURCE_DIR = HERE / "qkv_norm"

PREIMAGE_SHA = {
    "vllm_ascend/device/device_op.py":
        "6886fa3fe7855d4dc8c754a9564ed2fe3967efd0cc09c5edd487c57be371d3ce",
    "vllm_ascend/ops/gdn.py":
        "7bc9b8f19e1ab17f7b484121c4c88c7f365c5ddb3a9f7a0380592175c077c893",
}
POSTIMAGE_SHA = {
    "vllm_ascend/device/device_op.py":
        "19cf5d0f34015cc1e4f5f7639a084d3ad64ed24e0a7a1a3b1946b475eec14383",
    "vllm_ascend/ops/gdn.py":
        "c8c0d4d8a693658e075a2232b6938584760672bacf101bf50ca5fdfab2c0be7e",
}
CHUNK128_POSTIMAGE_SHA = {
    "vllm_ascend/device/device_op.py":
        "bf027ec99a3d069f913600ddfcb76283f73a4114febe46b151c113c62450c226",
}
ADAPTIVE_POSTIMAGE_SHA = {
    "vllm_ascend/device/device_op.py":
        "2abb3389f8502f17f0f0f0f567d50aed035623dae0740355a66aaa5d5a035b59",
}
CONV_QKV_POSTIMAGE_SHA = {
    "vllm_ascend/ops/gdn.py":
        "ead25cb2691cb9ab1c322353fc8e168f8bfc9c23b9e4d53de01c762943027e78",
}
CONV_QKV_VARLEN_POSTIMAGE_SHA = {
    "vllm_ascend/ops/gdn.py":
        "afe6d4e87c0eb47bfb325fb630620e47f845fb5a893cb1dada78069ad4e3ca07",
}
SOURCE_FILES = {
    "vllm_ascend/device/device_op.py": SOURCE_DIR / "qkv_device_op.py",
    "vllm_ascend/ops/gdn.py": SOURCE_DIR / "qkv_gdn.py",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_atomic(path: Path, data: bytes) -> None:
    temporary = path.with_name(path.name + ".qkv-norm.tmp")
    temporary.write_bytes(data)
    os.replace(temporary, path)


def status(path: Path, relative: str) -> str:
    if not path.is_file():
        return "missing"
    digest = sha256(path)
    if digest == PREIMAGE_SHA[relative]:
        return "preimage"
    if digest == POSTIMAGE_SHA[relative]:
        return "qkv"
    if digest == CHUNK128_POSTIMAGE_SHA.get(relative):
        return "qkv"
    if digest == ADAPTIVE_POSTIMAGE_SHA.get(relative):
        return "qkv"
    if digest == CONV_QKV_POSTIMAGE_SHA.get(relative):
        return "qkv"
    if digest == CONV_QKV_VARLEN_POSTIMAGE_SHA.get(relative):
        return "qkv"
    return "unknown"


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
    report = {"tree": str(tree), "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "action": "check" if args.check else "apply" if args.apply else "revert",
              "files": {}}
    writes: dict[Path, bytes | None] = {}
    try:
        fd = os.open(tree, os.O_RDONLY)
        try:
            fcntl.flock(fd, fcntl.LOCK_SH if args.check else fcntl.LOCK_EX)
            for relative, expected in PREIMAGE_SHA.items():
                path = tree / relative
                state = status(path, relative)
                item = {"status_before": state, "path": str(path)}
                backup = backup_dir / f"{path.name}.preimage"
                if state == "unknown":
                    raise ValueError(f"{relative}: unknown source content; refusing")
                if args.check:
                    item["status_after"] = state
                elif args.revert:
                    if state == "qkv":
                        if not backup.is_file() or sha256(backup) != expected:
                            raise ValueError(f"{relative}: verified preimage backup required")
                        writes[path] = backup.read_bytes()
                        item["status_after"] = "preimage"
                    else:
                        item["status_after"] = state
                else:
                    if state == "qkv":
                        item["status_after"] = state
                    else:
                        if not backup.exists():
                            backup.parent.mkdir(parents=True, exist_ok=True)
                            backup.write_bytes(path.read_bytes())
                        elif sha256(backup) != expected:
                            raise ValueError(f"{relative}: preimage backup SHA mismatch")
                        source = SOURCE_FILES[relative]
                        if not source.is_file() or sha256(source) != POSTIMAGE_SHA[relative]:
                            raise ValueError(f"{relative}: bundled postimage SHA mismatch")
                        writes[path] = source.read_bytes()
                        item["status_after"] = "qkv"
                report["files"][relative] = item
            if writes:
                originals = {p: (p.read_bytes() if p.is_file() else None) for p in writes}
                try:
                    for path, data in writes.items():
                        write_atomic(path, data)
                except OSError:
                    for path, data in originals.items():
                        if data is not None:
                            write_atomic(path, data)
                    raise
            report["ready"] = all(
                item.get("status_after") == "qkv"
                for item in report["files"].values()
            ) if args.check else True
        finally:
            os.close(fd)
    except (OSError, ValueError) as exc:
        report["error"] = str(exc)
        report["ready"] = False
        print(json.dumps(report, indent=1))
        return 1
    if args.manifest:
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
