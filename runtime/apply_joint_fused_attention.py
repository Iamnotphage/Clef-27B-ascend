#!/usr/bin/env python3
"""Install/revert the joint-head fused cross-attention candidate.

PREIMAGE is the installed single-field postimage (``f8101b98``).  The new
postimage adds ``CLEF_JOINT_FUSED_ATTN``: when set to 1, the six single-field
cross-attention calls use ``torch_npu.npu_fusion_attention`` while keeping the
q/k/v and output projections.  The default is 0, so installing alone does not
change behavior.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import time
from pathlib import Path

DEFAULT_TREE = Path("/workspace/yangchen/work/src/vllm")
DEFAULT_BACKUP = Path("/workspace/yangchen/logs/joint-fused-attention-20261119/phase0")
HERE = Path(__file__).resolve().parent
SOURCE = HERE / "joint_fused_attention" / "clef_joint_schema.py"
TARGET = "vllm/model_executor/models/clef_joint_schema.py"
PREIMAGE_SHA = "f8101b98228bbd472a63a1efe6827583e35c6cc9dede4379b925f2b016670863"
POSTIMAGE_SHA = "aa3e17585f8bd60a4d05e342abc7e2773c79a0b2d49f0370d15e29bcc6b72503"
ENV_VAR = "CLEF_JOINT_FUSED_ATTN"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_atomic(path: Path, data: bytes) -> None:
    temporary = path.with_name(path.name + ".fused-attn.tmp")
    temporary.write_bytes(data)
    os.replace(temporary, path)


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
    path = tree / TARGET
    backup = args.backup_dir.expanduser().resolve() / "clef_joint_schema.py.preimage"
    report = {
        "schema": "joint-fused-attention-installer/1",
        "tree": str(tree),
        "file": str(path),
        "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "action": "check" if args.check else "apply" if args.apply else "revert",
        "preimage_sha256": PREIMAGE_SHA,
        "postimage_sha256": POSTIMAGE_SHA,
        "env": {"name": ENV_VAR, "default": "0", "scope": "single-field cross-attention"},
    }
    try:
        fd = os.open(tree, os.O_RDONLY)
    except OSError as exc:
        report["error"] = str(exc)
        report["ready"] = False
        print(json.dumps(report, indent=1))
        return 1

    try:
        fcntl.flock(fd, fcntl.LOCK_SH if args.check else fcntl.LOCK_EX)
        if not path.is_file():
            raise ValueError("joint schema source missing")
        digest = sha256(path)
        state = "preimage" if digest == PREIMAGE_SHA else "postimage" if digest == POSTIMAGE_SHA else "unknown"
        report["state_before"] = state
        report["sha256_before"] = digest
        if state == "unknown":
            raise ValueError(f"unknown joint schema source SHA-256: {digest}")
        if args.check:
            report["state_after"] = state
            report["ready"] = state == "postimage"
        elif args.revert:
            if state == "postimage":
                if not backup.is_file() or sha256(backup) != PREIMAGE_SHA:
                    raise ValueError("verified single-field preimage backup required")
                write_atomic(path, backup.read_bytes())
                state = "preimage"
            report["state_after"] = state
            report["ready"] = sha256(path) == PREIMAGE_SHA
        else:
            if state == "postimage":
                report["state_after"] = state
            else:
                if not backup.exists():
                    backup.parent.mkdir(parents=True, exist_ok=True)
                    backup.write_bytes(path.read_bytes())
                elif sha256(backup) != PREIMAGE_SHA:
                    raise ValueError("preimage backup SHA-256 mismatch")
                if not SOURCE.is_file() or sha256(SOURCE) != POSTIMAGE_SHA:
                    raise ValueError("bundled fused postimage SHA-256 mismatch")
                write_atomic(path, SOURCE.read_bytes())
                if sha256(path) != POSTIMAGE_SHA:
                    raise ValueError("fused postimage SHA-256 mismatch after write")
                report["state_after"] = "postimage"
            report["ready"] = True
    except (OSError, ValueError) as exc:
        report["error"] = str(exc)
        report["ready"] = False
        print(json.dumps(report, indent=1))
        return 1
    finally:
        os.close(fd)

    if args.manifest:
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
