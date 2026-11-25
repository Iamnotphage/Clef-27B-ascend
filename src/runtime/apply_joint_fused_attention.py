#!/usr/bin/env python3
"""Install/revert the joint-head fused cross-attention candidate.

PREIMAGE is the installed single-field postimage (``f8101b98``).  The
postimage adds ``CLEF_JOINT_FUSED_ATTN`` (fused cross-attention) and
``CLEF_JOINT_RANK0_ONLY`` (skip the joint head on non-zero TP ranks after the
collective lexical embedding).  Both default to 0, so installing alone does not
change behavior.  ``aa3e1758`` is accepted as a legacy fused postimage and is
upgraded in place with a verified backup of its own.
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
LEGACY_POSTIMAGE_SHA = "aa3e17585f8bd60a4d05e342abc7e2773c79a0b2d49f0370d15e29bcc6b72503"
POSTIMAGE_SHA = "26d3ad047c42f47a7fa22e00d6bac53a8a063ed010e753cb79a673660f60d7c4"
ENV_VAR = "CLEF_JOINT_FUSED_ATTN"
RANK0_ENV_VAR = "CLEF_JOINT_RANK0_ONLY"


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
    backup_dir = args.backup_dir.expanduser().resolve()
    preimage_backup = backup_dir / "clef_joint_schema.py.preimage"
    legacy_backup = backup_dir / "clef_joint_schema.py.aa3e"
    restore_record = backup_dir / "install-state.json"
    report = {
        "schema": "joint-fused-attention-installer/2",
        "tree": str(tree),
        "file": str(path),
        "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "action": "check" if args.check else "apply" if args.apply else "revert",
        "preimage_sha256": PREIMAGE_SHA,
        "legacy_postimage_sha256": LEGACY_POSTIMAGE_SHA,
        "postimage_sha256": POSTIMAGE_SHA,
        "env": {
            "fused_attn": {"name": ENV_VAR, "default": "0",
                           "scope": "single-field cross-attention"},
            "rank0_only": {"name": RANK0_ENV_VAR, "default": "0",
                           "scope": "skip the joint head on TP ranks != 0"},
        },
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
        if digest == PREIMAGE_SHA:
            state, source_backup = "preimage", preimage_backup
        elif digest == LEGACY_POSTIMAGE_SHA:
            # Report the legacy fused file as the fused installer's preimage so
            # the launcher's apply branch upgrades it in place.
            state, source_backup = "preimage", legacy_backup
            report["legacy_postimage"] = True
        elif digest == POSTIMAGE_SHA:
            state, source_backup = "postimage", None
        else:
            state, source_backup = "unknown", None
        report["state_before"] = state
        report["sha256_before"] = digest
        if state == "unknown":
            raise ValueError(f"unknown joint schema source SHA-256: {digest}")
        if args.check:
            report["state_after"] = state
            report["ready"] = state == "postimage"
        elif args.revert:
            if state == "postimage":
                restore_sha = None
                if restore_record.is_file():
                    record = json.loads(restore_record.read_text())
                    restore_path = Path(record["backup"])
                    restore_sha = record["sha256"]
                else:
                    restore_path = preimage_backup
                    restore_sha = PREIMAGE_SHA
                if not restore_path.is_file() or sha256(restore_path) != restore_sha:
                    raise ValueError("verified pre-install backup required")
                write_atomic(path, restore_path.read_bytes())
                state = ("preimage" if restore_sha == PREIMAGE_SHA else "legacy")
                report["restored_sha256"] = restore_sha
            report["state_after"] = state
            report["ready"] = sha256(path) in (PREIMAGE_SHA, LEGACY_POSTIMAGE_SHA)
        else:
            if state == "postimage":
                report["state_after"] = state
            else:
                if not source_backup.exists():
                    source_backup.parent.mkdir(parents=True, exist_ok=True)
                    source_backup.write_bytes(path.read_bytes())
                elif sha256(source_backup) != digest:
                    raise ValueError("pre-install backup SHA-256 mismatch")
                if not SOURCE.is_file() or sha256(SOURCE) != POSTIMAGE_SHA:
                    raise ValueError("bundled fused postimage SHA-256 mismatch")
                write_atomic(path, SOURCE.read_bytes())
                if sha256(path) != POSTIMAGE_SHA:
                    raise ValueError("fused postimage SHA-256 mismatch after write")
                restore_record.parent.mkdir(parents=True, exist_ok=True)
                restore_record.write_text(json.dumps(
                    {"sha256": digest, "backup": str(source_backup),
                     "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())},
                    indent=1) + "\n")
                report["backup"] = str(source_backup)
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
