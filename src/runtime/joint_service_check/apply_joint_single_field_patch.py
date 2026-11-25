#!/usr/bin/env python3
"""Install/revert the single-field joint-head postimage.

The patch replaces vLLM's ``clef_joint_schema.py`` with a version that, when
``CLEF_JOINT_SINGLE_FIELD=1`` and a request has one question, folds the
Q=1 self-attention softmax to the V projection and simplifies option routing.
Multi-question requests keep the original path.
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
DEFAULT_BACKUP = Path("/workspace/yangchen/logs/gdn-data-path-20261115/joint-single-field-phase0")
HERE = Path(__file__).resolve().parent
SOURCE = HERE / "joint_single_field" / "clef_joint_schema.py"
TARGET = "vllm/model_executor/models/clef_joint_schema.py"
PREIMAGE_SHA = "197863d5019f8b5326421f7616fbda13b030c27023248bf130628af7722bf4be"
POSTIMAGE_SHA = "f8101b98228bbd472a63a1efe6827583e35c6cc9dede4379b925f2b016670863"
SERVICE_POSTIMAGE_SHA = "aa3e17585f8bd60a4d05e342abc7e2773c79a0b2d49f0370d15e29bcc6b72503"
SERVICE_ENV = "CLEF_JOINT_ALLOWED_POSTIMAGE_SHA"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_atomic(path: Path, data: bytes) -> None:
    temporary = path.with_name(path.name + ".single-field.tmp")
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
        "tree": str(tree),
        "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "action": "check" if args.check else "apply" if args.apply else "revert",
        "target": str(path),
        "service_postimage_sha256": SERVICE_POSTIMAGE_SHA,
        "service_env": SERVICE_ENV,
        "service_env_enabled": os.environ.get(SERVICE_ENV) == SERVICE_POSTIMAGE_SHA,
    }
    try:
        fd = os.open(tree, os.O_RDONLY)
        try:
            fcntl.flock(fd, fcntl.LOCK_SH if args.check else fcntl.LOCK_EX)
            if not path.is_file():
                raise ValueError("joint schema source missing")
            digest = sha256(path)
            allowed_postimage = POSTIMAGE_SHA
            if args.check and os.environ.get(SERVICE_ENV) == SERVICE_POSTIMAGE_SHA:
                allowed_postimage = SERVICE_POSTIMAGE_SHA
            if digest == PREIMAGE_SHA:
                state = "preimage"
            elif digest in (POSTIMAGE_SHA, allowed_postimage):
                state = "single-field"
            else:
                raise ValueError(f"unknown joint schema source SHA-256: {digest}")
            report["status_before"] = state
            if args.check:
                report["status_after"] = state
                report["ready"] = state == "single-field"
            elif args.revert:
                if state == "single-field":
                    if not backup.is_file() or sha256(backup) != PREIMAGE_SHA:
                        raise ValueError("verified preimage backup required")
                    write_atomic(path, backup.read_bytes())
                    report["status_after"] = "preimage"
                else:
                    report["status_after"] = state
                report["ready"] = True
            else:
                if state == "single-field":
                    report["status_after"] = state
                else:
                    if not backup.exists():
                        backup.parent.mkdir(parents=True, exist_ok=True)
                        backup.write_bytes(path.read_bytes())
                    elif sha256(backup) != PREIMAGE_SHA:
                        raise ValueError("preimage backup SHA-256 mismatch")
                    if not SOURCE.is_file() or sha256(SOURCE) != POSTIMAGE_SHA:
                        raise ValueError("bundled single-field postimage SHA-256 mismatch")
                    write_atomic(path, SOURCE.read_bytes())
                    report["status_after"] = "single-field"
                report["ready"] = True
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
