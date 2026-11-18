#!/usr/bin/env python3
"""Hash-gated installer for the service startup single-field check adapter.

PREIMAGE is the current production `runtime/apply_joint_single_field_patch.py`
(SHA 76595657).  The postimage adds exactly one allowance: when the service
environment sets CLEF_JOINT_ALLOWED_POSTIMAGE_SHA to the already approved
fused postimage aa3e1758, `--check` accepts that exact SHA as single-field.
Without the variable the original f810/197863 guard is unchanged.  A/B windows
apply this adapter, set the variable only in the arm service env, and revert
the adapter before restoring production.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import time
from pathlib import Path

DEFAULT_SERVICE_ROOT = Path("/workspace/yangchen/Clef-27B-ascend")
DEFAULT_BACKUP = Path("/workspace/yangchen/logs/joint-service-check-20261119/phase0")
HERE = Path(__file__).resolve().parent
SOURCE = HERE / "joint_service_check" / "apply_joint_single_field_patch.py"
TARGET = "runtime/apply_joint_single_field_patch.py"
PREIMAGE_SHA = "76595657d312699b4b1564d1bc054229808cd775c16702fb0e36fcdb1f9706d9"
POSTIMAGE_SHA = "20abe6800dfdf807a0ba54fbdf86b155a57d22bbce25dcc85ca3d5c36ca37e61"
ENV_VAR = "CLEF_JOINT_ALLOWED_POSTIMAGE_SHA"
SERVICE_POSTIMAGE_SHA = "aa3e17585f8bd60a4d05e342abc7e2773c79a0b2d49f0370d15e29bcc6b72503"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_atomic(path: Path, data: bytes) -> None:
    temporary = path.with_name(path.name + ".service-check.tmp")
    temporary.write_bytes(data)
    os.replace(temporary, path)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--service-root", type=Path, default=DEFAULT_SERVICE_ROOT)
    ap.add_argument("--backup-dir", type=Path, default=DEFAULT_BACKUP)
    group = ap.add_mutually_exclusive_group(required=True)
    for action in ("check", "apply", "revert"):
        group.add_argument("--" + action, action="store_true")
    ap.add_argument("--manifest", type=Path)
    args = ap.parse_args(argv)
    root = args.service_root.expanduser().resolve()
    path = root / TARGET
    backup = args.backup_dir.expanduser().resolve() / "apply_joint_single_field_patch.py.preimage"
    report = {"schema": "joint-service-check-installer/1",
              "service_root": str(root), "file": str(path),
              "action": "check" if args.check else "apply" if args.apply else "revert",
              "preimage_sha256": PREIMAGE_SHA, "postimage_sha256": POSTIMAGE_SHA,
              "env": {"name": ENV_VAR, "allowed_value": SERVICE_POSTIMAGE_SHA},
              "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    try:
        fd = os.open(root, os.O_RDONLY)
    except OSError as exc:
        report["error"] = str(exc)
        report["ready"] = False
        print(json.dumps(report, indent=1))
        return 1
    try:
        fcntl.flock(fd, fcntl.LOCK_SH if args.check else fcntl.LOCK_EX)
        if not path.is_file():
            raise ValueError("service joint installer missing")
        digest = sha256(path)
        state = "preimage" if digest == PREIMAGE_SHA else "postimage" if digest == POSTIMAGE_SHA else "unknown"
        report["state_before"] = state
        if state == "unknown":
            raise ValueError(f"unknown service installer SHA-256: {digest}")
        if args.check:
            report["state_after"] = state
            report["ready"] = state == "postimage"
        elif args.apply:
            if state == "preimage":
                if not SOURCE.is_file() or sha256(SOURCE) != POSTIMAGE_SHA:
                    raise ValueError("bundled service-check postimage SHA mismatch")
                if backup.exists() and sha256(backup) != PREIMAGE_SHA:
                    raise ValueError("existing backup is not the preimage")
                if not backup.exists():
                    backup.parent.mkdir(parents=True, exist_ok=True)
                    backup.write_bytes(path.read_bytes())
                write_atomic(path, SOURCE.read_bytes())
                if sha256(path) != POSTIMAGE_SHA:
                    raise ValueError("service-check postimage SHA mismatch after write")
                state = "postimage"
            report["state_after"] = state
            report["ready"] = sha256(path) == POSTIMAGE_SHA
        else:
            if state == "postimage":
                if not backup.is_file() or sha256(backup) != PREIMAGE_SHA:
                    raise ValueError("verified preimage backup required")
                write_atomic(path, backup.read_bytes())
                state = "preimage"
            report["state_after"] = state
            report["ready"] = sha256(path) == PREIMAGE_SHA
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
