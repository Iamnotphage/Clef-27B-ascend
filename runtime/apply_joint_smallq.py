#!/usr/bin/env python3
"""Hash-locked install/check/revert for the independent small-Q candidate.

No torch, NPU or Triton import. Service orchestration is owned by worker 21.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "joint_smallq"
TARGET = "vllm/model_executor/models"
PREIMAGE_SHA = "f8101b98228bbd472a63a1efe6827583e35c6cc9dede4379b925f2b016670863"
POSTIMAGE_SHA = "cf02419978ce62f18ca908b8d4c9646531fb00d3b07871b310576f963e5ec612"
KERNEL_SHA = "08ad88f6cebfb148aff8d973097d5bcf234e25d144a824385b18b8b078e82dff"
ENV_VAR = "CLEF_JOINT_SMALLQ"
FILES = {"clef_joint_schema.py": POSTIMAGE_SHA, "clef_smallq_kernel.py": KERNEL_SHA}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_atomic(path, data):
    temp = path.with_name(path.name + ".smallq.tmp")
    temp.write_bytes(data)
    os.replace(temp, path)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tree", type=Path, default=Path("/workspace/yangchen/work/src/vllm"))
    ap.add_argument("--backup-dir", type=Path,
                    default=Path("/workspace/yangchen/logs/joint-smallq-20261119/phase0"))
    group = ap.add_mutually_exclusive_group(required=True)
    for action in ("check", "apply", "revert"):
        group.add_argument("--" + action, action="store_true")
    ap.add_argument("--manifest", type=Path)
    args = ap.parse_args(argv)
    tree = args.tree.expanduser().resolve()
    directory = tree / TARGET
    head = directory / "clef_joint_schema.py"
    kernel = directory / "clef_smallq_kernel.py"
    backup = args.backup_dir.expanduser().resolve() / "clef_joint_schema.py.preimage"
    report = {"schema": "joint-smallq-installer/1", "tree": str(tree),
              "action": "check" if args.check else "apply" if args.apply else "revert",
              "preimage_sha256": PREIMAGE_SHA, "postimage_sha256": POSTIMAGE_SHA,
              "kernel_sha256": KERNEL_SHA, "env": {"name": ENV_VAR, "default": "0"},
              "ready": False}
    fd = None
    try:
        fd = os.open(tree, os.O_RDONLY)
        fcntl.flock(fd, fcntl.LOCK_SH if args.check else fcntl.LOCK_EX)
        current = sha256(head)
        if current not in (PREIMAGE_SHA, POSTIMAGE_SHA):
            raise ValueError("unknown head SHA-256: " + current)
        if kernel.exists() and sha256(kernel) != KERNEL_SHA:
            raise ValueError("unknown small-Q kernel SHA-256")
        state = "preimage" if current == PREIMAGE_SHA else "postimage"
        report["state_before"] = state
        if args.check:
            if state == "postimage" and (not kernel.exists() or sha256(kernel) != KERNEL_SHA):
                raise ValueError("postimage requires matching kernel")
            report["ready"] = state == "postimage"
        elif args.apply:
            for name, expected in FILES.items():
                if sha256(SOURCE / name) != expected:
                    raise ValueError("bundled source SHA-256 mismatch: " + name)
            if state == "preimage":
                if backup.exists() and sha256(backup) != PREIMAGE_SHA:
                    raise ValueError("preimage backup SHA-256 mismatch")
                if not backup.exists():
                    backup.parent.mkdir(parents=True, exist_ok=True)
                    write_atomic(backup, head.read_bytes())
                try:
                    write_atomic(kernel, (SOURCE / kernel.name).read_bytes())
                    write_atomic(head, (SOURCE / head.name).read_bytes())
                except OSError:
                    write_atomic(head, backup.read_bytes())
                    kernel.unlink(missing_ok=True)
                    raise
            if sha256(head) != POSTIMAGE_SHA or sha256(kernel) != KERNEL_SHA:
                raise ValueError("installed file verification failed")
            state = "postimage"
            report["ready"] = True
        else:
            if state == "postimage":
                if not backup.is_file() or sha256(backup) != PREIMAGE_SHA:
                    raise ValueError("verified preimage backup required")
                write_atomic(head, backup.read_bytes())
            kernel.unlink(missing_ok=True)
            state = "preimage"
            report["ready"] = sha256(head) == PREIMAGE_SHA
        report["state_after"] = state
        report["sha256_after"] = sha256(head)
    except (OSError, ValueError) as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        if fd is not None:
            os.close(fd)
    if args.manifest:
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        write_atomic(args.manifest, (json.dumps(report, indent=2) + "\n").encode())
    print(json.dumps(report, indent=2))
    return int("error" in report)


if __name__ == "__main__":
    raise SystemExit(main())
