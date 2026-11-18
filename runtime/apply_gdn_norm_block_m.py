#!/usr/bin/env python3
"""Install/revert the GDN output-norm ``BLOCK_M`` probe knob.

The frozen scope is recorded in ``docs/gdn-norm-block-m-2026-11-18.md``:

* target: ``vllm_ascend/ops/triton/layernorm_gated.py`` in the pinned
  vLLM-Ascend tree (commit ``89eb8a5e7a76d085b453a1aeadb6883e12d5ea2d``).
* change: ``BLOCK_M`` becomes ``int(os.environ.get("CLEF_GDN_NORM_BLOCK_M",
  "64"))`` with a ``(16, 32, 64, 128)`` guard, plus the ``import os`` the
  lookup needs.  The unset default stays 64.
* frozen: the wrapper signature, ``out``/``mean``/``rstd`` allocation,
  ``BLOCK_N``, ``grid = (triton.cdiv(M, BLOCK_M), ngroups)`` and the kernel
  body.  No ``num_warps``/``num_stages`` argument is added.

The installer is hash-gated: it only touches the exact preimage below, it
verifies the postimage hash after writing, and it refuses unknown content.
The bundled preimage lives under ``runtime/gdn_norm_block_m/`` so the
transform can be tested offline without a device.
"""

from __future__ import annotations

import argparse
import difflib
import fcntl
import hashlib
import json
import os
import time
from pathlib import Path

DEFAULT_TREE = Path("/workspace/yangchen/work/src/vllm-ascend")
DEFAULT_BACKUP = Path("/workspace/yangchen/logs/gdn-norm-block-m-20261118/phase0")
TARGET = "vllm_ascend/ops/triton/layernorm_gated.py"
PREIMAGE_SHA = "68f61d7f13547a11ca38eb33c0cd560dc6a569ad97c7fbc6681d1370db70a774"
POSTIMAGE_SHA = "a1093e1290afd48b546830a9ae0639f945910f44f9e22f12111b67aa72f8fe6c"
ENV_VAR = "CLEF_GDN_NORM_BLOCK_M"
ALLOWED_BLOCK_M = (16, 32, 64, 128)

_IMPORT_ANCHOR = "import torch\nfrom vllm.triton_utils import tl, triton\n"
_IMPORT_POSTIMAGE = "import os\n\n" + _IMPORT_ANCHOR
_BLOCK_M_ANCHOR = (
    "    # Choose BLOCK_M: e.g., 16, 32, 64 — depends on NPU vector core capacity\n"
    "    BLOCK_M = 64  # Tune this based on your NPU's register/shared memory\n"
)
_BLOCK_M_POSTIMAGE = (
    "    # Choose BLOCK_M: e.g., 16, 32, 64 — depends on NPU vector core capacity\n"
    "    # Tune this based on your NPU's register/shared memory\n"
    '    BLOCK_M = int(os.environ.get("CLEF_GDN_NORM_BLOCK_M", "64"))\n'
    "    if BLOCK_M not in (16, 32, 64, 128):\n"
    '        raise ValueError(f"unsupported CLEF_GDN_NORM_BLOCK_M={BLOCK_M}")\n'
)


def sha256_bytes(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def status(path: Path) -> str:
    if not path.is_file():
        return "missing"
    digest = sha256_file(path)
    if digest == PREIMAGE_SHA:
        return "preimage"
    if digest == POSTIMAGE_SHA:
        return "patched"
    return "unknown"


def patch_text(text: str) -> str:
    if text.count(_IMPORT_ANCHOR) != 1:
        raise ValueError("layernorm_gated.py: import anchor is not unique")
    if text.count(_BLOCK_M_ANCHOR) != 1:
        raise ValueError("layernorm_gated.py: BLOCK_M anchor is not unique")
    text = text.replace(_IMPORT_ANCHOR, _IMPORT_POSTIMAGE, 1)
    return text.replace(_BLOCK_M_ANCHOR, _BLOCK_M_POSTIMAGE, 1)


def unpatch_text(text: str) -> str:
    if text.count(_IMPORT_POSTIMAGE) != 1:
        raise ValueError("layernorm_gated.py: patched import anchor is not unique")
    if text.count(_BLOCK_M_POSTIMAGE) != 1:
        raise ValueError("layernorm_gated.py: patched BLOCK_M anchor is not unique")
    text = text.replace(_IMPORT_POSTIMAGE, _IMPORT_ANCHOR, 1)
    return text.replace(_BLOCK_M_POSTIMAGE, _BLOCK_M_ANCHOR, 1)


def unified_diff(before: str, after: str) -> str:
    return "".join(difflib.unified_diff(
        before.splitlines(True),
        after.splitlines(True),
        fromfile=TARGET,
        tofile=f"{TARGET} (patched)",
        n=3,
    ))


def write_atomic(path: Path, data: bytes) -> None:
    temporary = path.with_name(path.name + ".block-m.tmp")
    temporary.write_bytes(data)
    os.replace(temporary, path)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tree", type=Path, default=DEFAULT_TREE)
    parser.add_argument("--backup-dir", type=Path, default=DEFAULT_BACKUP)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--revert", action="store_true")
    parser.add_argument("--print-diff", action="store_true")
    parser.add_argument("--manifest", type=Path, default=None)
    args = parser.parse_args(argv)
    if sum((args.check, args.apply, args.revert)) != 1:
        parser.error("choose exactly one of --check, --apply, --revert")

    tree = args.tree.expanduser().resolve()
    backup_dir = args.backup_dir.expanduser().resolve()
    target = tree / TARGET
    backup = backup_dir / "layernorm_gated.py.preimage"
    report = {
        "schema": "gdn-norm-block-m-installer/1",
        "tree": str(tree),
        "file": str(target),
        "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "action": "check" if args.check else "apply" if args.apply else "revert",
        "preimage_sha256": PREIMAGE_SHA,
        "postimage_sha256": POSTIMAGE_SHA,
        "env": {"name": ENV_VAR, "default": "64", "allowed": list(ALLOWED_BLOCK_M)},
        "frozen": [
            "wrapper signature",
            "out/mean/rstd allocation",
            "BLOCK_N",
            "grid = (triton.cdiv(M, BLOCK_M), ngroups)",
            "kernel body and launch arguments (no num_warps/num_stages)",
        ],
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
        state = status(target)
        report["state_before"] = state
        report["sha256_before"] = sha256_file(target) if target.is_file() else None
        if state == "missing":
            raise ValueError(f"{target}: missing")
        if state == "unknown":
            raise ValueError(f"{target}: unknown content; refusing to touch it")

        before_text = target.read_text()

        if args.print_diff:
            after_text = patch_text(before_text) if state == "preimage" else unpatch_text(before_text)
            report["diff"] = unified_diff(before_text, after_text)

        if args.check:
            report["state_after"] = state
        elif args.apply:
            if state == "preimage":
                patched = patch_text(before_text)
                data = patched.encode()
                if sha256_bytes(data) != POSTIMAGE_SHA:
                    raise ValueError("patched text does not match POSTIMAGE_SHA; anchors drifted")
                original = target.read_bytes()
                if backup.exists() and sha256_file(backup) != PREIMAGE_SHA:
                    raise ValueError(f"{backup}: existing backup is not the preimage")
                if not backup.exists():
                    backup.parent.mkdir(parents=True, exist_ok=True)
                    backup.write_bytes(original)
                write_atomic(target, data)
                try:
                    if sha256_file(target) != POSTIMAGE_SHA:
                        raise ValueError("postimage SHA mismatch after write")
                except ValueError:
                    write_atomic(target, original)
                    raise
                state = "patched"
            report["state_after"] = state
        else:
            if state == "patched":
                if backup.is_file() and sha256_file(backup) == PREIMAGE_SHA:
                    data = backup.read_bytes()
                else:
                    data = unpatch_text(before_text).encode()
                    if sha256_bytes(data) != PREIMAGE_SHA:
                        raise ValueError("reverted text does not match PREIMAGE_SHA")
                write_atomic(target, data)
                if sha256_file(target) != PREIMAGE_SHA:
                    raise ValueError("revert did not restore PREIMAGE_SHA")
                state = "preimage"
            report["state_after"] = state

        report["sha256_after"] = sha256_file(target)
        report["knob_present"] = report["state_after"] == "patched"
        report["ready"] = report["state_after"] != "unknown"
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
