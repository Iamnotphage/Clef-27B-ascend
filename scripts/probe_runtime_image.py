#!/usr/bin/env python3
"""Report whether the installed vLLM trees match this submission's patches.

Run this inside the evaluation image before ``start.sh``:

    git clone <repo> && cd Clef-27B-ascend
    python3 scripts/probe_runtime_image.py

Every runtime installer (``src/runtime/apply_*.py``) refuses a tree whose
preimage SHA-256 differs, so a mismatch here means ``start.sh`` cannot start
until the image is replaced or the patches are ported.
"""

import hashlib
import importlib.metadata as md
import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path

# Hash prefixes from the pinned build (vllm 24420b7a, vllm-ascend 89eb8a5e).
KNOWN = {
    "vllm/model_executor/models/clef_joint_schema.py": {
        "197863d5": "joint preimage",
        "f8101b98": "single-field postimage",
        "26d3ad04": "fused-attention postimage",
        "aa3e1758": "legacy fused postimage",
    },
    "vllm/entrypoints/generate/structured_decisions/api_router.py": {
        "bdf11bee": "Tier 2 preimage",
        "1875f0e1": "Tier 2 postimage",
    },
    "vllm_ascend/worker/v2/model_runner.py": {
        "e866aeb2": "Tier 2 preimage",
        "a772213f": "Tier 2 postimage",
    },
    "vllm_ascend/ops/gdn.py": {
        "c8c0d4d8": "conv+QKV preimage",
        "ead25cb2": "conv+QKV postimage",
        "afe6d4e8": "conv+QKV varlen postimage",
    },
}

TARGETS = (
    "vllm/model_executor/models/clef_joint_schema.py",
    "vllm/entrypoints/generate/structured_decisions/api_router.py",
    "vllm/entrypoints/serve/instrumentator/health.py",
    "vllm/model_executor/model_loader/default_loader.py",
    "vllm_ascend/ops/gdn.py",
    "vllm_ascend/device/device_op.py",
    "vllm_ascend/ops/gdn_attn_builder.py",
    "vllm_ascend/worker/v2/model_runner.py",
    "vllm_ascend/ops/vocab_parallel_embedding.py",
)


def version(name):
    try:
        return md.version(name)
    except md.PackageNotFoundError:
        return "MISSING"


def pkgdir(name):
    spec = importlib.util.find_spec(name)
    if not spec or not spec.origin:
        return None
    return Path(spec.origin).resolve().parent


def sha256(path):
    if not path or not path.is_file():
        return "-"
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    print("python", sys.version.split()[0], sys.platform)
    for name in ("vllm", "vllm-ascend", "torch", "torch-npu", "triton",
                 "triton-ascend", "transformers", "flash-attn-npu"):
        print(f"  {name:24} {version(name)}")
    print("  ASCEND_HOME_PATH        ", os.environ.get("ASCEND_HOME_PATH", "unset"))
    print("  npu-smi                 ", shutil.which("npu-smi") or "missing")

    vllm, ascend = pkgdir("vllm"), pkgdir("vllm_ascend")
    print("  vllm tree               ", vllm)
    print("  vllm_ascend tree        ", ascend)
    if not vllm or not ascend:
        print("\nRESULT: vLLM or vLLM-Ascend is not importable in this environment.")
        return 1

    mismatches = 0
    print("\nfile hashes")
    for rel in TARGETS:
        root = vllm if rel.startswith("vllm/") else ascend
        path = root / rel.split("/", 1)[1]
        digest = sha256(path)
        note = ""
        if rel in KNOWN:
            note = KNOWN[rel].get(digest[:8], "UNKNOWN")
        print(f"  {rel:58} {digest[:16] if digest != '-' else 'ABSENT':16} {note}")
        if digest == "-" or (rel in KNOWN and digest[:8] not in KNOWN[rel]):
            mismatches += 1

    marker = "ClefFlashForJointSchema"
    found = []
    for root, name in ((vllm, "vllm"), (ascend, "vllm_ascend")):
        hit = subprocess.run(
            ["grep", "-rl", marker, str(root)], capture_output=True, text=True)
        if hit.returncode == 0 and hit.stdout.strip():
            found.append(name)
    print(f"\n{marker} registered in: {', '.join(found) or 'NOT FOUND'}")

    print("\nRESULT:", "MATCHES_THE_PINNED_BUILD" if mismatches == 0
          else f"{mismatches} file(s) differ from the pinned build")
    return 0 if mismatches == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
