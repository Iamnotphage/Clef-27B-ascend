#!/usr/bin/env python3
"""Install/revert the GDN chunk-128 optimization for the decision-only prefill.

The GDN decision path is numerically stable with L2-normalized q/k (the
service's ClefGdnQkvNorm output); chunk 128 is 12-20% faster on the fla_npu
kernel.  The patch runs after the QKV postimage and changes the two decision
branches plus the metadata chunk size.  At batch-1 OD it passes the official
2PP gates (OD three metrics and ERQA 217/400); the per-request differences are
recorded in the experiment docs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

DEFAULT_TREE = Path("/workspace/yangchen/work/src/vllm-ascend")
DEFAULT_BACKUP = Path("/workspace/yangchen/logs/gdn-chunk128-20261117/phase0")
PATCH_VERSION = 1

PREIMAGE_SHA = {
    "vllm_ascend/device/device_op.py":
        "19cf5d0f34015cc1e4f5f7639a084d3ad64ed24e0a7a1a3b1946b475eec14383",
    "vllm_ascend/ops/gdn_attn_builder.py":
        "1f464f9c3c09fc1f8b9f8727dffd53ee93cc251d9631e77b6561d91ae117bec0",
}
POSTIMAGE_SHA = {
    "vllm_ascend/device/device_op.py":
        "bf027ec99a3d069f913600ddfcb76283f73a4114febe46b151c113c62450c226",
    "vllm_ascend/ops/gdn_attn_builder.py":
        "f81eeefa957790398adbbde35fd3e7b2c5aaf3eeac9273400c7412f753ba4a8a",
}
#: The stacked adaptive-chunk postimage keeps the same default behavior
#: (threshold 0 is all-128), so this installer treats it as patched.
ADAPTIVE_POSTIMAGE_SHA = {
    "vllm_ascend/device/device_op.py":
        "2abb3389f8502f17f0f0f0f567d50aed035623dae0740355a66aaa5d5a035b59",
    "vllm_ascend/ops/gdn_attn_builder.py":
        "bf7ca3f93f8ed54957e0e012c14a932fd01d2c459e6dbce487c2909890d8de23",
}
TARGETS = {
    "vllm_ascend/device/device_op.py": "patch_device_op",
    "vllm_ascend/ops/gdn_attn_builder.py": "patch_gdn_attn_builder",
}


def sha256_bytes(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def status(path: Path, relative: str) -> str:
    if not path.is_file():
        return "missing"
    digest = sha256_file(path)
    if digest == PREIMAGE_SHA[relative]:
        return "preimage"
    if digest == POSTIMAGE_SHA[relative]:
        return "patched"
    if digest == ADAPTIVE_POSTIMAGE_SHA[relative]:
        # The stacked adaptive-chunk patch preserves the chunk-128 default at
        # threshold 0; do not overwrite it on every service start.
        return "patched"
    return "unknown"


def patch_device_op(text: str) -> str:
    first = "                chunk_size=64,\n"
    if text.count(first) != 1:
        raise SystemExit("device_op.py: first decision chunk anchor is not unique")
    text = text.replace(first, "                chunk_size=128,\n", 1)
    second = ("            output_final_state=not decision_only,\n"
              "            chunk_size=64,\n")
    if text.count(second) != 1:
        raise SystemExit("device_op.py: second decision chunk anchor is not unique")
    return text.replace(second,
                        "            output_final_state=not decision_only,\n"
                        "            chunk_size=128,\n", 1)


def patch_gdn_attn_builder(text: str) -> str:
    anchor = "_GDN_CHUNK_SIZE = 64\n"
    if text.count(anchor) != 1:
        raise SystemExit("gdn_attn_builder.py: chunk-size anchor is not unique")
    return text.replace(anchor, "_GDN_CHUNK_SIZE = 128\n", 1)


PATCHERS = {
    "patch_device_op": patch_device_op,
    "patch_gdn_attn_builder": patch_gdn_attn_builder,
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tree", type=Path, default=DEFAULT_TREE)
    parser.add_argument("--backup-dir", type=Path, default=DEFAULT_BACKUP)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--check", action="store_true")
    group.add_argument("--apply", action="store_true")
    group.add_argument("--revert", action="store_true")
    args = parser.parse_args()

    report = {
        "tree": str(args.tree),
        "patch_version": PATCH_VERSION,
        "action": "check" if args.check else ("apply" if args.apply else "revert"),
        "files": {},
    }
    for relative in TARGETS:
        path = args.tree / relative
        current = status(path, relative)
        item = {"status_before": current}
        if args.check:
            item["sha256"] = sha256_file(path) if path.is_file() else None
            item["preimage_sha256"] = PREIMAGE_SHA[relative]
            item["postimage_sha256"] = POSTIMAGE_SHA[relative]
            item["status_after"] = current
        elif args.apply:
            if current == "patched":
                item["action"] = "already-patched"
            elif current == "preimage":
                patched = PATCHERS[TARGETS[relative]](path.read_text()).encode()
                if sha256_bytes(patched) != POSTIMAGE_SHA[relative]:
                    raise SystemExit(f"{relative}: generated postimage SHA mismatch")
                backup = args.backup_dir / f"{path.name}.preimage"
                if backup.exists() and sha256_file(backup) != PREIMAGE_SHA[relative]:
                    raise SystemExit(f"{relative}: preimage backup SHA mismatch")
                if not backup.exists():
                    args.backup_dir.mkdir(parents=True, exist_ok=True)
                    backup.write_bytes(path.read_bytes())
                path.write_bytes(patched)
                item["action"] = "patched"
            else:
                raise SystemExit(f"{relative}: refused {current} source content")
            item["status_after"] = status(path, relative)
        else:
            if current == "preimage":
                item["action"] = "already-preimage"
            elif current == "patched":
                backup = args.backup_dir / f"{path.name}.preimage"
                if not backup.is_file() or sha256_file(backup) != PREIMAGE_SHA[relative]:
                    raise SystemExit(f"{relative}: verified preimage backup required: {backup}")
                path.write_bytes(backup.read_bytes())
                item["action"] = "reverted"
            else:
                raise SystemExit(f"{relative}: refused {current} source content")
            item["status_after"] = status(path, relative)
        report["files"][relative] = item
    report["ready"] = all(
        item.get("status_before") == "patched" or item.get("status_after") == "patched"
        for item in report["files"].values()
    )
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
