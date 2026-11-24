#!/usr/bin/env python3
"""Install/revert the adaptive GDN prefill chunk switch (64 / 128).

Stacks on the verified chunk-128 postimage.  The builder precomputes both chunk
index tables from the host ``cu_seqlens`` and the decision path picks chunk 64
when ``CLEF_GDN_CHUNK128_MIN_TOKENS`` is positive and the selected metric
(``CLEF_GDN_CHUNK_RULE=max_seq|total``) is below it.  The default threshold 0
keeps the measured all-128 behavior, so installing alone is not a change.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import time
from pathlib import Path

DEFAULT_TREE = Path("/workspace/yangchen/work/src/vllm-ascend")
DEFAULT_BACKUP = Path("/workspace/yangchen/logs/tp2-alwayson-20261124/phase2")
PATCH_VERSION = 1

PREIMAGE_SHA = {
    "vllm_ascend/device/device_op.py":
        "bf027ec99a3d069f913600ddfcb76283f73a4114febe46b151c113c62450c226",
    "vllm_ascend/ops/gdn_attn_builder.py":
        "f81eeefa957790398adbbde35fd3e7b2c5aaf3eeac9273400c7412f753ba4a8a",
}
POSTIMAGE_SHA = {
    "vllm_ascend/device/device_op.py":
        "2abb3389f8502f17f0f0f0f567d50aed035623dae0740355a66aaa5d5a035b59",
    "vllm_ascend/ops/gdn_attn_builder.py":
        "bf7ca3f93f8ed54957e0e012c14a932fd01d2c459e6dbce487c2909890d8de23",
}
TARGETS = {
    "vllm_ascend/device/device_op.py": "patch_device_op",
    "vllm_ascend/ops/gdn_attn_builder.py": "patch_gdn_attn_builder",
}

HELPER = '''def _clef_gdn_chunk(prebuilt_meta):
    """Pick the GDN prefill chunk size for the decision path.

    ``CLEF_GDN_CHUNK128_MIN_TOKENS=0`` keeps the measured chunk-128 default.
    A positive threshold selects chunk 64 when the host-side length metric is
    below it; ``CLEF_GDN_CHUNK_RULE`` chooses ``max_seq`` or ``total``.
    """
    raw = os.environ.get("CLEF_GDN_CHUNK128_MIN_TOKENS", "0").strip()
    try:
        threshold = int(raw or 0)
    except ValueError:
        threshold = 0
    if threshold > 0:
        rule = os.environ.get("CLEF_GDN_CHUNK_RULE", "max_seq").strip().lower()
        metric = (prebuilt_meta.total_prefill_tokens if rule == "total"
                  else prebuilt_meta.max_prefill_seq_len)
        if metric < threshold:
            return 64, prebuilt_meta.chunk_indices_64_host
    return 128, prebuilt_meta.chunk_indices_128_host


'''


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
        return "adaptive"
    return "unknown"


def patch_gdn_attn_builder(text: str) -> str:
    anchor_dataclass = "    chunk_indices_chunk64_host: tuple[int, ...]\n"
    if text.count(anchor_dataclass) != 1:
        raise SystemExit("gdn_attn_builder.py: dataclass anchor is not unique")
    text = text.replace(anchor_dataclass, anchor_dataclass +
                        "    chunk_indices_64_host: tuple[int, ...]\n"
                        "    chunk_indices_128_host: tuple[int, ...]\n"
                        "    max_prefill_seq_len: int\n"
                        "    total_prefill_tokens: int\n")
    anchor_return = "    return GDNChunkedPrefillMetadata(\n"
    if text.count(anchor_return) != 1:
        raise SystemExit("gdn_attn_builder.py: return anchor is not unique")
    text = text.replace(anchor_return,
        "    chunk_indices_64_host = tuple(\n"
        "        prepare_chunk_indices(cu_seqlens_cpu, 64).to(torch.int64).reshape(-1).tolist())\n"
        "    chunk_indices_128_host = tuple(chunk_indices_chunk64.to(torch.int64).reshape(-1).tolist())\n"
        "    prefill_seq_lens = cu_seqlens_cpu[1:] - cu_seqlens_cpu[:-1]\n"
        "    max_prefill_seq_len = (int(prefill_seq_lens.max().item())\n"
        "                           if prefill_seq_lens.numel() else 0)\n"
        "    total_prefill_tokens = int(cu_seqlens_cpu[-1].item())\n"
        + anchor_return)
    anchor_hosts = ("        chunk_indices_chunk64_host="
                    "tuple(chunk_indices_chunk64.to(torch.int64).reshape(-1).tolist()),\n")
    if text.count(anchor_hosts) != 1:
        raise SystemExit("gdn_attn_builder.py: host pack anchor is not unique")
    text = text.replace(anchor_hosts, anchor_hosts +
        "        chunk_indices_64_host=chunk_indices_64_host,\n"
        "        chunk_indices_128_host=chunk_indices_128_host,\n"
        "        max_prefill_seq_len=max_prefill_seq_len,\n"
        "        total_prefill_tokens=total_prefill_tokens,\n")
    return text


def patch_device_op(text: str) -> str:
    marker = "class A5DeviceAdaptor(BaseDeviceAdaptor):"
    if text.count(marker) != 1:
        raise SystemExit("device_op.py: A5 class anchor is not unique")
    head, tail = text.split(marker, 1)
    class_anchor = "class BaseDeviceAdaptor:"
    if head.count(class_anchor) != 1:
        raise SystemExit("device_op.py: Base class anchor is not unique")
    head = head.replace(class_anchor, HELPER + class_anchor)
    for indent in ("            ", "        "):
        anchor_meta = indent + "chunk_indices = prebuilt_meta.chunk_indices_chunk64_host\n"
        if head.count(anchor_meta) != 1:
            raise SystemExit(f"device_op.py: meta anchor at indent {len(indent)} is not unique")
        head = head.replace(anchor_meta,
                            indent + "chunk_size, chunk_indices = _clef_gdn_chunk(prebuilt_meta)\n")
    if head.count("chunk_size=128,") != 2:
        raise SystemExit("device_op.py: chunk_size anchors are not exactly two")
    head = head.replace("chunk_size=128,", "chunk_size=chunk_size,")
    return head + marker + tail


PATCHERS = {
    "patch_device_op": patch_device_op,
    "patch_gdn_attn_builder": patch_gdn_attn_builder,
}


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
    report = {
        "schema": "gdn-adaptive-chunk-installer/1",
        "tree": str(tree),
        "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "action": "check" if args.check else "apply" if args.apply else "revert",
        "patch_version": PATCH_VERSION,
        "env": {
            "threshold": {"name": "CLEF_GDN_CHUNK128_MIN_TOKENS", "default": "0"},
            "rule": {"name": "CLEF_GDN_CHUNK_RULE", "default": "max_seq"},
        },
        "files": {},
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
        ready = True
        for relative, patcher_name in TARGETS.items():
            path = tree / relative
            item = {
                "preimage_sha256": PREIMAGE_SHA[relative],
                "postimage_sha256": POSTIMAGE_SHA[relative],
                "path": str(path),
            }
            report["files"][relative] = item
            before = status(path, relative)
            item["status_before"] = before
            if before == "unknown":
                raise ValueError(f"{relative}: unknown source SHA-256")
            if args.check:
                item["status_after"] = before
                item["ready"] = before == "adaptive"
                ready = ready and item["ready"]
                continue
            if args.revert:
                backup = backup_dir / relative
                if before == "adaptive":
                    if not backup.is_file() or sha256_file(backup) != PREIMAGE_SHA[relative]:
                        raise ValueError(f"{relative}: verified preimage backup required")
                    path.write_bytes(backup.read_bytes())
                    item["status_after"] = "preimage"
                else:
                    item["status_after"] = before
                item["ready"] = sha256_file(path) == PREIMAGE_SHA[relative]
                ready = ready and item["ready"]
                continue
            if before == "adaptive":
                item["status_after"] = "adaptive"
            else:
                backup = backup_dir / relative
                backup.parent.mkdir(parents=True, exist_ok=True)
                if not backup.exists():
                    backup.write_bytes(path.read_bytes())
                elif sha256_file(backup) != PREIMAGE_SHA[relative]:
                    raise ValueError(f"{relative}: preimage backup SHA-256 mismatch")
                patched = PATCHERS[patcher_name](path.read_text()).encode()
                if sha256_bytes(patched) != POSTIMAGE_SHA[relative]:
                    raise ValueError(f"{relative}: patched SHA-256 mismatch")
                path.write_bytes(patched)
                if sha256_file(path) != POSTIMAGE_SHA[relative]:
                    raise ValueError(f"{relative}: postimage SHA-256 mismatch after write")
                item["backup"] = str(backup)
                item["status_after"] = "adaptive"
            item["ready"] = True
        report["ready"] = ready
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
