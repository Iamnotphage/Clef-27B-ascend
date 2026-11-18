"""CPU-only identity/diagnostic checks; no torch or device imports."""
from __future__ import annotations

import hashlib
from pathlib import Path

MAGIC = 0x4E312519


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def checksum(words):
    h = 2166136261
    for x in words:
        h = ((h ^ (x & 0xFFFFFFFF)) * 16777619) & 0xFFFFFFFF
    return h


def intervals(m, cores):
    return [(m * i // cores, m * (i + 1) // cores) for i in range(cores)]


def parse_diag(rows, m, cores, use_hccl, rank):
    errors = []
    if len(rows) != 48 or any(len(r) != 32 for r in rows):
        return {"pass": False, "errors": ["diag shape must be [48,32]"]}
    rows = [[int(w) & 0xFFFFFFFF for w in r] for r in rows]
    for i, (start, end) in enumerate(intervals(m, cores)):
        row = rows[i]
        expected = [MAGIC, 1, i, cores, start, end, use_hccl]
        if row[:7] != expected or row[11:13] != [m, 5120]:
            errors.append(f"slot {i}: identity/interval mismatch")
        if row[31] != checksum(row[:31]):
            errors.append(f"slot {i}: checksum mismatch")
        if any(row[13:31]):
            errors.append(f"slot {i}: reserved fields nonzero")
        expected_comm = [rank, 2, 0, 0] if use_hccl and i == 0 else [0xFFFFFFFF, 0xFFFFFFFF, 0, 0]
        if row[7:11] != expected_comm:
            errors.append(f"slot {i}: communication status mismatch")
    if any(any(r) for r in rows[cores:]):
        errors.append("unlaunched slots must remain zero")
    return {"pass": not errors, "errors": errors, "rows": rows,
            "note": "terminal diagnostic records; numerical writes are judged by poisoned outputs and fixed gate"}


def verify_identity(identity, vendor_root, adapter):
    root = Path(vendor_root).resolve(strict=True)
    if root.name != "clef_native_ar_norm25":
        raise ValueError("unexpected vendor directory")
    if sha256(adapter) != identity["adapter_sha256"]:
        raise ValueError("adapter SHA mismatch")
    for rel, expected in identity["vendor_files"].items():
        path = (root / rel).resolve(strict=True)
        if not path.is_relative_to(root):
            raise ValueError(f"vendor path escaped: {rel}")
        if sha256(path) != expected:
            raise ValueError(f"vendor SHA mismatch: {rel}")
    return root / "op_api/lib/libcust_opapi.so"
