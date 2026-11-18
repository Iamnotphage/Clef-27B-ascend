#!/usr/bin/env python3
"""Child process body for one D0 cell. Only d0_run.py starts this."""
import argparse
import json
import os
import sys
import traceback
from pathlib import Path

def _rank() -> int:
    return int(os.environ.get("RANK", "0"))


def write_failure(out_dir: Path, kind: str, exc: BaseException) -> None:
    """Rank-scoped so two ranks never overwrite each other's failure."""
    rank = _rank()
    (out_dir / f"{kind}-rank{rank}.json").write_text(json.dumps(
        {"rank": rank, "failure": type(exc).__name__, "message": str(exc)[:4000],
         "traceback": traceback.format_exc()}, ensure_ascii=False) + "\n")

sys.path.insert(0, str(Path(__file__).resolve().parent))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--rows", type=int, default=967)
    ap.add_argument("--seed", type=int, default=20261118)
    ap.add_argument("--mm-dim", type=int, default=1)
    ap.add_argument("--use-hccl", type=int, default=2)
    ap.add_argument("--adapter", type=Path, required=True)
    ap.add_argument("--adapter-sha", required=True)
    a = ap.parse_args()
    try:
        import torch
        import torch_npu  # noqa: F401
        import torch.distributed as dist
        from d0_executor import run, summarize_failure
    except Exception as exc:                                  # noqa: BLE001
        write_failure(a.out_dir, "import-failure", exc)
        return 3
    try:
        record = run(torch, torch_npu, a, a.out_dir, dist)
    except Exception as exc:                                  # noqa: BLE001
        write_failure(a.out_dir, "trace-unavailable", exc)
        return 4
    # A numeric gate failure is its own return code so the runner can tell it
    # apart from a crash, while raw trace and claims stay on disk.
    if record is not None and record.get("gate_pass") is False:
        return 6
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
