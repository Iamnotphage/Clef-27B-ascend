#!/usr/bin/env python3
"""Child process body for one D0 cell. Only d0_run.py starts this."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--rows", type=int, default=967)
    ap.add_argument("--seed", type=int, default=20261118)
    ap.add_argument("--mm-dim", type=int, default=1)
    ap.add_argument("--use-hccl", type=int, default=2)
    a = ap.parse_args()
    try:
        import torch
        import torch_npu  # noqa: F401
        from d0_executor import run, summarize_failure
    except Exception as exc:                                  # noqa: BLE001
        (a.out_dir / "import-failure.json").write_text(
            json.dumps({"failure": type(exc).__name__, "message": str(exc)}, ensure_ascii=False) + "\n")
        return 3
    try:
        run(torch, torch_npu, a, a.out_dir)
    except Exception as exc:                                  # noqa: BLE001
        (a.out_dir / "trace-unavailable.json").write_text(
            json.dumps(summarize_failure(exc), ensure_ascii=False) + "\n")
        return 4
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
