#!/usr/bin/env python3
"""Verify two local NPU ranks with HCCL, writing one JSON report per rank.

Run after sourcing scripts/activate_env.sh and releasing the baseline service:
  torchrun --standalone --nnodes=1 --nproc-per-node=2 scripts/verify_hccl.py \
      --output-dir /workspace/yangchen/logs/hccl-20261112

Each rank reduces [0..7] + rank with SUM in FP32 and BF16. The CPU-computed
expected result is [1, 3, 5, 7, 9, 11, 13, 15], exactly representable in both
dtypes. Successful reports require exact equality, barrier, synchronization
and process-group destruction. This script performs a functional smoke check.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def save(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=float, default=120)
    args = parser.parse_args()
    if not math.isfinite(args.timeout_seconds) or args.timeout_seconds <= 0:
        parser.error("--timeout-seconds must be positive and finite")

    environment = {key: os.environ.get(key) for key in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT")}
    rank = int(environment["RANK"]) if environment["RANK"] and environment["RANK"].isdigit() else 0
    output = args.output_dir / f"rank{rank}.json"
    report = {
        "schema_version": 1, "started_at": timestamp(), "passed": False,
        "status": "preparing", "phase": "validate_torchrun_environment",
        "rank": rank, "torchrun_environment": environment,
        "backend": "hccl", "timeout_seconds": args.timeout_seconds,
        "script": {"path": str(Path(__file__).resolve()), "sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
        "collectives": [], "barrier": {"passed": False},
        "destroy_process_group": {"attempted": False, "passed": False},
    }
    save(output, report)
    distributed = None
    operations_passed = False
    try:
        required = ("RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT")
        if any(environment[key] is None for key in required):
            raise ValueError("Launch with torchrun --standalone --nnodes=1 --nproc-per-node=2")
        world_size, local_world_size = int(environment["WORLD_SIZE"]), int(environment["LOCAL_WORLD_SIZE"])
        local_rank = int(environment["LOCAL_RANK"])
        if world_size != 2 or local_world_size != 2 or rank not in (0, 1) or local_rank != rank:
            raise ValueError("This smoke check requires one node with exactly two ranks, using NPU 0 and 1")
        report.update(local_rank=local_rank, world_size=world_size, device=f"npu:{local_rank}")

        import torch
        import torch_npu
        import torch.distributed as distributed

        report["packages"] = {
            "torch": {"version": torch.__version__, "path": torch.__file__},
            "torch_npu": {"version": torch_npu.__version__, "path": torch_npu.__file__},
        }
        available = bool(torch.npu.is_available())
        count = int(torch.npu.device_count())
        report["npu"] = {"available": available, "device_count": count}
        if not available or count < world_size:
            raise RuntimeError("Two visible NPU devices are required")
        torch.npu.set_device(local_rank)
        report["npu"]["device_name"] = torch.npu.get_device_name(local_rank)
        report.update(status="running", phase="init_process_group")
        save(output, report)
        distributed.init_process_group(
            backend="hccl", init_method="env://", rank=rank,
            world_size=world_size, timeout=timedelta(seconds=args.timeout_seconds),
        )
        report["initialized_backend"] = str(distributed.get_backend())
        if report["initialized_backend"] != "hccl":
            raise RuntimeError(f"Unexpected backend: {report['initialized_backend']}")

        input_values = [float(index + rank) for index in range(8)]
        expected_values = [float(world_size * index + sum(range(world_size))) for index in range(8)]
        expected = torch.tensor(expected_values, dtype=torch.float32, device="cpu")
        for dtype in (torch.float32, torch.bfloat16):
            result = {
                "operation": "all_reduce_SUM", "dtype": str(dtype),
                "input_values": input_values, "expected_values": expected_values,
                "comparison": "Exact CPU FP32 equality; all integer values are exactly representable",
                "passed": False, "synchronization_completed": False,
            }
            report["collectives"].append(result)
            report["phase"] = f"all_reduce/{dtype}"
            save(output, report)
            tensor = torch.tensor(input_values, dtype=dtype, device=f"npu:{local_rank}")
            torch.npu.synchronize(local_rank)
            started = time.perf_counter()
            distributed.all_reduce(tensor, op=distributed.ReduceOp.SUM)
            torch.npu.synchronize(local_rank)
            result["synchronized_elapsed_seconds"] = time.perf_counter() - started
            result["synchronization_completed"] = True
            actual = tensor.float().cpu()
            finite = bool(torch.isfinite(actual).all().item())
            result.update(
                output_device=str(tensor.device), output_dtype=str(tensor.dtype),
                actual_values=[value if math.isfinite(value) else str(value) for value in actual.tolist()], output_finite=finite,
                output_nan_count=int(torch.isnan(actual).sum().item()),
                output_inf_count=int(torch.isinf(actual).sum().item()),
                max_abs_error=float((actual - expected).abs().max().item()) if finite else None,
                passed=finite and torch.equal(actual, expected),
            )
            save(output, report)
            if not result["passed"]:
                raise RuntimeError(f"Rank {rank} {dtype} all_reduce result differs from the exact expectation")

        report["phase"] = "barrier_and_synchronize"
        save(output, report)
        distributed.barrier()
        torch.npu.synchronize(local_rank)
        report["barrier"] = {"passed": True, "device_synchronization_completed": True}
        operations_passed = True
    except Exception as exc:
        report.update(error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
    finally:
        if distributed is not None and distributed.is_initialized():
            report["phase"] = "destroy_process_group"
            report["destroy_process_group"]["attempted"] = True
            save(output, report)
            try:
                distributed.destroy_process_group()
                report["destroy_process_group"]["passed"] = True
            except Exception as exc:
                report["destroy_process_group"]["error"] = f"{type(exc).__name__}: {exc}"
                report.setdefault("error", f"destroy_process_group failed: {exc}")
        report["passed"] = bool(operations_passed and report["destroy_process_group"]["passed"] and "error" not in report)
        report.update(status="complete" if report["passed"] else "failed", finished_at=timestamp())
        save(output, report)
        print(json.dumps({"rank": rank, "passed": report["passed"], "status": report["status"], "report": str(output.resolve()), "error": report.get("error")}), flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
