#!/usr/bin/env python3
"""Two-rank correctness and interleaved timing for ClefArAddRmsNorm.

One device window:
  1. correctness with rank-different finite inputs (rank r partial carries a
     distinct offset), reference = same-process dist.all_reduce + native
     npu_add_rms_norm;
  2. interleaved timing, same tensors/layout/warmup/allocation, of
       baseline  : copy x -> dist.all_reduce -> npu_add_rms_norm
       candidate : torch.ops.clef_ar.ar_add_rms_norm
     with a copy-only arm to expose the baseline's restore cost;
  3. per-rank JSON files, per-iteration wall and device samples, no profiler.
Launch with torchrun --nproc_per_node=2.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch_npu
from torch.utils.cpp_extension import load


def build_adapter(adapter_dir: Path, op_api_dir: Path, build_dir: Path):
    venv = Path(torch.__file__).resolve().parents[1]
    include_paths = [
        str(venv / "torch_npu" / "include"),
        str(adapter_dir),
        str(op_api_dir / "include"),
        "/workspace/yangchen/work/src/vllm-ascend/csrc/aclnn_torch_adapter",
        "/usr/local/Ascend/cann-9.2.1/include",
        "/usr/local/Ascend/cann-9.2.1/aarch64-linux/include",
    ]
    return load(
        name="clef_ar_adapter",
        sources=[str(adapter_dir / "clef_ar_adapter.cpp")],
        extra_include_paths=include_paths,
        extra_ldflags=[
            f"-L{venv}/torch_npu/lib", "-ltorch_npu",
            f"-L{op_api_dir / 'lib'}", "-lcust_opapi",
            f"-Wl,-rpath,{op_api_dir / 'lib'}",
        ],
        extra_cflags=["-std=c++17"],
        build_directory=str(build_dir),
        is_python_module=False,
        verbose=False,
    )


def pct(values, q):
    if not values:
        return None
    ordered = sorted(values)
    idx = (len(ordered) - 1) * q
    lo, hi = int(idx), min(int(idx) + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (idx - lo)


def stats(values):
    return {
        "n": len(values),
        "median": statistics.median(values) if values else None,
        "mean": statistics.fmean(values) if values else None,
        "p90": pct(values, 0.90),
        "p99": pct(values, 0.99),
        "min": min(values) if values else None,
        "max": max(values) if values else None,
        "stdev": statistics.pstdev(values) if len(values) > 1 else 0.0,
    }


def timed(fn, warmup, iters):
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    wall, dev = [], []
    for _ in range(iters):
        start = torch.npu.Event(enable_timing=True)
        end = torch.npu.Event(enable_timing=True)
        begin = time.perf_counter()
        start.record()
        fn()
        end.record()
        torch.npu.synchronize()
        wall.append((time.perf_counter() - begin) * 1000.0)
        dev.append(start.elapsed_time(end))
    return wall, dev


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=967)
    parser.add_argument("--hidden", type=int, default=5120)
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--adapter-dir", type=Path, required=True)
    parser.add_argument("--op-api-dir", type=Path, required=True)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.npu.set_device(local_rank)
    dist.init_process_group("hccl", init_method="env://")
    backend = dist.group.WORLD._get_backend(torch.device("npu"))
    group_name = backend.get_hccl_comm_name(rank)
    if rank == 0:
        print(f"group_name={group_name}", flush=True)

    mod = build_adapter(args.adapter_dir, args.op_api_dir, args.build_dir)

    m, n = args.rows, args.hidden
    i = torch.arange(m * n, dtype=torch.float32)
    # Rank-different finite partials: rank r adds 0.1*r, so a correct
    # all-reduce must show the cross-rank term.
    x = ((torch.sin(i * 1e-3) * 0.75) + 0.1 * rank).reshape(m, n).to(torch.bfloat16).npu()
    r = (torch.cos(i * 7e-4) * 0.5).reshape(m, n).to(torch.bfloat16).npu()
    g = (0.5 + 0.5 * torch.sin(torch.arange(n, dtype=torch.float32) * 1e-2)).to(torch.bfloat16).npu()

    out, res_out, rstd, diag = torch.ops.clef_ar.ar_add_rms_norm(x, r, g, args.eps, 1, group_name)
    torch.npu.synchronize()

    total = x.float().clone()
    dist.all_reduce(total, op=dist.ReduceOp.SUM)
    total_bf = total.to(torch.bfloat16)
    ref_y, ref_rstd, ref_res = torch_npu.npu_add_rms_norm(total_bf, r, g, args.eps)
    torch.npu.synchronize()

    dy = (out.float() - ref_y.float()).abs()
    dres = (res_out.float() - ref_res.float()).abs()
    drstd = (rstd.float() - ref_rstd.float().reshape(-1)).abs()
    finite = bool(torch.isfinite(out.float()).all().cpu() and torch.isfinite(res_out.float()).all().cpu())
    correctness = {
        "finite": finite,
        "max_abs_y": float(dy.max().cpu()),
        "max_abs_residual_sum": float(dres.max().cpu()),
        "max_abs_rstd": float(drstd.max().cpu()),
        "y_gt_0.01": int((dy > 0.01).sum().cpu()),
        "elements": int(dy.numel()),
        "diag": [float(v) for v in diag.cpu()],
        "x_rank_offset": 0.1 * rank,
    }
    correctness["numeric_pass"] = bool(
        finite and correctness["max_abs_y"] <= 0.02
        and correctness["max_abs_residual_sum"] <= 0.01
        and correctness["max_abs_rstd"] <= 1e-3)

    # Symmetric preparation: both arms copy their input into a private scratch
    # buffer inside the timed region, so the harness restore cost is identical.
    scratch_b = torch.empty_like(x)
    scratch_c = torch.empty_like(x)

    def arm_baseline():
        scratch_b.copy_(x)
        dist.all_reduce(scratch_b, op=dist.ReduceOp.SUM)
        torch_npu.npu_add_rms_norm(scratch_b, r, g, args.eps)

    def arm_candidate():
        scratch_c.copy_(x)
        torch.ops.clef_ar.ar_add_rms_norm(scratch_c, r, g, args.eps, 1, group_name)

    # Primary comparison: input-ready. The restore copy and its completion sync
    # run OUTSIDE the timed region for both arms; the copy inside arm_baseline /
    # arm_candidate is therefore split out into arm_*_ready.
    def arm_baseline_ready():
        dist.all_reduce(scratch_b, op=dist.ReduceOp.SUM)
        torch_npu.npu_add_rms_norm(scratch_b, r, g, args.eps)

    def arm_candidate_ready():
        torch.ops.clef_ar.ar_add_rms_norm(scratch_c, r, g, args.eps, 1, group_name)

    def prepare(buf):
        buf.copy_(x)
        torch.npu.synchronize()

    for _ in range(args.warmup):
        prepare(scratch_b)
        arm_baseline_ready()
        prepare(scratch_c)
        arm_candidate_ready()
    torch.npu.synchronize()
    run_id = time.strftime("%Y%m%dT%H%M%S")
    rows = []
    for it in range(args.iters):
        order = (("baseline", arm_baseline_ready, scratch_b),
                 ("candidate", arm_candidate_ready, scratch_c)) if it % 2 == 0 \
            else (("candidate", arm_candidate_ready, scratch_c),
                  ("baseline", arm_baseline_ready, scratch_b))
        for arm, fn, buf in order:
            prepare(buf)  # outside the timed region
            start = torch.npu.Event(enable_timing=True)
            end = torch.npu.Event(enable_timing=True)
            begin = time.perf_counter()
            start.record()
            fn()
            end.record()
            torch.npu.synchronize()
            rows.append({"run_id": run_id, "pair_id": it, "arm": arm, "iter": it,
                         "rank": rank,
                         "wall_ms": (time.perf_counter() - begin) * 1000.0,
                         "event_ms": start.elapsed_time(end)})
    by_arm = {"baseline": [], "candidate": []}
    by_arm_dev = {"baseline": [], "candidate": []}
    for row in rows:
        by_arm[row["arm"]].append(row["wall_ms"])
        by_arm_dev[row["arm"]].append(row["event_ms"])
    timings = {arm: {"wall_ms": stats(by_arm[arm]), "device_ms": stats(by_arm_dev[arm]),
                     "wall_samples": by_arm[arm], "device_samples": by_arm_dev[arm]}
               for arm in ("baseline", "candidate")}
    del scratch_b, scratch_c

    import hashlib
    script_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    adapter_so = args.build_dir / "clef_ar_adapter.so"
    adapter_sha = hashlib.sha256(adapter_so.read_bytes()).hexdigest() if adapter_so.is_file() else None
    rank_summary = {
        "run_id": run_id,
        "script_sha256": script_sha,
        "adapter_sha256": adapter_sha,
        "op_api_dir": str(args.op_api_dir),
        "rank": rank,
        "group_name": group_name,
        "rows": m,
        "hidden": n,
        "eps": args.eps,
        "warmup": args.warmup,
        "iters": args.iters,
        "correctness": correctness,
        "timings": timings,
    }
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / f"rank{rank}.rows.jsonl").open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    (out_dir / f"rank{rank}.json").write_text(
        json.dumps(rank_summary, ensure_ascii=False, indent=1) + "\n")
    print(json.dumps({"rank": rank, "correctness": correctness,
                      "timing_median_wall_ms": {k: v["wall_ms"]["median"] for k, v in timings.items()},
                      "timing_median_dev_ms": {k: v["device_ms"]["median"] for k, v in timings.items()}},
                     ensure_ascii=False), flush=True)

    dist.barrier()
    dist.destroy_process_group()
    return 0 if correctness["numeric_pass"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
