#!/usr/bin/env python3
"""C-group probe: actual KFC all-reduce latency at the tile payloads.

For N in {1280, 2560, 5120} on M=967 the op is run twice on identical inputs:
use_hccl=1 (AR + Add + RMSNorm) and use_hccl=0 (Add + RMSNorm only). The
difference is the incremental cost of the real KFC collective at that payload
(2.47552 / 4.95104 / 9.90208 MB), measured with AB/BA interleaving in one
build and one cache state. Per-rank JSON with raw wall/event samples.
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
        str(venv / "torch_npu" / "include"), str(adapter_dir),
        str(op_api_dir / "include"),
        "/workspace/yangchen/work/src/vllm-ascend/csrc/aclnn_torch_adapter",
        "/usr/local/Ascend/cann-9.2.1/include",
        "/usr/local/Ascend/cann-9.2.1/aarch64-linux/include",
    ]
    return load(name="clef_ar_adapter", sources=[str(adapter_dir / "clef_ar_adapter.cpp")],
                extra_include_paths=include_paths,
                extra_ldflags=[f"-L{venv}/torch_npu/lib", "-ltorch_npu",
                               f"-L{op_api_dir / 'lib'}", "-lcust_opapi",
                               f"-Wl,-rpath,{op_api_dir / 'lib'}"],
                extra_cflags=["-std=c++17"], build_directory=str(build_dir),
                is_python_module=False, verbose=False)


def med(v):
    return statistics.median(v) if v else None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=967)
    parser.add_argument("--hidden", default="1280,2560,5120")
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--adapter-dir", type=Path, required=True)
    parser.add_argument("--op-api-dir", type=Path, required=True)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("on", "off", "both"), default="both")
    args = parser.parse_args()

    rank = int(os.environ["RANK"])
    torch.npu.set_device(int(os.environ.get("LOCAL_RANK", rank)))
    dist.init_process_group("hccl", init_method="env://")
    group_name = dist.group.WORLD._get_backend(torch.device("npu")).get_hccl_comm_name(rank)
    mod = build_adapter(args.adapter_dir, args.op_api_dir, args.build_dir)

    m = args.rows
    run_id = time.strftime("%Y%m%dT%H%M%S")
    results = {"run_id": run_id, "group_name": group_name, "rank": rank, "shapes": {}}
    for n in [int(v) for v in args.hidden.split(",")]:
        i = torch.arange(m * n, dtype=torch.float32)
        x = ((torch.sin(i * 1e-3) * 0.75) + 0.1 * rank).reshape(m, n).to(torch.bfloat16).npu()
        r = (torch.cos(i * 7e-4) * 0.5).reshape(m, n).to(torch.bfloat16).npu()
        g = (0.5 + 0.5 * torch.sin(torch.arange(n, dtype=torch.float32) * 1e-2)).to(torch.bfloat16).npu()

        def arm(hccl):
            return lambda: torch.ops.clef_ar.ar_add_rms_norm(x, r, g, args.eps, 1, group_name, hccl)

        if args.mode == "both":
            arms = (("on", arm(1)), ("off", arm(0)))
        else:
            arms = ((args.mode, arm(1 if args.mode == "on" else 0)),)
        for _, fn in arms:
            for _ in range(args.warmup):
                fn()
        # correctness of the measured mode against the same-process reference
        if args.mode == "off":
            exp_in = x
        else:
            exp_in = x.float().clone()
            dist.all_reduce(exp_in, op=dist.ReduceOp.SUM)
            exp_in = exp_in.to(torch.bfloat16)
        ref_y, ref_rstd, ref_res = torch_npu.npu_add_rms_norm(exp_in, r, g, args.eps)
        got_y, got_res, got_rstd, got_diag = torch.ops.clef_ar.ar_add_rms_norm(
            x, r, g, args.eps, 1, group_name, 1 if args.mode == "on" else 0)
        torch.npu.synchronize()
        check = {"max_abs_y": float((got_y.float() - ref_y.float()).abs().max().cpu()),
                 "max_abs_residual_sum": float((got_res.float() - ref_res.float()).abs().max().cpu()),
                 "max_abs_rstd": float((got_rstd.float() - ref_rstd.float().reshape(-1)).abs().max().cpu()),
                 "diag": [float(v) for v in got_diag.cpu()]}
        print(json.dumps({"rank": rank, "n": n, "mode": args.mode, "check": check}), flush=True)
        torch.npu.synchronize()

        rows = []
        for it in range(args.iters):
            order = arms if it % 2 == 0 else tuple(reversed(arms))
            for arm_name, fn in order:
                start = torch.npu.Event(enable_timing=True)
                end = torch.npu.Event(enable_timing=True)
                begin = time.perf_counter()
                start.record(); fn(); end.record()
                torch.npu.synchronize()
                rows.append({"arm": arm_name, "iter": it, "wall_ms": (time.perf_counter() - begin) * 1000.0,
                             "event_ms": start.elapsed_time(end)})
        on_wall = [r["wall_ms"] for r in rows if r["arm"] == "on"]
        off_wall = [r["wall_ms"] for r in rows if r["arm"] == "off"]
        on_dev = [r["event_ms"] for r in rows if r["arm"] == "on"]
        off_dev = [r["event_ms"] for r in rows if r["arm"] == "off"]
        if not on_wall or not off_wall:
            results["shapes"][str(n)] = {"payload_mb": m * n * 2 / 1024**2, "mode": args.mode,
                                         "wall_ms": med(on_wall or off_wall),
                                         "dev_ms": med(on_dev or off_dev), "rows": rows}
            print(json.dumps({"rank": rank, "n": n, "mode": args.mode,
                              "wall": round(med(on_wall or off_wall), 4),
                              "dev": round(med(on_dev or off_dev), 4)}), flush=True)
            del x, r, g
            torch.npu.empty_cache()
            continue
        results["shapes"][str(n)] = {
            "payload_mb": m * n * 2 / 1024**2,
            "ar_norm_wall_ms": med(on_wall), "norm_only_wall_ms": med(off_wall),
            "ar_increment_wall_ms": med([a - b for a, b in zip(on_wall, off_wall)]),
            "ar_norm_dev_ms": med(on_dev), "norm_only_dev_ms": med(off_dev),
            "ar_increment_dev_ms": med([a - b for a, b in zip(on_dev, off_dev)]),
            "rows": rows,
        }
        print(json.dumps({"rank": rank, "n": n,
                          "payload_mb": round(m * n * 2 / 1024**2, 4),
                          "ar_norm_wall": round(med(on_wall), 4),
                          "norm_only_wall": round(med(off_wall), 4),
                          "ar_increment_wall": round(med([a - b for a, b in zip(on_wall, off_wall)]), 4),
                          "ar_increment_dev": round(med([a - b for a, b in zip(on_dev, off_dev)]), 4)}),
              flush=True)
        del x, r, g
        torch.npu.empty_cache()

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"rank{rank}.json").write_text(json.dumps(results, ensure_ascii=False, indent=1) + "\n")
    dist.barrier()
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
