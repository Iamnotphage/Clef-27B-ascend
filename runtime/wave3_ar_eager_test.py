#!/usr/bin/env python3
"""Eager 2-rank test for ClefArAddRmsNorm through the OpCommand adapter.

Builds the torch adapter with torch.utils.cpp_extension, initialises an HCCL
process group, reads the real hcom group name from the backend, and runs the
op with repeat=1. The reference is dist.all_reduce + torch_npu.npu_add_rms_norm
on the same tensors. Must be launched with torchrun --nproc_per_node=2.
"""
from __future__ import annotations

import argparse
import json
import os
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=967)
    parser.add_argument("--hidden", type=int, default=5120)
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument("--adapter-dir", type=Path, required=True)
    parser.add_argument("--op-api-dir", type=Path, required=True)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.npu.set_device(local_rank)
    dist.init_process_group("hccl", init_method="env://")

    pg = dist.group.WORLD
    backend = pg._get_backend(torch.device("npu"))
    group_name = backend.get_hccl_comm_name(rank)
    if rank == 0:
        print(f"group_name={group_name}", flush=True)

    mod = build_adapter(args.adapter_dir, args.op_api_dir, args.build_dir)

    m, n = args.rows, args.hidden
    i = torch.arange(m * n, dtype=torch.float32)
    # Each rank gets a different partial so the SUM is non-trivial.
    x = ((torch.sin(i * 1e-3) * 0.75) + 0.1 * rank).reshape(m, n).to(torch.bfloat16).npu()
    r = (torch.cos(i * 7e-4) * 0.5).reshape(m, n).to(torch.bfloat16).npu()
    g = (0.5 + 0.5 * torch.sin(torch.arange(n, dtype=torch.float32) * 1e-2)).to(torch.bfloat16).npu()

    m_pad = (m + 7) // 8 * 8
    rstd_storage = torch.empty(m_pad, dtype=torch.float32, device=x.device)
    rstd_view = rstd_storage[:m]
    out, res_out, rstd, diag = torch.ops.clef_ar.ar_add_rms_norm(x, r, g, args.eps, 1, group_name)
    torch.npu.synchronize()

    total = x.float().clone()
    dist.all_reduce(total, op=dist.ReduceOp.SUM)
    total = total.to(torch.bfloat16)
    ref_y, ref_rstd, ref_res = torch_npu.npu_add_rms_norm(total, r, g, args.eps)
    torch.npu.synchronize()

    y_self = (res_out.float() * rstd.reshape(-1, 1).float() * g.float()).to(torch.bfloat16)
    d_self = (out.float() - y_self.float()).abs()
    if rank == 0:
        print("dbg ours_y[0:4]", [float(v) for v in out[0, :4].float().cpu()], flush=True)
        print("dbg ref_y[0:4]", [float(v) for v in ref_y[0, :4].float().cpu()], flush=True)
        print("dbg self-consistency max|out - z*rstd*g| =", float(d_self.max().cpu()),
              "gt0.01 =", int((d_self > 0.01).sum().cpu()), flush=True)
    dy = (out.float() - ref_y.float()).abs()
    dres = (res_out.float() - ref_res.float()).abs()
    drstd = (rstd.float() - ref_rstd.float().reshape(-1)).abs()
    summary = {
        "rank": rank,
        "group_name": group_name,
        "rows": m,
        "hidden": n,
        "max_abs_y": float(dy.max().cpu()),
        "max_abs_residual_sum": float(dres.max().cpu()),
        "max_abs_rstd": float(drstd.max().cpu()),
        "y_gt_0.01": int((dy > 0.01).sum().cpu()),
        "elements": int(dy.numel()),
    }
    if rank == 0:
        print("dbg x[0:4]", [float(v) for v in x[0, :4].float().cpu()], flush=True)
        print("dbg total[0:4]", [float(v) for v in total[0, :4].float().cpu()], flush=True)
        print("dbg ours_res[0:4]", [float(v) for v in res_out[0, :4].float().cpu()], flush=True)
        print("dbg ref_res[0:4]", [float(v) for v in ref_res[0, :4].float().cpu()], flush=True)
        print("dbg ours_rstd[0:4]", [float(v) for v in rstd[:4].cpu()], flush=True)
        print("dbg diag[rank,dim,rc,useHccl]", [float(v) for v in diag.cpu()], flush=True)
        print("dbg ref_rstd[0:4]", [float(v) for v in ref_rstd[:4].reshape(-1).cpu()], flush=True)
        print(json.dumps(summary, ensure_ascii=False), flush=True)
        args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=1) + "\n")
    dist.barrier()
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
