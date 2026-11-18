#!/usr/bin/env python3
"""Two-rank correctness for the E first version (MatMul+AR+AddRMS, ND weights)."""
from __future__ import annotations
import argparse, json, os
from pathlib import Path
import torch, torch.distributed as dist, torch_npu
from safetensors import safe_open
from torch.utils.cpp_extension import load


def build(adapter_dir, op_api_dir, build_dir):
    venv = Path(torch.__file__).resolve().parents[1]
    return load(name="clef_e_adapter", sources=[str(adapter_dir / "clef_e_adapter.cpp")],
                extra_include_paths=[str(venv / "torch_npu" / "include"), str(adapter_dir),
                                     str(op_api_dir / "include"),
                                     "/usr/local/Ascend/cann-9.2.1/include",
                                     "/usr/local/Ascend/cann-9.2.1/aarch64-linux/include"],
                extra_ldflags=[f"-L{venv}/torch_npu/lib", "-ltorch_npu", f"-L{op_api_dir / 'lib'}",
                               "-lcust_opapi", f"-Wl,-rpath,{op_api_dir / 'lib'}"],
                extra_cflags=["-std=c++17"], build_directory=str(build_dir),
                is_python_module=False, verbose=False)


def weight_slice(model_dir: Path, layer: int, rank: int, tp: int, key: str):
    idx = json.loads((model_dir / "model.safetensors.index.json").read_text())["weight_map"]
    with safe_open(model_dir / idx[key], framework="pt", device="cpu") as h:
        w = h.get_tensor(key)
    per = w.shape[0] // tp if key.endswith("gate_proj.weight") else w.shape[1] // tp
    if key.endswith("gate_proj.weight"):
        return w[rank * per:(rank + 1) * per, :], per
    return w[:, rank * per:(rank + 1) * per], per


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=967)
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--eps", type=float, default=1e-6)
    ap.add_argument("--block-dim", type=int, default=-1)
    ap.add_argument("--use-hccl", type=int, default=1)
    ap.add_argument("--mm-enable", type=int, default=1)
    ap.add_argument("--mm-dim", type=int, default=0)
    ap.add_argument("--mm-mode", type=int, default=2)
    ap.add_argument("--mm-bufspace", type=int, default=0)
    ap.add_argument("--adapter-dir", type=Path, required=True)
    ap.add_argument("--op-api-dir", type=Path, required=True)
    ap.add_argument("--build-dir", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    rank = int(os.environ["RANK"])
    torch.npu.set_device(int(os.environ.get("LOCAL_RANK", rank)))
    dist.init_process_group("hccl", init_method="env://")
    group = dist.group.WORLD._get_backend(torch.device("npu")).get_hccl_comm_name(rank)
    mod = build(a.adapter_dir, a.op_api_dir, a.build_dir)

    w_cpu, inter = weight_slice(Path("/workspace/yangchen/models/Clef-27B"), a.layer, rank, 2,
                                f"model.language_model.layers.{a.layer}.mlp.down_proj.weight")
    k, n = inter, 5120  # down: W[K=inter, N=hidden] per rank slice is [5120,8704] -> x@w needs [M,8704]x[8704,5120]
    # down_proj full weight is [hidden, inter]; our slice is [5120, 8704] (N,K).
    # For the op contract x[M,K] @ w[K,N], pass w = slice.T (contiguous).
    w = w_cpu.t().contiguous().npu()
    m = a.rows
    g = torch.Generator().manual_seed(20261118)
    x = torch.randn(m, k, generator=g).to(torch.bfloat16).npu()
    r = torch.randn(m, n, generator=g).to(torch.bfloat16).npu()
    gamma = torch.rand(n, generator=g).to(torch.bfloat16).npu()

    out, res_out, rstd, diag = torch.ops.clef_e.mm_ar_norm(x, w, r, gamma, a.eps, 1, group, a.use_hccl, a.block_dim, a.mm_enable, a.mm_dim, a.mm_mode, a.mm_bufspace)
    torch.npu.synchronize()

    partial = torch.matmul(x, w).float()
    dist.all_reduce(partial, op=dist.ReduceOp.SUM)
    ref_y, ref_rstd, ref_res = torch_npu.npu_add_rms_norm(partial.to(torch.bfloat16), r, gamma, a.eps)
    torch.npu.synchronize()
    dy = (out.float() - ref_y.float()).abs()
    dres = (res_out.float() - ref_res.float()).abs()
    drstd = (rstd.float() - ref_rstd.float().reshape(-1)).abs()
    summary = {"rank": rank, "group": group, "m": m, "k": k, "n": n,
               "diag": [float(v) for v in diag.cpu()],
               "max_abs_y": float(dy.max().cpu()), "max_abs_residual_sum": float(dres.max().cpu()),
               "max_abs_rstd": float(drstd.max().cpu()), "y_gt_0.01": int((dy > 0.01).sum().cpu()),
               "elements": int(dy.numel())}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(summary, ensure_ascii=False, indent=1) + "\n")
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    dist.barrier(); dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
