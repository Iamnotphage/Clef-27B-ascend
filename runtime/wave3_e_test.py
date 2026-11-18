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
    ap.add_argument("--norm-ub-late", type=int, default=1)
    ap.add_argument("--mm-btrans", type=int, default=0)
    ap.add_argument("--mm-step", type=int, default=0)
    ap.add_argument("--probe-partial", action="store_true")
    ap.add_argument("--probe-norm", action="store_true")
    ap.add_argument("--norm-from-residual", type=int, default=0)
    ap.add_argument("--norm-from-x", type=int, default=0)
    ap.add_argument("--w-nz", action="store_true")
    ap.add_argument("--seed", type=int, default=20261118)
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
    g = torch.Generator().manual_seed(a.seed)
    x = torch.randn(m, k, generator=g).to(torch.bfloat16).npu()
    r = torch.randn(m, n, generator=g).to(torch.bfloat16).npu()
    gamma = torch.rand(n, generator=g).to(torch.bfloat16).npu()

    if a.probe_norm:
        # Same-partial Norm isolation: the op runs with mm_enable=0 (no cube)
        # and receives this exact tensor as x, so both arms norm one identical
        # partial; nothing re-computes a GEMM.
        partial = torch.matmul(x, w)
        if a.use_hccl == 1:
            dist.all_reduce(partial, op=dist.ReduceOp.SUM)
        torch.npu.synchronize()
        # The host still validates w.shape[0] == K, so give it a dummy with
        # the partial's own K; no cube runs in this arm.
        w_dummy = torch.empty((partial.size(1), partial.size(1)), dtype=x.dtype,
                              device=x.device)
        out, res_out, rstd, diag = torch.ops.clef_e.mm_ar_norm(
            partial, w_dummy, r, gamma, a.eps, 1, group, a.use_hccl, a.block_dim,
            0, a.mm_dim, a.mm_mode, a.mm_bufspace, a.norm_ub_late, a.mm_btrans, a.mm_step, a.norm_from_residual, a.norm_from_x, 1 if a.w_nz else 0)
        torch.npu.synchronize()
        ry, rrs, rres = torch_npu.npu_add_rms_norm(partial, r, gamma, a.eps)
        torch.npu.synchronize()
        dy = (out.float() - ry.float()).abs()
        dres = (res_out.float() - rres.float()).abs()
        drstd = (rstd.float() - rrs.float().reshape(-1)).abs()
        summary_n = {"rank": rank, "probe": "norm", "use_hccl": a.use_hccl,
                     "same_partial": True,
                     "max_abs_y": float(dy.max().cpu()),
                     "max_abs_residual_sum": float(dres.max().cpu()),
                     "max_abs_rstd": float(drstd.max().cpu()),
                     "y_gt_0.02": int((dy > 0.02).sum().cpu()),
                     "y_gt_0.01": int((dy > 0.01).sum().cpu()),
                     "res_gt_0.01": int((dres > 0.01).sum().cpu()),
                     "elements": int(dy.numel()),
                     "max_abs_partial": float(partial.float().abs().max().cpu()),
                     "max_abs_gamma": float(gamma.float().abs().max().cpu()),
                     "diag": [float(v) for v in diag.cpu()]}
        a.output.parent.mkdir(parents=True, exist_ok=True)
        a.output.with_name(a.output.stem + f"-rank{rank}-norm.json").write_text(
            json.dumps(summary_n, ensure_ascii=False, indent=1) + "\n")
        print(json.dumps(summary_n, ensure_ascii=False), flush=True)
        dist.barrier(); dist.destroy_process_group()
        return 0
    if a.probe_partial:
        r = torch.zeros_like(r)
        gamma = torch.ones_like(gamma)
    w_op = torch_npu.npu_format_cast(w, 29) if a.w_nz else w
    out, res_out, rstd, diag = torch.ops.clef_e.mm_ar_norm(x, w_op, r, gamma, a.eps, 1, group, a.use_hccl, a.block_dim, a.mm_enable, a.mm_dim, a.mm_mode, a.mm_bufspace, a.norm_ub_late, a.mm_btrans, a.mm_step, a.norm_from_residual, a.norm_from_x, 1 if a.w_nz else 0)
    torch.npu.synchronize()

    # Reference A (fp32 reduce, as before) and Reference B (bf16 reduce, the
    # same rounding path as the op: bf16 partial, bf16 all-reduce).
    partial = torch.matmul(x, w).float()
    dist.all_reduce(partial, op=dist.ReduceOp.SUM)
    ref_y, ref_rstd, ref_res = torch_npu.npu_add_rms_norm(partial.to(torch.bfloat16), r, gamma, a.eps)
    partial_bf = torch.matmul(x, w)
    dist.all_reduce(partial_bf, op=dist.ReduceOp.SUM)
    refb_y, refb_rstd, refb_res = torch_npu.npu_add_rms_norm(partial_bf, r, gamma, a.eps)
    torch.npu.synchronize()
    dy = (out.float() - ref_y.float()).abs()
    dres = (res_out.float() - ref_res.float()).abs()
    drstd = (rstd.float() - ref_rstd.float().reshape(-1)).abs()
    dyb = (out.float() - refb_y.float()).abs()
    dresb = (res_out.float() - refb_res.float()).abs()
    drstdb = (rstd.float() - refb_rstd.float().reshape(-1)).abs()
    def bf16_gate(cand, ref):
        """Elementwise BF16 dev gate: |c-r| <= 1e-5 + 0.016*|r| for every element."""
        d = (cand.float() - ref.float()).abs()
        tol = 1e-5 + 0.016 * ref.float().abs()
        bad = d > tol
        n_bad = int(bad.sum().cpu())
        ratio = (d / tol)
        out = {"fail_count": n_bad, "max_abs": float(d.max().cpu()),
               "max_ratio": float(ratio.max().cpu()), "finite": bool(torch.isfinite(cand.float()).all().cpu())}
        if n_bad:
            flat = ratio.flatten()
            top = torch.topk(flat, min(3, flat.numel()))
            out["worst"] = [{"idx": int(i), "abs": float(d.flatten()[i].cpu()),
                             "ratio": float(v)} for i, v in zip(top.indices.cpu().tolist(),
                                                                top.values.cpu().tolist())]
        return out

    def y_max_rel(cand, ref):
        d = (cand.float() - ref.float()).abs()
        denom = ref.float().abs().clamp(min=1e-5)   # zero handling documented: |ref|<1e-5 excluded
        mask = ref.float().abs() >= 1e-5
        rel = torch.where(mask, d / denom, torch.zeros_like(d))
        return float(rel.max().cpu())

    gate = {"residual_sum": bf16_gate(res_out, ref_res),
            "y": bf16_gate(out, ref_y),
            "rstd": bf16_gate(rstd, ref_rstd.reshape(-1)),
            "y_max_rel": y_max_rel(out, ref_y),
            "criterion": "|c-r| <= 1e-5 + 0.016*|r| (torch.bfloat16 defaults), every element",
            "seed": a.seed}
    summary = {"rank": rank, "group": group, "m": m, "k": k, "n": n, "gate": gate,
               "bf16_ar_ref": {"max_abs_y": float(dyb.max().cpu()),
                               "max_abs_residual_sum": float(dresb.max().cpu()),
                               "max_abs_rstd": float(drstdb.max().cpu()),
                               "y_gt_0.02": int((dyb > 0.02).sum().cpu())},
               "diag": [float(v) for v in diag.cpu()],
               "max_abs_y": float(dy.max().cpu()), "max_abs_residual_sum": float(dres.max().cpu()),
               "max_abs_rstd": float(drstd.max().cpu()), "y_gt_0.01": int((dy > 0.01).sum().cpu()),
               "elements": int(dy.numel()), "w_format": "NZ" if a.w_nz else "ND"}
    if a.probe_partial:
        # res_out is z = partial + 0, i.e. the reduced matmul output itself.
        ref_p = torch.matmul(x, w)
        if a.use_hccl == 1:
            # rank-local reference must stay rank-local when AR is off (A arm)
            dist.all_reduce(ref_p, op=dist.ReduceOp.SUM)
        dp = (res_out.float() - ref_p.float()).abs()
        flat = dp.flatten()
        k = min(5, flat.numel())
        top = torch.topk(flat, k)
        idx = [[int(i // n), int(i % n)] for i in top.indices.cpu().tolist()]
        # Criterion 1: every mismatch must be an integer multiple of the bf16
        # lsb of its own magnitude (bf16: 8 mantissa bits, lsb = 2^(e-7)).
        mask = dp > 0
        av = ref_p.float().abs()
        expo = torch.floor(torch.log2(torch.clamp(av, min=2.0 ** -126)))
        lsb = torch.pow(2.0, expo - 7.0)
        ratio = torch.where(mask, dp / lsb, torch.zeros_like(dp))
        bad = ((ratio - ratio.round()).abs() > 1e-3) & mask
        rmax = ratio[mask]
        summary_p = {"rank": rank, "probe": "partial",
                     "criterion1": {"mismatch_count": int(mask.sum().cpu()),
                                    "ratio_max": float(rmax.max().cpu()) if rmax.numel() else 0.0,
                                    "non_integer_count": int(bad.sum().cpu())},
                     "max_abs_partial": float(flat.max().cpu()),
                     "gt_0.5": int((dp > 0.5).sum().cpu()), "gt_0.1": int((dp > 0.1).sum().cpu()),
                     "top": [{"row": r_, "col": c_, "diff": float(v)} for (r_, c_), v in
                             zip(idx, top.values.cpu().tolist())]}
        a.output.parent.mkdir(parents=True, exist_ok=True)
        a.output.with_name(a.output.stem + f"-rank{rank}-partial.json").write_text(
            json.dumps(summary_p, ensure_ascii=False, indent=1) + "\n")
        print(json.dumps(summary_p, ensure_ascii=False), flush=True)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.with_name(a.output.stem + f"-rank{rank}" + a.output.suffix).write_text(json.dumps(summary, ensure_ascii=False, indent=1) + "\n")
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    dist.barrier(); dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
