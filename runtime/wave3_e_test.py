#!/usr/bin/env python3
"""Two-rank correctness for the E first version (MatMul+AR+AddRMS, ND weights)."""
from __future__ import annotations
import argparse
import hashlib, json, os
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
                     "y_max_rel": float(((dy) / ry.float().abs().clamp(min=1e-5)).max().cpu()),
                     "y_min_abs_ref": float(ry.float().abs().min().cpu()),
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
    if a.w_nz:
        print("fmt_before_cast", torch_npu.get_npu_format(w), flush=True)
        torch.npu.config.allow_internal_format = True
        print("allow_internal_format=True", flush=True)
    w_op = torch_npu.npu_format_cast(w, 29) if a.w_nz else w
    if a.w_nz:
        print("fmt_after_cast", torch_npu.get_npu_format(w_op), flush=True)
    out, res_out, rstd, diag = torch.ops.clef_e.mm_ar_norm(x, w_op, r, gamma, a.eps, 1, group, a.use_hccl, a.block_dim, a.mm_enable, a.mm_dim, a.mm_mode, a.mm_bufspace, a.norm_ub_late, a.mm_btrans, a.mm_step, a.norm_from_residual, a.norm_from_x, 1 if a.w_nz else 0)
    torch.npu.synchronize()

    # Reference A (fp32 reduce, as before) and Reference B (bf16 reduce, the
    # same rounding path as the op: bf16 partial, bf16 all-reduce).
    partial = torch.matmul(x, w).float()
    partial_bf = torch.matmul(x, w)
    if a.use_hccl == 1:
        dist.all_reduce(partial, op=dist.ReduceOp.SUM)
        dist.all_reduce(partial_bf, op=dist.ReduceOp.SUM)
    ref_y, ref_rstd, ref_res = torch_npu.npu_add_rms_norm(partial.to(torch.bfloat16), r, gamma, a.eps)
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
        nz_mask = ref.float().abs() < 1e-5
        out = {"fail_count": n_bad, "max_abs": float(d.max().cpu()),
               "max_ratio": float(ratio.max().cpu()),
               "cand_finite": bool(torch.isfinite(cand.float()).all().cpu()),
               "ref_finite": bool(torch.isfinite(ref.float()).all().cpu()),
               "shape_ok": tuple(cand.shape) == tuple(ref.shape),
               "dtype_ok": cand.dtype == ref.dtype,
               "near_zero_count": int(nz_mask.sum().cpu()),
               "near_zero_max_abs": float(d[nz_mask].max().cpu()) if bool(nz_mask.any()) else 0.0,
               "near_zero_fail_count": int((bad & nz_mask).sum().cpu())}
        if n_bad:
            bad_idx = bad.flatten().nonzero().flatten()
            out["fail_total"] = int(bad_idx.numel())
            cf = cand.float().flatten(); rf = ref.float().flatten()
            tl = tol.flatten(); df = d.flatten(); rt = ratio.flatten()
            out["fail_points"] = [{"idx": int(i), "actual": float(cf[i].cpu()),
                                   "ref": float(rf[i].cpu()), "tol": float(tl[i].cpu()),
                                   "abs": float(df[i].cpu()), "ratio": float(rt[i].cpu())}
                                  for i in bad_idx.cpu().tolist()]
            out["fail_idx"] = [int(i) for i in bad_idx.cpu().tolist()]
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

    y_abs = (out.float() - refb_y.float()).abs()
    y_den = refb_y.float().abs().clamp(min=1e-5)
    y_mask = refb_y.float().abs() >= 1e-5
    y_rel = torch.where(y_mask, y_abs / y_den, torch.zeros_like(y_abs))
    y_strict = {"max_abs": float(y_abs.max().cpu()),
                "max_rel": float(y_rel.max().cpu()),
                "pass": bool(float(y_abs.max().cpu()) <= 0.02 and float(y_rel.max().cpu()) <= 0.01),
                "near_zero_count": int((~y_mask).sum().cpu())}
    gate = {"y_strict": y_strict,
            "reference": "bf16 matmul + bf16 SUM (refb_*), AR follows use_hccl",
            "residual_sum": bf16_gate(res_out, refb_res),
            "y": bf16_gate(out, refb_y),
            "rstd": bf16_gate(rstd, refb_rstd.reshape(-1)),
            "y_max_rel": y_max_rel(out, refb_y),
            "criterion": "|c-r| <= 1e-5 + 0.016*|r| (torch.bfloat16 defaults), every element",
            "hashes": {"x_sha256": hashlib.sha256(x.view(torch.uint16).cpu().numpy().tobytes()).hexdigest(),
                       "w_sha256": hashlib.sha256(w.view(torch.uint16).cpu().numpy().tobytes()).hexdigest(),
                       "r_sha256": hashlib.sha256(r.view(torch.uint16).cpu().numpy().tobytes()).hexdigest(),
                       "gamma_sha256": hashlib.sha256(gamma.view(torch.uint16).cpu().numpy().tobytes()).hexdigest(),
                       "eps": a.eps, "seed": a.seed, "w_format": "NZ" if a.w_nz else "ND",
                       "reference": "same r/gamma/eps tensors as the candidate"},
            "seed": a.seed}
    # One-shot export over the union of residual and Y failures: both outputs'
    # values at every index, plus row/col and the row/col factors.
    def _fail_idx(gg, key):
        return set(int(i) for i in gg[key].get("fail_idx", []))
    union = sorted(_fail_idx(gate, "residual_sum") | _fail_idx(gate, "y"))
    if union:
        rc = res_out.float().flatten(); rr = refb_res.float().flatten()
        yc = out.float().flatten(); yr = refb_y.float().flatten()
        rsc = rstd.float().flatten(); rsr = refb_rstd.float().flatten()
        gfl = gamma.float().flatten(); rfl = r.float().flatten()
        pfl = (res_out.float() - r).flatten()      # partial_cand = z - residual
        prfl = (refb_res.float() - r).flatten()
        def _pt(i):
            tol_r = 1e-5 + 0.016 * abs(float(rr[i].cpu()))
            tol_y = 1e-5 + 0.016 * abs(float(yr[i].cpu()))
            row, col = i // n, i % n
            return {"idx": int(i), "row": int(row), "col": int(col),
                    "residual": {"actual": float(rc[i].cpu()), "ref": float(rr[i].cpu()),
                                 "tol": tol_r, "ratio": abs(float(rc[i].cpu()) - float(rr[i].cpu())) / tol_r},
                    "y": {"actual": float(yc[i].cpu()), "ref": float(yr[i].cpu()),
                          "tol": tol_y, "ratio": abs(float(yc[i].cpu()) - float(yr[i].cpu())) / tol_y},
                    "rstd": {"cand": float(rsc[row].cpu()), "ref": float(rsr[row].cpu())},
                    "gamma": float(gfl[col].cpu()),
                    "partial": {"cand": float(pfl[i].cpu()), "ref": float(prfl[i].cpu())},
                    "residual_input": float(rfl[i].cpu())}
        export = {"rank": rank, "seed": a.seed, "union_count": len(union),
                  "residual_fail": gate["residual_sum"]["fail_count"],
                  "y_fail": gate["y"]["fail_count"],
                  "intersection": gate["residual_sum"]["fail_count"] + gate["y"]["fail_count"] - len(union),
                  "points": [_pt(i) for i in union]}
        a.output.parent.mkdir(parents=True, exist_ok=True)
        a.output.with_name(a.output.stem + f"-rank{rank}-union76.json").write_text(
            json.dumps(export, ensure_ascii=False, indent=1) + "\n")
        print(json.dumps({"rank": rank, "union": len(union)}, ensure_ascii=False), flush=True)

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
        # Real bytes for the CPU fp64 diagnostic: the A arm's own failures.
        tol_p = 1e-5 + 0.016 * ref_p.float().abs()
        badp = (dp > tol_p).flatten()
        a_bidx = [int(i) for i in badp.nonzero().flatten().cpu().tolist()]
        # Fixed source: the ORIGINAL full-chain union export, never this arm's
        # own failure set (no silent fallback).
        union_path = a.output.parent / f"e-full-rt-rank{rank}-union76.json"
        if not union_path.is_file():
            raise SystemExit(f"missing original union export: {union_path}")
        u = json.loads(union_path.read_text())
        union_sha = hashlib.sha256(union_path.read_bytes()).hexdigest()
        bidx = sorted({int(p_["idx"]) for p_ in u.get("points", [])})
        if len(bidx) != 76:
            raise SystemExit(f"union count {len(bidx)} != 76")
        # fixed control points (indices known not to fail in the full chain)
        controls = [0, n, 2 * n + 1, 500 * n + 7]
        sample = sorted(set(bidx) | set(controls))
        bytes_export = None
        if bidx:
            xh = x.view(torch.uint16).cpu()
            wh = w.view(torch.uint16).cpu()
            pc = res_out.float().flatten().cpu()
            pr = ref_p.float().flatten().cpu()
            rows = sorted({i // n for i in sample}); cols = sorted({i % n for i in sample})
            bytes_export = {
                "arm": "probe_partial r=0 use_hccl=2 (AR off, both sides)",
                "hashes": {
                    "x_sha256": hashlib.sha256(x.view(torch.uint16).cpu().numpy().tobytes()).hexdigest(),
                    "w_sha256": hashlib.sha256(w.view(torch.uint16).cpu().numpy().tobytes()).hexdigest(),
                    "gamma_sha256": hashlib.sha256(gamma.view(torch.uint16).cpu().numpy().tobytes()).hexdigest(),
                    "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    "dtype": "bfloat16", "byte_order": "little",
                    "layout": "x/w/gamma logical ND (w row-major [K,N])"},
                "shapes": {"x": list(x.shape), "w": list(w.shape), "n": n},
                "union76_idx": bidx, "union_source": str(union_path),
                "union_source_sha256": union_sha, "a_arm_fail_idx": a_bidx,
                "a_arm_fail_count": len(a_bidx), "controls": controls, "sample_idx": sample,
                "x_rows": {int(r_): [format(int(v), "04x") for v in xh[r_].tolist()] for r_ in rows},
                "w_cols": {int(c_): [format(int(v), "04x") for v in wh[:, c_].tolist()] for c_ in cols},
                "points": [{"idx": i, "row": i // n, "col": i % n,
                            "cand": float(pc[i]), "native": float(pr[i]),
                            "tol": float(tol_p.flatten()[i].cpu()),
                            "ratio": float(dp.flatten()[i].cpu() / tol_p.flatten()[i].cpu())}
                           for i in sample]}
            missing = [i for i in sample if (i // n) not in rows or (i % n) not in cols]
            assert not missing, f"bytes missing for {missing[:5]}"
            bytes_export["assert_all_points_covered"] = True
            a.output.with_name(a.output.stem + f"-rank{rank}-bytes.json").write_text(
                json.dumps(bytes_export, ensure_ascii=False) + "\n")
            print(json.dumps({"rank": rank, "union76": len(bidx), "a_fail": len(a_bidx),
                              "rows": len(rows), "cols": len(cols)}, ensure_ascii=False), flush=True)
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
