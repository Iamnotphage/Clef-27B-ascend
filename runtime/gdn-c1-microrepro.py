#!/usr/bin/env python3
"""C1 microrepro: FLA A3 chunk_gated_delta_rule_fwd layout candidates vs the
pinned A3 wrapper semantics (reference).

Paths:
  REF        current A3 wrapper: external l2norm, BNSD conversion, state
             transpose both directions, layout=BNSD, state_v_first=False.
  CAND_BSND  no q/k/v transposes, layout=BSND, l2norm still external,
             state transposes kept.
  CAND_FULL  BSND + use_qk_l2norm_in_kernel=True + state_v_first=True
             (state passed in cache layout [N,Hv,K,V]).

Cases: TP2/TP1 head counts, T in {63,64,65,127,128,129,307,1024,2048,3868},
B in {1,2,4} incl. heterogeneous lengths, zero/nonzero initial state,
request order permutation, strided (non-contiguous) inputs, magnitude
classes, NaN propagation. Timing: median host wall of kernel-path calls.
"""
import json
import statistics
import sys
import time

import torch

sys.path.insert(0, "/workspace/yangchen/work/src/vllm-ascend")
from vllm_ascend.ops.triton.fla.utils import prepare_chunk_indices  # noqa: E402
from vllm.third_party.flash_linear_attention.ops.l2norm import l2norm_fwd  # noqa: E402
from fla_npu.ops.ascendc import chunk_gated_delta_rule_fwd  # noqa: E402

DEV = "npu:0"
SCALE_EPS = 1e-6
CASES = []
FAILS = []


def meta(cu_cpu):
    cu_host = tuple(cu_cpu.to(torch.int64).reshape(-1).tolist())
    ci = tuple(prepare_chunk_indices(cu_cpu, 64).to(torch.int64).reshape(-1).tolist())
    return cu_host, ci


def ref_path(q, k, v, g, beta, initial_state, scale, cu_host, ci):
    """Exact pinned-wrapper semantics (device_op.py fla_gdn_prefill, keep_meta None)."""
    q = l2norm_fwd(q.contiguous()).contiguous()
    k = l2norm_fwd(k.contiguous()).contiguous()
    v = v.contiguous()
    g32 = g.to(torch.float32).contiguous()
    b16 = beta.to(v.dtype).contiguous()
    st = initial_state.contiguous()
    q = q.transpose(1, 2).contiguous()
    k = k.transpose(1, 2).contiguous()
    v = v.transpose(1, 2).contiguous()
    st_kern = st.transpose(-1, -2).contiguous()
    o, fs, *_ = chunk_gated_delta_rule_fwd(
        q, k, v, g32, b16, initial_state=st_kern, output_final_state=True,
        chunk_size=64, cu_seqlens=cu_host, chunk_indices=ci, scale=scale,
        layout="BNSD", use_exp2=False, use_qk_l2norm_in_kernel=False,
        allow_neg_eigval=False, disable_recompute=True, state_v_first=False)
    return o, fs.transpose(-1, -2).contiguous()


def cand_bsnd(q, k, v, g, beta, initial_state, scale, cu_host, ci):
    q = l2norm_fwd(q.contiguous()).contiguous()
    k = l2norm_fwd(k.contiguous()).contiguous()
    v = v.contiguous()
    g32 = g.to(torch.float32).contiguous()
    b16 = beta.to(v.dtype).contiguous()
    st_kern = initial_state.contiguous().transpose(-1, -2).contiguous()
    o, fs, *_ = chunk_gated_delta_rule_fwd(
        q, k, v, g32, b16, initial_state=st_kern, output_final_state=True,
        chunk_size=64, cu_seqlens=cu_host, chunk_indices=ci, scale=scale,
        layout="BSND", use_exp2=False, use_qk_l2norm_in_kernel=False,
        allow_neg_eigval=False, disable_recompute=True, state_v_first=False)
    return o, fs.transpose(-1, -2).contiguous()


def cand_full(q, k, v, g, beta, initial_state, scale, cu_host, ci):
    # candidate wrapper contract: inputs must be contiguous sequence-major;
    # state passed in cache layout [N, Hv, K, V].
    o, fs, *_ = chunk_gated_delta_rule_fwd(
        q, k, v, g, beta, initial_state=initial_state.contiguous(),
        output_final_state=True, chunk_size=64, cu_seqlens=cu_host,
        chunk_indices=ci, scale=scale, layout="BSND", use_exp2=False,
        use_qk_l2norm_in_kernel=True, allow_neg_eigval=False,
        disable_recompute=True, state_v_first=True)
    return o, fs


def make_inputs(T, hk, hv, dk, dv, state_kind="rand", gen=None):
    gen = gen or torch.Generator(device="cpu").manual_seed(hash((T, hk, hv)) % 2**31)
    q = torch.randn(1, T, hk, dk, generator=gen).to(DEV, torch.bfloat16)
    k = torch.randn(1, T, hk, dk, generator=gen).to(DEV, torch.bfloat16)
    v = torch.randn(1, T, hv, dv, generator=gen).to(DEV, torch.bfloat16)
    g = (-0.5 * torch.rand(1, T, hv, generator=gen)).to(DEV, torch.float32) - 1e-4
    beta = torch.rand(1, T, hv, generator=gen).to(DEV, torch.bfloat16)
    if state_kind == "zero":
        st = torch.zeros(1, hv, dk, dv)
    else:
        st = torch.randn(1, hv, dk, dv, generator=gen).to(DEV, torch.bfloat16) * 0.1
    return q, k, v, g, beta, st.to(DEV, torch.bfloat16)


def compare(name, ref, cand, out):
    ro, rs = ref
    co, cs = cand
    rec = {"case": name}
    for key, a, b in (("o", ro, co), ("state", rs, cs)):
        diff = (a.float() - b.float()).abs()
        rec[f"{key}_max_abs_diff"] = float(diff.max())
        rec[f"{key}_bitwise_equal"] = bool(torch.equal(a, b))
        rec[f"{key}_nan"] = int(torch.isnan(b.float()).sum())
        rec[f"{key}_inf"] = int(torch.isinf(b.float()).sum())
    out.append(rec)
    ok = (rec["o_max_abs_diff"] < 0.03 and rec["state_max_abs_diff"] < 0.03
          and rec["o_nan"] == int(torch.isnan(ro.float()).sum())
          and rec["o_inf"] == int(torch.isinf(ro.float()).sum()))
    print(f"  {name:44s} o_diff={rec['o_max_abs_diff']:.5f} st_diff={rec['state_max_abs_diff']:.5f} "
          f"bitwise={rec['o_bitwise_equal']}/{rec['state_bitwise_equal']} {'OK' if ok else 'FAIL'}")
    if not ok:
        FAILS.append(name)
    return rec


def run_case(name, q, k, v, g, beta, st, scale, cu_cpu, out):
    cu_host, ci = meta(cu_cpu)
    ref = ref_path(q, k, v, g, beta, st, scale, cu_host, ci)
    torch.npu.synchronize()
    try:
        c1 = cand_bsnd(q, k, v, g, beta, st, scale, cu_host, ci)
        compare(name + "/bsnd", ref, c1, out)
    except Exception as exc:
        print(f"  {name}/bsnd EXC: {repr(exc)[:120]}")
        out.append({"case": name + "/bsnd", "error": repr(exc)})
    try:
        c2 = cand_full(q, k, v, g, beta, st, scale, cu_host, ci)
        compare(name + "/full", ref, c2, out)
    except Exception as exc:
        print(f"  {name}/full EXC: {repr(exc)[:120]}")
        out.append({"case": name + "/full", "error": repr(exc)})


def time_path(fn, q, k, v, g, beta, st, scale, cu_cpu, iters=20):
    cu_host, ci = meta(cu_cpu)
    for _ in range(5):
        fn(q, k, v, g, beta, st, scale, cu_host, ci)
    torch.npu.synchronize()
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn(q, k, v, g, beta, st, scale, cu_host, ci)
        torch.npu.synchronize()
        ts.append((time.perf_counter() - t0) * 1000)
    return statistics.median(ts)


def main():
    torch.npu.set_device(0)
    out = []
    HK2, HV2, D = 8, 24, 128     # TP2 per-rank heads
    HK1, HV1 = 16, 48            # TP1 per-rank heads
    scale = D ** -0.5

    print("== single-sequence lengths, TP2 heads, nonzero state ==")
    for T in (63, 64, 65, 127, 128, 129, 307, 1024, 2048, 3868):
        q, k, v, g, beta, st = make_inputs(T, HK2, HV2, D, D)
        cu = torch.tensor([0, T], dtype=torch.int32)
        run_case(f"tp2-T{T}", q, k, v, g, beta, st, scale, cu, out)

    print("== TP1 heads (T=307, 1024) ==")
    for T in (307, 1024):
        q, k, v, g, beta, st = make_inputs(T, HK1, HV1, D, D)
        cu = torch.tensor([0, T], dtype=torch.int32)
        run_case(f"tp1-T{T}", q, k, v, g, beta, st, scale, cu, out)

    print("== zero initial state ==")
    q, k, v, g, beta, _ = make_inputs(307, HK2, HV2, D, D)
    st0 = torch.zeros(1, HV2, D, D).to(DEV, torch.bfloat16)
    run_case("tp2-T307-state0", q, k, v, g, beta, st0, scale,
             torch.tensor([0, 307], dtype=torch.int32), out)

    print("== heterogeneous batch B2/B4 + order permutation + tail chunks ==")
    for lens, tag in (((100, 207), "B2-hetero"), ((307, 64, 129, 205), "B4-hetero"),
                      ((64, 63, 65, 2), "B4-chunkedges")):
        tot = sum(lens)
        q = torch.randn(1, tot, HK2, D).to(DEV, torch.bfloat16)
        k = torch.randn(1, tot, HK2, D).to(DEV, torch.bfloat16)
        v = torch.randn(1, tot, HV2, D).to(DEV, torch.bfloat16)
        g = (-0.5 * torch.rand(1, tot, HV2)).to(DEV, torch.float32) - 1e-4
        beta = torch.rand(1, tot, HV2).to(DEV, torch.bfloat16)
        st = (torch.randn(len(lens), HV2, D, D) * 0.1).to(DEV, torch.bfloat16)
        cu = torch.tensor([0] + [sum(lens[:i + 1]) for i in range(len(lens))], dtype=torch.int32)
        run_case(tag, q, k, v, g, beta, st, scale, cu, out)

    print("== strided (non-contiguous) inputs: candidate contract must reject or handle ==")
    base = torch.randn(1, 307, HK2, D * 2, generator=torch.Generator().manual_seed(7))
    q_strided = base.to(DEV, torch.bfloat16)[..., :D]  # non-contiguous last dim
    q_c, k_c, v_c, g_c, beta_c, st_c = make_inputs(307, HK2, HV2, D, D)
    try:
        cu_host, ci = meta(torch.tensor([0, 307], dtype=torch.int32))
        ref = ref_path(q_strided, k_c, v_c, g_c, beta_c, st_c, scale, cu_host, ci)
        c2 = cand_full(q_strided, k_c, v_c, g_c, beta_c, st_c, scale, cu_host, ci)
        compare("strided-q/full", ref, c2, out)
    except Exception as exc:
        print(f"  strided-q/full EXC (expected if guard needed): {repr(exc)[:140]}")
        out.append({"case": "strided-q/full", "error": repr(exc)})

    print("== magnitude classes ==")
    for tag, mul in (("zeros", 0.0), ("tiny", 1e-3), ("large", 30.0)):
        q, k, v, g, beta, st = make_inputs(307, HK2, HV2, D, D)
        if mul == 0.0:
            q = torch.zeros_like(q); k = torch.zeros_like(k); v = torch.zeros_like(v)
        else:
            q = (q.float() * mul).to(torch.bfloat16)
            k = (k.float() * mul).to(torch.bfloat16)
            v = (v.float() * mul).to(torch.bfloat16)
        run_case(f"mag-{tag}", q, k, v, g, beta, st, scale,
                 torch.tensor([0, 307], dtype=torch.int32), out)

    print("== NaN propagation parity ==")
    q, k, v, g, beta, st = make_inputs(307, HK2, HV2, D, D)
    q[0, 150, 3, 5] = float("nan")
    run_case("nan-in-q", q, k, v, g, beta, st, scale,
             torch.tensor([0, 307], dtype=torch.int32), out)

    print("== timing (median ms, host wall incl. path conversions) ==")
    timing = {}
    for T in (307, 3868):
        q, k, v, g, beta, st = make_inputs(T, HK2, HV2, D, D)
        cu = torch.tensor([0, T], dtype=torch.int32)
        t_ref = time_path(ref_path, q, k, v, g, beta, st, scale, cu)
        t_c2 = time_path(cand_full, q, k, v, g, beta, st, scale, cu)
        timing[f"T{T}"] = {"ref_ms": round(t_ref, 3), "cand_full_ms": round(t_c2, 3),
                           "saving_ms": round(t_ref - t_c2, 3)}
        print(f"  T={T}: ref {t_ref:.3f} ms  cand_full {t_c2:.3f} ms  saving {t_ref - t_c2:.3f} ms")
        # per-layer per-request: 48 layers
    json.dump({"cases": out, "timing": timing, "failures": FAILS},
              open("/tmp/gdn_c1_microrepro.json", "w"), indent=1)
    print("FAILURES:", FAILS if FAILS else "none")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
