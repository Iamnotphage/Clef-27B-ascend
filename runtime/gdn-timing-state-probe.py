#!/usr/bin/env python3
"""C1' timing (BSND-only) + decode-path state orientation probe."""
import statistics
import sys
import time
import torch
sys.path.insert(0, "/workspace/yangchen/work/src/vllm-ascend")
from vllm_ascend.ops.triton.fla.utils import prepare_chunk_indices
from vllm.third_party.flash_linear_attention.ops.l2norm import l2norm_fwd
from fla_npu.ops.ascendc import chunk_gated_delta_rule_fwd, recurrent_gated_delta_rule

torch.npu.set_device(0)
HK, HV, D = 8, 24, 128
scale = D ** -0.5

def meta(T, lens=None):
    if lens:
        cu_cpu = torch.tensor([0] + [sum(lens[:i + 1]) for i in range(len(lens))], dtype=torch.int32)
    else:
        cu_cpu = torch.tensor([0, T], dtype=torch.int32)
    return (tuple(cu_cpu.to(torch.int64).reshape(-1).tolist()),
            tuple(prepare_chunk_indices(cu_cpu, 64).to(torch.int64).reshape(-1).tolist()))

def make(T):
    gen = torch.Generator().manual_seed(T)
    q = torch.randn(1, T, HK, D, generator=gen).to("npu", torch.bfloat16)
    k = torch.randn(1, T, HK, D, generator=gen).to("npu", torch.bfloat16)
    v = torch.randn(1, T, HV, D, generator=gen).to("npu", torch.bfloat16)
    g = (-0.5 * torch.rand(1, T, HV, generator=gen)).to("npu", torch.float32) - 1e-4
    beta = torch.rand(1, T, HV, generator=gen).to("npu", torch.bfloat16)
    st = (torch.randn(1, HV, D, D, generator=gen) * 0.1).to("npu", torch.bfloat16)
    return q, k, v, g, beta, st

def bnsd(x):
    return x.transpose(1, 2).contiguous()

def ref_path(q, k, v, g, beta, st, cu, ci):
    qn = l2norm_fwd(q.contiguous()).contiguous()
    kn = l2norm_fwd(k.contiguous()).contiguous()
    o, fs, *_ = chunk_gated_delta_rule_fwd(
        bnsd(qn), bnsd(kn), bnsd(v.contiguous()), g.contiguous(), beta.contiguous(),
        initial_state=st.contiguous().transpose(-1, -2).contiguous(),
        output_final_state=True, chunk_size=64, cu_seqlens=cu, chunk_indices=ci,
        scale=scale, layout="BNSD", use_exp2=False, use_qk_l2norm_in_kernel=False,
        allow_neg_eigval=False, disable_recompute=True, state_v_first=False)
    return o, fs.transpose(-1, -2).contiguous()

def cand_path(q, k, v, g, beta, st, cu, ci):
    qn = l2norm_fwd(q.contiguous()).contiguous()
    kn = l2norm_fwd(k.contiguous()).contiguous()
    o, fs, *_ = chunk_gated_delta_rule_fwd(
        qn, kn, v.contiguous(), g.contiguous(), beta.contiguous(),
        initial_state=st.contiguous().transpose(-1, -2).contiguous(),
        output_final_state=True, chunk_size=64, cu_seqlens=cu, chunk_indices=ci,
        scale=scale, layout="BSND", use_exp2=False, use_qk_l2norm_in_kernel=False,
        allow_neg_eigval=False, disable_recompute=True, state_v_first=False)
    return o, fs.transpose(-1, -2).contiguous()

def bench(fn, q, k, v, g, beta, st, cu, ci, iters=30):
    # Scope: host dispatch through device completion. Not device-only time.
    for _ in range(5):
        fn(q, k, v, g, beta, st, cu, ci)
    torch.npu.synchronize()
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn(q, k, v, g, beta, st, cu, ci)
        torch.npu.synchronize()
        ts.append((time.perf_counter() - t0) * 1000)
    return statistics.median(ts)

print("== C1' timing: wrapper synchronous wall (perf_counter + npu.synchronize; includes host dispatch and device wait) ==")
for T in (307, 1024, 2048, 3868):
    q, k, v, g, beta, st = make(T)
    cu, ci = meta(T)
    tr = bench(ref_path, q, k, v, g, beta, st, cu, ci)
    tc = bench(cand_path, q, k, v, g, beta, st, cu, ci)
    print(f"  T={T:5d}: ref {tr:7.3f} ms  bsnd {tc:7.3f} ms  saving {tr - tc:7.3f} ms ({(tr - tc) / tr * 100:.1f}%)")

# ---- decode state orientation probe ----
print("== decode recurrent_gated_delta_rule state orientation probe ==")
import inspect
print("sig:", inspect.signature(recurrent_gated_delta_rule))
ht = (1, HV, D, D)  # cache layout [1, Hv, K, V]
ht_t = (1, HV, D, D). __class__  # placeholder
q1 = torch.randn(1, HK, D).to("npu", torch.bfloat16)
k1 = torch.randn(1, HK, D).to("npu", torch.bfloat16)
v1 = torch.randn(1, HV, D).to("npu", torch.bfloat16)
g1 = (-0.5 * torch.rand(1, HV)).to("npu", torch.float32)
b1 = torch.rand(1, HV).to("npu", torch.bfloat16)
st_cache = (torch.randn(1, HV, D, D) * 0.1).to("npu", torch.bfloat16)
st_t = st_cache.transpose(-1, -2).contiguous()
for tag, s in (("cache-layout [N,Hv,K,V]", st_cache), ("transposed [N,Hv,V,K]", st_t)):
    try:
        o, fs, *_ = recurrent_gated_delta_rule(q1, k1, v1, g1, b1, initial_state=s.contiguous(), output_final_state=True, scale=scale)
        torch.npu.synchronize()
        print(f"  decode {tag:28s} OK o{tuple(o.shape)} fs{tuple(fs.shape)}")
        if tag.startswith("cache"):
            ref_fs = fs
        else:
            same = bool(torch.equal(fs.transpose(-1, -2), ref_fs)) if "ref_fs" in dir() else None
            print(f"    equals cache-layout result: {same}")
    except Exception as exc:
        print(f"  decode {tag:28s} FAIL {repr(exc)[:80]}")
