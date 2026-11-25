#!/usr/bin/env python3
"""Isolate which kernel flags the FLA stable ABI rejects on A3."""
import sys
import torch
sys.path.insert(0, "/workspace/yangchen/work/src/vllm-ascend")
from vllm.third_party.flash_linear_attention.ops.l2norm import l2norm_fwd
from fla_npu.ops.ascendc import chunk_gated_delta_rule_fwd
from vllm_ascend.ops.triton.fla.utils import prepare_chunk_indices

torch.npu.set_device(0)
T, HK, HV, D = 307, 8, 24, 128
gen = torch.Generator().manual_seed(3)
q = torch.randn(1, T, HK, D, generator=gen).to("npu", torch.bfloat16)
k = torch.randn(1, T, HK, D, generator=gen).to("npu", torch.bfloat16)
v = torch.randn(1, T, HV, D, generator=gen).to("npu", torch.bfloat16)
g = (-0.5 * torch.rand(1, T, HV, generator=gen)).to("npu", torch.float32)
beta = torch.rand(1, T, HV, generator=gen).to("npu", torch.bfloat16)
st = (torch.randn(1, HV, D, D, generator=gen) * 0.1).to("npu", torch.bfloat16)
scale = D ** -0.5
cu_cpu = torch.tensor([0, T], dtype=torch.int32)
cu = tuple(cu_cpu.to(torch.int64).reshape(-1).tolist())
ci = tuple(prepare_chunk_indices(cu_cpu, 64).to(torch.int64).reshape(-1).tolist())

def try_call(tag, q_, k_, v_, g_, b_, st_, **kw):
    try:
        o, fs, *_ = chunk_gated_delta_rule_fwd(
            q_, k_, v_, g_, b_, initial_state=st_, output_final_state=True,
            chunk_size=64, cu_seqlens=cu, chunk_indices=ci, scale=scale,
            allow_neg_eigval=False, disable_recompute=True, **kw)
        torch.npu.synchronize()
        print(f"{tag:52s} OK  o{tuple(o.shape)} fs{tuple(fs.shape)}")
        return o, fs
    except Exception as exc:
        print(f"{tag:52s} FAIL {repr(exc)[:90]}")
        return None

# orientation helpers
def bnsd(x):  # [1,T,H,D] -> [1,H,T,D] contiguous
    return x.transpose(1, 2).contiguous()

qn, kn = l2norm_fwd(q.contiguous()).contiguous(), l2norm_fwd(k.contiguous()).contiguous()
st_t = st.transpose(-1, -2).contiguous()

# reference baseline
try_call("REF bnsd l2norm-ext stateVfirst=False", bnsd(qn), bnsd(kn), bnsd(v.contiguous()), g.contiguous(), beta.contiguous(), st_t,
         layout="BNSD", use_qk_l2norm_in_kernel=False, state_v_first=False)
# flag isolation
try_call("bsnd l2norm-ext stateVfirst=False", qn, kn, v.contiguous(), g.contiguous(), beta.contiguous(), st_t,
         layout="BSND", use_qk_l2norm_in_kernel=False, state_v_first=False)
try_call("bsnd l2norm-KERNEL stateVfirst=False", q, k, v.contiguous(), g.contiguous(), beta.contiguous(), st_t,
         layout="BSND", use_qk_l2norm_in_kernel=True, state_v_first=False)
try_call("bsnd l2norm-ext stateVfirst=TRUE(cache layout)", qn, kn, v.contiguous(), g.contiguous(), beta.contiguous(), st.contiguous(),
         layout="BSND", use_qk_l2norm_in_kernel=False, state_v_first=True)
try_call("bsnd l2norm-ext stateVfirst=TRUE(transposed)", qn, kn, v.contiguous(), g.contiguous(), beta.contiguous(), st_t,
         layout="BSND", use_qk_l2norm_in_kernel=False, state_v_first=True)
try_call("bnsd l2norm-KERNEL stateVfirst=False", bnsd(q), bnsd(k), bnsd(v.contiguous()), g.contiguous(), beta.contiguous(), st_t,
         layout="BNSD", use_qk_l2norm_in_kernel=True, state_v_first=False)
try_call("bsnd l2norm-KERNEL + stateVfirst(cache)", q, k, v.contiguous(), g.contiguous(), beta.contiguous(), st.contiguous(),
         layout="BSND", use_qk_l2norm_in_kernel=True, state_v_first=True)
try_call("bnsd l2norm-ext stateVfirst=TRUE(cache)", bnsd(qn), bnsd(kn), bnsd(v.contiguous()), g.contiguous(), beta.contiguous(), st.contiguous(),
         layout="BNSD", use_qk_l2norm_in_kernel=False, state_v_first=True)
