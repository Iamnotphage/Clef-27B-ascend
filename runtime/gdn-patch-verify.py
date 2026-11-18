#!/usr/bin/env python3
"""Verify the patched DeviceOperator.fla_gdn_prefill with env off/on."""
import os
import sys
import torch

sys.path.insert(0, "/workspace/yangchen/work/src/vllm-ascend")
from vllm_ascend.ops.triton.fla.utils import prepare_chunk_indices

torch.npu.set_device(0)

def build_meta(T, lens=None):
    if lens:
        cu_cpu = torch.tensor([0] + [sum(lens[:i + 1]) for i in range(len(lens))], dtype=torch.int32)
    else:
        cu_cpu = torch.tensor([0, T], dtype=torch.int32)
    class M: pass
    m = M()
    m.cu_seqlens_host = tuple(cu_cpu.to(torch.int64).reshape(-1).tolist())
    m.chunk_indices_chunk64_host = tuple(prepare_chunk_indices(cu_cpu, 64).to(torch.int64).reshape(-1).tolist())
    m.cu_seqlens_kern = None
    m.keep_meta = None
    return m

from vllm_ascend.device.device_op import DeviceOperator
from fla_npu.ops.ascendc import chunk_gated_delta_rule_fwd as _fla_fwd
from vllm.third_party.flash_linear_attention.ops.l2norm import l2norm_fwd

fwd = _fla_fwd
HK, HV, D = 8, 24, 128
scale = D ** -0.5

def run(T, lens=None, env=None, strided=False):
    gen = torch.Generator().manual_seed(T * 7 + (sum(lens) if lens else 0))
    tot = sum(lens) if lens else T
    if strided:
        base = torch.randn(1, tot, HK, D * 2, generator=gen)
        q = base.to("npu", torch.bfloat16)[..., :D]
    else:
        q = torch.randn(1, tot, HK, D, generator=gen).to("npu", torch.bfloat16)
    k = torch.randn(1, tot, HK, D, generator=gen).to("npu", torch.bfloat16)
    v = torch.randn(1, tot, HV, D, generator=gen).to("npu", torch.bfloat16)
    g = (-0.5 * torch.rand(1, tot, HV, generator=gen)).to("npu", torch.float32) - 1e-4
    beta = torch.rand(1, tot, HV, generator=gen).to("npu", torch.bfloat16)
    n = len(lens) if lens else 1
    st = (torch.randn(n, HV, D, D, generator=gen) * 0.1).to("npu", torch.bfloat16)
    meta = build_meta(T, lens)
    if env is not None:
        os.environ["CLEF_GDN_BSND"] = env
        DeviceOperator  # import already done; flag reads lazily per process
    import vllm_ascend.device.device_op as do_mod
    do_mod._GDN_BSND_ENABLED = None if env is not None else do_mod._GDN_BSND_ENABLED
    if env is not None:
        do_mod._GDN_BSND_ENABLED = env == "1"
    return DeviceOperator.fla_gdn_prefill(q=q, k=k, v=v, g=g, beta=beta,
                                          initial_state=st, scale=scale,
                                          prebuilt_meta=meta, fused_fwd=fwd)

fails = []
cases = [("T307", dict(T=307)), ("T63", dict(T=63)), ("T1024", dict(T=1024)),
         ("B4-hetero", dict(T=0, lens=[307, 64, 129, 205])),
         ("strided-q", dict(T=307, strided=True))]
for name, kw in cases:
    off = run(env="0", **kw)
    on = run(env="1", **kw)
    do, dso = off
    dn, dsn = on
    md_o = float((do.float() - dn.float()).abs().max())
    md_s = float((dso.float() - dsn.float()).abs().max())
    bit = bool(torch.equal(do, dn)) and bool(torch.equal(dso, dsn))
    ok = md_o < 0.03 and md_s < 0.03 and (not kw.get("strided") or bit)
    print(f"{name:12s} off-vs-on max_diff o={md_o:.6f} state={md_s:.6f} bitwise={bit} {'OK' if ok else 'FAIL'}")
    if not ok:
        fails.append(name)
print("FAILURES:", fails if fails else "none")
sys.exit(1 if fails else 0)
