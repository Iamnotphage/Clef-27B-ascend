#!/usr/bin/env python3
"""Probe request isolation of the fused AscendC GDN prefill kernel.

Compares one fused call carrying several sequences with one call per sequence.
Both sides use the pinned A3 wrapper semantics (external l2norm, BNSD layout).
Differences indicate that the multi-sequence fused path does not produce the
same per-sequence outputs as isolated calls.
"""

import json
import sys

import torch

sys.path.insert(0, "/workspace/yangchen/work/src/vllm-ascend")
from vllm_ascend.ops.triton.fla.utils import prepare_chunk_indices  # noqa: E402
from vllm.third_party.flash_linear_attention.ops.l2norm import l2norm_fwd  # noqa: E402
from fla_npu.ops.ascendc import chunk_gated_delta_rule_fwd  # noqa: E402

DEV = "npu:0"
HK, HV, D = 8, 24, 128
SCALE = D ** -0.5


def fused(q, k, v, g, beta, state, cu_host):
    """Pinned wrapper semantics; q/k/v are [1, T, H, D] concatenated."""
    cu_cpu = torch.tensor(cu_host, dtype=torch.int32)
    chunk_indices = tuple(
        prepare_chunk_indices(cu_cpu, 64).to(torch.int64).reshape(-1).tolist()
    )
    q = l2norm_fwd(q.contiguous()).contiguous()
    k = l2norm_fwd(k.contiguous()).contiguous()
    v = v.contiguous()
    g32 = g.to(torch.float32).contiguous()
    beta16 = beta.to(torch.bfloat16).contiguous()
    state_kern = state.contiguous().transpose(-1, -2).contiguous()
    out, final_state, *_ = chunk_gated_delta_rule_fwd(
        q.transpose(1, 2).contiguous(),
        k.transpose(1, 2).contiguous(),
        v.transpose(1, 2).contiguous(),
        g32,
        beta16,
        initial_state=state_kern,
        output_final_state=True,
        chunk_size=64,
        cu_seqlens=cu_host,
        chunk_indices=chunk_indices,
        scale=SCALE,
        layout="BNSD",
        use_exp2=False,
        use_qk_l2norm_in_kernel=False,
        allow_neg_eigval=False,
        disable_recompute=True,
        state_v_first=False,
    )
    return out, final_state


def make(lengths, state_kind, seed):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    total = sum(lengths)
    q = torch.randn(1, total, HK, D, generator=generator).to(DEV, torch.bfloat16)
    k = torch.randn(1, total, HK, D, generator=generator).to(DEV, torch.bfloat16)
    v = torch.randn(1, total, HV, D, generator=generator).to(DEV, torch.bfloat16)
    g = (-0.5 * torch.rand(1, total, HV, generator=generator)).to(DEV, torch.float32) - 1e-4
    beta = torch.rand(1, total, HV, generator=generator).to(DEV, torch.bfloat16)
    if state_kind == "zero":
        state = torch.zeros(len(lengths), HV, D, D)
    else:
        state = torch.randn(len(lengths), HV, D, D, generator=generator) * 0.1
    return q, k, v, g, beta, state.to(DEV, torch.bfloat16)


def split_rows(tensor, lengths):
    return list(torch.split(tensor, lengths, dim=0))


def main():
    torch.npu.set_device(0)
    cases = [
        ("three-606", [606, 606, 606]),
        ("three-977", [977, 977, 977]),
        ("three-430", [430, 430, 430]),
        ("hetero-606-977-430", [606, 977, 430]),
        ("hetero-307-64-129-205", [307, 64, 129, 205]),
        ("pair-100-207", [100, 207]),
    ]
    results = []
    for state_kind in ("zero", "random"):
        for name, lengths in cases:
            q, k, v, g, beta, state = make(lengths, state_kind, abs(hash(name)) % 2**31)
            cu = [0] + list(torch.tensor(lengths).cumsum(0).tolist())
            batch_out, batch_state = fused(q, k, v, g, beta, state, cu)
            rows = list(torch.split(batch_out[0], lengths, dim=0))  # each [L, H, D]
            state_rows = list(batch_state)
            isolated = []
            for index, (start, length) in enumerate(
                zip(cu[:-1], lengths)
            ):
                end = start + length
                out_i, state_i = fused(
                    q[:, start:end], k[:, start:end], v[:, start:end],
                    g[:, start:end], beta[:, start:end],
                    state[index : index + 1], [0, length],
                )
                isolated.append({
                    "out": out_i[0],
                    "state": state_i[0],
                })
            out_diffs = [
                float((rows[i].float() - isolated[i]["out"].float()).abs().max())
                for i in range(len(lengths))
            ]
            state_diffs = [
                float((state_rows[i].float() - isolated[i]["state"].float()).abs().max())
                for i in range(len(lengths))
            ]
            record = {
                "case": name,
                "state": state_kind,
                "lengths": lengths,
                "out_max_abs_diff_per_seq": out_diffs,
                "state_max_abs_diff_per_seq": state_diffs,
                "out_all_equal": max(out_diffs) == 0.0,
                "state_all_equal": max(state_diffs) == 0.0,
            }
            results.append(record)
            print(json.dumps(record, ensure_ascii=False))
    print(json.dumps({"fails": sum(1 for r in results if not r["out_all_equal"]),
                      "cases": len(results)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
