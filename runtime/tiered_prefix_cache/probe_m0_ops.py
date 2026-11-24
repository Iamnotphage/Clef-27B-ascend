#!/usr/bin/env python3
"""M0 probes: continuous-KV FIA (sparse_mode=3) and conv resume equivalence.

M0-1 checks that a suffix query against a concatenated continuous
``(K_prefix, K_suffix)`` with ``input_layout="TND"`` + ``sparse_mode=3`` +
``block_table=None`` equals the suffix rows of a full one-pass FIA, for B=1
and B=8 varlen.

M0-2 checks that ``causal_conv1d_fn(..., conv_states=last3,
has_initial_state=True)`` reproduces the full convolution's suffix rows.

Both probes are read-only and use one logical NPU.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

import torch
import torch_npu


def ulp_bf16() -> float:
    return 0.001953125


def causal_mask(device: str):
    tril = torch.ones((2048, 2048), dtype=torch.bool, device=device).tril_()
    return ~tril


def fia(query, key, value, q_cu, kv_cu, mask):
    return torch_npu.npu_fused_infer_attention_score(
        query, key, value,
        atten_mask=mask,
        num_heads=12,
        num_key_value_heads=2,
        input_layout="TND",
        actual_seq_lengths=q_cu,
        actual_seq_lengths_kv=kv_cu,
        block_table=None,
        block_size=0,
        scale=1.0 / math.sqrt(256),
        sparse_mode=3,
    )[0]


def probe_fia(device: str) -> dict:
    head_dim, heads, kv_heads = 256, 12, 2
    torch.manual_seed(20261125)
    results = []
    mask = causal_mask(device)

    def case(name, lengths, split):
        total = sum(lengths)
        q = torch.randn(total, heads, head_dim, dtype=torch.bfloat16, device=device)
        k = torch.randn(total, kv_heads, head_dim, dtype=torch.bfloat16, device=device)
        v = torch.randn(total, kv_heads, head_dim, dtype=torch.bfloat16, device=device)
        cu = [0]
        for length in lengths:
            cu.append(cu[-1] + length)
        q_cu = []
        kv_cu = []
        suffix_rows = []
        start = 0
        suffix_total = 0
        for length in lengths:
            q_cu.append(suffix_total)
            suffix_total += length - split
            kv_cu.append(cu[lengths.index(length)]) if False else None
            start += length
        # per-sequence suffix cu and full kv cu
        suffix_cu = [0]
        kv_full_cu = [0]
        for length in lengths:
            suffix_cu.append(suffix_cu[-1] + (length - split))
            kv_full_cu.append(kv_full_cu[-1] + length)
        full = fia(q, k, v, cu, cu, mask)
        positions = []
        for index, length in enumerate(lengths):
            base = sum(lengths[:index])
            positions.extend(range(base + split, base + length))
        q_suffix = q[positions].contiguous()
        incremental = fia(q_suffix, k, v, suffix_cu, kv_full_cu, mask)
        reference = full[positions]
        diff = (reference.float() - incremental.float()).abs()
        exists = torch.isfinite(reference.float()).all()
        results.append({
            "name": name,
            "lengths": lengths,
            "split": split,
            "shape": {"q": list(q_suffix.shape), "kv": list(k.shape)},
            "max_abs": float(diff.max()),
            "mean_abs": float(diff.mean()),
            "bit_equal": bool(torch.equal(reference, incremental)),
            "finite": bool(exists),
        })

    case("b1_967", [967], 768)
    case("b1_10294", [10294], 10112)
    case("b8_450", [450] * 8, 256)
    case("b8_mixed", [1024, 307, 967, 450, 512, 700, 300, 128], 128)
    return {"cases": results,
            "ok": all(row["finite"] and row["max_abs"] <= ulp_bf16()
                      for row in results)}


def probe_conv(device: str) -> dict:
    # The production fused conv (fla_npu) does not expose a state-read prefill
    # on A3 with a suffix-only x (the workspace call rejects the state path),
    # so resume uses the vLLM-Ascend reference conv, which the reference test
    # below shows is bitwise exact against its own full pass.  The engine
    # records the cross-kernel delta against ClefGdnConvQkv separately.
    from vllm_ascend.ops.causal_conv1d import causal_conv1d_fn as ref_fn

    torch.manual_seed(20261125)
    results = []
    for name, lengths, split in (
            ("b1_967", [967], 768),
            ("b1_10294", [10294], 10112),
            ("b8_450", [450] * 8, 256)):
        total = sum(lengths)
        x = torch.randn(total, 5120, dtype=torch.bfloat16, device=device)
        xt = x.transpose(0, 1).contiguous()
        weight = torch.randn(4, 5120, dtype=torch.bfloat16, device=device)
        dummy = torch.zeros(1, 5120, 3, dtype=torch.bfloat16, device=device)
        ci = torch.tensor([0], dtype=torch.int32, device=device)
        full = ref_fn(
            xt, weight, None, conv_states=dummy,
            query_start_loc=torch.tensor([0, total], dtype=torch.int32, device=device),
            cache_indices=ci,
            has_initial_state=torch.tensor([False], dtype=torch.bool, device=device),
            activation="silu", pad_slot_id=-1)
        suffix = []
        suffix_rows = []
        for index, length in enumerate(lengths):
            base = sum(lengths[:index])
            state = x[base + split - 3:base + split].transpose(0, 1).unsqueeze(0).contiguous()
            out = ref_fn(
                xt[:, base + split:base + length].contiguous(), weight, None,
                conv_states=state,
                query_start_loc=torch.tensor([0, length - split], dtype=torch.int32, device=device),
                cache_indices=ci,
                has_initial_state=torch.tensor([True], dtype=torch.bool, device=device),
                activation="silu", pad_slot_id=-1)
            suffix.append(out)
            suffix_rows.extend(range(base + split, base + length))
        joined = torch.cat(suffix, dim=1)
        reference = full[:, suffix_rows]
        diff = (reference.float() - joined.float()).abs()
        results.append({
            "name": name, "lengths": lengths, "split": split,
            "shape": list(full.shape),
            "max_abs": float(diff.max()),
            "mean_abs": float(diff.mean()),
            "bit_equal": bool(torch.equal(reference, joined)),
            "finite": bool(torch.isfinite(reference.float()).all()),
        })
    return {"cases": results,
            "ok": all(row["finite"] and row["max_abs"] <= ulp_bf16()
                      for row in results)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.npu.set_device(args.device)
    report = {
        "schema": "tiered-prefix-cache-m0-ops/1",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "device": args.device,
        "ulp_bf16": ulp_bf16(),
        "fia": probe_fia(args.device),
        "conv": probe_conv(args.device),
    }
    report["ok"] = report["fia"]["ok"] and report["conv"]["ok"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps({
        "ok": report["ok"],
        "fia": report["fia"],
        "conv": report["conv"],
    }, indent=1))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
