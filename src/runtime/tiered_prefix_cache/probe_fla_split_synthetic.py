#!/usr/bin/env python3
"""Stage 0.2 probe B pre-check: fla_npu chunk op split vs full (synthetic).

Confirms the operator contract before the real-activation capture:

- ``chunk_gated_delta_rule_fwd`` accepts ``initial_state`` + ``output_final_state``
  in the TP2 GDN shape and layout the service uses;
- one full call over ``[0, T)`` and a prefix/suffix pair joined by the prefix's
  final state produce the same suffix output and final state, modulo the BF16
  rounding of the state at the split boundary;
- an FP32 initial_state is rejected by the contract, so BF16 is the only
  re-storable precision.

This script uses random normalized-scale tensors; the real-activation capture
in ``probe_gdn_capture`` repeats the same comparison on production tensors.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import torch


def chunk_indices(tokens: int, chunk: int = 64) -> list[int]:
    count = (tokens + chunk - 1) // chunk
    flat: list[int] = []
    for index in range(count):
        flat.extend((0, index))
    return flat


def tensor_digest(value: torch.Tensor | None) -> dict | None:
    if value is None:
        return None
    flat = value.float().flatten()
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "max_abs": float(flat.abs().max()),
        "finite": bool(torch.isfinite(flat).all()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, default=4096)
    parser.add_argument("--split", type=int, default=2048)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    from fla_npu.ops.ascendc import chunk_gated_delta_rule_fwd as fused_fwd

    torch.npu.set_device(args.device)
    torch.manual_seed(20261124)
    tokens, split = args.tokens, args.split
    if split % 64 or tokens % 64 or not 0 < split < tokens:
        raise SystemExit("tokens and split must be multiples of 64")

    device = args.device
    q = torch.randn(1, tokens, 16, 128, dtype=torch.bfloat16, device=device)
    k = torch.randn(1, tokens, 16, 128, dtype=torch.bfloat16, device=device)
    v = torch.randn(1, tokens, 24, 128, dtype=torch.bfloat16, device=device)
    g = torch.nn.functional.logsigmoid(
        torch.randn(1, tokens, 24, dtype=torch.float32, device=device))
    beta = torch.rand(1, tokens, 24, dtype=torch.bfloat16, device=device)

    def run(q_in, k_in, v_in, g_in, b_in, initial_state=None):
        return fused_fwd(
            q_in, k_in, v_in, g_in, b_in,
            initial_state=initial_state,
            output_final_state=True,
            chunk_size=64,
            cu_seqlens=[0, int(q_in.shape[1])],
            chunk_indices=chunk_indices(int(q_in.shape[1])),
            scale=128 ** -0.5,
            layout="BSND",
            use_exp2=True,
            use_qk_l2norm_in_kernel=True,
            allow_neg_eigval=False,
            disable_recompute=True,
            state_v_first=True,
        )

    started = time.perf_counter()
    full_out, full_state, *_ = run(q, k, v, g, beta)
    torch.npu.synchronize()
    full_ms = (time.perf_counter() - started) * 1000.0

    started = time.perf_counter()
    prefix_out, prefix_state, *_ = run(
        q[:, :split], k[:, :split], v[:, :split], g[:, :split], beta[:, :split])
    torch.npu.synchronize()
    prefix_ms = (time.perf_counter() - started) * 1000.0

    started = time.perf_counter()
    suffix_out, suffix_state, *_ = run(
        q[:, split:].contiguous(), k[:, split:].contiguous(),
        v[:, split:].contiguous(), g[:, split:].contiguous(),
        beta[:, split:].contiguous(), initial_state=prefix_state)
    torch.npu.synchronize()
    suffix_ms = (time.perf_counter() - started) * 1000.0

    out_diff = (full_out[:, split:].float() - suffix_out.float()).abs()
    state_diff = (full_state.float() - suffix_state.float()).abs()
    report = {
        "schema": "tiered-prefix-cache-fla-split-synthetic/1",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "device": device,
        "tokens": tokens,
        "split": split,
        "dtype": "bfloat16",
        "layout": "BSND",
        "timing_ms": {
            "full": full_ms,
            "prefix": prefix_ms,
            "suffix": suffix_ms,
        },
        "full_out": tensor_digest(full_out),
        "full_state": tensor_digest(full_state),
        "suffix_out": tensor_digest(suffix_out),
        "prefix_state": tensor_digest(prefix_state),
        "suffix_out_max_abs_diff": float(out_diff.max()),
        "suffix_out_mean_abs_diff": float(out_diff.mean()),
        "final_state_max_abs_diff": float(state_diff.max()),
        "final_state_mean_abs_diff": float(state_diff.mean()),
        "fp32_initial_state_error": None,
    }
    try:
        run(q[:, split:].contiguous(), k[:, split:].contiguous(),
            v[:, split:].contiguous(), g[:, split:].contiguous(),
            beta[:, split:].contiguous(),
            initial_state=prefix_state.float())
        report["fp32_initial_state_error"] = "accepted (unexpected)"
    except Exception as error:  # noqa: BLE001 - the rejection is the result
        report["fp32_initial_state_error"] = f"{type(error).__name__}: {error}"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
