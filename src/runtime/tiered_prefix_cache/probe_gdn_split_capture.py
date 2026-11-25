#!/usr/bin/env python3
"""Stage 0.2 probe B: production GDN prefill split vs 1-pass on captured data.

Consumes the tensors captured from the A3 service by
``probe_capture/sitecustomize.py`` and replays the exact Base adaptor
(``DeviceOperator.fla_gdn_prefill``) for each layer and TP rank:

1. production decision path: ``initial_state=None``, no final state;
2. full path from a zero state: returns the final state and checks that the
   zero-state start is the production result;
3. prefix ``[0, L_aligned)`` from the zero state, keeping the final state;
4. suffix ``[L_aligned, T)`` resumed from the prefix state.

The 2-pass split output and final state are compared with the 1-pass values.
The prefix length is a multiple of 128, which matches the kernel's chunk-128
boundary in the pinned A3 adaptor.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
from pathlib import Path
import time

import torch


def chunk_indices(tokens: int) -> tuple[int, ...]:
    indices: list[int] = []
    for chunk in range((int(tokens) + 127) // 128):
        indices.extend((0, chunk))
    return tuple(indices)


def stats(diff: torch.Tensor) -> dict:
    return {
        "max_abs": float(diff.max()),
        "mean_abs": float(diff.mean()),
        "n": int(diff.numel()),
    }


def load_capture(directory: Path) -> list[Path]:
    files = sorted(glob.glob(str(directory / "*-gdn.pt")))
    if not files:
        raise SystemExit(f"no captured gdn payloads under {directory}")
    return [Path(path) for path in files]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--split", type=int, required=True)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--layers", default="")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    from fla_npu.ops.ascendc import chunk_gated_delta_rule_fwd

    fused_fwd = chunk_gated_delta_rule_fwd
    if args.split % 128:
        raise SystemExit("--split must be a multiple of 128")

    torch.npu.set_device(args.device)
    wanted = {int(value) for value in args.layers.split(",") if value.strip()}
    report = {
        "schema": "tiered-prefix-cache-gdn-split/1",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "capture_dir": str(args.capture_dir),
        "split": args.split,
        "device": args.device,
        "layers": [],
    }
    totals = {"suffix_out": [], "final_state": [], "zero_vs_prod": []}

    for path in load_capture(args.capture_dir):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        layer = int(payload["layer"])
        if wanted and layer not in wanted:
            continue
        rank = int(payload["rank"])
        tokens = int(payload["tokens"])
        split = int(args.split)
        if not 0 < split < tokens:
            raise SystemExit(f"split {split} outside [0, {tokens}]")
        q = payload["q"].to(args.device)
        k = payload["k"].to(args.device)
        v = payload["v"].to(args.device)
        g = payload["g"].to(args.device)
        beta = payload["beta"].to(args.device)
        scale = float(payload["scale"])
        if not bool(payload.get("qkv_pre_normalized", False)):
            raise SystemExit(
                f"{path.name}: q/k are not pre-normalized; this probe only "
                "replays the ClefGdnQkvNorm path")

        def run(q, k, v, g, beta, initial_state, tokens_in, decision):
            # Exact Base-adaptor BSND branch for pre-normalized q/k/v.
            q = q.contiguous()
            k = k.contiguous()
            v = v.contiguous()
            g = g.to(torch.float32).contiguous()
            beta = beta.to(v.dtype).contiguous()
            initial_kernel = (
                None if initial_state is None else
                initial_state.transpose(-1, -2).contiguous())
            output, final_state, *_ = fused_fwd(
                q, k, v, g, beta,
                initial_state=initial_kernel,
                output_final_state=not decision,
                chunk_size=128,
                cu_seqlens=(0, int(tokens_in)),
                chunk_indices=chunk_indices(int(tokens_in)),
                scale=scale,
                layout="BSND",
                use_exp2=False,
                use_qk_l2norm_in_kernel=False,
                allow_neg_eigval=False,
                disable_recompute=True,
                state_v_first=False)
            if decision:
                return output, None
            return output, final_state.transpose(-1, -2).contiguous()

        states = {"layer": layer, "rank": rank, "tokens": tokens, "split": split}

        started = time.perf_counter()
        out_prod, _ = run(q, k, v, g, beta, None, tokens, True)
        torch.npu.synchronize()
        states["production_ms"] = (time.perf_counter() - started) * 1000.0

        state_shape = (1, int(v.shape[2]), int(v.shape[3]), int(q.shape[3]))
        zeros = torch.zeros(state_shape, dtype=torch.bfloat16, device=args.device)
        started = time.perf_counter()
        out_zero, state_zero = run(q, k, v, g, beta, zeros, tokens, False)
        torch.npu.synchronize()
        states["full_zero_ms"] = (time.perf_counter() - started) * 1000.0
        zero_diff = (out_prod.float() - out_zero.float()).abs()
        states["zero_vs_production_out"] = stats(zero_diff)
        totals["zero_vs_prod"].append(states["zero_vs_production_out"]["max_abs"])

        started = time.perf_counter()
        _, state_prefix = run(
            q[:, :split], k[:, :split], v[:, :split], g[:, :split],
            beta[:, :split], zeros, split, False)
        torch.npu.synchronize()
        states["prefix_ms"] = (time.perf_counter() - started) * 1000.0

        started = time.perf_counter()
        out_suffix, state_suffix = run(
            q[:, split:].contiguous(), k[:, split:].contiguous(),
            v[:, split:].contiguous(), g[:, split:].contiguous(),
            beta[:, split:].contiguous(), state_prefix, tokens - split, False)
        torch.npu.synchronize()
        states["suffix_ms"] = (time.perf_counter() - started) * 1000.0

        suffix_diff = (out_zero[:, split:].float() - out_suffix.float()).abs()
        state_diff = (state_zero.float() - state_suffix.float()).abs()
        states["suffix_out_vs_full"] = stats(suffix_diff)
        states["final_state_vs_full"] = stats(state_diff)
        states["prefix_state_finite"] = bool(
            torch.isfinite(state_prefix.float()).all())
        totals["suffix_out"].append(states["suffix_out_vs_full"]["max_abs"])
        totals["final_state"].append(states["final_state_vs_full"]["max_abs"])
        report["layers"].append(states)
        print(json.dumps(states), flush=True)

    report["summary"] = {
        "layers": len(report["layers"]),
        "suffix_out_max_abs": max(totals["suffix_out"], default=None),
        "final_state_max_abs": max(totals["final_state"], default=None),
        "zero_vs_production_max_abs": max(totals["zero_vs_prod"], default=None),
        "split_128_aligned": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report["summary"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
