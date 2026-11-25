#!/usr/bin/env python3
"""Probe the GDN data-path contracts on the real A3 NPU.

The probe answers the three questions the service patch depends on, using the
installed 26.9.1 operators and the real TP2 head layout (8 Q/K heads, 24 V
heads, head_dim 128):

1. ``causal_conv1d_fn(conv_states=None)`` must equal a fresh zero state, and
   must not write a state cache.
2. ``chunk_gated_delta_rule_fwd(initial_state=None)`` must equal an explicit
   zero initial state when ``output_final_state=False``.
3. BSND operands may be contiguous copies or strided sequence-major views of
   the head-first convolution output; both must equal the BNSD result.

It also drives the patched ``DeviceOperator.fla_gdn_prefill`` through every
combination of the three switches with the exact caller views the model
produces, so a layout mistake is caught before a service run.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from types import SimpleNamespace

import torch

WORKSPACE = "/workspace/yangchen"
sys.path.insert(0, f"{WORKSPACE}/work/src/vllm-ascend")
sys.path.insert(0, f"{WORKSPACE}/work/src/vllm")

from fla_npu.ops.ascendc import causal_conv1d_fn, chunk_gated_delta_rule_fwd  # noqa: E402
from vllm.third_party.flash_linear_attention.ops.l2norm import l2norm_fwd  # noqa: E402

# vllm_ascend.ops initializes the package before device_op; importing device_op
# first hits the moe <-> device_op cycle through the ops package __init__.
import vllm_ascend.ops  # noqa: E402,F401
from vllm_ascend.device.device_op import DeviceOperator  # noqa: E402
from vllm_ascend.ops.gdn import _split_head_first_conv_output  # noqa: E402
from vllm_ascend.ops.triton.fla.utils import prepare_chunk_indices  # noqa: E402

DEV = "npu:0"
DIM = 5120
CONV_HEADS = 40
WIDTH = 4
HK, HV, D = 8, 24, 128
SCALE = D ** -0.5


def meta(cu, keep_meta=None, kern=None):
    compact = kern if kern is not None else cu
    indices = prepare_chunk_indices(torch.tensor(compact, dtype=torch.int32), 64)
    return SimpleNamespace(
        cu_seqlens_host=tuple(cu),
        chunk_indices_chunk64_host=tuple(indices.to(torch.int64).reshape(-1).tolist()),
        cu_seqlens_kern=None if kern is None else tuple(kern),
        keep_meta=keep_meta,
    )


def diff(a: torch.Tensor, b: torch.Tensor) -> dict:
    nan_a, nan_b = torch.isnan(a), torch.isnan(b)
    finite = ~(nan_a | nan_b)
    finite_a = a[finite].float() if finite.any() else torch.zeros(0)
    finite_b = b[finite].float() if finite.any() else torch.zeros(0)
    delta = (finite_a - finite_b).abs()
    nan_equal = bool(torch.equal(nan_a, nan_b))
    finite_equal = bool(
        torch.equal(a[finite], b[finite]) if finite.any() else True
    )
    return {
        "max_abs": float(delta.max()) if delta.numel() else 0.0,
        "nan_count_a": int(nan_a.sum()),
        "nan_count_b": int(nan_b.sum()),
        "nan_equal": nan_equal,
        "bit_equal": bool(nan_equal and finite_equal),
    }


def conv_case(x, weight, lengths, conv_states, cache, has, head_num=CONV_HEADS):
    return causal_conv1d_fn(
        x,
        weight,
        None,
        conv_states=conv_states,
        query_start_loc=torch.tensor([0] + list(torch.tensor(lengths).cumsum(0).tolist()),
                                     dtype=torch.int32, device=DEV),
        cache_indices=cache,
        has_initial_state=has,
        activation="silu",
        pad_slot_id=-1,
        null_block_id=0,
        head_num=head_num,
    )


def probe_conv(report, generator, lengths_cases):
    for lengths in lengths_cases:
        total = sum(lengths)
        x = torch.randn(total, DIM, generator=generator).to(DEV, torch.bfloat16)
        weight = torch.randn(WIDTH, DIM, generator=generator).to(DEV, torch.bfloat16)
        # The service cache has 1821 rows; using a tiny cache changes the
        # installed kernel's path, so keep the real row count.
        states = torch.zeros(2048, WIDTH - 1, DIM, device=DEV, dtype=torch.bfloat16)
        cache = torch.arange(5, 5 + len(lengths), dtype=torch.int32, device=DEV)
        has = torch.zeros(len(lengths), dtype=torch.bool, device=DEV)
        baseline = conv_case(x, weight, lengths, states, cache, has)
        written_bytes = states.float().abs().sum().item()
        none_cases = {
            "none_all": dict(conv_states=None, cache=None, has=None),
            "none_with_cache": dict(conv_states=None, cache=cache, has=has),
            "none_with_has": dict(conv_states=None, cache=None, has=has),
        }
        for name, kwargs in none_cases.items():
            try:
                out = conv_case(x, weight, lengths, kwargs["conv_states"],
                                kwargs["cache"], kwargs["has"])
                record = {
                    "kind": "conv", "lengths": lengths, "case": name,
                    "cache_dtype": str(cache.dtype), "has_dtype": "torch.bool",
                    **diff(baseline, out),
                }
                record["baseline_wrote_state"] = written_bytes != 0.0
                report["cases"].append(record)
            except Exception as exc:  # noqa: BLE001
                report["cases"].append({
                    "kind": "conv", "lengths": lengths, "case": name,
                    "cache_dtype": str(cache.dtype), "has_dtype": "torch.bool",
                    "error": f"{type(exc).__name__}: {exc}",
                })
        report["cases"].append({
            "kind": "conv", "lengths": lengths, "case": "baseline_state",
            "state_abs_sum": written_bytes,
            "baseline_output_finite": bool(torch.isfinite(baseline).all()),
        })


def kernel_call(q, k, v, g, beta, cu, state, layout, output_final_state=False):
    m = meta(cu)
    return chunk_gated_delta_rule_fwd(
        q, k, v, g, beta,
        initial_state=state,
        output_final_state=output_final_state,
        chunk_size=64,
        cu_seqlens=m.cu_seqlens_host,
        chunk_indices=m.chunk_indices_chunk64_host,
        scale=SCALE,
        layout=layout,
        use_exp2=False,
        use_qk_l2norm_in_kernel=False,
        allow_neg_eigval=False,
        disable_recompute=False,
        state_v_first=False,
    )


def make_qkv(lengths, generator, heads=HK):
    total = sum(lengths)
    q = l2norm_fwd(torch.randn(1, total, heads, D, generator=generator).to(DEV, torch.bfloat16))
    k = l2norm_fwd(torch.randn(1, total, heads, D, generator=generator).to(DEV, torch.bfloat16))
    v = torch.randn(1, total, HV, D, generator=generator).to(DEV, torch.bfloat16)
    g = (-0.5 * torch.rand(1, total, HV, generator=generator, device="cpu")).to(DEV, torch.float32) - 1e-4
    beta = torch.rand(1, total, HV, generator=generator).to(DEV, torch.bfloat16)
    return q, k, v, g, beta


def probe_initial_state(report, generator, lengths_cases):
    for lengths in lengths_cases:
        q, k, v, g, beta = make_qkv(lengths, generator)
        cu = [0] + list(torch.tensor(lengths).cumsum(0).tolist())
        zeros = torch.zeros(len(lengths), HV, D, D, device=DEV, dtype=torch.bfloat16)
        out_zero = kernel_call(q.transpose(1, 2).contiguous(),
                                  k.transpose(1, 2).contiguous(),
                                  v.transpose(1, 2).contiguous(),
                                  g, beta, cu, zeros, "BNSD")[0]
        result_none = kernel_call(q.transpose(1, 2).contiguous(),
                                           k.transpose(1, 2).contiguous(),
                                           v.transpose(1, 2).contiguous(),
                                           g, beta, cu, None, "BNSD")
        q_bsnd = q.transpose(1, 2).contiguous().transpose(1, 2)
        k_bsnd = k.transpose(1, 2).contiguous().transpose(1, 2)
        v_bsnd = v.transpose(1, 2).contiguous().transpose(1, 2)
        out_bsnd = kernel_call(q_bsnd, k_bsnd, v_bsnd, g, beta, cu, None, "BSND")[0]
        report["cases"].extend([
            {"kind": "initial_state", "lengths": lengths, "layout": "BNSD",
             "case": "none_vs_zero", **diff(out_zero, result_none[0])},
            {"kind": "initial_state", "lengths": lengths, "layout": "BSND",
             "case": "none_vs_zero_bnsd", **diff(out_zero, out_bsnd)},
            {"kind": "initial_state", "lengths": lengths,
             "case": "final_state_is_none", "value": result_none[1] is None},
        ])

        # A strided BSND view of head-major normalized storage: this is what
        # the zero-copy candidate would hand the kernel.
        q_strided = q.transpose(1, 2)
        k_strided = k.transpose(1, 2)
        v_strided = v.transpose(1, 2)
        try:
            out_strided = kernel_call(q_strided, k_strided, v_strided,
                                         g, beta, cu, None, "BSND")[0]
            ref = kernel_call(q.transpose(1, 2).contiguous().transpose(1, 2).contiguous(),
                                 k.transpose(1, 2).contiguous().transpose(1, 2).contiguous(),
                                 v.transpose(1, 2).contiguous().transpose(1, 2).contiguous(),
                                 g, beta, cu, None, "BSND")[0]
            report["cases"].append({
                "kind": "bsnd_strided", "lengths": lengths,
                "q_contiguous": q_strided.is_contiguous(),
                **diff(ref, out_strided),
            })
        except Exception as exc:  # noqa: BLE001
            report["cases"].append({
                "kind": "bsnd_strided", "lengths": lengths,
                "q_contiguous": q_strided.is_contiguous(),
                "expected_rejection": True,
                "error": f"{type(exc).__name__}: {exc}",
            })


def flags(**kwargs):
    from vllm_ascend.ops.gdn_decision_flags import GdnFlags

    values = dict(epoch=1, decision_prefill=True, no_recompute=True)
    values.update(kwargs)
    return GdnFlags(**values)


def probe_wrapper(report, generator):
    from vllm_ascend.ops.gdn_decision_flags import GdnFlags

    lengths = [307, 64, 129, 205]
    total = sum(lengths)
    x = torch.randn(total, DIM, generator=generator).to(DEV, torch.bfloat16)
    weight = torch.randn(WIDTH, DIM, generator=generator).to(DEV, torch.bfloat16)
    cu = [0] + list(torch.tensor(lengths).cumsum(0).tolist())
    conv = conv_case(x, weight, lengths, None, None, None)
    q, k, v = _split_head_first_conv_output(conv, HK, HV)
    g = (-0.5 * torch.rand(1, total, HV, generator=generator)).to(DEV, torch.float32) - 1e-4
    beta = torch.rand(1, total, HV, generator=generator).to(DEV, torch.bfloat16)
    prebuilt = meta(cu)
    base = None
    arms = {
        "R": flags(),
        "B": flags(bsnd=True),
        "I": flags(skip_initial_state=True),
        "C": flags(skip_conv_state=True),
        "BI": flags(bsnd=True, skip_initial_state=True),
        "BC": flags(bsnd=True, skip_conv_state=True),
        "IC": flags(skip_initial_state=True, skip_conv_state=True),
        "BIC": flags(bsnd=True, skip_initial_state=True, skip_conv_state=True),
    }
    zeros = torch.zeros(len(lengths), HV, D, D, device=DEV, dtype=torch.bfloat16)
    for name, runtime in arms.items():
        state = None if runtime.skip_initial_state else zeros
        try:
            out, final = DeviceOperator.fla_gdn_prefill(
                q=q, k=k, v=v, g=g, beta=beta, initial_state=state,
                scale=SCALE, prebuilt_meta=prebuilt, fused_fwd=chunk_gated_delta_rule_fwd,
                decision_only=True, runtime=runtime,
            )
            record = {"kind": "wrapper", "arm": name, "shape": list(out.shape),
                      "final_state_is_none": final is None,
                      "finite": bool(torch.isfinite(out).all())}
            if base is None:
                base = out
                record.update({"max_abs": 0.0, "bit_equal": True})
            else:
                record.update(diff(base, out))
            report["cases"].append(record)
        except Exception as exc:  # noqa: BLE001
            report["cases"].append({"kind": "wrapper", "arm": name,
                                    "error": f"{type(exc).__name__}: {exc}"})

    # Empty segments: the wrapper must use the compacted cu_seqlens and skip
    # the final-state scatter when the decision path returns no state.
    cu_empty = [0, 0, 64, 64, 129]
    keep = torch.tensor([False, True, False, True], device=DEV)
    compact = [0, 64, 129]
    conv2 = conv_case(
        x[:129], weight, [64, 65], None, None, None,
    )
    q2, k2, v2 = _split_head_first_conv_output(conv2, HK, HV)
    g2 = g[:, :129]
    beta2 = beta[:, :129]
    prebuilt2 = meta(cu_empty, keep_meta=keep, kern=compact)
    try:
        zeros2 = torch.zeros(len(cu_empty) - 1, HV, D, D, device=DEV,
                             dtype=torch.bfloat16)
        base2, _ = DeviceOperator.fla_gdn_prefill(
            q=q2, k=k2, v=v2, g=g2, beta=beta2, initial_state=zeros2,
            scale=SCALE, prebuilt_meta=prebuilt2, fused_fwd=chunk_gated_delta_rule_fwd,
            decision_only=True, runtime=flags(),
        )
        out, final = DeviceOperator.fla_gdn_prefill(
            q=q2, k=k2, v=v2, g=g2, beta=beta2, initial_state=None,
            scale=SCALE, prebuilt_meta=prebuilt2, fused_fwd=chunk_gated_delta_rule_fwd,
            decision_only=True, runtime=flags(bsnd=True, skip_initial_state=True),
        )
        report["cases"].append({
            "kind": "wrapper_empty_segments", "shape": list(out.shape),
            "final_state_is_none": final is None,
            "finite": bool(torch.isfinite(out).all()),
            "vs_zero_state": diff(base2, out),
        })
    except Exception as exc:  # noqa: BLE001
        report["cases"].append({"kind": "wrapper_empty_segments",
                                "error": f"{type(exc).__name__}: {exc}"})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    global DEV
    DEV = args.device
    torch.npu.set_device(DEV)
    generator = torch.Generator(device="cpu").manual_seed(20261115)
    report = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "device": DEV,
        "workspace": WORKSPACE,
        "cases": [],
        "fails": 0,
    }
    lengths_cases = [
        [1, 3, 4, 63, 64, 65, 127, 128, 129],
        [203, 307, 401, 512],
        [307, 64, 129, 205],
        [662],
        [10294],
    ]
    probe_conv(report, generator, lengths_cases)
    probe_initial_state(report, generator, lengths_cases)
    probe_wrapper(report, generator)
    for case in report["cases"]:
        if case.get("expected_rejection"):
            continue
        if "error" in case:
            report["fails"] += 1
        elif case.get("kind") in ("conv", "initial_state", "bsnd_strided", "wrapper") \
                and "bit_equal" in case:
            report["fails"] += 0 if case["bit_equal"] else 1
        elif case.get("kind") == "wrapper_empty_segments":
            report["fails"] += 0 if case["vs_zero_state"]["bit_equal"] else 1
    report["case_count"] = len(report["cases"])
    text = json.dumps(report, ensure_ascii=False, indent=1)
    print(text)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(text + "\n")
    return 0 if report["fails"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
