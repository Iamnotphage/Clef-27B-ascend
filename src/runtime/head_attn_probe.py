#!/usr/bin/env python3
"""Compare nn.MultiheadAttention with a fused manual cross-attention path.

The joint head is 6.7-8.2 ms of host wall per request while its device compute
is only ~1.9 ms.  Each of its six attention blocks uses
``nn.MultiheadAttention``; this probe measures whether a manual
``q-projection + fused kv-projection + SDPA/bmm + out-projection`` path
reduces the dispatch cost for the real widths (W=1024, 16 heads, 967-token
memory) and whether its output matches the reference.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time

import torch
import torch.nn.functional as F


def time_call(fn, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    samples = []
    for _ in range(iters):
        started = time.perf_counter()
        fn()
        torch.npu.synchronize()
        samples.append((time.perf_counter() - started) * 1000.0)
    return statistics.median(samples)


def time_chain(fn, calls: int, repeats: int) -> float:
    for _ in range(2):
        for _ in range(calls):
            fn()
    torch.npu.synchronize()
    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        for _ in range(calls):
            fn()
        torch.npu.synchronize()
        samples.append((time.perf_counter() - started) * 1000.0)
    return statistics.median(samples)


def compare(reference: torch.Tensor, candidate: torch.Tensor) -> dict:
    delta = (reference.float() - candidate.float()).abs()
    return {
        "bit_equal": bool(torch.equal(reference, candidate)),
        "max_abs": float(delta.max()),
        "mean_abs": float(delta.mean()),
        "elements": reference.numel(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--memory", type=int, default=967)
    parser.add_argument("--query", type=int, default=4)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--chain", type=int, default=6)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    torch.npu.set_device(args.device)
    torch.manual_seed(20261117)
    device = f"npu:{args.device}"
    width, heads = args.width, args.heads
    head_dim = width // heads
    mha = torch.nn.MultiheadAttention(width, heads, batch_first=True, dropout=0.0).to(device).eval()
    memory = torch.randn(1, args.memory, width, device=device, dtype=torch.bfloat16)
    query = torch.randn(1, args.query, width, device=device, dtype=torch.bfloat16)
    weight = mha.in_proj_weight.detach().to(torch.bfloat16)
    bias = mha.in_proj_bias.detach().to(torch.bfloat16)
    out_weight = mha.out_proj.weight.detach().to(torch.bfloat16)
    out_bias = mha.out_proj.bias.detach().to(torch.bfloat16)

    def reference():
        with torch.no_grad():
            return mha(query, memory, memory, need_weights=False)[0]

    def project():
        q = F.linear(query, weight[:width], bias[:width])
        kv = F.linear(memory, weight[width:], bias[width:])
        k, v = kv.split(width, dim=-1)
        return (
            q.view(1, args.query, heads, head_dim).transpose(1, 2),
            k.view(1, args.memory, heads, head_dim).transpose(1, 2),
            v.view(1, args.memory, heads, head_dim).transpose(1, 2),
        )

    def manual_sdpa():
        with torch.no_grad():
            q, k, v = project()
            out = F.scaled_dot_product_attention(q, k, v)
            out = out.transpose(1, 2).reshape(1, args.query, width)
            return F.linear(out, out_weight, out_bias)

    def manual_math():
        with torch.no_grad():
            q, k, v = project()
            # Match F.multi_head_attention_forward: scale q before bmm.
            scores = torch.matmul(q * (head_dim ** -0.5), k.transpose(-1, -2))
            out = torch.matmul(scores.softmax(dim=-1), v)
            out = out.transpose(1, 2).reshape(1, args.query, width)
            return F.linear(out, out_weight, out_bias)

    functions = {"mha": reference, "manual_sdpa": manual_sdpa, "manual_math": manual_math}
    rows = {}
    reference_output = None
    for name, fn in functions.items():
        try:
            one = time_call(fn, args.warmup, args.iters)
            six = time_chain(fn, args.chain, 5)
            produced = fn()
            if reference_output is None:
                reference_output = produced
                equality = {"reference": True}
            else:
                equality = compare(reference_output, produced)
            rows[name] = {"one_ms": one, "six_ms": six, "equality": equality}
            print(json.dumps({"name": name, "one_ms": round(one, 4), "six_ms": round(six, 4),
                              "equality": equality}), flush=True)
        except Exception as error:  # noqa: BLE001 - report unsupported paths
            rows[name] = {"error": f"{type(error).__name__}: {error}"}
            print(json.dumps({"name": name, "error": rows[name]["error"]}), flush=True)

    with open(args.output, "w") as handle:
        json.dump({"args": vars(args), "rows": rows}, handle, indent=1)
        handle.write("\n")
    print(json.dumps({"output": args.output}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
