#!/usr/bin/env python3
"""Measure the pure BF16 matmul ceiling for the real TP2 MLP shapes.

The MLP chain timing includes SwiGLU and two matmuls.  This probe isolates each
matmul with the real per-rank weights:

* gate/up: ``[M, 5120] x [5120, 34816]``
* down:     ``[M, 17408] x [17408, 5120]``

It reports warm and cold (256 MiB L2 flush) times and TFLOPS, compares ND vs
FRACTAL_NZ29, and adds a 4096^3 square matmul as a practical Cube ceiling.  It
also pads M to 128-multiples to quantify tail-tile waste.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import time
from pathlib import Path

import torch
import torch_npu
import torch.nn.functional as functional


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


HERE = Path(__file__).resolve().parent
probe = load("probe_mlp_ffn_for_ceiling", HERE / "probe_mlp_ffn.py")
nz = load("probe_mlp_nz_for_ceiling", HERE / "probe_mlp_nz.py")


def tflops(m: int, k: int, n: int, seconds: float) -> float:
    return 2.0 * m * k * n / seconds / 1e12


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=Path("/workspace/yangchen/models/Clef-27B"))
    parser.add_argument("--layer", type=int, default=31)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--input-file", type=Path, default=None)
    parser.add_argument("--tokens", default="307,401,512,969,1595,10294")
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--flush-mib", type=int, default=256)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    torch.set_num_threads(1)
    torch.manual_seed(20261115)
    probe.initialize_npu(torch, torch_npu, args.device)
    device = f"npu:{args.device}"
    packed_cpu, down_cpu, identity = probe.load_weights(
        args.model_dir, torch, args.layer, args.rank
    )
    packed = packed_cpu.to(device)
    down = down_cpu.to(device)
    packed_nz = torch_npu.npu_format_cast(packed, 29)
    down_nz = torch_npu.npu_format_cast(down, 29)

    flush = None
    if args.flush_mib:
        buf = torch.empty(args.flush_mib * 1024 * 1024 // 2,
                          dtype=torch.bfloat16, device=device)
        buf.uniform_(-1.0, 1.0)
        flush = lambda: torch.sum(buf)

    token_values = [int(item) for item in args.tokens.split(",") if item.strip()]
    if args.input_file:
        args.tokens = token_values
        input_cpu, metadata, _ = probe.load_captured_input(
            args, identity["hidden_size"], torch
        )
    else:
        input_cpu = None
        metadata = None

    hidden = identity["hidden_size"]
    intermediate = identity["intermediate_size"] // 2
    down_in = intermediate
    report = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "layer": args.layer,
        "rank": args.rank,
        "device": device,
        "identity": identity,
        "shape": {
            "gate_up": [intermediate * 2, hidden],
            "down": [hidden, intermediate],
        },
        "input_file": str(args.input_file) if args.input_file else None,
        "input_metadata": metadata,
        "cases": [],
    }

    def record(tokens, case, fn, m, k, n, padded_m=None):
        warm, cold = nz.measured(fn, args.warmup, args.iters, flush)
        rec = {
            "tokens": tokens,
            "case": case,
            "m": m,
            "k": k,
            "n": n,
            "warm_ms": warm,
            "cold_ms": cold,
            "warm_tflops": tflops(m, k, n, warm / 1000.0),
            "cold_tflops": tflops(m, k, n, cold / 1000.0),
        }
        if padded_m is not None:
            rec["padded_m"] = padded_m
            rec["padded_cold_tflops"] = tflops(padded_m, k, n, cold / 1000.0)
        report["cases"].append(rec)
        print(json.dumps(rec, ensure_ascii=False), flush=True)

    for tokens in token_values:
        if input_cpu is not None:
            if tokens > input_cpu.shape[0]:
                continue
            x = input_cpu[:tokens].contiguous().to(device)
        else:
            x = (torch.randn(tokens, hidden) * 0.5).to(device, torch.bfloat16)
        y = (torch.randn(tokens, down_in) * 0.5).to(device, torch.bfloat16)
        record(tokens, "gate_up_nd", lambda: functional.linear(x, packed),
               tokens, hidden, intermediate * 2)
        record(tokens, "gate_up_nz", lambda: functional.linear(x, packed_nz),
               tokens, hidden, intermediate * 2)
        record(tokens, "down_nd", lambda: functional.linear(y, down),
               tokens, down_in, hidden)
        record(tokens, "down_nz", lambda: functional.linear(y, down_nz),
               tokens, down_in, hidden)
        del x, y
        torch.npu.empty_cache()

    # Pad M to the next 128-multiple: the same number of Cube tiles, but the
    # last tile is full.  Effective TFLOPS uses the real M, padded TFLOPS uses
    # the padded M.
    for tokens in token_values:
        padded = ((tokens + 127) // 128) * 128
        if padded == tokens:
            continue
        x = torch.zeros(padded, hidden, dtype=torch.bfloat16, device=device)
        x[:tokens] = (torch.randn(tokens, hidden) * 0.5).to(device, torch.bfloat16)
        record(tokens, "gate_up_nz_padded", lambda: functional.linear(x, packed_nz),
               tokens, hidden, intermediate * 2, padded_m=padded)
        del x
        torch.npu.empty_cache()

    # Practical Cube ceiling: a large square matmul, no edge effects.
    size = 4096
    a = (torch.randn(size, size) * 0.5).to(device, torch.bfloat16)
    b = (torch.randn(size, size) * 0.5).to(device, torch.bfloat16)
    record(size, "square_4096", lambda: torch.matmul(a, b), size, size, size)
    del a, b
    torch.npu.empty_cache()

    text = json.dumps(report, ensure_ascii=False, indent=1)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(json.dumps({"cases": len(report["cases"]), "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
