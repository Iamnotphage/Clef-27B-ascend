#!/usr/bin/env python3
"""Generate deterministic BF16 inputs and the native npu_add_rms_norm outputs.

The launcher (runtime/wave3_norm_launcher.cpp) then runs the custom op on the
same bytes and compares Y / rstd / residual_sum against these native outputs.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch_npu


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=967)
    parser.add_argument("--hidden", type=int, default=5120)
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    torch.npu.set_device(0)
    m, n = args.rows, args.hidden
    i = torch.arange(m * n, dtype=torch.float32)
    x = (torch.sin(i * 1e-3) * 0.75).reshape(m, n).to(torch.bfloat16).npu()
    r = (torch.cos(i * 7e-4) * 0.5).reshape(m, n).to(torch.bfloat16).npu()
    g = (0.5 + 0.5 * torch.sin(torch.arange(n, dtype=torch.float32) * 1e-2)).to(torch.bfloat16).npu()

    y, rstd, residual_sum = torch_npu.npu_add_rms_norm(x, r, g, args.eps)
    torch.npu.synchronize()

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    for name, tensor in (("x.bin", x), ("r.bin", r), ("g.bin", g),
                         ("y.bin", y), ("res.bin", residual_sum)):
        tensor.cpu().contiguous().view(torch.uint16).numpy().tofile(out / name)
    rstd.cpu().contiguous().float().numpy().tofile(out / "rstd.bin")
    (out / "meta.txt").write_text(
        f"rows={m}\nhidden={n}\neps={args.eps}\ndtype=bf16\n"
        f"x=bf16(sin(i*1e-3)*0.75)\nr=bf16(cos(i*7e-4)*0.5)\n"
        f"g=bf16(0.5+0.5*sin(c*1e-2))\nnative=torch_npu.npu_add_rms_norm\n")
    print(f"written {out} rows={m} hidden={n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
