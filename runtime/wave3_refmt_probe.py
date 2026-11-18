#!/usr/bin/env python3
"""Does the reference's physical weight format change torch.matmul output?

Same seed/inputs as the frozen full-chain run. No custom operator is called;
only the option readback and ref_ND vs ref_NZ at the 76 sampled points.
"""
from __future__ import annotations
import argparse, json, os
from pathlib import Path
import torch, torch.distributed as dist, torch_npu
from safetensors import safe_open


def opt(name):
    try:
        return str(torch_npu._C._npu_getOption(name))
    except Exception as e:  # noqa: BLE001
        return f"<err {e}>"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=967)
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--seed", type=int, default=20261118)
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    rank = int(os.environ["RANK"])
    torch.npu.set_device(int(os.environ.get("LOCAL_RANK", rank)))
    dist.init_process_group("hccl", init_method="env://")
    torch.npu.config.allow_internal_format = True

    idx = json.loads(Path("/workspace/yangchen/models/Clef-27B/model.safetensors.index.json").read_text())["weight_map"]
    key = f"model.language_model.layers.{a.layer}.mlp.down_proj.weight"
    with safe_open(Path("/workspace/yangchen/models/Clef-27B") / idx[key], framework="pt", device="cpu") as h:
        w_cpu = h.get_tensor(key)
    per = w_cpu.shape[1] // 2
    w_cpu = w_cpu[:, rank * per:(rank + 1) * per]
    w = w_cpu.t().contiguous().npu()
    w_nz = torch_npu.npu_format_cast(w, 29)
    k, n = w.shape
    g = torch.Generator().manual_seed(a.seed)
    x = torch.randn(a.rows, k, generator=g).to(torch.bfloat16).npu()

    ref_nd = torch.matmul(x, w)
    ref_nz = torch.matmul(x, w_nz)
    torch.npu.synchronize()
    d = (ref_nd.float() - ref_nz.float()).abs()
    union = Path(f"/workspace/yangchen/logs/perf-wave3-20261118/e-test/e-full-rt-rank{rank}-union76.json")
    pts = [int(p["idx"]) for p in json.loads(union.read_text())["points"]] if union.is_file() else []
    flat_nd, flat_nz = ref_nd.float().flatten(), ref_nz.float().flatten()
    diffs = [{"idx": i, "nd": float(flat_nd[i].cpu()), "nz": float(flat_nz[i].cpu()),
              "abs": float(d.flatten()[i].cpu())} for i in pts]
    out = {"rank": rank, "seed": a.seed, "k": k, "n": n,
           "options": {"ALLOW_MATMUL_HF32": opt("ALLOW_MATMUL_HF32"),
                       "ALLOW_INTERNAL_FORMAT": opt("ALLOW_INTERNAL_FORMAT"),
                       "CUBE_MATH_TYPE": opt("CUBE_MATH_TYPE")},
           "formats": {"x": torch_npu.get_npu_format(x), "w": torch_npu.get_npu_format(w),
                       "w_nz": torch_npu.get_npu_format(w_nz)},
           "full": {"max_abs": float(d.max().cpu()), "nonzero": int((d > 0).sum().cpu()),
                    "elements": int(d.numel())},
           "at76": {"nonzero": sum(1 for p in diffs if p["abs"] > 0), "n": len(diffs),
                    "max_abs": max([p["abs"] for p in diffs], default=0.0), "points": diffs}}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.with_name(a.output.stem + f"-rank{rank}.json").write_text(json.dumps(out, ensure_ascii=False) + "\n")
    print(json.dumps({"rank": rank, "max_abs": out["full"]["max_abs"], "nonzero_full": out["full"]["nonzero"],
                      "nonzero_76": out["at76"]["nonzero"], "opts": out["options"],
                      "fmt": out["formats"]}, ensure_ascii=False), flush=True)
    dist.barrier(); dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
