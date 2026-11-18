#!/usr/bin/env python3
"""N1 device cell. Execute only through worker21's supervised exclusive window.

Importing this file and --help do not import torch or touch a device.
No service lifecycle, installation, or automatic retries are implemented here.
"""
from __future__ import annotations

import argparse
from datetime import timedelta
import hashlib
import json
import os
from pathlib import Path
import traceback

from contract import parse_diag, sha256, verify_identity


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--owner", required=True, choices=["clef-27b-ascend-21"])
    p.add_argument("--identity", type=Path, required=True)
    p.add_argument("--vendor-root", type=Path, required=True)
    p.add_argument("--adapter", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--model", type=Path, default=Path("/workspace/yangchen/models/Clef-27B"))
    p.add_argument("--rows", type=int, default=967)
    p.add_argument("--seed", type=int, default=20261118)
    p.add_argument("--cores", type=int, default=1)
    p.add_argument("--use-hccl", type=int, choices=[0, 1], default=0)
    return p.parse_args()


def main():
    a = arguments()
    rank, local, world = (int(os.environ[k]) for k in ("RANK", "LOCAL_RANK", "WORLD_SIZE"))
    if world != 2 or rank not in (0, 1) or local != rank:
        raise RuntimeError("N1 requires rank/local_rank 0,1 and world_size=2")
    if os.environ.get("ASCEND_RT_VISIBLE_DEVICES") != "0,1":
        raise RuntimeError("owner must set ASCEND_RT_VISIBLE_DEVICES=0,1")
    if not (1 <= a.rows <= 16384 and 1 <= a.cores <= 48):
        raise ValueError("invalid rows/cores")
    identity = json.loads(a.identity.read_text())
    api = verify_identity(identity, a.vendor_root, a.adapter)
    out = a.output / f"rank{rank}"
    out.mkdir(parents=True, exist_ok=False)
    meta = {"rank": rank, "owner": a.owner, "stage": "prechecked",
            "identity_sha256": sha256(a.identity), "adapter_sha256": sha256(a.adapter),
            "api_path": str(api), "api_sha256": sha256(api),
            "seed": a.seed, "rows": a.rows, "cores": a.cores, "use_hccl": a.use_hccl,
            "eps": 1e-6, "inputs": {}, "files": {}}

    def flush():
        (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")

    flush()
    try:
        import torch
        import torch_npu
        import torch.distributed as dist
        from safetensors import safe_open
        from fixed_gate import evaluate

        torch.set_num_threads(1)
        torch.npu.set_device(local)
        dist.init_process_group("hccl", timeout=timedelta(seconds=90))
        group = dist.group.WORLD._get_backend(torch.device("npu")).get_hccl_comm_name(rank)
        torch.ops.load_library(str(a.adapter.resolve()))
        torch.npu.config.allow_internal_format = True
        meta["group"] = group
        meta["versions"] = {"torch": torch.__version__, "torch_npu": torch_npu.__version__}
        meta["stage"] = "adapter_loaded"
        flush()

        key = "model.language_model.layers.0.mlp.down_proj.weight"
        shard = json.loads((a.model / "model.safetensors.index.json").read_text())["weight_map"][key]
        with safe_open(a.model / shard, framework="pt", device="cpu") as f:
            full_w = f.get_tensor(key)
        width = full_w.shape[1] // 2
        w = full_w[:, rank * width:(rank + 1) * width].t().contiguous().npu()
        gen = torch.Generator().manual_seed(a.seed)
        x = torch.randn(a.rows, width, generator=gen).to(torch.bfloat16).npu()
        r = torch.randn(a.rows, 5120, generator=gen).to(torch.bfloat16).npu()
        g = torch.rand(5120, generator=gen).to(torch.bfloat16).npu()
        partial = torch.matmul(x, w)

        def cpu_bytes(t):
            c = t.detach().cpu().contiguous()
            return c, c.view(torch.uint8).numpy().tobytes()

        def save(name, t, write=True):
            c, raw = cpu_bytes(t)
            record = {"shape": list(t.shape), "dtype": str(t.dtype),
                      "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}
            if write:
                (out / f"{name}.bin").write_bytes(raw)
                meta["files"][name] = record
            return record

        meta["inputs"] = {name: save(name, t, name in ("partial", "r", "gamma"))
                          for name, t in (("x", x), ("w", w), ("r", r), ("gamma", g), ("partial", partial))}
        meta["weight"] = {"key": key, "shard": shard, "slice": [rank*width, (rank+1)*width],
                          "format": torch_npu.get_npu_format(w)}
        total = partial.clone()
        if a.use_hccl:
            dist.all_reduce(total, op=dist.ReduceOp.SUM)
        ref_y, ref_rs, ref_res = torch_npu.npu_add_rms_norm(total, r, g, 1e-6)
        torch.npu.synchronize()
        meta["stage"] = "before_candidate"
        flush()
        y, res, rs, reduced, diag = torch.ops.clef_n1_25.run(
            partial, r, g, 1e-6, group, a.use_hccl, a.cores, str(api), True)
        torch.npu.synchronize()
        cand = {"y": y.cpu(), "residual": res.cpu(), "rstd": rs.cpu()}
        ref = {"y": ref_y.cpu(), "residual": ref_res.cpu(), "rstd": ref_rs.reshape(-1).cpu()}
        gate = evaluate(cand, ref)
        reduced_ok = (reduced.shape == total.shape and reduced.dtype == total.dtype and
                      bool(torch.isfinite(reduced.float()).all()) and
                      bool(torch.isfinite(total.float()).all()) and torch.equal(reduced, total))
        diagnostic = parse_diag(diag.cpu().tolist(), a.rows, a.cores, a.use_hccl, rank)
        for name, tensor in (("reduced.reference", total), ("reduced.candidate", reduced),
                             ("y.reference", ref["y"]), ("y.candidate", cand["y"]),
                             ("residual.reference", ref["residual"]), ("residual.candidate", cand["residual"]),
                             ("rstd.reference", ref["rstd"]), ("rstd.candidate", cand["rstd"]), ("diag", diag)):
            save(name, tensor)
        result = {"fixed_gate": gate, "reduced_bit_equal": reduced_ok, "diag": diagnostic,
                  "pass": gate["pass"] and reduced_ok and diagnostic["pass"]}
        (out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        meta["stage"] = "complete"; meta["pass"] = result["pass"]; flush()
        dist.destroy_process_group()
        return 0 if result["pass"] else 1
    except BaseException:
        meta["error"] = traceback.format_exc()
        flush()
        raise


if __name__ == "__main__":
    raise SystemExit(main())
