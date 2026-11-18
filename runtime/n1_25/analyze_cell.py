#!/usr/bin/env python3
"""Read-only CPU analysis of N1 raw bytes; no device backend or NPU imports."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path

from contract import sha256


def analyze(folder):
    # Import CPU torch only. Never import torch_npu or execute stored objects.
    os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
    import torch
    from fixed_gate import evaluate
    torch.set_num_threads(1)
    meta = json.loads((folder / "meta.json").read_text())
    tensors = {}
    for name, spec in meta["files"].items():
        path = folder / f"{name}.bin"
        if path.stat().st_size != spec["bytes"] or sha256(path) != spec["sha256"]:
            raise ValueError(f"raw identity mismatch: {path}")
        dtype = {"torch.bfloat16": torch.bfloat16, "torch.float32": torch.float32,
                 "torch.int32": torch.int32}[spec["dtype"]]
        tensors[name] = torch.frombuffer(bytearray(path.read_bytes()), dtype=dtype).reshape(spec["shape"])
    cand = {key: tensors[f"{key}.candidate"] for key in ("y", "residual", "rstd")}
    ref = {key: tensors[f"{key}.reference"] for key in ("y", "residual", "rstd")}
    result = {"rank": meta["rank"], "source_meta_sha256": sha256(folder / "meta.json"),
              "fixed_gate": evaluate(cand, ref), "semantic_probes": {}, "residual_failures": []}
    p, r, g = (tensors[k].float() for k in ("partial", "r", "gamma"))
    for arm in ("reference", "candidate"):
        total = tensors[f"reduced.{arm}"]
        z = total.float() + r
        expected_res = z.to(torch.bfloat16)
        # Explicit recorded rstd is used; no CPU reduction substitutes for device.
        normalized = (z * tensors[f"rstd.{arm}"].reshape(-1, 1)).to(torch.bfloat16)
        expected_y = (normalized.float() * g).to(torch.bfloat16)
        res_diff = (expected_res.float() - tensors[f"residual.{arm}"].float()).abs()
        y_diff = (expected_y.float() - tensors[f"y.{arm}"].float()).abs()
        result["semantic_probes"][arm] = {
            "residual_from_recorded_reduced_equal": torch.equal(expected_res, tensors[f"residual.{arm}"]),
            "residual_max_abs": float(res_diff.max()),
            "y_from_recorded_rstd_equal": torch.equal(expected_y, tensors[f"y.{arm}"]),
            "y_max_abs": float(y_diff.max()),
            "note": "CPU reconstruction uses recorded reduced/rstd; it is a diagnostic, not a device gate substitute"}
    result["reduced_bit_equal"] = torch.equal(tensors["reduced.candidate"], tensors["reduced.reference"])
    result["aroff_partial_equal"] = (torch.equal(tensors["partial"], tensors["reduced.candidate"])
                                       if meta["use_hccl"] == 0 else None)
    diff = (cand["residual"].float() - ref["residual"].float()).abs()
    tol = 1e-5 + 0.016 * ref["residual"].float().abs()
    for row, col in torch.nonzero(diff > tol)[:16].tolist():
        result["residual_failures"].append({
            "row": row, "col": col, "partial": float(p[row, col]), "r": float(r[row, col]),
            "reduced_candidate": float(tensors["reduced.candidate"][row, col]),
            "reduced_reference": float(tensors["reduced.reference"][row, col]),
            "candidate": float(cand["residual"][row, col]), "reference": float(ref["residual"][row, col]),
            "abs_diff": float(diff[row, col]), "tolerance": float(tol[row, col])})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rank-dir", required=True, type=Path)
    a = parser.parse_args()
    print(json.dumps(analyze(a.rank_dir), indent=2))


if __name__ == "__main__":
    main()
