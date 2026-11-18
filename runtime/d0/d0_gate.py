#!/usr/bin/env python3
"""Fixed numeric gate for the D0 cell. CPU only; no device, no NPU import.

Protocol (unchanged, not relaxed here):
  every output: shape and dtype equal, both sides finite
  residual    : all elements |c - r| <= 1e-5 + 0.016*|r|
  Y           : max_abs <= 0.02 and, over elements with |ref| >= 1e-5,
                max_rel <= 0.01 ; near-zero elements still enter abs and finite
  rstd        : max_abs <= 1e-3

The reference is the existing ND native protocol on the same x/w/r/gamma/eps
with BF16 GEMM and the established add/rmsnorm. Nothing here changes formats,
inputs or tolerances to force a pass.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ATOL, RTOL = 1e-5, 0.016
Y_ABS, Y_REL, RSTD_ABS = 0.02, 0.01, 1e-3
NEAR_ZERO = 1e-5


def _finite(t) -> bool:
    import torch
    return bool(torch.isfinite(t.float()).all())


def gate_output(name: str, cand, ref, mode: str) -> dict:
    import torch
    out = {"name": name, "mode": mode,
           "shape_ok": tuple(cand.shape) == tuple(ref.shape),
           "dtype_ok": cand.dtype == ref.dtype,
           "cand_finite": _finite(cand), "ref_finite": _finite(ref)}
    if not (out["shape_ok"] and out["dtype_ok"] and out["cand_finite"] and out["ref_finite"]):
        out["pass"] = False
        out["reason"] = "precondition"
        return out
    d = (cand.float() - ref.float()).abs()
    af = ref.float().abs()
    if mode == "mixed":
        tol = ATOL + RTOL * af
        bad = d > tol
        out.update({"max_abs": float(d.max()), "fail_count": int(bad.sum()),
                    "elements": int(d.numel()),
                    "pass": bool(not bad.any())})
    elif mode == "y":
        mask = af >= NEAR_ZERO
        rel = torch.where(mask, d / af.clamp(min=NEAR_ZERO), torch.zeros_like(d))
        mx, mr = float(d.max()), float(rel.max())
        out.update({"max_abs": mx, "max_rel": mr,
                    "near_zero_count": int((~mask).sum()),
                    "near_zero_max_abs": float(d[~mask].max()) if bool((~mask).any()) else 0.0,
                    "pass": bool(mx <= Y_ABS and mr <= Y_REL)})
    elif mode == "abs":
        mx = float(d.max())
        out.update({"max_abs": mx, "pass": bool(mx <= RSTD_ABS)})
    else:
        raise ValueError(mode)
    return out


def normalise(obj: dict) -> dict:
    """Only the reference is normalised (rstd to 1-D); the candidate is judged
    in the shape the operator actually produced."""
    import torch
    out = dict(obj)
    if "rstd" in out:
        out["rstd"] = out["rstd"].reshape(-1)
    return out


def evaluate(cand, ref) -> dict:
    """cand/ref: CPU tensors with residual/y/rstd, same r/gamma/eps."""
    results = [gate_output("residual", cand["residual"], ref["residual"], "mixed"),
               gate_output("y", cand["y"], ref["y"], "y"),
               gate_output("rstd", cand["rstd"], ref["rstd"], "abs")]
    return {"gate_version": "fixed-mixed-v1",
            "tolerances": {"atol": ATOL, "rtol": RTOL, "y_abs": Y_ABS,
                           "y_rel": Y_REL, "rstd_abs": RSTD_ABS,
                           "near_zero": NEAR_ZERO},
            "outputs": results,
            "pass": all(r["pass"] for r in results),
            "note": ("claims and stage observations are reported separately and never "
                     "substitute for this gate")}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", type=Path, required=True)
    ap.add_argument("--reference", type=Path, required=True)
    ap.add_argument("--output", type=Path)
    a = ap.parse_args()
    import torch
    cand = {k: torch.load(a.candidate / f"{k}.pt") for k in ("residual", "y", "rstd")}
    ref = {k: torch.load(a.reference / f"{k}.pt") for k in ("residual", "y", "rstd")}
    res = evaluate(cand, ref)
    text = json.dumps(res, ensure_ascii=False, indent=1)
    if a.output:
        a.output.write_text(text + "\n")
    print(text)
    return 0 if res["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
