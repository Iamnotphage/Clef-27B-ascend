#!/usr/bin/env python3
"""Parse a D0 trace dump (72 x 8 uint32) into observations.

Contract, as specified by the design and NOT re-derived here:
  fields = 8 uint32 per participant:
    [0] magic, [1] id=(block<<8)|sub, [2] domain,
    [3] claim_start_m, [4] claim_m, [5] stage_bits, [6] reserved,
    [7] seal = 0xC0FFEE00 ^ id
  slot layout: AIC slot = block; AIV slot = AIC_CAP + block*SUB_CAP + sub
  slots = AIC_CAP * (1 + SUB_CAP)

What this module deliberately does NOT conclude:
  - a missing slot does not mean a missing participant;
  - the trace never proves the full launch set, SDK message receipt,
    partial writes, or GM visibility at the consumer.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

FIELDS = 8
SEAL_SEED = 0xC0FFEE00
# Final contract from the D0 interface owner (22, 2026-11-18 23:03 CST).
STAGE1 = {0: "K1", 1: "K2", 2: "K3", 3: "K4",
          4: "K5_enter", 5: "K5_return", 6: "K6_end_before", 7: "K6_end_after"}
STAGE2 = {0: "K7a", 1: "K7b", 2: "K8_final", 3: "is_worker", 4: "K8_scope_exit"}


def slots_for(aic_cap: int, sub_cap: int) -> int:
    return aic_cap * (1 + sub_cap)


def slot_of(domain: str, block: int, sub: int, aic_cap: int, sub_cap: int) -> int:
    if domain == "aic":
        return block
    return aic_cap + block * sub_cap + sub


def decode(raw: list[int], aic_cap: int, sub_cap: int) -> dict:
    n = slots_for(aic_cap, sub_cap)
    if len(raw) != n * FIELDS:
        raise SystemExit(f"trace length {len(raw)} != {n * FIELDS} (slots={n}, fields={FIELDS})")
    present, absent, partial = [], [], []
    for s in range(n):
        f = raw[s * FIELDS:(s + 1) * FIELDS]
        magic, pid, domain, start_m, claim_m, bits, _bits2, seal = f
        if magic == 0:
            absent.append(s)
            continue
        bits2 = f[6]
        rec = {"slot": s, "id": pid, "block": pid >> 8, "sub": pid & 0xFF,
               "domain": "aic" if domain == 0 else "aiv",
               "claim_start_m": start_m, "claim_m": claim_m,
               "stages": [STAGE1[i] for i in range(8) if bits >> i & 1],
               "stages2": [STAGE2[i] for i in range(5) if bits2 >> i & 1],
               "is_worker": bool(bits2 >> 3 & 1)}
        if seal == (SEAL_SEED ^ pid):
            present.append(rec)
        else:
            rec["seal_mismatch"] = True
            partial.append(rec)
    claims = sorted(((r["claim_start_m"], r["claim_m"]) for r in present + partial
                     if r["claim_m"]), key=lambda x: x[0])
    overlap, gaps = [], []
    end = 0
    for start, length in claims:
        if start < end:
            overlap.append({"start": start, "prev_end": end})
        elif start > end:
            gaps.append({"from": end, "to": start})
        end = max(end, start + length)
    return {
        "aic_cap": aic_cap, "sub_cap": sub_cap, "slots": n,
        "present": present, "absent_slots": absent, "seal_mismatch": partial,
        "claim_ranges": claims, "claim_overlap": overlap, "claim_gaps": gaps,
        "claimed_span_end": end,
        "scope": "capacity-internal observations only; absent slot is not a missing participant",
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace-json", type=Path, required=True, help="file holding a flat uint32 list")
    ap.add_argument("--aic-cap", type=int, required=True)
    ap.add_argument("--sub-cap", type=int, default=2)
    ap.add_argument("--output", type=Path)
    a = ap.parse_args()
    raw = json.loads(a.trace_json.read_text())
    if isinstance(raw, dict):
        raw = raw.get("trace") or raw.get("data") or []
    out = decode([int(v) for v in raw], a.aic_cap, a.sub_cap)
    text = json.dumps(out, ensure_ascii=False, indent=1)
    if a.output:
        a.output.write_text(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
