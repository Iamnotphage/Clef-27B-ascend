#!/usr/bin/env python3
"""Parse a D0 trace dump into strictly validated observations.

Contract in force (the proposed slot/seal changes await a ruling, so this
module keeps the ones below):

  fields = 8 uint32: [0] magic, [1] id=(block<<8)|sub, [2] domain,
    [3] claim_start_m, [4] claim_m, [5] stage_bits, [6] stage_bits2,
    [7] seal = 0xC0FFEE00 ^ id
  slot: AIC slot = block; AIV slot = AIC_CAP + block*SUB_CAP + sub

A record counts only when magic, seal, domain, the id bounds, the
slot-of-identity check and the stage bits are all valid AND it declares
itself a worker. Everything else is reported but never used as a claim.

Scope: capacity-internal observations only. An absent slot is not a
missing participant, and the trace never proves the full launch set, SDK
message receipt, partial writes, or GM visibility at the consumer.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

FIELDS = 8
SEAL_SEED = 0xC0FFEE00          # contract in force
DEFAULT_MAGIC = 0xD0D0C0DE      # proposed constant, overridable
UINT32_MAX = 0xFFFFFFFF

STAGE1 = {0: "K1", 1: "K2", 2: "K3", 3: "K4",
          4: "K5_enter", 5: "K5_return", 6: "K6_end_before", 7: "K6_end_after"}
STAGE2 = {0: "K7a", 1: "K7b", 2: "K8_final", 3: "is_worker", 4: "K8_scope_exit"}


def slots_for(aic_cap: int, sub_cap: int) -> int:
    return aic_cap * (1 + sub_cap)


def slot_of(domain: int, block: int, sub: int, aic_cap: int, sub_cap: int):
    if domain == 0:                      # AIC
        return block if sub == 0 else None
    if domain == 1:                      # AIV
        return aic_cap + block * sub_cap + sub
    return None


def _as_uint32(v, where: str) -> int:
    if isinstance(v, bool) or not isinstance(v, int):
        raise SystemExit(f"{where}: non-integer value {v!r}; refusing silent coercion")
    if v < 0 or v > UINT32_MAX:
        raise SystemExit(f"{where}: value {v} outside uint32 range")
    return v


def decode(raw: list, aic_cap: int, sub_cap: int, m: int, magic: int) -> dict:
    n = slots_for(aic_cap, sub_cap)
    if len(raw) != n * FIELDS:
        raise SystemExit(f"trace length {len(raw)} != {n * FIELDS} (slots={n})")
    present, rejected = [], []
    for s in range(n):
        f = [_as_uint32(v, f"slot{s}") for v in raw[s * FIELDS:(s + 1) * FIELDS]]
        got_magic, pid, domain, start_m, claim_m, bits, bits2, seal = f
        if got_magic == 0:
            continue                                   # never recorded
        why = []
        if got_magic != magic:
            why.append("magic")
        if seal != (SEAL_SEED ^ pid):
            why.append("seal")
        if domain not in (0, 1):
            why.append("domain")
        block, sub = pid >> 8, pid & 0xFF
        if domain == 0 and (block >= aic_cap or sub != 0):
            why.append("id_bounds")
        if domain == 1 and (block >= aic_cap or sub >= sub_cap):
            why.append("id_bounds")
        expect = slot_of(domain, block, sub, aic_cap, sub_cap) if domain in (0, 1) else None
        if expect != s:
            why.append(f"slot_mismatch(expect={expect})")
        rec = {"slot": s, "id": pid, "block": block, "sub": sub,
               "domain": "aic" if domain == 0 else ("aiv" if domain == 1 else "bad"),
               "claim_start_m": start_m, "claim_m": claim_m,
               "stages": [STAGE1[i] for i in range(8) if bits >> i & 1],
               "stages2": [STAGE2[i] for i in range(5) if bits2 >> i & 1],
               "is_worker": bool(bits2 >> 3 & 1)}
        if why:
            rec["rejected"] = why
            rejected.append(rec)
            continue
        if not rec["is_worker"]:
            rec["rejected"] = ["not_worker"]
            rejected.append(rec)
            continue
        present.append(rec)

    claims, bad_ranges = {"aic": [], "aiv": []}, []
    for r in present:
        start, length = r["claim_start_m"], r["claim_m"]
        end = start + length
        if not (0 <= start <= end <= m):
            bad_ranges.append({"slot": r["slot"], "start": start, "len": length, "end": end, "m": m})
            continue
        if length:
            claims[r["domain"]].append((start, end, r["slot"]))

    # AIC and AIV are MIX pairs: the same M range is reported by both domains,
    # so ranges are merged per domain and never across domains.
    per_domain = {}
    for dom, lst in claims.items():
        lst.sort()
        overlap, gaps, cursor = [], [], 0
        for start, end, slot in lst:
            if start < cursor:
                overlap.append({"slot": slot, "start": start, "prev_end": cursor})
            elif start > cursor:
                gaps.append({"from": cursor, "to": start})
            cursor = max(cursor, end)
        if lst and cursor < m:
            gaps.append({"from": cursor, "to": m, "tail": True})
        per_domain[dom] = {"ranges": [{"start": a, "end": b, "slot": c} for a, b, c in lst],
                           "overlap": overlap, "gaps": gaps, "covered_to": cursor}
    return {
        "aic_cap": aic_cap, "sub_cap": sub_cap, "slots": n, "m": m,
        "present": present, "rejected": rejected, "bad_ranges": bad_ranges,
        "claims_by_domain": per_domain,
        "scope": ("capacity-internal observations only; absent slot is not a missing "
                  "participant; launch set, SDK receipt, partial writes and consumer "
                  "visibility are outside this trace"),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace-json", type=Path, required=True)
    ap.add_argument("--aic-cap", type=int, required=True)
    ap.add_argument("--sub-cap", type=int, default=2)
    ap.add_argument("--m", type=int, required=True, help="row count M used for claim checks")
    ap.add_argument("--magic", type=lambda x: int(x, 0), default=DEFAULT_MAGIC)
    ap.add_argument("--output", type=Path)
    a = ap.parse_args()
    raw = json.loads(a.trace_json.read_text())
    if isinstance(raw, dict):
        raw = raw.get("trace") or raw.get("data") or []
    out = decode(raw, a.aic_cap, a.sub_cap, a.m, a.magic)
    text = json.dumps(out, ensure_ascii=False, indent=1)
    if a.output:
        a.output.write_text(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
