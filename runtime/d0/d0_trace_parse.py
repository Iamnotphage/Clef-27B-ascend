#!/usr/bin/env python3
"""Parse a D0 trace dump into strictly validated observations.

Contract in force (the proposed slot/seal changes await a ruling, so this
module keeps the ones below):

  fields = 8 uint32: [0] magic, [1] id=(block<<8)|sub, [2] domain,
    [3] claim_start_m, [4] claim_m, [5] stage_bits, [6] stage_bits2,
    [7] seal = 0xC0FFEE00 ^ id
  slot: AIC slot = block; AIV slot = AIC_CAP + block*SUB_CAP + sub
  AIV block is DERIVED by the probe from the public linear index and the
  ratio (host metadata marks identity_provenance=derived); the trace does
  not carry a raw linear or ratio readback, so this parser cannot
  re-derive it and only cross-checks the decoded (block, sub) against the
  slot that was written.

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


def check_derivation(block: int, sub: int, ratio: int, aiv_blocks: int) -> list:
    """Sanity checks for the derived AIV identity (host reports it as derived).

    The probe derives block from a public linear index; this only checks the
    result is internally plausible, it cannot re-derive it.
    """
    why = []
    if ratio <= 0:
        why.append("ratio_nonpositive")
    if sub < 0 or sub >= ratio:
        why.append("sub_out_of_ratio")
    if block < 0 or block >= aiv_blocks:
        why.append("block_out_of_domain")
    return why


REQUIRED_FOR_CLAIM = ("K1", "K4", "K8_final")   # prerequisites before a claim counts
STAGE_ORDER = ["K1", "K2", "K3", "K4", "K5_enter", "K5_return",
               "K6_end_before", "K6_end_after", "K7a", "K7b", "K8_final"]
ENTER_RETURN = ("K5_enter", "K5_return")


def decode(raw: list, aic_cap: int, sub_cap: int, m: int, magic: int,
           ratio: int | None = None, aiv_blocks: int | None = None) -> dict:
    n = slots_for(aic_cap, sub_cap)
    if len(raw) != n * FIELDS:
        raise SystemExit(f"trace length {len(raw)} != {n * FIELDS} (slots={n})")

    observations, identity_rejected, incomplete, absent, partial = [], [], [], [], []
    for s_ in range(n):
        f = [_as_uint32(v, f"slot{s_}") for v in raw[s_ * FIELDS:(s_ + 1) * FIELDS]]
        got_magic, pid, domain, start_m, claim_m, bits, bits2, seal = f
        if got_magic == 0:
            absent.append(s_)                       # never recorded, not "missing participant"
            continue
        block, sub = pid >> 8, pid & 0xFF
        sealed = seal == (SEAL_SEED ^ pid)
        why = []
        if got_magic != magic:
            why.append("magic")
        if domain not in (0, 1):
            why.append("domain")
        if domain == 0 and (block >= aic_cap or sub != 0):
            why.append("id_bounds")
        if domain == 1 and (block >= aic_cap or sub >= sub_cap):
            why.append("id_bounds")
        expect = slot_of(domain, block, sub, aic_cap, sub_cap) if domain in (0, 1) else None
        if expect != s_:
            why.append(f"slot_mismatch(expect={expect})")
        if domain == 1 and ratio is not None and aiv_blocks is not None:
            why.extend(check_derivation(block, sub, ratio, aiv_blocks))

        stages = [STAGE1[i] for i in range(8) if bits >> i & 1]
        stages += [STAGE2[i] for i in range(5) if bits2 >> i & 1]
        rec = {"slot": s_, "id": pid, "block": block, "sub": sub,
               "domain": "aic" if domain == 0 else ("aiv" if domain == 1 else "bad"),
               "claim_start_m": start_m, "claim_m": claim_m,
               "stages": stages, "is_worker": bool(bits2 >> 3 & 1),
               "sealed": seal == (SEAL_SEED ^ pid)}

        if why:
            rec["identity_errors"] = why
            identity_rejected.append(rec)
            continue
        if not sealed:
            # only the seal is missing: a stage record that never completed.
            rec["incomplete"] = ["unsealed"]
            partial.append(rec)
            continue

        # contract anomalies: AIC returns inside REGIST, so it can neither be a
        # worker nor carry a claim, and it never reaches the later stages.
        anomaly = []
        if rec["domain"] == "aic":
            if rec["is_worker"]:
                anomaly.append("aic_worker_bit")
            if rec["claim_m"]:
                anomaly.append("aic_claim")
            if any(x in stages for x in STAGE_ORDER[3:]):
                anomaly.append("aic_beyond_k3")
        if anomaly:
            rec["contract_anomaly"] = anomaly

        missing = [x for x in REQUIRED_FOR_CLAIM if x not in stages]
        if not rec["sealed"]:
            rec["incomplete"] = ["unsealed"]
        elif missing:
            rec["incomplete"] = [f"missing:{','.join(missing)}"]
        if rec.get("incomplete") or anomaly:
            # never qualifies for claims; kept as an observation only
            if not anomaly:
                incomplete.append(rec)
            else:
                observations.append(rec)
            if rec["domain"] == "bad":
                observations.append(rec)
            continue

        # enter must precede its return when both are present
        if all(x in stages for x in ENTER_RETURN) and stages.index("K5_enter") > stages.index("K5_return"):
            rec["incomplete"] = ["k5_order"]
            incomplete.append(rec)
            continue
        observations.append(rec)

    # claims come only from valid AIV worker records
    claims = []
    for r in observations:
        if r["domain"] == "aiv" and r["is_worker"] and not r.get("contract_anomaly"):
            claims.append((r["claim_start_m"], r["claim_start_m"] + r["claim_m"], r["slot"]))
    bad_ranges = [{"slot": c, "start": a, "end": b, "m": m}
                  for a, b, c in claims if not (0 <= a <= b <= m)]
    good = sorted(c for c in claims if 0 <= c[0] <= c[1] <= m and c[1] > c[0])
    overlap, gaps, cursor = [], [], 0
    for start, end, slot in good:
        if start < cursor:
            overlap.append({"slot": slot, "start": start, "prev_end": cursor})
        elif start > cursor:
            gaps.append({"from": cursor, "to": start})
        cursor = max(cursor, end)
    if good and cursor < m:
        gaps.append({"from": cursor, "to": m, "tail": True})
    covered = bool(good) and not overlap and not gaps and cursor == m

    return {
        "aic_cap": aic_cap, "sub_cap": sub_cap, "slots": n, "m": m,
        "observations": observations,
        "incomplete": incomplete,
        "partial_unsealed": partial,
        "identity_rejected": identity_rejected,
        "absent_slots": absent,
        "claim_ranges": [{"start": a, "end": b, "slot": c} for a, b, c in good],
        "bad_ranges": bad_ranges,
        "claim_overlap": overlap, "claim_gaps": gaps,
        "observed_coverage": covered,
        "unclaimed_span": None if covered else {"from": 0, "to": m},
        "coverage_basis": "valid sealed AIV worker records only (AIC cannot claim)",
        "scope": ("capacity-internal observations only; capacity is not an expected "
                  "count and an absent slot is not a missing participant; AIV block is "
                  "derived (identity_provenance=derived) with no raw linear/ratio "
                  "readback; launch set, SDK receipt, partial writes and consumer "
                  "visibility are outside this trace"),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace-json", type=Path, required=True)
    ap.add_argument("--aic-cap", type=int, required=True)
    ap.add_argument("--sub-cap", type=int, default=2)
    ap.add_argument("--m", type=int, required=True, help="row count M used for claim checks")
    ap.add_argument("--magic", type=lambda x: int(x, 0), default=DEFAULT_MAGIC)
    ap.add_argument("--ratio", type=int, help="declared sub ratio for AIV derivation checks")
    ap.add_argument("--aiv-blocks", type=int, help="AIV block domain size for derivation checks")
    ap.add_argument("--output", type=Path)
    a = ap.parse_args()
    raw = json.loads(a.trace_json.read_text())
    if isinstance(raw, dict):
        raw = raw.get("trace") or raw.get("data") or []
    out = decode(raw, a.aic_cap, a.sub_cap, a.m, a.magic, a.ratio, a.aiv_blocks)
    text = json.dumps(out, ensure_ascii=False, indent=1)
    if a.output:
        a.output.write_text(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
