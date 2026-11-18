#!/usr/bin/env python3
"""CPU counterexamples for the D0 trace parser. No device, no NPU import."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from d0_trace_parse import SEAL_SEED, decode, slots_for  # noqa: E402

AIC, SUB, M, MAGIC = 4, 2, 100, 0xD0D0C0DE
# stage_bits carries K1..K6 (bits 0..7 numbering starts at K1), stage_bits2
# carries K7a, K7b, K8_final, is_worker, K8_scope_exit.
W = 1 << 3                                   # is_worker in stage_bits2
K1_B, K2_B, K3_B, K4_B = 1 << 0, 1 << 1, 1 << 2, 1 << 3     # in stage_bits
K8_B2 = 1 << 2                               # K8_final in stage_bits2
K3_B = 1 << 2                                # K3 in stage_bits
K7a_B2, K7b_B2 = 1 << 0, 1 << 1              # in stage_bits2
K5E_B, K5R_B = 1 << 4, 1 << 5                # K5 enter / return in stage_bits
K6E_B, K6A_B = 1 << 6, 1 << 7                # K6 End before / after
DONE_B = K1_B | K4_B                         # required stage_bits for a claim
DONE_B2 = W | K8_B2                          # required stage_bits2 for a claim


def blank():
    return [0] * (slots_for(AIC, SUB) * 8)


def put(raw, slot, block, sub, domain, start=0, length=0,
        bits=0, bits2=0, magic=MAGIC, seal=None):
    pid = (block << 8) | sub
    base = slot * 8
    raw[base:base + 8] = [magic, pid, domain, start, length, bits, bits2,
                          SEAL_SEED ^ pid if seal is None else seal]


def run(raw):
    return decode(raw, AIC, SUB, M, MAGIC)


def main() -> int:
    fails = []

    def check(name, cond, extra=""):
        print(f"{name}: {'OK' if cond else 'FAIL ' + str(extra)}")
        if not cond:
            fails.append(name)

    # 1 normal: AIC sealed with K1/K2/K8 and no claim; AIV workers claim all M
    raw = blank()
    for b in range(AIC):
        put(raw, b, b, 0, 0, bits=K1_B | K2_B, bits2=K8_B2)   # AIC: K1,K2,K8 only
    for b in range(AIC):
        for s in range(SUB):
            idx = b * SUB + s
            span = M // (AIC * SUB)
            length = span + (M - span * (AIC * SUB) if idx == AIC * SUB - 1 else 0)
            put(raw, AIC + b * SUB + s, b, s, 1, start=idx * span,
                length=length, bits=DONE_B | K2_B | K3_B, bits2=DONE_B2)
    r = run(raw)
    check("normal aiv coverage", r["observed_coverage"] and not r["claim_overlap"]
          and not r["claim_gaps"], r["claim_gaps"])
    aic_obs = [o for o in r["observations"] if o["domain"] == "aic"]
    check("aic kept as observation", len(aic_obs) == AIC and not any(
        o.get("incomplete") for o in aic_obs), [o.get("incomplete") for o in aic_obs])
    check("aic never in claims", all(c["slot"] >= AIC for c in r["claim_ranges"]))

    # 2 tail gap fails the exit condition
    raw = blank()
    for b in range(AIC):
        put(raw, AIC + b * SUB, b, 0, 1, start=b * 20, length=20,
            bits=DONE_B | K2_B | K3_B, bits2=DONE_B2)
    r = run(raw)
    tail = any(g.get("tail") for g in r["claim_gaps"])
    check("tail gap not covered", (not r["observed_coverage"]) and tail, r["claim_gaps"])

    # 3 missing K4 -> incomplete, no claim
    raw = blank()
    put(raw, AIC, 0, 0, 1, start=0, length=M, bits=K1_B, bits2=DONE_B2)
    r = run(raw)
    check("missing K4 incomplete", not r["claim_ranges"]
          and any("missing:K4" in str(x.get("incomplete")) for x in r["incomplete"]), r["incomplete"])

    # 4 unsealed but pretends worker -> incomplete, no claim
    raw = blank()
    put(raw, AIC, 0, 0, 1, start=0, length=M, bits=DONE_B,
        bits2=DONE_B2, seal=0x1234)
    r = run(raw)
    check("unsealed no claim", (not r["claim_ranges"]) and len(r["partial_unsealed"]) == 1
          and r["partial_unsealed"][0]["incomplete"] == ["unsealed"])

    # 5 AIC erroneous claim -> contract anomaly, never a claim
    raw = blank()
    put(raw, 0, 0, 0, 0, start=0, length=M, bits=DONE_B | K2_B | K3_B, bits2=DONE_B2)
    r = run(raw)
    a0 = [o for o in r["observations"] if o["slot"] == 0][0]
    check("aic claim anomaly", "aic_claim" in a0.get("contract_anomaly", [])
          and not r["claim_ranges"])

    # 6 bad magic / bad domain / slot mismatch are identity rejects
    raw = blank()
    put(raw, 0, 0, 0, 0, magic=0xDEADBEEF)
    put(raw, 1, 0, 0, 7)
    put(raw, 2, 0, 0, 1)                       # AIV identity in an AIC slot
    r = run(raw)
    reasons = [w for x in r["identity_rejected"] for w in x["identity_errors"]]
    check("identity rejects", "magic" in reasons and "domain" in reasons
          and any("slot_mismatch" in w for w in reasons))

    # 7 out-of-range claim
    raw = blank()
    put(raw, AIC, 0, 0, 1, start=90, length=50, bits=DONE_B | K2_B | K3_B, bits2=DONE_B2)
    r = run(raw)
    check("out of range", bool(r["bad_ranges"]) and not r["claim_ranges"])

    # 8 no valid claim -> coverage false and unclaimed span reported
    raw = blank()
    put(raw, 0, 0, 0, 0, bits=K1_B, bits2=K8_B2)
    r = run(raw)
    check("no claim -> uncovered", r["observed_coverage"] is False
          and r["unclaimed_span"] == {"from": 0, "to": M})

    # 8a AIC with the full reachable set has no anomaly
    raw = blank()
    put(raw, 0, 0, 0, 0, bits=K1_B | K2_B, bits2=K8_B2)
    r = run(raw)
    a = [o for o in r["observations"] if o["slot"] == 0][0]
    check("aic final state no anomaly", not a.get("contract_anomaly"), a.get("contract_anomaly"))

    # 8a2 AIV worker with K4 but no K3 -> order violation, no claim
    raw = blank()
    put(raw, AIC, 0, 0, 1, start=0, length=M, bits=DONE_B, bits2=DONE_B2)
    r = run(raw)
    check("K4 without K3", not r["claim_ranges"]
          and any("order:" in str(x.get("incomplete")) for x in r["incomplete"]), r["incomplete"])

    # 8a3 K5 return without enter
    raw = blank()
    put(raw, AIC, 0, 0, 1, bits=K1_B | K5R_B, bits2=K8_B2)
    r = run(raw)
    check("return without enter", any("order:" in str(x.get("incomplete")) for x in r["incomplete"]))

    # 8a4 End after without before, barrier after without before
    raw = blank()
    put(raw, AIC, 0, 0, 1, bits=K6A_B, bits2=K8_B2 | K7b_B2)
    r = run(raw)
    inc = str([x.get("incomplete") for x in r["incomplete"]])
    check("end/barrier after without before", "K6_end_after->K6_end_before" in inc
          and "K7b->K7a" in inc, inc)

    # 8a5 unknown stage bit -> anomaly, no claim
    raw = blank()
    put(raw, AIC, 0, 0, 1, start=0, length=M, bits=DONE_B | K2_B | K3_B,
        bits2=DONE_B2 | (1 << 6))
    r = run(raw)
    check("unknown bit anomaly", not r["claim_ranges"]
          and any("unknown_stage_bits" in str(o.get("contract_anomaly"))
                  for o in r["observations"]), [o.get("contract_anomaly") for o in r["observations"]])

    # 8b AIV non-worker without K4 keeps identity, is complete and not a claim
    raw = blank()
    put(raw, AIC, 0, 0, 1, bits=K1_B, bits2=K8_B2)
    r = run(raw)
    aiv0 = [o for o in r["observations"] if o["slot"] == AIC][0]
    check("aiv non-worker identity kept", not aiv0.get("incomplete")
          and not r["claim_ranges"])

    # 8c K2 without K1 is an order violation
    raw = blank()
    put(raw, AIC, 0, 0, 1, bits=K2_B, bits2=K8_B2 | W)
    r = run(raw)
    inc = str([x.get("incomplete") for x in r["incomplete"]])
    check("K2 without K1", ("order:K2->K1" in inc or "missing:K1" in inc), inc)

    # 8d valid cover plus one out-of-range worker claim voids coverage
    raw = blank()
    for b in range(AIC):
        for s_ in range(SUB):
            idx = b * SUB + s_
            span = M // (AIC * SUB)
            length = span + (M - span * (AIC * SUB) if idx == AIC * SUB - 1 else 0)
            put(raw, AIC + idx, b, s_, 1, start=idx * span, length=length,
                bits=DONE_B | K2_B | K3_B, bits2=DONE_B2)
    put(raw, AIC + 1, 0, 1, 1, start=M, length=10,      # overwrite one slot
        bits=DONE_B | K2_B | K3_B, bits2=DONE_B2)
    r = run(raw)
    check("bad range voids coverage", bool(r["bad_ranges"]) and r["observed_coverage"] is False)

    # 8e magic zero with other words set is damaged, not absent
    raw = blank()
    raw[AIC * 8 + 3] = 7                        # a stray claim_start_m
    r = run(raw)
    check("torn write damaged", len(r["damaged_slots"]) == 1
          and (AIC not in r["absent_slots"]))

    # 9 coercion refusals
    for bad in (1.5, -1, 0x1_0000_0000):
        raw = blank()
        raw[3] = bad
        try:
            run(raw)
            check(f"reject {bad!r}", False)
        except SystemExit:
            check(f"reject {bad!r}", True)

    print("RESULT:", "ALL PASS" if not fails else f"FAILURES: {fails}")
    return 0 if not fails else 1


if __name__ == "__main__":
    raise SystemExit(main())
