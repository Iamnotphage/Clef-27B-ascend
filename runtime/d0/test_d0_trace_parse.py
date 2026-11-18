#!/usr/bin/env python3
"""CPU counterexamples for the D0 trace parser. No device, no NPU import."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from d0_trace_parse import SEAL_SEED, decode, slots_for  # noqa: E402

AIC, SUB, M, MAGIC = 4, 2, 100, 0xD0D0C0DE
WORKER_BIT = 1 << 3


def blank():
    return [0] * (slots_for(AIC, SUB) * 8)


def put(raw, slot, block, sub, domain, start, length, bits=0, bits2=WORKER_BIT,
        magic=MAGIC, seal=None):
    pid = (block << 8) | sub
    base = slot * 8
    raw[base:base + 8] = [magic, pid, domain, start, length, bits, bits2,
                          SEAL_SEED ^ pid if seal is None else seal]


def run(raw):
    return decode(raw, AIC, SUB, M, MAGIC)


def main() -> int:
    ok = True

    # 1 normal coverage: AIC 0..3 and one AIV, non-overlapping full cover
    raw = blank()
    for b in range(AIC):
        put(raw, b, b, 0, 0, b * 25, 25)
    put(raw, AIC + 0, 0, 0, 1, 0, 10)
    r = run(raw)
    aic = r["claims_by_domain"]["aic"]
    c1 = (len(r["present"]) == AIC + 1 and not aic["overlap"]
          and not aic["gaps"] and aic["covered_to"] == M)
    print("normal coverage:", "OK" if c1 else f"FAIL {r['claim_gaps']} {r['claim_overlap']}")
    ok &= c1

    # 2 tail gap: stop covering at 90 of 100
    raw = blank()
    for b in range(AIC):
        put(raw, b, b, 0, 0, b * 22, 22)
    r = run(raw)
    c2 = any(g.get("tail") for g in r["claims_by_domain"]["aic"]["gaps"])
    print("tail gap:", "OK" if c2 else "FAIL")

    # 3 bad magic with a claim -> rejected, no claim
    raw = blank()
    put(raw, 0, 0, 0, 0, 0, 50, magic=0xDEADBEEF)
    r = run(raw)
    c3 = (not r["present"] and r["rejected"] and r["rejected"][0]["rejected"] == ["magic"]
          and not r["claims_by_domain"]["aic"]["ranges"])
    print("bad magic:", "OK" if c3 else f"FAIL {r['rejected']}")
    ok &= c3

    # 4 bad seal with a claim
    raw = blank()
    put(raw, 0, 0, 0, 0, 0, 50, seal=0x12345678)
    r = run(raw)
    c4 = (not r["present"] and r["partial_unsealed"]
          and r["partial_unsealed"][0]["rejected"] == ["seal"]
          and not r["claims_by_domain"]["aic"]["ranges"])
    print("bad seal -> partial:", "OK" if c4 else f"FAIL {r['partial_unsealed']}")
    ok &= c4

    # 5 non-worker with a claim
    raw = blank()
    put(raw, 0, 0, 0, 0, 0, 50, bits2=0)
    r = run(raw)
    c5 = (not r["present"]
          and r["rejected"][0]["inconsistent"] == ["not_worker_but_has_claim"]
          and not r["claims_by_domain"]["aic"]["ranges"])
    print("non-worker no coverage:", "OK" if c5 else f"FAIL {r['rejected']}")
    ok &= c5

    # 6 wrong domain / wrong slot
    raw = blank()
    put(raw, 0, 0, 0, 1, 0, 10)          # AIV identity written into an AIC slot
    put(raw, 1, 0, 0, 7, 0, 10)          # invalid domain
    r = run(raw)
    reasons = [w for x in r["rejected"] for w in x["rejected"]]
    c6 = (any("slot_mismatch" in w for w in reasons) and "domain" in reasons
          and not r["claims_by_domain"]["aic"]["ranges"]
          and not r["claims_by_domain"]["aiv"]["ranges"])
    print("wrong domain/slot:", "OK" if c6 else f"FAIL {r['rejected']}")
    ok &= c6

    # 7 out-of-bounds claim range
    raw = blank()
    put(raw, 0, 0, 0, 0, 90, 50)          # 90+50 > M
    r = run(raw)
    c7 = bool(r["bad_ranges"]) and not r["claims_by_domain"]["aic"]["ranges"]
    print("out-of-bounds range:", "OK" if c7 else f"FAIL {r['bad_ranges']}")
    ok &= c7

    # 8 silent coercion must be refused
    raw = blank()
    raw[3] = 1.5                          # float in claim_start_m
    try:
        run(raw)
        c8 = False
    except SystemExit:
        c8 = True
    print("float rejected:", "OK" if c8 else "FAIL")
    ok &= c8

    raw = blank()
    raw[3] = -1
    try:
        run(raw)
        c9 = False
    except SystemExit:
        c9 = True
    print("negative rejected:", "OK" if c9 else "FAIL")
    ok &= c9

    raw = blank()
    raw[3] = 0x1_0000_0000
    try:
        run(raw)
        c10 = False
    except SystemExit:
        c10 = True
    print("over-uint32 rejected:", "OK" if c10 else "FAIL")
    ok &= c10

    print("RESULT:", "ALL PASS" if ok else "FAILURES PRESENT")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
