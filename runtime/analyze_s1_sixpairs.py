#!/usr/bin/env python3
"""Pooled six-pair S1 analysis per the pre-registration: P95 drop per pair
(primary), order split (3 OFF->ON + 3 ON->OFF), thirds-of-window latency,
process age, r2 ramp->formal timing anomaly, validity checks."""
import json
import math
import statistics
import sys

RUN = "/workspace/yangchen/logs/tuning/20261113-120651-perf3"
A1 = f"{RUN}/ab-gdn-s1-ramped"      # pairs 1-3 (OFF -> ON)
A2 = f"{RUN}/ab-gdn-s1-ramped2"     # pairs 4-6 (ON -> OFF)


def stats_of(path):
    d = json.load(open(path))
    s = d["summary"]
    reqs = sorted(d["requests"], key=lambda r: r["index"])
    lat_all = [r["elapsed_ms"] for r in reqs if r.get("passed")]
    thirds = []
    n3 = len(lat_all) // 3
    for k in range(3):
        seg = lat_all[k * n3:(k + 1) * n3] if k < 2 else lat_all[2 * n3:]
        thirds.append(round(statistics.mean(seg), 2))
    failed = s["failed_requests"]
    wall = s["measurement_wall_seconds"]
    cpu = None
    try:
        a = dict(l.split() for l in open(path.replace(".json", ".cpu-before.txt")) if len(l.split()) == 2)
        b = dict(l.split() for l in open(path.replace(".json", ".cpu-after.txt")) if len(l.split()) == 2)
        cpu = {"throttle_us": int(b.get("throttled_usec", 0)) - int(a.get("throttled_usec", 0)),
               "util_us": int(b.get("usage_usec", 0)) - int(a.get("usage_usec", 0))}
    except (OSError, ValueError):
        pass
    npu = {}
    for side in ("before", "after"):
        try:
            npu[side] = [l.strip() for l in open(path.replace(".json", f".npu-{side}.txt")) if "Temp" in l or "Power" in l][:4]
        except OSError:
            pass
    return {"rps": s["successful_requests_per_second"], "mean": l_mean(s), "p50": s["successful_request_latency_ms"]["p50"],
            "p95": s["successful_request_latency_ms"]["p95"], "p99": s["successful_request_latency_ms"]["p99"],
            "succ": s["successful_requests"], "failed": failed, "wall": wall,
            "thirds": thirds, "cpu": cpu, "npu": npu,
            "valid": failed == 0 and wall >= 30 and (cpu is None or cpu["throttle_us"] == 0)}


def l_mean(s):
    return s["successful_request_latency_ms"]["mean"]


def pair(name_a, name_b, base, a_is_off):
    """a_is_off: for pairs 1-3 the first element is OFF; pairs 4-6 first is ON."""
    try:
        sa = stats_of(f"{base}/{name_a}.json")
        sb = stats_of(f"{base}/{name_b}.json")
    except (OSError, json.JSONDecodeError, KeyError) as e:
        return {"pair": name_a.replace("-off-", "|").replace("-on-", "|") + f"({name_a}|{name_b})", "error": repr(e)}
    off, on = (sa, sb) if a_is_off else (sb, sa)
    return {
        "pair": f"{name_a}|{name_b}",
        "order": "OFF->ON" if a_is_off else "ON->OFF",
        "off": {k: off[k] for k in ("rps", "mean", "p50", "p95", "p99", "valid", "thirds", "cpu", "npu", "failed", "wall")},
        "on": {k: on[k] for k in ("rps", "mean", "p50", "p95", "p99", "valid", "thirds", "cpu", "npu", "failed", "wall")},
        "p95_drop_ms": round(off["p95"] - on["p95"], 2),
        "p95_drop_pct": round((off["p95"] - on["p95"]) / off["p95"] * 100, 2),
        "rps_gain_pct": round((on["rps"] / off["rps"] - 1) * 100, 2),
        "mean_drop_ms": round(off["mean"] - on["mean"], 2),
        "p50_drop_ms": round(off["p50"] - on["p50"], 2),
        "p99_drop_ms": round(off["p99"] - on["p99"], 2),
    }


pairs = [
    pair("s1r-off-1", "s1r-on-1", A1, True),
    pair("s1r-off-2", "s1r-on-2", A1, True),
    pair("s1r-off-3", "s1r-on-3", A1, True),
    pair("s1x-on-4", "s1x-off-4", A2, False),
    pair("s1x-on-5", "s1x-off-5", A2, False),
    pair("s1x-on-6", "s1x-off-6", A2, False),
]

valid = [p for p in pairs if "error" not in p and p["off"]["valid"] and p["on"]["valid"]]
out = {"pairs": pairs, "n_valid": len(valid)}

if valid:
    drops = [p["p95_drop_ms"] for p in valid]
    positives = sum(1 for d in drops if d > 0)
    first3 = [p for p in valid if p["order"] == "OFF->ON"]
    last3 = [p for p in valid if p["order"] == "ON->OFF"]
    def agg(ps):
        if not ps:
            return None
        return {
            "p95_drop_ms_mean": round(statistics.mean([p["p95_drop_ms"] for p in ps]), 2),
            "p95_off_mean": round(statistics.mean([p["off"]["p95"] for p in ps]), 2),
            "p95_on_mean": round(statistics.mean([p["on"]["p95"] for p in ps]), 2),
            "rps_gain_pct_mean": round(statistics.mean([p["rps_gain_pct"] for p in ps]), 2),
            "mean_drop_ms_mean": round(statistics.mean([p["mean_drop_ms"] for p in ps]), 2),
            "positives": sum(1 for p in ps if p["p95_drop_ms"] > 0),
            "n": len(ps),
        }
    out["summary"] = {
        "first3_OFF_to_ON": agg(first3), "last3_ON_to_OFF": agg(last3), "pooled": agg(valid),
        "p95_positive_pairs": f"{positives}/{len(valid)}",
        "pooled_p95_drop_pct": round(statistics.mean([p["p95_drop_pct"] for p in valid]), 2),
        "order_consistent": (agg(first3)["p95_drop_ms_mean"] > 0) == (agg(last3)["p95_drop_ms_mean"] > 0) if first3 and last3 else None,
    }
    # adoption rule (pre-registered): >=5/6 positive AND pooled p95 drop >=1%
    pdrop = out["summary"]["pooled_p95_drop_pct"]
    out["decision"] = ("adopt" if positives >= 5 and pdrop >= 1.0 else
                       "reject" if positives <= 2 else
                       "reject-no-third-round")
    # r2 anomaly: on-2 ramp vs formal thirds
    try:
        r2 = stats_of(f"{A1}/s1r-on-2.json")
        out["r2_anomaly"] = {"ramp_formal_rps": json.load(open(f"{A1}/s1r-on-2-ramp.json"))["summary"]["successful_requests_per_second"],
                             "formal_rps": r2["rps"], "formal_thirds": r2["thirds"]}
    except Exception as e:
        out["r2_anomaly"] = {"error": repr(e)}

# process age for each supplementary service
ages = {}
try:
    for p in pairs:
        if "error" in p:
            continue
        for side in ("off", "on"):
            tag = [k for k in p["pair"].split("|") if side in k]
            if tag:
                t = tag[0]
                try:
                    ages[t] = json.load(open(f"{A2}/{t}.age.json"))
                except OSError:
                    pass
except Exception:
    pass
out["process_age"] = ages

json.dump(out, open(f"{RUN}/ab-gdn-s1-sixpair-analysis.json", "w"), indent=1)
print(json.dumps(out["summary"], indent=1))
for p in pairs:
    if "error" in p:
        print(p)
    else:
        print(f"{p['pair']} order={p['order']} p95 {p['off']['p95']}->{p['on']['p95']} "
              f"drop={p['p95_drop_ms']}ms/{p['p95_drop_pct']}% rps_gain={p['rps_gain_pct']}% "
              f"mean_drop={p['mean_drop_ms']}ms valid={p['off']['valid']}/{p['on']['valid']} "
              f"thirds_off={p['off']['thirds']} thirds_on={p['on']['thirds']}")
print("decision:", out.get("decision"))
