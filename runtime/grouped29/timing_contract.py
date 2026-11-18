"""Strict CPU-only aggregation of the GroupedMMAR 80-call timing protocol.

Both ranks must supply all four ordered AB/BA/AB/BA rounds, ten pairs per
round. Cross-rank spans require matching host, boot and monotonic clock
implementation. Correctness admission and raw-file hashes belong to the
caller; this module validates the complete timing evidence with stdlib only.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
import math

ROUND_ORDERS = (("baseline", "candidate"), ("candidate", "baseline"),
                ("baseline", "candidate"), ("candidate", "baseline"))
REPS = 10
_ARMS = ("baseline", "candidate")
_SCHEDULE_KEYS = ("round", "rep", "arm")
_SAMPLE_KEYS = set(_SCHEDULE_KEYS) | {
    "rank", "start_ns", "end_ns", "wall_ms", "event_ms"}
_METRICS = ("critical_path_wall_ms", "slow_rank_wall_ms", "slow_rank_event_ms")


def schedule():
    """Fresh ordered formal-call protocol; first call/warmup are excluded."""
    return [{"round": round_id, "rep": rep, "arm": arm}
            for round_id, order in enumerate(ROUND_ORDERS)
            for rep in range(REPS) for arm in order]


def _two_ranks(value, label):
    if isinstance(value, Mapping):
        keys = set(value)
        if all(type(k) is int for k in keys) and keys == {0, 1}:
            return [value[0], value[1]]
        if keys == {"0", "1"}:
            return [value["0"], value["1"]]
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if len(value) == 2:
            return list(value)
    raise ValueError(f"{label}: exactly ranks 0 and 1 are required")


def _number(value, label):
    try:
        valid = type(value) in (int, float) and math.isfinite(value) and value >= 0
    except OverflowError:
        valid = False
    if not valid:
        raise ValueError(f"{label}: finite nonnegative number required")
    return value


def _duration_ms(start, end, label):
    try:
        return _number((end - start) / 1e6, label)
    except OverflowError:
        raise ValueError(f"{label}: timestamp duration is unrepresentable") from None


def _clock_identity(metas):
    identities = []
    for rank, meta in enumerate(metas):
        if not isinstance(meta, Mapping):
            raise ValueError(f"meta rank{rank}: object required")
        if "rank" in meta and (type(meta["rank"]) is not int or meta["rank"] != rank):
            raise ValueError(f"meta rank{rank}: rank mismatch")
        clock = meta.get("clock")
        if not isinstance(clock, Mapping) or clock.get("monotonic") is not True:
            raise ValueError(f"meta rank{rank}: monotonic clock required")
        identity = {"hostname": meta.get("hostname"), "boot_id": meta.get("boot_id"),
                    "implementation": clock.get("implementation")}
        if any(not isinstance(v, str) or not v.strip() for v in identity.values()):
            raise ValueError(f"meta rank{rank}: hostname/boot_id/clock implementation required")
        identities.append(identity)
    if identities[0] != identities[1]:
        raise ValueError("rank clocks differ in hostname, boot_id or implementation")
    return {**identities[0], "monotonic": True}


def _validated_samples(records, rank, expected):
    if (not isinstance(records, Sequence) or isinstance(records, (str, bytes))
            or len(records) != len(expected)):
        raise ValueError(f"rank{rank}: expected exactly {len(expected)} samples")
    previous_end = None
    result = []
    for index, (sample, want) in enumerate(zip(records, expected)):
        label = f"rank{rank} sample{index}"
        if not isinstance(sample, Mapping) or set(sample) != _SAMPLE_KEYS:
            raise ValueError(f"{label}: sample fields differ from the timing schema")
        for field in ("round", "rep", "rank", "start_ns", "end_ns"):
            if type(sample[field]) is not int:
                raise ValueError(f"{label}: {field} must be an integer")
        if sample["rank"] != rank or any(sample[k] != want[k] for k in _SCHEDULE_KEYS):
            raise ValueError(f"{label}: rank or ordered schedule mismatch")
        start, end = sample["start_ns"], sample["end_ns"]
        if start < 0 or end < start:
            raise ValueError(f"{label}: invalid start/end timestamps")
        if previous_end is not None and start < previous_end:
            raise ValueError(f"{label}: rank-local samples overlap or run backwards")
        previous_end = end
        wall = _number(sample["wall_ms"], f"{label} wall_ms")
        _number(sample["event_ms"], f"{label} event_ms")
        if wall != _duration_ms(start, end, f"{label} duration"):
            raise ValueError(f"{label}: wall_ms differs from timestamp duration")
        result.append(deepcopy(dict(sample)))
    return result


def _distribution(values):
    """p50/p95 use linear interpolation at (count-1)*q."""
    ordered = sorted(values)
    if not ordered:
        return {"count": 0, "mean": None, "p50": None, "p95": None}

    def percentile(q):
        index = (len(ordered) - 1) * q
        low, high = math.floor(index), math.ceil(index)
        return ordered[low] + (ordered[high] - ordered[low]) * (index - low)

    return {"count": len(ordered),
            "mean": math.fsum(value / len(ordered) for value in ordered),
            "p50": percentile(0.5), "p95": percentile(0.95)}


def _pairs(calls):
    pairs = []
    for index in range(0, len(calls), 2):
        arms = {call["arm"]: call for call in calls[index:index + 2]}
        baseline, candidate = arms["baseline"], arms["candidate"]
        pair = {k: baseline[k] for k in ("round", "rep")}
        pair["baseline"] = {k: baseline[k] for k in _METRICS}
        pair["candidate"] = {k: candidate[k] for k in _METRICS}
        pair["speedup_baseline_over_candidate"] = {}
        for metric in _METRICS:
            denominator = candidate[metric]
            ratio = baseline[metric] / denominator if denominator else None
            pair["speedup_baseline_over_candidate"][metric] = (
                ratio if ratio is not None and math.isfinite(ratio) else None)
        pairs.append(pair)
    return pairs


def _summary(calls):
    arms = {}
    for arm in _ARMS:
        selected = [call for call in calls if call["arm"] == arm]
        arms[arm] = {"count": len(selected), **{
            metric: _distribution([call[metric] for call in selected])
            for metric in _METRICS}}
    pairs = _pairs(calls)
    speedups = {}
    for metric in _METRICS:
        values = [pair["speedup_baseline_over_candidate"][metric] for pair in pairs]
        finite = [value for value in values if value is not None]
        speedups[metric] = {**_distribution(finite),
                            "undefined_count": len(values) - len(finite)}
    return {"arms": arms, "pair_count": len(pairs), "pairs": pairs,
            "paired_speedups": speedups}


def aggregate_samples(rank_records, metas):
    """Validate exactly 80 calls per rank and aggregate matched call pairs.

    Retain both original records in every call. Reject incomplete or malformed
    evidence; leave input objects unchanged. Raw first calls and warmups are
    intentionally outside this formal timing contract.
    """
    clock = _clock_identity(_two_ranks(metas, "metas"))
    expected = schedule()
    records = [_validated_samples(records, rank, expected)
               for rank, records in enumerate(_two_ranks(rank_records, "rank_records"))]
    calls = []
    for index, item in enumerate(expected):
        samples = [records[0][index], records[1][index]]
        calls.append({**item, "samples": samples,
                      "slow_rank_event_ms": max(s["event_ms"] for s in samples),
                      "slow_rank_wall_ms": max(s["wall_ms"] for s in samples),
                      "critical_path_wall_ms": _duration_ms(
                          min(s["start_ns"] for s in samples),
                          max(s["end_ns"] for s in samples), f"call{index} critical path")})
    return {
        "schema": "grouped29-timing-aggregate/1", "clock_identity": clock,
        "protocol": {"round_orders": [list(order) for order in ROUND_ORDERS],
                     "reps": REPS, "samples_per_rank": len(expected),
                     "measure_samples_per_rank": 80, "first_call_warmup_included": False,
                     "percentile": "linear interpolation at (count-1)*q",
                     "critical_path": "max(rank end_ns)-min(rank start_ns), per matched call",
                     "undefined_ratio": "null when candidate duration is zero or ratio overflows"},
        "calls": calls,
        "measure": {
            "overall": _summary(calls),
            "rounds": [{"round": i, "order": list(order), **_summary(
                [call for call in calls if call["round"] == i])}
                for i, order in enumerate(ROUND_ORDERS)]},
    }
