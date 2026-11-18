"""Strict, CPU-only contract for a two-rank input-ready N1 timing window.

``rank_records`` and ``metas`` are two-element sequences in rank order, or
mappings with exactly the keys 0/1 (or their JSON string equivalents).
First-call and warmup pairs use baseline then candidate. Formal measurements
use four rounds of AB, BA, BA, AB, with five pairs in each round.

Cross-rank wall spans require a common host, boot and monotonic implementation.
This module validates timing records only; the caller supplies the fixed-gate
admission result separately. No device package is imported.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
import math

ROUND_ORDERS = (("baseline", "candidate"), ("candidate", "baseline"),
                ("candidate", "baseline"), ("baseline", "candidate"))
REPS = 5
WARMUP = 3
_ARMS = ("baseline", "candidate")
_SCHEDULE_KEYS = ("phase", "round", "rep", "position", "arm")
_SAMPLE_KEYS = set(_SCHEDULE_KEYS) | {
    "rank", "start_ns", "end_ns", "wall_ms", "event_ms"}
_METRICS = ("critical_path_wall_ms", "slow_rank_wall_ms", "slow_rank_event_ms")


def schedule():
    """Return a fresh, ordered 48-call protocol without rank or time fields."""
    result = []
    for phase, round_no, reps, order in (
            ("first_call", -1, 1, _ARMS), ("warmup", -1, WARMUP, _ARMS),
            *(("measure", i, REPS, order) for i, order in enumerate(ROUND_ORDERS))):
        for rep in range(reps):
            for position, arm in enumerate(order):
                result.append({"phase": phase, "round": round_no, "rep": rep,
                               "position": position, "arm": arm})
    return result


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
        for field in ("round", "rep", "position", "rank", "start_ns", "end_ns"):
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
    """Median and p95 with linear interpolation at index (n-1)*0.95."""
    ordered = sorted(values)
    if not ordered:
        return {"count": 0, "median": None, "p95": None}
    index = (len(ordered) - 1) * 0.95
    low, high = math.floor(index), math.ceil(index)
    p95 = ordered[low] + (ordered[high] - ordered[low]) * (index - low)
    middle = len(ordered) // 2
    median = ordered[middle] if len(ordered) % 2 else (
        ordered[middle - 1] + (ordered[middle] - ordered[middle - 1]) / 2)
    return {"count": len(ordered), "median": median, "p95": p95}


def _pairs(calls):
    pairs = []
    for index in range(0, len(calls), 2):
        arms = {call["arm"]: call for call in calls[index:index + 2]}
        baseline, candidate = arms["baseline"], arms["candidate"]
        pair = {k: baseline[k] for k in ("phase", "round", "rep")}
        pair["baseline"] = {k: baseline[k] for k in _METRICS}
        pair["candidate"] = {k: candidate[k] for k in _METRICS}
        pair["speedup_baseline_over_candidate"] = {}
        for metric in ("critical_path_wall_ms", "slow_rank_event_ms"):
            denominator = candidate[metric]
            # Zero event durations are valid raw records. Their ratio is
            # undefined and remains explicit, never Infinity or an invented gain.
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
    for metric in ("critical_path_wall_ms", "slow_rank_event_ms"):
        values = [pair["speedup_baseline_over_candidate"][metric] for pair in pairs]
        finite = [value for value in values if value is not None]
        speedups[metric] = {**_distribution(finite),
                            "undefined_count": len(values) - len(finite)}
    return {"arms": arms, "pair_count": len(pairs), "pairs": pairs,
            "paired_speedups": speedups}


def aggregate_samples(rank_records, metas):
    """Validate both complete raw sequences, then aggregate matched calls.

    ``calls`` retains both original rank records for every scheduled call.
    ``measure`` alone holds formal per-round/overall distributions and paired
    baseline/candidate ratios. ``first_call`` and ``warmup`` are separate raw
    call/pair records, excluded from all formal distributions. Invalid input
    raises ValueError; the function neither modifies inputs nor uses torch.
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
    measure = [call for call in calls if call["phase"] == "measure"]
    phases = {phase: [call for call in calls if call["phase"] == phase]
              for phase in ("first_call", "warmup")}
    return {
        "schema": "n1-timing-aggregate/1", "clock_identity": clock,
        "protocol": {"round_orders": [list(order) for order in ROUND_ORDERS],
                     "reps": REPS, "warmup": WARMUP,
                     "samples_per_rank": len(expected), "measure_samples_per_rank": 40,
                     "percentile": "linear interpolation at (count-1)*0.95",
                     "undefined_ratio": "null when candidate duration is zero or ratio overflows"},
        "calls": calls,
        **{phase: {"calls": selected, "pairs": _pairs(selected)}
           for phase, selected in phases.items()},
        "measure": {
            "overall": _summary(measure),
            "rounds": [{"round": i, "order": list(order), **_summary(
                [call for call in measure if call["round"] == i])}
                for i, order in enumerate(ROUND_ORDERS)]},
    }
