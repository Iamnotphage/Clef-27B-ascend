#!/usr/bin/env python3
"""CPU integer model of per-core rstd FP32 writes, at most 8 values per DMA.

Validates logical DMA ranges only. No device/backend imports, floating-point
arithmetic changes, hardware transfer-granularity or synchronization proof.
"""
import argparse
import hashlib
import json
from pathlib import Path

BATCH = 8
ITEM_BYTES = 4
MAX_M = 16384
MAX_CORES = 48


def intervals(m, cores):
    assert 1 <= m <= MAX_M and 1 <= cores <= MAX_CORES
    return [(m * c // cores, m * (c + 1) // cores) for c in range(cores)]


def plans(m, cores):
    result = []
    for core, (start, end) in enumerate(intervals(m, cores)):
        transfers = []
        pending = 0
        batch_start = start
        for row in range(start, end):
            pending += 1
            if pending == BATCH or row + 1 == end:
                transfers.append({"row_begin": batch_start, "row_end": row + 1,
                                  "gm_byte_begin": ITEM_BYTES * batch_start,
                                  "gm_byte_end": ITEM_BYTES * (row + 1),
                                  "bytes": ITEM_BYTES * pending,
                                  "ub_byte_begin": 0, "ub_byte_end": ITEM_BYTES * pending})
                pending = 0
                batch_start = row + 1
        assert pending == 0
        result.append({"core": core, "row_range": [start, end],
                       "gm_byte_range": [ITEM_BYTES * start, ITEM_BYTES * end],
                       "transfers": transfers})
    return result


def verify_plan(m, cores, plan):
    assert len(plan) == cores
    global_cursor = 0
    count = total_bytes = 0
    for core, (entry, (start, end)) in enumerate(zip(plan, intervals(m, cores))):
        assert entry["core"] == core and entry["row_range"] == [start, end]
        assert entry["gm_byte_range"] == [ITEM_BYTES * start, ITEM_BYTES * end]
        cursor = ITEM_BYTES * start
        assert cursor == global_cursor
        transfers = entry["transfers"]
        assert len(transfers) == (end - start + BATCH - 1) // BATCH
        for i, transfer in enumerate(transfers):
            left, right = transfer["gm_byte_begin"], transfer["gm_byte_end"]
            size = transfer["bytes"]
            assert left == cursor and ITEM_BYTES * start <= left < right <= ITEM_BYTES * end
            assert size == right - left and 0 < size <= ITEM_BYTES * BATCH
            assert size % ITEM_BYTES == 0
            assert transfer["row_begin"] * ITEM_BYTES == left
            assert transfer["row_end"] * ITEM_BYTES == right
            assert transfer["ub_byte_begin"] == 0 and transfer["ub_byte_end"] == size
            assert i + 1 == len(transfers) or size == ITEM_BYTES * BATCH
            cursor = right
            total_bytes += size
            count += 1
        assert cursor == ITEM_BYTES * end
        global_cursor = cursor
    assert global_cursor == m * ITEM_BYTES and total_bytes == m * ITEM_BYTES
    return count


def exhaustive_arithmetic():
    """All 786432 M/cores cases; closed-form interval/batch/tail invariants."""
    cases = blocks = 0
    for m in range(1, MAX_M + 1):
        for cores in range(1, MAX_CORES + 1):
            cursor = covered = 0
            for c in range(cores):
                start, end = m * c // cores, m * (c + 1) // cores
                length = end - start
                assert start == cursor and 0 <= start <= end <= m
                full, tail = divmod(length, BATCH)
                assert BATCH * full + tail == length and 0 <= tail < BATCH
                full_end = start + BATCH * full
                assert start <= full_end <= end
                if full:
                    assert start + BATCH * (full - 1) < full_end <= end
                if tail:
                    assert full_end < end and end - full_end == tail
                    assert 0 < ITEM_BYTES * tail < ITEM_BYTES * BATCH
                else:
                    assert full_end == end
                assert full + bool(tail) == (length + BATCH - 1) // BATCH
                cursor = end
                covered += ITEM_BYTES * (BATCH * full + tail)
                blocks += 1
            assert cursor == m and covered == ITEM_BYTES * m
            cases += 1
    return {"cases": cases, "core_intervals": blocks, "M": [1, MAX_M],
            "cores": [1, MAX_CORES], "method": "exhaustive closed-form contiguous partition/full-batch/tail invariants"}


def run():
    assert __debug__, "Run without -O: validation uses assertions."
    exhaustive = exhaustive_arithmetic()
    representative_m = sorted(set(range(1, 66)) | {127, 128, 129, 307, 967, 1024, 1595, 16383, 16384})
    representative_cases = 0
    for m in representative_m:
        for cores in range(1, MAX_CORES + 1):
            verify_plan(m, cores, plans(m, cores))
            representative_cases += 1
    fixed = plans(967, 24)
    count = verify_plan(967, 24, fixed)
    histogram = {}
    for core in fixed:
        for transfer in core["transfers"]:
            size = str(transfer["bytes"])
            histogram[size] = histogram.get(size, 0) + 1
    # A padded 32-byte tail must be rejected because it crosses a core interval.
    bad = json.loads(json.dumps(fixed))
    tail = next(c["transfers"][-1] for c in bad if c["transfers"][-1]["bytes"] < 32)
    tail["gm_byte_end"] = tail["gm_byte_begin"] + 32
    tail["bytes"] = tail["ub_byte_end"] = 32
    rejected = False
    try:
        verify_plan(967, 24, bad)
    except AssertionError:
        rejected = True
    assert rejected
    return {"schema": "n1-rstd8-cpu-model/1", "scope": __doc__.strip(),
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "exhaustive": exhaustive, "enumerated_M": representative_m,
            "enumerated_cases": representative_cases, "padded_tail_negative_rejected": rejected,
            "fixed_967_24": {"old_dma_count": 967, "new_dma_count": count,
                             "dma_count_reduction": 967 - count, "total_bytes": 967 * 4,
                             "transfer_byte_histogram": histogram, "cores": fixed}, "pass": True}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("/tmp/n1-rstd8-model-result.json"))
    args = parser.parse_args()
    report = run()
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"pass": report["pass"], "output": str(args.output),
                      "output_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
                      "script_sha256": report["script_sha256"], "exhaustive": report["exhaustive"],
                      "enumerated_cases": report["enumerated_cases"],
                      "fixed_967_24": {k: v for k, v in report["fixed_967_24"].items() if k != "cores"}}, indent=2))
