#!/usr/bin/env python3
"""Pure-CPU check of the per-worker M slicing arithmetic (no NPU, no build).

This mirrors exactly what the kernel computes:

    workers   = (usedCoreNum == 0) ? 1 : usedCoreNum
    singleM   = (singleCoreM == 0) ? ceil(m / workers) : singleCoreM
    startM    = blockIdx * singleM
    curSingleM = (startM >= m) ? 0
                 : min(m - startM, singleM)

and asserts the properties the multi-core contract depends on:

  1. every worker's [startM, startM + curSingleM) is inside [0, m);
  2. the ranges are pairwise disjoint;
  3. the ranges of the participating workers cover [0, m) exactly once;
  4. the usedCoreNum == 1 case reproduces the previous single-core call:
     startM == 0 and curSingleM == m, i.e. A offset 0, C offset 0 and
     SetSingleShape(m, n, k) -- the fixed mm_dim=1 path is unchanged.

It says nothing about the device: it only rules out arithmetic mistakes in the
slice computation before anything is built or run.
"""

import sys


def worker_range(m: int, single_core_m: int, used_core_num: int, block_idx: int):
    workers = 1 if used_core_num == 0 else used_core_num
    if single_core_m == 0:
        single_m = (m + workers - 1) // workers
    else:
        single_m = single_core_m
    # Coverage guard, mirrored from the kernel: only trust the tiling's
    # singleCoreM while it still covers M with `workers` workers.
    if single_m == 0 or single_m * workers < m:
        single_m = (m + workers - 1) // workers
    start_m = block_idx * single_m
    if start_m >= m or single_m <= 0:
        cur = 0
    else:
        cur = min(m - start_m, single_m)
    participates = (block_idx < workers) and (single_m > 0) and (cur > 0)
    return participates, start_m, cur, single_m


def check(m: int, single_core_m: int, used_core_num: int) -> list:
    workers = 1 if used_core_num == 0 else used_core_num
    covered = [0] * m
    spans = []
    for b in range(max(workers, 1)):
        part, start, cur, single_m = worker_range(m, single_core_m, used_core_num, b)
        if not part:
            continue
        assert 0 <= start < m, (m, single_core_m, used_core_num, b, start)
        assert 0 < cur <= m - start, (m, single_core_m, used_core_num, b, cur)
        for i in range(start, start + cur):
            covered[i] += 1
        spans.append((start, cur, single_m))
    # 2 + 3: disjoint and covering
    assert all(c == 1 for c in covered), f"coverage broken for {m},{single_core_m},{used_core_num}: {spans}"
    return spans


def main() -> int:
    cases = 0

    # The production shape from the frozen runs, swept over the budgets that
    # matter (1 = the archived control arm, and the AIC counts around it).
    for m in (967, 1024, 1595, 7, 1):
        for single_core_m in (0, 128, 64, 1):
            for used in (0, 1, 2, 3, 8, 24, 48):
                spans = check(m, single_core_m, used)
                cases += 1
                if used in (0, 1):
                    # 4: the fixed mm_dim=1 path must be the same call as before:
                    # one worker owning all of M (startM 0, curSingleM m), which is
                    # A offset 0, C offset 0 and SetSingleShape(m, n, k).  The
                    # coverage guard is what makes singleCoreM collapse to m here.
                    assert [(s[0], s[1]) for s in spans] == [(0, m)], \
                        f"single-core path changed: m={m} singleCoreM={single_core_m} -> {spans}"
                # every worker gets the same singleCoreM except the tail
                single_vals = {s[2] for s in spans}
                assert len(single_vals) == 1, f"inconsistent singleM: {spans}"

    # Degenerate: more workers than rows must still cover exactly once.
    spans = check(5, 1, 24)
    assert len(spans) == 5 and sum(s[1] for s in spans) == 5, spans

    # Non-divisible tail: the last participating worker must take the remainder.
    spans = check(967, 128, 8)
    assert spans[-1][1] == 967 - spans[-1][0] < 128, spans
    assert sum(s[1] for s in spans) == 967, spans

    print(f"slicing_cpu_check: OK  ({cases} cases, disjoint + exact cover + single-core equivalence)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
