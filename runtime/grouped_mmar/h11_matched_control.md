# H11 matched Grouped control (CPU design only; not scheduled)

Purpose: the A14 v5 rerun (plan `b6429191`, run_probe `0f90fd23`) has produced a
complete numeric result, so run one Grouped `aclnnGroupedMatMulAllReduce` cell
matched to that cell on every axis the current harness can control, to decide
whether the Grouped hang is op-specific or tied to the larger v4r1-like
shape/tiling.

## Decision (20/32)

Reuse the existing `H11` arm. No new arm, no adapter change, no binding change.

`H11` already is `flags = KEEP_ALL`, `native = True`, grouped op through
`torch.ops.clef_mmar_probe.run`, one tensor per list, `group_pointer` from the same
HCCL comm, FP32 zero bias, and the shared 2202165e window. The only thing the plan
must add is the shape: `--rows 128 --k 256 --n 512`.

## Exact CLI (plan first; execution only after 32 GO)

```
torchrun --nproc_per_node=2 --master_port=<29531 preflight FREE> \
  runtime/grouped_mmar/run_probe.py \
  --arm H11 --rows 128 --k 256 --n 512 --seed 20261119 \
  --temp-root <abs fresh> --out <abs fresh probe-out-h11> \
  --adapter <abs path to binding adapter> --arm-timeout 240
```

Add `--execute` only in the gated execution step. The adapter must be the same
binary reused for the A14 v5 rerun (`d181b15a...`), loaded through
`torch.ops.load_library`; its `provider()` must report
`prepare/launch=libopapi.so;metadata=libnnopbase.so`.

Preparation status (CPU-only, not executed): 31 built plan `68cc5a79` with
identity `1311514/399079056`, out/run/restore `probe-out-h11` / `run-h11` /
`restore-h11`, `port 29531` preflight FREE, `flags=KEEP_ALL`, `plan_not_executed`.
The plan pins the frozen run_probe `0f90fd23`, binding `101660f8`, window
`2202165e` and adapter `d181b15a`; 32/33 contract review then 32 GO.

## Plan contract

- `arm=H11`, `flags=8` (`KEEP_ALL`), `flag_names=["keep_all"]`, `native=true`,
  `norm=false`, `rows=128`, `k=256`, `n=512`, `seed=20261119`.
- Provider/adapter: same `libopapi.so` prepare/launch and `libnnopbase.so`
  metadata as A14; same adapter SHA as the A14 v4 cell.
- Comm domain: `dist.init_process_group('hccl')`, group name from
  `get_hccl_comm_name(rank)`, both ranks must report the same `group_name_0`.
- One group only: `Lists.add(x)` / `add(weight)` / `add(bias)` / `add(y)` each hold a
  single tensor; `prepare(tx, tw, tb, nullptr, 0, group, "sum", 0, 1, ty, ...)`.
- Inputs: x `[128,256]` BF16, weight `[256,512]` BF16 (per-rank generated with
  `seed+rank`, as in A14), bias `[512]` FP32 zeros (Grouped contract; A14 passes
  nullptr).
- Output: `y [128,512]` BF16 (Grouped keeps M; A14 reduce-scatter output is
  `[64,512]`).
- Reference: local `x @ w` BF16 then `dist.all_reduce(..., SUM)`; zero bias makes the
  bias term vanish. Numeric comparison is recorded as `reduced.max_abs` and
  `reduced.bit_equal`; it is not a pass condition.
- Numeric gate: `numeric_pass = api/sync/finite/both_finite`; `pass`/exit follow the
  numeric gate; `harness_pass` additionally requires `harness_release_ok`. The release
  op is still unregistered in the reused adapter, so the expected record is
  `release_error=NotImplementedError`, `harness_release_ok=false`,
  `harness_pass=false`, numeric fields intact and child rc0.
- Timeout/cleanup: 240 s child timeout, own PGID, watchdog exit 124, exact cleanup,
  then the official `start.sh` aa3e-noop restore.

## What stays different from A14 (unavoidable op interface)

| Axis | A14 small | H11 matched |
| --- | --- | --- |
| op | `aclnnMatmulReduceScatter` V1 | `aclnnGroupedMatMulAllReduce` |
| bias | nullptr | FP32 zero `[N]` |
| output | `[M/rank, N]` reduce-scatter | `[M, N]` all-reduce |
| grouping | single tensor | single-element tensor lists (one group) |
| everything else | M128/K256/N512, BF16, same group, same adapter/window, KEEP_ALL, 240 s | same |

## Discriminability

- H11 hangs with stream47/507046 → the remaining difference is the Grouped op
  itself (kernel/tiling/rac_server/context), not shape or lifetime: shape, BF16,
  single group, comm domain and KEEP_ALL are matched.
- H11 returns synchronously with finite output → the small shape is not sufficient
  to trigger the hang; move to the v4r1-like M968/K8704/N5120 cell and compare
  against A14 large.
- H11 prepare fails → record the op shape constraint and pick the nearest legal
  matched shape; do not reinterpret a prepare failure as a hang.

## Result interpretation (one small cell)

Read `result.json` on both ranks together; no single field decides a cause.

- `numeric_pass=true`, `reduced.bit_equal=true` (or small `max_abs`), child rc0,
  window ok → the Grouped op returned at M128/K256/N512. The small shape is not
  sufficient to trigger the hang; the next cell is the v4r1-like
  M968/K8704/N5120. `release_error=NotImplementedError` and
  `harness_pass=false` are the known release-op limitation and are reported apart.
- `numeric_pass=true` but `bit_equal=false` with a small `max_abs` → return with
  benign BF16 reduction-order divergence; record, do not call it a hang.
- `prepare rc != 0` / `api_returned=false` with an aclnn error → op shape or
  interface constraint. Record the rc and `aclGetRecentErrMsg`, adjust one
  dimension to the nearest legal matched shape, and do not reinterpret it as a
  hang.
- Sync never returns / watchdog 124 / window timeout → hang. Expect
  `stream_id=47`, 507046 (or the 3-minute timeout line) in plog; Grouped-specific
  op/kernel/tiling/rac_server/context stays the prioritized candidate, but one
  cell does not single out a cause.
- `group_names` differ between ranks → comm-domain mismatch, not an MC2 result;
  fix the group and rerun.
- `harness_cleanup_error` present → cleanup issue only; the numeric result was
  already flushed and stays valid.

## Order and gates

1. A14 small v4 (run_probe fdbfeb67, adapter d181b15a, identity from root 1308632,
   fresh `probe-out-a14-small-v4`, port 29531 preflight) completes with numeric
   fields and full restore.
2. 32 reviews the v4 result and gates H11.
3. H11 small uses the post-v4 recovery root identity, a fresh
   `probe-out-h11-small-v1`, port 29531 preflight, same window 2202165e.
4. A14 large M968/K8704/N5120 only after a complete small result plus 32 limited
   review.

Device execution is not scheduled by this document. Invariants unchanged: official
aa3e/f810 defaults, entry, metrics and CSV untouched; no PR; dependency on the
refused `mc2_server` remains supported_but_unproven; H6 limited conclusion and A13
unexecuted.
