# MRS V1 + all_gather alternative for the down_proj TP sum (CPU design only)

Status: design for 32/20 review. No code change, no device cell, no official-entry or
CSV change. This is an independent alternative path, not evidence about the Grouped
root cause. The H4K freeze (`0d3f7fb` / manifest `21758ff5`) is unchanged.

## 1. Why this path exists

`aclnnMatmulReduceScatter` V1 (MRS) is the only MC2 primitive already proven on this
A3 for the real TP2 local down_proj shape family: A14 v5 ran M128/K256/N512 with
prepare/launch/sync rc0 and `reduced.max_abs=0.0`, `bit_equal=true`. Grouped
(`aclnnGroupedMatMulAllReduce`) still hangs at the Python device sync under the same
comm/KEEP_ALL/adapter conditions. MRS therefore provides a numerically correct
matmul-plus-collective building block; all_gather can reconstruct the full sum that
the current `RowParallelLinear` all-reduce produces.

## 2. Current production path and exact shapes

From the recorded model facts (`docs/dense-sp-mlp-probe-2026-11-16.md`,
`docs/current-progress-2026-11-15.md`):

- hidden `H = 5120`, intermediate `I = 17408`, TP2.
- gate/up are present per rank at `[17408, 5120]`; the down weight is row-parallel at
  `[5120, 8704]` per rank, i.e. `K_local = 8704`, `N = 5120`.
- The activation reaching down is the local half `x_local [T, 8704]` BF16; the current
  path computes `partial_i = x_local @ W_i^T` and all-reduces to `y [T, 5120]`.
- `Qwen2MoeMLP.down_proj` is a linear without a separate bias in this path; the probe
  passes `bias=nullptr`. If the deployed config has `mlp_bias=True`, a replicated bias
  must be added once after the full sum (see 4).

## 3. The alternative

For rank `i` with local `x_i [T, 8704]` and `W_i [8704, 5120]` (torch layout
`[out, in]` transposed to the aclnn `[K, N]` view):

1. Pad `x_i` along M to `T' = ceil(T / 2) * 2` with zero rows (BF16), contiguous.
2. `y_i [T'/2, 5120] = aclnnMatmulReduceScatter(x_i_pad, W_i, group, "sum", rank=2)`.
   This is the local partial matmul followed by a reduce-scatter over the M halves.
3. `full [T', 5120] = all_gather_into_tensor(y_0, y_1)` on the same HCCL group;
   every rank then holds `sum_i (x_i @ W_i)` for all rows.
4. Slice `full[:T]` and pass it to the unchanged native `npu_add_rms_norm`.

This reproduces the current all-reduce result because reduce-scatter distributes the
M chunks of the summed partial matmul and all-gather reassembles them.

Row-block mapping (32 item 1): with `T' = 2 * (T'/2)`, rank 0 contributes and receives
rows `[0, T'/2)`, rank 1 rows `[T'/2, T')`; `all_gather_into_tensor` concatenates the
rank outputs **in rank order**, so the reassembled `[T', N]` keeps the original row
order. A15 must record, per rank, which row block it owns and verify the full
`[T', N]` against a full matmul+all_reduce reference, not only its own block. The
vLLM TP group rank order must match this rank order; that mapping is unproven and is
a production-wiring item (see 4).

## 4. Bias and TP-rank semantics

- Each rank owns the K-slice `[K_local, N]` of the down weight and the matching
  activation columns; MRS sums the two partial matmuls, exactly the all-reduce sum.
- Bias (32 item 3): the local repo/docs contain **no verified `mlp_bias` value** and
  no local `config.json` copy, and the current probe passes `bias=nullptr`; that is a
  precondition, not a proven fact. vLLM `RowParallelLinear` convention adds a
  row-parallel bias **once, after the all-reduce** (not per rank). If the Clef config
  is `mlp_bias=True`, the equivalent design adds the same replicated bias once after
  step 3; if `False`, the branch is unreachable and must be marked so. 31 must read
  the model config and the installed `RowParallelLinear` source (read-only) before
  any device work; the bias placement is otherwise unproven.
- Group (32 item 4): the service TP world is size 2. The TP HCCL group name must come
  from the vLLM TP group's `get_hccl_comm_name(rank)` captured once at init; the MRS
  call and the all_gather must use that **same group**. The probe's `group_name_0`
  only validates numbers; the vLLM group-name mapping and the TP-rank order are
  production-wiring items that remain unproven.
- Stream (32 item 4): MRS captures the current NPU stream at prepare and launches on
  it; the all_gather must be issued on that same stream so the MRS output is ordered
  before it. No `torch.npu.synchronize()` in production; the probe sync is measurement
  only. This ordering is unproven in the service and must be validated before enabling
  the flag.

## 5. Padding, shapes, reproducibility

| T (tokens) | T' padded | MRS output per rank | full after all_gather |
| --- | --- | --- | --- |
| 307 | 308 | [154, 5120] | [308, 5120] |
| 967 | 968 | [484, 5120] | [968, 5120] |
| 1595 | 1596 | [798, 5120] | [1596, 5120] |

Pad rows are appended **at the tail**; the original rows keep their positions and
`[:T]` drops only the tail. Because the matmul is row-independent, a pad row cannot
change any real row's sum. The A15 probe must prove this with **non-zero pad rows**
(not zeros): compare the original rows after slice against a full all_reduce reference
computed on the original `[T, N]` only. Reproducible probe shapes: `M=968, K=8704,
N=5120` (the real local shape; the earlier A14 large plan used exactly this), plus the
small `M=128, K=256, N=512` control. Seeds and input hashing follow run_probe.

## 6. Communication and memory cost (TP2, BF16)

Let `S = T' * 5120 * 2` bytes.

- All-reduce today (ring, R=2): `2 * (R-1)/R * S = S`.
- MRS reduce-scatter: `(R-1)/R * S = 0.5 S`; all-gather: `0.5 S`; total `S`.
- So the communication **volume is at parity** with all-reduce, but it is two
  collectives (extra launch/latency) and the all-gather is not overlapped with the
  matmul. Performance is not expected to beat the current AllReduce; this path is a
  correctness-proven fallback, not an optimization claim.
- Buffers at `T'=968`: `x_pad [968,8704]` ≈ 16.85 MB, MRS partial `[484,5120]` ≈
  4.96 MB, all-gather full `[968,5120]` ≈ 9.91 MB, and a separate ND weight
  `[8704,5120]` ≈ 89.1 MB per rank if the production NZ copy is kept.
- MRS workspace at the real shape is **untested**; at the small cell it was
  17,170,432 B.

## 7. Weight-format and lifetime constraints

- The probe `run_mrs` requires contiguous ND BF16 x/weight. The production MLP keeps
  FRACTAL_NZ29 weights (`CLEF_MLP_BF16_NZ=1`), so the alternative needs either a
  separate ND copy of down (`+89.1 MB/rank`, preferred for a first test) or proof that
  the installed MRS kernel accepts NZ. NZ vs ND outputs were bit-identical in the
  earlier MLP tests, so the ND reference comparison is valid.
- The probe binding never destroys the `aclOpExecutor` and its `release` op is
  unregistered. A production wrapper must resolve executor destruction and per-call
  workspace ownership before this path can be enabled; this is a productionization
  gap, not part of the current probe.

## 8. Minimal change scope and step gates

1. **CPU design (this document)** → 32/20 review.
2. **Device validation, existing code only**: run the already-designed A14 large cell
   (`M=968, K=8704, N=5120`, KEEP_ALL) to prove MRS support/workspace/sync at the real
   shape. No new arm; fresh identity/plan. If it hangs or is unsupported, the
   alternative stops here.
3. **Probe reconstruction**: minimal new arm `A15` (≈15 lines in run_probe, no binding
   change) that reuses `run_mrs`, then `dist.all_gather_into_tensor`, and compares the
   full cropped `[T, N]` to a matmul+all_reduce reference; records max_abs/mean_abs/
   bit_equal, per-rank row blocks, pad/crop timing and comm timing.
   Threshold provenance (32 item 5): `fixed_gate.py` `041a66aa` is an **AddRMSNorm
   y/residual/rstd gate**, not an MLP/down gate. The numbers `0.03125` and `2e-5`
   come from the project's earlier probes (`docs/mc2-reduce-scatter-2026-11-16.md`
   reported max_abs 0.03125; `docs/dense-sp-mlp-probe-2026-11-16.md` reported
   max_abs 0.015625, mean_abs ≈2e-5). A15 therefore uses a **pre-registered
   equivalence diagnostic** of `max_abs <= 0.03125` and `mean_abs <= 1e-4` (bounded
   above the observed values) and labels it as such; it is not a new official gate.
   The authoritative gates remain `fixed_gate` for the norm boundary and the official
   2PP for the full model.
4. **Model wrapper** behind an off-by-default env flag (e.g. `CLEF_MLP_MRS_AG=1`), with
   padding/slice, optional bias, cached ND down weight, same stream, no sync. No
   official entry/default/CSV change in this step.
5. **Accuracy gate**: unchanged native `npu_add_rms_norm`; frozen `fixed_gate`
   thresholds; then full OpenDecisions 2352 and ERQA 400 within 2PP of the current
   BF16 NZ baseline, with exact response/probability diffs recorded.

## 9. Preconditions and unknowns (must stay explicit)

- **MRS large support and performance are untested.** Only M128/K256/N512 executed;
  M968/K8704/N5120 is the next gate.
- Workspace size, executor lifetime, and any NZ acceptance at the real shape are
  unknown.
- The model config's `mlp_bias` must be confirmed (read-only) before choosing the
  nullptr/zero-bias variant; vLLM's bias-after-all-reduce placement must be confirmed
  from the installed source.
- Group-name retrieval from the vLLM TP group, TP-rank order and stream ordering must
  be validated in the service before enabling the flag.
- The A15 probe itself uses the probe group `group_name_0` and its own sync; that only
  validates numbers and ordering in the harness, not the production wiring.
- This path does not explain the Grouped hang, does not establish the `mc2_server`
  dependency, and does not claim a throughput win. Grouped root-cause work and this
  equivalent substitution stay separate conclusions. H5K's later timeline shows its
  handler blocked inside `getCurrentNPUStream()` before launch, so H5K is not a valid
  Grouped device-path discriminator; that does not affect this design.

## 11. Full-layer HBM budget (32 item 6)

Per layer per rank the ND down weight is `[8704, 5120]` BF16 = 89,128,960 B
(85.0 MiB). The model has 64 layers with an MLP (`docs/current-progress-2026-11-15.md`),
so a full dual-format resident copy would add:

- `64 * 85.0 MiB = 5,440 MiB = 5.31 GiB per rank`, TP2 total **10.62 GiB**.

Current budget facts: 64 GiB HBM per rank; the recorded TP2 vLLM target budget at
`gpu_memory_utilization` 0.90 is about **55.14/55.15 GiB per rank**
(`docs/baseline-2026-11-12.md`), which already includes weights, activations and the
KV pool. The remaining KV/activation headroom is roughly the difference between that
budget and the per-rank weights; adding 5.31 GiB/rank of ND down copies would cut the
KV pool by the same amount (about a fifth of the observed KV headroom), which is not
free for 4096-token and multi-image ERQA workloads. A full dual-copy is therefore
**not the preferred plan**.

Preferred: **load-time format replacement for down only**. Keep the down weight in ND
and skip the NZ29 cast for down (gate/up keep NZ); zero extra resident bytes, and the
MRS path needs no runtime conversion. Cost: the down-NZ benefit is lost for this path.
The adopted full-NZ S1 gain was about 2% p95, and the down-only share is **not
measured**; it must be measured as part of any service A/B. Rollback: flip the MLP-NZ
flag back and restart with the AllReduce path; no weight conversion is required at
runtime.

Alternatives if both formats must coexist: per-layer ND cache for only the active
layers (89.1 MB × active layers), on-demand NZ→ND cast per call (format-cast cost per
layer per step, untested and likely expensive), or proving the installed MRS kernel
accepts NZ directly (unproven). The choice stays open until 31 reads the installed
kernel/config and 32/20 review the budget.

Executor lifetime and `release`: the probe binding does not destroy the
`aclOpExecutor`, and the `release` op is unregistered. That is acceptable for a
one-shot probe but **not** for a long-running service; a production wrapper must
resolve executor destruction and per-call workspace ownership. The A15 experiment and
the sustainable production integration remain separate gates.

## 10. Non-goals

No change to the official `start.sh`/`inference.py` defaults, no metrics/CSV edits, no
new device plan in this design step, and no change to the frozen H4K code
(`0d3f7fb` / run_probe `508e2931` / window `d7e73514`).

## 12. A15 rows=127 contract confirmation (CPU-only, no new code)

The current freeze (`6d09515`, run_probe `d54584b7`) already expresses the requested
A15 small contract with parameters only:

```
--arm A15 --rows 127 --k 256 --n 512
flags = 8 (KEEP_ALL), bias = nullptr, native grouped run
padded_rows(127, 2) = 128
row_block(0, 127, 2) = (0, 64)   # RS output per rank: [64, 512]
row_block(1, 127, 2) = (64, 128)
all_gather_into_tensor -> [128, 512]; crop [:127]
reference = matmul(127, 256) then all_reduce -> [127, 512]
```

The plan-only output was verified locally: arm A15, flags 8, rows 127, k 256, n 512,
bias nullptr, status plan_not_executed. The pad row is tail row 127 (rank1's block);
the original 127 rows keep their positions and the crop removes only that pad row.
No code or freeze change is needed to stage this cell.

## 13. Production lifecycle and NZ baseline plan (CPU design, not implemented)

The single-cell experiment uses `KEEP_ALL` and still hits the unregistered `release`
op; that is a diagnostic configuration, not a production implementation.

- **Executor/workspace lifetime**: for production, call the aclnn two-phase API with
  `flags=0` (no keep), destroy the `aclOpExecutor` after the launch returns, and let
  the workspace tensor be owned per call or by a small pool. The probe's
  "executor is never destroyed" comment and the missing `release` op must not be
  carried into the service.
- **Stream ordering**: prepare captures the current stream; MRS launch, all_gather and
  AddRMSNorm must stay on that stream. Production must not call
  `torch.npu.synchronize()`; the probe sync is measurement only.
- **Collective ordering**: reduce-scatter inside MRS and the following all_gather use
  the same TP HCCL group; no barrier beyond the two collectives.
- **NZ baseline comparison**: the current MLP keeps FRACTAL_NZ29 weights. Keep down in
  ND for the MRS path (skip the down NZ cast only), keep gate/up NZ, and compare the
  service against the full-NZ baseline on the same workload: S1 mean/p50/p95, S2
  throughput at the recorded concurrency, and OpenDecisions/ERQA 2PP. Rollback is the
  MLP-NZ flag plus a restart; no runtime conversion.
- **Memory**: a per-layer ND down copy for all 64 layers would be 5.31 GiB/rank, so
  the default plan is load-time format choice, not dual resident copies.
- **Adoption gate**: any service adoption still requires the model-level 2PP gate; the
  tensor-level A14/A15 cells are feasibility only.

## 14. Proposed error-stat fields (not implemented; no rerun now)

If a later cell needs scale-aware equivalence, add to `result.reduced`:

- `reference_abs_max`, `output_abs_max`
- `max_rel` (max |diff| / (|reference| + tiny)) and `mean_rel`
- `ulp_at_max` (BF16 ulp at the reference magnitude near the max diff)
- `diff_count_gt_1ulp`, `element_count`
- optionally a small diff histogram

These are field proposals only; they are not part of the current freeze and do not
change the A14 large v3 record, which remains numeric-complete but not a precision
pass.

## 15. Bounded steady-state benchmark plan (CPU design only; no device authorization)

Compare A15 (MRS V1 + all_gather, ND weights) with the current BF16 NZ
matmul+all_reduce down path. Same seed, x/w hashes, TP2 group, stream ordering and
per-iteration synchronization; the native AddRMSNorm boundary is measured as a
separate optional stage, not mixed into the matmul comparison.

- **Baseline arm (design, not implemented)**: a minimal `A16` baseline that keeps the
  weight in FRACTAL_NZ29 (record the ND source hash and the NZ cast result hash) and
  calls the production-shaped `F.linear`/matmul followed by `dist.all_reduce` on the
  same stream. It retains x/weight/output and the NZ weight until after sync,
  matching A15's KEEP_ALL lifetime; flags are not used by the binding because A16
  does not call it. A15 uses ND; before any benchmark, verify ND/NZ numerical
  equivalence at the benchmark shape (earlier MLP NZ vs ND output was bit-identical,
  but it must be re-recorded here).
- **Shapes and padding**: primary comparison at M968 (the A15 967+1 pad) for both
  arms, so M is identical; also record a secondary baseline at M967 and quantify
  the one-row pad effect rather than mixing the two M values.
- **Order and timing**: ABBA per pair; W warmups excluded from statistics; N measured
  iterations; each timing interval covers launch + collective + sync; report per-stage
  and total median/p95 for both arms plus paired deltas. No single-run adoption claim.
- **Process model and KEEP_ALL budget**: choose either one process alternating arms
  or one fresh process per arm. The A15 probe leaks one executor/workspace per call
  because release is unregistered. The earlier "62 small / 9 large" estimate counted
  only the workspace; the code-path budget in 15.1 includes the retained pad input and
  RS output, giving about 16.50 MiB small and 131.25 MiB large per call. Count W
  warmups, all A15 calls, retries and auxiliary calls against the same 1 GiB/rank cap.
  If both arms run in one process they share the A15 budget; separate processes
  isolate it and reclaim at exit. Suggested bounded cells after 15.1: small W=3/N=20
  (23 calls, about 398 MiB retained peak) in one process; large **W=1/N=4 (5 calls,
  about 807 MiB peak including shared weight/input and transient)** per arm in
  independent processes. Large cannot fit W2/N6 (8 calls, about 1,050 MiB retained
  alone) once the pad input and RS output are counted.
- **Production lifecycle**: a long-running service still needs executor destruction
  and workspace pooling/ownership; using flags=0 and relying on early release is not
  a solution and is not measured here.
- **Reporting**: raw per-iteration durations and amplitudes stay with the run; any
  comparison remains a diagnostic of this bounded cell, not a production throughput
  or precision claim. No official entry/CSV change.

### 15.1 KEEP_ALL budget from the actual retained objects

Code path: `run_mrs` allocates `workspace = at::empty({bytes})` under the input
options (binding_probe.cpp:615) and, when `KEEP_ALL` is set, stores
`keep_x = x`, `keep_w = weight`, `keep_y = y`, `keep_workspace = workspace` and
`keep_lists` (binding_probe.cpp:618-621). The state stays in `g_states` because `id`
is assigned for any keep flag; the release op is unregistered, so every call retains
its own objects until process exit.

| Object (per rank) | Small M127->128, K256, N512 | Large M967->968, K8704, N5120 |
| --- | --- | --- |
| workspace | 17,170,432 B = 16.375 MiB | 115,818,496 B = 110.458 MiB |
| pad input `x_mrs` (new each call) | 128*256*2 = 65,536 B = 0.0625 MiB | 968*8704*2 = 16,850,944 B = 16.071 MiB |
| MRS output `y` (new each call) | 64*512*2 = 65,536 B = 0.0625 MiB | 484*5120*2 = 4,956,160 B = 4.726 MiB |
| descriptors/lists | small | small |
| **retained per call** | **17,301,504 B = 16.500 MiB** | **137,625,600 B = 131.255 MiB** |

Shared, not multiplied per call: the weight tensor passed to `run_mrs` (large
8704*5120*2 = 85.0 MiB) is the same allocation every iteration, and the original
`x_n` [967,8704] is about 16.05 MiB; both are held once. The executor pointer is also
leaked but its allocator size is unknown, so keep a 20% margin.

Transient per call (freed after the comparison, not cumulative): `full` [968,5120]
9.91 MiB, `reference` [967,5120] 9.91 MiB, and diff/stat temporaries of the same
order; peak transient is roughly 40-60 MiB and belongs in peak RSS, not in the
cumulative leak sum.

Capacity under 1 GiB/rank (1,073,741,824 B):

- Small: floor(1 GiB / 17,301,504) = 62 calls; with 20% margin 49 calls. W3/N20 = 23
  calls (about 398 MiB retained including the shared weight/input).
- Large: floor(1 GiB / 137,625,600) = 7 calls of leak alone; with 20% margin 6.
  W2/N6 = 8 calls is 1,050 MiB of retained objects and does not fit. W1/N4 = 5 retains
  656 MiB, plus shared weight/input about 101 MiB and transient about 50 MiB, for a
  peak near 807 MiB (about 79% of 1 GiB). W1/N5 = 6 retains 788 MiB and peaks near
  938 MiB (about 91%), too tight once allocator overhead is included.

Process model:

- **Small**: one process alternating A15/A16 is feasible (23 A15 calls, about 398 MiB
  retained plus transient/shared); ABBA pairs are inside that process.
- **Large**: use one fresh process per arm (A15 and A16 each W1/N4), alternating the
  process launches A/B/B/A; this isolates the A15 leak and lets process exit reclaim.
  If a shared process is required, cap at W1/N3 = 4 A15 calls (525 MiB retained plus
  shared/transient, about 676 MiB peak) and keep A16 in its own process.
- A16 is design-not-implemented, so the benchmark cannot start until the baseline arm
  exists and its NZ/ND equivalence is verified; no device benchmark is authorized by
  this section.
