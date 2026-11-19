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

## 4. Bias and TP-rank semantics

- Each rank owns the K-slice `[K_local, N]` of the down weight and the matching
  activation columns; MRS sums the two partial matmuls, exactly the all-reduce sum.
- No bias is present in the current path (`bias=nullptr`). If the model config is
  `mlp_bias=True`, the bias is replicated: add it once after step 3, before the norm,
  on the full `[T, N]` tensor.
- Group: the service TP world is size 2, so the group name is the TP HCCL group name
  captured once at init (`get_hccl_comm_name(rank)`), not a per-call lookup.
- Stream: MRS captures the current NPU stream at prepare and launches on it; the
  all_gather must be issued on the same stream so ordering is preserved. No
  `torch.npu.synchronize()` in production; the probe sync is measurement only.

## 5. Padding, shapes, reproducibility

| T (tokens) | T' padded | MRS output per rank | full after all_gather |
| --- | --- | --- | --- |
| 307 | 308 | [154, 5120] | [308, 5120] |
| 967 | 968 | [484, 5120] | [968, 5120] |
| 1595 | 1596 | [798, 5120] | [1596, 5120] |

Zero rows produce zero partial rows and do not change any real row's sum; the slice
`[:T]` removes them. Reproducible probe shapes: `M=968, K=8704, N=5120` (the real
local shape; the earlier A14 large plan used exactly this), plus the small
`M=128, K=256, N=512` control. Seeds and input hashing follow run_probe.

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
   change) that reuses `run_mrs`, then `dist.all_gather_into_tensor`, and compares to a
   matmul+all_reduce reference; records max_abs/mean_abs/bit_equal and comm timing.
   Gate: `max_abs <= 0.03125`, `mean_abs <= 1e-4` (prior MC2 RS/Dense SP probes), no
   systematic bias.
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
  nullptr/zero-bias variant.
- Group-name retrieval from the vLLM TP group and stream ordering must be validated in
  the service before enabling the flag.
- This path does not explain the Grouped hang, does not establish the `mc2_server`
  dependency, and does not claim a throughput win. Grouped root-cause work and this
  equivalent substitution stay separate conclusions.

## 10. Non-goals

No change to the official `start.sh`/`inference.py` defaults, no metrics/CSV edits, no
new device plan in this design step, and no change to the frozen H4K code
(`0d3f7fb` / run_probe `508e2931` / window `d7e73514`).
