# NPU GEMM 融合与通信重叠技术方案（2026-11-16）

## 0. 结论先行

- **最大的剩余性能杠杆是通信与 GEMM 的融合/重叠**：969 token 时通信 32.1 ms，其中约 **13.4 ms 未与计算重叠**；307 token 时 allreduce 20.4 ms，约占客户端 16%。
- **但不能靠“换一个 MC2 算子”拿到收益**：`MatMulReduceScatter + AllGather` 还原完整输出只有 1.04–1.06×（1596 token 0.97×），单层 MLP-only SP 实测 **0.88–0.98×**。
- 真正能兑现的是**跨层 Dense SP 布局**：residual/norm/激活保持 sequence-sharded，column-parallel 前 AllGatherMatmul，row-parallel 后 MatMulReduceScatter，让整个模型不再每层 AllReduce 回全序列。
- 该路线工作量大、风险高，且上游 `c6857b4` 已有 Qwen3.5 Dense SP 实现；**建议优先移植/验证上游实现，而不是从零写算子**。
- 若不做完整 Dense SP，剩余 BF16 优化只有 MLP tiling（2–4%）、GDN 边界融合（4–7%）、full attention 跳过 KV 写回（1–3%），不建议为此重写主 GEMM/attention 算子。

## 1. 约束与当前基线

| 项 | 值 |
| --- | --- |
| 硬件 | A3 单物理卡内 2 个逻辑 NPU，TP2 |
| 精度 | BF16；**赛方不允许量化**，W8A8/W8A16/INT8/FP8 全部排除 |
| 框架 | vLLM `24420b7a` + vLLM-Ascend `89eb8a5e`，eager |
| 数据路径 | BIC 默认开启；GDN 决策专用 prefill |
| S1 | 198.42 ms（P95，2,352 题） |
| S2 | B8 C16 16.464 dec/s（候选）；B16 C64 17.110 |
| 质量门槛 | OD 三项 + ERQA accuracy，相对 baseline 绝对波动 ≤2 PP |

### 1.1 S1 预算（969 token）

| 项 | ms |
| --- | ---: |
| 设备 compute | 144.5 |
| 通信 union | 32.1 |
| 计算/通信 overlap | 18.7 |
| 通信 nonoverlap | **13.4** |
| 非 kernel | 12.5 |

### 1.2 307 token kernel 预算

| 项 | ms | 占比 |
| --- | ---: | ---: |
| MLP | 32.07 | 25.4% |
| allreduce | 20.40 | 16.2% |
| GDN 主核 | 10.52 | 8.3% |
| GDN 布局 | 6.73 | 5.3% |
| 其他 MatMul | 14.82 | 11.7% |
| 其他 kernel | 17.10 | 13.5% |

## 2. 候选方案与实测结论

| 方案 | 内容 | 实测 | 判断 |
| --- | --- | --- | --- |
| A. AllReduce MC2 | `npu_mm_all_reduce_base` | A3 缺 `MatmulAllReduce`，161001 | 不可用 |
| B. MatMulReduceScatter | `npu_mm_reduce_scatter_base(aiv)` | 本地 RS 1.10–1.23×；`RS+AllGather` 1.04–1.06×（1596 0.97×） | 需要 sequence-sharded 下游 |
| C. MLP-only SP | AllGatherMatmul gate/up + down RS | **0.88–0.98×**，1596 0.878× | 不采用 |
| D. Full Dense SP | 跨层 sequence-sharded | 未实现；上游 c6857b4 有 Qwen3.5 实现 | 唯一可能兑现 RS 的路线，高风险 |
| E. GEMM tiling/NZ | BF16 MLP | 969 已近 Cube 300–320 TFLOPS，理论上限 3–4% | 低收益 |
| F. GDN 边界融合 | conv + QKV 布局 + L2Norm + g cast | 短序列 2×，长序列 0.13× | 需重写 tiling，潜在 4–7% |
| G. Full attention cache skip | 跳过 KV 写回 | 未做 | 1–3% |

## 3. 方案 D：Full Dense SP 技术路线

### 3.1 目标布局

Megatron 式 sequence parallel：

```text
输入 hidden: [M/TP, H]（本地 token 分片）
  ↓
RMSNorm / residual / activation：本地 [M/TP, H]
  ↓
column-parallel GEMM（gate/up、QKV、O?）：
  AllGatherMatmul 把 [M/TP, H] gather 到 [M, H]，再乘本地权重分片
  ↓
row-parallel GEMM（down、attention O）：
  本地 [M, H] partial → MatMulReduceScatter → [M/TP, H]
  ↓
下一层 residual/norm 继续在本地 token 分片
```

关键点：

- **不是**把 GEMM 的计算量切一半；GEMM 仍然按 TP 权重分片，SP 主要让 residual/norm/激活和 row-parallel 输出保持在本地 token；
- 通信从每层 AllReduce 变成 AllGather（column-parallel 前）+ ReduceScatter（row-parallel 后），并与 GEMM 融合；
- 只有全链路保持布局，才能避免“RS 后再 AllGather 还原”的 1.0× 回退。

### 3.2 逐层改造清单

1. **MLP**
   - gate/up：`npu_all_gather_base_mm(local_x, gate_up.T, hcom, 2, comm_mode="aiv")`；
   - SwiGLU：本地/全序列按布局；
   - down：`npu_mm_reduce_scatter_base(act, down.T, hcom, 2, comm_mode="aiv")`。

2. **Full attention**
   - QKV 投影：column-parallel，输入需要 AllGather；
   - attention：本地 token 的 Q 需要跨 rank 的 K/V；需要 sequence-parallel attention（all-gather K/V 或 ring/context parallel）；
   - O 投影：row-parallel，输出 ReduceScatter；
   - KV cache：本地 token 的 KV 写入与 block table 要按 SP 布局；BIC 的 cache skip 逻辑需重新验证。

3. **GDN（48 层）**
   - GDN 是递归状态，序列切分后需要处理 segment 边界、跨 rank 状态、conv state；
   - BIC 当前跳过 `initial_state`/`conv_states` 的前提是“完整无历史 prefill”；SP 后每个 rank 只有部分 token，不能直接跳过跨 rank 状态；
   - 这是 Dense SP 在 Clef 上最大的不确定性，必须先做单层/小模型验证。

4. **联合头**
   - head 的 `hidden_states` 来自完整序列最后 hidden；SP 后需要把各 rank 的 token 分片按顺序 AllGather 回完整序列，或者让 head 支持 sequence-sharded 输入；
   - head 设备只约 2 ms，优先 AllGather 回完整序列，避免改 head 语义。

### 3.3 集成方式

- 优先**移植上游 vLLM-Ascend `c6857b4` 的 Qwen3.5 Dense SP 实现**，而不是从零写；
- 我们固定的是 `89eb8a5e`，需要评估 c6857b4 与 BIC/GDN patch 的兼容性；不直接升级固定栈；
- 采用环境开关 `CLEF_GEMM_SP=1/0`，默认关闭；保留原 eager/BIC 路径；
- 所有 collective 使用同一个 HCCL 通信域；记录 `hcom`、world_size、每层布局；
- 质量回归先 8 例冻结，再全量 OD/ERQA；任何 2PP 失败立即回退。

### 3.4 决策门

| Gate | 条件 | 不通过动作 |
| --- | --- | --- |
| G1 | MLP-only SP 单层 ≥1.05× | 停止，不再做 attention/GDN |
| G2 | 4 层连续 SP 数值 max_abs ≤0.02、有限 | 停止 |
| G3 | 全模型 OD 三项 + ERQA 2PP | 回退 |
| G4 | S1 P95 改善 ≥3%，或 S2 改善 ≥5% | 不接入 |
| G5 | GDN/attention 无质量异常 | 回退 |

**注意**：当前 MLP-only SP 实测 0.88–0.98×，G1 已不通过；因此该路线在现有栈上**不应直接开工**，除非上游完整实现证明端到端有效。

## 4. 方案 F：GDN 边界融合（低风险替代）

GDN 布局 6.73 ms + 主核 10.52 ms；QKV norm 原型短序列 2×、长序列 0.13×。下一版应：

- stride-aware 加载，避免逐行 `ReduceSum`/标量 `Muls`；
- tile 级归约与广播，一次完成 Q/K 归一化和 BSND 写出；
- V 直接拷贝；
- 按长度分派：短序列走融合，长序列走 BIC；
- 目标 1–3 ms，覆盖 1,595/10,294 token 不退化。

这是比 Full Dense SP 更现实的 BF16 优化，但仍需 Ascend C 重写与全量质量验证。

## 5. 方案 E/G：GEMM tiling 与 attention cache skip

- MLP：969 token 已约 256–299 TFLOPS，纯 matmul 上限 300–320；SwiGLU/启动 <1%，BF16 tiling 理论上限 3–4%；
- full attention：先做 `reshape_and_cache` 跳过/延迟写回的等价性验证，目标 1–3 ms；
- 不建议自写 attention/GEMM 主核，收益低于工程量。

## 6. 建议执行顺序

1. **不做 Full Dense SP 从零实现**；先核对上游 c6857b4 的 Dense SP 是否覆盖 GDN/attention，评估移植成本。
2. 若移植：先 MLP-only 端到端（不是单层），通过 G1–G3 后再扩 attention/GDN。
3. 并行推进低风险项：GDN 边界融合长度分派、full attention cache skip。
4. S2 继续用 B8 C16（16.464 dec/s，2PP）；B16 C64 仅在无 p95 约束时使用。
5. 任何方案先跑 8 例冻结 + 全量 OD/ERQA，再谈性能。

## 7. 目标与预期

| 方向 | 预期 | 风险 |
| --- | ---: | --- |
| 通信重叠（完整 Dense SP） | S1 5–10%（理论 13.4 ms 非重叠通信） | 高；单层 SP 已负 |
| GDN 边界融合 | 1–3 ms | 中；长序列退化风险 |
| full attention cache skip | 1–3 ms | 低–中 |
| MLP BF16 tiling/NZ | 2–4% | 低；服务级收益可能被淹没 |
| 量化 | ~10%（MLP） | **禁止**，不讨论 |

S1 170–180 ms 的目标在当前证据下**没有可靠路径**；通信全重叠的理论下限约 185 ms，仍需完整 Dense SP 且 GDN/attention 不退化。当前更稳妥的提交是 S1 198.42 + B8 C16 S2。

## 8. 证据

- 通信/SP 探针：`docs/mc2-reduce-scatter-2026-11-16.md`、`docs/dense-sp-mlp-probe-2026-11-16.md`
- profile：`docs/profile-residual-2026-11-16.md`
- MLP 上限：`docs/mlp-cube-ceiling-2026-11-15.md`
- GDN 原型：`docs/gdn-data-path-2026-11-15.md`
- 合批：`docs/batching-seqs8-16-2026-11-16.md`

## 9. 同层分块 AllReduce 探针补充（2026-11-16）

在计划之外又实现了“不改布局”的 down_proj 分块重叠：

- 把 `[M,K]` 按 token 切成 2/4 块，compute stream 算下一块，comm stream 对上一块做 async AllReduce；
- 结果：308 token 0.42×/0.29×，970 token 0.80×/0.46×，1596 token 0.79×/0.69×；
- 正确性 `max_abs=0.03125`、`mean_abs≈7e-7`；
- 失败原因：HCCL 小 AllReduce 固定开销重复、流/事件同步开销，同层没有足够计算可重叠。

结论：**同层 GEMM+AllReduce 分块也不成立**；这进一步支持“只有 Full Dense SP 的跨层重叠才可能兑现通信收益”。详见[分块重叠探针](npu-gemm-overlap-probe-2026-11-16.md)。

## 10. Full attention prefill cache skip 补充（2026-11-16）

在 `PrefillNoCache` 下跳过 `reshape_and_cache`（pooling、无 decode、无 KV sharing/producer）：

- 冻结 8/8 逐位一致；
- 307 token：skip mean 129.09 vs control 125.06（+3.2%，更慢）；
- 969 token：skip 170.85 vs control 172.51（−1.0% mean、−0.6% p95，噪声内）；
- 未达到 ≥3% 决策门，**不采用**，源码已恢复。

结论：cache 写入是语义死代码但不是性能瓶颈；full attention 的剩余空间不在这个单点。详见[Preprint cache skip 报告](attn-prefill-cache-skip-2026-11-16.md)。
