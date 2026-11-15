# 提交入口 profile：MLP、TP 通信与 GDN 预算（2026-11-15）

在 GDN 决策专用路径接入 S1 默认启用之后，对**提交入口配置**做一次设备时间 profile，回答“剩下的时间花在哪”。结论：MLP 是最大单项，TP2 allreduce 与 GDN 同量级，attention 本身很小。

## 1. 采集条件

| 项 | 值 |
| --- | --- |
| 服务 | `manage_systemone_profile_service.py start --profile --tp 2 --max-model-len 4096 --max-num-seqs 1` |
| 模型 | `/workspace/yangchen/models/Clef-27B`，BF16，eager，seqs1，缓存关 |
| 优化 | `CLEF_GDN_DECISION_PREFILL=1`（本轮 GDN 决策专用路径开启） |
| 请求 | 固定三字段 `/v1/systemone`，307 tokens；10 预热 + 20 正式，并发 1 |
| 采集 | vLLM `--profiler-config`（torch）+ 昇腾 profiler；停止服务后离线 `analyse` 逐 rank 导出 |
| 分析 | `scripts/analyze_systemone_profile.py --strict`，`operations.csv` / `top20.csv` / `op_statistic.csv` |

**口径**：307-token 固定模板，与 2026-11-12 profile 可直接对照；真实 OpenDecisions 平均 446.8 tokens，MatMul/attention 会按 token 放大。profiler 本身有开销，绝对值只用于占比，不用于性能声明。占比是 `op_statistic` 的 kernel 时间求和；通信与计算可能重叠，20.4 ms 不等于关键路径贡献。

## 2. 每请求预算（rank0，20 请求均值）

| 类别 | ms/请求 | 占 kernel 时间 |
| --- | ---: | ---: |
| MLP up/gate（`307×5120 × 17408×5120`） | 21.34 | 21.0% |
| MLP down（`307×8704 × 5120×8704`） | 10.73 | 10.6% |
| **MLP 小计** | **32.07** | **31.6%** |
| **allreduce（TP2）** | **20.40** | **20.1%** |
| **GDN 主核**（ChunkGatedDeltaRuleFwd） | **10.52** | 10.4% |
| GDN q/k/v/state 转置（5 个 shape） | 6.73 | 6.6% |
| 其他 MatMul（attention 投影等） | 14.82 | 14.6% |
| AddRmsNorm | 2.80 | 2.8% |
| CausalConv1d | 2.29 | 2.3% |
| SwiGlu | 1.48 | 1.5% |
| Slice | 1.13 | 1.1% |
| FusedInferAttentionScore | 0.92 | 0.9% |
| LayerNorm | 0.79 | 0.8% |
| 合计 rank0 / rank1 | **101.6 / 102.6** | 100% |

### 2.1 MLP

up/gate 21.34 + down 10.73 = **32.07 ms/请求**，与 2026-11-12 profile 的 32.16 ms 一致。历史微复现显示该 kernel `aic_mte2_ratio≈0.994`、cube 利用率约 98.8%，即**权重搬运受限**，不是算力打满。选择性 W8A8 直接减少权重字节，是当前最明确的杠杆。

### 2.2 TP2 通信

`allreduceAicpuKernel` 20.40 ms、2600 次调用/20 请求（约 130 次/请求）。它与 GDN 主核同量级，且是否在关键路径取决于重叠；需要单独的 union/关键路径分析，不能直接按 20.1% 计入收益。

### 2.3 GDN

- 主核 10.52 ms（与历史 10.9 ms 一致，本轮未改主核）。
- 布局 6.73 ms：q/k 转置两个方向 3.67、v 两个方向 2.19、**state 转置 0.87**。
- 本轮优化去掉的是 final-state 导出/转置/scatter 与反向中间量；**初始状态 gather 与输入状态转置仍在执行**，对应 0.87 ms 的 state 转置。q/k/v 布局 5.86 ms 是下一步融合的目标。

### 2.4 attention

`FusedInferAttentionScore` 仅 0.92 ms；QKV/O 投影计入“其他 MatMul”的 14.82 ms。attention 本身不是瓶颈。

## 3. 对下一步的含义

按预算排序：

1. **MLP 选择性 W8A8**：对准 32.07 ms 的搬运受限部分；若权重字节减半且 kernel 受 MTE2 限制，量级上可回收约 15 ms/请求。
2. **融合 FFN**：SwiGlu 1.48 + AddRmsNorm 2.80（部分属于 MLP/attention 前后处理）约 4.3 ms 的合并空间。
3. **TP 通信**：20.40 ms；先做 union/关键路径分析，再决定 MC2、通信与计算重叠或调度层优化。
4. **GDN 剩余布局**：5.86 ms 的 q/k/v 转置；融合 QK 归一化与打包可回收其中一部分。

## 4. 证据

| 内容 | 路径 |
| --- | --- |
| 预算汇总（两 rank、按 op 与 shape） | `Clef-27B/logs/gdn-decision-20261115/profile-entry-20261115/budget.json` |
| 原始 op 统计 | `.../op-statistic-rank0.csv`、`op-statistic-rank1.csv` |
| 分析器全量算子表 | `.../operations.csv.gz`、`top20.csv` |
| profiler 服务启动记录 | `.../launch.json` |
| 服务恢复核验 | `.../restore-snapshot.json` |

远端原始目录：`/workspace/yangchen/logs/gdn-decision-20261115/profile-entry-20261115/`（含两个 rank 的 `ASCEND_PROFILER_OUTPUT`、trace 与 16 MB `summary.json`）。

## 5. 服务恢复

- 采集结束后停止 profiling 服务，按 phase-0 快照恢复原服务为 **PID 533380**；argv、cwd、环境白名单与 phase-0 逐项一致。
- `GET /health` 200；8/8 冻结用例逐位一致。
- 当前服务指针更新为 533380。
