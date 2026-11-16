# MLP-only Dense SP 探针（2026-11-16）

Dense SP 的第一步是验证 MLP 的“序列切分 → AllGatherMatmul → SwiGLU → down ReduceScatter”是否比当前“全序列 MatMul + AllReduce”更快。结论：**数值等价可用，但 0.88–0.98×，比现有 AllReduce 慢**，因此不继续扩展 attention/GDN 的完整 Dense SP。

## 1. 配置

- A3 单物理卡两个逻辑 NPU，TP2，BF16；
- layer31 真实 MLP 权重：`H=5120`、`intermediate=17408`，每 rank `gate/up [17408,5120]`、`down [5120,8704]`；
- M=308/970/1596（由 307/969/1595 补齐到 2 的倍数）；
- 所有 rank 使用**相同的全局 x**，再按 rank 切本地 token；
- 基线：`gate/up` → `npu_swiglu` → `down` → `dist.all_reduce`；
- SP：`npu_all_gather_base_mm(x_local, gate_up.T, comm_mode="aiv")` → `npu_swiglu` → `npu_mm_reduce_scatter_base(act, down.T, comm_mode="aiv")`；
- 3 warmup + 10 次，rank device event 平均。

## 2. 数值

- `gathered_vs_x`：**逐位相等**（all-gather 的 token 顺序正确）；
- AllGatherMatmul 的 gate/up vs 基线：max_abs **0.015625**，mean_abs **≈2e-7**；
- SP 本地输出 vs 基线对应 token slice：max_abs **0.015625**，mean_abs **≈2e-5**；
- 差异来自 BF16 累加/融合顺序，不是布局错误。

> 第一次探针曾给每个 rank 生成不同随机 x，导致 mean_abs 0.46；修正为全 rank 相同 x 后得到上述结果。这个错误说明 SP 验证必须固定全局输入。

## 3. 性能

| M | rank | 基线 ms | SP ms | 加速 |
| ---: | ---: | ---: | ---: | ---: |
| 308 | 0 | 0.679 | 0.720 | **0.943×** |
| 308 | 1 | 0.691 | 0.718 | **0.962×** |
| 970 | 0 | 1.147 | 1.166 | **0.984×** |
| 970 | 1 | 1.110 | 1.166 | **0.952×** |
| 1596 | 0 | 1.681 | 1.914 | **0.878×** |
| 1596 | 1 | 1.679 | 1.912 | **0.878×** |

- 308/970 基本持平或略慢；
- 1596 明显变慢（0.878×）；
- 单层 MLP 的 AllGather + ReduceScatter 通信成本没有低于一次 AllReduce，融合算子也没有带来足够重叠；
- 全序列 MatMul 的计算量没有减少（Megatron 式 SP 主要切 norm/激活和通信，不是切 GEMM 计算）。

## 4. 判断

- **MLP-only SP 不采用**：数值虽然可接受，但单层就慢，完整 Dense SP 还要加 attention/GDN 的跨 token 语义、布局转换和 batching，收益更不确定；
- Dense SP 的价值需要跨层重叠/减少显存，单层探针不支持“能显著降 S1”的假设；
- 继续投入完整 Dense SP 的优先级应下调；当前 BF16 BIC + 合批仍是主线；
- 如果未来上游 vLLM-Ascend 的 Dense SP 已成熟，可直接在其实现上做端到端验证，而不是自己从 MLP 往上搭。

## 5. 证据与恢复

- 探针：`Clef-27B/logs/gdn-data-path-20261115/mlp-sp/probe2.json`、`probe2.log`
- 复现：`runtime/mlp_sp_probe.py`，`torchrun --standalone --nproc_per_node=2 ... --tokens 308,970,1596`
- 服务恢复：`.../restore-mlp-sp/`，baseline **PID 699006**，argv/cwd/env 与 phase-0 一致，`/health` 200，冻结 8 例 **8/8**
