# down_proj + TP AllReduce 的 MC2 原型（2026-11-16）

目标：把 MLP 的 down_proj 矩阵乘与 TP2 allreduce 融合，隐藏 20.4 ms 通信。结论：**当前 A3 / CANN 9.2.1 / torch_npu 2.10.0.post7 / eager 栈上不可用**，保留为负结果。

## 尝试 1：`torch_npu.npu_mm_all_reduce_base`

- 双 rank `torchrun` 原型：每个 rank 持有 down_proj 的 TP 分片 `[5120, 8704]`，输入 `[M, 8704]`；
- 基线：`F.linear + dist.all_reduce(SUM)`；
- MC2：`npu_mm_all_reduce_base(x, weight.T, hcom, reduce_op="sum")`；
- 结果：`aclnnMatmulAllReduce` 返回 **161001**：
  `SoC version ascend910_93 verification failed ... The operator package ... is not installed`；
- 检查 `opp/built-in/.../kernel/ascend910_93`：只有 `GroupedMatMulAllReduce`，没有 `MatmulAllReduce`。

## 尝试 2：`aclnnGroupedMatMulAllReduce`（单组）

CANN 9.2.1 为 `ascend910_93` 提供了 `aclnnGroupedMatMulAllReduce`，尝试用 aclnn ctypes 以“单组”方式调用：

- `aclCreateTensorList` 创建单元素 x/weight/y 列表；
- x `[M,8704]`、weight `[8704,5120]`、y `[M,5120]`；
- `splitItem` 0/1/2/3；bias `None`/零向量；groupList `None`/`[M]`；streamMode 0。
- 结果：
  - `bias=None, groupList=None, splitItem=0` → **SIGSEGV**；
  - `bias=zero, groupList=[M], splitItem=1` → **挂起**（collective 不返回），需终止进程。
- 说明该算子面向 MoE grouped matmul，不是 dense 单 matmul 的稳定接口；不能作为 down_proj MC2 的可用路径。

## 为什么没有别的直接替代

- `MLPRowParallelOp`（`mlp_tp_enable()`）使用 `reduce_scatter`，不是 allreduce；且 `finegrained_tp_config.mlp_tensor_parallel_size` 在当前配置中**只支持 graph mode**，服务为 eager。
- `npu_quant_mm_reduce_scatter` 是 sequence-parallel O-projection 的 reduce_scatter 路径，语义不是“标准 TP down_proj + allreduce”。
- `npu_mm_all_reduce_base` 在 A2/910B 可用，但 A3 的 `MatmulAllReduce` 内核未随当前 CANN 包提供。

## 结论

1. **MC2 down + allreduce 在当前栈上不采用**：不是模型或代码问题，而是 A3 算子包缺失/不兼容；继续调试单组 `GroupedMatMulAllReduce` 的收益不确定，且不属于受支持用法。
2. **可行替代**：
   - 等待/升级支持 `MatmulAllReduce` on `ascend910_93` 的 CANN/torch_npu；
   - 或使用 graph mode 的 MLP TP（`mlp_tensor_parallel_size=2`）走 reduce_scatter，但这会改变并行布局且当前服务是 eager；
   - 或自定义 Ascend C 融合 kernel（成本高）。
3. **本轮长序列优化应转向**：W8A8 gate/up + 平滑/按层选择（质量门槛），以及 BF16 NZ 的服务级集成（质量中性）。

## 证据

- 错误日志：`Clef-27B/logs/gdn-data-path-20261115/mc2/`（`npu-mm-all-reduce`、`grouped-segv`、`grouped-hang`）
- 原型：`runtime/mc2_down_probe.py`、`runtime/mc2_grouped_probe.py`
- 远端原始：`/workspace/yangchen/logs/gdn-data-path-20261115/mc2/`

服务已按 phase-0 快照恢复为 **PID 612539**：`argv`、`cwd`、白名单 env 与 phase-0 逐项一致，`/health` 200，冻结 8 例 **8/8**；`current-service.json` 已更新。恢复证据：`mc2/restore/`。

## 2026-11-16 补充：MatMulReduceScatter V2 可用

此前结论只覆盖 `npu_mm_all_reduce_base` / `aclnnGroupedMatMulAllReduce`。后续探针验证了另一条接口：

- `torch_npu.npu_mm_reduce_scatter_base(x, W.T, hcom, 2, comm_mode="aiv")` 在 A3、TP2、BF16、真实 down_proj `K=8704, N=5120` 上执行成功；
- 输出为本地 `[M/2, 5120]`，`all_gather_into_tensor` 可还原完整 `[M,5120]`；
- `fused+AllGather` 在 308/970 token 约 **1.04–1.06×**，1596 token **0.97×**；仅 reduce-scatter 有 **1.10–1.23×**，但下游必须保持 sequence-sharded；
- `comm_mode=None` 反而慢 0.68–0.69×，`aiv` 是关键；
- 输出与基线 `max_abs=0.03125`、`mean_abs≈7e-7`，不是逐位一致。

因此：

1. **“A3 没有 MC2”需要修正为**：A3 缺 `MatmulAllReduce`，但 `MatmulReduceScatterV2`（aiv）可用；
2. **RS+Gather 还不足以接入**：完整输出还原只有约 1.05×，长序列退化；要兑现单算子收益，需要跨层 Dense SP；
3. **当前不接入提交**，保留为后续 Dense SP 的接口证据。量化已被赛方禁止，本路线与量化无关。

详见[MatMulReduceScatter 探针](mc2-reduce-scatter-2026-11-16.md)。
