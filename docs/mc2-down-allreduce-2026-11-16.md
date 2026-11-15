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
