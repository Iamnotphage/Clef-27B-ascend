# 原生全融合 MatMul+AllReduce+Add+RMSNorm 存在性核查（只读，2026-11-19）

**范围**：仅头文件声明 / 库导出 / torch 注册三处只读核查；**未加载、未试跑**。
**不**由"头存在"推断本机 A3 支持、正确性或性能。

## 1) 真实存在的原生全融合入口（证据）
| 项 | 事实 |
| --- | --- |
| 头文件 | `.../cann-9.2.1/aarch64-linux/include/aclnnop/aclnn_matmul_all_reduce_add_rms_norm.h` |
| API | **`aclnnMatmulAllReduceAddRmsNormGetWorkspaceSize`** |
| 签名（实读 :53-57） | `(x1, x2, bias, residual, gamma, epsilon, group, reduceOp, commTurn, streamMode, y, normOut, workspaceSize, executor)` |
| 语义（注释） | "实现 MatmulAllReduce + AddRmsNorm 融合计算" |
| dtype 注释（:30） | `x1`: **float16, bfloat16** |
| 库导出 | **`libopapi.so` 导出 17 个 `MatmulAllReduceAddRmsNorm` 符号**（`nm -D --defined-only`），`libnnopbase.so` 无 |
| inplace 变体 | `aclnn_inplace_matmul_all_reduce_add_rms_norm.h` 亦存在（同族） |
| torch_npu 注册 | **无**（torch_npu 2.10 中 grep 不到 `MatmulAllReduceAddRmsNorm` / `mm_all_reduce_add_rms_norm` schema）⇒ 需自建 aclnn binding |

## 2) 关键限制：**官方已标记 deprecated**
两个头文件均含 `deprecated` 属性（各 2 处），原文（inplace 头 :45-50）：
> "...is scheduled to be deprecated in a post-December 2026 version update.
> Please migrate to **aclnnMatmulAllReduceGetWorkspaceSize** and **aclnnAddRmsNormGetWorkspaceSize**."

⇒ 官方方向是**拆开**用 `aclnnMatmulAllReduce` + `aclnnAddRmsNorm`，而不是用这个全融合入口。

## 3) 与已关闭方向的区别
- 旧已闭环题：**`aclnnMatmulAllReduce`**（仅 MatMul+AR，无 norm epilogue）与 MC2——
  **不等于**本入口的边界（本入口含 Add+RMSNorm epilogue）。
- 因此这是**不同 native 边界**，不属于"重复旧方向"。

## 4) 缺项（明示，未核）
1. **A3 / `ascend910_93` 是否在支持列表**：头文件未声明，**未查 op-info/支持矩阵**；
2. `bias` 是否可为空、`reduceOp`/`commTurn`/`streamMode` 的合法取值；
3. **是否支持 NZ 权重**（`x2` 的 format 约束未在头文件确认）；
4. 实际 kernel 实现与 tiling（未加载、未 profile）；
5. 正确性与性能：**零证据**，不声明收益。

## 5) 若后续要用（不由我实现）
最小可编译 binding 需：`aclnn_matmul_all_reduce_add_rms_norm.h` + `libopapi.so` 导出，
自定义 op 名注册，`group` 为真实 HCCL comm；**但 deprecated 属性意味着长期风险**，
与官方建议（拆成 `aclnnMatmulAllReduce` + `aclnnAddRmsNorm`）方向相反，需 20 决定是否投入。

## 6) 头与库 SHA
- 头：`include/aclnnop/aclnn_matmul_all_reduce_add_rms_norm.h`（SHA 未取，本次只读声明层面）
- 库：`aarch64-linux/lib64/libopapi.so`（SHA 未取）
> 若进入实现阶段，需补这两个文件的 SHA-256 作为证据基线。
