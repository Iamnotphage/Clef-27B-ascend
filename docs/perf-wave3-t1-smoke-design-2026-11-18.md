# T1 smoke 设计：AllReduce+Add+RMSNorm（2026-11-18）

> 状态：**设计完成，未编译、未上板、未获准启动**。本文只固定契约、门槛与审计项，避免开工后改判据。

## 0. 定位

- 这是 **plumbing smoke**：只验证「910_93 + CANN 9.2.1 上，eager 服务外能否把 HCCL KFC 的 AllReduce 塞进自定义算子、并完成 Add+RMSNorm」。
- **它的目的不是过 3% 端到端门槛。** 按 22 的估算，纯 `AR+Add+RMSNorm` 的天花板约 **2.5–2.6%**，过不了 3%，过不了是预期内的。
- 真正有量级的是第二段 `MatMul+AllReduce+Add+RMSNorm` 且 **N-tile 级 AR**（模型估算 6.2%）；两段共用同一 host 外壳，只换 `op_kernel`。

## 1. 契约（单算子，不接服务）

| 项 | 值 |
| --- | --- |
| 算子 | `ClefAllReduceAddRmsNorm`（临时名，vendor `clef_ar_rms_smoke`，独立目录） |
| 形态 | TP2（2 rank）、M=967、N=5120、BF16、`ReduceOp=SUM` |
| 输入 | `partial` `[M,N]`（每 rank 由同一全局 x 与各自 W 分片算出的 down_proj 输出）、`residual` `[M,N]`（两 rank 相同）、`gamma` `[N]`、`eps` |
| 输出 | `normed` `[M,N]` BF16、`rstd` `[M]` FP32 |
| 基准 | `npu_add_rms_norm(dist.all_reduce(partial, SUM), residual, gamma, eps)` |
| 对照形状 | M=967（KFC 128 对齐：4,951,040/128 = 38,680 ✓）与 M=1024（16,384 对齐对照） |
| 不包含 | matmul、服务集成、权重、KV/状态 |

## 2. 三道独立门（任一不过即判 T1-AR 不可行）

| 门 | 判据 |
| --- | --- |
| 1. KFC 运行包 | kernel 内 `Hccl::AllReduce` 返回 `handleId ≥ 0`、`Wait` 成功、结果为 2-rank 和；不能是 host 侧 `dist.all_reduce` |
| 2. 161001 | `Mc2CreateOpResCtx` 返回 SUCCESS；若返回非 SUCCESS 或运行时报 SoC verify/算子包缺失，与 MC2 同因，直接判不可行 |
| 3. 形状 | 967 与 1024 都过；若只有 1024 过、967 挂，说明被路由到 16384 对齐的 alg-template/superkernel 引擎 → 判不可行（除非能强制 AICPU KFC） |

**首个失败判定**：(a) `Mc2CreateOpResCtx` 非 SUCCESS；(b) `handleId < 0` 或 `Wait` 负值/超时 hang；(c) 与基准 `max_abs` 超过 1 ulp（BF16 最低 2 位）；(d) 不能强制非 mesh 算法且默认选到 `mesh_1d`（910_93 不支持）。

**通过后的性能门**（只对第二段有意义）：单次 AR < 161 µs × 2/3 = **107 µs**，且端到端可回收 ≥3%；此门不施加在 AR+norm 这一版。

## 3. 已知硬约束（来自 22 的只读核实）

- `mesh_1d` 仅 910_95 支持（`topo_match_1d.h:27`），910_93 必须用 AICPU KFC + NHR/ring；`libmc2_client.so` 中 `HcclAllocComResourceByTilingA5Mc2` 的 A5 后缀说明 CCU 资源是 A5 专用。
- 对齐随引擎而变：`aicpu_kfc_def.h:23` 的 `HCCL_MIN_SLICE_ALIGN=128`（KFC 路径）；`alg_template_base.h:32` 的 16384（alg-template/superkernel）。967×5120 只满足前者。
- 910_93 的 OPP kernel 目录无 hccl/mc2/rms 算子，不存在可直接调用的 `aclnnAllReduceAddRmsNorm`，必须自研。

## 3.1 编译期门（2026-11-18，已执行，零 NPU）

### 门 A：HCCL MC2 host args API —— **通过**

`/usr/local/Ascend/cann-9.2.1/include/hccl/hccl_mc2.h`（声明 `Hccl*` 前缀），实现由 `libhccl.so` 导出
（`libmc2_client.so` 导出的是旧命名 `Mc2*`，两者并存）。最小程序 `gate2`：

| 调用 | 返回 |
| --- | --- |
| `HcclKfcAllocOpArgs(&args)` | 0，指针有效 |
| `SetCount(4951040)` / `SetReduceType(0)` / `SetSrcDataType(11)` / `SetDstDataType(11)` / `SetAlgConfig("NHR")` | 均 0 |
| `HcclKfcFreeOpArgs` | 0 |

**引擎枚举更正**：`SetCommEngine` 只接受 **2（`COMM_ENGINE_AICPU`）与 4（`COMM_ENGINE_AIV`）**；
0（CPU）、1（CPU_TS）、3（AICPU_TS）全部返回 rc=5。22 方案里的「1 = AICPU」有误，AICPU 是 **2**。
（枚举见 `/usr/local/Ascend/cann-9.2.1/include/hcomm/hcomm_res_defs.h:108-112`。）

链接集合：`-L$H/lib64 -L$H/aarch64-linux/lib64 -lhccl -lhcomm -lascendcl -Wl,--allow-shlib-undefined`；
缺 `hcomm/ascendcl` 会出现 `aclrt*`/`Hcomm*` 未定义。

### 门 B：MC2 tiling API —— **未证实（需走官方 op 构建）**

- 头文件在 `asc/include/adv_api/hccl/hccl_tiling.h`（`AscendC::Mc2CcTilingConfig`，含 `GetTiling(Mc2InitTiling&)` / `GetTiling(Mc2CcTiling&)` 与 `SetCommEngine/SetAlgConfig/SetStepSize`）。
- 实现位于 `aarch64-linux/lib64/libtiling_api.a`；`optiling::TilingDef` 的依赖在 `aarch64-linux/lib64/device/lib64/libtilingdata_base.a`。
- 但用独立 `g++` 手动链接会失败：`Mc2CcTilingConfig` 构造函数未解析 + `TilingDefImpl::InitData` 为 hidden symbol 与 DSO 冲突。
- **判定：该接口只能通过官方 `npu_op_package` CMake 流程（op_host）验证**，不能用独立小程序替代。smoke 的构建必须走 `runtime/ascendc/clef_ar_add_rms_norm/build.sh`，并把 tiling 头包含进 op_host。

## 4. 构建与安装

- 复用 `runtime/ascendc/clef_gdn_conv_qkv/` 的 `build.sh` + `CMakeLists.txt` + `op_host`/`op_kernel` 布局，新建 `runtime/ascendc/clef_ar_rms_smoke/`，**不覆盖**已验收的 `Clef-27B/src/ascendc/vendors/{clef_conv_qkv,customize}`。
- 产物装到独立目录（如 `build_out/install-smoke/packages/vendors/clef_ar_rms_smoke`），测试时用 `ASCEND_CUSTOM_OPP_PATH` 指过去；服务默认环境不动。
- 编译前 `source scripts/activate_env.sh`；构建失败只影响 smoke 目录。

## 5. tiling 与同步审计清单（开工前逐项确认）

| 项 | 审计点 |
| --- | --- |
| 对齐 | M×N×2 字节是否 128 对齐；分核时每核 token 块是否按 128 字节对齐 |
| 通信域 | `Mc2CcTilingConfig` 的 group name 与训练/服务一致（TP2 组）；`SetStepSize`/`SetCommEngine` 明确选 AICPU KFC |
| 算法 | 强制非 mesh（`Mc2KfcOpArgsSetAlgConfig` 指定 NHR/ring），并断言选中的不是 `mesh_1d` |
| 生命周期 | `InitV2 → Prepare → Commit → Wait → Finalize` 每核每 tile 配对；不得跨迭代复用未 Finalize 的句柄 |
| 缓冲 | send/recv 缓冲是否同一块（in-place AllReduce）及其对 residual 读的影响；workspace 大小由 tiling 给出 |
| 同步 | `Wait` 之后的 Add+RMSNorm 是否真的等到数据；不同核之间的 reduce 结果可见性（GM 写回 + barrier） |
| 错误 | 每步返回值检查并写回 host 可读状态，避免 hang 被误判为超时 |
| 数值 | 与基准逐元素比较，记录 `max_abs`、`ulp`、是否逐位；BF16 累加顺序差异单独记录 |

## 6. 第二段（只有 smoke 通过才做）

`MatMul+AllReduce+Add+RMSNorm`，down_proj 形状 `X[M,8704] × W[8704,5120]`，**按 N 切 4 块（每块 1280）**做 tile 级 AR：

- 每块 AR ≈ 30.4 µs（fixed）+ 3.11 MB / 110.1 GB/s ≈ **58.6 µs**；
- 每块 GEMM ≈ 321/4 ≈ **80 µs**（967 token，需实测确认）；
- 前 3 块 AR 被下一块 GEMM 掩盖，只暴露末块 ≈ 58.6 µs；
- 对照现状 208 µs/层 → 省 ≈149 µs/层 → **≈9.6 ms/请求（6.2%）**。
- 仍需补测：M=967 时单块 GEMM 随 N 切分的墙钟曲线（N=1280/2560），确认 58.6 µs 真能藏在 80 µs 下。

## 7. 明确不声称的内容

- 本文所有收益都是模型推算，不是实测；smoke 通过也不代表端到端有收益。
- 不改默认、不改服务、不新建 PR；任何接入必须以同窗口交错 A/B、可回退开关、全量 OD+ERQA ≤2 pp 为前提。
- 若 smoke 失败，T1 以证据关闭，回到已交付的纯 host 优化结论。

## 8. 待办

1. 等用户对「是否投入 0.5–1 天做 smoke」给出明确授权；
2. ~~等 22 号的实现方案~~：2a/2 + 2b/2 已收到，本文已按它更新，并修正了 `SetCommEngine` 的取值（AICPU=2，不是 1）；
3. 授权后第一步是搭 `runtime/ascendc/clef_ar_add_rms_norm/` 的 op_host + op_kernel 骨架，用官方 `npu_op_package` 构建验证门 B（tiling），再做 2-rank comm（自建 `HcclCommInitRootInfo`，不依赖 torch.distributed）与 kernel 门 A/B。
