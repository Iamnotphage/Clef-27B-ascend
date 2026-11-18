# T1 smoke 设计：AllReduce+Add+RMSNorm（2026-11-18）

> 状态：**设计完成，未编译、未上板、未获准启动**。本文只固定契约、门槛与审计项，避免开工后改判据。

## 0. 定位

- 这是 **plumbing smoke**：只验证「910_93 + CANN 9.2.1 上，eager 服务外能否把 HCCL KFC 的 AllReduce 塞进自定义算子、并完成 Add+RMSNorm」。
- **3% 不是硬门槛。** 用户口径的硬要求是质量绝对变化 ≤2 pp；3% 只作为成本收益排序的参考，真实、稳定且质量合格的较小收益同样可采纳。
- 按 22 的粗预算，纯 `AR+Add+RMSNorm` 只重叠 Norm/可删的启动或中间读写，量级约 2.5–2.6%，本身大概率不划算；它存在的意义是打通 plumbing。
- 真正可能有量级的是第二段 `MatMul+AllReduce+Add+RMSNorm` 且 **N-tile 级 AR**，但该收益仍是待修正的条件估算，见 §6 与 §9。

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

**首个失败判定**：(a) `HcclCreateOpResCtx` 持续非 SUCCESS（参数原因需先排除，见门 C）；(b) `handleId < 0` 或 `Wait` 负值/超时 hang；(c) 与基准 `max_abs` 超过 1 ulp（BF16 最低 2 位）；(d) 选中的算法无法在日志中确认且行为异常。

**性能参考**（不是淘汰门）：单次 AR 显著低于现状 161 µs、且同口径完整链（GEMM→AR→Norm）确实变短，才值得继续；3% 只用于排序。

## 3. 已知硬约束（来自 22 的只读核实）

- **（已更正）** `topo_match_1d.h:27` 的「only 910_95」只约束 CCU/KFC adv_api 路径里的 **Mesh-1D CLOS 拓扑匹配器**，不能推广成「910_93 不能用任何含 mesh 的算法」。`libhccl_alg.so` 中存在大量 910_93 专用 executor（`AllReduceMeshAivFor91093Executor`、`AllReduceARSFor91093Executor`、`AllReduceFastDoubleRingFor91093Executor`、`AllReduceMidCountFor91093Executor`、`AllReduceOrderPreservedFor91093Executor` 等）。
- `libmc2_client.so` 的 `HcclAllocComResourceByTilingA5Mc2` 后缀说明 **CCU 资源分配**是 A5 专用，这一点仍需保留。
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

### 门 B：MC2 tiling API —— **通过（官方构建，链接级）**

**2026-11-18 实测**：在 `runtime/ascendc/clef_ar_add_rms_norm/` 搭官方 `npu_op_package` 脚手架（op_host 内真实调用
`AscendC::Mc2CcTilingConfig` 与 `GetTiling`，op_kernel 为占位实现），`bash build.sh` **构建成功**并产出
`custom_opp_ubuntu_aarch64.run`（247,090 B，sha256 `d8851ed7…c819`）：

- `cust_optiling` / `cust_opapi` / `cust_op_proto` 三个 target 全部 Built；
- 安装后的 `op_impl/.../op_tiling/liboptiling.so`（符号链接）内可检索到我们硬编码的
  `AllReduce=level0:ring` 与 `hccl_world_group` → tiling 代码确实进入产物；符号被 `-fvisibility=hidden` 隐藏属预期；
- `aic-ascend910_93-ops-info.json` 含 `ClefArAddRmsNorm`，kernel `.o` 已生成。

所需 CMake 适配仅两处（不改官方 target 结构）：`npu_op_code_gen(... COMPILE_OPTIONS -I<CANN>/asc/include -I<CANN>/aarch64-linux/include)`
与给三个 host target 加同名 `target_include_directories`。**运行时 GetTiling 返回码仍需在算子实际执行时验证**（下一步 kernel smoke）。

### （原）门 B 背景：为什么必须走官方构建

- 头文件在 `asc/include/adv_api/hccl/hccl_tiling.h`（`AscendC::Mc2CcTilingConfig`，含 `GetTiling(Mc2InitTiling&)` / `GetTiling(Mc2CcTiling&)` 与 `SetCommEngine/SetAlgConfig/SetStepSize`）。
- 实现位于 `aarch64-linux/lib64/libtiling_api.a`；`optiling::TilingDef` 的依赖在 `aarch64-linux/lib64/device/lib64/libtilingdata_base.a`。
- 但用独立 `g++` 手动链接会失败：`Mc2CcTilingConfig` 构造函数未解析 + `TilingDefImpl::InitData` 为 hidden symbol 与 DSO 冲突。
- **判定：该接口只能通过官方 `npu_op_package` CMake 流程（op_host）验证**，不能用独立小程序替代。smoke 的构建必须走 `runtime/ascendc/clef_ar_add_rms_norm/build.sh`，并把 tiling 头包含进 op_host。

### 门 C：通信域注入 —— **主机侧已打通，资源上下文待改参数**

**（已更正 2026-11-18）** 此前「libhccl.so 无实体符号 ⇒ 无公开 ABI」的推断作废。按 22 指出的类型来源
（`vllm_ascend/distributed/device_communicators/pyhccl_wrapper.py` 的 `HcclGetRootInfo` / `HcclCommInitRootInfo`
原型，`hcclUniqueId` = `char internal[4108]`），使用公共头 `aarch64-linux/include/hccl/hccl_comm.h` 实测：

| 步骤 | 结果 |
| --- | --- |
| `aclInit` / `aclrtSetDevice(0)`（需 `activate_env.sh` 的完整 CANN 运行时环境） | rc=0 / rc=0 |
| `HcclGetRootInfo(&root)` | rc=0 |
| `HcclCommInitRootInfo(2, &root, rank, &comm)` | **rc=0，两 rank 均拿到非空 comm** |
| `HcclKfcOpArgsSet*`（count/reduce/dtype/engine=2） | 全部 rc=0（引擎 2/4 可接受，0/1/3 rc=5） |
| `HcclCreateOpResCtx(comm, 2, opArgs, &ctx)`，algConfig 为空（自动选择） | **rc=1**，ctx=nil；日志 `[GetAlgInfo] invalid algConfig=[]` |
| 同上，`AllReduce=level0:ring` | **rc=0，ctx=0x12c081200000（两 rank 相同），PASS** |
| 同上，`AllReduce=level0:doublering` | **rc=0，ctx 非空，PASS** |
| 同上，`AllReduce=level0:fullmesh` | **rc=0，ctx 非空，PASS** |

- 结论：**公共 ABI 可用，2-rank comm 可自建；资源上下文在显式算法串下创建成功**。空串/自动选择被 `GetAlgInfo` 拒为 invalid，**只证明该参数组合未通过解析，不能据此判定 910_93 KFC 路径不可行**。
- 真实加载库（`dladdr`）：`HcclGetRootInfo` / `HcclCommInitRootInfo` ← `libhcomm.so`；`HcclKfcAllocOpArgs` / `HcclCreateOpResCtx` ← `libhccl.so`。
- HCCL 期望格式（`libhccl_fwk.so` 报错原文）：`level0:NA;level1:<algo>` 或 `<op>=level0:NA;level1:<algo>`；
  字面示例包含 `AllReduce=level0:ring`、`AllReduce=level0:doublering`、`AllReduce=level0:fullmesh`，
  词表为小写 `ring / doublering / fullmesh / pairwise / hierarchy / pipeline / star`，AP 级算法在 **level0**。
- 已执行顺序：空串 → `ring` → `doublering` → `fullmesh`；结果如上。**三种串被接受不等于三种 executor 都实际运行**：本次日志只有通用 `Executor*` 记录，未给出实际算法名，该项**待确认**（后续 kernel 侧以真实 AR 结果与日志再定）。
- 证据：`/workspace/yangchen/logs/perf-wave3-20261118/gate8/`（三种变体的 HCCL 日志目录 + 最后一次 fullmesh 的 rank0/rank1 stdout）。
- 证据分层：头文件存在（已核）；运行时报错串与字面示例（已核，仅证明词表）；**实际选中算法与拓扑尚未确认**。

### 证据分层约定（勿合并）

| 层 | 含义 | 当前状态 |
| --- | --- | --- |
| (a) 头文件存在 | `Hccl::AllReduce`（`hccl.h:77-78`）、`HcclKfcOpArgsSetAlgConfig`（`include/hccl/hccl_mc2.h:73`，**非 const char\***）、`Mc2CcTilingConfig`（`asc/include/adv_api/hccl/hccl_tiling.h:26-29`） | 已核 |
| (b) 运行时注册 | `libhccl_fwk.so` 报错串与示例、`libhccl_alg.so` 的 executor 名集合、已注册算法 DFX 行 | 仅证明词表/注册表存在，不证明本进程已成功创建资源上下文 |
| (c) 实际拓扑与算法 | 2 rank 单卡 node-internal，无 RDMA；最终由 HCCL 自动选择 | **未确认** |

## 4. 构建与安装

- 复用 `runtime/ascendc/clef_gdn_conv_qkv/` 的 `build.sh` + `CMakeLists.txt` + `op_host`/`op_kernel` 布局，新建 `runtime/ascendc/clef_ar_rms_smoke/`，**不覆盖**已验收的 `Clef-27B/src/ascendc/vendors/{clef_conv_qkv,customize}`。
- 产物装到独立目录（如 `build_out/install-smoke/packages/vendors/clef_ar_rms_smoke`），测试时用 `ASCEND_CUSTOM_OPP_PATH` 指过去；服务默认环境不动。
- 编译前 `source scripts/activate_env.sh`；构建失败只影响 smoke 目录。

## 5. tiling 与同步审计清单（开工前逐项确认）

| 项 | 审计点 |
| --- | --- |
| 对齐 | 每行 5120×2 = 10240 B = 80×128 B，按行切天然 128B 对齐；分核按 `(M+blockNum-1)/blockNum` 上取整并裁尾。**128/16384 两个常量的消费端与单位尚未证实**（`aicpu_kfc_def.h:23`、`alg_template_base.h:31-34`）：967 整块按元素或 BF16 字节都整除 128、不整除 16384；1024 两者都整除。须逐 task 检查 count/地址/stride；「仅 967 失败、1024 通过」**不能唯一归因于 16384 路由**。 |
| repeat | `AllReduce(..., uint64_t count, ..., uint8_t repeat=1)`（`aarch64-linux/asc/include/adv_api/hccl/hccl.h:77-78`）；`task[i]` 起址 = `sendBuf + count·sizeof(dtype)·i`（:67-71），**count 是每 task 元素数、每 task 等量**；repeat 上限 255，967 不可表达。整块 4,951,040 可按 repeat=4（1,237,760）或 repeat=8（618,880）等分，**算术上无须 padding**；运行时范围另验。「967 质数」只限制**等长整行分块**。 |
| 布局 | row-major `[967,5120]` 的 flat 四分跨行（241.75 行），**不会自动形成 `[967,1280]` 列块**；N-tile 需要 GEMM 直接产出 tile-major 连续缓冲，或显式 pack（pack 成本单列）。整行方案：不等长 242/242/242/241 各一次调用，或 pad 到 968 后 repeat=8×121 行；**pad 是候选布局选择，不是必需**。 |
| task 固定项 | 30.4 µs 是拟合截距，混有 host/API/task 成本，**不能当作 KFC 每 task 常数**，也不能断言 repeat 可免费摊薄；需在 C 组实测。 |
| 通信域 | `Mc2CcTilingConfig` 的 group name 与训练/服务一致（TP2 组）；`SetStepSize`/`SetCommEngine` 明确选 AICPU KFC |
| 算法 | `HcclKfcOpArgsSetAlgConfig` 用 HCCL 词表（小写、level0），先试空/nullptr 自动选择，再试 `AllReduce=level0:ring/doublering/fullmesh`；运行期用 HCCL 日志确认实际算法 |
| 生命周期 | `InitV2 → Prepare → Commit → Wait → Finalize` 每核每 tile 配对；不得跨迭代复用未 Finalize 的句柄 |
| 缓冲 | send/recv 缓冲是否同一块（in-place AllReduce）及其对 residual 读的影响；workspace 大小由 tiling 给出 |
| 同步 | `Wait` 之后的 Add+RMSNorm 是否真的等到数据；不同核之间的 reduce 结果可见性（GM 写回 + barrier） |
| 错误 | 每步返回值检查并写回 host 可读状态，避免 hang 被误判为超时 |
| 数值 | 与基准逐元素比较，记录 `max_abs`、`ulp`、是否逐位；BF16 累加顺序差异单独记录 |

## 6. 第二段（只有 smoke 通过才做）

`MatMul+AllReduce+Add+RMSNorm`，down_proj 形状 `X[M,8704] × W[8704,5120]`，**按 N 切 4 块（每块 1280）**做 tile 级 AR。

**以下为待修正的条件估算，不是收益预测**（23 号审计）：

- 单位先统一：每 rank 的 `[967,1280]` BF16 输出 = **2,475,520 B = 2.47552 MB**（2.36084 MiB），整块 `[967,5120]` = **9,902,080 B = 9.90208 MB**（9.44336 MiB）。（此前写的 3.11 MB 是把 MB/MiB 混用，已更正。）
- 载荷拟合的单位：纵轴是 **MiB**，slope = 0.009082167 ms/MiB ⇒ **115.454 GB/s（107.525 GiB/s）**；JSON 里的 110.105 是 MiB/ms。
- 单块 2.47552 MB 低于拟合下界（9.443 MiB）约 4 倍，**外推值 51.8 µs 不能当作 KFC tile 实测**；最小拟合点预测 116.2 µs vs 实测 124.9 µs，已有 8.7 µs 残差。
- 每块 GEMM ≈ 321/4 ≈ 80 µs 来自 NZ profile 的 MLP down 均值（约 320.9 µs/层），切 N 后 **kernel/tiling、Cube 波次、X 重复读取、W 布局、写回/pack、4 次 launch 都可能改变**，必须实测完整 GEMM 与单 N=1280/2560。
- 因此「前 3 块 AR 被下一块 GEMM 掩盖、只暴露末块」与 **≈9.6 ms（6.2%）** 都只是条件模型；仅按更正后的算术（(208.255−51.826) µs×64 ≈ 10.01 ms）也不能当收益。
- 正确比较口径：**同一口径的完整串行链（GEMM→AR→Add→RMSNorm）对比候选完整链（N-tile AR + 全 N 统计 + 输出尾部）**，逐项计入 pack/同步/launch/额外读。
- 仍需补测：M=967 时完整 GEMM 与单 N=1280/2560 的墙钟；新增的每-tile AR 真实延迟；RMSNorm 跨全部 N 的统计尾部。

## 7. 明确不声称的内容

- 本文所有收益都是模型推算，不是实测；smoke 通过也不代表端到端有收益。
- 不改默认、不改服务、不新建 PR；任何接入必须以同窗口交错 A/B、可回退开关、全量 OD+ERQA ≤2 pp 为前提。
- 若 smoke 失败，T1 以证据关闭，回到已交付的纯 host 优化结论。

## 8. 待办

1. 等用户对「是否投入 0.5–1 天做 smoke」给出明确授权；
2. ~~等 22 号的实现方案~~：2a/2 + 2b/2 已收到，本文已按它更新，并修正了 `SetCommEngine` 的取值（AICPU=2，不是 1）；
3. 授权后第一步是搭 `runtime/ascendc/clef_ar_add_rms_norm/` 的 op_host + op_kernel 骨架，用官方 `npu_op_package` 构建验证门 B（tiling），再做 2-rank comm（自建 `HcclCommInitRootInfo`，不依赖 torch.distributed）与 kernel 门 A/B。

## 9. 审计更正（2026-11-18，22/23 号只读核对）

### 9.1 数字口径

| 项 | 更正后 |
| --- | --- |
| 13.328 ms | 只是 **A−B**（`layer-serial-bound-v2.json`，M=967：A=0.667794840、B=0.459539937、A−B=0.208254904；×64）在**固定原 GEMM/Norm/计时方式**下的条件预算；不是普遍物理上界，也不能直接映射到全量 P95 收益 |
| 208−125 ≈ 83 µs | 不能唯一归因为同步；部分来自调度/launch，已含在 A−B 里，不能再同额加为收益 |
| “AR+norm ≈ 4 ms” | 只能标为**重叠 Norm / 可删启动或中间读写**的粗预算，且必须拆解覆盖范围：Wave 2 profile 的 `AddRmsNorm` 共 5.9298 ms / 128 次；只替换 64 个 MLP 链对应 Norm 时按均值为 **2.9649 ms**，尾部与必要工作仍在。若说覆盖 128 次必须另给证据 |
| 13.3 ms 与 4 ms | 覆盖范围有交集，**不能相加** |
| 9.6 ms / 6.2% | N-tile 融合的**待修正条件估算**：仅换个算术为 (208.255−51.826) µs×64≈10.01 ms，尚未扣 GEMM 膨胀 / pack / sync / RMSNorm 全 N 尾部 / 重复 task 固定项；不作为收益预测 |
| 单位 | 单块 `[967,1280]` = 2,475,520 B = **2.47552 MB**（2.36084 MiB）；整块 `[967,5120]` = 9,902,080 B = **9.90208 MB**（9.44336 MiB）；拟合纵轴是 MiB，slope 0.009082167 ms/MiB ⇒ **115.454 GB/s** |
| 拟合外推 | 2.47552 MB 低于拟合下界（9.443 MiB）约 4 倍，外推出的 51.8 µs **不能当 KFC tile 实测**；最小点已有 116.2 µs 预测 vs 124.9 µs 实测的 8.7 µs 残差 |

### 9.2 RMSNorm 依赖（23 号）

令 `P_i` 为 rank 局部 GEMM 输出、`U_i = AllReduce(P_i)`、`Z_i = U_i + residual_i`，
`q_i[row] = Σ_cols FP32(Z_i)²`，则 `rstd[row] = rsqrt((Σ_i q_i[row]) / 5120 + eps)`，`Y_i = Z_i · rstd · gamma_i`。

- 标准 RMSNorm **只需要 sumsq**，不需要普通 sum；只有语义变成 LayerNorm 才需要 sum。
- **平方必须在 AllReduce 完成后**，否则丢跨 rank 交叉项；**残差只在 AR 之后加一次**。
- TP2 下各 rank 拿到全部 N，统计归并**不需要再加一个跨 rank AR**；分核时仍需本卡依赖与可见性保证。
- 顺序：GEMM/Fixpipe 写回可见 → 对应 tile 发布/collective → Wait 并确保对消费核可见 → Add + partial sumsq → 该行四块 `q` 均 ready → 本卡跨 tile 统计归并/rstd → 最终输出。
- 前段 Add/q 可叠在后续 GEMM/AR 上；**rstd 与完整输出有尾部屏障**；slot 复用必须等 AR 及向量消费完成；`Z` 要留到最终乘 gamma（UB 容纳方案或 GM 落盘/重读成本需计入）。
- **已核对（只读）**：真实服务**需要 `residual_sum`**。`vllm_ascend/ops/layernorm.py:92-95`（bias 路径）与 `:112-114`（Gemma 路径）均为
  `x, _, residual = npu_add_rms_norm(...); return x, residual`，第三个输出被作为下一层的残差流继续传递；
  `vllm/model_executor/layers/layernorm.py:125-133` 的融合 add 路径同样保留残差。
  因此 T1 融合核必须同时产出 `normed` 与 `residual_sum`，不能只返回归一化结果。

### 9.3 收益比较与探针边界

- 正确比较是「**同口径串行完整链 − 候选完整链**」；均匀且资源独立时的递推为 `g + (p−1)·max(g,c) + c`，只有 `g ≥ c` 且 `Σg ≈ 完整 GEMM` 才退化为 `G_full + c`。
- 已测的 host 双流负结果**只覆盖其 torch/HCCL 调用方法**；设备内 producer-ready 的 N-tile 流水**仍未验证**，不能据此判定融合 N-tile 无效。
- `runtime/wave3_layer_serial_bound_probe.py` 的已知边界：各 rank 用不同随机 `residual[1,5120]`、无显式 NZ、异常时静默回退为分离 Add+Norm，且 JSON 未记录回退臂。**新对照必须**：同 rank 一致 `residual[M,N]`、实际 NZ 布局、显式记录所走算子/是否回退、完整链墙钟 + device timeline、两 rank 关键路径。

## 10. 可执行补测矩阵（23 号 A–E，待 gate8 恢复确认）

**执行依赖**：先解通信资源 ctx 与官方 op_host 构建；通过后用 A/B/C 收集实测并选分块；再做 D/E。只有同环境完整链出现稳定改善，才做服务级交错 A/B（OD/ERQA 绝对变化 ≤2 pp）；3% 仅作研发排序。

**通用测量规则**：预热 ≥5、稳态 20 次、A/B 交错、保存逐次值；**无 profiler 的完整墙钟用于核算收益，device timeline/profile 只用于解释重叠**；两个 rank 都记录并取关键路径；同一组内冷/热缓存状态一致。单例均值 ×64 或固定 967 样本不能替代全量请求 P95。

| 组 | 内容 | 关键记录 | 判定/用途 |
| --- | --- | --- | --- |
| **A 基线** | M=967 / K=8704 / N=5120，**实际服务 BF16 NZ 权重**、两 rank 相同 `residual[M,N]`，完整 `GEMM → AR → Add → RMSNorm` | 真实 kernel 名与布局、各阶段时间、是否发生异常回退（禁止静默 fallback） | 提供**同环境串行链分母**，后续所有候选与它比 |
| **B 计算** | M=967 / K=8704，N=5120/2560/1280；单块、累计 2/4 块总时与 ready 时间 | baseM/N/K、核数、波次数；权重预 pack 记为一次性；热路径 pack、输出重排、额外 launch 单列 | 确认 N 切分后 GEMM 是否近似线性；与 NZ profile 的 321 µs/层不可直接混比（那是 ND 热缓存口径） |
| **C 通信** | 独立连续输出 N=5120/2560/1280 各 1 组；相同整块总字节下比较 **4 个独立 call vs 1 个 repeat=4**（必要时 repeat=8） | 单 task 完成点、总时/尾部、两 rank 到达偏斜、实际选中算法；单次 wall 与 32-call 链式分列；M=1024 仅 1 组形状控制 | 判断小 payload 曲线、task 固定项、逐块 ready 能力；不把 30.4 µs 当每 task 常数 |
| **D 归一化** | 全 N `AddRMS` 参考 vs 4 块 `Add/FP32 sumsq` + 本卡归并 + rstd + 全输出 | partial 阶段与最终尾部分列；Y/rstd 误差分布；**必须核对 `residual_sum`**；覆盖跨 tile 边界、末行、buffer 轮换；两 rank 相同 residual、有限输入 | 证明 RMSNorm 语义与尾部代价；旧的 in-place SUM 链式 probe 会累积放大输入，**不作精度证据** |
| **E 完整流水** | p=2、p=4 两个候选对照 A | GEMM 写回 → AR 提交/完成 → q → rstd → 最终输出的时间线；扣除分块膨胀、搬运、同步；按两 rank 关键路径比较完整链 | 只有稳定改善才补 M=1595，并进入服务级交错 A/B 与 OD/ERQA ≤2 pp |

**不在本轮执行**：以上全部需要 NPU 与 gate8 恢复确认；当前只做本地文档与证据整理。

## 11. HCCL context 契约（2026-11-18，22 号随包源码核实）

> 更正：此前「`HcclCombineOpParam` 在任何头文件中都没有定义」的表述有误。定义在**实现头**而非 `include` 树：
> `/usr/local/Ascend/cann-9.2.1/aarch64-linux/asc/impl/adv_api/detail/hccl/common/hccl_inner_def.h`（arch=2201 走 `#else` 分支，字段含
> `workSpace / workSpaceSize / rankId / rankNum / winSize / windowsIn[] / windowsOut[] / res[8328] / multiFlag / data`）。

| 契约 | 源码依据 |
| --- | --- |
| `InitV2(context, initTiling)` 把 `context` 直接 `reinterpret_cast` 成 `__gm__ HcclCombineOpParam*`，随后只用 `hcclContext_->workSpace` 做 `InitInner(..., ONLINE_COMPILATION_TILING_VERSION)` | `impl/platform_v220/hccl_aicpu.h:146-157` |
| `tilingBaseAddr_ = reinterpret_cast<uint64_t>(initTiling)` | 同上 |
| `SetCcTilingV2(uint64_t offset)`：`opType = (Mc2CcTilingInner*)(tilingBaseAddr_ + offset)->opType`，即 **offset 相对 initTiling 的字节偏移**；要求 `curVersion_ == ONLINE_COMPILATION_TILING_VERSION`，**必须先 InitV2**；`opType < HCCL_CMD_ALL(21)` | `hccl_aicpu.h:182-195` |
| 已废弃的 `SetCcTiling(__gm__ void* ccOpTilingData)` 仍可用，语义是直接给 cc 块的 GM 指针 | `hccl_aicpu.h:160-180` |
| `offset` 由 host 侧 `Mc2CcTilingConfig::GetTiling` 写入 init 块的 `offset[]`（`Mc2InitTilingInner.mc2HcommCnt/offset[MAX_CC_TILING_NUM]`），**不要手工填**；两块在 device 侧必须连续 | `asc/include/adv_api/hccl/internal/hccl_tiling_msg.h:23-33`、`hccl_tilingdata.h:47-59` |
| 正确的契约校验：`InitV2` 后立即调 `GetRankId()` / `GetRankDim()`（公开 getter，分别读 `rankId` / `rankNum`），期望 `∈{0,1}` 与 `2`；**不要盲解引用结构体** | `hccl_aicpu.h:196-201` |
| 仍未证实：`HcclCreateOpResCtx` 返回的指针是否就是上述 device 缓冲；只能用 typed 运行（GetRankId/GetRankDim）判定 | 无静态结论 |

**对 kernel 的含义**：`InitV2` 的 context 必须来自框架注入（`MC2().HcclGroup("group")` + `GetHcclContext<HCCL_GROUP_ID_0>()`，见 vllm-ascend 的 MC2 参考算子）或我们自己把合法缓冲送进 device；`SetCcTilingV2` 的 offset 从 init 块的 `offset[]` 读取，不要猜。
