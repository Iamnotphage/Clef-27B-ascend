# Norm 验证证据（2026-11-18）

## 1. 接受准则（先明示，再对照）

| 输出 | 准则 | 依据 |
| --- | --- | --- |
| `residual_sum` | `max_abs ≤ 0.01` | 与原生逐元素比较；BF16 输出粒度在 |z|≈1 时为 0.0039（2⁻⁸ 相对） |
| `rstd` | `max_abs ≤ 1e-3` | FP32 输出，参考量级 1–2；1e-3 为 0.1% 相对 |
| `Y` | `max_abs ≤ 0.02` 且 `max_rel ≤ 1%` | BF16 相对粒度 2⁻⁸=0.39%，2 lsb≈0.78%；1% 覆盖 2 lsb |

- 这是**冒烟级数值准则**，不是质量门槛；正式质量仍以全量 OpenDecisions + ERQA 的 2 pp 为唯一验收。
- 撤回此前「逐位一致 / 约 2 ulp 已验证」的措辞：只保留实测的 `max_abs`；本轮未做逐位与逐对 ULP 计数。

## 2. 受控复测（2026-11-18 12:17 CST，唯一变量：gamma 描述符改为 `[N]`）

- 命令：`./wave3_norm_launcher 967 5120 /workspace/yangchen/logs/perf-wave3-20261118/norm-native`
- 身份：launcher SHA-256 `6c027992…`（本轮），kernel `.o` SHA-256 `71e16663…`，实际加载 vendor
  `build_out/install-norm6/packages/vendors/clef_ar_add_rms_norm`（经 `ASCEND_CUSTOM_OPP_PATH`）；
  原始退出码 **RAW_RC=0**（native 判据）。
- 描述符（launcher 自报）：`x[967,5120] str[5120,1] bf16`；`gamma[5120] str[1] bf16`；`out/residual_out[967,5120]`；`rstd[967] fp32 str[1]`。
- 结果：

| 统计 | 值 |
| --- | ---: |
| `residual_sum` max_abs | **0** |
| `rstd` max_abs | 4.77e-7 |
| `Y` max_abs | 0.015625 |
| `Y` 差异分位 p50 / p90 / p99 / p99.9 / max | 0 / 0.00195 / 0.00781 / 0.00781 / 0.015625 |
| `Y` max_rel | 0.00781（0.78%） |
| `Y` 绝对差 >0.01 的元素 | 403 / 4,951,040（0.0081%） |

判读：三项均在上述冒烟准则内；`Y` 的 p90 以下为 0 或 1 个 BF16 lsb，尾部 0.78% 相对差对应 2 lsb 量级，来源是
`z*rstd*gamma` 的运算顺序/舍入不同，属可解释的数值差异而非布局错误。

## 3. 历史三段记录（缺项如实标注）

| 阶段 | 时间（CST） | 身份 | 结果 | 记录状态 |
| --- | --- | --- | --- | --- |
| 早期 TBuf 无屏障 | 11:44 前后 | kernel 无 `PipeBarrier`，install-norm3/4 | `max_abs` 1e33 级垃圾、部分 block 未写 | 原始 stdout 未保存 |
| 旧 launcher 混淆 | 11:59 前后 | 重编前后的 launcher 混用 | 曾读出 `max_abs_Y=2.17`、首元素为 0 | 原始 stdout 未保存；重编后同输入恢复 0.0156 |
| 重建成功（本记录） | 12:17 | launcher `6c027992…` + kernel `71e16663…` | 见 §2 | 完整日志 `/workspace/yangchen/logs/perf-wave3-20261118/norm-native-run/run2.log` |

- 三段之间不合并归因：第一段是 kernel 同步/写回问题（Barrier + 8-float 对齐批次 + 独立 reduce work 修复）；
  第二段是二进制混用；第三段是当前锁定身份。
- 早期 CPU 参考口径：**BF16 输入的 CPU 高精度参考**（输入量化到 BF16 后以 double 累加 sumsq）；它与 native
  的输入字节是否跨阶段相同**不能默认成立**，仅在同一受控目录（`norm-native/`）下成立。

## 4. 证据路径

| 内容 | 路径 |
| --- | --- |
| native 输入与三输出 | `/workspace/yangchen/logs/perf-wave3-20261118/norm-native/`（x/r/g/y/res 各 9,902,080 B、gamma 10,240 B、rstd 3,868 B、meta.txt） |
| 受控 stdout + 哈希 | `/workspace/yangchen/logs/perf-wave3-20261118/norm-native-run/run2.log` |
| launcher 源码 | `runtime/wave3_norm_launcher.cpp` |
| native dump 脚本 | `runtime/wave3_norm_native_dump.py` |
| kernel | `runtime/ascendc/clef_ar_add_rms_norm/op_kernel/clef_ar_add_rms_norm.cpp` |

注意：gate-B 的 `d8851ed7…` 是 11:30 的**脚手架包**，不能代表本次数值构建；本次 kernel 身份以 `71e16663…` 为准。

## 5. KFC AllReduce 融合链实测（2026-11-18 12:40 CST）

**结论：eager 注入链打通、AR 返回成功，但归约结果没有落到提供的 GM 缓冲上；数值未通过。**

### 5.1 构建与接线

- op def：`Attr("group").String()` + `MC2().HcclGroup("group")`；tiling 用真实 group 构造 `Mc2CcTilingConfig`；
  `mc2InitTiling`(64 B) / `mc2CcTiling`(512 B) 连续置于 tiling 结构最前，kernel 用 `offsetof` 作为 `SetCcTilingV2` 的偏移。
- kernel：`InitV2(GetHcclContext<HCCL_GROUP_ID_0>(), &tiling->mc2InitTiling)` → `SetCcTilingV2(offsetof(...))` →
  `AllReduce<true>(x, recv, M*N, BFP16, SUM, 1)` → `Wait` → `Finalize` → Add+RMSNorm；AR 阶段 `blockDim=1`。
- eager 侧：自研 torch adapter（直接调用生成的 aclnn + `OpCommand.SetCustomHandler`），group 取自真实
  `ProcessGroupHCCL` 的 `get_hccl_comm_name(rank)`，实测值为 `group_name_0`。
- 身份：package `84d9cc18…`（255,016 B）、kernel `.o` `926c6e8a…`、adapter `.so` `781f07a4…`。

### 5.2 逐步结果（两 rank 均保存）

| 运行 | 结果 |
| --- | --- |
| run6 | 在 `GetWorkspaceSize` 内 **SIGSEGV**：根因是 group attr 取法错误（`GetAttrPointer<char*>` 解引用）；按固定范本改为 `GetAttrPointer<char>` 修复 |
| run7 | 原始 rc=0；两 rank `GetWorkspaceSize st=0/ws=0/exec≠null`、`OpCommand.Run` 返回；三输出产出，但 `max_abs_y=4.09`、`residual=1.61`、`rstd=0.787` |
| run9/10 | 增加独立 `diag` 输出：`[rankId=0, rankDim=2, rc=0, useHccl=1]`，**上下文有效且 AR 报成功** |
| run11 | 改为 out-of-place（workspace 作 recvBuf）：`ours_res = r`（workspace 为 0），仍无对端贡献 |

### 5.3 现象与解释边界

- in-place：`z = x_local + r`（无对端项）；out-of-place：`z = 0 + r`（workspace 未被写入）。
- 参考（同进程 `dist.all_reduce` + `npu_add_rms_norm`）在相同输入上正确：`total = x0 + x1`。
- 因此不是上下文缺失、不是 shape/类型错误，而是**归约数据没有出现在我们提供的 GM 地址**。
- 候选解释（未证实，按证据区分）：KFC `AllReduce` 的 send/recv 可能需要落在**通信窗口**（`GetWindowsInAddr/OutAddr`）
  或由 tiling 指定的窗口内，而不是任意 workspace；也可能需要额外的窗口注册/可见性步骤。
- **尚未做性能测量**，也没有任何质量结论；该链当前只到 “AR+Add+RMSNorm，不含 MatMul”。

### 5.4 通过（2026-11-18 12:56 CST）

在 AR 之后补两处修复后，**两 rank 全链数值通过**：

- 修复 1：`AllReduce` 改为 out-of-place（workspace 作 recvBuf），并在 `Wait` 后把 norm 的输入重绑到 workspace；
- 修复 2：AR 后对目标缓冲做 `DataCacheCleanAndInvalid<bfloat16_t, ENTIRE_DATA_CACHE, CACHELINE_OUT>`，
  解决读侧看到 AR 前旧值（in-place 表现为本地原值、out-of-place 表现为全 0）；
- 修复 3（Y 专项）：gamma 的 `DataCopy(gl, g_[0], n_)` 与 `Cast(gf, gl, ...)` 之间补 `PipeBarrier<PIPE_ALL>()`，
  此前 gamma 前若干行被读成未落地的数据，表现为 `out[2k]=0`、`out[2k+1]≈2·ref[k]`。

实测（M=967、N=5120、eps=1e-6、repeat=1、group=`group_name_0`，两 rank）：

| 输出 | max_abs（自研 vs 同进程 `dist.all_reduce`+`npu_add_rms_norm`） |
| --- | ---: |
| `residual_sum` | **0** |
| `rstd` | **2.38e-7** |
| `Y` | **0.0078125**（`Y>0.01` 元素数 **0**/4,951,040） |

- 身份：package `499e5302…`（255,286 B）、kernel `.o` `ac5e96ba…`；diag 输出显示 `rankId=0, rankDim=2, rc=0, useHccl=1`。
- 该链当前为 **AR+Add+RMSNorm（不含 MatMul）**；尚未做任何性能测量或质量声明。
- 服务在测试后已恢复（PID 1019618，health/文本/图文均通过）。

### 5.5 下一步

1. 只读核对 adv_api `Hccl` 对 sendBuf/recvBuf 的窗口要求与 `GetWindowsInAddr/OutAddr` 语义；
2. 若需要窗口，调整 tiling/kernel 让 AR 目标落在窗口内后重跑同一两 rank 测试（判据：`residual_sum` 与
   `dist.all_reduce` 参考一致）；
3. 通过后再做同条件完整链计时与 A–E；服务在阶段结束时已恢复（health/文本/图文均通过）。

## 6. AR 融合链同条件计时（2026-11-18 13:05 CST）

- 构建：package `499e5302…` / kernel `.o` `ac5e96ba…`（与 §5.4 通过的构建一致），adapter 增加按尺寸复用的 workspace 缓存。
- 方法：同一进程、同一输入/布局/预热（5）/迭代（20）、无 profiler；三臂顺序执行并各自记录逐次墙钟与 device event；
  两 rank 各写 `rank{0,1}.json`。
- 正确性（rank 各异有限输入：rank0 偏移 0、rank1 偏移 +0.1）：两 rank 均通过 —— `Y max_abs 0.0078125`、
  `residual_sum 0`、`rstd 2.38e-7`、`finite`、`diag rankId=0/1 & rankDim=2 & rc=0`。
- 计时（中位，ms；括号为 device event 中位）：

| 臂 | rank0 墙钟 | rank1 墙钟 |
| --- | ---: | ---: |
| copy-only（基线需要的输入恢复拷贝） | 0.200（0.040） | 0.275（0.034） |
| baseline：copy + `dist.all_reduce` + `npu_add_rms_norm` | 0.656（0.520） | 0.685（0.405） |
| candidate：融合 AR+Add+RMSNorm（**单 block 原型**） | **2.660（2.502）** | **2.656（2.451）** |

- 结论：**该单 block 原型实测慢**（约 4×），主因是 norm 阶段只使用 1 个 AIV 核（AR 阶段按契约也只有 core 0 可发起）。
  这是原型实现的限制，**不据此关闭 T1**。
- 成本对称性说明：baseline 含一次 `[967,5120]` BF16 拷贝（约 9.9 MB，copy-only 臂量化约 0.2–0.28 ms 墙钟）；
  candidate 不含该拷贝，但内部有 workspace 与 aclnn/OpCommand 调度开销。两者都包含同一次 2-rank AllReduce。
- 下一步：block0 发起 AR，其余核以 GM flag 同步后并行执行 norm（多核原型），再以同一方法复测；之后补
  NZ GEMM N=5120/2560/1280 与完整 MatMul 覆盖，最后才做服务级 A/B 与 OD/ERQA ≤2 pp。
- 本形状单例计时**不能外推**到 S1 p95。

## 7. 多核原型（block0 AR + GM flag 同步，2026-11-18 13:19 CST）

- 改动：block0 独占 `AllReduce/Wait/Finalize` 并在 workspace 尾部写完成标志；其余核轮询该标志（每条读前 DCCI），
  随后所有核按连续行区间并行执行 norm。tiling 恢复使用全部 AIV 核。
- 正确性：两 rank 仍全部通过（`Y 0.0078125`、`residual 0`、`rstd 2.38e-7`、`finite`、`diag rankId=0/1 & rankDim=2`）。
- 计时（中位，ms；括号为 device event 中位；warmup 5 / 20 次）：

| 臂 | rank0 墙钟 | rank1 墙钟 |
| --- | ---: | ---: |
| copy-only | 0.142（0.037） | 0.169（0.030） |
| baseline：copy + `dist.all_reduce` + `npu_add_rms_norm` | 0.465（0.352） | 0.461（0.352） |
| **candidate（多核）** | **0.457（0.356）** | **0.460（0.353）** |

- 结论：多核原型把候选从 2.66 ms 降到 **0.457–0.460 ms**，与基线持平（差异在噪声内）。
  由于基线还额外包含一次约 0.14–0.17 ms 的输入恢复拷贝，候选在**同口径下约省掉这次拷贝**，但净收益在
  AR+Add+RMSNorm 子链上仍接近于零 —— 真正的收益必须来自把 MatMul 一起融合（去掉中间写回/读回并让 AR 与
  GEMM 重叠），即 A–E 的 E 组。
- 仍未做服务级或质量结论；单形状数据不外推 S1 P95。

## 8. 计时口径修正与输入就绪主对照（2026-11-18 13:35 CST）

按 23 的审计要求修正：

- **主对照（input-ready）**：两臂的输入恢复 `copy_` 与完成同步都放在**计时区外**；计时区内 baseline 只做
  `dist.all_reduce + npu_add_rms_norm`，candidate 只做融合算子。AB/BA **逐轮交错**。
- **辅助臂**：两臂各含一次对称恢复拷贝（用于对照“含准备”的口径），保留但不用于净收益结论。
- 原始行保存 `run_id / pair_id / arm / iter / rank / wall_ms / event_ms`，脚本与 adapter SHA 写入 JSON；
  copy-only 仅作诊断，**不做减法**；`max(local duration)` 只是慢 rank 代理，不是严格 makespan。

主对照结果（中位，warmup 5 / 20 次，两 rank 均 `numeric_pass=true`）：

按 (run_id, pair_id, iter, arm) 取两 rank 的较大耗时为慢 rank 代理（**不是严格 makespan**，缺跨 rank 共同起止时刻）：

| 统计 | wall | event |
| --- | ---: | ---: |
| baseline 慢-rank 中位 | 0.459440 ms | 0.367230 ms |
| candidate 慢-rank 中位 | 0.496970 ms | 0.408060 ms |
| **paired delta 的中位**（逐对 candidate−baseline） | **+0.044170 ms** | **+0.039910 ms** |
| candidate 更慢的配对 | 16/20 | 16/20 |

- 注意区分：上表是「差的中位」，与「中位之差」（0.037530 / 0.040830）不是同一统计量；20 对全部保留。

- 辅助臂（两侧都含拷贝）：baseline 0.5098、candidate 0.5150 ms（交错）。
- **结论：在该实现下，融合子链比 native 路径慢约 8–11%**（墙钟与 device 同向）。这是当前 kernel 实现的结果，
  不是 T1 整体的结论；收益只能从更大的融合范围（MatMul）或其他路径取得，需由 B/E 实测判定。

## 9. B 组：真实权重 NZ GEMM 分块实测（2026-11-18 13:44 CST）

- 来源：真实 checkpoint `model.language_model.layers.0.mlp.{gate_proj,down_proj}.weight`，按 TP2 切片
  （gate/up per rank `[8704,5120]`、down per rank `[5120,8704]`），`npu_format_cast(..., 29)` 转 NZ；
  M=967，warmup 5 / 20 次，device event 中位。
- pack 为一次性成本（如 gate/up 全量约 2.8 s），服务中权重已是 NZ，**不计入热路径**。

| 形状 | 完整 N | N=4352/2560 两块（每块） | N=2176/1280 四块（每块） | 四块累计 vs 完整 |
| --- | ---: | ---: | ---: | ---: |
| gate/up（K=5120, N=8704） | **0.2976 ms** | 0.1568 / 0.1520 | 0.0813（四块一致） | 0.3252（**+9.3%**） |
| down（K=8704, N=5120） | **0.2963 ms** | 0.1655 / 0.1668 | 0.0866–0.0875 | 0.3484（**+17.6%**） |

- 单块 GEMM device 中位：**gate/up N=2176 为 0.0811–0.0813 ms；down N=1280 为 0.0866–0.0875 ms**（两者不可混写）。
- 四块数值是**四个独立中位数之和**：down 0.348310 ms（较 full +0.052020/+17.56%）、gate/up 0.324850 ms
  （较 full +0.027250/+9.16%）。JSON 未保存四块连续链的逐次值，因此**累计四块链尚未实测**，不能称“已测四块链耗时”。
- **撤回**此前“E 收益 ≈6.2 ms / p95 3.9% / 预计 2–4%”的估算：0.296（新 GEMM device）、0.208（旧 wall 聚合差）、
  0.046（旧 profile 平均）与 0.052（旧大 payload 拟合外推）混合了计时/输入/窗口，不能相加为同条件 baseline 或预测 P95。
- 每块 AR（2.47552 MB）的旧拟合外推 0.052 ms **低于拟合范围 4 倍**，必须由 C 组小 payload 实测替代；E 的可行性
  以 C 与 E 完整链同口径双 rank 实测为准。本表只提供分块 GEMM 成本。
- 局限：单层（layer 0）、单 rank、单形状；未含 pack 热路径与写回布局开销；跨层差异未测。

## 10. 排队候选：GDN 输出 Norm 的 BLOCK_M 单变量探针（22 号方案，未执行）

- 范围：只改 `vllm_ascend/ops/triton/layernorm_gated.py` 的 `BLOCK_M`；接口、rstd 保存、out 分配、grid 推导、
  `num_warps` 全部保持不变。不做 rstd 删除、缓冲复用或联合扫描。
- 契约（现行唯一口径）：`x/z` BF16 `[23208,128]`、`weight[128]`、`group_size=None → 128`、`ngroups=1`、M=23208、N=128；
  四臂 BM16/32/64/128 的 grid 分别为 1451/726/363/182，padding 8/24/24/88 行。**不使用** 967 行、groups24 或旧 padding 论据。
- 探针：取真实调用点的一份 x/z/weight（四臂共用）；按真实每请求模式以 **48 次连续调用**为一组；四臂 **交错 ≥3 轮**；
  每轮记录 48 次总 device 时间、单次均值/最大值、相对 BM64 的逐元素比较（逐位一致是目标而非保证）。
- 判据：仅当某臂在 ≥3 轮交错中稳定优于 BM64，且全量 S1 p95 不回归、OD/ERQA ≤2 pp，才进入下一步；四臂噪声内则关闭该方向。
- 版本与 patch 身份：固定 vLLM-Ascend `89eb8a5e`；改动前后文件 SHA-256、一行 diff 原文；Triton Ascend 3.6.0.post2 /
  CANN 9.2.1 / torch_npu 2.10.0.post7；**每臂实际解析到的 num_warps/num_stages** 必须从编译产物读出；eps 取模型实际值；
  实施时按仓库惯例做成带 `--check` 的 `runtime/apply_*.py` 安装器。
- 状态：**排队，待 E 组阶段完成后由 20/22 择时执行**；不占当前 NPU 窗口。

## 11. C 组部分结果与可复现阻塞（2026-11-18 13:58 CST）

**已获得**（分进程单臂，N=1280，与 on 臂同构建）：

| 臂 | rank0 墙钟（device） | rank1 墙钟（device） |
| --- | ---: | ---: |
| AR + Add + RMSNorm（use_hccl=1） | 0.4026（0.3141）ms | 0.4021（0.3132）ms |

- 测法变化：因同进程混跑两模式触发 AICPU 异常，C 改为**分进程单臂**；两臂中位差**不能**当作纯 AR 单 task 时长，
  也与原 AB/BA 交错条件不同。

**阻塞（可复现，日志已保留）**：

1. ON 臂在 N=1280 成功后，N=2560 触发 **AIV 故障**：`aivec error`，core id 4/22/36（两 rank 不同），
   `error code = 0`，vec error info 两 rank 不同；发生在 norm/轮询路径（ON/OFF 两臂均出现，**与 HCCL 未必相关**）。
2. 同一窗口另见 **KFC 参数故障**：`ProcLogicCqReport ... task exception`、`execute kernel param invalid`、
   `retCode=0x2a`、`kernel_type=TS_AICPU_KERNEL_AICPU_KFC(5)`、`argsSize=852`；host 侧 `GetWorkspaceSize` 与
   `OpCommand.Run` 均返回 0，错误在 `synchronize` 暴露。
3. 两类故障互相独立；失败时具体 N 无日志标记（已给 adapter 增加 `N`/`use_hccl` 打印，下个窗口生效）。

**下一步最小判别**（未执行）：

1. 用新标记单 N 单次跑 OFF 臂，仅 N=1280：若也挂 → 与 shape 无关；若过 → 故障在更大 N。
2. 每臂独立 `ASCEND_PROCESS_LOG_PATH`，抓第一条 `ProcLogicCqReport` 与 `[AICPU_INFO]` 段。
3. 核对 AIV core id（4/22/36）与实际 block dim 是否匹配；按官方 AI Core Error 文档解 `vec error info`。
4. 确认 `use_hccl=0` 是否仍走 KFC ctx（当前未证实）。

**E 组状态**：E 的 cube GEMM 实现依赖上述 kernel 缺陷先定位；C 的完整 2/4 块/尾部与 ready 粒度仍未测。

## 12. C 判别结果：OFF 臂仍触发 KFC，且是 host 注入（2026-11-18 14:38 CST）

单 shape（N=1280）、单 mode（`use_hccl=0`）、单次调用，独立日志目录，两 rank 均失败于**首个 device 任务**：

- `[adapter] before GetWorkspaceSize group=group_name_0 N=1280 use_hccl=0`（两 rank）
- `an exception occurred during AICPU execution, stream_id:13, task_id:3, errcode:1, msg:execute kernel param invalid`
  （chipId 7 / dieId 0 与 1，serial 105 / 9）
- `RAW_RC=1`

**结论（有证据）**：

1. `use_hccl=0` 在 device 分支里跳过了 `InitV2/AllReduce`，但 **host 侧仍提交了 `RunAicpuKfcSrvLaunch`**，且参数块被拒。
   这与静态证据一致：`NnopbaseSetMc2/SetHcomGroup` 与 `RunAicpuKfcSrvLaunch` 都在 host 侧库，launch 由 opdef 的
   `MC2().HcclGroup("group")` 声明触发，device 分支无法关闭。
2. 因此 OFF 臂不是“无 KFC 对照”，之前把 ON/OFF 当作 HCCL 开/关两臂的做法**作废**；C 的两臂差异无法用该开关获得。
3. `task_id=3`（首个 device 任务）即失败，说明失败发生在 KFC server 启动的参数校验，不是后续 AR 数据路径。
4. 早前 ON N=1280 打印的 “成功” 只是 host 调用返回；KFC 任务异步失败后在 `synchronize` 暴露——两者并不矛盾。

**含义**：自定义 Op 只要声明 `MC2().HcclGroup`，框架就会注入带参数的 KFC 启动；要让这条路径可用，必须按框架契约
提供合法的 MC2 参数块（当前 tiling 的 `mc2InitTiling/mc2CcTiling` + group 未满足），或改用不声明 MC2 的路径（则拿不到 context）。
这是 E 组实现前必须解决的前置：**不是 kernel 语法问题，而是 MC2 参数契约问题**。

## 13. C 判别二：有效 tiling + `use_hccl=0` → 挂起（2026-11-18 14:44 CST）

最小修复假设（保留两臂一致且有效的 MC2 tiling，仅 device 跳过 AllReduce）实测：

- 改动：`TilingFunc` 不再按 `use_hccl` 跳过 `Mc2CcTilingConfig::GetTiling`，两臂都生成有效 init/cc tiling；
  device 分支仍按 `useHccl` 跳过 `InitV2/AllReduce/Finalize`。
- 运行：单 shape N=1280、单 mode OFF、单次；两 rank 均打印到
  `[adapter] after GetWorkspaceSize st=0 ws=2475648` 与 `[adapter] after OpCommand.Run`，随后**挂起**，
  无 `RAW_RC`，无 `param invalid`。按有限超时以 SIGTERM 终止（日志内 `got signal: 15`）。
- 对比：改动前 OFF 为**立即** `execute kernel param invalid`；改动后变为**等待**。

**判读（待验证假设，非根因）**：host 侧注入的 KFC 服务在等待 device 侧 `InitV2/Finalize` 配对；
仅跳过 AllReduce 而跳过整个生命周期会让服务悬挂。下一步应在 OFF 臂里只跳过“数据归约调用”，
仍执行 `InitV2` + `SetCcTilingV2` + `Finalize`（不调用 `AllReduce`/`Wait`），观察是否能正常结束。

**服务**：测试进程已按超时清理；默认服务已恢复，`/health` ready，文本 od-1534 → score、图文 erqa-0000 → choice 均通过。

## 14. E 组端到端与 Norm 归因（2026-11-18 20:22 CST）

### 14.1 链路打通

`bb300298`（End + waitIterateAll + 接线去 `usedCoreNum` 门控）首次让 MatMul+AR+AddRMS 在设备上完整执行：
两 rank `RAW_RC=0`，diag `mmRc=0`、`usedCoreNum=1`、`baseM=128`、`depthA1=16`。

### 14.2 原始测量（只记录，不作验收）

| 臂 | 配置 | 实测 |
| --- | --- | --- |
| A（rank-local GEMM） | `use_hccl=2`、residual=0、rank-local 参考 | max_abs **0.015625**，>0.1 计 0 |
| B（AR 后） | `use_hccl=1`、residual=0 | max_abs **0.03125**，>0.1 计 0 |
| 完整链 | `use_hccl=1`、真实 r/γ | residual_sum 0.03125、rstd 6.14e-6、Y 0.0625、Y>0.02 计 116 |
| 同 partial Norm（normx） | `norm_from_x=1`、`mm_enable=0`、`use_hccl=2`，x=`[967,5120]`、dummy w `[5120,5120]` | residual_out **0.0**、rstd 1.19e-7/1.79e-7、Y 0.0625、>0.02 计 98/75 |
| 同 partial Norm（normsp） | 同上，方差改整行单次 ReduceSum | residual_out **0.0**、rstd 1.79e-7/2.38e-7、Y 0.0625、>0.02 计 98/75 |

### 14.3 已排除与未排除

- 已排除（对 Y 尾部）：GEMM、AllReduce、方差求和树、线性通路 z 的 BF16 化（`residual_out` 精确为 0）。
- **未**唯一锁定"最后那次乘法"：`rstd` 的绝对差（1e-7 级）在 BF16 边界上同样可能翻转，不能把 `|δY|/(rstd·γ)` 之类反推当证明。
- 完整链 `residual_sum 0.03125`（>0.01）**仍然存在**；norm-only 臂的 `res=0` 不可外推为整个 E 数值门闭合。

### 14.4 参考语义（SDK 直接证据，只读）

`opp/built-in/op_impl/ai_core/tbe/impl/ops_nn/ascendc/add_rms_norm/add_rms_norm.h:201-244` 的 BF16 Compute：
`Muls(x_fp32, rstd)` → `Cast_RINT` 到 BF16 → `Cast_NONE` 回 FP32 → gamma 转 FP32 → `Mul` → `CAST_RINT` 到 BF16。
即**归一化后、乘 gamma 前有一次 BF16 round-trip**；"全程 FP32 最后只舍入一次"不是该分支语义。
边界：入口 `add_rms_norm.cpp:39-41/63-65/69-70` 存在 key 30/33/34 等多分支，实际运行分支尚未闭合，
因此这只是**有据的最小改动方向**，不是已证实根因。

### 14.5 口径

- 未宣布任何数值门、质量或收益结论；A/B 臂与 Norm 臂数字一律标 descriptive。
- 修复后须同时复核同 partial Norm 格与**完整链三输出**，两者都过才谈数值门；之后才切 NZ 与同条件 ABBA 计时。

## 15. 完整 E 链数值判据（2026-11-18 20:49 CST 定稿）

### 15.1 有范围的判据版本（2026-11-18 21:28 CST 整段修正）

| 范围 | 判据 |
| --- | --- |
| 同 partial AddRMS 隔离 smoke | `residual max_abs ≤ 0.01`、`rstd ≤ 1e-3`、`Y max_abs ≤ 0.02 且 max_rel ≤ 1%` |
| **完整 E 的 GEMM / AR 后 / residual** | 逐元素 `|cand − ref| ≤ 1e-5 + 0.016·|ref|`（显式 `rtol=0.016, atol=1e-5`，来源 `torch/testing/_comparison.py:54-57` 的 BF16 默认值） |
| **完整 E 的 Y** | 仍用 `max_abs ≤ 0.02` **且** `max_rel ≤ 1%`（**不并入** mixed 门） |
| **完整 E 的 rstd** | 仍用 `max_abs ≤ 1e-3`（**不并入** mixed 门） |

- mixed 门**只**用于 GEMM 输出 / AR 后输出 / residual；Y 与 rstd 的 BF16 统计仅作**辅助描述**，单列报告，不能替代各自的门。
- `Y max_rel` 定义：`|Δ| / max(|ref|, 1e-5)`；`|ref| < 1e-5` 的元素**仅从相对统计中排除**，其 abs/finite/shape/dtype 检查**仍然全纳**。
- 这是**本轮研发准则选择**，不是厂家 GEMM 误差保证，也不等于服务 2 pp。

### 15.2 旧 run（seed 20261119）的记录更正

- 该 run 的参考是 **FP32 SUM**，按约定的 **BF16 SUM 协议并未在该 run 上执行**；78 项只作**当时观测**，既非新协议的 pass 也非 fail。
- 当时只保存了 `fail_count` 与 top3，**没有完整索引集合**；因此此前"residual 与 Y 的失败索引完全相同、故 Y 由 z 驱动"的表述**撤回**，因果不作断言。
- 该 run 的 `finite` 仅检查了 candidate，**未检查 reference**；`shape/dtype` 检查当时也**未实现**，不得追写为"已测"。
- 近零相对 mask 仅用于相对统计描述；任何点都**未从** abs/finite 检查中移除。

### 15.3 结论边界

- 旧 run 的 78 项是**旧 FP32 SUM 参考下的观测**（seed 20261119），BF16 SUM 协议**未在该 run 执行**；它既不是新协议的通过，也不是新协议的失败，需在新协议下重跑才有结论。
- 同 partial Norm 格：**绝对误差项**满足（`Y>0.02` 为 0、`residual_out` 精确 0、`rstd` 1e-7 级）；**`max_rel ≤ 1%` 未记录，隔离门未闭合**。
- 未宣布任何收益或质量结论。

## 16. Gate v2 新 run（2026-11-18 21:05 CST，脚本 SHA 02d03a18）

### 16.1 协议变更（先记后跑）

- 总 gate 参考改为 **BF16 MatMul + BF16 SUM**（`refb_*`）；`ref`/`refb` 两处 AR 均受 `use_hccl` 控制；
- 判据：逐元素 `|cand − ref| ≤ 1e-5 + 0.016·|ref|`（BF16 默认，显式传参）；
- `Y` 另跑严格门：`max_abs ≤ 0.02` **且** `max_rel ≤ 0.01`，`max_rel = |Δ| / max(|ref|, 1e-5)`（`|ref| < 1e-5` 不计入相对统计，但全部元素仍进入 abs/finite/shape/dtype 检查）；
- 失败集输出**完整索引**（不再截断 200）；记录 `shape_ok`、`dtype_ok`、`cand_finite`、`ref_finite`。

### 16.2 结果（`ndrt_2ee1c9ce`，ND，seed **20261118**，script `02d03a18…`，`RAW_RC=0`）

| 输出 | 失败数 | max_abs | 备注 |
| --- | ---: | ---: | --- |
| `residual_sum` | 68 | 0.03125 | max_ratio 见原始日志 |
| `Y`（BF16 门） | 68 | 0.015625 | — |
| `Y`（严格门） | — | 0.015625 | **`max_rel = 1.0` → 严格门 fail** |
| `rstd` | **0** | 6.08e-6 | — |

- `shape_ok`、`dtype_ok`、`cand_finite`、`ref_finite` 两 rank 均为 **true**。
- **失败集合不同**：residual 与 Y 各 68 项，交集 60，各自另有 8 项
  （residual-only：974922, 1664739, 1742133, 2113619, 2137955, 2355728, 2364236, 2899646；
  Y-only：926323, 1208484, 1738104, 2878969, 3024019, 3098558, 3456082, 3722519）。
  即"residual 与 Y 失败集合相同"的此前观察**被本 run 否定**，归因继续未定。
- 旧 run 的 78 项：实际是 **FP32 SUM 参考 + seed 20261119**；按约定的 **BF16 SUM 协议在该 run 上并未执行**，因此它既不是新协议下的 pass，也不是新协议下的 fail，与本 run 不做比较、不追认。

### 16.3 待补

- 逐点 `actual / ref / tol / ratio` 落盘（当前日志含完整索引与 top3，逐点值待导出）；
- 预固定 seed 20261119 的同协议确认格；
- 同 seed 的 rank-local 诊断格（AR 关闭，参考同步关闭）。


## 17. 并集导出与字段来源说明（2026-11-18 21:13 CST，script `83679100`）

- 导出文件：`e-full-rt-rank{0,1}-union76.json`（各 40,023 B），并集 **76 点**（residual 68 ∪ Y 68，交集 60）。
- **字段来源必须标注**：
  - `residual.actual/ref`、`y.actual/ref`、`rstd.cand/ref`、`gamma`、`residual_input` —— 均为本次 run 实际输出/参考，可直接使用；
  - **`partial.cand/ref` 是由已舍入的 BF16 z 反推（`z − residual`），不是真实 GEMM/AR 前或 AR 后 partial**。BF16 舍入不可逆，因此该字段**不得**用于证明上游误差或抵消机制，只作占位。
- 口径更正：此前"所有失败点都是相消、差额恒定、BF16 内不可能满足"的推断**撤回**——实测 `|ref|` 跨 0.0029–0.89、`abs_diff` 有 4 种取值、Y 的 diff 有 37 种，不支持该概括。
- 真实 partial 只能来自 `residual=0` 的 probe 臂，且必须与该臂的输入/hash/AR 开关绑定，不能与本 run 混用。

### 17.1 交给 22 的 CPU FP64 诊断所需数据（尚未采集）

| 组 | 内容 | 状态 |
| --- | --- | --- |
| A 点级 | `x_row[row][0..K-1]`、`w_col_logical[col][0..K-1]` 的 **BF16 uint16 位模式**；真实 `partial`（每 rank）、`residual_in`、`gamma`、`eps` | 未采集 |
| B 行级 | 去重 row 的 `z_cand_row`/`z_ref_row`（各 5120 个 BF16 位模式）、`rstd_cand/ref` | 未采集 |
| C 控制点 | 4 个未失败 idx，字段同 A+B | 未采集 |

- 位模式（`"3f80"` 形式）而非十进制，避免参考自身引入截断误差；只针对现有失败坐标与去重行，不导出整模型、不扩形状矩阵。

## 18. 受控 A 臂（AR off、r=0）与字节导出（2026-11-18 21:21 CST）

### 18.1 运行身份（实际值，日志头为静态旧值，以本节为准）

| 项 | 值 |
| --- | --- |
| 包 | `pkgs/clef_matmul_ar_add_rms_norm_ndrt_2ee1c9cec900c160.run`，sha256 `2ee1c9ce…` |
| 脚本（运行时真实） | `5269aa0db4d2bb6f05870b4aa2df3f53c144a86fcf5efa92e2888ae5963c8c8e`（**日志 header 静态写 `def11e5f`，以本行为准**） |
| 实参 | `mm_enable=1, mm_dim=1, use_hccl=2, probe_partial(r=0), seed=20261118`，双方 AR off |
| 启动 | UTC `13:21:22`，`RAW_RC=0`；运行前已清空旧 8 个 PID（进程前后证据保留）。当时的端口检查读的是 `:40BA`（= **16570**，**错读端口**，见 §19）；16666 的正确键是 `:411A` |
| 来源缺口 | **运行中的完整 PID 树快照未落盘**，不能靠 `exclusive_run` 之类声明补证 |

### 18.2 结果

- 覆盖：**原完整链 union76 + 4 控制点 = 80 点 / 73 行 / 79 列**；两 rank 每点行列均存在，每行/列 8704 个 4 位 uint16 hex；`assert_all_points_covered=true`。
- **A 臂自身失败数 = 0**（`fail_idx` 空），即 rank-local GEMM 在固定 BF16 门下**通过**。
- dump：`e-A-bytes-rank{0,1}-bytes.json`（各约 10.6 MB），本体 SHA256 rank0 `3629c841…`、rank1 `725e55b5…`；
  `x` 完整 hash 两 rank 同为 `220586f0…`，`W` 分别 `1fd0e09d…` / `12539641…`。

### 18.3 归因边界（2026-11-18 21:26 CST 更正）

**撤回**此前"GEMM 已排除""完整链 68 点失败只能来自 AR 或 partial+residual 舍入"的表述。更正后的事实：

- A 臂 `RAW_RC=0`，全元素逐点 `badp` 计数为 **0**（统计范围 = 该臂自身全部元素，不只是 80 个采样点）；这只说明**该臂在固定 mixed 容差下满足条件**。
- 候选与原生在该臂上**仍有非零差**：22 的 CPU FP64 诊断（**按 rank 分开**）为 —— rank0 候选 57/80、原生 65/80；**rank1 候选 63/80、原生 59/80**；合计 **120/160 与 124/160**。160 个 rank-point 全部只落在 {−1, 0, +1} 个结果 lsb，**从不在同一点同时偏离**。
  - 该统计**仅覆盖这 80 个坐标 × 2 rank**，不外推为全量正确舍入保证，也不是厂家 GEMM 误差保证。
- 因此"门内差"仍可能在 **AR 求和、`partial + residual` 相消、以及 Norm 传播**之后放大到下游容差之外——**不能用 A 臂通过来排除 GEMM 作为完整链失败的来源**。
- 旧完整链 run **未保存输入 hash**，跨 run 因果**不能闭合**；完整链归因保持**未定**。
- 跨臂关联只核**每 rank 的 `x_sha256` 与 `w_sha256`**（§20）；`gamma` 在诊断臂为 ones，不能与完整链共享 hash。


## 19. 端口常量更正（2026-11-18 21:27 CST）

- HCCL 默认端口 **16666 = `/proc/net/tcp` 的 `:411A`**。
- 此前部分检查用了 `:40BA`，那是 **16570**，**属读错端口**；相关"port free"证据标注为无效读数。
- A 臂窗格本身有前后进程清理与单次启动记录，其运行结论不因此重跑；后续检查统一改用 `:411A`。

## 20. 输入 hash 的适用范围（2026-11-18 21:27 CST）

- `probe_partial` 臂令 `r=0` 且 `gamma=ones`，所以 A/B dump 里的 `gamma_sha256` 来自 **ones**。
- GEMM 跨臂关联应核 **`x_sha256` 与 `w_sha256`**（A 与 B 已确认一致：`x 220586f0…`，`w 1fd0e09d…` / `12539641…`）。
- 完整链若使用随机 `gamma` 与真实 `residual`，**不能**声称与 A 臂 gamma 同 hash；完整链需另存自己的 `r`/`gamma` 实际 hash，且候选与参考使用同一份张量。
- 记录范围已定：**完整链保留原随机 `r`/`gamma`**，另存其自身的 `r`/`gamma`/`eps` 实际 hash；候选与参考使用**同一份**张量。**不得为凑同 hash 而静默修改完整链输入**。

## 21. 同 x/w 完整链与 CPU 传播对照（2026-11-18 21:34 CST）

### 21.1 冻结证据目录

`/workspace/yangchen/logs/perf-wave3-20261118/e-full-final-20261118-s20261118/`
（含 `run-full-final.log` `24c9629c…`、`e-full-rt-rank0.json` `390b31d4…`、`e-full-rt-rank1.json` `d701802e…`、
两份 union76、`runner.sh`、`SHA256SUMS`）。身份：seed **20261118**、script **`c71ad736`**、包 **ndrt `2ee1c9ce`**。

### 21.2 该 run 的数值（random r/gamma，候选参考同份）

| 输出 | 值 | 门 | 判定 |
| --- | --- | --- | --- |
| `residual_sum` | fail 68，max_abs 0.03125 | mixed BF16 | fail |
| `Y` | max_abs 0.015625，**max_rel 1.0** | `≤0.02` 且 `≤1%` | fail |
| `rstd` | max_abs 6.08e-6 | `≤1e-3` | pass |
| finite/shape/dtype | 两 rank true | — | pass |

`r_sha256 62a80d25…`、`gamma_sha256 5cadbc9f…`、`eps 1e-6`；每 rank `x`/`w` 与 A/B 哈希一致。

### 21.3 CPU 传播对照（22，仅限 76 点）

| 检验 | rank0 | rank1 |
| --- | ---: | ---: |
| `RNE(B.cand + r)` == `residual.actual` | 76/76 | 76/76 |
| `RNE(B.native + r)` == `residual.ref` | 76/76 | 76/76 |
| `Y` 链（候选）复现 | 76/76 | 76/76 |
| `Y` 链（原生）复现 | 76/76 | 76/76 |
| `rstd.cand == rstd.ref` | **12/76** | **12/76** |

- 使用 B 臂 **r=0 的真实 post-AR partial**，`z = fp32(B.cand) + fp32(r)`，**未**把 BF16 `residual_out` 当未舍入 z，**未**用 `z − r` 派生值。
- 结论（限本次 76 点、且给定各自 partial 与已测 rstd）：**加法与缩放两步的复算吻合**；
  **`rstd` 的生成尚未独立核验**，`|Δrstd|` 中位 1.19e-7、最大 9.54e-7。
- **不得**表述为"整个 E 只剩 rstd 一个问题"：完整链 `residual` 的 68 项全量失败与 rstd 核验无关，仍需处理。

### 21.4 明确缺口

`rstd` 依赖整行 `mean(z²)`，当前只有点值。需要按 row 导出**整行未舍入 z**（FP32 或整行 BF16），才能判断 rstd 的差是否完全由 z 的差驱动。


## 22. seed 20261119 确认格（2026-11-18 21:36 CST）

| 输出 | 本 run 实际值 |
| --- | --- |
| `residual_sum` | fail **78**，max_abs **0.125**（BF16 SUM 参考 `refb_*`） |
| `Y` | max_abs **0.0625**、max_rel **1.2427325**，严格门 **fail** |
| `rstd` | max_abs **1.14023685e-4**，fail 0 |

- 身份：seed **20261119**、script 与包同 §21（`ndrt 2ee1c9ce`）；运行 `RAW_RC=0`，独立新输出路径 `e-full-2119.json`。
- **可比范围**：以上仅为**本 run 的实际值**。此前 seed 20261119 的 78 项来自 **FP32 SUM 参考**，摘要数字相同**不能**证明失败集合或协议可复现；两者不合并、不互相追认。

## 23. NZ 格契约失败（2026-11-18 21:39 CST）

### 23.1 结果：在 OpDef 前置被拒，未进入 tiling/kernel

- 运行：`nzguard_9a1c20cc`（`wave3 @ 65399e6`），实际 vendor **`wave3-e-install18`**（runner stdout 静态头写 `nzfmt_11eda865/install17`，**以本行为准**，原 stdout 不改）。
- adapter 实读描述符：`format=2`、`storage=[8704,5120,-1,-1]`、`strides=[5120,1]`、`offset=0`（两 rank 一致）。
- 最早异常：**`GetWorkspaceSize` 161002 / EZ1009**，`actual w BF16 ND` 与 `OpDef w FRACTAL_NZ` 不匹配 → **只支持在 OpDef 格式前拒绝**。
- 因此：**host 自定义 tiling 未执行、无有效 `diag` 输出**，`diag[7]` 无值；不得写成"tiling 已执行/已校验"。

### 23.2 直接线索（23 独立实读，冻结目录内可复现）

`run-nzguard.log` 开头两 rank 均有：

- `FormatCastKernelNpu.cpp:167`：**"Cannot create tensor with internal format while allow_internal_format=False, tensor will be created with base format"**
- `TensorFactories.cpp:437` 重复同一警告

即 `npu_format_cast(w, 29)` 在 `allow_internal_format=False` 的创建路径下**静默回落到 base format（ND）**，与 adapter 实读的 `format=2` 完全一致。

### 23.3 边界

- 该配置分支是**可核实的直接线索**，但尚未由我方独立确认修复路径；
- 不据 4D 文档示例推断"2D 不支持"，也不以手写 29 伪装 NZ；
- 冻结目录：`/workspace/yangchen/logs/perf-wave3-20261118/nz-contract-fail-20261118/`（`SHA256SUMS` 校验 2/2 OK）。

## 24. ND 与 NZ 逐点比对（2026-11-18 21:47 CST）

### 24.1 结果（同 seed 20261118、同 x/w/r/gamma）

| 项 | rank0 | rank1 |
| --- | --- | --- |
| `residual` fail | ND 68 / NZ 68，交集 **68**，各自独有 **0** | 同 |
| `Y` fail | ND 68 / NZ 68，交集 **68**，各自独有 **0** | 同 |
| union 集合 | **76 = 76，完全相同** | 同 |
| 逐点六字段（residual / y / rstd / gamma / residual_input / partial） | **76/76 相等** | 76/76 相等 |
| 汇总 | res max_abs 0.03125、Y max_abs 0.015625 / max_rel **1.0**、rstd 6.079673767e-6 | 同 |

### 24.2 能说与不能说

- **能说**：本次同输入的 76 点采样上，NZ 臂与 ND 臂的失败集合与逐点值**完全一致**，即这些失败**不是 NZ 消费路径特有**的。
- **不能说**：不做"NZ 引起/非 NZ 引起"的排他因果（仅 76 点，非全张量比对）；ND 臂的 68 个失败仍是**开放问题**，此前只到"每 rank ≤1 ulp + 跨 rank 抵消放大"的描述层面。

### 24.3 身份与缺口

| 环节 | 状态 |
| --- | --- |
| 安装对应 | **已闭合**：包 `9a1c20cc…`；install18 的 `libcust_opapi.so 9d62051f…`、`libcust_opmaster_rt2.0.so 4111ffc5…`、kernel `.o 68014382…` 三项与包内值逐一一致 |
| adapter descriptor | `format=29`、`storage=[320,544,16,16]`（= 8704×5120 精确相等） |
| tiling 闸门 | 非 `FORMAT_FRACTAL_NZ` 即 `GRAPH_FAILED`；本 run `ws=36581504`、`RAW_RC=0` |
| **运行时加载** | **未证明**：loader / mmap 记录未留存，静态证据不能证明进程实际加载了这三个文件 |

- `diag[7]=16` 已由 22 更正为 **`depthA1`**，不携带格式语义，本 doc 早前相关表述作废。

## 25. 原生 dispatch 采集（2026-11-18 22:04 CST）

### 25.1 运行与冻结

- 脚本 `wave3_dispatch_probe_22.py`，SHA256 `3cb3de30151256b861556873b7c6e0418599510df71379a387361a0bb5cbdad6`（26,990 B，本机直传，双方 SHA 一致）。
- 两 rank `RC=0`：rank0 → npu0（PID 1141568，`14:02:18–25Z`），rank1 → npu1（PID 1141768，`14:02:31–37Z`）。
- 冻结目录 `/workspace/yangchen/logs/perf-wave3-20261118/dispatch-probe-20261118/`：含 `rank{0,1}/{meta,exported_fields,numeric}.json`、`maps_filtered.txt`、`trace/`、`probe.py`、`runner.sh`、`run-dispatch.log`；`SHA256SUMS` **177 条，177/177 OK**（排除自身）。

### 25.2 已取得的字段（两 rank 一致）

| 字段 | REF_ND | REF_NZ |
| --- | --- | --- |
| Name | `aclnnMatmul_MatMulCommon_MatMulV2` | `aclnnMatmulWeightNz_MatMulCommon_MatMulV2` |
| Type | `MatMulV2` | `MatMulV2` |
| Accelerator Core | AI_CORE | AI_CORE |
| Block Num | **24** | **24** |
| Mix Block Num | 0 | 0 |
| **HF32 Eligible** | **NO** | **NO** |
| Input Formats | `ND;ND` | `ND;FRACTAL_NZ` |
| Input Data Types | `DT_BF16;DT_BF16` | 同 |

- 时间线关联按 `trace_view.json` 的**窗口包含**判定（每标签窗口内恰好各含一个 API 与一个 kernel 事件），非按 Step 顺序推断。
- 输入 hash：X `220586f0…`、W0 `1fd0e09d…`、W1 `12539641…`，与 A/B/完整 E 的 `x`/`w` 一致。

### 25.3 缺口（不再重复采集）

- **本次导出未取得 tiling key**：`tiling_columns: []`；`api_statistic.csv` 的 `MatMulV2_Tiling` 是 **API 名 + 调用耗时**，不是 key。
- kernel 名后缀（`98514`/`229586`）是名字内嵌标识，**不能当 tiling key**；名中 `FP16` 与实际 `DT_BF16` 字段冲突时**以字段为准**。
- **设备 kernel 二进制身份未闭合**：本轮 maps 为**主机库**映射（各 136 文件、0 个 deleted）；maps 只能证明**这些主机库存在于进程内**（路径 + 落盘 SHA 说明被映射），**单凭同时映射不能判定符号解析选中了哪一份 tiling**，也不能追证旧 E 实际加载，且已发现同机存在**多份** `libcust_opapi.so`/`libcust_opmaster_rt2.0.so`（`batch_invariant`、`fla_npu_transformer`、`libcustom_op_runtime.so`）——安装 identity 与实际加载 identity 必须分开记录。
- 22 已更正其 `exported_fields.json` 的误报：`core_type`/`op_type`/`block_dim` 实际**存在**（提示词下划线 vs 列名空格），真正缺失的只有 tiling key。

### 25.4 数值（背景，非 dispatch 证据）

`numeric.json` 两 rank 各 4,951,040 元素、`max_abs_diff = 0`、`nonzero = 0` —— 表述为**本窗口全元素数值差 0**（该文件只算 float 差与 `nonzero_count`，**未做位模式比较、未独立检查 finite**，故不称"逐位相同"）；不外推其他形状。此前 ND/NZ 两臂 76 点"全等"同样是 JSON 十进制往返后的**数值**相等，一并按此表述。

### 25.5 与自定义算子的对照（仅记录）

原生 `Block Num = 24`；自定义 E 算子 `diag[5]`（`cubeTiling.usedCoreNum`）在 ND/NZ 两次全量 run 中均为 **1**。

**来源已查明（22 只读源码 `op_host:180-194`）**：`mm_dim` 属性被 runner 显式传为 **1** → `SetDim(1)` → `GetTiling` 返回 `usedCoreNum=1`。这是**传参结果，不是 tiling 能力不足**。

**三个数字不可等同**：

| 数字 | 来源 | 值 |
| --- | --- | --- |
| `mm_dim` 属性（`SetDim` 预算） | runner 传参 | 1 |
| `TCubeTiling.usedCoreNum` | 我们 kernel 写入 `diag[5]` | 1 |
| 原生 `Block Num` | profiler `kernel_details.csv` | 24（AIC） |
| 我们算子实际 launch block 数 | **本轮无证据** | 未知 |

**默认映射问题（源码级，未上板）**：`mm_dim=0` 时走 `cores`，而 `block_dim=0` 时 `cores = GetCoreNumAiv()`（Vector 核数），cube 却只能按 AIC 核数划分（本机 AIC=24）。最小修复是把未指定时的 `mmDim` 默认取 `GetCoreNumAic()`；需改代码与协调包身份，**当前不动**。

**可检验的单变量控制臂（未执行）**：只把 `mm_dim: 1 → -1`（走已有 `mmDimReq==-1 → GetCoreNumAic()` 路径，**无需改代码/重建包**），预期 `usedCoreNum=24`。**预先登记预测**：多核切分切 M/N 不切 K，数值预期不变（全元素数值差 0）——若变则说明"M/N 划分影响累加"模型不成立，属新证据；若不变则多核配置被排除。**该差异不预判为精度根因，仅是把并行划分拉回预期。**

**其他源码判断**：每 block 的 A/B/C 偏移由 `IterateAll` 依 `TCubeTiling` 自行推导，我们未写也不需要写（非缺口）；`SetSingleShape`/`SetTail` 未调用且该路径不需要，但 **tail 行为属未证明**（M=967、baseM=128 确有尾部）；`REGIST_MATMUL_OBJ` 在所有 block 执行，仅计算调用受 `GetBlockIdx() < usedCoreNum` 门控。

## 26. 多核 M 分片修复（22 交付，2026-11-18 22:26 CST，未安装未上板）

### 26.1 实现要点（最小差异）

```
workers = usedCoreNum (0→1)
singleM = singleCoreM ?: ceil(m/workers)
if (singleM==0 || singleM*workers < m) singleM = ceil(m/workers)   // 覆盖守卫
startM  = blockIdx * singleM;  curSingleM = min(m-startM, singleM)
SetOrgShape(m,n,k); SetSingleShape(curSingleM, n, k)
SetTensorA(aG[startM*k]); SetTensorB(bG, btrans); IterateAll(cG[startM*n],0,false,true)
```

- **B 完全不偏移**（不切 N、不切 K）；直接依据：`grouped_matmul_swiglu_quant.h:222-231` 的 `GetWOffset(tailN,k)` 在 `tailN=0` 时恒为 0，其调用方仅在 N 切分时加它（`:174`），M 切分只偏移 A 与 C（`:169-172`）。
- 参与者映射：同栈生产算子把 matmul 放在 **AIC**（`CubeProcess` 在 `if ASCEND_IS_AIC` 内，用 AIC `blockIdx`）；我们的包走 **AIV 客户端**范式，matmul 头里**没有** `GetBlockIdx`（server 只认 `subBlockID`，`matmul_server.h:57`）→ **不依赖 AIV→AIC 索引映射**。

### 26.2 CPU 覆盖与控制流

- `tests/slicing_cpu_check.py` **140 例**：区间不重叠、恰好覆盖 `[0,m)`、单核等价；**测试先抓到真 bug**（`singleCoreM < ceil(m/workers)` 会漏行），因此加覆盖守卫；`workers=1` 时守卫把 `singleCoreM` 折成 `m` → **固定 `mm_dim=1` 路径行为不变**。
- cube 段后 `SyncAll<true>()`（`kernel_operator_block_sync_intf.h:77-87`）。

### 26.3 未证部分（需设备核）

| 项 | 状态 |
| --- | --- |
| 该 AIV-only 屏障是否把 fixpipe/GM 写序化到其他 AIV 的 GM 读之前 | **未证**（头文件无内存序契约） |
| `usedCoreNum` 超过 AIC 数时，屏障集合是否等于实际发起集合 | **未证** |
| 本点是否还需 AIC 参与 | **未证**（旧"AIC 已退出"因果已撤回） |

- 模板默认值本身**不构成**可用性证明。

### 26.4 diag ABI 与三个数字

- `diag` 仍 8 float（`InitBuffer(diagT_,32)`、三处 `DataCopy(...,8)` 未改）；
- 新增可选属性 **`mc_diag=1`**：`diag[5..7] = workers / singleCoreM / GetBlockNum()`；缺省仍 `usedCoreNum / baseM / depthA1`；
- **三者不混同**：请求预算 `mmDim`（host tiling）/ tiling 回答 `cubeTiling.usedCoreNum`（diag[5]）/ 实际 launch `GetBlockNum()`（`mc_diag=1` 的 diag[7]）。
- **不预保证** `usedCoreNum=24`，**不预保证**数值不变（`GetTiling` 只有声明、实现预编译，无法判断是否改 `baseK`/`step`）。

### 26.5 构建归档与集成前置

| 产物 | SHA-256 |
| --- | --- |
| `custom_opp_ubuntu_aarch64.run` | **`873ef8fb56df860978802584b3cbdea8ca9b2435d479a43239847abe29ee81d1`**（重编后；旧 `65260db7…` 作废） |
| kernel `.o` | **`05cf6bd218ce15ac483a4f7ac338d1ef60edbf03d5bd71bca03c2249c12574c2`** |
| `libcust_opmaster_rt2.0.so` | **`1ecdb91fb2f12f85dca27c74ad64c314d839d5df90039a8377c2e54c37c5db18`**（注：此前记录的 `290cc31e…` 作废） |
| `libcust_opapi.so` | `9d62051f0d7da2efafd741ed68640becd9b9ea88eb8d08e07d050271d56cc227`（与 nzguard 包**逐字节相同**） |

- **`mc_diag` 缺陷的成因（22 自述）**：属性只加进包 json，**未进 OpDef** 的 `this->Attr(...)`；生成头因此没有它（`grep -c mcDiag` = 0、`grep -c int64_t` = 14），而 tiling 读 `GetAttrPointer<int64_t>(14)` 会**越界**。已整条移除，diag 恢复 `usedCoreNum/baseM/depthA1`，**8 float、14 属性、adapter 无需改动**。
- **集成前置（2026-11-18 22:33 CST 更正，已闭）**：22 实际生成的 aclnn 头**没有 `mcDiag`**；随后确认**第 15 属性从未进入 OpDef**，现已**移除该属性并保持原 14 参数 ABI**。
  因此：**不要**按此前说法把 adapter 改到 15 参数，**也不要**安装旧 `65260db7…` 包（仅归档待替换）。等 22 的**最终新包与生成头证据**再核对。
- vendor 名 `clef_matmul_ar_add_rms_norm_mc`、op 名仍 `ClefMatmulArAddRmsNorm`。**安装目录共存**与**运行时实际选用**必须分开记录；**不得**仅据同名自行卸载既有包，也不**推断**唯一装载方式。
- **授权状态**：新包仅 CPU 准备，**20 尚未放行任何 E 设备格**；GDN 完成后亦须有明确排队消息才可执行——**队列顺序不等于上板授权**。
- 源提交 `aacf371`，MANIFEST 在 `836b8b6`；已归档的 nzguard/ND 包与单核对照**未动**。

## 27. 多核包 2201 契约核准：两个高优先问题（22 只读，2026-11-18 22:34 CST）

### 27.1 关键对应表（每项带 SDK 直接依据）

| # | 项 | 本包位置 | 依据 |
| --- | --- | --- | --- |
| 1 | **AIV 的 `GetBlockIdx()` = `aicIdx*2 + subBlockId`**，AIC 的是 `aicIdx` | `op_kernel:220` | `dav_c220/kernel_operator_sys_var_impl.h:56-72` |
| 2 | `GetSubBlockIdx()`：AIV 0~1，AIC 0 | 未使用 | 同上 `:31-42` |
| 3 | `MIX_NUM = 2` | — | `dav_c220/kfc/kfc_comm.h:94-96` |
| 4 | AIC↔AIV 分组 `aicId = blockIdx / MIX_NUM` | — | `dav_c220/core_mng/roc/kernel_operator_cube_group_handle_impl.h:53-69` |
| 5 | 本包 worker 判据用 **AIV 下标** `myBlock < usedCoreNum` | `op_kernel:221-238` | 由 #1 推导；**实际 launch 数未记录** |
| 6 | `REGIST_MATMUL_OBJ` 在所有 block 执行 | `op_kernel:180` | `kfc_register_obj.h:210/250/297/321` |
| 7 | server 退出 `quitSize < MIX_NUM` | — | `kernel_kfc.h:60-63`、`:123-128`、`:199-218` |
| 8 | `End()` 仅 `isSyncGetC` 时投 `MMFUN_END` | `op_kernel:229` 附近 | `matmul_client.h:695-701` |
| 9 | `WaitIterateAll()` 断言 `!isSyncGetC` | `IterateAll(...,true)` | `matmul_client.h:772-773` |
| 10 | **`SyncAll<true>` 是 AIC 直接返回的 AIV-only 屏障** | `op_kernel:246` | `dav_c220/kernel_operator_sync_impl.h:63-83` |
| 11 | GM 生产→消费的**内存序** | `op_kernel:246-265` | **SDK 未给契约** |

### 27.2 问题 ①：worker 用 AIV 下标 ⇒ 24 worker 只落 12 个 cube

- `GetBlockIdx()` 在 AIV 为 `aicIdx*2+subBlockId`；`usedCoreNum=24` 时参与的是 AIV 0..23 ⇒ **aicIdx 0..11**，每 AIC 两个 client 各认领不同 M 段。
- **不漏行、不重复写**（M 段互斥且覆盖全区间；CPU 140 例只证这一点），但**不是 24 路 cube 并行**，只是 12 个 cube 各做两段。
- 若要 `usedCoreNum` 路 cube，worker 判据应改为 `subBlockId==0 && aicIdx<usedCoreNum`（`aicIdx = GetBlockIdx()/MIX_NUM`）——**需改代码，本轮未做**。

### 27.3 问题 ②：`SyncAll<true>` 两 AIV 子块共用同一 GM 槽 ⇒ 可能永久等待

- `:77` 在 `isAIVOnly` 时用 `get_block_idx()`（AIC 下标 0..23）算槽位，`:72` 总块数用 `GetBlockNum()`。
- 若 `GetBlockNum()` 在 AIV 返回 **AIV 数（48）**，则 48 个到达者只写 24 个槽 ⇒ **屏障永不满足 → 挂死**。
- `:66-68` 明确 **AIC 直接 return**，所以旧注释"AIC 已退出导致超时"不成立；但**新屏障有另一种挂死路径**，`GetBlockNum()` 的 AIV 语义未定。

### 27.4 拟议的最小可证伪诊断（只准备，未执行，待 20 决定）

| 格 | 配置 | 通过判据 | 停止条件 |
| --- | --- | --- | --- |
| **D1**（判 ②，风险最高，先跑） | `usedCoreNum=1`，保留新增 `SyncAll<true>` | `RAW_RC=0` 且三输出与归档单核对照差 0 | 出现 `507014 / 0x25` 或超时 ⇒ **判定 `SyncAll<true>` 在本包范式下不可用**，立即停，回到 GM flag 手工屏障方案 |
| **D2**（判 ①） | `mm_dim=2`（最小偶数 worker） | `RAW_RC=0`；`partial` 与单核对照全元素差 0；diag[5] 报 `usedCoreNum` | `partial` 未写行 ⇒ 漏行；两段相同 ⇒ 重复写；超时 ⇒ 集合问题 |

- **不做**：baseK/shape 扫描，不一次跑多个 `usedCoreNum`。
- 本包**未安装、未上板**；①的改法属代码改动，需另行授权。

## 28. GDN BLOCK_M 方向关闭（2026-11-18 22:36 CST）

### 28.1 run4 汇总（24 号）

- 阶段：ENV / CHECK / APPLY / PROBE / REVERT **全 0**；目标已回 preimage。
- 可执行臂：**BM16 / BM32 / BM64**；**BM128 = warmup 期 `MLIRCompilationError` / UB overflow**（完整 traceback 4,985 B 存于 `run4/probe.json.arm-128.traceback.txt`）。
- 正确性：三臂对同输入 BM64 的 `out`（2,970,624 元素）与 `rstd`（23,208）**全部逐位一致**、finite、0 NaN/Inf、`max_abs 0`。
- metadata：`warps32/stages2/coalesce1/axis-1/shared1` 无缺失；三臂 kernel hash 不同且 `BLOCK_M` constexpr 与请求臂匹配。
- **48 调用批 3 轮 device 配对差**：BM16 `+0.1798 / −0.2302 / −0.0418 ms`；BM32 `+0.0659 / −0.2418 / −0.0141 ms` → **正负翻转，无稳定优于 BM64**；`per_call=null`，不报单次 max。

### 28.2 决定

**保持默认 BM64；BLOCK_M 方向关闭；不启动服务 A/B，不开新窗口。**

### 28.3 证据

| 项 | 值 |
| --- | --- |
| 分支/提交 | `perf/wave3-gdn-block-m-prep` @ `dd1f7539d2beddcaae75f389de81d4a2cfc4b651` |
| `run4/probe.json` | `552c00e8…1ec94` |
| `run-gdn4.log` | `719d6182…d4d5` |
| 缺口 | run4 **无 PID 树记录**，保留 |

- 冻结目录 `/workspace/yangchen/logs/gdn-norm-block-m-20261118-run4-freeze/`（`SHA256SUMS` 10/10 OK，含 `provenance.txt` 记录 PID 树缺项）。
