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
