# Clef-27B Ascend 工程约定

## 沟通与目标

- 使用中文直接陈述结论、证据和下一步，遵循用户关于肯定表达的要求。
- 2026-11-17 最终两项非量化优化（已采用）：`ACLNN_CACHE_LIMIT=1000000` 修复 aclnn executor 缓存耗尽导致的 +55 ms 形状慢路径；MLP BF16 NZ（FRACTAL_NZ，GDN 安装器 v4，`CLEF_MLP_BF16_NZ=1`）再降 p95。官方 `start.sh --config s1` 全量 S1 **mean 121.705 / p50 113.656 / p95 166.002 ms、8.213 dec/s**；2,352 条响应逐位一致，OD Capability 74.3107、ERQA 217/400、S2 B8 C16 16.376。证据 `docs/final-summary-2026-11-17.md`、`docs/aclnn-cache-shape-fix-2026-11-17.md`、`Clef-27B/logs/tuning5-20261117/`。
- 当前任务为在现有昇腾 A3 环境建立可复现的 Clef-27B 比赛提交版本，包含完整主干、联合决策头、文本与图文接口、质量及性能评测证据。
- 2026-11-13 用户补充的比赛交付规则见 `docs/submission-rules-2026-11-13.md`。当前按该版本准备验收；后续以主办方最终协议与模板更新，规则来源和版本均需保留。
- 2026-11-16 用户明确 **2PP 精度口径**：比赛**不需要温度校准**，使用原始服务输出；“baseline 精度范围不超过 2PP”指 **OpenDecisions Intelligence、Calibration、Capability 与 ERQA 总体准确率**相对 baseline 的**绝对波动均不超过 2 个百分点**（ERQA 2PP = 0.02）。该口径已写入 `docs/submission-rules-2026-11-13.md` 第 4.3 节；主办方最终细则更新时保留本条来源与日期。
- 2026-11-16 用户转述赛方要求：**不允许量化**。W8A8、W8A16、INT8/FP8 等降低权重/激活精度的方案不得作为提交配置；2PP 不能放行量化。历史量化结果仅保留为负结果，后续只做 BF16 非量化优化（BIC/GDN、布局、graph、调度/合批、数据路径、BF16 融合算子）。该限制写入规则文档第 4.4 节。
- 2026-11-14 第四轮复核：seqs4 合批全量质量未通过 0.03/0.05 工程门槛（Capability 74.200178；4 条 choice 变化；最大概率差 0.097399），最终提交配置改为 **seqs1/B1**；全量质量 74.425479 与 baseline 逐位一致，S1 P95 217.036 ms，S2 5.715 decisions/s @ C16。根因、GDN 覆盖、启动/补丁修复见 `docs/tuning4-2026-11-14.md`。
- 2026-11-14 ERQA 官方 400 题全量图文 baseline 完成：400/400 成功、0 失败、0 重试，accuracy **219/400 = 0.5475**；单图 0.5923、多图 0.4336，8 类分项见 `docs/erqa-baseline-2026-11-14.md`。服务为 TP2/BF16/eager/16384/seqs1/缓存关、`image` 上限 16、图像像素 65,536～1,048,576；C1 wall 101.099 s、p95 323.476 ms。数据固定官方 commit `5bba488`、TFRecord SHA-256 `1974f5e5…b4c93e2`。EvalScope ERQAAdapter 图文补丁在 19 题交叉验证中与直接 `/v1/decide` 逐题一致。原服务已恢复为 PID 432954、原 argv/cwd/4096 配置，8/8 冻结用例通过。
- 2026-11-14 CPU 独立参考：发布方 `joint_schema_model.py` 以 CPU BF16（eager、`use_kernels=False`、batch1、32 线程）重建输入与逐题前向。两套数据全量编码 token 数与 NPU 实测逐题相等（OpenDecisions 2,352/2,352；ERQA 400/400，含 16 图 10,294 token）；参考媒体布局与 `ClefFlashStrategy` 一致、无需补丁；8 条冻结请求在当前服务复放 24/24 逐位一致。逐题前向在 OpenDecisions 154/2352 时按用户要求终止，未生成完整 CPU 评分。证据见 `Clef-27B/logs/cpu-reference-20261114/`，结论与终止范围见 `docs/cpu-npu-reference-2026-11-14.md`。
- 2026-11-15 MLP ND 路径复核：局部 MLP 链 ND 慢于 NZ（短形状 63 点仅 2 点 ND 更快），但服务级 63 条短形状子集 ND 60/63 更快（中位 −11.65 ms）；ND 全量 OD 5.7711 req/s、P95 205.76 ms、Capability 74.0716，ERQA 219/400、P95 305.716 ms（完整 400 预热）。ND 修复了 NZ 的短形状端到端退化，但 W8A8 质量仍低于 BF16（74.4255）；默认提交保持 BF16。ND/W8A8 现因赛方**不允许量化**不可提交，仅保留为历史负结果。证据见 `Clef-27B/logs/mlp-20261115/nd-format-ab.json` 与主报告第 10 节。
- 2026-11-15 GDN 决策专用 prefill：纯 prefill、无 decode 的服务跳过无人读取的 GDN 循环状态读写（`output_final_state`/scatter）与反向中间量（`disable_recompute`），guard 只用宿主侧字段。OpenDecisions 全量 2,352 题 OFF/ON 逐位一致；同进程交错 A/B 6/6 对更快，p95 中位 −3.80%（−7.68 ms），吞吐 +0.17 decisions/s；已接入 `--config s1` 默认启用（`inference.py` 启动时安装/校验并写入 manifest），正常入口 S1 P95 213.56 ms，ERQA 400 ON/OFF 逐题一致。中间版本 v2 的逐层设备同步导致 +14.8 ms，作为负结果保留。证据见 `Clef-27B/logs/gdn-decision-20261115/`，方法与设计迭代见 `docs/gdn-decision-prefill-2026-11-15.md`。
- 2026-11-15 HBM 带宽微基准：A3 单逻辑 NPU 纯 HBM 读 **1.26–1.32 TB/s**（`copy` 1.26–1.27、AI Core `sum` 1.30–1.32），超过 L2 的 256 MiB–1 GiB 工作集稳定；MLP 有效权重带宽 **0.53 TB/s**，仅约 **40–42%**，说明 `aic_mte2_ratio=0.994` 是 MTE2 通路忙而非 HBM 上限，瓶颈在访问效率/tiling/格式。按 1.27 TB/s 粗算 MLP 理论上限约 13.5 ms（当前 32.07），但受 Cube tiling/依赖限制，需用 BF16 NZ/tiling/预取验证。证据见 `Clef-27B/logs/gdn-data-path-20261115/hbm-bandwidth.json`，方法与结果见 `docs/hbm-bandwidth-2026-11-15.md`。
- 2026-11-15 MLP BF16 NZ 局部实验：`F.linear` 接受 FRACTAL_NZ29，ND/NZ 输出全部逐位一致；cold 307 token 1.26–1.32×（名义 509–532→671–677 GB/s）、969 1.06×、1595/10294 约 1.00–1.03×。S1 P95（969 token）的 MLP 已约 256–299 TFLOPS、转为 compute-bound，NZ 对 P95 约 1.9%，是质量中性候选而非主杠杆；服务级 A/B 未执行。证据见 `Clef-27B/logs/gdn-data-path-20261115/mlp-nz/`，报告见 `docs/mlp-bf16-nz-2026-11-15.md`。
- 2026-11-15 MLP 纯 matmul 上限：BF16 Cube 约 300–320 TFLOPS（10294/4096 方阵）；969 token 的 gate/up 282、down 258，MLP 链 0.954 ms/层 ≈ 纯 matmul 之和 0.948，SwiGLU/启动开销 <1%。BF16 tiling 的理论上限约 3–4% 端到端，M padding 仅 969→1024 约 2%，1595/10294 无收益；长序列优先 INT8 gate/up（约 10%，质量门槛）与 MC2 down+allreduce（约 10%，质量中性）。证据见 `Clef-27B/logs/gdn-data-path-20261115/matmul-ceiling.json`，报告见 `docs/mlp-cube-ceiling-2026-11-15.md`。
- 2026-11-16 输入分布：OD token p50/p95/p99=357.5/969.25/1211.96（max 1595），cap 1200 覆盖 98.85%，全部单字段、候选 2–7；ERQA p50/p95/p99=933/1653/3559（max 10294），cap 1200 覆盖 68.75%，3+ 图 53 题。支持 cap 1200 与 B8 C16。报告 `docs/input-distribution-2026-11-16.md`。
- 2026-11-17 最终版 profile（v2+head）：307 CPU 每层 1.78–2.18 ms×64=114–139 ms，高于设备 compute 75 ms，CPU dispatch 临界；967 设备 compute 144.6 ms、CPU backbone 126–145 ms、通信非重叠 13.5–17.2 ms、非 kernel 12.5–15.7 ms；joint pooler（含 head）8.9–11.5 ms，head 6.7–9.0 ms，schema+tokenizer 约 3.9 ms。S1 P95 仍设备侧；head/pooler 再融合约 1–2 ms，S1 乐观 185–188 ms。报告 `docs/profile-final-2026-11-17.md`。
- 2026-11-17 conv+QKV 融合算子 `ClefGdnConvQkv`（**已采纳为 s1 默认**）：一次调用完成 causal conv1d + q/k L2 norm + v 直拷，替代 conv（3.79 ms）+ `ClefGdnQkvNorm`（4.66 ms）两个 kernel；v2 起支持 packed QKVZ `[T,8192]`（宿主 `as_strided` + 存储边界校验，免去逐层 5120 列连续化拷贝），v3 改 FLA arch22 tile 归约。探针 v1/v2/v3 共 177 组（10 长度 × seed × tile 32/64 × 5120/8192）全部逐位一致、0 失败；单算子 967 token 0.2113 vs 0.3143 ms、1595 0.1296 vs 0.3357。服务级同窗口全量 A/B（唯一变量 `CLEF_GDN_CONV_QKV`）：**p95 165.091 → 160.150 ms（−2.99%）**、mean −4.41%、p50 −5.26%、p99 −3.24%、吞吐 8.137 → 8.510 dec/s（+4.59%）；质量 **Capability 74.3107 → 73.9636（−0.347 pp，2PP 内）**、accuracy 1902/2352 不变、Score MAE 0.401545 → 0.401678；ERQA **217 → 218/400 = 0.5450（+0.25 pp）**。注意：输出**不是逐位一致**（三个固定样本最大概率差 0.00095/0.00240/0.00029；全量 max_prob_diff 0.1447、1 条 choice 变化、7 条 >0.03），项目内部 0.03/0.05 门槛不达标，按官方 2PP 口径采纳。vendor 内置于 `Clef-27B/src/ascendc/vendors/clef_conv_qkv`，`inference.py` 自动注入 env；`CLEF_GDN_CONV_QKV=0` 复现对照臂。通用 `ClefGdnQkvGeneric`（独立 vendor，TP1/TP2 任意长度）36 组检查通过，TP2 逐位一致、TP1 最大差 0.001953125，作 TP1 与回退候选。报告 `docs/gdn-conv-qkv-2026-11-17.md`，证据 `Clef-27B/logs/perf-next-20261117/`。
- 2026-11-17 GDN **chunk 128 改为默认**（用户口径“2PP 以内就行”）：`inference.py` 在决策 prefill 启用时自动 apply，`CLEF_GDN_CHUNK128=0` 回到 64。叠加 conv+QKV 融合后同窗口 A/B：mean 120.926→118.733、p95 160.419→**158.301**（−1.32%）、8.259→**8.411 dec/s**；OD Capability 73.9636→**74.2476**、accuracy 1902→1904、ERQA 218→**216/400**（−0.50 pp，2PP）；S2 B8 C16 **16.727 dec/s**、p95 1768.9、Capability 73.8426。相对改造前 12:20 对照（p95 164.347、8.223）为 −3.68% / +2.29%。逐题漂移更大（3 条 choice 变化、max 0.360、24 条 >0.05），项目内部 0.03/0.05 门槛不达标，按官方 2PP 采纳；证据 `docs/gdn-conv-qkv-2026-11-17.md` 第 4.3 节、`s1-chunk128-conv.json.gz`、`s2-chunk128-conv.json.gz`、`chunk128-conv-quality.json`、`erqa-chunk128-conv/`。
- 2026-11-17 MLP matmul 上限复核（负结果）：真实 TP2 权重下 gate/up 与 down 冷态 **285.9–303.4 TFLOPS**，已等于/超过 4096³ 方阵冷态 294.6 TFLOPS；warm 下 1595 token 的 ND 领先 3–5% 是 L2 驻留伪影，cold 下 NZ 全面不差；M padding 967→1024 仅 0.4%。**MLP 自定义 tiling 无可挖空间**，不投入 Ascend C matmul 重写。证据 `Clef-27B/logs/perf-next-20261117/mlp-ceiling.json.gz`，报告 `docs/gdn-conv-qkv-2026-11-17.md` 第 6 节。
- 2026-11-17 通信/归一化复核（负结果）：TP2 allreduce `[967,5120]` 隔离 0.1433 ms / 69.1 GB/s，`HCCL_OP_EXPANSION_MODE=AIV` 隔离 +7.3%（74.5 GB/s）但服务级相邻窗口 S1 全部在噪声内（p95 −0.24%、mean −0.19%），不采纳；`npu_add_rms_norm` 967 token 0.0355 ms / 1117 GB/s、1595 1266 GB/s，已贴 HBM 上限，profile 的 5.96 ms 是含间隙聚合值。证据 `Clef-27B/logs/perf-next-20261117/{allreduce-967,add-rmsnorm,s1-aiv,s1-aiv-control}.json.gz`，报告第 7 节。
- 2026-11-17 修正 host load 口径：`uptime` 的 load average 是**宿主机全局**统计（含其他租户、含不可中断 I/O 任务），不代表本容器 CPU 争用。实测 `vmstat` id 99%、`r=0`、`st=0`，容器 `cpu.max=3200000`（32 核配额）、宿主机 640 核，空闲时本项目 vLLM 进程合计约 13% 单核。同一配置 1.5 小时内 4 次全量 S1：p95 **157.32–158.30 ms（差 0.62%）**、mean 115.4–118.7（差约 2.8%）、8.41–8.65 dec/s。**结论：p95 报数可直接采用，mean/吞吐按 ±3% 波动、只用同窗口 A/B 比较；此前“load>10 不测”过于保守。**
- 2026-11-16 GDN 边界融合（已采用）：`ClefGdnQkvNorm` 把 conv 输出直接写成归一化 BSND q/k/v；969 token 融合核 1.27×，kernel v2 + 单字段 head 后正常入口 S1 p95 **190.405 ms**（v2 同窗口 v1 −2.48%，head 同窗口 head-off −2.91%）、mean 153.761、吞吐 6.503；OD Capability **74.310693**（Δ−0.115，2PP）、accuracy 1902、Score MAE 0.401545；ERQA **217/400=0.5425**（−0.50 pp，2PP）。cap **1600**，token >1600 回退 BIC；head 安装器 `runtime/apply_joint_single_field_patch.py`。vendor 在 `Clef-27B/src/ascendc/vendors/customize`，v4 安装器 `runtime/apply_qkv_norm_patch.py`。报告 `docs/gdn-boundary-fusion-2026-11-16.md`、`docs/gdn-kernel-v2-2026-11-16.md`。
- 2026-11-16 BF16 B10/B12 补测：B10 C20 16.670 dec/s/p95 2249、Capability 74.3011；B12 C24 16.841/p95 2598、Capability 74.2014。两者均未支配 B8 C16（16.464/p95 1804），B8 保持 p95 约束下的 S2 推荐点；B16 C64 17.110 仍是无 p95 约束的最大吞吐。报告 `docs/batching-b10-b12-2026-11-16.md`。
- 2026-11-16 BF16 B8/B16 合批：B8 C16 **16.464 dec/s**、p95 1804 ms，Capability 74.2676（−0.158）；B16 C64 **17.110 dec/s**、p95 6088，Capability 74.2241（−0.201）；OD 三项均在 2PP 内。ERQA 合批 B8 C16 **218/400=0.5450**（−0.25 pp）、B16 C32 216/400（−0.75 pp），均在 2PP 内。S2 推荐 B8 C16（最终 v2+head 16.398 dec/s、p95 1803、Capability 74.2510），B10/B12 未支配，B16 只按最大吞吐。证据 `Clef-27B/logs/gdn-data-path-20261115/seqs8-test/`、`seqs16-test/`、`erqa-b8/`、`erqa-b16/`，报告 `docs/batching-seqs8-16-2026-11-16.md`。
- 2026-11-16 上游 Dense SP 移植：把 `b65e7c718`（Qwen3.5 dense SP，对接 vLLM `24420b7a`）移植到 `89eb8a5`，保留 GDN/BIC，`enable_flashcomm1` 启用；质量 2PP（Capability 74.4235、choice 0 变化），但全量 S1 p95 **212.96 vs 198.42**、5.718 vs 6.226 dec/s，307 更慢；**不采用**，源码恢复 SHA `1b82b52e…`/`52585cf4…`/`65bc9b18…`，服务 PID 711199。报告 `docs/dense-sp-upstream-port-2026-11-16.md`。
- 2026-11-16 Full attention prefill cache skip：`PrefillNoCache` 下跳过 `reshape_and_cache`，冻结 8/8 逐位；307 mean 129.09 vs 125.06（更慢），969 170.85 vs 172.51（−1.0% mean，噪声内）；未过 ≥3% 门，**不采用**，源码恢复 SHA `a221dd76…`。证据 `Clef-27B/logs/gdn-data-path-20261115/attn-skip-test/`，报告 `docs/attn-prefill-cache-skip-2026-11-16.md`。
- 2026-11-16 GEMM+AllReduce 同层分块重叠：down_proj 按 token 切 2/4 块、compute/comm 双 stream；308 token 0.42×/0.29×、970 0.80×/0.46×、1596 0.79×/0.69×，均慢于一次完整 AllReduce；不采用。证据 `Clef-27B/logs/gdn-data-path-20261115/mlp-overlap/`，报告 `docs/npu-gemm-overlap-probe-2026-11-16.md`。
- 2026-11-16 MLP-only Dense SP 探针：`npu_all_gather_base_mm` + down ReduceScatter 数值可用（max_abs 0.015625、mean_abs≈2e-5），但端到端 **0.88–0.98×**（1596 token 0.878×），比当前 AllReduce 慢；不采用，完整 Dense SP 优先级下调。证据 `Clef-27B/logs/gdn-data-path-20261115/mlp-sp/`，报告 `docs/dense-sp-mlp-probe-2026-11-16.md`。
- 2026-11-16 Joint head 跨请求批处理：实现 padding/mask 的 `forward_batch` 与批量 lexical embedding；B4 C4 13.442 vs off 13.637 dec/s（变慢），B8 小样本 18.614 vs 18.158（+2.5%），但全量 OD C16 16.4685 vs 16.4638（+0.03%）无实际收益；质量 2PP（Capability 74.2174，Δ−0.208）。**不采用**，源码已恢复 SHA `197863d5…`，服务 PID 692708。报告 `docs/joint-head-batch-2026-11-16.md`。
- 2026-11-16 Host 微优化复测：schema cache v2 在 s1/307 反而慢约 5%、969 持平；BF16 NZ 服务化真实 hook（`Qwen2MoeMLP.forward`）307 mean −1.6% 但 p95 +1.4%，969 mean/p95 −0.7%/−0.8%；均逐位一致但**不采用**。第一版 hook 放在未使用的 `AscendQwen3_5DecoderLayer` 上是死代码，修正后数据才有效。证据 `Clef-27B/logs/gdn-data-path-20261115/schema-cache-test/`、`nz-test/`，报告 `docs/host-opt-2026-11-16.md`。
- 2026-11-16 Graph/ACL graph 复核：默认 `max_num_seqs=1` 只捕获 `[1]`，307/967 实际未 replay；显式 `[307,967]` 后 307 mean/p95 **94.34/96.38 ms**（邻近 eager 109.95–119.90/123.36–130.84），967 持平；粗桶全量 S1 p95 **244.53 ms**、密桶 265.80（eager 198.42），质量 2PP 通过（Capability 74.2822、Δ−0.1433）。Graph 仅 <384 短请求有效，**S1 P95 保持 eager，不采用**；证据 `Clef-27B/logs/gdn-data-path-20261115/graph-sizes-test/`、`graph-buckets-test/`、`graph-dense-test/`，报告 `docs/graph-acl-2026-11-16.md`。
- 2026-11-16 18.8 ms 非 kernel 残余 profile：307 token CPU 每层约 2.0 ms×64≈129 ms，高于设备 compute 77.3 ms，CPU dispatch 为临界路径；967 token 设备 compute 144.5 ms、CPU backbone 128.5 ms，非 kernel 约 12.5 ms。joint pooler（含 head）CPU 10.9/9.4 ms，head 设备仅约 2 ms；scheduler 0.13–0.17 ms。结论：残余主要在 vLLM eager 逐层 dispatch 与模型侧 joint head/pooler/schema，不在 vLLM-Ascend kernel；优先级为 graph/ACL graph、head/pooler 向量化、schema cache；S1 P95 仍设备侧。证据见 `Clef-27B/logs/gdn-data-path-20261115/profile-residual/`，报告见 `docs/profile-residual-2026-11-16.md`。
- 2026-11-16 MC2 down+allreduce 原型：`npu_mm_all_reduce_base` 在 `ascend910_93` 缺 `MatmulAllReduce` 内核（161001）；`aclnnGroupedMatMulAllReduce` 单组调用 SIGSEGV/挂起；`MLPRowParallelOp` 的 reduce_scatter 仅 graph mode + `mlp_tensor_parallel_size>1` 可用，当前 eager 服务不可用。**MC2 down+allreduce 负结果，不采用**；量化已被赛方禁止，长序列转向 BF16 NZ、布局/融合与调度优化。证据见 `Clef-27B/logs/gdn-data-path-20261115/mc2/`，报告见 `docs/mc2-down-allreduce-2026-11-16.md`。
- 2026-11-16 MatMulReduceScatter V2 探针：`torch_npu.npu_mm_reduce_scatter_base(..., comm_mode="aiv")` 在 A3/TP2/BF16 真实 down_proj 上可用；`RS+AllGather` 308/970 token 约 1.04–1.06×、1596 0.97×；仅本地 RS 1.10–1.23×，但需跨层 Dense SP；输出 max_abs 0.03125，不逐位。当前不接入；证据 `Clef-27B/logs/gdn-data-path-20261115/mc2-rs/`，报告 `docs/mc2-reduce-scatter-2026-11-16.md`。
- 2026-11-15 GDN 数据路径：在决策路径之上默认启用 **BIC**（`bsnd` + `skip_initial_state` + `skip_conv_state`）。NPU 契约探针（含 10,294 token、空 segment）与全量 OpenDecisions 均逐位一致；6 对交错 A/B 6/6 更快，p95 中位 **−7.01%（−13.81 ms）**，吞吐 +0.378 decisions/s。正常入口 S1 P95 **198.42 ms**（上一提交入口 213.56，−7.09%），Capability 74.425479；ERQA 400 题 R/BIC 概率逐位一致（219/400）。seqs4 C8 **14.652 dec/s**、p95 913.93 ms，C16 14.689。Ascend C `ClefGdnQkvNorm` 原型短序列 2×、长序列慢；2026-11-16 kernel v2 + cap1600 已接入（见上一条）。v3 安装器支持 v2 树从已验证 preimage 升级。证据见 `Clef-27B/logs/gdn-data-path-20261115/`，报告见 `docs/gdn-data-path-2026-11-15.md`。
- 2026-11-19 candidate S1 最小验收组合（cc34 CPU 冻结，设备未执行）：f810 正式全 S1≥3 与四 phase 仍用 cc7 runner `7d32e5c3…`/manifest `0338f87b…`、v5 监督器 `172c2838…`、phase v3 `888a8d7f…`，单轮 flag 仅 f810 且非最终。candidate aa3e 只冻结 S1 最小组合：独立入口 `runtime/candidate_service_phase.py`（SHA `1f98dba9…64d7`，默认 plan-only，复用原 d0_stop/launcher/两个已核 installer/phase 门，停后未知端口或客户端或源状态只记 blocked）、派生 manifest `31202d27…d2a4`（base `0338f87b…`，仅 13 处白名单）、v5 同协议 3 轮与 finally 回 f810 clean 恢复。CPU 25 项故障测试、真实 plan-only CLI、远端 13 项 pin 通过；执行 stage `/workspace/yangchen/work/final-acceptance-cc34-640ac23cac9aaa9a937832c2361dddd74007d195`，报告 `docs/final-acceptance-combo-2026-11-19.md`。候选 S1 无收益则不采用，S2/ERQA 用 f810 原组合；四 phase 候选变体待 S1 有收益后再补。
- 2026-11-13 用户补充资源限制：允许并行，全部推理与评测进程合计限用 **A3 单物理卡内的两个逻辑 NPU**，禁止跨物理卡。TP1、TP2 及并发调优均在此范围内进行，记录实际设备映射和进程配置。
- 2026-11-12 用户已选择新机器现有的 CANN 9.2.1，并采用 PyTorch 2.10 配套环境。
- 环境安装、服务运行及精度性能结论使用实际日志和检查结果标注完成状态。

## 阶段提交与推送

- 每完成一个可验证的小阶段，或取得有实际证据的阶段性进展，及时整理相关代码、文档和验证记录，执行 `commit + push`，保持小步、可回溯的提交。
- 小阶段提交与推送按本约定直接执行。提交说明写清本阶段的具体改动和验证结果；提交前完成对应的必要检查。
- 提交内容聚焦当前阶段，保留用户已有修改，凭据及模型权重大文件按仓库现有约定管理。推送使用正常快进流程，遇到远端更新先核对并整合。

## 当前机器与工作目录

| 项目               | 当前值                                                  |
| ------------------ | ------------------------------------------------------- |
| SSH 目标           | `root@199.88.54.221`                                    |
| 跳板               | `113.47.8.48:2234`，使用平台即时发放的跳板账号及令牌    |
| 远端工程           | `/workspace/yangchen/Clef-27B-ascend`                   |
| 模型目录           | `/workspace/yangchen/models/Clef-27B`                   |
| 日志目录           | `/workspace/yangchen/logs`                              |
| 下载及构建中间文件 | `/workspace/yangchen/work`                              |
| 计划 Python 环境   | `/workspace/yangchen/venvs/clef-py312`                  |
| 共享数据           | `/workspace/shared_assets/datasets`，按平台说明只读使用 |

SSH 模板如下。凭据通过当前会话或平台提供，仓库保存脱敏连接信息。

```bash
ssh -J '<跳板账号>:<临时令牌>@113.47.8.48:2234' root@199.88.54.221
```

新机实测为 Ubuntu 22.04.5 / aarch64，A3 单物理卡内两个 `Ascend910_9382` 逻辑 NPU，各 64 GiB HBM。`npu-smi` 卡号为 7，Chip 0 / 1 对应逻辑设备 0 / 1、`Phy-ID` 14 / 15；资源统计按 **1 物理卡、2 逻辑 NPU** 记录。容器资源为 128 GiB 内存和 32 核等价 CPU 时间配额。设备编号和资源配额在实例变化后重新查询。

## 固定的软件组合

| 组件                             | 目标                                       |
| -------------------------------- | ------------------------------------------ |
| CANN Toolkit / A3 Ops / NNAL ATB | 9.2.1，新机器已经安装                      |
| Python                           | 独立 CPython 3.12 环境，安装时记录补丁版本 |
| torch                            | `2.10.0+cpu`                               |
| torch_npu                        | `2.10.0.post7`                             |
| torchvision / torchaudio         | `0.25.0` / `2.10.0`，采用对应 CPU 构建     |
| Transformers                     | `5.19.0`                                   |
| Triton Ascend                    | `3.6.0.post2`                              |
| FLA NPU A3                       | `flash-linear-attention-npu-a3==26.9.1`    |
| flash_attn_npu                   | `0.5.0`                                    |
| vLLM                             | `24420b7a341af37a9bc3186e6fc6062512e0aa58` |
| vLLM-Ascend                      | `89eb8a5e7a76d085b453a1aeadb6883e12d5ea2d` |

构建前加载所选 CANN 与 ATB 的环境脚本，安装 CMake 至少 3.26.1、Ninja 及官方 A3 配方要求的编译依赖。按固定源码的 `empty` 构建流程安装 vLLM，随后构建 Ascend 插件与 A3 扩展。运行依赖由 constraints 固定，构建隔离环境的依赖单独记录。验收包含 `pip check`、实际版本、加载路径和 NPU 运算结果。

## 模型与入口

- 模型以 `Clef-27B/logs/model_manifest.json` 中的逐文件 revision、大小和 SHA-256 为准。
- `scripts/download_model_manifest.py` 使用系统 Python 标准库下载，支持 `.part` 续传、并行 HTTP Range 和 SHA-256 校验。成功报告保存到工程或远端日志目录。
- `Clef-27B/start.sh` 和 `inference.py` 是比赛入口。目标命令为 `bash start.sh --model-dir xxx --output-dir xxx`，模型路径支持相对和绝对路径；保留 `MODEL_DIR` 与 `PYTHON_BIN` 调用方式。参数、补丁初始化、服务加载与日志产物均须实测验收。
- 目标默认地址为 `http://127.0.0.1:18143`，提供 `GET /health`、`GET /v1/decide/info`、文本 `POST /v1/systemone` 与图文 `POST /v1/decide`。当前启动器的 `--output-dir`、默认端口及新增接口规范列为待适配项，完成状态以代码和实际请求证据为准。
- 模型加载必须覆盖 `joint_head_config.json`、`joint_head.safetensors` 和完整主干，确认识别为 `ClefFlashForJointSchema`。
- 首轮配置为 BF16、TP1、eager、max model len 1024、batch 1、纯文本，关闭 prefix caching 和 chunked prefill。
- 服务模型名统一为 `clef`。请求中的 `joint` 采用默认值，三种决策由模型联合决策头计算。
- 在相同输入上检查 `choice`、`noul`、`score`、候选概率、多字段和 token/span 对齐，再扩展 TP2 与多模态。

就绪响应包含 `status=ready`、`model`、`revision`；`/v1/decide/info` 返回模型身份、任务能力、协议版本和缓存配置。图文路径包含图像处理及视觉推理适配。

## 比赛质量、性能与交付

- Clef-27B 按多模态型号准备 **OpenDecisions + ERQA**。OpenDecisions 提交完整 Calibration、Intelligence、Capability 及 accuracy、score_mae；ERQA 提交图文准确率和分项结果。固定数据 revision、split、模板、评分器、样本范围及失败处理。
- `accuracy` 按 0～1 记录，Calibration / Intelligence / Capability 按 0～100 分记录，`capability=(calibration+intelligence)/2`；Score MAE 单独保留等级单位，执行正确率保留有效性判定及分母。
- 性能提交 **S1：单并发端到端 P95（ms）**，**S2：最大有效吞吐（decisions/s）及对应并发**，并保留完整并发扫描、逐请求起止时间和优化前后对照。有效性阈值、扫描范围及统计参数按主办方最终协议固定。
- 提交 `start.sh`、`inference.py`、`README.md`、`metrics.csv`、`results.csv`、`result.png`、`npu_device_call.png`、`logs/`、`optimization.md`；`src/` 与 `model_agent_pr.md` 为可选。日志须含命令、实际配置、逐题预测与完整概率、执行状态、错误和原始输出；截图来自真实执行并与 CSV 和日志对应。
- 每模型、每数据集一行记录提交版本的实测汇总，保留基线/优化值、样本数、准入阈值与结论。用户提供的 `metrics.csv` 五列长表与七列宽表格式并列保存在规则文档，最终导出格式待主办方模板确认；现有 CSV 的历史格式单独保留。
- 规则正文要求 910C，示例资源为 A3；当前机器原始标识为 `Ascend910_9382`。继续保存真实硬件、设备编号及资源输出，910C/A3 的赛事认可映射待核实。

## 数据、验证与记录

- 转换数据与缓存放在 `/workspace/yangchen` 下，保留共享数据来源、revision、split、模板和哈希。
- 比赛验收按本次补充规则的 OpenDecisions / ERQA 与 S1 / S2 组织。Decision Index 0.2.1、Typesafe WorkflowEvals 保留为模型发布方对照评测，BFCL、API-Bank 等按 Decision Index 子项报告；保留各套件原生评分和聚合，既有执行说明见 `docs/evaluation-protocol.md`。
- 早期探索评测结果仅作历史归档，赛事提交采用指定协议下的提交版本实测值。OpenDecisions 数据集作者公布的 Clef 原始服务输出参考为 Calibration 82.3、Intelligence 65.8、Capability 74.1；论文 Table 3 的作者额外温度校准后参考为 83.4、65.4、74.4。引用时注明来源和校准条件，评测配置分别记录原始与校准后结果；主办方正式 baseline 按赛事资源确认。
- OpenDecisions 当前公开版本为 1.0.1（2026-10-15），历史初版为 1.0（2026-10-12），均为 2,352 条；1.0.1 公告确认题目、标签及原有评分保持一致。复现同时固定数据与评分器 revision、哈希和 split，版本及论文分数差异的证据见 `docs/submission-rules-2026-11-13.md` 第 10 节。
- 发布方对照评测的两套数据快照已于 2026-11-13 下载并校验，目录为 `/workspace/yangchen/datasets/clef-official-20261113/`。逐文件来源、revision、大小和服务器 SHA-256，以及 Decision Index 官方解压哈希与行数对照见 `Clef-27B/logs/official-datasets-download-20261113.json`；模型评测状态单独记录。
- 2026-11-13 发布方对照运行保持 A3 / TP2 / BF16 / eager / 4096 / 单序列服务。WorkflowEvals 采用官方 `questions=one` 完成全部 705 case，原生评分与配置见 `Clef-27B/logs/workflowevals-baseline-20261113.json`；Decision Index 0.2.1 已于 15:55 按用户要求终止评测及 supervisor，部分输出保留在 `/workspace/yangchen/logs/official-baseline-20261113/decision-index/`，终止证据见 `Clef-27B/logs/decision-index-stopped-20261113.json`。该次运行按未完成记录。
- 2026-11-13 15:58—16:05 全新 OpenDecisions 1.0.1 全量原始概率评测已完成，2,352/2,352 成功、0 错误。Calibration 82.7941、Intelligence 66.0568、Capability 74.4255；accuracy 1906/2352，Score MAE 0.401389（588 条）。固定版作者 `score.py` 与 EvalScope 未舍入三项指标完全一致。输出目录为 `/workspace/yangchen/logs/opendecisions-baseline-20261113-1600/`，证据见 `Clef-27B/logs/opendecisions-baseline-20261113/`，复现方式见 `docs/opendecisions-baseline-2026-11-13.md`。每次重跑使用新空目录，保存完整原始概率与执行覆盖。
- 下载完成以逐文件 SHA-256 校验报告为准。依赖完成以版本与 NPU smoke 日志为准。完整 baseline 以服务、精度和性能记录为准。
- 新机器的当前状态与后续执行计划见 `docs/new-machine-plan.md`，旧机器记录在文档中保留采集时间和来源。
- 修改任务涉及的代码和文档，保留用户已有修改。提交前检查文档链接、命令语法以及本次实际执行产生的结果。
