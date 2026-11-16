# Clef-27B Ascend 当前进展与优化记录

更新：2026-11-15（UTC）。本文面向未参与本项目的外部专家，目标是让读者在不看历史对话的情况下理解：当前提交配置、实测结果、端到端时间花在哪、已经排除了哪些方向、以及需要外部判断的问题。

---

## 0. 摘要

- **模型**：Clef-27B，Qwen3.5 主干（64 层：48 层 GDN 线性注意力 + 16 层 full attention），hidden 5120，intermediate 34816，BF16 权重约 51 GB，27.48B 参数。
- **环境**：昇腾 A3 单物理卡内 **2 个逻辑 NPU**（`Ascend910_9382`，各 64 GiB HBM），容器 32 核等价 CPU、128 GiB 内存；CANN 9.2.1、PyTorch 2.10.0+cpu、torch_npu 2.10.0.post7、vLLM `24420b7a…`、vLLM-Ascend `89eb8a5e…`。
- **当前提交配置**：TP2 / BF16 / eager / max_model_len 4096 / max_num_seqs 1 / prefix cache 关 / chunked prefill 关；GDN 决策专用 prefill 路径默认开启，并在 2026-11-15 默认启用 **BIC** 数据路径（BSND、`initial_state=None`、`conv_states=None`）。
- **S1（单并发端到端 P95，2,352 题）**：baseline **217.036 ms** → GDN 决策路径 **213.560 ms** → 2026-11-15 正常入口 BIC **198.420 ms**（同窗口受控 A/B BIC vs R p95 中位 **−7.01%**；另一 flag 文件窗口 BIC 185.62、同窗口 R 197.95）。跨运行绝对值受机器漂移影响，采用结论以同进程 A/B 为准。
- **2026-11-15 数据路径**：BIC 全量 2,352 逐位一致（Capability 74.425479）；ERQA 400 题 R/BIC 概率逐位一致（219/400）；seqs4 C8 **14.652 dec/s**、p95 913.93 ms，C16 14.689；融合 Ascend C 原型短序列 2×、长序列变慢且 Q/K 1 ulp，不接入。[报告](gdn-data-path-2026-11-15.md)
- **S2（最大有效吞吐）**：baseline **5.715 dec/s @ C16**（seqs1，实际 batch 恒为 1）→ seqs4 **14.125 dec/s @ C16**（推荐工作点 C8：14.049 dec/s、p95 950.7 ms），约 **2.47×**。
- **质量**：C1 与 baseline **逐位一致**（Capability 74.425479、accuracy 1906/2352、Score MAE 0.401389@588）；合批后 C8 为 Capability 74.195104（−0.230）、accuracy 1903、max 概率差 0.097399、6 条 choice 翻转、2 条 noul 跨 0.5。
- **主要瓶颈**：设备 kernel 约占客户端 80%；其中 MLP 32.07 ms（31.6%）、TP2 allreduce 20.40 ms（20.1%）、GDN 主核 10.52 ms + 布局 6.73 ms、其他 MatMul（attention 投影等）14.82 ms。MLP 的 kernel 是 **MTE2 权重搬运受限**（占用 0.994），有效带宽约 0.53 TB/s；2026-11-15 微基准测得纯 HBM 读带宽 **1.26–1.32 TB/s**，即 MLP 只达到约 **40–42%**，瓶颈在 MTE2 访问效率/tiling/格式，而非 HBM 上限。[HBM 微基准](hbm-bandwidth-2026-11-15.md)
- **18.8 ms 非 kernel 残余（2026-11-16）**：307 token 时 CPU 每层约 2.0 ms ×64≈129 ms，高于设备 compute 77.3 ms，CPU dispatch 是临界路径；967 token 时设备 compute 144.5 ms、CPU backbone 128.5 ms，非 kernel 约 12.5 ms。joint pooler（含 head）CPU 10.9/9.4 ms，head 设备计算仅约 2 ms；scheduler 仅 0.13–0.17 ms。残余主要在 vLLM eager 逐层 dispatch 与模型侧 joint head/pooler/schema，不在 vLLM-Ascend kernel。[Profile 定位](profile-residual-2026-11-16.md)
- **Graph/ACL graph（2026-11-16，显式尺寸复核）**：默认 `max_num_seqs=1` 只捕获 `[1]`，307/967 实际未 replay；显式捕获 `[307,967]` 后，307 mean/p95 **94.34/96.38 ms**（邻近 eager 109.95–119.90 / 123.36–130.84），967 基本持平（169.23/170.07）。但粗桶全量 S1 p95 **244.53 ms**、密桶 265.80 ms（eager 198.42），质量 2PP 通过（Capability 74.2822、Δ−0.1433）。Graph 只对 <384 短请求有效，**S1 P95 保持 eager，不采用**。[Graph 复核](graph-acl-2026-11-16.md)
- **Host 微优化复测（2026-11-16）**：schema cache v2 在 s1/307 反而慢约 5%（126.23/137.61 vs eager 119.90/130.84），969 持平；BF16 NZ 服务化（真实的 `Qwen2MoeMLP.forward` hook）307 mean −1.6% 但 p95 +1.4%，969 mean/p95 −0.7%/−0.8%；均逐位一致但**不采用**。第一版 NZ hook 放在未使用的 `AscendQwen3_5DecoderLayer` 上，是死代码；修正后才得到真实数据。[Host 复测](host-opt-2026-11-16.md)
- **Full attention prefill cache skip（2026-11-16）**：`PrefillNoCache` 下跳过 `reshape_and_cache`，冻结 8/8 逐位；307 mean 129.09 vs control 125.06（更慢），969 170.85 vs 172.51（−1.0%，噪声内）；未过 ≥3% 门，**不采用**。[Cache skip 报告](attn-prefill-cache-skip-2026-11-16.md)
- **GEMM+AllReduce 同层分块重叠（2026-11-16）**：down_proj 按 token 切 2/4 块、compute/comm 双 stream；308 token 0.42×/0.29×、970 0.80×/0.46×、1596 0.79×/0.69×，全部慢于一次完整 AllReduce；不采用。[分块重叠探针](npu-gemm-overlap-probe-2026-11-16.md)
- **MLP-only Dense SP（2026-11-16）**：`npu_all_gather_base_mm` + down ReduceScatter 数值可用（max_abs 0.015625、mean_abs≈2e-5），但端到端 **0.88–0.98×**（1596 token 0.878×），比当前 AllReduce 慢；**不采用**，完整 Dense SP 优先级下调。[MLP SP 探针](dense-sp-mlp-probe-2026-11-16.md)
- **Joint head 跨请求批处理（2026-11-16）**：实现 padding/mask 的 `forward_batch` 与批量 lexical embedding；B4 C4 **13.442 vs off 13.637 dec/s**（变慢），B8 小样本 18.614 vs 18.158（+2.5%），但全量 OD C16 **16.4685 vs 16.4638**（+0.03%）无实际收益；质量 2PP（Capability 74.2174，Δ−0.208）。**不采用**，源码已恢复。[报告](joint-head-batch-2026-11-16.md)
- **BF16 B8/B16 合批（2026-11-16）**：B8 C16 **16.464 dec/s**、p95 1804 ms，Capability 74.2676（−0.158）；B16 C64 **17.110 dec/s**、p95 6088，Capability 74.2241（−0.201）；OD 三项均在 2PP 内。ERQA 合批已补：B8 C16 **218/400=0.5450**（−0.25 pp）、B16 C32 216/400（−0.75 pp），均在 2PP 内。S2 推荐 B8 C16，B16 只在只看最大吞吐时使用。[B8/B16 报告](batching-seqs8-16-2026-11-16.md)
- **已排除且不可提交（赛方不允许量化）**：W8A16、W8A8、W8A8+SmoothQuant、seqs4+W8A8 叠加等量化方案，即使质量落在 2PP 内也不得提交；历史数据仅作负结果。非量化方向中 TP1（S1 P95 264.3 ms，劣于 TP2）已排除。
- **待判断（禁止量化前提下）**：MLP 的 MTE2 效率能否通过 BF16 NZ/tiling/预取从 0.53 提到 0.8+ TB/s（纯 HBM 已测 1.26–1.32 TB/s）；allreduce 能否与计算重叠；batch-invariant GEMM 是否可行；`S1 ~198 ms` 是否已接近该硬件上 27B 模型单请求的物理下限。

---

## 1. 任务、指标与约束

### 1.1 比赛与数据

- 文本质量：**OpenDecisions 1.0.1**，完整 2,352 题，作者评分器给出 Calibration / Intelligence / Capability（0–100），另记 accuracy（0–1）、Score MAE（等级单位，588 条）与校准 TVD（129 条）。
- 图文质量：**ERQA** 官方 400 题，accuracy 与 8 类分项。
- 性能：**S1 = 单并发端到端 P95（ms）**；**S2 = 最大有效吞吐（decisions/s）及对应并发**。
- 质量门槛：2026-11-16 用户明确 **2PP 口径**：比赛**不需要温度校准**；OpenDecisions **Intelligence、Calibration、Capability** 与 **ERQA 总体准确率**相对 baseline 的**绝对波动均不超过 2 个百分点**（ERQA 2PP = 0.02）。项目早期自设的 0.03（概率差）/ 0.05（score 期望值）工程门槛仍单独保留，不与 2PP 混用；主办方正式 baseline 与模板仍待确认。

### 1.2 资源与软件

| 项 | 值 |
| --- | --- |
| NPU | A3 单物理卡，2 个逻辑 `Ascend910_9382`，各 64 GiB HBM；禁止跨卡 |
| CPU / 内存 | 32 核等价（cgroup `3200000/100000`）、128 GiB |
| CANN | 9.2.1（Toolkit / A3 Ops / NNAL ATB） |
| Python | 3.12（独立 venv `clef-py312`） |
| PyTorch / torch_npu | 2.10.0+cpu / 2.10.0.post7 |
| Transformers | 5.19.0 |
| vLLM / vLLM-Ascend | `24420b7a341af37a9bc3186e6fc6062512e0aa58` / `89eb8a5e7a76d085b453a1aeadb6883e12d5ea2d`（editable 安装） |
| FLA NPU A3 | `flash-linear-attention-npu-a3==26.9.1` |

### 1.3 模型

| 项 | 值 |
| --- | --- |
| 架构 | Qwen3.5 + 独立联合决策头（`ClefFlashForJointSchema`） |
| 层 | 64 层：48 `linear_attention`（GDN）+ 16 `full_attention`（每 4 层 1 层） |
| hidden / intermediate | 5120 / 34816（TP2 每 rank 17408） |
| GDN | 16 key heads、48 value heads、key/value head_dim 128、conv kernel 4、每序列状态约 1.5 MiB |
| 参数 | 27,484,784,884（BF16 约 51.2 GiB；TP2 每 rank 约 25.7 GiB） |
| 视觉 | Qwen3_VisionTransformer，约 4.6 亿参数 |

---

## 2. 服务与入口

- 统一入口：`bash Clef-27B/start.sh --model-dir … --output-dir …`，默认端口 18143；`--config s1` 为提交配置（TP2/4096/seqs1/缓存关）。
- `inference.py` 在启动时：
  - 安装并校验 **GDN 决策专用 prefill 补丁**（preimage SHA 校验、幂等、可回退）；
  - 写入 `CLEF_GDN_DECISION_PREFILL=1`；
  - 在 `launch-manifest.json` 记录 `gdn_decision_prefill` 的启用状态、三个源码文件的 SHA-256、补丁模块哈希。
- 源码树为固定 vLLM-Ascend 89eb8a5 的 editable 安装；GDN 补丁默认应用（新的提交状态）。
- 接口：`GET /health`、`GET /v1/decide/info`、`POST /v1/systemone`（文本）、`POST /v1/decide`（文本/图文）。

---

## 3. 当前结果

### 3.1 S1（单并发、全量 2,352、端到端）

| 项 | baseline（2026-11-14） | 当前（2026-11-15 正常入口） | 变化 |
| --- | ---: | ---: | ---: |
| 配置 | TP2/4096/seqs1/eager/缓存关 | 同 + GDN 决策路径 | — |
| 成功 / 失败 | 2,352 / 0 | 2,352 / 0 | — |
| wall | 423.846 s | 408.328 s | −3.66% |
| mean | 180.069 ms | 173.600 ms | −3.59% |
| p50 | 205.034 ms | 195.567 ms | −4.61% |
| **p95** | **217.036 ms** | **213.560 ms** | **−1.60%** |
| p99 | 222.139 ms | 223.563 ms | +0.64% |
| C1 吞吐 | 5.549 dec/s | 5.760 dec/s | +3.80% |

- 受控 A/B（同一服务进程内 6 对、588 题等距抽样、交替顺序）：p95 中位 **−3.80%（−7.68 ms）**，6/6 对 ON 更快，吞吐中位 +0.17 dec/s。
- 同日另一次 C1 全量测得 p95 201.4 ms、6.000 dec/s（同代码路径），说明单次绝对值有约 ±3–5% 的机器漂移；**结论以受控 A/B 为准**。

### 3.2 S2（并发扫描、全量 2,352）

| 并发 | baseline seqs1 | 当前 seqs4 | 吞吐变化 |
| ---: | ---: | ---: | ---: |
| C1 | 5.549 dec/s，p95 217.0 ms | 6.000 dec/s，p95 201.4 ms | +8.1% |
| C2 | 5.626，p95 426.5 | 6.208，p95 398.5 | +10.3% |
| C4 | 5.673，p95 831.2 | 9.819，p95 570.7 | +73.1% |
| **C8** | 5.584，p95 1723.8 | **14.049，p95 950.7** | **2.52×** |
| **C16** | **5.715，p95 3292.7** | **14.125，p95 1848.3** | **2.47×** |

- **S2 最大值**：baseline 5.715 @ C16 → 当前 **14.125 @ C16**（2.47×）。
- **推荐工作点 C8**：14.049 dec/s、p95 950.7 ms；C16 只多 0.5% 吞吐，p95 几乎翻倍。
- 合批证据：C1→C8 吞吐 2.34× 无法由排队解释；C16 服务日志出现 `Waiting: 8 reqs`；C1/C2 逐位、C4 起出现质量差异。

### 3.3 质量（作者评分器，全量 2,352）

| 点 | Capability | accuracy | Score MAE（588） | max 概率差 | >0.03 | >0.05 | choice 翻转 | noul 跨 0.5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline C1 | 74.425479 | 1906 | 0.401389 | 0 | 0 | 0 | 0 | 0 |
| 当前 C1 | 74.425479 | 1906 | 0.401389 | **0** | 0 | 0 | 0 | 0 |
| 当前 C2 | 74.425479 | 1906 | 0.401389 | **0** | 0 | 0 | 0 | 0 |
| 当前 C4 | 74.444325 | 1904 | 0.401281 | 0.102702 | 18 | 11 | 4 | 0 |
| 当前 C8 | 74.195104 | 1903 | 0.400693 | 0.097399 | 25 | 7 | 6 | 2 |
| 当前 C16 | 74.224612 | 1905 | 0.400941 | 0.097399 | 21 | 8 | 4 | 2 |

- **C1 与 C2 逐位一致**：seqs4 在单并发/低并发下不改变输出；S1 与任何 C1 质量评测不受影响。
- 合批差异从 C4 开始，根因是**批内位置相关的 GEMM 累加顺序**（BF16 舍入差经 64 层与联合头放大），不是输入、候选映射或模型逻辑错误。
- ERQA（GDN 路径 ON/OFF）：400/400 逐题一致，accuracy 219/400 两边相同。

---

## 4. 端到端阶段预算

### 4.1 阶段表（TP2，固定 307 tokens，关闭采集）

| 阶段 | ms | 占客户端 |
| --- | ---: | ---: |
| 客户端完整请求 | 126.25 | 100% |
| HTTP（含服务队列与执行） | 126.14 | 99.9% |
| 服务端队列 | 0.006 | 0.005% |
| 引擎 e2e | 123.07 | 97.5% |
| RUNNING（执行+完成等待） | 122.42 | 97.0% |
| **设备 kernel 合计** | **101.64** | **80.5%** |
| ├ MLP（up/gate + down） | 32.07 | 25.4% |
| ├ allreduce（TP2） | 20.40 | 16.2% |
| ├ GDN 主核 | 10.52 | 8.3% |
| ├ GDN q/k/v/state 布局 | 6.73 | 5.3% |
| ├ 其他 MatMul（attention 投影等） | 14.82 | 11.7% |
| └ 其他 kernel（Norm/SwiGLU/Conv/Slice 等） | 17.10 | 13.5% |
| 联合头 | 1.98 | 1.6% |
| RUNNING 内非 kernel 残余（调度/同步/等待） | 18.80 | 14.9% |
| 队列/引擎残余/HTTP 前端 | 3.83 | 3.0% |

> 设备行来自 2026-11-15 的 `op_statistic` 求和（307-token 固定模板，GDN 决策路径开启）；阶段行来自 2026-11-12 原生 before/after 差分。两者在 307 tokens 下吻合：81.23（compute）+ 20.40（comm）= 101.63。

### 4.2 设备 kernel 关键项（307 tokens，TP2 rank0）

| 项 | ms/请求 | 说明 |
| --- | ---: | --- |
| MLP up/gate（`307×5120 × 17408×5120`） | 21.34 | `aic_mte2_ratio≈0.994`，权重搬运受限 |
| MLP down（`307×8704 × 5120×8704`） | 10.73 | 同上 |
| allreduce | 20.40 | 约 130 次/请求；与计算可能重叠 |
| GDN 主核（ChunkGatedDeltaRuleFwd） | 10.52 | 48 层 × 20 请求 = 960 次 |
| q/k 转置两向 | 3.67 | BSND 候选可去掉 |
| v 转置两向 | 2.19 | BSND 候选可去掉 |
| state 转置 | 0.87 | 初始状态侧，仍保留 |
| AddRmsNorm | 2.80 | |
| CausalConv1d | 2.29 | |
| SwiGlu | 1.48 | |
| attention score | 0.92 | QKV/O 投影在“其他 MatMul”内 |

### 4.3 口径与 caveat

- 307-token 是固定探针；生产 S1 的真实输入平均 **446.8 tokens**（p50 约 401，最大 1,595），因此 213.56 ms 是真实分布上的 P95，而阶段表是 307-token 的可比分解。
- profiler 有开销；kernel 时间是求和时间（通信与计算可能重叠），不等于关键路径贡献。
- 2026-11-12 的 TP2 客户端 126.25 ms 与 2026-11-15 的 307-token profile 设备 101.64 ms 是同一量级但不同窗口，不能与 446.8-token 的 213.56 ms 直接相减。

---

## 5. 本轮优化：做了什么、结果、结论

### 5.1 GDN 决策专用 prefill（已采用，S1 默认启用）

**背景**：服务是 pooling 决策模型，一次前向直接出结果（`output_tokens=0`），**从不 decode**。但固定 vLLM-Ascend 的 GDN prefill 分支仍无条件：

1. `gather_ssm_states` 取初始状态（无历史时为全零，零填充不读 cache）并对输入状态做 `transpose(-1,-2).contiguous()`；
2. `output_final_state=True` 计算并返回 final state；
3. final state 转置后 `scatter_ssm_states_` 写回；
4. fused kernel 的 `disable_recompute=True` 每层分配并写出 `g_cumsum` 与 `A`（只供反向）。

**改动**：在 guard 成立时消除 2/3/4（保留 1）。guard 为宿主侧条件：`runner_type == "pooling"`、chunked prefill / prefix cache / spec decode 全关、`num_prefills>0`、`num_decodes=0`、`num_decode_tokens=0`、`num_spec_decodes=0`。

**正确性**：8/8 冻结用例；全量 2,352 题 OFF vs ON **逐位一致（max 0.0）**，与冻结 baseline 逐位一致；ERQA 400 题 ON/OFF 400/400 一致。

**性能**：同进程 6 对交错 A/B，p95 中位 **−3.80%（−7.68 ms）**，6/6 对更快。

**负结果（重要）**：第一版 guard 读设备张量 `prefill_has_initial_state.any().item()`，按 `(id(metadata), data_ptr)` 缓存；因为 GDN metadata 是逐层的，缓存每层失效，等于每请求 48 次设备同步——6/6 对**变慢**，p95 中位 **+6.75%（+14.8 ms）**。最终版改为纯宿主侧判断才转正。

### 5.2 seqs4 批处理（建议采用）

- 只把 `--max-num-seqs` 从 1 改为 4，其余不变；客户端 C1–C16 各跑全量。
- 吞吐 C1 6.000 → C8 14.049 → C16 14.125 dec/s；C8 是拐点。
- 质量：C1/C2 逐位；C4+ 出现合批差异；C8 Capability 74.195（−0.230）、max 0.0974、6 choice 翻转、2 noul 跨 0.5。
- **S1 不受影响**（C1 下 batch 恒为 1），S2 提升约 2.47×。
- 建议：提交服务配置改为 `--max-num-seqs 4`；S2 报 C8（或按规则报 C16=14.125），质量报 C1（逐位）与 C8（合批边界）两组。

> 2026-11-16 赛方不允许量化；以下 W8A16/W8A8/W8A8-ND/W8A8-NZ/SmoothQuant 仅作为历史负结果，不再作为提交候选。

### 5.3 W8A16（拒绝）

- 复用 W8A8 候选的 INT8 权重，只把 128 个 gate/up 的 `quant_model_description.json` 条目改为 `W8A16`；服务以 `--quantization ascend` 正常加载。
- 局部 MLP 链（layer31/rank0，307 tokens）：BF16 0.5101 ms vs W8A16 0.4829 ms，**1.057×**，6/6 轮更快；relative RMSE 1.303%。
- 全量：Capability **74.2434（−0.182）**、accuracy 1902（−4）、p95 216.71 ms、5.633 dec/s（单次，噪声内）。
- 结论：质量优于 W8A8，但局部加速太小，全量无收益；**不作为性能候选**。

### 5.4 W8A8（拒绝；部分工作来自并行会话）

- gate/up 选择性 W8A8_DYNAMIC（NZ 格式）：局部 MLP 链 1.20–1.38×，layer31 局部 relative RMSE 10.9–12.5%。
- 全量（另一会话）：Capability **74.0716（−0.354）**、accuracy 1902；ND 格式候选 +0.28% 吞吐、P95 −0.94%（−1.95 ms）、p50 −3.02 ms；ERQA 219/400。
- 本轮在 NZ 候选上测 C1：4.961 dec/s（比 BF16 C1 6.000 慢）、Capability 74.0881、max 0.3969——**NZ 的格式/调度在单请求下反而拖累**，与 ND 候选的 +0.28% 相反。
- 结论：W8A8 在 batch 1 的端到端收益约 1%，与 1.35× 的局部收益不匹配；NZ 不可用；ND 质量代价 −0.35，且现因赛方量化限制不可提交。

### 5.5 激活平滑（仿真，未实现）

- 对 gate/up 输入按通道做 SmoothQuant 式缩放（`x/s`、`W*s`），再走 W8A8。CPU 仿真（真实捕获输入，float32 matmul，隔离量化误差）：

| 点 | W8A16 | W8A8 | W8A8+smooth（shared α=0.7） |
| --- | ---: | ---: | ---: |
| layer0 / od-0105 | 0.48% | 1.49% | 1.00% |
| layer31 / od-0105 r0 | 1.29% | 11.78% | **2.37%** |
| layer31 / od-0105 r1 | 1.28% | 11.84% | 2.27% |
| layer63 / od-0105 | 0.31% | 1.12% | 0.68% |
| layer31 / erqa-0089 | 1.26% | 11.04% | 2.38% |

- 平滑把最坏层误差降 5×，同时保留 INT8 速度；实现需要逐层激活捕获、把 `1/s` 折进 MLP 前 RMSNorm、把 `s` 折进 gate/up 权重列，并重新量化转换（预计约 2 小时）。
- 未实现、未做全量质量验证。

### 5.6 seqs4 + W8A8 叠加（拒绝）

| 点 | 吞吐 | p95 | Capability | 相对 baseline | max 概率差 | choice 翻转 | noul 跨 0.5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| seqs4 BF16 C8 | 14.049 | 950.7 | 74.1951 | −0.230 | 0.0974 | 6 | 2 |
| **seqs4 W8A8 C8** | **14.771** | **834.0** | **73.6987** | **−0.727** | 0.2692 | 14 | 9 |
| seqs1 BF16 C1 | 6.000 | 201.4 | 74.4255 | 0 | 0 | 0 | 0 |
| seqs1 W8A8 C1 | 4.961 | 220.1 | 74.0881 | −0.337 | 0.3969 | 15 | 5 |

- 批处理下 MLP 转为计算受限，INT8 生效：C8 吞吐 +5.1%、p95 −12.3%。
- 但质量代价叠加后不可接受（−0.727、max 0.269、14 翻转、9 noul 跨 0.5）。
- 结论：**seqs4 + W8A8(NZ) 不采用**；当前最优质量/性能点是 **seqs4 + BF16**。

### 5.7 TP1 vs TP2（TP2 保持）

同配置、同 2,352 请求、同计时工具，只改 TP：

| 指标 | TP1 | TP2 |
| --- | ---: | ---: |
| wall | 421.61 s | 408.33 s |
| 吞吐 | 5.579 dec/s | 5.760 dec/s |
| mean | 179.25 ms | 173.60 ms |
| p50 | **180.10 ms** | 195.57 ms |
| p95 | 264.34 ms | **213.56 ms** |
| p99 | 319.90 ms | **223.56 ms** |
| max | 399.09 ms | 246.21 ms |

- TP1 中位数更快（短请求不付 allreduce/同步），但尾部显著更差（单卡承担全部计算与权重带宽）。
- S1 看 P95、S2 看吞吐，TP2 均胜出；且 ERQA 16,384 + 16 图配置下 TP1 显存余量不足。**保持 TP2**。

---

## 6. 已排除的方向

| 方向 | 实测 | 排除原因 |
| --- | --- | --- |
| W8A16（权重 INT8、激活 BF16） | 局部 1.057×；全量 Capability −0.182、无吞吐提升 | 加速太小 |
| W8A8（gate/up，NZ） | 局部 1.20–1.38×；C1 约 1%；NZ C1 反而更慢 | 端到端收益低、NZ 格式惩罚 |
| W8A8（ND） | +0.28% 吞吐、P95 −0.94%；Capability −0.354 | 质量代价大、收益小 |
| 激活平滑（W8A8） | 仿真误差 11.8%→2.4% | 未实现；预计端到端上限约 1%（C1）或 +5%（C8） |
| seqs4 + W8A8 | C8 14.771、p95 834；Capability −0.727、max 0.269 | 质量不可接受 |
| TP1 | S1 p95 264.3 ms | 尾部与显存劣势 |
| 整模型 OM/ATC | 未尝试 | vLLM 动态 shape/合批与固定 OM 冲突，风险高 |
| C++ 扩展替 CPU 热点 | 未尝试 | CPU 编码仅 1.07 ms（307）/7.87 ms（3868），收益远小于 200 ms 延迟 |
| 设备张量 guard | p95 +14.8 ms（6/6 更慢） | 逐层同步吞掉收益 |

---

## 7. 当前瓶颈与需要外部判断的问题

### 7.1 MLP（32.07 ms，31.6% kernel / 25.4% 客户端）

- kernel 是 **MTE2 权重搬运受限**（`aic_mte2_ratio≈0.994`，cube 利用率约 98.8%）。
- 每请求每 rank 要流约 **17.1 GB** MLP 权重（up/gate 11.4 + down 5.7），实测有效带宽约 **0.53 TB/s**。
- 2026-11-15 纯 HBM 微基准：`copy` 1.26–1.27 TB/s、AI Core `sum` 读 1.30–1.32 TB/s；MLP 仅约 40–42%，**不是 HBM 上限**。
- 2026-11-15 BF16 NZ 局部实验：`F.linear` 接受 FRACTAL_NZ29，全部输出与 ND **逐位一致**；cold 307 token **1.26–1.32×**（名义 509–532→671–677 GB/s）、969 **1.06×**、1595/10294 约 **1.00–1.03×**。S1 P95（969 token）的 MLP 已约 256–299 TFLOPS、转为 **compute-bound**，NZ 对 P95 约 **1.9%**，是质量中性候选而非主杠杆。[报告](mlp-bf16-nz-2026-11-15.md)
- 2026-11-15 纯 BF16 matmul 上限：Cube 约 **300–320 TFLOPS**（10294/4096 方阵）；969 的 gate/up 282、down 258，MLP 链 0.954 ms/层 ≈ 纯 matmul 之和 0.948，链开销 <1%。BF16 tiling 的理论上限约 **3–4% 端到端**，M padding 仅 969→1024 约 2%；长序列大杠杆是 **INT8 gate/up**（约 10%，质量门槛）与 **MC2 down+allreduce**（约 10%，质量中性）。[报告](mlp-cube-ceiling-2026-11-15.md)
- 问题：
  1. BF16 NZ 预打包、更大 tile、双缓冲、L2 预取能把 MLP 有效带宽从 0.53 提到多少？
  2. 纯 HBM 1.27 TB/s 对应的 gate/up 理论上限约 9.0 ms（当前 21.34）、MLP 合计约 13.5 ms（当前 32.07）；matmul 实际能兑现多少？
  3. 量化已被赛方禁止，W8A8/SmoothQuant 不再考虑；BF16 下剩余空间主要来自 tiling/预取、NZ 布局和融合算子，是否值得继续？

### 7.2 TP2 allreduce（20.40 ms，20.1%）

- 与 MLP 同量级；是否在关键路径取决于重叠。
- 2026-11-16 MC2 原型：`npu_mm_all_reduce_base` 在 `ascend910_93` 上缺 `MatmulAllReduce` 内核（161001）；`aclnnGroupedMatMulAllReduce` 单组调用分别 SIGSEGV / 挂起；`MLPRowParallelOp` 的 reduce_scatter 仅在 graph mode 的 `mlp_tensor_parallel_size>1` 下可用，当前 eager 服务无法使用。**MC2 down+allreduce 在当前 A3 栈不可用**，保留为负结果。[报告](mc2-down-allreduce-2026-11-16.md)
- 2026-11-16 补充：`npu_mm_reduce_scatter_base(..., comm_mode="aiv")` 的 MatMulReduceScatter V2 在 A3/TP2/BF16 真实 down_proj 上可用；RS+AllGather 308/970 token 约 1.04–1.06×、1596 0.97×；仅本地 RS 1.10–1.23×，但需要跨层 Dense SP。输出 max_abs 0.03125，不逐位；当前不接入。[RS 探针](mc2-reduce-scatter-2026-11-16.md)

### 7.3 GDN 剩余布局（6.73 ms，6.6%）

- q/k 转置 3.67 + v 转置 2.19 + state 转置 0.87。
- **已存在但未做受控 A/B 的候选**：`CLEF_GDN_BSND=1` 传 sequence-major q/k/v、`layout="BSND"`，去掉 4 组 q/k/v 转置；tuning4 已确认真实服务命中 1000/rank、8/8 逐位一致，但因 A/B 顺序混淆被拒绝。
- 问题：是否值得把 BSND 改成运行时开关，做一次同进程 6 对 A/B（预期 3–4%）？还是直接做 Ascend C 融合算子（conv 输出 → q/k/v 布局 + l2norm + g cast）？

### 7.4 合批数值（S2 的质量代价）

- C1/C2 逐位；C4+ 出现批内位置相关的 BF16 累加差异，C8 Capability −0.23、max 0.0974。
- 问题：CANN/AscendC 是否存在可用的 batch-invariant GEMM/内核？若没有，S2 的 2.47× 是否只能以 C8 质量边界为代价？

### 7.5 S1 的物理下限

- 51 GB 权重、batch 1、每请求每 rank 流 25.7 GB；按不同 HBM 峰值假设（0.8/1.2/1.6 TB/s），仅权重搬运的理論下限约 32/21/16 ms/rank。
- 当前 S1 p95 213.56 ms（真实 446.8 tokens）。问题：这个水平是否已接近该硬件上 27B 模型单请求的合理下限？剩余空间主要在 kernel 效率、通信重叠还是调度？

---

## 8. 可借鉴的外部经验（graspnet-npu）

已阅读队友的 `graspnet-npu` 仓库（Atlas 300V Pro / 310P3），可迁移的模式：

1. **按阶段拆延迟预算**（1000-run，p50/p90；本项目的阶段表同理）。
2. **CPU/NPU 按实测放置**（FPS 放 CPU、three_nn/interp 放 CPU、group_points CPU + OM NPU）。
3. **单遍融合**（Cylinder 多遍 → C++ single-pass 4-depth，5.9 ms）——对应 GDN q/k/v 布局融合。
4. **固定 shape OM + CPU 预算索引**——对应按长度分桶的 MLP/FFN 子图 OM，但 vLLM 集成风险高。
5. **近似算法 + 质量实测**（QuickFPS 替代精确 FPS，AP 反而 +0.96pp）。
6. **Ascend C 自定义算子 skill**：msopgen → ccec → OPP → ACLNN ctypes 全流程与坑（INT64→INT32、opset、format），可直接用于融合 GDN/FFN 算子。
7. **首跑 vs 稳态分开**（OM 首跑 9.4 s 含初始化，稳态 68 ms）。

> `G00R-npu` 仓库返回 404（两个账号下均未找到），未能分析。

---

## 9. 证据、提交与复现

### 9.1 关键文档

| 文档 | 内容 |
| --- | --- |
| `docs/baseline-2026-11-12.md` | 首轮工程 baseline、TP1/TP2、多模态、HCCL |
| `docs/opendecisions-baseline-2026-11-13.md` | OpenDecisions 全量与作者评分 |
| `docs/erqa-baseline-2026-11-14.md` | ERQA 400 题与适配 |
| `docs/tuning4-2026-11-14.md` | seqs1/B1 最终配置、S1/S2、合批根因 |
| `docs/gdn-decision-prefill-2026-11-15.md` | GDN 决策路径：改动、guard 迭代、A/B、入口集成 |
| `docs/profile-entry-2026-11-15.md` | 提交入口 profile、TP1 vs TP2 |
| `docs/mlp-optimization-2026-11-15.md` | 并行会话的 W8A8/FFN/ND 工作 |
| `docs/mlp-w8a16-2026-11-15.md` | W8A16 候选、激活平滑仿真、MLP 收益上限 |
| `docs/batching-seqs4-2026-11-15.md` | seqs4 C1–C16 扫描、质量、seqs4+W8A8 叠加 |
| `docs/cpu-npu-reference-2026-11-14.md` | CPU 独立参考与输入等价 |

### 9.2 证据目录（仓库内）

- `Clef-27B/logs/gdn-decision-20261115/`：补丁、8 例、全量逐位、6 对 A/B、入口 manifest、服务恢复。
- `Clef-27B/logs/gdn-decision-20261115/profile-entry-20261115/`：设备预算、op_statistic、operations.csv.gz。
- `Clef-27B/logs/mlp-w8a16-20261115/`：W8A16 候选、微基准、全量质量、平滑仿真。
- `Clef-27B/logs/mlp-20261115/`：并行会话的 W8A8/FFN/ND 证据。
- `Clef-27B/logs/batching-s1-20261115/`：C1–C16 原始报告与评分。
- `Clef-27B/logs/batching-w8a8-20261115/`：seqs4+W8A8 原始报告与评分。
- `Clef-27B/logs/tp1-s1-20261115/`：TP1 全量 S1。

### 9.3 提交（分支 `codex/tuning-perf2a-20261113`）

`a13a39e`、`b32eef8`、`ffae77b`（CPU 参考）；`6fd128d`、`525a1dd`（GDN 决策路径与入口）；`98b41bf`、`3700c9b`（profile/W8A16）；`1726cc6`（TP1）；`dbc4804`（批处理）；`603ea19`（seqs4+W8A8）。另有并行会话的 MLP 提交 `3cddd6f`、`cba9ea5`、`24a6460`、`120e3b8`、`6b69089`。

### 9.4 关键复现命令

```bash
cd /workspace/yangchen/Clef-27B-ascend
source scripts/activate_env.sh

# 提交配置启动（s1：TP2/4096/seqs1/缓存关，GDN 决策路径默认启用）
bash Clef-27B/start.sh --model-dir /workspace/yangchen/models/Clef-27B \
  --output-dir <run>/entry --config s1 --port 18143

# 全量 S1（C1）
python runtime/od_full_benchmark.py \
  --sample-file /workspace/yangchen/logs/tuning/20261114-013000-fix4/opendecisions/all2352.json \
  --url http://127.0.0.1:18143/v1/systemone --concurrency 1 --report <run>/c1.json

# 批量并发扫描（seqs4 服务，--max-num-seqs 4）
for C in 1 2 4 8 16; do
  python runtime/od_full_benchmark.py --sample-file <all2352.json> \
    --url http://127.0.0.1:18143/v1/systemone --concurrency $C --report <run>/c$C.json
done

# 作者评分与逐题差异
python runtime/od_raw_acceptance.py \
  --data /workspace/yangchen/datasets/clef-full-20261112/full-data/open-decisions2352.jsonl \
  --scorer /workspace/yangchen/logs/opendecisions-baseline-20261113-1600/author-score.py \
  --template /workspace/yangchen/logs/opendecisions-baseline-20261113-1600/evalscope/predictions/clef/clef_open_decisions_full2352_default.jsonl \
  --run baseline=/workspace/yangchen/logs/tuning/20261114-013000-fix4/opendecisions/s1-final-full.json \
  --run candidate=<run>/c8.json --reference baseline --output <run>/c8-quality.json
```

---

## 10. 当前环境状态

- 8000 端口：按 phase-0 快照恢复的 baseline 服务（argv/cwd/env 与原始一致，`/health` 200，8/8 冻结用例通过）；当前 PID 记录在 `/workspace/yangchen/logs/current-service.json`。
- 源码树：固定 vLLM-Ascend 89eb8a5 + 已应用的 GDN 决策补丁（`device_op.py`、`gdn.py`、`patch_qwen3_5.py`）；`gdn_decision_flags.py` 在位。
- 候选 checkpoint（未部署为默认）：
  - `/workspace/yangchen/models/Clef-27B-mlp-gateup-w8a8`（gate/up W8A8，NZ）
  - `/workspace/yangchen/models/Clef-27B-mlp-gateup-w8a16`（同权重，description 改为 W8A16）
- 运行中服务只使用 A3 单物理卡内的两个逻辑 NPU，未跨卡。
