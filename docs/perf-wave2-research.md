# Clef-27B A3 性能第二轮调研（2026-11-18）

> 本文只做联网调研与只读代码分析，**不占用 NPU、不启动或停止任何服务、不运行推理/kernel**。
> 所有"预期收益"都是基于 profile 数值的推算，**不是实测值**；每条都标注证据等级。
> 基线：`ao/docs-new-handoff/perf-next`（`df7cb1a`）。原始撰写窗口的 S1 p95 为 157.745 ms（969 token）、S2 17.446 dec/s @ B16 C64。
>
> **整合更新（2026-11-18，perf/wave2-npu 实测后）**：§0.1 的设备预算已替换为 conv+QKV 融合与 chunk128 生效后的窗口
> （`docs/perf-wave2-20261118.md`；原始 144.60 ms 窗口保留在 `docs/profile-final-2026-11-17.md`）。
> 当前提交口径：S1 p95 **155.86 ms**（五轮中位，155.72–157.50）、**8.849 dec/s**；S2 B16 C64 **17.462 / 17.475 dec/s**。
> T3 已由服务内打点实测（§3），结论从"建议"降级为"已测关闭（除内核化）"；T1/T2/T4/T5 仍为建议，未实测。

## 0. 结论摘要

### 0.1 当前 p95 的预算（967 token，单请求，rank0，已折算为每请求）

数据来自 2026-11-18 的 torch-profiler 窗口（967 token、10 请求、双 rank，converter 后每请求），
即 conv+QKV 融合与 chunk128 **生效后**的默认栈；旧窗口（`analysis-969b`，144.60 ms）见 `docs/profile-final-2026-11-17.md`：

| 类别 | ms | 说明 |
| --- | ---: | --- |
| 设备 compute union | **132.69** | 下面是它的分项 |
| ├ MLP gate/up `967,5120×17408` | 41.50 | cube_utilization 97.3%、mac_ratio 0.80，**已到 Cube 上限** |
| ├ GDN `ChunkGatedDeltaRuleFwd` ×48 | 17.97 | cube 99.2% 但 mac_ratio 0.067、aiv_scalar 0.37 |
| ├ MLP down `967,8704×5120` | 20.54 | cube 94.3%，上限 |
| ├ MatMul `5120→8192`（GDN in_proj_qkvz）×48 | 14.88 | ≈266 TFLOPS，**MTE2 ratio 0.93，访存受限** |
| ├ MatMul `5120→3072`（q_proj + GDN out_proj）×64 | 8.02 | |
| ├ AddRmsNorm ×128 | 5.93 | 真实算子 0.0355×128 ≈ 4.5 ms，贴 HBM 上限 |
| ├ `ClefGdnConvQkv` ×48（融合） | 4.17 | 原 `ClefGdnQkvNorm` 4.68 + `CausalConv1d` 3.82 已合一 |
| ├ MatMul `5120→7168` ×16 | 4.47 | |
| ├ SwiGlu ×64 / FusedAttn ×16 | 2.39 / 1.67 | |
| └ 其余小算子（layernorm、gating、split_qkv_rmsnorm、Transpose、Index…） | ≈11.2 | |
| allreduce comm union | **32.85** | **非重叠（暴露）14.17 ms**；AICPU kernel 130 次/请求、均值 161 µs |
| 非 kernel 空隙（设备空转等 CPU） | **11.84** | joint head/pooler + schema/tokenizer |
| **kernel window span** | **158.70** | 含 profiler 开销；生产 p95 为 155.86 |

**关键结构判断**：S1 p95 已基本是纯设备侧问题。可分的三块是
① 暴露的 allreduce ≈14 ms、② 设备空转等 CPU ≈12 ms、③ 设备 compute ≈133 ms（**MLP/AddRmsNorm/matmul 都已到各自硬件上限**）。
wave-1 已把 ③ 里的可选项全部吃完（见第 6 节复核），所以 wave-2 的可行空间**只在 ① 和 ②**，再加上 S2 的通信份额。

### 0.2 排序后的建议（详见第 1–5 节）

| # | 杠杆 | 目标 | 预期 S1 p95 | S2 | 成本 | 置信度 |
| --- | --- | --- | ---: | --- | --- | --- |
| **T1** | 把 MLP `down_proj` 做成 **MatMul+AllReduce+Add+RMSNorm 单核**（Ascend C） | 吃掉 64 次 allreduce 的暴露延迟 + 64 次 AddRmsNorm | **−5 ~ −9 ms（3–6%）** | +3~8% | 高 | 机制有上游同构实现，**本形状未证实** |
| **T2** | **K 维切分** allreduce ∥ matmul 流水（区别于已证伪的 M 维分块） | 隐藏 ① 的 50% | −3 ~ −7 ms（2–4.5%） | +2~6% | 中 | **未证实**（M 维版本已实测退化） |
| ~~T3~~ | ~~压缩 head/pooler 与前端的关键路径往返~~ | ~~吃 ② 的 4–6 ms~~ | **已测关闭** | — | — | **实测：head 只剩 ≤2 ms，且需内核化（并入 T1 批次）** |
| **T4** | `ATB_OPSRUNNER_KERNEL_CACHE_*` 从 5/1 放大（对标已生效的 ACLNN cache 修复） | 消除 ATB 算子的新 shape 首用开销 | 未知，0 ~ −2 ms | 0~1% | **极低（纯 env）** | **未证实，先探查** |
| **T5** | 分桶 ACL graph 阶梯 `{128…1600}` + `enable_static_kernel` / SuperKernel | 只服务 <768 token 与 S2 | **≈0**（969 设备受限） | +2~5% | 中 | 短请求收益已证实，**对 p95 预期为 0** |

如果只做一件事：**T1**。如果只做一件便宜事：**T4 探查**。T3 已由 2026-11-18 的服务内打点关闭，见第 3 节。

---

## 1. T1 融合 `MatMul + AllReduce + Add + RMSNorm`（MLP down_proj）

### 机制

当前每层有 **2 次 TP allreduce**（attention/GDN out_proj 后 1 次、MLP down_proj 后 1 次），
实测每请求 **130 次**、单次 **0.162 ms**、comm union 35.0 ms、**非重叠 17.2 ms**。
把 down_proj 的「GEMM epilogue → AllReduce → 残差加 → RMSNorm」收进一个 Ascend C 核：

1. GEMM 的输出 tile 是**逐步完成**的，allreduce 可以按 tile 流式发起，其延迟被同一个核里的 MAC 掩盖，
   而不是等整个 GEMM 结束后再暴露 0.162 ms × 64；
2. 顺带消掉 64 次独立 launch 与 64 次 `AddRmsNorm`（5.96 ms / 真实 4.5 ms）的读回；
3. 少一次 `[967,5120]` 的中间写回 + 读回（各 9.9 MB）。

### 对本栈的适用性

- 结构完全对得上：`Qwen3_5DecoderLayer` 的 MLP 是「column-parallel gate/up → SwiGLU → row-parallel down → allreduce → AddRmsNorm」，
  64 层同构，形状固定（TP2 后 `967×8704×5120`，`K=8704, N=5120`）。
- 已在 `llama/` 以外的 Qwen3 系上有上游同构实现（见下），说明可行域存在；但**上游那个版本是 decode 的 row=1 形状**，
  我们这里是 prefill 的 M=967，需要重写 tiling。

### 上游证据

- **vllm-ascend PR #12649**（open，target `releases/v0.22.1rc`，分支 `matmul-allreduce-rmsnorm-910c`）：
  「Add 910C matmul-allreduce-rmsnorm fusion op for Qwen3 down-projection」，Python 侧 pass 名 `allreduce_rmsnorm_fusion_pass.py`，
  语义正是 matmul→allreduce→add→rmsnorm，挂在 `ascend_compilation_config` 后面按硬件门控。
  **注意两个负面信息**：① 尚未合并；② review 指出 host adapter 允许 batch=2 而 AIV kernel 只支持 row=1，
  即**它是 decode 形状的算子，不能直接套到我们的 967 token prefill**。
  <https://github.com/vllm-project/vllm-ascend/pull/12649>
- **vllm-ascend PR #19033**（open，「V0.23.0: Optimization of TPOT under low concurrency conditions」）：
  910C / Qwen3.5-35B-A3B / TP2 / full-decode graph / static kernel，报告
  `AllReduce+RMSNorm` 融合 + SuperKernel 计算通信重叠 + side-stream 共享专家重叠，合起来把 **c=1 TPOT 从 ~8.2 ms 降到 ~7.6 ms（−7%）**、
  MTP=4 时 7.0→5.1（−27%）；`fuse_allreduce_rms: true` 放在 `ascend_compilation_config`。
  这是**低并发下通信融合值 7% 量级**的最直接外部佐证，和我们 C1 的处境同构。
  <https://github.com/vllm-project/vllm-ascend/pull/19033>

### 与现有代码的关系（已核查）

- 固定版 `89eb8a5` 树里**没有** `fuse_allreduce_rms` / `allreduce_rmsnorm` 相关代码
  （`grep -rn "fuse_allreduce_rms\|allreduce_rmsnorm" vllm_ascend/` 无结果），
  `AscendCompilationConfig` 只有 `enable_npugraph_ex / enable_static_kernel / enable_super_kernel / fuse_norm_quant / fuse_qknorm_rope / fuse_muls_add`
  （`vllm_ascend/ascend_config.py:116-131`）。所以**不能靠开关拿到**，要么自己写核，要么等上游。
- 我们的 MLP NZ hook（`CLEF_MLP_BF16_NZ`）已经把 down_proj 权重转成 FRACTAL_NZ，
  自己写 Ascend C 核时必须沿用 NZ 布局，否则会丢掉那 −1.56% / −3.06% 的收益。

### 预期收益

- 保守：非重叠通信从 14.17 ms 中吃掉 64/130 次的一半 → 14.17 × 0.5 ≈ **−5 ms**，再减 AddRmsNorm 中约 64 次（≈2.2 ms 真实值）→ **约 −7 ms**。
- 上限：把 64 次 down_proj allreduce 的暴露全部吃掉 → **−9 ms 上下**（−6%）。
- （2026-11-18 整合更新：按融合后窗口的 14.17 ms 非重叠通信与 5.93 ms AddRmsNorm 重算；原文的 −6~−12 ms 基于旧窗口 17.2 ms。）
- S2：B16 下通信份额更大（payload 71.7 MB/次），但 fixed cost 摊薄，融合主要省 launch；估计 **+3~8%**。

### 成本与风险

- 成本**高**：需要 Ascend C 写一个 M∈[1,4096] 的 GEMM + 核间 allreduce 的融合核，还要处理 TP2 的 rank 分组、
  workspace、NZ 布局、以及 chunk 128 / cap 1600 的现有分支。
- 风险：**中高**。数值必须保持 allreduce 的加法顺序（否则逐位比对会破，但 2PP 口径大概率仍过）；
  与 `ACLNN_CACHE_LIMIT` 类似，新算子会有自己的 shape 缓存问题，M=967 有 755 个不同长度，必须一开始就验证冷/热两态。

### 验证方法

1. 先做**离线微基准**（不接服务）：`down_proj(M=967) + allreduce + add + rmsnorm` 融合核 vs 现有 4 个算子串联，
   M 取 307 / 967 / 1595，warm/cold 各 10 次。门槛：cold 时间 < 现有串联的 0.85 倍，且 `max_abs ≤ 1 ulp`（BF16）。
2. 单单算子过关后再接服务：固定 307/967 窗口 → 全量 S1 → OD 全量 2,352 逐题概率 → ERQA 400。
3. 决策门：S1 p95 改善 ≥ 3% **且** 三项指标绝对波动 ≤ 2PP **且** choice 翻转 ≤ 现有 chunk128 的 9 条。

---

## 2. T2 按 K 维切分的 allreduce ∥ matmul 流水

### 机制

已证伪的是 **M 维（按 token）分块**：`docs/npu-gemm-overlap-probe-2026-11-16.md` 把 `[M,K]` 按行切 C 块，
C=2/4 在 M=970 时是 **0.80× / 0.46×**（更慢），原因是小 allreduce 固定开销重复 + event/stream 同步。

**K 维切分是另一回事**：row-parallel 层的 `allreduce(h)` 结果要作为**下一个 column-parallel matmul 的 A 矩阵**，
而 matmul 沿 K 自然累加。因此可以把 allreduce 结果按 hidden 切成 K 的两段：
`allreduce(h[:, 0:2560])` 一回来就发起 `matmul(A[:, 0:2560], W[0:2560, :])`，
同时让 `allreduce(h[:, 2560:5120])` 在通信流上跑。

- collective 次数翻倍（130 → 260），单次 payload 减半；
- 但 A3 上单次 allreduce 是**固定开销主导**：9.9 MB 只要 0.143–0.162 ms ≈ 61 GB/s，
  而 `quantized_all_reduce.py` 的注释给出「**A3 (910C) bf16 HCCS 带宽约为 A2 的 2.6 倍，而 int8 路径的每次 collective 固定成本不变**」，
  说明 9.9 MB 的 0.14 ms 里绝大部分不是带宽。**固定开销翻倍几乎必然吃掉收益** —— 这是这条路线最大的风险，也是它和 T1 的本质差别（T1 是把 allreduce 塞进已有核，不增加 collective）。

### 适用性与预期

- 适用：`allreduce → AddRmsNorm → next layer in_proj`（GDN `in_proj_qkvz` `K=5120`、attention qkv `K=5120`）这条链上。
- 预期：乐观 **−3 ~ −7 ms**（融合后窗口暴露通信 14.17 ms 的一半），悲观 **0 或负**（固定开销翻倍）。
- 成本：中（host 侧切片 + 双流 + event，不改 kernel）。
- 置信度：**未证实**，且相邻证据（M 维分块 0.46–0.80×）是负面的。
- 状态（2026-11-18）：**未测**。前置探针需要占用 NPU，本轮整合窗口不再启动新上板实验，留待下一轮按 §9 执行。

### 更省的做法（建议先做这个再决定）

先跑一个 **纯通信探针**，只回答一个问题：A3 上 `dist.all_reduce` 的耗时随 payload 的曲线。
若 `[967,2560]`（4.95 MB）的耗时 ≥ `[967,5120]`（9.9 MB）的 0.9 倍，则固定开销主导，**T2 直接放弃**；
若接近 0.55 倍，说明带宽主导，T2 才值得做。这个探针只要几分钟、不接服务、不改代码。

---

## 3. T3 head/pooler 与前端的关键路径往返（**已测–关闭，除内核化**）

### 原假设（2026-11-18 之前）

原调研假设 `joint_pooler` 8.835 / 11.470 ms（307/967 rank0/1）、其中 `joint_head` 6.728 / 8.989 ms，
而这些数值来自开启 CPU observation 与 trace 的 profile 窗口，**含采集开销**；
后续服务内打点（下节）证明真实 head 只有 4.78–5.05 ms、pooler 5.90–6.25 ms。

单字段 head 后像（`runtime/joint_single_field/clef_joint_schema.py`）**已经做过一轮向量化**：
`segment_mean` 代替逐 span `_mean_span`、`_indices()` 预计算索引、`async_tensor_h2d` 做 H2D、
逐 option 的 Python 循环已消除。所以"再向量化一遍"空间有限，剩下的集中在**同步与往返**上。

### 新证据：Ascend 上的同步代价被严重低估

- vllm-ascend issue **#21740**（open）：「Qwen3.6 long-context prefill is ~36% slower on Model Runner V2:
  the blocking seq-len sync removes stream overlap」。算子时间几乎不变（+0.4%），
  但 **compute stream 空闲从 6.07 s 涨到 23.22 s**，损失约 17 s 的 allreduce/compute 重叠；
  根因是 `_update_seq_lens_cpu()` 里的 `num_computed_tokens_event.synchronize()`。
  <https://github.com/vllm-project/vllm-ascend/issues/21740>
- 这条直接对应我们 profile 里的 **非 kernel 12.48 ms**：那不是"CPU 太慢"，而是**流被同步点排空**。

### 2026-11-18 服务内实测（结论：关闭）

用环境变量 `CLEF_JOINT_TIMING=1` 在单字段后像内打点（默认关闭，测后已还原，SHA 回到 `f8101b98…`），
967 token、4 候选、12 次请求：

| 段 | ms（每请求） |
| --- | ---: |
| `parse_joint_schema` | 0.01 |
| `_indices`（numpy + H2D） | 0.23–0.28 |
| `lexical`（embed + segment_mean） | 0.62–0.71 |
| **head 合计** | **4.78–5.05** |
| ↳ `norm_mem` / `options` / `fields_route` / `decoder` / `final` | 0.26 / 1.10–1.18 / 0.42–0.45 / 2.23–2.33 / 0.62–0.67 |
| `act`（softmax + 收集） | 0.15–0.17 |
| **pooler 合计** | **5.90–6.25** |

- 原文档引用的 `joint_pooler` 8.835 / 11.470、`joint_head` 6.728 / 8.989 来自开启了 CPU observation + trace 的
  `profile-final` 窗口，**含采集开销**，不能作为优化基线。真实服务内 head 是 4.78–5.05 ms，pooler 5.90–6.25 ms。
- 结构上约 100 个小算子，`decoder` 每层约 15 个（3 norm + 2 自注意力投影 + 跨注意力 in/attn/out + FFN 3 + 4 残差），
  耗时与 FLOP 无关，全部是下发开销；设备侧对应算子合计仅约 1.5–2 ms。
- **CPU 卸载否决**：`memory_projection` 是 967×5120×1024 ≈ **10.1 GFLOP**，必须留在 NPU；
  搬到 CPU 的剩余部分仍有约 70 MFLOP（4 层 FF 4096 + 2 层 routing 跨注意力），`OMP_NUM_THREADS=1` 下估算 3–4 ms，
  加上 9.9 MB D2H（约 0.4 ms），净收益不足且数值路径改变。
- 跨请求 head 批处理已在 2026-11-16 实测无 S2 收益（`docs/joint-head-batch-2026-11-16.md`）。

### 原建议项（保留为后续评估记录）

| 项 | 机制 | 原预期 | 实测状态 |
| --- | --- | ---: | --- |
| 去掉 head 内任何 `.item()` / `.tolist()` 中转 | 一次 sync = 排空整条流 | 1–3 ms | **已排除**：`act` 0.15–0.17 ms，`tolist()` 0.009 ms（`docs/frontend-timing-2026-11-17.md`） |
| logits 只回传 `[n_options]` fp32 | 减少 D2H 副本 | 0.3–1 ms | **已排除**：pooler 输出已是 `[O]`，D2H 可忽略 |
| 把 `evidence_routing` 与小 matmul 合并成 1–2 个 `baddbmm` | 减少 `aclnnAddmm` 条数 | 1–2 ms | **未做**；`options` 段 1.10–1.18 ms 是全部 routing 成本的上界 |
| schema 索引/长度跨请求复用 | 避免每请求重算 | 0–1 ms | **已排除**：`indices` 0.23–0.28 + `parse` 0.01 ms |

- **结论**：在纯 host 侧没有可拿的 3–5 ms；剩余只有把整个 head 融成 Ascend C 算子（预计 ≤2 ms），
  与 T1 属于同一批内核工作，**合并到 T1 一起评估**。
- 外部 issue 佐证同步代价量级（#21740），但本服务的 head 路径没有对应的阻塞同步点。

---

## 4. T4（**先探查**）ATB ops-runner kernel cache 只有 5/1

### 线索

服务环境里有：

```
ATB_MATMUL_SHUFFLE_K_ENABLE=1
ATB_OPSRUNNER_KERNEL_CACHE_GLOABL_COUNT=5      # 注意上游拼写就是 GLOABL
ATB_OPSRUNNER_KERNEL_CACHE_LOCAL_COUNT=1
ATB_STREAM_SYNC_EVERY_KERNEL_ENABLE=0
ATB_WORKSPACE_MEM_ALLOC_ALG_TYPE=1
```

（来源：`/workspace/yangchen/logs/current-service.json` 的 `environment`；本仓库 `inference.py` 的白名单只放行 `ATB_` 前缀，没有改这些值。）

这和 wave-1 **ACLNN executor cache 只覆盖约 145 个 shape、新 shape 首用固定多约 55 ms** 是同一类问题：
OD 的 prompt 长度有 **755 个不同值**，任何按 shape 建缓存的 executor，容量小于长度种类数就会持续 thrash。
ACLNN 侧把 limit 提到 1e6 之后拿到 **−21.8 ms p95**（`docs/aclnn-cache-shape-fix-2026-11-17.md`）。
**ATB 侧仍是 5/1，从未调过。**

### 适用性判断（关键不确定点）

profile 的 op 列表里没有显式的 ATB 算子名，但 `SwiGlu`（2.39 ms，64 次）与 `CausalConv1d` 的命名风格不像 aclnn，
`SwiGlu` 正是 ATB 的算子名。若 SwiGlu 走 ATB ops-runner，则它每遇到一个新 token 数都要重选 tiling ——
这和 ACLNN 的病症同构。

> **2026-11-18 交叉核对（负面先验）**：conv+QKV 融合与 chunk128 生效后，`SwiGlu` 仍是 **2.39 ms**（与 ACLNN 修复前完全相同），
> 这削弱了「ATB 在热路径且按 shape thrash」的先验。做 A/B 之前应先确认该 env 是否真被 ATB ops-runner 读取、
> `SwiGlu` 是否由 ATB 下发（例如打开 ATB 日志或按算子名查 profiler 的 module 归属），否则可能只是改了一个无人读的变量。
> 本项仍为**建议/未测**，且需要上板，本轮不执行。

### 做法与门槛

1. **探查（零风险、几分钟）**：设 `ATB_OPSRUNNER_KERNEL_CACHE_GLOABL_COUNT=1000000`、`LOCAL_COUNT=1000000`，
   跑固定 307/967 窗口 + 打乱顺序的全量 S1。若 p95 无变化 → 说明 ATB 不在热路径，**立即关闭该方向**。
2. 若有效，再确认 `ATB_MATMUL_SHUFFLE_K_ENABLE=1` 与 NZ 布局是否冲突。
- 预期：0 ~ −2 ms。成本：**极低**（纯 env，可回退）。
- 置信度：**未证实**。之所以仍然列进来，是因为它的成本/收益比是全部候选里最好的，且 wave-1 的同构经验收益极大。

---

## 5. T5 分桶 ACL graph 阶梯（只对 <768 token 与 S2 有意义）

### 机制与已证伪的边界

`docs/graph-acl-2026-11-16.md` 第 6 节已经做过正确的那一步：显式 `cudagraph_capture_sizes=[307,967]`，
结果 **307 token 快 14–21%（109.95 → 94.34 mean，p95 123.36 → 96.38）**，**967 token 持平**，
但**全量 S1 p95 反而变差**。

原因可以从 dispatcher 行为直接推出：`_is_compatible` 允许「请求 token 数 ≤ 捕获尺寸」，
所以 307 与 967 之间的一切请求（例如 500、700 token）都会 **replay 967 的 graph 并 padding 到 967**，
白算近一倍。**这不是"graph 不适合"，而是"捕获集只有两个点"。**

### 未做过的版本：阶梯捕获

- 捕获 `{128, 192, 256, 320, 384, 512, 640, 768, 896, 1024, 1280, 1600}`（12 个点，步长约 12–25%）。
  每个请求 replay 最小的 ≥ 自身长度的桶，**padding 浪费 ≤ 一个步长**。
- padding 的代价已有实测：967→1024 的 MLP `M padding` 冷态**只多出 0.4% 的收益**（`docs/gdn-conv-qkv-2026-11-17.md` 第 7 节），
  即额外的 5.9% 计算几乎被更好的 tiling 抵消，M padding 在冷态近似免费。
  注意该测量只覆盖 MLP 单层，全模型 padding 代价仍需在 T5 立项时单独测。
- 内存：2 个尺寸 0.24 GiB → 12 个约 1.5 GiB，64 GiB HBM 可承受；捕获时间约 3 s × 6 ≈ 20 s 启动开销。

### 收益与局限（必须诚实）

| 区间 | 请求占比 | 预期 | 依据 |
| --- | ---: | --- | --- |
| 0–768 token | 85.9% | **−10 ~ −20%** 延迟 | 307 token 实测 −14~−21% |
| 768–1200 | 12.9% | **≈0** | 967 实测持平（设备受限） |
| **S1 p95（969 token）** | — | **≈0** | 同上，**不要指望它降 p95** |
| 全量 mean / 吞吐 | — | −5 ~ −12% | 上面的加权 |
| S2 | — | +2 ~ +5% | dispatch 占比在合批下更低，收益打折 |

**结论定位**：T5 是 **mean/吞吐/S2 杠杆，不是 S1 p95 杠杆**。如果主办方按 `results.csv` 报 mean 或按 S2 排名，它有价值；
如果只看 p95，优先级排在 T1/T3 之后。

### 配套的 graph-模式专用特性（当前 eager 拿不到）

`https://docs.vllm.ai/projects/ascend/en/latest/user_guide/feature_guide/graph_mode.html`：

- **Npugraph_ex**（`ascend_compilation_config.enable_npugraph_ex`，FULL/FULL_DECODE_ONLY 默认开）：
  编译期把 add+RMSNorm 合成 `npu_add_rms_norm` 等，减少 launch；
- **Static kernel**（`enable_static_kernel`，默认关）：预编译定 shape 算子二进制，降低运行期开销；
- **Super Kernel**（`enable_super_kernel`）：在 ACLGraph 捕获期融合合格算子二进制，**要求 static kernel + npugraph_ex**；
  PR #19033 进一步说明它在 **CANN 9.1+** 上「随 static kernel/npugraph_ex 自动开启」，并做**流融合，让计算与通信重叠**。

这三项都只在 graph/compile 路径生效，**eager 下无法使用**；也就是说 T5 若不做，这些也一并拿不到。
反过来说，`fuse_qknorm_rope` / `fuse_muls_add` 已经在 `AscendCompilationConfig` 里默认 True，
但对我们**同样因为没有 graph 而完全不生效** —— 这是"eager 决策"的隐性代价，值得在 T5 里一起重新评估。

---

## 6. wave-1 已关闭项的外部复核（新增证据，不建议重开）

| 项 | 我们的结论 | 本次新增的外部/代码证据 |
| --- | --- | --- |
| int8/量化 allreduce | 未做 | `vllm_ascend/distributed/device_communicators/quantized_all_reduce.py` 头注释**明确排除 A3**：「A3 (910C) is rejected. Its bf16 HCCS bandwidth is about 2.6x that of A2 … the quantized path is a net loss at every payload size (measured 5–320 MiB across four different groups of 910C cards)」→ **结构性关闭，且顺带证明 A3 通信带宽充足、瓶颈是固定延迟** |
| Dense SP / FlashComm1 | 0.88–0.98×，全量 p95 +7.3% | 上游 PR #20756 的 dense SP 已移植实测；上游自己也写「A3 may be negligible」 |
| M 维 GEMM+AR 分块 | 0.29–0.80× | 见 T2 分析，固定开销主导的解释被 `quantized_all_reduce.py` 的 A2/A3 对比独立印证 |
| all-linear NZ | p95 +0.61 ms | 冷/热态差异的解释成立（warm 是 L2 伪影） |
| MLP 自定义 tiling | 冷态 285.9–303.4 TFLOPS ≈ 上限 | 与 4096³ 方阵 294.6 对比；另可用 profile 的 `aic_mac_ratio≈0.77 / aic_mte2_ratio≈0.94` 佐证 MAC 与 MTE2 已平衡 |
| AddRmsNorm 自定义 | 1117–1266 GB/s，贴 HBM | 一致 |
| HCCL `AIV` 展开 | 隔离 +7.3%、端到端噪声内 | 一致 |
| GDN chunk 32 | aclnn 161002 不支持 | 一致 |
| `TASK_QUEUE_ENABLE` / `OMP_NUM_THREADS` | 已设 1/1 | `Clef-27B/inference.py:943-944` 已确认 |

### 额外确认：prefix caching 对本服务是**语义阻塞**，不是性能问题

- `docs/upstream-npu-baseline.md:91` 记录：Mamba prefix caching 需要同时开 chunked prefill 并处理 block size 对齐。
- 更硬的阻塞在语义侧：本服务是 **pooling**，`JointSchemaHead.forward` 需要**整条序列每个 token 的 hidden states**
  来算 span 均值、候选 embedding 与 evidence routing。prefix cache 命中意味着被缓存的前缀**不再重新前向**，
  也就拿不到它们的 hidden states；KV/mamba state 缓存无法替代。
- **2026-11-18 实测确认**：在关闭 chunked prefill 的默认栈上追加 `--enable-prefix-caching`，服务启动即失败：
  `pydantic ValidationError: Chunked prefill is required for Mamba prefix caching.`；
  打开 chunked prefill 又会关闭 GDN 决策快速路径。公共 prompt 前缀仅约 **45 token**，收益上限远小于该组合的风险。
- 因此**不要投入** prefix caching；这一条从"未探索"改判为"已关闭（语义 + 启动约束）"。

### 2026-11-18 追加的两条硬事实（并入已关闭项）

- **GDN chunk**：96 / 160 / 192 / 256 全部返回 `aclnnChunkGatedDeltaRuleFwdGetWorkspaceSize failed: 161002`（FLA NPU 不支持）；
  128 是唯一快于 64 的受支持取值（1595 token device 0.521 vs 0.608 ms），且已是默认。**该项彻底关闭**。
- **S2 token 预算**：B16 C64 把 `max_num_batched_tokens` 从 16384 放大到 24576 后为 17.374 dec/s（16384 为 17.462 / 17.475），
  预算不是约束，**吞吐在 17.4–17.5 饱和**。

---

## 7. 其他 BF16 非量化方向（评估后不建议）

| 方向 | 判断 |
| --- | --- |
| 权重预取 / L2 驻留 | 每 chip 每 forward 从 HBM 流 27 GB ≈ 21 ms（1.27 TB/s），相对 136 ms compute 只占 15% 且大部分被 MTE2/MAC 流水掩盖；MLP warm(ND) 比 cold 快 3–5% 是 L2 伪影，不能当预取红利。**不建议** |
| SwiGLU 融合 | 2.39 ms（1.6%），且 SwiGlu 已是单算子。**收益 < 风险** |
| rope 融合 | 只有 16 个 full-attention 层，`split_qkv_rmsnorm_mrope_kernel` 已 0.92 ms。**不建议** |
| 减少 head 的候选计算 | 输入分布显示 4 候选占 43%，全量单字段、跨请求 head 批处理已实测无收益。**关闭** |
| 提示词/模板裁短 | 直接线性减 compute（969 token 是设备受限），但属于**质量改动**，需要完整 2PP + 逐题复核。**列为风险项，不作为性能手段** |
| 提高 conv+QKV / chunk128 的 cap（1600 → 更长） | S1 全部 2,352 条请求为 **145–1595 token**，已全部落在 cap 内；该方向只影响 ERQA 长尾。**S1 无关** |
| `max_num_batched_tokens` 放大 | B16 已用 16384（17.462 / 17.475），24576 实测 17.374，B32 C128 只到 17.364。**关闭** |

---

## 8. 上游观察清单（不要现在依赖，但要盯）

| 项 | 内容 | 状态 |
| --- | --- | --- |
| PR #12649 | 910C matmul+allreduce+add+rmsnorm 融合算子（Qwen3 down_proj） | **open，未合并**；row=1 形状（decode），需为 prefill 重写 |
| PR #19033 | 低并发 TPOT 优化：AllReduce+RMSNorm 融合、SuperKernel 计算通信重叠、side-stream | **open**；910C c=1 TPOT −7% |
| Issue #21740 | MRV2 长 prefill 因 `num_computed_tokens_event.synchronize()` 掉 36%，compute stream 空闲 6.07→23.22 s | **open**；直接支撑 T3 |
| PR #12317 | DP metadata 同步缓冲复用，NPU/HCCL 同步延迟 −25.5%~−31.5% | open，910B2 |
| PR #20454 | PCP prefill 通信重叠 + KDA AllReduce→ReduceScatter，TTFT −4.12% | merged |
| v0.30.0 release notes | 「Qwen3.8-27B: FLA NPU backend for GDN prefill; non-contiguous KV cache for hybrid linear attention」 | 已在我们的树上（`_cann_ops_custom` 有 `chunk_gated_delta_rule_fwd_h_vllm`） |
| v0.32.0rc1 | Qwen3.8-Flash-Next 的 GDN/PLE 缓存与 CPU 同步削减 | 值得跟进同类改动是否回移到 Qwen3.5 |

---

## 9. 建议执行顺序（含决策门）

1. **T4 探查（0.5 h，零风险）**：先确认 ATB ops-runner cache env 是否真被读取，再放大做固定窗口 A/B。无变化就永久关闭。
2. **T2 前置探针（0.5 h，零风险）**：`all_reduce` 耗时 vs payload 曲线。若固定开销主导，T2 关闭。
3. **T1 立项（3–5 d，高风险）**：先离线微基准过 0.85× 门槛再谈接服务；T3 剩余的「head 内核化」并入这一批一起评估。
4. **T5 视目标决定**：只有当报数包含 mean/吞吐或 S2 权重上升时才做阶梯 graph。

> 2026-11-18 整合更新：原第 3 步「T3 落地 0.5–1 d」已删除——服务内打点证明 host 侧无 3–5 ms 可拿，
> 剩余 ≤2 ms 必须内核化，与 T1 同批。

决策门沿用现有口径：**任一改动必须同时满足** ① S1 p95 改善 ≥ 3%（或 S2 ≥ 3%）② OD 三项 + ERQA 绝对波动 ≤ 2PP ③ 同窗口 A/B，host load 记录在案。

---

## 10. 本次勘查的证据与命令范围

- 本地只读：`AGENTS.md`、`docs/gdn-conv-qkv-2026-11-17.md` §7–9、`docs/profile-800plus-2026-11-17.md`、
  `docs/profile-final-2026-11-17.md`、`docs/final-summary-2026-11-17.md`、`docs/graph-acl-2026-11-16.md` §6、
  `docs/input-distribution-2026-11-16.md`、`docs/dense-sp-upstream-port-2026-11-16.md`、
  `docs/npu-gemm-overlap-probe-2026-11-16.md`、`docs/gdn-chunk128-2026-11-17.md`、
  `docs/batching-seqs8-16-2026-11-16.md`、`Clef-27B/inference.py`、`runtime/joint_single_field/clef_joint_schema.py`。
- 2026-11-18 整合更新引用的实测证据：`docs/perf-wave2-20261118.md`、
  `Clef-27B/logs/perf-wave2-20261118/`（profile 紧凑表、head 打点、chunk 扫描、S1/S2 汇总）、
  `docs/frontend-timing-2026-11-17.md`、`docs/joint-head-batch-2026-11-16.md`。
- 远端只读（`ssh clef`，仅 `cat/grep/ls/find/python3 -c` 读取）：
  `/workspace/yangchen/work/src/vllm-ascend`（`89eb8a5e7`）的 `ascend_config.py`、`compilation/*`、
  `distributed/device_communicators/*`；`/workspace/yangchen/models/Clef-27B/{config.json,joint_schema_model.py}`；
  `/workspace/yangchen/logs/gdn-data-path-20261115/final-profile/analysis-969b/{operations.csv,top20.csv}`；
  `/workspace/yangchen/logs/current-service.json`。
- **未执行**：任何 NPU 任务、服务启停、推理、benchmark、对 `/workspace` 的写操作。

### 参考链接

- vLLM-Ascend graph mode（ACLGraph / Npugraph_ex / static kernel / Super Kernel）：<https://docs.vllm.ai/projects/ascend/en/latest/user_guide/feature_guide/graph_mode.html>
- vLLM-Ascend PR #12649：<https://github.com/vllm-project/vllm-ascend/pull/12649>
- vLLM-Ascend PR #19033：<https://github.com/vllm-project/vllm-ascend/pull/19033>
- vLLM-Ascend issue #21740：<https://github.com/vllm-project/vllm-ascend/issues/21740>
- vLLM-Ascend PR #12317：<https://github.com/vllm-project/vllm-ascend/pull/12317>
- vLLM-Ascend PR #20454：<https://github.com/vllm-project/vllm-ascend/pull/20454>
- vLLM-Ascend releases（v0.23/v0.30/v0.32 notes）：<https://github.com/vllm-project/vllm-ascend/releases>
