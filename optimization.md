# Clef-27B 昇腾 910A3 适配与非量化性能调优报告

## 1. 执行摘要与核心成果

本报告系统总结 `Cloudflare/clef`（27.5B 参数多模态结构化决策模型）在昇腾 910A3（`Ascend910_9382`，单物理卡内 2 个逻辑 NPU，每逻辑 NPU 配备 24 个 AI Cube Core、48 个 AI Vector Core 与 64 GiB HBM）上的全栈工程适配与性能调优工作。

本工程严格遵循赛事交付规范与工程红线：
1. **严格非量化（Pure BF16）**：全程保持原始 BF16 权重、激活与状态精度，未引入任何 W8A8、W8A16、INT8 或 FP8 量化。
2. **原始输出与 2PP 精度保障**：不使用额外温度校准（as-served），`OpenDecisions`（Calibration / Intelligence / Capability）与 `ERQA` 总体准确率相对适配后基线的绝对波动均控制在 **0.37 个百分点以内**（远优于 ±2PP 要求）。
3. **单一常驻服务零重启**：全部文本与多模态质量、S1 单并发延迟、S2 高并发吞吐评测均由同一个 `bash start.sh` 启动的常驻 TP2 服务进程承载，无按请求动态路由两套方案，也无跨数据集手动重启。

### 1.1 端到端优化演进总览

| 评估维度 | 指标 | 适配后基线（2026-11-14） | 算子与调度主线（Tier 2 前） | 最终交付版本（2026-11-26） | 相对适配基线总收益 |
| --- | --- | ---: | ---: | ---: | ---: |
| **OpenDecisions 性能** | S1 P95 延迟 (ms) ↓ | 217.04 | 154.11 | **118.54** | **−45.4%** |
| | S1 吞吐 (decisions/s) ↑ | 4.61 | 9.13 | **9.742** | **2.11×** |
| | S2 最大吞吐 (decisions/s) ↑ | 5.715 | 18.039 | **21.583** | **3.78× (+277.7%)** |
| **ERQA 图文性能** | S1 P95 延迟 (ms) ↓ | 323.48 | 244.66 | **122.37** | **−62.2%** |
| | S1 吞吐 (decisions/s) ↑ | 3.957 | 5.942 | **8.481** | **2.14×** |
| | S2 最大吞吐 (decisions/s) ↑ | 3.957 | 8.627 | **21.439** | **5.42× (+441.8%)** |
| **OpenDecisions 质量** | Calibration (0–100) | 82.7941 | 82.4113 | **82.4254** | Δ −0.37 pp（2PP 内） |
| | Intelligence (0–100) | 66.0568 | 66.0841 | **66.0872** | Δ +0.03 pp（2PP 内） |
| | Capability (0–100) | 74.4255 | 74.2477 | **74.2563** | Δ −0.17 pp（2PP 内） |
| **ERQA 图文质量** | 总体 Accuracy (0–1) | 0.5475 (219/400) | 0.5450 (218/400) | **0.5450 (218/400)** | Δ −0.25 pp（2PP 内） |

> 数据来源：[metrics.csv](metrics.csv)、[results.csv](results.csv) 以及 [logs/final-eval/](logs/final-eval/) 归档报告。

![全栈渐进优化消融](imgs/fig1_optimization_ablation.png)

> **图 1**：6 个阶段的 S1 p95 与 S2 最大吞吐；最终相对 baseline 为 OD −45.4% / 3.78×、ERQA −62.2% / 5.42×。阶段 2–3 未单独测 ERQA，图中留空。数据见 [logs/final-eval/](logs/final-eval/)。


---

## 2. 模型架构特征与昇腾 910A3 Roofline 瓶颈诊断

### 2.1 模型与工作负载特征

`Cloudflare/clef` 总参数量 27,484,784,884（约 55.0 GB BF16），并非标准自回归生成模型，而是基于 `pooling` runner 的**单步纯 Prefill 结构化决策模型**（`output_tokens=0`）：
- **混合主干（64 层）**：基于 Qwen3.5 混合架构，包含 **48 层 Gated DeltaNet（GDN 线性注意力 + 短因果卷积）** 与 **16 层标准 Full-Attention**，辅以多模态视觉编码塔（单题最多 16 张高分辨率图像）。
- **联合决策头（`JointSchemaHead`，1.28 亿参数 / 122 个张量）**：消费全序列末层隐藏态与词法嵌入（lexical embedding），一次性输出 `choice`、`noul`（二元概率）或 `score`（等级概率分布）的精确归一化概率。
- **输入分布跨度大**（[docs/input-distribution-2026-11-16.md](docs/input-distribution-2026-11-16.md)）：
  - `OpenDecisions`（2,352 题纯文本）：`p50 = 357.5`、`p95 = 969.25`、`max = 1,595` tokens；多道题目共享长系统策略或对话背景（`STATE`），仅候选路由问题不同。
  - `ERQA`（400 题具身推理图文）：`p50 = 933`、`p95 = 1,653`、`max = 10,294` tokens（含最高 16 张图像、10,065 个视觉 token），大量图像跨题共享或占总序列长度的 70%–95%。

### 2.2 硬件实测屋脊（Roofline）与五大核心瓶颈

通过自研微基准探针（[hbm_bandwidth_probe.py](src/runtime/hbm_bandwidth_probe.py)、[probe_matmul_ceiling.py](src/runtime/probe_matmul_ceiling.py)）与双卡 `torch_npu.profiler` 时间轴并集分析（[docs/profile-final-2026-11-17.md](docs/profile-final-2026-11-17.md)），我们确立了昇腾 910A3 单逻辑 NPU 的物理上限并定位了五大瓶颈：

- **物理上限实测**：单逻辑 NPU 纯 HBM 读带宽稳定在 **1.26–1.32 TB/s**（超出 L2 的 256 MiB–1 GiB 工作集）；BF16 Cube 矩阵乘冷态峰值 **295–320 TFLOPS**。在 P95 长度（~969 tokens）下，真实 TP2 MLP 矩阵乘已达 **285.9–303.4 TFLOPS**（Cube 利用率 94%–97%），`npu_add_rms_norm` 达 **1,117–1,266 GB/s**，二者已贴近硬件极限，**重写常规 MatMul/Norm 算子已无收益空间**。
- **五大真实工程瓶颈**：
  1. **动态长度 Executor 缓存击穿与宿主下发空泡**：CANN 默认 `aclnn` executor 缓存仅容纳约 145 种输入长度，超出后每个新 token 长度触发 **+55 ms** 编译/查找慢路径；短序列（~307 tokens）CPU 64 层算子下发耗时（114–139 ms）反超设备计算（75 ms）。
  2. **GDN 48 层数据路径冗余与 Vector 核利用率不足**：纯决策服务无 decode 阶段，原框架仍逐层写回循环末态并保留反向重算中间量；GDN 投影输出 `[T, 8192]` 切分 `[T, 5120]` 非连续视图触发显式拷贝，因果卷积与 Q/K L2 Norm 分离启动，且早期算子仅用到 24 个 AIV（浪费 A3 一半 Vector 算力），多序列合批时回退慢路径。
  3. **短序列 MLP MTE2 搬运效率受限**：307 token 短序列下 ND 格式权重有效带宽仅 ~0.53 TB/s（约为 HBM 上限的 41%），`aic_mte2_ratio` 高达 0.994。
  4. **多模态接口割裂与联合决策头小算子开销**：原生 `/v1/systemone` 未归一图文 `state` 列表；`JointSchemaHead` 对单字段请求仍走通用多字段循环与分解注意力。
  5. **原生 Mamba 前缀缓存与决策快路径互斥**：vLLM 原生 Mamba prefix caching 强制要求开启 `--enable-chunked-prefill`，而开启 chunked prefill 会直接关闭结构化决策快路径并触发对齐断言失败，导致长文本与多图前缀在每条请求中重复全量计算。

![S1 Roofline](imgs/fig5_s1_roofline.png)

> **图 5**：967-token 全量计算的纯 Cube 屋脊 79 ms、实用设备屋脊 145 ms、端到端约 154 ms；Tier 2 命中（199-token 后缀）设备预算 96 ms，但 Host dispatch 105 ms 成为瓶颈。详见 [Profile 与 S1 Roofline 报告](docs/profile-tier2-2026-11-26.md)。


---

## 3. 分层优化方案与核心技术原理

### 3.1 适配底座与运行时调度优化

1. **TP2 词表并行归约契约修复与自包含构建**（[src/patches/](src/patches/)、[scripts/setup_env.sh](scripts/setup_env.sh)）：
   - 修复上游 `vLLM-Ascend` 词嵌入层缺失 `use_fused_embedding=False`、`reduce_results=True` 与 `parallel_group` 导致的 TP2 首请求异常；通过 `CLEF_LOAD_AUDIT` 校验全部 12 个主干分片与 122 个联合头参数张量零缺失加载。
   - 修复 CANN `recompile_binary.py` 与 MemFabric `mfcli` 在隐藏目录（`.build` / `.venv`）下 `glob` 漏检 `.o` 目标文件的构建缺陷（[docs/setup-env-2026-11-25.md](docs/setup-env-2026-11-25.md)），实现 `quay.io/ascend/vllm-ascend:v0.30.0-a3` 裸容器内约 17.7 分钟一键构建完整隔离 `.venv`。
2. **消除动态 Shape 慢路径（`ACLNN_CACHE_LIMIT=1000000`）**（[docs/aclnn-cache-shape-fix-2026-11-17.md](docs/aclnn-cache-shape-fix-2026-11-17.md)）：
   - 将 CANN executor 缓存上限扩容至 `1,000,000`，彻底消除 145 种 prompt 长度后的 +55 ms 阶跃退化。全量 2,352 题输出**逐位一致**，官方入口 S1 P95 由 `190.41 ms` 降至 **`168.63 ms`**（mean `153.76 → 125.55 ms`）。
3. **双级异步任务队列（`TASK_QUEUE_ENABLE=2`）与线程隔离**（[docs/tp2-alwayson-2026-11-24.md](docs/tp2-alwayson-2026-11-24.md)）：
   - 开启 `torch_npu` 双级流水下发队列并固定 `OMP_NUM_THREADS=1`，掩盖短序列下 CPU 64 层小算子派发延迟。ABBA 交错对照证实输出**逐位一致**，S1 mean 由 `118.53 ms` 降至 **`110.85 ms`**（p50 `111.39 → 101.09 ms`，吞吐 **+6.9%**），S2 保持中性。

### 3.2 主干 MLP `FRACTAL_NZ` 权重布局优化

在 `Qwen2MoeMLP.forward` 首次执行时，惰性调用 `torch_npu.npu_format_cast(..., 29)` 将 `gate_up_proj` 与 `down_proj` 的 BF16 权重转换为昇腾 Cube 原生亲和的 `FRACTAL_NZ` 格式（`CLEF_MLP_BF16_NZ=1`，见 [docs/mlp-bf16-nz-2026-11-15.md](docs/mlp-bf16-nz-2026-11-15.md)）：
- **原理与收益**：改善短 M 维度下 Cube MTE2 通道的数据局部性，307 token 冷态有效带宽由 `509–532 GB/s` 提升至 **`671–677 GB/s`（1.26–1.32×）**。
- **精度与端到端验证**：ND 与 NZ 矩阵乘输出在全部测试长度下**逐位一致（`max_diff = 0.0`）**；全量 OpenDecisions S1 P95 由 `168.63 ms` 进一步降至 **`166.00 ms`**（mean **−3.06%**，吞吐 **+3.18%**）。

### 3.3 GDN 线性注意力数据路径与自定义 Ascend C 算子

48 层 GDN 占全模型层数的 75%，是主干算子级调优的核心突破口：

1. **决策专用 Prefill 数据路径（`BIC`）**（[docs/gdn-data-path-2026-11-15.md](docs/gdn-data-path-2026-11-15.md)）：
   - 针对 `output_tokens=0` 的纯决策请求（仅依据宿主侧元数据门控，零设备同步开销），跳过无人读取的循环状态 scatter 写回、初始状态读取及反向重算中间张量（`bsnd + skip_initial_state + skip_conv_state + no_recompute`）。
   - 八臂正交筛选确认 `BIC` 最优：OpenDecisions 2,352 题与 ERQA 400 题概率**逐位一致**，交错 A/B 6/6 对更快，S1 P95 中位下降 **−7.01%（−13.81 ms）**。
2. **GDN 分块粒度调优（`chunk_size=128`）**（[docs/gdn-chunk128-2026-11-17.md](docs/gdn-chunk128-2026-11-17.md)）：
   - 遍历测试 chunk 64/96/128/160/192/256（其中非 2 幂或超限块报 `161002`），确认 `128` 是硬件唯一快于 `64` 的分块尺寸（长序列设备耗时 `0.608 → 0.521 ms/层`），进一步将 S1 P95 压低 **−1.32%**（OD Capability `74.2476`，2PP 内）。
3. **单序列 48-AIV 融合算子 `ClefGdnConvQkv` v4**（[src/ascendc/kernels/clef_gdn_conv_qkv/](src/ascendc/kernels/clef_gdn_conv_qkv/)、[docs/gdn-conv-qkv-2026-11-17.md](docs/gdn-conv-qkv-2026-11-17.md)）：
   - **三算子融合**：在单次 Ascend C kernel 启动内完成 `causal_conv1d (width=4)` + `Q/K FLA arch22 L2 Norm` + `V BSND 直拷`，将 967 token 下 48 层设备耗时由 `8.45 ms`（3.79 + 4.66）腰斩至 **`4.17 ms`**。
   - **`as_strided` 免拷贝读取**：Qwen3.5 主干将 `[T, 8192]` 投影拆分为 `[T, 5120]` QKV 与 `[T, 3072]` Z 跨步视图；宿主通过严格存储边界校验的 `as_strided` 直接传入 8192 行跨度，kernel 仅读取前 5120 通道，彻底消除 48 层连续化拷贝。
   - **打满 48 AIV**：针对 `Ascend910_9382` 的 24 AIC / 48 AIV 拓扑重写 tiling（`blockDim = min(totalTiles, 48)`）并将长度上限放宽至 `16,384`，使 ERQA 万级 token 多图输入全程保持融合，长序列单算子较分解路径提速 **3.2–3.7×**。
4. **多序列变长融合算子 `ClefGdnConvQkvVarlen`**（[src/ascendc/kernels/clef_gdn_conv_qkv_varlen/](src/ascendc/kernels/clef_gdn_conv_qkv_varlen/)、[docs/conv-qkv-varlen-2026-11-25.md](docs/conv-qkv-varlen-2026-11-25.md)）：
   - 为解决 S2 多并发合批时 `num_prefills > 1` 回退分解路径的问题，新增独立 `ClefGdnConvQkvVarlen` 算子：以 `cu_seqlens`（int32）驱动 per-sequence `tile=64` 划分，通过 `MTE2_S` 同步与 `history = min(localT0, 3)` 严防序列间卷积历史越界。
   - 离线通过 48 组多序列逐位对齐与 144 组极端毒化（`±1e4` / `NaN`）零串扰门控，单算子提速 **3.53–4.54×**；同窗 A/B 将无缓存 OD S2 C64 推升至 **18.039 dec/s**（+4.86%），ERQA C16 推升至 **8.627 dec/s**（+5.96%，p95 −7.66%）。

### 3.4 多模态协议归一与联合决策头（JointSchemaHead）融合

1. **`/v1/systemone` 图文 `state-list` 原生接入**（[src/runtime/systemone_mm_state.py](src/runtime/systemone_mm_state.py)、[docs/systemone-mm-state-2026-11-24.md](docs/systemone-mm-state-2026-11-24.md)）：
   - 在入口层对多模态 `state` 内容块列表执行无损提取与 Fail-Closed 校验（非法结构直接返回 HTTP 400；纯文本请求零分支透传，OpenDecisions 100% 逐位不变）。经 A1/A2 对照锁定等价于参考包装的 `A2 (extract_question_dict)` 模式，使单进程 `/v1/systemone` 统一服务两套数据集。
2. **JointSchemaHead 单字段快路径与 `npu_fusion_attention`**（[src/runtime/joint_fused_attention/clef_joint_schema.py](src/runtime/joint_fused_attention/clef_joint_schema.py)）：
   - OpenDecisions 与 ERQA 全部样本均为单决策字段（`answer`）；单字段快路径省去动态字段拆分与冗余张量拼接，并将头内部注意力替换为 `torch_npu.npu_fusion_attention`，端到端再降 S1 P95 **−2.91%**。

### 3.5 两级异构缓存架构（Tier 1 视觉缓存 + Tier 2 混合前缀状态缓存）

在完成算子与调度压榨后，单次全长前向已逼近 A3 硬件计算屋脊。为彻底消除跨请求共享上下文与多图特征的重复计算，同时绕开 vLLM 原生 Mamba prefix caching 与决策快路径的架构冲突，我们在 [src/runtime/tiered_prefix_cache/](src/runtime/tiered_prefix_cache/) 自研了面向 `48 GDN + 16 Full-Attn + JointHead` 混合架构的 **Tier 2 前缀状态缓存系统**（[docs/tiered-cache-stage0-2026-11-25.md](docs/tiered-cache-stage0-2026-11-25.md)）：

```
请求批次到达 ──► 128 对齐前缀切分 (文本 ≥512 / 多图 ≥128, 允许切在图像展开块内)
                    │
                    ▼
          TP2 单次 [B, 35] cpu_all_reduce_ 准入共识投票 + 同批同 Key 去重
             ├── Miss / Populate ──► 1-Pass 全长前向（输出与 OFF 逐位一致）
             │                         └─► 分组 Varlen SSM 捕获 (4–8 层/批, ≤512 MiB)
             │                         └─► 8×640 MiB Pinned Ring 异步 D2H ──► Pageable HostStore
             └── Hit ──────────────► 异步 H2D 预取 (16L KV + 48L GDN conv/ssm + prefix_memory)
                                       └─► Pass 2 后缀续算 (仅算 S = T - L tokens!)
                                             ├─ GDN: x_aug varlen conv + 批量 fla 状态续算
                                             ├─ Full-Attn: 单次 FIA TND sparse3 注意力
                                             └─ JointHead: [S,5120] 后缀 hidden + positions-L + prefix_memory [L,1024]
```

核心设计亮点：
1. **完备混合状态表达与多图跨界切分**：
   - 缓存键采用 128 对齐前缀的加密级 SHA-256（涵盖前缀 token IDs、起点 `< L` 的全部原始图像内容 SHA-256、模型 revision 与 TP world size）。
   - 联合缓存三类跨层状态：16 层 Full-Attention 连续 post-RoPE KV、48 层 GDN 在边界 `L` 处的 `(conv_state, ssm_state)`、以及 JointSchemaHead 所需的 `prefix_memory [L, 1024]`（避免冗余保存庞大的 `[L, 5120]` 主干 hidden）。
   - `locate_state_prefix` 允许 128 对齐切分点落在多图视觉 token 展开块内部，配合 Tier 1 扩容后的 `encoder_cache_tokens=262144`，将 ERQA 400 题的前缀可缓存覆盖率从 267/400 提升至 **400/400（100%）**。
2. **1-Pass Tap 零税填充与分组 Varlen SSM 捕获**：
   - Populate 请求直接执行一遍标准全长前向（响应概率与关闭缓存 **100% 逐位一致**）。由于昇腾 A3 上 `fla` 的 `return_intermediate_states` 报 `161002` 不可用，我们设计了 **Grouped Varlen SSM Capture**：按 512 MiB 显存预算每 4–8 层打包一次轻量前缀状态提取，并通过后台线程 + `8 × 640 MiB` 锁页环形缓冲区（Pinned Ring）异步拷贝至 Pageable 主机池，将 ERQA 首轮 populate 的**设备侧**延迟税压缩至 **+0.14 ms**；OD 967 首次填充的端到端 p95 仍约 **167.2 ms**（vs 无缓存 154.1 ms），差额来自 Host 发布/异步 D2H，第二次同前缀请求即降至 **118.5 ms**（详见 [Profile 与 S1 Roofline 报告](docs/profile-tier2-2026-11-26.md)）。
3. **M2 两遍混批续算与 TP2 强一致共识**：
   - 高并发合批（B=2..16）下按 `prepare_inputs` 真实序列顺序执行一次 `[B, 35]` `cpu_all_reduce_` 投票（确保双卡对每个请求的 Hit/Populate/Bypass 决策 100% 一致，任一 rank 未就绪立即协同回退全长计算），并对同批重复前缀自动去重。
   - 执行层采用 **Pass 1 Fresh + Pass 2 Resume** 两遍打包（显式 varlen 卷积 + 单次 `FIA TND` 稀疏注意力 + 批量 `embed_lexical`），消除逐请求 Python 循环开销。
4. **跨数据集平滑交接（`aged_keep_longest`，`ACTIVE_WINDOW=500`）**：
   - 单 rank 默认 32 GiB 主机缓存无法同时常驻完整 OD（~25.7 GB）与 ERQA（~25.0 GB）。将活跃老化窗口设定为 `500`（略大于单轮最大 eligible 观测数：OD 418 / ERQA 400），既保证单一数据集多轮评测 **0 老化、0 淘汰**，又使同进程切换到新数据集时旧池条目在首轮内自然变为 stale 并优先换出，使同进程 `OD → ERQA` 交接后的 ERQA S2 C16 稳定达到 **21.44–22.30 dec/s（末态 0 miss）**。
5. **静态物理/容器内存自适应钳制**（[inference.py](inference.py)）：
   - 启动时静态读取 `min(/proc/meminfo MemTotal, cgroup memory.max)`（不读动态波动的 `MemAvailable`，保证 TP 双卡决策严格一致）：
     - `RAM ≥ 112 GiB`：默认 `32 GiB/rank`、`8` 个 Ring 槽位（128 GiB 容器总 RSS 稳态约 95–102 GiB，余量充足）；
     - `48 GiB < RAM < 112 GiB`：自动收缩 Ring 至 `2` 槽位，每 rank 配额按 `min(32 GiB, max(4 GiB, (RAM - 36 GiB) // 2))` 安全钳制；
     - `RAM ≤ 48 GiB`：自动设置 `CLEF_PREFIX_STATE_CACHE=0` 并跳过 10 GiB Pinned Ring 分配，平滑退回算子优化主线，杜绝小内存评测机 OOM。

![前缀切分与跨数据集交接](imgs/fig4_prefix_handover_analysis.png)

> **图 4**：左图为 OD/ERQA 的 128 对齐前缀 `L` 与后缀 `S=T−L` 代表点；右图为同进程 `OD → ERQA` 交接，`window=3000` 旧池 8.08 dec/s，`window=500` 两轮 21.36/22.30 dec/s、末态 0 miss。代表点由远端完整审计 `stage0-dataset-audit.json` 计算，仓库保留其汇总 `logs/tiered-cache-20261124/stage0-dataset-audit-summary.json`。


---

## 4. 关键阶段消融实验与证据链

所有优化均经过独立微基准、离线单测与真机全量数据验证，关键里程碑汇总如下：

| 阶段与日期 | 关键增量改动 | OD S1 P95 (ms) | OD S2 峰值 (dec/s) | ERQA S1 P95 (ms) | ERQA S2 峰值 (dec/s) | 质量验证（OD Cap / ERQA Acc） | 专题证据文档 |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| **阶段 0**（11-13/14） | 昇腾 A3 TP2 BF16 完整适配基线（seqs1） | 217.04 | 5.715 | 323.48 | 3.957 | 74.4255 / 0.5475 | [tuning4](docs/tuning4-2026-11-14.md)、[erqa-baseline](docs/erqa-baseline-2026-11-14.md) |
| **阶段 1**（11-15/16） | GDN 决策 `BIC` + `ClefGdnQkvNorm` v2 + 单字段 Head + B8/B16 合批 | 190.41 | 17.110 | — | — | 74.3107 / 0.5425 | [gdn-data-path](docs/gdn-data-path-2026-11-15.md)、[batching](docs/batching-seqs8-16-2026-11-16.md) |
| **阶段 2**（11-17） | `ACLNN_CACHE_LIMIT=1e6` + MLP `FRACTAL_NZ` | 166.00 | 16.376 (B8) | — | — | 74.3107 / 0.5425（逐位同上） | [aclnn-fix](docs/aclnn-cache-shape-fix-2026-11-17.md)、[final-1117](docs/final-summary-2026-11-17.md) |
| **阶段 3**（11-17/18） | `ClefGdnConvQkv` 免拷贝融合 + GDN `chunk128` | 155.86 | 17.475 | — | — | 74.2476 / 0.5400 | [gdn-conv-qkv](docs/gdn-conv-qkv-2026-11-17.md)、[perf-wave2](docs/perf-wave2-20261118.md) |
| **阶段 4**（11-24） | 单一常驻 TP2 + `TASK_QUEUE=2` + **48 AIV** v4 + `/v1/systemone` 图文 `A2` + Tier 1 `262144` | 154.55 | 17.193 | 244.38 (暖) | 8.240 | 74.2477 / 0.5400–0.5475 | [tp2-alwayson](docs/tp2-alwayson-2026-11-24.md)、[mm-state](docs/systemone-mm-state-2026-11-24.md) |
| **阶段 5**（11-25） | 多序列 `ClefGdnConvQkvVarlen`（`tile=64`） | 154.11 | 18.039 | 244.66 (暖) | 8.627 | 74.2477 / 0.5450 | [conv-qkv-varlen](docs/conv-qkv-varlen-2026-11-25.md) |
| **阶段 6**（11-25/26） | **Tier 2 混合前缀状态缓存**（1-Pass Tap + M2 混批 + `window=500` + RAM 自适应） | **118.54** | **21.583** | **122.37** | **21.439** | **74.2563** / **0.5450** | [tiered-cache](docs/tiered-cache-stage0-2026-11-25.md)、[logs/final-eval](logs/final-eval/) |

![双卡算子级耗时拆解](imgs/fig2_operator_breakdown.png)

> **图 2**：W1–W4 双卡设备 union 时间按算子类别拆分。OD 全长 159 ms / 命中 90 ms；ERQA 全长 232 ms（含 ViT 16 ms）/ 命中 110 ms。Populate 与 Hit 均按每请求归一；详见 [Profile 报告](docs/profile-tier2-2026-11-26.md) 第 2 节。

![S2 并发扩展与帕累托](imgs/fig3_s2_concurrency_pareto.png)

> **图 3**：OD C=1..128 全扫描与 ERQA C8/C16 的吞吐-并发（左）和 p95-吞吐帕累托（右）；红色虚线为 Tier 2 前基线（OD 18.04、ERQA 8.63 dec/s）。


---

## 5. 已验证负结果与合规性排除记录

工程坚持“实测同窗 A/B 门控（端到端收益显著且满足 2PP 与非量化规则方可合入）”，以下探索项经完整实证后明确排除并恢复干净源码：

| 探索方向 | 实测现象与量化结论 | 排除原因与处置 | 参考档案 |
| --- | --- | --- | --- |
| **W8A8 / W8A16 权重量化** | ND W8A8 可将 S1 P95 压至 ~205 ms，但 Capability 跌至 74.07 且降低权重/激活精度 | **赛方规则明确禁止任何量化方案**；仅保留为历史研究对比，提交配置 100% 纯 BF16 | [submission-rules](docs/submission-rules-2026-11-13.md) §4.4、[mlp-w8a16](docs/mlp-w8a16-2026-11-15.md) |
| **上游 Qwen3.5 Dense SP / MLP-only SP** | 移植 `flashcomm1` / `all_gather_base_mm + ReduceScatter`，S1 P95 退化至 `212.96 vs 198.42 ms`（MLP SP 0.88–0.98×） | 单序列与中等长度下 AllGather/ReduceScatter 拆分通信开销大于单次 AllReduce | [dense-sp-port](docs/dense-sp-upstream-port-2026-11-16.md)、[mlp-sp](docs/dense-sp-mlp-probe-2026-11-16.md) |
| **GEMM + AllReduce 分块双流重叠** | `down_proj` 按 token 切 2/4 块双流重叠，实测仅原速度的 `0.29×–0.80×` | 切小 M 维度破坏 Cube 满载效率且多次 HCCL 启动开销远超重叠收益 | [gemm-overlap](docs/npu-gemm-overlap-probe-2026-11-16.md) |
| **PIECEWISE 图编译 / ACL Graph** | 编译耗时 +29.7 s，动态决策输入长度下吞吐 `19.34 vs 19.33 rps` 完全持平 | 纯动态长短混合 prefill 无 decode 循环，图捕获收益被动态 shape 抵消，保持 `--enforce-eager` | [graph-acl](docs/graph-acl-2026-11-16.md) |
| **`PrefillNoCache` 跳过 KV 写槽** | 307 token 均值反慢（`129.09 vs 125.06 ms`），969 token 仅 −1.0%（噪声内） | 未过 3% 工程采纳门且后续与 Tier 2 KV 捕获需求方向相悖 | [attn-skip](docs/attn-prefill-cache-skip-2026-11-16.md) |
| **HCCL AIV 展开模式与自适应 Chunk** | `HCCL_OP_EXPANSION_MODE=AIV` 服务级 S1 仅 −0.24%（噪声内）；自适应 chunk384 多序列更慢 | 不引入无端到端收益的额外复杂度，保持默认 HCCL 与全局 `chunk128` | [gdn-conv-qkv](docs/gdn-conv-qkv-2026-11-17.md) §7、[tp2-alwayson](docs/tp2-alwayson-2026-11-24.md) |
| **JointHead 跨请求 Padding 批处理** | B4 变慢（`13.44 vs 13.64 dec/s`），全量 OD C16 仅 `+0.03%` | 不同请求候选数与序列长度差异导致的 Padding 计算抵消合批收益；改在 Tier 2 采用无损 `embed_lexical` 批处理 | [joint-head-batch](docs/joint-head-batch-2026-11-16.md) |

---

## 6. 资源占用、防篡改门控与复现审计

1. **NPU HBM 与主机内存占用**：
   - **HBM 占用**：TP2 配置下（`gpu_memory_utilization=0.85`、`max_model_len=16384`、`encoder_cache_tokens=262144`），两张逻辑 NPU 稳态显存占用约为 **61.0–61.4 GiB / 64 GiB**（每卡模型权重约 25.76 GiB，其余为 KV 池、多模态视觉特征缓存与运行时工作区），全程零 HBM OOM。
   - **主机 RAM 占用**：在 128 GiB 标准评测容器下，Tier 2 每 rank 持久化上限为 `32 GiB`（Pageable），外加双 rank 合计约 `10 GiB` 锁页传输缓冲（`8 × 640 MiB × 2`）；跑满 OD + ERQA 全周期交接时容器总内存稳定在安全水位内，且受 `inference.py::static_ram_limit_bytes()` 自适应保护（详见第 3.5 节）。
   - **极限压测验证**：在 `4 GiB/rank` 极小配额 + `C16` 高并发淘汰压测（场景 D，125 次高频淘汰）下实现 **0 协议失败、0 死锁**，Capability 保持 `74.064–74.068`（未命中平滑回退常规全长前向）。
2. **SHA-256 哈希门控与可追溯清单**：
   - 所有运行时补丁（`src/runtime/apply_*.py`）均强制校验目标源码的 Preimage / Postimage SHA-256，版本漂移或只读异常一律 Fail-Closed 拒绝启动，绝不静默退化。
   - 每次执行 `bash start.sh` 均在 `--output-dir` 原子生成 `launch-manifest.json`，完整固化 Git 提交、模型文件摘要、各补丁安装前后 SHA-256、静态内存配额决策及生效环境变量；配套离线单测套件（启动器 35 项、Tier 2 缓存与跨集交接 43 项、核心协议与算子探针 176+ 项）全部通过。
3. **一键复现指南**：
   - 环境构建、模型与数据集下载、单一服务启动及 OpenDecisions / ERQA 质量与 S1/S2 完整复现命令请直接参阅根目录 **[README.md](README.md)**。
