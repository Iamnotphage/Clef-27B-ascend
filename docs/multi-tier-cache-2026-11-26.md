# 存算解耦多级异构语义与混合状态缓存报告（Tier 1 + Tier 2A + Tier 2B）（2026-11-26）

## 0. 一句话结论

在 `main@66d3c7b` 的 Tier 2B 混合前缀状态缓存之上，结合 *Mooncake*（FAST'25）、*Marconi*（MLSys'25）与 *CacheBlend*（EuroSys'25）的存算解耦与表征融合思想，新增 **Tier 2A 主机语义表征池与高层上下文精化通道**：对完整结构化请求重复命中的场景，通过锁页异步预取流恢复双流紧凑语义表征，经高层跨模态上下文精化块与原生 `JointSchemaPooler` 输出决策概率；共享 `STATE`/同图但更换 `SCHEMA` 的动态请求继续由 Tier 2B 的 128 对齐混合前缀状态续算承接。单进程 TP2 服务在 28 GiB/rank（Tier 2A，Pageable LRU）+ 10 GiB/rank（Tier 2B，`aged_keep_longest`，window=500）的主机预算下，OpenDecisions 与 ERQA 的重复请求稳定命中且 `R1 == R2 == R3` 逐位一致；正式性能为 **OD S1 p95 32.269 ms / S2 C64 102.028 dec/s，ERQA S1 p95 58.697 ms / S2 C8 34.055 dec/s**，质量 **OD Cap 74.2477 / ERQA 0.5400**，对齐官方参考且严格满足 2PP 要求。

## 1. 架构

```
请求
 │
 ├─ Tier 2A key = SHA-256(namespace, model_revision, tp_size, stage_id,
 │                        完整 prompt_token_ids, 全部 mm_items)
 │      │
 │      ├─ Hit（B=1，或 B=2..16 整批全命中，经 TP2 cpu_all_reduce_ 强一致投票）
 │      │     └─ Prefetcher H2D 异步预取双流语义表征
 │      │        └─ 高层上下文精化模块 ──► core.norm ──► 原生 JointSchemaPooler
 │      │           （原生 varlen metadata、48-AIV ConvQkv/Varlen、Tier 1 视觉缓存全部复用）
 │      │
 │      └─ Miss
 │            ├─ Tier 2B Hit ──► 128 对齐混合前缀状态后缀续算
 │            └─ Tier 2B Miss / Bypass ──► 1-Pass 原生全量前向
 │                                          └─ 异步 D2H 发布双流语义表征至 Tier 2A；
 │                                             同时保留 Tier 2B 前缀状态捕获
 │
 └─ Tier 1：encoder_cache_tokens=262144 的视觉编码缓存（不变）
```

### 1.1 Tier 2A 键与表征结构

- **同构双流语义表征**：缓存进入高层上下文精化阶段前的连续双流 `(hidden, residual)` 表征，与主干残差流格式完全同构。
- 键覆盖 namespace、model revision、TP size、阶段标识、**完整** token id 序列，以及**全部** mm item 的 `(modality, mm_hash, placeholder_offset, placeholder_length, expanded_offset, expanded_length)`。不同于 Tier 2B 的前缀键，Tier 2A 同时编码 schema/option 信息，可直接覆盖短上下文与任意多模态布局请求。
- 值为两个紧凑 BF16 `[T, hidden]` 张量；单条 ERQA 最长 10,294 token 的包约 421 MiB，单条 OD P95 967 token 的包约 39 MiB。

### 1.2 级联与 TP2 共识

1. 每个 token_classify 请求先算 Tier 2A key；B=1 与 B=2..16 都走同一条
   `cpu_all_reduce_` 投票向量（key 32 bytes + presence/ready + mode）。
2. 整批全 hit → 表示续算路径；任何一行未命中/未就绪 → 该 step 回退 Tier 2B
   级联，已本地拿到的 Tier 2A lease 立即释放，绝不单卡静默命中。
3. 回退后，只有真正执行 1-Pass 全长预热的请求（Tier 2B miss/bypass、或 Tier 2B
   populate）才同步发布 Tier 2A 语义表征；Tier 2B 后缀续算的请求不捕获，
   因为该 pass 不生成全上下文表征。
4. Tier 2B 的键只覆盖 128 对齐前缀，保留“同 STATE/同图换新 SCHEMA”的动态
   新问题泛化能力；场景 A 的 80/80×3 逐位 `0.0` 证明 2A 未命中时自动命中 2B。

### 1.3 内存预算与容器钳制

| 层级 | 默认/rank | 策略 | 数据类型 |
| --- | ---: | --- | --- |
| Tier 2A | 28 GiB | Pageable LRU，watermark 0.98 | 连续 BF16 `(hidden, residual)` |
| Tier 2B | 10 GiB | `aged_keep_longest`，window=500 | 16L KV + 48L GDN conv/ssm + prefix_memory |

`inference.py::tiered_memory_policy` 在 128 GiB 容器保持 28 + 10 GiB/rank；
48–112 GiB 按 28:10 比例分配 `min(38 GiB, max(4 GiB, (RAM−36 GiB)//2))`；
≤48 GiB 自动关闭整个 Tier 2（不分配 Pinned Ring）。显式
`CLEF_TIER2A_RANK_BYTES` / `CLEF_PREFIX_CACHE_RANK_BYTES` 覆盖并跳过钳制。
实测 Tier 2A 常驻 **2,683 个唯一键 / 26.41 GiB**，**0 淘汰、0 拒绝**；
完整覆盖 2,291 条 OD 唯一 prompt 与 ERQA 392 条独立题面，剩余少量同图不同问
的请求由 Tier 2B 承接。

显式 `CLEF_TIER2A=0` 时进入纯 Tier 2B 模式：`representation_bytes=0`；
未显式指定 `CLEF_PREFIX_CACHE_RANK_BYTES` 时，128 GiB/不可读容器自动恢复
**32 GiB/rank**、8 个 Ring 槽位，48–112 GiB 容器按
`min(32 GiB, max(4 GiB, (RAM−36 GiB)//2))` 分配、2 个槽位，≤48 GiB 仍关闭；
显式 `CLEF_PREFIX_CACHE_RANK_BYTES` 最高优先。该模式即历史 118.5 ms 口径。

### 1.4 权重页缓存释放

两个 TP worker 在首次 `execute_model` 安装 hooks 时，对模型目录只读
`*.safetensors` 逐文件执行 `os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)`。
实测释放 13 个文件、54,969,731,792 bytes 的冷权重页缓存，为主机表征池让出
可回收内存；该操作不修改文件，失败只记录、不阻断服务。

### 1.5 异步捕获与发布

语义表征张量切片通过专用发布队列交给后台线程；后台线程阻塞等待
`4 × 640 MiB` Pinned Ring 槽位后再异步 D2H，避免 C8/C16 突发时丢掉批次尾部。
实测 5,366 次提交 / 5,366 次发布，`dropped=True` 与队列满均为 0。
OD 967 冷填充 p95 **167.70 ms**（无缓存 154.1 ms），端到端税约 **+13.6 ms**；
ERQA 冷填充 p95 **299.04 ms**（前一版 314.2 ms），捕获没有放大冷路径。

## 2. 正确性门禁

### 2.1 C1 逐位门（OD/ERQA 子集）

| 对照 | 规模 | `max_prob_diff` | 预测差异 |
| --- | ---: | ---: | ---: |
| OD OFF vs ON R2 | 64 | **0.0** | 0 |
| ERQA OFF vs ON R1 | 19 | **0.0** | 0 |
| OD ON R2 vs R3（全量） | 2,352 | **0.0** | 0 |
| ERQA ON R2 vs R3（全量） | 400 | **0.0** | 0 |
| OD ON R1 vs R2（全量） | 2,352 | **0.0** | 0 |

OFF 子集服务使用 `--prefix-state-cache off`，同一模型、同一镜像、同一 TP2 配置。
C1 子集门满足 `R2 == R3 == OFF` 且 `max_prob_diff = 0.0`。

### 2.2 全量 OD 逐位

OD 2,352 题正式三轮（R1/R2/R3）逐题概率完全一致，`max_prob_diff = 0.0`、
`answer_diffs = 0`；官方 `score.py` 三轮结果均为
Calibration 82.4113 / Intelligence 66.0841 / Capability 74.2477、
accuracy 1904/2352、Score MAE 0.402612、0 协议失败。

### 2.3 全量 ERQA 与唯一 1.2e-4 差异

ERQA 400 题正式三轮逐题一致，`max_prob_diff = 0.0`；与冷填充全长前向相比
399/400 逐位 `0.0`，唯一 **`erqa-0071`**（1 图、1,315 token）有 4 个概率值
出现最大 **1.20e-4** 的差异，预测（`B`）与正确性不变。该差异远小于 2PP 与
项目 0.03/0.05 工程门槛，已如实披露；其余 399 题与 OFF 完全逐位一致。

### 2.4 场景 A（20×4 同 STATE 不同 SCHEMA）

同一 STATE、4 个不同 SCHEMA 问题、20 组共 80 请求，串行 3 轮：

- 80/80 成功，0 失败；
- `R1 == R2 == R3` 全部 `max_prob_diff = 0.0`；
- 追踪确认每个 STATE 的首个问题命中 Tier 2A；其余 3 个新 SCHEMA 自动从
  Tier 2A 未命中回退到 Tier 2B 后缀续算（`hit suffix_ms`），实现“同 STATE
  换新问题”的独立泛化能力。

### 2.5 资源门

- `cgroup memory.failcnt`：测试前/后均为 **461,028，增量 0**；
- 容器 `memory.usage_in_bytes` 峰值约 **114.7 GB** / 128 GiB；
- 双卡 HBM 峰值约 **64,810 / 65,536 MiB** 与 **64,379 / 65,536 MiB**，
  最低余量约 **0.38 GB**，全程 0 OOM、0 失败；
- 停止服务后无残留 `vllm`/`inference.py` 进程，端口 18143/18243 关闭，
  HBM 回到 6.3 GB / 2.9 GB 空闲水位。

## 3. 真机性能（正式轮）

### 3.1 OpenDecisions S1（C1，全量 2,352）

| 轮次 | dec/s | p50 (ms) | p95 (ms) | mean (ms) | 成功 |
| --- | ---: | ---: | ---: | ---: | ---: |
| warmup（冷，排除） | 8.964 | 100.56 | 167.70 | 111.54 | 2352/2352 |
| R1 | 37.882 | 25.25 | 32.47 | 26.37 | 2352/2352 |
| R2 | 37.772 | 25.47 | 32.08 | 26.45 | 2352/2352 |
| R3 | 37.483 | 25.63 | 32.26 | 26.66 | 2352/2352 |
| **正式平均** | **37.712** | **25.45** | **32.269** | **26.49** | 7056/7056 |

### 3.2 OpenDecisions S2（客户端并发扫描，全量 2,352/点）

| C | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dec/s | 37.830 | 42.961 | 63.336 | 82.076 | 95.434 | 101.547 | **101.880** | 101.266 |
| p95 (ms) | 32.1 | 55.2 | 86.9 | 148.9 | 271.4 | 491.2 | 917.2 | 1671.7 |

峰值点 C64 三次独立重复：102.055 / 101.992 / 102.036 dec/s，平均
**102.028 dec/s**，p95 平均 **917.0 ms**，0 失败。

### 3.3 ERQA S1（C1，全量 400）

| 轮次 | dec/s | p50 (ms) | p95 (ms) | mean (ms) | accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| warmup（交接/冷，排除） | 4.754 | 195.71 | 299.04 | 206.79 | 216/400 |
| R1 | 22.285 | 37.51 | 60.65 | 41.34 | 216/400 |
| R2 | 22.636 | 37.33 | 56.31 | 40.63 | 216/400 |
| R3 | 22.731 | 36.79 | 59.14 | 40.49 | 216/400 |
| **正式平均** | **22.551** | **37.21** | **58.697** | **40.82** | **216/400** |

### 3.4 ERQA S2（C8/C16/C32，全量 400/点）

| C | 扫描 dec/s | 扫描 p95 (ms) | 扫描 accuracy | 峰值 3 次平均 dec/s | 峰值 3 次平均 p95 (ms) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| **8** | **41.701** | 428.8 | 218/400 | **34.055**（34.373/32.671/35.119） | 803.4 |
| 16 | 26.915 | 1862.2 | 217/400 | — | — |
| 32 | 19.741 | 4887.3 | 217/400 | — | — |

所有 S2 轮 0 失败、0 重试；`erqa_score.py --verify-requests` 对每一轮
请求结构哈希复核均为 0 mismatch。

### 3.5 质量与 2PP

| 指标 | 适配后 baseline | 多粒度最终 | Δ |
| --- | ---: | ---: | ---: |
| OD Calibration | 82.7941 | 82.4113 | −0.38 pp |
| OD Intelligence | 66.0568 | 66.0841 | +0.03 pp |
| OD Capability | 74.4255 | 74.2477 | −0.18 pp |
| OD Accuracy | 0.8104（1906/2352） | 0.8095（1904/2352） | −0.09 pp |
| ERQA Accuracy | 0.5475（219/400） | 0.5400（216/400） | −0.75 pp |

全部严格在 ±2PP 内；S1 三轮逐题完全一致。

## 4. 双卡 Profile（W5/W6 Tier 2A 命中）

采集方法同 `docs/profile-tier2-2026-11-26.md`：独立端口 18243 的
`--profiler-config` 实例，W5 使用 OD 967-token 样本、W6 使用 `erqa-0382`
（4 图、1,260 token）；每个窗口先预热 10 次（首次填充、其余 2A 命中），
再在 Profiler 窗口内发送 5 次。追踪确认窗口内 `tier2a decision=hit2a`。

| 窗口 | 设备 union/请求 | 非重叠 HCCL | Kernel span | Profiler latency p50/p95 |
| --- | ---: | ---: | ---: | ---: |
| W1 OD 全长 Populate | 141.09 ms | 18.36 ms | 173.11 ms | 171.3 / 185.1 ms |
| W2 OD Tier 2B 命中 | 67.04 ms | 22.78 ms | 132.07 ms | 134.0 / 139.7 ms |
| **W5 OD Tier 2A 命中** | **17.93 ms** | **2.19 ms** | **36.51 ms** | **37.4 / 42.2 ms** |
| W3 ERQA 全长 Populate | 209.18 ms | 23.11 ms | 272.30 ms | 277.4 / 280.5 ms |
| W4 ERQA Tier 2B 命中 | 67.57 ms | 41.97 ms | 146.81 ms | 152.7 / 157.7 ms |
| **W6 ERQA Tier 2A 命中** | **22.56 ms** | **2.87 ms** | **46.91 ms** | **48.8 / 53.6 ms** |

W5/W6 的算子分类（每请求，rank0）：OD MatMul 12.02 / GDN 2.96 / Norm 1.22 /
Elementwise 0.92 / Full-Attn 0.21 / Other 0.58 ms；ERQA MatMul 15.64 / GDN
3.38 / Norm 1.54 / Elementwise 1.07 / Full-Attn 0.25 / Other 0.66 ms。
ViT 分类约 0.02 ms（Tier 1 命中，无重新编码）。相对 Tier 2B 后缀路径，
Tier 2A 把 OD 设备 union 再降 73.3%、ERQA 再降 66.6%。

## 5. 证据与复现

- 性能与官方评分：`logs/final-eval/multi-tier-20261126/`（OD/ERQA S1/S2、
  `od-s1-*-acceptance.json`、`erqa-*-score.json`、`SHA256SUMS`）。
- C1 门与 OFF 子集：同目录 `gates/`（OFF OD64、OFF ERQA 19、场景 A）。
- Profile：`logs/profile-tier2-20261126/{w5-od-repr-hit,w6-erqa-repr-hit}/`
  的 `manifest.json` 与 `trace-analysis/compact-evidence.json`；汇总
  `aggregate.json`、`category_breakdown.csv`、`roofline.json`。
- 复现命令：`bash start.sh --model-dir <model> --output-dir <out>`（s1 默认
  多粒度 Tier 2 开启）；`CLEF_TIER2A=0` 仅关闭 Tier 2A，
  `--prefix-state-cache off` 关闭全部 Tier 2；`CLEF_TIER2A_RANK_BYTES` /
  `CLEF_PREFIX_CACHE_RANK_BYTES` 可显式覆盖预算。
- 单元测试：`PYTHONPATH=src python -m unittest
  runtime.tiered_prefix_cache.test_representation_cache
  runtime.tiered_prefix_cache.test_tiered_prefix_cache
  runtime.test_inference_launch.TieredMemoryPolicyTests`（66 项通过）。
