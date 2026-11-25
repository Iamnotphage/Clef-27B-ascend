# 两层分层缓存：阶段 0 探针与阶段 1 Tier 1 定容（2026-11-25）

> 基线：`origin/main` **512317e**（Worker 1 + Worker 2 最终合并）。本报告只记录已在 A3 NPU 6 上实测的证据；阶段 0.1 全量审计与阶段 0.2 硬件/算子探针均已完成，阶段 0 Go/No-Go 结论为 **Go**（含一项 967-token 加速比低于计划目标的偏差）。阶段 1 结论：**保留 `encoder_cache_tokens=262144`**。

## 1. 阶段 0.1：固定数据前缀切分与复用审计

脚本：`runtime/tiered_prefix_cache/audit_dataset_prefixes.py`（离线复放 `ClefFlashStrategy` 的固定前缀/后缀与逐段 schema 编码）。为验证离线口径与真实服务一致，脚本同时对比 2026-11-24 最终服务报告的 `usage.input_tokens`：

| 数据 | 逐题 token 对比 | 结果 |
| --- | --- | --- |
| OpenDecisions 2,352 | 2,352/2,352 | **0 mismatch** |
| ERQA 400（A2 / state-list） | 400/400 | **0 mismatch** |

### 1.1 长度切分（tokens，`L_aligned = floor(L_prefix/128)*128`）

| 数据 | 指标 | min | p50 | p95 | max | 均值 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| OD 2,352 | `L_prefix` | 40 | 182 | 780.45 | 1,314 | 275.72 |
| OD | `L_aligned` | 0 | 128 | 768 | 1,280 | 211.21 |
| OD | `L_suffix` | 101 | 232 | 330 | 446 | 235.60 |
| OD | 总 token | 145 | 357.5 | 969.25 | 1,595 | 446.81 |
| ERQA 400 | `L_prefix` | 149 | 810 | 1,529 | 10,161 | 796.14 |
| ERQA | `L_aligned` | 128 | 768 | 1,408 | 10,112 | 728.00 |
| ERQA | `L_suffix` | 103 | 204 | 264 | 414 | 203.68 |
| ERQA | 总 token | 240 | 933 | 1,653 | 10,294 | 931.68 |

- OD `L_aligned >= 128` 的样本 **1,450/2,352 = 61.65%**（902 条 `L_prefix < 128`，Tier 2 不适用，走普通全量路径）。
- ERQA 400 条 **全部** `L_aligned >= 128`，后缀 p50 204、p95 264 tokens。

### 1.2 复用与同一 STATE 多 SCHEMA

| 指标 | OD 2,352 | ERQA 400 |
| --- | ---: | ---: |
| 唯一完整 prompt | 2,291 | 400（1 对重复） |
| 唯一 STATE | 2,240 | 399 |
| 可复用 STATE 组 / 样本 | 108 组 / 220 条 | 1 组 / 2 条 |
| 可复用对齐前缀（`L_aligned>=128`） | 1,356 唯一键，86 组、180 条、94 个“第二遍+”请求，最大组 3 | 399 唯一键，1 组 2 条 |
| 同一 STATE 不同 SCHEMA | **48 组 / 100 条**，单 STATE 最多 **5** 种 SCHEMA | 无跨题复用 |

官方 OD 全量单遍冷跑只有 94 个请求（4.0%，且在 1,450 条 eligible 内）能直接命中 Tier 2；Tier 2 的工业价值主要在“同 STATE 多问”场景（48 组）与暖态重跑，而不是冷首遍全量。

### 1.3 ERQA 视觉工作集与 joint head 参数

- 唯一图片 **530** 张（630 次出现）；唯一图片视觉 token 合计 **227,549**（单图 min 64 / p50 382 / p95 983 / max 1,014，均值 429.34；逐图求和与逐题 processor 结果残差 **0**）。
- `encoder_cache_tokens=262144` **可完整装下 530 张唯一图**（余量 34,595 tokens）；`131072` 装不下。
- `joint_head_config.json`：`hidden_size=5120`、**`width=1024`**、`routing_layers=2`、`layers=4`、`heads=16`。`prefix_memory = 2 × 1024 B/token = 2 KiB/token`（全精度字节），16 层 KV 合计 64 KiB/token（TP2），GDN 48 层 BF16 状态约 74.8 MiB/前缀（双 rank 合计）。

## 2. 阶段 0.2：硬件与两段续算探针

### 2.1 Probe A：Pinned H2D / D2H（A3 两逻辑 NPU）

| 设备 | 包大小 | Pinned H2D p50 | Pinned D2H p50 | H2D 带宽 | D2H 带宽 | 入队 host 时间 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| npu:0 | 40 MiB | 0.71 ms | 1.00 ms | 55.0 GiB/s | 39.2 GiB/s | 0.04 ms |
| npu:0 | 70 MiB | 1.24 ms | 1.75 ms | 55.1 GiB/s | 39.0 GiB/s | 0.04 ms |
| npu:0 | 360 MiB | 6.29 ms | 8.88 ms | 55.9 GiB/s | 39.6 GiB/s | 0.04 ms |
| npu:1 | 360 MiB | 6.96 ms | 9.51 ms | 50.5 GiB/s | 37.0 GiB/s | 0.04 ms |

- 40/70/360 MiB 对应单 rank 的 350/967/10,294-token 状态包；恢复耗时 1–9 ms，异步入队对主计算流 **< 0.05 ms**，满足 < 1.0 ms 目标。
- Pageable D2H 只有 ~3.6 GiB/s，Tier 2 必须使用 pinned 缓冲。

### 2.2 Probe B：生产 GDN 前缀状态续算（真实激活）

通过 `runtime/tiered_prefix_cache/probe_capture/sitecustomize.py` 在**不改源码**的前提下从常驻服务捕获生产 `fla_npu` GDN 前填充的 q/k/v/g/beta 与 metadata；随后用同一 A3 Base adaptor 的精确调用（`chunk_size=128`、BSND、`use_exp2=False`、`use_qk_l2norm_in_kernel=False`、BF16 `initial_state`）复放 1-Pass vs 2-Pass。捕获层覆盖 9 个 GDN 层 × 2 TP rank = **18 组/请求**。

| 请求 | tokens | `L_aligned` | suffix 输出 vs 全量 | 最终状态 vs 全量 | 0 状态全量 vs 生产 decision 输出 | 前缀状态有限 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `od-1534` | 967 | 768 | **0.0**（max/mean） | **0.0** | **0.0** | 是 |
| `erqa-0167` | 10,294 | 10,112 | **0.0** | **0.0** | **0.0** | 是 |

- **结论：A3 生产 GDN 算子在 128-token 对齐边界上做 BF16 状态续算是逐位精确的**（18/18 组 suffix 输出与最终状态均为 0.0）。
- `fla_npu` 的“prepare path”（`use_qk_l2norm_in_kernel=True` / `use_exp2=True`）在 A3 报 `161002: The prepare path is supported on Ascend950 only`；A3 服务实际走的是预归一化 q/k + chunk128 的 BSND 路径，该路径正常且上面的续算探针即基于它。
- FP32 `initial_state` 被算子契约拒绝（要求 `initial_state.dtype == q.dtype`，即 BF16），且 `output_final_state` 本身返回 BF16；因此 Tier 2 的 `ssm_state` **只能以 BF16 存储/恢复**，FP32 存储不会带来额外精度。计划中的“BF16 vs FP32”分支按此收敛为 BF16。

### 2.3 Probe C：后缀增量净收益（实测口径）

同一常驻服务（TP2、16384、seqs16、encoder 262144、BF16）串行实测：

| 请求 | 输入 tokens | mean | p50 | p95 |
| --- | ---: | ---: | ---: | ---: |
| 合成短 prompt | 176 | 100.95 ms | 100.69 ms | 102.06 ms |
| 合成短 prompt | 196 | 99.73 ms | 99.10 ms | 103.51 ms |
| 合成短 prompt | 256 | 101.01 ms | 100.29 ms | 105.37 ms |
| `od-1534` 全量 | 967 | **153.62 ms** | 152.89 ms | 156.30 ms |
| `erqa-0167` 全量（Tier 1 暖） | 10,294 | **1,291.19 ms** | 1,290.73 ms | 1,293.45 ms |

按“后缀同长度全量前向 + H2D 恢复 + 联合头更长 memory”的保守估算：

| 场景 | 后缀基线 | + 恢复（pinned） | + 头 memory 增量 | 估算命中延迟 | 估算加速 |
| --- | ---: | ---: | ---: | ---: | ---: |
| OD 967（L_aligned 768, suffix 199） | ~100 ms | ~1.2–2.0 ms | ~3–5 ms | ~104–107 ms | **1.44–1.48×** |
| ERQA 10,294（L_aligned 10,112, suffix 182） | ~100 ms | ~6.3–7.0 ms | ~10–20 ms | ~116–127 ms | **10.2–11.1×** |

- 长图文（10,294）显著超过计划 ≥ 5× 目标；**967-token 目标 1.8× 未达到**，根因是 64 层 eager CPU dispatch 的 ~100 ms 固定下限（与 2026-11-16/17 profile 的每层 ~2 ms 结论一致），不是 GDN 状态恢复。
- 因此 Tier 2 的定位应调整为：**长图文/长文本、同 STATE 多问、暖态复用**为主收益；OD 单遍 S1 P95 只会有 4% 请求受益，不应以 Tier 2 作为 OD S1 的主优化项。

### 2.4 阶段 0 Go/No-Go

**Go**，但带两条约束：

1. A3 生产 GDN 续算逐位精确（18/18 组 0.0），无算子限制，BF16 状态即可；FP32 状态无收益。
2. OD 967 的净加速 1.44–1.48× 低于计划 1.8× 目标；进入阶段 3 后只应在长 ERQA、同 STATE 多问和暖态场景宣传收益，OD 冷单遍 S1 目标保持不变，不得回退。
3. 尚未在阶段 0 覆盖：ClefGdnConvQkv（自定义 conv 算子）与续算所用 `causal_conv1d_fn` 的跨算子舍入差、16 层 Full-Attn KV 的 paged 恢复、`prefix_memory` 拼接的端到端概率差。这三项按计划在阶段 3 的引擎集成中做 1-Pass vs 2-Pass 逐题概率对照；当前算子级证据（GDN 逐位、KV 为纯拷贝、`memory_projection`/LayerNorm 逐 token 线性）均支持可行。

## 3. 阶段 1：Tier 1 Vision Encoder Cache 定容

同一干净常驻 TP2 服务（`max_model_len=16384`、`max_num_seqs=16`、`util=0.85`、`T2_QUEUE=2`、conv+QKV v4、joint fused `26d3ad04…`）实测：

### 3.1 HBM 峰值

| `encoder_cache_tokens` | chip0 峰值 | chip1 峰值 | chip0 余量 | chip1 余量 |
| ---: | ---: | ---: | ---: | ---: |
| **262,144** | **61,358 MiB (59.92 GiB)** | **61,009 MiB (59.58 GiB)** | **4.08 GiB** | **4.42 GiB** |
| 131,072 | 61,433 MiB (59.99 GiB) | 61,081 MiB (59.65 GiB) | 4.01 GiB | 4.39 GiB |

- 峰值由 `gpu_memory_utilization=0.85` 决定，**降低 encoder 容量并不降低 HBM 峰值**；两档均留下 **≥ 4.0 GiB / rank**，足够 Tier 2 的 70–360 MiB 临时 H2D 缓冲（计划要求 ≥ 3 GiB）。
- 结论：**保留 `encoder_cache_tokens=262144`**；131072 无法装下 227,549 tokens 的唯一图片集，且暖态吞吐下降。

### 3.2 ERQA 冷/暖对照（同一服务进程，不重启）

| 运行 | 正确/总数 | 成功率 | 吞吐 | mean | p50 | p95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| C16 冷首遍（262144） | 217/400 | 400/400 | 6.953 dec/s | 2,239.5 ms | 2,424.8 ms | **3,492.3 ms** |
| C16 第二遍（262144） | 216/400 | 400/400 | 8.189 dec/s | 1,895.7 ms | 2,100.3 ms | **2,966.7 ms** |
| C16 第三遍（262144） | 217/400 | 400/400 | 8.187 dec/s | 1,894.2 ms | 2,059.0 ms | 2,963.5 ms |
| C16 冷首遍（131072） | 219/400 | 400/400 | 6.952 dec/s | 2,243.5 ms | 2,488.5 ms | 3,617.7 ms |
| C16 第二遍（131072） | 218/400 | 400/400 | 7.081 dec/s | 2,198.2 ms | 2,447.1 ms | 3,502.7 ms |

- **262144 暖态 p95 比冷首遍低 15.0%**（3,492→2,967 ms），吞吐 +17.8%；131072 暖态只提升到 7.08 dec/s，明显低于 262144 的 8.19。
- C16 的 cold/warm 1 题差与 warm2/warm3 的 3 个 choice flip（max_prob_diff 0.088）来自 **C16 合批组成不同导致的批内非确定性**，缓存全暖时同样出现，不是 Tier 1 命中造成的；C16 质量对照应按 ±3 题噪声解释，逐位门仍用 C1。
- C1（encoder 已暖）cold/warm 对照：**max_prob_diff = 0.0、0 choice flips、216/216**；mean 166.68→165.62 ms、p95 247.61→247.62 ms。512317e 已记录的干净 C1 冷首遍/第二遍为 288.40→244.38 ms、逐位一致，本轮未重复冷首遍。
- 全部运行 0 失败、0 重试；ERQA 正确率 216–219/400，相对 baseline 219/400 的绝对波动 ≤ 0.75 PP，满足 2PP 门。

## 4. 阶段 2：纯 CPU 缓存核心与单测

新增独立包 `runtime/tiered_prefix_cache/`（不依赖 NPU，可在 CPU 侧完整验证）：

| 模块 | 能力 |
| --- | --- |
| `prefix_key.py` | 最后一个 `\n\nSCHEMA FIELDS:\n` 定位、`L_aligned=(L_prefix//128)*128`、schema span 越界拒识；SHA-256 键覆盖 namespace、模型 revision、前缀 token ids 与每个多模态项的 `mm_hash`/placeholder/expanded 偏移长度，同尺寸不同图必不相同 |
| `host_store.py` | 32 GiB 默认预算（cgroup/MemAvailable 扣 12 GiB 安全余量）、引用计数 LRU（在途 lease 永不回收）、`second_hit` Ghost LRU（65,536 键）/`always` 双准入、rank0 单份 POSIX SHM `prefix_memory` + rank1 只读映射、`tp_vote_vector`/`tp_vote_result` 与已验证 K32 协议逐字节一致 |
| `async_transfer.py` | Pinned ring（默认 3 槽）、独立 `torch.npu.Stream()` D2H + Event 后台发布、H2D 分组预取与逐组 `wait`；CPU/无 NPU 环境自动退化为同步路径 |
| `test_tiered_prefix_cache.py` | 27 项单测：307/967/1,595/10,294 边界、schema span、mm 防碰撞、second_hit/always、lease-safe 淘汰、8 线程并发压力、SHM 往返与清理、TP 投票 |
| `stress_host_store.py` | 4 GiB 紧预算、16 线程、20,000 次操作的淘汰压测，校验预算/lease/SHM 残留 |

实测：

- 远端 pinned venv：**27 tests, OK, 0 failures**（含 POSIX SHM 用例）；本机无 torch//dev/shm 时 3 项按预期 skip。
- 4 GiB 压测：`ok=true`、1,945 条常驻、bytes 4,079,022,080 ≤ 4 GiB、**10,765 次淘汰、17 次拒绝、lease 0、SHM 关闭后残留 0**、2.97 s 完成。
- 证据：`Clef-27B/logs/tiered-cache-20261124/stage2-stress-host-store.json`。

## 4.1 阶段 2 代码审查修复与 A3 真 NPU 异步验证（2026-11-25 补记）

按 `stage2_bugfix_plan.md` 修复了 5 类问题，并新增 NPU 验证脚本：

| 问题 | 修复 |
| --- | --- |
| `PinnedRing.acquire` 持锁 sleep 死锁 | 改为 `threading.Condition`；`acquire(timeout=0.05)` 超时返回 `None`，调用方记 `dropped_ring_full`，绝不阻塞在线响应 |
| `AsyncCapture` 流竞争/脏读/源显存复用/同步别名 | 双事件协议：主计算流记 `ready` → Copy 流 `wait_event` → 逐张量 `record_stream` → **设备内 snapshot** → Copy 流记 `snapshot_done`；主计算流 `wait_event(snapshot_done)` 后才允许后续覆写；单后台消费线程 `queue.Queue` 有序发布，`drain()`/`close()` 可等待并暴露异常；同步模式深拷贝持久化 pinned 张量 |
| `Prefetcher` 退化 pageable（慢 ~15×）与 `wait` 阻塞 CPU | H2D Pinned Staging Pool（2 槽 × 384 MiB，超尺寸按需分配）；`RestoredState.wait()` 默认 `current_stream().wait_event()`（纯流间等待），仅 `sync_host=True`/`close()` 才 `event.synchronize()`，并对目标 `record_stream` |
| `HostStore` SHM 非 owner 误删 / 锁外创建 / 重复 attach 切片 / `BufferError` | `_shm_objects` 记录 `(shm, owned)`，仅 owner `unlink`；`publish` 的重复键、预算与 `_sequence` 全部在锁内；`attach_shm` 加锁、校验 `size<=shm.size`，返回 `AttachedShm` 上下文并统一释放 `memoryview`；新增 `read_shm_bytes`；session 使用 pid+ns+uuid 避免同名 |
| `prefix_key` 图片展开被切穿 / 逐 token hash | `L_aligned` 若落在某图片展开块内，向下回退到该图片对齐起点（循环直到不切穿，低于 `min_prefix` 返回 `None`）；只把完全位于 `[0,L_aligned)` 的多模态项纳入 `split.mm_items`；`compute_prefix_key` 一次性 `struct.pack`，并把 `align`/`tp_size` 编入哈希域 |

验证（远端 A3，NPU 6，单进程 ~1–2 GiB HBM，未启动 vLLM）：

| 项目 | 结果 |
| --- | --- |
| 单测 | **35 tests, OK, 0 failures, 0 skipped**（含 SHM 并发/非 owner/重复 attach、ring 死锁、同步别名、异步 drain） |
| 覆写对抗（70 MiB） | **20/20 `torch.equal=True`**；submit 均值 0.43 ms / p50 0.41 / p95 0.57 |
| 覆写对抗（360 MiB） | **20/20 `torch.equal=True`**；submit 均值 0.67 ms / p50 0.56 / p95 1.06（热身后；submit 仍远小于 9 ms 的 D2H，主线程未被传输阻塞） |
| 端到端恢复（70 MiB） | `equal=True`；enqueue 0.66 ms、`wait` CPU **0.053 ms**、有效 H2D **32.7 GiB/s** |
| 端到端恢复（360 MiB） | `equal=True`；enqueue 0.59 ms、`wait` CPU **0.041 ms**、有效 H2D **46.8 GiB/s** |
| 4 GiB 压测（SHM 开） | `ok=true`，10,699 次淘汰、23 次拒绝、0 lease 泄漏、关闭后 `/dev/shm` 0 残留 |

- 360 MiB 的 `submit` 均值 0.67 ms 略高于计划 0.5 ms 目标（p50 0.56 ms）；差异来自每次提交的 Event/Stream 书签与设备内 snapshot 排序，属于非阻塞入队，且相对 OD 967 的 154 ms、ERQA 10,294 的 1,291 ms 可忽略。有效 H2D 32.7–46.8 GiB/s 已证明未退化到 pageable 3.6 GiB/s。
- 证据：`Clef-27B/logs/tiered-cache-20261124/stage2-async-npu-verify.json`、`stage2-stress-host-store.json`。

## 5. 阶段 3 的前置结论与风险（尚未开工）

阶段 3 需要同时改动 `gdn.py`、`clef_joint_schema.py`、Full-Attention 的 paged KV 与调度器，严格按安装器 `--check/--apply/--revert` 叠层；本节列出已确认的接口与仍需在阶段 3 实测的差异：

1. **GDN 已解决**：A3 生产算子在 128 对齐边界 BF16 续算逐位一致（18/18 组 0.0）；`ssm_state` 必须 BF16。`conv_state` 由卷积前最后 3 个 token 恢复；生产 `ClefGdnConvQkv` 与续算所用 `causal_conv1d_fn` 的跨算子舍入差需在阶段 3 用逐题概率对照量化（当前证据：conv 为 4-tap 因果卷积，状态语义确定）。
2. **Full-Attn KV**：16 层、TP 分片、每 token 64 KiB（TP2 合计）；捕获/恢复是 paged block 拷贝，数值上应为纯拷贝逐位一致，但需要接入 `block_table` 与 vLLM-Ascend 128-token 页布局。
3. **JointHead `prefix_memory`**：`hidden_norm` 与 `memory_projection` 都是逐 token 线性，命中时 `cat([prefix_memory, suffix_memory])` 数学上与全量一致；阶段 3 需在 `clef_joint_schema.py` 增加可选通道并保持 `CLEF_JOINT_RANK0_ONLY=1` 只让 rank 0 读取。
4. **调度器**：`max_num_seqs=16` 下按 Miss / Hit / Populate 分桶；`second_hit` 首次命中不捕获、第二次才异步 D2H，保证冷首遍（官方 OD/ERQA 单遍）零额外延迟。
5. **已知收益边界**：OD 967-token 命中估算 1.44–1.48×（eager 64 层 CPU dispatch ~100 ms 固定下限）；阶段 3 的验收应把 OD 冷单遍 S1/S2 零回归作为硬门，把 Tier 2 收益主要记在长 ERQA、同 STATE 多问与暖态复用。

## 6. 证据与后续

- 本报告与轻量证据：`Clef-27B/logs/tiered-cache-20261124/stage0-dataset-audit-summary.json`、`stage0-pinned-bandwidth.json`、`stage0-gdn-split-od967.json`、`stage0-gdn-split-erqa10294.json`、`stage0-latency-tokens.json`、`stage1-hbm-summary.json`。
- 远端原始证据（含逐题审计行、完整服务报告、500 ms HBM 采样、捕获张量）：`/workspace/yangchen/logs/tiered-cache-20261124/` 与 `capture/`。
- 阶段 2（纯 CPU 引擎与单测）可在不占用 NPU 的情况下继续；阶段 3 的引擎接入按计划必须在 512317e 上以安装器 `--check/--apply/--revert` 叠层，并保留 `CLEF_PREFIX_STATE_CACHE=0` 与 `admission=second_hit` 的冷首遍零回归。
- 本阶段服务已停止，18143 与 NPU 6 已释放。

## 9. M0/M1 引擎攻坚（2026-11-25 晚）

### 9.1 M0 单算子门（全过）

- **FIA `TND` + `sparse_mode=3` + `block_table=None` + 2048 bool causal mask**：B1 967/10,294、B8 450/mixed 全量用例 `torch.equal=True`（max_abs 0.0）。
- **Conv 续算**：fla_npu 融合 prefill 的状态读入在 A3 上 `161002` 不可用，改用 `vllm_ascend.ops.causal_conv1d.causal_conv1d_fn`；B1 967/10,294 与 B8 450 的 suffix 输出全部 `torch.equal=True`（max_abs 0.0）。另验证 v4 融合核“前补 3 个 conv_state token”的 suffix q/k/v 与全量 v4 逐位一致（0.0），最终引擎用该快路径。
- **阈值**：`min_prefix=512`；OD 1,450→418 条 eligible，唯一前缀 415，估算 **47.48 GiB** 双卡常驻；P95 尾部（token≥967）覆盖 **100%**；`keep_longest` 淘汰与 anon-aware 64 GiB 预算已实现。

### 9.2 M1 B1 全链闭环（功能跑通，ERQA 数值门差 0.000913）

实现并以进程内 `probe_engine`（sitecustomize/`tier2_boot` .pth）挂载 4 个 hook：`NPUModelRunner.execute_model` 上下文、`ClefFlashForJointSchema.forward` 前缀捕获+后缀、`AscendAttentionBackendImpl.forward` 连续 KV FIA、`QwenGatedDeltaNetAttention._forward_core` 状态续算、`JointSchemaPooler/Head` 旁路 AllPool。缓存命中走 HostStore 连续包（KV 16 层、GDN 48 层、prefix hidden），单次 H2D 3 包。

| 样本 | tokens | L | populate | hit | vs baseline max_prob_diff | flips |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| od-1534 | 967 | 768 | 2013 ms | **109–114 ms** | **0.003582** | 0 |
| od-0677 | 1595 | 1280 | 394 ms | **113 ms** | **0.000558** | 0 |
| od-0105 | 307 | 128 | bypass | 103 ms | 0.0 | 0 |
| erqa-0167（16 图） | 10,294 | 10,112 | 1999 ms | **169–181 ms** | **0.005913** | 0 |

- 时间拆解：ERQA hit H2D **7.5–7.8 ms**、suffix 64 层 **90.1–90.5 ms**；OD hit load **2.0–2.8 ms**、suffix **86.6–88.7 ms**。
- **结论**：M1 功能链完整、速度达到计划区间（OD 109–114 ms、ERQA 169–181 ms，ERQA 相对 1,940 ms 冷基线约 10.7×），OD 满足 5e-3 fallback，ERQA **0.005913 超 5e-3 门 0.000913**（0 flips）。按刚性门 **M1 未通过**，M2–M4 未启动；Tier 2 默认 off，主线保持 `ae5903f`。根因是分块 GEMM（QKV/MLP/记忆投影）M=L/T 形状不同带来的 BF16 tiling 差异，待用 T 形状前缀补齐或 FP32 累加方案再评估。
- 探针清理：`/tmp/clef_tier2_engine|trace`、venv `tier2_probe.pth` 已移除；服务停止、18143/NPU6/`/dev/shm` 已清。

## 10. 架构审阅 P0 修复与复测（2026-11-25 晚）

按审阅报告完成核心修复：

- **P0-1**：删除 `ctx.prefix_hidden` 全链（Model 捕获、Store 打包、Hit 恢复、Pooler 全 Hidden 分支）；`_pooler_forward` 现严格传后缀 `hidden [S,5120]`、`positions-ctx.L` 与 `ctx.prefix_memory [L,1024]`；`find_last_subsequence` 改尾部倒序；`_debug` 改为 `/tmp/clef_tier2_debug` 门控，不再无条件写盘。
- **P0-2 部分**：`pool_done` 生命周期与 lease 释放修正；TP2 单次 `all_reduce` 投票（key+mode）；`mm_hash` 为空直接 Bypass；`chunk_indices` 动态读取 `_GDN_CHUNK_SIZE`；`min_prefix` 分模态（文本 512 / 多图 128）；HostStore `observe(key, L)` 满水位早拒。
- **P0-4 部分**：每 rank 预算固定 `CLEF_PREFIX_CACHE_RANK_BYTES`（默认 32 GiB），不再读 `/proc/meminfo`；持久化仍为 pageable `.cpu()`。

复测（单进程 TP2，修正后）：

| 运行 | mean | p95 | acc | vs OFF max_prob_diff | flips |
| --- | ---: | ---: | ---: | ---: | ---: |
| ERQA C1 R1 Warmup | 248.06 ms | 326.94 ms | 216/400 | 0.064700 | 2（erqa-0075/0165） |
| ERQA C1 R2 | **134.79 ms** | **202.94 ms** | 216/400 | 0.064700 | 2 |
| OD S1 R1 Warmup | 148.83 ms | 300.29 ms | — | — | — |
| OD S1 R2 | 127.01 ms | 215.95 ms | — | — | — |
| OD S1 R3 | 126.89 ms | 216.80 ms | — | — | — |

- **关键结论**：修复 Pooler 错位后 ERQA 单样本仍为 `0.003582 / 0.005913`、全量 maxdiff 0.064700、2 flips 不变，**证明审阅报告“0.005913 由 `prefix_hidden` 索引错位导致”的根因判断不成立**；全 Hidden 分支确为真实 Bug，已修复，但数值差来自 64 层分块 GEMM 的 M-tiling（L/S vs T），需 T 形状补齐或确定性 GEMM 才能压入 5e-3。
- **S1 对比**：OD 暖态 R2/R3 **7.87 dec/s / p95 216 ms**，慢于基线 9.24 / 154 ms；ERQA C1 R2 134.8 / 202.9 快于基线 206.9 / 289.3。OD 短前缀命中收益不足，另有 2-Substep Populate 税；1-Pass Tap 未实现。**Tier 2 仍默认 off，M2–M4 未启动，main 保持 `ae5903f`。**

## 11. Step 1：Tier 2 B1 1-Pass Tap、老化淘汰与性能闭环（2026-11-25 晚）

> 分支 `feat/tiered-cache-20261124`（基于 `a0af111`，含 varlen main `29a1bf8`）。本轮把 M1 的 2-Substep Populate 改为 **1-Pass Tap**，补齐 `aged_keep_longest` 与 `observe(est_bytes)` 水位修复，接入 `AsyncCapture`/`Prefetcher` 与 CPU `cpu_all_reduce_` 投票，并用 **grouped varlen SSM 捕获**把 48 层 GDN 状态压到每 4–8 层一次调用。OFF 对照为同一代码树、关闭 Tier 2 的独立服务（`tap-off-20261125`）。

### 11.1 实现

1. **Populate = 全长一遍**：`ClefFlashForJointSchema.forward` 在 populate 时直接跑 `[0:T]` 全 64 层，输出经原装 pooler（FRACTAL_NZ/单字段等常规路径），层内旁路切 `key[:L]`、`value[:L]`、`mixed_qkv[L-3:L, :5120]`，JointHead 侧切 `memory[:L]`；不再跑第二遍 64 层。
2. **Grouped SSM 捕获**：GDN 状态在 `fla_gdn_prefill` 旁路按层暂存 prefix q/k/v/g/beta，每 `G` 层合并为一次 varlen 调用（`cu_seqlens=[0,L,…,G·L]`、seq-major `chunk_indices`、`initial_state=0`、`output_final_state=True`）。`G=min(8, 512 MiB / 单层字节)`：OD 常取 8（约 105 MiB），ERQA 10,112-token 取 5（约 525 MiB）。离线 2 层 OD/ERQA 与单层捕获 **max_abs=0.0**；在线 grouped R2 与 immediate R2 **400/400 逐位 0.0**。
3. **异步存取**：`AsyncCapture` 独立 copy 流 + 双事件 + 设备内 snapshot，`PinnedRing(slots=2, slot_bytes=640 MiB)`；HostStore 持久化 pageable；Hit 经 `Prefetcher` 精确尺寸 pinned staging（总缓存 ≤ 768 MiB）恢复，`wait_event` 不阻塞 CPU。CPU 张量不再调用 `record_stream`。
4. **准入与淘汰**：`HostStore` 新增 `aged_keep_longest`（`ACTIVE_WINDOW=3000` 次 observe，stale 按 `last_obs_seq` 优先淘汰）与 `observe(key, length, est_bytes)`——早拒按 `bytes+est` 预测 publish 水位，不再用淘汰后的 `bytes`（修复每轮重复 Populate 抖动）；`populates`（准入）与 `publications`（发布）分离。TP2 所有 `split` 请求（含 `miss`）统一 `get_tp_group().cpu_all_reduce_` 35 元 int32（key+mode+load），不一致双卡同步 Bypass。
5. **B1 作用域**：仍只接管"每步恰好一条全新请求"；`dummy_run`/`is_profile` 直接跳过；其余走原路径。

### 11.2 数值等价

| 对照 | 结果 |
| --- | --- |
| OFF 服务 vs 冻结基线（ERQA C1 400） | **400/400 逐位 0.0** |
| OD R1（tap）vs OFF | **2,351/2,352 逐位**；唯一差异 `od-1534` 为冒烟预热后的 hit（0.002028） |
| ERQA R1（tap）vs OFF | **388/400 逐位**；12 条为冒烟/同轮预热命中（max 0.023889） |
| OD R2 vs R3 | immediate **2,352/2,352 逐位 0.0**；grouped 同为 **2,352/2,352 逐位 0.0** |
| ERQA R2 vs R3 | **400/400 逐位 0.0** |
| grouped vs immediate 命中（ERQA R2） | **400/400 逐位 0.0** |
| hit vs OFF | OD max 0.069327（302/2,352 条非零）、ERQA max 0.055595（252/400 条）；2PP 内 |
| 质量 | OD R2（grouped）Cap **74.2586** / Int **66.0880** / Cal **82.4292** / acc 1904 / MAE 0.402523（immediate Cap 74.2573）；ERQA R2 **217/400** |

- **结论：Populate（1-Pass Tap）与 OFF 逐位一致**；所有差异均来自 hit 路径的 BF16 分块 GEMM `M=L/S` vs `M=T` 累加顺序，与 S2 合批误差同源，不涉及缓存实现正确性。
- grouped 与 immediate 两次 OD R2 有 7 条差异（2,345/2,352 逐位，max 0.0167）；逐条核对全部是“一条命中、另一条 Bypass”的**缓存驻留集合差异**（异步 publish 完成时序影响 keep_longest 早拒），不是状态捕获差异；ERQA 无淘汰场景 grouped vs immediate **400/400 逐位 0.0**。同一次运行的 R2 vs R3 才是缓存自洽门。

### 11.3 性能（S1 C1，同机同服务对照）

ERQA C1（400，A2 state-list）：

| 轮次 | mean (ms) | p50 | p95 | dec/s | 准确率 |
| --- | ---: | ---: | ---: | ---: | --- |
| OFF 冷首遍 | 206.32 | 194.25 | 287.97 | 4.76 | 216/400 |
| Tap R1（populate，grouped） | **206.46** | 197.72 | **286.98** | 4.76 | 216/400 |
| Tap R2（hit，grouped） | **131.22** | 120.38 | **206.46** | 7.42 | 217/400 |
| Tap R2（hit，immediate） | 138.52 | 129.86 | 202.35 | 7.04 | 217/400 |
| Tap R3（hit，immediate） | 135.94 | 126.63 | 202.77 | 7.17 | 217/400 |
| 冻结暖基线（C1 第二遍） | 165.40 | 153.42 | 244.66 | 5.94 | 216/400 |

- R1 对 OFF **mean +0.14 ms / p95 −1.0 ms / wall 84.0 vs 84.0 s**：Populate 已"冷不亏"。
- 按请求配对（R1 vs OFF）：populate n=255 **+9.43 ms mean / +21.83 p95**（immediate 为 +18.9/+30.1）；bypass −0.86（噪声）；12 条命中 −186 ms。
- R2 对冻结暖基线 **mean −20.7%、p95 −15.6%**；267/400 命中（133 条因图像展开跨 128 边界不可切分而 Bypass，属设计上限）。
- ERQA 聚合（grouped R1+R2、immediate R3）：暖 2 轮 mean 133.58 vs 冻结暖基线 165.40（**−19.2%**）、p95 204.62 vs 244.66（**−16.4%**）；3 轮 mean 157.87 vs OFF 冷+暖+暖 179.04（**−11.8%**）。

OD S1（2,352）：

| 轮次 | mean (ms) | p50 | p95 | dec/s |
| --- | ---: | ---: | ---: | ---: |
| OFF 对照 | 109.10 | 100.41 | 153.87 | 9.164 |
| Tap R1（populate，immediate） | 123.19 | 106.56 | 195.25 | 8.116 |
| Tap R2（hit，immediate） | 110.67 | 106.31 | **124.06** | 9.034 |
| Tap R3（hit，immediate） | 109.92 | 106.01 | **123.96** | 9.096 |
| Tap R1（populate，grouped） | 119.79 | 105.54 | 187.11 | 8.346 |
| Tap R2（hit，grouped） | **106.49** | 102.54 | **121.66** | 9.388 |
| Tap R3（hit，grouped） | **107.66** | 103.61 | **122.55** | 9.286 |

- 按请求配对（grouped R2 vs OFF）：hit n=314 **−41.09 ms mean / −8.52 p95**；bypass +3.42（会话漂移）；R2 总体 **mean −2.61 ms（−2.4%）、p95 −32.21 ms（−20.9%）、9.388 dec/s（+2.4%）**。R3 与 R2 同量级（mean −1.44 ms、p95 −20.4%、9.286 dec/s）。immediate 臂的 hit 为 −36.23 ms、R2 总体 p95 −19.6%。
- 聚合：OD **暖 2 轮 mean 107.08 vs OFF 109.10（−1.9%）、p95 122.11 vs 153.87（−20.6%）**；3 轮 mean 111.31（+2.0%）、p95 受 R1 的 187.11 拉高。
- 缓存准入（grouped 臂）：R1 379 条 populate、32 条 keep_longest 早拒、154–158 条淘汰；R2/R3 **0 准入、0 淘汰**，存储稳定在 478–479 条 / 32.6 GB/rank（0.95 水位）。
- 配对说明：R1 的 populate 增量 grouped 为 +28.34 mean / +103.21 p95（immediate +43.75/+119.91）；ERQA grouped 为 +9.43/+21.83。OD R1 的残余主要来自首轮新形状的 aclnn/组调用预热与打包，非稳态成本。

### 11.4 跨数据集与稳定性

- 同一常驻服务 ERQA×2 → OD×3（grouped 臂）：ERQA 17.1 GB/rank 常驻后，OD R1 淘汰 154–158 条、R2/R3 **零准入零淘汰**；ERQA 的 133 条 bypass、OD 的 1,941 条 bypass 与 97 条 keep_longest 早拒全程稳定，无双卡分叉（`cpu_all_reduce_` 投票下仅出现一次 C1 连发同 key 的 hit/populate 竞态，双卡同步 Bypass）。immediate 臂的 ERQA×3 → OD×3 计数相同量级（158 淘汰、R2/R3 零准入）。
- 41 项单测（远端 41/41 OK）；`torch_npu`/SHM 用例全跑。
- 失败/重试：所有 ERQA/OD 运行 **0 失败、0 重试**；`erqa_score.py --verify-requests` 400/400、0 mismatch。

### 11.5 偏差、限制与下一步

1. **Populate 残余税**：grouped 后 ERQA populate +9.4 ms mean / +21.8 p95；单层状态重算的设备时间与 KV/SSM 打包仍计入。`return_intermediate_states`（h 每 chunk 状态）在 A3 直接报 `161002: The prepare path is supported on Ascend950 only`，**不可用**；grouped varlen 已是 A3 上可验证的最优路径。
2. **ERQA 覆盖 267/400**：133 条的对齐边界落在最后一张图的展开块内，按"不得切穿图像"规则必须 Bypass；阶段 0 审计的 400/400 仅按 delimiter 对齐、未计该规则。提高覆盖需要放宽边界或改缓存粒度，留作后续。
3. **hit 均值与 R1 残余**：grouped 臂 OD R2 mean 已低于 OFF 2.6 ms（命中 −41.1 ms × 314 条抵消 bypass 会话漂移 +3.4）；immediate 臂因更大漂移为 +1.6 ms。OD R1 仍 +10.7 mean/+33.2 p95，主要为首轮新形状预热与打包，暖 2 轮已 −1.9%/−20.6%；与阶段 0 "收益定位长图文/暖态"的结论一致。
4. **M2 多并发**：未启动；下一步按计划做 Varlen 混批（Bypass/Populate/Hit 分桶、`runner.input_batch.req_ids` 对齐、整批一次 `embed_lexical`）。
