# 当前版本（多粒度 Tier 2 常驻 TP2）Profile 与 S1 Roofline（2026-11-26）

> 基准：`main` @ `66d3c7b`（W1–W4）与 `feat/multi-tier-cache-20261126` @ `91dbe02`
> （W5–W6，Tier 2A 命中）；远端 `/workspace/final-test/Clef-27B-ascend`。
> 采集服务：`bash start.sh --config s1`，端口 `18243`，`--profiler-config` 仅用于本报告；
> 所有端到端性能口径仍以 `results.csv` 与 `logs/final-eval/multi-tier-20261126/` 的
> 无 Profiler 正式轮次为准。原始 trace 在采集后已清理，仓库只保存轻量汇总
> `logs/profile-tier2-20261126/`（约 700 KiB）。

## 1. 采集方法

### 1.1 六个窗口

| 窗口 | 场景 | 样本 | 采集方式 |
| --- | --- | --- | --- |
| W1 | OD 全长 967 token（Populate） | 5 条不同前缀，统一 967 token | 先用另一条 967-token 样本预热 10 次，再在 Profiler 窗口内各发送 1 次 |
| W2 | OD Tier 2B 命中（后缀 199 token） | 同一条 967-token 样本 | 预热 10 次（首次 populate，其余命中），再发送 5 次 |
| W3 | ERQA 多图全长 1,260 token（Populate） | `erqa-0382`（4 图）的 5 个像素微扰副本 | 另一扰动副本预热 10 次，再发送 5 个互不相同的冷副本 |
| W4 | ERQA Tier 2B 命中（后缀 236 token） | 原始 `erqa-0382` | 预热 10 次，再发送 5 次 |
| **W5** | **OD Tier 2A 命中（boundary=56，完整 967 token）** | **同一条 967-token 样本** | **预热 10 次（首次填充 2A，其余 2A 命中），再发送 5 次** |
| **W6** | **ERQA Tier 2A 命中（boundary=56，完整 1,260 token）** | **原始 `erqa-0382`** | **预热 10 次，再发送 5 次** |

OD 样本在 state 末尾追加 filler，使 5 条请求的 `usage.input_tokens` 均为 **967**；filler 位于
128 对齐前缀之后，不改变缓存键。ERQA 副本只修改每张图 (0,0) 像素的 G 通道并转存 PNG，
宽高不变，`usage.input_tokens` 均为 **1,260**；原始请求与 15 个扰动副本的 Tier 1 图像哈希
互不相同，保证 W3 的 5 条测量请求全部走 ViT + Populate。

### 1.2 解析口径

`profile_systemone.py` 自动完成「窗口外 warmup → `/start_profile` → 5 次请求 →
`/stop_profile` → 独立进程 analyse」，随后由 `analyze_systemone_profile.py --strict`
计算双 rank 的设备时间区间并集（`union_us`）。**本报告的设备耗时均为 union，不是
kernel duration 直接相加**；算子分类按 `operations.csv` 的 `interval_union_us`
归并，非通信分类之和与 `compute_kernel_union_us` 逐项闭合（W1 141.0 vs 141.1，W3
209.1 vs 209.2）。ViT 按权重/激活维度（1152/2152/1728/576/2304/4608）与 `aclnnConvolution`
单独归组。

## 2. W1–W6 双卡算子级拆解（每请求，rank0）

### 2.1 W1–W4：全长 / Tier 2B 后缀

| 指标 | W1 OD Populate | W2 OD Hit | W3 ERQA Populate | W4 ERQA Hit |
| --- | ---: | ---: | ---: | ---: |
| 输入 token | 967 | 967（S=199） | 1,260 | 1,260（S=236） |
| **设备 compute union** | **141.1 ms** | **67.0 ms** | **209.2 ms** | **67.6 ms** |
| MatMul (Cube) | 91.2 | 42.8 | 120.1 | 41.9 |
| GDN Core (FLA) | 27.6 | 11.1 | 33.3 | 11.4 |
| Fused Conv+QKV | 2.6 | 1.1 | 3.2 | 1.2 |
| Full-Attn (FIA) | 1.6 | 1.0 | 6.9 | 1.2 |
| Norm | 8.3 | 3.5 | 10.5 | 3.8 |
| Elementwise | 6.8 | 6.0 | 10.9 | 6.3 |
| Other | 2.9 | 1.6 | 8.2 | 1.7 |
| ViT Vision | — | — | 16.0 | — |
| **HCCL/传输非重叠** | **18.4 ms** | **22.8 ms** | **23.1 ms** | **42.0 ms** |
| **设备总预算** | **159.5 ms** | **89.8 ms** | **232.2 ms** | **109.5 ms** |
| 客户端 p50 / p95（含 Profiler） | 171.3 / 185.1 | 134.0 / 139.7 | 277.4 / 280.5 | 152.7 / 157.7 |

rank1 的 compute union 为 141.2 / 66.7 / 208.6 / 67.1 ms，与 rank0 一致；差异集中在
HCCL 非重叠（W2 36.0 ms、W4 25.0 ms），说明命中路径的预取/通信在 rank 间并不完全对称。
W3 的 ViT 本体约 16 ms（4 图 ×1,024 patch），其余 120 ms MatMul 为 1,260-token 混合主干。

### 2.2 W5–W6：Tier 2A 边界 56 命中

| 指标 | W5 OD Tier 2A | W6 ERQA Tier 2A |
| --- | ---: | ---: |
| 输入 token | 967（完整 prompt） | 1,260（完整 prompt） |
| **设备 compute union** | **17.9 ms** | **22.6 ms** |
| MatMul (Cube) | 12.0 | 15.6 |
| GDN Core (FLA) | 2.6 | 3.0 |
| Fused Conv+QKV | 0.3 | 0.4 |
| Full-Attn (FIA) | 0.2 | 0.3 |
| Norm | 1.2 | 1.5 |
| Elementwise | 0.9 | 1.1 |
| Other | 0.6 | 0.7 |
| ViT Vision | 0.0（Tier 1 命中） | 0.0（Tier 1 命中） |
| **HCCL/传输非重叠** | **2.2 ms** | **2.9 ms** |
| **设备总预算** | **20.1 ms** | **25.4 ms** |
| 客户端 p50 / p95（含 Profiler） | 37.4 / 42.2 | 48.8 / 53.6 |

追踪确认窗口内 56 次 `tier2a decision=hit2a`（两 rank），5 次测量请求全部走
`core.layers[56:64] + norm + 原生 JointSchemaPooler`；`repr resume_ms` 约 11–15 ms。
W5/W6 的 ViT 分类约 0.02 ms，说明 Tier 1 视觉缓存命中、没有重新编码。
设备 union 相对 W2/W4 再降 **73.3% / 66.6%**。

## 3. 结果解读

1. **1-Pass Tap Populate 的设备侧几乎零额外开销，但端到端首次填充仍有约 +12 ms 税。**
   W1 的 141.1 ms 与 2026-11-17 无缓存 profile 的 144.6 ms 同量级；W3 的 209.2 ms 覆盖了
   ViT、全长主干与分组 SSM 捕获，说明 Populate 不是一条额外的设备慢路径，而是原来就
   存在的全长前向。但在同一临时服务上对 5 条全新 967-token 前缀做无 Profiler 冷测
   （`logs/profile-tier2-20261126/unprofiled_od_cold.json`），p50/p95/mean 为
   **166.5 / 167.2 / 166.4 ms**，相对 Tier 2 前全长 p95 **154.11 ms** 高约 8%。差值
   不在设备 compute（159.5 ms 与旧 profile 的 161.8 ms 同量级），而在首次捕获后的
   Host 侧发布/异步 D2H 与簿记；从第二条同前缀请求开始，118.5 ms 的命中路径即覆盖该税。
2. **Tier 2 命中把 Cube 计算削去一半以上，但固定逐层开销仍在。** W2 的 MatMul 42.8 ms
   对全量 91.2 ms 为 **46.9%**，远高于 token 比 199/967 = 20.6%；GDN 12.1 vs 30.2 为 40.1%。
   这说明 MLP/GDN 的每层 kernel 启动、tiling 与 TP2 allreduce 存在明显的固定成本，命中
   并不能按 token 数线性缩短设备时间。
3. **ERQA 命中同时消除 ViT 与长图前缀。** W3 → W4 设备总预算 232.2 → 109.5 ms（−52.8%），
   其中 ViT 16.0 ms 完全消失、MatMul 120.1 → 41.9 ms；Profiler 内客户端 p95 280.5 →
   157.7 ms（−43.8%），与 Tier 2B-only 生产口径 ERQA S1 p95 244.7 → 122.4 ms（−50.0%）方向一致。
4. **Tier 2A 把 2B 仍保留的 64 层后缀固定开销也移出关键路径。** W2 → W5 设备 union
   67.0 → 17.9 ms（−73.3%）、客户端 p95 139.7 → 42.2 ms；W4 → W6 67.6 → 22.6 ms
   （−66.6%）、p95 157.7 → 53.6 ms。W5 的 MatMul 12.0 ms 约为 W1 91.2 ms 的 13.2%，
   与 8/64 层数比接近；HCCL 非重叠从 22.8/42.0 ms 降到 2.2/2.9 ms。多粒度生产口径
   的 OD/ERQA S1 p95 为 **32.27 / 58.70 ms**，与 Profiler 窗口的方向一致（Profiler 自身
   增加约 5–10 ms 开销）。

## 4. S1 Roofline：如果 967 个 token 全部参与计算

### 4.1 计算口径

以 `config.json` 与权重张量逐项统计的 64 层主干为基准（TP2，BF16）：

- MatMul 活跃参数：MLP 17.113 B + GDN 投影 5.613 B + Full-Attn 投影 1.695 B；
- 每 token FLOPs ≈ `2 × 24.42 B`；T=967 时全模型 **47.4 TFLOP**（含 attention core 0.18 TFLOP），
  每 rank **23.71 TFLOP**；
- BF16 Cube 实测峰值 **300 TFLOPS/rank**（历史微基准 285.9–303.4 TFLOPS，取 300）；
- 权重读取：每 rank **27.49 GB**，HBM 实测 **1.27 TB/s**；
- HCCL：每层 2 次 allreduce × `[967,5120]` BF16，合计 **1.267 GB/rank**，实测 **69.1 GB/s**。

### 4.2 分层屋脊

| 层级 | 数值 | 说明 |
| --- | ---: | --- |
| **纯 Cube 屋脊** | **79.0 ms** | 23.71 TFLOP/rank ÷ 300 TFLOPS |
| 权重读取屋脊（可重叠） | 21.6 ms | 27.49 GB ÷ 1.27 TB/s；小于 Cube，故不绑定 |
| + 非 Cube 实测（归一化/逐元素/GDN core/FIA） | +49.9 ms | W1 实测 141.1 − MatMul 91.2 |
| + HCCL 非重叠 | +16.1 ms | W1 双 rank 平均（18.4 / 13.8） |
| **实用设备屋脊** | **145.0 ms** | Cube 79.0 + 非 Cube 49.9 + HCCL 16.1 |
| Host 逐层 dispatch 下限 | 105 ms | 2026-11-17 每层 1.78–1.81 ms ×64 ≈ 114–116 ms，扣除 `TASK_QUEUE=2` 约 7.7 ms 后取 ~105 ms；W2 命中仍约 118.5 ms 佐证 |
| **S1 端到端屋脊** | **≈ 154 ms** | `max(145, 105) + 前端/联合头 ≈ 9 ms` |
| 实测全长 S1 p95（Tier 2 前） | **154.11 ms** | `optimization.md` 阶段 5 |
| 实测当前版本 Populate p95（n=5，无 Profiler） | **167.2 ms** | 5 条全新 967-token 前缀；设备 159.5 ms + 约 8 ms Host 发布/前端 |
| 实测 Tier 2 命中 S1 p95 | **118.54 ms** | `results.csv` / `logs/final-eval/od-s1`（Tier 2B-only 历史） |
| 实测 Tier 2A 命中 S1 p95 | **32.269 ms** | `results.csv` / `logs/final-eval/multi-tier-20261126/od-s1` |

### 4.3 解读

1. **如果 967 个 token 全部参与计算，S1 的纯 Cube 屋脊是 79 ms，实用设备屋脊约 145 ms，
   端到端约 154 ms。** 当前 Tier 2 前的全长路径实测 154.11 ms，已经贴在实用屋脊上；
   把 MatMul 从 91.2 ms 压到 79.0 ms 最多只值 12 ms，单靠算子重写无法再显著降低全长 S1。
   当前版本带 Populate 的冷路径 p95 167.2 ms 比 154.11 高约 8%，差额主要来自首次捕获的
   Host 发布/异步 D2H，而不是设备 compute；它只发生在每个前缀第一次出现时。
2. **Tier 2 命中把 Cube 部分压到 16.3 ms（199/967），非 Cube 仍约 50.6 ms（逐层固定开销），
   设备预算 96.3 ms；但 Host dispatch 下限 105 ms 反过来成为瓶颈，端到端 118.5 ms
   已经低于“全长计算”的 145 ms 设备屋脊。**
3. 因此 S1 的下一步不在缓存或 Cube，而在 **Host 逐层 dispatch**（graph/ACL graph 已证明
   仅对 <384 token 有效）与 **GDN core / Norm 的固定逐层开销**；这两项在命中路径上
   合计约 50 ms，是 118.5 ms 中最大的可争取部分。W2 的 MatMul 仍有 42.8 ms，其中约
   19 ms 是 199-token 的线性部分、约 24 ms 是 64 层固定部分——这部分只能靠减少层内
   小算子数量或更激进的图捕获来削。

### 4.4 Tier 2A 边界 56 的分层预算

| 层级 | 数值 | 说明 |
| --- | ---: | --- |
| 8/64 层 Cube 理论 | 9.9 ms | 79.0 × 8/64 |
| 实测 MatMul（W5） | 12.0 ms | 固定 tiling/启动开销使略高于理论 |
| + 非 Cube 实测 | +5.9 ms | 17.9 − 12.0 |
| + HCCL 非重叠 | +2.2 ms | W5 双 rank 平均（2.19 / 3.23） |
| **设备总预算** | **20.1 ms** | 17.9 compute + 2.2 HCCL |
| Host 8 层 dispatch + pooler/schema | ≈ 24 ms | 8 × ~1.7 ms + ~10 ms |
| **实测 S1 p95** | **32.269 ms** | 正式 3 轮平均；p50 25.45 ms |

Tier 2A 命中已把设备侧压到 20 ms 量级，Host 的 8 层 dispatch、joint pooler 与 schema
处理成为新的主要成本；这也解释了为什么 p50（25.45 ms）与设备预算（20.1 ms）只差约
5 ms，而 p95 的尾部（32.27 ms）主要来自较长的 pooler/attention 形状。

## 5. 图表

- `imgs/fig1_optimization_ablation.{png,pdf}`：全栈渐进优化消融（S1 p95 / S2 吞吐）。
- `imgs/fig2_operator_breakdown.{png,pdf}`：W1–W6 双卡算子级耗时拆解（全长 / 2B / 2A）。
- `imgs/fig3_s2_concurrency_pareto.{png,pdf}`：S2 并发扫描与吞吐-延迟帕累托。
- `imgs/fig4_prefix_handover_analysis.{png,pdf}`：前缀切分收益与跨数据集交接。
- `imgs/fig5_s1_roofline.{png,pdf}`：S1 全长、Tier 2B 命中与 Tier 2A 命中的分层 roofline。

图注：

- **图 1**：6 个阶段的 S1 p95（左）与 S2 最大吞吐（右）；ERQA 仅在有实测的阶段绘制斜线柱，
  阶段 2–3 未单独测 ERQA。最终相对 baseline：OD −85.1% / 17.9×，ERQA −81.9% / 8.6×。
- **图 2**：设备 union 时间按算子类别拆分。Populate 为全长前向，Tier 2B 为后缀续算，
  Tier 2A 为 boundary=56 的完整 prompt 续算；ERQA Populate 的紫色段为 ViT。图中数值与
  第 2 节表格一致。
- **图 3**：OD 为 C=1..128 全扫描，ERQA 为 C8/C16/C32；点线为 Tier 2 前基线
  （OD 18.04、ERQA 8.63 dec/s），虚线为 Tier 2B-only（OD 21.58、ERQA 21.44 dec/s）。
  右图标注 OD C64 与 ERQA C8。
- **图 4**：左图为前缀审计的 128 对齐 `L` 与后缀 `S=T−L`（P50/P95 代表点）；右图为同进程
  `OD → ERQA` 交接，`window=3000` 旧池锁死 8.08 dec/s，`window=500` 两轮 21.36/22.30 dec/s。
- **图 5**：以 967-token 全长（左）、199-token 后缀命中（中）与 Tier 2A 边界 56 命中（右）
  的分层预算；绿色虚线为 Host dispatch 下限，黑色点线为正式 S1 p95。全长设备预算 145 ms
  接近实测 154.1 ms；Tier 2B 受 Host 105 ms 限制（实测 118.5 ms）；Tier 2A 设备预算
 约 20 ms、Host 约 24 ms，实测 32.3 ms。

## 6. 证据与清理

- `category_breakdown.csv` 的 `hccl_union_ms_per_req` 是 HCCL 父区间的并集（可与计算重叠）；
  图表使用 `communication_nonoverlapped_ms_per_req`。
- 轻量汇总：`logs/profile-tier2-20261126/`（`aggregate.json`、`category_breakdown.csv`、
  `roofline.json`、`fig4_data.json`、`unprofiled_od_cold.json`、W1–W6 六个窗口的 manifest
  与 compact evidence）。
- W5/W6 采集服务：端口 18243、`--profiler-config`、`CLEF_TIER2_TRACE=1`；追踪确认窗口内
  全部走 `tier2a decision=hit2a`。原始 trace 在采集后已从远端清理（曾位于
  `/tmp/profile-tier2-traces`），不进入 Git。
- 采集完成后关闭临时 Profiler 服务并确认 `npu-smi info` 与 `ps` 无残留进程。
- 正式性能/质量证据：[logs/final-eval/multi-tier-20261126/](../logs/final-eval/multi-tier-20261126/)；
  专题报告：[docs/multi-tier-cache-2026-11-26.md](multi-tier-cache-2026-11-26.md)。
