# 当前版本（Tier 2 常驻 TP2）Profile 采集与论文级图表生成任务书

**日期**：2026-11-26  
**基准分支与提交**：`main` @ `5e7352d`（及后续纯文档提交）  
**目标目录**：Profile 数据归档于 `logs/profile-tier2-20261126/`，汇总报告写入 `docs/profile-tier2-2026-11-26.md`，论文级图表输出至 `imgs/`。

---

## 1. 任务背景与目标

当前仓库之前的算子级拆解报告（`docs/profile-final-2026-11-17.md` 与 `docs/perf-wave2-20261118.md`）采集于 **48-AIV Conv+QKV v4、`ClefGdnConvQkvVarlen` 以及 Tier 2 混合前缀状态缓存** 合入之前。为了在 `README.md`（附录 2）与 `optimization.md` 中直观展示最终提交版本的全栈优化效果与设备侧耗时结构，需要由 Worker 完成两项闭环任务：

1. **远端昇腾 A3 真机补充采集当前版本的 NPU Profile**：重点对比 **Cold/Populate（全长前向 + 分组 Varlen SSM 捕获）** 与 **Tier 2 Hit（后缀续算 Pass）** 在 OpenDecisions（代表长度 ~967 tokens）和 ERQA（多图长序列 ~1,650+ tokens）上的双卡设备计算并集（Union）、算子分类耗时、HCCL 非重叠通信及 Host/Device 间隙；**注意：开启 Profiler 的服务仅用于耗时占比拆解，严禁将其端到端延迟混入 `results.csv` 正式性能数据**。
2. **使用本机兄弟仓库 `../figures4papers` 生成 4 张出版级高清矢量/位图（`.png` + `.pdf`）**：严格遵循 `/Users/chenclay/Documents/code/projects/figures4papers/scientific-figure-making/SKILL.md` 及其 `references/api.md`、`references/design-theory.md` 的配色、字体、线宽与无上/右边框规范，将生成的图片放入 `imgs/` 并嵌入 `README.md` 与 `optimization.md`。

---

## 2. 阶段一：昇腾 A3 真机 Profile 采集规范

### 2.1 采集矩阵（4 个核心窗口）

在远端 A3 容器中激活 `.venv`（`source scripts/activate_env.sh`），使用带 `--profiler-config` 的临时实例（独立 `--output-dir /tmp/profile-tier2-svc` 与端口，避免干扰任何正式评测服务；若已有常驻进程占卡，先确认空闲后再停启，任务结束后清理进程）：

| 窗口编号 | 场景 | 样本选择 | 采集目的 |
| --- | --- | --- | --- |
| **W1: OD-Cold/Populate** | OpenDecisions ~967 tokens（P95 代表样本，L≥512 可切分前缀） | 单样本冷态（设置 `CLEF_PREFIX_STATE_CACHE=1`，每次使用不同 dummy 后缀或独立缓存冷窗口，采集 5 次） | 量化全长 1-Pass Tap 前向 + Grouped Varlen SSM 捕获 + 异步 D2H 触发的真实设备耗时构成 |
| **W2: OD-Tier2-Hit** | 与 W1 同一 ~967 tokens 样本 | 预热 ≥10 次确保 100% 命中 Tier 2 后，采集 5 次请求窗口 | 量化仅算后缀 `S = T - L` 时的 MLP Cube、GDN `fla`、`FIA TND`、H2D 状态恢复与 JointHead 耗时 |
| **W3: ERQA-MultiImg-Cold** | ERQA 多图代表样本（如 3–4 图、`T ≈ 1,200–1,650` tokens，例如 `erqa-0382`） | Tier 1 & Tier 2 均未命中（使用 `scripts/prepare_systemone_uncached_media.py` 生成像素微扰副本以保留真实 ViT + Populate 开销） | 量化多图 ViT 编码 + 全长混合主干 + 跨图边界前缀捕获耗时 |
| **W4: ERQA-Tier2-Hit** | 与 W3 同形状原样本 | 预热 ≥10 次确保 Tier 1 + Tier 2 100% 命中后，采集 5 次请求窗口 | 验证 ViT 编码与长图像前缀主干计算从关键路径彻底消除后的后缀耗时结构 |

### 2.2 标准采集与解析命令

仓库已内置完整的双卡 Profile 控制与离线并集解析脚本（切勿简单把所有 kernel duration 直接相加充当延迟，必须使用 `analyze_systemone_profile.py` 计算双 rank 时间区间并集 `union_us`！）：

```bash
source scripts/activate_env.sh

TRACE_ROOT=/tmp/profile-tier2-traces
mkdir -p "$TRACE_ROOT"

# 1. 启动带 torch_npu HTTP Profiler 控制端点的服务（通过 -- 透传 --profiler-config）
bash start.sh \
  --model-dir /workspace/models/Clef-27B \
  --output-dir /tmp/profile-tier2-svc \
  --port 18243 \
  -- --profiler-config "{\"profiler\":\"torch\",\"torch_profiler_dir\":\"$TRACE_ROOT\",\"torch_profiler_with_stack\":false,\"torch_profiler_use_gzip\":true}" &

# 等待 /health 就绪后，调用仓库标准采集入口（自动执行采集外 warmup -> POST /start_profile -> 5 次测量 -> POST /stop_profile -> 独立进程 analyse -> analyze_systemone_profile.py）
python scripts/profile_systemone.py \
  --base-url http://127.0.0.1:18243 \
  --output-dir /tmp/profile-tier2-out/w2-od-hit \
  --trace-root "$TRACE_ROOT" \
  --expected-ranks 2 \
  --sample-file /tmp/profile-samples/od-967-eligible.json \
  --warmup 10 \
  --requests 5 \
  --concurrency 1
```

> **注意**：
> - 对于 **W1（Cold/Populate）**，因为 `profile_systemone.py` 强制 `--warmup >= 10`，可通过 `--warmup-sample-file` 传入另一条**不同前缀**的同长度样本完成算子/executor 预热，而 `--sample-file` 传入 5 条各自拥有唯一前缀（但长度同为 ~967 tokens）的样本，从而在采集窗口内精准抓到真实的无 executor 编译干扰的 **Tier 2 Populate** 路径！
> - 解析完成后，只将紧凑的 JSON 汇总（`manifest.json`、`trace-analysis/*.json`、算子分类聚合 CSV/JSON）拉回本地 `logs/profile-tier2-20261126/`，**严禁把数百 MB 的原始 `*_ascend_pt` Chrome trace 提交进 Git 仓库**。

---

## 3. 阶段二：基于 `figures4papers` 生成 4 张论文级图表

### 3.1 绘图规范检查单（必读）

绘图脚本必须在绘图前先完整阅读 `/Users/chenclay/Documents/code/projects/figures4papers/scientific-figure-making/SKILL.md`、`references/api.md` 与 `references/design-theory.md`，并可使用 `/Users/chenclay/Documents/code/projects/figures4papers/.venv/bin/python` 运行绘图脚本（脚本归档于 `scripts/plot_paper_figures.py` 或临时沙盒）：

1. **全局样式（`PUBLICATION_RCPARAMS`）**：
   - `matplotlib.use("Agg")`
   - 字体栈：`["Arial", "Helvetica", "DejaVu Sans", "sans-serif"]`（为避免中文字体缺失乱码，**图内坐标轴、图例、柱顶数值统一使用简洁规范的英文术语**，在 Markdown 正文图注中配中文说明）。
   - 隐藏上边框与右边框：`axes.spines.top = False`、`axes.spines.right = False`；无框图例：`legend.frameon = False`。
   - 线宽与字号：大图柱状面板 `font.size = 18–22`、`axes.linewidth = 2.5`；柱体统一加黑色描边 `edgecolor="black", linewidth=1.5`。
2. **语义配色（严格取自 `figures4papers` `PALETTE`）**：
   - 核心最终方案（Tier 2 Final）：`blue_main = "#0F4D92"` / `blue_secondary = "#3775BA"`
   - 渐进优化中间阶段：`teal = "#42949E"`、`green_3 = "#8BCF8B"`、`green_2 = "#AADCA9"`
   - 基线或冷态对照（Baseline / Cold）：`red_strong = "#B64342"` / `red_2 = "#E9A6A1"` / `neutral = "#CFCECE"`
3. **导出规格**：每张图同时导出 `dpi=300` 的 `.png`（供 `README.md` / `optimization.md` 直接渲染）与矢量 `.pdf`（备论文/答辩 PPT 使用）至 `imgs/` 目录。

### 3.2 四张核心图表规格设计

#### 图 1：`imgs/fig1_optimization_ablation.{png,pdf}` — 全栈渐进优化消融对比图
- **布局**：`1 × 2` 宽幅双面板（`figsize=(16, 5.8)`），左面板为 **S1 Tail Latency (`p95 ms`, Lower is Better ↓)**，右面板为 **S2 Max Throughput (`decisions/s`, Higher is Better ↑)**。
- **横轴阶段（6 阶段演进，数据均已固化于 `optimization.md` 第 4 节）**：
  1. `Baseline` (OD: `217.0 ms / 5.72 dec/s`, ERQA: `323.5 ms / 3.96 dec/s`) — `#B64342`
  2. `+BIC & ACLNN & NZ` (OD: `166.0 ms / 17.11 dec/s`) — `#CFCECE`
  3. `+ConvQKV & Chunk128` (OD: `155.9 ms / 17.48 dec/s`) — `#AADCA9`
  4. `+TQ2 & 48-AIV & MM-A2` (OD: `154.5 ms / 17.19 dec/s`, ERQA: `244.4 ms / 8.24 dec/s`) — `#8BCF8B`
  5. `+ConvQKV Varlen` (OD: `154.1 ms / 18.04 dec/s`, ERQA: `244.7 ms / 8.63 dec/s`) — `#3775BA`
  6. `+Tier 2 Cache (Final)` (OD: **`118.5 ms / 21.58 dec/s`**, ERQA: **`122.4 ms / 21.44 dec/s`**) — `#0F4D92`
- **视觉编码**：分组柱状图（OpenDecisions 实心 vs ERQA 斜线填充 `hatch="//"`），柱顶直接标注精确数值与最终总加速比徽标（`−45.4%` / `−62.2%` 延迟，`3.78×` / `5.42×` 吞吐）。

#### 图 2：`imgs/fig2_operator_breakdown.{png,pdf}` — 双卡算子级耗时拆解对比（Cold/Populate vs Tier 2 Hit）
- **数据来源**：由阶段一采集的 W1–W4 真实双卡 `union` 耗时填充（并可附带 967 token 历史分解前基线作对照）。
- **布局**：`1 × 2` 水平堆叠条形图或分组堆叠柱状图（`figsize=(15, 5.5)`），分别展示 **OpenDecisions (~967 tok)** 与 **ERQA Multi-Image (~1,500 tok)** 从 `Pre-Fusion Baseline` → `Tier 2 Populate (1-Pass Tap)` → `Tier 2 Hit (Suffix Resume)` 的设备耗时构成：
  - `ViT Vision Encoder`、`MLP Cube (FRACTAL_NZ)`、`GDN Core (FLA chunk128)`、`Fused Conv+QKV (48-AIV)`、`Full-Attn (FIA)`、`JointSchemaHead`、`HCCL Non-Overlapped`、`Cache Transfer / Capture`。
- **核心叙事**：直观呈现（1）1-Pass Tap Populate 相比无缓存几乎零额外开销；（2）Tier 2 Hit 将 ViT 与绝大部分前缀主干计算完全削去。

#### 图 3：`imgs/fig3_s2_concurrency_pareto.{png,pdf}` — S2 并发扩展与吞吐-延迟帕累托曲线
- **数据来源**：`logs/final-eval/od-s2/`（`C = 1, 2, 4, 8, 16, 32, 64, 128`）与 `logs/final-eval/erqa-s2/`（`C = 1, 8, 16`），对比 Tier 2 开启前后（`18.04 vs 21.58 dec/s`、`8.63 vs 21.44 dec/s`）。
- **布局**：`1 × 2` 双面板（`figsize=(15, 5.5)`）：
  - 左图：Client Concurrency `C`（对数刻度 `1..128`）vs Throughput (`decisions/s`)；
  - 右图：Latency-Throughput Trade-off (`p95 Latency ms` vs `decisions/s`)，高亮标注 `OD C32`（低延迟拐点 `21.54 dec/s @ 2.07 s`）、`OD C64`（峰值吞吐 `21.58 dec/s`）与 `ERQA C16`（`21.44 dec/s @ 823 ms`）。

#### 图 4：`imgs/fig4_prefix_handover_analysis.{png,pdf}` — 前缀切分收益与单进程跨数据集交接曲线
- **数据来源**：`logs/input-distribution-2026-11-16.json`、前缀审计脚本 `src/runtime/tiered_prefix_cache/audit_dataset_prefixes.py` 输出，以及 `docs/tiered-cache-stage0-2026-11-25.md` §11.8–11.10 的 `window=500` vs `window=3000` 跨集交接实测数据。
- **布局**：`1 × 2` 双面板（`figsize=(15, 5.5)`）：
  - 左图：OpenDecisions 与 ERQA 的总输入长度 `T` vs Tier 2 缓存命中的 128 对齐前缀 `L` vs 实际后缀计算长度 `S = T - L` 对比；
  - 右图：单进程零重启 `OD → ERQA` 连续轮次下，`ACTIVE_WINDOW=500`（平滑换出旧池，ERQA S2 跃升至 `21.44–22.30 dec/s`，末态 `0 miss`）对比旧 `window=3000`（旧池锁死仅 `8.08 dec/s`）的吞吐与命中率演化。

---

## 4. 交付验收清单

1. [ ] 远端采集完成并彻底关闭临时 `--profiler-config` 进程，`npu-smi info` 确认不残留多余调试进程；
2. [ ] `logs/profile-tier2-20261126/` 仅保留轻量汇总 JSON/CSV（单目录总体积 `< 2 MB`），无原始 trace 大文件；
3. [ ] 新增专题报告 `docs/profile-tier2-2026-11-26.md`，完整记录 W1–W4 双卡算子耗时表与采样方法；
4. [ ] `imgs/` 下生成 `fig1_optimization_ablation.{png,pdf}`、`fig2_operator_breakdown.{png,pdf}`、`fig3_s2_concurrency_pareto.{png,pdf}`、`fig4_prefix_handover_analysis.{png,pdf}`，目视检查每张 `.png` 无文字重叠、无裁切、无乱码；
5. [ ] 在 `README.md`（`## 优化概要` 与 `## 附录 -> 2. 性能剖析`）和 `optimization.md` 对应章节嵌入上述 `imgs/fig*.png` 图片引用，不改动任何业务源码与 `metrics.csv` / `results.csv`。

---

## 5. 可直接发送给 Worker 的任务指令（复制即用）

```text
请阅读并严格按 docs/profile-and-figures-plan-2026-11-26.md 执行当前版本（main @ 5e7352d）的补充 Profile 采集与论文级图表生成任务：
1. 先阅读 /Users/chenclay/Documents/code/projects/figures4papers/scientific-figure-making/SKILL.md 及其 references/{api.md,design-theory.md}，掌握出版级配色与版式规范；
2. 在远端昇腾 A3 容器激活项目 .venv，按任务书第 2 节使用独立端口与 --profiler-config 采集 W1–W4 四个代表窗口（OD ~967 tok Cold/Populate vs Hit，ERQA 多图 Cold/Populate vs Hit），用 scripts/analyze_systemone_profile.py 计算双 rank union 算子拆解，清理临时进程后将轻量汇总存入 logs/profile-tier2-20261126/ 并撰写 docs/profile-tier2-2026-11-26.md（切勿提交原始 trace 大文件，切勿改动任何模型/缓存业务源码或 results.csv）；
3. 使用 /Users/chenclay/Documents/code/projects/figures4papers/.venv/bin/python 生成任务书第 3.2 节定义的 4 张图表（同时导出 300 DPI .png 与矢量 .pdf 至 imgs/），逐张读取 .png 自检排版无遮挡/裁切后，将图片嵌入 README.md 与 optimization.md 对应章节并汇报结果。
```
