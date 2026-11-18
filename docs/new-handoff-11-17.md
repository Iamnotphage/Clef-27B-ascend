# Clef-27B Ascend 交接文档（2026-11-17 第二轮）

## 0. 交接摘要

当前可交付提交版本（chunk 64 + aclnn cache + MLP NZ）：

- **S1（seqs1，2,352 题）mean 121.705 / p50 113.656 / p95 166.002 / p99 204.324 ms，8.213 dec/s**；3 轮短预热平均 121.01 / 111.60 / 165.95 / 204.04，8.26 dec/s；
- baseline（2026-11-14）S1 180.069 / 205.034 / 217.036 / 222.139 ms、5.549 dec/s；相对 baseline p95 **−23.5%**、吞吐 **+48.0%**；
- **S2 B8 C16 16.376 dec/s、p95 1803.2 ms、mean 973.3**（baseline 5.715，**2.87×**）；B16 C64 17.110 只按最大吞吐；
- OpenDecisions Calibration **82.7092** / Intelligence **65.9122** / Capability **74.3107**、accuracy **1902/2352**、Score MAE **0.401545**；ERQA **217/400 = 0.5425**（400/400、0 重试）；
- 相对官方公开 as-served（82.3/65.8/74.1）三项 **+0.409 / +0.112 / +0.211 pp**，2PP 内；相对本地 baseline 最大变化 0.145 pp；无温度校准、无量化；
- 质量逐位：aclnn cache 与 MLP NZ 均为逐位中性（2,352 条 max prob diff 0.0、0 choice 翻转）。

分支 `codex/tuning-perf2a-20261113`，HEAD `ef4de9a`，已 push；本地仅用户原有 `docs/profiling-2026-11-12.md` 未提交（不要动）。

## 1. 机器与环境

| 项 | 值 |
| --- | --- |
| 机器 | `root@199.88.54.221`，A3 单物理卡内 2 个逻辑 NPU（`Ascend910_9382`，各 64 GiB） |
| 资源约束 | 全部推理/评测进程合计限用这 2 个逻辑 NPU，禁止跨物理卡 |
| OS / CPU / RAM | Ubuntu 22.04.5 aarch64，32 核等价 CPU，128 GiB |
| CANN / Python | 9.2.1 / `/workspace/yangchen/venvs/clef-py312` |
| vLLM / vLLM-Ascend | `24420b7a` / `89eb8a5e` |
| 模型 | `/workspace/yangchen/models/Clef-27B`（官方 BF16） |
| 工程 | `/workspace/yangchen/Clef-27B-ascend`，editable 安装 |

SSH 使用会话提供的跳板凭据；不要把 `jt_账号:令牌` 写入仓库、日志或聊天。重连用 `! ./connect.sh`。

**重要环境风险**：2026-11-17 上午 host CPU load 从 34 升到 **210**（外部租户争用，非本项目进程；服务停掉后仍上升）。此状态下所有性能测试会被整体抬高 20–25 ms，任何 S1/S2 数字都无效；开测前先看 `uptime`，要求 load 稳定在个位数。

## 2. 当前提交配置与入口

### 2.1 配置

- TP2 / BF16 / eager / max_model_len 4096 / max_num_seqs 1 / max_num_batched_tokens 4096 / prefix cache 关 / chunked prefill 关；
- GDN 决策专用 prefill + BIC 数据路径 + QKV norm v2（cap 1600，>1600 回退 BIC）+ 单字段 joint head；
- **`ACLNN_CACHE_LIMIT=1000000`**（CANN aclnn executor 缓存）；
- **MLP BF16 NZ**（`CLEF_MLP_BF16_NZ=1`，GDN 安装器 **v4**，`Qwen2MoeMLP` gate_up/down 转 FRACTAL_NZ）；
- **GDN chunk 128 已接入但默认关闭**（`CLEF_GDN_CHUNK128=1` 才启用，staged；见第 5 节）。

`ACLNN_CACHE_LIMIT`、`CLEF_MLP_BF16_NZ` 写入 `Clef-27B/inference.py` 的 service environment，launch manifest 记录。

### 2.2 入口

```bash
# S1
bash Clef-27B/start.sh --model-dir /workspace/yangchen/models/Clef-27B \
  --output-dir <run>/s1 --config s1 --port 18143

# S2 最终 B8（仍用 --config s1 覆盖；不要用 --config s2）
bash Clef-27B/start.sh --model-dir /workspace/yangchen/models/Clef-27B \
  --output-dir <run>/s2 --config s1 \
  --max-num-seqs 8 --max-num-batched-tokens 8192 --port 18143
```

S2 的 `--config s2` 是历史 schema-cache-on 候选，未采用。

### 2.3 全量评测

```bash
python scripts/benchmark_systemone.py --base-url http://127.0.0.1:18143 \
  --sample-file /workspace/yangchen/logs/tuning/20261114-013000-fix4/opendecisions/all2352.json \
  --warmup 0 --requests 2352 --concurrency 1 --report <run>/s1.json

python scripts/official_eval/erqa_baseline.py \
  --data-dir /workspace/yangchen/datasets/erqa-official-20261114 \
  --output-dir <run>/erqa --api-url http://127.0.0.1:18143/v1 --scope full
python scripts/official_eval/erqa_score.py \
  --run-dir <run>/erqa --data-dir /workspace/yangchen/datasets/erqa-official-20261114 \
  --output <run>/erqa/offline-score.json --verify-requests
```

ERQA 服务需 `--max-model-len 16384 --max-num-batched-tokens 16384 --limit-mm-per-prompt '{"image":16,"video":1}'`，图像像素 65,536–1,048,576。

## 3. 本轮采用的优化

### 3.1 aclnn executor cache（S1 p95 190.40 → 168.63 ms）

- 根因：CANN aclnn executor 缓存默认只覆盖约 **145 个不同 prompt 长度**（OD 有 755 个）；缓存耗尽后新形状走慢路径，固定 +约 55 ms，且逐位质量不变。
- 修复：`ACLNN_CACHE_LIMIT=1000000`；755 个长度全快，308–600 从约 175 降到约 119 ms。
- 证据：`docs/aclnn-cache-shape-fix-2026-11-17.md`、`Clef-27B/logs/tuning5-20261117/`（`s1-aclnn-cache-full`、`s1-official-entry-full`、`s1-aclnn-repeat-*`、`s1-official-shuffled-full`）。
- 诊断要点：同长度 within-length sd 中位 4.7 ms；快/慢档输出逐位相同；离线探针排除 MLP/GDN/conv/QKV 算子时间跳变；旧 handoff 的“307 CPU dispatch 临界”是 307 profile 被采集开销推慢（od-0105 生产 114.38 vs profile 169.55），不可用；967 profile 仍有效。

### 3.2 MLP BF16 NZ（S1 p95 168.63 → 166.00 ms）

- `Qwen2MoeMLP.gate_up_proj/down_proj` 首次前向转 FRACTAL_NZ；GDN 安装器升级为 **v4**（`patch_qwen3_5.py` 追加 `CLEF_MLP_BF16_NZ` hook）。
- 官方入口 S1 mean 121.705 / p50 113.656 / p95 166.002 ms、8.213 dec/s；相对 cache-only p95 −1.56%、mean −3.06%、吞吐 +3.18%；2,352 条逐位一致；ERQA 217/400；S2 16.376（neutral）。
- 证据：`docs/final-summary-2026-11-17.md`、`Clef-27B/logs/tuning5-20261117/s1-official-nz-full.json.gz`、`s2/s2-nz-*`、`erqa-nz/`。

### 3.3 Warmup（测量协议，未内置）

- 32 条短预热（8 样本 ×4，200–1500 token，约 5.6 s）+ 3 轮全量平均：mean 121.01 / p50 111.60 / p95 165.95 / 204.04、8.26 dec/s；p95 与无预热 166.00 持平，首请求 677.9 ms 冷启动被清掉，但后续轮仍有 500–647 ms 瞬时离群。
- 全量预热 2,352 条变体（启动多 5 分钟）在 host load 138–187 下作废；低负载下此前第二遍为 119.37 / 109.10 / 165.43 / 8.373。
- 结论：若比赛有 warmup 阶段，32 条短预热即可；无 warmup 阶段不要放后台预热（会与首批测量请求竞争）。
- 证据：`docs/warmup-2026-11-17.md`、`Clef-27B/logs/tuning5-20261117/warmup-multiround.json`。

## 4. 800+ token 专项分析与负结果

967 token profile（`docs/profile-final-2026-11-17.md`，cache-only、NZ 前）：

- compute **144.6 ms / 窗口 174.3 ms（83%）**；通信非重叠 17.2 ms（10%）；非 kernel 12.5 ms（7%）。
- compute：MLP gate/up 42.65（29.5%）、GDN 主核 23.31（16.1%）、MLP down 21.23（14.7%）、其他 matmul 约 28（19%）、AddRmsNorm 5.96、QKV norm 4.66、conv 3.79、SwiGlu 2.39、attention 1.67。
- 分桶 mean：0–400 约 108–112、401–600 约 116、601–800 约 131、801–1000 约 157、1001–1200 约 186、1201–1400 约 210、1401–1600 约 237 ms。

已试、不采用：

| 方向 | 结果 |
| --- | --- |
| MLP M padding | 1,084/1,200/1,258/1,595 全部净收益 ≈ 0 |
| GDN chunk 128（未归一化探针） | NaN；后确认是探针输入问题（见第 5 节） |
| All-linear BF16 NZ | 每层 5 个权重，全量 p95 166.61 vs 166.00，无收益 |
| QKV norm v3（V 直拷） | V 逐位，单算子 v2/v3 持平；服务 165.54/166.06，无收益；源码恢复 v2 |
| 前端 8 ms 缺口 | encode 2.35 ms、tolist/构建 <0.02 ms，其余约 5.8 ms 在 vLLM engine client 内部 |
| 单字段 head attention | 手工 SDPA 仅省约 0.5 ms 且非逐位 |

高成本剩余方向：MLP 自定义 BF16 tiling（端到端 2–3%，1,200 token 已 294 TFLOPS / 平台 300–320）、QKV Q/K `WholeReduceSum` tile 归约（≤1%，v1 遇 UB 对齐/NaN）、conv+QKV norm 融合（1–2 ms）、GDN 主核重写；通信重叠（Dense SP/MC2/RS）历史上全部退化或不可用。

证据：`docs/profile-800plus-2026-11-17.md`、`docs/qkv-v3-2026-11-17.md`、`docs/frontend-timing-2026-11-17.md`、`Clef-27B/logs/tuning5-20261117/shape-probe/`。

## 5. GDN chunk 128（已 staged，默认关闭）

- 关键更正：之前 NaN 是探针用**未归一化随机 q/k**；服务 q/k 已过 `ClefGdnQkvNorm` L2Norm。归一化后 chunk 128 完全可用：NaN=0、与 64 仅 1 ulp，单算子 967 0.594→0.470（−20.9%）、1,200 0.587→0.515（−12.3%）、10,294 3.019→2.621（−13.2%）。
- 服务全量（chunk 128 + cache + MLP NZ，host load 34）：mean 124.37 / p50 118.18 / **p95 164.20** / p99 202.24、8.036 dec/s；相对 chunk 64 的 166.00，p95 −1.80 ms（−1.08%），mean/p50 受负载影响偏高。
- 质量：OD Calibration 82.7056 / Intelligence 66.1169 / Capability **74.4113**（2PP），ERQA **217/400 = 0.5425**（一致）；但逐题 **9 条 choice 翻转、max 概率差 0.361、1,087 条 >1e-3**，过不了项目内部 0.03/0.05 门槛（比赛官方 2PP 通过）。
- 官方入口在 host load 44+ 下 S1 为 mean 137.58 / p95 169.31 / 7.256（负载污染，不可与 166.00 直接比）；当前 host load 210，**clean S1/S2 待补**。
- 接入方式：
  - `runtime/apply_gdn_chunk128_patch.py`：在 GDN/QKV 安装器之后 apply；改 `device_op.py` 两个决策分支 `chunk_size=128` 和 `gdn_attn_builder.py` `_GDN_CHUNK_SIZE=128`；
  - GDN/QKV 安装器已识别 chunk128 postimage（device_op `bf027ec9…`、builder `f81eeefa…`），从 chunk128 源码重启可通过；
  - 开关 `CLEF_GDN_CHUNK128=1`（默认关闭）；`--revert` 回 QKV 状态；
  - 启用命令：`CLEF_GDN_CHUNK128=1 bash Clef-27B/start.sh ... --config s1`；S2 加 `--max-num-seqs 8 --max-num-batched-tokens 8192`。
- 决策：默认保持 chunk 64；等 host load 回到个位数，跑 clean 官方入口 S1 + S2 B8 C16 + 质量；若 clean p95 <166 且 S2 2PP，再打开默认。
- 证据：`docs/gdn-chunk128-2026-11-17.md`、`Clef-27B/logs/tuning5-20261117/s1-chunk128-full.json.gz`、`chunk128-quality.json`、`erqa-chunk128/`。

## 6. 当前源码/机器状态

| 项 | 状态 |
| --- | --- |
| 服务 | 无进程，18143 空闲，两个逻辑 NPU 空闲 |
| `device_op.py` | QKV postimage，SHA `19cf5d0f…`（chunk 64） |
| `gdn_attn_builder.py` | `_GDN_CHUNK_SIZE=64`，SHA `1f464f9c…` |
| GDN 决策补丁 | v4（含 MLP NZ hook），`--check` ready |
| QKV norm v2 | vendor 已恢复 v2（kernel `.o` `e0ca1454…`），`--check` ready |
| 单字段 head | `--check` ready |
| chunk 128 | installer staged，默认关闭；GDN/QKV 识别其 postimage |
| host load | **210**（2026-11-17 10:09，外部争用）；不要在此状态测性能 |
| QKV v3 | 源码归档 `runtime/qkv_norm_v3_kernel.cpp`，vendor v2，未采用 |

远端提交包（`/workspace/yangchen/Clef-27B-ascend`，main 工作区）已更新但未提交：
- `Clef-27B/inference.py`：`ACLNN_CACHE_LIMIT`、`CLEF_MLP_BF16_NZ`、chunk128 gated installer；
- `Clef-27B/results.csv`：S1 `166.0024710232392`、S2 `16.37641244039108`，measurement id 含 `aclnn-cache-mlp-nz`；
- `Clef-27B/README.md`、根 `README.md`、`Clef-27B/optimization.md`：S1/S2/质量表与 2026-11-17 两节（aclnn cache、MLP NZ）。
- 远端 origin 是本地 bundle；这些提交包修改在远端 main 工作区未提交。

本地仓库：分支 `codex/tuning-perf2a-20261113`，HEAD `ef4de9a`，已 push；仅 `docs/profiling-2026-11-12.md` 未提交（用户原有）。

## 7. 关键提交

- `c54d042` aclnn executor cache；
- `e435ae7` 官方入口缓存验证；`4024c9f` 最终汇总；
- `81e57bc`/`1994f98` S2 复测/打乱顺序；
- `af6fec1`/`2f73ffd` 前端定位；
- `e1d3005` warmup；`2785e76` 3 轮平均；
- `b7d1be3`/`b734092` 800+ 分析、all-linear NZ 负结果；
- `f3d9004` QKV v3；
- `59866ca`/`43502fe` chunk128 质量/ERQA；
- `ef4de9a` chunk128 staged + 安装器幂等。

## 8. 下一步 / 开放问题

1. **等 host load 回落到个位数**，跑 clean 官方入口：
   - `CLEF_GDN_CHUNK128=1` 全量 S1；
   - S2 B8 C16；
   - OD 三项 + ERQA 质量复核（预期 Capability 74.4113、ERQA 217/400）。
   - 若 clean p95 <166 且 S2 在 2PP，打开 chunk128 默认；否则保持 chunk64。
2. 主办方正式 baseline、ERQA 官方公开值仍未取得；S2 有效性规则（B8 vs B16）未明确。
3. 剩余高成本 lever：MLP 自定义 tiling（2–3%）、QKV Q/K tile 归约（≤1%）、conv+QKV 融合（1–2 ms）、GDN 主核；通信重叠全部失败。
4. 不要量化（赛方禁止）；不要用温度校准。

## 9. 交接注意

- 不要提交 `docs/profiling-2026-11-12.md`；
- 性能测试前先 `uptime`，load >10 不要测；
- 不要跨物理卡；不要 kill 同门/其他进程；
- 模型权重、SSH 凭据、跳板令牌不入库；
- 服务恢复用 `/workspace/yangchen/logs/gdn-data-path-20261115/phase0/snapshot.json`（若需要），恢复后跑冻结 8 例；
- chunk128 的 `--revert` 会回到 QKV 状态；GDN/QKV 安装器已识别 chunk128 postimage，重启不会失败。
