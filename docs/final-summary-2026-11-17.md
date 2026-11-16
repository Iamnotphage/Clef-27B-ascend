# Clef-27B 提交版最终汇总（2026-11-17）

## 0. 版本与配置

| 项 | 值 |
| --- | --- |
| 分支 / HEAD | `codex/tuning-perf2a-20261113` / `e435ae7` |
| 服务入口 | `bash Clef-27B/start.sh --model-dir <dir> --output-dir <dir> --config s1` |
| 硬件 | A3 单物理卡内 2 个逻辑 NPU（`Ascend910_9382`，各 64 GiB） |
| CANN / Python | 9.2.1 / CPython 3.12 |
| 主干 | vLLM `24420b7a`、vLLM-Ascend `89eb8a5e`、BF16 官方权重 |
| 服务配置 | TP2、BF16、eager、max_model_len 4096、seqs1、prefix cache 关、chunked prefill 关 |
| 已采用的优化 | GDN 决策专用 prefill、BIC 数据路径、QKV norm v2（cap 1600）、单字段 joint head、**`ACLNN_CACHE_LIMIT=1000000`** |
| 规则合规 | 无量化、无温度校准、单物理卡内 2 逻辑 NPU |

`ACLNN_CACHE_LIMIT=1000000` 已写入 `Clef-27B/inference.py` 的 service environment（默认值，可由外部覆盖），并在 `launch-manifest.json` 的 `service_environment` 中记录。

## 1. 质量指标

| 指标 | baseline（2026-11-14） | final（2026-11-17） | 官方公开 as-served | final − baseline | final − official |
| --- | ---: | ---: | ---: | ---: | ---: |
| OpenDecisions Calibration | 82.7941 | **82.7092** | 82.3 | −0.0849 | +0.4092 |
| OpenDecisions Intelligence | 66.0568 | **65.9122** | 65.8 | −0.1446 | +0.1122 |
| OpenDecisions Capability | 74.4255 | **74.3107** | 74.1 | −0.1148 | +0.2107 |
| OpenDecisions accuracy | 1906/2352 | **1902/2352** | — | −4 题（−0.17 pp） | — |
| OpenDecisions Score MAE | 0.401389 | **0.401545** | — | +0.000156 | — |
| ERQA accuracy | 219/400 = 0.5475 | **217/400 = 0.5425** | 未公布 | −0.50 pp | — |

- **2PP 口径**（比赛不需要温度校准；三项绝对波动 ≤ 2 pp）：OD 三项最大变化 0.145 pp，ERQA 0.50 pp，全部通过；相对官方 as-served 三项为 **+0.409 / +0.112 / +0.211 pp**，也在 2PP 内。
- `ACLNN_CACHE_LIMIT` 修复为**逐位质量中性**：2,352 条 OD 响应的概率、choice、noul、score 全部与修复前逐位相同（max abs diff 0.0、0 flips）；ERQA 仍为 217/400。
- 官方正式 baseline 与 ERQA 公开值仍未取得；上表同时保留本地 baseline 与数据卡 as-served 两种对照。

## 2. 性能指标

### 2.1 S1：单并发端到端 P95（全量 2,352，C1）

| 版本 | mean | p50 | p95 | p99 | 吞吐 |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline（2026-11-14） | 180.07 ms | 205.03 ms | 217.04 ms | 222.14 ms | 5.549 dec/s |
| final 前一版（head-entry） | 153.76 ms | 168.49 ms | 190.40 ms | 203.20 ms | 6.503 dec/s |
| **final + cache（官方入口）** | **125.55 ms** | **118.17 ms** | **168.63 ms** | **203.45 ms** | **7.960 dec/s** |
| 相对 baseline | −30.3% | −42.4% | **−22.3%** | −8.4% | **+43.4%** |
| 相对前一版 | −18.3% | −29.9% | **−11.4%** | +0.1% | **+22.4%** |

- 四组独立全量复测的 p95 为 **167.28 / 169.01 / 169.02 / 168.63 ms**，mean 123.67–128.99 ms，吞吐 7.75–8.08 dec/s；官方 `start.sh` 入口本轮为 168.63 ms / 7.960 dec/s。
- 分长度 p50：140–307 117.4；308–500 119.5；501–700 125.4；701–1000 156.4；1001–1600 192.8 ms。p95 由 1,000–1,300 token 的长请求主导，已回到设备计算极限；中短请求已从慢档约 175 ms 降到约 120 ms。
- 首个请求仍有约 0.5–0.6 s 冷启动，1/2,352 样本，不影响 p95。

### 2.2 S2：最大有效吞吐（全量 2,352）

| 配置 | 并发 | 吞吐 | p95 | mean |
| --- | ---: | ---: | ---: | ---: |
| baseline（seqs1） | C16 | 5.715 dec/s | 3292.7 ms | — |
| **final B8 C16（`--config s1 --max-num-seqs 8 --max-num-batched-tokens 8192`）** | C16 | **16.398 dec/s** | 1802.9 ms | 971.6 ms |
| final + cache B8 C16（复测） | C16 | 16.319 dec/s | 1807.2 ms | 976.6 ms |
| final + cache B16 C64（复测） | C64 | 16.928 dec/s | 6120.2 ms | 3743.0 ms |

- 最终 S2 相对 baseline **2.87×**；`ACLNN_CACHE_LIMIT` 对批处理形状无收益（B8 16.319、B16 16.928，均与 pre-cache 16.398/17.110 在噪声内），S2 报告沿用提交版 16.398 dec/s。
- S2 B8 cache 复测质量：Capability **74.1516**（提交 B8 74.2510，Δ−0.099）、accuracy 1908/2352（−1）、2 条 choice 翻转、max 概率差 0.1002；与既有合批波动同量级，2PP 通过。
- 若规则只看最大吞吐，B16 C64 为 17.110 dec/s；若看 p95/有效延迟，B8 C16 是推荐点。`results.csv` 按 B8 C16 记录。

## 3. 本轮新增：aclnn executor cache 修复

- **现象**：提交版 S1 按精确 input token 数分成快/慢两档，慢档固定多约 55 ms；同长度请求 within-length sd 中位 4.7 ms；快/慢档输出逐位相同。
- **根因**：CANN aclnn executor 缓存默认容量只覆盖前约 145 个不同 prompt 长度；之后首次出现的新 shape 走慢路径。离线探针排除了 MLP、GDN chunk、conv、QKV norm 的算子时间跳变。
- **修复**：`ACLNN_CACHE_LIMIT=1000000`；755 个 OD 长度暖机全部快档，308–600 从约 175 ms 降到约 119 ms。
- **验证**：官方 `start.sh --config s1` 入口全量 2,352 条，2,352 成功；响应与提交版逐位一致；ERQA 217/400 不变。
- 详细报告：`docs/aclnn-cache-shape-fix-2026-11-17.md`；证据：`Clef-27B/logs/tuning5-20261117/`。

## 4. 当前瓶颈与后续

- S1 p95（168.63 ms）由 1,000–1,300 token 的长请求主导，设备 compute 约 150–170 ms；BF16 不量化条件下的剩余杠杆是 MLP tiling（上限 3–4%）、GDN 主核、通信重叠（Dense SP/MC2 已实测不可用/退化）。
- 中短请求约 118–125 ms，已接近 CPU/设备平衡；head/pooler 与前端仍有约 6–8 ms host 成本，但单字段 head 的 attention 优化实测仅约 0.5 ms 且非逐位，未采用。
- 不建议：量化（赛方禁止）、完整 graph（P95 变差）、Dense SP（退化）、MC2（A3 不可用）、joint head 跨请求批处理（无收益）。
- 开放项：主办方正式 baseline、ERQA 公开值、S2 有效性规则（B8 vs B16）。

## 5. 提交物与证据

- 入口：`Clef-27B/start.sh`、`Clef-27B/inference.py`（含 `ACLNN_CACHE_LIMIT` 默认值）。
- 服务补丁：`runtime/apply_gdn_decision_patch.py`、`runtime/apply_qkv_norm_patch.py`、`runtime/apply_joint_single_field_patch.py`（三者 `--check` 均为 ready）。
- 质量证据：`Clef-27B/logs/gdn-data-path-20261115/head-entry/`、`Clef-27B/logs/tuning5-20261117/erqa/`。
- S1 证据：`Clef-27B/logs/tuning5-20261117/s1-official-entry-full.json.gz`、`s1-aclnn-cache-full.json.gz`、`s1-aclnn-repeat-run1/2.json.gz`。
- 逐位比对：`Clef-27B/logs/tuning5-20261117/quality-compare.json`。
- 复现：`docs/aclnn-cache-shape-fix-2026-11-17.md` 第 5 节；完整运行目录 `/workspace/yangchen/logs/tuning5-20261117/`。
