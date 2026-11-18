# Clef-27B 提交版最终汇总（2026-11-17）

## 0. 版本与配置

| 项 | 值 |
| --- | --- |
| 分支 / HEAD | `codex/tuning-perf2a-20261113`（本轮最终提交见第 6 节） |
| 服务入口 | `bash Clef-27B/start.sh --model-dir <dir> --output-dir <dir> --config s1` |
| 硬件 | A3 单物理卡内 2 个逻辑 NPU（`Ascend910_9382`，各 64 GiB） |
| CANN / Python | 9.2.1 / CPython 3.12 |
| 主干 | vLLM `24420b7a`、vLLM-Ascend `89eb8a5e`、BF16 官方权重 |
| 服务配置 | TP2、BF16、eager、max_model_len 4096、seqs1、prefix cache 关、chunked prefill 关 |
| 已采用的优化 | GDN 决策专用 prefill、BIC、QKV norm v2（cap 1600）、单字段 joint head、`ACLNN_CACHE_LIMIT=1000000`、**MLP BF16 NZ（FRACTAL_NZ）** |
| 安装器 | GDN 决策补丁 **v4**（含 MLP NZ hook，`CLEF_MLP_BF16_NZ=1`）、QKV norm v2、单字段 head |
| 规则合规 | 无量化、无温度校准、单物理卡内 2 逻辑 NPU |

`ACLNN_CACHE_LIMIT=1000000` 与 `CLEF_MLP_BF16_NZ=1` 均写入 `Clef-27B/inference.py` 的 service environment，并在 `launch-manifest.json` 的 `service_environment` 中记录；MLP NZ hook 由 GDN 安装器 v4 写入 `patch_qwen3_5.py`。

## 1. 质量指标

| 指标 | baseline（2026-11-14） | final（2026-11-17） | 官方公开 as-served | final − baseline | final − official |
| --- | ---: | ---: | ---: | ---: | ---: |
| OpenDecisions Calibration | 82.7941 | **82.7092** | 82.3 | −0.0849 | +0.4092 |
| OpenDecisions Intelligence | 66.0568 | **65.9122** | 65.8 | −0.1446 | +0.1122 |
| OpenDecisions Capability | 74.4255 | **74.3107** | 74.1 | −0.1148 | +0.2107 |
| OpenDecisions accuracy | 1906/2352 | **1902/2352** | — | −4 题（−0.17 pp） | — |
| OpenDecisions Score MAE | 0.401389 | **0.401545** | — | +0.000156 | — |
| ERQA accuracy | 219/400 = 0.5475 | **217/400 = 0.5425** | 未公布 | −0.50 pp | — |

- **2PP 口径**（无温度校准；三项绝对波动 ≤ 2 pp）：OD 三项最大变化 0.145 pp，ERQA 0.50 pp，全部通过；相对官方 as-served 为 **+0.409 / +0.112 / +0.211 pp**，2PP 内。
- `ACLNN_CACHE_LIMIT` 与 MLP BF16 NZ 均为**逐位质量中性**：官方入口全量 2,352 条响应的概率、choice、noul、score 与修复前逐位相同（max abs diff 0.0、0 flips）；ERQA 仍为 217/400（400/400 成功、0 重试）。
- 官方正式 baseline 与 ERQA 公开值仍未取得；上表同时保留本地 baseline 与数据卡 as-served 两种对照。

## 2. 性能指标

### 2.1 S1：单并发端到端 P95（全量 2,352，C1）

| 版本 | mean | p50 | p95 | p99 | 吞吐 |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline（2026-11-14） | 180.07 ms | 205.03 ms | 217.04 ms | 222.14 ms | 5.549 dec/s |
| 前一版（BIC + GDN v2 + 单字段 head） | 153.76 ms | 168.49 ms | 190.40 ms | 203.20 ms | 6.503 dec/s |
| + aclnn cache（官方入口） | 125.55 ms | 118.17 ms | 168.63 ms | 203.45 ms | 7.960 dec/s |
| **+ aclnn cache + MLP NZ（官方入口）** | **121.71 ms** | **113.66 ms** | **166.00 ms** | **204.32 ms** | **8.213 dec/s** |
| 相对 baseline | −32.41% | −44.57% | **−23.51%** | −8.02% | **+48.0%** |
| 相对前一版 | −20.85% | −32.54% | **−12.82%** | +0.55% | **+26.3%** |
| 相对 cache-only | −3.06% | −3.82% | **−1.56%** | +0.43% | **+3.18%** |

- MLP NZ 三组全量复测 p95 **165.53 / 165.84 / 166.00 ms**，mean 119.37–121.71 ms，吞吐 8.21–8.37 dec/s；cache-only 多组 p95 167.28–169.02 ms。
- 分长度 p50：140–307 117.4；308–500 119.5；501–700 125.4；701–1000 156.4；1001–1600 192.8 ms（cache-only 测量）；p95 由 1,000–1,300 token 的长请求主导，已接近设备计算极限。
- 首个请求仍有约 0.5–0.7 s 冷启动（1/2,352 样本），不影响 p95。

### 2.2 S2：最大有效吞吐（全量 2,352）

| 配置 | 并发 | 吞吐 | p95 | mean |
| --- | ---: | ---: | ---: | ---: |
| baseline（seqs1） | C16 | 5.715 dec/s | 3292.7 ms | — |
| 前一版 B8 C16（v2+head） | C16 | 16.398 dec/s | 1802.9 ms | 971.6 ms |
| **final B8 C16（aclnn cache + MLP NZ）** | C16 | **16.376 dec/s** | 1803.2 ms | 973.3 ms |
| final B16 C64（无 p95 约束） | C64 | 17.110 dec/s | 6088.1 ms | — |

- 最终 S2 相对 baseline **2.87×**；MLP NZ 对批处理无额外收益（16.376 vs 16.398，噪声内），报告沿用最终配置实测值 16.376。
- S2 B8 + NZ 质量：Capability **74.1572**（提交 B8 74.2510，Δ−0.094）、accuracy 1908/2352、2 条 choice 变化、max 概率差 0.1002；与既有合批波动同量级，2PP 通过。
- 若规则只看最大吞吐，B16 C64 为 17.110 dec/s；若看 p95/有效延迟，B8 C16 是推荐点。`results.csv` 按 B8 C16 记录。

## 3. 本轮新增优化

### 3.1 aclnn executor cache（−21.8 ms p95）

- 默认 aclnn executor 缓存只覆盖约 145 个不同 prompt 长度；之后首次出现的新 shape 固定多约 55 ms，输出与快档逐位相同。
- `ACLNN_CACHE_LIMIT=1000000` 后 755 个 OD 长度全部快档：308–600 中位从约 175 ms 降到约 119 ms。
- 官方入口全量 S1：mean 125.55 / p50 118.17 / p95 168.63 ms、7.960 dec/s（原 153.76 / 168.49 / 190.40、6.503）；打乱数据集顺序复测 p95 168.33，证明不依赖请求顺序。
- 详见 `docs/aclnn-cache-shape-fix-2026-11-17.md`。

### 3.2 MLP BF16 NZ（再 −2.6 ms p95）

- `Qwen2MoeMLP.forward` 首次调用时把 `gate_up_proj` / `down_proj` 权重转为 FRACTAL_NZ（`torch_npu.npu_format_cast(..., 29)`），`CLEF_MLP_BF16_NZ=1` 控制；GDN 安装器升级为 v4。
- 官方入口全量 S1：**mean 121.71 / p50 113.66 / p95 166.00 ms、8.213 dec/s**；相对 cache-only p95 −1.56%、mean −3.06%、吞吐 +3.18%。
- 全量 2,352 条逐位一致；ERQA 217/400；S2 16.376（neutral）。
- 详见 `Clef-27B/optimization.md` 的 2026-11-17 MLP BF16 NZ 节。

## 4. 当前瓶颈与后续

- S1 p95（166.00 ms）由 1,000–1,300 token 的长请求主导，设备 compute 约 150–170 ms；不量化条件下的剩余杠杆是 MLP tiling（上限 3–4%）、GDN 主核、通信重叠（Dense SP/MC2 已实测不可用/退化）。
- 中短请求约 110–125 ms，已接近 CPU/设备平衡；前端打点显示 encode 2.35 ms、`tolist()`/构建 <0.02 ms，其余约 5.8 ms 在 vLLM engine client 内部（`docs/frontend-timing-2026-11-17.md`）。
- 单字段 head 的 attention 手工路径实测仅省约 0.5 ms 且非逐位，未采用。
- 不建议：量化（赛方禁止）、完整 graph（P95 变差）、Dense SP（退化）、MC2（A3 不可用）、joint head 跨请求批处理（无收益）。
- 开放项：主办方正式 baseline、ERQA 公开值、S2 有效性规则（B8 vs B16）。

## 5. 提交物与证据

- 入口：`Clef-27B/start.sh`、`Clef-27B/inference.py`（含 `ACLNN_CACHE_LIMIT`、`CLEF_MLP_BF16_NZ`）。
- 服务补丁：`runtime/apply_gdn_decision_patch.py`（v4，含 MLP NZ hook）、`runtime/apply_qkv_norm_patch.py`、`runtime/apply_joint_single_field_patch.py`（三者 `--check` 均为 ready）。
- 质量证据：`Clef-27B/logs/gdn-data-path-20261115/head-entry/`、`Clef-27B/logs/tuning5-20261117/erqa-nz/`。
- S1/S2 证据：`Clef-27B/logs/tuning5-20261117/s1-official-nz-full.json.gz`、`s1-official-nz-launch.json`、`s2/s2-nz-c16.json.gz`、`s2/s2-nz-c16-quality.json`。
- 逐位比对：`Clef-27B/logs/tuning5-20261117/quality-compare.json`（缓存）与本轮 NZ 全量比对（max 0.0）。
- 完整运行目录：`/workspace/yangchen/logs/tuning5-20261117/`。
- 复现：`docs/aclnn-cache-shape-fix-2026-11-17.md` 第 5 节；MLP NZ 由 GDN v4 安装器在启动时写入。
