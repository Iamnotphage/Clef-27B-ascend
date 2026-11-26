# 多粒度 Tier 2 最终评测归档（2026-11-26）

## 目的与协议

本目录是 `feat/multi-tier-cache-20261126`（基线 `main@66d3c7b`）的正式评测证据：
单一常驻 `bash start.sh --model-dir ... --output-dir ...` TP2 服务、零重启，顺序完成：

1. OpenDecisions S1：warmup + R1/R2/R3（C1，全量 2,352）；
2. OpenDecisions S2：C=1,2,4,8,16,32,64,128 扫描 + 峰值 C64 三次重复；
3. ERQA S1：warmup + R1/R2/R3（C1，全量 400）；
4. ERQA S2：C=8,16,32 扫描 + 峰值 C8 三次重复。

服务配置：`quay.io/ascend/vllm-ascend:v0.30.0-a3`、A3 1 物理卡 2 逻辑设备、BF16、
TP2、eager、`max_model_len=16384`、`max_num_seqs=16`、
`max_num_batched_tokens=16384`、`limit_mm_per_prompt={"image":16,"video":0}`、
`encoder_cache_tokens=262144`、vLLM 原生 prefix caching 关闭、
Tier 2A 28 GiB/rank + Tier 2B 10 GiB/rank（`CLEF_TIER2_ACTIVE_WINDOW=500`）。
服务日志确认 `[tier2-engine] tier2a=True boundary=56`，权重页缓存释放
13 个文件 / 54,969,731,792 bytes。

## 结果

### OpenDecisions S1

| 轮次 | dec/s | p50 (ms) | p95 (ms) | mean (ms) | 成功 |
| --- | ---: | ---: | ---: | ---: | ---: |
| warmup（冷，排除） | 8.964 | 100.56 | 167.70 | 111.54 | 2352/2352 |
| R1 | 37.882 | 25.25 | 32.47 | 26.37 | 2352/2352 |
| R2 | 37.772 | 25.47 | 32.08 | 26.45 | 2352/2352 |
| R3 | 37.483 | 25.63 | 32.26 | 26.66 | 2352/2352 |
| **正式 3 轮平均** | **37.712** | **25.45** | **32.269** | **26.49** | 7056/7056 |

### OpenDecisions S2

| C | dec/s | p95 (ms) | C | dec/s | p95 (ms) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 37.830 | 32.1 | 16 | 95.434 | 271.4 |
| 2 | 42.961 | 55.2 | 32 | 101.547 | 491.2 |
| 4 | 63.336 | 86.9 | **64** | **101.880** | **917.2** |
| 8 | 82.076 | 148.9 | 128 | 101.266 | 1671.7 |

峰值 C64 三次：102.055 / 101.992 / 102.036 dec/s，平均 **102.028**，0 失败。

### ERQA S1

| 轮次 | dec/s | p50 (ms) | p95 (ms) | mean (ms) | accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| warmup（交接/冷，排除） | 4.754 | 195.71 | 299.04 | 206.79 | 216/400 |
| R1 | 22.285 | 37.51 | 60.65 | 41.34 | 216/400 |
| R2 | 22.636 | 37.33 | 56.31 | 40.63 | 216/400 |
| R3 | 22.731 | 36.79 | 59.14 | 40.49 | 216/400 |
| **正式 3 轮平均** | **22.551** | **37.21** | **58.697** | **40.82** | **216/400** |

### ERQA S2

| C | 扫描 dec/s | 扫描 p95 (ms) | accuracy | 峰值 3 次平均 dec/s | 峰值 3 次平均 p95 (ms) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| **8** | **41.701** | 428.8 | 218/400 | **34.055** | 803.4 |
| 16 | 26.915 | 1862.2 | 217/400 | — | — |
| 32 | 19.741 | 4887.3 | 217/400 | — | — |

## 质量与 2PP

- OpenDecisions 官方 `score.py`（R1/R2/R3 完全一致）：
  Calibration **82.4113**、Intelligence **66.0841**、Capability **74.2477**、
  accuracy **1904/2352 = 0.809524**、Score MAE **0.402612**、0 协议失败。
- ERQA 官方 `erqa_score.py --verify-requests`：S1 四轮均 **216/400 = 0.5400**，
  S2 C8/C16/C32 为 218/217/217，峰值三轮 217/218/217；0 失败、0 重试、0 请求哈希
  mismatch。
- 相对适配后 baseline：OD Cal −0.38 pp、Int +0.03 pp、Cap −0.18 pp、Acc −0.09 pp；
  ERQA Acc −0.75 pp；全部严格在 ±2PP 内。

## 正确性门

`gates/c1-gate.json`：OD 全量 R1/R2/R3 逐位 0.0；OD 64 子集对 OFF 0.0；ERQA 19
子集对 OFF 0.0；ERQA R1/R2/R3 全量 0.0。ERQA 全量与冷填充全长前向对比 399/400
逐位 0.0，唯一 `erqa-0071` 的 4 个概率值最大差 **1.2004e-4**，预测不变。
`gates/resource.json`：`memory.failcnt` 增量 0、峰值 cgroup 114.7 GB / 128 GiB、
HBM 最低余量 382/813 MiB、0 OOM、退出无残留。

## 文件

| 路径 | 说明 |
| --- | --- |
| `od-s1/` | OD warmup/R1/R2/R3 原始报告（gzip）、客户端日志、官方 acceptance JSON |
| `od-s2/` | 8 个并发点 + C64 峰值三次原始报告（gzip）、`scan.csv`、`peak-c64.csv` |
| `erqa-s1/` | warmup/R1/R2/R3 的 summary、predictions、score、日志 |
| `erqa-s2/` | C8/C16/C32 扫描与 C8 峰值三次的 summary、predictions、score |
| `gates/` | OFF OD64、OFF ERQA 19、场景 A、C1 逐位门、资源门 |
| `profile/` | W5/W6 Tier 2A 命中窗口的 manifest 与 compact evidence；`aggregate.json` |
| `SHA256SUMS` | 本目录归档文件 SHA-256 |

> 原始 ERQA 每轮 `samples/` 目录约 124 MB（含请求图片副本），未纳入仓库；
> 预测与 summarize 已足以复核逐题结果。原始 trace 不进入 Git。

校验：

```bash
cd logs/final-eval/multi-tier-20261126 && shasum -a 256 -c SHA256SUMS
```
