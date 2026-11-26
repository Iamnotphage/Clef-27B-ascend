# ERQA S1 warmup + 3 formal runs（2026-11-26）

## 目的与协议

ERQA 官方 400 题、单并发（C1）的 S1 验收，服务为同一 OD→ERQA 进程
（PID 734299，启动时间 `2026-11-26T00:41:59Z`），未重启：

- 环境：`quay.io/ascend/vllm-ascend:v0.30.0-a3`、A3 1 物理卡 2 逻辑设备、
  BF16、TP2、eager；
- 配置：`max_model_len=16384`、`max_num_seqs=16`、`max_num_batched_tokens=16384`、
  `limit_mm_per_prompt={"image":16,"video":0}`、`encoder_cache_tokens=262144`；
- 调度/缓存：`TASK_QUEUE_ENABLE=2`、Tier 2 prefix-state cache 开启
  （32 GiB/rank、8 slots、`CLEF_TIER2_ACTIVE_WINDOW=500`）、vLLM 原生 prefix
  caching 关闭；
- 接口：`state-list` 报文、`POST /systemone`；
- 数据：`/tmp/final-test-erqa-data`（官方 400 题，manifest/samples/images）；
- 统计：warmup 1 轮（完整 400，排除）+ 正式 3 轮；S1 为正式三轮平均。

## 结果

| 轮次 | wall | dec/s | p50 | p95 | p99 | mean | accuracy | 成功/重试 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| warmup（交接/冷，排除） | 88.9 s | 4.499 | 206.3 ms | 314.2 ms | 533.3 ms | 217.6 ms | 216/400 | 400/400 / 0 |
| R1 | 47.3 s | 8.459 | 113.4 ms | 122.6 ms | 132.7 ms | 114.4 ms | 218/400 | 400/400 / 0 |
| R2 | 46.8 s | 8.538 | 112.4 ms | 121.1 ms | 133.7 ms | 113.3 ms | 218/400 | 400/400 / 0 |
| R3 | 47.4 s | 8.446 | 112.8 ms | 123.4 ms | 142.7 ms | 114.4 ms | 218/400 | 400/400 / 0 |
| **正式三次平均** | **47.17 s** | **8.481** | **112.86 ms** | **122.37 ms** | **136.36 ms** | **114.05 ms** | **218/400** | **1200/1200 / 0** |

- Warmup 的高 p95 来自 OD→ERQA 的 Tier 2 交接与 ERQA 首轮 populate，已排除；
  正式三轮 p95 121.1–123.4 ms。
- R1/R2/R3 的逐题 prediction 完全一致；accuracy 218/400 = 0.545，和
  `metrics.csv` 一致，2PP 内。
- 0 失败、0 重试，400/400 覆盖。

## 文件

| 文件 | 说明 |
| --- | --- |
| `erqa-s1-{warmup,r1,r2,r3}-summary.json` | 每轮汇总（latency/accuracy/coverage） |
| `erqa-s1-{warmup,r1,r2,r3}-predictions.jsonl` | 每轮逐题预测 |
| `erqa-s1-{warmup,r1,r2,r3}.log` | 客户端逐轮日志 |
| `erqa-s1-loop.log` | 运行时间线 |
| `SOURCE_SHA256SUMS` / `SHA256SUMS` | 原始/归档文件校验 |

> 每轮的 `samples/` 目录约 124 MB（含请求图片副本），未纳入仓库；原始目录仍保留在
> 服务器 `/tmp/erqa-s1-*`，预测/汇总已足够复核逐题结果。

```bash
shasum -a 256 -c SHA256SUMS
```
