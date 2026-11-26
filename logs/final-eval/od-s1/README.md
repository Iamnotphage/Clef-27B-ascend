# OD S1 warmup + 3 formal runs（2026-11-26）

## 目的与协议

记录 OpenDecisions 全量 2,352 题、单并发（C1）的 S1 性能验收：

- 服务：`bash start.sh --model-dir ... --output-dir /tmp/manual-accept-svc`，PID 734299，
  启动时间 `2026-11-26T00:41:59Z`；
- 环境：`quay.io/ascend/vllm-ascend:v0.30.0-a3`、A3 1 物理卡 2 逻辑设备、BF16、TP2、
  `max_num_seqs=16`、`max_num_batched_tokens=16384`、eager；
- 调度：`TASK_QUEUE_ENABLE=2`、`OMP_NUM_THREADS=1`、`ACLNN_CACHE_LIMIT=1000000`；
- 缓存：vLLM 原生 prefix caching 关闭；Tier 2 prefix-state cache 开启
  （32 GiB/rank、8 slots、`CLEF_TIER2_ACTIVE_WINDOW=500`）；
- 工作负载：`logs/final-eval/od-s1/all2352.json`（2,352 条，SHA-256
  `74ce0f4fde81c1fa1f125dc6b51db2f46fb6d358a37d9faf9c54b362884a2141`；随本目录归档）；
- 统计：**warmup 1 次（不计入）+ 正式 3 次**；S1 指标为正式三次的 p95
  算术平均，同时保留每轮与汇总。

## 结果

| 轮次 | wall | dec/s | p50 | p95 | p99 | mean | 成功 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| warmup（冷，排除） | 266.1 s | 8.839 | 99.30 ms | 167.31 ms | 207.24 ms | 112.97 ms | 2352/2352 |
| R1 | 240.6 s | 9.774 | 99.24 ms | 118.57 ms | 121.41 ms | 102.16 ms | 2352/2352 |
| R2 | 241.5 s | 9.741 | 99.24 ms | 118.49 ms | 121.30 ms | 102.52 ms | 2352/2352 |
| R3 | 242.2 s | 9.712 | 99.88 ms | 118.56 ms | 121.32 ms | 102.83 ms | 2352/2352 |
| **正式三次平均** | **241.42 s** | **9.742** | **99.45 ms** | **118.54 ms** | **121.35 ms** | **102.51 ms** | **7056/7056** |

- 正式三次总吞吐（合并 7,056 决策 / 总 wall）：9.742 decisions/s。
- 正式三次输入吞吐：4,352.98 tokens/s。
- 失败/重试：0。

## 文件

| 文件 | 说明 |
| --- | --- |
| `od-s1-500-warmup.json.gz` | warmup 原始逐请求报告（gzip） |
| `od-s1-500-r1.json.gz` / `r2` / `r3` | 正式三轮原始逐请求报告（gzip） |
| `od-s1-500-warmup.log` / `r1` / `r2` / `r3.log` | 客户端逐轮日志 |
| `od-s1-500-loop.log` | 运行时间线 |
| `od-s1-500-summary.json.gz` / `.csv` | 正式三轮汇总（canonical，JSON 为 gzip） |
| `od-s1-500-runs.csv` | warmup + R1/R2/R3 + 正式平均的逐轮明细 |
| `od-s1-500-summary-2run.*` | 早期两轮汇总，保留备查 |
| `od-s1-500-summary-r3.*` | 与 3 轮 canonical 汇总内容相同的别名副本 |
| `all2352.json` | 本次 2,352 条请求序列（S1/S2 共用） |
| `SOURCE_SHA256SUMS` | 服务器原始文件 SHA-256 |
| `SHA256SUMS` | 本目录归档文件 SHA-256 |

查看 gzip 报告：

```bash
gzip -dc od-s1-500-r1.json.gz | python3 -m json.tool | head
```

校验：

```bash
shasum -a 256 -c SHA256SUMS
```
