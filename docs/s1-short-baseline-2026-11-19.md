# S1 短基线（2026-11-19 01:50 CST）

## 用途与限定

**仅取前 64 条固定短样本，不可与全量 S1 比较，不作性能结论。**
用途：本轮真实性与后续同窗口参考锚点。

## 命令（完整参数）

```
python3 /workspace/yangchen/Clef-27B-ascend/runtime/od_full_benchmark.py \
  --sample-file /workspace/yangchen/logs/tuning/20261114-013000-fix4/opendecisions/all2352.json \
  --url http://127.0.0.1:18143/v1/systemone \
  --limit 64 --concurrency 1 --timeout 120 \
  --report <out>/s1-short.json
```

外层 `timeout 900`；`RC=0`。

## 身份

| 项 | 值 |
| --- | --- |
| 运行时刻（UTC） | `2026-11-18T17:50:15Z`（CST 2026-11-19 01:50） |
| 样本文件 full SHA-256 | `74ce0f4fde81c1fa1f125dc6…`（`all2352.json`，完整值见远端原件） |
| 服务 api PID | **1176049** |
| PGID / starttime | 1176049 / 394031057 |
| boot_id | `393ecc15-721f-46f7-b119-21fa327a2b3b` |
| 服务配置 | `--config s1`（TP2 / BF16 / eager / max-model-len 4096 / max-num-seqs 1 / max-num-batched-tokens 4096 / 无 prefix caching / 无 chunked prefill） |

## 结果

| 指标 | 值 |
| --- | ---: |
| 请求 | **64 / 64 成功，0 失败** |
| 吞吐 | **8.937 dec/s** |
| mean | **111.84 ms** |
| p50 | **108.49 ms** |
| p95 | **115.08 ms** |
| p99 | **189.70 ms** |
| min / max | 101.96 / 307.94 ms |
| `client_inflight_max` | **1**（确认未并发） |

逐请求起止与原始响应保存在远端原件的 `requests` 字段。

## 远端原件

| 项 | 值 |
| --- | --- |
| 目录 | `/workspace/yangchen/logs/s1-short-20261118T175015Z/` |
| 文件 | `s1-short.json`（含 `requests`/`summary`/`configuration`）、`run.log` |

## 口径

- 未改服务 / 软件 / 库；未并发其它设备 probe；
- 使用既有已核入口，参数与原运行一致；
- 正式完整 S1（≥3 轮）留到最终阶段。
