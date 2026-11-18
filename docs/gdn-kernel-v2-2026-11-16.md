> **2026-11-16 后续：** 在 v2 之上接入单字段 joint head 后，正常入口最终 S1 p95 **190.405 ms**、mean 153.761、6.503 dec/s；head 同窗口 head-off 为 203.102，head-on 197.184（−2.91%）。详见[单字段 head](joint-single-field-2026-11-16.md)。

# GDN QKV Norm kernel v2（2026-11-16）

## 1. 改了什么

v1 每行做两次 `ReduceSum(64)`，再用 `GetValue` + 逐行标量 `Muls`。v2：

- 每行一次 `ReduceSum(..., 128)`，去掉半段相加；
- 新增 `scaleBuf`，用 `Broadcast<float,2,1>` 把每行 `rsqrt(sum+eps)` 扩成 `[rows,128]`；
- 一次 `Mul(xf, xf, scales, rows*128)` 替代逐行标量 `Muls`。

Q/K 仍是 1 bf16 ulp 差异（归约顺序），V 逐位不变。

## 2. 单算子

| tokens | v2 fused ms | baseline ms | v2 加速 |
| ---: | ---: | ---: | ---: |
| 307 | 0.1233 | 0.2469 | **2.00×** |
| 969 | 0.1231 | 0.2477 | **2.01×** |
| 1024 | 0.1250 | 0.2510 | **2.01×** |
| 1595 | 0.1539 | 0.2534 | **1.65×** |

连续 event p50/p95：

| tokens | v2 p50 / p95 | v1 p50 / p95 | v2 p95 加速 |
| ---: | ---: | ---: | ---: |
| 307 | 0.1587 / 0.1871 ms | 0.1968 / 0.2333 ms | 1.25× |
| 969 | 0.1895 / **0.2098 ms** | 0.1925 / 0.2228 ms | **1.06×** |
| 1024 | 0.1900 / 0.2132 ms | 0.1937 / 0.2252 ms | 1.06× |
| 1595 | 0.1896 / 0.2062 ms | —（v1 已退化） | — |

v2 把 1595 从 v1 的 0.86× 提升到 **1.65×**，因此 cap 从 1200 扩到 **1600**，覆盖 OD 最大 1,595 token 和 ERQA 1200–1595 的 102 条请求。

## 3. 服务级

正常入口 `start.sh --config s1`，TP2/BF16/eager/4096/seqs1，全量 2,352：

| 指标 | v1 最佳窗口 | **v2 最佳窗口** | v1 同窗口 control | v2 同窗口 control |
| --- | ---: | ---: | ---: | ---: |
| p95 | 197.083 ms | **196.193 ms** | 209.616 ms | **204.430 ms** |
| mean | 157.699 ms | 160.547 ms | 163.116 ms | 162.731 ms |
| p50 | 174.169 ms | 177.810 ms | 177.755 ms | 176.446 ms |
| 吞吐 | 6.340 dec/s | 6.228 dec/s | 6.130 dec/s | 6.144 dec/s |

- v2 同窗口相对 v1 control **p95 −2.48%（−5.186 ms）**，mean −0.24%，吞吐 +0.23%；
- 最佳窗口 v2 196.193 vs v1 197.083（−0.45%）；跨运行有 ±3–5% 机器漂移，采用 v2 同窗口 control 的 **−2.48%**；
- ERQA 400：accuracy **217/400 = 0.5425**（同 v1），p95 311.349 ms。

质量（v2 正常入口全量 OD）：

| 指标 | baseline | v2 | Δ |
| --- | ---: | ---: | ---: |
| Capability | 74.425479 | **74.311197** | −0.1143 |
| Calibration | 82.794115 | 82.710226 | −0.0839 |
| Intelligence | 66.056843 | 65.912169 | −0.1446 |
| accuracy | 1906/2352 | 1902/2352 | −4 题 |
| Score MAE | 0.401389 | 0.401552 | +0.000163 |
| choice 变化 | — | 4 | — |

三项与 accuracy 均在 2PP 内。

## 4. 状态

- vendor：`Clef-27B/src/ascendc/vendors/customize` 为 v2 构建；
- cap 默认 **1600**（`inference.py` 的 `CLEF_GDN_QKV_NORM_MAX_TOKENS`）；
- `results.csv` S1 **196.1930829 ms**、`s1_gdn_data_path=BIC+QKV-v2`；`metrics.csv` 为 v2 质量；
- baseline 服务恢复 **PID 743800**，`/health` 200，冻结 8/8；
- 证据：`qkv-v2-entry/`、`qkv-v1-control/`、`qkv-v2-control/`、`qkv-v2-erqa/`、`qkv-v2-probe.json`、`qkv-v2-latency.json`、`restore-qkv-v2/`。
