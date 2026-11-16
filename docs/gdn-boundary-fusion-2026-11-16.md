# GDN 边界融合：ClefGdnQkvNorm（2026-11-16）

## 1. 结论

`ClefGdnQkvNorm` 把 GDN prefill 的卷积输出 `[40,T,128]` 一次写成连续的 BSND q/k/v：Q/K 在 head_dim=128 上做 L2Norm，V 直接拷贝。它替代了 BIC 路径上的 split/l2norm/transpose/contiguous 多次操作。

- 单算子：969 token **1.27×**、1024 **1.29×**、1200 **1.11×**、1595 **0.86×**；
- 正常入口 S1（全量 2,352）**p95 197.083 ms vs 同窗口 control 203.091 ms（−2.96%）**，mean 157.699 vs 166.552，吞吐 6.340 vs 6.003 dec/s；
- 质量 2PP 通过：OD Capability **74.313479**（Δ−0.1121）、ERQA **217/400 = 0.5425**（−0.50 pp）；
- **已接入 `--config s1` 默认**；token 总数 >1200 自动回退 BIC。

## 2. Kernel 与 cap

| tokens | fused ms | baseline ms | 加速 | Q/K max_abs | V |
| ---: | ---: | ---: | ---: | ---: | --- |
| 307 | 0.1217 | 0.2444 | 2.01× | 0.001953（1 ulp） | 逐位 |
| 401 | 0.1232 | 0.2536 | 2.06× | 1 ulp | 逐位 |
| 512 | 0.1255 | 0.2538 | 2.02× | 1 ulp | 逐位 |
| 969 | 0.1954 | 0.2485 | **1.27×** | 1 ulp | 逐位 |
| 1024 | 0.1999 | 0.2586 | **1.29×** | 1 ulp | 逐位 |
| 1200 | 0.2331 | 0.2580 | **1.11×** | 1 ulp | 逐位 |
| 1595 | 0.3054 | 0.2613 | 0.86× | 1 ulp | 逐位 |
| 10294 | 1.9556 | 0.2608 | 0.13× | 1 ulp | 逐位 |

因此 cap 取 **1200**，覆盖 S1 P95 的 969；长文本/ERQA 10,294 直接走 BIC。Q/K 的 1 ulp 来自归约顺序（64+64 两半 vs Triton 单次 128），V 逐位保持。

## 3. 服务接入

### 3.1 数据路径

`vllm_ascend/ops/gdn.py` 在 `runtime.preprocess_fusion` 且总 token ≤ `CLEF_GDN_QKV_NORM_MAX_TOKENS` 时：

- 调用 `aclnnClefGdnQkvNorm` 得到已归一化的 BSND q/k/v；
- 传给 `DeviceOperator.fla_gdn_prefill(..., qkv_pre_normalized=True)`；
- 该函数跳过重复 l2norm/transpose，只做 g/beta cast 和 fused GDN prefill；
- 任何 vendor/op 缺失、shape 不匹配或异常都回退 BIC。

### 3.2 打包

- vendor：`Clef-27B/src/ascendc/vendors/customize`（456 KB，含 `libcust_opapi.so`）；
- v4 companion installer：`runtime/apply_qkv_norm_patch.py`，preimage = GDN v3 postimage，postimage 在 `runtime/qkv_norm/`；
- v3 GDN installer 接受 `qkv` 状态；
- `Clef-27B/inference.py` 在 `s1` 中设置 `preprocess_fusion=True`，自动设置 `CLEF_QKV_NORM_VENDOR`、`LD_LIBRARY_PATH`、`ASCEND_CUSTOM_OPP_PATH`；
- 默认 cap 1200，可用 `CLEF_GDN_QKV_NORM_MAX_TOKENS` 覆盖。

## 4. 正常入口结果

服务：`start.sh --config s1`，TP2/BF16/eager/4096/seqs1/BIC+QKV；全量 2,352：

| 指标 | 同窗口 control（BIC） | **BIC+QKV** | 变化 |
| --- | ---: | ---: | ---: |
| S1 P95 | 203.091 ms | **197.083 ms** | **−6.008 ms（−2.96%）** |
| S1 mean | 166.552 ms | **157.699 ms** | −8.852 ms |
| S1 p50 | 184.868 ms | **174.169 ms** | −10.699 ms |
| S1 p99 | 213.656 ms | 205.583 ms | −8.073 ms |
| 吞吐 | 6.003 dec/s | **6.340 dec/s** | **+5.6%** |

另一直接服务窗口融合为 190.98 ms；跨运行有 ±3–5% 机器漂移，**采用同窗口 control 的 −2.96%**。

质量（正常入口全量 OD）：

| 指标 | baseline | BIC+QKV | Δ |
| --- | ---: | ---: | ---: |
| Capability | 74.425479 | **74.313479** | −0.1120 |
| Calibration | 82.794115 | 82.713420 | −0.0807 |
| Intelligence | 66.056843 | 65.913538 | −0.1433 |
| accuracy | 1906/2352 | 1902/2352 | −4 题 |
| Score MAE | 0.401389 | 0.401512 | +0.000122 |
| choice 变化 | — | 4 | — |
| max 概率差 | — | 0.0851 | — |

三项与 accuracy 均在 2PP 内。

ERQA 400（同一融合 kernel；≤1200 走融合，长序列回退）：

| 指标 | baseline | BIC+QKV |
| --- | ---: | ---: |
| accuracy | 219/400 = 0.5475 | **217/400 = 0.5425** |
| 单图 | 170/287 = 0.5923 | 168/287 = 0.5854 |
| 多图 | 49/113 = 0.4336 | 49/113 = 0.4336 |
| P95 | 323.476 ms | 310.949 ms |

总体 −0.50 pp，2PP 通过。

## 5. 状态

- 正常入口服务已用 `start.sh --config s1` 实测；提交默认 `s1` 已开启 preprocess fusion；
- `results.csv` S1 更新为 **197.083265 ms**，`s1_gdn_data_path=BIC+QKV`；`metrics.csv` 更新为当前质量；
- S2 仍使用未融合 seqs4 的 14.652/14.689，尚未与融合组合复测；
- baseline 服务恢复 **PID 726791**，`/health` 200，冻结 8 例 **8/8**；
- 证据：`Clef-27B/logs/gdn-data-path-20261115/qkv-norm-test/`（直接服务）、`qkv-norm-erqa/`（ERQA）、`qkv-norm-entry5/`（正常入口）、`qkv-norm-control/`（同窗口 control）、`restore-qkv-norm-final/`；vendor 与 v4 installer 在 `Clef-27B/src/`、`runtime/`。
