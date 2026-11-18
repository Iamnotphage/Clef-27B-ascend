# 上游 Dense SP 移植与 A3 实测（2026-11-16）

## 1. 来源

上游 vLLM-Ascend 提交：

```text
b65e7c718b93e47fc8a2a3be759b9cc64b1079bd
[BugFix][Model] Support dense sequence parallelism for the Qwen3.5 family (Qwen3.8-27B) (#20756)
```

它明确对接 vLLM `24420b7a`（我们的固定 vLLM），新增：

- `enable_dense_sp()`：TP > 1、PP == 1、`qwen3_5_text`、FlashComm switch；
- `Qwen3_5DecoderLayer` 的 sequence-parallel 布局：
  - hidden/residual 保持本地 token 分片；
  - attention（full + GDN）前 AllGather，输出投影 `reduce_results=False` 后 ReduceScatter；
  - dense MLP 在 TP 分片权重前 AllGather，down 后 ReduceScatter；
- `utils.enable_sp` 覆盖 dense family；
- 通过 `--additional-config '{"enable_flashcomm1": true}'` 启用。

## 2. 移植

固定栈 vLLM-Ascend `89eb8a5e` 没有该 dense 分支。本轮把上述 decoder/MLP SP 逻辑移植到当前树：

- `patch_qwen3_5.py`：新增 `enable_dense_sp`、`_sequence_parallel_dense_mlp_forward`、SP decoder forward、`Qwen3_5DecoderLayer.__init__/forward` 安装；
- `utils.py`：`enable_sp` 覆盖 dense；
- `ascend_config.py`：FlashComm 日志分支；
- 保留现有 GDN 决策/BIC 补丁与 v3 安装器语义。

服务启动日志确认：

```text
FlashComm1 dense sequence parallelism is enabled for the Qwen3.5 dense family.
```

## 3. 质量

- 冻结 8 例：**6/8 逐位**，两例 choice 最大概率差 **2.1e-5**，choice 不变；
- 全量 OD 2,352：

| 指标 | baseline | Dense SP | Δ |
| --- | ---: | ---: | ---: |
| Capability | 74.4255 | **74.4235** | −0.0020 |
| Calibration | 82.7941 | 82.7953 | +0.0012 |
| Intelligence | 66.0568 | 66.0517 | −0.0052 |
| accuracy | 1906/2352 | 1906/2352 | 0 |
| Score MAE | 0.401389 | 0.401501 | +0.000112 |
| max 概率差 | — | 0.0343 | — |
| choice 变化 | — | 0 | — |

质量完全在 2PP 内。

## 4. 性能

同一服务配置（TP2/BF16/eager/4096/seqs1/BIC）：

| 点 | baseline | Dense SP | 变化 |
| --- | ---: | ---: | ---: |
| 307 mean / p95 | 125.06 / 140.77 | 133.90 / 146.69 | **+7.1% / +4.2%** |
| 969 mean / p95 | 172.51 / 175.50 | 168.24 / 173.05 | −2.5% / −1.4% |
| **全量 S1 p95** | **198.42** | **212.96** | **+14.54 ms（+7.3%）** |
| 全量 mean | 160.59 | 174.86 | +14.27 ms |
| 全量 p50 | 179.86 | 191.59 | +11.73 ms |
| 吞吐 | 6.226 dec/s | 5.718 dec/s | −8.2% |

## 5. 结论

- 上游 dense SP 在 A3 上**质量可用但性能退化**，与上游文档“A3 may be negligible / decode-heavy may regress”一致；
- 307/中等请求变慢，969 略快，但全量 P95 明显变差；
- 主要成本：AllGather/ReduceScatter 的 collective 次数与 padding、elementwise 缩小不足以抵消；
- **不采用**，恢复 `patch_qwen3_5.py`/`utils.py`/`ascend_config.py` 原始 SHA；
- 这条“try best”的高风险路线已实测闭环：完整 Dense SP 不是当前 A3 的收益方向。

## 6. 状态与证据

- 源码恢复：
  - `patch_qwen3_5.py` `1b82b52e…`
  - `utils.py` `52585cf4…`
  - `ascend_config.py` `65bc9b18…`
- baseline 服务 **PID 711199**，argv/cwd/env 与 phase-0 一致，`/health` 200，冻结 8 例 **8/8**；
- 证据：`Clef-27B/logs/gdn-data-path-20261115/dense-sp-test/`、`restore-dense-sp/`。
