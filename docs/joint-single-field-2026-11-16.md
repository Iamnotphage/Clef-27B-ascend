# Joint head 单字段特化（2026-11-16）

## 1. 实现

OD 全部 2,352 条都是 **单字段**，且 `JointSchemaHead` 的 decoder 有 4 层 `TransformerDecoderLayer`。当 `CLEF_JOINT_SINGLE_FIELD=1` 且 `num_questions == 1`：

- self-attention 的 Q=1，softmax 恒为 1；只保留 **V 投影 + out_proj**，跳过 Q/K 投影和 softmax；
- cross-attention、FFN、LayerNorm、残差保持不变；
- option routing 的 Q=1 只对候选做一次加权平均，避免 `qow,qw->qo` 的通用 einsum；
- 多字段请求自动回退原路径。

默认 `s1`/`s2` 启用，`custom` 关闭；postimage 和安装器在 `runtime/`。

## 2. 数值

- 冻结 8 例：1/8 逐位，其余最大概率差 **0.0019**，choice 全部不变；
- 全量 OD：Capability **74.310693**、Calibration 82.709163、Intelligence 65.912223、accuracy **1902/2352**、Score MAE 0.401545；2PP 通过，4 条 choice 变化，max 0.0851；
- ERQA 400：**217/400 = 0.5425**，与 v2 相同。

## 3. 性能

同窗口 head-on vs head-off（都是 v2 kernel、cap 1600、seqs1）：

| 指标 | head-off | **head-on** | 变化 |
| --- | ---: | ---: | ---: |
| S1 p95 | 203.102 ms | **197.184 ms** | **−2.91%（−5.918 ms）** |
| S1 mean | 161.917 ms | **157.038 ms** | −3.01% |
| 吞吐 | 6.175 dec/s | **6.367 dec/s** | +3.1% |

最终正常入口全量（v2 + head）：

| 指标 | baseline | 当前 |
| --- | ---: | ---: |
| S1 p95 | 217.036 ms | **190.405 ms（−12.27%）** |
| S1 mean | 180.069 ms | **153.761 ms** |
| S1 p50 | 205.034 ms | **168.485 ms** |
| 吞吐 | 5.549 dec/s | **6.503 dec/s** |
| Capability | 74.4255 | **74.3107** |
| ERQA | 219/400 | **217/400** |

跨运行有 ±3–5% 机器漂移；采用同窗口 head-on/off 的 −2.91%。

## 4. 打包

- postimage：`runtime/joint_single_field/clef_joint_schema.py`，SHA-256 `f8101b98…`；
- installer：`runtime/apply_joint_single_field_patch.py`，preimage `197863d5…`；
- `inference.py` 在 `s1`/`s2` 默认启用，`custom` 关闭；dry-run 不改源码；
- 服务环境设置 `CLEF_JOINT_SINGLE_FIELD=1`。

## 5. 状态

- `results.csv` S1 **190.40488475002348 ms**，`s1_gdn_data_path=BIC+QKV-v2+SF`；
- `metrics.csv` 为 v2+head 质量；
- baseline 服务恢复 **PID 755361**，`/health` 200，冻结 8/8；
- 证据：`head-entry/`、`head-control/`、`single-field-test3/`、`single-field-erqa/`、`restore-head-final/`。
