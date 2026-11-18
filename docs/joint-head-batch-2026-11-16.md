# Joint head 跨请求批处理原型（2026-11-16）

实现了 `JointSchemaHead.forward_batch` 和 `JointSchemaPooler` 的批路径：一次批量 lexical embedding，head 输入 padding 到 `[B, Lmax, H]`，问题/候选用 mask 对齐，最后按每个请求的原始 option 顺序切片。环境变量 `CLEF_JOINT_BATCH=0` 可回退到逐请求路径。

结论：**实现正确、质量在 2PP 内，但全量 S2 没有收益**；B4 反而变慢，B8 小样本的 +2.5% 没有在 2,352 题上复现。已回退源码。

## 1. 实现

- `EvidenceRoutingLayer.forward` 增加可选 `key_padding_mask`；
- `JointSchemaHead.forward_batch`：
  - hidden 按 `Lmax` padding，memory mask 屏蔽 padding；
  - `segment_mean` 按每个请求真实 span 计算，再 pad 到 `Qmax/Omax`；
  - evidence routing 的 `MultiheadAttention` 使用 `key_padding_mask`；
  - 4 层 `TransformerDecoderLayer` 使用 `tgt_key_padding_mask` / `memory_key_padding_mask`；
  - 候选 routing 的 `option_mask` 对 padding 问题用有限 `-1e4`，避免全 `-inf` 的 NaN；
  - 最终按 `option_question` gather，逐请求返回 `[O_i]`。
- `JointSchemaPooler.forward`：
  - 先收集所有有效请求的 lexical IDs，一次 `embed_lexical`，再按请求 `segment_mean`；
  - `len(entries)>1` 且 `CLEF_JOINT_BATCH!=0` 时走 batched head；否则保持原逐请求路径。

## 2. 数值

- B4 30 次重复请求：batch on/off 最大概率差 **0.0024**（同一 backbone batch，差异来自 head padding/mask）；
- 全量 OD 2,352：Capability **74.2174**（baseline 74.4255，Δ**−0.2081**）、Calibration 82.3497（Δ−0.4444）、Intelligence 66.0851（Δ+0.0283）、accuracy 1909（+3）、Score MAE 0.401428；**三项均在 2PP 内**，2 条 choice 变化，max 概率差 0.1361。

## 3. 性能

设备：TP2/BF16/eager/BIC；`max_num_seqs=4` 或 8。

| 场景 | batch off | batch on | 变化 |
| --- | ---: | ---: | ---: |
| B4 C4，30×307 | 13.637 dec/s，mean 273.43，p95 287.56 | 13.442 dec/s，mean 279.73，p95 296.27 | **−1.4% 吞吐，p95 +3.0%** |
| B8 C8，30×307 | 18.158 dec/s，mean 408.57，p95 439.28 | **18.614 dec/s**，mean 398.22，p95 438.06 | 小样本 +2.5% |
| B8 C16，全量 2,352 | 16.4638 dec/s，p95 1804.06 | 16.4685 dec/s，p95 1799.77 | **+0.03%，无实际收益** |

解释：

- head 在完整请求中只占约 8–10 ms CPU wall，跨请求批处理能省一部分 launch/lexical embedding，但 padding、mask、`MultiheadAttention`/decoder 的额外开销抵掉了收益；
- B4 的 padding/mask 比例更高，直接变慢；
- B8 小样本正好都是相同的 307-token/单字段 schema，padding 小，所以出现 +2.5%；全量 OD 的字段/长度分布更杂，收益消失。

## 4. 决定

- **不采用**：实现复杂度高、维护面大，S2 全量无收益，B4 退化；
- 源码已从备份恢复为原始 `clef_joint_schema.py`（SHA-256 `197863d5…`）；
- baseline 服务恢复为 **PID 692708**，argv/cwd/env 与 phase-0 一致，`/health` 200，冻结 8 例 **8/8**；
- 若未来要重新考虑，前提是 head 占比显著上升（例如主干更快后），并且只对相同 schema/短请求启用；当前不作为提交候选。

## 5. 证据

- B4：`Clef-27B/logs/gdn-data-path-20261115/joint-batch-on/`、`joint-batch-off/`
- B8 小样本：`.../joint-batch-b8-on/`、`joint-batch-b8-off/`
- 全量质量：`.../joint-batch-b8-on/od-c16-quality.json`
- 恢复：`.../restore-joint-batch/`
