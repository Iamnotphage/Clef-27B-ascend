# A15-large-v1 diagnostic_gate 更正（2026-11-19）

- 事实（原始 result.json，逐键复核）：rank0 与 rank1 的 `diagnostic_gate.within_bounds` 均为 **false**。
  - `max_abs = 2.0` > `max_abs_le = 0.03125`
  - `mean_abs = 5.1305116357980296e-05` <= `mean_abs_le = 1e-4`
  - 既有 AND 组合结果为 **false**；组合表达式以 32/35 代码核实为准。
- 更正范围：此前 owner31 在回传消息/摘要中把本格写成 `within_bounds=true`，系误将 A15-small-v3（rows127，zero diff，gate=true）的值带入本格；v3 小格的 true 仍成立，本格为 false。
- 原始 `rank0-result.json`、`rank1-result.json`、`trace.jsonl`、`window.json` 与既有 `SHA256SUMS`（`2adc7ef6…`，17/17）均未改动；本文件为更正记录，未纳入原 SUMS。
- 统一记录口径：max_abs 2.0 超 0.03125、mean 5.13e-5 在 1e-4 内、既有 AND 诊断结果 false；该 gate 仅 diagnostic，不宣称精度通过；无需设备重跑。

## 32 联审补充口径（同日）

- `max_abs=2.0` 仅在 `ref=400` 样本（index 1040755、pos [203,1395]）处为 1 个 BF16 ulp；不能据此归因累加机制，也不代表其余 509 个非零差元素的分布。
- `max_rel_abs` **仅记录数值 1.0，其位置未落盘**，不得与 `max_error` 的位置混用。
- `starttime 400405818` 仅来自执行前对 `/proc` 的实扫记录；`restore-a15-large-v1` 归档中没有该字段。
- 该 gate 表达式为 AND（max_abs≤0.03125 且 mean_abs≤1e-4），本格 false 属真实结果，非实现错误；不称精度通过、不重跑、不改门。
