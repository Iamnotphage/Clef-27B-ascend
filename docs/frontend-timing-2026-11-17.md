# 前端/引擎输出 8 ms 缺口定位（2026-11-17）

## 方法

在 `vllm/entrypoints/openai/decisions/strategies.py` 的 `JointSchemaStrategy.read` 内临时打点：

- `encode()`（含 tokenizer 与 schema 编码）；
- `async for result in engine_client.encode(...)`（引擎往返）；
- `result.outputs.data.tolist()`（D2H/结果取回）；
- `QuestionRead` 列表构建。

请求为 967 token（`od-1534`），2 预热 + 10 正式，C1；服务为最终 s1 配置 + `ACLNN_CACHE_LIMIT=1000000`。测后恢复 `strategies.py`，SHA-256 回到 `0ddd92b44b38ae7a64d6057cacb205b5164b47dfeb330ea3fcab2bf2f2198a4c`。

## 结果（每请求，ms）

| 段 | 中位 | 范围（正式 9 条） |
| --- | ---: | ---: |
| `encode` | **2.35** | 2.30–2.49 |
| 引擎 `async for` | **166.5** | 164.8–171.5（首条冷启动 594） |
| `tolist()` | **0.009** | 0.008–0.012 |
| `QuestionRead` 构建 | **0.008** | 0.007–0.013 |

## 结论

- `docs/profile-final-2026-11-17.md` 中 `frontend_read_including_engine − worker_execute_model ≈ 8.1 ms` 的缺口，**只有 2.35 ms 在策略侧 `encode`**；`tolist()` 与 `QuestionRead` 构建合计 <0.02 ms。
- 其余约 **5.8 ms 在 engine client 的调度/输出处理路径内**（vLLM 内部），不在 `strategies.py`，没有单点可改。
- `encode` 的 2.35 ms 主要是 tokenizer 1.4 ms + schema 循环；此前的 schema token cache v2 在 969 持平、307 更慢，不作为候选。
- **不采用任何改动**；本项作为负结果归档，提交配置不变。

## 证据

- 打点日志：`Clef-27B/logs/tuning5-20261117/frontend-timing.log`
- 原始服务/运行目录：`/workspace/yangchen/logs/tuning5-20261117/shape-ab/frontend-diag/`
- 恢复后 SHA-256：`0ddd92b4…`；三份安装器 `--check` 仍为 ready。
