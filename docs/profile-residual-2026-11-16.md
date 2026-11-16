# 18.8 ms 非 kernel 残余的 profile 定位（2026-11-16）

此前阶段表的 RUNNING 122.42 ms 中，设备 kernel 101.64、联合头设备计算 1.98，剩余 18.80 ms 未分解。本轮在 BIC S1 服务上同时采集 CPU observation scopes 与 torch/NPU trace，分别用 307-token 和 967-token 请求定位。

## 1. 采集条件

| 项 | 值 |
| --- | --- |
| 服务 | `start.sh --config s1 --port 18143`，TP2/BF16/eager/4096/seqs1 |
| 观测 | `CLEF_PROFILE_MARKERS=1` + `CLEF_PROFILE_OBSERVE_DIR`（14 个 vLLM/vLLM-Ascend scope） |
| trace | `--profiler-config` torch，`ignore_frontend=true`，Level1 |
| 请求 | 307-token（`od-0105`）与 967-token（`od-1534`），10 预热 + 10 正式，C1 |
| 分析 | `analyze_systemone_profile.py`，trace 导出后按 rank 分析；`--observations-root` 关联 CPU scope |

**口径**：profiler 与 observation 都会增加开销（307 profiled 164.8 ms vs 未采集 126.25 ms；967 profiled 170.0 ms），因此绝对值只用于归因，不作性能声明；CPU scope 为 inclusive wall，嵌套 scope 不能相加。

## 2. CPU scope（每请求均值）

| scope | 307 rank0 | 307 rank1 | 967 rank0 | 967 rank1 |
| --- | ---: | ---: | ---: | ---: |
| `clef::worker_execute_model` | 160.11 | 151.56 | 145.08 | 150.38 |
| `clef::backbone` | 139.86 | 134.40 | 128.52 | 133.59 |
| `clef::backbone_decoder_layer`（单层） | **2.014** | 1.951 | **1.853** | 1.939 |
| `clef::joint_pooler`（含 head） | 10.86 | 10.09 | 9.42 | 9.82 |
| `clef::joint_head`（嵌套） | 8.40 | 7.92 | 7.22 | 7.74 |
| `clef::schema_tokenize_encode` | 1.51 | — | 2.47 | — |
| `clef::tokenizer` | 0.14 | — | 1.42 | — |
| `clef::candidate_embedding_tp` | 0.78 | 0.64 | 0.59 | 0.58 |
| `clef::schema_indices_h2d` | 0.40 | 0.37 | 0.42 | 0.37 |
| `clef::joint_evidence_routing`（单次） | 0.85 | 0.78 | 0.72 | 0.76 |
| `clef::scheduler`（单次） | 0.166 | — | 0.132 | — |
| `clef::api_handler_including_json` | 164.76 | — | 170.01 | — |

- `backbone_decoder_layer`：64 层 × 2.0 ms ≈ **129 ms**（307）与 64 × 1.85 ≈ **119 ms**（967），是 CPU 侧最大项；这是 vLLM eager 逐层 Python/dispatch，不是 vLLM-Ascend kernel。
- `joint_pooler` 10.9/9.4 ms，其中 `joint_head` 8.4/7.2 ms；head 设备计算只有约 2 ms，其余是 CPU dispatch、D2H/tolist、schema 循环。
- 前端 `schema_tokenize_encode`+`tokenizer` 1.5–3.9 ms；scheduler 仅 0.13–0.17 ms。

## 3. 设备侧（每请求，10 请求均值）

| 指标 | 307 rank0 | 307 rank1 | 967 rank0 | 967 rank1 |
| --- | ---: | ---: | ---: | ---: |
| kernel window span | 164.76 | 165.18 | 170.37 | 170.39 |
| kernel union | 93.09 | 140.84 | 157.86 | 157.85 |
| compute union | 77.35 | 77.11 | 144.45 | 144.00 |
| communication union | 28.04 | 74.79 | 32.09 | 33.20 |
| compute/comm overlap | 12.30 | 11.06 | 18.68 | 19.35 |
| comm nonoverlapped | 15.74 | 63.73 | 13.41 | 13.85 |
| no device kernel within window | 71.67 | 24.34 | 12.51 | 12.54 |

307 的 rank0/rank1 差异来自采集窗口内的 rank 不平衡与 profiler 开销，不能直接当作生产设备空隙；967 的两个 rank 一致：**非 kernel 约 12.5 ms/请求**。

## 4. 结论

1. **307 token：CPU dispatch 是临界路径**
   - CPU 每层 2.0 ms ×64 ≈ 129 ms，高于设备 compute 77.3 ms；
   - 设备 kernel union 93.1 ms，CPU backbone 139.9 ms，host 侧更长；
   - 18.8 ms 残余的主要来源是 **逐层 dispatch + joint pooler/head + 前端 schema**，不是某个设备 kernel 缺失。

2. **967 token（S1 P95）：设备重新成为临界路径**
   - 设备 compute 144.5 ms，CPU backbone 128.5 ms；非 kernel 约 12.5 ms（7.3%）；
   - joint pooler（含 head）9.4 ms，仍是可见的 host 成本，但不再是最大项；
   - 因此对 **P95**，继续压 CPU 的收益有限；对 **P50/均值**，CPU/graph 优化的价值更大。

3. **“联合头 1.98 ms”只是设备计算**
   - head 的 CPU wall 是 7.2–8.4 ms，包含 enqueue、D2H、tolist、schema 循环；
   - 这部分在 `vllm/model_executor/models/clef_joint_schema.py` 与 strategies，不在 vLLM-Ascend kernel 内。

4. **vLLM-Ascend 模块的定位**
   - `scheduler` 0.13–0.17 ms，重写调度器没有意义；
   - `worker_execute_model` 是外壳，真正的逐层 CPU 在 vLLM eager 模型执行；
   - 值得动的 vLLM-Ascend 模块是 **`compilation/`（graph/ACL graph）** 与 `worker/model_runner` 的 host 路径，不是 attention/GDN/linear kernel。

## 5. 下一步建议

| 方向 | 目标 | 预期 | 风险 |
| --- | --- | --- | --- |
| Graph mode / ACL graph | 降低 64 层 Python dispatch | 短/中请求 10–20 ms；P95 较小 | 动态 shape、pooling、质量回归 |
| Joint head/pooler CPU | 向量化 `_indices`、减少 Python 循环、D2H/tolist | 3–6 ms（各长度） | 中；需逐位回归 |
| Schema token cache v2 | 省 tokenize/encode | 1–4.6 ms（字段数相关） | 低；需 ID/span 一致 |
| MLP/INT8、MC2 | 967+ 设备侧 | 见 MLP 报告 | 质量/平台限制 |

**不建议**：重写 vLLM-Ascend 的 scheduler（0.17 ms）、attention/GDN/linear kernel（设备侧已接近平台）；这些不解决 18.8 ms 残余。

## 6. 证据

- CPU scope：`Clef-27B/logs/gdn-data-path-20261115/profile-residual/analysis-307/cpu_observations.csv`、`analysis-969/…`
- 设备窗口：`analysis-307/summary-compact.json`、`analysis-969/summary-compact.json`
- 采集记录：`analysis-307/manifest.json`、`analysis-969/manifest.json`、`instrumentation-manifest.json`
- 样本：`sample-307.json`、`sample-969.json`
- 服务恢复：`restore/compare.json`、`restore/verify.json`
- 远端原始 trace：`/workspace/yangchen/logs/gdn-data-path-20261115/profile-residual/traces-307/`（4 个 rank 目录，约 270 MB）

服务已按 phase-0 快照恢复为 **PID 638964**，argv/cwd/env 逐项一致，`/health` 200，冻结 8 例 **8/8**。
