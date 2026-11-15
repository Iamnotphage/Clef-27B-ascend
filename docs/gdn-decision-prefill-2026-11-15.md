# GDN 决策专用 prefill 路径（2026-11-15）

本轮把 GDN 的“纯 prefill、无 decode”特化实现成可选路径：服务端不再为没有读者的循环状态和反向中间量付成本。改动以版本化补丁 + 安装器 + 默认关开关交付，源码在验证后已回退到原始哈希，原服务按 phase-0 快照恢复。

**结论**

- 数值完全不变：8 例冻结回归 OFF/ON 均 8/8；OpenDecisions 全量 2,352 题在 OFF 与 ON 下**逐位一致**（最大概率差 0.0），也与冻结 baseline 逐位一致。
- 同进程交错 A/B（588 题分层子集，6 对，交替顺序）：**6/6 对 ON 更快**，p95 中位 **−3.80%（−7.68 ms）**，均值 −4.13%，吞吐中位 +0.17 decisions/s。
- 全量 2,352 单次：wall 413.65→408.30 s（−1.29%），p50 195.26→191.03 ms（−2.17%），p99 216.66→211.32 ms（−2.47%），p95 204.65→204.86 ms（单次噪声内，不作结论）。
- **S1 默认启用**：`inference.py` 在启动时安装并校验补丁，manifest 记录实际状态；`CLEF_GDN_DECISION_PREFILL=0` 可关闭。

## 1. 改了什么

服务配置为 seqs1、无 prefix caching、无 chunked prefill、无 speculative decoding，模型一次 pooling 前向直接出决策（`output_tokens=0`），因此**从来不会有 decode 步**。但固定 vLLM-Ascend 的 GDN prefill 分支仍无条件：

1. `gather_ssm_states` 取出初始状态（无历史时为全零），并对输入状态做 `transpose(-1,-2).contiguous()`；
2. `output_final_state=True` 计算并返回 final state；
3. `final_state.transpose(-1,-2).contiguous()` 转置后 `scatter_ssm_states_` 写回 `ssm_state`；
4. fused kernel 的 `disable_recompute=True` 每层分配并写出 `g_cumsum [B,T,Hv] fp32` 与 `A [B,Hv,T,64]`，这两个张量只供反向重算使用。

本轮的决策专用路径**保留第 1 项**（初始状态 gather 与输入状态转置仍在执行），在满足 guard 时消除第 **2、3、4** 项：最终状态导出、其转置与 scatter 写回，以及反向中间量输出。按 446 tokens、48 层估算，第 4 项的 `A` 约 2.7 MB/层、约 130 MB/请求。

因此收益来自“无读者的输出”，而不是初始状态读取；若后续要再省第 1 项，需要让 kernel 接受 `initial_state=None` 并单独验证。

### 1.1 代码落点

| 文件 | 改动 |
| --- | --- |
| `vllm_ascend/ops/gdn_decision_flags.py`（新增） | 进程默认开关 + 显式 opt-in 的 `CLEF_GDN_FLAG_FILE` 运行时覆盖（1 秒 mtime 缓存），用于同进程 A/B |
| `vllm_ascend/device/device_op.py` | A3 `fla_gdn_prefill` 两处调用：`output_final_state=not decision_only`、`disable_recompute=not skip_recompute`；`decision_only` 时返回 `(output, None)` |
| `vllm_ascend/ops/gdn.py` | prefill 分支：宿主侧 guard；`last_recurrent_state is None` 时跳过状态转置与 scatter |
| `vllm_ascend/patch/worker/patch_qwen3_5.py` | 在实际执行的 `QwenGatedDeltaNetAttention.__init__` 包装器里记录 `_decision_safe_config` |

### 1.2 Guard

```text
decision_safe_config                      # chunked prefill / prefix cache / spec decode 全部关闭
and decision_prefill_enabled()            # 开关
and num_prefills > 0
and num_decodes == 0
and num_decode_tokens == 0
and num_spec_decodes == 0
and num_spec_decode_tokens == 0
```

全部是宿主侧整数/布尔量，**不读设备张量**。这一点是第 4 节负结果的核心。

### 1.3 开关

- `CLEF_GDN_DECISION_PREFILL=1` 进程默认开启；`CLEF_GDN_NO_RECOMPUTE` 可单独控制第 4 项。
- 实验覆盖：服务以 `CLEF_GDN_FLAG_FILE=<路径>` 启动时才读取该 JSON（`{"decision_prefill": true}`），用于不重启服务切换；提交入口不设置该变量。
- 默认值均为关；关闭时行为与原始代码逐字节等价。

## 2. 正确性

| 检查 | 结果 |
| --- | --- |
| 8 例冻结回归（v1 / v2 / v5，OFF 与 ON） | 各 8/8 逐位一致 |
| OpenDecisions 全量 2,352：OFF vs 冻结 baseline | **2,352/2,352 逐位，max 0.0** |
| 全量：A-only vs OFF | 2,352/2,352 逐位，max 0.0 |
| 全量：A+B vs OFF | 2,352/2,352 逐位，max 0.0 |
| 全量：ON vs OFF（v5 最终代码） | **2,352/2,352 逐位，max 0.0** |
| token 数 | 2,352/2,352 与冻结运行相等 |

说明：`disable_recompute=False` 只停止写出反向中间量，`output_final_state=False` 只停止返回/写回无人读取的 final state；两者都不改变前向输出，逐位结果与此一致。

## 3. 性能

### 3.1 同进程交错 A/B（判定依据）

588 题按**等距抽样**（每 4 题取 1，覆盖三类与不同长度），同一服务进程内交替 OFF/ON，奇偶对交换顺序。

| 对 | 顺序 | p95 OFF (ms) | p95 ON (ms) | 变化 |
| ---: | --- | ---: | ---: | ---: |
| 1 | OFF→ON | 209.145 | 194.739 | −6.89% |
| 2 | ON→OFF | 202.709 | 195.178 | −3.72% |
| 3 | OFF→ON | 199.684 | 196.270 | −1.71% |
| 4 | ON→OFF | 199.878 | 193.636 | −3.12% |
| 5 | OFF→ON | 204.146 | 193.035 | −5.44% |
| 6 | ON→OFF | 201.782 | 193.961 | −3.88% |
| **中位** | | | | **−3.80%（−7.68 ms）** |

- ON 更快：**6/6**；p95 降幅 ≥1% 的预注册条件满足。
- 吞吐中位变化 **+0.170 decisions/s**；均值降幅 4.13%。

### 3.2 全量 2,352 单次

| 配置 | wall (s) | mean (ms) | p50 (ms) | p95 (ms) | p99 (ms) | decisions/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| OFF | 413.65 | 175.86 | 195.26 | 204.65 | 216.66 | 5.686 |
| A+B | 408.30 | 173.59 | 191.03 | 204.86 | 211.32 | 5.760 |
| 变化 | −1.29% | −1.29% | −2.17% | +0.10% | −2.47% | +1.31% |

单次全量的 p95 差异落在本机噪声内（同代码的 OFF 在不同时间测得 204.6–228.8 ms），因此 p95 结论只采用 3.1 的交错 A/B。

### 3.3 v1 三次全量（仅作参考，顺序混杂）

| 顺序 | 配置 | wall (s) | mean (ms) | p95 (ms) |
| ---: | --- | ---: | ---: | ---: |
| 1 | OFF | 436.61 | 185.62 | 228.82 |
| 2 | A-only | 433.67 | 184.37 | 220.38 |
| 3 | A+B | 415.71 | 176.74 | 215.97 |

三次运行随顺序单调变快，说明存在明显的 ramp/位置效应；这组数字不作为效果判定，只用于说明“跨服务/跨顺序比较不可靠”。

## 4. 负结果与设计迭代

| 版本 | 方案 | 结果 |
| --- | --- | --- |
| v1 | 在 `device_op` 内只按开关决定 | 逐位正确；但没有元数据 guard，误在含 decode 的服务开启会破坏状态 |
| v2 | guard 读 `prefill_has_initial_state.any().item()`，按 `(id(metadata), data_ptr)` 缓存 | **6/6 对更慢，p95 中位 +6.75%（+14.8 ms）**；GDN 的 metadata 是逐层的，缓存每层失效，等于 48 次设备同步/请求 |
| v3 | 在 `AscendGatedDeltaNetAttention.__init__` 记录配置 | 实际实例是 `QwenGatedDeltaNetAttention`，属性不存在，500 |
| v4 | 用 `attn_metadata.attn_state` | `GDNAttentionMetadata` 无此字段，500 |
| **v5** | 在真正执行的 `__init__` 包装器记录配置；guard 只用宿主整数 | 8/8、逐位一致、A/B 6/6 更快 |

v2 的负结果是本轮最有价值的教训：**推理热路径上任何逐层设备同步都会吞掉状态写回省下的时间**。

## 5. 复现

```bash
cd /workspace/yangchen/Clef-27B-ascend

# 状态与安装（源码树为固定 vLLM-Ascend 89eb8a5 的 editable 安装）
python runtime/apply_gdn_decision_patch.py --check
python runtime/apply_gdn_decision_patch.py --apply
python runtime/test_gdn_decision_patch.py

# 服务（实验端口 18143；生产入口仍为 8000）
source scripts/activate_env.sh
CLEF_GDN_DECISION_PREFILL=0 bash Clef-27B/start.sh \
  --model-dir /workspace/yangchen/models/Clef-27B \
  --output-dir /workspace/yangchen/logs/gdn-decision-20261115/service \
  --config s1 --port 18143

# 同进程交错 A/B（服务须以同一 CLEF_GDN_FLAG_FILE 启动）
python runtime/gdn_decision_ab.py \
  --sample-file /workspace/yangchen/logs/gdn-decision-20261115/ab/ab-subset-stride4.json \
  --url http://127.0.0.1:18143/v1/systemone --limit 10000 --pairs 6 \
  --flag-file /dev/shm/clef-gdn-ab.json --service-pid <service-pid> \
  --output-dir /workspace/yangchen/logs/gdn-decision-20261115/ab/pairs

# 回退
python runtime/apply_gdn_decision_patch.py --revert
```

## 6. 采用决定

- 用户已决定采纳并**推进 S1 默认启用**；集成方式见第 9 节。
- 补丁仍保留 `CLEF_GDN_DECISION_PREFILL=0` 关闭路径，便于回退与对照。
- 本轮 A/B 测的是 A+B 合并路径；A-only 与 B-only 的单独贡献未用最终代码重复测量，v1 的逐次数据受顺序混杂，不作归因。
- S2 收益未测量：当前 A/B 是单并发顺序回放，+0.170 decisions/s 不能外推到并发场景。

## 7. 证据

| 内容 | 路径 |
| --- | --- |
| phase-0 快照与补丁清单 | `Clef-27B/logs/gdn-decision-20261115/phase0/` |
| 8 例冻结 OFF/ON | `.../quality/v5-frozen-off.json`、`v5-frozen-on.json` |
| 全量 2,352 OFF / A+B 原始报告 | `.../quality/v5-od-off.json.gz`、`v5-od-a-b.json.gz` |
| 逐题逐位对照 | `.../quality/v5-compare.json.gz` |
| 6 对交错 A/B | `.../ab/ab-summary.json`、`.../logs/ab-v5.log` |
| 服务恢复核验 | `.../restore/snapshot.json` |
| 工具与补丁 | `runtime/apply_gdn_decision_patch.py`、`runtime/gdn_decision_flags.py`、`runtime/od_full_benchmark.py`、`runtime/compare_od_runs.py`、`runtime/gdn_decision_ab.py`、`runtime/test_gdn_decision_patch.py` |

远端原始目录：`/workspace/yangchen/logs/gdn-decision-20261115/`。

## 8. 服务恢复

- 原始服务停止后，按 phase-0 快照恢复为 **PID 498272**；argv、cwd、环境白名单与 PID 432954 逐项一致。
- `GET /health` **200**；8/8 冻结用例逐位一致。
- 三个被补丁的源码文件 SHA-256 回到 preimage：`device_op.py a08d44b4…`、`gdn.py b235e34d…`、`patch_qwen3_5.py a9a71634…`；`gdn_decision_flags.py` 已移除。
- 当前服务指针已更新为 498272。

## 9. 启动入口集成（S1 默认启用）

`Clef-27B/inference.py` 在构造启动参数时调用 `gdn_decision_prefill(config)`：

- `--config s1` 默认请求启用；`CLEF_GDN_DECISION_PREFILL=0` 可显式关闭。
- 先运行安装器 `--check`：源码处于 preimage 时按需 `--apply`，已 patched 时跳过；逻辑幂等。
- 源码树与预期 preimage 不符（例如缺少既有 BSND 补丁）时**拒绝启动**并列出缺失项，避免“以为启用、实际没生效”。
- 启用后把 `CLEF_GDN_DECISION_PREFILL=1` 写入服务环境。
- `launch-manifest.json` 新增 `gdn_decision_prefill`：`enabled` / `requested` / `source` / `patch_state` / 每个文件的 SHA-256 / `flag_module_sha256` / `installer_sha256`。

**BSND 依赖**：本补丁的 preimage SHA 对应“已应用 C1′ BSND 补丁”的源码树。全新部署须先应用 `runtime/gdn-bsnd.patch`；入口检查会把缺失暴露为启动失败，而不是静默降级。

**实验开关隔离**：`/dev/shm` 文件覆盖改为显式 opt-in——服务必须以 `CLEF_GDN_FLAG_FILE=<路径>` 启动才会读取该文件；提交入口不设置该变量，散落的标志文件无法影响提交运行。A/B 驱动要求显式传入同一路径，并在 `finally` 中恢复原文件（含异常路径）。

**Guard 增加 pooling 条件**：普通生成服务的首个 prefill 批也满足其余条件，随后 decode 需要被跳过的循环状态；`model_config.runner_type == "pooling"` 在构造期排除该场景，并配生成路径回退测试。

## 10. 最终验收（正常入口）

### 10.1 启动清单

`bash Clef-27B/start.sh --model-dir ... --output-dir ... --config s1 --port 18143` 生成的 `launch-manifest.json` 记录：

- `gdn_decision_prefill.enabled = true`，`source = config:s1`；
- `patch_state`：`device_op.py`、`gdn.py`、`patch_qwen3_5.py` 均为 `patched`；
- 三个文件的 SHA-256：`ad92a322…`、`509dafb0…`、`1b82b52e…`；
- `flag_module_sha256`：`cf1ede3f…`；`installer_sha256` 一并记录；
- 服务环境 `CLEF_GDN_DECISION_PREFILL=1`，进程 `/proc/<pid>/environ` 已核对。

### 10.2 正确性

| 检查 | 结果 |
| --- | --- |
| 8 例冻结回归（入口服务，ON） | 8/8 |
| 全量 2,352 vs 冻结 baseline | **2,352/2,352 逐位，max 0.0** |
| ERQA 400 ON vs OFF | **400/400 逐题一致，max 概率差 0.0** |
| ERQA accuracy | 219/400 = 0.5475（两边相同） |

### 10.3 S1（正常入口，全量 2,352）

| 项 | 值 |
| --- | ---: |
| 成功 / 失败 | 2,352 / 0 |
| wall | 408.33 s |
| decisions/s | 5.760 |
| mean / p50 / p95 / p99 | 173.60 / 195.57 / **213.56** / 223.56 ms |

`results.csv` 的 S1 行更新为 213.5597 ms；S2 仍是优化前的 seqs1 扫描值（5.715 @ C16），本轮未做并发复测，需在后续并发测量中更新。

### 10.4 服务恢复

- 实验服务停止后，按 phase-0 快照恢复原服务为 **PID 498272**；argv、cwd、环境白名单与 phase-0 逐项一致。
- `GET /health` 200；8/8 冻结用例逐位一致。
- 源码树保留已应用的补丁（这是新的默认入口状态）；原 8000 服务进程未设置 `CLEF_GDN_DECISION_PREFILL`，运行的是 baseline 路径，输出与冻结结果一致。
- 当前服务指针更新为 498272。
