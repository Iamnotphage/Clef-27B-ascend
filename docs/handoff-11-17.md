# Clef-27B Ascend 交接文档（2026-11-17）

> **2026-11-17 调优后更新**：`ACLNN_CACHE_LIMIT=1000000` 把官方 `start.sh --config s1` 入口的全量 S1 降到 **mean 125.55 / p50 118.17 / p95 168.63 ms、7.960 dec/s**（原 153.76 / 168.49 / 190.40、6.503），再叠加 **MLP BF16 NZ**（GDN 安装器 v4）后为 **mean 121.71 / p50 113.66 / p95 166.00 ms、8.213 dec/s**；2,352 条响应逐位一致，ERQA 217/400，S2 B8 C16 16.376 dec/s。本文的 S1/S2 是调优前快照；最新结论与表格见 `docs/final-summary-2026-11-17.md` 和 `docs/aclnn-cache-shape-fix-2026-11-17.md`。另：307 profile 的 CPU/设备绝对值受采集开销污染，不可用于生产结论（967 profile 仍有效）。

## 0. 交接摘要

当前提交版本已经完成并验证：

- **S1（seqs1，2,352 题）P95 190.405 ms**，baseline 217.036 ms，**−12.27%**；
- **S2（B8 C16，2,352 题）16.398 dec/s**，baseline 5.715，**2.87×**；
- OpenDecisions Capability **74.3107**，accuracy 1902/2352，Score MAE 0.401545；
- ERQA **217/400 = 0.5425**；
- 相对官方公开 as-served（82.3/65.8/74.1）三项偏差 **+0.409 / +0.112 / +0.211 pp**，全部在 2PP 内。

当前分支 `codex/tuning-perf2a-20261113`，HEAD `9446cda`，与 origin 一致。唯一未提交的是用户原有的 `docs/profiling-2026-11-12.md`，不要覆盖或提交它。

## 1. 机器与环境

| 项 | 值 |
| --- | --- |
| 机器 | `root@199.88.54.221`，A3 单物理卡内 **2 个逻辑 NPU**（`Ascend910_9382`，各 64 GiB HBM） |
| 资源约束 | 全部推理/评测进程合计限用这 2 个逻辑 NPU，**禁止跨物理卡** |
| OS / CPU / RAM | Ubuntu 22.04.5 aarch64，32 核等价 CPU，128 GiB |
| CANN | 9.2.1（Toolkit / A3 Ops / NNAL ATB） |
| Python | 3.12，`/workspace/yangchen/venvs/clef-py312` |
| torch / torch_npu | 2.10.0+cpu / 2.10.0.post7 |
| Transformers | 5.19.0 |
| vLLM | `24420b7a341af37a9bc3186e6fc6062512e0aa58` |
| vLLM-Ascend | `89eb8a5e7a76d085b453a1aeadb6883e12d5ea2d` |
| 模型 | `/workspace/yangchen/models/Clef-27B`（官方 BF16，12 shards + joint head） |
| 工程 | `/workspace/yangchen/Clef-27B-ascend`，editable 安装 |

SSH 使用会话提供的跳板凭据；**不要把 `jt_账号:令牌` 写入仓库、日志或聊天**。重连用 `! ./connect.sh` 或平台提供的 `CLEF_JUMP_AUTH`。

## 2. 当前服务与提交入口

### 2.1 当前 baseline 服务

- 8000 端口，**PID 769624**，phase-0 恢复版；
- argv/cwd/env 与 `/workspace/yangchen/logs/gdn-data-path-20261115/phase0/snapshot.json` 逐项一致；
- `/health` 200，冻结 8 例 8/8；
- 当前 PID 记录：`/workspace/yangchen/logs/current-service.json`。

### 2.2 最终提交配置

S1：

```bash
bash Clef-27B/start.sh \
  --model-dir /workspace/yangchen/models/Clef-27B \
  --output-dir <run> --config s1
```

- TP2 / BF16 / eager / max_model_len 4096 / max_num_seqs 1 / max_num_batched_tokens 4096；
- prefix cache 关、chunked prefill 关；
- GDN 决策专用 prefill + BIC；
- GDN QKV norm v2（cap **1600**，总 token >1600 回退 BIC）；
- 单字段 joint head 特化。

S2（最终 B8 C16，仍使用 `--config s1` 覆盖调度参数，**不要用 `--config s2`**；`s2` 是历史 schema-cache-on 候选，未采用）：

```bash
bash Clef-27B/start.sh \
  --model-dir /workspace/yangchen/models/Clef-27B \
  --output-dir <run> --config s1 \
  --max-num-seqs 8 --max-num-batched-tokens 8192
```

## 3. 与 baseline 的对比

### 3.1 定义

- **baseline**：2026-11-14 适配后的 BF16 版本，TP2/BF16/eager/4096/seqs1/缓存关；
- **final**：上述 baseline + GDN 决策路径 + BIC + QKV kernel v2 + 单字段 head；
- **官方公开**：OpenDecisions 数据卡榜单的 Cloudflare/clef **as-served**（不做温度校准），不是主办方正式 baseline。

### 3.2 精度

| 指标 | baseline | final | 变化 | 官方 as-served |
| --- | ---: | ---: | ---: | ---: |
| OpenDecisions Calibration | 82.7941 | **82.7092** | −0.0849 | 82.3 |
| OpenDecisions Intelligence | 66.0568 | **65.9122** | −0.1446 | 65.8 |
| OpenDecisions Capability | 74.4255 | **74.3107** | −0.1148 | 74.1 |
| OpenDecisions accuracy | 1906/2352 | **1902/2352** | −4 题（−0.17 pp） | 未公布 |
| OpenDecisions Score MAE | 0.401389 | **0.401545** | +0.000156 | 未公布 |
| ERQA accuracy | 219/400 = 0.5475 | **217/400 = 0.5425** | −0.50 pp | 未找到 |
| ERQA 单图 | 170/287 = 0.5923 | **168/287 = 0.5854** | −0.69 pp | — |
| ERQA 多图 | 49/113 = 0.4336 | **49/113 = 0.4336** | 0 | — |

- 相对官方公开 as-served：**+0.409 / +0.112 / +0.211 pp**，2PP 通过；
- 相对自己的 baseline：最大变化 0.145 pp，2PP 通过；4 条 choice 变化、max 概率差 0.0851；
- **没有温度校准**；2PP 口径见 `docs/submission-rules-2026-11-13.md` 第 4.3 节。

### 3.3 性能

S1（seqs1，全量 2,352）：

| 指标 | baseline | final | 变化 |
| --- | ---: | ---: | ---: |
| P95 | 217.036 ms | **190.405 ms** | **−26.631 ms（−12.27%）** |
| Mean | 180.069 ms | **153.761 ms** | −26.308 ms（−14.61%） |
| P50 | 205.034 ms | **168.485 ms** | −36.549 ms（−17.83%） |
| P99 | 222.139 ms | **203.202 ms** | −18.937 ms（−8.52%） |
| 吞吐 | 5.549 dec/s | **6.503 dec/s** | **+17.2%** |

S2（全量 2,352）：

| 版本 | 配置 | 并发 | 吞吐 | P95 |
| --- | --- | ---: | ---: | ---: |
| baseline | seqs1（实际 B1） | C16 | 5.715 | 3292.7 ms |
| seqs4 | max_num_seqs 4 | C8 | 14.652 | 913.9 ms |
| B8 最终 | max_num_seqs 8 / 8192 | C16 | **16.398** | **1802.9 ms** |
| B10 | max_num_seqs 10 / 10240 | C20 | 16.670 | 2248.9 ms |
| B12 | max_num_seqs 12 / 12288 | C24 | 16.841 | 2597.5 ms |
| B16 | max_num_seqs 16 / 16384 | C64 | 17.110 | 6088.1 ms |

- 最终 S2 相对 baseline **2.87×**；相对 seqs4 C16 **+11.6%**，p95 基本持平；
- 如果 S2 规则只看最大吞吐，B16 C64 是 17.110；如果看 p95/有效延迟，B8 C16 是推荐点。

## 4. 已采用的优化

### 4.1 GDN 决策专用 prefill

- 纯 prefill、无 decode 的 pooling 服务跳过 `output_final_state`/scatter 和反向中间量；
- guard 全宿主侧，不读设备张量；
- 全量 OD OFF/ON 逐位一致；6 对 A/B p95 中位 **−3.80%**；
- 入口 S1 P95 213.56 ms；
- 提交 `6fd128d`、`525a1dd`；报告 `docs/gdn-decision-prefill-2026-11-15.md`。

### 4.2 BIC 数据路径

- BSND + `initial_state=None` + `conv_states=None`；
- 全量 2,352 逐位一致，ERQA 400 逐位一致；
- 6 对 A/B p95 中位 **−7.01%（−13.81 ms）**；
- S1 198.42 ms；
- 提交 `811b220`；报告 `docs/gdn-data-path-2026-11-15.md`。

### 4.3 GDN QKV norm v1 / v2

v1：

- conv 输出 `[40,T,128]` 一次写成 BSND q/k/v，Q/K L2Norm，V 直接拷贝；
- 短序列 2×，长序列 v1 退化，因此只做长度分派。

v2（最终采用）：

- 每行一次 `ReduceSum(128)`，替代两次 64；
- `Broadcast<float,2,1>` + 一次 `Mul` 替代逐行标量 `Muls`；
- 969 单算子 **2.01×**，1595 **1.65×**；969 p95 0.2098 vs baseline 0.3244 ms（1.55×）；
- cap **1600**，覆盖 OD 最大 1,595 和 ERQA 1200–1595；
- 服务 v2 同窗口 v1 p95 **−2.48%**；
- 质量 2PP；
- 提交 `a76f3ba`；报告 `docs/gdn-kernel-v2-2026-11-16.md`；
- vendor：`Clef-27B/src/ascendc/vendors/customize`；v4 安装器 `runtime/apply_qkv_norm_patch.py`；postimage `runtime/qkv_norm/`。

### 4.4 单字段 joint head

- OD 全部单字段；Q=1 时 self-attention softmax 恒为 1，跳过 Q/K 投影和 softmax，保留 V/out_proj、cross-attn、FFN、norm、residual；
- option routing 单字段化简；
- 多字段自动回退；
- head 同窗口 head-off 203.102 → head-on **197.184 ms**，**−2.91%**；
- 最终 head-entry S1 **190.405 ms**、6.503 dec/s；
- 质量 2PP；
- 提交 `b725c2f`；报告 `docs/joint-single-field-2026-11-16.md`；
- 安装器 `runtime/apply_joint_single_field_patch.py`，postimage `runtime/joint_single_field/`，preimage `197863d5…`，postimage `f8101b98…`。

### 4.5 S2 B8 C16

- 最终 B8 C16 **16.3985 dec/s**、p95 1802.93、mean 971.62；
- OD Capability 74.2510、accuracy 1909、2PP；
- 提交 `82df587`、`ed65bd4`；报告 `docs/batching-b10-b12-2026-11-16.md`、`docs/batching-seqs8-16-2026-11-16.md`。

### 4.6 输入分布分析

- OD token p50/p95/p99 = 357.5 / 969.25 / 1211.96，max 1595；cap 1200 覆盖 98.85%；全单字段、候选 2–7；
- ERQA p50/p95/p99 = 933 / 1653 / 3559，max 10,294；cap 1200 仅覆盖 68.75%；3+ 图 53 题；
- 该数据支持 cap=1600、B8 C16，不支持 joint head 跨请求批处理；
- 提交 `a7cc1cc`；报告 `docs/input-distribution-2026-11-16.md`。

## 5. 负结果：不要再重复

| 方向 | 结果 | 结论 |
| --- | --- | --- |
| W8A8 / W8A16 / INT8 / FP8 | 赛方禁止量化；2PP 不能放行 | **不可提交** |
| ACL graph / graph | capture [1] 未 replay；显式 [307,967] 短请求快，但全量 S1 p95 244.5/265.8 | 不采用 |
| schema token cache v2 | s1/307 慢约 5%，969 持平 | 不采用 |
| BF16 NZ 服务化 | 307 p95 +1.4%，969 基本持平 | 不采用 |
| MC2 down+allreduce | A3 缺 MatmulAllReduce（161001）；Grouped 单组 SIGSEGV/挂起 | 不可用 |
| MatMulReduceScatter V2 | 本地 RS 1.10–1.23×，RS+AllGather 仅 1.04–1.06×，1596 0.97× | 需 Dense SP，不采用 |
| MLP-only Dense SP | 0.88–0.98× | 不采用 |
| 完整 Dense SP 移植 | 质量 2PP，但 S1 p95 212.96 vs 198.42 | 不采用 |
| joint head 跨请求批处理 | B4 变慢，B8 小样本 +2.5%，全量 +0.03% | 不采用 |
| attention prefill cache skip | 307 更慢，969 噪声内 | 不采用 |
| 同层 GEMM+AllReduce 分块 | 0.29–0.80× | 不采用 |
| TP1 | S1 p95 264.3 ms | 保持 TP2 |

历史量化/合批数据仍保留在 `docs/mlp-*.md`、`docs/batching-*.md`，但量化已因赛方规则退出候选。

## 6. 最终 profile 与瓶颈

数据：`docs/profile-final-2026-11-17.md`，`Clef-27B/logs/gdn-data-path-20261115/final-profile/`。

### 6.1 CPU（每请求，inclusive）

- 307：每层 1.78–2.18 ms ×64 = **114–139 ms**，设备 compute **75 ms**；CPU dispatch 临界；
- 967：每层 1.81–2.11 ms ×64 = **116–135 ms**，设备 compute **144.6 ms**；设备临界；
- joint pooler（含 head）8.9–11.5 ms，head 6.7–9.0 ms；
- schema tokenize 1.6–2.5 ms，tokenizer 0.14–1.41 ms；
- scheduler 0.29–0.36 ms。

### 6.2 设备（967，每请求）

- compute 144.6 ms；
- communication 32–35 ms，非重叠 13.5–17.2 ms；
- no device kernel 12.5–15.7 ms。

### 6.3 结论

- S1 P95 967 是设备侧临界；head/pooler 再优化约 1–2 ms；
- 307/中短请求是 CPU dispatch 临界；完整 graph 已证明 P95 变差；
- 现实的下一步 S1 区间 **185–188 ms**；170–180 ms 在当前 BF16/不量化/TP2 栈没有可靠路径。

## 7. 可复现命令

```bash
cd /workspace/yangchen/Clef-27B-ascend
source scripts/activate_env.sh

# 检查三个安装器
python runtime/apply_gdn_decision_patch.py --check
python runtime/apply_qkv_norm_patch.py --check
python runtime/apply_joint_single_field_patch.py --check

# S1
bash Clef-27B/start.sh --model-dir /workspace/yangchen/models/Clef-27B \
  --output-dir <run>/s1 --config s1 --port 18143

# S2 最终 B8
bash Clef-27B/start.sh --model-dir /workspace/yangchen/models/Clef-27B \
  --output-dir <run>/s2 --config s1 \
  --max-num-seqs 8 --max-num-batched-tokens 8192 --port 18143
```

全量 OD：

```bash
python runtime/od_full_benchmark.py \
  --sample-file /workspace/yangchen/logs/tuning/20261114-013000-fix4/opendecisions/all2352.json \
  --url http://127.0.0.1:18143/v1/systemone --concurrency 1 --report <run>/od.json
```

ERQA 参考：

```bash
python scripts/official_eval/erqa_baseline.py \
  --data-dir /workspace/yangchen/datasets/erqa-official-20261114 \
  --output-dir <run>/erqa --api-url http://127.0.0.1:18143/v1 --scope full
python scripts/official_eval/erqa_score.py \
  --run-dir <run>/erqa --data-dir /workspace/yangchen/datasets/erqa-official-20261114 \
  --output <run>/erqa/offline-score.json --verify-requests
```

## 8. 当前源码/补丁状态

| 项 | 状态 |
| --- | --- |
| GDN 决策补丁 | v3，`apply_gdn_decision_patch.py --check` true |
| QKV norm v2 | vendor v2 + qkv postimage，`apply_qkv_norm_patch.py --check` true |
| 单字段 head | postimage `f8101b98…`，`apply_joint_single_field_patch.py --check` true |
| vLLM | `24420b7a` |
| vLLM-Ascend | `89eb8a5e` |
| `inference.py` s1 | preprocess_fusion true、joint_single_field true、cap 1600 |
| `inference.py` s2 | 历史候选，schema cache on；**最终 S2 用 s1 覆盖，不要用 s2** |
| 量化 | 无，官方 BF16 权重 |

## 9. 开放问题 / 下一步

1. **主办方正式 baseline 仍未取得**；当前对比是官方公开 as-served 与本地 baseline；
2. **ERQA 官方公开值缺失**，ERQA 2PP 只能相对自己的 baseline 判断；
3. **S2 有效性规则未明确**：若只看最大吞吐，B16 C64 17.110；若有 p95 约束，B8 C16 16.398；`results.csv` 暂按 B8；
4. **如继续 S1 优化**：
   - GDN kernel v3 tile/向量归约（0.5–1.5%）；
   - head/pooler 再融合（1–2 ms）；
   - schema/tokenizer CPU（最多 2–3 ms，需重测 cache）；
   - MLP BF16 tiling/NZ（1–2%）；
5. **不建议**：量化（禁止）、完整 Dense SP（退化）、MC2（不可用）、完整 graph（P95 变差）、joint head 跨请求批处理（无收益）；
6. 任何新优化必须跑全量 OD 三项 + ERQA accuracy + 2PP，再谈性能。

## 10. 交接注意

- 不要提交 `docs/profiling-2026-11-12.md`（用户原有修改）；
- 不要跨物理卡；不要 kill 同门/其他进程；
- 模型权重、SSH 凭据、跳板令牌不入库；
- 服务恢复用 `/workspace/yangchen/logs/gdn-data-path-20261115/phase0/snapshot.json`，恢复后跑冻结 8 例；
- 认证/挂载目录以平台说明为准；模型路径和日志路径见 AGENTS.md。
