# 最终版本 S1 profile：BIC + GDN v2 + 单字段 head（2026-11-17）

对最终提交配置做一次 S1 profile，确认 190.405 ms 的最后一段花在哪里。服务：TP2/BF16/eager/4096/seqs1/BIC + GDN kernel v2（cap 1600）+ 单字段 head；307/967 token 各 10 预热 + 10 正式，C1。

> 采集开启 CPU observation 和 torch/NPU trace，绝对值有开销；下面 CPU scope 是 inclusive wall，嵌套项不可相加。设备窗口为 10 请求窗口，表中已换算为 **每请求**。

## 1. CPU scope（每请求，ms）

| scope | 307 rank0 | 307 rank1 | 967 rank0 | 967 rank1 |
| --- | ---: | ---: | ---: | ---: |
| `worker_execute_model` | 139.518 | 169.130 | 141.969 | 164.989 |
| `backbone` | 123.679 | 149.241 | 125.904 | 145.169 |
| `language_backbone` | 123.527 | 149.081 | 125.736 | 144.994 |
| `backbone_decoder_layer`（单层） | **1.784** | **2.175** | **1.811** | **2.109** |
| 64 层合计（估算） | **114.17** | **139.22** | **115.93** | **135.00** |
| `joint_pooler`（含 head） | 8.835 | 11.470 | 8.871 | 10.809 |
| `joint_head`（嵌套） | 6.728 | 8.989 | 6.685 | 8.210 |
| `joint_evidence_routing`（单次） | 1.462 | 2.014 | 1.465 | 1.751 |
| `schema_tokenize_encode` | 1.580 | — | 2.494 | — |
| `tokenizer` | 0.142 | — | 1.407 | — |
| `candidate_embedding_tp` | 0.582 | 0.773 | 0.614 | 0.842 |
| `schema_indices_h2d` | 0.382 | 0.426 | 0.393 | 0.455 |
| `scheduler`（单次/请求） | 0.361 | — | 0.287 | — |
| `state_render` | 0.033 | — | 0.032 | — |
| `api_handler_including_json` | 168.323 | — | 173.694 | — |

## 2. 设备侧（每请求，10 请求窗口 / 10）

| 指标 | 307 rank0 | 307 rank1 | 967 rank0 | 967 rank1 |
| --- | ---: | ---: | ---: | ---: |
| kernel window span | 168.93 | 169.72 | 174.26 | 174.11 |
| kernel union | 157.55 | 90.56 | 161.78 | 158.44 |
| compute union | **75.09** | 75.51 | **144.60** | 144.93 |
| communication union | 92.17 | 26.42 | 35.49 | 32.32 |
| compute/comm overlap | 9.71 | 11.37 | 18.30 | 18.81 |
| comm nonoverlapped | 82.46 | 15.06 | **17.18** | 13.51 |
| no device kernel | 11.38 | 79.16 | **12.48** | 15.68 |

307 的 rank0/rank1 差异主要来自 profiler 窗口内的 rank 不平衡和采集开销，不应直接当作生产环境设备空隙；967 两个 rank 一致，**非 kernel 约 12.5–15.7 ms**。

## 3. 结论

1. **307/中等长度：CPU dispatch 仍是临界路径**
   - 每层 1.78–2.18 ms × 64 = **114–139 ms**，高于设备 compute **75 ms**；
   - `worker_execute_model` 139.5–169.1 ms，`backbone` 123.7–149.2 ms；
   - 这解释了为什么 graph 只对短请求有效；当前 P95 的 969 不能靠单纯压 CPU 获得大收益。

2. **967（S1 P95）：设备重新成为临界路径**
   - 设备 compute **144.6 ms**，CPU backbone 125.9–145.2 ms；
   - 通信 union 32–35 ms，非重叠 13.5–17.2 ms；
   - 非 kernel 约 12.5–15.7 ms，仍以 joint pooler/head、schema/tokenizer 和前端为主。

3. **joint head 仍占 6.7–9.0 ms CPU，pooler 8.9–11.5 ms**
   - 单字段特化后 head 比旧 profile 略低，但仍是 967 非 kernel 的最大单项；
   - head 设备计算约 2 ms，剩余主要是 CPU dispatch、D2H/tolist、schema 循环。

4. **前端/schema 在 967 有 3.9 ms 左右**
   - `schema_tokenize_encode` 2.49 + `tokenizer` 1.41；
   - 之前 schema cache 在 307 更慢、967 持平，不能指望它显著降 P95。

5. **没有再出现新的设备 kernel 空白**
   - 967 非 kernel 12.5–15.7 ms 与旧 profile 的 12.5 ms 同量级；
   - 设备侧仍是 MLP/Cube、通信和 GDN 主核；`allreduce` 非重叠 13.5–17.2 ms 是剩余可争取的通信部分，但 Dense SP/MC2 已实测不成立。

## 4. 对后续工作的含义

- **S1 P95（967）：** 继续做 CPU head/pooler 最多约 1–2 ms；要再明显下降必须动设备侧（MLP/通信），而量化被禁止、Dense SP 退化、MC2 不可用；
- **短/中请求：** CPU per-layer 仍是临界路径；graph/compile 只在 <384 有效，完整 graph 已证明 P95 变差；
- **优先顺序：**
  1. joint head/pooler 再向量化/融合（1–2 ms，低风险）；
  2. GDN kernel v3 tile 级归约（0.5–1.5%）；
  3. MLP BF16 tiling/NZ（1–2%，收益有限）；
  4. S2 工作点已确定为 B8 C16；
- **现实目标：** S1 乐观 185–188 ms；170–180 ms 在当前 BF16/不量化/TP2 栈上仍无可靠路径。

## 5. 服务与证据

- 服务已恢复 baseline **PID 769624**，argv/cwd/env 与 phase-0 一致，`/health` 200，冻结 8/8；
- CPU：`Clef-27B/logs/gdn-data-path-20261115/final-profile/analysis-307/`、`analysis-969/`（`cpu_scopes.csv`、`cpu-decomposition.csv`、`summary-compact.json`）；
- 设备：同上 `compact-evidence.json`、`communication.csv`；
- 采集：`window-*b/manifest.json`、`instrumentation-manifest.json`、`restore/`；
- 原始 trace：`/workspace/yangchen/logs/gdn-data-path-20261115/final-profile/traces/`（4 个 rank 目录）。
