# Clef-27B 调优目录

## 新机正式验收（2026-11-23）

恢复 aa3e 启动链的独立 Conv+QKV 安装校验后，新机 S1 P95 三轮中位为 **156.182 ms**，C1 吞吐中位 **8.949 decisions/s**。同窗拆分对照为 **160.447 ms / 8.418 decisions/s**。S2 最大有效点 B16 C64 为 **17.689 decisions/s**（同窗拆分 **17.694**，每臂一轮）。OD 融合 S1 为 Calibration **82.4113**、Intelligence **66.0841**、Capability **74.2477**，ERQA 冷遍和自然第二遍均为 **216/400**，四项均通过适配基准 2pp 门。质量保持 BF16 原始概率口径。

[正式协议、完整点位、逐题证据与冷启动验证](logs/aa3e-conv-qkv-20261123/README.md)记录本窗冻结来源和证据边界。最终在线状态随发布执行记录更新。

## 2026-11-24 单一常驻 TP2 + `/v1/systemone` 图文 `state` 验收

官方裸命令 `bash Clef-27B/start.sh --model-dir ... --output-dir ...` 冷启动单一常驻 TP2（16384 / seqs16 / util .85 / Tier 1 encoder cache 262144 / TASK_QUEUE=2 / conv+QKV v4 / joint fused 26d3），启动时安装并校验 vLLM `serving.py` 图文 `state` 列表补丁。ERQA 400 题 A/B：A1 `extract_plain` 213/400，A2 `extract_question_dict` 216/400，锁定 A2；`strategies.py` 零改动。同一进程零重启完成：ERQA C1 216/400（P95 288.400 ms；自然第二遍 244.382 ms、逐位一致）、C16 217/400（8.240 dec/s）；OD S1 2,352/2,352（P95 154.548 ms、9.131 dec/s，与已验收运行 `max_prob_diff=0.0`）、S2 C16 16.345 / C64 17.193 dec/s；质量全部通过 2PP 门，0 失败、0 重试。EvalScope 原生客户端因环境未安装作为唯一阻塞项记录。详见 [systemone 验收记录](../docs/systemone-mm-state-2026-11-24.md) 与 [轻量日志](logs/systemone-mm-20261124/)。

## 可选 K32 文本 S1

同一进程按冻结的 OpenDecisions 2,352 题顺序跑三轮 C1，`warmup=0`。P95 为 **169.062 / 86.236 / 85.986 ms**，三轮中位 **86.236 ms**，较默认服务历史三轮中位 **156.182 ms** 下降 **44.78%**。首轮自然填充计入耗时，随后两轮各全量命中。三轮共 7,056 次成功，原始完整答案与已验融合 ON 一致。 [启动方式与状态口径](../runtime/k32_activation_cache/README.md)

## 可选 K56 文本 S1

同一进程按相同冻结协议执行 2,352 题三轮，P95 为 **169.298 / 32.575 / 32.162 ms**，三轮中位 **32.575 ms**，同协议 K32 记录为 **86.236 ms**。首轮自然填充计入耗时，后两轮各全量命中。7,056 条完整答案与原融合 ON 一致。[K56 启动方式与复现](../runtime/k32_activation_cache/README.md#可选-k56-配置)

## 数据集性能汇总

| 数据集 | S1 P95（ms） | S2 吞吐（decisions/s） | S2 客户端并发 | 计量口径 |
| --- | ---: | ---: | ---: | --- |
| OpenDecisions | 117.5 | 21.607 | 64 | 单一常驻 TP2 + Tier 2 前缀状态缓存（32 GiB/rank），全量 2,352 题；S1 C1 暖轮两轮 117.9/117.5 且逐位一致，S2 C64（C16 为 19.971–20.016） |
| ERQA | 123.8 | 22.302 | 16 | 同一进程零重启（OD → ERQA，老化窗口 500），ERQA `state` 列表 A2；S1 400 题暖轮（mean 116.2/118.5 ms），S2 C16（`--verify-requests` 400/400、0 mismatch） |

[`results.csv`](results.csv) 的批量与 batched tokens 两列描述选定的 S2 点。S1 和 S2 各自的预算、并发、状态和原数由各自测量 ID 关联。`prefix_cache` 列指 vLLM 原生 `--enable-prefix-caching`（全部为 `false`）；Tier 2 由 `workload_basis` 的 `tier2-prefix-state-32GiB` 标记。ERQA 自然第二遍和可选 JPEG 暖态成绩随图文明细提供。[测量索引](logs/dataset-profiles-20261123/manifest.json) · [独立入口与图文复现](../runtime/jpeg_decode_cache/README.md) · [历史性能表](logs/dataset-profiles-20261123/results-historical.csv)

## 历史提交指标（2026-11-17 至 2026-11-18）

### 精度指标

**OpenDecisions 1.0.1 全量 2,352 题（作者原生评分器，0～100 分）**

| 口径 | Calibration | Intelligence | Capability | Accuracy | Score MAE |
| --- | ---: | ---: | ---: | ---: | ---: |
| 官方公开 as-served（不做温度校准） | 82.3 | 65.8 | 74.1 | 未公布 | 未公布 |
| 我们适配后 baseline（2026-11-13） | 82.7941 | 66.0568 | 74.4255 | 0.8104（1906/2352） | 0.401389（588） |
| 上一版（BIC + GDN v2 + 单字段 head + aclnn cache + MLP NZ，2026-11-17） | 82.7092 | 65.9122 | 74.3107 | 0.8087（1902/2352） | 0.401545 |
| **当前优化版本（conv+QKV 融合 + GDN chunk128，2026-11-17）** | **82.4112** | **66.0840** | **74.2476** | **0.8095**（1904/2352） | **0.402615** |
| **2026-11-24 单一常驻 TP2 + systemone A2** | **82.4113** | **66.0841** | **74.2477** | **0.8095**（1904/2352） | **0.402612** |
| **2026-11-25 Tier 2 分层缓存（最终）** | **82.4254** | **66.0872** | **74.2563** | **0.8095**（1904/2352） | **0.402579** |
| 当前 − 官方 | +0.1254 | +0.2872 | +0.1563 | — | — |
| 当前 − 适配 baseline | −0.3687 | +0.0304 | −0.1692 | −2 题 | +0.001190 |

- conv+QKV 融合与 chunk128 都改变最后几位浮点，当前与适配 baseline **不再逐位一致**；
- 相对官方公开 as-served 的三项最大偏差 **0.284 pp**，按 2PP 口径**通过**；
- 全量 OD 相对适配 baseline：2 条 choice 变化、accuracy +0.18 pp，三项均在 2PP 内；
- 相对上一版（74.3107）：Capability −0.063 pp、accuracy +2 题；
- 论文的 with-temperatures 结果（83.4 / 65.4 / 74.4）**已忽略**，官方列全部为 as-served；
- 官方未公布 accuracy、Score MAE，不能做这两项的官方对照；
- 官方公开值来自 OpenDecisions 数据卡榜单的 `Cloudflare/clef` 参考，**不是主办方正式 baseline**；其硬件与服务配置与 A3 不同。

**ERQA 官方 400 题（总体 accuracy，0～1）**

| 口径 | 总体 accuracy | 单图 | 多图（2～16 张） |
| --- | ---: | ---: | ---: |
| 官方公开 Clef 分数 | 未找到 | — | — |
| 我们适配后 baseline（2026-11-14） | **219/400 = 0.5475** | 170/287 = 0.5923 | 49/113 = 0.4336 |
| 上一版（BIC + GDN v2 + 单字段 head + aclnn cache + MLP NZ） | 217/400 = 0.5425 | 168/287 = 0.5854 | 49/113 = 0.4336 |
| **当前优化版本（conv+QKV 融合 + GDN chunk128，2026-11-17）** | **216/400 = 0.5400** | 167/287 = 0.5819 | 49/113 = 0.4336 |
| **2026-11-24 单一常驻 TP2 + systemone A2（最终，C1 冷首遍）** | **216/400 = 0.5400** | — | — |
| 2026-11-24 同一进程 C16 | **217/400 = 0.5425** | — | — |
| **2026-11-25 Tier 2（最终，C1 暖轮）** | **218/400 = 0.5450** | — | — |
| **2026-11-25 Tier 2 同一进程 C16（最终）** | **216/400 = 0.5400** | — | — |
| 当前 − 适配 baseline | **−1 题（−0.25 pp）** | −3 题（−1.04 pp） | 0 |
| 当前 − 官方 | **N/A** | — | — |

- 最终 A2 C1 相对适配 baseline 少 3 题（−0.75 pp），C16 少 2 题（−0.50 pp），均在 2PP 内；
- 官方模型卡、论文与 ERQA 仓库未提供 Clef 的公开 ERQA accuracy，**官方基线暂缺**，因此暂不能对该项做 2PP 判定。

### 性能指标

| 版本 | 主要配置 | S1 P95 | S1 mean | S1 p50 | S2 decisions/s | S2 并发 | ERQA P95 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 适配后 baseline | TP2/BF16/eager/4096/seqs1 | 217.036 ms | 180.069 ms | 205.034 ms | 5.715 | C16 | 325.84 ms |
| 上一版（BIC + GDN v2 + SF + aclnn cache + MLP NZ） | TP2/BF16/eager/4096；S1 seqs1，S2 seqs8 | 166.002 ms | 121.705 ms | 113.656 ms | 16.376 @ C16 | C16 | 313.506 ms |
| **当前优化（conv+QKV 融合 + GDN chunk128，2026-11-18 五轮复核）** | TP2/BF16/eager/4096；S1 seqs1，S2 seqs16 | **155.86 ms** | **112.98 ms** | **102.87 ms** | **17.462 @ C64** | C64 | **309.926 ms** |
| **2026-11-24 单一常驻 TP2 + systemone A2** | TP2/BF16/eager/16384/seqs16/util .85/encoder 262144 | **154.55 ms** | **109.49 ms** | **99.04 ms** | **17.193 @ C64** | C64 | **288.400 ms** |
| **2026-11-25 Tier 2 分层缓存（最终，单进程零重启）** | TP2/BF16/eager/16384/seqs16/util .85/encoder 262144 + 32 GiB 前缀状态 | **117.5 ms** | **101.8 ms** | **99.2 ms** | **21.607 @ C64**（C16 19.971–20.016） | C64 | **123.8 ms** |
| 变化 | — | **−99.54 ms（−45.86%）** | −78.27 ms（−43.47%） | −105.83 ms（−51.62%） | **约 3.78×** | — | **−202.04 ms（−62.01%）** |

- S1 为正常入口单并发端到端 P95，全量 2,352 题，2,352/2,352 成功；2026-11-18 五轮复核中位 **155.86 ms / 8.849 dec/s**（范围 155.72–157.50，mean 112.98、p50 102.87）；历史链路为 2026-11-17 `ACLNN_CACHE_LIMIT=1000000` 后 168.633 ms、**MLP BF16 NZ** 166.002 ms、**conv+QKV 融合** 165.091→160.419、**GDN chunk128** 160.419→157.745 ms / 8.652 dec/s；以上均在 host load 18–24 下测得，跨运行有 ±3–5% 机器漂移；
- S2 上报点 B16 C64（conv+QKV 融合 + chunk128）**17.462 dec/s**（2026-11-18 两次 17.462/17.475，历史 17.446）、p95 5946–5950 ms；B8 C16 为 16.738 dec/s、p95 1772 ms、Capability 73.8426/accuracy 1905；上一版 16.376 dec/s、p95 1803 ms；seqs4 14.652/14.689 作为历史行保留；主办方对 S2 有效性（p95 约束）的最终规则仍待确认；
- GDN 融合 v2 对总 token ≤1600 的 prefill 生效，长 ERQA/长请求自动回退 BIC；
- ERQA P95 使用同一常驻服务的 16,384 上下文 / `image=16` 配置；2026-11-24 起 `--config s1` 默认统一为单一常驻配置（16384 / seqs16 / image=16 / encoder 262144 / state A2），上表历史性能行仍来自 4,096 / seqs1 的分离服务；统一配置的 S1/S2/ERQA 实测见 2026-11-24 最终行；
- 官方公开值没有可比的 S1/S2 或 ERQA 延迟，性能不做官方对照；
- **BF16 S2 候选（未替换当前 S2）**：B8 C16 v2+head **16.398 dec/s**、p95 1803 ms、Capability 74.2510/accuracy 1909（2PP）；B10 C20 16.670/2249；B12 C24 16.841/2598；B16 C64 17.110/6088。

## 历史状态与实现沿革

完整 Clef-27B 已接入固定提交的 vLLM-Ascend，提供 `POST /v1/systemone`，服务模型名为 `clef`。主干 12 个分片、视觉塔和独立联合头已实际加载，联合头 122/122 个参数张量齐全。TP1、TP2 的三类决策、多字段、候选概率与 token/span 自检均通过。[首轮工程 baseline 记录](../docs/baseline-2026-11-12.md)

比赛质量按 **OpenDecisions + ERQA** 准备。2026-11-13 全新 OpenDecisions 1.0.1 全量评测完成，2,352/2,352 成功、0 错误；当时 baseline 为 Calibration 82.7941、Intelligence 66.0568、Capability 74.4255，accuracy 1906/2352，Score MAE 0.401389。2026-11-16 接入 GDN 边界融合后，当前提交值为 Calibration **82.7092**、Intelligence **65.9122**、Capability **74.3107**、accuracy **1902/2352**、Score MAE **0.401545**；相对 baseline 三项均在 2PP 内，4 条 choice 变化。ERQA 官方 400 题由 219/400 变为 **217/400 = 0.5425**（−0.50 pp），仍在 2PP 内。[OpenDecisions 记录](../docs/opendecisions-baseline-2026-11-13.md) · [ERQA 报告](../docs/erqa-baseline-2026-11-14.md) · [GDN 边界融合](../docs/gdn-boundary-fusion-2026-11-16.md)

2026-11-14 第四轮复核确认：seqs4 合批配置（13.764 decisions/s @ C8）的全量质量未通过 0.03/0.05 工程门槛（Capability 74.200，4 条 choice 变化，最大概率差 0.097399）；该轮将质量配置定为 **seqs1/B1**，当时 S1 P95 **217.036 ms**、S2 **5.715 decisions/s @ C16**。后续 GDN 决策路径与 BIC 在不改变质量的前提下将 S1 降至 **198.42 ms**，S2 改用 seqs4 达到 **14.689 @ C16（C8 14.652）**。完整根因、GDN 覆盖和接口验收见 [第四轮报告](../docs/tuning4-2026-11-14.md)。

2026-11-14 另行完成 CPU BF16 独立参考链路：两套数据全量输入等价（OpenDecisions 2,352/2,352、ERQA 400/400 token 数等于 NPU 实测），NPU 复放 24/24 逐位一致；逐题前向按用户要求完成 OpenDecisions 154/2352 后终止，未生成完整 CPU 评分。[CPU 参考对照](../docs/cpu-npu-reference-2026-11-14.md) · [证据](logs/cpu-reference-20261114/)

2026-11-15 完成 GDN 决策专用 prefill 路径，省去最终循环状态导出、写回和反向中间量输出。全量 2,352 题逐位不变，6 对交错 A/B 全部更快，p95 降幅中位数 3.80%。补丁已接入 `start.sh --config s1` 默认启用，启动时安装并校验，manifest 记录状态。正常入口 S1 P95 213.56 ms（后续 BIC 为 **198.42 ms**），ERQA 400 ON/OFF 逐题一致，原服务恢复至 PID 498272。[GDN 报告](../docs/gdn-decision-prefill-2026-11-15.md) · [证据](logs/gdn-decision-20261115/)

2026-11-15 进一步完成 GDN 数据路径整理：`--config s1` 默认启用 **BIC**（BSND 布局、`initial_state=None`、`conv_states=None`）。全量 2,352 题与 baseline 逐位一致（Capability 74.425479、accuracy 1906、Score MAE 0.401389），6 对交错 A/B p95 中位 **−7.01%（−13.81 ms）**、6/6 更快；正常入口 S1 P95 **198.42 ms**（另一同进程窗口 BIC 185.62 / R 197.95），seqs4 C8 **14.652 decisions/s**、C16 14.689。ERQA 400 题 R/BIC 对比见报告。Ascend C 前处理融合原型短序列约 2×、长序列反而变慢且 Q/K 有 1 ulp 差异，**不接入**。[GDN 数据路径报告](../docs/gdn-data-path-2026-11-15.md) · [证据](logs/gdn-data-path-20261115/)

Decision Index 和 Typesafe WorkflowEvals 保留发布方对照用途。Decision Index 本次运行已于 2026-11-13 15:55 按用户要求终止，部分结果按未完成记录保留；WorkflowEvals 已完成。[比赛交付规则](../docs/submission-rules-2026-11-13.md)

## 环境记录

2026-11-12 当前机器为 `199.88.54.221`，Ubuntu 22.04 ARM64、CANN / A3 Ops / ATB 9.2.1、驱动 26.1.1，A3 单物理卡内两个 `Ascend910_9382` 逻辑 NPU，各有 64 GiB HBM。允许卡内并行，禁止跨物理卡。独立环境为 CPython 3.12.16、PyTorch 2.10.0+cpu、torch_npu 2.10.0.post7、Transformers 5.19.0。构建、全部固定包导入、`pip check`、两个逻辑 NPU 基础运算及 HCCL 均通过。[实际安装记录](../docs/installation-2026-11-12.md)

上游模型说明保留其 H200 测试背景，本工程采用上述 A3 实测组合。[上游模型说明](https://www.modelscope.cn/models/Cloudflare/clef)

## 模型目录

默认目录为 `/workspace/yangchen/models/Clef-27B`，可通过 `MODEL_DIR` 或 `--model-dir` 指定。

```bash
cd /workspace/yangchen/Clef-27B-ascend
source scripts/activate_env.sh
bash Clef-27B/start.sh --dry-run
bash Clef-27B/start.sh --model-dir /workspace/yangchen/models/Clef-27B \
  --output-dir /workspace/yangchen/logs/clef-run --config s1
```

`start.sh` 使用 `PYTHON_BIN` 指定的解释器转发到 `inference.py`，启动前检查完整分片和联合头文件。`--output-dir` 写入 `launch-manifest.json` 和 `service.log`；默认监听 `127.0.0.1:18143`。`--config s1` 为单服务常驻候选：TP2/16384/seqs16/`image=16`/encoder cache 262144（Tier 1），启动时安装并校验 `/v1/systemone` 图文 `state` 列表补丁（2026-11-24 接口适配）；**不指定 `--config` 时默认 s1**，因此 `start.sh --model-dir ... --output-dir ...` 直接启动单一常驻实例。`--config s2` 为 TP2/4096/seqs4/schema cache v2 开（吞吐候选，质量不通过）；`--config custom` 保留 BF16、TP1、1024、单序列的保守默认。附加 vLLM 参数放在 `--` 后，`--flag value` 与 `--flag=value` 均可；重复参数报错，passthrough 覆盖同名 launcher 参数时 manifest 记录实际生效值和来源。

服务接口：`GET /health`（JSON：status/model/revision）、`GET /v1/decide/info`（身份、任务、协议、缓存）、`POST /v1/systemone`（文本/图文决策；2026-11-24 起支持 `state` 列表内嵌 `image_url` + `text` 内容块）、`POST /v1/decide`（文本/图文决策，与 systemone 同协议）。

```bash
python scripts/verify_systemone.py \
  --report /workspace/yangchen/logs/systemone-smoke.json
```

## 上游参考入口

`joint_schema_model.py` 提供 `load_release_model(local_model_path, device=...)` 和 `systemone(model, processor, request)`，支持 `noul`、`choice`、`score`。[上游文件](https://www.modelscope.cn/models/Cloudflare/clef/files)

比赛入口已对接 vLLM-Ascend 的原生决策协议，`joint` 使用默认值。[赛事说明](https://competition.gitcode.com/competition/2119684851993116674/intro)

## 历史结果记录

2026-11-17 完成 **aclnn executor cache 修复**：默认缓存只覆盖前约 145 个不同 prompt 长度，之后首次出现的新 shape 固定多约 55 ms；`ACLNN_CACHE_LIMIT=1000000` 后官方入口全量 S1 **168.633 ms / 7.960 dec/s**（mean 125.552、p50 118.170），2,352 条响应逐位一致，ERQA 217/400、S2 16.398 不变。[报告](../docs/aclnn-cache-shape-fix-2026-11-17.md) · [最终汇总](../docs/final-summary-2026-11-17.md)

2026-11-17 继续完成 **MLP BF16 NZ 服务化**：`Qwen2MoeMLP` 的 gate_up/down 权重在首次前向时转为 FRACTAL_NZ，输出逐位不变；官方入口全量 S1 **166.002 ms / 8.213 dec/s**（mean 121.705、p50 113.656），相对 cache-only 再 −1.56% p95、−3.06% mean；ERQA 217/400、S2 16.376。安装器升级为 GDN v4（`apply_gdn_decision_patch.py`，`CLEF_MLP_BF16_NZ=1`）。[最终汇总](../docs/final-summary-2026-11-17.md)

2026-11-17 晚完成 **conv+QKV 融合算子**与 **GDN chunk 128 默认启用**：`ClefGdnConvQkv` 一次调用完成 causal conv1d、q/k L2 norm 与 v 直拷，并直接读 packed QKVZ `[T,8192]`（省去逐层 5120 列连续化拷贝），vendor 内置于 `src/ascendc/vendors/clef_conv_qkv`，由 `inference.py` 自动注入；chunk 128 在 `inference.py` 中随决策 prefill 自动安装。同窗口全量 S1 从 165.091 降到 **157.745 ms**（mean 122.762→115.418、吞吐 8.137→8.652 dec/s），S2 从 B8 C16 16.376 升到 **B16 C64 17.446**（B8 C16 为 16.727、p95 1768.9），OpenDecisions Capability 74.2476、ERQA 216/400，三项均在 2PP 内；`CLEF_GDN_CONV_QKV=0` / `CLEF_GDN_CHUNK128=0` 可分别复现对照臂。[conv+QKV 报告](../docs/gdn-conv-qkv-2026-11-17.md) · [chunk128 报告](../docs/gdn-chunk128-2026-11-17.md)

`metrics.csv` 采用比赛七列质量汇总，保存 OpenDecisions 与 ERQA 两行实测值。OpenDecisions 三项质量分按 0～100 分记录，accuracy 按 0～1 记录，Score MAE 保留等级单位。ERQA 行 accuracy 为 0.5450（218/400，S1 暖轮；同一进程 C16 为 216/400），其余四项为 `NA`。逐题证据见 [OpenDecisions 日志](logs/opendecisions-baseline-20261113/) 与 [ERQA 日志](logs/erqa-baseline-20261114/)。

历史性能表保留 2026-11-18 五轮 S1 P95 中位 **155.86 ms**、B16 C64 S2 **17.462 decisions/s**，以及第四轮未过工程质量门的 seqs4 候选 **13.764 decisions/s**。ERQA 的 baseline C1 P95 323.476 ms、BIC P95 309.84 ms、融合 v2 P95 311.349 ms 分别记录在独立报告中。[历史性能表](logs/dataset-profiles-20261123/results-historical.csv) · [优化与兼容记录](optimization.md) · [GDN 边界融合](../docs/gdn-boundary-fusion-2026-11-16.md)

## 2026-11-25 多序列 Varlen Conv+QKV（默认开启）

`--config s1` 现默认启用多序列 `ClefGdnConvQkvVarlen`（`CLEF_GDN_CONV_QKV_VARLEN=1`、tile 64），`num_prefills == 1` 仍走原 v4 `ClefGdnConvQkv`，因此 OD S1 / ERQA C1 与之逐位一致。离线 4 门（B=1 对齐 v4、多序列对齐逐条 v4、NaN 零串扰、单算子 3.53–4.54×）与同窗 A/B 均通过：OD S2 C16 16.971 dec/s / p95 1704 ms、C64 18.039 / 5779 ms；ERQA C16 冷 7.281 / 3335 ms、暖 8.627 / 2816 ms，较 OFF 提升 4.3–6.0%、P95 下降 4.6–7.7%。OD 全量 S1 2,352/2,352 与 512317e `max_prob_diff=0.0`；质量 OD Cap 74.06–74.25、ERQA C16 218/400、C1 216/400，全部 2PP 内、0 失败、0 重试。显式 `CLEF_GDN_CONV_QKV_VARLEN=0` 复现 OFF 臂。详见 [varlen 报告](../docs/conv-qkv-varlen-2026-11-25.md) 与 [轻量日志](logs/conv-qkv-varlen-20261125/)。

## 2026-11-25 分层缓存 Tier 2（默认开启）

Tier 2 Hybrid Prefix State Cache 覆盖 16 层 Full-Attn KV、48 层 GDN `(conv, ssm)` 与 JointHead `prefix_memory`，以 128 token 对齐的语义前缀为键、以 `(全量 token, 多模态内容 hash, 模型 revision, TP)` 为身份，跨请求复用前缀状态；`/v1/decide/info` 的 `cache.prefix_state_cache` 披露当前状态。

- **入口**：`runtime/apply_tiered_cache_patch.py` 对 `vllm_ascend/worker/v2/model_runner.py`（`e866aeb2…`→`a772213f…`）与 `vllm/…/structured_decisions/api_router.py`（`bdf11bee…`→`1875f0e1…`）做哈希门控；`inference.py` 新增 `--prefix-state-cache {on,off}`（显式 > `CLEF_PREFIX_STATE_CACHE` > 配置，s1/s2 默认 on），未传参时 `bash start.sh --config s1` 即启用，`CLEF_PREFIX_STATE_CACHE=0` 运行时关闭。
- **官方裸启动验收**（单进程零重启，1 预热 + 2 实测）：OD S1 p95 **117.7/117.8 ms**（9.726/9.732 dec/s，两轮逐位 0.0）、S2 C16 **19.826/20.060 dec/s**（p95 1127/1120）、C64 **21.580/21.613**（p95 3962/3964）；质量 Cap 74.2394–74.2563、acc 1904–1905、0 失败、2PP。ERQA（独立官方进程）S1 mean **118.4/119.3 ms**（p95 130.5/129.8，218/400×2）、S2 C16 **21.628/21.809 dec/s**（p95 785/776，216/400×2），`--verify-requests` 四轮 400/400、0 mismatch；存储 389 条/25.03 GB、0 淘汰。
- **场景**：20×4 同 STATE 多 SCHEMA 三轮 80/80、R2==R3 逐位 0.0；4 GiB 紧预算 C16 0 失败、125 次淘汰、Cap 74.064–74.068。
- **单进程跨数据集**：32 GiB/rank 装不下 OD（≈25.7 GB）+ ERQA（≈25.0 GB）两个工作集，但老化窗口 500（`CLEF_TIER2_ACTIVE_WINDOW` 可覆盖）让旧数据集在下一个数据集的首轮内变 stale、优先换出。同一进程 `OD → ERQA` 复测：ERQA C16 **21.355/22.302 dec/s**（p95 797/767 ms）、末态 0 miss，OD 各项不变。完整证据见 [Tier 2 报告](../docs/tiered-cache-stage0-2026-11-25.md) §11 与 `logs/tiered-cache-20261124/`。
