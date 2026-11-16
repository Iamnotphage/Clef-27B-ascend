# Clef-27B 调优目录

## 当前提交指标（2026-11-16）

### 精度指标

**OpenDecisions 1.0.1 全量 2,352 题（作者原生评分器，0～100 分）**

| 口径 | Calibration | Intelligence | Capability | Accuracy | Score MAE |
| --- | ---: | ---: | ---: | ---: | ---: |
| 官方公开 as-served（不做温度校准） | 82.3 | 65.8 | 74.1 | 未公布 | 未公布 |
| 我们适配后 baseline（2026-11-13） | 82.7941 | 66.0568 | 74.4255 | 0.8104（1906/2352） | 0.401389（588） |
| 当前优化版本（BIC，2026-11-15） | **82.7941** | **66.0568** | **74.4255** | **0.8104**（1906/2352） | **0.401389** |
| 当前 − 官方 | +0.4941 | +0.2568 | +0.3255 | — | — |
| 当前 − 适配 baseline | 0 | 0 | 0 | 0 | 0 |

- 当前优化与适配 baseline **2,352/2,352 逐位一致**；
- 相对官方公开 as-served 的三项最大偏差 **0.494 pp**，按 2PP 口径**通过**；
- 论文的 with-temperatures 结果（83.4 / 65.4 / 74.4）**已忽略**，官方列全部为 as-served；
- 官方未公布 accuracy、Score MAE，不能做这两项的官方对照；
- 官方公开值来自 OpenDecisions 数据卡榜单的 `Cloudflare/clef` 参考，**不是主办方正式 baseline**；其硬件与服务配置与 A3 不同。

**ERQA 官方 400 题（总体 accuracy，0～1）**

| 口径 | 总体 accuracy | 单图 | 多图（2～16 张） |
| --- | ---: | ---: | ---: |
| 官方公开 Clef 分数 | 未找到 | — | — |
| 我们适配后 baseline（2026-11-14） | **219/400 = 0.5475** | 170/287 = 0.5923 | 49/113 = 0.4336 |
| 当前优化版本（BIC，2026-11-15） | **219/400 = 0.5475** | 同上 | 同上 |
| 当前 − 适配 baseline | **0（400/400 概率逐位一致）** | 0 | 0 |
| 当前 − 官方 | **N/A** | — | — |

- R/BIC 400 题概率逐位一致，预测变化 0、最大概率差 0.0；
- 官方模型卡、论文与 ERQA 仓库未提供 Clef 的公开 ERQA accuracy，**官方基线暂缺**，因此暂不能对该项做 2PP 判定；
- 单图/多图为适配 baseline 的同一组预测，BIC 不改变预测。

### 性能指标

| 版本 | 主要配置 | S1 P95 | S1 mean | S1 p50 | S2 decisions/s | S2 并发 | ERQA P95 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 适配后 baseline | TP2/BF16/eager/4096/seqs1 | 217.036 ms | 180.069 ms | 205.034 ms | 5.715 | C16 | 325.84 ms |
| 当前优化（BIC） | TP2/BF16/eager/4096；S1 seqs1，S2 seqs4 | **198.42 ms** | **160.59 ms** | **179.86 ms** | **14.652（C8）/ 14.689（C16）** | C8 / C16 | **309.84 ms** |
| 变化 | — | **−18.62 ms（−8.58%）** | −19.48 ms | −25.17 ms | **约 2.56×（C8）** | — | **−16.00 ms（−4.91%）** |

- S1 为单并发端到端 P95，全量 2,352 题，2,352/2,352 成功；
- S2 为 seqs4 并发扫描；C8 是推荐工作点，C16 只多约 0.25% 吞吐但 p95 接近翻倍；
- ERQA P95 使用 ERQA 服务的 16,384 上下文 / `image=16` 配置，与 S1 的 4,096 / `image=1` 不是同一服务配置；
- 官方公开值没有可比的 S1/S2 或 ERQA 延迟，性能不做官方对照；
- **BF16 B8/B16 S2 候选（未替换当前 S2）**：B8 C16 **16.464 dec/s**、p95 1804 ms；B16 C64 **17.110 dec/s**、p95 6088 ms；OD 三项均在 2PP 内，ERQA 合批质量待补。

## 当前状态

完整 Clef-27B 已接入固定提交的 vLLM-Ascend，提供 `POST /v1/systemone`，服务模型名为 `clef`。主干 12 个分片、视觉塔和独立联合头已实际加载，联合头 122/122 个参数张量齐全。TP1、TP2 的三类决策、多字段、候选概率与 token/span 自检均通过。[首轮工程 baseline 记录](../docs/baseline-2026-11-12.md)

比赛质量按 **OpenDecisions + ERQA** 准备。2026-11-13 全新 OpenDecisions 1.0.1 全量评测完成，2,352/2,352 成功、0 错误；原始概率 Calibration 82.7941、Intelligence 66.0568、Capability 74.4255，accuracy 0.810374，Score MAE 0.401389（588 条）。作者固定版评分脚本与 EvalScope 未舍入指标完全一致。2026-11-14 完成 ERQA 官方 400 题全量图文评测，400/400 成功、0 重试，accuracy **219/400 = 0.5475**；单图 0.5923、多图 0.4336，8 类分项见报告。[本轮复现说明](../docs/opendecisions-baseline-2026-11-13.md) · [结果汇总](logs/opendecisions-baseline-20261113/summary.json) · [ERQA 报告](../docs/erqa-baseline-2026-11-14.md)

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

`start.sh` 使用 `PYTHON_BIN` 指定的解释器转发到 `inference.py`，启动前检查完整分片和联合头文件。`--output-dir` 写入 `launch-manifest.json` 和 `service.log`；默认监听 `127.0.0.1:18143`。`--config s1` 为最终采用的 TP2/4096/seqs1/缓存关；**不指定 `--config` 时默认 s1**，因此 `start.sh --model-dir ... --output-dir ...` 直接启动验收配置。`--config s2` 为 TP2/4096/seqs4/schema cache v2 开（吞吐候选，质量不通过）；`--config custom` 保留 BF16、TP1、1024、单序列的保守默认。附加 vLLM 参数放在 `--` 后，`--flag value` 与 `--flag=value` 均可；重复参数报错，passthrough 覆盖同名 launcher 参数时 manifest 记录实际生效值和来源。

服务接口：`GET /health`（JSON：status/model/revision）、`GET /v1/decide/info`（身份、任务、协议、缓存）、`POST /v1/systemone`（文本决策）、`POST /v1/decide`（文本/图文决策，与 systemone 同协议）。

```bash
python scripts/verify_systemone.py \
  --report /workspace/yangchen/logs/systemone-smoke.json
```

## 上游参考入口

`joint_schema_model.py` 提供 `load_release_model(local_model_path, device=...)` 和 `systemone(model, processor, request)`，支持 `noul`、`choice`、`score`。[上游文件](https://www.modelscope.cn/models/Cloudflare/clef/files)

比赛入口已对接 vLLM-Ascend 的原生决策协议，`joint` 使用默认值。[赛事说明](https://competition.gitcode.com/competition/2119684851993116674/intro)

## 结果记录

`metrics.csv` 采用比赛七列质量汇总，保存 OpenDecisions 与 ERQA 两行实测值。OpenDecisions 三项质量分按 0～100 分记录，accuracy 按 0～1 记录，Score MAE 保留等级单位。ERQA 行 accuracy 为 0.5475，其余四项为 `NA`。逐题证据见 [OpenDecisions 日志](logs/opendecisions-baseline-20261113/) 与 [ERQA 日志](logs/erqa-baseline-20261114/)。

`results.csv` 当前 S1 为 seqs1、缓存关、GDN 决策优化与 BIC 开启的 P95 **198.420 ms**。S2 当前为 seqs4/BIC 的 **14.652 decisions/s @ C8**（C16 14.689 见 GDN 数据路径报告），`gdn_decision_prefill`、`gdn_data_path` 与 `measurement_id` 列标明运行身份。seqs4 的 13.764 dec/s 行保留第四轮工程门槛判定；ERQA 的 baseline C1 P95 323.476 ms、BIC P95 309.84 ms 分别记录在独立报告中。[优化与兼容记录](optimization.md) · [当前 GDN 数据路径验收](../docs/gdn-data-path-2026-11-15.md)
