# Clef-27B 调优目录

## 当前状态

完整 Clef-27B 已接入固定提交的 vLLM-Ascend，提供 `POST /v1/systemone`，服务模型名为 `clef`。主干 12 个分片、视觉塔和独立联合头已实际加载，联合头 122/122 个参数张量齐全。TP1、TP2 的三类决策、多字段、候选概率与 token/span 自检均通过。[首轮工程 baseline 记录](../docs/baseline-2026-11-12.md)

比赛质量按 **OpenDecisions + ERQA** 准备。2026-11-13 全新 OpenDecisions 1.0.1 全量评测完成，2,352/2,352 成功、0 错误；原始概率 Calibration 82.7941、Intelligence 66.0568、Capability 74.4255，accuracy 0.810374，Score MAE 0.401389（588 条）。作者固定版评分脚本与 EvalScope 未舍入指标完全一致。2026-11-14 完成 ERQA 官方 400 题全量图文评测，400/400 成功、0 重试，accuracy **219/400 = 0.5475**；单图 0.5923、多图 0.4336，8 类分项见报告。[本轮复现说明](../docs/opendecisions-baseline-2026-11-13.md) · [结果汇总](logs/opendecisions-baseline-20261113/summary.json) · [ERQA 报告](../docs/erqa-baseline-2026-11-14.md)

2026-11-14 第四轮复核确认：seqs4 合批配置（13.764 decisions/s @ C8）的全量质量未通过 0.03/0.05 工程门槛（Capability 74.200，4 条 choice 变化，最大概率差 0.097399）；最终提交配置改为 **seqs1/B1**，全量质量与 baseline 逐位一致，S1 P95 **217.036 ms**，S2 **5.715 decisions/s @ C16**。完整根因、GDN 覆盖和接口验收见 [第四轮报告](../docs/tuning4-2026-11-14.md)。

2026-11-14 另行完成 CPU BF16 独立参考链路：两套数据全量输入等价（OpenDecisions 2,352/2,352、ERQA 400/400 token 数等于 NPU 实测），NPU 复放 24/24 逐位一致；逐题前向按用户要求完成 OpenDecisions 154/2352 后终止，未生成完整 CPU 评分。[CPU 参考对照](../docs/cpu-npu-reference-2026-11-14.md) · [证据](logs/cpu-reference-20261114/)

2026-11-15 完成 GDN 决策专用 prefill 路径，省去最终循环状态导出、写回和反向中间量输出。全量 2,352 题逐位不变，6 对交错 A/B 全部更快，p95 降幅中位数 3.80%。补丁已接入 `start.sh --config s1` 默认启用，启动时安装并校验，manifest 记录状态。正常入口 S1 P95 213.56 ms，ERQA 400 ON/OFF 逐题一致，原服务恢复至 PID 498272。[GDN 报告](../docs/gdn-decision-prefill-2026-11-15.md) · [证据](logs/gdn-decision-20261115/)

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

`results.csv` 当前 S1 为 seqs1、缓存关、GDN 决策优化开启的 P95 **213.560 ms**。S2 **5.715 decisions/s @ C16** 来自优化前 GDN 关闭的扫描。两项各有 `gdn_decision_prefill` 与 `measurement_id` 列标明运行身份，优化后的并发扫描待完成。seqs4 的 13.764 dec/s 行保留第四轮工程门槛判定，ERQA 的历史 C1 P95 323.476 ms 保存在其独立报告中。[优化与兼容记录](optimization.md) · [当前 GDN 验收](../docs/gdn-decision-prefill-2026-11-15.md)
