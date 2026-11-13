# Clef-27B 调优目录

## 当前状态

完整 Clef-27B 已接入固定提交的 vLLM-Ascend，提供 `POST /v1/systemone`，服务模型名为 `clef`。主干 12 个分片、视觉塔和独立联合头已实际加载，联合头 122/122 个参数张量齐全。TP1、TP2 的三类决策、多字段、候选概率与 token/span 自检均通过。[首轮工程 baseline 记录](../docs/baseline-2026-11-12.md)

比赛质量按 **OpenDecisions + ERQA** 准备。2026-11-13 全新 OpenDecisions 1.0.1 全量评测完成，2,352/2,352 成功、0 错误；原始概率 Calibration 82.7941、Intelligence 66.0568、Capability 74.4255，accuracy 0.810374，Score MAE 0.401389（588 条）。作者固定版评分脚本与 EvalScope 未舍入指标完全一致。[本轮复现说明](../docs/opendecisions-baseline-2026-11-13.md) · [结果汇总](logs/opendecisions-baseline-20261113/summary.json)

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
bash Clef-27B/start.sh --model-dir /workspace/yangchen/models/Clef-27B
```

`start.sh` 使用 `PYTHON_BIN` 指定的解释器转发到 `inference.py`，启动前检查完整分片和联合头文件。默认 BF16、TP1、eager、1024、单序列、`127.0.0.1:8000`，关闭 prefix caching 和 chunked prefill；保留视觉塔并跳过视觉预热，初始请求全部使用纯文本。附加 vLLM 参数放在 `--` 后。

```bash
python scripts/verify_systemone.py \
  --report /workspace/yangchen/logs/systemone-smoke.json
```

## 上游参考入口

`joint_schema_model.py` 提供 `load_release_model(local_model_path, device=...)` 和 `systemone(model, processor, request)`，支持 `noul`、`choice`、`score`。[上游文件](https://www.modelscope.cn/models/Cloudflare/clef/files)

比赛入口已对接 vLLM-Ascend 的原生决策协议，`joint` 使用默认值。[赛事说明](https://competition.gitcode.com/competition/2119684851993116674/intro)

## 结果记录

`metrics.csv` 采用比赛七列质量汇总，保存本次 OpenDecisions 实测值；三项质量分按 0～100 分记录，accuracy 按 0～1 记录，Score MAE 保留等级单位。逐题原始概率、作者评分、EvalScope 报告、数据版本、运行命令和退出记录见 [logs/opendecisions-baseline-20261113/](logs/opendecisions-baseline-20261113/)。`results.csv` 保留已有性能记录，S1/S2 与 ERQA 按正式协议继续补齐。原始请求、完整响应与 reviews 在远端本轮输出目录保留。[优化与兼容记录](optimization.md)
