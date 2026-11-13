# OpenDecisions 全量 baseline：2026-11-13

本轮按用户要求先终止 Decision Index，再重新请求完整 OpenDecisions，保留原始概率。北京时间 15:58:06—16:05:24 执行完成，退出码 0，总耗时 437.878 秒。该耗时用于说明本次质量评测，S1/S2 性能按指定协议单独测试。

## 结果

| 指标 | 实测结果 | 单位或分母 |
| --- | ---: | --- |
| Calibration | 82.7941149442 | 0～100 分 |
| Intelligence | 66.0568428085 | 0～100 分 |
| Capability | 74.4254788763 | 两项均值，0～100 分 |
| Accuracy | 0.8103741497 | 1,906 / 2,352，按原生题型规则判定 |
| Score MAE | 0.4013893127 | 588 条 Score 的期望等级绝对误差均值，等级单位 |
| 执行成功率 | 100% | 2,352 / 2,352，错误 0、incomplete=false |

| 题型 | 样本数 | 正确数 | Accuracy |
| --- | ---: | ---: | ---: |
| Choice | 1,178 | 981 | 83.2767% |
| Yes/no | 586 | 502 | 85.6655% |
| Score | 588 | 423 | 71.9388% |

Score 的 accuracy 使用最大概率等级；Score MAE 使用 `abs(sum(level * probability) - gold)`。原生三项质量评分保留按题型、难度、机会水平和概率校准计算的定义。[结果汇总](../Clef-27B/logs/opendecisions-baseline-20261113/summary.json) · [提交质量 CSV](../Clef-27B/metrics.csv)

## 数据、服务与评分条件

- 数据：OpenDecisions 1.0.1，`test`，完整 2,352 条；revision `ff0b640be2b72c7d06d9d656a34ad6eed79ec7fe`；原始 JSONL 的 SHA-256 为 `78614972006c5e5efbbfc121ce31e15cf576ef6c0cdf24a229b33d34d2bfcda6`。
- 数据路径：`/workspace/yangchen/datasets/clef-full-20261112/full-data/open-decisions2352.jsonl`，入口重新校验哈希与样本数。
- 服务：现有 `clef`，BF16、TP2、eager、max model len 4096、max num seqs 1、max num batched tokens 4096，关闭 prefix caching 和 chunked prefill。
- 资源：A3 单物理卡，`npu-smi` 卡号 7、Chip 0 / 1、逻辑 NPU 0 / 1、Phy-ID 14 / 15，遵循卡内两逻辑 NPU 上限。
- 请求：单并发、每题一次 `POST http://127.0.0.1:8000/v1/systemone`；原始 state、question 和候选标签保持一致，`joint` 采用服务默认值。全新目录保存本次推理，保留完整原始概率，额外温度校准关闭。
- EvalScope：固定源码 `3d39bab09c1d63095490a8b0f4a6dcbcc5ea843e`，继承原生 OpenDecisions 评分及聚合，仅替换本地 JSONL 加载方式。`eval_batch_size=1`、timeout 300 秒、retries 1；逐题日志确认实际 HTTP 请求为 2,352 次，每题均一次尝试。

将全部保存的响应按 `metadata.source_id` 关联原始题目，导出 `{"id": ..., "probs": ...}`。转换器验证数据哈希、原始请求、标签、重复 ID、概率有效性和评分器哈希，概率保持原值。原作者固定版 `score.py` 的 SHA-256 为 `478d14ae7a779c72e0dcd6c3225a3462e24faa8d7e8092afb0d6422ff84e4377`，来源为[固定 revision](https://huggingface.co/datasets/jgeuter/OpenDecisions/raw/ff0b640be2b72c7d06d9d656a34ad6eed79ec7fe/score.py)。

作者脚本与 EvalScope 报告 `metadata.upstream_metric` 的三项未舍入值完全一致。EvalScope 顶层 ratio 保留四位小数，精确汇总及 CSV 使用作者未舍入的 0～100 分值。129 条真实概率校准样本的 mean TVD 为 0.2375644578，单独记录其定义和分母。

## 实际运行与复测

本次由 `nohup` 托管评测入口，supervisor PID 162905、EvalScope PID 162906。输出目录为 `/workspace/yangchen/logs/opendecisions-baseline-20261113-1600/`，两个进程已正常退出。启动命令及实际参数见 [launch.json](../Clef-27B/logs/opendecisions-baseline-20261113/launch.json)。

复测时登录服务器，选择新的空输出目录：

```bash
ssh clef
CLEF_OD_RUN=/workspace/yangchen/logs/opendecisions-repeat-01
/workspace/yangchen/venvs/clef-eval-py312/bin/python \
  /workspace/yangchen/Clef-27B-ascend/scripts/official_eval/open_decisions_baseline.py \
  --output-dir "$CLEF_OD_RUN"
```

入口实际调用如下，可按同样参数直接运行。`--benchmark open_decisions --scope full` 固定本次数据范围：

```bash
HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 \
/workspace/yangchen/venvs/clef-eval-py312/bin/python \
  /workspace/yangchen/Clef-27B-ascend/scripts/run_evalscope_offline.py \
  --data-dir /workspace/yangchen/datasets/clef-full-20261112/full-data \
  --work-dir "$CLEF_OD_RUN/evalscope" \
  --api-url http://127.0.0.1:8000/v1 \
  --benchmark open_decisions --scope full
```

作者评分复核使用已经保存的概率；仅执行转换和 CPU 评分：

```bash
CLEF_OD_DATA=/workspace/yangchen/datasets/clef-full-20261112/full-data/open-decisions2352.jsonl
CLEF_OD_SCORER=/workspace/yangchen/logs/opendecisions-baseline-20261113-1600/author-score.py
CLEF_OD_PY=/workspace/yangchen/venvs/clef-eval-py312/bin/python
"$CLEF_OD_PY" /workspace/yangchen/Clef-27B-ascend/scripts/official_eval/convert_open_decisions_predictions.py \
  --predictions "$CLEF_OD_RUN/evalscope/predictions/clef/clef_open_decisions_full2352_default.jsonl" \
  --data "$CLEF_OD_DATA" --scorer "$CLEF_OD_SCORER" \
  --output "$CLEF_OD_RUN/author-predictions.jsonl" > "$CLEF_OD_RUN/author-conversion.json"
"$CLEF_OD_PY" "$CLEF_OD_SCORER" "$CLEF_OD_RUN/author-predictions.jsonl" \
  --data "$CLEF_OD_DATA" --json > "$CLEF_OD_RUN/author-score.json"
```

## 产物与执行状态

| 产物 | 内容 |
| --- | --- |
| `launch.json`、`environment.json`、`exit.json` | 数据版本和哈希、实际命令、服务配置、设备映射、时间及退出码 |
| `native.log` | EvalScope 执行与进度日志 |
| `evalscope/predictions/clef/clef_open_decisions_full2352_default.jsonl` | 本次逐题请求、原始响应、完整概率及传输尝试信息 |
| `evalscope/reviews/clef/clef_open_decisions_full2352_default.jsonl` | 逐题正确性、Score MAE、gold 与概率 |
| `evalscope/reports/clef/clef_open_decisions_full2352.json` | 原生指标、分项、执行成功及错误数 |
| `author-predictions.jsonl`、`author-score.json`、`author-conversion.json` | 全部原始概率导出、作者评分及转换核对 |
| `summary.json`、`metrics.csv`、`artifact-manifest.json`、`raw-artifacts.json` | 精确指标汇总、比赛七列质量表、文件路径和哈希、请求次数与 reviews 独立核对 |

本地可复核的作者概率导出、报告及汇总见 [本轮证据目录](../Clef-27B/logs/opendecisions-baseline-20261113/)。原生完整报告在本地使用 `evalscope-report.json.gz` 无损保存，解压哈希与服务器原件一致；完整原始请求、响应及 reviews 在服务器本轮目录保留。reviews 独立重算的 1,906 个正确答案和 Score MAE 与作者概率导出的统计一致。

Decision Index 的评测 PID 146869 与 supervisor PID 146512 已于 15:55 按用户要求终止，部分输出保留，状态按未完成记录。[终止证据](../Clef-27B/logs/decision-index-stopped-20261113.json)
