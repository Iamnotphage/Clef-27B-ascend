# 比赛要求与 baseline 数据集

核查日期为 2026-11-12，时区为 Asia/Shanghai。赛事要求来自官方介绍页的公开 API 和已登录 Chrome 中的赛题发布页。本文的数据建议用于本工程构建 baseline，赛事正式数据集和官方 baseline 配置仍待取得。

## 1. 已确认的比赛要求

比赛为“昇腾 Jev 开源决策模型适配与调优大赛”，包含部署适配和性能调优两条独立赛道。Clef-27B 在调优赛题列表中列为多模态类，规模约 27–28B。可自由认领，同一模型允许多个团队提交。官方模型清单由飞书表格维护。[比赛介绍](https://competition.gitcode.com/competition/2119684851993116674/intro) · [赛题发布](https://competition.gitcode.com/competition/2119684851993116674/publish) · [模型清单](https://vj22qe6kl3.feishu.cn/wiki/HKn8wk4QGiuNiAkOe9Lcfroyni0?from=from_copylink&sheet=b6dc86)

| 项目 | 已确认要求 | 对本工程的影响 |
| --- | --- | --- |
| 适配赛道 | 在昇腾 NPU 稳定运行，检查提交完整性、跑通情况和精度 | 保留可复现的输入、输出和 NPU 调用截图 |
| 调优赛道 | 精度相对 Baseline 波动不超过 2 个百分点，再比较吞吐和延迟 | 先固定 baseline 输出、软件版本、数据版本和测量配置 |
| 服务协议 | `start.sh` 暴露 `POST /v1/systemone`，由 EvalScope 正式评测调用 | 需要完整 System One 服务和决策输出 |
| 推理框架 | `inference.py` 调用 vLLM-Ascend 完成完整决策路径 | 当前骨架须接入主干、联合决策头和服务层 |
| 提交形式 | 平台拉取代码并自动校验，赛题页另要求向 `Ascend/model-agent` 提交 PR | 最终按赛题页规定的入口、目录和仓库提交 |
| 调优产物 | 性能、精度、原始日志、复现说明和截图 | 现有 CSV 表头须填入实测结果，截图须来自真实运行 |

上述要求可在[官方正文 API](https://web-api.gitcode.com/aihub/api/v1/activity/2119684851993116674/content)及本地[脱敏证据摘要](evidence/competition-2026-11-12.json)复核。介绍正文最后更新时间为 2026-11-12 12:03:33（北京时间）。公开规则给出了 2PP 门槛，精度指标名称、性能加权公式、基准硬件和具体 workload 尚待赛题细则确认。

调优赛道目录如下，现有仓库采用这一结构。

```text
Clef-27B/
├── README.md
├── start.sh
├── inference.py
├── results.csv
├── metrics.csv
├── logs/
├── optimization.md
├── result.png
└── npu_device_call.png
```

适配赛道还要求小写 `readme.md` 和 `assets/agent_workflow.png`、`assets/npu_device_call.png`、`assets/model_result.png`。如同时参赛，需要补齐这一组产物。[赛题发布要求](https://competition.gitcode.com/competition/2119684851993116674/publish)

时间口径存在页面内差异。正文关键节点写报名截止为 11 月 21 日 24:00，元数据时间线写 11 月 22 日。作品截止分别为适配赛道 11 月 22 日 24:00、调优赛道 11 月 26 日 24:00。赛题页显示调优从 11 月 12 日开始，介绍正文的命题区间从 11 月 13 日开始。执行前按平台最新页面核对。[官方介绍](https://competition.gitcode.com/competition/2119684851993116674/intro)

## 2. 正式评测数据的确认进度

赛题页展开 Clef-27B 后，当前可见详情只有“多模态类（约 27–28B）”。已读取的公开赛事正文包含 EvalScope、接口和目录要求，尚未提供正式数据集 ID、split、revision、样本数、baseline 数值及压测参数。匿名读取 `ascend_problem_publish`、`subject-traces` 和 `problem_intro_files` API 均返回 HTTP 401，浏览器已能显示赛题列表。后续需要从已报名的赛题资源或正式发布的附件取得评测配置。[赛题发布](https://competition.gitcode.com/competition/2119684851993116674/publish)

指定提交仓库的默认分支于本次检查指向 `c75448c13fb4a35e7ec0883ce48d05c23eb5b7fa`。按路径检索 `Clef`、`systemone` 和 `evalscope`，未定位到 Clef 专用评测入口。该检查覆盖目录树及根 README，正式评测细节仍以赛事资源为准。[指定仓库](https://gitcode.com/Ascend/model-agent)

## 3. 可以直接用于工程验证的公开数据

以下数据集均有公开来源，赛事采用情况待确认。

| 数据集 | 规模与题型 | 用途 | 获取入口 |
| --- | --- | --- | --- |
| OpenDecisions | `test` 2,352 条，1,178 Choice、586 Noul、588 Score，多语言文本 | 第一轮三类决策精度与概率验证，EvalScope 已有适配 | [ModelScope 镜像](https://modelscope.cn/datasets/evalscope/OpenDecisions/summary) · [原始数据](https://huggingface.co/datasets/jgeuter/OpenDecisions) |
| JevBench 公开子集 | `test` 231 条，139 Choice、74 Noul、18 Score | 较小的流程验证集，保留公开题评分规则 | [ModelScope 镜像](https://modelscope.cn/datasets/evalscope/jevbench/summary) · [固定原始快照](https://github.com/fstandhartinger/jevbench/tree/609a664e81f46e03731600bd7bd9f1b7a61257c5) |
| Typed Decisions | 4 个工作流，每个 300 train、100 test；合计 test 400 个 case、2,000 个决策 | 每条请求同时回答 5 个问题，适合检查联合决策头与跨字段交互 | [LocalLLaMA/typed-decisions](https://huggingface.co/datasets/LocalLLaMA/typed-decisions) |
| safeagents judge bench v2 | `test` 661 条，每条同时包含 Noul 和 7 选 1 Choice | 补充多字段输出和概率回归，包含已发布 Clef 结果 | [harness-hub/safeagents-judge-bench v2](https://huggingface.co/datasets/harness-hub/safeagents-judge-bench/tree/v2) |
| STS-B Decision | 连续相似度标准分数，Score 任务 | 专门检查按概率计算的期望分数与 MAE | [EvalScope 文档](https://github.com/modelscope/evalscope/blob/3d39bab09c1d63095490a8b0f4a6dcbcc5ea843e/docs/zh/benchmarks/stsb.md) |

OpenDecisions 的数据许可为 CC BY-SA 4.0，JevBench 公开记录采用 MIT。Typed Decisions 与 safeagents v2 的数据卡标为 Apache-2.0，Hugging Face 元数据均为 `private=false`、`gated=false`。这两个仓库的元数据和 README 可匿名读取，下载时仍需检查目标服务器网络连通性。首次下载应保存 revision 和文件校验值。[OpenDecisions 评测说明](https://github.com/modelscope/evalscope/blob/3d39bab09c1d63095490a8b0f4a6dcbcc5ea843e/docs/zh/benchmarks/open_decisions.md) · [JevBench 评测说明](https://github.com/modelscope/evalscope/blob/3d39bab09c1d63095490a8b0f4a6dcbcc5ea843e/docs/zh/benchmarks/jevbench.md)

本次核实的 Hugging Face revision 如下。

| 数据集 | revision |
| --- | --- |
| `LocalLLaMA/typed-decisions` | `cc1a2e795f5160d417d4d7131a71d1ab60ea33ca` |
| `harness-hub/safeagents-judge-bench` 的 `v2` | `037bfaaee6039d339a24f664957999bc9ac8461a` |

Typed Decisions 的 `state`、`questions`、`gold` 字段是 JSON 字符串，发送前解析前两者即可组成请求，`gold` 仅供评分。四个工作流为 `agent_trace_observability`、`customer_service`、`invoice_processing` 和 `security_incidents`，`all` 配置汇总四者。保持每个 case 的 5 个问题一起送入模型，才能检验其联合计算行为。[数据说明](https://huggingface.co/datasets/LocalLLaMA/typed-decisions)

模型下载包里的两个 `.eval_results/*.yaml` 提供已发表指标和数据来源，可用来追溯基准。它们只含指标元数据，原始样本需要从上表入口另行取得。

| 模型自带文件 | 已发表结果 | 来源环境 |
| --- | --- | --- |
| `.eval_results/Cloudflare__clef.yaml` | Typed Decisions accuracy 0.7275，KL 0.1907，Brier 0.1032，ECE 0.0253 | 2026-10-16 的第三方评测记录 |
| `.eval_results/safeagents-judge-bench.yaml` | balanced accuracy 0.808，AUROC 0.993，false block rate 0.384，category macro-F1 0.563 | 2026-11-10，NVIDIA A100 80GB |

这些数值可用于回归检查的背景参考，竞赛的 2PP 判断需要使用主办方指定 baseline 和相同评测协议。[Typed Decisions 指标文件](https://modelscope.cn/models/Cloudflare/clef/resolve/master/.eval_results/Cloudflare__clef.yaml) · [safeagents 指标文件](https://modelscope.cn/models/Cloudflare/clef/resolve/master/.eval_results/safeagents-judge-bench.yaml)

## 4. 服务器已经具备的共享数据目录

远端 `/workspace/shared_assets/datasets/` 已有真实数据文件。平台 README 将它们说明为只读公共资源，用于训练和评测。下表大小为排除 `.cache`、`.git` 后的文件字节数汇总，采用十进制 GB/MB。此次完成文件树扫描和部分 README 检查，尚未逐文件校验、解压或运行数据加载器。[数据与网络采集证据](evidence/data-and-network-2026-11-12.json)

| 远端相对路径 | 已确认内容 | baseline 复用方向与所需处理 |
| --- | --- | --- |
| `cais/mmlu` | 180 个文件，约 270 MB，包含 175 个 Parquet 及 `data.tar` | 文本 Choice，固定学科、split、few-shot 设置后接 EvalScope |
| `MMMU___mmmu/` | 121 个文件，约 3.68 GB，包含 91 个 Arrow 和 30 个学科元数据 JSON | 图片选择题，核对图像载荷与题目类型，优先使用带公开标签的 split |
| `lmms-lab/Video-MME` | 27 个文件，约 101 GB，包含 21 个 ZIP 和 1 个 Parquet，数据卡记载 test 2,700 条 | 视频 Choice，按需解压到工程数据目录，固定采帧、字幕和分辨率 |
| `MTEB/Daily-Omni` | 14 个文件，约 5.16 GB，包含 10 个 Parquet，数据卡记载 test 1,196 条，含 video、audio、question、candidates、answer | 后续多模态样例候选，筛选模型支持的输入和可评分任务 |
| `LLaVA-Instruct-150K` | 8 个文件，约 1.62 GB，数据文件为 6 个 JSON，当前目录扫描未见图像文件 | 可先检查文本与长度分布，图片验证需补齐对应媒体并定义决策题 |
| `tatsu-lab/alpaca` | 3 个文件，约 24.3 MB，含 1 个 Parquet，数据卡记载 52,002 条 train | 文本输入素材，为 System One 定义问题和候选项，遵循数据卡 CC BY-NC 4.0 |
| `tatsu-lab/alpaca_eval` | 12 个文件，约 76.3 MB，含样本、模型输出和评审 JSON | 文本输入素材，定义适合决策模型的评分任务，数据卡为 CC BY-NC 4.0 |

服务器上还存在 `CowboyZ/seed-tts-eval`，包含 7 个文件、约 2.54 GB。其语音合成评测可以留待其他模型使用。任何转换结果、解压文件和索引应写入本工程的数据目录，共享源文件按平台要求只读使用。平台将这批资源说明为公共数据，赛事正式评测采用情况仍待确认。

当前容器可写盘约剩余 151 GiB，Video-MME 压缩资源本身约 94 GiB，完整解压所需空间还要另行估算，并给环境安装和编译保留空间。第一轮视频验证宜按选定样本提取所需媒体。上述扫描验证了文件存在和大小，Arrow、Parquet 及视频的逐样本解码仍待执行。[磁盘证据](evidence/environment-2026-11-12.json)

远端本次访问 PyPI 的 torch-npu 索引、ModelScope 小文件均得到 HTTP 200，Hugging Face 数据集 API 发生连接重置，GitHub raw 文件读取超时。本地已能读取 Hugging Face 数据卡和元数据。后续公开数据下载可先尝试远端 ModelScope 镜像，也可在本地固定 revision 下载后传入服务器，随后验证 SHA-256 并记录传输后的文件清单。[网络采集证据](evidence/data-and-network-2026-11-12.json)

## 5. EvalScope 接入与最小验证顺序

本次读取 EvalScope 提交 `3d39bab09c1d63095490a8b0f4a6dcbcc5ea843e` 的文档，已提供 `eval_type='systemone_api'`、原生三题型数据集和 `evalscope perf --api systemone`。该文档声明当前 System One 评测适配支持文本决策，图片和视频需要另做服务调用检查或使用之后提供的赛事适配器。[System One 文档](https://github.com/modelscope/evalscope/blob/3d39bab09c1d63095490a8b0f4a6dcbcc5ea843e/docs/zh/user_guides/systemone.md) · [压测示例](https://github.com/modelscope/evalscope/blob/3d39bab09c1d63095490a8b0f4a6dcbcc5ea843e/docs/zh/best_practice/decision_evaluation.md)

1. 使用模型 README 的短文本请求，分别覆盖 Choice、Noul、Score，再发一条混合三题型的请求。确认概率、期望分数、问题 ID 和 `usage` 正常。
2. 取 JevBench 或 OpenDecisions 的少量样本检查 EvalScope 与服务连通性，并核对所抽样本确实包含三种题型。
3. 跑完整 JevBench/OpenDecisions，保存原始请求、响应、逐题评分和汇总报告。
4. 加入 Typed Decisions 的完整 400 case test，保留每条请求中的联合问题，检查多字段、变长请求及概率漂移。
5. 使用相同输入和软件环境做 TP1/TP2 对照，再扩展并发、图片与视频。
6. 正式赛题配置取得后，按其数据、指标和压测定义补齐竞赛 baseline。

以下命令是服务跑通后的工程验证模板，尚未在本机执行。模型名 `clef` 与[上游启动示例](upstream-npu-baseline.md)中的 `--served-model-name clef` 保持一致。`api_url` 使用 `/v1` 基地址，由 EvalScope 追加 `/systemone`。

```bash
evalscope eval \
  --model clef \
  --eval-type systemone_api \
  --api-url http://127.0.0.1:8000/v1 \
  --api-key EMPTY \
  --datasets jevbench \
  --eval-batch-size 1 \
  --limit 12

evalscope perf \
  --api systemone \
  --model clef \
  --url http://127.0.0.1:8000/v1 \
  --api-key EMPTY \
  --dataset open_decisions \
  --parallel 1 \
  --number 24 \
  --warmup-num 2 \
  --no-stream
```

正式精度比较时去掉 `--limit` 并固定数据 revision。EvalScope 的 `eval_batch_size` 控制并发请求数，服务端实际 batching 由推理引擎控制。性能记录至少保留成功率、成功吞吐、请求延迟 p50/p95、输入 token 数、题目数、候选项数、预热次数和设备配置，以便解释相同模型在不同 workload 下的差异。[EvalScope 配置说明](https://github.com/modelscope/evalscope/blob/3d39bab09c1d63095490a8b0f4a6dcbcc5ea843e/docs/zh/user_guides/systemone.md)

## 6. 构建 baseline 前待取得的信息

- 正式评测数据集的 ID、下载地址、访问条件、revision、split 和样本数，以及多模态占比。
- 官方 baseline 的代码提交、完整依赖、设备型号与设备数，以及真实精度和性能结果。
- 2PP 门槛对应哪个指标、如何汇总多数据集、如何处理失败或超时样本。
- 性能评分的吞吐与延迟定义、权重、并发、输入长度、预热、重复次数和时间窗口。
- `results.csv`、`metrics.csv` 的正式列名和导出规范，以及提交 PR 的目标分支与目录。

本轮已经取得可用于文本功能、联合决策及后续多模态探索的数据入口。环境安装、样本下载、格式转换、推理执行和评分仍属于下一轮 baseline 实现工作。
