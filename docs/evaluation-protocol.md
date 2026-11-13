# Clef-27B 统一评测口径

决策日期：2026-11-13，北京时间。用户确定本工程主评测采用 Cloudflare 已发表结果对应的 **Decision Index 0.2.1** 和 **Typesafe WorkflowEvals**。两者均为包含多个测试任务、数据和评分器的评测套件。[发布方结果](https://huggingface.co/Cloudflare/clef#results) · [发布文章](https://blog.cloudflare.com/clef-decision-models/)

## 套件、子项与结果的关系

```text
Clef-27B 主评测
├── Decision Index 0.2.1
│   ├── BFCL：工具调用决策，case exact accuracy
│   ├── API-Bank：API 选择等决策，accuracy
│   ├── BANKING77 / CLINC150+OOS：意图分类，macro-F1
│   └── 其他子项：保留所选 edition 的任务、评分与聚合
└── Typesafe WorkflowEvals
    ├── 发票处理：exact_actions，另列 primary_action
    ├── 客服：exact_actions
    ├── 安全事件：exact_actions
    └── Agent 运行审查：primary_action
```

BFCL、API-Bank 是 Decision Index 内的具体 benchmark。每个子项有自己的样本和指标；套件按固定 edition 的规则产生汇总。Decision Index 使用经适配的决策题格式，引用成绩时同时注明套件 edition 与子项指标。[Decision Index 说明](https://github.com/apolinario/decision-index) · [任务与样本来源](https://github.com/apolinario/decision-index/blob/main/docs/suite.md)

WorkflowEvals 将业务输入和决策问题组成工作流，使用共识参考标签评分。`exact_actions` 检查动作集合及参数，`primary_action` 检查主动作及参数。跨工作流汇总采用官方脚本支持且四工作流齐全的同名指标，沿用工作流等权规则，并保留四工作流的独立结果。[WorkflowEvals 说明](https://github.com/typesafe-ai/WorkflowEvals)

## 已发表结果与本工程状态

以下为 Cloudflare 已发表的 Clef-27B 结果，单位为百分比；本工程对应的两套评测均待接入与实际运行。

| 套件 | 子项 | 指标 | Cloudflare 已发表值 | 本工程 A3 |
| --- | --- | --- | ---: | --- |
| Decision Index 0.2.1 | BFCL | case exact accuracy | 98.47 | 待运行 |
| Decision Index 0.2.1 | API-Bank | accuracy | 91.93 | 待运行 |
| Decision Index 0.2.1 | BANKING77 | macro-F1 | 94.20 | 待运行 |
| Decision Index 0.2.1 | CLINC150+OOS | macro-F1 | 97.43 | 待运行 |
| WorkflowEvals | 发票处理 | exact_actions | 64.7 | 待运行 |
| WorkflowEvals | 发票处理 | primary_action | 86.2 | 待运行 |
| WorkflowEvals | 客服 | exact_actions | 76.3 | 待运行 |
| WorkflowEvals | 安全事件 | exact_actions | 62.9 | 待运行 |
| WorkflowEvals | Agent 运行审查 | primary_action | 68.5 | 待运行 |

上表列出部分 Decision Index 子项用于解释层级关系；完整主评测覆盖 0.2.1 公开套件。原始数值来自[Cloudflare 发布文章](https://blog.cloudflare.com/clef-decision-models/)与[模型卡](https://huggingface.co/Cloudflare/clef#results)。工作流比较还需确认相同数据 revision、策略和 case cohort，保留发表值与本次实际配置的对应记录。

Cloudflare 另公布套件请求延迟 median 209.3 ms、p95 238.6 ms。此类性能汇总独立于子项精度保存，当前来源的硬件、并发和长度等完整条件待补齐。本工程 A3 吞吐、延迟和 HBM 按实际 workload 与测量范围报告。

## 执行与固定条件

1. **模型与服务**：固定完整模型 manifest、主干和联合头哈希、软件提交及补丁；记录 dtype、A3 设备数、TP 和服务参数。沿用 `POST /v1/systemone`，服务模型名为 `clef`。
2. **Decision Index**：显式选择 `--edition 0.2.1`，固定 harness 提交、原始数据 revision、生成样本与排除列表哈希。使用原生 HTTP 引擎，`base_url` 指向服务根地址，发送 `model=clef`；数据按公开配方重建，并完成各来源要求的访问条件。[HTTP 实现](https://github.com/apolinario/decision-index/blob/main/decision_index/engines/http.py)
3. **WorkflowEvals**：固定四工作流的代码、数据 revision、策略、case cohort 与 `consensus` 参考标签。当前客户端对 TypeSafe 模型名有限定，接入时增加 Clef HTTP 支持并记录补丁，保留问题分组 `all`、业务逻辑及原评分器。评测客户端采用独立 Python 3.13+ 环境，通过 HTTP 连接模型服务。[模型限制与问题分组](https://github.com/typesafe-ai/WorkflowEvals/blob/main/core/providers.py) · [客户端](https://github.com/typesafe-ai/WorkflowEvals/blob/main/core/clients.py)
4. **输入与计分**：保存原始 state/questions、候选顺序、编码长度、上下文上限及截断信息；按原 harness 规则统计超长、拒绝、超时和错误，逐子项列出覆盖率。
5. **性能**：记录客户端位置、并发、输入长度、字段数、候选数、媒体设置、预热与重复次数，以及 HTTP/服务/设备各自计时范围。TP1/TP2 优化前后使用相同数据和测量条件。
6. **交付证据**：保存逐请求输出、原生评分、聚合报告和版本清单；全量执行与评分结束后更新完成状态。所选公开套件的完成状态与榜单维护者的私有测试分别记录。

## 既有结果与赛事口径

2026-11-12 已完成的 JevBench 公开文本 231 条、OpenDecisions 2,352 条及 Typed Decisions 400 case / 2,000 决策，继续作为工程回归证据。固定 307-token 压测、红蓝图片和视频帧序素材用于性能与功能自检，保留其原始测量范围。[首轮工程 baseline](baseline-2026-11-12.md) · [profiling](profiling-2026-11-12.md)

后续主评测完成标准为：**完成 Decision Index 0.2.1 公开评测与 WorkflowEvals 四工作流评测，并保存逐请求结果、原生评分与版本证据。** 赛事正式提交按主办方指定的数据、指标及性能协议执行，主办方参数的取得状态见[赛事资料](competition-and-datasets.md)。
