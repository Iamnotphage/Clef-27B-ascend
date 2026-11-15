# Clef-27B baseline 调查与执行计划

更新：2026-11-14（北京时间）。完整环境与首轮工程 baseline 已完成；新增端到端 profiling 覆盖各900请求的TP1/TP2对照、14窗口/24 rank trace、全输入矩阵、冷/热多模态和batching。比赛质量交付已完成 OpenDecisions 1.0.1 全量 2,352 条与 ERQA 官方 400 题，最终性能配置为 seqs1/B1。CPU 独立参考完成两套数据的全量输入等价验收与部分逐题对照（OpenDecisions 154/2352），按用户要求终止。2026-11-15 完成 GDN 决策专用 prefill 路径并接入 S1 默认启用：全量 2,352 题逐位不变，同进程交错 A/B 6/6 对更快（p95 中位 −3.80%），正常入口 S1 P95 213.56 ms；ERQA 400 ON/OFF 逐题一致。提交入口 profile 显示 MLP 占 31.6%、TP2 allreduce 20.1%、GDN 主核 10.4%、布局 6.6%。[profiling实测](profiling-2026-11-12.md) · [实测 baseline](baseline-2026-11-12.md) · [OpenDecisions 全量](opendecisions-baseline-2026-11-13.md) · [ERQA 全量](erqa-baseline-2026-11-14.md) · [CPU 参考对照](cpu-npu-reference-2026-11-14.md) · [GDN 决策专用路径](gdn-decision-prefill-2026-11-15.md)

比赛质量验收采用 **OpenDecisions + ERQA**，性能验收采用 **S1 单并发 P95 / S2 最大有效 decisions/s**；Decision Index 与 Typesafe WorkflowEvals 保留为发布方对照记录。[比赛提交规范](submission-rules-2026-11-13.md)

当前机器已切换为 `root@199.88.54.221`，用户选择沿用新机的 CANN 9.2.1 并配套 PyTorch 2.10。当前状态、SSH 模板、环境查看命令及分阶段实施顺序见新机器执行计划。[新机器执行计划](new-machine-plan.md)

## 调查结论

跑通 baseline 的主要工作是**配套运行环境、接入现成服务、准备决策评测数据、验证 A3 输出**。上游已经实现模型注册、独立联合决策头加载和 `/v1/systemone`，可作为首轮复用路径。

- 实测为两个 `Ascend910_9382` 计算单元，各 64 GiB HBM；当前实例拥有 32 核等价 CPU 时间配额和 128 GiB RAM。
- 55.0 GB 模型已在服务器，独立决策头和模型参考实现齐全；本地入口仍为骨架。
- 新机器已配备 CANN Toolkit、A3 Ops 和 ATB 9.2.1。后续创建 Python 3.12 环境并安装 PyTorch 2.10 配套栈，CMake 更新至至少 3.26.1。系统 Python 3.10.12 的 pip 和推理依赖待配置。
- vLLM-Ascend 的 Clef 支持已合入；已找到配套 vLLM 提交。公开实测覆盖 950DT 单设备文本/图片；A3 和 TP2 需要本项目补测。
- EvalScope 已有 `systemone_api` 和原生决策数据适配；服务器另有通用文本、多模态数据。比赛最终评测集与官方数值基线仍待明确。

## 文档导航

| 文档 | 内容 |
| --- | --- |
| [新机器执行计划](new-machine-plan.md) | 当前目标机器、实测环境、环境检查命令与后续安装顺序 |
| [实际安装与验收](installation-2026-11-12.md) | Python 与 wheel 哈希、真实版本、双卡误差、依赖补丁、复现命令与构建日志 |
| [实测 baseline](baseline-2026-11-12.md) | 完整加载、TP1/TP2 精度性能、完整公开评测、HCCL 和多模态证据 |
| [统一评测口径](evaluation-protocol.md) | Decision Index 0.2.1 子项、WorkflowEvals 四工作流、同协议对照与执行状态 |
| [OpenDecisions 全量](opendecisions-baseline-2026-11-13.md) | 2,352 条原始概率、作者评分复核、EvalScope 固定源码与复现命令 |
| [ERQA 全量](erqa-baseline-2026-11-14.md) | 官方 400 题数据清单、图文协议、图像预算、8 类分项、离线复核与适配补丁 |
| [CPU 参考对照](cpu-npu-reference-2026-11-14.md) | 发布方参考实现的 CPU BF16 独立前向、全量输入等价、逐题概率差与终止范围 |
| [GDN 决策专用路径](gdn-decision-prefill-2026-11-15.md) | 纯 prefill 特化：状态与反向中间量裁剪、guard 设计、6 对 A/B 与服务恢复 |
| [提交入口 profile](profile-entry-2026-11-15.md) | 优化后设备预算：MLP、TP2 allreduce、GDN 主核与布局、attention |
| [MLP W8A16 与平滑](mlp-w8a16-2026-11-15.md) | W8A16 候选全量结果、激活平滑仿真与 MLP 量化收益上限 |
| [seqs4 批处理](batching-seqs4-2026-11-15.md) | C1–C16 吞吐/质量扫描、S1 不变性与合批质量边界 |
| [当前进展总览](current-progress-2026-11-15.md) | 供外部专家评审的自包含状态：结果、预算、已排除方向与开放问题 |
| [第四轮复核](tuning4-2026-11-14.md) | seqs1/B1 最终配置、S1/S2、合批精度根因与恢复记录 |
| [端到端 profiling](profiling-2026-11-12.md) | 稳定TP对照、Top20、CPU/NPU/通信分解、全矩阵、内存、优化排序和服务恢复 |
| [环境调查](environment.md) | 实测资源配额、NPU 映射、软件缺项、模型资产、网络与磁盘约束 |
| [上游与 NPU 适配](upstream-npu-baseline.md) | 固定提交、版本组合、模型语义、启动候选命令、三类请求与 A3/TP2 风险 |
| [比赛与数据集](competition-and-datasets.md) | 比赛交付要求、评分已知项、服务器现有数据、决策数据源与评测接入 |
| [原始证据](evidence/) | 脱敏环境快照、数据统计和比赛页面证据 |

## baseline 分阶段工作清单

| 顺序 | 工作 | 完成标准 | 当前状态 |
| --- | --- | --- | --- |
| 0 | 固定模型、源码和数据版本 | manifest、源码 SHA、数据 revision 与 SHA-256 可追溯 | 已完成，运行和构建快照已保存 |
| 1 | 建立隔离运行环境 | ARM64 / A3 一致版本栈和实际导入 | 完整环境与 `pip check` 通过 |
| 2 | 基础 NPU 运算验证 | 双卡 FP16 / BF16 2048 matmul、同步及有限值检查 | 四项通过，误差与 CPU FP32 对应 |
| 3 | 完整 Clef 加载 | 主干、视觉塔、LMHead 和独立联合头有加载审计 | TP1/TP2 均通过，head 122/122 |
| 4 | 项目入口 | MODEL_DIR / PYTHON_BIN、健康检查及原生服务 | 已接通，`POST /v1/systemone` |
| 5 | 三类与多字段自检 | 概率、候选重排、token/span、变长请求 | API 5/5，编码 5/5，七个决策一致 |
| 6 | 精度和公开回归评测 | CPU 参考对齐与逐题结果 | TP1/TP2 对齐通过，2,983 请求执行错误 0 |
| 7 | 性能基线 | 固定长度/问题/候选/并发/预热，记录延迟吞吐与采样 HBM | TP1/TP2 各 30 次通过，CSV 与 raw 已保存 |
| 8 | TP2 与多模态 | HCCL、TP2、图片/视频及长输入 | HCCL 与 MM 4/4 通过，4096 完整输入无截断；更高并发留作扩展 |
| 9 | 比赛质量评测 | OpenDecisions 1.0.1 全量 2,352 条与 ERQA 官方 400 题 | 均已完成；Decision Index 与 WorkflowEvals 保留发布方对照 |
| 10 | 比赛交付整理 | 入口、说明、CSV、图和日志 | OpenDecisions 与 ERQA 结果、复现命令和轻量证据已归档 |

阶段 5 是“最小功能跑通”的完成点；阶段 7 形成可复现的首轮工程 baseline；阶段 9 补齐与 Cloudflare 发布成绩对应的主评测。赛题完整多模态路径和正式指标覆盖继续按主办方要求补齐。

## 建议首轮配置和排障顺序

首轮实测采用 **BF16、TP1、eager、batch 1、max model len 1024、纯文本**，关闭 prefix caching 和 chunked prefill。TP2 使用相同短请求比较；完整公开集和多模态阶段扩展为 TP2/4096。实际启动参数和接口样例见[实测 baseline](baseline-2026-11-12.md)。

1. 先解决环境配套。优先评估平台是否提供含指定栈的 A3 / ARM64 镜像，并确认工作目录持久化；也可使用当前容器内的独立安装前缀。保留现有系统 CANN，记录实际采用的构建方式。
2. 模型加载失败时，检查架构识别、vLLM 插件注册、joint head 文件路径和权重映射。
3. A3 算子失败时，按日志定位 Gated DeltaNet / causal conv、Attention / RoPE、联合决策头的 MultiheadAttention / LayerNorm / FFN。优先复用 A3 已有算子实现，确认缺口后再补兼容。
4. 内存不足时，核对实际 HBM 峰值、上下文上限、最大 batch 和视觉预算；调整后再评估 TP2。128 GiB 主机内存也需同步监测。
5. 输出差异时，先核对请求模板、候选顺序、`joint` 参数语义、score 等级起点、yes/no 概率和舍入规则，再定位数值误差。参考实现与 vLLM 的完整 JSON 结构存在差异，对比时提取共同语义字段。

## 主评测与工程回归数据安排

| 层次 | 数据 | 用途 |
| --- | --- | --- |
| 主评测：决策任务 | Decision Index 0.2.1，包含 BFCL、API-Bank 等子项 | 按发布方采用的任务协议、数据版本和原生指标复现，待接入与运行 |
| 主评测：业务工作流 | Typesafe WorkflowEvals 的发票、客服、安全事件、Agent 运行审查 | 保留官方工作流、参考标签与评分逻辑，待接入与运行 |
| 最小自检 | 固定的三类决策 + 多字段 + 不同长度请求 | 验证服务协议、联合决策头和概率语义 |
| 首轮公开回归 | EvalScope 的 OpenDecisions / JevBench 小样本后扩至完整集 | 保存已完成的工程回归指标 |
| 联合决策回归 | Typed Decisions；SafeAgents Judge Bench v2 | 多字段分布、概率校准与安全决策场景 |
| 服务器现有文本数据 | MMLU 的带标签 split | 转为 `choice` 并固定模板，验证较大规模文本任务 |
| 多模态扩展 | MMMU 带标签 split、Daily-Omni、Video-MME 小样本 | 图片/视频输入、视觉 token 和 span 对齐；按实际媒体可用性筛选 |

两套主评测与历史回归数据的范围、指标和状态按[统一评测口径](evaluation-protocol.md)记录。比赛专用正式测试包、基准数值及聚合权重按主办方发布的协议对齐。首轮回归数据规模、下载源、版本和转换限制详见[数据集文档](competition-and-datasets.md)。

## 本轮执行边界

本轮已完成环境安装、固定源码及兼容补丁、完整模型服务、CPU 精度对齐、固定请求性能、首轮公开文本回归评测与多模态自检。保留系统 CANN 9.2.1；全部指标取自实际报告。Decision Index 0.2.1 与 WorkflowEvals 主评测待接入与运行。比赛专用输入、官方基线与最终交付规范继续按主办方协议对齐。
