# 2026-11-13 比赛提交与评测规则整理

来源为用户于 2026-11-13 在本会话粘贴的“提交内容”及两份 CSV 字段说明。本文保存该版本的要求、单位和待明确项；本轮整理依据为会话原文，官网对应版本尚待独立核对。原文注明“当前交付物非最终版本，后续会进一步完善相关内容，请持续关注官网”。

同日用户另行补充资源规则：**允许并行，限用 A3 单物理卡内两个逻辑 NPU，禁止跨卡**。该限制按本会话授权执行，详见第 7 节。

## 1. 赛事验收范围

| 范围 | 本轮规则要求 |
| --- | --- |
| 文本质量评测 | 提交完整 OpenDecisions 结果，包含 Calibration、Intelligence、Capability，并保留 accuracy、score_mae、样本数及执行正确率等记录 |
| 图文质量评测 | 多模态型号同时提交 ERQA 和 OpenDecisions；ERQA 提供图文准确率及分项结果 |
| 性能评测 | S1 单并发 P95 端到端时延；S2 最大有效吞吐（decisions/s）及对应并发；完整并发扫描、逐请求记录及优化前后对比 |
| 环境与证据 | 原文要求昇腾 910C 环境，记录模型精度、并行配置、缓存设置、测试数据、统计方式及资源占用 |
| 结果复现 | 每个模型、每个数据集一行，填写提交版本的实际结果；最终效果由主办方作品复现确定 |

Clef-27B 包含视觉分支，按本轮多模态要求准备 **ERQA + OpenDecisions** 的质量交付，并包含图像处理与视觉推理适配。Decision Index 和 WorkflowEvals 的已有评测继续保留其发布方对照用途，赛事验收按照本轮指定的数据与指标组织。

## 2. 统一入口与服务接口

统一启动命令：

```bash
bash start.sh --model-dir xxx --output-dir xxx
```

| 文件或接口 | 职责与返回要求 |
| --- | --- |
| `start.sh` | 接收 `--model-dir` 和 `--output-dir`，将参数传递给 `inference.py`；模型目录同时支持相对路径、绝对路径 |
| `inference.py` | 完成模型准备、加载、补丁初始化和服务启动，默认监听 `http://127.0.0.1:18143` |
| `GET /health` | 就绪检查，返回 `status=ready`、`model`、`revision` |
| `GET /v1/decide/info` | 返回模型身份、任务能力、协议版本及缓存配置 |
| `POST /v1/systemone` | 标准文本评测接口，接收题目并返回决策与完整概率 |
| `POST /v1/decide` | 图文评测及 Demo 交互接口 |

提交的完整代码还应覆盖框架配置、Benchmark 执行脚本、Demo 启动脚本、依赖及环境配置；实际文件名和调用方式写入项目 README。`--output-dir` 内的具体文件布局与接口请求/响应细节按最终协议补齐。

## 3. 提交目录与文件职责

```text
Clef-27B/
├── start.sh
├── inference.py
├── README.md
├── metrics.csv
├── results.csv
├── result.png
├── npu_device_call.png
├── logs/
├── optimization.md
├── src/                    # 可选：框架优化代码或补丁
└── model_agent_pr.md        # 可选：ModelAgent 贡献说明
```

| 交付项 | 文件职责 |
| --- | --- |
| `README.md` | 调优项目介绍、环境与依赖、服务和 Demo 用法、质量与性能评测命令、复现入口 |
| `src/`（可选） | 框架优化代码、模型适配补丁；图文型号包含图像处理和视觉推理适配 |
| `metrics.csv` | 模型与场景的质量结果，关联样本数、执行正确率、基线值、优化值、准入阈值和达标结论 |
| `results.csv` | S1/S2 性能汇总、实际推理配置及优化前后对照；完整扫描和逐点记录随日志提供 |
| `result.png` | 真实运行结果截图，清晰展示模型标识、版本、主要评测结果及运行环境 |
| `npu_device_call.png` | 测试期间真实 NPU 状态截图，展示型号、编号、进程、HBM 占用及利用率 |
| `logs/` | 服务启动、就绪检查、接口调用、质量与性能测试的命令、配置、原始输出和汇总 |
| `optimization.md` | 原始瓶颈证据、优化原理、具体改动、每项改动的质量/性能变化、资源要求、计时范围、缓存与特殊配置、完整复现步骤 |
| `model_agent_pr.md`（可选） | 向 AutomGit/AtomGit 社区 ModelAgent 项目贡献的 PR 链接与简介，内容可为模型优化沉淀的 skill 或 ModelAgent 优化点；项目名称拼写及提交地址按最终发布页面核对 |

## 4. 质量结果：`metrics.csv`

### 4.1 字段明细表给出的宽表

```csv
model,dataset,accuracy,score_mae,calibration,intelligence,capability
```

| 字段 | 描述与单位 |
| --- | --- |
| `model` | 提交作品使用的模型名称或模型仓库标识 |
| `dataset` | 文本型号为 `OpenDecisions`；多模态型号分别提供 `ERQA`、`OpenDecisions` 两个评测集的结果 |
| `accuracy` | 答案准确率，范围 **0～1**，越高越好 |
| `score_mae` | Score 期望等级与标准等级的平均绝对误差，越低越好；原文未给固定上限 |
| `calibration` | 概率校准分，范围 **0～100 分** |
| `intelligence` | 决策能力分，按题型、难度加权并校正机会水平，范围 **0～100 分** |
| `capability` | 综合质量分，`(calibration + intelligence) / 2`；两输入采用相同的 0～100 分量纲 |

依原文“每个模型、每个数据集一行”组织提交版本的汇总。ERQA 的分项结果、质量字段适用性与缺失值表示，随最终模板确认并保留原生报告。

### 4.2 总表同时给出的长表表头

原文在提交总表中又给出以下表头：

```csv
模型名称,评测场景,指标名称,执行正确率（%）,指标值
```

该表为五列长表，4.1 节为七列宽表；总表还要求记录样本数、基线值、优化值、准入阈值与达标结论，而两个表头均未完整包含这些信息。两种要求在此并列保留，最终 `metrics.csv` 格式待主办方模板确认。准备期间在原始质量报告中保留全部字段、单位和分母，便于按最终模板导出。

执行正确率的分子、分母、超时/拒绝处理及其与答案准确率的关系，按最终协议明确。指标记录保留原始单位；0～1 的概率/准确率与 0～100 的质量分分别标注。

## 5. 性能结果：`results.csv`

### 5.1 汇总字段

原文字段明细给出以下十二列：

```csv
model,dataset,image,resources,precision,tensor_parallel_size,max_num_seqs,max_num_batched_tokens,prefix_cache,s1_p95_ms,s2_decisions_per_second,s2_concurrency
```

| 字段 | 描述 | 原文示例 |
| --- | --- | --- |
| `model` | 模型名称或仓库标识 | `autotrust/JEV-27B-VL` |
| `dataset` | 评测数据集名称 | `OpenDecisions` |
| `image` | 推理环境镜像及版本 | `quay.io/ascend/vllm-ascend:v0.30.0-a3` |
| `resources` | 硬件类型、物理卡及逻辑设备数量 | `A3;1物理卡;2逻辑设备` |
| `precision` | 实际推理精度 | `bfloat16` |
| `tensor_parallel_size` | 张量并行度 TP | `2` |
| `max_num_seqs` | 引擎同时调度的最大序列数 | `8` |
| `max_num_batched_tokens` | 引擎每批调度的 token 上限 | `4096` |
| `prefix_cache` | 是否开启前缀缓存 | `false` |
| `s1_p95_ms` | 单并发端到端 P95 时延，毫秒 | `2000` |
| `s2_decisions_per_second` | 最大有效吞吐，决策数/秒 | `5` |
| `s2_concurrency` | 最大有效吞吐对应的压测并发数 | `32` |

示例数值用于说明字段，提交时填写真实运行值；环境镜像与资源字段同样按实际运行信息记录。

### 5.2 S1、S2 与扫描明细

- **S1**：单并发 P95 端到端时延，单位 ms；保留请求起止时间及 P95 的统计方法。
- **S2**：满足有效性要求的最大吞吐，单位 **decisions/s**，同时记录对应压测并发数。
- 完整并发扫描需保留每个测试点的并发、请求数、P95、有效吞吐、执行正确率、有效性判定，并标记 S1 与最终 S2。
- 优化前后使用可比较的测试输入和统计口径，在报告及日志中列出对照。

十二列汇总表未覆盖请求数、逐点执行正确率、有效性判定和完整扫描记录。准备时将每模型/每数据集的提交版本汇总与逐测试点明细分别保存，明细随 `logs/` 及性能报告交付；逐点文件名与最终 `results.csv` 模板待官方规范确认。

原文尚未给出有效吞吐的具体准入阈值、并发扫描范围、预热次数、测试窗口、重复次数、超时及失败样本处理公式。当前文档将这些列为待明确参数，实测记录保存实际采用值及依据。

## 6. 原始日志与真实执行证据

质量和性能结果均需提供测试命令、参数配置、逐题预测、完整候选概率、执行状态、错误信息、原始输出及结果汇总。日志还应记录请求起止时间，使质量指标、执行正确率、P95 和吞吐可重新核算。

截图与日志需要共同覆盖：

1. 服务启动、模型标识与版本、就绪检查。
2. 正式文本/图文评测接口调用、决策结果与完整概率。
3. 昇腾 910C 设备信息、设备编号、运行进程、HBM 占用及利用率。
4. 主要质量与性能结果，数值与提交 CSV、原始日志一致。

`result.png` 与 `npu_device_call.png` 使用真实执行截图，并保留对应采集时间和原始日志路径。图文型号同时保存图像预处理、视觉推理和 ERQA 分项结果的复核材料。

## 7. 当前工程事实与硬件称谓

规则正文要求 **昇腾 910C**；`results.csv` 示例的 `resources` 使用 **A3**，镜像示例也带 `-a3`。两种称谓之间的赛事认可映射和提交平台要求，需按主办方最终说明核对。

**2026-11-13 用户补充的资源约束：允许并行，全部推理与评测进程合计限用 A3 单物理卡内两个逻辑 NPU，禁止跨物理卡。** TP1、TP2、请求并发和其他并行方式均遵循该资源上限；启动与性能记录保存实际设备列表、并行度和全部相关进程。变更设备或实例后，先核对逻辑设备到物理卡的映射，再执行测试。

当前工程已有的机器事实为：A3 单物理卡内两个 `Ascend910_9382` 逻辑 NPU，各 64 GiB HBM；`npu-smi` 卡号 7，Chip 0 / 1 对应逻辑设备 0 / 1、`Phy-ID` 14 / 15。资源字段按 `A3;1物理卡;2逻辑设备` 记录。[已采集设备映射](evidence/new-machine-2026-11-12.json) 中的 `Phy-ID` 是芯片设备标识，物理卡归属按 `NPU ID` 和 `Chip ID` 核对。后续证据继续保留原始设备输出，910C 身份及其与平台资源说明的关系取得实际或官方证据后标注。

入口改造、CSV 导出、ERQA 适配、S1/S2 正式扫描及截图完成状态，分别以代码检查和实际执行证据更新。后续按用户要求于 15:55 终止 Decision Index，并于 15:58—16:05 完成一次全新 OpenDecisions 全量原始概率评测；质量结果与执行证据见[本轮运行记录](opendecisions-baseline-2026-11-13.md)。

## 8. 最终模板与规则待明确项

| 待明确项 | 需要取得的定义 |
| --- | --- |
| `metrics.csv` 格式 | 五列长表与七列宽表的最终模板；样本数、基线/优化值、执行正确率、阈值及结论的承载位置 |
| `results.csv` 格式 | 最终提交汇总与完整扫描明细的列名、关联方式及优化前后结果的记录方式 |
| OpenDecisions | 数据集 revision、split、样本范围、模板、原生评分器版本、失败处理及发布方 baseline 对照条件 |
| ERQA | 数据入口、revision、split、图像处理规则、准确率定义与分项格式 |
| 执行正确率 | 计算分母、有效响应判定、超时/拒绝/错误的处理 |
| S1 / S2 有效性 | 数据与请求构造、计时边界、预热、窗口、重复次数、并发范围、准入阈值和有效吞吐计算 |
| 硬件要求 | 已按用户补充限定 A3 单物理卡内两个逻辑 NPU；继续核实 910C 与 A3 称谓的赛事认可映射、正式复现设备 |
| 服务细节 | 四接口完整协议、默认配置、`--output-dir` 产物规范及最终 Demo 启动方式 |
| 最终规则版本 | 官网对应版本、更新时间及后续修改内容 |

官方后续发布模板或协议后，应记录来源和版本，再同步工程约定、执行配置及提交产物。

## 9. 当前待对齐项（代码与产物静态核对）

本节为 2026-11-13 的本地静态核对，保持运行中的服务和后台评测配置。

| 项目 | 当前证据 | 后续验收 |
| --- | --- | --- |
| 入口参数 | `start.sh` 透传参数，`inference.py` 已支持 `--model-dir` 和相对/绝对路径；参数表待增加 `--output-dir` | 输出目录参数、日志和结果落盘 |
| 端口与接口 | 启动器默认端口为 8000，直接调用上游 vLLM；已保留 `/v1/systemone` 调用证据 | 目标端口 18143，四接口及各自返回格式逐项验收 |
| 图文质量 | 视觉塔和图像功能自检已有记录 | ERQA 数据协议、请求适配、完整准确率及分项报告 |
| CSV | 现有两份 CSV 使用内部记录格式 | 按最终比赛模板导出，核对质量分单位和完整扫描明细 |
| S1 / S2 | 已有固定请求和有限并发的探索性性能记录 | 指定比赛数据下的 S1、完整并发扫描和有效 S2 |
| 截图与硬件 | 已有两张报告导出图，硬件原始标识为 A3 / `Ascend910_9382` | 正式提交版本的真实截图、910C 认可信息及 CSV/日志一致性 |

本轮完成规则整理和待办记录，以上适配与正式验收均按实际代码和执行证据更新状态。

## 10. Clef 在 OpenDecisions 上的公开参考

### 10.1 原始服务输出与论文温度校准结果

联网核查日期为 2026-11-13。OpenDecisions 数据集作者在 **v1.0.1、test 2,352 条、2026-10-15 榜单**和论文中公布了 `Cloudflare/clef` 的两种结果，分数为 0～100：

| 评测条件 | Calibration | Intelligence | Capability | 依据 |
| --- | ---: | ---: | ---: | --- |
| 原始服务输出（as served） | 82.3 | 65.8 | 74.1 | 数据卡榜单三项分数；论文 Table 19 的 as-served Capability |
| 作者额外温度校准后 | 83.4 | 65.4 | 74.4 | 论文 Table 3 三项分数；Table 19 的 with-temperatures Capability |

来源：[原始数据集榜单](https://huggingface.co/datasets/jgeuter/OpenDecisions#leaderboard-october-15-2026) · [ModelScope 镜像](https://modelscope.cn/datasets/evalscope/OpenDecisions)。榜单行还给出 **TVD 0.245**，对应 129 条具有真实概率分布的校准样本；TVD 与提交字段 `score_mae` 各有定义，分别记录。该榜单行的 accuracy、score_mae、硬件和完整运行配置仍需另行取得。

上述成绩属于**数据集作者的公开评测**。本轮已读 Cloudflare 模型卡、公开发布文章及 `.eval_results` 中尚未定位到模型发布方自行公布的 OpenDecisions 数值；官方 dashboard 本轮读取受限，检索结论限定于已取得资料。[Cloudflare 模型卡](https://huggingface.co/Cloudflare/clef) · [模型评测元数据目录](https://huggingface.co/Cloudflare/clef/tree/main/.eval_results)

[论文 v1](https://arxiv.org/pdf/2610.17205v1) 的具体依据如下：

- **第 7 页 Table 3** 的表注说明，公开系统采用原始服务输出与拟合温度结果中 OpenDecisions Capability 较高的一组；Clef 该行采用 74.4。
- **第 22 页 Table 19** 明列 Clef 的 as served 74.1 与 with temperatures 74.4。Choice / Yes-no / Score 的拟合温度分别为 0.87 / 1.15 / 1.12；作者将小于 1 的温度设为 1，实际应用为 1 / 1.15 / 1.12。温度使用独立的 held-out 校准集拟合。
- **第 20 页 Appendix G，Where temperatures act** 解释温度保持最大概率选项，同时改变 Score 的期望等级。Score competence 按期望等级误差计算，因此 Intelligence 也会变化。

[固定 revision 评分源码](https://huggingface.co/datasets/jgeuter/OpenDecisions/blob/ff0b640be2b72c7d06d9d656a34ad6eed79ec7fe/score.py) 中 Score 使用 `expected_level = sum(k * p_k)`，归一化误差为 `abs(expected_level - gold) / (K - 1)`，再校正机会水平；连续期望值直接参与计算。Score 的 argmax accuracy 单独统计。温度变化能同时提高 Calibration、改变 Intelligence，并最终改变两项均值 Capability。精确到 Intelligence 下降 0.4 分的逐题贡献，仍需作者原始与校准后预测复算。

工程对照原始服务输出时采用 74.1 及其配套三项分数；复现论文时单独记录概率后处理、校准数据和温度参数，并报告校准后的整组三项指标。比赛正式准入继续使用主办方指定 baseline、数据版本、评分器及配置。本轮记录文献条件，现有服务和后台任务配置保持原状。

### 10.2 OpenDecisions 版本与复现定位

截至核查日，原始仓库公开命名的语义版本共 **2 个**：

| 版本 | 发布日期 | 条目数 | 主要变化 |
| --- | --- | ---: | --- |
| 1.0 | 2026-10-12 | 2,352 | 初始发布，初版 split 名称为 `train` |
| 1.0.1（当前） | 2026-10-15 | 2,352 | 清理多余字段、增加 `trap_answer`、说明 `target_probs`、增加 TVD 报告与榜单、split 改名为 `test` |

当前题型为 Choice 1,178 条、Yes/no 586 条、Score 588 条。[Changes](https://huggingface.co/datasets/jgeuter/OpenDecisions#changes) 明确声明 **“Items, labels, and scores are unchanged.”** 两版评分脚本的原有 Calibration / Intelligence / Capability 计算保持一致；1.0.1 新增 129 个真实概率校准样本的 mean TVD 报告，该值单独输出。因此公告中的版本变更支持保留相同题目、标签与原有评分；论文与榜单的两组 Clef 分数按额外温度校准条件解释。

原始仓库当前 main revision 为 `ff0b640be2b72c7d06d9d656a34ad6eed79ec7fe`，仓库无 Git tags。语义版本、Git revision、评分器内容和论文修订各自记录；[arXiv 提交历史](https://arxiv.org/abs/2610.17205) 当前仅有 2026-10-12 的 v1。

| 当前文件 | 字节数 | SHA-256 |
| --- | ---: | --- |
| `opendecisions.jsonl` | 4,931,936 | `78614972006c5e5efbbfc121ce31e15cf576ef6c0cdf24a229b33d34d2bfcda6` |
| `score.py` | 11,463 | `478d14ae7a779c72e0dcd6c3225a3462e24faa8d7e8092afb0d6422ff84e4377` |

本轮核查的 [ModelScope 镜像](https://www.modelscope.cn/datasets/evalscope/OpenDecisions) 与原始仓库当前 README、JSONL 的 SHA-256 一致。下载与复现固定实际 revision、split 和文件哈希，保留来源及评分器，后续版本变更据实更新。[原始提交历史](https://huggingface.co/datasets/jgeuter/OpenDecisions/commits/main)
