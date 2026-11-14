# Clef-27B Ascend 工程约定

## 沟通与目标

- 使用中文直接陈述结论、证据和下一步，遵循用户关于肯定表达的要求。
- 当前任务为在现有昇腾 A3 环境建立可复现的 Clef-27B 比赛提交版本，包含完整主干、联合决策头、文本与图文接口、质量及性能评测证据。
- 2026-11-13 用户补充的比赛交付规则见 `docs/submission-rules-2026-11-13.md`。当前按该版本准备验收；后续以主办方最终协议与模板更新，规则来源和版本均需保留。
- 2026-11-14 第四轮复核：seqs4 合批全量质量未通过 0.03/0.05 工程门槛（Capability 74.200178；4 条 choice 变化；最大概率差 0.097399），最终提交配置改为 **seqs1/B1**；全量质量 74.425479 与 baseline 逐位一致，S1 P95 217.036 ms，S2 5.715 decisions/s @ C16。根因、GDN 覆盖、启动/补丁修复见 `docs/tuning4-2026-11-14.md`。
- 2026-11-14 ERQA 官方 400 题全量图文 baseline 完成：400/400 成功、0 失败、0 重试，accuracy **219/400 = 0.5475**；单图 0.5923、多图 0.4336，8 类分项见 `docs/erqa-baseline-2026-11-14.md`。服务为 TP2/BF16/eager/16384/seqs1/缓存关、`image` 上限 16、图像像素 65,536～1,048,576；C1 wall 101.099 s、p95 323.476 ms。数据固定官方 commit `5bba488`、TFRecord SHA-256 `1974f5e5…b4c93e2`。EvalScope ERQAAdapter 图文补丁在 19 题交叉验证中与直接 `/v1/decide` 逐题一致。原服务已恢复为 PID 432954、原 argv/cwd/4096 配置，8/8 冻结用例通过。
- 2026-11-14 CPU 独立参考：发布方 `joint_schema_model.py` 以 CPU BF16（eager、`use_kernels=False`、batch1、32 线程）重建输入与逐题前向。两套数据全量编码 token 数与 NPU 实测逐题相等（OpenDecisions 2,352/2,352；ERQA 400/400，含 16 图 10,294 token）；参考媒体布局与 `ClefFlashStrategy` 一致、无需补丁；8 条冻结请求在当前服务复放 24/24 逐位一致。逐题前向在 OpenDecisions 154/2352 时按用户要求终止，未生成完整 CPU 评分。证据见 `Clef-27B/logs/cpu-reference-20261114/`，结论与终止范围见 `docs/cpu-npu-reference-2026-11-14.md`。
- 2026-11-13 用户补充资源限制：允许并行，全部推理与评测进程合计限用 **A3 单物理卡内的两个逻辑 NPU**，禁止跨物理卡。TP1、TP2 及并发调优均在此范围内进行，记录实际设备映射和进程配置。
- 2026-11-12 用户已选择新机器现有的 CANN 9.2.1，并采用 PyTorch 2.10 配套环境。
- 环境安装、服务运行及精度性能结论使用实际日志和检查结果标注完成状态。

## 阶段提交与推送

- 每完成一个可验证的小阶段，或取得有实际证据的阶段性进展，及时整理相关代码、文档和验证记录，执行 `commit + push`，保持小步、可回溯的提交。
- 小阶段提交与推送按本约定直接执行。提交说明写清本阶段的具体改动和验证结果；提交前完成对应的必要检查。
- 提交内容聚焦当前阶段，保留用户已有修改，凭据及模型权重大文件按仓库现有约定管理。推送使用正常快进流程，遇到远端更新先核对并整合。

## 当前机器与工作目录

| 项目               | 当前值                                                  |
| ------------------ | ------------------------------------------------------- |
| SSH 目标           | `root@199.88.54.221`                                    |
| 跳板               | `113.47.8.48:2234`，使用平台即时发放的跳板账号及令牌    |
| 远端工程           | `/workspace/yangchen/Clef-27B-ascend`                   |
| 模型目录           | `/workspace/yangchen/models/Clef-27B`                   |
| 日志目录           | `/workspace/yangchen/logs`                              |
| 下载及构建中间文件 | `/workspace/yangchen/work`                              |
| 计划 Python 环境   | `/workspace/yangchen/venvs/clef-py312`                  |
| 共享数据           | `/workspace/shared_assets/datasets`，按平台说明只读使用 |

SSH 模板如下。凭据通过当前会话或平台提供，仓库保存脱敏连接信息。

```bash
ssh -J '<跳板账号>:<临时令牌>@113.47.8.48:2234' root@199.88.54.221
```

新机实测为 Ubuntu 22.04.5 / aarch64，A3 单物理卡内两个 `Ascend910_9382` 逻辑 NPU，各 64 GiB HBM。`npu-smi` 卡号为 7，Chip 0 / 1 对应逻辑设备 0 / 1、`Phy-ID` 14 / 15；资源统计按 **1 物理卡、2 逻辑 NPU** 记录。容器资源为 128 GiB 内存和 32 核等价 CPU 时间配额。设备编号和资源配额在实例变化后重新查询。

## 固定的软件组合

| 组件                             | 目标                                       |
| -------------------------------- | ------------------------------------------ |
| CANN Toolkit / A3 Ops / NNAL ATB | 9.2.1，新机器已经安装                      |
| Python                           | 独立 CPython 3.12 环境，安装时记录补丁版本 |
| torch                            | `2.10.0+cpu`                               |
| torch_npu                        | `2.10.0.post7`                             |
| torchvision / torchaudio         | `0.25.0` / `2.10.0`，采用对应 CPU 构建     |
| Transformers                     | `5.19.0`                                   |
| Triton Ascend                    | `3.6.0.post2`                              |
| FLA NPU A3                       | `flash-linear-attention-npu-a3==26.9.1`    |
| flash_attn_npu                   | `0.5.0`                                    |
| vLLM                             | `24420b7a341af37a9bc3186e6fc6062512e0aa58` |
| vLLM-Ascend                      | `89eb8a5e7a76d085b453a1aeadb6883e12d5ea2d` |

构建前加载所选 CANN 与 ATB 的环境脚本，安装 CMake 至少 3.26.1、Ninja 及官方 A3 配方要求的编译依赖。按固定源码的 `empty` 构建流程安装 vLLM，随后构建 Ascend 插件与 A3 扩展。运行依赖由 constraints 固定，构建隔离环境的依赖单独记录。验收包含 `pip check`、实际版本、加载路径和 NPU 运算结果。

## 模型与入口

- 模型以 `Clef-27B/logs/model_manifest.json` 中的逐文件 revision、大小和 SHA-256 为准。
- `scripts/download_model_manifest.py` 使用系统 Python 标准库下载，支持 `.part` 续传、并行 HTTP Range 和 SHA-256 校验。成功报告保存到工程或远端日志目录。
- `Clef-27B/start.sh` 和 `inference.py` 是比赛入口。目标命令为 `bash start.sh --model-dir xxx --output-dir xxx`，模型路径支持相对和绝对路径；保留 `MODEL_DIR` 与 `PYTHON_BIN` 调用方式。参数、补丁初始化、服务加载与日志产物均须实测验收。
- 目标默认地址为 `http://127.0.0.1:18143`，提供 `GET /health`、`GET /v1/decide/info`、文本 `POST /v1/systemone` 与图文 `POST /v1/decide`。当前启动器的 `--output-dir`、默认端口及新增接口规范列为待适配项，完成状态以代码和实际请求证据为准。
- 模型加载必须覆盖 `joint_head_config.json`、`joint_head.safetensors` 和完整主干，确认识别为 `ClefFlashForJointSchema`。
- 首轮配置为 BF16、TP1、eager、max model len 1024、batch 1、纯文本，关闭 prefix caching 和 chunked prefill。
- 服务模型名统一为 `clef`。请求中的 `joint` 采用默认值，三种决策由模型联合决策头计算。
- 在相同输入上检查 `choice`、`noul`、`score`、候选概率、多字段和 token/span 对齐，再扩展 TP2 与多模态。

就绪响应包含 `status=ready`、`model`、`revision`；`/v1/decide/info` 返回模型身份、任务能力、协议版本和缓存配置。图文路径包含图像处理及视觉推理适配。

## 比赛质量、性能与交付

- Clef-27B 按多模态型号准备 **OpenDecisions + ERQA**。OpenDecisions 提交完整 Calibration、Intelligence、Capability 及 accuracy、score_mae；ERQA 提交图文准确率和分项结果。固定数据 revision、split、模板、评分器、样本范围及失败处理。
- `accuracy` 按 0～1 记录，Calibration / Intelligence / Capability 按 0～100 分记录，`capability=(calibration+intelligence)/2`；Score MAE 单独保留等级单位，执行正确率保留有效性判定及分母。
- 性能提交 **S1：单并发端到端 P95（ms）**，**S2：最大有效吞吐（decisions/s）及对应并发**，并保留完整并发扫描、逐请求起止时间和优化前后对照。有效性阈值、扫描范围及统计参数按主办方最终协议固定。
- 提交 `start.sh`、`inference.py`、`README.md`、`metrics.csv`、`results.csv`、`result.png`、`npu_device_call.png`、`logs/`、`optimization.md`；`src/` 与 `model_agent_pr.md` 为可选。日志须含命令、实际配置、逐题预测与完整概率、执行状态、错误和原始输出；截图来自真实执行并与 CSV 和日志对应。
- 每模型、每数据集一行记录提交版本的实测汇总，保留基线/优化值、样本数、准入阈值与结论。用户提供的 `metrics.csv` 五列长表与七列宽表格式并列保存在规则文档，最终导出格式待主办方模板确认；现有 CSV 的历史格式单独保留。
- 规则正文要求 910C，示例资源为 A3；当前机器原始标识为 `Ascend910_9382`。继续保存真实硬件、设备编号及资源输出，910C/A3 的赛事认可映射待核实。

## 数据、验证与记录

- 转换数据与缓存放在 `/workspace/yangchen` 下，保留共享数据来源、revision、split、模板和哈希。
- 比赛验收按本次补充规则的 OpenDecisions / ERQA 与 S1 / S2 组织。Decision Index 0.2.1、Typesafe WorkflowEvals 保留为模型发布方对照评测，BFCL、API-Bank 等按 Decision Index 子项报告；保留各套件原生评分和聚合，既有执行说明见 `docs/evaluation-protocol.md`。
- 早期探索评测结果仅作历史归档，赛事提交采用指定协议下的提交版本实测值。OpenDecisions 数据集作者公布的 Clef 原始服务输出参考为 Calibration 82.3、Intelligence 65.8、Capability 74.1；论文 Table 3 的作者额外温度校准后参考为 83.4、65.4、74.4。引用时注明来源和校准条件，评测配置分别记录原始与校准后结果；主办方正式 baseline 按赛事资源确认。
- OpenDecisions 当前公开版本为 1.0.1（2026-10-15），历史初版为 1.0（2026-10-12），均为 2,352 条；1.0.1 公告确认题目、标签及原有评分保持一致。复现同时固定数据与评分器 revision、哈希和 split，版本及论文分数差异的证据见 `docs/submission-rules-2026-11-13.md` 第 10 节。
- 发布方对照评测的两套数据快照已于 2026-11-13 下载并校验，目录为 `/workspace/yangchen/datasets/clef-official-20261113/`。逐文件来源、revision、大小和服务器 SHA-256，以及 Decision Index 官方解压哈希与行数对照见 `Clef-27B/logs/official-datasets-download-20261113.json`；模型评测状态单独记录。
- 2026-11-13 发布方对照运行保持 A3 / TP2 / BF16 / eager / 4096 / 单序列服务。WorkflowEvals 采用官方 `questions=one` 完成全部 705 case，原生评分与配置见 `Clef-27B/logs/workflowevals-baseline-20261113.json`；Decision Index 0.2.1 已于 15:55 按用户要求终止评测及 supervisor，部分输出保留在 `/workspace/yangchen/logs/official-baseline-20261113/decision-index/`，终止证据见 `Clef-27B/logs/decision-index-stopped-20261113.json`。该次运行按未完成记录。
- 2026-11-13 15:58—16:05 全新 OpenDecisions 1.0.1 全量原始概率评测已完成，2,352/2,352 成功、0 错误。Calibration 82.7941、Intelligence 66.0568、Capability 74.4255；accuracy 1906/2352，Score MAE 0.401389（588 条）。固定版作者 `score.py` 与 EvalScope 未舍入三项指标完全一致。输出目录为 `/workspace/yangchen/logs/opendecisions-baseline-20261113-1600/`，证据见 `Clef-27B/logs/opendecisions-baseline-20261113/`，复现方式见 `docs/opendecisions-baseline-2026-11-13.md`。每次重跑使用新空目录，保存完整原始概率与执行覆盖。
- 下载完成以逐文件 SHA-256 校验报告为准。依赖完成以版本与 NPU smoke 日志为准。完整 baseline 以服务、精度和性能记录为准。
- 新机器的当前状态与后续执行计划见 `docs/new-machine-plan.md`，旧机器记录在文档中保留采集时间和来源。
- 修改任务涉及的代码和文档，保留用户已有修改。提交前检查文档链接、命令语法以及本次实际执行产生的结果。
