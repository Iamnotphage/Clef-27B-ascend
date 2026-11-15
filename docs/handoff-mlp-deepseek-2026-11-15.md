# Clef MLP 优化交接 prompt · 2026-11-15

将下文完整复制给 DeepSeek-v4.1-flash 后继续执行。

---

请接续 Clef-27B 昇腾 A3 优化工作，依据实际证据完成候选评估、服务恢复和小阶段提交。使用中文肯定表达。本轮已按用户要求暂停并恢复实验前服务。恢复 PID 为520642，8000端口健康200，冻结8/8通过，argv/cwd/环境与快照完全一致。恢复记录位于 `Clef-27B/logs/mlp-20261115/restore/verification.json`，开始时结合现场状态和 `git log` 核对最新身份。开始时先确认恢复记录和当前服务身份，再开展下一轮实验。

## 工作目录和约束

本地仓库 `/Users/chenclay/Documents/code/projects/Clef-27B-ascend`，远端通过 `ssh clef` 连接。远端仓库 `/workspace/yangchen/Clef-27B-ascend`，BF16 模型 `/workspace/yangchen/models/Clef-27B`，gate/up INT8 候选 `/workspace/yangchen/models/Clef-27B-mlp-gateup-w8a8`。本轮日志 `/workspace/yangchen/logs/mlp-20261115-1027`，归档在仓库 `Clef-27B/logs/mlp-20261115/`，主报告为 `docs/mlp-optimization-2026-11-15.md`。

先读 `AGENTS.md`、主报告、最新 restore 记录，核对两端 `git status --short`、HEAD、远端分支。本地用户已有修改为 `docs/profiling-2026-11-12.md`。远端HEAD曾为 `3e83ae1` 且存在大量已有工作区修改，本轮采用针对性文件同步。保留两端已有修改，通过核对差异和备份后同步目标文件。传输使用 `scp -O`，默认SFTP传输曾失败。CPU全量独立参考此前因预计20多小时已按用户要求终止，本轮推理和评分在NPU适配版本上继续。只提交本阶段文件。每完成可验证小阶段执行必要检查与 `commit + push`，正常快进整合远端更新。模型大文件和凭据沿用仓库管理约定。

软件固定 CANN 9.2.1、Python 3.12、torch 2.10.0+cpu、torch_npu 2.10.0.post7、Transformers 5.19.0，vLLM `24420b7a341af37a9bc3186e6fc6062512e0aa58`、vLLM-Ascend `89eb8a5e7a76d085b453a1aeadb6883e12d5ea2d`。资源上限为一张 A3 物理卡内的两个 Ascend910_9382 逻辑NPU，各64 GiB。物理卡号7，逻辑设备0/1，对应Chip0/1、Phy-ID14/15，全部推理与评测合计使用该范围。实验前加载工程环境并核对设备状态。

```bash
ssh clef
cd /workspace/yangchen/Clef-27B-ascend
git status --short
git log -8 --oneline
source scripts/activate_env.sh
npu-smi info
```

## 已完成的证据

历史提交 `011fa2d` 完成 profile 归档与原 TP2/4096 多模态服务恢复，`b06fd34` 完成 GDN patch 安装器可靠性修复及34项测试。已推送 `df2a1f6` 为GDN远端复验、`3cddd6f` 为微基准及converter、`cba9ea5` 为启动器参数修复。最终收尾提交见 restore 报告与 `git log`。

完整 MLP W8A8_DYNAMIC 微基准完成12点，device配对收益1.354–1.633×，wall 1.284–1.604×。选择性 gate/up 量化12点的device收益1.205–1.376×、wall 1.176–1.371×，layer31局部相对RMSE为10.90%–12.50%。计时输入来自真实服务，三层0/31/63、四个T=307/1595/662/10294，测量TP2单rank局部输出，终点位于all-reduce之前。局部收益须通过完整服务数据确认。

FP16 `npu_ffn` 已完成负结果验证。BF16→FP16→FFN→BF16完整链耗时为BF16分解实现的1.777–2.573倍，四输入每点六轮方向一致。保留该证据，下一轮优先排查现有量化路径。

64层gate/up已转换，128份原始权重转为INT8，scale为逐输出通道BF16，offset为零。1056个保留主干张量及联合头逐项哈希一致。正常入口的双token参数透传已修复，10项启动测试通过。候选每rank加载64个gate_up INT8参数，共5,704,253,440个元素，联合头122个参数全部加载。使用原生量化loader，`strict_tracking_enabled=false`。额外开启严格跟踪会提前检查48个 `ascend_conv1d_weight` 派生缓存，缓存在后处理由源卷积权重转置生成。原始失败集合、成功加载审计和转换核验已保留。

OD18冒烟全部成功，ERQA19全部成功且请求哈希/token对齐，ERQA正确数BF16为7/19、候选为6/19。完整OD2352/ERQA400的最终数值见下方接续记录。旧逐概率0.03、score期望0.05阈值用于诊断，候选质量依据作者原生Calibration、Intelligence、Capability和ERQA官方exact-match、逐题变化共同判断。

## 下一轮执行顺序

首选评估原生 `weight_nz_mode=0` 的ND路径。短形状INT8路径已出现端到端退化，先比较ND/NZ布局及调度开销，再决定进一步实现。当前探针强制NZ/format29，先扩展显式format选项，保留默认NZ及格式实测断言，支持ND/format2。固定输入、权重、scale、预热、轮数与计时终点，保存实际格式、原始时序和源码SHA。先做短形状及代表性长输入的成对局部实验，再运行服务端对照。

重点补齐T小于512的真实输入，并覆盖145–255、256–383、384–511三档。分别统计动态量化、quant_matmul、SwiGLU及完整MLP链，保持权重驻留和同步边界一致。按照ND/NZ交替顺序重复测量，报告每轮收益和波动。取得稳定收益后再扩展完整模型，按同一冻结集合重算质量。新kernel的需求由原生路径的剩余瓶颈决定，具体候选算子待办参阅已有 `short-shape-followup.md`。

候选服务参数与BF16评分对照统一为TP2、seqs1、eager、maxlen/budget16384、16图、pixels65536–1048576、prefix/chunked/schema cache关闭、GDN决策prefill开启。默认提交配置为BF16。候选S2仍待测，先完成S1路径判断。

```bash
clef_nd_run="/workspace/yangchen/logs/mlp-nd-$(date -u +%Y%m%d-%H%M%S)"
bash Clef-27B/start.sh \
  --model-dir /workspace/yangchen/models/Clef-27B-mlp-gateup-w8a8 \
  --output-dir "$clef_nd_run" --config s1 -- \
  --quantization ascend \
  --additional-config '{"weight_nz_mode":0}' \
  --max-model-len 16384 --max-num-batched-tokens 16384 \
  --limit-mm-per-prompt '{"image":16,"video":0}' \
  --mm-processor-kwargs '{"images_kwargs":{"size":{"shortest_edge":65536,"longest_edge":1048576}}}'
```

执行服务切换前，保存当前完整argv、launch manifest、PID、模型身份、环境和验收结果，并确定恢复命令。每个实验使用独立日志目录，串行管理服务与NPU实验。复用 `runtime/run_mlp_dataset_eval.py` 和 `runtime/run_mlp_throughput_scan.py`，先检查参数及现有调用记录。失败时诊断、修复并重试，保留原始失败证据。

ERQA性能现有预热条件有差异，BF16仅冻结8条预热，候选经历19条smoke。两侧统一完整ERQA预热，再按相同400题、顺序、并发、输入哈希、图像和token预算重测。记录全流程wall、HTTP p50/p95/p99、吞吐、失败、CPU配额与内存采样，注明视觉预处理及缓存状态。预热和正式测量分目录归档。

完成每轮后恢复实验前原服务快照，验证 `/health`、`/v1/decide/info`、`/v1/systemone` 以及冻结文本和多模态请求，核对模型身份、协议、全部概率及token/span。实际恢复PID和最终commit写入restore报告。最后给出可采用配置、负结果、尚待验证项及下一步，提交相关代码、文档和验证记录。

## 完整评分与性能接续记录

以下为完整集合真实结果。OD两侧均先执行2352条预热，正式评测各一个窗口、client concurrency 1，2352/2352成功且请求与token全部对齐。候选预热与正式轮2352条概率数值逐项一致。

| OD2352指标 | BF16 | gate/up W8A8 |
| --- | ---: | ---: |
| Calibration | 82.7941149442 | 82.6493687743 |
| Intelligence | 66.0568428085 | 65.5268266235 |
| Capability | 74.4254788763 | 74.0880976989 |
| 正确数 | 1906/2352 | 1903/2352 |
| p50 / p95 / p99，ms | 199.514535 / 207.708020 / 213.290811 | 195.245415 / 207.104697 / 210.494835 |
| wall，s | 408.703621 | 416.646155 |
| 吞吐，req/s | 5.754781 | 5.645078 |

OD候选吞吐下降1.9063%，配对延迟累计增加7.9458s。T小于512的1636题累计增加10.5728s，T至少512的716题累计减少2.6270s。145–255档平均增加10.2846ms，256–383档增加5.3578ms。OD正确迁移为18条正确变错、15条错误变正确。BF16的588题探索扫描在C16观测峰值5.860027req/s，候选扫描及S2仍待完成。该扫描的窗口和输入集合单独解释。

ERQA两侧均400/400成功、219/400正确，即54.75%。BF16/candidate p95为331.495474/292.617537ms，吞吐3.528630/3.870919req/s。21条choice变化，包括7条正确变错、7条错误变正确、7条错答互换。按前述统一预热方案复测后，再确认性能收益的稳定性。每次报告同时列出质量迁移和性能统计，保留完整概率以便按字段追溯误差。

先阅读归档的 `reference-summary.json`、`formal-paired-latency.json`、`candidate-repeatability.json` 与主报告，获取完整分桶、数据哈希和请求身份。它们均在 `Clef-27B/logs/mlp-20261115/`。BF16资源采样覆盖OD前段存在约38秒边界，后续重测应在预热前启动采样并覆盖正式窗口。当前默认BF16作为提交和质量对照配置，候选是否采用由完整评分、稳定性能和恢复验收共同决定。
