# 2026-11-19 最终验收冻结计划

状态：**计划已准备，实际验收未运行**。28 仅进行 CPU 实现、只读 SSH 核验及本地 HTTP 边界模拟；27 独占设备、服务生命周期和真实请求执行。北京时间 06:30 停新增候选，07:40 前收口，08:00 授权窗口结束。所有 UTC 时间保存原值，再按 UTC+08:00 换算；远端 timezone 标签不参与判断。

本文件与 [`runtime/final_acceptance_manifest.json`](../runtime/final_acceptance_manifest.json)、[`runtime/final_acceptance.py`](../runtime/final_acceptance.py) 构成同一交付。机器清单保留完整路径、SHA、有效 argv/env、数据版本、分母、阈值、阶段命令参数及待填字段。服务生命周期由 27 的已审流程执行，封装仅对已就绪服务发评测请求。

## 1. 真实来源与冻结身份

初次核对 `date -u` 为 `2026-11-18T18:46:11Z`，即北京时间 `2026-11-19 02:46:11`；清单采集原值为 `2026-11-18T18:48:21Z`，即 `02:48:21`。

| 对象 | 真实来源 / 精确身份 | 核验范围 |
| --- | --- | --- |
| 生产工程 | `/workspace/yangchen/Clef-27B-ascend`，HEAD `3e83ae17df7b559ec2dd6b2a38aa029c2777e061` | 只读 `git rev-parse`，评测封装另行部署 |
| 生产 head | `/workspace/yangchen/work/src/vllm/vllm/model_executor/models/clef_joint_schema.py`；`f8101b98228bbd472a63a1efe6827583e35c6cc9dede4379b925f2b016670863` | 当前文件 SHA 实测 |
| 原 guard | `runtime/apply_joint_single_field_patch.py`；`76595657d312699b4b1564d1bc054229808cd775c16702fb0e36fcdb1f9706d9` | 当前文件 SHA 实测 |
| 软件树 | vLLM `24420b7a341af37a9bc3186e6fc6062512e0aa58`；Ascend `89eb8a5e7a76d085b453a1aeadb6883e12d5ea2d` | Git HEAD、全部 tracked 修改文件与运行用 untracked Python、两组 vendor 文件 SHA 在 manifest |
| 软件包 | Python 3.12.16；CANN 9.2.1；torch 2.10.0+cpu；torch_npu 2.10.0.post7；Triton Ascend 3.6.0.post2；Transformers 5.19.0 | 包版本只读 `importlib.metadata`，未 import torch/NPU；CANN 来自成功 launch 的实际路径 |
| 模型 | `/workspace/yangchen/models/Clef-27B`，`local-65e5a3c5ffb6` | `model_manifest.json` SHA `98c104a9d27227ee7c86a2713fdd5aa7e4256fd7826a4a1fc7168f982cad6b68`，清单含逐文件 revision/大小/预期 SHA；本轮未重新读取全部权重计算 SHA |
| OD 原数据 | `/workspace/yangchen/datasets/clef-full-20261112/full-data/open-decisions2352.jsonl` | 1.0.1 / test / revision `ff0b640be2b72c7d06d9d656a34ad6eed79ec7fe`；SHA `78614972006c5e5efbbfc121ce31e15cf576ef6c0cdf24a229b33d34d2bfcda6` |
| OD 冻结请求 | `/workspace/yangchen/logs/tuning/20261114-013000-fix4/opendecisions/all2352.json` | 2352 条；SHA `74ce0f4fde81c1fa1f125dc6b51db2f46fb6d358a37d9faf9c54b362884a2141` |
| OD 模板 | `/workspace/yangchen/logs/opendecisions-baseline-20261113-1600/evalscope/predictions/clef/clef_open_decisions_full2352_default.jsonl` | SHA `29fe80b8062b9966277f0d34b1aed76d8ef80ac260eed3c53f2bd0a83f25e915`；执行前逐题 request 比较 |
| OD 作者评分器 | `/workspace/yangchen/logs/opendecisions-baseline-20261113-1600/author-score.py` | SHA `478d14ae7a779c72e0dcd6c3225a3462e24faa8d7e8092afb0d6422ff84e4377`；调用既有 `od_raw_acceptance.py` |
| ERQA | `/workspace/yangchen/datasets/erqa-official-20261114/` | 官方 commit `5bba488252c5b345a43335bfb9239b6bf12bd91c`，400 条、630 图、单题最多16图 |
| ERQA TFRecord | `/workspace/yangchen/work/src/ERQA/data/erqa.tfrecord` | 91,402,921 字节；SHA `1974f5e587eac144cc3ba325477477b8ad6a56c22ce4e419c34229520b4c93e2` |
| ERQA 提取文件 | `manifest.json` / `samples.jsonl` | SHA `a3e9cd3e4449141e252c35b347c32fea410a65fe1e3b98b52ff2af5f57577bfb` / `35bd36560dcc228c02a8e336cc57b7b8bb0f1c6752521dc63fc482f48a4168af`；原顺序 `erqa-0000..0399`，每图 SHA 由原请求构建器检查 |

当前 f810 生产对照与 aa3e 候选严格分开记录。`aa3e17585f8bd60a4d05e342abc7e2773c79a0b2d49f0370d15e29bcc6b72503` 在 `CLEF_JOINT_FUSED_ATTN=0` 时已共享 routing KV；flag 0/1 仅说明内部增量。采用候选前须有 f810 总收益对照、完整质量和审定 guardadapter。当前清单冻结 f810；选定候选后由 27 与 24 更新清单的实际 source/guard/env/argv，重新审查并保存新的清单 SHA。

## 2. 原始成功协议与统计

原件优先级：成功运行原始 JSON/launch manifest → 对应源码 → 已提交报告。历史报告中的复现命令用于校验参数；以下显式命令由这些来源组装，**没有声称存在完整历史 shell transcript**。

| 阶段 | 真实成功原件（均在 `/workspace/yangchen/logs/perf-wave2-20261118/`） | 最终计划 |
| --- | --- | --- |
| S1 | `s1-final/r1..r5/s1.json`，`s1-final-r1/launch-manifest.json` | C1，全量2352，至少3轮；每轮独立统计，报告 p95 中位与范围，保留各轮 mean/p50/p95/p99/吞吐 |
| S2 B8 | `s2-b8/s2-b8c16.json`，`s2-b8-c16/launch-manifest.json` | B8/token8192，C8、C16 各2352；历史完整扫描范围见 `docs/batching-seqs8-16-2026-11-16.md` |
| S2 B16 | `s2-b16/s2-b16c64.json`，`s2-b16-c64/launch-manifest.json` | B16/token16384，C32、C64 各2352；每点独立离线质量，不提前确定提交点 |
| OD 质量 | `s1-final/r3/quality.json`；原 baseline `opendecisions-baseline-20261113-1600/author-score.json` | 每个正式 S1/S2 结果用冻结作者 scorer 计算原始概率质量 |
| ERQA | `erqa/summary.json`、`erqa/offline-score.json`、`erqa-final/launch-manifest.json` | B1/C1，完整400；原 `erqa_baseline.py` 后接 `erqa_score.py --verify-requests` |

OD 沿 `runtime/od_full_benchmark.py`：固定顺序每题一次，`--warmup 0 --timeout 300`；C>1 由 ThreadPoolExecutor 同时排队全部 futures。历史 raw 未写入 warmup/timeout 字段，零预热/300秒来自已成功复现命令未传覆盖项及当时源码默认值；此次显式冻结并写入 configuration。百分位是排序后 `(n-1)*q` 线性插值；延迟包括客户端序列化、HTTP、解析、答案提取；吞吐分母是正式测量 wall time，成功响应数为分子。失败记录保留；覆盖不足或失败时不输出通过验收结论。预热策略保持零，不将全量首轮暗中移作预热。

ERQA 沿原协议：完整题干与选项、图片按 visual_indices 稳定排序、原始 choice 精确匹配规则；3 次最大尝试、每次600秒、重试间隔5秒，客户端C1、无resume、零额外预热。原成功记录400/400、0重试。准确率固定400分母，失败计错；始终保留重试数、失败分类和每题全部概率。保留8类、单/多图、候选数及visual模式分项。

新增记录层仅补证据：OD 逐完成请求 JSONL、微秒 UTC 起止、HTTP 错误原文与成功响应文本；ERQA 每次尝试响应 `.bin`。原请求构造、调度、延迟测量范围和评分逻辑沿用。JSONL 落盘在客户端完成收集处，wall time 包含其开销，最终报告需保留该脚本版本。异常中止留下已完成响应、冻结输入和步骤日志；未完成题按固定输入索引核对，整轮记为不完整。

历史 `od_raw_acceptance.py` 对新 runner 的性能 schema 读出部分 null；最终封装从 raw `summary` 直接读性能，质量沿作者评分结果，避免将 null 当作成功证据。

## 3. 质量阈值与源配置

原 baseline：Calibration **82.79411494415206**、Intelligence **66.0568428084581**、Capability **74.42547887630508**；OD accuracy **1906/2352**；Score MAE **0.4013893127**（588题，等级单位）；ERQA **219/400 = 0.5475**。

按 [规则4.3/4.4](submission-rules-2026-11-13.md)：三项 OD 绝对波动各≤2.0分，ERQA总体 accuracy 绝对波动≤0.02；原始概率、无温度校准、BF16固定、禁止答案缓存。OD accuracy 和 Score MAE 继续报告。逐题概率差与choice变化可离线追加比较，历史0.03/0.05只作诊断记录。

共通：TP2/BF16/eager、pooling、模型名clef、prefix/chunked/schema缓存关、物理卡7/chip0、1/逻辑0、1/phy14、15。GDN decision/BIC/convQKV、chunk128源码、MLP BF16 NZ、ACLNN cache limit1000000以 source SHA 与实际 launch/env 双重冻结。清单含完整 argv、service_environment；额外 profile/候选开关默认拒绝。

ERQA 必须显式使用 `--max-model-len 16384 --max-num-batched-tokens 16384 --limit-mm-per-prompt '{"image":16,"video":0}'`，以及

```json
{"images_kwargs":{"size":{"shortest_edge":65536,"longest_edge":1048576}},"videos_kwargs":{"size":{"shortest_edge":4096,"longest_edge":65536},"do_sample_frames":false}}
```

默认 s1 的图像 longest_edge 是65536；上列1048576来自 ERQA 成功 launch 原件。ERQA最长实测10294 tokens。

## 4. 单命令执行接口（仅27执行）

先将本提交的文件以固定 commit staging 到独立目录，例如 `/workspace/yangchen/work/final-acceptance-<完整提交SHA>`。保留生产源码与已有成功日志原位；清单冻结的新 runner SHA 必须匹配 staging 文件。`CLEF_ACCEPT_CODE`、`CLEF_ACCEPT_RUN`、`CLEF_READY_IDS`、`CLEF_READY_LAUNCH`、`CLEF_OWNER_EVIDENCE` 为**27填写的实际路径**，下列模板具有显式依赖，尚未运行。

```bash
# 单阶段单命令；phase取 s1 / s2_b8 / s2_b16 / erqa。
# --execute去掉后仅输出完整命令计划，不创建输出目录、不访问服务。
/workspace/yangchen/venvs/clef-py312/bin/python \
  "$CLEF_ACCEPT_CODE/runtime/final_acceptance.py" \
  --manifest "$CLEF_ACCEPT_CODE/runtime/final_acceptance_manifest.json" \
  --phase s1 --output-dir "$CLEF_ACCEPT_RUN" \
  --identities "$CLEF_READY_IDS" --launch-manifest "$CLEF_READY_LAUNCH" \
  --owner-evidence "$CLEF_OWNER_EVIDENCE" --execute
```

`--identities` 接受 ready 后 root 在首位的列表（27的`service-identities.json`），或 `{root, descendants}`。每个元素必含 pid/ppid/pgid(int)、starttime(str)、boot_id、完整空格连接argv的cmdline；实际 argv另从/proc读取。封装对全/proc扫描、树成员、五身份、launch/api.pid、源码/guard、配置/env、port、health/info做前后校验。资源证据按原样保存，**NPU同卡与无竞争的解释、审定及准入由27执行**；封装未实现新的npu-smi文本解析器。

输出目录必须不存在；每次失败换新目录，旧目录完整保留。步骤stdout/stderr、实际argv、UTC起止和退出码写入`execution.json`及每步log；输出包括`frozen-manifest.json`、before/after身份、每轮raw+JSONL+quality、`acceptance.json`。ERQA保留每题request/meta/record/attempts/response与离线核验。

## 5. 合法切换与恢复顺序（27生命周期边界）

06:30冻结赢家；先保存最终源码、guard、完整运行态与实际env，之后 S1三轮 → B8两点 → B16两点 → ERQA400 → 恢复选定最佳服务。S1/质量失败保留原始证据并回到已验证生产栈；不得把64条短样本p95作为全量S1对照。若候选最终采用，恢复目标必须与通过全量验收的源/配置身份一致。

每次切换前：fresh全/proc、源/guard SHA、唯一label/api.pid、运行期root+descendants、port状态与NPU卡7两chip表归档；状态冲突立即阻断。初始bash PID在exec后必须证明 pid/starttime/boot_id/PGID 连续，更新为最终vLLM cmdline并重抓后代。

已审停服模板（动态变量由27的fresh记录取得）：

```bash
timeout 120 /workspace/yangchen/venvs/clef-py312/bin/python \
  /tmp/d0cc/d0_stop.py --api-pid "$CLEF_READY_PID" \
  --saved-identity "$CLEF_SWITCH_RUN/root-identity.json" \
  --out "$CLEF_SWITCH_RUN/stop.json"
```

stop SHA：`d6d49d75012a0789d3bff3e7b522e9cd496844ca0ee0424608ab8e6b43a33886`。非零或残留阻断；0=open、111=closed，其它port结果为unknown并阻断。禁用旧宽pgrep与旧`final-default-s1`label。

已核启动入口 `/workspace/yangchen/logs/perf-wave2-20261118/launch.sh` SHA `a737386119b89c5bc787e6f69e5017b83391934148528d08d498ff943a958321`，内部source `/workspace/yangchen/Clef-27B-ascend/scripts/activate_env.sh` SHA `3b363394536fbe2433a84b666170568fcb58848b1d8d24551ff6ae9f53b4044a`。唯一label由27生成并先断言其目录不存在；下面四条使用第8节已核clean seed，分别选择配置，逐阶段运行；父环境不预source：

```bash
# S1与最后恢复：选定源码/原guard先由27恢复到已验身份。
env -i PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin CLEF_LAUNCH_CONFIG=s1 /bin/bash /workspace/yangchen/logs/perf-wave2-20261118/launch.sh "$CLEF_UNIQUE_LABEL"
# S2 B8，仍使用s1优化栈。
env -i PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin CLEF_LAUNCH_CONFIG=s1 /bin/bash /workspace/yangchen/logs/perf-wave2-20261118/launch.sh "$CLEF_UNIQUE_LABEL" --max-num-seqs 8 --max-num-batched-tokens 8192
# S2 B16。
env -i PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin CLEF_LAUNCH_CONFIG=s1 /bin/bash /workspace/yangchen/logs/perf-wave2-20261118/launch.sh "$CLEF_UNIQUE_LABEL" --max-num-seqs 16 --max-num-batched-tokens 16384
# ERQA。
env -i PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin CLEF_LAUNCH_CONFIG=s1 /bin/bash /workspace/yangchen/logs/perf-wave2-20261118/launch.sh "$CLEF_UNIQUE_LABEL" --max-model-len 16384 --max-num-batched-tokens 16384 --limit-mm-per-prompt '{"image":16,"video":0}' --mm-processor-kwargs '{"images_kwargs":{"size":{"shortest_edge":65536,"longest_edge":1048576}},"videos_kwargs":{"size":{"shortest_edge":4096,"longest_edge":65536},"do_sample_frames":false}}'
# 每次launch后：
timeout 600 bash /workspace/yangchen/logs/perf-wave2-20261118/wait.sh
```

wait SHA `e3fb08882211aef6ea0fd4f3db3d37bba0a591034cc397ba95559e09d1f1ffa0`。上面启动模板使用完整两键父环境，并由27先选定审定源码；清单中的运行时env断言继续逐字节检查。源恢复必须使用27已审D0恢复流程，原guard恢复顺序保持其现有约定。

恢复原件入口：`/workspace/yangchen/work/scratch-24/joint-fused-5c045eb/runtime/joint_fused_attention_window.py` SHA `9245e1a6504742dc4706eb466f8053cb3999ff3d80abfac960ed218a37a747d5`；六D0模块哈希由它钉住。该wrapper入口还承担候选AB，本文不猜测新的restore-only参数。27直接复用已成功恢复操作及`restore.json`的health/workers/text/image真实RC/ok、进程树与同卡检查。历史可核身份：`/workspace/yangchen/logs/handoff27-restore-20261118T183212Z-dc78478b/service-identities.json`；该PID只作历史证据，执行前取fresh。

## 6. 验证与待完成事项

CPU验证涵盖真实CLI参数、原请求/评分入口、HTTP错误/无效JSON保存、目录复用拒绝、清单漂移阻断、S1三轮与2pp汇总逻辑。NPU性能、真实服务身份断言与最终恢复仍由27实测。最终权重完整内容SHA复核沿用原已校验下载证据，本封装检查冻结manifest、文件存在和大小；本轮未冒充新的全权重哈希扫描。

尚待实际填写：最终选中的源/配置、每阶段fresh身份与资源证据、至少3轮S1原件、全部S2扫描/质量、ERQA400原件、最终恢复结果与提交点。所有新输出保持`plan_not_executed`，直到实际子步骤与验收完成。

执行监督边界：封装的子进程调用没有总墙钟timeout，也未实现signal转发或进程组cleanup。**27必须用既有外层监督器绑定评测客户端进程组与本阶段截止时间**，记录runner及子进程身份；中断/超时后确认该组客户端全部退出，再切换或恢复服务。HTTP单请求timeout 300/600秒不能替代阶段截止。06:30后的验收预算由27按剩余时间分配，07:40收口前先清理评测客户端，再进行最终恢复；阶段失败保留已有原始文件，汇总状态按不完整处理。

CPU验证记录：真实CLI/本地HTTP测试覆盖成功、HTTP503完整body、非法JSON、NaN/Infinity及数值溢出原文保存；ERQA ±8题（恰好2pp）通过、±9题失败，失败覆盖阻断；目录复用与源漂移拒绝。既有Wave2 r3响应在本地以原作者评分器重算，Cal **82.4112222748262** / Int **66.08398222582059** / Cap **74.2476022503234**、1904/2352、MAE **0.4026147325655452**，与原质量JSON精确一致。聚合器以同一历史r3复制到三个临时夹具路径验证三轮汇总代码，明确属于CPU测试数据；实际最终三轮仍未执行。

24交付候选接口（已完成内部flag A/B，生产采用待完整对照，当前生产清单保持f810）：head installer SHA `b09cd6025b0aa63ae0410781f3c611e9d512f7ef939cb78dab41b8413a19d406`；guardadapter installer `1ed35f0a24f8b23d8ac099992d703aca20d7f1460c89ff02c30635dfde6a219c`，guard post `36eaff440fd58063a3311b2efabfd1604b4a284cd24f3b7ac3cc4e7fe177fcc7`，仅`--check`及`CLEF_JOINT_ALLOWED_POSTIMAGE_SHA=aa3e17585f8bd60a4d05e342abc7e2773c79a0b2d49f0370d15e29bcc6b72503`放行。23已放行的最新AB runner commit `c0a176d14d66836226202fbb2602b510776d103a` / SHA `127e47aec47b30c461de69f37355c8149a045ae32127f7a2afde39d7a3caa648`，staging `/workspace/yangchen/work/scratch-24/joint-fused-service-ab-c0a176d`；`ab.json` schema `joint-fused-service-ab/1`，每臂provenance/label/samples/error/stop_error先落盘。该AB内部flag0/1数据与f810生产对照分开引用。

补充执行前输入冻结：已只读核对23个模型运行文件大小，10个非safetensors文件（含tokenizer.json、processor_config.json、chat_template.jinja）SHA全部匹配下载manifest；13个权重文件维持原完整哈希证据与本轮大小核对。封装在首个请求前复制OD完整请求列表或ERQA完整样本JSONL到`expected-inputs.json/jsonl`，中断后据此核对固定分母及缺题。


## 7. 候选实测证据更新：aa3e 内部 flag A/B

27已完成 `/workspace/yangchen/logs/joint-fused-service-ab-27-c0a176d-first`，runner为上述 `c0a176d` / `127e47ae…`。UTC `2026-11-18T18:52:21Z`～`19:07:03Z`，对应北京时间 `2026-11-19 02:52:21`～`03:07:03`。4轮ABBA顺序、每臂每输入3次预热+5次正式，overall RC0；8次stop均无alive/unknown/zombies。24结果文档位于commit `61199e2765a843fe6e2f716fa12ff6cca163167e` 的 `docs/joint-fused-service-ab-result-2026-11-19.md`，完整commit和文档SHA已写入manifest。

28只读取得原始 `ab.json` 与 `restore.json`，按每轮5次正式样本中位计算配对差，再取四轮中位，复算如下：

| 输入 | 配对中位 Δms（flag1−flag0） | 配对中位变化 | 固定输入质量 |
| --- | ---: | ---: | --- |
| od-0105 短文本 | +0.067995 | +0.065613% | 两臂各20个正式样本，全valid，答案签名集合相同 |
| od-1534 长文本 | −1.039965 | −0.668169% | 两臂各20个正式样本，全valid，答案签名集合相同 |
| erqa-0000 图文 | −0.820335 | −0.750630% | 两臂各20个正式样本，全valid，答案签名集合相同 |

三个输入两臂各只有一个唯一答案签名，最大数值差0.0。图文其中一轮+9.198ms，保留四轮全部差值。该结果明确限定于同源aa3e的flag0/1增量；生产f810总收益与最终全量S1/S2、OD2352/ERQA400验收仍待执行，winner继续pending。

恢复label为 `d0-restore-20261118T190530Z-150b9178`，`verify_ok=true`；revert/service_check_revert/restore全ok，health/workers/text/image均RC0且ok；恢复源f810、原guard765956。历史PID只作原件身份，下一次执行重新抓fresh运行期树。

原件身份：`ab.json` 294,790字节、SHA `c160c95074acd679f72501eda5f9d6e132aa99cb5d836e37b5368faa90d9d003`；`restore.json` 5,375字节、SHA `fe08de720b3fef8dd61fbbe6dd0cc07e06aa8ea2e20edcb3cfb055d40919080f`。此次更新仅扩充候选证据元数据；冻结phase、生产源码、guard、env、argv、评分门和所有runner保持原值。

23独立复核已并入上述61199e2文档：24份raw JSONL共192行（120正式、72预热），HTTP200/valid/finite、bodySHA对齐及跨8臂答案/概率精确相等；28另外只读重计192/120/72与全valid均匹配。此次只更新结果文档引用及覆盖元数据。

## 8. clean-parent 路径复核与显式 baseline 第1轮

27的只读预检原件 `/workspace/yangchen/logs/final-acceptance-preflight-27-20261118T192040Z/selected-env-mismatch.json` 显示当前OPP/LD字符串累积；`frozen_check`全过，`service_check`严格阻断。恢复流程的 `env=dict(prod_env)` 继承运行态，launcher重复source CANN/ATB，入口再次前置vendor，解释了新增重复链。历史成功四phase的expected保持原值；相同unique路径顺序本身不构成实际库变化证据。

28已完成CPU只读构造验证：从仅含下列两项的父环境实际source原`activate_env.sh`，启动venv Python，抽取生产`inference.py`原AST中的`qkv_norm_environment()`执行其stdlib路径构造。没有执行整个模型入口、installer、推理或服务生命周期；`sys.modules`确认torch/torch_npu/fla_npu子模块/vllm/triton均未加载。四历史phase的OPP **9项**、LD **12项**与expected逐字节一致。全局`/usr/bin/python3`的torch spec为空；原ATB探测选择历史已采用的cxx_abi_1。

```json
{"PATH":"/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin","CLEF_LAUNCH_CONFIG":"s1"}
```

FLA的`.pth`钩子先加入一组FLA路径，入口再加入一组；因此成功expected中的两组FLA具有源码依据。干净父环境保持`ASCEND_CUSTOM_OPP_PATH`/`LD_LIBRARY_PATH`未设置，CANN/ATB/入口自行构造完整链，保留CANN与三个driver库目录；历史LD链上的`libhccl.so`路径已通过只读文件存在检查。此处完整执行建议供27下一合法切换窗口使用，**本轮没有执行launch**：

```bash
env -i PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin CLEF_LAUNCH_CONFIG=s1 \
  /bin/bash /workspace/yangchen/logs/perf-wave2-20261118/launch.sh "${CLEF_UNIQUE_LABEL:?本阶段新唯一label}"
```

B8/B16/ERQA在此命令末尾使用第5节原覆盖参数。27监督器中等价调用是 `subprocess.run(['/bin/bash', launch_path, fresh_label, *phase_overrides], env=seed, ...)`；父环境只取上列显式seed，每次由launcher完成一次activation。当前服务继续受原门保护，等待其自然合法恢复窗口。

20授权闲时先做正式f810完整S1第1轮后，新增 `--single-baseline-round` 参数：仅与`--phase s1`组合，并在请求前后额外校验head等于清单的生产f810 SHA。清单`phases.s1.rounds=3`与默认最终至少3轮门保持原值。该参数从原命令序列中选择第一轮benchmark及作者评分，原2352请求/顺序/C1/warmup0/timeout300/全概率/journal/严格身份与质量门全部沿用：

```bash
/workspace/yangchen/venvs/clef-py312/bin/python "$CLEF_ACCEPT_CODE/runtime/final_acceptance.py" \
  --manifest "$CLEF_ACCEPT_CODE/runtime/final_acceptance_manifest.json" \
  --phase s1 --single-baseline-round --output-dir "$CLEF_BASELINE_ROUND1" \
  --identities "$CLEF_READY_IDS" --launch-manifest "$CLEF_READY_LAUNCH" \
  --owner-evidence "$CLEF_OWNER_EVIDENCE" --execute
```

上列实际执行仍置于27已审客户端监督器之下。监督器最小适配为显式接受并转发`--single-baseline-round`，并固定本次新runner SHA；其进程组/截止/cleanup逻辑继续保持。未带`--execute`时打印完整argv而不执行。

新目录输出仍为 `c1-r1.json`、`c1-r1.jsonl`、`c1-r1-quality.json`及所有输入/身份/步骤记录。`execution.json.scope=baseline_single_round`；`acceptance.json`明确写入`scope=baseline_single_round`、`completed_rounds=1`、`required_final_rounds=3`、`final_s1_complete=false`与`baseline_p95_ms`。正式第1轮可保留为后续相同冻结协议/源配置下三轮汇总的原件，最终汇总须核验至少三份独立完整运行的身份及时间窗口。该参数不建立新的生命周期，也不把单轮标记为最终完成。

验证：9项CPU回归全部通过；新增真实CLI测试验证单轮argv恰好等于默认S1前两步、原manifest未变、S2组合拒绝；文件夹具验证单轮完整质量通过仍`final_s1_complete=false`，默认最终缺轮报错，篡改rounds=1仍被默认最终至少3轮门拒绝。已有27监督器v4来源为27提交`ab2903db592dd1e690f0f9ac9ad0b437ef96365e`，supervisor `03409c78…db41` / test `f9034f72…17f`；单轮转发适配后的身份以27后续审定tuple为准。

## 9. 06:30–08:00 执行预算（计划值）

下表的实测耗时只取历史成功原件，预算和截止为建议调度值，最终由27的绝对deadline监督器落实。S1沿Wave2五轮，B8 C8与B16 C32沿11月16日完整扫描；后两者的软件优化版本较早，明确作为耗时估计依据保留。

| phase | 历史请求wall（秒） | 建议北京时间窗口 | 窗口预算 |
| --- | --- | --- | --- |
| S1完整3轮 | 单轮260.316–272.046；三轮约780.95–816.14 | 06:30–06:48 | 18分钟，含冻结/必要恢复/前后断言/3次作者评分 |
| S2 B8 C8、C16 | 175.3876 + 140.5223 =315.9099 | 06:48–07:00 | 12分钟，含切换与两点质量；client建议最多6分钟，其余留生命周期 |
| S2 B16 C32、C64 | 138.3264 +134.6933 =273.0197 | 07:00–07:12 | 12分钟，含切换与两点质量；client建议最多6分钟 |
| ERQA400 | 86.7478，0失败/0重试 | 07:12–07:24 | 12分钟，含16图配置切换/完整原始质量/verify-requests；client建议最多5分钟 |
| 最佳配置恢复与汇总 | 最近restore从label时间至health约92秒，属于一次历史窗口 | 07:24–07:40 | 16分钟，保留绑定stop/启动/wait/health/workers/text/image及原件封存 |
| 交接保留 | — | 07:40–08:00 | 20分钟供最终核对和异常恢复；保持新增候选已停止 |

历史请求总wall约24.9分钟；上表总执行/恢复预算70分钟。每阶段将启动、客户端、清理分别限制在本阶段剩余预算内，下一阶段只在`clients_exited=true`、严格身份和资源门通过后进入。单次HTTP 300/600秒与wait脚本上限继续按原协议记录，外层绝对截止可以提前终止本阶段；到期原件保留并记不完整，成功覆盖与质量门仍按完整分母判断。未完成项据实上报，恢复窗口保留。

具体原件：Wave2 `s1-final/r1..r5/s1.json`、`s2-b8/s2-b8c16.json`、`s2-b16/s2-b16c64.json`、`erqa/summary.json`；早期B8 C8 `/workspace/yangchen/logs/gdn-data-path-20261115/seqs8-test/c8.json` SHA `2f4ab895363213abbae9514ab6989d45f65353fe07b36d06d7cb9befd67286bd`；B16 C32 `.../seqs16-test/c32.json` SHA `2ff82db4074c9dc39f47bf4fbe649a4f9302ce86cb598a5ed1dbc63055991825`。最近恢复为第7节 `150b9178` 历史原件；本计划没有把历史耗时记为最终测量。

## 10. 独立f810 source-control入口（已审计划，待27执行）

23有限核通过24提交 `ebca7694ac8e956ec4781e2112c27aa9910ef6be`，stage为 `/workspace/yangchen/work/scratch-24/joint-fused-source-control-ebca769`；runner SHA `a1c502a3a709b836d31dc994a156f702d6a826ce5f2854beeec9ac50701f7299`，test SHA `bd1a50343b4e6851d25d0ab74161ead640d323c423527abe6a3166e7857b16b5`。24报告17/17 CPU通过，23实读真实restore成功路径与端口门；28核对git blob及远端字节一致。9fb/148/683已被此版本替代，不作为执行入口。

该独立模式固定A=f810/flag0/无allow，B=aa3e/flag1/exact allow，四轮ABBA、每输入5次正式+3次预热。已有第7节结果仍限定aa3e内部增量。两实验臂env、输入、评分保持原协议；最终restore采用第8节完整clean seed，由原launcher一次activate。当前最终验收清单仍固定f810，winner仍pending。

27执行前须在外层fresh目录保存expected完整副本并核对，再以真实 `--clean-env-expect` 参数传入。原件 `/workspace/yangchen/logs/perf-wave2-20261118/s1-final-r1/launch-manifest.json`：12933 bytes，SHA `fcbdcbf11521e1ae499c1ac5329eed990d63f3e8a75a99b5fbb6b4df1dcb0230`。两项均读取 `service_environment` 中的原字符串，其value SHA按UTF-8字节、不加换行：

| 键 | 项数 | value SHA-256 |
| --- | ---: | --- |
| ASCEND_CUSTOM_OPP_PATH | 9 | `7b79c5dbfac4fa09692cf902c3cb6f4dc6b9db4ace099144e4df488515869ef1` |
| LD_LIBRARY_PATH | 12 | `ca27dbb7cd79489e2e2c0fc25185f5aa35b83ee3d05a424c482c0a5758be6b6c` |

该外层前置检查补足runner在最终restore时才读取expected的时序。restore四检后，runner从fresh root `/proc/PID/environ`逐字节比较，保存expected/actual/compare三份JSON，失败阻断且不fallback。生产源、guard、最终三轮门和四phase执行参数均未因本次候选元数据更新而改变。
