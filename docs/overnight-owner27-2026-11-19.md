# 2026-11-19 夜间优化：owner27 实测记录

用户授权持续优化至 08:00 CST，06:30 停止新候选，随后执行至少三轮最终 S1、S2 和完整 OpenDecisions / ERQA，07:40 前恢复最佳服务。质量约束为 OpenDecisions 三项指标及 ERQA accuracy 相对 baseline 的绝对波动均不超过 2pp；固定 BF16 与既有软件组合，禁止答案缓存。全部设备操作由 owner27 串行执行，限 A3 物理卡 7 的逻辑 NPU 0/1。

## 交接恢复

20 明确确认 21 已退出并激活 27 后，27 执行经 23 审核的单次恢复脚本。旧恢复 shell PID1190564 已自然退出，没有发送信号。成功读取 `/proc`、端口查询和 `npu-smi` 后确认 18143 closed、卡7无运行进程。旧候选源码 `aa3e1758…` 通过原 apply manifest、installer 和已核 f810 备份回退。

恢复脚本 SHA-256 `f0b679f65b9fcaef0a59aa9fa0a84d9ed978393c55738c9e5716a74aa4decff5`。原件：[handoff](../Clef-27B/logs/overnight-20261119-owner27/handoff/complete.json)；远端 `/workspace/yangchen/logs/handoff27-restore-20261118T183212Z-dc78478b/`。归档脚本保留此次具体身份和路径，仅用于审计复现依据。

远端记录 02:32:12–02:33:45 CST 完成恢复，外层 SSH rc0。唯一标签 `d0-restore-20261118T183212Z-36d81642`，root1192060，双 worker1192211/1192212。health ready、服务树内双 worker、文本 score、图文 choice 四项均 rc0/ok=true。源码 SHA-256 `f8101b98228bbd472a63a1efe6827583e35c6cc9dede4379b925f2b016670863`，smallQ kernel 不存在。启动为 TP2/BF16/eager/maxlen4096/maxseqs1/tokens4096，关闭 prefix caching 与 chunked prefill。

## smallQ 首格：数值通过，整头负收益

候选 commit `2e8af5d59fc88de8922276ed4f55db187b7e2fea`，监管 wrapper `5c045eb1427e9cb902867bd7147c0d636ce91e1c`。完整参数与逐文件 SHA 见 [冻结命令](../Clef-27B/logs/overnight-20261119-owner27/smallq-first/run-frozen.sh) 和 [window](../Clef-27B/logs/overnight-20261119-owner27/smallq-first/window.json)。远端 `/workspace/yangchen/logs/joint-smallq-27-2e8af5d-first/`。

真实 joint head 权重、合成输入。六组 core、三组完整 head（每组调用顺序 Q=[4,4,1,1,1,1]）和两组 fallback 冻结门全部通过；有限值检查通过。core 最大绝对误差 ≤0.001953125，head 最大 0.0009765625，概率最大 0.00019699335，fallback 逐位一致。首调用/JIT 单列，未计入稳态。

各形状四轮 AB/BA/BA/AB，每臂每轮20个原始 wall/device 样本。下表为逐轮中位数之差的中位数，candidate−reference，单位 ms。

| 范围 | L | Q/候选数 | device Δ | wall Δ | 四轮方向 |
| --- | ---: | ---: | ---: | ---: | --- |
| core | 307 | 1 | -0.029190 | -0.036215 | 4/4 更快 |
| core | 307 | 4 | -0.063095 | -0.059882 | 4/4 更快 |
| core | 967 | 1 | -0.042280 | -0.043498 | 4/4 更快 |
| core | 967 | 4 | -0.054865 | -0.052645 | 4/4 更快 |
| core | 1595 | 1 | -0.037155 | -0.027660 | 4/4 更快 |
| core | 1595 | 4 | -0.031550 | -0.026165 | 4/4 更快 |
| head | 307 | 4 | +0.162455 | +0.163128 | 4/4 更慢 |
| head | 967 | 4 | +0.140460 | +0.154033 | 4/4 更慢 |
| head | 1595 | 4 | +0.119930 | +0.122842 | 4/4 更慢 |

结论：该版本保持未采纳；完整 head 的稳定负收益支持先做 CPU 归因，暂不进入服务 A/B 或全量质量验证。

probe rc0、gate_passed=true，清理 reaped=true/unknown=[]/residual=[]，随后 revert f810 并删除 kernel。窗口 overall_rc0；02:36:15 CST 恢复四检闭合，标签 `d0-restore-20261118T183442Z-d3301487`，root1193641，worker1193787/1193788。完整 [原始 probe（gzip）](../Clef-27B/logs/overnight-20261119-owner27/smallq-first/probe/head-probe.json.gz)、[配对汇总](../Clef-27B/logs/overnight-20261119-owner27/smallq-first/paired-summary.json)、[恢复原件](../Clef-27B/logs/overnight-20261119-owner27/smallq-first/restore.json) 已归档；23 已独立从 raw samples 重算并核验。

## N1 ARon/cores1：数值与生命周期通过

owner27 入口 commit `397d8fb4751230438b4c11012e9a1db49b00558b`，payload `f548b3689b91cfa2e81f5ba5282ef022aa050cc8cc58542b214c0d5b32eb409e`，plan `59da5843df72fdd1b668302d5bd1d0c711768360beb14b7488e5802910f0f8a6`。实际执行 owner27，冻结 runner 元数据的 owner21 作为兼容标记保留。输入 rows967、seed20261118、cores1、use_hccl1；包、adapter、runner 和固定门保持原冻结字节。

远端 `/workspace/yangchen/logs/n1-v2-aron-c1-owner27-20261119T0230/`。两 rank 的输入元数据与先前 ARoff 格相同；每 rank 12 个真实二进制的 SHA/大小均通过独立校验，reduced、Y、residual、rstd 的 candidate/ref 逐字节相等。两 rank reduced SHA 均为 `02215064e0e393b3ab21a3bf0aca14c51281435ce74b3840e26647810adbfcd1`，固定 shape/dtype/finite/混合误差门全部通过。诊断槽覆盖 [0,967)，FNV 校验和通过，余47槽全零。23 与25分别完成原件/CPU复核。

probe child_rc0、无 timeout、reaped=true、unknown/residual为空；独立 vendor 已退役。窗口 overall_rc0，02:38:31 CST 完成四检恢复：label `d0-restore-20261118T183704Z-b8572d6f`，root1195216，双 worker1195493/1195494。源码 f810，guard 76595657，health/workers/text/image 均 rc0/ok=true。

原件：[window](../Clef-27B/logs/overnight-20261119-owner27/n1-aron-c1/window.json)、[probe](../Clef-27B/logs/overnight-20261119-owner27/n1-aron-c1/probe/head-probe.json)、[rank0](../Clef-27B/logs/overnight-20261119-owner27/n1-aron-c1/probe/cell/rank0/result.json)、[rank1](../Clef-27B/logs/overnight-20261119-owner27/n1-aron-c1/probe/cell/rank1/result.json)、[恢复](../Clef-27B/logs/overnight-20261119-owner27/n1-aron-c1/restore.json)。大体积原始二进制保留远端；每文件哈希/大小在 rank 元数据中。

本格证明相同输入下 cores1 ARon 的数值与生命周期通过；多核和性能继续单独验证。

## head 服务 A/B 首错：请求类型与 exec 身份快照

首次服务窗口 `/workspace/yangchen/logs/joint-fused-service-ab-27-8b46-first/` 使用 `8b46a2d57f0c91493d614e64218f53190983fc74`，rounds4/reps5/warmup3，外层 SSH rc1。原始 [ab.json](../Clef-27B/logs/overnight-20261119-owner27/head-ab-first-failure/ab.json) 与首臂响应已保留。

启动 guard 适配通过，第一臂 OFF 成功启动。冻结 `od-0105` 请求实际为 choice，8 个原始响应均 HTTP200/choice；runner 对文本统一写入 expect=score，触发响应门拒绝。该错误发生在包装契约层，本窗口没有完整 A/B 样本或服务收益结论。

清理的 [严格 stop 原件](../Clef-27B/logs/overnight-20261119-owner27/head-ab-first-failure/stop-cleanup-arm.json) 记录 `saved cmdline mismatch`：同 PID1196772、start394415908、boot393ecc…、PGID1196772，从启动 `setsid nohup bash Clef-27B/start.sh` exec 为 `python -m vllm.entrypoints.cli.main`，runner 保留了早期 cmdline。严格 stop 正确拒绝，未发送信号。runner 已将磁盘 head 与 guard 回退到 f810/76595657；运行中服务仍加载候选源码，须完成一次精确停服和新标签四检恢复。

24 负责一次修复请求 schema 类型契约与预期 exec 转换后的身份快照。27 的恢复增量经23核验后执行，脚本 SHA `8ee5efaeeee0fb53f742c94784330ea482033e278a512b848160f0171c810489`，绑定身份文件 SHA `d436fef70394227cd9309caec02a48a2dd373f0b2f825d2c2a898d6b6c0f5fd5`。原标签 api.pid、PID/PPID/start/boot/PGID、完整运行 argv 与 flag0/allowaa3e/s1 均 fresh 核验；仍调用原 d0_stop 严格契约。

恢复远端目录 `/workspace/yangchen/logs/handoff27-ab-recover-20261118T184552Z-428a533f/`。精确树1196772/1196897/1196898/1196925/1196926均由 TERM 退出，killed/refused/survivors均为空。02:47:28 CST 恢复四检完成，外层rc0、source f810、guard76595657；新 label `d0-restore-20261118T184556Z-5392095c`，root1198436，worker1198610/1198611。原件：[完成](../Clef-27B/logs/overnight-20261119-owner27/head-ab-first-recovery/complete.json)、[stop](../Clef-27B/logs/overnight-20261119-owner27/head-ab-first-recovery/stop-record.json)、[四检](../Clef-27B/logs/overnight-20261119-owner27/head-ab-first-recovery/restore.json)。后续 N1 cores2 已按20授权与23核定开始；head A/B 等待新版本核验。

## N1 ARon/cores2：两段边界通过

20 授权且23核准后，使用独立 bundle `clef-n1-25-window-v2-aron-c2-owner27`，来源 `f16a15107e970bf1d4bfee72f8f7a7345daa6767`，payload `697a7538b218b6c059b6b6f162fc75266ac53fdb85e4e6ecfb27139e9415b478`。相对 cores1 仅变更 cores=2；输入、包、adapter、固定门与监管流程保持。

两 rank 各12个真实 bin 的 SHA/大小核验通过；reduced/Y/residual/rstd 与本格 reference 及 cores1 原始字节相等，固定门全过。diag 直接解码证明 slot0 [0,483)、slot1 [483,967)，四个有效槽的 FNV 校验和通过，剩46槽全零。rstd 第481–484行在两rank、reference、cores1间均逐位相等。该格覆盖两段行边界和已观测的跨核 reduced 可见性，性能保持未测。

远端 `/workspace/yangchen/logs/n1-v2-aron-c2-owner27-20261119T0245/`；窗口 overall_rc0，probe child_rc0、reaped、无未知/残留。vendor退役16文件逐SHA匹配。02:49:50 CST 恢复四检闭合：label `d0-restore-20261118T184817Z-ea064140`，root1200025，worker1200187/1200188。原件：[window](../Clef-27B/logs/overnight-20261119-owner27/n1-aron-c2/window.json)、[probe](../Clef-27B/logs/overnight-20261119-owner27/n1-aron-c2/probe/head-probe.json)、[恢复](../Clef-27B/logs/overnight-20261119-owner27/n1-aron-c2/restore.json)。23已独立闭合数值、diag、vendor与恢复证据。

## head 服务 A/B 四轮完成：长文本内部增量，整体待定

最终 runner commit `c0a176d14d66836226202fbb2602b510776d103a`，runner SHA `127e47aec47b30c461de69f37355c8149a045ae32127f7a2afde39d7a3caa648`。远端 `/workspace/yangchen/logs/joint-fused-service-ab-27-c0a176d-first/`，外层 SSH rc0，ab overall_rc0、四轮八臂全部完成，八次臂 stop 成功。固定请求 body SHA 不变，24份原始 JSONL 共192条（120正式、72预热），全部 HTTP200/valid/finite；三条固定输入在八臂间的答案与概率精确相等。

每臂每轮5个正式样本，逐轮中位数做 ON−OFF 配对，再取四轮中位数：

| 输入 | 配对中位 Δ ms | 配对中位 Δ % | 更快轮数 |
| --- | ---: | ---: | ---: |
| od-0105 / 307 token | +0.067995 | +0.065613 | 2/4 |
| od-1534 / 967 token | -1.039965 | -0.668169 | 4/4 |
| erqa-0000 | -0.820335 | -0.750630 | 3/4 |

ERQA末轮 Δ +9.19843 ms，短文本方向各半。结论限于同 aa3e postimage 上 flag0/1 的内部增量；aa3e OFF 已共享 routing KV，生产 f810 总收益和完整质量仍待单独对照。当前证据不足以升级为全域稳定收益或采用结论。

03:07:03 CST 恢复四检闭合：label `d0-restore-20261118T190530Z-150b9178`，root1212699，双 worker1212859/1212860；source f810、guard76595657。ab SHA `c160c95074acd679f72501eda5f9d6e132aa99cb5d836e37b5368faa90d9d003`，restore SHA `fe08de720b3fef8dd61fbbe6dd0cc07e06aa8ea2e20edcb3cfb055d40919080f`。23已独立重算全部raw并确认恢复。

原件：[ab](../Clef-27B/logs/overnight-20261119-owner27/head-ab-c0a/ab.json)、[配对汇总](../Clef-27B/logs/overnight-20261119-owner27/head-ab-c0a/paired-summary.json)、[恢复](../Clef-27B/logs/overnight-20261119-owner27/head-ab-c0a/restore.json)，24份raw gzip及stop/launch/wait一并归档。

## N1 ARon/cores24：正确性通过，计时未见稳定收益

来源 `0e935d06e1368c788e9d41c1e61bb1ed9e217cd0`；payload b782e490…14977、plan8a380481…4a5e3。远端 `/workspace/yangchen/logs/n1-v2-aron-c24-timing-owner27-20261119T0310/`。先双rank固定门与raw/diag，再同窗口条件计时，全部通过。23独立校验42个bin的大小/SHA/shape，candidate=reference=cores2；计时终态与本窗cell相等。24个有效diag槽覆盖 [0,967)，17段40行、7段41行；FNV、rank、通信字段和余24零槽通过，rstd967行均有限正值。

每rank48calls，first2、warmup6、正式40分列。96条raw经23重算与summary逐项一致。正式20pair：critical wall pooled median 0.567595→0.578880 ms，paired delta +0.012930 ms、速度比0.977116；slow-rank event 0.449630→0.478040 ms，paired delta +0.027530 ms、速度比0.941371。四轮critical中位差 +0.02558/+0.03787/+0.00660/−0.28653 ms，末轮两臂均升高。首call critical baseline10.57658/candidate303.83746 ms独列。该格没有稳定收益证据，保持未采纳，不扩cores48。

外层rc0，window overall0、六step rc0，probe reaped/unknown0/residual0，vendor完整退役，四检恢复通过。label `d0-restore-20261118T191020Z-4516976e`，root1214582、workers1214738/1214739。原件：[window](../Clef-27B/logs/overnight-20261119-owner27/n1-aron-c24/window.json)、[timing summary](../Clef-27B/logs/overnight-20261119-owner27/n1-aron-c24/probe/timing/summary.json.gz)、[restore](../Clef-27B/logs/overnight-20261119-owner27/n1-aron-c24/restore.json)；大体积bin保留远端并由原始metadata绑定。

## smallQ A/C/B：接入路径净开销

诊断来源 `13fd807a2349aad36f3cf972dc82dbc118c01f17`，profile `06ede1a10230cc3b83d66180e99f51faf3fbb3dd6a0a6544cfa954cf68d4043c`，B四个冻件不变。A为f810原MHA，B为smallQ adapter+Triton，C为相同adapter/guard/projection且进程内显式换SDPA backend。同实例/同真实权重storage、9格四轮三臂各20样本，共2160正式raw，均有限正值。11个gate、原域及两个fallback全部通过，same_weight_storage_verified/frozen_files_unchanged为true。

| 完整head长度 | C−A wall ms | B−C wall ms | B−A wall ms | B−A device ms |
| --- | ---: | ---: | ---: | ---: |
| 307 | +0.1538175 | +0.0150325 | +0.1688500 | +0.172285 |
| 967 | +0.1704150 | +0.0381725 | +0.2004000 | +0.211025 |
| 1595 | +0.1513450 | +0.0569075 | +0.2082525 | +0.211830 |

表中为四轮中位数差的中位数。C−A/B−A全部4/4变慢，B−C各3/4变慢且范围跨0；六core B−C全部4/4更快。结果支持接入/投影路径净开销这一归因范围，host_submit包含内部同步，当前不能指认纯Python或单条guard。26后续仅提出每调用元数据读取去重的单变量方案；本诊断无服务采用结论，生产保持off。

远端 `/workspace/yangchen/logs/joint-smallq-acb-27-v1-first/`；probe SHA `79d115accb3822425db30c40ccd11775213e45ed5e3ffbecbf5d46c09c203caa`。外层rc0、window overall0、cleanup无unknown/residual，revert f810且移除kernel，四检恢复通过，label `d0-restore-20261118T191302Z-5067eaf0`。23与26均独立重算raw。原件：[probe gzip](../Clef-27B/logs/overnight-20261119-owner27/smallq-acb/probe/head-probe.json.gz)、[配对统计](../Clef-27B/logs/overnight-20261119-owner27/smallq-acb/paired-summary.json)、[恢复](../Clef-27B/logs/overnight-20261119-owner27/smallq-acb/restore.json)。

## 最终验收准备

28冻结tuple `02546ae2088d76127ac11a0fb6c51a82dbb67c9a` 已部署至同名独立远端work目录，8文件逐SHA一致，manifest `b42cc6f4a456e2ffade5479c392735aa69f7ef6ca33028800deec89c1d70b2b7`、runner `e6ddb887c92c4f34a4d05502b7ff3317b1b10845f20bd7d63e8d95d34b049ff3`。只读 frozen_check 全过；当前服务的 OPP/LD 路径从多轮restore继承累积，service_check按原严格门拒绝。首错保留在远端 `final-acceptance-preflight-27-20261118T192040Z`，由28核干净启动环境；正式最终评测尚未执行。


最终客户端监督器 [final_acceptance_supervisor.py](../runtime/final_acceptance_supervisor.py) v4 已经23有限核通过，SHA `03409c78b0e1d39f44299785247586a545c171a020ed78d886c441559c59db41`。默认plan-only，固定28 runner与5c helper文件SHA；隔离加载5c清理函数，并明确使用严格/proc查询适配器，清理前核整组完整cmdline、逐信号核PID/start/boot/PGID。服务PID保护、未知/查询失败阻断、最终clients_exited确认后才能切服务。此脚本仅监管评测客户端，不操作服务。

绝对monotonic截止锚点在main准备前固定，准备、Popen及循环共享；到期拒绝启动、慢启动计入预算。真实Linux CPU 11项测试通过：正常/非0退出、超时双进程、已观测孤儿清理、查询失败、身份变化、保护服务、清理查询失败、PGID/cmdline前置门、准备过期、慢启动。v2在组KILL后的cmdline清空瞬态误记unknown，首错原件保留；v3将完整cmdline门置于清理前，v4收紧绝对deadline。测试仅创建自身CPU短进程，未导入NPU模块。证据 [CPU日志](../Clef-27B/logs/overnight-20261119-owner27/final-supervisor-cpu/final-supervisor-cpu-v4.log)，原failed日志与超时5次复核另列。正式评测尚未执行。


单轮基线转发 v5 经23有限核通过：supervisor SHA `172c283827d75074bf339fd5ba6326006d2b3755dd6b9c5c31d3f030d6f31876`、test `044a4ef7033a8a2d7929aa3ea3413a0b1f0b1883274a18ba3cd70fada82e3ff0`。本体只更新28单轮runner pin为 `7d32e5c3e7977efa7222d527792887e50b541349d3ef6390663470180e011cab` 并显式转发仅s1的 `--single-baseline-round`；所有监督/清理函数AST保持v4。新增3项CLI CPU测试通过。默认完整phase与最终S1≥3轮门保持，单轮输出单独标记 baseline_single_round/final_s1_complete=false。旧v4独立stage与commit保留。单轮仍待合法clean恢复和严格env检查通过后执行。


## smallQ B/D：元数据去重取得局部收益

来源 `9b803fefba4d1aa3beab1125bb4cbc7a3daae12f`，profile71f7e7a1…51bf3b；D只在每调用内复用shape/device/stride，同一KV对象校验一次，动态env/training/grad/参数dtype-device/原domain继续检查。kernel/tile/projection不变、无跨调用缓存。原11gate与额外D/B bitwise、defaultoff、fallback通过，single_variable/storage/frozen verified均为true。

9格×4轮×2臂×20共1440条正式raw，BD/DB/DB/BD。完整head D−B四轮配对中位wall：L307 −0.0458875 ms、L967 −0.0369825 ms、L1595 −0.0495500 ms；device分别 −0.0439849/−0.0390600/−0.0471551 ms，各4/4更快。范围仅相对冻结B的局部改善，生产f810总路径差待同窗A/D直接对照，暂不升级服务。

远端 `/workspace/yangchen/logs/joint-smallq-bd-27-v1-first/`，外层rc0；probe8aea817c…dfcd，window862af0c0…f136 overall0、cleanup reaped/unknown0/residual0，revert f810、删除kernel，四检恢复通过。label `d0-restore-20261118T193157Z-1b3839a2`，root1220123，workers1220262/1220263。原件：[probe gzip](../Clef-27B/logs/overnight-20261119-owner27/smallq-bd/probe/head-probe.json.gz)、[配对重算](../Clef-27B/logs/overnight-20261119-owner27/smallq-bd/paired-summary.json)、[恢复](../Clef-27B/logs/overnight-20261119-owner27/smallq-bd/restore.json)。

## Grouped API首格超时与独立clean恢复

来源 `e529a5ec858273482a14355a7e36d456d0935b34`，仅固定rows967、seed20261118，两rank，合法streamMode1与冻结NZ生产参考。远端 `/workspace/yangchen/logs/grouped29-first967-owner27-20261119-v3/`。NZ baseline和ND diagnostic原始输出均已落盘；candidate未产生输出。240s边界触发原5c监管清理，probe rc−9/timed_out=true/reaped=true。原saved只有leader1221671，清理记录unknown成员1221672/1221680/1221681，residual为空；恢复按unknown门被阻断。原window overall1及所有错误保留，不将本窗标为成功。

03:42:39 CST strict全/proc、端口和npu查询证明四PID与PGID1221671全部gone、无候选runner、18143=111、卡7无运行进程；sourcef810/guard765956，smallQ不存在。该fresh证据与原record/window分别绑定SHA，恢复没有新增signal。23核准一次脚本17f8d804…9826后，27调用已核ebca clean restore_prod及原六d0四检，外层fresh核activate3b363、历史expected12933B/fcbdcb及OPP/LD value SHA；原失败窗口不覆盖。

恢复远端 `/workspace/yangchen/logs/handoff27-grouped-recover-20261118T194732Z-e36660ba/`，外层rc0，complete.ok=true/signals=[]；label `d0-restore-20261118T194732Z-3e31e017`，root1222521，workers1222677/1222678，五PID同PGID/SID1222521及boot393ecc。四检均rc0/oktrue，OPP9项、LD12项与历史expected逐字节相等，23独立闭合。原件：[失败window](../Clef-27B/logs/overnight-20261119-owner27/grouped29-first-timeout/window.json)、[失败record](../Clef-27B/logs/overnight-20261119-owner27/grouped29-first-timeout/probe/record.json)、[fresh gone](../Clef-27B/logs/overnight-20261119-owner27/grouped29-first-recovery/earlier-fresh-gone.json)、[恢复complete](../Clef-27B/logs/overnight-20261119-owner27/grouped29-first-recovery/complete.json)、[四检](../Clef-27B/logs/overnight-20261119-owner27/grouped29-first-recovery/restore.json)、[env严格比对](../Clef-27B/logs/overnight-20261119-owner27/grouped29-first-recovery/restore-env-compare.json)。

结论限于固定首调用未完成，prepare/launch/sync尚未定位；没有Grouped候选精度、timing或profile证据。ND/NZ诊断差异单独记录，保持原NZ参考与冻结门。当时排程为ebca真f810 source-control后接rstd8和A/D；截止时执行范围见下文收口记录。


最终服务phase入口 [final_service_phase.py](../runtime/final_service_phase.py) 已完成CPU准备，尚未设备执行。默认plan-only；仅f810/原guard，固定28四phase覆盖参数与干净两键父环境。旧restore SHA、label/api.pid、PID/start/boot/PGID/完整final argv绑定后调用原d0_stop；停前strict旧PGID所有非Z成员须已归属树，停后整个旧PGID须空；未知端口或未清空卡7禁止launch。客户端前置为互斥必选：实际prior supervision记录，或显式first-phase-no-client且仅S1→S1；已观测client PID及PGID须全部退出。新服务完整argv/env/launch/info经cc7检查，再做原六d0四检与卡7双worker归属。

v3 SHA `888a8d7fac79bd243fe894c25324aca0f498e311cab91be77df1dc2cfdc0de44`，test `901d50e3226d418b91d4abe801ab5d6e254d24ee9775e1e07dbbda3c2ac22bd1`，17项CPU测试通过；所有停启/NPU命令均mock。包含停前PPID1同PGID orphan零stop/launch、停后同PGID残留零launch、缺失/非法/冲突客户端模式、query/port/source/stop失败、clean parent成功路径。23已有限核通过提交 `d4464ac2c32fc122a9c6a391a45806cdaac17916`；平铺独立stage为 `/workspace/yangchen/work/final-phase-owner27-d4464ac2c32fc122a9c6a391a45806cdaac17916`，本入口尚无实际服务切换证据。28后续文档元数据 `1e42a93610ad4338c3cdb67385a17b307bdf18c3` 经23核准，执行组合继续固定cc7/0338与监督器063109a/172c。

## source-control传输中断：远端终态待核

真f810/aa3eON窗口使用已核 `ebca7694ac8e956ec4781e2112c27aa9910ef6be`、runner `a1c502a3a709b836d31dc994a156f702d6a826ce5f2854beeec9ac50701f7299`，远端run为 `/workspace/yangchen/logs/joint-fused-source-control-27-ebca-first`。本地04:00保存的进度副本处于 `round-2-start`，已有两轮完整pair；该运行中快照的 `overall_rc=1` 保留原值，候选结束码未取得。23另报告最后一次远端观察为04:01:11 CST的 `round-3-start/error=null`，已完成三轮；该观察来源与本地快照分列。

04:15 CST回收原外层SSH会话69391，得到传输退出码255；本地 `/tmp/clef27-handoff/source-control-ebca-outer.log` 为0字节，匹配的原SSH/ebca本地进程为空。唯一owner通过既有clef入口做一次20秒有界只读 `true` 探测，返回255及 `Connection closed by UNKNOWN port 65535`。本次未修改SSH配置或凭据，未重复候选命令、未发送设备或服务信号。该错误尚不足以定位凭据失效，远端实验完成情况、源/guard状态、客户端退出与恢复四检/env终态均待实读。

04:15原件：[owner连接探测](../Clef-27B/logs/overnight-20261119-owner27/source-control-transport-interruption/owner-probe.json)，345 bytes、SHA `386824134a47603a81b64f09731a682cd02cdb8502922b227bef7a0f09839734`。完整ab/raw/stop/restore/envcompare尚无本地副本，无法据此判定winner或恢复完成。20接手平台入口核验后，27停止重连及所有后续窗口。

07:57:12 CST按20截止前授权，通过原clef入口唯一一次有界只读探测，0.0295秒返回SSH rc255，stderr仍为 `Connection closed by UNKNOWN port 65535`；随后停止重试。原件：[截止前连接探测](../Clef-27B/logs/overnight-20261119-owner27/source-control-transport-interruption/owner-probe-075712.json)，237 bytes、SHA `ce81a56b02d6e0d4b696fdc25dd13072820fbc75a2ece05a38f18ac100299ffa`。两次探测均无signal、无配置或凭据修改；传输失败与候选运行结果分列。

## 本轮截止范围与后续入口

| 范围 | 截止状态及证据边界 |
| --- | --- |
| 前述head内部flag四轮AB、N1 ARon cores1/2/24、smallQ A/C/B及B/D | 已归档各窗口原件与恢复四检。内部flag增量、ND固定参考、局部head诊断各保持原结论范围，未据此采纳生产候选。 |
| Grouped v3首格及其独立clean恢复 | 首格保持240秒超时失败、candidate无raw。独立恢复已由23核闭合四检及OPP/LD；该历史恢复服务随后进入ebca实验，其证据不代表截止时的服务现场。 |
| ebca真f810 source-control | 第四轮、完整8次stop、最终restore、envcompare及总体退出码均为unknown。最后独立观察仅覆盖三轮完成、04:01:11 CST进入第四轮；没有完整四轮结果或winner结论。 |
| 当前远端现场 | 当前服务label/root/完整身份、源码、guard、health、卡7进程与资源、客户端退出及恢复状态均为unknown，待恢复连接后fresh读取。 |
| rstd8、A/D、Grouped设备重试 | 本轮未执行；06:30新增候选截止已过，本轮不再启动。 |
| 正式f810 S1首轮、首次final_service_phase服务切换、最终全量验收 | 均未执行。CPU测试和有限源码审阅仅证明入口准备状态；本轮无S1≥3、S2 B8/B16或ERQA全量验收结果。 |

历史生产baseline与既往验收数据按其原日期、来源和配置保留，见[2026-11-17历史汇总](final-summary-2026-11-17.md)及本报告引用的既有原件；本轮状态为未完成验收。执行冻件与指标CSV保持原样，本次归档仅更新本报告、连接证据和证据manifest。

整体任务保留未完成。当前继续停止重连与新窗口；平台入口恢复后，唯一优先动作是只读同一 `/workspace/yangchen/logs/joint-fused-source-control-27-ebca-first` 的ab/raw/8stop/restore/envcompare及精确现场。发现未恢复或异常时先将实际身份与原件交23；必要恢复沿已核授权、原身份门和严格清理/恢复检查执行。恢复完成须由fresh原件证明。

## 2026-11-19 owner31 续跑收口：采用 aa3eON 为最终提交默认

连接恢复后由 31 独占执行，完成 ebca 原 run 实读（phase=finished、rc0、四轮 8 臂 192 raw 行）、rstd8 C24 timing（4/4 轮更慢，不采纳）、smallQ A/D 直接对照（core 4/4 快、完整 head 4/4 慢，默认 off）、Grouped29 v4r1 一次诊断（240 s 超时、无 candidate raw）、f810 正式 S1/S2/ERQA 全链与候选 aa3eON 同协议 S1 及 typed S2/ERQA 三格，最后 clean 恢复。20 于全部结果与 32/33 独立核验后决定 **采用 aa3eON 作为最终提交默认**。

采用理由（同窗口、同协议）：

| 指标 | f810（对照） | aa3eON（采用） | 差 |
| --- | ---: | ---: | ---: |
| S1 p95 中位（3 轮，ms） | 156.4051（156.144–157.513） | 156.4417（156.116–156.888） | +0.0366（逐轮 2 快 1 慢） |
| S1 吞吐（dec/s） | 8.5063/8.8317/8.9059 | 8.5184/8.8209/8.7227 | 中性 |
| S2 B8 C8 / C16（dec/s） | 14.7148 / 16.8259 | 14.7832 / 16.9037 | +0.47% / +0.46% |
| S2 B16 C32 / C64（dec/s） | 17.5432 / 17.5996 | 17.6572 / 17.7047 | +0.65% / +0.60% |
| ERQA accuracy | 216/400 = 0.54 | 216/400 = 0.54 | 0 题（acceptance 字节相同） |
| OD Capability（各点） | 74.2476 / 74.2790 / 73.8473 / 74.1398 / 74.1412 | 74.2477 / 74.2531 / 73.8250 / 74.1469 / 74.1418 | 全部 2PP 内 |

- 质量：aa3e 各点相对 11-13 基线最大 Δ 为 B16 C64 Cap −0.2837 pp、C16 −0.6005 pp（f810 同点），全部远在 2PP 内；ERQA Δ0。
- 性能：S1 中性；S2 四点单窗口一致小幅正（吞吐 +0.46~0.65%、p95 −0.24~−0.49%）；ERQA 同值。
- 上报口径：**只使用官方 JSON benchmark 的 `decisions_per_second`（=successful/wall_seconds）**；journal span（首 start→末 finish）重算仅作交叉核对，末位差单注，不另立指标，也不拼 f810 的 S1 数字。最终 profile 全部标注 aa3e：S1=156.4417、S2=17.7047@B16C64、OD/ERQA 同版本。

已关闭/透明保留项：

- N1/smallQ 线关闭：rstd8 C24 timing 4/4 轮更慢（配对中位 speedup 0.8606/0.8151，不采纳）；smallQ A/D core 4/4 快、完整 head 4/4 慢（+0.125~+0.152 ms），保持默认 off，无后续 D 候选。
- Grouped29 v4r1：一次诊断 240 s 超时（final_sync_enter 后设备完成等待未返回），无 candidate raw；三个 unknown 组员与 leader 均已 gone、无新增 signal，已按已核 clean restore 恢复（label cec60d1c）并四检闭合；240 s 不延长、不重试。
- 34 越界时间线：00:57:51Z 其 CPU 测试误把生产 head 切至 aa3e 并立即 revert（报告 f7a2e6d1…，state_before=postimage→preimage，ready=true）；我的候选 source-deploy 记录（01:00:13Z，state_before=preimage、sha256_before=f8101b98）证明回退先于正式候选切源；f810 基线 S1 测量（00:22–00:35Z）早于事件，测量未被污染。
- 候选 S1 两次失败：v1 为 `--repo` 参数缺少 staging 文件（未停服）；v2 为 entry `ab.state` 未初始化（停服后失败，entry recovery 回 source/guard、无 signal）；均精确 clean 恢复（c4a4a6af/be55b18c）后由修复版 00e3bdb8 成功。
- 环境继承事实：5c 窗口 restore 会继承 prod_env（重复链），不得用于最终验收；最终验收一律走 phase/typed clean 两键父环境，OPP 9/LD 12 与冻结 expected 逐字节。

本轮证据：`Clef-27B/logs/overnight-20261119-owner31/`（pre-window、rstd8、smallq-ad、grouped29、grouped-recovery-and-candidate-attempts、s1-final-f810、phase-s1-first-clean、final-chain-results、typed-chain-results）；提交 `934fff0`（f810 全链）、`da5a6d1`（typed 链）。独立核验：32（生命周期/性能/raw/恢复）、33（逐 index 逐题质量）。

最终服务目标：官方 `start.sh` 入口验证通过后为**所选 aa3e/s1**（当前临时状态仍为 f810 clean s1，label `b57139dc`，仅作为入口验证前的健康基线；入口验证与最终切换见下节计划）。
