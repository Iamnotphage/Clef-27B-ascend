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
