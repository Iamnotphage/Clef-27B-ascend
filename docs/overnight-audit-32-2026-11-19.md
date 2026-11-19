# 2026-11-19 独立取证与验收核验（worker32）

角色：只读 SSH + CPU 独立核验（不执行设备/服务操作、不 import torch_npu、不发候选请求）。设备/服务唯一 owner 为 clef-27b-ascend-31，协调 clef-27b-ascend-20。本文为单一汇总记录，逐项以远端原件重算为准。

## 1. ebca source-control 原 run（已完成，非 unknown）

远端 `/workspace/yangchen/logs/joint-fused-source-control-27-ebca-first`：

- ab.json 298600 B、SHA `fd222a22…`；overall_rc=0、phase=finished、finished_at `2026-11-18T20:05:29Z`。本地拉取 71 个原件与远端逐文件 SHA-256 相等。
- 24 份 raw JSONL × 8 行 = 192 行（120 正式 + 72 warmup）。逐行核验 0 异常：全部 status 200/valid/finite/error=null；body SHA 与 ab.inputs 一致；expect 为 od-0105=choice、od-1534=score、erqa-0000=choice。
- 八个臂 × 三个输入的答案与概率载荷各只有 **1 个唯一签名**（跨臂逐位一致）。
- 逐轮 median wall（aa3e−f810，单位 ms；每臂每轮 5 个正式样本）：

| 输入 | R0 | R1 | R2 | R3 | 四轮中位 | 更快轮数 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| od-0105 (307) | +1.6908 | −12.4632 | −2.2164 | −3.3032 | −2.7598 | 3/4 |
| od-1534 (967) | +0.0425 | −1.2442 | −0.9280 | +1.2722 | −0.4428 | 2/4 |
| erqa-0000 | +4.7577 | −4.5381 | −5.7792 | −5.3048 | −4.9215 | 3/4 |

- 8 个臂 stop 各 1628 B：stop_root_identity=verified、targets=该臂 5 PID、termed 全、refused/killed/survivors 空；initial stop 精确停旧 1222521 树（5 PID）。
- 最终 restore：label `d0-restore-20261118T200407Z-ed140cf3`、api_pid 1234686、五 PID；health/workers/text/image 四检 rc0/ok；restore_env 为 clean 两键父环境；OPP 9 项 SHA `7b79c5db…`、LD 12 项 SHA `ca27dbb7…`，与 expected `fcbdcbf11521e1ae499c1ac5329eed990d63f3e8a75a99b5fbb6b4df1dcb0230`（12933 B）逐字节一致；最终 source f810、guard 76595657。
- 该服务随后由 31 的 rstd8 窗口按同五 PID 精确停止（stop identity verified、无 survivors），交接合法。
- 结论：四轮证据完整；性能混向，不足以采用，需正式全 S1 比较（已在第 5 节执行）。

## 2. rstd8（7092657 / payload 70e639 / plan 10b2）——negative，不采用

- 绑定：commit `7092657b0da55bcebd4b811a64e1e7f0b3c3ece5`；bundle plan.json SHA `10b2dd3f…`、payload_sha256 `70e6390e…`；window 六步 rc0（`2026-11-19T00:14:27Z`→`00:16:27Z`），精确停止 ebca restore 五 PID。
- cell（967 行、cores 24）：reduced/Y/residual/rstd 在两 rank 上 reference 与 candidate **逐字节相等**；rstd 3868 B=967 fp32，全有限正值 0.541852–0.606094；diag 每 rank 24 个有效槽（slot_id 0–23、行区间连续覆盖 [0,967)、其余 24 槽全零）；固定门 3/3 pass、reduced_bit_equal=true。
- timing 96 raw（48/rank = first 2 + warmup 6 + measure 40），0 异常。三种统计口径分别成立、不可混称：
  - pooled 臂中位：baseline 0.493690 → candidate 0.532265，差 **+0.038575 ms**；pooled 比值 baseline/candidate **0.927527**。
  - 逐 pair 差中位（20 对）：critical **+0.073370 ms**；slow_rank_event **+0.076180 ms**；逐 pair 比值中位 baseline/candidate **0.860647**（event 0.815067）——即 31 报的“配对中位 speedup”。
  - 逐轮臂中位差（5 样本/臂/轮）：+0.074410 / +0.059780 / +0.078010 / +0.018540；四轮同向为正。
  - 逐轮 5 个 pair 差的中位：+0.071020 / +0.077500 / +0.078010 / +0.014810（与上一项口径不同）。
- first_call 单列：candidate 281.734 ms vs baseline 6.671 ms（JIT 首调用，不计稳态）。
- 结论：四轮全部更慢，不采用；不外推 NZ 生产结论。

## 3. A/D（5e6daee / profile a7c418af）——negative，保持 off

- commit `5e6daeeecff11b58a7449fa8b823162803eff05d`；profile.json SHA `a7c418af865a3482bec1a60be0173355ddc2bc7c478e68dc65995f8ff2a51a6d`；stage f810_reference=`f8101b98…`、b_reference=`cf024199…`、head=`086357bd…`、kernel=`08ad88f6…`；apply f810→086357bd、revert→f810 双 ready。
- 1440 raw = 9 格 × 4 轮 × 2 臂 × 20，全有限正值、0 异常；orders AD/DA/DA/AD；内嵌 paired_deltas 与逐轮重算逐位一致。
- 完整 head（D−A，四轮中位差的中位）：wall +0.128125 / +0.124488 / +0.151755 ms（0/4 更快）；device +0.124540 / +0.129950 / +0.157295 ms。core 六项 4/4 D 更快（wall −0.0289～−0.0666 ms）。
- 11 个 gate 全过；A−A 与 **D−B bit_equal=true（0 差异）**；B/D 相对 A 为预期 smallQ 近似（411 元素、max_abs 0.001953125、prob 阈值内）。
- restore label `d0-restore-20261119T001747Z-f1796dbe` 五 PID 四检 ok（随后被 final phase 精确停止）。
- 结论：完整 head 退化，保持默认 off；不新造 D 后续候选。

## 4. Grouped29 v4r1（33 冻结包）——tuple ready

- work `/workspace/yangchen/work/grouped29-20261119-v4r1-cc33`（run dir 预留且执行前不存在）；branch `ao/resume-cc/grouped-20261119` commit 18666b2。
- 绑定：identity `ea042e60…`、payload `64d5a816…`、plan `66e3fda6…`、entry `4dc2b059…`、runner `dde68f34…`、adapter `23d9ab07…`（v4 原件 byte-identical）、binding.cpp `5db40ef9…`、fixed_gate `041a66aa…`、wrapper `9245e1a6…`、stop `d6d49d75…`、preimage f810。
- plan 本地重跑 print-only：输出 SHA 精确 `66e3fda6…`、rc0、argv 85 项；N1 参考 cell rank0/rank1 meta `5b2261c6…`/`a14f9649…` 与 identity 绑定一致。
- v4→v4r1 增量仅 run_probe.py 最外层异常处理器的 `trace.close()` try/except（记录 `trace_close_error`，不掩盖原异常）+ test_first_trace 真实 handler 用例 + identity/payload 重生成；其余 source/gate/adapter/binding 逐字节不变。
- 正式参数不变：80 raw/rank（AB/BA/AB/BA×10、warmup 2/臂）、固定门、NZ 参考、240 s probe、同 supervised probe PGID；hook 为 validation-only、无 vendor/系统变更、无 OPP/LD prepend。
- 唯一命令（31 执行、240 s 不扩不重试）：
  `/workspace/yangchen/venvs/clef-py312/bin/python /workspace/yangchen/work/grouped29-20261119-v4r1-cc33/supervised_window.py --owner clef-27b-ascend-27 --expect-payload-sha 64d5a81657a60f32c9fa700efb4035a3c653fcb3075381eb960e78691dd3812c --run-dir /workspace/yangchen/logs/grouped29-first967-owner27-20261119-v4r1 --execute`
- 设备执行（20 排程、S1 退出后，**失败、不重试**）：outer `/workspace/yangchen/logs/owner31-outer-grouped-20261119T003655Z`；run `/workspace/yangchen/logs/grouped29-first967-owner27-20261119-v4r1`，window `e785336e…`、overall_rc=1。
  - preflight/stop/apply ok；probe rc=−9、timed_out=true（00:37:00→00:41:00Z，240 s 上限）；cleanup killed leader 1255353、unknown 1255354/1255361/1255362（unattributed group member）、residual []、reaped=true；revert ok；restore rc=1、action=blocked_probe_residual，无 restore.json。
  - 两 rank first-call trace（rank0 `b2360475…`、rank1 `3fdfe026…`）均止于 `final_sync_enter`，无 `final_sync_return`/`first_call_complete`；prepare/workspace（16 MiB，274/283 ms）、launch、handler 均 rc0；candidate 无落盘、error=null、performance_valid=false；首调用墙钟 baseline 84.8/84.1 ms、nd_native 13.86/13.85 ms。
  - 归因（33 plog，引用）：RUNTIME `SynchronizeExecutedTask` 3 分钟看门狗在 stream 47/task 36 pending（入队后 +180 s），无 AICore trap、无 HCCL 错误 → 设备侧完成等待（kernel/MC2 通信）未返回；host 侧无失败。
  - 恢复前置（我独立 fresh 复核）：4+1 PID 全 gone、18143 connect_ex=111、无 probe/vllm/torchrun、source f810、guard 76595657、卡7 无进程。`fresh-inventory.json` `101df87c…` 的 `no_probe_like=false` 为采集 shell（1256404，含 run 路径）自匹配，该 PID 已 gone。

## 5. 最终验收：phase first clean 切换 + f810 正式 S1 三轮

- phase first：`/workspace/yangchen/logs/final-phase-s1-first-owner31-20261119T0021`，complete ok=true、phase=s1、label `d0-final-s1-20261119T002025Z-ca972eaf`、root 1252696；clean parent 仅 PATH+CLEF_LAUNCH_CONFIG=s1；旧身份=AD restore 1250992 树（5 PID）；四检 ok；**live OPP 9 项 `7b79c5db…` / LD 12 项 `ca27dbb7…` 与 expected 12933 B/fcbdcbf1 完全一致**，candidate 键无残留。clean first 门有效。
- S1 冻结组合实读命中：supervisor `172c2838…`、runner `7d32e5c3…`、manifest `0338f87b…`、phase stage `888a8d7f…`（四文件平铺，含两 test `044a4ef7…`/`901d50e3…`）；cmdline 含 `--phase s1 --execute`、**无 `--single-baseline-round`**、deadline `2026-11-19T01:10:00Z`。
- S1 三轮（每轮全量 OD 2352 @ C1），独立重算：
  - 每轮 2352 行、0 异常、index/name/request 与 expected-inputs 逐条相等、inflight 恒 1；journal 重算分位数与报告 0 偏差。
  - p50/p95/p99：r1 109.999/157.513/194.178、r2 104.129/156.405/192.821、r3 101.759/156.144/192.341 ms；dec/s 8.506/8.832/8.906。
  - 质量三轮同值（响应确定）：Calibration 82.4112、Intelligence 66.0840、Capability 74.2476、accuracy 1904/2352、MAE 0.402615；相对 baseline 74.4255 / 1906 / 0.401389 的 Δ = −0.178 pp / −2 题 / +0.001226，均在 2PP 内。r1 用冻结算分器独立重打分逐位复现。
  - supervision `1da03a19…`：status complete、clients_exited=true、client_returncode 0、cleanup rc0/reaped、unknown/residual 空、remaining_group []；execution 六步全 rc0。
- 结论：f810 正式 S1≥3 轮真实闭合；candidate（cc34 组合）与 S2/ERQA 阶段由 20 排程、31 执行，我逐相位只读核 raw。

## 6. cc34 candidate 正式 S1 组合（CPU 联检 ready）

- stage `/workspace/yangchen/work/final-acceptance-cc34-640ac23cac9aaa9a937832c2361dddd74007d195`：entry `1f98dba9…`（448 行）、candidate manifest `31202d27…`、diff `646ba8d8…`、combo `bf32b3e9…`、builder `6348ab52…`；v5 `172c2838…`、phase `888a8d7f…`、runner `7d32e5c3…`、base `0338f87b…`；f810 冻结入口未改。
- 逐路径 diff 恰 13 处：source_files[5] head f810→aa3e、[47] guard 76595657→36eaff44、source_selection.production_head_sha/guard_sha、forbidden 移除 CLEF_JOINT_FUSED_ATTN、四 phase 各 +2 env（FLAG=1、ALLOW=aa3e）。13 处逆补丁回 base 后序列化**逐字节等于 0338f87b**。
- 门与失败路径：build_deps 全 pin 命中（AB base 127e47ae、source-control a1c502、window 700cf10c、installers b09cd602/1ed35f0a 等）；deploy/restore 的旧根身份/argv/PGID 归属、pending client、端口 0→111、d0 精确 stop、整 PGID 空、fresh label、启动 env 恰 2+2 键、候选 head/guard/env、service_check+d0 四检；失败仅 stopped 后 recovery（未知→blocked 不发信号）；restore 绑 deploy record SHA/phase 且 prior client 必需，最终 clean 两键 + OPP/LD 对 fcbdcbf1 逐字节。
- 独立复跑 CPU：25 tests OK（1 skip）+ 真实 base FrozenManifestTest OK；真实路径 plan-only deploy rc0。
- 执行注意：`--ab-code` 必须为 `/workspace/yangchen/work/scratch-24/joint-fused-source-control-ebca769/runtime`；真实 deploy 用 f810 S1 最后 supervision（clients_exited=true）作 `--prior-client-record`；candidate restore 用候选 S1 自己最后 supervision；远端部署仓库 HEAD 3e83ae1 不含 640ac23，stage 按文件 SHA 冻结执行。
- 结论：ready，无阻断修复；待 31 设备执行后核实际 raw。
- 31 外层组合脚本（本机 `/tmp/owner31_candidate_s1.sh`，SHA `78be7737…`）已有限核：只顺序调用两条冻结入口（deploy → v5 → restore），无 kill/pgrep/signal/source 写；deploy 失败不 restore、v5 非 0 仍 restore、restore 的 unknown/blocked 由 entry 记录。GO 条件：OUT 全新；`--current-restore` 用本次恢复 restore.json+fresh SHA；deploy prior=f810 supervision `1da03a19…`；v5 后 restore 前确认候选 supervision `clients_exited=true`；外层 rc0 仅代表 deploy+restore，S1 看 v5.rc/acceptance.json。
- 恢复驱动（`/tmp/owner31_clean_restore.py`，SHA `ffd75bd6…`）已有限核：只 import ebca source-control（a1c502）调用未改 `restore_prod()`（clean 两键、原 launch/wait、d0 discover+四检、OPP/LD 对 12933 B/`fcbdcbf1…` 逐字节），无 stop/signal/source/guard 写；run-dir 必须不存在。驱动自身无端口/进程/source 前置门，故执行前紧邻 fresh 四查并保留 `fresh-inventory`/原 window/原 record 绑定。
- **已发生的具体 finding 1（调用参数，非代码缺陷）**：31 首次候选 deploy（OUT `owner31-candidate-s1-20261119T0048Z`）0.5 s 内 rc1，`stage_candidate()` 报 `FileNotFoundError: .../joint-fused-5c045eb/runtime/apply_joint_service_check.py`。原因：`--repo` 用了 5c045eb，而 stage 需要 ebca bundle 根的 `runtime/{apply_joint_fused_attention.py, apply_joint_service_check.py, joint_fused_attention/, joint_service_check/}`。正确 `--repo=/workspace/yangchen/work/scratch-24/joint-fused-source-control-ebca769`；该处四件与 d0 六件 SHA 全部命中 pin（b09cd602/1ed35f0a/aa3e1758/36eaff44）；helper 仍须显式 5c `9245e1a6…`。失败发生在停服前：未发信号、source/guard 未动，服务五 PID 仍存活。
- **finding 2（entry 代码缺陷，阻断候选）**：修正 `--repo` 后第二次 deploy（OUT `owner31-candidate-s1-20261119T0051Z`）在 stop+apply 成功后，entry:306 `deps.ab.switch_source('aa3e','deploy')` 抛 `AttributeError: 'SourceControl' object has no attribute 'state'`（source_control.py:100）。entry 构造 `SourceControl(ns)` 后既未设 `state`（switch_source:100 / verify_restored_env:168 / restore_prod:181 需要），也未设 `prod_env`（restore_prod:179、arm_environment 需要）；ebca main 在调用前显式赋值 `state`（schema/at/args/rounds/steps/source_switches/overall_rc）与 `prod_env`（当前服务 env）。entry 的 recover() 在 `switch_source('f810','recover')` 同错，故未走到 restore_prod，服务 down；source 已回 f810、guard 已回 pre。
- 已向 34 提出最小修复：build_deps 构造后初始化 `ab.state` 与 `ab.prod_env`，并用真实 a1c502 SourceControl 实例补 CPU 契约测试（成功/失败/recovery/restore，仅外部边界替身）；fba7b00 阶段仅文档更正、entry 仍 1f98dba9，不得执行。
- 复核 34 第一次修复 stage `dfa20f93…`（entry `d3aead16…`）：已补 `ab.state = initial_state(...)`，但**全文仍无 `ab.prod_env`**（a1c502 仅在 restore_prod 内部 180/217 赋值，179 先读取）。entry 的 restore 与 deploy 失败后的 recover 都会调用 restore_prod，故仍会在停服后 AttributeError；新测试也只断言 state。结论：仍不可执行，等补 `ab.prod_env = {}`（31 的 clean-restore 驱动即此先例）后的新 entry。
- 复核 34 第二次修复 stage `00e3bdb8…`（entry `a1ca26cf…`）：build_deps 恰新增 `ab.state=initial_state(...)` 与 `ab.prod_env={}` 两行，其余逐字节不变（manifest 31202d27/diff 646ba8d8/builder 6348ab52/supervisor 172c2838/phase 888a8d7f 均未变）。26 项 CPU 测试 OK（1 skip）+ 真实 base 用例 OK；我用真实路径调 build_deps（真实 a1c502 + 真实 manifest，无设备）确认 state 七键齐全、prod_env={}；AST 覆盖显示 entry 可达方法的 self 数据属性缺口恰为 state/prod_env，args 全覆盖；restore_prod 以 prod_env={} 运行已由两次真机 clean restore（driver `ffd75bd6…`，label `cec60d1c`/`c4a4a6af`）实证。**结论：ready**。
- 候选执行（20 裁决后）：OUT `owner31-candidate-s1-20261119T0059Z`、runner `61355d60…`（v3，仅 STAGE 改为 00e3bdb8）。**deploy 成功并已核**：complete ok=true、label `d0-cand-20261119T010007Z-ead542f1`、root 1260810、source aa3e、guard 36eaff44；restore.json 五 PID [1260810,1260958,1260959,1261007,1261008] 同 PGID、stable=true、四检 rc0/ok；live env OPP 9/`7b79c5db…`、LD 12/`ca27dbb7…`、FLAG=1、ALLOW=aa3e、CLEF_LAUNCH_CONFIG=s1，joint 键仅 FLAG/ALLOW/SINGLE_FIELD。
- **候选 S1 三轮完成（结论：中性）**：每轮 2352/2352、0 异常；p95 逐轮 candidate 156.888/156.442/156.116 ms，f810 同轮 157.513/156.405/156.144 → **同轮对应逐轮差 −0.625/+0.037/−0.028 ms（快/慢/快，2 快 1 慢；逐轮差中位 −0.028 ms）**；两组组中位分别为 156.4417 与 156.4051，组中位差 **+0.0366 ms（+0.02%）** 为另一统计量，不得替代逐轮方向。p50 110.156/104.790/106.087，dps 8.519/8.821/8.723。质量三轮同值：Cal 82.411344（Δ−0.3828 pp）、Int 66.084076（Δ+0.0272 pp）、Cap 74.247710（Δ−0.1778 pp）、1904/2352（−2 题）、MAE 0.402612（+0.001223），全部 2PP 内；r1/r2/r3 作者 scorer 独立重打分与记录逐位一致。supervision complete/clients_exited=true/client_rc0/cleanup 干净；acceptance `604a68b4…`、supervision `a0f78be9…`；deploy/v5/restore/composition rc 全 0。
- finally 候选 restore（已核）：label `d0-restore-20261119T011516Z-95656516`、source f810、guard 76595657、五 PID [1263216,1263346,1263347,1263387,1263388] 同 PGID、四检 rc0/ok、envcompare match=true（OPP 9/LD 12 哈希与 expected 12933 B/`fcbdcbf1…` 一致）、live env clean、candidate 键清零；restore.json `4820faad…`、complete `cd476546…`。
- 下一步按 20：31 先跑 f810 S2_b8(C8/C16) → S2_b16(C32/C64) → ERQA(400@16384/16 图/1048576) 原链；候选 typed S2/ERQA 变体由 35 纯本地准备、20 另行排程。
- 安全披露（34，已独立佐证，20 已退出该 agent 并保留其 tree/stage 与测试证据）：其补测初版 **未覆盖 `--tree` 默认值**，于 **00:57:51Z** 用真实 installer 把生产 head f810→aa3e；随即以已核 installer revert（报告 SHA `f7a2e6d1…`，postimage→preimage，ready=true，备份 preimage=`f8101b98…`）。独立佐证：31 候选 deploy 的 `source-deploy-aa3e.json`（**01:00:13Z**）`state_before=preimage`、`sha256_before=f8101b98…`，证明回退在正规候选切换前已生效；**基线 S1 三轮 00:22–00:35 早已结束**，事件晚于基线测量，无污染。**当前真实候选 head=aa3e / guard=36eaff44 是候选窗口预期态**，不得以“当前 f810”的旧状态推断窗口内状态；生产写/installer 仅 31 执行，34 已退出。
- 34 rev4.1 补测（audit-only stage `91314b3…`）独立复跑：unit `53d984ec…` + integration `7d209b26…` + manifest 共 **30 项 OK（1 skip）**；测试在 `/tmp/clef-real-*` **临时 tree/guard** 运行（fixture 用备份 preimage f810 与 bundle 内 guard pre），我前后实读生产 tree/guard 保持 aa3e/36eaff44 未变；**未执行任何生产 installer，执行代码 00e3bdb8/a1ca26cf 未变**。此为补测/文档证据，执行阻断与否仍按 20 裁决。

## 7. 环境事实（最终验收清单）

- clean 两键父环境恢复（ebca `ed140cf3`、final phase `ca972eaf`）：OPP 9 / LD 12，值 SHA `7b79c5db…` / `ca27dbb7…`，与 expected 12933 B/`fcbdcbf1…` 逐字节一致。
- 5c 窗口 restore 继承 prod 环境（31 实测 OPP 23 / LD 28；rstd8 window 的 env-prod-selected 已含 OPP 9/LD 12 重复链），**不得用于最终验收**；最终验收必须走 phase first clean 切换并保持严格 env 门。

## 8. 未闭合项

- Grouped29 v4r1：已按 240 s 单次执行并超时，无 candidate raw；结论为设备侧 final_sync 等待（见第 4 节），不重试、不采用。
- Grouped 后第一次 clean 恢复：`/workspace/yangchen/logs/owner31-grouped-recover-20261119T0044Z/restore-run`，label `d0-restore-20261119T004611Z-cec60d1c`、restore.json `4e022bce…`、四检 ok、envcompare match（OPP 9/`7b79c5db…`、LD 12/`ca27dbb7…`）。
- 候选 deploy 两次失败后的第二次 clean 恢复：`/workspace/yangchen/logs/owner31-candidate-recover-20261119T0052Z/restore-run`，driver rc0、label `d0-restore-20261119T005148Z-c4a4a6af`、五 PID [1258713,1258853,1258854,1258880,1258881] 同 PGID、四检 rc0/ok、envcompare match=true；restore.json **`be55b18c…`**（后续候选 current_restore）、complete `67b9511a…`、envcompare `c9a73e77…`、inventory `ea47954e…`；live clean env、source f810、guard 76595657。旧 `4e022bce…` 对应服务已不存在。
- S2 b8/b16 完整扫描、ERQA 400 与最终最佳服务恢复——待 31 执行。
- cc34 candidate 正式 S1 三轮与 finally restore——**暂停**：等 34 修复 entry 的 state/prod_env 构造契约并出新 stage/entry SHA；重跑绑定 `be55b18c…` restore + prior `1da03a19…`，之后核 3×2352 raw/质量/P95 对新基线 156.4051。

## 8.1 f810 正式验收 S2_b8（已完成并核）

- phase 切换：`owner31-phase-s2b8-20261119T0116Z`，complete ok=true、phase s2_b8、label `d0-final-s2_b8-20261119T011723Z-77231922`、root 1264634、五 PID 同 PGID、四检 rc0/ok；launch argv 与 live cmdline 均含 `--max-num-seqs 8 --max-num-batched-tokens 8192`（effective 同）；live env OPP 9/LD 12 clean、仅 SINGLE_FIELD。
- v5 两点（每点 2352，journal 为**完成序**；以 index 集合完整 + 逐行 index→request 绑定核验，均通过、0 异常）：
  - C8：inflight_max 8；p50 479.306 / p95 940.082 / p99 1021.762 ms；14.7148 dec/s；Cal 82.6932 / Int 65.8648 / Cap 74.278957（Δ−0.1465 pp）/ acc 1902 / MAE 0.402839。
  - C16：inflight_max 16；p50 855.177 / p95 1773.543 / p99 1865.495 ms；16.8259 dec/s；Cal 81.8729 / Int 65.8217 / Cap 73.847316（Δ−0.5782 pp）/ acc 1905 / MAE 0.402753。
  - 两点作者 scorer 独立重打分与记录逐位一致；均 2PP 内。
- supervision complete、clients_exited=true、client_rc0、timed_out false、remaining_group []。
- 待续：S2_b16（C32/C64）→ ERQA 400 → s1 clean 恢复。

## 8.2 f810 正式验收 S2_b16（已完成并核）

- phase：`owner31-phase-s2b16-20261119T0125Z`，complete ok=true、label `d0-final-s2_b16-20261119T012444Z-ad6e0ec7`、root 1266382、五 PID 同 PGID、四检 rc0/ok；launch-manifest effective 16/16384 且 argv 含 overrides；clean 两键父环境（切服时 service-check rc0/ok）。该服务随后被 ERQA 相位替换，live env 不可事后读，以上为切换期证据。
- v5 两点（每点 2352，完成序 journal；index 集合完整 + 逐行绑定，0 异常）：
  - C32：inflight_max 32；p50 1679.734 / p95 3125.361 / p99 3610.798 ms；17.5432 dec/s；Cal 82.5362 / Int 65.7434 / Cap 74.139804（Δ−0.2857 pp）/ acc 1902 / MAE 0.402396。
  - C64：inflight_max 64；p50 3534.276 / p95 5937.589 / p99 7194.449 ms；17.5996 dec/s；Cal 82.5394 / Int 65.743 / Cap 74.141169（Δ−0.2843 pp）/ acc 1902 / MAE 0.402382。
  - 两点作者 scorer 独立重打分与记录逐位一致；均 2PP 内。
- supervision complete、clients_exited=true、client_rc0、timed_out false；s2_b16 phase restore.json SHA `5683086c…7ace`。
- 待续：ERQA 400（16384/16 图/1048576）→ s1 clean 恢复。

## 8.3 f810 正式验收 ERQA（已完成并核）

- phase：`owner31-phase-erqa-20261119T0136Z`，complete ok=true、label `d0-final-erqa-20261119T013135Z-dca5097e`、root 1268324、五 PID、四检 rc0/ok；launch-manifest effective max_model_len 16384、limit_mm `{"image":16,"video":0}`、mm 像素上限 1048576（argv 实测）。
- v5：400/400 预测、0 失败；我逐题独立核对 gold=expected answer、correct 标志、argmax=prediction、图像 SHA 与 expected 绑定、概率有限 → 0 issue。
- **216/400 = 0.5400**；多图 49/113、单图 167/287；选项 4:203/382、2:11/14、3:2/4；与 acceptance breakdown 逐项一致。对基线 219/400=0.5475 的 Δ = −3 题（−0.75 pp），在 2PP（±8 题）内。
- supervision complete、clients_exited=true、client_rc0、timed_out false、remaining_group []。
- 待续：最终 s1 clean 恢复（最佳服务）与四检/envcompare。

## 8.4 最终 s1 恢复（最佳服务）与全链结论

- phase：`owner31-phase-s1-restore-20261119T0140Z`，complete ok=true、label `d0-final-s1-20261119T013520Z-047e2bb5`、root 1269846、五 PID 同 PGID、四检 rc0/ok；effective 4096/seqs1/batched4096、limit_mm `{"image":1,"video":1}`；plan clean_parent 仅 PATH+CLEF_LAUNCH_CONFIG=s1。
- live 复核：OPP 9/`7b79c5db…`、LD 12/`ca27dbb7…`、joint 仅 SINGLE_FIELD（候选键清零）、source f810、guard 76595657。
- **全链结论**：f810 正式验收 S1≥3 轮、S2_b8（C8/C16）、S2_b16（C32/C64）、ERQA 400 全部真实完成并逐项独立核验；各点 2352/400 完整、0 协议失败、作者 scorer 重打分与记录逐位一致；OD 三项与 ERQA 相对 baseline 的绝对波动均 ≤0.578 pp / ≤3 题（2PP 内）；最佳服务已恢复。候选 aa3e S1 中性（逐轮差 −0.625/+0.037/−0.028 ms，2 快 1 慢；Cap 74.247710，2PP 内）。

## 8.5 35 typed 变体（e20e128）本地 ready

- 新 tuple 实读 SHA 全部命中：typed entry `abb7c383…`、unit test `4681a7e1…`、diff `8b96aa89…`、evidence `a00cf2bc…`、stage `644be6b4…`、doc `9191eaf2…`、combo `d72e7957…`；branch 远端 e20e128（在 45745a1 之上）。
- derived-manifest 复核：restore 恒强制 `record.manifest_sha` == 入口 manifest SHA；record 有 `manifest` 时强制解析相等；缺失时从 prior supervision `argv --manifest` 推导并同样强制等于入口 manifest；`manifest_source` 记 deploy_record|prior_supervision；`verify_prior_supervision` 仍以入口 manifest 校验 prior 的 --manifest/三路径/phase/clients_exited/observed/root。无弱化。
- 独立实跑：typed-fixtures 布局 60/60 OK；typed-records 布局 60 OK（1 布局性 skip）；新用例覆盖 missing-manifest 正常路径、supervision manifest 错误阻断、record manifest 不匹配阻断。
- 原语零改参：新 diff 14 hunk 全为校验/plan 字段；stage_d0/stage_candidate/apply_service_check/switch_source/restore_prod 调用无 +/- 行。
- 结论：本地 tuple ready；执行前 31 代 remote staging，我核远端字节与本地 SHA 一致后给 remote ready，再由 20 排程按每候选 phase deploy→v5→restore f810/s1 执行。
- 31 远端 staging：`/workspace/yangchen/work/final-acceptance-cc34-e20e128`。补齐 2 个审计文件后 **16/16 文件逐字节与本地 tuple 一致、无多余/缺失**（typed entry abb7c383、unit 4681a7e1、integration cfda281d、a1ca26cf、S1 tests 53d984ec、manifest 31202d27、builder 6348ab52、test_manifest 8ba9cdb5、evidence harness 25735efb、evidence a00cf2bc、diff 8b96aa89、stage 644be6b4、combo d72e7957、doc 9191eaf2、phase 888a8d7f、supervisor 172c2838）。**remote ready**：20 排程后按每候选 phase（s2_b8/s2_b16/erqa）deploy(from s1)→v5(同 phase、无 single)→restore(f810/s1) 执行；首格 current_restore=最终 s1 restore.json（label 047e2bb5，SHA `92b33040…`）；`--repo`=ebca bundle 根、`--helper`=5c `9245e1a6…`；禁 candidate→candidate。
- 31 报的原件 SHA 已全部实读命中：final restore complete `cdf40a37…`、restore `92b33040…`、ready-identities `8de57b1b…`、service-check `1f96a40c…`；phase restore s2b8 `931a46b4…`、s2b16 `5683086c…`、erqa `fe4db104…`；acceptance s2b8 `e2de2986…`、s2b16 `2f16f492…`、erqa `1f119678…`。


## 8.6 候选 typed 相位（e20e128）执行与核验

- 20 排程 31 串行执行 s2_b8→s2_b16→erqa；每格新 OUT：plan-only→deploy(from f810/s1)→同 phase v5（无 single）→自身 client 退出→restore(f810/s1)→新 current_restore 交接；禁 candidate→candidate。
- **typed s2_b8（cell 闭合，已核）**：OUT `owner31-typed-s2b8-20261119T0150Z`。
  - deploy：label `d0-cand-20261119T014340Z-ede30b98`、root 1271820、aa3e/36eaff44、from s1、current_restore=f810 s1 restore `92b33040…`、overrides 8/8192、manifest e20e128/31202d27、launch_env 2+2 键；五 PID 同 PGID、四检 ok；live argv/live 8/8192、OPP9 `7b79c5db…`/LD12 `ca27dbb7…`、FLAG=1/ALLOW=aa3e。
  - v5：唯一队列（supervisor 1273095/client 1273096/bench 1273109；output `v5-s2b8`、records `v5-s2b8-records`、deadline 02:05Z、无 single）。
  - C8：2352/0 异常、inflight 8；p50 474.575/p95 937.613/p99 1017.324；**14.7833 dec/s**（report 14.7832，wall 159.099 s，官方=successful/wall）；Cap 74.253107/Cal 82.6424/Int 65.8638/acc 1903/MAE 0.402792。
  - C16：2352/0 异常、inflight 16；p50 851.255/p95 1764.774/p99 1861.114；**16.9038 dec/s**（report 16.9037，wall 139.141 s）；Cap 73.82501/Cal 81.8361/Int 65.814/acc 1905/MAE 0.402936。
  - 对同窗 f810：吞吐 +0.47%/+0.46%、p95 −0.26%/−0.49%、Cap −0.0259/−0.0223 pp、acc +1/0；2PP 内；作者 scorer 重打分一致；supervision complete/clients_exited=true/rc0/cleanup 干净。
  - restore：label `d0-restore-20261119T015105Z-05a535b7`、manifest_source=deploy_record、prior=候选自身 v5 supervision、source f810/guard 76595657、五 PID 四检 ok、envcompare match=true（9/12 哈希）；新 current_restore=restore.json `2f62529b…`。restored 服务随后由下一格 s2_b16 deploy 正常停止。
- **typed s2_b16（cell 闭合，已核）**：OUT `owner31-typed-s2b16-20261119T0155Z`；label `d0-cand-20261119T015300Z-75005c96`、root 1274993、aa3e/36eaff44、from s1、current_restore=`2f62529b…`、prior=s2b8 supervision `70b1834f…`、overrides 16/16384；五 PID 四检 ok、effective 16/16384；v5 `v5-s2b16`、deadline 02:25Z；supervision 干净。
  - C32：2352/0 异常、inflight 32；p50 1668.498/p95 3117.81/p99 3608.027；17.6576 dec/s（report 17.6572，wall 133.203 s）；Cap 74.14685/Cal 82.5512/Int 65.7425/acc 1902/MAE 0.40242。
  - C64：2352/0 异常、inflight 64；p50 3511.907/p95 5913.838/p99 7164.443；17.7050 dec/s（report 17.7047，wall 132.846 s）；Cap 74.141828/Cal 82.5383/Int 65.7454/acc 1902/MAE 0.402338。
  - 对 f810 同点：+0.65%/+0.60% dec/s、p95 −7.55/−23.75 ms、Cap +0.0070/+0.0007 pp；重打分一致；restore label `d0-restore-20261119T020026Z-6fc6f7cc`、envmatch true、新 current_restore `06dfcc0f…`。
- **typed erqa（cell 闭合，已核）**：OUT `owner31-typed-erqa-20261119T0202Z`；label `d0-cand-20261119T020219Z-8ec9bc97`、root 1278447、overrides 16384/16 图/1048576、五 PID 四检 ok、deploy complete `c6742044…`；v5 `v5-erqa`、deadline 02:45Z、supervision `5ec2942e…` 干净。
  - 400/400、0 失败；我逐题核 gold/argmax/图像 SHA/概率/状态 0 issue；**216/400 = 0.5400**，single 167/287、multi 49/113、8 类分项与 f810 全等；acceptance `1f119678…` 与 f810 **字节相同**；33 另核仅 5/400 概率差 max 0.001328、argmax 0 变化。
  - restore label `d0-restore-20261119T020614Z-b57139dc`、source f810/guard pre、manifest_source=deploy_record、envmatch true、新 current_restore `a61bdf31…`。
- **typed 全链结论**：三格 deploy→v5→restore 全 rc0、全部 client 退出/cleanup 干净；S2 两点小幅正收益（+0.46~0.65% 吞吐、p95 −0.24~−0.49%）、ERQA 完全同值、S1 中性；所有质量 Δ 在 2PP 内（B8 C8 −0.0259pp 最大负向，B16 略正）。最终 f810/s1 最佳服务已恢复：root 1279960、4096/seqs1/batched4096、OPP9/LD12 clean、仅 SINGLE_FIELD、source f810、guard 76595657。

## 8.8 20 采用 aa3eON 为最终默认（2026-11-19）

- 依据：S1 中性（逐轮 2 快 1 慢、组中位 +0.0366 ms）；S2 同窗口四点吞吐 +0.46~0.65%、p95 −0.24~−0.49%；ERQA 216/400 与 f810 同值；所有质量 Δ 在 2PP 内。20 明确不宣称稳定显著。
- 实施：35 获授权按 rev3 + 两条实现细节做最小官方入口集成（仅本地 CPU、无 SSH）；31 唯一部署执行；旧 765/a1ca/31202/phase/v5 资产冻结不动；新 entry/inference SHA 作为新 tuple 项；无 PR 范围。
- 我已给 31 最小必要验证闭合标准：**可继承**（aa3e/36/b09/f810 源资产与候选服务数值结论、manifest 31202d27/评分 2PP/S2 并发点/ERQA 分母）；**必须新测**（新入口 SHA 与 stage、197863d5→f810→aa3e 冷启动状态机、幂等重启三连、tree 解析/独立 backup/原子 manifest/fail-closed、bash start.sh 实际服务四检+双 worker、live env 恰为 FLAG=1/ALLOW=aa3e/SINGLE_FIELD=1 + OPP9/LD12、回退边界、客户端全空）。
- 指标口径：只用官方 benchmark JSON `decisions_per_second`（successful/wall_seconds）；journal span（首 start→末 finish）重算仅交叉核对、末位差单注。
- 收口：新 tuple 文件/SHA、冷启动/幂等/失败证据、服务身份与 env、selected profile 全 aa3e（不拼 f810 数字）、最终服务为所选 aa3e/s1、clients 全空、旧 f810 资产未改。我负责新 tuple 逐 SHA 与运行等价门；实施代码由 20 触发原生 review。

## 8.9 官方入口准备：197863d5 fixture、等价口径与 review 替代

- **197863d5 原始 preimage 已交付**：远端只读来源 `/workspace/yangchen/logs/tuning/20261114-013000-fix4/clef_joint_schema.py.pre-trace`（cat 取回、本地复算 SHA 命中）；本地只读副本 `/Users/chenclay/.ao/data/artifacts/clef-27b-ascend-32/original-joint-schema-197863d5/`（14144 B、SHA `197863d5…2bf4be`、SHA256SUMS、README），供 35 本地临时 tree 三起点测试，不写提交树/生产。
- **guard 口径（20）**：新官方入口使用独立 `runtime/apply_joint_single_field_patch_fused.py`=36eaff44；旧 `runtime/apply_joint_single_field_patch.py`=76595657 保留冻结；aa3e 运行期不读 guard；新 tuple 分别记录官方 36 与旧验收 765 的角色，不以旧路径必须为 36 作为官方启动成功门。
- **数值等价口径（20）**：3 个固定 smoke 请求仅作接口交叉佐证；全量质量/性能继承的门是完整运行 recipe 逐项等价（aa3e/b09/36 资产、FLAG/ALLOW/SINGLE_FIELD、s1 argv、OPP9/LD12、模型/tokenizer/配置与加载路径、既有 GDN/QKV/chunk128/NZ/custom .so）；recipe 差异项须补对应全量验收，不得用旧 manifest 掩盖新入口 SHA。
- **31 计划调整**：真实官方入口仅覆盖 f810→fused 起点，197863d5 起点由 CPU 临时夹具覆盖（真实冷启动如需额外生产 revert 须 20 授权）；Release tuple 需单列新 entry/inference/stage/资产 SHA、两条 guard、selected profile 全 aa3e、A 继承/B 新测清单与冻结资产零改动 diff。
- **review 替代**：原生 review 无关联 PR 且 gh 不可用 → REVIEW_INVALID；按无 PR 范围不新建 PR。我将在新 tuple 交付后做直接本地冻结 diff + 真实夹具独立审查（重点 rev3 全部门与 35 新增分支），记录该替代及局限，finding 发 35、不改其代码。

## 8.10 官方 aa3e 入口（f9e601c）本地独立审查 = PASS

- 新 tuple SHA 全命中：inference `de5b0d1e…`、b09 `b09cd602…`、aa3e `aa3e1758…`、独立 guard `36eaff44…`（两处）、1ed `1ed35f0a…`、test_joint `dcc545c5…`、test_launch `b4f18279…`、harness `385b6324…`、evidence `1cad8ba1…`、diff `c40725a0…`、stage `d7516e8a…`、doc `23ac3c2d…`；冻结资产未改（765/f810/a1ca/31202d27/888a8d7f/172c2838/start.sh d880a01e）。
- **diff 可复现**：`official_entry.diff` 与 `git diff e20e128 f9e601c -- inference.py test_inference_launch.py` 字节相同；应用到 e20e128 精确产出 `de5b0d1e…` + `b4f18279…`。
- **基点复现**（按真实 runtime 资产修正 /tmp 布局）：e20e128 inference+test 恰为 1 error（KeyError `pending_install`）+ 1 failure（SystemExit not raised），其余 15 通过——与 7ea8aa6 决策门旧契约诊断一致。
- **独立复跑**：`test_inference_launch` 17/17；`test_joint_fused_entry` 用交付的 197863d5 fixture 11/11（无 fixture 10+1 布局性 skip）。11 项覆盖三起点、重复 no-op、unknown fail-closed、失败不启动、非 fused pop、f810 dry-run pop、默认 s1 dry-run、探针/布局校验、资产无 scratch。
- **代码审查**：guard 判 `status_before`、fused 判 `state_before`；仅 preimage 才 apply；独立 backup；原子 `joint-patch-record`（含 no-op/失败）；tree 同解释器探针 + 目标/可写/布局校验 + vllm 前 fail-closed；36 支持 `--manifest`；1ed SOURCE=36、36 SOURCE=f810、b09 SOURCE=aa3e 齐备；environment 白名单且非 fused 显式 pop。**无需改代码**。
- **完整 release 清单要求（非阻断）**：base 运行依赖须绑定 SHA——GDN `124884fb…`、QKV `37b0698e…`、chunk128 `f1283218…`、schema_cache `0d8d495c…`、旧 guard 765、f810 源、activate_env `3b363394…`、start.sh `d880a01e…`、vendor .so（customize/clef_conv_qkv/clef_gdn_qkv_norm）与模型目录 manifest；新根/打包排除 `._*`。delta stage 的 frozen 列表仅覆盖 7 项，完整 release 由 31 新根补齐。
- 旧 `/workspace/yangchen/Clef-27B-ascend`（3e83ae1+dirty、缺 delta）仅作 baseline inventory，不再作为执行根重复报告；等 31 报 0b869dd 重建的新根后核字节与 recipe；OPP/LD 若变化须保持 fcbdcbf1 精确门不放宽。

## 8.11 完整 release 根（2dcc029）与 Plan B = READY

- **326/326 逐字节**：对 `/workspace/yangchen/work/final-release-aa3e-0b869dd` 326 文件逐 SHA 与本地 commit `2dcc029` 同路径 git blob 比对，无缺失、无 mismatch；含 `Clef-27B/logs/model_manifest.json` `98c104a9…`，无 `._`/`__pycache__`/bulk logs。
- recipe SHA 全中：inference `de5b0d1e…`、start.sh `d880a01e…`、activate `3b363394…`、GDN `124884fb…`、QKV `37b0698e…`、chunk128 `f1283218…`、schema_cache `0d8d495c…`、旧 guard 765、f810 源、b09、aa3e、36×2、1ed、stage `048dec31…`、evidence `1cad8ba1…`、diff `c40725a0…`、tests `dcc545c5…`/`b4f18279…`、harness `385b6324…`、doc `ba0e97e9…`；vendor 六 .so `4d2440a3/64b5dda7/8f8c3dbb/2185231c/25063307/b8e60b5c`。
- **vendor 加载角色（源码实证）**：`vllm_ascend/ops/gdn.py` 用 `ctypes.CDLL(<CLEF_QKV_NORM_VENDOR>/op_api/lib/libcust_opapi.so)` 与 `CLEF_CONV_QKV_VENDOR` 加载 `vendors/customize` 与 `vendors/clef_conv_qkv`；无 `torch.ops.load_library`；`clef_gdn_qkv_norm` 无任何运行引用（仅历史构建源/日志），release 将构建源放 `runtime/ascendc/`、不含生产副本，依据成立。
- **OPP/LD 路径判定（确定性）**：`qkv_norm_environment` 从运行中 `inference.py` 所在目录推导 vendor 路径；独立 release 根使 OPP 0–4 段变为 release 前缀，无法与 fcbdcbf1（内嵌生产绝对路径）精确一致。dry-run 实测：argv 35/35 一致、OPP 9（0–4 前缀不同）、LD 2（未 source activate_env）。故 Plan B（生产路径 overlay）是唯一保持 exact 门的执行方式。
- **Plan B（20 认可，32 ready）**：先备份被替换文件（inference 081b5623、test_inference_launch 99baa718）到独立 backup 目录并记录 SHA/来源；部署 13 个已审对象（2 替换 + 11 新增）；旧 765/f810/a1ca/31202d27/phase/v5/GDN/QKV/chunk/NZ/vendor 六 .so 零改；部署后核生产根 SHA；同一 activate_env 下 exact dry-run 断言 OPP9/LD12/argv35；再真实 start.sh 首启 f810->fused、重启 aa3e-noop，核 record/两 report/独立 backup/health/info/五 PID/双 worker/四检/env/joint 三键/clients 空；失败 fail-closed。数值继承仍以 full run recipe 等价为门。
- **官方入口两次启动（已执行，32 独立核验通过）**：
  - 首启（绝对参数）：`official-aa3e-abs-20261119T0250Z`。joint-patch-record ok=true、branch **f810->fused**、guard noop/fused apply、target f810→aa3e、guard 36、env FLAG=1/ALLOW=aa3e；launch-manifest argv 35/35、effective 全等、OPP9/LD12 逐字节等于 fcbdcbf1、joint 三键；patch-backup joint-fused preimage=f810（guard 无备份，符合 noop）；服务 health ready、info config s1/4096/TP2、根 1285953 五 PID 同 PGID、卡7 双 worker、无 clients。根 ppid 为调用 shell（官方直调 start.sh 的预期形态）。
  - 重启（相对参数）：`official-aa3e-rel-20261119T0254Z`。record branch **aa3e-noop**、guard/fused 均 noop、before/after 恒 aa3e、patch-backup 新增 0 文件；manifest argv35/effective/OPP-LD 全等、pid 1287336；live OPP 9 `7b79c5db…`/LD 12 `ca27dbb7…`、joint 三键、PGID 1287336 五成员、卡7 双 worker；旧首启根 1285953 gone、无重复 vllm 树、无 clients。树 aa3e、旧 guard 765 未变、fused guard 36。
  - 结论：官方入口状态机（f810→fused、aa3e no-op）、幂等、env 精确门、服务双 worker/四检前置均真实闭合；最终服务 aa3e/s1（root 1287336）。
- **Plan B 执行前核验（已通过，GO 已发）**：备份 `official-entry-backup-20261119T024617Z`（manifest `73972463…`，`at_utc=20261119T024617Z` = 10:46:17 CST，31 报文 14:46:17Z 为转录错误待纠正）；备份副本 inference 081b5623、test 99baa718 原字节一致。生产根 13/13 SHA 与 release 一致、冻结与 vendor 六 .so 未变。**生产根 exact dry-run `official-entry-dryrun-prod-20261119/dry-run.json` SHA `f8c506aa…`：argv 35/35、effective 全等、OPP 9/9 与 LD 12/12 逐字节等于 fcbdcbf1、joint_patch.installer=生产根、config s1/4096/seqs1**；旧 `official-entry-dryrun-20261119/dry-run.json`（`1d66304d…`）是 release 根 run，不再引用。当前服务 root 1279960（starttime 397095494、boot 393ecc15）身份与保存记录一致、端口 open、无 clients。GO 条件：停 1279960（同身份门）→ 绝对参数首启（branch=f810->fused、record/两 report/backup、树 aa3e/guard 36、五 PID/双 worker/四检/OPP9-LD12/joint 三键/clients 空）→ 停首启 → 相对参数重启（aa3e-noop、树恒定、无重复/残留）→ 最终 aa3e/s1 健康。

## 8.12 官方 aa3e 入口总 PASS 与最终服务（commit 0db4d21）

- **两次启动/两次 stop/four checks（原件已核）**：停旧 f810 root 1279960（02:50:08–11Z，identity verified、5 targets、termed 5、survivors 空）；首启 abs `official-aa3e-abs-…0250Z` root 1285953（branch f810->fused、guard noop/fused apply、target f810→aa3e、guard 36、backup preimage=f810、argv35/effective/OPP9/LD12 全等、GET health/info + POST systemone/decide 全 200）；首启 stop 1285953（02:52:36–39Z，verified/5 targets/无 survivors）；重启 rel `official-aa3e-rel-…0254Z` root 1287336（branch aa3e-noop、guard/fused noop、0 新 backup、argv35/OPP-LD 全等、fourcheck-rel.txt 四检 ok）；wrapper 1287335 verified/gone，root reparent **ppid=1**。
- **P3 metadata 增量（0db4d21，已 push，origin==HEAD）**：`fourcheck-rel.json→.txt` 为 R100 原字节 rename（SHA `e33f040f…` 不变、json 不存在）；SHA256SUMS 78 行 **78/78 OK**；entry 下实际 **14 份 .json 全部严格解析通过**（修正 31 报的“15 份”，为该 rename 前口径）；仅 2 路径变更，无代码/服务/数值变化；工作树 clean。33 独立补核一致。
- **最终服务（live 已核）**：root 1287336、ppid 1、PGID/SID 1287336、start 397375124；五成员同组；health ready、info config s1/4096/seqs1/batched4096/TP2/eager；source aa3e、fused guard 36、旧 guard 765 未变；OPP9 `7b79c5db…`/LD12 `ca27dbb7…` 逐字节等于 fcbdcbf1；joint 键 FLAG=1/ALLOW=aa3e/SINGLE_FIELD=1；卡7 chip0/1 双 worker；clients/stray 空。
- **全 recipe 继承（release-binding `ready_official_entry_verified`）**：A 复用 aa3e/b09/36/f810 源与 aa3e 服务数值（S1 156.4417、B8 14.7832/16.9037、B16 17.6572/17.7047、ERQA 216/400）+ 31202d27/scorer/2PP/S2 点/ERQA 分母；实测绑定 S1 `604a68b4…`/`a0f78be9…`、B8 `c54c0958…`/`70b1834f…`、B16 `8f90bddd…`/`700361df…`、ERQA `1f119678…`/`5ec2942e…`、final f810 baseline `b57139dc`/`a61bdf31…`；B 新测项全部由 CPU 证据 + 真实窗口 + 32 独立核验闭合。release 根 326/326 vs 2dcc029；entry `de5b0d1e…`/stage `048dec31…`/doc `ba0e97e9…`/evidence `1cad8ba1…`；Plan B backup `73972463…`（at_utc 20261119T024617Z = **10:46:17 CST**，31 已更正转录）；prod dry-run `f8c506aa…` argv35/OPP9/LD12/effective 全等。
- **selected 一致性**：README/metrics.csv/results.csv/optimization.md 当前行全为 aa3e（Cal 82.4113/Int 66.0841/Cap 74.2477/1904/MAE 0.402612；ERQA 216/400；S1 156.4417；S2 17.7047@B16C64），f810 数字仅在明确标注的同窗口对照/历史；图片声明为历史 baseline、用户本人补拍。
- **结论：总 PASS**。官方 aa3e 入口、Plan B、selected、全 recipe 与最终服务真实闭合；图片为用户自截图项，不阻塞代理收口。

## 9. 本地核对副本

本机 `/tmp/clef32-audit/` 保留 ebca 71 原件、rstd8 window/probe/timing 与 cell/timing bin（diag/rstd）、AD window/probe、cc34 双 manifest 与 entry；均与远端逐 SHA 对齐（见各节哈希）。远端原件路径如各节所示。
