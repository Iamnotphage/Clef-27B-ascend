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

## 8.13 新阶段：Grouped 超时深挖（2026-11-19 起）

- 授权与角色：31 唯一设备/生产 owner；35 本地实现；33 只读契约研究；32 独立证据/有限代码协议审核。**32 不再 SSH 计算实例**，只核 31 的只读证据包/本地工作树与 CPU 临时夹具；旧正式 0db4d21/aa3e 数值与原件冻结；无 PR/无合并。
- **Snapshot 恢复门**：root/ppid=1/PGID/SID/start/boot/完整 argv；tree=aa3e、old guard=765、fused guard=36（独立路径）、vendor 六 .so、GDN/QKV/chunk128/schema_cache、activate/start.sh、model manifest 98c104a9；OPP9 `7b79c5db…`/LD12 `ca27dbb7…` 逐字节、joint 三键 FLAG=1/ALLOW=aa3e/SINGLE_FIELD=1、health/info s1、双 worker、clients 空。
- **恢复计划**：禁止旧 Grouped/5c finally 或 restore_prod 自动 revert f810；停后恢复必须走当前官方 start.sh（clean+activate，tree 保持 aa3e、branch=aa3e-noop、guard 36/765 不变）；finally 四检+双 worker+clients 空+exact 身份/env；f810 回退需单独授权。
- **探针隔离**：仅临时 AOT/child env（ASCEND_RT_VISIBLE_DEVICES=0,1、owner/payload sha）、hook 仅 validation（system_mutations=[]）、不写全局 tree/guard/vendor；fresh run-dir/唯一 manifest；旧失败 raw 保留。
- **首包判据**：Cell0 无设备 snapshot+bundle SHA+“无全局写”证明；Cell1 小合法形状双 rank、单变量三臂（MM+独立 AR+norm / native MMAR 单独 / MMAR+norm），输入 shape/BF16/seed 固定且 SHA 不变；通过=两 rank 有限+对**生产 NZ 参考**的原冻结门 `041a66aa…`（不弱化）+**API rc0 且观察到 sync 返回**；只 rc0 无 sync=失败；≥3 次 fresh 进程成功；trace/plog 保留。
- **后续**：restore 桥接+norm → 真实 967 行 shape/权重（生产 NZ 参考、原门、240 s 上限与 clean 恢复）；性能仅正确后同窗 ABBA≥4（最慢 rank、sync 后 wall/event、含分配/host/通信/后置 norm；冷启动单列；局部 native 不外推服务）。若采纳再全量 S1≥3/S2/OD2352/ERQA400，2PP 不变。
- 证据包：31 提供只读本地副本+SHA256SUMS（snapshot/bundle/plan/trace/result/bins/gate/cleanup/recovery/命令 rc/时间/boot/clock）；32 逐项核原件。

## 8.14 Grouped binding 代码级生命周期（v4r1 binding.cpp 5db40ef9，本地只读复核）

- 文件 SHA 与 33 本地副本一致（5db40ef9…）；run_probe.py `dde68f34…`。
- **H1（组名）**：handler lambda **不捕获 group**；`group.c_str()` 仅传入同步 `prepare`。Python 侧 `group` 为 `get_hccl_comm_name(rank)` str，binding 签名 `const std::string&` → pybind 每次调用生成**临时 std::string**，调用返回即释放。若 executor 保存该指针，生产路径即悬垂；A1 的保留对象必须是 C++ static/global string、在 prepare 前创建并直接传其 c_str、保活到 sync 后（同时是候选修复形态）。
- **H3（descriptor/list，强候选）**：handler 在 `launch()` 入队后**立即 `lists->close()`**（销毁 aclTensorList 及成员描述符），随后 handler 返回、Python 才 sync；异步 MC2 若仍引用描述符可能挂起。A3=close 延后到 sync 后，记录 lists/descriptor/executor 地址与存活区间。
- **H2（workspace，候选）**：workspace tensor 为 C++ 局部、被 lambda 按值捕获；binding 返回的只是 int64 字节数；handler 返回后 tensor 释放（sync 前）。A2=static/global 保活到 sync 后并记录 data_ptr。
- **H5/H4**：stream 在 prepare 后、dispatch 前捕获一次，handler 用该 stream；Python sync 用自己的 current stream；handler 线程/context 与 prepare 不同。A9 每臂记录 prepare/handler/sync stream id、thread id、aclrtContext/device、`/proc/self/maps` 实际加载库路径+SONAME+SHA、group 指针/字节。
- **H11（tiling host buffer）**：executor 从未销毁（泄漏），tiling 非“被释放”，H11 降级；除非头文件说明 executor 引用调用方描述符（并入 H3）。
- **现有证据**：baseline（F.linear+all_reduce）与 nd_native（matmul+all_reduce）首调用均成功，仅 candidate（原生 MMAR）挂起 → comm/组健康已间接证明；A8 仅作显式基线。
- 首批顺序建议：A9 instrumentation 随每臂 + A0 小 M(128) → A1(static string) → A3(close 延后) → A2(workspace 保活) → H4/H5 诊断 → H7 sweep；H6/H11 诊断项后置。每臂双 rank/240 s/fresh/固定 seed；成功=API rc0+两 rank sync 返回+finite；首包还须附 aclnn 头文件/op-info/binary config/kernel .o/version/platform/torch_npu HCCL 符号的只读副本。

## 8.15 Grouped 深挖 preflight（snapshot + restore-spec + API 包）

- **Snapshot PASS**：`snapshot-and-restore-spec.json` SHA `39e52eb5…` 本地实核；root 1287336/ppid1/PGID-SID/start 397375124/boot 393ecc15、五成员；health/info s1/4096/seqs1/TP2/eager；aa3e、fused guard 36、old guard 765、inference de5b0d1e、start.sh d880a01e、activate 3b363394、b09；OPP9 `7b79c5db…`/LD12 `ca27dbb7…`、joint 三键、无 SMALLQ；vendor 六 .so、GDN/QKV/chunk128/schema_cache、model_manifest 98c104a9 全中；双 worker、clients 空。
- **Restore-spec 收紧**：routine finally 必须 branch=**aa3e-noop**、tree 恒 aa3e；`f810->fused` 仅限已知源异常的救援臂并记录 source 漂移/该格隔离失败，不得作为正常成功；unknown 源严格门停报；已知源紧急恢复已获用户授权。更新版 SHA `76355dab…` 本地已核：target aa3e/s1、before_restore=exact stop+fresh identity+端口111/整PGID空/卡7空/clients 空、branch_policy 三态正确、success_assertions 含最终“branch == aa3e-noop (normal cell)”。**metadata 清理建议**：success_assertions 仍保留旧条目“branch in {f810->fused, aa3e-noop}”，建议删除或限定为 exception_only 不计成功，避免自动判定放行救援分支（非 blocking）。
- **API 包**（本地 `/tmp/aa3e-grouped-api-pack-20261119`，64 文件）：按构建前缀映射逐项核 **63/63 内容 SHA 一致**；打包问题：PACK_SHA256SUMS.txt 用构建机绝对路径并把自身记为 e3b0c442（实际 b1699d03）→ 应改相对路径并去自条目/最后写自身；tar `685d9a33…` 不在本地 /tmp（无 tar 文件），待 31 给本地路径或以目录+清单为准。
- **头文件契约**：`aclnnGroupedMatMulAllReduceGetWorkspaceSize(..., const char *group, ...)`、launch=(workspace,size,executor,stream)，未声明 group 生命周期/是否拷贝 → H1 开放；官方样例保活 hcom_name 到 sync 之后（33 C1）支持 H1 高优先。A12（官方 A3 样例）为分层正对照，不能单独作为 C1 单因证据。
- **设备/注册**：Ascend910_9382（cube24/vector48/L2 192MB/HBM 65.8GB）、driver 26.1.1 fw 9.0.0.9.220、CANN 9.2.1；ops-info a16b41d0 与两 Grouped kernel .o 存在仅证注册；platform 无 910_93.ini、公开文档 A3 支持自相矛盾——如实保留，A12 实测裁决。
- **首批 ready 要求**：A8/A12/A0 + 必要 A9 埋点 + H1(A1)/H2(A2) 即可 preflight，不要求全 11 假设；A3/A4/A5 随后；每臂双 rank/240 s/fresh/固定 seed；成功=rc0+两 rank sync 返回+finite；A9 记录 stream/thread/context/group 指针/maps 实际库。
- **A12 官方样例包（33 交付，本地已核）**：`/Users/chenclay/.ao/data/artifacts/clef-27b-ascend-33/grouped-a12-sample/`，SHA256SUMS **5/5 OK**；源 `test_aclnn_grouped_mat_mul_all_reduce.cpp` SHA `6c1a90ab…`（21,723 B，f5703e4）；行号核对：aclInit L163、hcom_name 栈 L355-356、GetWorkspaceSize L370、launch L380、`aclrtSynchronizeStreamWithTimeout(…,10000)` L383、aclrtFree L385、rank 线程 context L423-424、finally sync L452/461；FP16 默认/2 group（x0[256,256]/x1[1024,256] 等，data_size 10485760）。**pending（具体）**：包内无官方 examples/CMakeLists.txt（compile 文档为通用 FlashAttentionScore 模板）；主张“官方原样”须补 f5703e4 的 CMakeLists 及 SHA，否则 A12 标注为“官方样例源+重建 CMake（派生构建）”。A12 无数值参考答案 → 成功判据=API rc0+sync 返回（liveness/通路控制），不单因证明 C1；A12b（BF16+FP32 bias、v4r1-like shape）仅作派生 delta；编译/加载 provider 差异（libopapi_math/transformer+hccl vs dlopen libopapi/libnnopbase）须记录。

## 8.16 Grouped 深挖：包定稿、op-info 与 A12 preflight

- **API 包定稿**：`PACK_SHA256SUMS` SHA `ecffc1ff…`，63 条目全相对路径、**63/63 OK**、无自条目；目录 64 文件（63 payload+manifest）；派生 tar `f40628b7…` 仅记录不作权威。
- **restore-spec**：SHA `224b0036…`；success_assertions 已删旧 `{f810->fused, aa3e-noop}` 并列项，正常成功仅 `branch == aa3e-noop (normal cell)`；`exception_note` 明确 f810->fused 永不满足成功门；branch_policy 三态保留。
- **op-info（本机 9.2.1 a16b41d0）**：GroupedMatMulAllReduce 实测 x/w `bfloat16,float16` ND、bias `float32,float16` ND、y 同 x；group required str；splitItem/commTurn 默认 0；reduceOp 默认 sum；group_list int64 optional；`mc2.ctx=mc2_context_0`；`needCheckSupport=false`；dynamic shape/compile true；opFile grouped_mat_mul_all_reduce；multi_kernel；precision_reduce=true——支持 33 契约矩阵；op-info 无 128 对齐约束（M=967 不由其排除，tiling 仍待 A0/A7 实测）。
- **A12 构建 preflight**：source `6c1a90ab…`（官方原字节）、glue `22edb89f…`（manifest 明确为 derived CMake glue，官方 examples/CMakeLists 不在冻结包）、binary `37929ec4…`/204,744 B；ldd 链 libascendcl/libnnopbase/libopapi_math/libopapi_transformer/libhccl/libhcomm/libstdc++；FP16 原样、成功=rc0+sync 返回（无数值参考）；deltas/provider_delta 已记录。**pending-1**：本地归档 stage-manifest SHA `088f05f6…` 与 31 报文 `1d529ca8…` 不一致，待指明权威。**pending-2**：本地归档仅 ldd.log+stage-manifest.json，缺 glue CMakeLists/cmake.log/make.log/source/binary 副本，待补齐加 SHA256SUMS。
- 首批 readiness：包+A12 接近 ready；35 的 A8/A0(M=128)/H1/H2 仍待；两 pending 解决后合并 preflight，31 按恢复门停服执行。

## 8.17 Grouped 深挖：原生 review 未运行 + fallback 有限审查（NO-GO）

- **原生 review 未运行**：对 35 研究分支 `REVIEW_INVALID`（无关联 PR、gh 不在 PATH）；按 20 决定不为此新建 PR。以下是**本地静态 + CPU 临时夹具的有限审查，明确不视为 PASS**。
- **冻结件**：首包 commit `ced48f3`（binding_probe 8ec3642d、run_probe b27a954e、compile_probe 53e518db、fixed_gate 041a66aa 未改、snapshot_contract a954e1a0、validate_snapshot 81737532、test_probe_cpu a8426375、README 375c1673、BUILD_NOTE d606aa22、A12 源 6c1a90ab）；监督冻结 commit `abfe38a`（window `e55fcc00` = draft 同字节、manifest `0ef5005c`、cell_supervisor `6473a011`、README `8ef3da00`）。**CPU 独立复跑 27/27**（11+8+8），但测试 mock `win.run`/用会退出的假 restore，不覆盖前台 start.sh。
- **NO-GO（P0）**：
  1. `start.sh` 末尾 `exec python inference.py` 前台常驻；window `run(restore_argv, timeout=900)` 用 `subprocess.run` → 阻塞 900 s 并**杀掉刚恢复的服务**；`cell_supervisor.run_aux` 同样等 rc。必须 Popen 分离、不等待、wait_health、从 launch-manifest 读身份。
  2. restore 只在 child try/finally 内；stop rc!=0、停后端口/PGID/NPU 检查失败、d0_stop 异常都在其之前 `return 1` → **不恢复**。必须顶层 finally 覆盖 stop 之后的全部路径。
  3. A12 样例 `ACL_CHECK` 在线程内 `return retcode` 且打印 `[ERROR] ... retcode: N`，std::thread 忽略返回值，main 可能 rc0；window 只判 program rc，未解析日志、未捕获 launch_rc/sync_rc。
  4. A12 独立二进制的实际 maps（opapi_math/transformer/hccl_fwk 等）未采集；probe 的 library_evidence 只读 Python 进程。
  5. restore_output 非 fresh，可能读旧 launch-manifest/joint-record。
- **P1**：binding_probe 的 KEEP_* 保留整个 RunState（含 workspace）→ H1/H3 与 H2 混淆；run_probe 默认 arm-timeout 60 s（协议 240）；recovery 文案仍写 “aa3e-noop or f810->fused”；stop-script SHA 检查条件化且 d0_stop CLI 未证；env_expect 未 gate 冻结 fcbdcbf1；protected 未含 supervisor pid；finally 中 rc 可能未初始化。
- **当前状态**：NO-GO 已发 31/35/20；服务 aa3e/s1 保持在线未动；修复后重冻结（新 window/manifest SHA）再 preflight。A12 body（text a7dd3f64/image f3229f5c）与 identity 需 31 提供本地副本/SHA256SUMS 方可核。

## 8.18 A12 preflight 输入核验（31 本地副本）

- **a12-preflight 目录**：SHA256SUMS **4/4 OK**；text-body `a7dd3f64…`（2,849 B，model/questions/state）、image-body `f3229f5c…`（22,570 B，images/model/questions/state）、root-identity `109795ae…`（pid 1287336/ppid1/pgid 1287336/start 397375124/boot 393ecc15，与 snapshot `39e52eb5…` 一致）、plan-a12 `97ae37df…`。
- **plan**：child_timeout 240、port 18143、stop_argv=d0_stop --api-pid/--saved-identity/--out、child_argv=run_a12.sh+binary `37929ec4…`、restore_argv=env -i+activate+官方 start.sh（前台 exec，P0-1 仍成立）、success 含 child_rc0/not_timed_out/no_survivors/source aa3e/branch aa3e-noop/four/env/joint/two_workers/clients；status=plan_not_executed。
- **仍未满足**：plan success 不含 A12 `[ERROR]`/retcode/线程完成日志门（P0-3），无实际 maps 要求（P0-4）；window `e55fcc00` 的 restore 仍 `subprocess.run`、stop 失败绕过 finally、restore_output 非 fresh（P0-1/2/5）——代码级，plan-only 不能证明修复。
- **P1 新增**：clean_expect 仅比 OPP/LD 并记录路径；需 gate 其 service_environment == 冻结 `7b79c5db…`/`ca27dbb7…`（或提供该 launch-manifest 本地副本/SHA），否则可换文件放宽。
- **GO=否**；等 35 修复版新 commit → 31 重 stage+plan-only → 32 核 diff/SHA、CPU 复跑、restore/A12 日志门/maps 后重给 GO。服务在线未动。
- 本地 a12-preflight 目录核验：5 文件、SHA256SUMS **5/5 OK**（clean-env `fcbdcbf1…` 12,933 B）；但本地只有旧 plan `97ae37df…`，**plan-v2 `0482225e…` 无本地副本**——已请 31 在最终窗冻结后把最终 plan/window/manifest/stage 写入本地目录并更新 SUMS，再核。

## 8.19 监督窗 P0/E 复审（d619501 + 7e1cf51）

- **d619501（window 80fd76f9，manifest 817ed3aa）**：restore=Popen(start_new_session/stdin DEVNULL/不 wait)、顶层 finally、A12 `[ERROR]`/retcode/≥2 `mc2 costtime`、PGID maps、restore_output fresh+output_dir/mtime——P0 A–D 代码 PASS；CPU 独立 11+8+12=31/31。
- **7e1cf51（binding 11baa2c6、run_probe c7e0410e、cell 3daa1455、manifest fbce54cd；window 同 80fd76f9）**：binding 按资源拆分（H1 仅 group_keep、H2 仅 keep_workspace、H3 仅 keep_lists；handler 仅 !keep_descriptors 时 close；release 仅释放对应对象）→ 单变量成立；run_probe 默认 arm-timeout 240、recovery 仅 aa3e-noop；cell 明确拒绝 start.sh restore；CPU 独立 **11+9+12=32/32**。
- **clean_expect**：31 本地副本 SHA `fcbdcbf1…`（12,933 B；OPP9 `7b79c5db…`、LD12 `ca27dbb7…`）——P1 env gate 满足；此前 fcbdc7b1 为转录错。
- **仍缺（A12 GO 前）**：① 恢复状态机安全门：old_alive 时绝不 Popen（端口 open+health ready → skip+该格失败；端口非 0/未知 → restore_blocked 停报）；仅 old_root gone 时进入恢复，且 Popen 前再门 port==111/旧 PGID 非 Z 空/无 probe 残留/source==aa3e；skip/blocked 记 identity/health/port/PGID/source。② A12 maps 按 PGID 子树每 PID 记录（pid/cmdline/命中库 path SHA/SONAME），识别实际 binary 进程并记录 opapi_math/transformer/hccl_fwk/hcomm 实际加载（未链上记 absent）。③ CPU 补真实前台常驻 fixture（Popen 后窗口完成、服务仍活、fixture 清理）与 stop 失败/旧服务活不双启；不得整段 mock win.run。④ 小项：stop-script build_plan 强制 is_file；skip 记 health。
- **skip 语义决定**：接受“旧服务活+端口 open+health ready → skip 且判失败”；不接受旧 root 活时因 port==111 启动第二棵树（必须 blocked/停报）。A12 inputs/identity/plan/clean_expect 已 PASS；完成①②③后重冻结再核。
- **对 31“安全门已存在”的复核**：`needs_restore=(stopped or port==111 or not old_alive)` 只覆盖 old_alive+port open；**old_alive+port111（partial）仍 Popen**，且 Popen 前未检查旧 PGID/探针残留/source。要求：old_alive=True 绝不 Popen（health ready+port open→skip+fail；否则 blocked 停报）；old_alive=False 时 Popen 前再门 port==111/旧 PGID 非 Z 空/无 child-probe 残留/source==aa3e；任一不满足→blocked。A12 maps 需 per-PID `{pid,cmdline,libs[]}` 并识别实际 binary（a12_grouped_mmar/timeout），记录 opapi_math/transformer/hccl_fwk/hcomm/ascendcl/nnopbase 实际 loaded/absent；硬门保持 libhccl+opapi_math/transformer（hccl_fwk 未链上不硬门，hcomm 记录）。CPU 真实前台 fixture（Popen+health+进程存活+清理；old_alive+port111→blocked 无 Popen）仍必须补。

## 8.20 A12 增量复核与 GO（4653278）

- **冻结件**：window `51d6bc0c…`、manifest `e55eac00…`；binding/run_probe/cell 与 7e1cf51 相同（未回归）。
- **增量核对**：
  - `recovery_gate`：old_alive 先判且**绝不 Popen**；health ready+port open→`skipped_service_alive_healthy`（context 含 health/identity/port/old_pgid/residual/source）；port111/unknown/health 非 ready→`old_root_alive_partial` blocked；old_root gone 才进入 Popen，前门 port==111/旧 PGID 非 Z 空/无 child-probe 残留/无 runner 残留/source==aa3e；skip/blocked 写 window.json 返回 1。测试 `started==[]` 证不 Popen；healthy skip 记录 health。
  - maps：per-PID `{pid,cmdline,libs[path,sha256,soname]}`；map_coverage 7 类 loaded/absent；binary_processes 识别 a12_grouped_mmar/timeout；硬门 libhccl+opapi_math/transformer，hccl_fwk/hcomm 仅记录。stop-script 在 build_plan 强制 is_file。
  - 真实前台 fixture：单测 OK，window.json ok=True/gate=restore/branch=aa3e-noop/pid=87508，restore.log 0 字节（预期），proxy 用 ProxyHandler({}) 绕过；测试末 killpg(pid)。ResourceWarning 为 Popen 未 wait 的 cosmetic。
  - CPU 独立 **11+9+20=40/40**。
- **A12 preflight 本地 7/7**：plan `063e794b…`；stage-identity：commit 4653278/window 51d6bc0c/manifest e55eac00/binding 11baa2c6/run_probe c7e0410e/cell 3daa1455/adapter a9b01fd9/A12 binary 37929ec4/glue 22edb89f/bodies a7dd3f64+f3229f5c/clean_env fcbdcbf1。
- **GO**：已下发 31 单格 A12（d0_stop→A12≤240s→精确清理→官方 aa3e-noop 恢复；fresh run/restore dirs；skip/blocked/restore 失败停报且绝不双启）。回传 window.json/stop.json/child stdout+stderr/restore.log/per-PID maps/launch-manifest+joint-record/final identity/plog/A12 rc+error+costtime。判定仅当 ok=true+A12 证据+maps 硬门+branch aa3e-noop+五 PID/双 worker/四检/env/joint/clients。**A12 仅 liveness/通路正对照，不外推 BF16 精度、不单因归因**；production restore（setsid）设备首次真实执行，异常按门停报。

## 8.21 A12 单格结果（决定性负结果，2026-11-19 06:24Z）

- **执行**：官方样例（6c1a90ab，派生 CMake）在 4653278/51d6bc0c 窗口下执行；旧 aa3e root 1287336 经 d0_stop 停止（stop.json identity verified、5 targets、refused 空）；A12 child 运行约 114 s（10 s 同步超时 + 2×50 s HCCL DestroyAicpuComm）<240 s；恢复官方 start.sh 成功，**新 root 1301295**、五 PID、branch aa3e-noop、四检/env/joint/clients 全 true。
- **结果**：window ok=false（正确）；child rc0 是假成功（样例线程内 ACL_CHECK 被忽略，main rc0），被 `a12_evidence` 日志门拦下（errors=3、cost=0）。child.stdout：两 rank hcom 名对称、workspace 16777216、`[ERROR] ...:383 retcode 507046` + `TestCall execute AICPU failed`。plog-debug：两 device `stream_id=47, timeout=10000ms`、`current task type_name=KERNEL_AICORE, task_id=1`、`ErrCode=507046 InnerCode=0x7030010`；**HCCL DestroyAicpuComm 50 s×2 timeout**；AICPU scheduler 日志止于 KFC ResInit/SrvLaunch API 加载，无 task 处理进展。maps per-PID：libhccl/opapi_math/opapi_transformer/hcomm/ascendcl/nnopbase loaded，hccl_fwk absent。
- **意义**：官方样例生命周期全部保活、显式 context/stream、单进程 HcclCommInitAll，仍复现同一 stream47/AICORE pending → **H1/H2/H3/H11 对“挂起”被决定性削弱**；H4/H5 亦削弱；H12 的“单 vs 多进程 comm 创建”不再是唯一差异（两者都挂），重心转为 **MC2 AICPU/KFC 公共路径**；H7 被样例 aligned shapes 削弱；H6 workspace 初值仍低成本可测；环境/SoC/op 路径上升。A12 不裁决 A3 不支持，但显著提高环境/op 风险；**不外推 BF16 精度、不单因归因**。
- **监督/恢复链 PASS**：假 rc0 被日志门拦截；gate allow=restore（port111、旧 root gone、PGID/residual/runners 空、source aa3e）；恢复 identity/五 PID/双 worker/四检全过——真实失败下窗口与恢复链得到验证。
- **下一步**：A8 纯 torch all_reduce+sync（基座 comm/stream）→ H6 workspace 零初始化（单变量）→ **A13 MatMulAllReduce（非 grouped MC2）控制臂**（API pack 有 v1/v2/v3 头；同 HCCL/AICPU 路径）区分“MC2 路径坏”vs“grouped op 坏”；AICPU/KFC 日志升 debug；H1–H3/H7 降为确认/低优先。
- **新身份**：下一格必须基于恢复后的 root 1301295 重新出 fresh root-identity/snapshot（旧 1287336 已终止）；所有 plan/stop identity 需更新。

## 8.22 Grouped 深挖收口：mc2_server 环境阻塞（决定性）

- **独立实测 plog-run-1301080.log**：06:24:20.366 `insert package:mc2_server.tar.gz config`；device0 06:24:20.502 与 device1 06:24:20.927 均 `current device chip:5 does not support package:mc2_server.tar.gz`、`does not need to load to device:0/1`；同段 `cann-tsch-compat.tar.gz` 亦不支持。发生在进程启动 TDT 阶段、op launch 之前。
- **因果链**：MC2 AICPU server 包未加载 → AICORE kernel 等待 AICPU/KFC → stream47 KERNEL_AICORE pending → RUNTIME 10 s 507046 / v4r1 无超时挂起 → DestroyAicpuComm 50 s×2。官方样例（生命周期全保活、显式 context/stream、单进程 HcclCommInitAll）与 v4r1 同因；**H1–H12 对“挂起”不再有解释力**。
- **结论**：GroupedMatMulAllReduce 在本机 A3/CANN 9.2.1 **环境阻塞**（mc2_server 不支持 chip:5）；不改变官方 aa3e/f810 默认、不影响提交；不新建 PR/不改指标。H6/H7/H12/A8/A13 暂停（A8 基座 comm 此前 baseline/nd_native 已通过；A13 若同需 mc2_server 会同样被拒）。
- **收口证据（已核，31 mc2-support 5/5）**：`mc2_server.tar.gz` SHA `71263835…`（873,250 B，头 `aa55aa55` 私有格式）；`ascend_package_load.ini` 中 mc2_server `optional:true`、`package_path opp/built-in/op_impl/aicpu/kernel`、`load_as_per_soc:false`（ini 无 soc/chip 支持字段，匹配决策在运行时包内）；`libmc2_server.json` 明确 `HcclLaunchAicpuKernel/Mc2AicpuTaskCacheClearKernel/Mc2ServerKernel` 均在 `libmc2_server.so`；`mc2-load-evidence-lines.txt` 与 plog 行逐字一致（device0/1 chip:5 不支持 mc2_server/tsch-compat；AICPU 仅从 libccl_kernel.so 取 KFC ResInit/SrvLaunch）。→ **环境阻塞结论成立**：MC2 server 包未加载，MC2 KFC 任务无实现，AICORE 等待 → stream47 pending/507046/DestroyAicpuComm 超时；与绑定/生命周期/comm 创建无关。
- **结案**：Grouped MMAR 在本机 A3/CANN 9.2.1 不可运行（平台包支持限制）；暂停 H6/H7/H12/A8/A13 及所有 Grouped 设备臂；官方 aa3e/f810 默认、入口、指标不变；服务 aa3e root 1301295 健康。33 的支持矩阵/chip 映射确认留作文档收尾（loader 实测拒绝已足够操作结论）。35 记录 commit d4b600c 存在，无代码/参数改动。
- **A12 监督/恢复链 PASS**：rc0 假成功被日志门拦、gate allow、官方 start.sh aa3e-noop、root 1301295/五 PID/四检全过；服务当前健康在线。

## 8.23 A13（MatMulAllReduce 控制臂）审查与处置

- **交付**：35 commit 365cb95（binding 52287e3a、run_probe 94f3b9a1、window 706d802b、manifest 460757b1）；CPU 独立 **42/42**。设计：`aclnnMatmulAllReduce` v1（x1/x2/bias BF16、group、sum、commTurn0、streamMode1、output），torch 多进程 comm，M=967/K=8704/N=5120。
- **契约审查**：v1 头明确 x1/x2/bias 支持 float16/bf16 → bias BF16 合规；Grouped 的 FP32 bias 是 op 差异（记录即可）。v1 未标 bias optional，零 bias 张量是安全形态；null-bias 只可作单独诊断。v1 为最小控制，v2/v3 引入 x3/commMode 变量，不作为控制臂。
- **关键 preflight 缺口**：打包的 A3 ops-info/binary_index 中**无精确 MatMulAllReduce 条目**（只有 GroupedMatMulAllReduce 索引；精确名 grep 0 命中）；若未先确认本机 CANN 的 A3 注册/kernel，A13 的失败可能是 op-not-found 而非 MC2 路径。
- **处置（与 35 一致）**：**不做设备执行**；mc2_server 在 TDT 启动阶段因 chip:5 被拒且与 op 无关，MatMulAllReduce 同属 MC2，A13 极可能同拒；为 A13 停掉健康 aa3e 无新根因信息。仅保留代码（未来平台/CANN 支持 A3 mc2_server 时首个控制臂）；CPU-only stage/plan 可选、不触设备。
- **解释边界**：A13 仅 MC2 公共路径控制；返回→MatMulAllReduce 无 mc2_server 可用（隔离 Grouped）；同挂/同拒→平台 MC2 阻塞泛化；不外推精度、不单因。
- **结案记录**：31 conclusion `d95a2d52…`（confirmed_environment_blocked，commit cb90bc3）；35 `98fc132`；32 `34ccde7`；服务 aa3e/s1 root 1301295 在线；入口/指标/默认/CSV 不变；无新设备臂。
- **33 最终记录（已核）**：`docs/grouped-contract-research-2026-11-19.md` commit `87ffb96`、文件 SHA `f2326783…`，第 9–11 节确认 mc2_server/chip:5 证据与结案。**A13 更正**：`aclnnMatmulAllReduce`（V1/V2/V3）在 A3 文档不支持且本机 A3 op-info 未注册（仅 AllGatherMatmul/V2、MatmulReduceScatter/V2），运行会停在 op 查找（161001 类），不能作 MC2 路径控制；A13 代码保留未执行。若未来需要 A3 AICPU 控制，用 `aclnnMatmulReduceScatter` V1（A3 仅 AI CPU/KFC 路径，bias 非 0 不支持）与 V2 commMode="aiv"（A3 仅 aiv、通信缓冲 ≥200MB）；但 loader 直接拒绝证据已足够，当前不安排设备臂。

## 8.24 结论收窄（按 20，2026-11-19）：环境首因线索，非最终结案

- **修订说明**：8.21–8.23 的“decisive/closed/环境阻塞结案”表述为**历史轨迹**，按 20 的复核收窄；不删除原件与旧结论，只在此声明当前状态。
- **已确立事实（仅两项）**：① A12 官方样例失败（stream47/KERNEL_AICORE pending，507046@10s；两 rank）；② `mc2_server.tar.gz` 在 TDT 启动阶段因 chip:5 被 skip-load（device0/1，op launch 前），`cann-tsch-compat` 同被 skip。
- **未确立（不得当作已证）**：`mc2_server` 是 **optional**，且 AICPU scheduler 已从 `libccl_kernel.so` 加载 `RunAicpuKfcResInit/RunAicpuKfcSrvLaunch` 并完成 `AicpuRpcResInit`——**Grouped 是否实际依赖被拒的 mc2_server 包尚未证明**；H6（workspace 零初始化）与 MC2 context/AICPU 绑定（H12 的核心）仍未证伪。A13 的 MatMulAllReduce 在 A3 未注册（161001 类），**不能**作为公共 MC2 反证，停跑。
- **当前状态**：**环境首因线索（environment first-cause clue）；服务依赖/合法控制待核**。任何“H1–H12 全部结案”“Grouped 不可运行”的最终定性暂不成立。
- **后续必要控制（preflight 判据）**：
  - **A14（MatmulReduceScatter V1，A3 合法 AICPU 控制）**：33 先在本机 CANN 确认该 op 的 A3 op-info/kernel 注册与所需 AICPU 包（mc2_server vs aicpu_hccl）及 header/样例同版；35 实现最小 A14（x1/x2/bias/group/sum/commTurn0/streamMode1/output；A3 bias 非 0 不支持→零/空 bias；先小合法 shape，再 v4r1-like shape）；成功=API rc0+两 rank sync 返回+finite+实际 binary maps（libhccl/opapi/apapi 族/ascendcl/nnopbase）+无错误行+完成标记；失败同 stream47/mc2_server skip 记录。解释：A14 通过→合法 AICPU MC2 控制可用，被拒 mc2_server 非普遍阻断；A14 同挂→路径阻塞更广，但仍须先证明 A14 与 Grouped 同依赖该被拒包。
  - **A12-H6（最小派生）**：35 在官方 A12 源（6c1a90ab 不变）上只加 `aclrtMemsetAsync(workspace,0,size,stream)`（launch 前、同 stream、全 16,777,216 B、ACL_CHECK 记录 rc），diff/source/binary SHA 冻结、明确“官方源+最小派生”标签；其余（10s sync、args 2 1 0,1、build glue）不变；用 A12 臂同一窗口/a12_evidence（错误行/≥2 costtime/maps）。H6 返回=workspace 初值被支持；同挂=H6 削弱。单变量仅为 memset。
  - 设备门（任何新格）：fresh root-identity/snapshot（当前 1301295）、run/restore fresh、safe window（old_alive 绝不 Popen/pre-Popen 门/官方 aae-noop 恢复/240s/精确 PGID）、32 ready 后 31 执行；服务保持 aa3e。
- **A13**：停跑（未注册）；代码保留；未来 A3 控制只用确认注册的 MatmulReduceScatter V1/V2。

## 8.25 A12-H6 最小派生静态 preflight（未执行）

- **交付**：35 commit b4a658c；derivative source `4b8e6e2a…`、`h6.diff 754e2f24…`、compile `e7a59de6…`、manifest `ceb76693…`。`diff 官方源(6c1a90ab) vs h6` 与 h6.diff 完全一致，**仅新增 9 行 memset 块**（aclrtMalloc 后、launch 前）；`aclrtMemsetAsync(workspaceAddr, workspaceSize, 0, workspaceSize, rtStream)` 参数顺序正确、与 launch 同 stream（同流顺序保证先 memset 后 kernel）；rc 打印并 CHECK_RET；其余 provider/FP16 shape/2 设备 1 loop/生命周期未改。single-variable 成立，A12 臂/a12_evidence 可复用。
- **待修/待核（3 项）**：
  - **记录一致性**：manifest 顶层 status 仍 `closed_environment_blocked`，与 20 收窄冲突 → 改为“environment first-cause clue / dependency & legal control pending”，H6 条目保留 prepared_not_executed；31 conclusion.json 同样收窄（旧文本留轨迹）。
  - **编译脚本注释**：`compile_a12_h6.sh` 头注释仍写 “Build the frozen official A12 sample unchanged”，应改为 “official source + h6.diff，derived build”（脚本验 source SHA 4b8e6e2a 正确）。
  - **API 可用性/约定**：31 编译前 grep 本机 CANN `include/acl/acl.h`（或 acl_rt.h）确认 `aclrtMemsetAsync` 声明；若无则改用同步 `aclrtMemset` 并记变体；33 确认 workspace 零初始化约定与是否需额外 sync（同流已足够；跨 stream/context 则加 sync 属另一变量）。
- **判据**：launch/sync rc0 + 无 `[ERROR]`/retcode + ≥2 mc2 costtime + per-PID maps 硬门 + memset rc0；仍 507046/stream47 → H6 削弱（非平台否定）；返回 → workspace 初值被支持（仍需区分 mc2_server 依赖）。
- **执行门**：不编译/不执行；待 33 约定 + 31 头检查 + 32 ready + 20 排程；fresh identity 1301295/safe window/A12 臂；服务保持 aa3e。

## 8.26 依赖分析与 H6 条件 ready（2026-11-19）

- **依赖分析（31 dependency-analysis `31f8f1ee…`）**：磁盘无 `libmc2_server.so`（仅在拒绝包内）；仅 `libmc2_server.json` 引用 `Mc2ServerKernel`，无 op-info/binary_info/op 配置引用；`libccl_kernel.so` 导出 46 个 KFC/HCCL kernel（RunAicpuKfcResInit/SrvLaunch、RunAicpuRpcSrvLaunchV2_* 等）及 MC2 机制符号，但**无 Mc2ServerKernel/RacServer**；已装 AICPU .so 无该符号；Grouped AscendC 源码含 `rac_server.h`（期望 AICPU server）；运行日志未指名 grouped 具体等待哪个 server。→ `dependency_status=**supported_but_unproven**`；判别待 A12-H6 与 A14。
- **31 conclusion 收窄**：`conclusion.json` `bfef1c02…`，status=`suspected_environment_blockage_pending_dependency_control`，revision_trail 记录 20 的收窄理由（旧 confirmed 文本保留）。
- **H6 修复核验（35）**：commit 1e090d7/1641360；manifest `a35d5321…`，顶层 status=`narrowed_environment_first_cause_pending`；source `4b8e6e2a…`/diff `754e2f24…` 未变；compile `8ef604f1…`，注释已改“official source 6c1a90ab + h6.diff, derived build”；H6 条目 `prepared_not_executed_pending_33_32_contract`（含 api_check/static_preflight/pending）。
- **剩余措辞项**：manifest `closure.root_cause` 仍把“MC2 KFC AICPU task has no implementation”写成事实；应改为“suspected first-cause: mc2_server skipped; grouped dependency unproven”或标为历史/证据摘要（非阻塞 H6）。
- **H6 条件 ready**：静态单变量/同 stream/9 行 memset/Async API（31 核 acl_rt.h:2563、libascendcl 导出）均通过；仅待 33 的 workspace 约定——同流顺序足够则不改；若需额外 sync，作为单独变体（不与 H6 控制混）。之后 + 20 排程，31 以 1301295 fresh identity、safe window、A12 臂（--a12-binary H6）执行。
- **A14** 等 33 的 A3 注册/依赖包结论；A13 停跑；服务 aa3e/s1 在线，无设备动作。

## 8.27 反证核验、H6 ready 与 A14 preflight（2026-11-19）

- **35 措辞修正**：36f8141/manifest `ab2a1855…`；source `4b8e6e2a…`/diff `754e2f24…`/compile `8ef604f1…` 未变；closure.root_cause=“suspected first-cause…unproven”、decision 已去泛化；manifest 无显式 revision 字段（旧 confirmed 文本在 git 历史，可加 revision_trail 作审计，非阻塞）。
- **33 反证**：依赖 `supported_but_unproven`（Grouped tiling taskType=KFC_TASK_HCC_TASK_DELIVER；A12 日志无 Mc2ServerKernel/HcclLaunchAicpuKernel/libmc2_server 请求/失败行；libccl_kernel 提供 KFC/HCCL 家族）。未解释线索：A12 plog 两 device 启动 TDT 有 `LoadSysOpKernel: cannot find aicpu packages`（06:24:20.367/.807），31 待判正常/相关。**独立核**：31 pack 的 A3 op-info/binary_index **确有 MatmulReduceScatter 与 V2**（A14 注册成立；MatMulAllReduce 确无）。
- **A12-H6 从 32 侧 READY**：33 约定 GetUserWorkspace 仅 GEMM scratch、MC2 消息区在 `GetHcclContext<0>()->WorkSpace`，memset 清零契约安全；同流顺序足够、**不加额外 sync**；Async 本机可用（acl_rt.h:2563、libascendcl 导出）。待 31 在 Linux stage/编译 H6 binary（AOT、记录 binary/ldd/log SHA）→ 1301295 fresh identity、fresh run/restore、A12 臂（--a12-binary H6）、safe window 出 plan-only → 32 核 binary/plan/identity → 20 排程后执行。判据：launch/sync rc0+无 `[ERROR]`/retcode+≥2 costtime+maps 硬门+memset rc0；507046/stream47→H6 削弱；返回→H6 上升（仍与 mc2_server 依赖分开）。
- **A14 preflight 待件**：31 提供本机 `aclnn_matmul_reduce_scatter*.h`（pack 缺；33 上游 `6e49c20f…`）、V1 kernel .o 符号与 tiling taskType（对照 KFC_TASK_HCC_TASK_DELIVER）、`opp/built-in/op_impl/aicpu/kernel/` 全量清单、libccl_kernel.so 路径/SHA；35 最小 A14（bias null/零；x1(m,k)、x2(k,n)或(n,k)；output(m/2,n)；sum/commTurn0/streamMode1；m 为 rank_size 整数倍：v4r1-like **M=968**，小格 M=128/K=256/N=512）；判据同 A12；驱动/固件兼容（compatible_fw [7.0.0,9.9.9]）列为待核。
- **A13 停跑**；不改白名单强载；服务 aa3e/s1 在线；未获 20 排程前无设备动作。

## 8.28 H6 GO 与 A14 KEEP_ALL 契约修正（2026-11-19）

- **H6 final-check PASS**：identity root 1301295/start 398656442（`03c437ee…`，与 plan 一致）、SUMS `4b4afbd4…` 6/6、window `51d6bc0c…`、H6 binary `cdf48553…`（209,656 B；compile log `d1fa7fbc…` 的 sha256sum 一致、ldd `fe47d0a0…`）、source `4b8e6e2a…`/diff `754e2f24…`/compile `8ef604f1…`。**20/用户已授权，32 GO**：31 立即执行单格 H6（d0_stop 1301295→H6≤240s→精确清理→官方 aa3e-noop 恢复）；回传两线程 memset rc、launch/sync/错误/costtime、per-PID maps、window.json、plog、恢复身份/四检。判据：rc0+无 `[ERROR]`/retcode+≥2 costtime+maps 硬门+memset rc0；507046/stream47→H6 削弱；返回→H6 上升；不外推平台。
- **A14 旧 plan 废弃**：31 的 d9b9070b（small）/8cdbebd5（large）基于旧 run_probe `331f9929…`（A14 flags=0）→ 早释放混淆，**不得执行**。35 已在 e0b06df 修为 **A14 flags=KEEP_ALL**（binding `101660f8…`、run_probe `6ae866b1…`、window `2202165e…`、manifest `1c8d2682…`；group/workspace/descriptors/tensors 保活到 sync 后 release，handler 不早 close），CPU 独立 **44/44**。
- **A14 流程**：31 从 e0b06df 重 stage/编译（adapter 若仍 101660f8 可复用 d181b15a，需重出 compile log）/重出 small plan M128/K256/N512；若 H6 先跑，用新 AA3E root 重建 identity/plan。小格 plan/binary/identity + 32 核 KEEP_ALL 契约后 **GO**（无需二次许可）；large M968/K8704/N5120 按小格结果与 32 有限核再定。
- **A14 判定**：通过→KFC 路径可用、被拒包非普遍阻断；同挂→KFC 路径阻塞更广；均不单独裁决被拒包依赖（仍 supported_but_unproven）。A13 编译-only 不执行；服务 aa3e 在线。

## 8.29 A14 重建核验与执行顺序（2026-11-19）

- **A14 stage-identity v2 `99a2dc04…`**：e0b06df / manifest `1c8d2682…` / binding `101660f8…` / run_probe `6ae866b1…`（A14 flags=KEEP_ALL，group/workspace/descriptors/tensors 保活到 sync 后）/ window `2202165e…` / adapter `d181b15a…` / compile log `7ce25dad…`。small plan `2ee38a28…`（M128/K256/N512）、large `bc1efa2d…`（M968/K8704/N5120），argv/240 s/adapter/identity 均核；root-identity `03c437ee…`=1301295/398656442。KEEP_ALL 契约正确（无早释放混淆）。
- **顺序**：**H6 GO 已发出，先执行 H6**（plan aedd3f68/window 51d6bc0c/binary cdf48553/identity 1301295）。H6 恢复后 AA3E root 变；**当前 A14 plans 基于 1301295，届时作废**：H6 后重建 fresh root-identity → 重出 A14 small plan（M128/K256/N512，KEEP_ALL，window 2202165e，adapter d181b15a）→ 32 核 identity/plan（代码/args 不变）→ A14 small **GO**（无需二次许可）；large 等 small 结果 + 32 有限核。若选择 A14 先于 H6，当前 small plan 有效，但 H6 必须用 A14 恢复后的新 root；不得跨格复用旧 plan。
- **A14 small 判据**：run_probe pass（API rc0+两 rank sync+finite+both）+ window child_ok（rc0+maps adapter/libhccl/ascendcl）+ reference diff 记录；失败记录 stream47/mc2_server。minor：input metadata 记 bias zero 但 run_mrs 实际传 nullptr（op-info bias optional），参考无 bias，数值口径一致。
- **解释**：A14 通过→KFC 路径可用（被拒包非普遍阻断）；同挂→KFC 路径阻塞更广；均不单独裁决被拒包依赖（仍 supported_but_unproven）。A13 不执行；服务 aa3e。

## 8.30 H6 结果（限定）与 A14 small GO（2026-11-19）

- **H6 cell（a12-h6-cell 10/10，SUMS `e88d9d23…`）**：两 rank `h6 workspace memset api=aclrtMemsetAsync rc=0`；随后 h6.cpp:392 `507046` + `TestCall execute AICPU failed`；plog-debug 两 device `stream_id=47, KERNEL_AICORE, task_id=2, 10s, 0x7030010`；无 costtime；window ok=false、child rc0、maps_ok=true；恢复 root **1305145**/start 398845277、五 PID、aa3e-noop、四检/env/joint/source/clients 全 true。**结论限定**：user workspace 初值假设对该挂起被削弱/证伪；**HcclContext 的消息 workspace 属另一资源域，未被该格检验**；平台/服务包因果仍待 A14 与依赖证据，不外推。
- **A14 small v2（plan-only）核验 PASS**：plan `18fdf185…`、identity `4ace4a79…`=1305145/398845277、stage-identity-v3 `50b659a6…`（e0b06df/manifest 1c8d2682/binding 101660f8/run_probe 6ae866b1 KEEP_ALL/window 2202165e/adapter d181b15a/compile 7ce25dad）、SUMS `2e1086eb…` 13/13；旧 1301295 plans 作废。**A14 small GO 已下发**。
- **torchrun/PGID/安全窗语义**：window 以 `start_new_session=True` 启动 child=torchrun（session/PGID leader）；`--nproc_per_node=2` 两 rank 在同一 PGID；window 身份校验 `pgid==child.pid`、cleanup `killpg` 精确该组；probe 240 s watchdog + window 240 s child timeout；child rc0 需两 rank pass；window child_ok 加 maps（adapter+libhccl/ascendcl）；固定 `--master_port 29531` 为 torchrun rendezvous 端口——**31 执行前须预检其空闲**（有监听则 fail/记录，未获准不得改 plan 参数）；恢复门/identity/aa3e-noop 同前。
- **证据**：成功/失败均保留 rank0/1 result.json（api/sync/finite/both/pass、reduced max_abs/bit_equal、group/world、versions）、trace、child 日志、window.json/per-PID maps、plog；结束完整恢复（五 PID/双 worker/四检/env/joint/clients/branch aa3e-noop）。
- **解释**：A14 通过→KFC 路径可用、被拒包非普遍阻断；同挂→KFC 路径阻塞更广；均不单独裁决被拒包依赖（仍 supported_but_unproven）。large M968/K8704/N5120 等 small 结果 + 32 有限核，且须用 small 恢复后的新 root identity 重建 plan。A13 不执行；服务 aa3e。

## 9. 本地核对副本

本机 `/tmp/clef32-audit/` 保留 ebca 71 原件、rstd8 window/probe/timing 与 cell/timing bin（diag/rstd）、AD window/probe、cc34 双 manifest 与 entry；均与远端逐 SHA 对齐（见各节哈希）。远端原件路径如各节所示。
