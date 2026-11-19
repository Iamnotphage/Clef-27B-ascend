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

## 8.31 A14 small 首次执行 = harness 失败（非 MC2），GO 暂停（2026-11-19）

- **执行**：plan `18fdf185…`；d0_stop 1305145 成功、torchrun 2 ranks 启动；约 4 s 后 rank1 exit1；child.stdout 仅 `ERR99999`；child.stderr `run_probe.py:243 out.mkdir(parents=True, exist_ok=False)` → `FileExistsError`（rank0 先创建 `probe-out-a14-small-v2`），rank0 SIGTERM；probe-out 仅 rank0 stub/空 trace；**无 507046/stream47，未进入 MC2 调用**。该格不作为 A14 判据。
- **根因/修复**：harness mkdir 竞争；最小修复 `out.mkdir(parents=True, exist_ok=True)`，per-rank `rank_dir.mkdir()` 保持 `exist_ok=False` 以保留 fresh 检查。35 重冻 run_probe 新 SHA；31 先完成 safe window 官方 aa3e 恢复（回传新 root/五 PID/branch/四检/restore 原件），再从新冻重 stage/编译/重出 small plan（新 root identity、新 run_probe SHA、新 probe-out 路径），预检 `master_port 29531` 空闲；32 核后重发 A14 small GO。
- **状态**：A14 仍未检验；H6 限定结论（user workspace 初值削弱）不变；依赖 `supported_but_unproven`；A13 不执行；服务恢复 aa3e（新 root **1306695**）。
- **失败格归档核验**：a14-small-cell 10/10 OK；window ok=false、child rc1、无 survivor；child.stderr FileExistsError（run_probe.py:243）；无 507046/stream47（未进 MC2）；rank0 stub/空 trace；restore manifest pid 1306695、joint-record branch aa3e-noop/ok true；恢复五 PID/四检全 true。
- **修复最终冻结**：f087746/manifest `02fdd342…`（binding `101660f8…`、run_probe `b7f255cb…`（prepare_out exist_ok=True + rank_dir fresh 门、A14 KEEP_ALL）、window `2202165e…`）；代码与 d571641 相同，CPU 独立 **45/45**；a14_control 记录 small_v2_aborted_pre_launch/fixed_d571641_v3_pending。
- **v3**：31 从 f087746 重 stage；adapter d181b15a 可复用（binding 未变，重出 compile log）；root 1306695 fresh identity；probe-out 全新（probe-out-a14-small-v3）；master_port 29531 预检空闲；small plan M128/K256/N512/KEEP_ALL/window 2202165e/240 s；stage-identity v4/SUMS → 32/33 核后 GO；证据与完整恢复要求同前；large 等 small。
- **v3 最终核验/GO**：不可变 tuple = commit **f087746**/manifest **02fdd342**（代码与 d571641 逐字节相同：binding `101660f8…`、run_probe `b7f255cb…`、window `2202165e…`）、adapter `d181b15a…`（compile `fc8efe47…`）、identity **1306695/398875478**（`a85ff9b8…`）、plan **8782537a…**（M128/K256/N512/KEEP_ALL/out probe-out-a14-small-v3/run+restore v3 fresh/240 s/plan_not_executed）；stage-v5 `a3b12870…`、SUMS 17/17（`4eaf108e…`）、master_port 29531 FREE。早前 d88250e4 为 v4 旧 plan，已 superseded。**A14 small v3 GO 已发**；回传 rank0/1 result.json（api/sync/finite/both/pass、reduced max_abs/bit_equal、group/world、versions）、trace、child 日志、window.json/per-PID maps、plog、完整恢复；large 等 small 结果 + 32 有限核，用 small 恢复后新 root 重建；A13 不执行。
- **33 修复独立核**：run_probe `b7f255cb…`（20,278 B）diff 仅新增 `prepare_out(exist_ok=True)` 并保留 `rank_dir.mkdir()` fresh 门；binding/window 未变；`test_probe_cpu.py ea514245…`（7,708 B）含 `test_out_dir_shared_by_two_ranks`，实际 **14/14**、总 **45/45**（14+9+22，32 独立跑）。**metadata finding（非阻断）**：manifest cpu_tests 仍写 13/13、role tests(11)，与实际不符；35 在 docs-only v3 metadata 修正，但不使已冻结执行 tuple 失效（无需重编译/重跑）。

## 8.32 A14 small v3 liveness = KFC 路径可用（2026-11-19）

- **trace（rank0/1，probe-out-a14-small-v3）**：prepare_rc=0/executor_nonnull；handler workspace `0x12c09aa00000`、bytes 17,170,432、rows128/k256/n512/rank2；**launch_rc=0**；**sync_rc=0**；sync_ms 18.66/8.16；**无 stream47、无 507046** → MRS V1 MC2 KFC 路径在小形状正常同步返回。A14 liveness 成立：**被拒 mc2_server 包非普遍阻断**。
- **崩溃点（harness）**：sync 后 `run_probe.py:353 release` 的零 Tensor 参数自定义算子 dispatch `NotImplementedError: clef_mmar_probe::release ... no fallback`；result.json 仅 stub、reference diff 未落盘；与 MC2 无关。
- **修复/重跑**：35 最小修 release schema（占位 token tensor 或 fallback）+ 先 flush result 再 release + try/except；重冻 run_probe 新 SHA。31 先完成官方 aa3e 恢复（新 root **1308632**）并回传身份/五 PID/branch/四检/原件；修后以新 root 重建 identity/plan、fresh out（v4）、master_port 预检，重跑同小格取完整 result.json（api/sync/finite/both/pass、reduced max_abs/bit_equal、group/world、versions）。large 等完整 small + 32 有限核。
- **解释**：Grouped 仍 507046/stream47；A14 可用 → 差异转向 Grouped-specific（op/kernel/tiling/rac_server 协议/上下文）与 H12/依赖证据；**仍不单独裁决 mc2_server 依赖（supported_but_unproven）**。H6 限定结论、A13 不执行不变；服务恢复 aa3e。
- **35 记录核（7bb4dd8）**：run_probe `b7f255cb…`、manifest `6517c34c…`、cpu_tests **14/14+9/9+22/22=45/45**（与 32 独立跑一致）；v2 invalid/旧 plan superseded 记录正确。**但 a14 状态仍写 “small_v3_plan_8782537a_pending_32_review”**，与已执行的 v3 不符（liveness pass + release harness crash；新 root 1308632）——已请 35 改为 `small_v3_executed_liveness_pass_release_harness_crash; numeric rerun pending release fix` 并保留 trace/原始日志引用。
- **release 修复**：schema 占位 token tensor 或 fallback；先 flush result 再 release；try/except；重冻 run_probe 新 SHA；31 完成 1308632 恢复回传后以新 root 重建 identity/plan、fresh out（v4）、port 预检，重跑同小格取完整 result.json；large 等完整 small + 32 有限核。
- **v3 cell 原件核验（a14-small-v3-cell 16/16）**：rank0/1 trace prepare_rc0/executor、workspace 17,170,432、launch_rc0、handler cleanup0/sync0、sync_ms 18.66017/8.16125；无 stream47/507046；result.json api/sync=true、finite/pass=false（stub）、error 为 release 零 Tensor dispatch；window ok=false/child rc1；恢复 root **1308632**/start 398926792、五 PID、aa3e-noop、四检/env/joint/source/clients 全 true。33 独立核一致（trace/plog/result/window/restore）。
- **release 修复（33 细化）**：① schema 占位 Tensor 或 fallback；② **先完整计算 reduced/reference/bit_equal/finite/versions 并原子落盘 result，再 release**；release 包 try/except 记录 release_rc/error 后重 flush；③ 重冻 run_probe 新 SHA/stage identity；31 用 1308632 重建 identity/plan v4、fresh out、port 29531 预检，重跑同小格取完整结果；large 等完整 small + 32 有限核。H6/A13/official 不变；服务 aa3e root 1308632。
- **33 最终归档核（v3 收口）**：SUMS `88b823a4…` 16/16 OK；commit `064d469`（17 文件）；window `c291b877…`；rank0 result `33410194…`/trace `1fb6bf87…`、rank1 `300eebfd…`/`7e7b05a1…`、child stdout `cb1aa6ed…`/stderr `37177294…`、restore manifest `5deee436…`/joint `861044f4…` 均命中；restore branch aa3e-noop、root 1308632/start 398926792、五 PID、四检全 true、无残留。**A14 v3 收口 = KFC liveness 成立 + release harness 待修**；后续 35 修复重冻 → 31 v4 重跑完整数值 → large；Grouped-specific/依赖口径不变（supported_but_unproven）。
- **release 修复方案（33 提出，32 确认采用）**：b7f255cb 中 release 在 reference/finite 之前；改为**先计算 reduced/reference/bit_equal/finite/versions 并原子写完整 result，再 try/except release**（记录 release_rc/release_error 后重写）；数值门只看 api/sync/finite/reference，release 失败记 harness_error 不抹数值；子进程正常退出、资源 teardown 回收。**run_probe-only，不改 binding/adapter**（d181b15a 复用）；schema 占位 Tensor/fallback 仅作未来显式 release 门，非本次必需。重冻 run_probe 新 SHA；31 以 root 1308632 重建 identity/plan v4、fresh out、port 29531 预检，重跑同小格取完整 result；large 等完整 small + 32 有限核。

## 8.33 v4 审核清单与 matched Grouped H11 小格设计（2026-11-19）

- **v4（run_probe-only release 修复）增量审核清单**（不重开 stage/不重编译）：release 移到 reference/finite/gate 之后；先原子写完整 result 再 release；release try/except 记录 `release_rc`/`release_error` 后原子重写；`pass`/numeric 只看 api/sync/finite/reference，release 失败记 harness_error、不抹 numeric；trace release 标记与 child rc 一致；window child_ok 仍 rc0+maps；KEEP_ALL 与 fresh out/rank_dir 门保留（prepare_out exist_ok=True + rank_dir exist_ok=False）；仅核新 run_probe SHA/identity 1308632/plan v4（fresh out、port 29531、adapter d181b15a 复用、window 2202165e）；binding 未变不重编译。35 新冻 + 31 v4 identity/plan 齐后直接给同小格 GO（无需二次确认）。
- **matched BF16 单 group 小形状 Grouped（H11）**：flags=KEEP_ALL，M128/K256/N512、BF16 x/w、**FP32 zero bias**（Grouped 契约）、单 group、torch 多进程 comm、KEEP_ALL、240 s、fixed run_probe。与 A14 small 的 op 接口差异记录：bias null vs FP32 zero、output reduce-scatter vs all-reduce、op MRS vs Grouped。证据：trace prepare/launch/sync + result（api/sync/finite/both/pass、reduced max_abs/bit_equal、group/world/versions、release_rc/error）+ maps + plog + 恢复；reference=matmul+all_reduce（bias zero）。
- **可判别性**：H11 小格同挂 stream47/507046 → Grouped op/kernel/tiling/rac_server/上下文为差异源（shape/comm/KEEP_ALL 已匹配）；若返回 → 挂起与较大 shape/tiling（M967/K8704/N5120）相关；若 prepare 失败 → 记录 op 形状约束、换合法匹配形状。顺序：A14 v4 完整 small → matched Grouped H11 小格（A14 v4 恢复后新 root identity、fresh out、port）→ large（A14 M968/K8704/N5120 等完整 small + 32 有限核）。
- 口径：依赖 supported_but_unproven；H6 仅 user workspace 初值；A13 停跑；official/指标不变。

## 8.34 v4 pass 语义阻断（方案①）与 H11 matched 合法性（2026-11-19）

- **v4 阻断**：5ada848 的 `meta['pass']=numeric_pass and harness_release_ok`（L413）、`return 0 if pass else 1`（L420）、destroy_process_group 仅 pass 下；release 零 Tensor 必失败 → rc1/window false，重演 v3 门。**采用方案①（run_probe-only，推荐）**：`pass=numeric_pass`；`harness_pass=numeric_pass and harness_release_ok` 单独记录；`destroy_process_group()` 移到 numeric_pass/finally；`return 0 if numeric_pass else 1`。binding/adapter/window 不变（101660f8/d181b15a/2202165e），仅 run_probe 再冻；方案②（schema/fallback+重编）暂不做。31 当前 v4 plan `47907073…`/stage-v6 基于 strict pass，作废；新冻后重 stage/identity/plan（1308632、fresh out、port）。验收：numeric_pass=true + reduced/reference/bit_equal/versions + trace 齐全；release_error/harness_release_ok=false 单独报告；rc0/window 由 numeric 决定；恢复完整。
- **H11 matched 合法性（33）**：M128/K256/N512、BF16 x/w、**FP32 zero bias**、单 group、H11 flags=KEEP_ALL、同 adapter d181b15a/同 torch comm/240s；与 A14 仅 op 变化（bias null vs FP32 zero、output rs vs ar 记录）。A8 native=false 仅 reference/comm health，A0 flags=0 不匹配；选 H11。顺序 A14 v4 完整 numeric → H11 fresh identity/plan → large。H11 同挂→只列 Grouped 特异 op/协议/rac_server/context 候选；返回→shape/tiling 逐维；不单格定因。
- **plog 反证（33）**：A14 v3 同 chip:5 skip mc2_server 但 prepare/launch/sync rc0 → **skip 非充分原因**；Grouped 必要依赖仍 supported_but_unproven；Grouped tiling KFC_TASK_HCC_TASK_DELIVER/HcclServer+rac_server vs A14 fullmesh 标准 MC2 client 路径，列为优先候选（不单因）。v4r1 plog 本地缺口不依赖它。
- **v4 精确 diff（33，32 采纳）**：fdbfeb67 L408-420 → `meta['numeric_pass']=numeric_pass`（保留）；`meta['harness_pass']=bool(numeric_pass and meta['harness_release_ok'])`；**`meta['pass']=numeric_pass`**（window/child 看 numeric，严格 harness 另字段）；`trace.emit('done', passed=meta['pass'], harness_release_ok=...)`；`if numeric_pass: dist.destroy_process_group()`（或 finally，**两 rank 必须同走**，避免单侧悬挂）；`return 0 if numeric_pass else 1`。预期：release NotImplementedError 时 numeric_pass=true、harness_pass=false、child rc0、window 可 ok（rc0+maps），result 保留 reduced/reference/bit_equal/versions 与 release_error。35 重冻 run_probe 新 SHA/manifest/test（46/46 预期）；binding 101660f8/adapter d181b15a/window 2202165e 不变；31 旧 v4 plan 47907073/stage-v6 作废，新冻后 stage-identity v7（仅 run_probe SHA/路径增量）、root 1308632 重建 v4 identity/plan、fresh out、port；32 核后直接 GO。
- **v4r1-plog 核验（6/6）**：device0 plog-1255361 L42/L91-93、device1 plog-1255362 L42/L91-93 均 chip:5 skip mc2_server；同 plog device0 L409/device1 L388 00:40:09 `three minutes timeout! stream_id=47, task_id=36, pendingNum=2`；33 独立核一致（含 KFC task/tiling trace 对象已建但 AICORE pending，不能据此判 server 已响应）。三路：A12/v4r1 挂、A14 sync rc0、skip 相同 → **skip 非充分原因**；Grouped-specific op/kernel/tiling/rac_server/context 为优先候选；依赖 supported_but_unproven。
- **v4 清理覆盖（20）**：`destroy_process_group()` 不能 `if numeric_pass`（单 rank 数值不同会单侧跳过）；应在 **finally** 中、以 `dist_ready=True`（init 成功后置位）覆盖所有已初始化 rank 的成功/异常路径；**不新增 barrier/其他 collective**。`pass`/exit=numeric；`harness_pass`/`release_error` 单列；trace done 带 harness_release_ok。35 按 33 diff + finally 覆盖重冻；31 旧 v4 plan 47907073/stage-v6 保留 superseded_not_executed（不删原件），新冻后 stage-identity v7、root 1308632 重建 v4 identity/plan v5（fresh out v5、port 29531）；32 增量核后直接 small GO。H11 顺序不变；H6 限定、A13 不执行、official 不变；服务 aa3e root 1308632。

## 8.36 v4 方案①终冻 f3e76cc 核验（2026-11-19）

- **冻结**：commit f3e76cc；manifest `5531e104…`；run_probe `0f90fd23…`（21,854 B）；test_probe `a8bf075f…`（9,091 B）；binding `101660f8…`、window `2202165e…`、adapter `d181b15a…` 未变（不重编）；CPU 独立 **46/46**。35/33 独立核一致（33 研究 commit 6787b0a）。
- **代码（0f90fd23）**：result 先 flush 再 call_release；release_error/harness_release_ok 记录并重 flush；numeric_pass=api+sync+finite+all(both)（norm 含 gate）；harness_pass=numeric_pass and harness_release_ok；**pass/退出码=numeric_pass**；trace done 写 passed+harness_release_ok；finally 中 `dist.is_initialized()` 时 destroy_process_group（两 rank 成功/异常都走，**无新增 barrier/collective**），cleanup 异常写 `harness_cleanup_error` 并 flush、不改退出码；240 s 窗口兜底。
- **v4 预期**：release 无 fallback → release_error=NotImplementedError、harness_pass=false，但 numeric_pass=true、child rc0、window 可 ok（rc0+maps）；result 保留 reduced/reference/bit_equal/finite/versions/group/world。
- **31**：从 f3e76cc stage-identity **v7**（仅 run_probe SHA/路径增量）、root **1308632/398926792** 重建 identity/plan **v5**、fresh out **probe-out-a14-small-v5**、port 29531 预检；旧 47907073/stage-v6 标 superseded_not_executed（原件不删）；32 核 run_probe/身份/路径增量后直接 small GO。H11 文档 `1294559d…` 已核（现有 H11 臂、无新代码/臂、未排设备）；v4 完整 numeric 后 H11 matched small。H6 限定、A13 不执行、official 不变；服务 aa3e root 1308632。
- **v5 最终核验/GO**：plan **b6429191…**（f3e76cc run_probe、A14、M128/K256/N512、KEEP_ALL、out/temp/run/restore **v5** 全新、240 s、plan_not_executed）；stage-identity-v7 **06b7c600…**（small_plan_v5=b6429191、preconditions 已改 **probe_out_v5: absent at plan time (verified)**，33 字面 finding 解决）；identity **eb3a4b42…**=1308632/398926792；SUMS 22/22；port 29531 FREE。旧 v4 **6da7309c**/out v4 保留未执行（superseded_not_executed）。**v5 GO 已发**；验收 numeric_pass=true + reduced/reference/bit_equal/finite/versions/group/world + trace；release_error/harness_pass=false/harness_cleanup_error 单列；rc0/window 按 numeric；恢复完整。

## 8.37 A14 v5 完整 numeric（2026-11-19）

- **原件核（a14-small-v5-cell 16/16）**：rank0/1 `numeric_pass=true、pass=true、api/sync/finite/both=true`；`reduced.max_abs=0.0、bit_equal=true`（与 reduce_scatter 参考逐位一致）；`release_rc=null、release_error=NotImplementedError、harness_pass=false、cleanup=null`（数值先 flush、不受影响）；result 的 libraries 含 libascendcl `1c06b9c4…`/libhccl `da6dd17f…` 等、provider libopapi+libnnopbase；trace sync 20.65 ms、无 stream47/507046；window child rc0、survivors []、**ok=false 仅 maps 硬门**（只捕 torchrun 父 1311355、binary_pids 空，rank 1311385/1386 采样前已退出；per-rank libraries 在 result 内）；恢复 root **1311514**、五 PID、aa3e-noop、四检全过、无残留。
- **有限结论**：A14（MRS V1）在同 skip mc2_server 下 **KFC 路径 + 数值均可用（bit_equal）**；Grouped 仍 stream47 pending → Grouped-specific 候选优先；依赖 supported_but_unproven。maps 门失败是窗口对快速 torchrun 进程的采样伪影，numeric 证据完整。
- **H11 准备（31 CPU-only）**：以新 root **1311514** 重建 fresh identity；H11 matched small（M128/K256/N512、BF16 x/w、FP32 zero bias、单 group、KEEP_ALL、adapter d181b15a、window 2202165e、fresh out probe-out-h11-small-v1、port 29531 预检、240 s）；stage-identity/plan/SUMS；33/35 合同（h11_matched_control.md `1670bc9b…` 已核）；32 核新鲜 identity/plan 增量后直接给一次 H11 GO。
- **H11 验收/解释**：api/sync/finite/reduced max_abs/bit_equal + trace/plog；release_error/harness_pass 单列；window maps 门可能同伪影（per-rank libraries 在 result）；恢复完整。同挂→只列 Grouped 特异 op/协议/rac_server/context 候选（保留 bias/output/collective 差异）；返回→shape/tiling 逐维；不单格定因。large 继续等待。H6 限定、A13 不执行、official 不变；服务 aa3e root 1311514。

## 8.38 maps 加载门接受与 H11 GO（2026-11-19）

- **maps 事实（32/33/35 独立核）**：A14 v5 rank0 pid 1311385/rank1 1311386 的 `result.libraries` 各 17 项，来自各自 `/proc/self/maps`（run_probe `library_evidence` :211-241 读本进程 maps；:330 在 op 前写入 result），含 adapter **d181b15a**、libhccl **da6dd17f**、libascendcl **1c06b9c4**、libopapi **eb1c788e**、libnnopbase **72cd43b4**、torch_npu **65e6be19**；provider 经 binding `symbol()` **dladdr** 校验 + realpath。window.ok=false 仅因外部采样先捕 torchrun 父、rank 已退出 → **采样时序伪影**；现有实测证据补齐加载门，原样保留 window.ok=false 并注明。**不改 maps 代码、不重跑 A14、不重命名**（35 的 rank-local maps-self 持久化仅作未来硬化）。
- **H11 GO（唯一 plan e159c897）**：plan `e159c897…`（torchrun 2 + f3e76cc run_probe --arm H11、k256/n512/rows128、BF16 x/w、FP32 zero bias、单 group、KEEP_ALL、240/90、out probe-out-h11-small-v1、run/restore h11-small-v1 fresh、plan_not_executed）；stage-identity `e45db9da…`（H11 arm=KEEP_ALL、reference=matmul+all_reduce、matched differences、precondition port free、root identity `9802c322…`=1311514/399079056、旧 68cc5a79 renamed/superseded_not_executed）；SUMS 4/4。freeze f3e76cc/5531e104/0f90fd23/a8bf075f/101660f8/2202165e/d181b15a 保持。
- **H11 验收**：numeric_pass=api/sync/finite/both；**精度按实际 reduced max_abs/bit_equal 与原精度口径单列，finite≠精度通过**；release_rc/release_error/harness_pass/harness_cleanup_error 单列；per-rank libraries + provider 补加载门；恢复完整。判别：同挂 stream47/507046→Grouped-specific op/kernel/tiling/rac_server/AivAicpu 候选（A14 无 KFCtaskAndTiling/AivAicpuOpParam）；返回→shape/tiling 逐维；prepare 失败→记录形状约束；不单格定因。若已启动则继续、不第二轮；large 待后续；H6 限定、A13 不执行、official 不变；服务 aa3e root 1311514。

## 8.39 AICPU 协议证据（33）与 H11 状态（2026-11-19）

- **协议痕迹对照（33 只读）**：A12/v4r1 各 device 出现 2× `AicpuComTraceData`/`SqeBatchInfo`/`End AicpuRpcResInit`/`RunAicpuKfcSrvLaunch`/`KFCtaskAndTilingTraceData`/`AivAicpuOpParam`；A14 除 `RunAicpuKfcSrvLaunch` API 加载外均为 0，且使用 `AicpuRpcResInitV2`。结合 Grouped kernel 的 `HcclServer`/`AivAicpuOpParam` 源码 → **rac_server/AivAicpu 为 Grouped-specific 候选**；**日志缺失不作未执行证明**；mc2_server 必要依赖仍 `supported_but_unproven`。
- **H11**：plan `e159c897…`/stage-identity `e45db9da…`/SUMS `8f781003…` 4/4/root `9802c322…`=1311514/399079056/port FREE/adapter d181b15a 与 31/33 一致；旧 68cc5a79/out probe-out-h11 保留 superseded_not_executed；H11 GO 已发（一次），31 执行中/待执行，若已启动不第二轮。
- **验收不变**：numeric_pass + 实际 reduced diff/bit_equal 单列（finite≠精度）；release/harness 单列；per-rank libraries；restore；同挂→Grouped-specific 候选，返回→shape/tiling 逐维；不单格定因。等 H11 raw（31 回传，33 只读核）；large 待后续；H6 限定、A13 不执行、official 不变；服务 aa3e root 1311514。

## 8.40 H11 运行中：sync_rc 语义与等待（2026-11-19）

- **源码事实（f3e76cc/binding 101660f8）**：`int sync_rc=0; if (flags & DIRECT_LAUNCH) sync_rc=aclrtSynchronizeStream(...)`。H11 flags=**KEEP_ALL(8)**，不含 DIRECT_LAUNCH(16) → handler_return 的 **sync_rc=0 是默认值、未调用同步 API**；cleanup_rc=0 仅清理由 keep_descriptors 决定。实际同步是 run_probe 的 **Python `torch.npu.synchronize()`**（trace sync_enter→sync_return），当前尚未返回。**不得把 handler_return.sync_rc 当作已完成同步**（35 核对一致）。
- **当前口径**：host handler 已返回（launch/cleanup rc0、sync_rc 默认 0）；device-wide sync 等待中；未见最终 plog 前**不预判 stream47/507046**。等原 240 s：probe watchdog os._exit(124)/window child timeout→精确 PGID 清理→官方 aa3e-noop 恢复；不中断、不第二轮。
- **H11 最终证据/判定**：rank0/1 trace（prepare/launch/handler_return、sync_enter 是否 sync_return、timeout）、result/stub、window.json/child rc/cleanup、plog（stream/task/context/KFC/AivAicpu）、restore 身份/五 PID/四检/branch。若 device sync 始终未返回 → H11（matched small Grouped）悬挂，在同形状/同 comm/同 KEEP_ALL 下指向 **Grouped-specific 候选**；若返回 → 按实际 reduced diff/bit_equal 单列（finite≠精度）、shape/tiling 逐维；均不单格定因。
- v5 已收口；large 不排；不发新计划；H6 限定、A13 不执行、official 不变；恢复后服务保持 aa3e。

## 8.41 H11 matched small 结果（2026-11-19）

- **原件核（h11-small-v1-cell 15/15，SUMS `57932c67…`，commit f576d23）**：trace rank0/1 flags=8（KEEP_ALL）、prepare_rc0/executor、workspace 16,777,216、launch_rc0、handler cleanup0/sync_rc0（**默认、无同步 API**）、Python `sync_enter` 无 `sync_return`；240 s watchdog child rc **-9**/timed_out=true/survivors []；plog 两 device `stream_id=47, task_id=13, sqHead=13, pendingNum=1`（07:50:08.015/.016）；device 日志 L108/L111 建 **KFCtaskAndTilingTraceData + AivAicpuOpParam** + RunAicpuKfcResInit/V2/SrvLaunch（A12/v4r1 同型，A14 无）；mc2_server chip:5 skip；恢复 branch aa3e-noop、root **1313827**、五 PID、EngineCore+TP0/TP1、四检全 true、无残留。33 独立核一致（研究 commit dbc9a28）。
- **有限结论**：A14（MRS，同 shape/comm/KEEP_ALL/adapter/240 s）bit_equal 成功；H11（Grouped，同条件）在 Python device-wide sync 挂起、stream47/AICORE pending → **排除“仅大 shape/tiling”解释**；Grouped-specific op/kernel/tiling/rac_server/AivAcpu/KFC task 投递为优先候选（pending task_id=13）；bias/output/collective 固有差异保留；**不单格定因**；mc2_server 依赖仍 supported_but_unproven；H11 numeric 未产出（stub），精度不适用。
- **后续**：H4K=24/H5K=40 设计已批准（保活同 H11；H4K 仍线程+真实同步组合、H5K 单 stream/context 变量），当前不写码/不排设备，待 20/32 排程、用 H11 恢复后新 root 1313827 的 fresh identity/plan 再议；large 暂停；A13 不执行；H6 限定；official 不变；服务 aa3e root 1313827。

## 8.42 H5K-first 排程（2026-11-19）

- **授权**：H11 raw/恢复闭合后，20 批准 35 最小加入 **H5K=REFETCH_STREAM|KEEP_ALL=40** 与 **H4K=DIRECT_LAUNCH|KEEP_ALL=24**（仅 run_probe ARMS/window choices；binding/flags/adapter 零改）。先 H5K（唯一重取流变量）；H4K 仅备用。
- **H5K 单变量**：对 H11（flags=8）仅加 handler 内 `getCurrentNPUStream()` 重取；仍 OpCommand 线程、handler 无同步（sync_rc 默认）；实际同步仍 Python `torch.npu.synchronize()`；保活同 H11。
- **流程**：35 重冻（新 run_probe/window SHA/manifest/CPU）→ 33 合同核 → 31 以 H11 恢复后 root **1313827** 的 fresh identity 重出 H5K plan（M128/K256/N512、KEEP_ALL、fresh out/run/restore、port 29531 预检、adapter d181b15a、新 window SHA）→ 32 核 freeze/CPU/plan 增量后**直接给一次 H5K GO**（无需二次确认）。
- **验收**：rank0/1 result/trace（numeric 或 stub、sync_enter/sync_return、`source=cpp|python` 区分、prepare vs handler stream 同/异关系+device、KEEP_ALL sync_rc 默认）、per-rank libraries/provider、plog（stream/task/context/KFC/AivAicpu）、window/child rc/cleanup、restore 身份/五 PID/四检/branch；numeric 与精度/harness 分离；window maps 伪影如旧。
- **判定**：H5K 同挂 + 完整恢复 → 原件有限核后可放**一次 H4K=24** 组合对照（同形状；必须用 H5K 恢复后新 root 重建 plan；H4K 线程+真实同步组合，不单证 context）；H5K 返回 → 先核真实 sync/refdiff，不自动跑 H4K，先报告候选修复。large 暂停；A13 不执行；metadata 不重冻；official 不变；服务 aa3e root 1313827。

## 8.43 H5K/H4K 合同核清单与 Grouped 源码链补充（2026-11-19）

- **33 合同核清单（32 对齐）**：H5K=40/H4K=24；run_probe ARMS + window choices；binding **101660f8**/adapter **d181b15a** 不变不重编；KEEP_ALL 继承（keep_group/workspace/descriptors 全 true、id/run_id/release 语义保留）；同 shape/dtype/comm/bias（M128/K256/N512、BF16 x/w、FP32 zero bias、单 group_name_0、Grouped run）；H5K 仍 OpCommand 线程/无 handler sync、stream_compare 记 prepare vs handler；H4K 调用线程真实 C++ sync、trace source=cpp；Python `torch.npu.synchronize()` 为设备边界；240 s watchdog/window 兜底不变。顺序先 H5K；同挂→恢复后一次 H4K；不自动归因 context；H5K 返回先看 stream_compare 与 reduced/reference。
- **Grouped 源码链补充（33，commit 0e5f34c）**：Grouped Process 每轮 `TurnNotifyRun`、core0 `TurnWait` 轮询 `AivAicpuOpParam` rcvCnt；`HcclCombinOpParam` 定义与 libccl_kernel RPC V1/V2 源缺失，映射仅日志/符号推断——强化 **AivAicpu 投递候选**，不单格定因；H11 结论不重复。
- 当前无新 GO/plan；等 35 新冻（ARMS/choices + 新 run_probe/window SHA/CPU）→ 33 合同 + 32 增量核 → 31 以 root 1313827 fresh identity 重出 H5K plan → 32 直接一次 H5K GO。large 暂停；A13 不执行；metadata 不重冻；official 不变；服务 aa3e root 1313827。

## 8.44 H5K/H4K freeze 合同核、33 producer/consumer 事实与 H5K GO（2026-11-19）

- **freeze 0d3f7fb**：manifest `21758ff5…`、run_probe `508e2931…`（22,275 B）、window `d7e73514…`（31,650 B）、binding `101660f8…`/adapter `d181b15a…` 未变、tests 16/23/9=**48/48**（32 独立复跑）；diff 仅 ARMS（H4K=24、H5K=40）、choices、tests/manifest/README；33 合同核 **PASS**（flags/保活/单变量/跨进程规则/同一 shape/dtype/comm/bias/release/maps/240 s）。
- **H5K plan/GO**：plan `f10bfa01…`（arm H5K、flags 40、M128/K256/N512、out/run/restore h5k-small-v1 fresh、240/90、plan_not_executed）；stage-identity `5ff8a4aa…`（H5K=40 KEEP_ALL|REFETCH、H4K=24 standby、tests、preconditions）；identity `467b7c7c…`=1313827/399164390；SUMS 3/3；port 29531 FREE。**H5K GO 已发**（一次）；回传 trace（sync_enter/return、source=cpp/python、stream_compare+device）、result/stub、libraries/provider、plog、window/child/cleanup、restore；numeric 与精度/harness 分离；maps 伪影如旧。同挂+完整恢复→下一次 H4K=24；返回→先核真实 sync/reduced diff，不自动 H4K。
- **33 producer/consumer 事实（f5703e4）**：Grouped kernel `user1=GetUserWorkspace(workspace)`（GEMM scratch）与 `context->WorkSpace+notifyOff`（KFC 消息区）不同，HcclServer::Init send 在 win、response 在 win+64B；producer TurnNotifyRun 先 CrossCoreSet/WaitFlag，blockIdx0/AIC 写 sendCnt=curTurn/valid mask/Barrier+DCCI；consumer core0 TurnWait 轮询 `AivAicpuOpParam` rcvCnt>=totalTurn（totalTurn=Ceil(m/(splitM*singleM))），满足后清零/DCCI。**AivAicpuOpParam 的 commType/opType/sendBuffer/recvBuffer/count/stride/hcDataType/funID/everyTurn*/totalTurnCnt 在 official Grouped kernel/header/rac_server 无赋值点**（operation 参数应在 host tiling KFCMsgBody，kernel→AICPU 路径未证）；rcvCnt 初始化/递增/响应可见性在 AICPU/libccl 侧、源缺失。
- **H6 区分**：H6 memset 的是 aclnn user workspace，**不覆盖** `context->WorkSpace+notifyOff` 的 KFC 区；H6 负结果**不能证明**该区初值无影响。
- **ABI 证据**：已证 f5703e4 上游 Grouped kernel/tiling/def/rac_server 与 A14 main/tiling/fullmesh 字节 SHA、本机 op-info/kernel .o 名、libccl 符号/日志 API 名；未证本机 9.2.1 kernel/tiling 是否由 f5703e4 构建、AivAicpuOpParam/HcclCombinOpParam 字段偏移/大小/KFC task/funID 数值、V1/V2 RPC→具体 server 映射；**不把上游字段当本机 ABI 事实**。31 只读请求 ①-④（HcclCombinOpParam 头、本机 Grouped kernel/tiling .so strings/objdump、libccl 符号/反汇编、rac_server/aicpu param 头）可在 H5K 执行/恢复后补，不阻塞。
- 候选顺序：H5K →（同挂）H4K →（均挂）AivAicpu KFC producer/consumer（rcvCnt 初始化/递增/可见性）优先；H5K 返回先看 stream_compare/reduced，不自动唯一归因 context。large 暂停；A13 不执行；mc2_server 依赖仍 supported_but_unproven；official 不变；服务 aa3e root 1313827。

## 8.61 a00edad P0/P1/P2 修复审阅（2026-11-19）

- **freeze**：commit a00edad；manifest 778c37af；window **9a31091b**（46,020 B）；test_window 5fdc498f（34，5 Linux-only）；run_probe 0bd1fde8；cell 9/9、probe 17/17；binding 101660f8/adapter d181b15a、H4K tuple 未动；Mac **60 OK（skipped=5）**。
- **P0/F-A**：`OWNER_ENV='CLEF_WINDOW_OWNER'`（L132），window spawn 生成 `token_hex(16)` 注入 child_env（L827-831）；registry subreaper 分支仅 `_has_env_token(pid, token)`（`/proc/<pid>/environ` NUL 精确匹配，L132-142/L165-167）才登记；无 token 的无关 `python -c` 不登记/不杀；fixture 断言无关进程不在 snapshot、仍存活。
- **P1/F-B**：SIGKILL 后 `_reap(entries)` 遍历所有登记候选（含 Z，L307-312/L334-336），settle 后最终 scan；不依赖 rescan verified；Z `/proc` 残留消除待 Linux 验证。
- **P2/F-C**：`stop()` 新增记录时 sleep 0.1、最多 3 次（L228-242）；chain fixture 等 intermediate 孙代 pid 后再 start/stop。
- **未证（待 31 Linux 重跑）**：真实 torchrun/worker 是否继承 token、`/proc/<pid>/environ` 可读、Z reap 真机成立。33 代码级复核 PASS（commit af2acb4），无新阻塞；设备继续暂缓；A14 large fac2d948 未执行；A15 仅 CPU；large/A13 暂停。

## 8.62 Linux 安全门 PASS 与 A14 large GO（2026-11-19）

- **Linux fixtures PASS（32/33）**：fixtures-a00edad SUMS 3/3；`unittest.log` SHA `659754bd…`，`Ran 34 tests / OK`（含 5 Linux-only 真实）；F2 token worker 1323101 registered→killed→**reaped**（/proc 不存在/state null），**无 token 无关 1323102 存活且未登记**，registry 仅 [1323101]，survivors []；F3 登记 [1323106,1323107]（中间+setsid 孙代），cleanup 后两者均不存在，survivors []；无残留；freeze a00edad/778c37af/window 9a31091b/test 5fdc498f；服务未触 root 1319033/399382752/ppid1/pgid/health/config/TP0+TP1/card7 phy14/15/boot 未变。
- **A14 large GO（一次，原授权）**：31 更新 stage-identity（window **9a31091b**）、确认 identity **1319033/399382752** fresh；**fac2d948 保留不重命名**（child 参数不变）；执行 d0_stop(1319033)→A14 M968/K8704/N5120（KEEP_ALL）→240 s→新 window 精确清理（DescendantRegistry/subreaper/rank 组/survivors/recovery）→官方 aa3e-noop 恢复。
- **验收**：actual reduced max_abs/bit_equal 单列（不仅 finite）、workspace、MRS large 支持、KEEP_ALL、executor/资源、restore；失败/成功均保留；用途=替代路径 large 支持/workspace/数值可行性（非生产 NZ/非性能）。Grouped large/A13 暂停；A15 无设备 GO；official/CSV 未动。

## 8.63 A14 MRS large 事实与两 harness 缺陷（2026-11-19）

- **设备事实 PASS（32/33）**：a14-large-cell 19/19（SUMS a560be11，commit 4372c7f）；两 rank flags=8、prepare_rc0/executor、**workspace 115,818,496 B（110.5 MiB）**、launch_rc0、**Python sync_return 36.61861/9.46084 ms**、无 stream47/507046 → **MRS V1 large 支持/workspace/同步成立**；数值未取得。
- **缺陷1（数值阻断）**：run_probe 0bd1fde8 的 `output=y` 仅在 `arm not in (A14,A15)`；A14 未赋值 → `finite` 处两 rank `UnboundLocalError: output`，reduced/reference/bit_equal 未产出。修复：A14 分支在 norm/gate 前 `output=y`，重冻。
- **缺陷2（恢复误报）**：window 9a31091b 在 wait_health 后立即要求 manifest pid ppid==1 && pgid==pid；setsid wrapper **1323588** 仍父，root **1323589** 已健康却抛 `restored root identity mismatch`（restore checks/branch None）。manual：root 1323589/start 399707126/ppid1/pgid 1323589/sid 1323589、tree 五 PID、branch aa3e-noop、OPP9/LD12、health/info/text/image 200、clients 空、无第二树。修复：restore_timeout 内重试到 wrapper 退出/reparent，或接受已登记 wrapper 父+tree 同 PGID 后再复核 ppid1；**不得启动第二 restore**。
- **服务**：root **1323589** 健康。**决定**：35 修两缺陷重冻后**重跑一次 A14 large 取数值**（同 shape、新 run_probe+window、fresh identity 1323589、新 plan、一次 GO）；验收 actual reduced max_abs/bit_equal（不仅 finite）、workspace、sync、restore。Grouped large/A13 暂停；A15 无设备 GO；official/CSV 未动。

## 8.64 subreaper 恢复边界：120d4e7/3d8667a 审计（2026-11-19）

- **根因确认（原始事实）**：a14-large-cell/window.json 记录 `subreaper=true`、restore.wrapper_pid 1323588、manifest root 1323589、error=`restored root identity mismatch`、restore_checks=null；报错时 root 实际 ppid **未记录**，31 未推定。旧 window 52c10dce 全程保持 subreaper → wrapper 退出后 daemon 被 window 收养，`ppid==1` 在 window 存活期不可达；报错后 window 退出才 manual 见 ppid1，与此相容。
- **修复（120d4e7 → 3d8667a）**：新增 `enable/disable_subreaper`（prctl 36）与 `begin_restore()`：**先 `registry.stop()` 最终记录、`cleanup_attributed(..., reap=subreaper)` 清理/reap、survivors 为空、`recovery_gate` 通过，才 disable**；disable 失败置 `blocked_subreaper_restore`/`restore_blocked` 并 raise，**不 Popen、不第二 restore**；成功后 `start_restore`。后续 wait_health→manifest/identity→`wait_restored_root`（严格 ppid1/pgid-self）→tree PGID→branch/env/http/clients 检查保持不变。
- **调用点审计**：全文件仅一个 `begin_restore`/`start_restore` 入口（L998/L130/L138）；normal/timeout/exception 路径全部汇入同一 finally，cleanup 先于 gate；所有元组无重试循环。`recovery_gate` 的 `stopped` 参数未使用，门按实际状态（old_alive/port/old_pgid/residual/attributed/runners/source）判定，属状态式安全门。`disable_subreaper` 无条件置 0（无法读取前值；本部署 window 非 subreaper，0 即前态），已在记录中注明。
- **测试**：120d4e7 漏更 manifest（test_probe 实为 01fd2f54/11,437 而登记 c11f4284/11,152，29 项中 1 项不符）→ **3d8667a 修正，29/29 全匹配**。3d8667a：manifest 4dfb719b、window b77530bc（未变）、test_window 0c81786a（39）、test_probe 01fd2f54、run_probe d54584b7、binding 101660f8；Mac 我独立复跑 **66/66 OK（skipped=7）**。新增 Linux-only `test_fake_restore_reaches_pid1_after_disable`：真实 enable→begin_restore(disable)→fake wrapper fork setsid daemon 写 manifest 退出→`wait_restored_root` 得 ppid1/pgid-self，且仅一次 start_restore；另有 daemon lifecycle fixture。两 fixture 待 31 Linux 跑。
- **31 旧证据**：a99c93d/52c10dce 的 Linux 36 项 OK（log a3d26fdd）、120d4e7/6f6a3cc9 的 Linux 38 项 OK（log 17bd78d9，fixture 遗留 daemon 1325527/1325530 已按身份 SIGKILL）；均只作历史证据。
- **F5（33 指出，v3 前阻断）**：`set_subreaper` 无 `PR_GET_CHILD_SUBREAPER`（prctl 37）捕获，`disable_subreaper` 固定写 0、`execute` 结束不恢复原值；若前置 flag=1，最终状态被改变。按 20“恢复原 subreaper flag”要求 35：捕获 original 入 result；begin_restore 仍置 0（daemon→PID1）；restore 分支结束（含异常）`prctl(36, original)` 恢复并记录；补 original=1 Linux-only fixture；重冻后 window SHA 变化，31 重跑最终套件。
- **Linux 40 测试循环（fixture 缺陷，产品/F5 不受影响）**：0cea506 实测 39 OK/1 ERROR（log b7ea26ad）：(a) `original` 变量遮蔽（L795 存 start_restore，L804 被 `get_subreaper()` int 覆盖 → `counted()` TypeError）；(b) fake_restore 双 fork → manifest 孙代 pgid≠pid（3d8667a 已见 1325752!=1325753）。F5 代码 PASS。35 的 6dae03f 已改单 fork（manifest 83dd8e85 29/29、window 1f780bf8/48,448、test 22f8297d、Mac 67/67 skip7），但 (a) 仍存在；已要求改名 `real_start`/`original_flag` 后再冻。产品门（pgid==pid、ppid1、单一 restore）不改。
- **最终候选 6d09515（已核）**：manifest f82f44ad **29/29**；window 1f780bf8/48,448 B（未变）；test_window 228150e4/41,364 B（单 fork + `original_start_restore`/`original_subreaper` 分名 + finally 兜底取 pid/只杀该 daemon/按 original 恢复 flag）；run_probe d54584b7；test_probe 01fd2f54；binding 101660f8；Mac 67/67（skip 7）。d5555a1 的提交 blob 是 2cfea8e2（旧），6d09515 为最终版；冻结链自洽。
- **状态**：v2 56441483 未执行、标 superseded；31 将 stage 6d09515 → Linux **先单跑 fake_restore** 再全 40（失败则按 20 用独立全流程 CPU probe 等效；原 40 失败保留、不得宣称全过） → 用 fresh identity 1323589/399707126 出 v3 plan（记录实际 child/runner/supervisor SHA、同 M968/K8704/N5120、KEEP_ALL、fresh out、port 预检）→ 32/33 核 stage 原件 → 一次 A14 large 数值 GO。设备暂停；A15 无设备授权；official/CSV 未动。

## 8.65 A14 large v3 一次 GO（2026-11-19）

- **核验原件**：plan-a14-large-v3 **efb6fb19**（arm A14/KEEP_ALL、M968/K8704/N5120、fresh run/temp/out/restore-a14-large-v3、240/90、master 29531、source `aa3e17…`、plan_not_executed）；root identity **ed2e262b**（1323589/399707126/boot 393ecc15）；stage-identity-v4 47ec314f（commit **6d09515**、manifest **f82f44ad** Linux 29/29、window **1f780bf8**、run_probe **d54584b7**、binding 101660f8、adapter d181b15a、test 228150e4/01fd2f54）；preflight SUMS **90557c94**、fixtures SUMS **90f20867** 均匹配；Linux 单跑 fake_restore **6a369a97 OK**、全 **40/40 OK（53cb7baf）**；33 合同核 PASS、35 复核一致。
- **GO 已发（32）**：31 按 plan 原样执行**唯一一次** window；执行前自检身份/端口/stage SHA；失败不启第二棵 restore、保留现场。回传 actual reduced max_abs/mean_abs/**bit_equal**（不仅 finite）、workspace、sync、trace/plog、descendants/cleanup、subreaper original/cleared/final、restore 五 PID/四检/branch、source 三段 SHA。
- **验收**：bit_equal=true 或精确差值、finite=true、workspace≈115,818,496、sync 与 36.62/9.46 ms 同量级、无 survivors、restore 全检/五 PID/branch aa3e-noop、无第二树。
- 设备唯一格；Grouped large/A13 暂停；A15 无设备；official/CSV 未动。

## 8.66 A14 large v3 执行结果与判读（2026-11-19）

- **tuple/原件**：6d09515 / manifest f82f44ad / run_probe d54584b7 / window 1f780bf8 / binding 101660f8 / adapter d181b15a / plan efb6fb19；commit 691a826、SUMS 18/18（自校验 **774bbdf5**）。两 rank A14/KEEP_ALL、prepare_rc0、executor 非空、**workspace 115,818,496 B（110.5 MiB）**、launch_rc0、sync **9.97 / 36.81 ms**、run_id=1、provider libopapi/libnnopbase、libraries 17/rank；plog 见 aicpu/mc2_server 包路径、device 无错误。
- **数值**：`reduced.max_abs=2.0`（两 rank）、`mean_abs=5.446110662887804e-05 / 4.80431153846439e-05`、**bit_equal=false**；finite/both_finite=true、numeric_pass/pass=true；harness_pass=false 仅 release 空 tensor 的已知 harness 问题。
- **判读（33 纠正后口径）**：result 仅有 `reduced={max_abs,mean_abs,bit_equal}` 与输入 shape/hash，**无 reference 幅值、无逐元素 diff/分位/直方图、无 max_rel/mean_rel/ulp_at_max/counts**。可证：N=484×5120=2,478,080，`total_abs=mean×N=134.959/119.055`，故非零差元素数**至少 68/60 个**（下界，非精确计数）；不可证：2.0=峰值 1 BF16 ulp、输出峰值 420–510、平均相对 7e-7（σ≈93.3 只是解析估计，非本格实测）。**记录为 numeric-complete（finite/workspace/sync 成立）/ not precision pass；差异原因未证**。不满足小形状 0.03125（非尺度不变、非 A14 门），A14 对照臂不新增绝对门；任何采用仍需模型级 2PP。后续必要 probe 可顺带采 ref/output 幅值、diff 分位/非零计数、relative/ULP；不为 A14 报表单独加格。
- **恢复**：branch aa3e-noop、root **1326845**、五 PID [1326845,1326991,1326992,1327006,1327007]、health/info/text/image/env/joint/source/clients 全 true、clients 空；`subreaper_original=0 → cleared=true → final_restored=true → final=0`；无第二树。window `ok=false` 仅 maps_ok 采样（result.libraries 17/rank 为实际映射证据），child rc0、survivors=[]、cleanup_final killed/空。

## 8.67 A15-small-v1 授权与静态合同（2026-11-19）

- **20 授权**：下一次诊断设备格 A15-small-v1，rows127/k256/n512（padded128、RS64→AG128→crop127、非零 pad、flags8、bias null、同 6d09515/d54584b7/1f780bf8/d181b15a）；只检验全行 rank 序/pad 裁回/真实 sync/actual diff；不自动 A15 large/模型接线；不新增/不放宽既定门。
- **静态合同核 PASS**：`arm_plan` CLI rows/k/n 覆盖；A15 flags=KEEP_ALL、bias=nullptr；`padded_rows(127,2)=128`；pad 用 `torch.ones`（非零，行 127，rank1 块尾）；`row_block(0)=(0,64)`、`row_block(1)=(64,128)`；`all_gather_into_tensor` rank 序 → [128,512]；`output=full[:127]`；reference=`matmul(127,256)+all_reduce`；`equivalence` 记录 reference/shape/max_abs/mean_abs/bit_equal；`diagnostic_gate` 0.03125/1e-4 原样 diagnostic-only。设计文档 §12 已记录该精确合同；metadata-only 10398cb 未动 runtime。
- **preflight 已核 + 一次 GO 已发（32）**：plan **99db4369**（A15/rows127/k256/n512/seed/KEEP_ALL/bias null/fresh a15-small-v1/240/90/master 29531/plan_not_executed）；root-identity **c1370f1c**（pid **1326845**、starttime **399871182**、boot 393ecc15、ppid1/pgid self；31 消息中 399707126 为旧 root 笔误，以原件为准）；stage-identity-v5 **d33de953**（6d09515 runtime：window 1f780bf8/run_probe d54584b7/binding 101660f8/adapter d181b15a；metadata-only 70e43be/10398cb 未重 stage）；SUMS **277d82ed**；29531 FREE。GO 要求：只跑一次、失败不启第二棵；回传 pad/row_block/gather/equivalence/diagnostic_gate（原样仅 diagnostic）/workspace/sync/trace、window survivors/cleanup、restore 五 PID/四检/branch/subreaper 三值/source 三段 SHA；不扩 A15 large/不接模型/不新增门。

## 8.68 A15-small-v1 失败：harness pre-launch NameError（2026-11-19）

- **原件**：plan 99db4369、identity 1326845/399871182；两 rank `arm_enter flags=8` 后 `python_exception: NameError: name 'k' is not defined`（run_probe.py:386 `torch.ones((m_pad, k), ...)`，A15 奇数 rows127→pad128 才触发）；api/sync/finite=false，无 pad_ready/gather/equivalence/diagnostic_gate/数值。
- **window/恢复**：child rc1、timed_out=false、survivors=[]、cleanup_final killed 且各集空；restore branch **aa3e-noop**、root **1328734**/start 399924959、五 PID、四检全 true、clients 空；`subreaper 0→cleared→final0`；无第二树；stop verified、source 三段 `aa3e17…` 不变。证据 SUMS 18/18（自校验 **e185265f**），commit 2cb8fdd。
- **判定**：harness pre-launch 产品缺陷，本格 invalid/无数值，**不算算子负结果**；A15 MRS+AG 设备判别尚未发生。
- **审阅教训**：32/33 的 A15 静态合同核只覆盖参数/分支结构，**漏检该未定义名**。20 要求补：`k→a.k`、未定义名静态检查（须对旧 d54584b7 报错）、真实奇数 pad 路径 CPU 替身冒烟（pad→MRS 替身→AG→crop→reference→result/equivalence，不得只验 ARMS/helper）。
- **后续**：window 1f780bf8/adapter d181b15a 字节未变 → 沿用既有安全证据、不重跑 Linux40；35 修 run_probe 重冻 → 31 用 fresh **1328734/399924959** 与新 runner 出 a15-small-v2 plan → 32 核后一次 GO（rows127/k256/n512、flags8、仅诊断；large/生产未授权）。

## 8.69 A15 NameError 修复与上机前检查（22346ec → 0146129）

- **修复**：run_probe **7afd7799**（27,430 B）新增 `a15_flow(rows,k,n,world,rank,x,w,*,ones,empty,run_mrs,sync,all_gather,final_sync,matmul,all_reduce,flags,trace)`：pad（显式 `k`，`ones((m_pad,k))`）→ run_mrs → sync → AG → crop `full[:rows]` → reference `all_reduce(matmul(x_n))`；execute L432 实际调用并传 `a.k/a.n`，meta.pad/row_block/workspace/run_id/trace 事件与 KEEP_ALL 生命周期保持；A14/其它分支不变。
- **上机前检查**：test_probe **4418a835**（22/22）新增 F821 等价检查（symtable，覆盖 run_probe.py 与 supervised_probe_window.py，负例内嵌旧 `torch.ones((m_pad, k))` 并断言 `('execute','k')`）；A15 rows127 CPU 替身真实执行 a15_flow（pad 行 ones、原 127 行不变、run_mrs 128、sync 1 次、AG 128/64、crop 127、reference 127、result/equivalence 占位 max_abs=8.0）。
- **独立验证（32）**：用**真实旧 d54584b7** 跑同一 F821 得 `[('execute','k')]`，新 runner 与 window 均空；Mac 本地 71/71 OK（skip 7）。冻结 0146129 / manifest **47976c76（29/29）** / run_probe 7afd7799 / test 4418a835 / window 1f780bf8、test_window 228150e4、binding 101660f8、adapter d181b15a 均未变；33 复核一致。
- **后续**：window/adapter 字节未变 → 不重跑 Linux40；31 re-stage 0146129 → Linux CPU-only 新 probe **22/22** → fresh **1328734/399924959** 出 a15-small-v2 plan → 32 核 stage/plan/identity/log → 一次 GO（同小格诊断；large/生产未授权）。

## 8.70 A15-small-v2 preflight 与一次 GO（2026-11-19）

- **preflight 核 PASS（32/33）**：commit **0146129**、manifest **47976c76**（Linux stage 逐文件 **29/29**）、run_probe **7afd7799**、window **1f780bf8**（未重跑 40）、binding 101660f8、adapter d181b15a（未重编）、test_probe 4418a835（Linux **22/22 OK，log 3429b5f9**，含 a15_flow 桩与 F821 负例）；root identity **8ecd96b6**（pid **1328734**、starttime **399924959**、boot 393ecc15、ppid1/pgid self）；plan **a9ea5fe2**（A15/rows127/k256/n512/KEEP_ALL/bias null/fresh a15-small-v2/29531/plan_not_executed）；stage-identity-v6 **bf21fabb**；SUMS **9bceab86**（9/9）；31 commit 8cbf840。
- **GO（32，仅一次）**：按 plan a9ea5fe2 与实际 stage SHA 执行；若已启动则继续该一次格，不重复、不启动第二 window/第二棵 restore。范围 rows127/pad128/crop127、k256/n512、flags8、仅诊断（diagnostic_gate 0.03125/1e-4 原样，不宣称精度通过）；large/生产未授权。
- 旧 99db4369（v1）invalid、99abe352 superseded；A15 v1 失败证据保留；A14 v3 口径不变。

## 8.71 A15-small-v2：设备/恢复成立，diff 记录缺失（第二产品缺陷，2026-11-19）

- **tuple**：0146129 / run_probe 7afd7799 / window 1f780bf8 / adapter d181b15a / plan a9ea5fe2 / identity 1328734/399924959；commit 32ef246、SUMS 18/18（自校验 **05b2ccc6**）。
- **成立**：两 rank pad `{rows127,padded_rows128,pad_value1.0}`、row_block `0:(0,64)/1:(64,128)/padded_end128`、gather **1.16421/1.15801 ms**、workspace **17,170,432 B**、sync **7.776 ms**、finite/both_finite/**numeric_pass=true**；child rc0、survivors=[]、cleanup_final killed/空；trace pad_ready/handler rows128/api_return/sync_return/gather_return/result 齐全。
- **缺失**：`reduced/equivalence/diagnostic_gate` **整键不存在**（不是零差异）。根因（产品）：`a15_flow` `reference = all_reduce(matmul(x,w))`，而 execute lambda 的 `dist.all_reduce` 原地操作返回 None → reference=None → 后置 `if reference is not None` 跳过 diff；CPU 桩因 `all_reduce` 返回张量而漏检。
- **恢复**：branch aa3e-noop、root **1330921**、五 PID [1330921,1331056,1331057,1331089,1331090]、四检全 true、clients 空；subreaper 0→cleared→final0；无第二树；source 三段不变。
- **判定**：v2 **numeric-fields-incomplete**，不是精度结论。修复要求（35）：①`reference=matmul(...); all_reduce(reference)` 原地；②CPU 桩 `all_reduce` 返回 None 且原地修改，断言 reference/等价记录（建议抽纯函数测 None fail-closed）；③A15 `numeric_pass` 要求 reference/diff 记录存在（记录完整性 gate，不是误差门）。重冻 run_probe/test；window/adapter 未变 → 不重跑 Linux40。
- **后续**：31 用 fresh **1330921**（starttime 实扫 /proc；restore manifest 无该字段）出 a15-small-v3 plan → 32 核 → 一次 GO；A15 large/生产未授权。

## 8.72 A15 v2 修复 cdc8324 审阅（2026-11-19）

- **冻结**：commit **cdc8324**（9ac47a3 为中间提交，不采用）、manifest **eedd36c9**（29/29）、run_probe **103b9775**（28,844 B）、test_probe **f6fe2c99**（24/24）、window **1f780bf8**/test_window 228150e4/binding 101660f8/adapter d181b15a 未变；Mac 73/73（skip 7）。
- **修复核 PASS**：①`a15_flow` 改为 `reference=matmul(x,w); all_reduce(reference)` 原地，execute lambda 的 `dist.all_reduce` 返回 None 不被依赖；②`a15_record()` 纯函数：reference None→None，正常生成 reduced/equivalence/diagnostic_gate；execute 缺三键写 `numeric_error=missing_equivalence_record` 且 `numeric_pass=false`（记录完整性 fail-closed）；③CPU 桩 all_reduce **返回 None 且原地修改**，断言 reference 非 None/数值正确；A15RecordTest 正常/None 两态。
- **最小 finding（未闭）**：`a15_record` 未校验 `reference.shape == output.shape`，`equivalence.shape` 硬编码 `[rows,n]`；广义不符会在 diff 抛错（fail-closed），但广播兼容错 shape（如 `(1,n)`）会静默产出误导记录。要求 35 加 shape 校验（不符→None + `numeric_error=invalid_equivalence_shape`）与错 shape CPU 用例。分类：不阻塞 v3 实跑（真实路径两侧固定 `(rows,n)`），但**记录完整性闭合前必须补**。
- **后续**：31 stage cdc8324（Linux probe **24/24**）→ fresh **1330921/400000296** 出 a15-small-v3 plan → 32 核 stage/plan/identity/log → 一次 GO；范围 127/pad128/crop127、k256/n512、flags8、仅诊断；large/生产未授权。

## 8.73 A15-small-v3 preflight 与一次 GO（2026-11-19）

- **preflight 核 PASS**：plan **2fcd101b**（A15、rows127/k256/n512、seed 20261119、KEEP_ALL、bias null、runner grouped-deepdive-35-cdc8324 run_probe **103b9775**、adapter d181b15a、fresh run/temp/out/restore a15-small-v3、master 29531、240/90、source `aa3e17…`、plan_not_executed）；root identity **78b95dc0**（pid **1330921**、starttime **400000296**、boot 393ecc15、ppid1/pgid self）；stage-identity-v7 **0e65c9e4**（commit cdc8324、manifest eedd36c9 Linux **29/29**、window 1f780bf8、binding 101660f8、test f6fe2c99）；Linux probe **24/24 OK（log 0b71b25a）**；SUMS **68c1660f** 全匹配。
- **GO（32，唯一一次）**：按 plan 2fcd101b 执行；tuple 锁定 cdc8324/run_probe 103b9775/test f6fe2c99，shape guard 候选 8807597 **不用于本格**、不重 stage/不重 plan；失败不启第二棵 restore。验收 actual `max_abs/mean_abs/bit_equal`（shape[127,512]）、diagnostic_gate 原样（仅 diagnostic）、pad/row_block/gather/workspace/sync/trace、window survivors/cleanup、restore 五 PID/四检/branch/subreaper/source 三段 SHA；不宣称精度通过。
- **停止说明**：期间一次只读 `ls`+`shasum` 组合命令被平台拒绝（提示 user does not want / STOP）；经澄清无人类总体任务停止，按 20 指示恢复并完成只读核验；未发生设备动作或 tuple 变更。

## 8.74 A15-small-v3 结果：本格 zero diff / bit_equal（2026-11-19）

- **tuple/原件**：cdc8324 / run_probe 103b9775 / test f6fe2c99 / window 1f780bf8 / binding 101660f8 / adapter d181b15a / plan 2fcd101b / identity 1330921→restore root **1334752**；commit 1d1daef、SUMS **17/17（自校验 d7e8b90f）**。
- **数值（两 rank）**：pad `{rows127,padded_rows128,pad_value1.0}`、row_block `0:(0,64)/1:(64,128)/padded_end128`、gather **1.22224/1.21543 ms**、workspace **17,170,432 B**、**reduced max_abs=0.0 / mean_abs=0.0 / bit_equal=true**、equivalence shape **[127,512]** 零差、diagnostic_gate `within_bounds=true`（0.03125/1e-4，`diagnostic_only_not_an_official_gate`）、finite/both/numeric_pass/pass=true；trace arm_enter flags8/pad_ready/handler rows128/api_return/sync 13.17 ms/gather/result 齐全；plog/device 零 `[ERROR]`。
- **window/恢复**：child rc0、timed_out=false、survivors=[]、cleanup_final killed/空；`ok=false` 仅 maps 采样伪影。restore：branch **aa3e-noop**、root **1334752**、五 PID [1334752,1334888,1334889,1334910,1334911]、四检全 true、clients 空；subreaper 0→cleared→final0；无第二树；source 三段不变；stop verified。
- **结论（range-limited）**：rows127/pad128/crop127、k256/n512 下，全行 rank 序、pad 裁回、真实 sync 与 actual diff 均成立（本格 zero diff）。**仅本格诊断事实，不宣称精度通过、不外推大形状/生产**；shape guard 8807597 仍为后续候选（记录完整性），不替换本格 tuple。A15 large/生产未授权；设备暂停待新授权。

## 8.75 A15-large-v1 有限授权与前置（2026-11-19）

- **20 授权一次 A15-large-v1**：M967/K8704/N5120（内部 pad968、RS484/AG968/crop967）、flags8（KEEP_ALL）、bias null、同 seed/group/provider。目标仅原 967 形状的功能/全行数学参考差异与尺度实测；**不接模型、不做性能/生产结论**。
- **前置**：①保留 **8807597** shape guard（a15_record 返回 (record,error)，None→missing_equivalence_record、错 shape→invalid_equivalence_shape，numeric fail-closed）；②a15_record 增补**真实误差记录**：`ref_abs_max`/`out_abs_max`、`diff_nonzero_count`/`diff_total_count`、`relative_l2`（可选 `max_rel_abs`），均从真实 output/reference 计算、逐 rank、不做 pass/fail 推断；CPU 用例断言精确 count/l2，None/错 shape 三态保持；重冻 run_probe/test（window 1f780bf8/adapter d181b15a 未变）。既有 diagnostic **0.03125/1e-4 不变、仅诊断，不得为既有差值放宽**；large 无生产精度结论。
- **另**：已请 35 提交 **CPU benchmark 方案**（ND/NZ 基线、公平同步、内存/生命周期、ABBA）供 32 审；设备 benchmark 在具体计划获批前不跑。
- **流程**：35 新冻 → 31 stage + Linux probe 套件 → fresh **1334752/400276164** 出 a15-large-v1 plan（967/8704/5120、fresh dirs、29531 预检、plan_not_executed）→ 32 核 stage/plan/identity/log → 一次 GO。设备保持暂停。

## 8.76 误差统计候选 c482f3f/ed12a12 与 benchmark §15 审阅（2026-11-19）

- **ed12a12（最终字段名版）**：manifest **8e5846de** 29/29、run_probe **8e308804**、test **07e5a5b4**（27/27）、window 1f780bf8/test_window 228150e4 未变；Mac 76/76（skip 7）。`tensor_error_stats` 字段：max_abs/mean_abs/bit_equal、ref_abs_max/out_abs_max、diff_nonzero_count/diff_total_count、max_error_index/position/ref/output、max_rel_abs、relative_l2、relative_l2_denominator、relative_l2_zero_reference；规则 `diff/max(ref_norm,1e-12)`（非零/0→有限大值 + zero_reference=true，**不是 null**）。diagnostic 0.03125/1e-4 与 numeric_pass 未变。
- **Finding（重冻前置）**：CPU 测试仍用 test-local `list_error_stats`，**从未执行生产 `tensor_error_stats`/`a15_record` 真实张量路径**；与此前两次设备漏检同类。要求 35 加 torch-available 直测（`find_spec("torch")`+skipUnless；macOS 无 torch 跳过，31 Linux venv 会执行），覆盖正常/零 reference/None/错 shape，并与 list helper 交叉一致。
- **Benchmark §15 findings（设计文档 cccaccc8）**：①NZ 基线未落到可执行 arm/算子/权重布局与哈希/flags，须与 A15 同 KEEP_ALL/stream/sync，并说明 ND↔NZ 转换与等价验证；②M pad 不对称（A15 M968 vs baseline M967），需统一或量化 1 行 pad；③≤1 GiB/rank 预算须计 **W warmup+所有 A15/基线调用+重试/辅助调用**，按进程模型（同进程交替或每 arm 独立）共享，重算“32 小/8 大”；④明确 W/N、计时区间含 launch+collective+sync、warmup 不计统计、ABBA/median/p95/配对差；⑤生产 executor/workspace 生命周期 pending、flags=0 早释放非解法（PASS）。
- **31 的 c482f3f preflight（plan d1b02bcd、stage afef4c25、run_probe 21d4ebc7）已过期**：ed12a12 的 run_probe 8e308804 不同；已通知不执行、等最终冻重做 stage/plan；identity 保持 **1334752/400276164**。设备暂停。

## 8.77 A15-large-v1 ed12a12 preflight 结构核验（hold，2026-11-19）

- **plan 34bfb3aa**：identity **1334752/400276164**（ppid1/pgid self、boot 393ecc15）、run_dir `run-a15-large-v1`、child argv A15/rows967/k8704/n5120/seed 20261119、runner `grouped-deepdive-35-ed12a12` run_probe **8e308804**、adapter d181b15a、fresh temp/out/restore a15-large-v1、240/90、master 29531、source `aa3e17…`、plan_not_executed。
- **stage 39ffb333**：commit ed12a12、manifest **8e5846de Linux 29/29**、run_probe 8e308804、window 1f780bf8、binding 101660f8、adapter d181b15a、test 07e5a5b4；Linux probe **27/27 OK（log d21e8520）**；SUMS 匹配。
- **hold 原因**：35 工作树未提交的 `test_probe_cpu.py` 已加入 `skipUnless(find_spec("torch"))` 并直调生产 `run_probe.tensor_error_stats`（32 要求的直测），design doc §15 也在按 findings 修改；当前 27 项 Linux log 未包含直测。待 35 提交最终冻后，31 重跑 Linux probe（新计数）并按新 test/manifest 出最终 stage/plan（run_probe 预计仍 8e308804；若变则重做），32 核后一次 GO。
- 当前 plan/stage **未执行**；设备暂停；A15 large 仅功能/误差诊断，不接模型、不做性能/生产结论。

## 8.78 e586d49 最终冻审（2026-11-19）

- **冻结**：commit **e586d49**、manifest **87ff44a9**（29/29）、run_probe **8e308804**（未变）、test_probe **9abfbcab**（30/30）、window 1f780bf8/test_window 228150e4/binding 101660f8/adapter d181b15a 未变；design doc **f6ad74fb**；Mac 79/79（skip 10：7 Linux-only + 3 torch-direct）。
- **torch-direct**：`TensorErrorStatsTest` 用 `find_spec('torch')`，小 CPU 张量直调生产 `tensor_error_stats` 与 `a15_record(stats=..., equal=torch.equal)`，覆盖 normal/零 reference/None/错 shape，并与 list helper 交叉；补上了此前“只测 helper、未执行生产统计函数”的缺口。
- **Benchmark §15 五项 PASS**：①A16 明确 design-not-implemented（FRACTAL_NZ29、记录 ND 源/NZ cast 哈希、先验 ND/NZ 等价、同 stream、保留到 sync、不用 flags）；②主比较统一 M968，另测 M967 量化 1 行 pad；③预算计 W warmup+全部 A15 调用+重试/辅助，≤1 GiB/rank ⇒ ≈62 小/9 大；同进程共享或每臂独立；④W/N、ABBA、计时含 launch+collective+sync、warmup 不计统计；⑤生产 executor/workspace 生命周期 pending、flags=0 非解法。
- **Clarity note**：large W2/N6=8 calls/arm 与 9-call 上限意味着**大形状每 arm 独立进程**（或共享进程 W1/N3）；A16 尚未实现是设备 benchmark 前置；后续具体计划中写明。
- **后续**：31 stage e586d49 → Linux probe **30/30**（含 3 torch-direct）→ 最终 plan（identity **1334752/400276164**、M967/K8704/N5120、fresh dirs）→ 32 核 → 一次 GO；设备暂停。

## 8.79 e586d49 Linux torch-direct：test-only 容差失败（2026-11-19）

- **31 原件**（fixtures-e586d49，log bd4652b5）：有 CANN env 时 `Ran 30 / 29 OK / 1 FAIL`；失败 `test_tensor_stats_match_the_list_helper` 第 317 行精确相等：`relative_l2` **1.8708287477493286（torch float32）vs 1.8708286933869707（double list helper）**，差 ~5.4e-8。整数/位置字段全部通过；生产 `tensor_error_stats` 口径不变（torch 实测值）。
- **无 env**：venv torch 自动加载 torch_npu backend entry point，无 CANN env 下 triton `TORCH_LIBRARY` 重复注册 → 3 error/segfault；属**环境前提**，Linux CPU 套件须先 `source activate_env.sh`（仅 CPU 张量，不调 NPU）。
- **判定/修复**：test-only；要求 35 将整数/位置字段保持严格相等，浮点字段（relative_l2、mean_abs、max_rel_abs、abs_max 等）用 `assertAlmostEqual(delta=1e-6)` 或 float32 期望；README 注明 activate_env。test/manifest 重冻，run_probe 8e308804/window/adapter 不变。
- **后续**：31 用 activate_env 重跑 Linux **30/30** → 最终 stage/plan（identity 1334752/400276164、M967/K8704/N5120）→ 32 核 → 一次 GO；设备暂停。

## 8.80 ba81adb 最终 test 冻（浮点容差，2026-11-19）

- **冻结**：commit **ba81adb**、manifest **c727b10f**（29/29）、run_probe **8e308804**（未变）、test_probe **23409fcb**（30/30；3 torch-direct）、window 1f780bf8/test_window 228150e4/binding 101660f8/adapter d181b15a 未变；README **483c871c** 注明 Linux 需 `source activate_env.sh`（否则 torch_npu entry point 在无 CANN env 下 segfault，属环境前提）；Mac 79/79（skip 10）。
- **修复**：`TensorErrorStatsTest` 浮点字段（max_abs/mean_abs/ref_abs_max/out_abs_max/max_error_reference/output/max_rel_abs/relative_l2/relative_l2_denominator）改 `assertAlmostEqual(delta=1e-5)`；counts/indices（diff_nonzero_count/diff_total_count/max_error_index）与 max_error_position 保持严格相等；生产 `tensor_error_stats` 字段/阈值未改。
- **后续**：31 stage ba81adb → `source activate_env.sh` 跑 Linux 30/30（含 3 torch-direct）→ 最终 plan（identity 1334752/400276164、M967/K8704/N5120）→ 32 核 Linux log + stage/plan/identity → 一次 GO；设备暂停。

## 8.81 e38b541 最终 test 冻（delta=1e-6，2026-11-19）

- **冻结**：commit **e38b541**、manifest **8e324bea**（29/29）、run_probe **8e308804**（未变）、test_probe **ad257b5f**（30/30；3 torch-direct）、window 1f780bf8/test_window 228150e4/binding 101660f8/adapter d181b15a 未变；Mac 79/79（skip 10）。相对 ba81adb 仅浮点容差 1e-5→1e-6（观测差 ~5.4e-8），counts/indices/position 仍严格；生产统计与阈值未改。
- **后续**：31 stage e38b541 → `source activate_env.sh` 跑 Linux **30/30**（含 3 torch-direct）→ 最终 plan（identity 1334752/400276164、M967/K8704/N5120、pad968/crop967、flags8、bias null、fresh dirs、29531 预检、plan_not_executed）→ 32 核 Linux log + stage/plan/identity → 一次 GO；设备暂停。

## 8.82 A15-large-v1 preflight 与一次 GO（2026-11-19）

- **preflight 核 PASS**：plan **8befce63**（A15、M967/K8704/N5120、pad968/RS484/AG968/crop967、flags8、bias null、runner `grouped-deepdive-35-e38b541` run_probe **8e308804**、adapter d181b15a、fresh run/temp/out/restore a15-large-v1、240/90、master 29531、source `aa3e17…`、plan_not_executed）；root identity **9bac5010**（pid **1334752**、starttime **400276164**、boot 393ecc15、ppid1/pgid self）；stage **4583cc60**（commit e38b541、manifest 8e324bea Linux **29/29**、window 1f780bf8、binding 101660f8、adapter d181b15a、test ad257b5f）；Linux probe **30/30 OK（log 48efc257，含 3 torch-direct）**；SUMS **1a1f30c8**。
- **GO（32，唯一一次）**：按 plan 8befce63 执行；失败不启第二棵 restore。回传逐 rank 实测：max_abs/mean_abs/bit_equal、ref_abs_max/out_abs_max、diff_nonzero_count/diff_total_count、max_error_index/position + ref/output、max_rel_abs、relative_l2/denominator/zero_reference；pad/row_block/equivalence shape [967,5120]、diagnostic_gate 原样（0.03125/1e-4、diagnostic_only）、workspace/sync/trace；window child rc/timed_out/survivors/cleanup_final；restore 五 PID/四检/branch aa3e-noop/subreaper original/cleared/final/source 三段 SHA。
- **口径**：仅功能/全行参考差异与尺度实测；不宣称精度通过、不做性能/生产结论；benchmark/模型未授权；不新增/放宽门。设备唯一格；服务 aa3e/s1 root 1334752 健康。

## 8.83 Benchmark 完整保留内存预算审阅（CPU-only，2026-11-19）

- **现状（§15 f6ad74fb）**：1 GiB/rank 仅按 per-call workspace 计——large 115,818,496 B → **9 calls**；small 17,170,432 B → **62 calls**；W warmup+全部 A15 调用+重试计入；large 建议 W2/N6=8/arm（须每 arm 独立进程或共享降至 W1/N3）。
- **Finding（完整保留口径）**：预算应含三类：①进程常驻：x（967×8704 bf16 ≈16.85 MB）、w（8704×5120 bf16 ≈89.13 MB）、adapter/其它，是否只建一次；②每次调用被 KEEP_ALL 保留：workspace + **descriptors/group metadata/handles（当前未计量，需实测或上界）**；③每迭代临时量（full/output/reference/y）：确认迭代结束释放、未留列表，否则计入。公式 `max_calls = floor((1 GiB − 常驻) / 每次保留)`；两臂共享进程求和；当前“9 大/62 小”仅是 workspace-only，若 descriptors 有 X MB 上限会下降。35 需更新 §15 并标注实测/上界。
- **max_rel_abs 语义**：分母 `clamp_min(|ref|, TINY)`，近零参考会产生大值；报告须同时给 `max_error_reference` 以便解释，且只解释该位置，不当作相对误差/门。
- **范围**：设备 benchmark 未授权；不影响当前 A15-large-v1 单格。

## 8.84 A15-large-v1 原件联审（2026-11-19）

- **tuple/原件**：e38b541 / run_probe 8e308804 / test ad257b5f / window 1f780bf8 / binding 101660f8 / adapter d181b15a / plan 8befce63；cell `a15-large-v1-cell`（window d75f4745、rank0 result f50dba4f、rank1 995a4443 等），SUMS **17/17（自校验 2adc7ef6）**，commit 0d8cc1a。
- **diagnostic_gate 更正**：原件 rank0/1 均为 **`within_bounds=false`**；run_probe L379-380 表达式是 **AND**（`max_abs <= 0.03125 and mean_abs <= 1e-4`）。本格 max 条件 false（2.0 > 0.03125）、mean 条件 true（5.1305e-5 ≤ 1e-4）→ false。**不是 OR 约定、不是语义/实现错误**；31 报文与转述的 true 是**报告笔误**，以原件为准；门不变、不重跑。
- **可证数值**：两 rank 完全一致——pad `{967,968,1.0}`、row_block `0:(0,484)/1:(484,968)`、equivalence **[967,5120]**；`max_abs=2.0`、`mean_abs=5.1305116357980296e-05`、`ref_abs_max=out_abs_max=744.0`、`diff_nonzero=510/4,951,040`（0.0103%）、`max_error_index=1,040,755`（=203×5120+1395）、position **[203,1395]**、ref **400.0**/out **402.0**、`max_rel_abs=1.0`、`relative_l2=4.9198e-05`（denom 293,496.6875、zero_reference=false）；gather 1.139/1.210 ms、workspace **115,818,496 B**、sync **7.846 ms**、numeric_pass/pass=true。
- **派生**：total_abs≈**254.014**；非零均值≈**0.498**；非零数下界 ceil(254.014/2.0)=**128**（实际 510）；||diff||2≈**14.44**；非零 RMS≈**0.639**。BF16：ref=400 在 [256,512) → ulp=2，该 max 样本差 2.0 = **1 ulp**；ref_abs_max=744 在 [512,1024) → ulp=4（2.0=0.5 ulp）。**仅解释该样本**，不能归因累加或代表其余 509 差异。
- **max_rel_abs=1.0**：逐元素 |diff|/max(|ref|,1e-12)；其最大位置/参考值**未落盘**，不可与 max_error 位置 [203,1395] 混用，不能据此推断尺度级相对误差。
- **恢复/清理**：child rc0、timed_out=false、survivors=[]、cleanup_final killed 且 members/groups/singles/unknown/drift 全空；restore branch **aa3e-noop**、manifest pid **1337814**、五 PID [1337814,1337961,1337962,1337979,1337980]、四检全 true、clients 空；subreaper 0→cleared→final0、无第二树；source 三段 `aa3e17…` 不变；stop verified、unavailable 0。start **400405818** 为 31 的 /proc 实扫，归档 window/manifest 无该字段，属 31 报告、不可独立复核。
- **libraries/错误**：per-rank 17 项（adapter d181b15a、libhccl da6dd17f、libascendcl 1c06b9c4、libopapi eb1c788e、libnnopbase 72cd43b4、libtorch_npu 65e6be19）；provider 一致；plog×2/device×2 零 `[ERROR]`。
- **未证/口径**：510 差异的逐元素分布/ULP 归因、max_rel 位置/参考、其他形状/生产/性能均未证；within_bounds=false 表示该小形状绝对诊断不通过（仅诊断，不是生产门）。**不称精度通过、不改门、不重跑**；设备暂停。

## 8.85 §15.1 完整保留预算 CPU 复核（条件式 PASS，2026-11-19）

- **台账（large，per rank）**：per-call retained = workspace 115,818,496 + pad 输入 16,850,944 + RS 输出 4,956,160 = **137,625,600 B（131.255 MiB）**；resident = x 16,833,536 + w 89,128,960 = **105,962,496 B（101.05 MiB）**；**瞬态代码下限 = bf16 full/ref 18.9 MiB + 4 个 [967,5120] float32 统计张量（out.float/ref.float/diff/pow-sum）75.55 MiB = 94.44 MiB**，不是文档 ~50 MiB（实际 pow/div/clamp 可能更多）。
- **重算**：W1/N4=5 + X=1 MiB/call 峰值 = **856.75 MiB（83.7%）**，非 807；6 calls 峰值 = **989.0 MiB（96.6%）**；5 calls 要 ≤1 GiB 需 X≤**34.45 MiB**，6 calls 需 X≤**6.83 MiB**。**X=1 MiB 是规划预留，未实测/未由代码证明 executor/descriptor 上界**；W1/N4=5 对 X 较稳健，6-call 上限对 X 敏感。
- **1 GiB 口径**：现公式减 resident(x+w) 未含 adapter/runtime 常驻，属**实验新增分配预算**，不是进程总峰值；若指总峰值须加 adapter/runtime RSS 与 allocator 高水位，文档须二选一。
- **allocator/两臂**：释放的临时量可能留在 NPU caching allocator，峰值 RSS 可高于 live；需 `memory_allocated/reserved` 实测或非计时区间 empty_cache。共享进程最坏点 = A15 累积 + A16 高水位 + 两者瞬态；A16 sync 后释放、A15 进程退出释放；per-arm 进程隔离。
- **ABBA 可执行**：A1(A15,W1+N4)→B1(A16,W1+N4)→B2(A16,W1+N4)→A2(A15,W1+N4)；配对 A1-B1/A2-B2，B1/B2 漂移；每进程 5 calls≤6（X 前提下）；W1/launch 不计统计；同 seed/x/w hash/TP group/stream/逐迭代 sync；A16 未实现前不可启动。
- **max_rel 语义**：`max_error_reference/output` 只标识**最大绝对误差**样本，不能解释 `max_rel_abs=1.0` 的成因；max_rel 位置/参考未落盘，应标未证或新增字段。
- **小格**：23 calls×17,301,504=379.5 MiB（+X=402.5），文档“398 MiB”需给准确组成。
- **结论**：方向与 W1/N4=5/W3/N20=23 可**条件式**收口；上述台账/口径/max_rel 文本需修正；A16 design-not-implemented、设备 benchmark 未授权、无设备动作/代码冻结。31 的 within_bounds=false 更正文件已核，原 SUMS 2adc7ef6 未变。

## 8.86 §15.1 最终收口（bd7685a / e91d4df8 / 069c701a，2026-11-19）

- **已闭（CPU 文档）**：1 GiB=2^30 B/rank 明确定义为**实验新增分配预算**（保留张量+瞬态峰值+预留，不含 adapter/runtime RSS/allocator 高水位，非进程总峰值）；字节台账 per-call 137,625,600 B、+X 138,674,176 B、resident 105,962,496 B；瞬态**下界 94.44 MiB**；W1/N4=5 **856.75 MiB 条件台账**；6 calls **989.0 MiB（96.6%，X≤6.83 MiB）**；X 上限 34.45/6.83 MiB；y 已在 per-call 计一次，94.44 瞬态不含 y（“含 y 约104 MiB”为替代口径，不可再加）；ABBA A1(A15,W1+N4)→B1(A16,W1+N4)→B2(A16,W1+N4)→A2(A15,W1+N4)，配对 A1-B1/A2-B2，W1/launch 不计；max_rel 位置/参考未落盘、max_error_* 只标绝对最大样本；A16 design-not-implemented、设备 benchmark 未授权。
- **待具体 benchmark 计划验证（前置）**：X=1 MiB 与 descriptor/executor 尺寸未测；adapter/runtime RSS 与 allocator `memory_allocated/reserved` 高水位需实测或非计时区 empty_cache；max_rel 的 div/clamp 临时量未单列，会使瞬态下界上升；若 1 GiB 改指进程总峰值需重算。
- **最终文档（12da7e3 / e1059ff8 / manifest 911cc150 / README 9b2ac523）**：y 明确属于 per-call retained、不属瞬态；替代“同时存活张量总量”103,987,200 B=**99.17 MiB** 且明确 **not additive/不得与 per-call 重复相加**；旧 50/807 残留已清除；X 未验证预留、max_rel 位置未落盘、A16 design-not-implemented 均保持。
- **最终数值文档（c050655 / a1df799c / manifest 5ebe652a / README fbb71fa0）**：103,987,200 B 明确为 **99.17 MiB（约104 MB）**、含 y 一次且 **not additive**；94.44 MiB 瞬态下界**不含 y**；两峰值读数互斥、绝不相加；W1/N4=**856.75**（94.44）为条件台账、非已证峰值，99.17 替代下 **861.5**；6 calls=**989.0**（X≤6.83）仅限 94.44 口径（若用 99.17 代入则为 993.7、X≤5.05）。旧 “about 104 MiB” 已清除。
- **结论**：预算口径/max_rel/ABBA 一致，**条件式 CPU PASS 收口**；待具体 benchmark 计划实测 descriptor/executor、adapter/runtime RSS、allocator high-water、max_rel div/clamp 临时量。不新增设备动作、不改运行代码、不重 stage；设备暂停。31 的 abi-preflight4 属 33 的 Grouped 研究，不在本 A15 审核内。

## 8.87 H11 观测草案 CPU/只读预审（2026-11-19，无 GO）

- **H11 身份/契约（可闭）**：`ARMS['H11']` 为真实 CLI arm——flags `KEEP_ALL(8)`、native=true、rows=128；`arm_plan` 对 H11 用 **FP32 zero bias**；execute 走 native grouped `run`（带 bias），reference=local matmul+all_reduce。与 `h11_matched_control.md` 一致（M128/K256/N512、BF16 x/w、FP32 zero bias、单 group、同 adapter/comm、240/90）。本机 plan-only 实测：H11、flags8、native true、rows128/k256/n512、bias fp32_zero、timeout 240/90。
- **精确 argv（当前冻 e38b541/run_probe 8e308804/window 1f780bf8/adapter d181b15a）**：
  `torchrun --nproc_per_node=2 --master_port 29531 <stage>/run_probe.py --arm H11 --rows 128 --k 256 --n 512 --seed 20261119 --temp-root <fresh> --out <fresh> --adapter <stage>/probe_adapter.so --arm-timeout 240 --comm-timeout 90 --execute`；plan-only 同参去掉 `--execute`。
- **env/restore（可闭）**：window 进程环境设 `ASCEND_GLOBAL_LOG_LEVEL=0`、`ASCEND_PROCESS_LOG_PATH=<fresh abs dir>`；`child_env={**os.environ, CLEF_WINDOW_OWNER}` → torchrun→ranks 继承；执行中只读 `/proc/<rank>/environ` 验证；fresh dir 预创建。restore argv 为 `env -i PATH=... CLEF_LAUNCH_CONFIG=s1 setsid ... start.sh`，观察 env **不进入**恢复服务；source 三段 SHA 检查；`env_evidence` 仅比 ASCEND_CUSTOM_OPP_PATH/LD_LIBRARY_PATH vs clean_expect，隔离靠 env -i 结构。
- **window 兼容（可闭）**：1f780bf8 choices 含 H11；plan-only/有限窗口/240s/descendants cleanup/唯一 restore 相容；须用 e38b541 runtime；fresh identity/port/dirs 由 31 后续预检（当前服务 root 1337814/400405818）。
- **采集（可闭）**：完整保留 fresh PROCESS_LOG_PATH + 默认 `/root/ascend/log` 时间窗 + device 日志；**不得按 probe PID 严格过滤**（device/AICPU daemon PID 不同）；按 time/device/file/host 派生视图，SUMS；正向标记（start to read/reset valid/Check msg equal/ReadValidMsg timeout/Init addr）出现才算采到，否则记缺测。级别证据 cf9f1d6/rodata 33a2226；guard 映射/新 xref 未闭，不猜 MODULE/HCCL_DEBUG。
- **阻断/前置**：①无设备 GO、31 不得运行；②PROCESS_LOG_PATH/GLOBAL_LOG_LEVEL 对 device/AICPU daemon 传播未证，须正标记或记缺测；③guard 级别映射未证；④fresh log dir 预创建/空间；⑤采集不得 PID 过滤；⑥H11 hang 未解，观测仅诊断；⑦H11 文档仍引旧 run_probe/window，计划文档需更新到 e38b541（doc-only，不改代码/不重冻）。
- **33 §79 argv 复核（2026-11-19）**：**阻断1**——probe child 写成 `python run_probe.py --execute`，缺 torchrun 注入的 RANK/WORLD_SIZE/LOCAL_RANK，execute 会 ValueError；实际 child 必须由 window `build_child_argv` 生成 torchrun（独立文档用同 argv 的 torchrun）。**阻断2**——window argv 缺必填 `--text-body`/`--image-body`（argparse required + build_plan 检查文件存在），须补现有 cell body。可闭：H11 真实 arm/flags8/native/fp32_zero bias；window child 传 `arm-timeout=child_timeout(240)`/`comm-timeout=90`/adapter；env 须在启动 window 前导出 `GLOBAL_LOG_LEVEL=0`+`PROCESS_LOG_PATH` fresh（不在 argv），child_env 继承，restore 用 env -i 不进入服务；aa3e-noop/五 PID/source 三段检查保留；采集 §79.3 与 32 口径一致。前置：fresh identity/port/dirs、stop-script SHA、text/image body、log dir、31 preflight。
- **§81 并入复核（2026-11-19）**：33 已记录大部分预审，但 §79.2 命令块仍差两条：①probe child 仍 `python run_probe.py --execute`（缺 torchrun 注入的 RANK/WORLD_SIZE/LOCAL_RANK，会 ValueError），须改 torchrun 或明确标注为不可执行形状（实际 child 由 window `build_child_argv` 生成）；②window argv 仍缺必填 `--text-body`/`--image-body`（argparse required + build_plan `is_file` 检查），须补现有 cell body。其余 §81 记录正确（H11 真实 arm/flags8/native/fp32_zero、冻结 tuple、env 继承与 restore env -i、采集 §79.3、guard/daemon 阻断、缺测判据）。
- **§82 修正核验与闭环（2026-11-19）**：probe child 已改 torchrun（`--master_port` 下划线，与 window 生成一致）；window 已补 required `--text-body/--image-body` 占位。回答两点：① `--master-port` 不重复/不冲突——window 的 `--master-port` 是 window 层，`build_child_argv` 生成 child 的 `torchrun --master_port 29531 run_probe.py ...`；run_probe 无 master-port 参数，独立 probe 命令须把 `--master_port` 放在 torchrun 后/脚本前，放脚本后会被拒；两层同值、只有一个 torchrun，31 预检 29531 空闲。② body 用已验证 A15-large plan 的 `/workspace/yangchen/work/grouped-deepdive-cell-a15-large/text-body.json` 与 `image-body.json`（A14-large 同名亦可），31 preflight `test -f`；window build_plan 也检查 `is_file`。
- **真实 window plan-only（32 本机，/tmp 夹具，2026-11-19）**：用完整 parser 跑 `supervised_probe_window.py --arm H11 ...`（临时 stop/source/identity/body，不执行/NPU），rc=0；生成的 `child_argv` 为 `torchrun --nproc_per_node=2 --master_port 29531 run_probe.py --arm H11 --temp-root/out <fresh> --k 256 --n 512 --rows 128 --seed 20261119 --arm-timeout 240.0 --comm-timeout 90.0 --adapter <frozen> --execute`；`child_timeout=240.0`；`restore_argv=env -i PATH=... CLEF_LAUNCH_CONFIG=s1 setsid /bin/bash -c ...`；text/image probes 均被接受。**master-port 不重复**：window 层消费，child torchrun 生成 `--master_port`，run_probe 无该参数，只有一个 torchrun；独立命令须放 torchrun 后/脚本前。body 用 A15-large cell 的 text/image（A14 替代亦可），31 `test -f`；本机仅验证 parser，不代表远端存在。
- **e7d1782/eb50f10 修正版复验（2026-11-19）**：字面远端 body 路径在本机 `build_plan` 按预期报 `text/image probe bodies must exist`（macOS 无 `/workspace`，非命令缺陷；`| tail` 会把管道 rc 显示为 0，实际 Python 失败）；用 /tmp 同名 body 副本复跑 **rc=0**，child_argv 为 `torchrun --nproc_per_node=2 --master_port 29531 run_probe.py --arm H11 --temp-root/out <fresh> --k 256 --n 512 --rows 128 --seed 20261119 --arm-timeout 240.0 --comm-timeout 90.0 --adapter <frozen> --execute`，probes 指向 body。master-port 无重复/冲突（window 层消费，child torchrun 仅一个 `--master_port`，run_probe 无该参数）；独立命令须放 torchrun 后/脚本前。env 白名单（GLOBAL_LOG_LEVEL/PROCESS_LOG_PATH/CLEF_WINDOW_OWNER）与 restore env_evidence（ASCEND_CUSTOM_OPP_PATH/LD_LIBRARY_PATH + env -i）不冲突。
- **结论**：命令/计划兼容性**可闭**；剩余为 31 只读 preflight（真实 body 存在、stop-script SHA、source AA3E、fresh identity 1337814/400405818、29531、fresh dirs、adapter、log dir）+ 3 xref/条件项（AICPU env 传播/guard 映射/正标记）。当前**不发执行许可/GO**、无设备动作。

## 8.90 H11-observe-v1 只读 preflight 核验（2026-11-19，无 GO）

- **原件**：commit 01e8631、SUMS **6/6（cf59ed66）**；plan c95ae5c3、stage-identity 76332ae6、root-identity de780872、env-sidecar a6a3c4a4、live-checks 466702e9。plan **status=plan_not_executed**；无服务停止、无 torchrun、无设备动作。
- **plan**：arm H11、rows128/k256/n512、world2、KEEP_ALL/native/FP32 zero bias、seed 20261119、arm-timeout 240.0/comm-timeout 90.0、master_port 29531；child=`torchrun --nproc_per_node=2 --master_port 29531` + e38b541 run_probe **8e308804** + adapter **d181b15a** + `--execute`（未运行）；restore=`env -i PATH=... CLEF_LAUNCH_CONFIG=s1 setsid /bin/bash -c ... start.sh`，输出 fresh。
- **identity/服务**：pid **1337814**、starttime **400405818**、boot 393ecc15、ppid1、pgid/sid self、state S；五 PID 1337814/1337961/1337962/1337979/1337980 全同 pgid/sid；/health ready（local-65e5a3c5ffb6）、/v1/decide/info s1/TP2/4096/cache off。
- **端口/目录**：29531 TCP/UDP × 127.0.0.1/0.0.0.0 均 FREE（python socket bind；host 无 ss）；cell 与 logs-h11-observe-v1 已建；run/probe-temp/probe-out/restore 四路径 ABSENT（plan 前后复查）。
- **冻结/输入**：window 1f780bf8、run_probe 8e308804、binding 101660f8、adapter d181b15a、test ad257b5f、manifest 8e324bea、stop d6d49d75、source aa3e1758；body text a7dd3f64/image f3229f5c（A15-large cell，test -f+SHA）。env sidecar：GLOBAL=0+PROCESS_LOG_PATH<cell>/logs，**applied=false**；restore 不导出。
- **残余（非阻断）**：端口用 python bind（无 ss，README 称 plain bind）；stage identity 依赖 A15-large 既有 29/29 核验 + 本次六键复核（未全量重核/未重 stage）；body SHA 为 31 现场实测（body 未归档）；按指示未重跑 Linux40。
- **执行准备度（2026-11-19）**：plan/identity/冻结/body/env sidecar 齐备（plan 完整 SHA **c95ae5c366a370423fdc42c99d5c94d35892c78f98a7b40dacf9b5bf85b21d15**），但**尚缺 4 项具体步骤**（已转 31/33 补齐，doc-only）：①完整 window `--execute` 模板（含 `GLOBAL_LOG_LEVEL=0`+`PROCESS_LOG_PATH` 导出与全部 argparse；plan 是输出不是输入；建议同模板去 `--execute` 重跑 plan-only 验 SHA 仍为 c95ae5c3）；②raw fresh+默认 `/root/ascend/log` 时间窗+device 采集命令/时间关联/SUMS（不按 PID 过滤）；③rank env 3 白名单（GLOBAL_LOG_LEVEL/PROCESS_LOG_PATH/CLEF_WINDOW_OWNER）读取步骤（descendants→rank PID→`/proc/<pid>/environ`，缺失记缺测）；④执行前重扫清单（identity/starttime/boot/五 PID/health/info/clients、29531 TCP FREE、四目录 ABSENT、logs 可写、stop/source/body/frozen SHA、服务未重启、无其他设备格）。
- **§85 最终计划复核（947ddb3，2026-11-19）**：plan c95ae5c3 绑定、env 全名+unset+child 继承、rank PID→`/proc/<pid>/environ` 三键白名单、采集对象（fresh LOGDIR+/root/ascend/log device-0/1+cell+meta、不按 PID 过滤、correlation 另存）、四类判读（未定 xref 0x14ee680/0x1613f58/0x1614098 不用）、唯一 restore env -i 均 PASS。**仍缺 3 个执行准备步骤**（已转 33/31，doc-only）：①完整 window `--execute` 命令模板（plan 是输出不是输入；建议同模板去 `--execute` 重跑 plan-only 验 SHA 仍 c95ae5c3）；②采集 commands/时间窗/SUMS；③执行前完整重扫清单（identity/starttime/boot/五 PID/health/info/clients、29531 TCP FREE、四目录 ABSENT、logs 可写、stop/source/body/frozen SHA、服务未重启、无其他设备格）。
- **§86 执行准备模板复核（63bda3a，2026-11-19）**：骨架齐，但**未闭**。阻断：①§86.1 运行 repo 根 `runtime/grouped_mmar/supervised_probe_window.py`，**不是 e38b541 stage 的冻 window**；须绝对 stage 路径 + venv 绝对 python。②§86.2 采集用相对 `run-h11-observe-v1`（cwd 为 repo 根）且 `meta/start-time.txt` 在 mkdir 前写入；须 `CELL`/`ARCHIVE` 绝对路径、先 mkdir、START/END marker（替代 `-newermt "@$START"`）；device 路径/脚本仍占位。③§86.3 读 `descendants.json` 的 `records`，实际键为 **`entries`**，rank 列表会空、白名单检查失效。Fill-in：§86.4 `ps -o sid` 兼容性（改 `/proc`/`sess`）、端口 0.0.0.0、补 test_probe `ad257b5f`/plan `c95ae5c3`/root identity `de780872` SHA、health/info、五 PID/sid、clients、无其他设备格；相对 root-identity 路径绝对化。
- **结论**：preflight **PASS**；执行准备**未闭**（§86 上述 3 阻断+fill-ins）；剩余设备授权与条件项（AICPU env 传播、guard 映射、正标记）。**31 保持不执行、无 GO**；设备暂停。

## 8.93 h11-observe-execution-v1 包状态（2026-11-19，无 GO）

- **已有（untracked，未提交）**：`window-execute-argv.json`、`window-planonly-argv.json`、`README.txt`。两份 argv 数组绝对路径正确：venv python、**e38b541 stage** 的 `supervised_probe_window.py`、显式 `--python`、run-dir/root identity/stop/expect-stop-sha/arm H11/adapter/probe-temp/out/k-n-rows-seed/master/comm/child/stop/restore timeouts/port/source/expected-source-sha/official/activate/service-root/model-dir/restore-output/clean-env-expect/text/image；execute 版仅多顶层 `--execute`；env 仅两项 sidecar、无 unset。
- **缺失（README 自己索引）**：`window-launch-template.txt`、`planonly-rerun.txt`、`collect-raw.sh`、`rank-env-read.py`、`preflight-rescan.sh`、`restore-and-failure.txt`；**无 SUMS、无 commit**（目录 `??` untracked）。因此 README 中的“plan-only 复核通过 c95ae5c3”“collect-raw/rank-env/preflight 脚本就绪”等声明**尚无原件可核**。
- **待 31 补齐后复审**：plan-only 结果与 c95ae5c3 逐字节一致；collect-raw 绝对 CELL/ARCHIVE、先 mkdir、START/END marker、fresh LOGDIR+/root/ascend/log device-0/1/plog/vllm_ascend 完整文件、不按 PID、SUMS/correlation；rank-env `entries`+ranks 存活期三键；preflight-rescan 完整清单；唯一 restore env -i/失败留现场。**当前不执行、无 GO**；设备暂停。

## 8.94 execution-v1 9 文件内容审阅（2026-11-19，未提交/无 GO）

- **已到 9 文件（untracked）**：README、window-execute/planonly argv、window-launch-template、planonly-rerun、collect-raw.sh、rank-env-read.py、preflight-rescan.sh、restore-and-failure.txt。
- **PASS**：argv 绝对路径（stage window/venv/全 argparse，execute 仅多 flag）；launch template 仅两项 env、无 unset；planonly-rerun 记录 c95ae5c3/diff IDENTICAL；rank-env-read 用 descendants `entries`+pgid、ranks 存活期三键；preflight-rescan 覆盖 identity/boot/五 PID/health/info/clients/29531 tcp+udp×127+0.0.0.0/四路径/logs/stop+source+body+window+run_probe+binding+adapter+test+manifest+plan+root identity SHA/无既有 child；restore-and-failure env -i 唯一、成功门、失败留现场。
- **仍缺/需修**：①**提交 + SHA256SUMS**（当前 `??` untracked，无包 SHA）；②collect-raw 的 `$CELL/env-sidecar`、`$CELL/plan`、`$CELL/text-body.json`/`image-body.json` 可能不在 H11 cell（env sidecar/plan 在 h11-observe-preflight/，body 在 a15-large cell），需改实际来源或显式 source→archive 映射，必需 meta 不得 `|| true` 静默缺失；③建议 exact 五 PID、`attributed survivors` 措辞、planonly-rerun 原始输出/`sha256sum`/`diff` 证据。
- **结论**：内容基本 PASS，但包未提交/无 SUMS 且有 collect-raw 源路径缺口；**暂不给 readiness/包 SHA**；当前不执行、无 GO。

## 8.95 固定包 39f30b9 一致性核验（2026-11-19，未闭/无 GO）

- **从包目录复跑 SUMS**：10 项中 **9 OK，`./collect-raw.sh: FAILED`**；SUMS 自哈希 **b88f9242** 与 31 报一致。HEAD 的 `collect-raw.sh` blob 为 **954b708c**，与 SUMS 条目一致 → **39f30b9 提交本身自洽**，但该版本仍是旧脚本：`$CELL/env-sidecar`、`$CELL/plan`、`$CELL/text-body.json`/`image-body.json` 源 + `|| true` 静默缺失。
- **工作树另有未提交修复** `collect-raw.sh` = **13c9da04**：`BODIES=/workspace/.../a15-large`、`need_file/need_dir`、`meta/sources.txt`、必需输入缺失 `exit 1`、plan/body 实际来源——方向正确，但未提交、未入 SUMS。
- **结论**：39f30b9/SUMS b88f9242 与“远端副本 SHA 一致”的说法需澄清——若远端是 13c9da04，则**不匹配** SUMS；若远端是 954b708c，则 source 路径缺口仍在。**readiness/包 SHA 未闭**：31 须提交 13c9da04（或等价修复）+ 重新生成 10 文件 SUMS + 新 commit/包 SHA，并确认远端与新 SUMS 一致。其余 9 文件（argv/template/planonly/rank-env/preflight ALL PASS/restore/env/limits 120-240-900）PASS；35 的 rank-env CPU 替身结果待并入。当前不执行、无 GO。

## 8.96 execution-v1 最终 readiness（a31e044/afb741c7，2026-11-19，无 GO）

- **包 SHA**：commit **a31e044**；SUMS **afb741c7e9d77dc1ffc819995d251f5df2f44920ac483babfdc1d9ee43991803**（包目录 **11/11 OK**，自哈希一致）；plan **c95ae5c366a370423fdc42c99d5c94d35892c78f98a7b40dacf9b5bf85b21d15**；rank-env-read.py **a4b766765ba8137b0fbb0b37556ac159409804c9c80a2b2adb9e423b2a15d1e0**（与 35 目标同 SHA）。远端 11 文件逐一同 SHA 为 31 报。
- **PASS**：argv 60/61（execute 仅多顶层 `--execute`）；window e38b541 stage 绝对 + venv 绝对 python + 全 argparse；env 仅两项 sidecar、无 unset；collect-raw 源→archive 映射（CELL sidecar/plan/root-identity + A15-large body a7dd3f64/f3229f5c）、必需缺失 **exit 1**、meta/sources.txt；START/END+slack marker、完整文件不按 PID、SUMS/correlation；rank-env `entries`+pgid+三键+MISSING；preflight-rescan **ALL PASS rc=0** 且精确五 PID 1337814/1337961/1337962/1337979/1337980；restore 唯一 env -i、attributed survivors、120/240/900 上限；planonly-rerun-output.json SHA c95ae5c3。
- **条件式闭合**：①35 rank-env CPU 替身（同 SHA a4b76676）待并入，失败仅修该文件；②AICPU env 传播/guard 映射/正标记仍实验有效性项，缺失只记缺测；③真实执行前必须重跑 preflight-rescan（当前为时点扫描）。执行前重扫：identity/starttime/boot/ppid/pgid-sid/五 PID exact/health/info/clients；29531 tcp+udp×127+0.0.0.0；四路径 ABSENT；logs 可写；全部冻结 SHA；无既有 child；服务未重启；无其他格。
- **当前不发 GO、不执行设备**；等 20 明确执行授权。

## 8.97 rank-env CPU 替身并入 readiness（2026-11-19，无 GO）

- **sidecar 核验**：`/Users/chenclay/.ao/data/artifacts/clef-27b-ascend-35/h11-rank-env-cpu-standin-2026-11-19.md` SHA **e0511e91**、**4,363 B**；helper **a4b76676…** 在 39f30b9 与 a31e044 **同 blob**，window 1f780bf8；方法为 byte-identical 复制 + mock /proc + 真实 descendants.json；**18/18 PASS**（A 11/11 helper 真实逻辑；B 7/7 registry entries/start/stop）。
- **边界（并入判读）**：fallback `scan_pgid(root_pid)` 只覆盖 `pgid==root_pid`；torchrun rank 独立 PGID **只能靠 descendants `entries`**；**空/缺失 entries → 空行 rc0 = 缺测**，不是“无 rank”/排除支路；死 PID `alive=false`+`MISSING(not_alive)`；缺键 `MISSING`；helper 无 kill/signal/subprocess/torch。
- **最终 readiness**：包 **a31e044 / SUMS afb741c7（11/11）**、plan **c95ae5c3**；argv/template/collect-raw/rank-env/preflight-rescan ALL PASS/restore 均 PASS；命令准备**闭合（条件式）**。剩余：AICPU env 传播/guard/正标记（缺失记缺测）；真实执行前重跑 preflight-rescan；等明确执行授权。**当前不发 GO、不执行设备**。

## 8.98 execution-v1 最终包 953eb2d 复审（2026-11-19，无 GO）

- **包 SHA**：commit **953eb2d**（branch ao/resume-cc/execute-20261119，31 报远端=953eb2d）；SUMS **754859dccd2438a735f028e669263b9c69de3254bb07ffb8384c54431e13134a**（包目录 **11/11 OK**，自哈希一致）；plan **c95ae5c3**；helper **a4b76676**；rank-env CPU 替身 sidecar **e0511e91**（4,363 B，18/18 PASS，同 helper SHA）。
- **PASS**：argv 60/61（execute 仅多顶层 `--execute`）；stage/venv/全 argparse；env 仅两项、无 unset；collect-raw 显式 `SRC_ID/SRC_SIDECAR/SRC_PLAN/BODIES`（可 env 覆盖）、`need_file/need_dir`、`meta/sources.txt`、缺失 **exit 1**；START/END marker、完整文件不按 PID、SUMS；preflight-rescan ALL PASS 精确五 PID；restore 唯一 env -i、child PGID + attributed survivors、120/240/900；planonly 完整原始命令 + sha256sum/diff + 输出 c95ae5c3。
- **条件式闭合**：AICPU env 传播/guard/正标记仍实验有效性项（缺失记缺测）；真实执行前必须重跑 preflight-rescan；等待 20 明确执行授权。**当前不发 GO、不执行设备**；服务 aa3e/s1 root 1337814 健康。

## 8.99 最终引用统一（2026-11-19，无 GO）

- **最终命令包文本**：commit **953eb2d**、SUMS **754859dccd2438a735f028e669263b9c69de3254bb07ffb8384c54431e13134a**（11/11），远端 execution-v1 已同步（31 报现场 sha256sum 11/11 OK）。**a31e044/afb741c7 为前一提交**，执行参数与 953eb2d 完全相同（argv 60/61、env 两项、240/120/900/90、window/run_probe/adapter、helper a4b76676）；差异仅为 collect-raw SRC_* 显式化、restore 措辞、planonly 完整命令等 doc 收口。
- 已请 33 将 §91 引用从 a31e044 更新到 953eb2d；不重开、不改包。
- 条件项：AICPU env/guard/正标记、执行前重跑 preflight-rescan、等待 20 明确执行授权。**当前不发 GO、不执行设备**。

## 8.100 953eb2d delta 核验（2026-11-19，无 GO）

- `git diff a31e044..953eb2d` 仅 5 文件：README.txt、SHA256SUMS、collect-raw.sh、planonly-rerun.txt、restore-and-failure.txt。变更即要求的收口：collect-raw 显式 `SRC_ID/SRC_SIDECAR/SRC_PLAN/BODIES` 默认路径 + 缺失 exit1、restore child PGID+attributed survivors、planonly 完整原始命令。
- **关键文件 SHA 未变**（a31e044 == 953eb2d）：window-planonly-argv `001c44c2`、window-execute-argv `68670524`、rank-env-read.py `a4b76676`、window-launch-template `6cb3e877`、preflight-rescan `98a92998`；复用此前 PASS。
- 953 包目录 SUMS **11/11 OK**、自哈希 **754859dc**；无真实新增阻断。**命令准备闭合**；剩余 AICPU env/guard/正标记、执行前重跑 preflight-rescan、明确执行授权。**当前无 GO/无设备动作**。

## 8.101 H11-observe-v1 唯一 GO 已发（2026-11-19）

- 20 依人类原始授权（session20 seq3882）明确一次执行；我确认本格**此前未发过 GO**。
- 唯一 GO 发往 31，`--steer --client-message-id **h11-observe-v1-go-20261119-1**`；固定包 **953eb2d/SUMS 754859dc（11/11）**、plan **c95ae5c3**、frozen e38b541/8e308804/1f780bf8/101660f8/d181b15a。
- 条件：启动前 `preflight-rescan.sh` **ALL PASS rc=0**；任一 identity/port/SHA/路径漂移即停报；env 仅两项 sidecar；stop120/child240/comm90/restore900；一次 `--execute`；禁止第二棵 restore/同格自动重试；child 失败也 cleanup+唯一 restore；collect-raw begin/end、rank-env 三白名单、正向标记/空=缺测；回传数值/超时/raw+SUMS/survivors/cleanup/restore 五 PID/四检/source/subreaper 三值。范围仅 H11 small。
- **交付状态**：`ao send` 返回“No active turn to steer; accepted as a normal Chat turn in running state with delivery handle …; provider delivery is not confirmed”；`--recover-only` 同 handle 返回“already accepted; delivery state unchanged”。**未重发、未换 id**；已报 20 确认 31 是否收到。
- **20 已确认交付**：31 明确回报“收到唯一 GO”，预扫 **ALL PASS（含包 SUMS 11/11）**，正在 `collect-raw begin` + 同 shell 两项 env + 单次 window。保持 handle **h11-observe-v1-go-20261119-1**，不重发/不换 id/不再 recover。
- 当前等待 31 的实际 raw 与恢复结果；无设备动作。

## 8.102 H11 单次执行启动失败（2026-11-19，无重跑/无第二 GO）

- **事实**：单次 window 在 child 启动前 `FileNotFoundError: torchrun`（SSH shell PATH 未含 venv）；探针尚未启动、无 MRS/Grouped 设备路径，**不作算子负结果**。
- **处置**：不重跑 window、不发第二 GO；31 优先核服务/唯一 restore 并保留原件（window.json/child stdout/stderr/exec、collect archive+SUMS、descendants/stop、restore 证据）。
- **若 window 未触发 restore**：仅在“已停服、无 probe/归属进程、source/身份/端口等条件确认”后，按 20 许可使用原 plan `restore_argv` **第一次唯一恢复**；**不得重启 window 间接重跑**。
- **后续修复项**：CPU 准备审计遗漏 PATH/baseline activate（launch shell 未 source activate 或 PATH 未含 venv bin），记后续 doc/流程修正；当前不启动新 NPU。
- 等待 31 回传失败原件与恢复结果后只读核验。

## 8.103 H11 attempt-1 失败/恢复原件核验（2026-11-19，无 GO）

- **原件**：`h11-observe-v1-cell-attempt1/` 13 文件，SUMS **13/13 OK**、自哈希 **fd41ac86**（commit 50241c2）；archive **51 文件**、SUMS **4fbf9653**（archive-SHA256SUMS.txt 50 条 + SUMS 自身）。
- **失败**：window `ok=false`，error `FileNotFoundError: torchrun`；steps preflight/stop/stopped/child；`child.started=false`；probe 未启动、无 MRS/Grouped 设备证据 → **harness PATH 启动失败，非算子负结果**。
- **恢复（window 唯一发起）**：`restore` started=true、wrapper_pid **1347085**、argv env -i + 官方 start.sh；restore root **1347086**/starttime **401212026**/ppid1/pgid-sid self/boot 393ecc15；五 PID [1347086,1347229,1347230,1347258,1347259]；manifest pid 1347086/config s1/started_at 2026-11-19T13:32:24Z/revision local-65e5a3c5ffb6；health ready/info s1；branch **aa3e-noop**、isolation_failed 无、**9/9 checks true**、clients []、source 三段 aa3e1758、subreaper 0→cleared→final0；stop verified；29531 tcp/udp×127+0.0.0.0 FREE；**无第二 restore/无残留**。
- **rank-env**：26 次采样 entries=0/rows=[] → **缺测**（非“无 rank”/排除）。
- **修复项（后续）**：启动 shell PATH 前置 venv bin 或用绝对 torchrun；rescan 增加 torchrun 可用性检查；若再次授权需 fresh identity/新 plan（当前 plan identity 1337814 已过期）。当前不启动新 NPU、不重跑、不发第二 GO；服务 root 1347086 健康（31 现场）。

## 8.104 attempt-1 最终证据（14 文件，2026-11-19，无 GO）

- **证据**：commit **3d54f70**、14 文件 SUMS **14/14 OK**、自哈希 **f3733350**；`original-artifacts-manifest.txt` 记录远端：window **6060e263**、stop **0bbc14bd**、child.stdout/stderr/stop.log/restore.log 全空 `e3b0c442`、restore joint **a31ad8ce**/launch **431d6105**/service.log **cba4f8b5**、archive SUMS **4fbf9653**（51 文件）；`descendants.json`、`probe-temp`、`probe-out` **ABSENT**；proc scan 空。
- **恢复**：window 唯一发起；root 1347086/start 401212026/五 PID/branch aa3e-noop/9-9/source 三段/subreaper 0→0/29531 free；无第二 restore/残留。
- **结论**：attempt-1 为 harness PATH/torchrun 启动失败，非算子负结果；原件完整保留。后续：PATH 前置 venv 或绝对 torchrun、rescan 增加 torchrun 检查、若再授权需 fresh identity/新 plan。当前服务 root 1347086 健康；无新 NPU、无重跑、无第二 GO。

## 8.105 H11-observe-v2 CPU 准备 readiness（2026-11-19，无 GO）

- **包/plan**：commit **ea30cec**、包 SUMS **e30e221cec97ae5a2c68b13fb0052e6fc137b7e44fcd23fa95176e45e87c570b（19/19 OK）**、自哈希一致；plan **9c2936c240112599209c3aa396d7695bbac10b91ca908733777df1bb36a13137**，plan_not_executed；plan-only 同 argv 输出逐字节同 SHA。
- **新 identity/路径**：root **1347086/401212026**/ppid1/pgid-sid self/boot 393ecc15；五 PID exact 1347086/1347229/1347230/1347258/1347259；新 CELL `h11-observe-v2`，run/temp/out/restore fresh ABSENT、logs 已建可写；root identity 文件 84c9edb2、env sidecar 2474b487。
- **argv**：60/61（仅多顶层 `--execute`），全部 `-v2` 路径；H11/128/256/512/seed/240/120/900/90；adapter d181b15a；restore -v2。
- **PATH/torchrun（实核）**：`activate_env.sh` SHA **3b363394**，第 15/23 行 source `$CLEF_VENV/bin/activate`，**会加入 venv bin**；v1 根因是启动 shell **未 source activate**，不是脚本不加 PATH；v2 显式 `source activate → export PATH=venv/bin:$PATH`（幂等加固）。基线 `env -i` clean PATH 下 `command -v torchrun` 空/rc1（复现）；effective env 解析 `/workspace/yangchen/venvs/clef-py312/bin/torchrun`、executable、SHA **fd801bde…0903**、解释器 venv python、Python 3.12.16、**`torchrun --help` rc=0**（CPU-only，154 行独立日志）。
- **preflight-rescan**：实跑 **ALL PASS rc=0**，含 identity/boot/五 PID exact/health/info/clients/29531 tcp+udp×127+0.0.0.0/四路径 ABSENT/logdir/全部冻结 SHA/plan-v2/root-identity-v2/env-sidecar-v2/torchrun 基线缺失+effective+help；无其它 probe/torchrun。
- **冻结/复用**：e38b541/8e308804/1f780bf8/101660f8/d181b15a/ad257b5f/8e324bea 不变；helper a4b76676 同 blob；v1 包与 attempt-1 未改；无 restage/Linux40/18-helper 重跑；远端 19/19 为 31 报。
- **结论**：v2 CPU 准备 **readiness 闭合**，无新增阻断；剩余 AICPU env/guard/正标记、真实执行前重跑 preflight-rescan、20 明确 v2 执行授权。**我不发 v2 GO**。

## 8.106 v2 85ab351/b5cec871 增量审：2 阻断（2026-11-19，无 GO）

- **包/plan**：commit 85ab351（ea30cec 后 doc-only 收口）、SUMS **b5cec871**（19/19 OK）；plan **9c2936c2**、identity 1347086/401212026、argv 60/61、frozen e38b541/8e308804/1f780bf8/101660f8/d181b15a 未变；helper 逻辑 a4b76676、18/18 可复用；v1 未改；无 Linux40。硬顺序模板与 window 1f780bf8 line 894/896（child_env=os.environ）、546-548（--activate 仅 restore）、876（--execute 先 stop）一致。
- **阻断 A（preflight 假阳性，35 指出成立）**：preflight-rescan 的 torchrun 段在嵌套 `/bin/bash -c` 自建 env（source activate+export PATH+command-v/--help），baseline 用 `env -i`；**不检查调用 shell 自身 PATH/command -v**。裸 shell 可 ALL PASS，随后同 shell 启动 window → child 继承裸 os.environ → 裸 torchrun FileNotFound，stop 已发生，v1 复现。修复：caller `command -v torchrun` fail-closed 检查（不嵌套）或单一 wrapper（env→caller→preflight→exec --execute），模板只允许 wrapper。
- **阻断 B（helper 未绑定 v2）**：v2 rank-env-read.py 同 blob a4b76676，但 `RUN=CELL/run-h11-observe-v1` **硬编码 v1**、默认 CELL 也是 v1；传 v2 CELL 仍读 v1 run → 空/缺测；v2 README 无 v2 调用。需 v2 helper 副本（逻辑同 18/18）或 `--run`/env 覆盖，并写明 v2 CELL 命令。
- **小项 C**：v2 collect-raw 标题/注释仍 v1、注释源 SHA 为 v1（de780872/a6a3c4a4/c95ae5c3）；功能路径已是 v2，需改注释为 v2（84c9edb2/2474b487/9c2936c2）。
- **复审更新（35 口径 + 32 复核）**：A 归为**程序性门**——模板 step4 在同一 shell 手工 `command -v torchrun`+SHA+`--help` rc0，严格按模板视为 PASS；preflight 脚本仍是嵌套 recipe，非脚本级 caller fail-closed（残余改进，不强制）。B 仍是**真实阻断且比 35 建议更强**：helper `RUN=CELL/run-h11-observe-v1` 硬编码 v1，仅传 v2 CELL 仍读 `...cell-v2/run-h11-observe-v1`（不存在）→ 空/缺测；需 v2 helper 副本（RUN 后缀 v2，逻辑 18/18 复用）/`--run` 覆盖/显式 v2 调用。C 注释 v1 为小项。
- **结论**：v2 readiness **未闭**（B+C）；修后复审。当前不执行、不发 v2 GO；设备暂停。

## 8.107 v2 5923f7e/f5cf839e 复审（2026-11-19，A/D 闭，B/C 未闭，无 GO）

- **A 已闭（脚本级）**：preflight 新增 caller `command -v torchrun` 必须 venv、caller PATH 含 venv、caller `torchrun --help` rc0，recipe 检查保留；bare shell **RESULT FAIL rc1**（validation-caller-bare-fail.txt），prepared shell **ALL PASS rc0**（validation-prepared-pass.txt）；wrapper `run-h11-observe-v2.sh` 同 shell env→caller→help→preflight(rc0)→exec window --execute。
- **D 已闭**：wrapper/validation 已入包；SUMS **21/21 OK**、自哈希 **f5cf839e**；commit 5923f7e（85ab351 之后）；v1 未改；无 Linux40。
- **B 仍开放（真实阻断）**：`rank-env-read.py` 同 blob a4b76676，`RUN=CELL/run-h11-observe-v1` 硬编码；v2 README 仅描述、无实际调用命令；仅传 v2 CELL 仍读 `...cell-v2/run-h11-observe-v1`（不存在）→ 空/缺测。需 v2 helper 副本（默认 CELL/RUN 后缀 -v2，逻辑 18/18 复用）或 `--run` 覆盖 + 精确 v2 调用（ranks 存活期、--watch）。
- **C 小项**：collect-raw 标题/注释仍 v1、注释源 SHA v1；功能路径已 v2。
- 包/plan（9c2936c2）/identity（1347086）/argv（60/61）/frozen（e38b541/8e308804/1f780bf8/101660f8/d181b15a）PASS。**v2 readiness 未闭（B+C）**；修后复审；当前不执行、不发 v2 GO。
- **冻结例外（2026-11-19）**：31 称 5923f7e 后“包冻结不再追加 commit”，但 B 是真实阻断（helper RUN 硬编码 v1 + 无 v2 调用），属冻结例外；已明确要求 31 修 B（v2 helper 副本/--run + 精确 v2 命令）与 C（collect 注释 v1），重生成 SUMS/commit 后复审。A 已升级为脚本级闭（caller 强制 + bare FAIL/prepared PASS + wrapper），不再是仅程序性门；D 已闭。

## 8.108 v2 B/C 中间态（2026-11-19，未提交/SUMS 旧，无 GO）

- **工作树（HEAD 5923f7e，4 文件 M）**：`rank-env-read.py` **68dc9e2c** 已改默认 CELL=v2、`RUN=<CELL>/run-h11-observe-v2`、支持 `--run` 覆盖；README/模板已给精确 v2 调用（v2 CELL + `--watch 180`）；collect-raw 标题已 v2。方向正确。
- **E 未闭**：SUMS 仍为 21 条旧值，`shasum -c` 对 README/collect-raw/rank-env-read（及 template）FAILED；未 commit。
- **要求**：31 提交 B/C + 重生成 21 文件 SUMS/新 commit；35 对**提交后 helper SHA** 做真实调用路径 CPU 回归（不 patch RUN 常量，spy 实际 v2 descendants 访问），PASS 后我复审 delta/SUMS/README 调用并给完整 plan SHA（预计 9c2936c2）/最终 readiness。
- 当前不执行、不发 v2 GO；设备暂停。

## 8.109 B 负控与新 helper 回归口径（2026-11-19，无 GO）

- **35 旧 helper a4b76676 负控（/tmp/h11_verify/test_call_binding.py，不 patch CELL/RUN）**：显式传 v2 CELL 仍实际打开 `<CELL>/run-h11-observe-v1/descendants.json`（v1 run，rows 仅 2001、无 2002）；默认（不带 CELL）读 v1 根 + v1 run。→ 仅传 v2 CELL 不够，B 确认为真实阻断。
- **31 工作树 68dc9e2c**：默认 CELL=v2、`RUN=<CELL>/run-h11-observe-v2`、支持 `--run`；README/模板精确 v2 调用；collect 标题 v2。仍在 HEAD 5923f7e 之外未提交，SUMS 旧（README/collect/rank-env/template FAILED）→ E 未闭。
- **验收（新 helper SHA，35）**：①显式 v2 CELL → 实际打开 `...-v2/run-h11-observe-v2/descendants.json`，rows 见 2002、不见 2001，v1 路径不访问；②默认模式路径必须指向 v2 run 或明确要求显式 CELL；③新 SHA 重跑原 18/18 小夹具（entries/独立 PGID/缺键/退出 MISSING/三白名单/只读）；不 monkeypatch CELL/RUN；旧 a4b76676 证据标注旧 source。
- **待 31**：提交 68dc9e2c + 重生成 21 文件 SUMS + C/D；之后 35 回归；PASS 后 32 给完整 plan SHA（预计 9c2936c2）/最终 readiness。当前不执行、不发 v2 GO；设备暂停。

## 8.110 v2 b04f8fe/dd5aac02 增量审（2026-11-19，B/C/D 闭；35 回归待；无 GO）

- **包**：commit **b04f8fe**、SUMS **dd5aac02f7a57fefb46e58c4283721fcb95f9bbe0e8740964775b6c88d7aa7ae（22/22 OK，自哈希一致）**；plan **9c2936c2**、root identity 84c9edb2、env sidecar 2474b487、argv 60/61、stage identity 7ee4933e 未变；frozen e38b541/8e308804/1f780bf8/101660f8/d181b15a；v1 未改；无 Linux40。
- **delta 5923f7e→b04f8fe**：README、SUMS、collect-raw、helper-v2-fixture-check.txt（新）、rank-env-read.py、window-launch-template.txt。
- **B 闭**：helper **68dc9e2c** 默认 `CELL=/workspace/.../cell-h11-observe-v2`、`RUN=<CELL>/run-h11-observe-v2`、支持 `--run`；README/模板精确 v2 调用（v2 CELL + --watch 180）；31 fixture 证据默认读 v2 run/alt-run/死 PID MISSING。
- **C 闭**：collect-raw 标题 v2、源注释 SHA 84c9edb2/2474b487/9c2936c2。**D 闭**：22 文件与 README 索引一致、无悬空 validation-readonly-v2。
- **待 35**：对提交后 helper 68dc9e2c 做独立真实调用路径回归（显式 v2 CELL 打开 v2 run、2002 不见 2001、v1 不访问；默认 v2；新 SHA 18/18；不 patch CELL/RUN）。PASS 后给最终 readiness/plan SHA。当前不执行、不发 v2 GO。

## 8.111 v2 991c21c/6075c4c0 增量（2026-11-19，35 回归待，无 GO）

- **包**：commit **991c21c**、SUMS **6075c4c0a68ac13d64da730a04bc8968baae90139a56f5200ea313981d187509（22/22 OK，自哈希一致）**；delta b04f8fe→991c21c 仅 README/SUMS/stage-identity；stage `rank_helper_sha256` = **68dc9e2c**；plan 9c2936c2、argv 60/61、helper 68dc9e2c 未变；frozen/v1 未改。
- **README 旧 helper 文本**：唯一 `a4b76676` 为“非 v1 a4b76676”的历史澄清，**不是活动绑定**；31 的“grep 无命中”表述略不精确，可选删括注，不构成阻断。
- **待 35**：对 68dc9e2c 独立真实调用路径回归（显式 v2 CELL→v2 run、2002 不见 2001、v1 不访问；默认 v2；新 SHA 18/18；不 patch CELL/RUN）。PASS 后 final readiness/plan SHA。
- 当前不执行、不发 v2 GO；设备暂停。

## 8.112 991c21c 固定快照复验（2026-11-19，35 回归待，无 GO）

- **从 commit tree（`git show 991c21c:`）复算**：SUMS 自哈希 **6075c4c0a68ac13d64da730a04bc8968baae90139a56f5200ea313981d187509**；22 条目逐项匹配 commit blobs（ok=22/bad=0）；helper **68dc9e2c**；stage `rank_helper_sha256`=**68dc9e2c**；delta b04f8fe→991c21c 仅 README/SHA256SUMS/stage-identity。
- A/D/argv/plan（9c2936c2）/identity（1347086/401212026）/frozen（e38b541/8e308804/1f780bf8/101660f8/d181b15a）沿用此前 PASS；v1 未改；无 Linux40；不再以工作树中间态作为固定版问题。
- **待 35**：对同 68dc9e2c 做真实调用绑定 + 小 fixture（显式 v2 CELL→v2 run、2002/2001、默认 v2、新 SHA 18/18、不 patch CELL/RUN）。PASS 后给新 plan 完整 SHA/最终 readiness。当前不执行、不发 v2 GO。

## 8.113 H11-observe-v2 最终 readiness（2026-11-19，PASS，无 GO）

- **包 SHA**：commit **991c21c**；SUMS **6075c4c0a68ac13d64da730a04bc8968baae90139a56f5200ea313981d187509（22/22，git show 991c21c 逐项匹配）**；**plan 9c2936c240112599209c3aa396d7695bbac10b91ca908733777df1bb36a13137**；helper **68dc9e2c**；identity 1347086/401212026、五 PID；argv 60/61；frozen e38b541/8e308804/1f780bf8/101660f8/d181b15a。
- **A/D 闭**：caller torchrun 强制（bare FAIL rc1/prepared PASS rc0）、wrapper 同 shell、README/SUMS 22 文件一致。**C 闭**：collect 注释 v2。
- **B 闭 + 35 独立回归 PASS**：artifact **d3a0787a**（3,584 B）；显式 v2 CELL→v2 run（rows 2002、无 2001、无 v1 路径）、默认 v2、--run alt、旧 a4b76676 负控保留；新 SHA 逻辑 **13/13**；35 独立复核 SUMS **22/22**。
- **条件**：①AICPU env 传播/guard/正标记（空/缺失=缺测，不排除路径）；②真实执行前重跑 preflight-rescan（caller torchrun/identity/端口/目录/SHA）；③等 20 明确 v2 GO。
- **执行口径**：唯一 wrapper 同 shell；env 仅两项 sidecar；collect-raw begin/end；rank-env 68dc9e2c 精确 v2 调用；唯一 restore/无第二棵；stop120/child240/comm90/restore900。
- 当前不执行、不发 v2 GO；服务 root 1347086 健康。

## 8.114 H11-observe-v2 唯一 GO（2026-11-19）

- 20 依人类 seq3882 明确一次 v2 设备日志观测；确认 **v2 此前从未 GO**。唯一 GO 经 `--steer --client-message-id **h11-observe-v2-go-20261119-1**` 发往 31。
- 固定引用：包 **991c21c / SUMS 6075c4c0…d187509（22/22）**、plan **9c2936c2…3137**、helper **68dc9e2c…42d9**；H11 M128/K256/N512/world2/BF16/KEEP_ALL/native/FP32zero/seed20261119；frozen e38b541/8e308804/1f780bf8/101660f8/d181b15a。
- 条件：唯一 wrapper（同 shell source activate+venv PATH→caller torchrun which/SHA/help→preflight ALL PASS rc0→window --execute）；禁手抄/split；启动前 identity 1347086/401212026/boot/五 PID、29531、fresh 四路径、source/stop/body/frozen/plan/bundle SHA 漂移即停；collect begin/end、rank-env v2 精确调用、空=缺测；stop120/child240/comm90/restore900；失败/超时 cleanup+原 plan 唯一 restore，禁同格重试/第二棵；范围仅 H11 small。
- **交付状态**：`--steer` 返回 accepted as a normal Chat turn in running state、provider delivery 未确认；`--recover-only` 同 handle 返回 already accepted/unchanged。**未重发、未换 id**；已报 20 确认 31 是否收到。

## 8.115 v2 GO 未执行：991 collect-raw identity 缺陷（2026-11-19）

- **31 停报**：按 GO 启动前采集，`collect-raw.sh begin` 发现 991 包 L56 **硬编码 `pid=1337814`（v1）**，v2 root 为 **1347086**；identity python 读 `/proc/1337814/stat` 失败 → `meta/identity.json` **0 字节**；脚本未对 identity 捕获 fail-closed（L93 只检查 need_file/need_dir 的 FAIL），begin 仍 **rc0**。
- **现场**：archive 已建（START/START_SLACK、sources/plan/body/sidecar/health/info/port），identity 空；run/probe-temp/probe-out/restore 仍 ABSENT；**未启动 wrapper/window、无 stop/torchrun/设备动作**；服务 root 1347086 健康。
- **定性**：包/harness collect 缺陷，不是 Grouped 算子证据；本次唯一 v2 GO **未执行 window，不视为完成**；旧空 archive 保留为证据。
- **修复**：collect-raw 从 `CELL/root-identity-h11-observe-v2.json` 读 pid，校验 pid/starttime/boot 与 /proc 一致；identity 捕获失败必须 FAIL/非零；**begin 与 end 都校验 identity.json（非空/可解析/与 SRC_ID 一致）**——35 补证 end 也不兜底、0 字节会被 SUMS 打包且 rc0；其它硬编码 18143/29531 与 v2 一致。重建 SUMS/commit 新固定包；20 另发 **v2 attempt-2 新 GO**（新 handle），当前 handle h11-observe-v2-go-20261119-1 保持未执行/不重发。下次 begin 前清理/重建空 archive。当前无设备动作。

## 8.116 4cbe23a collector identity fix（2026-11-19，PASS + 2 准备缺口，无 GO）

- **包**：commit **4cbe23a**、SUMS **8f4bca74c32b97e2214ddf7680c6dc04dc5ec919e0a2dc779bbd28edab3b660d（23/23，commit-tree 匹配）**；delta 991c21c→4cbe23a 仅 README/SUMS/collect-identity-fix-check/collect-raw；plan 9c2936c2、identity 1347086/401212026、argv 60/61、helper 68dc9e2c、frozen 未变。
- **Identity fix PASS**：从 `SRC_ID` 读 pid，/proc 校验 pid/starttime/boot；identity.tmp 成功才 mv；begin/end `need_identity` 非空/可解析/与 SRC_ID 一致，失败 FAIL/非零；fixture good begin/end rc0、bad start begin rc1、empty end rc1；**end 仅比对保留 before 快照，不要求旧 pid 存活**（阶段边界正确）；无 1337814 引用。
- **缺口 A（pre-stop begin 硬门）**：唯一 wrapper 仍只 env→caller→preflight→exec window，未调用 collect-raw begin/未强制 identity 后即可能 `--execute`；仍靠外部手工 begin。需 wrapper 内、任何 window --execute 前 `bash collect-raw.sh begin` rc0 硬门。
- **缺口 B（prestart1/clean archive）**：旧失败 canonical archive 仍存在（0B identity）；begin 用 mkdir -p 复用、无 fresh 门。需先以原字节+SUMS 保全为独立 prestart1，再干净重建 canonical（或 begin 拒绝非空 ARCH）。
- 已交 31 补 A/B；35 独立 identity fail-closed 审待；20 另发 attempt-2 GO。当前不执行、不发 GO。

## 8.117 4cbe23a identity 双审 PASS；两缺口确认（2026-11-19，无 GO）

- **35 独立复核**：4cbe23a/8f4bca74 **23/23**；SRC_ID→pid + `/proc` 校验、identity.tmp→mv、begin/end `need_identity`、**end 不访问旧 PID/不要求存活**、无 `1337814`/`h11-observe-v1` 绑定、helper 68dc9e2c 未变 → identity 修复 PASS。
- **缺口 A（真实，需 31 修）**：wrapper `run-h11-observe-v2.sh` 全文无 collect/begin，模板/README 无 begin 前置；`--execute` 内部先 stop，begin 后置则 before identity 无法捕获。最小修：wrapper 在 preflight 前、同 shell 执行 `collect-raw.sh begin`，非零即 abort（不进 preflight/stop/exec）；模板 step5 前加该步，README 写明仅 wrapper 路径、begin 一次且成功才继续。
- **缺口 B（真实，需 31 修）**：包内无“原字节保全+fresh 重建”说明；需先把 `archive-h11-observe-v2` 原字节复制到保留路径（或保留原目录），再用 fresh archive 路径 begin，不删除旧证据。
- **非阻断**：health/info `|| true` 与 29531 探针为 best-effort；identity/sources 为硬门（如需 fail-closed 由 20 明确）。
- 修 A/B 重冻后 20 另发 attempt-2 GO；当前不执行、无设备、不发 GO。

## 8.118 3c4bdc5 wrapper begin/metadata/prestart1（2026-11-19，PASS + C/D 未闭，无 GO）

- **包**：commit **3c4bdc5**、SUMS **7f24f88b1f048768695caa0534ac7c8f9c70772fc3844c2d898fdd484d36d182（24/24，commit-tree 匹配）**；delta 4cbe23a→3c4bdc5 13 文件（README/SUMS/collect-begin-smoke/collect-raw/wrapper/template/prestart1 证据）。
- **PASS**：wrapper env→caller→preflight→`collect-raw.sh begin`（`ARCH=...attempt2`、identity/metadata 有效）→`exec window --execute`，begin 失败 **exit6**、stop 未发生；begin health/info/port/plan/sidecar/body fail-closed；smoke good rc0、bad start/dead pid/invalid plan rc1；**end 仅 before 快照**；prestart1 仓库证据 SUMS **73963a05**（6/6），旧 archive 原字节保留；plan 9c2936c2/root 1347086/helper 68/frozen 未变。
- **未闭 C（ARCH 一致性）**：`collect-raw.sh` 默认 `ARCH=$CELL/archive-h11-observe-v2`（旧 0B archive）；wrapper begin 与 template end 覆盖 attempt2，但 README 无 attempt2 end 命令，**bare begin/end 会误回旧 archive**。需默认 ARCH 改 attempt2（或显式 ARCH 未设即失败），README/template/wrapper 四处统一。
- **未闭 D（attempt2 fresh）**：preflight 未检查 `archive-h11-observe-v2-attempt2` fresh；wrapper begin mkdir -p 复用。需执行前 record/check ABSENT（或 begin 拒绝非空 ARCH），仅此具体检查。
- health/info 新校验可保留；不再扩 check。修 C/D 重冻+35 复核后由 20 发续行 GO。当前不执行、不发 GO。

## 8.119 顺序裁定与 C/D 状态（2026-11-19，无 GO）

- **20 裁定**：保留 wrapper `env/caller→preflight→collect begin rc0→exec window`；preflight 只读、begin 在 stop/window 前，**停服前 begin 硬门已满足**，不互换 preflight/begin。33 的“字面 before-preflight”问题关闭。
- **D 当前证据（85f3ae4/4ef99a4c）**：`archive-h11-observe-v2-attempt2` **ABSENT**；旧 canonical EXISTS（0B identity）、prestart1 EXISTS；prestart1 SUMS 文件本体 d9217518、仓库证据 SUMS 292cf7fe（4c370713 为 start-utc 内容，已更正）。
- **C 仍待**：collect-raw 默认 ARCH 仍 `$CELL/archive-h11-observe-v2`、README 无 attempt2 end 命令；需默认/README/template/wrapper 统一 attempt2（31 正做最小修复）。
- 待 31 新固定 commit/SUMS：仅对 commit tree 核 C/D delta、SUMS、plan 9c2936c2/identity 1347086/helper 68/frozen 未变，不重开 A/B；35 并行。两审 PASS 后 20 授权再发绑定新包单次续行 GO。当前不执行、无设备。

## 8.120 8ecf8f9 最终 readiness PASS（2026-11-19，无 GO）

- **包**：commit **8ecf8f9**、SUMS **bed3a6e670d2ecd245e156ffe495c8e9480740a1b9e9bb0d97dbe291210c5b6c（24/24，commit-tree 逐项匹配）**；delta 3c4bdc5→8ecf8f9 仅 README/SUMS/collect-raw/preflight/validation + 证据目录/prestart1 说明，plan/argv/helper 未变。
- **C 闭**：collect-raw 默认 `ARCH=…attempt2`；wrapper begin（`ARCH=…attempt2`、exit6）；template end（`ARCH=…attempt2`）；README begin/end 均 attempt2；四处统一，无旧 ARCH 默认。
- **D 闭**：preflight 在任何 window `--execute` 前检查 `archive-h11-observe-v2-attempt2` 必须 ABSENT（`PASS attempt2 archive fresh (absent)`）；validation-prepared-pass RESULT ALL PASS rc0。
- **PASS 复用**：plan **9c2936c2**、identity **1347086/401212026**、helper **68dc9e2c**、argv 60/61、frozen e38b541/8e308804/1f780bf8/101660f8/d181b15a；prestart1 原目录/快照保全（仓库 SUMS 292cf7fe、远端 SUMS 本体 d9217518）；v1 未改；无 Linux40。
- **结论**：32 侧最终 readiness **PASS**；待 35 并行复核后 20 发绑定新包单次续行 GO。旧 GO 挂起、不重发；当前不执行、无设备。

## 8.121 H11-observe-v2 attempt2 原件与恢复验收（2026-11-19，无新 GO）

- **固定原件**：31 commit **80c73d9**，证据目录 `h11-observe-v2-cell-attempt2/`；SHA256SUMS **21/21 OK**、自哈希 **56a99868c13f68a1a4c5df944d2ea75b84a916e3c44267e4ba8b69b8e59b746d**；包 8ecf8f9 仍为 SUMS **bed3a6e6**（未改）。
- **window 生命周期**：stop 14:48:54–56Z，旧五 PID 全部 termed/gone、survivors=[]；child torchrun **1354277**、ranks **1354310/1354311**（各自 pgid=pid）；`timed_out=True`、`child.rc=0`；child.stderr 14:52:59.631 收 15、14:52:59.633 向两 rank 发 SIGTERM。
- **cleanup/gate**：`cleanup` 与 `cleanup_final` 均 `groups=[1354310,1354311]`、`killed=true`、`unknown/drift/singles=[]`、`members` 两 rank；`survivors=[1354310,1354311]`、`error=child group survivors`；`recovery_gate allow=false reason=blocked_probe_residual`；`restore.skipped/blocked=true`、**无 restore.started**、无 restore 目录（window 未发起 restore）。
- **gate 语义核对**：`_live_verified` 明确跳过 `state=Z`；gate 的 `attributed_survivors` 用新的 `/proc` 读取（非旧 scan 表）；故 gate 时刻两 rank 是**非 Z 的存活进程**，不是 zombie/采样假阳性。cleanup 已发 SIGTERM→1s→SIGKILL→1s→reap→0.05s→final scan，signal 已送达但 1s 有界等待内未退出；具体 state/退出时刻未采集，**D-state 不可断言**。
- **rank 结果语义更正**：两 rank `result.json` 的 `api_returned=false/sync_returned=false/finite=false/pass=false` 是 **run_probe 启动时的初始快照**（仅在 sync 返回后才再次 flush）；trace 证明 python **`api_return` 已发生**（workspace_bytes=16,777,216、run_id=1），随后 `sync_enter`，cpp `launch_return rc0`、`handler_return cleanup_rc0/sync_rc0`，python sync 240s 未返回。`timeout=false` 因外部 SIGTERM 终止且无 timeout.json，不是“未超时”。
- **人工第一次（唯一）restore**：plan `restore_argv` 7 段原件为 `env -i … CLEF_LAUNCH_CONFIG=s1 setsid /bin/bash -c "cd … && source activate_env.sh && exec bash start.sh --model-dir … --output-dir …/restore-h11-observe-v2"`；launch-manifest 内层 argv/output_dir=s1/pid=1355077/started 14:56:49Z 一致；joint-patch **branch=aa3e-noop**、target before/after 均 aa3e；root **1355077/starttime 401718579/pgid=sid self/boot 393ecc15**、**ppid=1355076 偏差如实记录**；source 三段均 aa3e；window subreaper original=0/final=0/final_restored=true。
- **归档**：archive 92 文件、SHA256SUMS 自哈希 **52d145ee…**（92 行，cann-plog 13/device 3/fresh-logs 40/meta 19/cell 17）；**probe-temp 0 条**，collect end **rc1**，原样保留，**不判完整采集 PASS**。
- **证据缺口（明确标注）**：①9/9 检查与五 PID 仅见 `manual-restore-record.txt` 引用 JSON，无独立原件；②实际外层 `env -i/setsid` argv 无独立 capture，只有 plan 原件+记录引用+launch-manifest 内层一致性；③两 rank 的 gate 时刻 `/proc` state 与精确退出时刻无原件；manual restore 启动前“两 rank 已消失”为 31 只读扫描陈述，包内无快照；④92 文件仅 SUMS 入 commit，payload 未随 commit；⑤`steps.child.maps_ok=false`（maps 覆盖 opapi/hccl 等为 false）为启动早期采样，非库缺失结论；⑥AICPU 生效正标记仍缺测。
- **验收结论**：window 的 restore **未发起**（fail-closed，原件）；人工第一次且唯一 restore 按 plan argv 来源执行，**服务恢复健康（9/9/source 三段/clients 空/端口 free 均为记录引用）**，但存在 ppid≠1 偏差与上述证据缺口，**不能等同 window 自动 restore 成功口径**；archive 为 **rc1 不完整**；本次为 **H11 sync 挂起复现**，无 H11 数值结论、无 A/B 结论、无第二 window/restore 痕迹。无新 GO、无设备动作、包未改。

## 8.122 cleanup 归因收窄：parent-exit-first + 即时 gate（2026-11-19，无 GO）

- **31 补证（只读）**：rank-env 44 samples 覆盖 14:48:59–14:52:35Z、5 s 间隔，torchrun/两 rank 全部 alive=True；最后样本早于 SIGTERM，**未覆盖退出时刻**。gate 之后一次残留扫描 residual_count=3，但无 pid/state/时间戳原件；restore 前复核（≈14:55:5x）三 PID 均已 gone。**精确退出时刻与 `/proc` state 无原件，D-state 不可断言**。
- **35 代码路径（1f780bf8）**：超时分支 951 调 cleanup_final；`child.wait` 返回 rc0 后，exception(981)/finally(998) 仅在 `child.poll() is None` 时再 cleanup → torchrun 已退出，**finally 跳过**；1022 的 recovery_gate 立即读取，仍见两 rank 存活 → `blocked_probe_residual`，**无 post-exit 等待/重扫**。
- **收窄后的归因**：本格不是 signal 集合不足（unknown/drift 空，两 PGID 均 killpg SIGTERM/SIGKILL），而是 **parent 先退出 + 固定等待结束后即时 gate**；退出延迟/僵尸-reap/采样时序仍可能，D-state 未证。
- **32 结论不变**：window restore 未发起；人工唯一 restore 按人工口径闭合但带 ppid 偏差与引用型证据缺口；archive rc1 不完整；H11 为 sync 挂起复现；`result.json api_returned=false` 仍按初始快照解释（8.121）。
- F1–F6 CPU 修复方案属 20 决策项；当前无产品/包改动、无设备、无新 GO。

## 8.123 bce9ce2 增量：9-check 当前复核、outer argv 语义、marker 12/12（2026-11-19，无 GO）

- **固定件**：31 commit **bce9ce2**，证据 SUMS **1aae8f57e27b4dc363279d889856cb438577483ffda5bc57e70a591bdbefba9e（38/38 OK、自哈希一致）**；相对 80c73d9 仅 README/SUMS 修改，21 个原 attempt2 原件未改写。
- **9-check/五 PID 缺口（增量关闭，性质=当前复核）**：`recheck-9checks-current-20261119T151015Z.json`（schema `h11-restore-recheck/1`，`sampled_at_utc=2026-11-19T15:09:22.210186+00:00`，note 明示 CURRENT-MOMENT、非恢复瞬间）含 `checks` **9/9 true**、`pids=[1355077,1355208,1355209,1355231,1355232]`、health/info/text/image 原始返回、root 身份（ppid 1355076、pgid/sid 1355077、starttime 401718579、boot）与 wrapper 1355076 cmdline。**不再仅有记录引用**；但仍非首次恢复瞬间历史原件。
- **outer restore argv（缺口收窄）**：wrapper `/proc/1355076/cmdline`（15:09:22Z 采样）为 `bash -c C=…; env -i PATH=… CLEF_LAUNCH_CONFIG=s1 setsid /bin/bash -c "cd … && source activate_env.sh && exec bash start.sh --model-dir … --output-dir …" > … 2>&1; echo restore_exec_rc=$? >> …`；剥离 shell 包装/重定向并重拼 inner -c 后 `head[0:7]` 与 `[7]` 均等于 plan `restore_argv`，**semantic match=true**。无独立历史启动脚本，不补造；ppid≠1 偏差保持。
- **marker 12 原件（已本地复核）**：MANIFEST 12 条；6 个 debug plog `.gz` 逐条 gunzip 后 SHA256/大小与 MANIFEST 全部一致，**12/12 OK**。重算命中行数/文件数与报告完全一致：`AicpuKfcRpcServer::Init addr` 2/2、`KFCtaskAndTiling` 4/4、`AivAicpuOpParam` 4/4、`RunAicpuKfcSrvLaunch` 8/6、`stream_id=47` 451/8、`task_id=` 761/8；`ReadValidMsg[`/`ReadValidMsg timeout 6s`/`Check msg equal failed`/`ReadAddrMsg`/`TwoStage`/`PostMsg`/`507046` 等均 0。
- **marker 解释边界**：可证两设备均出现 AICPU KFC RPC server init 与 KFC task/AivAicpuOpParam/RunAicpuKfcSrvLaunch 标记；**不能**据此证明完整 KFC message/read 路径或挂起根因（多个关键标记 0 命中，0 命中≠路径排除）。stream/task 活动跨度 14:49:05.110→14:52:06.287，**结束于 SIGTERM（14:52:59.6）前约 53 s**，最后约 53 s 无 marker 活动。
- **结论不变**：window restore 未发起；人工唯一 restore 按人工口径闭合（9/9 现有当前复核原件，ppid 偏差保留）；archive 仍 92 文件、SUMS 52d145ee、probe-temp 缺失、end rc1 **不完整**；H11 为 sync 挂起复现（API 已返回/handler rc0/sync 未返回），无数值、无 A/B；包 8ecf8f9/bed3a6e6 未改；无新 GO、无设备。
- 35 已获 CPU 候选实现授权（默认总 settle 30 s，不动远端/设备）；待其固定 commit 后我只读增量审代码/fixture。

## 8.124 settle 候选 7d371a1 代码审：Linux orphan fail-open（2026-11-19，无 GO）

- **固定件**：35 research commit **7d371a1**（仅 README/manifest/window/test 4 文件）；window **5bbda3cc…**（52,752 B，base **1f780bf8…**）、test **8336a663…**（51,814 B，base **228150e4…**）均逐字节核对；实际 diff window **+101/−10**、test **+234**（35 报 +111/−10，计数口径差 10 行）。
- **本地 CPU**：macOS `python3 -m unittest -q test_supervised_window_cpu` → **Ran 48 tests, OK (skipped=7)**（9.658 s），与 35 报一致。
- **设计符合项**：settle 每轮单一 `table_reader()` 快照（survivor 与 state 同表）、连续两空才 clean、默认 30 s 上限、逐 pid state/starttime（无 state=unknown）、超期 fail-closed、gate 复用同一 settle table、唯一 `begin_restore`、child timeout `ok=False` 保持；`_signal_targets`/`_attributed_targets`/`cleanup_attributed`/protected/unknown/drift/PID 复用规则未改。
- **阻断发现（fail-open 回退）**：`recovery_gate` L822-823 把 gate 的 `table`（无 settle 时来自 `scan()`）与 `reader=table.get` 传给 `attributed_survivors`（L836）；base 1f780bf8 是 `attributed_survivors(registry, child_pid)`，内部走 `_scan_direct()` 真实 `/proc`。Linux test `test_orphan_after_immediate_parent_exit_is_registered_and_cleaned` 将 `scan()` mock 为 `{}`，candidate 因此 `attributed=[]、residual=[]、gate allow=True`，与 31 Linux 48 rc1 的 `orphan-after-immediate-parent-exit (gate allow=True)` 完全对应。生产含义：无 settle 的 gate 失去第二次独立 `/proc` 读取，对 scan 竞态/缺失不再 fail-closed，属相对 base 的回退，不只测试问题。
- **最小修复**：仅当调用方显式传 `table`（settle 路径）时把 table/reader 传给 `attributed_survivors`；`table is None` 时保持 base（`table=scan()` 后 `attributed_survivors(registry, child_pid)` 内部 `_scan_direct`）；settle 路径不变。
- **其它审查项**：①默认 30 s 对 attempt2 退出延迟（gate 时 >1.4 s，最长可能约 174 s，精确未知）不保证覆盖，若 >30 s 仍 fail-closed，不能声称已修复该格；②manifest `files` 的 window/test SHA 已改为候选 5bbda3cc/8336a663，而 README/manifest 又称 frozen 1f780bf8/228150e4 不动——manifest-driven 校验会与 v2 包不一致，建议 files 保持 frozen 或明确候选分支；③31 另有 2 个 CPU 测试残留待精确身份清理。
- **结论**：7d371a1 **NOT READY**；等 35 修复后的新固定 commit + 31 macOS/Linux 48 复跑。无设备、无新 GO、不改 v2 包。

## 9. 本地核对副本

本机 `/tmp/clef32-audit/` 保留 ebca 71 原件、rstd8 window/probe/timing 与 cell/timing bin（diag/rstd）、AD window/probe、cc34 双 manifest 与 entry；均与远端逐 SHA 对齐（见各节哈希）。远端原件路径如各节所示。
