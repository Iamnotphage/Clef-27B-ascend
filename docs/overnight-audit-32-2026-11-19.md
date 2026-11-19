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
- 候选执行（20 裁决后）：OUT `owner31-candidate-s1-20261119T0059Z`、runner `61355d60…`（v3，仅 STAGE 改为 00e3bdb8）。**deploy 成功并已核**：complete ok=true、label `d0-cand-20261119T010007Z-ead542f1`、root 1260810、source aa3e、guard 36eaff44；restore.json 五 PID [1260810,1260958,1260959,1261007,1261008] 同 PGID、stable=true、四检 rc0/ok；live env OPP 9/`7b79c5db…`、LD 12/`ca27dbb7…`、FLAG=1、ALLOW=aa3e、CLEF_LAUNCH_CONFIG=s1，joint 键仅 FLAG/ALLOW/SINGLE_FIELD。v5 三轮运行中。
- 安全披露（34，已独立佐证）：00:57:51Z 其补测初版误用真实 installer 将生产 head f810→aa3e，随即以已核 installer revert（报告 SHA `f7a2e6d1…`，postimage→preimage，ready=true，备份 preimage=`f8101b98…`）。独立佐证：候选 deploy 的 `source-deploy-aa3e.json`（01:00:13Z）`state_before=preimage`、`sha256_before=f8101b98…`，证明回退在正规候选切换前已生效；S1 基线测量 00:22–00:35 早于事件，无污染；现 head=aa3e/guard=36eaff44 属候选窗口预期态。34 已承诺窗口内不再触碰生产 tree/guard。
- 34 rev4.1 补测（audit-only stage `91314b3…`）独立复跑：unit `53d984ec…` + integration `7d209b26…` + manifest 共 **30 项 OK（1 skip）**；测试在 `/tmp/clef-real-*` 临时 tree/guard 运行，我前后实读生产 tree/guard 均保持 aa3e/36eaff44 未变（未执行 installer）。

## 7. 环境事实（最终验收清单）

- clean 两键父环境恢复（ebca `ed140cf3`、final phase `ca972eaf`）：OPP 9 / LD 12，值 SHA `7b79c5db…` / `ca27dbb7…`，与 expected 12933 B/`fcbdcbf1…` 逐字节一致。
- 5c 窗口 restore 继承 prod 环境（31 实测 OPP 23 / LD 28；rstd8 window 的 env-prod-selected 已含 OPP 9/LD 12 重复链），**不得用于最终验收**；最终验收必须走 phase first clean 切换并保持严格 env 门。

## 8. 未闭合项

- Grouped29 v4r1：已按 240 s 单次执行并超时，无 candidate raw；结论为设备侧 final_sync 等待（见第 4 节），不重试、不采用。
- Grouped 后第一次 clean 恢复：`/workspace/yangchen/logs/owner31-grouped-recover-20261119T0044Z/restore-run`，label `d0-restore-20261119T004611Z-cec60d1c`、restore.json `4e022bce…`、四检 ok、envcompare match（OPP 9/`7b79c5db…`、LD 12/`ca27dbb7…`）。
- 候选 deploy 两次失败后的第二次 clean 恢复：`/workspace/yangchen/logs/owner31-candidate-recover-20261119T0052Z/restore-run`，driver rc0、label `d0-restore-20261119T005148Z-c4a4a6af`、五 PID [1258713,1258853,1258854,1258880,1258881] 同 PGID、四检 rc0/ok、envcompare match=true；restore.json **`be55b18c…`**（后续候选 current_restore）、complete `67b9511a…`、envcompare `c9a73e77…`、inventory `ea47954e…`；live clean env、source f810、guard 76595657。旧 `4e022bce…` 对应服务已不存在。
- S2 b8/b16 完整扫描、ERQA 400 与最终最佳服务恢复——待 31 执行。
- cc34 candidate 正式 S1 三轮与 finally restore——**暂停**：等 34 修复 entry 的 state/prod_env 构造契约并出新 stage/entry SHA；重跑绑定 `be55b18c…` restore + prior `1da03a19…`，之后核 3×2352 raw/质量/P95 对新基线 156.4051。

## 9. 本地核对副本

本机 `/tmp/clef32-audit/` 保留 ebca 71 原件、rstd8 window/probe/timing 与 cell/timing bin（diag/rstd）、AD window/probe、cc34 双 manifest 与 entry；均与远端逐 SHA 对齐（见各节哈希）。远端原件路径如各节所示。
