# 2026-11-19 最终验收执行组合（cc34 冻结准备）

状态：**CPU 准备完成，设备未执行**。本文件、[`runtime/final_acceptance_combo.json`](../runtime/final_acceptance_combo.json) 与候选部署入口构成同一交付。生产 f810 组合全部引用既有冻结原件，原始字节未改动；候选组合只在独立 namespace 新增最小入口，用于 aa3e ON 的同协议正式验收准备。winner 不由本文件决定。

角色：20 指挥与排程；31 唯一 NPU/服务 owner；32 独立原件与有限核审计；34 仅 CPU 实现、只读 SSH 与独立 staging。所有设备动作仍由 31 在空窗执行。

## 1. 冻结身份

| 对象 | 路径 | SHA-256 |
| --- | --- | --- |
| 验收 runner（cc7） | `/workspace/yangchen/work/final-acceptance-cc7f32842a78786a8f0e5da373f66597752ea9ab/runtime/final_acceptance.py` | `7d32e5c3e7977efa7222d527792887e50b541349d3ef6390663470180e011cab` |
| 验收 manifest（cc7，f810） | 同目录 `runtime/final_acceptance_manifest.json` | `0338f87b326a1b375dbb757b71acfa1d4bf1b037acb77a70dbbe176b7b5786b2` |
| 客户端监督器 v5 | `/workspace/yangchen/work/final-owner27-063109a4b1c8b2c3611b5850d7f1915046fd10df/final_acceptance_supervisor.py` | `172c283827d75074bf339fd5ba6326006d2b3755dd6b9c5c31d3f030d6f31876` |
| v5 test | 同目录 `test_final_acceptance_supervisor.py` | `044a4ef7033a8a2d7929aa3ea3413a0b1f0b1883274a18ba3cd70fada82e3ff0` |
| f810 phase v3 | `/workspace/yangchen/work/final-phase-owner27-d4464ac2c32fc122a9c6a391a45806cdaac17916/final_service_phase.py` | `888a8d7fac79bd243fe894c25324aca0f498e311cab91be77df1dc2cfdc0de44` |
| phase v3 test | 同目录 `test_final_service_phase.py` | `901d50e3226d418b91d4abe801ab5d6e254d24ee9775e1e07dbbda3c2ac22bd1` |
| AB/source-control base | `/workspace/yangchen/work/scratch-24/joint-fused-source-control-ebca769/runtime/joint_fused_service_ab.py` | `127e47aec47b30c461de69f37355c8149a045ae32127f7a2afde39d7a3caa648` |
| fusion installer | 同目录 `apply_joint_fused_attention.py` | `b09cd6025b0aa63ae0410781f3c611e9d512f7ef939cb78dab41b8413a19d406` |
| guardadapter installer | 同目录 `apply_joint_service_check.py` | `1ed35f0a24f8b23d8ac099992d703aca20d7f1460c89ff02c30635dfde6a219c` |
| cleanup helper | `/workspace/yangchen/work/scratch-24/joint-fused-5c045eb/runtime/joint_fused_attention_window.py` | `9245e1a6504742dc4706eb466f8053cb3999ff3d80abfac960ed218a37a747d5` |
| 原精确 stop | `/tmp/d0cc/d0_stop.py` | `d6d49d75012a0789d3bff3e7b522e9cd496844ca0ee0424608ab8e6b43a33886`（不修改） |
| 原 launcher / wait / activate | `perf-wave2-20261118/launch.sh` / `wait.sh`；`Clef-27B-ascend/scripts/activate_env.sh` | `a7373861…8321` / `e3fb0888…ffa0` / `3b363394…044a` |
| clean expected | `/workspace/yangchen/logs/perf-wave2-20261118/s1-final-r1/launch-manifest.json` | `fcbdcbf11521e1ae499c1ac5329eed990d63f3e8a75a99b5fbb6b4df1dcb0230`（12,933 B） |
| 生产 head | `/workspace/yangchen/work/src/vllm/vllm/model_executor/models/clef_joint_schema.py` | `f8101b98228bbd472a63a1efe6827583e35c6cc9dede4379b925f2b016670863` |
| 原 guard | `/workspace/yangchen/Clef-27B-ascend/runtime/apply_joint_single_field_patch.py` | `76595657d312699b4b1564d1bc054229808cd775c16702fb0e36fcdb1f9706d9` |
| 候选 head | 同一生产 head 的 aa3e postimage | `aa3e17585f8bd60a4d05e342abc7e2773c79a0b2d49f0370d15e29bcc6b72503` |
| 候选 guard post | 同一 guard 的 adapter postimage | `36eaff440fd58063a3311b2efabfd1604b4a284cd24f3b7ac3cc4e7fe177fcc7` |
| 候选 manifest（本提交派生） | `runtime/final_acceptance_candidate_manifest.json` | `31202d27b49c43279ecbdf71bb2881930ab3aaaa1729fa30361f65d0e19dd2a4`（77,763 B） |

候选 manifest 由 [`runtime/candidate_manifest.py`](../runtime/candidate_manifest.py)（SHA `6348ab52…c2b5`）从 cc7 的 `0338f87b…` 逐字节派生，白名单校验 13 个叶节点，见 [`runtime/final_acceptance_candidate_manifest.diff.json`](../runtime/final_acceptance_candidate_manifest.diff.json)（SHA `646ba8d8…a9ab`）。其余字段（数据、分母、作者 scorer、2pp 门、runner、argv、effective、OPP/LD）逐字节不变。

本提交新增文件 SHA-256：

| 文件 | SHA-256 |
| --- | --- |
| `runtime/candidate_service_phase.py` | `a1ca26cf34cd2ca0ff47787d10e06bf6e395b794882c59c938691b05d62e1d44` |
| `runtime/candidate_manifest.py` | `6348ab527ce03e025263e23f6745a304351e8ab72e34ce04d4af3f43f256c2b5` |
| `runtime/test_candidate_service_phase.py` | `53d984ec043518e50bde24f60a55446b0a4807703feb7e0c6b91dbae40c16d96` |
| `runtime/test_candidate_service_integration.py` | `7d209b26d271651c884c93ff60ceb4c4f456d9a0f6d02916901df0e4bc773a64` |
| `runtime/test_candidate_manifest.py` | `8ba9cdb5a2f96f7253fcc0a09fc833428b8b61c9510beba56951108fac5b6d5a` |
| `runtime/final_acceptance_candidate_manifest.json` | `31202d27b49c43279ecbdf71bb2881930ab3aaaa1729fa30361f65d0e19dd2a4` |
| `runtime/final_acceptance_candidate_manifest.diff.json` | `646ba8d89a07c9a647bea1a6429b617d22b409fb6aa9ca9f68b1fc37c6c9a9ab` |
| `runtime/verify_final_acceptance_combo.py` | `3f367b8844b036a7531aa2f4e9dad0f39328a46058d039deabe4bc675ee26231` |

一次自检：`python3 "$CLEF_CC34_CODE/runtime/verify_final_acceptance_combo.py" --stage "$CLEF_CC34_CODE" --ssh-host clef --json`，本地 prep 与远端 pin 全部 ok 才继续。

## 2. 现有 f810 正式全 S1（≥3 轮）与四 phase 组合

原 f810 组合无需任何新代码，全部使用上表已审原件。**`--single-baseline-round` 只产生单轮 `final_s1_complete=false`，不是正式全 S1，也不得用于候选。**

### 2.1 服务切换（每 phase 一次，默认 plan-only）

首格（S1→S1，无上一客户端）：

```bash
/workspace/yangchen/venvs/clef-py312/bin/python \
  /workspace/yangchen/work/final-phase-owner27-d4464ac2c32fc122a9c6a391a45806cdaac17916/final_service_phase.py \
  --code /workspace/yangchen/work/final-acceptance-cc7f32842a78786a8f0e5da373f66597752ea9ab \
  --manifest /workspace/yangchen/work/final-acceptance-cc7f32842a78786a8f0e5da373f66597752ea9ab/runtime/final_acceptance_manifest.json \
  --manifest-sha 0338f87b326a1b375dbb757b71acfa1d4bf1b037acb77a70dbbe176b7b5786b2 \
  --runner-sha 7d32e5c3e7977efa7222d527792887e50b541349d3ef6390663470180e011cab \
  --current-restore "${CLEF_CURRENT_RESTORE:?仅31现场填写已核四检restore原件}" \
  --current-restore-sha "${CLEF_CURRENT_RESTORE_SHA:?仅31现场填写fresh SHA256}" \
  --run-dir "${CLEF_PHASE_RUN:?全新phase输出目录}" \
  --from-phase s1 --phase s1 --first-phase-no-client
```

后续格把末两参换为 `--prior-client-record "${CLEF_PRIOR_SUPERVISION:?上一客户端supervision.json}" --from-phase <当前> --phase <目标>`。31 在全部门闭合后追加 `--execute`。成功输出 `complete.json`，其 `identities`、`launch_manifest`、`owner_evidence` 分别指向 `ready-identities.json`、该 label 的 `launch-manifest.json`、`restore.json`，直接作为下一段客户端监督的输入。

### 2.2 完整 S1 客户端（3 轮，正式）

```bash
/workspace/yangchen/venvs/clef-py312/bin/python \
  /workspace/yangchen/work/final-owner27-063109a4b1c8b2c3611b5850d7f1915046fd10df/final_acceptance_supervisor.py \
  --code /workspace/yangchen/work/final-acceptance-cc7f32842a78786a8f0e5da373f66597752ea9ab \
  --manifest /workspace/yangchen/work/final-acceptance-cc7f32842a78786a8f0e5da373f66597752ea9ab/runtime/final_acceptance_manifest.json \
  --manifest-sha 0338f87b326a1b375dbb757b71acfa1d4bf1b037acb77a70dbbe176b7b5786b2 \
  --phase s1 \
  --output-dir "${CLEF_S1_OUT:?全新评测输出目录}" \
  --records-dir "${CLEF_S1_RECORDS:?全新监督输出目录}" \
  --identities "${CLEF_READY_IDS:?phase complete.json identities}" \
  --launch-manifest "${CLEF_READY_LAUNCH:?该label launch-manifest.json}" \
  --owner-evidence "${CLEF_OWNER_EVIDENCE:?phase restore.json}" \
  --deadline-utc "${CLEF_S1_DEADLINE_UTC:?含时区绝对截止}" \
  --helper /workspace/yangchen/work/scratch-24/joint-fused-5c045eb/runtime/joint_fused_attention_window.py \
  --execute
```

S2/ERQA 客户端完全相同，只把 `--phase` 换成 `s2_b8`/`s2_b16`/`erqa` 并使用各自 fresh 目录、服务证据与截止；`--single-baseline-round` 不得出现。每轮 `c1-r{n}.json`、`.jsonl`、`-quality.json` 与最终 `acceptance.json` 全部保留；S1 默认 3 轮门、`p95_median_ms`/范围、OD 2pp 与 ERQA 400/219±8 门均在 manifest 内，不改。

### 2.3 phase 顺序与输出接线

`S1 → S2 B8(C8,C16) → S2 B16(C32,C64) → ERQA → S1 恢复`。每次客户端完成并确认 `clients_exited=true` 后，才进入下一格 phase 切换；下一格以本格 `supervision.json` 作为 prior 记录，以下一格 `complete.json` 的三件作为新服务证据。最终恢复后的服务身份、argv、env 由下一次 phase 或候选 restore 的检查再次证明。

## 3. candidate aa3e ON 同协议组合（本提交新增，独立 namespace）

### 3.1 最小差异

候选 manifest 相对 cc7 f810 manifest 只允许 13 个叶节点变化：head 与 guard 的 `source_files[].sha256`、`source_selection.production_head_sha/guard_sha`、`forbidden_environment`（移除 `CLEF_JOINT_FUSED_ATTN`），以及四个 phase 的 `service_environment` 各新增 `CLEF_JOINT_FUSED_ATTN=1` 与 `CLEF_JOINT_ALLOWED_POSTIMAGE_SHA=aa3e…`。argv/effective/数据/分母/scorer/门/OPP/LD 全部不变。

### 3.2.1 `--repo` 必填值与停前自检（32 finding 更正）

`--repo` 必须是 **bundle 根** `/workspace/yangchen/work/scratch-24/joint-fused-source-control-ebca769`。entry 的 `stage_candidate()`/`stage_d0()` 从 `repo/runtime` 复制 fusion installer、guardadapter、aa3e postimage、guard postimage 与 d0 六件；`joint-fused-5c045eb` 只用于 `--helper joint_fused_attention_window.py`（`9245e1a6…`），不能作 `--repo`。

部署前先做停前自检（任一行不匹配即不执行）：

```bash
B=/workspace/yangchen/work/scratch-24/joint-fused-source-control-ebca769
sha256sum "$B/runtime/apply_joint_fused_attention.py" \
          "$B/runtime/apply_joint_service_check.py" \
          "$B/runtime/joint_fused_attention/clef_joint_schema.py" \
          "$B/runtime/joint_service_check/apply_joint_single_field_patch.py"
# 期望：b09cd602… / 1ed35f0a… / aa3e1758… / 36eaff44…
```

该错误发生在 `stage_d0/stage_candidate` 阶段、早于停服，不会发信号或改 source/guard；但必须按上述值调用，避免重复空跑。

### 3.2.2 已修复：`SourceControl.state` 初始化（31 实跑 finding）

31 的 v2 deploy 在停服+apply adapter 后调用 `switch_source` 时报 `AttributeError: 'SourceControl' object has no attribute 'state'`。根因：ebca `main()` 在使用前给 `ab.state` 赋值，库式调用不会。最小修复：`build_deps()` 构造 `SourceControl` 后立即写入 `initial_state()`（schema/at/args/rounds/steps/source_switches/overall_rc），不改变其它行为；新增回归测试覆盖该初始化与首次 `switch_source`。修复前 recovery 也因 `restore_prod` 写 `state` 失败而停止在“服务已停、source f810、guard pre、端口关闭、卡7空”；31 须先用既有已核 clean restore 恢复服务，再做新 deploy。

### 3.2 部署（每个候选 phase 一次，默认 plan-only）

```bash
/workspace/yangchen/venvs/clef-py312/bin/python "${CLEF_CC34_CODE:?cc34独立stage}/runtime/candidate_service_phase.py" \
  --action deploy \
  --code /workspace/yangchen/work/final-acceptance-cc7f32842a78786a8f0e5da373f66597752ea9ab \
  --owner-code /workspace/yangchen/work/final-phase-owner27-d4464ac2c32fc122a9c6a391a45806cdaac17916 \
  --ab-code /workspace/yangchen/work/scratch-24/joint-fused-source-control-ebca769/runtime \
  --repo /workspace/yangchen/work/scratch-24/joint-fused-source-control-ebca769 \
  --manifest "${CLEF_CC34_CODE}/runtime/final_acceptance_candidate_manifest.json" \
  --manifest-sha 31202d27b49c43279ecbdf71bb2881930ab3aaaa1729fa30361f65d0e19dd2a4 \
  --base-manifest /workspace/yangchen/work/final-acceptance-cc7f32842a78786a8f0e5da373f66597752ea9ab/runtime/final_acceptance_manifest.json \
  --run-dir "${CLEF_CAND_RUN:?全新deploy输出目录}" \
  --phase s1 --from-phase s1 \
  --current-restore "${CLEF_CURRENT_RESTORE:?当前f810四检restore原件}" \
  --current-restore-sha "${CLEF_CURRENT_RESTORE_SHA:?fresh SHA256}" \
  --first-phase-no-client
```

- 后续候选 phase 用 `--prior-client-record "${CLEF_PRIOR_SUPERVISION}" --from-phase <上一候选phase> --phase <目标>`，不再用 `--first-phase-no-client`。
- 31 在门闭合后追加 `--execute`。成功输出 `complete.json`（action=deploy，phase/source/guard/flag env、`identities`、`launch_manifest`、`owner_evidence`）。
- 客户端：完全复用 2.2 的 v5 命令，只把 `--manifest` 换成本候选 manifest、`--manifest-sha` 换成 `31202d…`，其余 fresh 目录/证据/截止不变。**3 轮 S1 默认门保持。**

### 3.3 恢复（最后一个候选 phase 之后一次）

```bash
/workspace/yangchen/venvs/clef-py312/bin/python "${CLEF_CC34_CODE}/runtime/candidate_service_phase.py" \
  --action restore \
  --code <cc7 stage> --owner-code <d4464ac stage> \
  --ab-code /workspace/yangchen/work/scratch-24/joint-fused-source-control-ebca769/runtime \
  --repo /workspace/yangchen/work/scratch-24/joint-fused-source-control-ebca769 \
  --manifest "${CLEF_CC34_CODE}/runtime/final_acceptance_candidate_manifest.json" \
  --manifest-sha 31202d27b49c43279ecbdf71bb2881930ab3aaaa1729fa30361f65d0e19dd2a4 \
  --base-manifest <cc7 manifest> \
  --run-dir "${CLEF_CAND_RESTORE:?全新restore输出目录}" \
  --from-phase <最后一个候选phase> \
  --deploy-record "${CLEF_LAST_DEPLOY_COMPLETE:?最后一次deploy complete.json}" \
  --deploy-record-sha "${CLEF_LAST_DEPLOY_SHA:?fresh SHA256}" \
  --prior-client-record "${CLEF_LAST_SUPERVISION:?最后一个候选phase的supervision.json}"
```

成功后 `complete.json` 记录 f810/原 guard、`restored-identities.json`、launch manifest、`restore.json` 与 OPP/LD 逐字节比较；该服务即可作为后续 f810 phase 的 current restore（`--from-phase s1`）。

### 3.4 入口边界（不新增设备动作）

`candidate_service_phase.py` 只编排既有已审原语：原 `d0_stop.py`、原 launcher/wait、`apply_joint_service_check.py`、`apply_joint_fused_attention.py --check/--apply/--revert`（owned manifest）、d0 四检、cc7 `service_check`、phase v3 的 PGID/客户端/端口/NPU 门。没有新的 kill、pgrep、信号、source 写入或 guard 绕过。

安全顺序：停前核对旧 root/api.pid/argv/整 PGID 归属与无 pending client；停后旧树与整 PGID 非 Z 为空、端口 111、卡 7 空才继续；候选启动为 clean 两键父环境 + 仅两个候选变量；ready 后核对 head/guard/env/身份/health/info/四检/双 worker；任何停后失败进入显式 recovery gate（端口、客户端、已知 source/guard 全部闭合才回原），否则只记录 blocked，不发信号、不 fallback 启动。

## 4. 不变量

1. 原 `/tmp/d0cc/d0_stop.py` 字节与用法不变；不使用宽 pgrep 或旧 label。
2. 所有新服务使用全新 label，启动前断言目录不存在；api.pid 必须等于新 root。
3. 评测客户端只由 v5 监督器绑定进程组与绝对截止；`clients_exited=true` 前不切换服务。
4. `--single-baseline-round` 仅 f810 单轮；正式 S1 至少 3 轮，candidate 不接受单轮替代。
5. candidate manifest 只允许第 3.1 节的 13 处差异；生成器与 CPU 测试会拒绝任何其它漂移。
6. 数据 revision/SHA、2352/400 分母、作者 scorer、2pp 门、ERQA 219±8（0.02 绝对）不改。
7. 生产 f810 的 phase v3、v5、cc7 runner/manifest 原始字节未改；本提交只新增候选文件。
8. 远端 staging 只写 `/workspace/yangchen/work/final-acceptance-cc34-<commit>/`，不与 31 活动目录混写。

## 5. CPU 验证记录（本提交实际执行）

- `runtime/test_candidate_manifest.py`：8 项通过（含真实 cc7 基线时 8/8），验证派生确定性、候选 SHA `31202d…`、13 项白名单、非白名单/额外 env/错误 preimage 全部拒绝。
- `runtime/test_candidate_service_phase.py`：18 项通过，覆盖 plan-only 零变更、成功部署顺序与 clean env、停服失败不安装、启动失败回原、端口未知阻断、source 未知阻断、停前 PGID 未归属阻断、pending client 阻断、restore 成功/错误记录/phase 不匹配/停服失败不启动、adapter 失败不误启动、CLI 互斥门，以及 `build_deps` 必须为 ebca `SourceControl` 初始化 `state` 的回归。
- `runtime/candidate_manifest.py --check` 对真实 cc7 manifest 复算候选 SHA 一致。
- `runtime/test_candidate_service_integration.py`：4 项真实方法级 CPU 集成通过（需真实冻结 bundle fixtures）：成功 deploy（真实 source/guard installer 在临时 tree/guard 上运行）、实际 `main()` CLI deploy 成功、停后 switch 异常→真实 recovery→restore 成功、独立 restore 成功。CLI 测试显式 `--tree/--service-log-root` 指向临时夹具，并在结束后断言生产 f810 head 未变。仅替换 stop/launch/wait、/proc/NPU 扫描、HTTP 四检与 d0 服务发现/verify 等外部边界；真实 `SourceControl`/`ServiceAB` 的 `switch_source`/`apply_service_check`/`revert_service_check`/`restore_prod`/`verify_restored_env` 原样执行。
- 字段契约审计：本入口调用的真实方法只依赖 `state`（`source_switches`、`steps`）与 `prod_env`（`restore_prod` 保存/恢复，launch 使用显式 clean env）；`__init__` 已提供 args/run_dir/win、active_source/source_switches、service_check_applied、attribution_unknown、cleanup_unconfirmed。`build_deps()` 现已同时初始化 `state` 与 `prod_env`。
- 全部测试不 import torch/NPU、不访问生产服务或生产文件、不发模型请求；设备侧真实执行仍由 31 完成。

## 6. 待现场填写与剩余设备步骤

现场变量（每次全新，不复用历史）：current restore + SHA、deploy/restore run dir、identities/launch/owner 三件、prior supervision、绝对 UTC 截止、候选每个 phase 的 fresh 输出与监督目录。

设备顺序（31）：确认当前 f810 服务四检与严格 env 闭合 → 需要时先完成 f810 正式 S1≥3 轮 → 若 20/31 判定候选值得比较，按第 3 节部署候选并跑同协议 S1≥3 轮 → 候选胜出则继续 S2 四点与 ERQA 400 → 最后按 3.3 恢复 f810（或候选胜出时恢复候选）并闭合四检/严格 env/新身份。

质量与性能结论只来自实际全量原件；本文件不预先宣称 winner、S1/S2/ERQA 结果或采纳。
