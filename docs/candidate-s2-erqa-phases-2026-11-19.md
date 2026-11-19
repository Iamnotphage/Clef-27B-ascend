# 2026-11-19 candidate aa3e S2/ERQA typed phase 变体（CPU 准备，设备未执行）

状态：**本地 CPU 准备完成，待 32 独立核，31 唯一上传/执行**。本文件与
[`runtime/candidate_phase_stage.json`](../runtime/candidate_phase_stage.json)、
[`runtime/candidate_phase_variant.diff`](../runtime/candidate_phase_variant.diff)、
[`runtime/candidate_phase_cpu_evidence.json`](../runtime/candidate_phase_cpu_evidence.json) 构成同一交付。
生产 f810 四 phase 组合与当前 S1 候选阶段（`00e3bdb8…`，entry `a1ca26cf…`）全部保持原字节，不重跑、不改写。

角色：20 指挥与排程；31 唯一 NPU/服务 owner；32 独立原件与有限核；35 本变体纯 CPU 实现与本地证据。
所有设备动作仍由 31 在空窗执行；本文件不宣称 winner，也不主张采用。

## 1. 范围与不变量

- 只新增 S2/ERQA typed 入口；**不修改** `runtime/candidate_service_phase.py`（a1ca）与
  `runtime/final_acceptance_candidate_manifest.json`（31202d27）。
- 每个候选 phase 的固定闭环：`fresh f810 s1（四检、f810 head、guard pre）→ deploy 目标 phase →
  同 phase v5 正式客户端 → restore 回同一 clean f810 s1 → 下一 phase`。
  **不接受 candidate→candidate 直接转换**。
- 不新增扫描点、不改数据/分母/scorer/2pp 门：候选 manifest 与 frozen f810 manifest 仅差 13 个白名单叶节点
  （head/guard 绑定、`forbidden_environment`、四个 phase 的两个 env 键），argv/effective/config/
  concurrencies/rounds/launch_overrides 逐项相等。
- 原 `d0_stop.py`、launcher/wait、guard adapter、source installer、cc7 runner、v5 supervisor、
  phase v3 全部原字节复用；typed 入口只是编排既有原语。

## 2. 相对 a1ca 的最小 diff（`runtime/candidate_phase_typed.diff`）

新增 `runtime/candidate_service_phase_typed.py`，从 a1ca 派生；`build_deps` 首先要求同目录
`candidate_service_phase.py` SHA = `a1ca26cf…`，因此 a1ca 缺失或被改即阻断。相对 a1ca 只加以下门：

1. **phase 绑定**：`verify_phase_bindings()` 要求四 phase 集合一致、
   `phase.OVERRIDES` 与 manifest `launch_overrides` 逐项相等、`expected_argv`/`effective`/`config`/
   `concurrencies`/`rounds` 与 frozen base 逐项相等、每 phase env 仅多
   `CLEF_JOINT_FUSED_ATTN=1` 与 `CLEF_JOINT_ALLOWED_POSTIMAGE_SHA=aa3e…`、
   `forbidden_environment` 仅移除该 flag。
2. **deploy 起点**：`from_phase == s1`；启动前 `TARGET == f810`、`GUARD == guard-pre`；
   `current_restore` 必须 SHA 匹配、四检 `rc=0/ok=true`、label 与 `service_log_root/<label>/api.pid`
   等于该记录 root（兼容 d0 的 `label`/`service_label` 两种真实字段）。
3. **restore 绑定**：deploy record 必须 `action=deploy/ok/source=aa3e/guard=post`，
   `phase == from_phase`，`label` 精确，`manifest`+`manifest_sha` 等于本次入口参数，
   `root` 等于 ready identities 的 root；`prior-client-record` 必须是**本次 deploy 的最后客户端组**：
   `clients_exited=true`，argv 的 `--phase`、`--manifest`、`--identities`、`--launch-manifest`、
   `--owner-evidence` 与 deploy record 逐一相等，且 protected root 不在 `observed` 中。
4. **restore 收尾**：d0 `restore.json` 必须四检 `rc=0/ok=true`、`service_discovery.api_identity.pid`
   等于新 root；`restore_prod` 的 OPP/LD 比较必须 `match=true`；`complete.json` 输出
   `current_restore` + `current_restore_sha` 作为下一 phase 的 fresh f810 s1 记录。
5. 其余 stop/端口/整 PGID/NPU/clean env/finally recovery 逻辑与 a1ca 相同，未扩任何参数或配置。

deploy 旧服务的 prior client 允许来自之前完成的不同 phase，**不**与当前新 root 做强绑定；
其安全性由严格退出、全 client runner 为空、fresh current_restore 的 label/root 绑定共同保证。

## 3. 冻结身份与新增对象 SHA-256

| 对象 | 路径 | SHA-256 |
| --- | --- | --- |
| 原 S1 entry（不改） | `runtime/candidate_service_phase.py` | `a1ca26cf34cd2ca0ff47787d10e06bf6e395b794882c59c938691b05d62e1d44` |
| 原 S1 entry 测试（不改） | `runtime/test_candidate_service_phase.py` | `53d984ec043518e50bde24f60a55446b0a4807703feb7e0c6b91dbae40c16d96` |
| 候选 manifest（不改） | `runtime/final_acceptance_candidate_manifest.json` | `31202d27b49c43279ecbdf71bb2881930ab3aaaa1729fa30361f65d0e19dd2a4` |
| manifest 生成器 | `runtime/candidate_manifest.py` | `6348ab527ce03e025263e23f6745a304351e8ab72e34ce04d4af3f43f256c2b5` |
| manifest 测试 | `runtime/test_candidate_manifest.py` | `8ba9cdb5a2f96f7253fcc0a09fc833428b8b61c9510beba56951108fac5b6d5a` |
| **typed 入口（新增）** | `runtime/candidate_service_phase_typed.py` | `d93ee93463454db56673a3aefe261d1b598b52acd588dff67216c430370588d6` |
| typed 单元测试（新增） | `runtime/test_candidate_phase_typed.py` | `245fdc5cc693c49d5e3b19f996a97f4a8b2c04d8b2de842399e31ed86bb38e9a` |
| typed 真实方法集成（新增） | `runtime/test_candidate_phase_typed_integration.py` | `cfda281d694afc1d6f09c0b7bba4b7e6dd6045ce586e9de42faaa266d32ebb76` |
| 本地证据 harness（新增） | `runtime/run_typed_phase_evidence.py` | `25735efbf988a406f4df2555a25288112d2fb3b155c802bc8dcd91285785dd3c` |
| 本地 CPU 证据（新增） | `runtime/candidate_phase_cpu_evidence.json` | `4a74fee85d864c11b20ea2e42ad9b9504b6166a2857d57d17efcd67706140c9e` |
| 相对 a1ca 的 diff（新增） | `runtime/candidate_phase_variant.diff` | `cae4dfb5e75813dead8a01e6589d1422b0f433727a4260f566d87a8cd489781b` |
| stage 清单（新增） | `runtime/candidate_phase_stage.json` | 由提交后 SHA 记录 |
| 本文件（新增） | `docs/candidate-s2-erqa-phases-2026-11-19.md` | 由提交后 SHA 记录 |
| frozen f810 manifest | cc7 `runtime/final_acceptance_manifest.json` | `0338f87b326a1b375dbb757b71acfa1d4bf1b037acb77a70dbbe176b7b5786b2` |
| cc7 runner | cc7 `runtime/final_acceptance.py` | `7d32e5c3e7977efa7222d527792887e50b541349d3ef6390663470180e011cab` |
| v5 supervisor | owner27 d4464ac stage | `172c283827d75074bf339fd5ba6326006d2b3755dd6b9c5c31d3f030d6f31876` |
| phase v3 | 同 owner27 stage | `888a8d7fac79bd243fe894c25324aca0f498e311cab91be77df1dc2cfdc0de44` |
| f810 head | 生产 head 原字节 | `f8101b98228bbd472a63a1efe6827583e35c6cc9dede4379b925f2b016670863` |
| guard pre | 生产 guard 原字节 | `76595657d312699b4b1564d1bc054229808cd775c16702fb0e36fcdb1f9706d9` |
| aa3e head | 候选 postimage | `aa3e17585f8bd60a4d05e342abc7e2773c79a0b2d49f0370d15e29bcc6b72503` |
| guard post | 候选 adapter postimage | `36eaff440fd58063a3311b2efabfd1604b4a284cd24f3b7ac3cc4e7fe177fcc7` |
| clean expected | `s1-final-r1/launch-manifest.json` | `fcbdcbf11521e1ae499c1ac5329eed990d63f3e8a75a99b5fbb6b4df1dcb0230`（12,933 B） |
| AB bundle | ebca769 runtime | `127e47ae…` / `a1c502a3…` / `700cf10c…` / `b09cd602…` / `1ed35f0a…` |
| cleanup helper | 5c045eb `joint_fused_attention_window.py` | `9245e1a6504742dc4706eb466f8053cb3999ff3d80abfac960ed218a37a747d5` |

## 4. 执行模板（31 唯一执行；默认 plan-only，先审 plan 再 `--execute`）

变量每次全新：`CLEF_STAGE`（本提交在远端的独立 stage 根）、`CLEF_F810_RESTORE`（fresh f810/s1
四检记录，来自上一段 clean restore）、`CLEF_F810_RESTORE_SHA`、`CLEF_LAST_SUPERVISION`
（最近一次已完成客户端组的 `supervision.json`）、各 phase 的 fresh `CLEF_DEP_*` / `CLEF_OUT_*` /
`CLEF_REC_*` / `CLEF_RES_*` 目录、绝对 UTC 截止。

### 4.1 deploy（每个候选 phase 一次）

```bash
/workspace/yangchen/venvs/clef-py312/bin/python \
  "${CLEF_STAGE}/runtime/candidate_service_phase_typed.py" \
  --action deploy \
  --code /workspace/yangchen/work/final-acceptance-cc7f32842a78786a8f0e5da373f66597752ea9ab \
  --owner-code /workspace/yangchen/work/final-phase-owner27-d4464ac2c32fc122a9c6a391a45806cdaac17916 \
  --ab-code /workspace/yangchen/work/scratch-24/joint-fused-source-control-ebca769/runtime \
  --repo /workspace/yangchen/work/scratch-24/joint-fused-source-control-ebca769 \
  --manifest "${CLEF_STAGE}/runtime/final_acceptance_candidate_manifest.json" \
  --manifest-sha 31202d27b49c43279ecbdf71bb2881930ab3aaaa1729fa30361f65d0e19dd2a4 \
  --base-manifest /workspace/yangchen/work/final-acceptance-cc7f32842a78786a8f0e5da373f66597752ea9ab/runtime/final_acceptance_manifest.json \
  --run-dir "${CLEF_DEP_PHASE:?fresh deploy dir}" \
  --phase <s2_b8|s2_b16|erqa> --from-phase s1 \
  --current-restore "${CLEF_F810_RESTORE}" \
  --current-restore-sha "${CLEF_F810_RESTORE_SHA}" \
  --prior-client-record "${CLEF_LAST_SUPERVISION}"
# 审 plan.json/输出后同一 run-dir 追加 --execute（plan-only 不创建目录）
```

成功 `complete.json` 关键字段：`action=deploy, phase, from_phase=s1, label, root,
identities, launch_manifest, owner_evidence, manifest, manifest_sha, overrides`。

### 4.2 同 phase 正式客户端（v5，原协议不变）

与 `docs/final-acceptance-combo-2026-11-19.md` §2.2 完全相同，仅把
`--manifest` 换成 `${CLEF_STAGE}/runtime/final_acceptance_candidate_manifest.json`、
`--manifest-sha` 换成 `31202d27…`，`--phase` 用与 deploy 相同的 `<phase>`；
`--identities/--launch-manifest/--owner-evidence` 取该 phase `complete.json` 三件；
S2 扫描点 C8/C16 与 C32/C64、ERQA C1、轮数、截止、helper 均不变。

### 4.3 restore（每个候选 phase 之后一次）

```bash
/workspace/yangchen/venvs/clef-py312/bin/python \
  "${CLEF_STAGE}/runtime/candidate_service_phase_typed.py" \
  --action restore \
  --code <cc7 stage> --owner-code <d4464ac stage> \
  --ab-code /workspace/yangchen/work/scratch-24/joint-fused-source-control-ebca769/runtime \
  --repo /workspace/yangchen/work/scratch-24/joint-fused-source-control-ebca769 \
  --manifest "${CLEF_STAGE}/runtime/final_acceptance_candidate_manifest.json" \
  --manifest-sha 31202d27b49c43279ecbdf71bb2881930ab3aaaa1729fa30361f65d0e19dd2a4 \
  --base-manifest <cc7 manifest> \
  --run-dir "${CLEF_RES_PHASE:?fresh restore dir}" \
  --from-phase <与 deploy 相同 phase> \
  --deploy-record "${CLEF_DEP_PHASE}/complete.json" \
  --deploy-record-sha "${CLEF_DEP_PHASE_COMPLETE_SHA}" \
  --prior-client-record "${CLEF_REC_PHASE}/supervision.json"
```

成功 `complete.json` 关键字段：`action=restore, phase, label, source=f810, guard=pre,
restore_env.match=true, identities, launch_manifest, owner_evidence, current_restore,
current_restore_sha`。`current_restore` 即下一 phase 的 fresh f810/s1 四检记录。

### 4.4 顺序

1. 31 先按 20 排程跑原 f810 冻结四 phase（S2B8 → S2B16 → ERQA，最后 S1 恢复），不改任何原件。
2. 若 20 排定候选比较：`候选 S2B8 → v5 → restore`，`候选 S2B16 → v5 → restore`，
   `候选 ERQA → v5 → restore`；每次 restore 后的新 s1 记录作为下一 deploy 的 current_restore。
3. 任一 stage 失败/未知即停：typed 入口只在旧服务已确认停止且 source/guard 已知时 finally 恢复，
   否则仅记录 blocked，不发信号、不 fallback 启动。

## 5. CPU 证据（本机实际执行，全部通过）

命令（本地只读源 → mkdtemp 夹具；不 ssh、不触生产、不 import torch/NPU）：

```bash
python3 runtime/run_typed_phase_evidence.py \
  --bundle-source <ebca769>/runtime --owner-source runtime \
  --code-source <cc7 runner 所在的本地旧树> \
  --f810-source runtime/joint_single_field/clef_joint_schema.py \
  --guard-source runtime/apply_joint_single_field_patch.py \
  --clean-source Clef-27B/logs/overnight-20261119-owner27/grouped29-first-recovery/clean-env-expected.frozen.json \
  --records-dir /Users/chenclay/.ao/data/artifacts/clef-27b-ascend-32/typed-fixtures-20261119 \
  --out runtime/candidate_phase_cpu_evidence.json
```

结果（`candidate_phase_cpu_evidence.json`，SHA `4a74fee8…`）：

| 套件 | 结果 |
| --- | --- |
| 原 a1ca 单元测试 | 18/18 通过 |
| manifest 白名单/冻结 base | 8/8 通过（含真实 0338f87b） |
| typed 单元测试 | **58/58 通过，0 跳过**（含 6 项 32 真实记录 shape/绑定测试） |
| 原 a1ca 真实方法集成 | 4/4 通过（真实 SourceControl/AB + 临时 installer） |
| typed 真实方法集成 | 6/6 通过（含 S2B8/S2B16 exact argv、restore 交接、链式 ERQA、CLI、恢复） |

隔离保证（测试内强制）：所有 tree/guard/service-log-root 在测试 mkdtemp 内；fixture 源在 harness
mkdtemp 内；外部进程守卫拒绝 `ssh/scp/sftp/curl/npu-smi` 与任何 `/workspace`、`/tmp/d0cc` 路径；
`s2_*` 的 argv/并发/轮数直接取自真实 manifest。

32 真实记录（只读副本，已逐 SHA 校验；无任何请求/响应 raw）：
目录 `typed-fixtures-20261119`，`manifest.json` SHA `590ec04c…`、`SHA256SUMS` SHA `1242319d…`，
17 个受校验文件。`RealRecordShapeTest` 已实际运行并确认：

- `supervision.json`：schema/argv/`clients_exited`/`observed` 与实现一致；candidate S1 的 argv 与
  `cand-deploy-complete.json` 的 phase/identities/launch-manifest/owner-evidence 字符串完全一致；
- 历史 a1ca `complete.json` **没有** `manifest` 路径字段（typed 新增该字段），restore 绑定因此只接受
  typed deploy 记录；旧 S1 阶段继续用 a1ca 入口；
- d0 `restore.json`（deploy/finally/phase s2b8）的 `label`、`service_discovery.api_identity`、
  `api_pid`、四检 `rc/ok` 与实现一致；
- finally restore complete 的 OPP/LD compare `match=true`、counts 9/12、expected SHA
  `fcbdcb…`，且历史记录没有 `current_restore`（typed 新增交接字段）；
- phase s2b8 的冻结 v3 complete 不含 `source`/`manifest_sha`，不是 candidate deploy 记录。

实现按真实字段修正了 `label`/`service_label` 别名兼容；其余字段名与 32 README 契约一致。

## 6. 32 审核清单

1. `runtime/candidate_phase_variant.diff` 相对 a1ca 只含第 2 节 5 类门；无新增信号/停服/安装器/参数。
2. a1ca 与 31202d27 字节未变；typed `build_deps` 对 a1ca 的 SHA 门存在。
3. 四 phase 的 manifest/base/phase v3 三方逐项相等；仅两 env 键与 forbidden flag 差异。
4. restore 的 deploy record + prior supervision 绑定字段齐全，旧 S1 helper 的“仅 record 退出不绑 label”缺口已关闭。
5. `candidate_phase_cpu_evidence.json` 五套件全过；隔离守卫与 mkdtemp 约束符合要求。
6. 32 真实记录副本已逐 SHA 校验并驱动 `RealRecordShapeTest` 6 项实际运行；设备新产生的 typed
   `complete.json` 必须含 `manifest`、`current_restore`、`current_restore_sha`，31 首次执行后由 32 复核这三项。

## 7. 结论边界

- 当前 S1 候选最终为中性（中位 156.4417 ms vs f810 156.4051 ms，+0.0366 ms；三轮质量通过）。
- 本变体只用于 20 排定后的 S2/ERQA 同协议比较准备，不预设任何 S2/ERQA 结果，不宣 winner。
- 设备结果只来自 31 的实际全量原件；本文件所有未执行项均标注为待执行。
