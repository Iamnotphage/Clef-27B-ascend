# 2026-11-19 官方入口 aa3e 默认集成（CPU 准备，设备未执行）

状态：**本地实现与真实 installer 临时夹具测试完成，待 31 唯一 stage/执行、32 独立核**。
20 已决定最终提交默认采用 **aa3eON**；证据口径为：S1 中性（156.4417 vs 156.4051 ms），
S2 四点单窗 +0.46%～0.65%，ERQA 同值（216/400），全部在 2PP 内；**不声称稳定显著收益**。
旧 765 / a1ca / manifest 31202d27 / phase v3 / v5 冻结资产零改；本轮指标 CSV 未改（由 31 统一收口）。

## 1. 基点

| 对象 | 变更前 | 变更后 |
| --- | --- | --- |
| `Clef-27B/inference.py` | `081b56235fca53f9787e1e13a5667167f62e2dd09cdee2309c4f0936f7e4dfc4` | `de5b0d1ea95194dcb7e039e65cb192c3fb6ad98e5c8217579040ecce309e7b29` |
| `Clef-27B/start.sh`（不改） | `d880a01ea1e4625388317b9cd617470fc11c11db8900fa390c82133e0ea46b02` | 同左 |

变更前入口按 `GDN decision → QKV norm → chunk128 → joint single-field(f810) → vllm serve` 顺序启动，
但 b09/aa3e/36/1ed 四个候选资产只存在于 scratch24 bundle，官方入口无法独立复现 aa3e。

## 2. 最小 diff（`runtime/official_entry.diff`，SHA `c40725a0…`）

`Clef-27B/inference.py`：`+226 / -5`（其余为新增文件）。

1. 常量：fused installer（b09）、独立 guard（36）、service-check installer（1ed）、原始/f810/aa3e/guard SHA、
   `DEFAULT_CONFIG="s1"`、FLAG/ALLOW 键名。
2. `CONFIGS["s1"]` 增加 `joint_fused_attention: True`（仅默认配置）；s2/custom 保持 f810 兼容路径。
3. `resolve_vllm_tree()`：同解释器子进程 `import vllm; print(vllm.__file__)`，短超时（60s）；
   校验 `<tree>/vllm/model_executor/models/clef_joint_schema.py` 为常规文件、target 与所在目录可写；
   rc≠0/空输出/非 `vllm/__init__.py`/zip/namespace/不可写 → `ValueError`（由 fused 分支转 fail-closed）。
4. `fused_joint_patch()`：置 FLAG/ALLOW → guard(36) check-first（`status_before` 为 preimage 才 apply，
   期望 `status_after=single-field`）→ fused(b09) check-first（`state_before` 为 preimage 才 apply，
   期望 `state_after=postimage`）；独立 backup；最终目标 SHA 必为 aa3e、guard 文件 SHA 必为 36；
   原子写 `joint-patch-record.json`（含 no-op/失败/当前 SHA/分支/默认配置名/两个 report/installer+SHA/
   backup/env/时间戳）；任何失败先记录再 `SystemExit`，**绝不启动 vLLM**。
5. `build_launch()`：默认配置走 fused 分支，f810 配置走原 `joint_single_field_patch()` 且显式 `pop` FLAG/ALLOW；
   fused 时 service env 增加 `CLEF_JOINT_FUSED_ATTN=1`、`CLEF_JOINT_ALLOWED_POSTIMAGE_SHA=aa3e`，保留
   `CLEF_JOINT_SINGLE_FIELD=1`；返回 manifest 增加 `joint_patch`。
6. `--config` 默认值改用 `DEFAULT_CONFIG`（s1）。

`runtime/test_inference_launch.py`：`+12 / -7`，仅修正 7ea8aa6 引入 `decision_enabled` 门后的过期测试契约
（HEAD 上同样失败，非功能缺陷）；运行时零改。

未改：GDN/QKV/chunk128 数值路径、模型/请求/评分/数据路径、f810 安装器调用、`start.sh`、指标 CSV。

## 3. 资产与来源（原字节入库）

| 提交路径 | SHA-256 | 来源 |
| --- | --- | --- |
| `runtime/apply_joint_fused_attention.py` | `b09cd6025b0aa63ae0410781f3c611e9d512f7ef939cb78dab41b8413a19d406` | ebca769 bundle，原字节 |
| `runtime/joint_fused_attention/clef_joint_schema.py` | `aa3e17585f8bd60a4d05e342abc7e2773c79a0b2d49f0370d15e29bcc6b72503` | 同上 |
| `runtime/apply_joint_single_field_patch_fused.py` | `36eaff440fd58063a3311b2efabfd1604b4a284cd24f3b7ac3cc4e7fe177fcc7` | 同上（独立文件名，SOURCE 解析到现有 f810） |
| `runtime/apply_joint_service_check.py` | `1ed35f0a24f8b23d8ac099992d703aca20d7f1460c89ff02c30635dfde6a219c` | 同上（审计/回退工具，不参与启动） |
| `runtime/joint_service_check/apply_joint_single_field_patch.py` | `36eaff440fd58063a3311b2efabfd1604b4a284cd24f3b7ac3cc4e7fe177fcc7` | 同上（1ed 的 SOURCE） |
| `runtime/joint_single_field/clef_joint_schema.py`（不改） | `f8101b98228bbd472a63a1efe6827583e35c6cc9dede4379b925f2b016670863` | 现有冻结资产 |
| `runtime/apply_joint_single_field_patch.py`（不改） | `76595657d312699b4b1564d1bc054229808cd775c16702fb0e36fcdb1f9706d9` | 现有冻结资产 |

197863d5 原始 preimage 只来自 32 的只读副本（`original-joint-schema-197863d5/`），**不进入提交树**，
仅用于三起点测试。

## 4. 启动语义

| 起点树 SHA | guard(36) | fused(b09) | record branch |
| --- | --- | --- | --- |
| `197863d5` | `status_before=preimage` → apply f810 | `state_before=preimage` → apply aa3e | `original->f810->fused` |
| `f810` | `status_before=single-field`，no-op | `state_before=preimage` → apply aa3e | `f810->fused` |
| `aa3e` | `single-field`，no-op | `state_before=postimage`，no-op | `aa3e-noop` |

- guard 的 ALLOW 豁免只在 `--check` 生效；apply 仅在该 installer 自己的 `status_before=preimage` 时调用。
- f810 vs aa3e 只由 fused `state_before` 区分；guard report 在 ALLOW 下把 aa3e 也报 single-field。
- guard apply 后 fused 仍 unknown = 真实失败，fail-closed，不重试、不回退。
- backup 分目录：`<output>/patch-backup/joint-single-field` 与 `.../joint-fused`（文件名相同，必须分开）。
- 每次启动（含 no-op）原子写 `<output>/joint-patch-record.json`；失败先写 `ok=false` 再非零退出。
- 重启幂等：197863d5 首启→aa3e；后续 aa3e 启动双 no-op，目标 SHA 恒 aa3e。
- f810 配置（s2/custom）显式清理 FLAG/ALLOW；aa3e 树遇 f810 配置在 guard check 阶段 unknown → fail-closed；
  新 output-dir 无 f810 备份，不自动跨运行回退（需同 backup-dir 显式 `--revert`）。

## 5. 官方命令

```bash
bash Clef-27B/start.sh --model-dir /path/to/Clef-27B --output-dir /path/to/output
```

- 默认 `--config s1`（aa3e）；`--dry-run` 只输出 manifest，不探树、不安装、不写 output。
- 成功启动前 output 下应有：`joint-patch-record.json`（ok=true/branch）、`patch-manifest-joint-*.json`、
  `patch-backup/*`、`launch-manifest.json`（含 `joint_patch` 与 env 两键）。
- 服务 env 精确为 `CLEF_JOINT_FUSED_ATTN=1`、`CLEF_JOINT_ALLOWED_POSTIMAGE_SHA=aa3e…`、
  `CLEF_JOINT_SINGLE_FIELD=1`（另加既有 GDN/QKV 键）；source=aa3e、guard 文件=36。

## 6. CPU 证据（本地实际执行）

命令：

```bash
python3 runtime/run_official_entry_evidence.py \
  --original "${CLEF_ORIGINAL_JOINT_SCHEMA:?32 original-joint-schema-197863d5 copy}" \
  --out runtime/official_entry_cpu_evidence.json
```

结果（`runtime/official_entry_cpu_evidence.json`，SHA `1cad8ba1…`）：

| 套件 | 结果 |
| --- | --- |
| `test_joint_fused_entry`（真实 installer，mkdtemp tree） | **11/11 通过，0 跳过** |
| `test_inference_launch`（入口回归，含 chunk128 契约修正） | **17/17 通过** |

覆盖：三起点（197863d5/f810/aa3e）、重复启动 no-op、unknown fail-closed（树未变、未启动）、
独立 backup、f810 配置 env 清理、dry-run 默认 s1 不探树/不写 output、失败不调用 `execv`、
probe 的 rc/空输出/非 vllm/缺目标/不可写校验、资产 SHA 与无 scratch 依赖。

## 7. Stage 清单（31 平铺到独立远端 stage 后逐 SHA 核）

| 文件 | SHA-256 |
| --- | --- |
| `Clef-27B/inference.py` | `de5b0d1e…e7b29` |
| `runtime/apply_joint_fused_attention.py` | `b09cd602…d406` |
| `runtime/joint_fused_attention/clef_joint_schema.py` | `aa3e1758…503` |
| `runtime/apply_joint_single_field_patch_fused.py` | `36eaff44…cc7` |
| `runtime/apply_joint_service_check.py` | `1ed35f0a…19c` |
| `runtime/joint_service_check/apply_joint_single_field_patch.py` | `36eaff44…cc7` |
| `runtime/test_joint_fused_entry.py` | `dcc545c5…6b1e` |
| `runtime/test_inference_launch.py` | `b4f18279…60ec` |
| `runtime/run_official_entry_evidence.py` | `385b6324…2bf5` |
| `runtime/official_entry_cpu_evidence.json` | `1cad8ba1…2c4d` |
| `runtime/official_entry.diff` | `c40725a0…ef3f` |
| `runtime/official_entry_stage.json` | 提交后记录 |
| 本文件 | 提交后记录 |

基础运行依赖（GDN/QKV/chunk128/schema-cache 安装器、`scripts/activate_env.sh`、
`Clef-27B/src/ascendc/vendors` 下的 vendor `.so`、`Clef-27B/logs/model_manifest.json`）
逐项 SHA 见 `runtime/official_entry_stage.json` 的 `base_runtime_dependencies`；全部为现有冻结字节。

冻结未改：`start.sh` `d880a01e…`、`runtime/apply_joint_single_field_patch.py` `76595657…`、
`runtime/joint_single_field/clef_joint_schema.py` `f8101b98…`、a1ca `a1ca26cf…`、manifest `31202d27…`、
phase v3 `888a8d7f…`、v5 `172c2838…`。

## 8. 设备检查（31 执行、32 独立核；CPU 侧不声称已执行）

1. 冷启动状态机：197863d5→guard apply→f810→fused apply→aa3e；f810→fused；aa3e→双 no-op；
   unknown/不可写 → fail-closed 且不启动 vLLM；核对每步 report 键名与 record。
2. 连续三次启动（首启→重启→再重启）：树 SHA 恒 aa3e；后两次 no-op 且有记录；无残留进程。
3. `bash Clef-27B/start.sh --model-dir … --output-dir …`：health ready、`/v1/decide/info` config=s1、
   端口 18143；五 PID 同 PGID、卡 7 chip0/1 各一 worker；四检 rc0/ok。
4. live env：OPP9 `7b79c5db…` / LD12 `ca27dbb7…` 与 `fcbdcbf1…` 逐字节；joint 三键精确；
   source=aa3e、guard=36；无 SMALLQ/其它候选键。
5. 回退边界：f810 配置遇 aa3e 树 fail-closed；f810 重启不携带 ALLOW；运行期不写 repo。
6. 指标只用官方 benchmark JSON 的 `decisions_per_second`（successful/wall_seconds）；全部标 aa3e/s1，
   不拼 f810 数字；journal span 仅作交叉核对。

## 9. 单默认单版本

最终提交默认仅 `s1/aa3e`。S1/S2/ERQA 指标必须全部来自该默认（aa3e）；f810 结果保留为对照，
不得把 candidate S2 与 f810 S1 混称同一版本。s2/custom 仅作为 f810 兼容/回退配置，不是提交默认。
