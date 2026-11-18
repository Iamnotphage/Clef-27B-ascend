# Joint head 跨注意力内核化候选（2026-11-19）

状态：**CPU 实现/安装器/数学等价检查完成，未上板**。真实收益由 21 在同一窗口 ABBA 验证；本文不报告未实测收益。

## 1. 基线与闭题边界

- 远端工程 HEAD `3e83ae17df7b559e`；vLLM-Ascend `89eb8a5e`；21 服务分支 `0bd6f6a`（`perf/wave2-npu-wave3`）。
- 当前安装的 head postimage：`vllm/model_executor/models/clef_joint_schema.py`
  SHA-256 `f8101b98228bbd472a63a1efe6827583e35c6cc9dede4379b925f2b016670863`（23 只读确认与 preimage 一致）。
- T3「head/pooler 与前端去同步/序列化/简单向量化」已服务内打点关闭（`docs/perf-wave2-research.md` §3）：
  967 token 4 候选 pooler 5.90–6.25 ms、head 4.78–5.05 ms；CPU 卸载、跨请求批处理、tolist/D2H 均无收益。
- 本候选不重做上述路径；只改 **kernel 边界**：单字段路径的 6 处跨注意力（4 个 decoder 层 + 2 个
  evidence routing 层）由 `nn.MultiheadAttention` 改为 `npu_fusion_attention`，保留 q/k/v 与 out 投影。

## 2. 候选实现

新 postimage：`runtime/joint_fused_attention/clef_joint_schema.py`
SHA-256 `aa3e17585f8bd60a4d05e342abc7e2773c79a0b2d49f0370d15e29bcc6b72503`。

- 环境开关 `CLEF_JOINT_FUSED_ATTN`，默认 **0**；为 1 且 `CLEF_JOINT_SINGLE_FIELD=1` 时，6 处
  cross-attention 使用 `torch_npu.npu_fusion_attention(..., "BSH", scale=1/sqrt(64), keep_prob=1.0)`。
- 融合仅替换 attention 内部的 softmax + 两次 bmm；q/k/v、out_proj、LayerNorm、FFN、残差和最终打分不变。routing 层由 `single_field` 标记启用，Q>1 多字段仍回退。
- 多字段路径不启用该融合；`in_proj_weight` 缺失时回退原 MHA。
- 默认关闭时行为与 `f8101b98` 完全一致；安装器只换文件，不改变默认配置。

## 3. 收益预算与关闭条件

- 预算（假设，未实测）：6 个 attention 块（2 routing + 4 decoder），每块减少 2–3 次下发/中间读写；
  head 5 ms 中的 **0.3–0.8 ms/请求**；S1 967 p95 影响约 ≤0.5%。
- 关闭条件：head probe 三轮交错中 device 或 wall 中位收益 `<0.3 ms`，或方向不一致；关闭后不启服务 A/B。

## 4. 安装/回退

```bash
PY=/workspace/yangchen/venvs/clef-py312/bin/python
TREE=/workspace/yangchen/work/src/vllm
MAN=/workspace/yangchen/logs/joint-fused-attention-20261119
$PY runtime/apply_joint_fused_attention.py --tree "$TREE" --check
$PY runtime/apply_joint_fused_attention.py --tree "$TREE" --apply --manifest "$MAN/apply.json"
# 回退到 f8101b98
$PY runtime/apply_joint_fused_attention.py --tree "$TREE" --revert --manifest "$MAN/revert.json"
```

安装器只接受 preimage `f8101b98`/postimage `4e5c450f`，拒绝未知内容；`CLEF_JOINT_FUSED_ATTN` 默认 0。

## 5. CPU 检查（已完成）

- `python3 -I -m py_compile` 三个脚本通过。
- 数学等价测试（fake `torch_npu.npu_fusion_attention`，CPU torch，`TORCH_DEVICE_BACKEND_AUTOLOAD=0`）：
  合法 Q1+O3 schema（4 span）；`shape=True, close=True, max_abs=5.96e-08, on_calls=6`（2 routing+4 decoder），
  `off_calls=0`，`disabled_none=True`。该测试只证接线/数学，不证真实 NPU 精度或性能。
- 安装器在临时树对 `f8101b98` 做 check→apply→check→revert，最终恢复 `f8101b98`，全部 RC0。

## 6. 21 执行：head probe（第一道 A/B）

服务停止、单逻辑 NPU、三检后：

```bash
PY=/workspace/yangchen/venvs/clef-py312/bin/python
MAN=/workspace/yangchen/logs/joint-fused-attention-20261119
$PY runtime/joint_fused_attention_probe.py \
  --tree /workspace/yangchen/work/src/vllm --device 0 \
  --output "$MAN/head-probe.json"
```

- 输入：随机 BF16 `hidden[967,5120]`、`lexical[4,5120]`、合法 Q1+O4 的 5 span schema；probe 记录两输入 SHA-256。
- 设备先核接口：probe 在计时前对 Q=1 与 Q=4 各做一次 `npu_fusion_attention` 小调用，显式记录 `BSH`、
  `scale=1/sqrt(64)`、`keep_prob=1.0`、BF16、head_num/head_dim 与 finite。
- 两臂共用权重/输入/`CLEF_JOINT_SINGLE_FIELD=1`，仅 `CLEF_JOINT_FUSED_ATTN=0/1`；每臂 5 warmup +
  50 次逐调用 NPU event，报告 mean/median/p95 device 与 wall、`bit_equal/finite/max_abs`。
- 判读：4 轮 ABBA（A/B、B/A、B/A、A/B）；JSON 的每 round 每 arm 保存 `device_samples`、`wall_samples`
  与带 `index` 的 `samples` 数组，另有 stats。先看 interface.ok、candidate_error 为空、correctness finite；
  再看 per-round 配对中位与总中位是否 ≥0.3 ms 且同向。未过则关闭。

## 7. 21 执行：服务 A/B（probe 通过后）

固定输入：`od-1534`，canonical body 2,835 B，SHA-256
`568ea1583377dcc6044d3fcd4a099c3beb4cd0e118155f95b595bda72b1c76bf`。

```bash
# A 臂：CLEF_JOINT_FUSED_ATTN=0 启动 s1 服务（21 的标准 start.sh）
CLEF_JOINT_FUSED_ATTN=0 bash start.sh --model-dir /workspace/yangchen/models/Clef-27B \
  --output-dir <run-A> --config s1
$PY runtime/joint_fused_attention_client.py --output "$MAN/client-A.json" --iters 20
# 精确 PID 停止 A；B 臂同一命令改 CLEF_JOINT_FUSED_ATTN=1
CLEF_JOINT_FUSED_ATTN=1 bash start.sh --model-dir /workspace/yangchen/models/Clef-27B \
  --output-dir <run-B> --config s1
$PY runtime/joint_fused_attention_client.py --output "$MAN/client-B.json" --iters 20
```

- 客户端强制校验 `body_sha256`，保留每请求开始/结束、HTTP 状态、完整响应与 answer type；只有 200 且
  类型匹配才计入延迟。每臂 3 warmup + 20 次，ABBA 至少 3 轮。
- 服务级通过后再由 21 跑同窗口 S1 ≥3 轮、S2、完整 OpenDecisions+ERQA；质量仍按 ≤2pp，未通过不采用。
- 若 head probe 收益不足或服务 A/B 不同向，关闭候选并回退 `f8101b98`。

## 8. 未做与边界

- 24 未触 NPU/真实服务/安装/加载；未在真实设备上运行 probe 或客户端。
- 未改输入、参考、容差、协议、token/span 或多字段语义；未缓存答案。
- 未测质量；fused attention 的 BF16 数值差异需完整 OD/ERQA 2pp 判定。

## 9. 完整真实执行包装（21 单命令）

文件：`runtime/joint_fused_attention_window.py`
SHA-256 `069d16e08e2d34842613634c2e5208a8bc12c68b8c8758bef0e6ac4b7ead8157`。
CPU 故障分支测试：`runtime/test_joint_fused_window.py`（SHA `da928dda20c0b363e7caa38db6cfe8eb2fd5bfafc0984cc939654a07080aa9cd`），
8/8 通过（env 过滤、LISTEN/refused、stop 失败保留既有服务、apply/probe 失败 revert→restore）。

单次命令（21 执行；run-dir 必须不存在）：

```bash
PY=/workspace/yangchen/venvs/clef-py312/bin/python
REPO=<perf/wave4-joint-fused-20261119 检出>
RUN=/workspace/yangchen/logs/joint-fused-attention-$(date -u +%Y%m%dT%H%M%SZ)
$PY "$REPO/runtime/joint_fused_attention_window.py" \
  --run-dir "$RUN" --repo "$REPO" \
  --stop-script /tmp/d0cc/d0_stop.py \
  --expect-stop-sha <21 修好后的 stop SHA> \
  --tree /workspace/yangchen/work/src/vllm --device 0
```

- preflight：stage installer+postimage+probe+client+stop，逐文件 SHA；activate/launch/wait SHA；
  source activate 后 CPU loader 预检（libhccl + torch_npu ldd，无 torch_npu import）；target 必须
  `f8101b98`；LISTEN 用 socket（0=open、ECONNREFUSED=closed、其它 unknown 阻断）；保存 root
  start/boot/PGID/cmdline/descendants、`npu-smi info` 与**选定** env 键。
- stop：调用 21 的 `d0_stop.py --api-pid <saved root> --saved-identity <root-identity.json>`，非 0 立即阻断；
  wrapper 再验 saved 身份全退出且端口 closed。当前 stop `3c55ee5` 的正常退出误拒绝由 21 修，等新 SHA 再跑。
- apply → probe：probe 独立进程组、900s timeout；超时只 kill 该 PGID；raw 日志在 `probe/`。
- finally：已 apply 必 revert 并验 `f8101b98`；已 stop 必 restore（新 `d0-restore-…` label、
  `CLEF_LAUNCH_CONFIG=s1`、launch.sh/wait.sh、`d0_restore.discover/verify` 四检）；stop 失败且旧服务仍
  健康则保留旧服务，不再启动第二个。
- 每步 RC 与阶段 flush 在 `window.json`；test/probe/apply/revert/restore 分列，任一必要失败整体非 0。
- env：无独立备份文件声明；运行时从 `/proc/<root>/environ` 只保存选定性能/库键到
  `env-prod-selected.json`，与 activate clean env 比较；候选键不进入 restore。公开日志不展开无关变量。
