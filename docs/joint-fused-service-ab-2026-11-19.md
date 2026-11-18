# 服务 A/B runner（head fused-attention，2026-11-19）

执行 owner：21。两臂同一已核 head postimage `aa3e1758`，仅 `CLEF_JOINT_FUSED_ATTN=0/1` 不同；真实权重、
真实服务、固定短/长文本与图文输入。24 只做 CPU 文件与测试，不执行服务。

## 身份

- 基座：head 已核 `89ebd3f02b17aef100d4fc3cae036138fa94b40e`（wrapper `700cf10d…e8473`，probe `fda7cc8d…3e15f3`）；runner 通过 `Path(win.__file__)` 核 wrapper SHA，不依赖 wrapper 常量。
- runner：`runtime/joint_fused_service_ab.py`，SHA-256 `15ff998ef9b9c5e68ff4bd1a02cb2f7e6f32fb532ee92f6b2ffcbb9b47a2367d`
- runner CPU 测试：`runtime/test_joint_fused_service_ab.py`，SHA `8a591cbe9a779f55b51915726977780554e08c7a11a2700ba9aea66d22048341`，8/8
- stop：`/tmp/d0cc/d0_stop.py`，SHA `d6d49d75012a0789d3bff3e7b522e9cd496844ca0ee0424608ab8e6b43a33886`

## 命令（新不存在 run-dir）

```bash
PY=/workspace/yangchen/venvs/clef-py312/bin/python
REPO=<本提交完整 runtime 部署>
RUN=/workspace/yangchen/logs/joint-fused-service-ab-$(date -u +%Y%m%dT%H%M%SZ)
$PY "$REPO/runtime/joint_fused_service_ab.py" --run-dir "$RUN" --repo "$REPO" \
  --tree /workspace/yangchen/work/src/vllm --python "$PY" \
  --stop-script /tmp/d0cc/d0_stop.py --rounds 4 --reps 5
```

## 固定输入

| 输入 | 端点 | 长度 | SHA-256 |
| --- | --- | ---: | --- |
| od-0105 | `/v1/systemone` | 800 B | `25b8acaf74a0d2b9fc210e08687e3a08c3d5fff82e6785c20611418e8cc72bf0` |
| od-1534 | `/v1/systemone` | 2,835 B | `568ea1583377dcc6044d3fcd4a099c3beb4cd0e118155f95b595bda72b1c76bf` |
| erqa-0000 | `/v1/decide` | 22,570 B | `f3229f5c120098ac4eebcc6f4025c28254e35f959fd01998cac5db7b8d209aab` |

runner 读取后重新计算 SHA，任一不符即中止；每请求记录 status/answer_type/valid/ms。

## 流程与安全

- 初始服务按保存 root 身份精确 stop；preflight 校验 activate/launch/wait/stop SHA、head target 必须
  `f8101b98`，再安装 `aa3e1758`；两臂同 postimage。
- 4 轮 ABBA：`[0,1],[1,0],[1,0],[0,1]`；每臂唯一 `d0-ab-…` label，run 固定输入 `--reps` 次，
  再精确 stop；每轮/每臂写 `ab.json` 与 raw 日志。
- 启动后半途失败：只有本 label `api.pid` 绑定且 start/boot 身份匹配的 root 才可 exact stop；来源未知 listener 不发信号，记录后阻断恢复。
- finally：以实际 target SHA + revert 成功 + 保存 root/后代/PGID 全退出守卫；任一不成立不启动恢复；恢复唯一 `d0-restore-…` label、
  `CLEF_LAUNCH_CONFIG=s1`，`d0_restore.discover/verify` 四检（health/2 worker/text/image）。
- 每请求即时写 raw JSONL（bodySHA/start/end/response/error），warmup 与正式样本分列；每臂完成即 append/flush `ab.json`。
- 每臂固定 `CLEF_LAUNCH_CONFIG=s1`，记录实际 root env flag/source SHA/npu-smi 设备原件。
- 候选 env 仅在两臂启动/probe child；生产恢复不继承候选键。
- 任一必要步骤失败整体非 0；`ab.json` 分列 rounds/steps/error。

## CPU 测试范围

8/8：ABBA 顺序、≥4 轮、固定输入 hash、旧 run-dir 拒绝、stop 失败、输入 hash 不符拒绝。仅局部 CPU；
真实服务启动/性能/质量由 21 现场执行、20 判定；本文件不报告服务收益。

## 服务启动 guard 适配（精确 aa3e）

启动阻断：`Clef-27B/inference.py` 在启动时调用
`runtime/apply_joint_single_field_patch.py --check`，原 guard 只认 preimage 197863 和 single-field
f810；head 候选 postimage aa3e 被判 unknown。

- adapter installer：`runtime/apply_joint_service_check.py`，SHA
  `1ed35f0a24f8b23d8ac099992d703aca20d7f1460c89ff02c30635dfde6a219c`
- 服务端 postimage：`runtime/joint_service_check/apply_joint_single_field_patch.py`，SHA
  `36eaff440fd58063a3311b2efabfd1604b4a284cd24f3b7ac3cc4e7fe177fcc7`
- preimage guard：`76595657…`；adapter 只在 **`--check`** 且 env `CLEF_JOINT_ALLOWED_POSTIMAGE_SHA`
  **精确等于** `aa3e1758…` 时把该 SHA 作为 single-field；apply/revert 仍保持原契约，未设/其它值原 guard 不变。
- AB runner 停服前要求 SERVICE_ROOT guard 精确为原 `76595657…`；head postimage 安装后 apply adapter；两臂服务 env 仅注入该变量；restore env 移除；以实际 guard SHA 决定 revert；
  finally 先 revert adapter 并核 preimage SHA，再恢复；adapter revert 失败则阻断 restore。

基线恢复单命令（target 必须 f810、端口 closed）：

```bash
$PY $REPO/runtime/joint_fused_service_ab.py --run-dir "$RUN" --repo "$REPO" \
  --tree /workspace/yangchen/work/src/vllm --restore-only
```

CPU 证据：`runtime/test_joint_service_check.py` 8/8（aa3e 无 env 被拒、精确 env 通过、其它值拒绝；apply/revert 不接受 aa3e）；
AB runner 8/8。adapter 源码 diff 即上述两个常量与 `allowed_postimage` 判定，无其它行为变化。
