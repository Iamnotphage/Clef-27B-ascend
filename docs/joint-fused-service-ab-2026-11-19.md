# 服务 A/B runner（head fused-attention，2026-11-19）

执行 owner：21。两臂同一已核 head postimage `aa3e1758`，仅 `CLEF_JOINT_FUSED_ATTN=0/1` 不同；真实权重、
真实服务、固定短/长文本与图文输入。24 只做 CPU 文件与测试，不执行服务。

## 身份

- 基座：head 已核 `89ebd3f02b17aef100d4fc3cae036138fa94b40e`（wrapper `700cf10d…e8473`，probe `fda7cc8d…3e15f3`）；runner 通过 `Path(win.__file__)` 核 wrapper SHA，不依赖 wrapper 常量。
- runner：`runtime/joint_fused_service_ab.py`，SHA-256 `8638224f51e56a4b89c16f5d1d29a386278f0a7dd7b12beaa320c0fa2a6fde7f`
- runner CPU 测试：`runtime/test_joint_fused_service_ab.py`，SHA `1c03c4cfb654bffddb106c5f19a027c9cffce03c45808278ee2212fb727682a9`，8/8
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
