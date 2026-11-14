# ERQA baseline 证据索引（2026-11-14）

正式结果：官方 ERQA 400 题，**219/400 = 0.5475**，400/400 成功、0 失败、0 重试。服务 TP2 / BF16 / eager / 16384 / seqs1 / 缓存关，客户端并发 1，端点 `POST /v1/decide`。完整说明见 [docs/erqa-baseline-2026-11-14.md](../../../docs/erqa-baseline-2026-11-14.md)。

| 文件 | 内容 |
| --- | --- |
| `summary.json` | 总体、分项、延迟、token、吞吐、服务身份与数据清单 |
| `predictions.jsonl` | 400 题逐题：ID、类别、gold、预测、完整候选概率、图片/请求哈希、tokens、耗时 |
| `offline-score.json` | 独立离线重算 219/400，请求哈希 400/400 通过 |
| `metrics-breakdown.csv` | 类别、单/多图、候选数、`visual_indices` 形态分项 |
| `data-manifest.json` | 官方 TFRecord commit/大小/SHA-256、400 条、630 图、类别/答案/尺寸分布 |
| `token-budget.json` | 逐题 prompt 预算：prefix/media/state/schema/suffix 与总 tokens |
| `phase0-snapshot.json` | 本轮开始前 PID 424593 的 argv、cwd、环境白名单、进程树、NPU 状态、源码哈希与 8/8 冻结回放 |
| `service-launch-manifest.json` | ERQA 服务 PID 430985 的完整 argv、生效值与来源、模型 revision |
| `service-info.json` | `/v1/decide/info` 返回 |
| `smoke-summary.json`、`smoke-offline-score.json` | 19 题 smoke 功能验证 |
| `evalscope-smoke-comparison.json`、`evalscope-smoke-report.json` | EvalScope 原生路径与直接路径 19/19 逐题一致 |
| `adapter-patch/erqa-evalscope.patch` | EvalScope ERQAAdapter 与 SystemOne 图文转换补丁 |
| `adapter-patch/patch-manifest.json` | 补丁前后文件 SHA-256 与补丁 SHA-256 |
| `adapter-patch/apply-check.log` | 补丁对 preimage 的 `patch --dry-run` 全部通过 |
| `restore/snapshot.json` | 恢复后 PID 432954 的快照、`/health` 200（空 body）与 8/8 冻结回放 |
| `restore/comparison.json` | 恢复前后 argv/cwd/env 逐项一致、关键文件哈希与回放对比 |
| `restore/current-service.json` | 当前服务指针（PID 432954，原 argv/cwd/4096 配置） |
| `restore/final-replay.json` | 恢复后再次执行 8/8 冻结用例，全部逐位一致 |
| `delivery-state.json` | 本地/远端/推送提交对应关系、文件哈希核对与服务状态 |
| `service-stop.json` | ERQA 服务 PID 430985 的停止记录 |
| `artifact-manifest.json` | 本目录文件大小/SHA-256 与远端原始路径 |

远端原始证据（大体积，不进入仓库）：

- `/workspace/yangchen/logs/erqa-baseline-20261114/full/samples/<id>/`：完整 wire 请求（含 base64 图像）、原始响应、每次尝试与逐题记录。
- `/workspace/yangchen/logs/erqa-baseline-20261114/full/`：`predictions.jsonl`、`summary.json`、`offline-score.json`。
- `/workspace/yangchen/datasets/erqa-official-20261114/`：`samples.jsonl`、`images/`、`manifest.json`。
- `/workspace/yangchen/work/src/ERQA/`：官方仓库 checkout（commit `5bba488`）。
- `/workspace/yangchen/logs/erqa-baseline-20261114/service/service.log`：ERQA 服务日志。
