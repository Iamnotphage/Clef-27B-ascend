# ERQA 全量 baseline：2026-11-14

本轮在 A3 单物理卡两个逻辑 NPU 上完成 ERQA 官方 400 题全量图文评测，服务配置沿用已验证的 **TP2 / BF16 / eager / max_num_seqs=1 / 缓存关**，客户端并发 1。全部 400 题成功，无失败与重试；总体准确率 **219/400 = 0.5475**。

## 1. 数据版本与冻结清单

| 项 | 值 |
| --- | --- |
| 官方仓库 | `https://github.com/embodiedreasoning/ERQA` |
| 固定 commit | `5bba488252c5b345a43335bfb9239b6bf12bd91c`（2025-03-12） |
| 数据文件 | `data/erqa.tfrecord` |
| 大小 | 91,402,921 字节 |
| SHA-256 | `1974f5e587eac144cc3ba325477477b8ad6a56c22ce4e419c34229520b4c93e2` |
| 许可证 | CC BY 4.0（仓库 `LICENSE`） |
| split | 官方单文件全量；无训练/测试划分，按 400 条完整记录评测 |
| 样本范围 | 全部 400 条，原始 TFRecord 顺序，ID `erqa-0000`～`erqa-0399` |
| 图像 | 630 张，530 个唯一 SHA-256；全部 JPEG；单题 1～16 张 |
| 原图尺寸 | 宽 200～3,840，高 150～4,080 |
| 类别 | Action Reasoning 72、Multi-view Reasoning 37、Other 14、Pointing 34、Spatial Reasoning 84、State Estimation 55、Task Reasoning 38、Trajectory Reasoning 66 |
| 答案分布 | A 123、B 96、C 98、D 83 |

数据由项目脚本 `scripts/official_eval/erqa_extract_tfrecord.py` 用 Python 标准库直接解析 TFRecord 与 `tf.train.Example` protobuf，不依赖 TensorFlow。输出 `samples.jsonl`、`images/` 与 `manifest.json`，逐题保存原题、答案、类别、`visual_indices`、每题图片顺序、尺寸、格式与 SHA-256。完整清单见 [data-manifest.json](../Clef-27B/logs/erqa-baseline-20261114/data-manifest.json)。

远端数据目录：`/workspace/yangchen/datasets/erqa-official-20261114/`。数据从官方 GitHub 仓库固定 commit 获取，未使用镜像。

## 2. 图文协议与适配

### 2.1 选项解析

官方题目文本内嵌选项，存在 `A.`、`A:`、`A)` 三种标记，选项数为 2、3 或 4。解析器按 A→D 顺序匹配标记，保留空选项文本；400/400 条均可解析，gold 答案始终在解析出的候选内。候选映射为 `{"A": 文本, "B": 文本, ...}`，顺序与题目原文一致。

### 2.2 `visual_indices` 与图片顺序

官方 `eval_harness.py` 按字符位置把图片插入 question 文本：索引为空或全 0 时图片放在 question 前；否则按索引排序后切分文本插入。本快照分布为：空索引 183 条、全 0 索引 187 条、交错索引 30 条。

Clef 的原生请求协议把 `images` 作为独立字段，`joint_schema_model.py` 的 `encode_record` 将图片占位符固定放在 `state` 之前，模型训练格式不支持把图片插入 `state` 文本内部。因此本轮：

- 图片按官方 harness 的规则排序（按 `visual_indices` 稳定排序；空/全 0 保持原顺序）；
- 排序后的全部图片放在 `state` 之前发送，随后是完整原题与结构化候选；
- 逐题记录原始图片顺序、发送顺序与每条 `visual_indices`，便于复核。

这是与官方 harness 的唯一位置差异，已在逐题记录中保留原始索引；图片数量与顺序不变。交错索引 30 条单独统计，准确率 16/30 = 0.5333。

### 2.3 请求模板

端点为 `POST /v1/decide`，协议为 vLLM-Ascend 的 `StructuredDecisionRequest`：

```json
{
  "model": "clef",
  "state": {"question": "<官方原题全文>"},
  "questions": {
    "answer": {
      "type": "choice",
      "instructions": "Which option correctly answers `question`?",
      "criteria": {"A": "<选项 A>", "B": "<选项 B>", "...": "..."}
    }
  },
  "images": ["data:image/jpeg;base64,..."]
}
```

`state` 保留官方原题全文（含题干、选项文本与作答指令），`criteria` 同时给出结构化候选，`instructions` 与 OpenDecisions 基线一致。服务返回 `answers.answer.choice` 与完整候选概率；原始 choice 直接映射为 ERQA 答案字母。

### 2.4 图像处理与上下文预算

| 项 | 值 | 依据 |
| --- | --- | --- |
| 图像像素上限 | `shortest_edge=65536`，`longest_edge=1048576` | `JointSchemaStrategy.IMAGE_SIZE` 参考服务边界 |
| 每请求图片上限 | 16 | 快照单题最大图片数 |
| `max_model_len` | 16384 | `ClefFlashStrategy.MAX_LENGTH` 上限 |
| `max_num_batched_tokens` | 16384 | 单序列、关闭 chunked prefill |
| 实测输入 tokens | 最小 240、最大 10,294、均值 931.7、p95 1,653 | 400 题全部成功请求的 `usage.input_tokens` |
| 实测图像 tokens | 最大 10,065（16 图题） | processor 逐题统计 |

全量前用模型 processor 对 400 题逐题复算 prompt 长度，最大 10,294 tokens，全部低于 16,384；实测 `usage.input_tokens` 与预算逐题一致（例如 erqa-0000 366、erqa-0167 10,294）。逐题预算见 [token-budget.json](../Clef-27B/logs/erqa-baseline-20261114/token-budget.json)。

## 3. 服务配置

| 项 | 值 |
| --- | --- |
| 模型 | `/workspace/yangchen/models/Clef-27B`，revision `local-65e5a3c5ffb6` |
| 并行 | TP2，A3 单物理卡内 2 个逻辑 NPU |
| 精度 / 执行 | BF16 / eager |
| 上下文 / 序列 | `max_model_len=16384`、`max_num_seqs=1`、`max_num_batched_tokens=16384` |
| 缓存 | prefix caching 关、chunked prefill 关、schema token cache 关 |
| 图像 | `--limit-mm-per-prompt {"image":16,"video":0}` |
| 端口 / PID | `127.0.0.1:8000`，PID 430985 |
| 启动入口 | `bash Clef-27B/start.sh --config custom ...` |

完整 argv、生效值与来源见 [service-launch-manifest.json](../Clef-27B/logs/erqa-baseline-20261114/service-launch-manifest.json)，`/v1/decide/info` 返回见 [service-info.json](../Clef-27B/logs/erqa-baseline-20261114/service-info.json)。

本轮开始前原服务 PID 424593（TP2/4096/seqs1/`image=1`）已保存完整快照并停止；快照见 [phase0-snapshot.json](../Clef-27B/logs/erqa-baseline-20261114/phase0-snapshot.json)。

## 4. 全量结果

| 指标 | 结果 |
| --- | ---: |
| 正确 / 固定总数 | **219 / 400** |
| accuracy | **0.5475**（0～1） |
| 成功 / 失败 | 400 / 0 |
| 重试次数 | 0 |
| 失败分类 | 无 |
| wall | 101.099 s |
| 吞吐（C1） | 3.9565 decisions/s |
| p50 / p95 / p99 | 267.849 / **323.476** / 551.516 ms |
| 平均端到端 | 249.871 ms |

准确率分母固定为官方 400 题；失败题按错误计入。本轮无失败，因此固定分母与成功响应分母相同（400）。

### 4.1 按 question_type

| 类别 | 正确 / 总数 | accuracy |
| --- | ---: | ---: |
| Action Reasoning | 42 / 72 | 0.5833 |
| Multi-view Reasoning | 10 / 37 | 0.2703 |
| Other | 3 / 14 | 0.2143 |
| Pointing | 25 / 34 | 0.7353 |
| Spatial Reasoning | 50 / 84 | 0.5952 |
| State Estimation | 34 / 55 | 0.6182 |
| Task Reasoning | 21 / 38 | 0.5526 |
| Trajectory Reasoning | 34 / 66 | 0.5152 |

### 4.2 单图 / 多图

| 分组 | 正确 / 总数 | accuracy |
| --- | ---: | ---: |
| 单图 | 170 / 287 | 0.5923 |
| 多图（2～16 张） | 49 / 113 | 0.4336 |

### 4.3 候选数与图片索引形态

| 分组 | 正确 / 总数 | accuracy |
| --- | ---: | ---: |
| 2 候选 | 11 / 14 | 0.7857 |
| 3 候选 | 2 / 4 | 0.5000 |
| 4 候选 | 206 / 382 | 0.5393 |
| 空 `visual_indices` | 93 / 183 | 0.5082 |
| 全 0 `visual_indices` | 110 / 187 | 0.5882 |
| 交错 `visual_indices` | 16 / 30 | 0.5333 |

逐题预测、完整候选概率、请求哈希、耗时与 token 数见 [predictions.jsonl](../Clef-27B/logs/erqa-baseline-20261114/predictions.jsonl)；分项表见 [metrics-breakdown.csv](../Clef-27B/logs/erqa-baseline-20261114/metrics-breakdown.csv)。

## 5. 离线复核

独立脚本 `scripts/official_eval/erqa_score.py` 从 `predictions.jsonl` 重算：

- 准确率 **219/400 = 0.5475**，与运行汇总一致；
- 分母固定 400，失败 0；
- 官方匹配规则 `prediction.replace(".", "").strip().lower() == gold.strip().lower()`；
- `--verify-requests` 从固定数据重建 400 条请求并比对结构化 SHA-256：**400/400 通过，0 不一致**。

复核报告见 [offline-score.json](../Clef-27B/logs/erqa-baseline-20261114/offline-score.json)。完整 wire 请求（含 base64 图像）、原始响应与每次尝试保存在远端 `full/samples/<id>/`；仓库只保留轻量预测与哈希。

## 6. EvalScope ERQAAdapter 适配

固定源码 `3d39bab09c1d63095490a8b0f4a6dcbcc5ea843e` 的 ERQAAdapter 存在两个缺口：

1. `record_to_sample` 把整段 question 作为文本后追加全部图片，未读取 `visual_indices`；
2. 候选直接固定为 `['A','B','C','D']`，且 `MultiChoiceAdapter.build_systemone_messages` 要求 `sample.input` 为纯文本，无法转换图文样本。

本轮补丁（[erqa-evalscope.patch](../Clef-27B/logs/erqa-baseline-20261114/adapter-patch/erqa-evalscope.patch)）：

- `erqa_adapter.py`：按官方 `eval_harness.py` 语义切分 question 与图片；解析 2～4 个候选；`supports_choice=True`；在 `ChatMessageUser.internal["systemone"]` 中携带完整图文 DecisionInput；
- `multi_choice_adapter.py`：识别图文样本的已标注转换数据；
- `api/model/decision.py`：`DecisionInput` 增加 `images`，允许 ERQA 中的空候选文本；
- `models/utils/decision.py`：wire 请求携带 `images`，解析时读回。

补丁应用前后的文件 SHA-256 见 [patch-manifest.json](../Clef-27B/logs/erqa-baseline-20261114/adapter-patch/patch-manifest.json)。

**交叉验证**：在固定 19 题 smoke 子集上，EvalScope 原生 `systemone_api` 路径（服务同 `StructuredDecisionRequest` 处理器，端点 `/v1/systemone`）与项目直接 `/v1/decide` 路径的逐题预测 **19/19 完全一致**，accuracy 均为 7/19 = 0.3684；对比见 [evalscope-smoke-comparison.json](../Clef-27B/logs/erqa-baseline-20261114/evalscope-smoke-comparison.json)，原生报告见 [evalscope-smoke-report.json](../Clef-27B/logs/erqa-baseline-20261114/evalscope-smoke-report.json)。补丁后另跑 OpenDecisions 文本路径 12 条 smoke，accuracy 12/12，确认新增的图文分支未改变纯文本转换。正式全量结果采用直接 `/v1/decide` 路径。

## 7. 复现命令

```bash
ssh clef
cd /workspace/yangchen/Clef-27B-ascend
source scripts/activate_env.sh

# 1. 解析官方 TFRecord（全新目录）
/workspace/yangchen/venvs/clef-eval-py312/bin/python \
  scripts/official_eval/erqa_extract_tfrecord.py \
  --tfrecord /workspace/yangchen/work/src/ERQA/data/erqa.tfrecord \
  --output-dir /workspace/yangchen/datasets/erqa-official-20261114

# 2. 启动 ERQA 服务（全新 output-dir）
nohup bash Clef-27B/start.sh --model-dir /workspace/yangchen/models/Clef-27B \
  --output-dir /workspace/yangchen/logs/erqa-baseline-20261114/service \
  --config custom --tensor-parallel-size 2 --max-model-len 16384 \
  --max-num-seqs 1 --max-num-batched-tokens 16384 \
  --limit-mm-per-prompt '{"image":16,"video":0}' --port 8000 \
  -- '--mm-processor-kwargs={"images_kwargs":{"size":{"shortest_edge":65536,"longest_edge":1048576}},"videos_kwargs":{"size":{"shortest_edge":4096,"longest_edge":65536},"do_sample_frames":false}}' \
  > /workspace/yangchen/logs/erqa-baseline-20261114/service/nohup.log 2>&1 &

# 3. 全量评测（全新输出目录；--resume 可按请求哈希续跑）
/workspace/yangchen/venvs/clef-eval-py312/bin/python \
  scripts/official_eval/erqa_baseline.py \
  --data-dir /workspace/yangchen/datasets/erqa-official-20261114 \
  --output-dir /workspace/yangchen/logs/erqa-baseline-20261114/full \
  --scope full

# 4. 离线复核与请求哈希校验
/workspace/yangchen/venvs/clef-eval-py312/bin/python \
  scripts/official_eval/erqa_score.py \
  --run-dir /workspace/yangchen/logs/erqa-baseline-20261114/full \
  --data-dir /workspace/yangchen/datasets/erqa-official-20261114 \
  --output /workspace/yangchen/logs/erqa-baseline-20261114/full/offline-score.json \
  --verify-requests
```

EvalScope 适配交叉验证：

```bash
/workspace/yangchen/venvs/clef-eval-py312/bin/python \
  scripts/official_eval/erqa_evalscope_smoke.py \
  --data-dir /workspace/yangchen/datasets/erqa-official-20261114 \
  --work-dir /workspace/yangchen/logs/erqa-baseline-20261114/evalscope-smoke \
  --direct-run /workspace/yangchen/logs/erqa-baseline-20261114/smoke
```

## 8. 证据路径

| 内容 | 仓库路径 | 远端原始路径 |
| --- | --- | --- |
| 运行汇总 | `Clef-27B/logs/erqa-baseline-20261114/summary.json` | `.../full/summary.json` |
| 逐题预测 | `Clef-27B/logs/erqa-baseline-20261114/predictions.jsonl` | `.../full/predictions.jsonl` |
| 离线复核 | `Clef-27B/logs/erqa-baseline-20261114/offline-score.json` | `.../full/offline-score.json` |
| 数据清单 | `Clef-27B/logs/erqa-baseline-20261114/data-manifest.json` | `.../datasets/erqa-official-20261114/manifest.json` |
| Token 预算 | `Clef-27B/logs/erqa-baseline-20261114/token-budget.json` | `.../phase0/token-budget.json` |
| 服务快照 | `Clef-27B/logs/erqa-baseline-20261114/phase0-snapshot.json` | `.../phase0/snapshot.json` |
| 启动清单 | `Clef-27B/logs/erqa-baseline-20261114/service-launch-manifest.json` | `.../service/launch-manifest.json` |
| 完整 wire 请求/响应/尝试 | 轻量预测与哈希 | `.../full/samples/<id>/` |
| EvalScope 补丁 | `Clef-27B/logs/erqa-baseline-20261114/adapter-patch/` | `.../adapter-patch/` |

远端根目录：`/workspace/yangchen/logs/erqa-baseline-20261114/`。

## 9. 服务恢复

ERQA 服务停止后按本轮 phase-0 快照恢复原服务：

- 恢复 PID **432954**，argv、cwd、环境白名单与 phase-0 的 PID 424593 逐项一致；
- 12 个关键文件 SHA-256 全部未变；
- 8/8 冻结用例逐位一致，恢复后再次回放仍为 8/8；
- `GET /health` **200**（空 body），`GET /v1/decide/info` 200；phase-0 快照辅助脚本当时把 `/health` 拼在 `/v1` 基址后，实际请求了 `/v1/health` 并记录 404，已在 [restore/comparison.json](../Clef-27B/logs/erqa-baseline-20261114/restore/comparison.json) 中注明；
- 当前服务指针已更新为 PID 432954，原 4096/seqs1/`image=1` 配置保持不变。

## 10. 已知边界

1. **图文位置**：Clef 原生协议只支持图片置于 `state` 之前，未按官方 harness 的字符索引交错插入。图片顺序与数量保持官方排序；30 条交错样本的分项准确率单独列出。
2. **图像像素上限**：固定为 1,048,576 像素，以同时满足 16 图与 16,384 token 上限；未尝试更高分辨率策略。
3. **单次全量**：本轮为一次全量运行，未做多轮重复；置信区间与跨次波动未估计。
4. **EvalScope 原生验证**：仅覆盖 19 题 smoke 子集，用于适配正确性交叉验证；正式指标来自直接 `/v1/decide` 全量运行。
5. **硬件称谓**：实际设备为 `Ascend910_9382`（A3 单物理卡两个逻辑 NPU），规则正文物为 910C；赛事认可映射仍待官方确认。
6. **CSV 缺失值**：`metrics.csv` 中 ERQA 行的 `score_mae`、`calibration`、`intelligence`、`capability` 记为 `NA`，表示该数据集不适用这些字段；accuracy 范围 0～1。
