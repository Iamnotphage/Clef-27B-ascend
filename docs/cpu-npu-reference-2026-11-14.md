# CPU 独立参考与 NPU baseline 对照（2026-11-14）

本轮用发布方固定参考实现 `joint_schema_model.py` 在 CPU BF16 上独立重跑 OpenDecisions 1.0.1 与 ERQA 官方数据的逐题前向，并与已冻结的 NPU B1 baseline 比较，目标是把输入处理、候选映射、模型加载和数值实现逐层分开。

**本轮按用户要求在 2026-11-14 15:12 UTC 终止**：OpenDecisions 完成 154/2352 题，ERQA 全量未启动，两套数据均未完成全量评分。已完成的输入等价、NPU 复放、pilot 与部分全量对照见下文；未覆盖范围在第 9 节明确列出。

**已确认的结论**

- 输入等价：两套数据全部样本的 CPU 编码 token 数与 NPU 实测 `usage.input_tokens` **逐题相等**（OpenDecisions 2,352/2,352；ERQA 400/400，含 16 图、10,294 token 极值题）。
- 模型加载：CPU 侧 27,484,784,884 个参数全部 `cpu/bfloat16`，与 2026-11-12 参考一致；权重文件、参考源码与联合头配置 SHA-256 固定。
- NPU 稳定：8 条冻结请求在当前服务重复 3 次，**24/24 逐位一致**（最大概率差 0.0）。
- 数值差异随边界收紧放大：154 题连续样本最大概率差 p50 **0.00048**、max 0.0385、1 次 choice 翻转；pilot 的边界抽样子集 p50 0.0136、max 0.1346、5 次 choice 翻转。翻转全部出现在 NPU margin ≤ 0.05 的样本。
- 无候选映射错误：非竞争候选的最大概率差 p95 为 0.0016（pilot 0.0064），差异集中在前两名候选之间。

## 1. 冻结条件

### 1.1 数据与评分器

| 项 | OpenDecisions | ERQA |
| --- | --- | --- |
| 版本 | 1.0.1 | 官方 commit `5bba488252c5b345a43335bfb9239b6bf12bd91c` |
| 文件 | `open-decisions2352.jsonl` | `samples.jsonl` + `images/`（由官方 TFRecord 提取） |
| SHA-256 | `78614972006c5e5efbbfc121ce31e15cf576ef6c0cdf24a229b33d34d2bfcda6` | `35bd36560dcc228c02a8e336cc57b7b8bb0f1c6752521dc63fc482f48a4168af` |
| 样本 | 2,352 | 400（630 张图，113 道多图） |
| 评分 | 作者固定 `score.py`，SHA-256 `478d14ae…e4377` | 官方 exact-match 规则 |

### 1.2 NPU 对照运行

| 项 | OpenDecisions | ERQA |
| --- | --- | --- |
| 运行目录 | `logs/tuning/20261114-013000-fix4/opendecisions/s1-final-full.json` | `logs/erqa-baseline-20261114/full/` |
| 服务配置 | TP2 / BF16 / eager / 4096 / seqs1 / 缓存关 | TP2 / BF16 / eager / 16384 / seqs1 / 缓存关，`image=16`，图像 65,536–1,048,576 像素 |
| 结果 | Capability 74.42547887630508，accuracy 1906/2352，Score MAE 0.4013893127（588） | accuracy 219/400 = 0.5475 |

### 1.3 参考实现与权重

- 模型目录 `/workspace/yangchen/models/Clef-27B`，权重 12 分片 + `joint_head.safetensors`，逐文件大小与 SHA-256 前缀记入 `run-identity.json`。
- 参考源码 SHA-256 `0e304cf7c6500e8bb59bef7e2afd2c6373f82596dfb3b57d1aa93c175e2dc3a3`，未修改。
- 实际执行：CPU、BF16、`eager`、`use_kernels=False`、batch 1、`torch.inference_mode()`、32 线程；参数审计 **27,484,784,884** 个元素全部 `cpu/bfloat16`。
- 环境：PyTorch 2.10.0+cpu、Transformers 5.19.0、accelerate 1.10.1、safetensors 0.8.0；`causal_conv1d` 与 `flash-linear-attention` 未安装，Transformers 使用参考 PyTorch 实现（日志显式记录）。
- 本轮向 `clef-py312` 补装 `accelerate==1.10.1`（历史 CPU 参考使用的版本；此前缺失导致 `device_map` 加载失败），`pip check` 通过。
- `activate_env.sh` 默认 `OMP_NUM_THREADS=1`；参考入口在导入 torch 前把它设为 `--threads`，避免单核运行（286-token 样本：16 线程 32.31 s，32 线程 19.58 s）。

## 2. CPU 参考入口

`scripts/official_eval/cpu_reference_baseline.py`

- 直接调用固定参考实现的 `encode_record` / `collate_records` / 模型前向，不经过服务。
- 输入来自已保存的 NPU 请求：OpenDecisions 取 benchmark 的 `samples[i].request`；ERQA 取 `full/samples/<id>/request.json`（完整 wire body，含 base64 图像）。
- 每题记录 `input_ids`、`input_ids_sha256`、问题/候选 span、媒体身份、原始 logits、未舍入概率、耗时与峰值 RSS。
- 媒体只保留身份：`pixel_values` 形状、dtype、字节 SHA-256 与 `mm_token_type_ids` 计数/哈希，不写像素值。ERQA pilot 首版曾写入 `pixel_values`，16 图单题记录 1.8 GiB；修正后单题约 12 KiB，并在 4 图样本上复核 token 数仍与 NPU 一致。
- 逐题原子写入 `samples/<id>.json`；`run-identity.json` 固定数据、权重、源码、版本、线程、长度与媒体配置；`--resume` 只有执行身份与请求身份都匹配才复用。
- `--encode-check` 只跑 processor，先用全量样本核对编码 token 数。

### 2.1 媒体布局

发布方参考的 `_encode_media` 生成 `PLACEHOLDER * n + "\n"`。固定服务使用的 `ClefFlashStrategy._render_media` 覆盖了基类占位符，生成完全相同的文本形式，因此**无需补丁**；`--encode-check` 在 400 题（1–16 图）上逐题验证。

## 3. 输入等价证据

| 数据集 | 样本 | CPU token 数 = NPU | 不一致 | 说明 |
| --- | ---: | ---: | ---: | --- |
| OpenDecisions | 2,352 | 2,352 | 0 | 长度 145–1,595，4096 上限 |
| ERQA | 400 | 400 | 0 | 240–10,294 tokens，16 图题 `erqa-0167` 10,294 |

编码之外还核对：

- OpenDecisions：保存请求的 `state` 与原始记录逐字相等，`questions.answer` 的 `type`/`instructions`/`criteria` 与原始记录一致（加载时校验，不一致即中止）。
- ERQA：wire 请求的图片数量、顺序与 SHA-256 与提取清单及 NPU 预测行 `image_order`/`image_sha256` 一致；候选数与空候选文本保持请求原样。

## 4. 逐题比较方法

对每题输出：两套实现的预测、完整概率、正确性、最大与平均概率绝对差、TVD、choice/noul/score 的 argmax 变化、score 期望值差、top1–top2 margin、token 数、图片数、题型与样本 ID。

为区分“数值扰动”和“候选映射错误”，额外记录：

- **top-2 占总绝对差的比例**：数值扰动集中在争胜的两个候选；
- **非竞争候选（双方概率均 < 0.1）的最大差**：映射/顺序错误会同时移动所有候选。

工程门槛 0.03/0.05 在本轮仅作为历史诊断统计，不作为 CPU/NPU 适配的通过条件。

## 5. Pilot

### 5.1 OpenDecisions pilot（96 题，56 分钟）

选择规则：三类各 32，覆盖已知合批差异样本、最低/最高 margin、最长/最短输入、NPU 正确与错误样本。该子集刻意向边界与错误样本倾斜，差异率不代表全量。

| 项 | 结果 |
| --- | ---: |
| CPU 完成 / 失败 | 96 / 0 |
| token 数与 NPU 一致 | 96 / 96 |
| 最大概率差 p50 / p95 / p99 / max | 0.01361 / 0.05452 / 0.08672 / 0.13463 |
| > 0.03 / > 0.05 / > 0.1 | 17 / 9 / 1 |
| choice 翻转 | 5 |
| noul 跨 0.5 | 2 |
| score argmax 变化 | 3 |
| score 期望值差 > 0.05 | 4（最大 0.13139） |
| 正确性（都对 / 仅 CPU / 仅 NPU / 都错） | 42 / 5 / 4 / 45 |
| top-2 占总绝对差 p50 | 0.958 |
| 非竞争候选最大差 p95 / max | 0.00643 / 0.01884 |

概率质量只在前两名候选之间转移；翻转全部落在 NPU margin ≤ 0.05 的样本（如 `jb16-llm_router-d03976bffa`：CPU margin 0.0077、NPU 0.0241）。

### 5.2 ERQA pilot

首轮 64 题 pilot 在存储修正前运行，完成 41 题后停止并替换：16 图题把 `pixel_values` 写进 JSON，单题 1.8 GiB。修正后按同一冻结清单重跑的计划随本轮终止取消。

存储修正前的 41 题提供成本与极值证据（`pilot/erqa-v1-timing.json`）：

| 项 | 结果 |
| --- | ---: |
| 完成 / 失败 | 41 / 0 |
| token 数与 NPU 一致 | 41 / 41 |
| 单题耗时均值 | 113.2 s |
| 41 题合计（仅前向） | 4,640.0 s |
| 最长单题 `erqa-0167`（16 图，10,294 tokens） | **1,805.5 s** |
| 该题峰值 RSS | **76.94 GiB** |

### 5.3 存储修正验证

`erqa-0089`（4 图，662 tokens）：token 数与 NPU 一致，记录 11.5 KiB，含 `pixel_values` 形状 `[1728,1536]`、dtype `float32`、字节 SHA-256 与 `mm_token_type_ids` 哈希。

## 6. OpenDecisions 部分全量结果（154/2,352）

终止时完成前 154 条连续记录；这 154 条全部属于 `jb16-llm_router` 的 choice 题，因此本节只覆盖 choice 题型。

| 项 | 结果 |
| --- | ---: |
| 完成 / 失败 | 154 / 0 |
| 墙钟 | 0.89 h |
| 平均 | 20.8 s/题 |
| token 数与 NPU 一致 | 154 / 154 |
| 最大概率差 p50 / p95 / p99 / max | 0.00048 / 0.01527 / 0.02476 / 0.03849 |
| > 0.01 / > 0.03 / > 0.05 | 13 / 1 / 0 |
| choice 翻转 | 1 |
| 正确性（都对 / 仅 CPU / 仅 NPU / 都错） | 146 / 0 / 1 / 7 |
| 该子集 accuracy（CPU / NPU） | 0.9481 / 0.9545 |
| top-2 占总绝对差 p50 | 0.797 |
| 非竞争候选最大差 p95 / max | 0.00160 / 0.00529 |

唯一翻转 `jb16-llm_router-d03976bffa`：229 tokens，最大概率差 0.0161，CPU margin 0.0077、NPU margin 0.0241，CPU 判错、NPU 判对。

## 7. 异常定位

按“输入 → 映射 → 加载 → 数值”顺序检查：

1. **输入编码**：两套数据全量 token 数逐题相等；OpenDecisions 请求与原始记录逐字校验；ERQA 图片顺序与哈希逐题校验。排除输入差异。
2. **候选映射**：非竞争候选最大差 p95 0.0016（pilot 0.0064）；若候选顺序或 id 映射错误，所有候选会同时大幅移动。排除映射错误。
3. **模型加载**：参数总数、dtype、权重文件与源码 SHA-256 固定；加载日志无 missing/unexpected。排除加载错误。
4. **数值实现**：差异集中在前两名候选（top-2 占比 p50 0.80–0.96），随 margin 收紧放大；翻转样本的 margin 均 ≤ 0.05。CPU 为单进程 BF16 参考实现，NPU 为 TP2 + Ascend 融合算子 + vLLM 侧独立实现的联合头（`segment_mean` 以 float32 累加、einsum 池化），两者的归约顺序与算子实现不同，边界处 logit 差被 softmax 放大是预期行为。
5. **服务稳定性**：8 题 × 3 次复放逐位一致，排除服务侧随机性。

**结论**：未发现输入处理、候选映射或模型加载错误；剩余差异属于两套执行栈的数值实现差异，且只在决策边界产生可见影响。FP32/局部 FP64 复核未执行（本轮终止），也未出现需要触发该复核的、超出历史 0.03/0.05 门槛的系统性异常。

## 8. 耗时、内存与复现

| 项 | 实测 |
| --- | ---: |
| OpenDecisions pilot 96 题 | 56 分钟，均约 35 s/题 |
| OpenDecisions 部分全量 154 题 | 0.89 h，均 20.8 s/题 |
| ERQA pilot 41 题 | 均 113.2 s/题 |
| ERQA 16 图极值题 | 1,805.5 s，峰值 RSS 76.94 GiB |
| 常规题峰值 RSS | 约 47.5–50 GiB |

复现命令（远端环境）：

```bash
cd /workspace/yangchen/Clef-27B-ascend
source scripts/activate_env.sh
export TORCH_DEVICE_BACKEND_AUTOLOAD=0 OMP_NUM_THREADS=32

# 输入等价（全量，只跑 processor）
python scripts/official_eval/cpu_reference_baseline.py \
  --dataset opendecisions \
  --data-dir /workspace/yangchen/datasets/clef-full-20261112/full-data \
  --npu-run /workspace/yangchen/logs/tuning/20261114-013000-fix4/opendecisions/s1-final-full.json \
  --output-dir /workspace/yangchen/logs/cpu-reference-20261114/encode-check-od \
  --encode-check

# 逐题前向（可 --resume 续跑）
python scripts/official_eval/cpu_reference_baseline.py \
  --dataset opendecisions \
  --data-dir /workspace/yangchen/datasets/clef-full-20261112/full-data \
  --npu-run /workspace/yangchen/logs/tuning/20261114-013000-fix4/opendecisions/s1-final-full.json \
  --output-dir /workspace/yangchen/logs/cpu-reference-20261114/full/od \
  --threads 32 --resume

# 对照与评分
python scripts/official_eval/cpu_reference_compare.py --dataset opendecisions \
  --cpu-run /workspace/yangchen/logs/cpu-reference-20261114/full/od \
  --npu-run /workspace/yangchen/logs/tuning/20261114-013000-fix4/opendecisions/s1-final-full.json \
  --data-dir /workspace/yangchen/datasets/clef-full-20261112/full-data \
  --scorer /workspace/yangchen/logs/opendecisions-baseline-20261113-1600/author-score.py \
  --output /workspace/yangchen/logs/cpu-reference-20261114/full/od-compare
```

ERQA 的同组命令使用 `--dataset erqa --data-dir /workspace/yangchen/datasets/erqa-official-20261114 --npu-run /workspace/yangchen/logs/erqa-baseline-20261114/full`，默认 `max_length=16384`。

## 9. 证据路径

仓库 `Clef-27B/logs/cpu-reference-20261114/`：

| 内容 | 文件 |
| --- | --- |
| phase-0 服务快照与 8/8 冻结回归 | `phase0/snapshot.json` |
| 两套数据全量编码等价 | `encode-check/encode-check-od-summary.json`、`encode-check-erqa-summary.json`（完整版 `.json.gz`） |
| pilot 清单、规则与理由 | `pilot/pilot-od-96.*`、`pilot/pilot-erqa-64.*` |
| ERQA pilot v1 成本与极值 | `pilot/erqa-v1-timing.json` |
| 存储修正后的媒体记录 | `smoke/erqa-0089.json`、`smoke/summary.json`、`smoke/run-identity.json` |
| NPU 复放稳定性 | `repeat/npu-repeat.json`、`repeat/npu-repeat-ids.json` |
| OpenDecisions 部分全量（154 题） | `full/od-summary-partial.json`、`full/od-comparison-partial.json` |
| 终止记录 | `termination.json` |

远端原始目录：`/workspace/yangchen/logs/cpu-reference-20261114/`（含 154 份逐题记录、pilot 记录、日志与运行身份）。

## 10. 终止时的未完成范围

1. **OpenDecisions 全量**：完成 154/2352（6.55%），仅覆盖 choice 题；noul 与 score 的全量对照未执行。
2. **ERQA 全量**：0/400；存储修正后的 64 题 pilot 未重跑。
3. **完整评分**：两套数据的 CPU 端 Calibration / Intelligence / Capability / accuracy / Score MAE / TVD 与 ERQA accuracy 及分项均未生成；154 题子集的 accuracy 仅作参考，不可作为评分结论。
4. **FP32/FP64 复核与 CPU 重复确定性**：未执行。
5. **服务状态**：未改动。终止后复核 PID 432954、`GET /health` 200、8/8 冻结用例逐位一致。
6. **CPU 参考结果不进入比赛提交指标**；OpenDecisions 与 ERQA 的提交质量仍采用 NPU 运行值及其运行身份。
