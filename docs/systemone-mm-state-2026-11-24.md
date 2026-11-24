# `POST /v1/systemone` 图文 `state` 列表实现与验收记录（2026-11-24）

> 状态：**完成**。阶段 1–3 离线实现与阶段 0/4 远端验收均已完成；阶段 4 于 2026-11-24 12:09–12:52 UTC 在 A3 NPU 6 的单一常驻 TP2 服务上执行，服务已停止，18143 与两个逻辑 NPU 已释放。EvalScope 原生客户端因当前环境未安装且无离线包/网络，作为唯一环境阻塞项记录；完整 2,352 题已由固定 `/v1/systemone` 客户端覆盖。

## 1. 背景与目标

主办方 2026-11-24 规则要求 `POST /v1/systemone` 同时承接：

- OpenDecisions 文本评测（EvalScope System One 格式，`state` 为字符串或 JSON 对象）；
- ERQA 图文评测（原题顺序的 `image_url` + `text` 内容块放入 `state` 数组，响应与文本选择题一致）。

原服务只从顶层 `images` 构造 `media`，`JointSchemaStrategy._render` 对非字符串 `state` 直接 `json.dumps`，因此新报文会把 base64 当普通文本分词并截断题干，图片完全不进入视觉编码器。本阶段的目标是让默认 `bash Clef-27B/start.sh --model-dir <dir> --output-dir <out>` 单服务常驻实例同时通过 OpenDecisions 与 ERQA。

## 2. 固定源码契约（阶段 0 部分已完成离线核对）

从公开固定提交 `vllm-project/vllm@24420b7a` 获取的 4 个前端文件 SHA-256 与历史快照一致：

| 文件 | SHA-256 | 说明 |
| --- | --- | --- |
| `vllm/entrypoints/generate/structured_decisions/serving.py` | `5a700bb79c5db36f46d096d9153bb000357644bc77938fbbf667c3b4ef4c400a` | 补丁 preimage |
| `.../structured_decisions/protocol.py` | `c4ea3c8a35e1e1be58c3fe34fa590f92ea832233a3eb47644e5ca9bf964492f6` | `state: Any`，**无需修改** |
| `.../structured_decisions/api_router.py` | 上游 `b773cbe5…`；打 `clef-api-routes.patch` 后 `bdf11bee…` | 与 `clef-api-routes.patch` 快照一致 |
| `vllm/entrypoints/openai/decisions/strategies.py` | `d7e25f1d879cd0dc2d8e38629805a66018a9c63e364572d6cbfee6a9fedd97bb` | **零改动**，方案 A 不触碰 `strategies.py` |

`protocol.py` 的 `StructuredDecisionRequest.state` 为 `Any`，`images: list[str] | None`，因此本方案不需要放宽 Pydantic 类型。

## 3. 协议规范（阶段 1 冻结）

### 3.1 识别与归一化

`state` 仅在同时满足以下条件时进入图文解析：

1. `isinstance(state, list)` 且非空；
2. 每个元素都是 `dict` 且 `type` 为 `str`；
3. 至少一个元素 `type == "image_url"`。

其余值一律原样透传，保持 OpenDecisions 文本路径逐位不变：`str`、`dict`（包括恰好含 `"type": "image_url"` 的 dict）、数字、布尔、`None`、`[]`、纯字符串列表、普通对象列表、没有 `image_url` 的列表。

对已识别列表采取 Fail-Closed：

- `image_url` 必须是 `{"url": <非空 str>}` 或字符串简写；缺失、空串、非字符串、非 `data:image/(jpeg|jpg|png|webp);base64,` 且非 `http://` / `https://` / `file://` 的 URL → `ValueError` → HTTP 400；
- `text` 必须包含字符串 `text`；
- 混入 `audio_url`、`video_url` 等未声明类型 → `ValueError` → HTTP 400；
- 含 `image_url` 但结构不完整的混合列表也直接 400，绝不回退 `json.dumps`（避免 base64 进入分词器）。

### 3.2 文本拼接与模式

环境变量 `CLEF_STATE_MM_MODE`：

| 模式 | 行为 | 状态 |
| --- | --- | --- |
| `extract_plain`（A1） | 所有 `text` 段 `"".join` 为纯字符串 `state`；图片按原顺序进入 `media["image"]` | A/B 臂：ERQA 213/400 |
| `extract_question_dict`（A2，默认） | 同上，但 `state = {"question": <joined>}`，等价 2026-11-14 基线 | **锁定默认**：ERQA 216/400 |
| `interleave` | 保留 `[(kind, value), ...]` 段序列，供方案 B 原生交错渲染 | 解析器支持；前端在方案 B 落地前对图文请求明确 400 |

官方 `eval_harness.py` 的字符下标切分经 `"".join(text_parts)` 可逐字符还原原题；解析器保留 `segments` 供后续原生交错方案使用。

### 3.3 媒体合并

`[*顶层 images, *state 内图片]`；顶层 `images` 与 `state` 列表可同时存在，统一受 `--limit-mm-per-prompt` 校验。仅传 `state` 列表时不传顶层 `images`。

### 3.4 响应

与文本选择题共用 `ServingStructuredDecisions.create_decision` 出口：`id`、`object`、`created`、`model`、`answers`、`usage`、`diagnostics`、`joint` 不变；概率为原始服务输出，不做温度校准。

## 4. 实现与 SHA

| 文件 | SHA-256 | 作用 |
| --- | --- | --- |
| `runtime/systemone_mm_state.py` | `a8b87841cc777a73dd682fd165739dc047ce2925a80879dbb3bf5dd60753e7f5` | 零依赖归一化核心 |
| `runtime/systemone-mm-state.patch` | `f92cc769a1b5d6c0c951aaada1963d302f86de719af515f3aba36903dc7e16d0` | `serving.py` 统一 diff |
| `runtime/apply_systemone_mm_state_patch.py` | `903dc0644338109035b2b82ecdf57a1edc61cb5709c100c4f6ea26a9a0f68e09` | `--check / --apply / --revert` 幂等安装器 |
| `runtime/testdata/systemone_mm_state/serving.py` | `5a700bb79c5db36f46d096d9153bb000357644bc77938fbbf667c3b4ef4c400a` | 固定 preimage 测试夹具 |
| `runtime/test_systemone_mm_state.py` | `8ae2a2ad80b1eb41b449e872a5d50fa11eeb5d9ea179b9c5900f47657929d385` | 归一化与安装器离线测试 |
| `scripts/official_eval/test_erqa_state_list.py` | `9dd7814c566e4460819e0a77503fbc54fb909b3857e6a1a9994a28be7c0943c1` | ERQA 官方交错与线格式测试 |
| `Clef-27B/inference.py` | `d46aff0f745708eca9f6018c2d764ec6817ff295543388595febce3489a3c770` | 安装器调用、统一常驻配置与 A2 默认 |

安装器把 `systemone_mm_state.py` 逐字节复制为 `vllm/entrypoints/generate/structured_decisions/_clef_mm_state.py`，并在 `serving.py` 中：

- 导入 `normalize_systemone_state`；
- 在 `create_decision` 内构造 `effective_request = request.model_copy(update={"images": normalized.images})`，不就地修改 Pydantic 实例；
- 用 `normalized.state` 调用 `strategy.render_state`；
- 将解析期 `ValueError` 转为现有 `StructuredDecisionError`（HTTP 400）。

postimage SHA-256：`99a8e1b73dd55800736de4c2b29cf300aa7f819e4e5495f59805733dbe62994a`；`git apply` 与 `patch -p1` 均已验证产出该精确字节。

## 5. 启动器默认配置

`Clef-27B/inference.py` 的 `s1` 从独立 OD 配置改为单服务常驻 TP2：

| 参数 | 旧值 | 新值 |
| --- | --- | --- |
| `--max-model-len` | 4096 | **16384** |
| `--max-num-seqs` | 1 | **16** |
| `--max-num-batched-tokens` | 4096 | **16384** |
| `--gpu-memory-utilization` | 0.90 | **0.85** |
| `--limit-mm-per-prompt` | `{"image":1,"video":1}` | **`{"image":16,"video":0}`** |
| `--mm-processor-kwargs` `longest_edge` | 65536 | **1048576**（`shortest_edge` 65536） |
| `--encoder-cache-tokens` | 0 | **262144**（Worker 1 阶段 0 实测 OD S1 零劣化 ≤0.7 ms，默认启用；显式 0 可回退） |

`--encoder-cache-tokens` 已支持 `CONFIGS` 配置与 CLI/`sources` 记录；Worker 1 阶段 0 的同窗口 A/B 已证明 `262144` 对 OD S1 零劣化（≤0.7 ms），因此 `s1` 默认启用，显式 `--encoder-cache-tokens 0` 可复现关闭臂。s1 的 TP2、joint fused、GDN BIC、conv+QKV、chunk128/adaptive 等既有默认保持不变。

启动时 `systemone_mm_state_patch()`：

- `--dry-run` 严格只读、不调用子进程、不探测 vLLM，manifest 写 `pending_install: true`；
- 正式启动先 `--check`，未就绪则 `--apply` 后复核，记录安装器/模块 SHA、`patch_state`、`CLEF_STATE_MM_MODE`；
- 非法 `CLEF_STATE_MM_MODE` 在启动期报错。

## 6. ERQA 评测脚本

`scripts/official_eval/erqa_protocol.py`、`erqa_baseline.py`、`erqa_score.py` 与 `runtime/erqa_s2_concurrent.py` 新增：

- `--wire-format {top-level-images,state-list}`：默认 `top-level-images` 保持历史基线可复核；新规则使用 `state-list`；
- `--endpoint {/decide,/systemone}`：默认 `/decide`；新规则使用 `/systemone`；
- `state-list` 按官方 `eval_harness.py` 的 `visual_indices` 稳定排序与字符切片生成内容块，**不传顶层 `images`**；
- `request_summary` 对 `state-list` 增加 `wire_format` 字段；历史 `top-level-images` 的 structural hash 保持原算法，`--verify-requests` 兼容新旧运行；
- predictions 记录 `wire_format` 与 `endpoint`；`erqa_s2_concurrent.py` 的 frozen SHA 已更新为 `erqa_baseline.py` 新 SHA `a3a7a961…`。

## 7. 阶段 0 与阶段 4 实测（2026-11-24，A3 NPU 6，单实例 TP2）

### 7.1 前置核验

- 远端 `/workspace/yangchen/work/mix-ws-20261123/src/vllm` HEAD `24420b7a…`，`serving.py` `5a700bb7…`、`protocol.py` `c4ea3c8a…`、`strategies.py` `d7e25f1d…`、`api_router.py` `bdf11bee…`；`vllm-ascend` HEAD `89eb8a5e…`，GDN conv+QKV/chunk128 补丁就绪。
- 启动前 18143 空闲、无 vLLM 进程、NPU 6 无进程。
- 远端干净 checkout 全量单测：**59 tests, 0 failures, 0 skipped**；OpenDecisions 2,352 条多模态误判 **0/2,352**；ERQA 400 条 `"".join(text_parts) == question` **400/400**，`image_count >= 1` 全部成立。
- 官方裸命令（无额外覆盖参数）冷启动 `http://127.0.0.1:18143`，75 s 就绪；manifest 记录 `systemone_mm_state.patch_state=postimage`、模块 `a8b87841…`、`CLEF_STATE_MM_MODE=extract_question_dict`、encoder `262144`、joint fused `26d3ad04…`、conv+QKV ready、`TASK_QUEUE_ENABLE=2`。

### 7.2 新接口冒烟（静默锁定 A2 后）

| 项目 | 结果 |
| --- | --- |
| `erqa-0000` 单图 `state=[image_url,text]` | HTTP 200；`input_tokens=366`；choice C；概率完整 |
| `erqa-0167` 16 图 | HTTP 200；`input_tokens=10,294`；choice D；无 OOM/超长 |
| `erqa-0115` 交错题 | HTTP 200；`text_join_matches=true`；6 个内容块顺序与官方一致 |
| 旧 `/v1/decide` 顶层 `images` + `{"question":...}` | HTTP 200；与 state-list A2 概率逐位一致 |
| 恶意报文 `url=""` / 缺 `url` / 混入 `audio_url` | 均 HTTP 400；进程存活，随后 `/health` 200 |
| 冻结 8 例文本回归 | **8/8 逐位一致**（5 个验收 case + 3 条 OD 请求对已验收 2026-11-24 运行） |

### 7.3 ERQA 全量 400 A/B（`--wire-format state-list --endpoint /systemone`）

| 臂 | 正确/总数 | 准确率 | 成功/失败/重试 | S1 P95 | S1 mean | 吞吐 |
| --- | --- | ---: | --- | ---: | ---: | ---: |
| A1 `extract_plain` | 213/400 | 0.5325 | 400/0/0 | 281.70 ms | 200.54 ms | 4.900 dec/s |
| A2 `extract_question_dict` | 216/400 | 0.5400 | 400/0/0 | 284.93 ms | 205.08 ms | 4.794 dec/s |
| **锁定 A2** | **216/400** | **0.5400** | **400/0/0** | — | — | — |

A2 比 A1 高 3 题，接近 2026-11-14 基线 219/400，且达到计划决策门 216/400；交错题 16/30 与基线一致，未触发方案 B。锁定 A2 后启动器默认 `SYSTEMONE_MM_DEFAULT_MODE="extract_question_dict"`，`strategies.py` 零改动。

### 7.4 锁定配置同一常驻进程零重启

| 运行 | 正确/总数 | 准确率/质量 | 成功/失败/重试 | 吞吐 | mean | p50 | p95 | p99 |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| ERQA C1 冷首遍 | 216/400 | 0.5400 | 400/0/0 | 4.790 dec/s | 205.17 ms | 194.52 ms | 288.40 ms | 598.37 ms |
| ERQA C1 自然第二遍 | 216/400 | 0.5400 | 400/0/0 | 5.942 dec/s | 164.71 ms | 153.42 ms | 244.38 ms | 465.71 ms |
| ERQA C16 | 217/400 | 0.5425 | 400/0/0 | 8.240 dec/s | 1881.56 ms | 2054.23 ms | 2953.75 ms | 3381.23 ms |
| OD S1 C1 | 2,352/2,352 | Cap **74.2477**、Int 66.0841、Cal 82.4113、acc 1904/2352、MAE 0.402612 | 2352/0/0 | 9.131 dec/s | 109.49 ms | 99.04 ms | 154.55 ms | 191.12 ms |
| OD S2 C16 | 2,352/2,352 | Cap 74.1510、Int 65.8929、Cal 82.4090、acc 1904、MAE 0.401672 | 2352/0/0 | 16.345 dec/s | 978.29 ms | 878.71 ms | 1783.72 ms | 1982.78 ms |
| OD S2 C64 | 2,352/2,352 | Cap 74.0514、Int 65.9840、Cal 82.1188、acc 1904、MAE 0.402581 | 2352/0/0 | 17.193 dec/s | 3685.32 ms | 3617.99 ms | 6087.10 ms | 7362.68 ms |

- ERQA 冷/暖两遍 **400/400 概率逐位一致、0 choice 翻转、max_prob_diff=0.0**；P95 **288.40 → 244.38 ms（−15.27%）**、mean **205.17 → 164.71 ms（−19.72%）**，证明 Tier 1 encoder cache 命中。
- OD S1 与已验收 `official-s1-r1` 全量报告逐位一致：**max_prob_diff=0.0、0 choice 翻转、2,352/2,352 token 数相等**；质量与 Worker 1 验收值一致。
- ERQA `erqa_score.py --verify-requests`：C1 与 C16 均 **400/400 通过、0 mismatch**。
- 全部质量相对 baseline（OD Cap 74.4255、ERQA 0.5475）绝对波动 ≤2PP；所有运行 0 失败、0 重试。

### 7.5 环境阻塞

EvalScope 未安装于 `/workspace/yangchen/venvs/clef-mix-ws-py312`，且镜像内无离线 wheel/源码，`pip index` 不可达。`open_decisions_baseline.py` 在读取包版本时 `PackageNotFoundError` 退出。作为唯一遗留项，待提供 EvalScope 包后可直接执行：

```bash
source scripts/activate_env.sh
python scripts/official_eval/open_decisions_baseline.py \
  --output-dir <fresh-dir> --api-url http://127.0.0.1:18143/v1 \
  --data-dir /workspace/yangchen/datasets/clef-fixed-20261123/opendecisions-1.0.1
```

完整 2,352 题已通过固定 `/v1/systemone` 客户端（`runtime/od_full_benchmark.py` + `od_raw_acceptance.py`）验证，含逐请求原始概率。

## 8. 验收结论与后续

1. **已验收**：单服务常驻 TP2（16K/seqs16/util .85/Tier 1 encoder 262144/TASK_QUEUE=2/conv+QKV v4/joint fused 26d3）同时承接 OpenDecisions 与 ERQA `state` 图文列表；A2 锁定；`strategies.py` 零改动；全部 2PP 质量门通过；0 失败/0 重试。
2. **已归档**：`Clef-27B/logs/systemone-mm-20261124/` 保存 smoke、A/B、C1/C16、OD S1/S2 的轻量摘要、质量 JSON、ERQA 逐题预测与服务 manifest；服务已停止，18143 与 NPU 6 已释放。
3. **剩余**：EvalScope 原生客户端（环境包缺失）；主办方最终模板与 910C/A3 认可映射仍按规则文档持续跟踪。
4. 本记录中的默认参数已由官方裸命令实测确认，不再作为待验证候选。
