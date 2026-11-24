# `POST /v1/systemone` 图文 `state` 列表离线实现记录（2026-11-24）

> 状态：**阶段 1–3 离线完成；阶段 0 / 4 远端验证待执行**。性能优化 Worker 仍占用远端 A3 与 18143，本阶段未登录远端、未启动/停止服务、未修改 `/workspace/yangchen/work/src/vllm`。恢复远端验证前按 `p0_execution_plan.md` 与 `systemone_multimodal_state_plan.md` 的协作纪律执行。

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
| `extract_plain`（默认） | 所有 `text` 段 `"".join` 为纯字符串 `state`；图片按原顺序进入 `media["image"]` | 本次默认 |
| `extract_question_dict` | 同上，但 `state = {"question": <joined>}`，等价 2026-11-14 基线 | A/B 对照 |
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
| `runtime/test_systemone_mm_state.py` | `042f4e29f18c253dee13983cf5edf21a323577c678726a89dd26fe8742d7c157` | 归一化与安装器离线测试 |
| `scripts/official_eval/test_erqa_state_list.py` | `36884add96332dabf78159eed2ada2a2e4ad459c5e9a4f14d01d2e5a012ae8ee` | ERQA 官方交错与线格式测试 |

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
| `--encoder-cache-tokens` | 0 | **0（待阶段 0 证据）** |

`--encoder-cache-tokens` 已支持 `CONFIGS` 配置与 CLI/`sources` 记录；候选值 `262144` 必须等性能 Worker 阶段 0 的同窗口 A/B 证明无 S1 退化后再启用，因此本次保持 0。s1 的 TP2、joint fused、GDN BIC、conv+QKV、chunk128 等既有默认不变。

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

## 7. 离线验证结果

执行：

```bash
python3 -m unittest runtime/test_systemone_mm_state.py \
  runtime/test_inference_launch.py scripts/official_eval/test_erqa_state_list.py
```

结果：**59 tests, 0 failures, 2 skipped**（跳过项为数据集缺失，非关键代码项）：

- OpenDecisions 2,352 全量误判扫描：本机无 `/workspace/yangchen/.../open-decisions2352.jsonl`，测试自动 skip；测试存在数据时硬断言 0/2,352，需在远端阶段 0 补跑。
- ERQA 400 交错切分审计：本机无 `samples.jsonl`，测试自动 skip；远端阶段 0 对全部 400 条断言 `image_count >= 1` 且 `"".join(text_parts) == question`。

已离线覆盖：

1. 标准单图/16 图/交错/纯图/多 MIME/detail/字符串简写；
2. 旧协议 `state` dict + 顶层 `images` 原样透传；顶层与 state 图片确定性合并；
3. 零误判：`str`、`dict`、`None`、`[]`、字符串列表、文本列表、普通对象列表、含 `image_url` 的 dict 全部不识别；200 组随机交错题 `"".join` 恒等；
4. Fail-Closed：缺失/空/非字符串 URL、非法 scheme、非字符串 text、未知 type、含图片的畸形混合列表均 `ValueError`；
5. 安装器生命周期：`--check(preimage)` → `--apply` → `--check(ready)` → 幂等 `--apply`（inode/mtime 不变）→ `--revert`（serving 恢复 preimage、模块删除）；未知 serving/模块内容拒绝修改；
6. 启动器：s1 新默认、dry-run 零子进程、`CLEF_STATE_MM_MODE` 记录/校验、安装器报告 SHA 校验；
7. ERQA：两种 wire format 的结构哈希可复算、endpoint 路由、官方交错切片、历史报文 key 不变。

## 8. 待执行（远端，需性能 Worker 释放后）

1. **阶段 0**：核对远端 4 文件 SHA（应全部匹配第 2 节）、运行 OpenDecisions 2,352 与 ERQA 400 的离线预审计；确认 18143 与双逻辑 NPU 空闲。
2. **阶段 4**：官方命令裸启动；6 项冒烟（`/health`、`/v1/decide/info`、冻结 8 例文本、`erqa-0000` 单图 token≈366、`erqa-0167` 16 图 token≈10,294、非法报文 400、旧 `/v1/decide` 兼容）；ERQA 400 全量 A1/A2（必要时方案 B）与 S2 C16；同进程 OpenDecisions 2,352 逐位回归与 EvalScope 联调。
3. **阶段 5**：按实测更新 `docs/submission-rules-2026-11-13.md`、`Clef-27B/README.md`、`optimization.md`、`metrics.csv`、`results.csv`，补全本记录并合并交付。

在阶段 4 通过前，`s1` 的 16384/16/0.85 等默认值与 `encoder_cache_tokens=0` 均视为待验证候选，不写入最终提交结论。
