# Aclnn executor cache 与 S1 形状慢路径修复（2026-11-17）

## 0. 结论

- 单行服务环境 `ACLNN_CACHE_LIMIT=1000000`（已写入提交入口 `Clef-27B/inference.py` 的 service environment，默认值，可由外部覆盖）把 S1 从 **mean 153.76 / p50 168.49 / p95 190.40 ms** 降到 **mean 128.99 / p50 120.96 / p95 169.02 ms**，吞吐 **6.503 → 7.748 decisions/s（+19.2%）**，2,352/2,352 成功。
- 质量逐位不变：OpenDecisions 全量 2,352 条相对提交版 **max 概率差 0.0、0 个 choice 翻转、noul/score 差 0.0**；ERQA **217/400 = 0.5425** 不变，400/400 成功、0 重试。
- S2 B8 C16 **16.319 dec/s**、p95 1807 ms（提交版 16.398 / 1803，噪声内）；该修复只影响 C1 的按长度 executor 选择，不改变批处理吞吐。
- 规则合规：只调整 CANN aclnn executor 缓存容量，无量化、无温度校准、无数值变化。

## 1. 现象

提交版 S1 的逐请求延迟按**精确 input token 数**分成两档：

- 快档约 105–130 ms（短/中长度）；
- 慢档约 165–190 ms，固定多约 **55 ms**；
- 同一 token 长度的所有请求落在同一档，within-length 标准差中位 **4.7 ms**（p90 8.5）；相邻同长度请求不会切换档；
- head-entry / head-control / qkv-v2-control 三个运行 207/207 个长度标签一致；11-15 的 `batching-s1` C1 也 256/256 一致 → 与 head/qkv 补丁无关。

profile 中的 307 样本就是 `od-0105`：生产 **114.38 ms**，profile 窗口 **169.55 ms**；967 样本 `od-1534` 生产 **177.85 ms**，profile **174.99 ms**。因此 `docs/profile-final-2026-11-17.md` 的 307 CPU/设备绝对值不可用于生产结论（采集把快档推入慢档），967 设备归因仍然有效。

## 2. 机制

1. **全新服务**：逐个首次使用 203/204/230/306/307/308/309/310/311/312/313/315/320/330/350/385/389/400/426/438/447/498/600/967，全部约 110–145 ms（600 以上受设备计算自然增长），没有 308 硬边界。
2. **按 755 个不同长度依次首次使用**（OD 全量长度集合，升序）：前约 **145 个不同长度**保持快档；之后首次出现的新长度锁进慢档，固定 +约 55 ms；已锁定的长度不再随历史恢复/切换。
3. **快/慢档数值完全一致**：`od-0383`（203）与 `od-0182`（204）在快档和慢档下概率逐位相同；全量 2,352 条与提交版逐位相同。
4. **离线算子探针排除 kernel 计算**：MLP gate/up、down 及全部模型 matmul 形状（5120→8192/3072/7168/48、8192/3072/7168→5120）、GDN `chunk_gated_delta_rule_fwd`、`causal_conv1d_fn` 在 307/308/坏/好形状上的纯算子时间均平滑（见 `Clef-27B/logs/tuning5-20261117/shape-probe/`）。
5. **CANN 侧缓存**：`ACLNN_CACHE_LIMIT` 由 `libnnopbase.so`、`libop_compile_adapter.so` 等消费；默认容量只够覆盖前约 145 个 prompt 长度所对应的 aclnn op shape。缓存耗尽后，新 shape 的 executor 走慢路径。把上限设为 **1,000,000** 后，755 个长度的暖机全部为快档：308–600 中位 **约 119 ms**（原约 175 ms），只有 >600 token 因设备计算自然变慢（701–1000 中位 156 ms，1001–1600 中位 193 ms）。

## 3. 结果

### 3.1 S1（seqs1，全量 2,352，C1）

| 指标 | 提交版 | `ACLNN_CACHE_LIMIT=1000000` | 变化 |
| --- | ---: | ---: | ---: |
| mean | 153.76 ms | **128.99 ms** | **−16.11%** |
| p50 | 168.49 ms | **120.96 ms** | **−28.21%** |
| p95 | 190.40 ms | **169.02 ms** | **−11.23%** |
| p99 | 203.20 ms | 205.91 ms | +1.33% |
| 吞吐 | 6.503 dec/s | **7.748 dec/s** | **+19.16%** |
| 成功 | 2,352/2,352 | 2,352/2,352 | — |

- 分长度：140–307 p50 117.4；308–500 p50 119.5；501–700 p50 125.4；701–1000 p50 156.4；1001–1600 p50 192.8 ms。
- 首个请求仍有约 617 ms 冷启动（max），不影响 p95；p99 由长序列设备计算主导。

### 3.2 S2（B8 C16，全量 2,352，C16）

| 版本 | 吞吐 | p95 | mean |
| --- | ---: | ---: | ---: |
| 提交版 B8 C16 | 16.398 dec/s | 1802.9 ms | 971.6 ms |
| 本次 + cache | 16.319 dec/s | 1807.2 ms | 976.6 ms |

批处理形状由 batch 组合决定，不走按 prompt 长度的 executor 慢路径，故无收益；保持提交版 S2 数字，环境变量对 S2 无副作用。

### 3.3 质量

| 指标 | 提交版 | 本次 | 变化 |
| --- | ---: | ---: | ---: |
| OpenDecisions Calibration | 82.7092 | 82.7092 | 0 |
| OpenDecisions Intelligence | 65.9122 | 65.9122 | 0 |
| OpenDecisions Capability | 74.3107 | 74.3107 | 0 |
| OpenDecisions accuracy | 1902/2352 | 1902/2352 | 0（逐位） |
| Score MAE | 0.401545 | 0.401545 | 0 |
| ERQA accuracy | 217/400 = 0.5425 | 217/400 = 0.5425 | 0 |

- 2,352 条逐样本比较：0 missing、0 choice flips、max abs probability diff **0.0**、noul/score 差 0.0；证据 `Clef-27B/logs/tuning5-20261117/quality-compare.json`。
- ERQA 400/400 成功、0 重试；mean 223.1 ms、p95 312.1 ms。

## 4. 提交入口改动

`Clef-27B/inference.py` 的 `environment` 字典新增：

```python
"ACLNN_CACHE_LIMIT": os.environ.get("ACLNN_CACHE_LIMIT", "1000000"),
```

该值在 `os.execv` 进入 vLLM 之前写入 `os.environ`，并在 `launch-manifest.json` 的 `service_environment` 中记录。dry-run 已验证：

```
ACLNN_CACHE_LIMIT = 1000000
config = s1 port 18143
```

`start.sh` 无需修改；外部仍可用同名环境变量覆盖。

## 5. 复现

```bash
# 实验启动器（不改动 start.sh/inference.py，只复放已验证 manifest）
python runtime/launch_from_manifest.py \
  --manifest /workspace/yangchen/logs/gdn-data-path-20261115/head-entry/launch-manifest.json \
  --run-dir <run> --port 18143 --set ACLNN_CACHE_LIMIT=1000000

# S1 全量
python scripts/benchmark_systemone.py --base-url http://127.0.0.1:18143 \
  --sample-file /workspace/yangchen/logs/tuning/20261114-013000-fix4/opendecisions/all2352.json \
  --warmup 0 --requests 2352 --concurrency 1 --report <run>/full.json
```

ERQA 服务需使用 `--max-model-len 16384 --max-num-batched-tokens 16384 --limit-mm-per-prompt '{"image":16,"video":1}'` 与图像像素 65,536–1,048,576；命令同 `docs/handoff-11-17.md` 第 7 节。

## 6. 证据

- S1 报告：`Clef-27B/logs/tuning5-20261117/s1-aclnn-cache-full.json.gz`（2,352 条逐题响应、概率、耗时）。
- 启动记录：`Clef-27B/logs/tuning5-20261117/s1-aclnn-launch.json`（env 含 `ACLNN_CACHE_LIMIT`）。
- 质量比对：`Clef-27B/logs/tuning5-20261117/quality-compare.json`。
- ERQA：`Clef-27B/logs/tuning5-20261117/erqa/summary.json`、`offline-score.json`。
- 算子探针：`Clef-27B/logs/tuning5-20261117/shape-probe/matmul-sweep.json`、`gdn-shapes.json`、`all-matmul-shapes.json`。
- 实验脚本：`runtime/launch_from_manifest.py`、`runtime/shape_history_probe.py`、`runtime/shape_gdn_probe.py`、`runtime/shape_matmul_sweep.py`。
- 服务端日志与完整运行目录：`/workspace/yangchen/logs/tuning5-20261117/`。

## 7. 注意事项

- 每次“全新服务”实验必须确认新 PID 存活且端口空闲；本轮曾因一个未退出的服务占用 18143，使后续启动器静默失败、请求全部打到污染服务，产生过一批错误结论。`runtime/launch_from_manifest.py` 现已内置端口占用与启动存活检查。
- 约 145 个不同长度是实测预算；`ACLNN_CACHE_LIMIT` 的默认数值未在本机文档中给出。1,000,000 对 755 个 OD 长度足够，host 内存从 135 GB 到 136 GB，无明显增长。
- 首个请求冷启动（约 500–620 ms）仍在；S1 指标为 P95，不影响。若后续需要 p99，可在服务 ready 前加一轮覆盖代表性长度的暖机。
- 307-token 的 profile 绝对值不可用；后续 profile 应使用 967 或 1,200-token 形状，并同时记录未采集的对照延迟。
