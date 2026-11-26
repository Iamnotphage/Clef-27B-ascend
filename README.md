# Clef-27B-ascend

`Cloudflare/clef`（27.5B 混合线性注意力多模态联合决策模型）在昇腾 910A3（单物理卡内 2 个逻辑 NPU，64 GiB HBM × 2）上的全栈适配与非量化性能调优工程。

本仓库通过**单一常驻 TP2 服务（纯 BF16、零量化、零重启、无按请求切库路由）**统一承载纯文本 `OpenDecisions` 与多模态图文 `ERQA` 的质量评测、S1 单并发低延迟评测及 S2 高并发吞吐评测。

## 项目结构

```text
Clef-27B-ascend/
├── start.sh                     # 官方提交启动入口（汇总默认运行开关并转发至 inference.py）
├── inference.py                 # 服务启动编排器（SHA-256 门控安装运行时补丁、RAM 自适应钳制、生成 launch-manifest.json）
├── metrics.csv                  # 质量指标汇总表（OpenDecisions 与 ERQA）
├── results.csv                  # 性能指标汇总表（S1 P95 延迟与 S2 最大吞吐）
├── optimization.md              # 架构瓶颈剖析、分层优化原理、消融数据与负结果完整报告
├── result.png                   # 服务启动、NPU/内存状态与实测运行截图
├── npu_device_call.png          # 昇腾 NPU 设备调用与算子执行实测截图
├── requirements.txt             # Python 纯依赖冻结锁（配合 scripts/setup_env.sh 构建自包含 .venv）
├── src/
│   ├── ascendc/                 # 自定义 Ascend C 融合算子源码（kernels/）与预编译算子包（vendors/）
│   ├── patches/                 # vLLM / vLLM-Ascend 固定提交构建期补丁
│   └── runtime/                 # 运行时哈希门控补丁安装器、Tier 2 前缀状态缓存引擎、并发压测与离线评分工具
├── scripts/
│   ├── setup_env.sh             # 一键构建项目内自包含 .venv（含固定提交 vLLM/vLLM-Ascend/EvalScope 与自定义算子）
│   ├── activate_env.sh          # 激活 CANN / ATB 与项目 .venv 环境
│   ├── download_model.sh        # 基于 ModelScope 下载并校验 Cloudflare/clef 权重
│   ├── benchmark_systemone.py   # /v1/systemone 标准库压测与协议校验脚本（OD S1/S2）
│   ├── profile_systemone.py     # 双卡 torch_npu Profiler 窗口采集与离线解析入口
│   └── official_eval/           # OpenDecisions 与 ERQA 官方协议解包、评测及请求哈希复核脚本
├── logs/
│   └── final-eval/              # 最终提交版原始评测产物与 SHA-256 校验清单（od-s1, od-s2, erqa-s1, erqa-s2）
├── docs/                        # 阶段性优化专题报告、基线档案与 Profile 出图任务书
└── imgs/                        # 性能剖析与消融对比图表目录
```

## 质量指标

OpenDecisions 公开参考来源：[数据集作者榜单](https://huggingface.co/datasets/jgeuter/OpenDecisions#leaderboard-october-15-2026)（`Cloudflare/clef` 原始服务输出，as-served，无额外温度校准）。

| 数据集        | 质量指标      | 公开参考 |     适配后 baseline |            当前实测 |
| ------------- | ------------- | -------: | ------------------: | ------------------: |
| OpenDecisions | Calibration   |     82.3 |             82.7941 |             82.4254 |
| OpenDecisions | Intelligence  |     65.8 |             66.0568 |             66.0872 |
| OpenDecisions | Capability    |     74.1 |             74.4255 |             74.2563 |
| OpenDecisions | Accuracy      |        — | 0.8104（1906/2352） | 0.8095（1904/2352） |
| OpenDecisions | Score MAE     |        — |            0.401389 |            0.402579 |
| ERQA          | 总体 accuracy |        — |   0.5475（219/400） |   0.5450（218/400） |

当前实测取自 [metrics.csv](metrics.csv)；适配后 baseline 详见 [OpenDecisions 评测记录](docs/opendecisions-baseline-2026-11-13.md) 与 [ERQA 评测记录](docs/erqa-baseline-2026-11-14.md)。三项综合指标按 0～100 分、accuracy 按 0～1 记录，Score MAE 使用等级单位（588 条 Score 题）；`—` 表示公开榜单未披露该子项。评测全程采用原始服务输出（未做温度校准）且严格保持 **BF16 非量化**；OpenDecisions 三项指标（Δ 分别为 −0.37 / +0.03 / −0.17 pp）与 ERQA 总体准确率（Δ −0.25 pp）相对适配后 baseline 的绝对波动均远小于 2PP 门槛。

## 性能指标

以下四组测试均在**同一常驻 `start.sh` 服务进程（零重启）**上顺序完成：

### OpenDecisions S1（单并发，全量 2,352）

配置：`TASK_QUEUE_ENABLE=2`、Tier 2 prefix-state cache（32 GiB/rank、`window=500`）、TP2、BF16、`max_num_seqs=16`。第 1 轮 warmup（冷填充）不计入，正式连续运行 3 轮取算术平均；每轮 2,352/2,352 成功、0 失败。

| 指标        | warmup（冷，排除） |     R1 |     R2 |     R3 | 正式 3 轮平均 |
| ----------- | -----------------: | -----: | -----: | -----: | ------------: |
| decisions/s |              8.839 |  9.774 |  9.741 |  9.712 |     **9.742** |
| p50 (ms)    |              99.30 |  99.24 |  99.24 |  99.88 |     **99.45** |
| p95 (ms)    |             167.31 | 118.57 | 118.49 | 118.56 |    **118.54** |
| mean (ms)   |             112.97 | 102.16 | 102.52 | 102.83 |    **102.51** |

### OpenDecisions S2（最大吞吐，B16 并发扫描）

配置：`max_num_seqs=16`、`max_num_batched_tokens=16384`；复用上述热态服务，扫描客户端并发 `C=1..128`，每点全量 2,352 题、0 失败。

| 客户端并发 C |     1 |      2 |      4 |      8 |     16 |     32 |         64 |    128 |
| ------------ | ----: | -----: | -----: | -----: | -----: | -----: | ---------: | -----: |
| decisions/s  | 9.691 | 10.143 | 14.223 | 17.361 | 19.981 | 21.544 | **21.578** | 21.560 |
| p95 (ms)     | 118.5 |  220.5 |  399.3 |  658.4 | 1124.1 | 2068.9 |     3969.0 | 7375.4 |

最大有效吞吐点为 **B16 C64**：三次独立 C64 平均 **21.583 decisions/s**（p50 3,033.1 ms、p95 3,968.5 ms，0 失败）。`C32` 吞吐仅低 0.2%（21.544 decisions/s），而 p95 降至约一半（2,068.9 ms）；若评测带有尾部延迟约束，推荐选用 `C32`。

### ERQA S1（单并发，全量 400）

配置：`max_model_len=16384`、`max_num_seqs=16`、`max_num_batched_tokens=16384`、`image=16`、`encoder_cache_tokens=262144`、Tier 2 `window=500`；报文格式 `state-list`、端点 `POST /v1/systemone`。第 1 轮 warmup 不计入，正式 3 轮取平均；每轮 400/400 成功、0 失败/重试。

| 指标        | warmup（交接/冷，排除） |      R1 |      R2 |      R3 | 正式 3 轮平均 |
| ----------- | ----------------------: | ------: | ------: | ------: | ------------: |
| decisions/s |                   4.499 |   8.459 |   8.538 |   8.446 |     **8.481** |
| p50 (ms)    |                   206.3 |   113.4 |   112.4 |   112.8 |    **112.86** |
| p95 (ms)    |                   314.2 |   122.6 |   121.1 |   123.4 |    **122.37** |
| mean (ms)   |                   217.6 |   114.4 |   113.3 |   114.4 |    **114.05** |
| accuracy    |                 216/400 | 218/400 | 218/400 | 218/400 |   **218/400** |

Warmup 轮的较高延迟来自同进程 `OD → ERQA` 跨数据集交接淘汰与首轮状态捕获（populate）；凭借 `window=500` 老化窗口，旧数据集前缀在首轮内平滑换出，正式三轮命中稳定且逐题预测完全一致（accuracy 218/400 = 0.545，2PP 内）。

### ERQA S2（最大吞吐，C8/C16）

使用 [erqa_s2_concurrent.py](src/runtime/erqa_s2_concurrent.py)（单次尝试、有界并发），在同一热态服务上运行全量 400 题。`C16` 为最大吞吐点，三次平均 **21.439 decisions/s**（中位 21.863、最高 22.246；R3 含单条约 2.1 s 尾部离群）。

|                并发 |      dec/s |  p50 (ms) |  p95 (ms) | mean (ms) |    accuracy |
| ------------------: | ---------: | --------: | --------: | --------: | ----------: |
|      C8（2 次平均） |     17.821 |     431.1 |     476.8 |     431.1 |     216/400 |
| **C16（3 次平均）** | **21.439** | **697.5** | **823.2** | **712.6** | **216/400** |

`C16` 三次预测逐题一致；`C8` 与 `C16` 间存在 4/400 题因动态合批组成不同产生的微小边界差异，两者准确率均为 216/400 = 0.540（2PP 内）。

原始日志与校验归档：[logs/final-eval/od-s1](logs/final-eval/od-s1/) · [logs/final-eval/od-s2](logs/final-eval/od-s2/) · [logs/final-eval/erqa-s1](logs/final-eval/erqa-s1/) · [logs/final-eval/erqa-s2](logs/final-eval/erqa-s2/)；汇总表见 [results.csv](results.csv)（每个数据集一行：S1 取正式 3 轮平均，S2 取最大吞吐点平均）。

## 环境准备

### 1. 构建并激活项目自包含环境

克隆本仓库后，执行 `scripts/setup_env.sh` 在仓库内自动构建隔离的 `.venv`（包含 Python 3.12.16、`torch_npu`、固定提交的 `vLLM` / `vLLM-Ascend` 自定义算子、`EvalScope` 及 `modelscope` CLI，全程约 15–18 分钟）：

```bash
# 通过 GitHub 克隆
git clone https://github.com/Iamnotphage/Clef-27B-ascend.git
# 或通过 AtomGit / GitCode 克隆
git clone https://atomgit.com/m0_74196153/Clef-27B-ascend.git

cd Clef-27B-ascend

# 一键下载依赖并编译相关源码（幂等执行，完成后写入 .venv/.clef-env-complete）
bash scripts/setup_env.sh

# 激活 CANN / ATB 与项目虚拟环境（后续每个新开终端均需执行）
source scripts/activate_env.sh
```

验证环境激活无误：

```bash
which python      # <repo>/.venv/bin/python
python --version  # Python 3.12.16
python -m pip check
```

### 2. 下载模型权重与评测数据集

执行 `source scripts/activate_env.sh` 后，环境中已内置 `modelscope` 命令行工具，可直接下载模型权重与两套评测数据集（以下以存放至 `/workspace/models` 和 `/workspace/datasets` 为例，可按实际磁盘路径调整）：

```bash
# 1) 下载 Cloudflare/clef 模型权重（约 55 GB，含完整主干 12 分片与联合决策头）
modelscope download --model Cloudflare/clef --local_dir /workspace/models/Clef-27B
# 也可调用仓库封装脚本：bash scripts/download_model.sh /workspace/models/Clef-27B

# 2) 下载 OpenDecisions 数据集（ModelScope 官方镜像，含 opendecisions.jsonl 与作者 score.py）
OD_DATA_DIR=/workspace/datasets/OpenDecisions
modelscope download --dataset evalscope/OpenDecisions --local_dir "$OD_DATA_DIR"
ln -sf "$OD_DATA_DIR/opendecisions.jsonl" "$OD_DATA_DIR/open-decisions2352.jsonl"

# 从数据集目录直接生成标准库压测脚本所需的 2,352 条请求序列 JSON（零外部依赖）
python - "$OD_DATA_DIR" <<'PY'
import json, pathlib, sys
d = pathlib.Path(sys.argv[1])
samples = []
for i, line in enumerate((d / "opendecisions.jsonl").open(encoding="utf-8")):
    if not line.strip(): continue
    t = json.loads(line)
    q = {k: t["question"][k] for k in ("type", "instructions", "criteria") if t["question"].get(k) is not None}
    samples.append({"name": f"od-{i:04d}", "request": {"model": "clef", "state": t["state"], "questions": {"answer": q}}})
(d / "all2352.json").write_text(json.dumps({"samples": samples}, ensure_ascii=False) + "\n", encoding="utf-8")
print(f"Generated {len(samples)} requests -> {d / 'all2352.json'}")
PY

# 3) 下载并解包 ERQA 图文数据集（ModelScope 镜像 + 官方 TFRecord 标准库解包）
ERQA_DATA_DIR=/workspace/datasets/ERQA
# 若使用 EvalScope Hub 数据源可直接下载：
modelscope download --dataset evalscope/ERQA --local_dir /workspace/datasets/ERQA-ms
# S1/S2 原生图文评测脚本读取由官方 erqa.tfrecord（commit 5bba488）解包生成的 samples.jsonl + images/ 目录：
curl -fL https://raw.githubusercontent.com/embodiedreasoning/ERQA/5bba488252c5b345a43335bfb9239b6bf12bd91c/data/erqa.tfrecord \
  -o /workspace/datasets/erqa.tfrecord
python scripts/official_eval/erqa_extract_tfrecord.py \
  --tfrecord /workspace/datasets/erqa.tfrecord \
  --output-dir "$ERQA_DATA_DIR"
```

> 说明：仓库 [logs/final-eval/od-s1/all2352.json](logs/final-eval/od-s1/all2352.json) 亦归档了上述由 `opendecisions.jsonl`（SHA-256 `78614972…`）生成的完全一致的 2,352 条请求载荷，方便离线直接校验。

## 启动命令

激活环境后，通过根目录 `start.sh` 启动单一常驻推理服务（默认端口 `18143`，默认启用 `s1` 统一配置）。请将 `--model-dir` 与 `--output-dir` 替换为实际目录：

```bash
source scripts/activate_env.sh

# 方式 A：前台启动（推荐用于独立服务终端，自动显示 12 分片加载进度并在就绪时打印 Service ready）
bash start.sh \
  --model-dir /workspace/models/Clef-27B \
  --output-dir /tmp/final-eval

# 方式 B：后台启动并实时追踪详细运行日志（模型加载与运行时补丁自检约需 1.5～3 分钟）
bash start.sh \
  --model-dir /workspace/models/Clef-27B \
  --output-dir /tmp/final-eval &
tail -f /tmp/final-eval/service.log
```

服务就绪后（`<output-dir>/launch-manifest.json` 会完整记录生效参数、补丁 SHA-256 与静态内存预算决策），可向 `http://127.0.0.1:18143` 发送健康检查与推理请求（配合 `jq` 格式化输出）：

```bash
# 1. 健康检查
curl -sf http://127.0.0.1:18143/health | jq .

# 2. 查询服务配置与 Tier 2 缓存开启状态（确认 prefix_state_cache: true）
curl -sf http://127.0.0.1:18143/v1/decide/info | jq .

# 3. 发送单条结构化决策 Demo 请求
curl -sf http://127.0.0.1:18143/v1/systemone \
  -H "Content-Type: application/json" \
  -d '{
    "model": "clef",
    "state": "Pretty-print this JSON for me: {\"id\":7,\"tags\":[\"a\",\"b\"],\"ok\":true}",
    "questions": {
      "answer": {
        "type": "choice",
        "instructions": "Pick the model tier the router should send this request to.",
        "criteria": {
          "nano": "Trivial formatting or lookup with one clear output.",
          "standard": "Everyday explanation or simple code, no investigation.",
          "pro": "Investigation or analysis across several inputs."
        }
      }
    }
  }' | jq .
```

## 复现步骤

在开始复现前，请确保：
1. 当前评测终端已执行 `source scripts/activate_env.sh`；
2. `bash start.sh` 启动的服务已就绪（`curl -sf http://127.0.0.1:18143/health` 返回 HTTP 200）；
3. **全程保持单一服务进程零重启**：OpenDecisions（S1 / S2 / 质量）与 ERQA（S1 / S2 / 质量）均直接请求同一 `http://127.0.0.1:18143` 实例，无需修改环境变量或重启服务。

以下复现命令统一使用环境准备阶段设置的数据集目录变量：

```bash
OD_DATA_DIR=/workspace/datasets/OpenDecisions
ERQA_DATA_DIR=/workspace/datasets/ERQA
```

### OpenDecisions S1 全量评测

使用标准库压测器运行全量 2,352 条请求（第 1 轮 `warmup` 排除，正式计 `r1/r2/r3` 三轮平均；逐条输出重定向至 `.log`，每轮结束后用 `jq` 打印核心吞吐与分位延迟）：

```bash
OD_SAMPLE="$OD_DATA_DIR/all2352.json"   # 亦可使用仓库归档 "$PWD/logs/final-eval/od-s1/all2352.json"

for r in warmup r1 r2 r3; do
  python scripts/benchmark_systemone.py \
    --base-url http://127.0.0.1:18143 \
    --sample-file "$OD_SAMPLE" \
    --warmup 0 --requests 2352 --concurrency 1 \
    --report "/tmp/od-s1-${r}.json" > "/tmp/od-s1-${r}.log"
  jq -c --arg round "$r" '{
    round: $round,
    passed: .passed,
    dps: (.summary.decisions_per_second | .*1000 | round / 1000),
    p50_ms: (.summary.successful_request_latency_ms.p50 | .*100 | round / 100),
    p95_ms: (.summary.successful_request_latency_ms.p95 | .*100 | round / 100),
    mean_ms: (.summary.successful_request_latency_ms.mean | .*100 | round / 100)
  }' "/tmp/od-s1-${r}.json"
done
```

### OpenDecisions S2 全量评测

在同一热态服务（`max_num_seqs=16`、`max_num_batched_tokens=16384`）上扫描并发 `C=1..128`，并在最大吞吐点 `C=64` 额外重复两次取三轮平均：

```bash
OD_SAMPLE="$OD_DATA_DIR/all2352.json"

for c in 1 2 4 8 16 32 64 128; do
  python scripts/benchmark_systemone.py \
    --base-url http://127.0.0.1:18143 \
    --sample-file "$OD_SAMPLE" \
    --warmup 0 --requests 2352 --concurrency "$c" \
    --report "/tmp/od-s2-c${c}.json" > "/tmp/od-s2-c${c}.log"
  jq -c --argjson c "$c" '{
    concurrency: $c,
    dps: (.summary.decisions_per_second | .*1000 | round / 1000),
    p50_ms: (.summary.successful_request_latency_ms.p50 | .*10 | round / 10),
    p95_ms: (.summary.successful_request_latency_ms.p95 | .*10 | round / 10)
  }' "/tmp/od-s2-c${c}.json"
done

# 最大吞吐点 C64 再重复两次，取三次算术平均
for r in 2 3; do
  python scripts/benchmark_systemone.py \
    --base-url http://127.0.0.1:18143 \
    --sample-file "$OD_SAMPLE" \
    --warmup 0 --requests 2352 --concurrency 64 \
    --report "/tmp/od-s2-c64-r${r}.json" > "/tmp/od-s2-c64-r${r}.log"
done
```

### ERQA S1 全量评测

仍对同一服务进程直接发起 ERQA 官方 400 题单并发评测（`warmup` 完成跨数据集缓存交接后排除，取 `r1/r2/r3` 三轮平均）：

```bash
for r in warmup r1 r2 r3; do
  python scripts/official_eval/erqa_baseline.py --scope full \
    --api-url http://127.0.0.1:18143/v1 \
    --wire-format state-list --endpoint /systemone \
    --data-dir "$ERQA_DATA_DIR" \
    --output-dir "/tmp/erqa-s1-${r}" > "/tmp/erqa-s1-${r}.log"
  jq -c --arg round "$r" '{
    round: $round,
    accuracy: .accuracy,
    dps: (.throughput.decisions_per_second | .*1000 | round / 1000),
    p50_ms: (.latency_ms.p50 | .*100 | round / 100),
    p95_ms: (.latency_ms.p95 | .*100 | round / 100),
    mean_ms: (.latency_ms.mean | .*100 | round / 100)
  }' "/tmp/erqa-s1-${r}/summary.json"
done
```

### ERQA S2 全量评测

在同一热态服务上以有界并发 `C=16`（及可选 `C=8`）运行三次全量 400 题取平均：

```bash
for r in 1 2 3; do
  PYTHONPATH="$PWD/scripts/official_eval" \
  python src/runtime/erqa_s2_concurrent.py \
    --data-dir "$ERQA_DATA_DIR" \
    --output-dir "/tmp/erqa-s2-c16-r${r}" \
    --api-url http://127.0.0.1:18143/v1 \
    --wire-format state-list --endpoint /systemone \
    --concurrency 16
done
```

### 质量指标评测

质量指标既可直接从上述 S1 评测保存的完整原始响应离线精确重算（零额外推理开销），也可通过 `EvalScope` 离线流水线独立运行：

```bash
# 1. OpenDecisions：调用数据集作者官方 score.py 离线重算 Calibration / Intelligence / Capability / Accuracy / Score MAE
python - "$OD_DATA_DIR" /tmp/od-s1-r1.json <<'PY'
import importlib.util, json, pathlib, sys
data_dir, run_path = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
tasks = [json.loads(l) for l in (data_dir / "opendecisions.jsonl").open(encoding="utf-8") if l.strip()]
reqs = sorted(json.loads(run_path.read_text(encoding="utf-8"))["requests"], key=lambda r: r["index"])
rows = []
for task, req in zip(tasks, reqs):
    ans = req["response"]["answers"]["answer"]
    probs = ans.get("probabilities") or {"yes": ans["noul"], "no": 1.0 - ans["noul"]}
    rows.append({"id": task["id"], "probs": probs})
spec = importlib.util.spec_from_file_location("od_score", data_dir / "score.py")
scorer = importlib.util.module_from_spec(spec); spec.loader.exec_module(scorer)
scores = scorer.score(tasks, rows)
outcomes = [scorer.item_outcome(t, r["probs"]) for t, r in zip(tasks, rows)]
valid = [o for o in outcomes if o["valid"]]
score_items = [o for o in valid if o["type"] == "score"]
print(json.dumps({
    "calibration": scores["calibration"],
    "intelligence": scores["intelligence"],
    "capability": scores["capability"],
    "accuracy": sum(o["correct"] for o in valid) / len(valid),
    "correct": f"{sum(o['correct'] for o in valid)}/{len(valid)}",
    "score_mae": sum(o["nmae"] * (o["k"] - 1) for o in score_items) / len(score_items),
}, indent=2))
PY

# （可选）通过 EvalScope 原生适配器独立发起全量 OpenDecisions 质量评测
python scripts/official_eval/open_decisions_baseline.py \
  --data-dir "$OD_DATA_DIR" \
  --api-url http://127.0.0.1:18143/v1 \
  --output-dir /tmp/od-evalscope-quality

# 2. ERQA：离线重算 400 题准确率、8 类分项指标并逐题校验请求结构化 SHA-256
PYTHONPATH="$PWD/scripts/official_eval" \
python scripts/official_eval/erqa_score.py \
  --run-dir /tmp/erqa-s1-r1 \
  --data-dir "$ERQA_DATA_DIR" \
  --verify-requests
```

## 优化概要

针对 Clef-27B“纯 prefill（`output_tokens=0`）、48 层 GDN 线性注意力 + 16 层 Full-Attention 混合主干 + 独立联合决策头”的架构特点与昇腾 910A3（24 AIC / 48 AIV）硬件特征，工程自底向上实施了 7 层非量化优化（全部保持纯 BF16 精度，严格满足 2PP 质量红线）。完整瓶颈诊断、算子设计与消融实验见 **[optimization.md](optimization.md)**。

| 优化层级 | 核心优化手段 | 数值与精度特性 | 端到端实测收益 |
| --- | --- | --- | --- |
| **1. 调度与算子缓存底座** | `ACLNN_CACHE_LIMIT=1000000` 消除 ~145 种动态长度后的 executor 缓存击穿；`TASK_QUEUE_ENABLE=2` 双级异步下发队列；`OMP_NUM_THREADS=1` 消除宿主线程争用；统一 `B16` 连续合批 | **逐位一致**（`max_diff = 0.0`） | 消除新 shape **+55 ms** 退化（S1 P95 `190.4 → 168.6 ms`）；`TASK_QUEUE=2` 再降 S1 mean **−7.7 ms**（吞吐 **+6.9%**） |
| **2. 主干 MLP 权重布局** | `Qwen2MoeMLP` 首次前向惰性将 `gate_up_proj` / `down_proj` 权重转为 `FRACTAL_NZ`（format 29），消除短序列 Cube MTE2 搬运瓶颈 | **逐位一致**（`max_diff = 0.0`） | 307 token MTE2 有效带宽 `509 → 677 GB/s`（**1.32×**）；全量 OD S1 P95 `168.6 → 166.0 ms`（mean **−3.1%**） |
| **3. GDN 决策专用数据路径与分块** | 跳过无 decode 场景下无人消费的循环末态写回与反向重算（`BIC` 数据路径）；GDN `chunk_size` 由 `64` 升级为硬件最优的 `128` | `BIC` **逐位一致**；`chunk128` 纯 BF16 重排（2PP 内，Cap `74.25`） | `BIC` 降 S1 P95 **−7.0%**（`−13.8 ms`）；叠加 `chunk128` 后 S1 P95 进一步降至 **158.3 ms** |
| **4. Ascend C 48-AIV Conv+QKV 双模融合算子** | 单序列 `ClefGdnConvQkv` v4（`as_strided` 免 `[T,8192]` 切片拷贝，因果卷积 + FLA L2 Norm + V 直拷合一，铺满 **48 AIV**，`cap=16384`）+ 多序列 `ClefGdnConvQkvVarlen`（`cu_seqlens` 严格边界隔离，`tile=64`） | 纯 BF16 融合（v3/v4/varlen 离线逐位一致、144 组毒化零串扰；全量 2PP 内） | 单算子提速 **3.2–4.5×**；无缓存主线 OD S1 P95 压至 **154.1 ms**，OD S2 C64 提至 **18.04 dec/s**，ERQA S2 提至 **8.63 dec/s** |
| **5. 多模态前端与联合决策头融合** | `/v1/systemone` 原生支持图文 `state` 内容列表（`A2` 模式，文本零开销、异常 Fail-Closed）；JointHead 单字段快路径 + `npu_fusion_attention` 融合 | 单字段快路径逐位一致；融合注意力在 2PP 内 | 消除 JointHead 冗余字段循环与小算子下发（S1 P95 **−2.9%**），实现单进程原生 `/v1/systemone` 图文与文本共池 |
| **6. Tier 1 视觉缓存 + Tier 2 混合前缀状态缓存** | Tier 1 扩容 `encoder_cache_tokens=262144`；自研 Tier 2 跨请求缓存 128 对齐前缀的 `16 层 Full-Attn KV + 48 层 GDN (conv, ssm) + JointHead prefix_memory`：1-Pass Tap 设备侧零税填充、分组 varlen SSM 异步捕获、M2 两遍混批后缀续算、`aged_keep_longest (window=500)` 跨集平滑交接 | Populate 与 OFF **逐位一致**；同进程命中复用自洽 `0.0`；OD Cap `74.2563`、ERQA `0.5450`（均在 2PP 内） | OD S1 P95 `154.1 → 118.5 ms`（**−23.1%**），OD S2 C64 `18.04 → 21.58 dec/s`（**+19.6%**）；ERQA S1 P95 `244.7 → 122.4 ms`（**−50.0%**），ERQA S2 C16 `8.63 → 21.44 dec/s`（**+148.5%**） |
| **7. 静态内存自适应安全钳制** | 启动时读取 `min(MemTotal, cgroup memory.max)`，按 `≥112 GiB` / `48–112 GiB` / `≤48 GiB` 三档自动调节每卡缓存配额与 Pinned Ring 槽位（`≤48 GiB` 自动关闭 Tier 2 免除 10 GiB 锁页分配） | 不改变算子数值路径，满水位或关闭时自动 Fail-Closed 回退常规前向 | 保障未知规格评测容器（32～128+ GiB RAM）100% 零 OOM 安全启动与平滑降级 |

### 优化总览图

![全栈渐进优化消融](imgs/fig1_optimization_ablation.png)

> **图 1**：6 个阶段的 S1 p95（左）与 S2 最大吞吐（右）。最终相对 baseline：OD −45.4% / 3.78×，ERQA −62.2% / 5.42×。数据来源为 `optimization.md` 第 4 节与 `logs/final-eval/`。


## 附录

### 1. 可选主机激活缓存方案（K32 / K56 / K64 探索归档）

除默认开启、同时支持共享前缀文本与多图混批的 **Tier 2 混合前缀状态缓存**外，仓库在 [src/runtime/k32_activation_cache](src/runtime/k32_activation_cache/README.md) 中还保留了一套针对**纯文本、单序列（B1）、完整 Prompt 精确重复场景**的显式可选（opt-in）主机激活缓存方案。该方案将第 31 层（`K32`）、第 55 层（`K56`）或末层 Final Norm（`K64`）输出的全序列 BF16 隐藏态缓存于主机内存（双卡逻辑上限合计 ≤41 GiB，保留 ≥10 GiB cgroup 余量，任何图文、合批或越界自动旁路回原前向）。

在 OpenDecisions 全量 2,352 题同进程三轮测试（R1 空池自然填充 + R2/R3 同进程自然复用）中，各配置实测对比如下（证据归档于 [logs/dataset-profiles-20261123](logs/dataset-profiles-20261123/)）：

| 配置（显式 opt-in） | 缓存切分点 | 并发模式 | 冷首轮 P95 / 吞吐 | 暖复用 P95 (ms) | 暖复用吞吐 (dec/s) | 质量指标（Capability / Acc） |
| --- | --- | --- | ---: | ---: | ---: | --- |
| `od-s1-k32` | Layer 31 `(hidden, residual)` | B1 / C1 | 169.06 ms / 7.85 | **86.24**（R2） / **85.99**（R3） | **14.91** / **15.12** | 74.2477 / 1904（逐题同 ON） |
| `od-s1-k56` | Layer 55 `(hidden, residual)` | B1 / C1 | 169.30 ms / 8.24 | **32.57**（R2） / **32.16**（R3） | **35.70** / **36.20** | 74.2477 / 1904（逐题同 ON） |
| `od-s1-k64` | Post-FinalNorm `hidden`（保留真实 Pooler + JointHead） | B1 / C1（640 题筛选） | 183.67 ms / 8.13 | **17.14** / **17.18** | **64.73** / **64.49** | 逐题概率完全一致 |
| `od-s1-k64` | Post-FinalNorm `hidden`（全量 2,352 乱序 ABA） | B1 / C2 | 306.27 ms / 8.36 | **27.16**（R2） / **27.18**（R3） | **79.33** / **77.89** | 74.2477 / 1904（2PP 内） |

> 注：由于 `K32/K56/K64` 仅命中完整 Prompt 相同的重复请求且限定 B1 文本模式，而比赛主入口要求单一配置同时高效服务共享 `STATE` 不同 `SCHEMA`、多模态 `ERQA` 以及 `B16` 高并发合批，因此默认 `start.sh` 采用通用性更强的 **Tier 2 前缀状态缓存**，未启用 `K32/K56/K64`。

### 2. 性能剖析（Profiling）摘要

基于昇腾 910A3 双卡 `torch_npu.profiler` 与微基准实测（详见 [optimization.md](optimization.md)、[HBM 带宽报告](docs/hbm-bandwidth-2026-11-15.md)、[MLP Cube 上限报告](docs/mlp-cube-ceiling-2026-11-15.md) 及 [Wave 2 Profile 报告](docs/perf-wave2-20261118.md)）：

- **硬件实测屋脊（Roofline）**：单逻辑 NPU 纯 HBM 读带宽 **1.26–1.32 TB/s**，BF16 Cube 矩阵乘峰值 **295–320 TFLOPS**（真实 TP2 MLP 权重冷态已达 **285.9–303.4 TFLOPS**，贴近硬件计算上限；`npu_add_rms_norm` 达 **1,117–1,266 GB/s**，贴近 HBM 带宽上限）。
- **当前版本 967-token 全长 Populate 拆解（双卡 Union，每请求）**：设备 compute **141.1 ms**（MatMul 91.2 ms、GDN core 27.6 ms、Fused Conv+QKV 2.6 ms、Full-Attn 1.6 ms、Norm+Elementwise+Other 18.0 ms），非重叠 HCCL/传输 **18.4 ms**，设备总预算 **159.5 ms**；Tier 2 命中（199-token 后缀）设备 compute 降至 **67.0 ms**、总预算 **89.8 ms**。无 Profiler 的当前版本 Populate p95 为 **167.2 ms**（5 条全新 967-token 前缀），命中路径 S1 p95 **118.5 ms**。详见 [Profile 与 S1 Roofline 报告](docs/profile-tier2-2026-11-26.md)。
- **Tier 2 命中后缀前向收益机理**：OpenDecisions 共享上下文与 ERQA 多图前缀（`L` 对齐至 128 的倍数，ERQA 最高达数千视觉 token）在首轮全长前向中被一次性捕获；后续同前缀请求仅对短后缀 `S = T - L` 执行 64 层后缀计算与单次 `FIA TND` 注意力，将前缀 Cube/GDN 计算与多图 ViT 处理移出关键路径（W2 MatMul 仍为 42.8 ms，约为全量 91.2 ms 的 47%，说明每层固定开销仍在），使 OD 与 ERQA 的热态 S1 P95 收敛至 **~118–122 ms**，S2 并发吞吐突破 **21.4+ decisions/s**。
- 当前版本的补充算子级对比与 S1 Roofline 详见 **[Profile 与 S1 Roofline 报告](docs/profile-tier2-2026-11-26.md)**；四张设备侧图表如下。

![双卡算子级耗时拆解](imgs/fig2_operator_breakdown.png)

> **图 2**：W1–W4 双卡设备 union 时间按算子类别拆分。OD 全长 159 ms / 命中 90 ms；ERQA 全长 232 ms（含 ViT 16 ms）/ 命中 110 ms。

![S2 并发扩展与帕累托](imgs/fig3_s2_concurrency_pareto.png)

> **图 3**：OD C=1..128 全扫描与 ERQA C8/C16 的吞吐-并发（左）和 p95-吞吐帕累托（右）；红色虚线为 Tier 2 前基线（OD 18.04、ERQA 8.63 dec/s）。

![前缀切分与跨数据集交接](imgs/fig4_prefix_handover_analysis.png)

> **图 4**：左图为 128 对齐前缀 `L` 与后缀 `S=T−L` 的代表点；右图为同进程 `OD → ERQA` 交接，`window=3000` 旧池 8.08 dec/s，`window=500` 两轮 21.36/22.30 dec/s、末态 0 miss。

![S1 Roofline](imgs/fig5_s1_roofline.png)

> **图 5**：967-token 全长（左）与 199-token 后缀命中（右）的 S1 分层预算。全长实用设备屋脊约 145 ms、端到端约 154 ms，实测 154.1 ms 已贴住屋脊；命中路径受 Host dispatch 105 ms 限制，实测 118.5 ms。

### 3. 最终采用优化项源码位置索引

| 模块分类 | 源码 / 安装器路径 | 作用说明 |
| --- | --- | --- |
| **服务入口与静态内存策略** | [start.sh](start.sh)、[inference.py](inference.py)、[src/runtime/startup_progress.py](src/runtime/startup_progress.py) | 环境变量汇总、SHA-256 门控补丁链编排、静态 RAM 三档自适应钳制、终端加载进度与 `launch-manifest.json` 审计 |
| **单序列 Conv+QKV 48-AIV 算子** | [src/ascendc/kernels/clef_gdn_conv_qkv/](src/ascendc/kernels/clef_gdn_conv_qkv/)、[src/ascendc/vendors/clef_conv_qkv/](src/ascendc/vendors/clef_conv_qkv/)、[src/runtime/apply_conv_qkv_patch.py](src/runtime/apply_conv_qkv_patch.py) | `ClefGdnConvQkv` v4 算子源码、预编译二进制包与 `gdn.py` 安装器（`as_strided` 免拷贝 + 48 AIV tiling） |
| **多序列 Varlen Conv+QKV 算子** | [src/ascendc/kernels/clef_gdn_conv_qkv_varlen/](src/ascendc/kernels/clef_gdn_conv_qkv_varlen/)、[src/ascendc/vendors/clef_conv_qkv_varlen/](src/ascendc/vendors/clef_conv_qkv_varlen/)、[src/runtime/apply_conv_qkv_varlen_patch.py](src/runtime/apply_conv_qkv_varlen_patch.py) | `ClefGdnConvQkvVarlen` 多序列融合核（`cu_seqlens` 序列隔离、`tile=64`），加速 S2 合批与 Tier 2 混批 |
| **GDN 决策路径 / Chunk128 / MLP NZ** | [src/runtime/apply_gdn_decision_patch.py](src/runtime/apply_gdn_decision_patch.py)、[src/runtime/gdn_decision_flags.py](src/runtime/gdn_decision_flags.py)、[src/runtime/apply_gdn_chunk128_patch.py](src/runtime/apply_gdn_chunk128_patch.py) | GDN 决策专用 `BIC` 路径、`chunk_size=128` 切换及 `Qwen2MoeMLP` 权重惰性 `FRACTAL_NZ` 转换 |
| **联合决策头与多模态 State** | [src/runtime/joint_fused_attention/clef_joint_schema.py](src/runtime/joint_fused_attention/clef_joint_schema.py)、[src/runtime/apply_joint_fused_attention.py](src/runtime/apply_joint_fused_attention.py)、[src/runtime/systemone_mm_state.py](src/runtime/systemone_mm_state.py)、[src/runtime/apply_systemone_mm_state_patch.py](src/runtime/apply_systemone_mm_state_patch.py) | JointSchemaHead 单字段快路径 + `npu_fusion_attention`；`/v1/systemone` 图文 `state-list`（`A2` 模式）归一化 |
| **Tier 2 混合前缀状态缓存** | [src/runtime/tiered_prefix_cache/engine_hooks.py](src/runtime/tiered_prefix_cache/engine_hooks.py)、[host_store.py](src/runtime/tiered_prefix_cache/host_store.py)、[async_transfer.py](src/runtime/tiered_prefix_cache/async_transfer.py)、[prefix_key.py](src/runtime/tiered_prefix_cache/prefix_key.py)、[src/runtime/apply_tiered_cache_patch.py](src/runtime/apply_tiered_cache_patch.py) | 1-Pass Tap 捕获、分组 varlen SSM 提取、8×640 MiB 异步流水线、M2 两遍混批续算与 `aged_keep_longest (window=500)` 主机池管理 |
| **上游固定源码构建补丁** | [src/patches/](src/patches/)、[scripts/setup_env.sh](scripts/setup_env.sh) | TP2 词表并行归约契约修复、OpenCV/Triton-Ascend 构建兼容与项目自包含 `.venv` 编译脚本 |
