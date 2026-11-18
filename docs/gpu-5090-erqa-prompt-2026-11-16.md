# 5090 GPU 执行 Prompt：Clef-27B + ERQA 400 题质量参考

> 这是交给 `lab-5090` 上执行 agent 的完整 prompt。目标是下载官方 Clef-27B 与 ERQA 官方数据，跑 ERQA 400 题，输出质量指标；性能可记录，但不作为 A3 NPU 调优依据。

## 0. 机器与环境（已确认）

| 项 | 值 |
| --- | --- |
| 主机 | `lab-5090`（`bangbanglab-5090`） |
| GPU | 4 × NVIDIA GeForce RTX 5090，每张 **32,607 MiB** |
| 驱动 | 580.126.20 |
| 系统 Python | 3.12.3 |
| 工具 | `/snap/bin/uv`、`/home/yangchen/miniconda3/bin/conda` |
| 允许目录 | `/disk/yangchen`、`/home/yangchen` |
| 磁盘 | `/disk` 约 4.1 TiB 可用；`/home` 约 2.1 TiB 可用 |
| 网络 | ModelScope / GitHub / download.pytorch.org 均可达 |
| 注意 | 系统 `nvcc` 是 CUDA 12.0，**不能直接编译 sm_120（5090）**；需要 CUDA 12.8+ 工具链和 cu128 的 PyTorch |

当前 4 张卡都有其他用户进程，**不要 kill 任何非本任务进程**。开始前先看空闲显存，等足够的卡空闲再跑；如果不足，报告并等待。

## 1. 目标与约束

### 目标

1. 从 ModelScope 下载官方 `Cloudflare/clef` 到 `/disk/yangchen/models/Clef-27B`；
2. 从官方 ERQA 仓库下载 `5bba488` 的 TFRecord 到 `/disk/yangchen/datasets/erqa-official-20261114`；
3. 在 4×5090、**BF16、不量化** 下跑官方 ERQA 400 题；
4. 输出总体 accuracy、单图/多图、8 类分项、逐题 prediction 和服务配置；
5. 记录性能（wall、p50/p95）仅作参考，不用于 A3 调优比较。

### 硬约束

- **禁止量化**：不得使用 W8A8/W8A16/INT8/FP8/4bit；必须 BF16。
- **禁止** kill 同门进程、占用未授权 GPU、写 `/disk/yangchen` 和 `/home/yangchen` 之外的目录。
- 模型和数据都放 `/disk/yangchen`，不要放 `/home`。
- 使用与官方一致的 ERQA 数据、图片顺序、prompt 模板、答案字母映射；不做温度校准。
- 不修改 A3 工程或它的提交产物；GPU 结果只作为参考，不写成正式 `metrics.csv` 的 A3 成绩。
- 如果 vLLM CUDA 构建失败，不要偷偷改成量化或单卡 CPU 跑 400 题；先报告 blocker。

## 2. 工作目录

```bash
export WORK=/disk/yangchen/clef-gpu
export MODEL=/disk/yangchen/models/Clef-27B
export DATA=/disk/yangchen/datasets/erqa-official-20261114
export RESULTS=$WORK/results/erqa-400
mkdir -p "$WORK" "$MODEL" "$DATA" "$RESULTS" "$WORK/logs"
```

已预置：

```text
$WORK/Clef-27B/logs/model_manifest.json   # ModelScope 官方文件清单（含 SHA-256）
$WORK/scripts/official_eval/              # erqa_extract_tfrecord.py / erqa_baseline.py / erqa_score.py / erqa_protocol.py
```

为方便使用，可复制到根目录：

```bash
cp "$WORK/Clef-27B/logs/model_manifest.json" "$WORK/model_manifest.json"
```

## 3. GPU 占用检查

```bash
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
  --format=csv,noheader
```

- 若 4 张卡都基本空闲（每张 used < 2 GiB），用 TP4；
- 若只有 2 张空闲，可先做 smoke，但 ERQA 16K/16 图可能 OOM，不要强跑；
- 若不足 2 张，停止并报告，不要 kill 其他进程；
- 记录开始时的 `nvidia-smi -L` 和每卡 used MiB 到 `$WORK/logs/gpu-before.txt`。

TF/TP 使用 `CUDA_VISIBLE_DEVICES=0,1,2,3` 前请确认这 4 个 index 都是空闲的；如果只有部分空闲，用对应的 index 并调整 `--tensor-parallel-size`，不要假设卡号固定。

## 4. 下载并校验模型

### 4.1 创建下载环境

```bash
/home/yangchen/miniconda3/bin/conda create -p /disk/yangchen/venvs/clef-download python=3.12 -y
source /home/yangchen/miniconda3/bin/activate /disk/yangchen/venvs/clef-download
pip install -U pip modelscope
```

### 4.2 ModelScope 下载

```bash
modelscope download \
  --model Cloudflare/clef \
  --local_dir "$MODEL" \
  --revision master
```

如果 CLI 参数不同，使用 Python API：

```bash
python - <<'PY'
from modelscope import snapshot_download
snapshot_download(
    "Cloudflare/clef",
    local_dir="/disk/yangchen/models/Clef-27B",
    revision="master",
)
PY
```

模型约 **52 GB**，下载到 `/disk/yangchen`。不要用 `/home`，不要下载到当前目录。

### 4.3 必需文件

```text
config.json
generation_config.json
model-00001-of-00012.safetensors ... model-00012-of-00012.safetensors
model.safetensors.index.json
joint_head.safetensors
joint_head_config.json
joint_schema_model.py
tokenizer.json
tokenizer_config.json
chat_template.jinja
processor_config.json
README.md
LICENSE
```

检查：

```bash
ls -lh "$MODEL" | head -30
python - <<'PY'
import json, struct, glob, collections, os
root = "/disk/yangchen/models/Clef-27B"
config = json.load(open(f"{root}/config.json"))
assert config.get("quantization_config") is None, "模型带量化配置，停止"
assert config.get("dtype") == "bfloat16", config.get("dtype")
files = sorted(glob.glob(f"{root}/model-*.safetensors"))
assert len(files) == 12, len(files)
counts = collections.Counter()
total = 0
for path in files + [f"{root}/joint_head.safetensors"]:
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
    for name, info in header.items():
        if name != "__metadata__":
            counts[info["dtype"]] += 1
            total += 1
print("tensors", total, "dtypes", dict(counts))
assert set(counts) == {"BF16"}, counts
print("model dtype check: OK")
PY
```

预期：12 个主干分片 + 1 个联合头，全部 **BF16**，无 `quantization_config`，联合头 122 个张量。

### 4.4 逐文件 SHA-256

`$WORK/model_manifest.json` 是 ModelScope API 的文件清单，字段为 `Data.Files[]`，每个 blob 有 `Path`、`Size`、`Sha256`。至少校验：

- `config.json`
- `joint_head_config.json`
- `joint_head.safetensors`
- `model.safetensors.index.json`
- 全部 `model-*-of-00012.safetensors`

示例：

```bash
python - <<'PY'
import hashlib, json
from pathlib import Path
root = Path("/disk/yangchen/models/Clef-27B")
manifest = json.load(open("/disk/yangchen/clef-gpu/model_manifest.json"))
files = manifest["Data"]["Files"]
checked = 0
for item in files:
    path = root / item["Path"]
    if item.get("Type") != "blob" or not path.is_file():
        continue
    if item["Path"].startswith(("model-", "joint_head", "config.json", "model.safetensors.index.json")):
        size = path.stat().st_size
        assert size == item["Size"], (path, size, item["Size"])
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest == item["Sha256"], (path, digest, item["Sha256"])
        checked += 1
print("verified files", checked)
PY
```

如果 ModelScope 的 revision/文件与清单不一致，停止并报告，不要带着未知权重继续。

## 5. 下载并解析 ERQA 官方数据

### 5.1 固定官方 commit

```bash
git clone https://github.com/embodiedreasoning/ERQA.git "$WORK/ERQA"
git -C "$WORK/ERQA" checkout 5bba488252c5b345a43335bfb9239b6bf12bd91c
```

校验：

```bash
python - <<'PY'
import hashlib
from pathlib import Path
p = Path("/disk/yangchen/clef-gpu/ERQA/data/erqa.tfrecord")
assert p.stat().st_size == 91402921, p.stat().st_size
h = hashlib.sha256(p.read_bytes()).hexdigest()
assert h == "1974f5e587eac144cc3ba325477477b8ad6a56c22ce4e419c34229520b4c93e2", h
print("ERQA TFRecord OK", h)
PY
```

### 5.2 解析为官方 400 题

```bash
python "$WORK/scripts/official_eval/erqa_extract_tfrecord.py" \
  --tfrecord "$WORK/ERQA/data/erqa.tfrecord" \
  --output-dir "$DATA"
```

检查：

```bash
python - <<'PY'
import json
from pathlib import Path
root = Path("/disk/yangchen/datasets/erqa-official-20261114")
manifest = json.load(open(root / "manifest.json"))
samples = [json.loads(x) for x in (root / "samples.jsonl").read_text().splitlines() if x.strip()]
assert len(samples) == 400, len(samples)
assert manifest["source"]["sha256"] == "1974f5e587eac144cc3ba325477477b8ad6a56c22ce4e419c34229520b4c93e2"
images = sum(len(s["images"]) for s in samples)
assert images == 630, images
print("ERQA samples", len(samples), "images", images)
PY
```

预期：400 题、630 张图、单题 1～16 图，官方 `erqa-0000`～`erqa-0399` 顺序。

## 6. GPU 运行时：优先 vLLM CUDA

### 6.1 建环境（CUDA 12.8 + torch cu128）

系统的 CUDA 12.0 不够 5090（sm_120）。用 conda 安装 CUDA 12.8 工具链：

```bash
/home/yangchen/miniconda3/bin/conda create -p /disk/yangchen/venvs/clef-vllm python=3.12 -y
source /home/yangchen/miniconda3/bin/activate /disk/yangchen/venvs/clef-vllm
conda install -p /disk/yangchen/venvs/clef-vllm -c nvidia -c conda-forge \
  cuda-toolkit=12.8 cmake ninja -y
export CUDA_HOME=/disk/yangchen/venvs/clef-vllm
export PATH="$CUDA_HOME/bin:$PATH"
nvcc --version
```

安装与 pinned vLLM 一致的 PyTorch：

```bash
pip install --upgrade pip
pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cu128
python - <<'PY'
import torch
print(torch.__version__, torch.version.cuda, torch.cuda.is_available())
assert torch.cuda.is_available()
assert torch.cuda.get_device_capability(0)[0] >= 12  # 5090 sm_120
PY
```

如果 PyTorch 2.14.0 cu128 wheel 不存在，使用该 vLLM commit 官方要求且带 cu128 的最近版本；**不要降到 CPU**，也不要用 12.0 编译。

### 6.2 构建 pinned vLLM

```bash
cd "$WORK"
git clone https://github.com/vllm-project/vllm.git vllm
git -C vllm checkout 24420b7a341af37a9bc3186e6fc6062512e0aa58
cd vllm
pip install -r requirements/cuda.txt -r requirements/common.txt
TORCH_CUDA_ARCH_LIST=12.0 VLLM_TARGET_DEVICE=cuda \
  pip install -e . --no-build-isolation
```

构建失败时保存完整的 `pip`/`nvcc` 日志到 `$WORK/logs/vllm-build.log`，报告失败点；不要改用其它引擎/量化来“绕过”。

### 6.3 启动服务

确认至少有 4 张空闲卡后：

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
vllm serve "$MODEL" \
  --served-model-name clef \
  --runner pooling \
  --dtype bfloat16 \
  --tensor-parallel-size 4 \
  --max-model-len 16384 \
  --max-num-seqs 1 \
  --max-num-batched-tokens 16384 \
  --gpu-memory-utilization 0.90 \
  --enforce-eager \
  --no-enable-prefix-caching \
  --no-enable-chunked-prefill \
  --limit-mm-per-prompt '{"image":16,"video":0}' \
  --mm-processor-kwargs '{"images_kwargs":{"size":{"shortest_edge":65536,"longest_edge":1048576}},"videos_kwargs":{"size":{"shortest_edge":4096,"longest_edge":65536},"do_sample_frames":false}}' \
  --host 127.0.0.1 --port 18000 \
  > "$WORK/logs/vllm-serve.log" 2>&1 &
echo $! > "$WORK/logs/vllm.pid"
```

如果日志显示模型类不是 `ClefFlashForJointSchema`（例如只加载了 Qwen3_5 backbone、没有 joint head），在启动参数加：

```bash
--hf-overrides '{"architectures":["ClefFlashForJointSchema"]}'
```

并检查日志中出现 `ClefFlashForJointSchema`、`pooler.head`、`joint_head.safetensors` 加载记录。`config.json` 的 `architectures` 是 `Qwen3_5ForConditionalGeneration`，这一步是必要的诊断。

### 6.4 就绪与 smoke

```bash
for i in $(seq 1 120); do
  code=$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:18000/health)
  [ "$code" = 200 ] && break
  sleep 5
done
curl -s http://127.0.0.1:18000/health
curl -s http://127.0.0.1:18000/v1/decide/info
```

如果 `/v1/decide/info` 404，说明 CUDA vLLM 没有注册 decisions API；记录日志并停止，不要把 ERQA 发到错误端点。

先用 `--scope smoke` 跑 19 题：

```bash
python "$WORK/scripts/official_eval/erqa_baseline.py" \
  --data-dir "$DATA" \
  --output-dir "$RESULTS/smoke" \
  --api-url http://127.0.0.1:18000/v1 \
  --scope smoke
```

检查 19/19 成功、choice 解析正常、图片顺序与官方一致。

## 7. 全量 ERQA 400

```bash
python "$WORK/scripts/official_eval/erqa_baseline.py" \
  --data-dir "$DATA" \
  --output-dir "$RESULTS/full" \
  --api-url http://127.0.0.1:18000/v1 \
  --scope full

python "$WORK/scripts/official_eval/erqa_score.py" \
  --run-dir "$RESULTS/full" \
  --data-dir "$DATA" \
  --output "$RESULTS/full/offline-score.json" \
  --verify-requests
```

要求：

- 400/400 成功、0 失败（失败按错误计入固定分母）；
- 记录总体 accuracy、单图、多图、8 类分项；
- 保留 `predictions.jsonl`、完整响应/概率、请求哈希、服务配置、`nvidia-smi` 快照；
- 性能记录 wall、p50/p95、吞吐，标注“GPU 参考，不可与 A3 比较”。

### 与 A3 参考的对照

A3 当前结果：**219/400 = 0.5475**，单图 170/287 = 0.5923，多图 49/113 = 0.4336。

GPU 跑完后：

1. 总体 accuracy 与 A3 对比；
2. 逐题 choice 对比，统计：
   - 相同预测数；
   - GPU 正确/A3 错误、GPU 错误/A3 正确；
   - 概率最大差；
3. 若差距 > 2PP，优先检查：
   - 图片是否按 `visual_indices` 排序；
   - `state` 是否包含完整原题；
   - `criteria` 字母/文本映射；
   - 是否错误加载了 Qwen3_5 backbone 而非 `ClefFlashForJointSchema`；
   - 图像像素边界（65,536～1,048,576）和图片上限 16。

不要用温度校准；不要用 GPU 性能数据去改 A3 的 S1/S2。

## 8. 交付物

```text
/disk/yangchen/clef-gpu/results/erqa-400/
├── summary.md                  # 环境、命令、准确率、分项、与 A3 对照
├── full/summary.json
├── full/predictions.jsonl
├── full/offline-score.json
├── full/samples/...            # 请求/响应/attempt（脚本生成）
├── smoke/
└── logs/
    ├── gpu-before.txt
    ├── vllm-serve.log
    └── vllm-build.log
```

`summary.md` 至少包含：

- GPU 型号/数量、驱动、CUDA、torch、vLLM commit；
- 模型文件数与 BF16 校验结果、manifest 校验结果；
- ERQA commit/TFRecord SHA-256/400 题；
- 服务 argv、TP、max_model_len、image 上限；
- 总体/单图/多图/8 类 accuracy；
- 与 A3 219/400 的逐题差异；
- 性能 wall/p50/p95（标注不可比）；
- 任何失败、重试、OOM 或环境阻塞。

## 9. 失败处理

| 情况 | 处理 |
| --- | --- |
| GPU 被占用 | 等待空闲；记录 `nvidia-smi`；**不要 kill** |
| 模型/数据哈希不一致 | 停止，报告具体文件 |
| vLLM CUDA 构建失败 | 保存日志，报告 blocker；不要量化绕过 |
| `/v1/decide` 404 | 检查 decisions API/模型类；不要改发 `/v1/systemone` 替代图文 |
| 16K/16 图 OOM | 先 smoke，记录显存；不要截断图片或改小像素边界来“跑通” |
| 400 题失败 | 保留失败分母，报告失败分类；不要丢弃失败样本 |

## 10. 已知参考值

| 项 | 值 |
| --- | --- |
| 模型 | `Cloudflare/clef`，BF16，约 51.2 GiB，12 shards + joint head |
| ERQA | commit `5bba488252c5b345a43335bfb9239b6bf12bd91c`，TFRecord SHA-256 `1974f5e587eac144cc3ba325477477b8ad6a56c22ce4e419c34229520b4c93e2`，400 题、630 图 |
| A3 参考 | 219/400 = 0.5475；单图 0.5923；多图 0.4336 |
| 允许 GPU | 4 × RTX 5090 32GB；至少 2 张空闲才 smoke，4 张空闲再跑全量 |
