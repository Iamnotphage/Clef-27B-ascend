# Clef-27B 上游复用与 NPU baseline 方案

核查日期为 2026-11-12，时区为 Asia/Shanghai。本文件保留当日上板前的源码与安装研究。后续 A3 实测已完成 TP1/TP2、CPU 精度、完整公开文本评测和多模态自检，实际命令及证据见[实测 baseline](baseline-2026-11-12.md)与[安装记录](installation-2026-11-12.md)。

当前机器为 `199.88.54.221`，已预装 CANN、A3 Ops、NNAL 9.2.1，用户选择采用本文的 PyTorch 2.10 配套方案。文中 CANN 9.0.0 的服务器段落保留了早期实例的调查背景，当前部署顺序见新机器计划。[新机器执行计划](new-machine-plan.md)

## 已确认的复用基础

vLLM-Ascend PR #21445 于北京时间 2026-11-11 17:35:02 合入，合并提交为 `89eb8a5e7a76d085b453a1aeadb6883e12d5ea2d`。改动注册 `ClefFlashForJointSchema`，组合上游联合决策实现与 Ascend Qwen3.5 多模态预处理。`Cloudflare/clef` 和 `Cloudflare/clef-flash` 共用这套架构名。[PR #21445](https://github.com/vllm-project/vllm-ascend/pull/21445) · [Ascend 适配类](https://github.com/vllm-project/vllm-ascend/blob/89eb8a5e7a76d085b453a1aeadb6883e12d5ea2d/vllm_ascend/models/clef_flash.py)

| 项目 | 已核查的行为 | baseline 所需工作 |
| --- | --- | --- |
| 模型识别 | `Qwen3_5ForConditionalGeneration` 配合 `joint_head_config.json` 自动映射到 `ClefFlashForJointSchema` | 保留模型目录中的独立 head 配置和权重，核对启动日志实际架构 |
| 主干 | 复用 Qwen3.5，包含 Gated DeltaNet 和 full attention | 验证 A3 上 BF16 前向与对应算子加载 |
| 决策头 | `JointSchemaPooler` 读取每个 prompt 的全序列 hidden states，执行字段与选项联合评分 | 检查完整 hidden states、候选词向量、token/span 映射和最终概率 |
| head 权重 | `secondary_weights` 显式读取 `joint_head.safetensors`，参数前缀为 `pooler.head.` | 检查 missing/unexpected keys 和 head 张量实际加载 |
| 服务 | vLLM 原生注册 `POST /v1/systemone` | 把本项目入口接到服务，补健康检查、请求样例、结果与日志保存 |
| 多模态 | Ascend 适配类复用 `AscendQwen3_5VLProcessor` | 文本成功后继续验证图片、视频及媒体 token 展开 |

[模型自动识别](https://github.com/vllm-project/vllm/blob/24420b7a341af37a9bc3186e6fc6062512e0aa58/vllm/transformers_utils/model_arch_config_convertor.py#L622) · [head 加载](https://github.com/vllm-project/vllm/blob/24420b7a341af37a9bc3186e6fc6062512e0aa58/vllm/model_executor/models/clef_flash.py) · [联合决策头](https://github.com/vllm-project/vllm/blob/24420b7a341af37a9bc3186e6fc6062512e0aa58/vllm/model_executor/models/clef_joint_schema.py) · [服务路由](https://github.com/vllm-project/vllm/blob/24420b7a341af37a9bc3186e6fc6062512e0aa58/vllm/entrypoints/generate/structured_decisions/api_router.py)

主干继续使用 Qwen3.5 的因果序列计算。决策头以全序列 hidden states 为 memory，通过 `MultiheadAttention` 和 `TransformerDecoderLayer` 完成证据读取、字段交互与候选评分。适配需保留两层计算语义。[主干实现](https://github.com/vllm-project/vllm/blob/24420b7a341af37a9bc3186e6fc6062512e0aa58/vllm/model_executor/models/qwen3_5.py) · [模型参考实现](https://modelscope.cn/models/Cloudflare/clef/resolve/master/joint_schema_model.py)

## 上游实测范围

PR 的公开硬件结果来自 **单个 Ascend 950DT、BF16**。作者报告 Clef 与名为 `CPU reference` 的参考结果有 11/11 个决策一致，最大概率绝对差为 0.01645，文本和图片加载、服务通过。正文另提到 CPU reference 运行时未使用 `accelerate`。其参考脚本、主干与 head 的设备分配、CPU dtype 及中间张量来源仍待核实，整模型跨设备精度对齐需在本项目独立复现。A2、A3、多 NPU 配置和 Clef 的请求延迟、吞吐、峰值 HBM 均待验证。[PR 验证说明](https://github.com/vllm-project/vllm-ascend/pull/21445)

同一 PR 的 Clef Flash 数据包含 36/36 CPU 决策一致、两个纯色图片案例，以及 2,200 个决策子集上的 72.05% 标签准确率。该子集对应的公开同协议基准分数尚待确认。169 秒是 Flash 三个图模式端到端案例的总测试时间。Clef-27B 的图模式结果和本比赛基准仍需单独采集。[PR 验证说明](https://github.com/vllm-project/vllm-ascend/pull/21445)

## 建议固定的软件组合

合并提交中的 `.github/vllm-main-verified.commit` 指向 `24420b7a341af37a9bc3186e6fc6062512e0aa58`，与该 PR 的硬件验证版本相同。这两个 SHA 适合作为首轮可复现候选。[配套 vLLM 提交记录](https://github.com/vllm-project/vllm-ascend/blob/89eb8a5e7a76d085b453a1aeadb6883e12d5ea2d/.github/vllm-main-verified.commit)

| 组件 | 候选固定值 | 证据与实际含义 |
| --- | --- | --- |
| vLLM-Ascend | `89eb8a5e7a76d085b453a1aeadb6883e12d5ea2d` | 已包含 Clef 注册与预处理 |
| vLLM | `24420b7a341af37a9bc3186e6fc6062512e0aa58` | Ascend 合并提交记录的已验证上游 SHA |
| 系统 / Python | Ubuntu 22.04 / Python 3.12 | `Dockerfile.a3` 基础镜像采用该组合。固定 vLLM SHA 的 `requires-python` 为 `>=3.11,<3.15`，与 Ascend 文档范围 `>=3.10,<3.13` 的交集为 3.11、3.12 |
| CANN Toolkit + A3 Ops | 9.2.1 | 配套镜像包含 A3 算子包 |
| NNAL / ATB | 9.2.1 | 检查 `libatb.so` 和对应 `set_env.sh` |
| Ascend HDK | 文档验证组合为 26.1.1 | 本机驱动、固件与 9.2.1 的具体兼容状态仍需复测 |
| PyTorch | 2.10.0 | 合并提交 `requirements.txt` |
| torch_npu | 2.10.0.post7 | 合并提交 `requirements.txt` |
| Transformers | 5.19.0 | 合并提交 `requirements.txt` |
| torchvision / torchaudio | 0.25.0 / 2.10.0 | 合并提交 `requirements.txt` |
| Triton Ascend | 3.6.0.post2 | 合并提交 `requirements.txt` |
| FLA NPU | `flash-linear-attention-npu-a3==26.9.1` | A3 Dockerfile 的 Gated DeltaNet / convolution 配套包 |
| flash_attn_npu | 0.5.0 | A3 Dockerfile 指定，构建使用 `FLASH_ATTN_BUILD_NPU=910` |

[requirements.txt](https://github.com/vllm-project/vllm-ascend/blob/89eb8a5e7a76d085b453a1aeadb6883e12d5ea2d/requirements.txt) · [Dockerfile.a3](https://github.com/vllm-project/vllm-ascend/blob/89eb8a5e7a76d085b453a1aeadb6883e12d5ea2d/Dockerfile.a3) · [配套版本常量](https://github.com/vllm-project/vllm-ascend/blob/89eb8a5e7a76d085b453a1aeadb6883e12d5ea2d/mkdocs.yml) · [安装与 HDK 组合说明](https://github.com/vllm-project/vllm-ascend/blob/89eb8a5e7a76d085b453a1aeadb6883e12d5ea2d/docs/source/getting_started/installation.md)

官方 CANN A3 基础镜像为 `quay.io/ascend/cann:9.2.1-a3-ubuntu22.04-py3.12`。对应 Dockerfile 明确支持 `linux/arm64 → aarch64`，同时下载 9.2.1 Toolkit、A3 Ops、NNAL。镜像的 ARM64 manifest、实际拉取结果和最终 digest 应在部署时记录。HDK 26.1.1 是上游验证组合，最低驱动版本与各固件配套关系还需要对照华为兼容矩阵。[CANN 镜像目录](https://github.com/Ascend/cann-container-image/blob/main/OVERVIEW.md) · [A3 ARM64 Dockerfile](https://github.com/Ascend/cann-container-image/blob/main/cann/9.2.1-a3-ubuntu22.04-py3.12/Dockerfile) · [华为兼容性查询](https://info.support.huawei.com/computing/tools/compatibility-query/enterprise/ascend-computing/system-compatibility)

本次服务器探索确认当前 SSH 环境本身处于容器内，Docker、podman、conda、uv、pip、ensurepip 均待配备，Python 为 3.10.12。实际 cgroup 配额为 128 GiB RAM 和 32 核等价 CPU 时间配额，cpuset 可见范围为 0–639，安装、源码编译与参考推理应据此规划资源。ACL 初始化后可见两个逻辑设备，映射到 `/dev/davinci14`、`/dev/davinci15`。当前 CANN、A3 Ops、NNAL 为 9.0.0，驱动为 26.1.1，固件为 9.0.0.9.220。[环境证据](evidence/environment-2026-11-12.json) · [补充检查](evidence/environment-details-2026-11-12.json)

环境准备优先评估由算力平台选择匹配的 A3 CANN 9.2.1 镜像，并在操作前确认 `/workspace` 持久卷和已有模型可继续挂载。随后在该环境构建固定 SHA 的软件栈。使用 `Dockerfile.a3` 作为构建配方时应显式传入上表 `VLLM_COMMIT`，文件默认的 `VLLM_TAG=v0.33.0` 与本次选定提交需要分别核对。构建目标也应由本机查询到的 `SOC_VERSION` 和构建脚本支持列表决定。文件中的 `ascend910_9391` 为默认值，本机报告型号为 `Ascend910_9382`。[A3 Dockerfile](https://github.com/vllm-project/vllm-ascend/blob/89eb8a5e7a76d085b453a1aeadb6883e12d5ea2d/Dockerfile.a3)

保留当前容器的下一候选是在独立前缀安装匹配 CANN/NNAL 并建立独立 Python 环境，用进程级环境脚本选择版本。该路径需要先准备 Python 包管理工具，并核对安装器支持的路径和动态库解析。现有 CANN 9.0.0 的官方旧版兼容行指向 vLLM-Ascend v0.20.2rc1，Clef 新增实现需要进一步回移评估，首轮优先采用上述 9.2.1 组合。[版本兼容表](https://github.com/vllm-project/vllm-ascend/blob/89eb8a5e7a76d085b453a1aeadb6883e12d5ea2d/docs/source/community/versioning_policy.md)

固定 vLLM 提交要求 Python `>=3.11`，因此当前 Python 3.10.12 需要升级到独立的 3.12 环境。当前 CMake 3.22.1 也需要在构建环境升级到至少 3.26.1。vLLM 的构建声明包括 Ninja、`packaging>=24.2`、`setuptools>=77.0.3,<81.0.0`、`setuptools-scm>=8.0`、`setuptools-rust>=1.9.0`、wheel 和 jinja2。Python 3.12 的通用运行依赖进一步要求 `setuptools>=78.1.1,<81.0.0`。Ascend 插件构建声明还包含 `pybind11` 及上表 PyTorch/TorchNPU 栈。[vLLM pyproject](https://github.com/vllm-project/vllm/blob/24420b7a341af37a9bc3186e6fc6062512e0aa58/pyproject.toml) · [vLLM 通用依赖](https://github.com/vllm-project/vllm/blob/24420b7a341af37a9bc3186e6fc6062512e0aa58/requirements/common.txt) · [Ascend pyproject](https://github.com/vllm-project/vllm-ascend/blob/89eb8a5e7a76d085b453a1aeadb6883e12d5ea2d/pyproject.toml)

上游 NPU 源码安装流程先以 `VLLM_TARGET_DEVICE=empty python -m pip install -e .` 安装 vLLM，再在已加载 CANN/ATB 环境的进程安装 vLLM-Ascend。`empty` 分支使用 vLLM 通用依赖并清空 vLLM 的设备 C++ 扩展列表，NPU 扩展由 Ascend 插件构建。A3 插件需要初始化 git submodules，实际运行构建使用 `COMPILE_CUSTOM_KERNELS=1`。Ascend 构建脚本支持 `MAX_JOBS`，应根据上述 CPU 时间配额和 128 GiB 内存显式设置并发。Ninja 属于 vLLM 的构建依赖，Ascend C kernels 的构建脚本当前使用 CMake 常规构建路径。[官方源码安装流程](https://github.com/vllm-project/vllm-ascend/blob/89eb8a5e7a76d085b453a1aeadb6883e12d5ea2d/docs/source/getting_started/installation/install_vllm_ascend.inc.md) · [vLLM empty 构建分支](https://github.com/vllm-project/vllm/blob/24420b7a341af37a9bc3186e6fc6062512e0aa58/setup.py) · [Ascend 构建脚本](https://github.com/vllm-project/vllm-ascend/blob/89eb8a5e7a76d085b453a1aeadb6883e12d5ea2d/setup.py)

该 vLLM 提交的 PEP 517 构建声明含 `torch==2.14.0`，Ascend 插件运行栈固定为 `torch==2.10.0`。应保留构建隔离和安装日志，确认最终运行环境的 PyTorch/TorchNPU 版本符合上表并执行 `pip check`。如选择关闭构建隔离，需要针对已预装依赖验证构建结果。`Dockerfile.a3` 另外执行 Rust 前端构建，完全复用该配方时还需其 Rust、protobuf 构建工具。[vLLM 构建声明](https://github.com/vllm-project/vllm/blob/24420b7a341af37a9bc3186e6fc6062512e0aa58/pyproject.toml) · [A3 构建配方](https://github.com/vllm-project/vllm-ascend/blob/89eb8a5e7a76d085b453a1aeadb6883e12d5ea2d/Dockerfile.a3)

## 第一轮启动与请求

以下命令已同步为本轮 A3 首次完整服务的实测配置：短文本、batch 1、BF16、TP1、eager。设置 image 1 / video 0 保留共用视觉塔，`--skip-mm-profiling` 跳过视觉预热，首轮请求使用纯文本。`--max-model-len 1024` 包含 state、schema 和固定 prompt 的全部 token。测试样例应控制在该范围内。[CLI 参数实现](https://github.com/vllm-project/vllm/blob/24420b7a341af37a9bc3186e6fc6062512e0aa58/vllm/engine/arg_utils.py) · [实际加载与性能](baseline-2026-11-12.md)

```bash
export MODEL_DIR=/workspace/yangchen/models/Clef-27B

vllm serve "$MODEL_DIR" \
  --served-model-name clef \
  --runner pooling \
  --dtype bfloat16 \
  --tensor-parallel-size 1 \
  --enforce-eager \
  --max-model-len 1024 \
  --max-num-seqs 1 \
  --max-num-batched-tokens 1024 \
  --gpu-memory-utilization 0.90 \
  --no-enable-prefix-caching \
  --no-enable-chunked-prefill \
  --limit-mm-per-prompt '{"image":1,"video":0}' \
  --skip-mm-profiling \
  --host 127.0.0.1 \
  --port 8000
```

`--enforce-eager` 在该版本统一设置 `CompilationMode.NONE` 和 `CUDAGraphMode.NONE`，对 pooling 生效，Ascend 平台更新逻辑也保留 eager 设置。上述命令显式关闭 prefix caching 与 chunked prefill。Mamba 配置据此采用 `mamba_cache_mode=none`，默认 `mamba_block_size=max_model_len`。后续开启 Mamba prefix caching 时，框架要求同时开启 chunked prefill，并处理 block size 对齐。[vLLM 编译配置](https://github.com/vllm-project/vllm/blob/24420b7a341af37a9bc3186e6fc6062512e0aa58/vllm/config/vllm.py#L1782) · [Ascend eager 配置](https://github.com/vllm-project/vllm-ascend/blob/89eb8a5e7a76d085b453a1aeadb6883e12d5ea2d/vllm_ascend/platform.py#L1320) · [Mamba 配置](https://github.com/vllm-project/vllm/blob/24420b7a341af37a9bc3186e6fc6062512e0aa58/vllm/model_executor/models/config.py#L823)

首先确认服务日志识别到 Ascend 插件、`ClefFlashForJointSchema`、pooling、BF16，以及独立 head 的加载。CANN 和 ATB 环境使用所选镜像的初始化脚本。Clef 适配类自身没有定义额外必需的模型专用环境变量，设备可见性应按实际分配的 NPU 配置。多模态测试时把 `--limit-mm-per-prompt` 改为实际需要的媒体数量。[Ascend Clef 适配类](https://github.com/vllm-project/vllm-ascend/blob/89eb8a5e7a76d085b453a1aeadb6883e12d5ea2d/vllm_ascend/models/clef_flash.py)

```bash
curl --fail-with-body http://127.0.0.1:8000/v1/systemone \
  -H 'Content-Type: application/json' \
  --data-binary @- <<'JSON'
{
  "model": "clef",
  "state": "The customer was charged twice for one order and requests a refund.",
  "questions": {
    "department": {
      "type": "choice",
      "instructions": "Which team should handle this ticket?",
      "criteria": {
        "billing": "Payments and refunds",
        "shipping": "Deliveries"
      }
    },
    "refund_requested": {
      "type": "noul",
      "instructions": "Does the customer request a refund?"
    },
    "severity": {
      "type": "score",
      "instructions": "How severe is the payment problem?",
      "criteria": ["none", "minor", "major"]
    }
  }
}
JSON
```

样例的结构验收应包含 HTTP 200、三个答案、有限且归一化的候选概率、`noul` 落在 `[0,1]`、`score` 落在 `[0,2]`，以及 `usage.output_tokens=0`。具体语义和数值以同一权重、同一 prompt 的参考实现输出为准。[类型与输出计算](https://github.com/vllm-project/vllm/blob/24420b7a341af37a9bc3186e6fc6062512e0aa58/vllm/entrypoints/openai/decisions/question_types.py)

## 服务协议与精度对齐

| 项目 | 固定源码的行为 | 本项目应验证的内容 |
| --- | --- | --- |
| 联合字段 | Clef 一次 pooling read 处理请求中的全部问题 | 多字段结果与参考实现一致 |
| `joint` 请求参数 | 默认 `false`。该版本的 `true` 分支用于 DiffusionGemma，Clef 会被校验拒绝 | Clef 请求省略 `joint` 或使用 `false`，联合决策由模型 head 完成 |
| `choice` | 按候选名称排序构建模型 prompt，输出再映射回请求候选 | 候选打乱顺序后的键值对应、同概率 tie 行为 |
| `noul` | 内部语义顺序为 true、false，HTTP 的 `noul` 为 P(yes)，概率键为 yes/no | 与参考实现内部 true/false 的映射 |
| `score` | 返回 `sum(index * probability)`，index 从 0 起 | 等级期望、legend、候选概率 |
| 概率 | 每个字段内部 softmax，head 输出先转 float32 | 逐选项绝对误差、最大误差和决策一致率 |
| token/span | 按问题与候选片段分别 tokenize，span 从 prompt 末尾计数 | 中文、空描述、变长请求、媒体占位符展开后的边界 |
| 长度 | Clef 策略最大为 `min(16384,max_model_len)`，state 截断到剩余预算，schema 超限报错 | 明确记录截断策略和实际 token 数 |
| `instructions` | 每个 question 的 instructions 有效，顶层非空 instructions 被固定 prompt 策略拒绝 | 测试数据字段位置正确 |
| 图像 / 视频 | 支持 image/video，视频按 2 fps 读取 | 实际媒体解码、预处理和模型路径 |
| 响应格式 | vLLM 加入 id、object、diagnostics 等，模型参考实现部分数值四舍五入到 4 位 | 按评测器期望解析字段和精度，保留原始概率作比较 |

[Clef 策略实现](https://github.com/vllm-project/vllm/blob/24420b7a341af37a9bc3186e6fc6062512e0aa58/vllm/entrypoints/openai/decisions/strategies.py#L841) · [joint 参数校验](https://github.com/vllm-project/vllm/blob/24420b7a341af37a9bc3186e6fc6062512e0aa58/vllm/entrypoints/openai/decisions/serving.py#L40) · [官方服务说明](https://github.com/vllm-project/vllm/blob/24420b7a341af37a9bc3186e6fc6062512e0aa58/docs/models/structured_decisions_models.md#clef-flash) · [模型参考实现](https://modelscope.cn/models/Cloudflare/clef/resolve/master/joint_schema_model.py)

本次读取的 `joint_schema_model.py` SHA-256 为 `0e304cf7c6500e8bb59bef7e2afd2c6373f82596dfb3b57d1aa93c175e2dc3a3`，与本仓库 `model_manifest.json` 相同。head 配置为 hidden size 5120、width 1024、routing layers 2、decoder layers 4、heads 16、feedforward 4096。独立 head 文件为 256,125,024 字节，约 244.3 MiB。[本仓库 manifest](../Clef-27B/logs/model_manifest.json) · [head 配置](https://modelscope.cn/models/Cloudflare/clef/resolve/master/joint_head_config.json)

## NPU 适配工作与 TP2 验证

首轮复用上游模型、服务和已有 Ascend 算子，按实际加载或执行异常补兼容修正。需要重点观察 Gated DeltaNet / FLA、full attention、RMSNorm、RoPE，以及 head 中的 `MultiheadAttention`、`TransformerDecoderLayer`、`LayerNorm`、GELU、`index_add_`、einsum 和 softmax。记录失败算子、dtype、shape、调用栈和输入样例，再用最小复现定位。[主干实现](https://github.com/vllm-project/vllm/blob/24420b7a341af37a9bc3186e6fc6062512e0aa58/vllm/model_executor/models/qwen3_5.py) · [head 实现](https://github.com/vllm-project/vllm/blob/24420b7a341af37a9bc3186e6fc6062512e0aa58/vllm/model_executor/models/clef_joint_schema.py)

两个 64 GiB 计算单元提供 TP1 和 TP2 两种试验条件。约 51.2 GiB 的权重占用只覆盖静态权重，TP1 还需容纳运行时工作区、状态缓存、全序列 hidden states、head 激活和多模态 profile。`gpu_memory_utilization=0.90` 对 64 GiB 的预算约为 57.6 GiB，首轮是否足够需要实测加载与 warmup 峰值。TP1 若出现 OOM，应保存失败阶段和显存记录，进入 TP2 分摊主干的验证路径。

TP2 的主干沿用 vLLM 分片，head 中的普通 `nn.Linear` 等模块按源码结构会在各 rank 持有完整副本。候选词向量通过 `VocabParallelEmbedding.forward(self.language_model.lm_head, token_ids)` 获取，底层会处理词表分片及 all-reduce。TP2 需同时验证 HCCL、完整 hidden states、head 副本一致性和包含不同词表分片 token 的候选。最终逐字段比较 TP1/TP2 的决策和概率，并记录各 rank 峰值 HBM。[词向量读取](https://github.com/vllm-project/vllm/blob/24420b7a341af37a9bc3186e6fc6062512e0aa58/vllm/model_executor/models/clef_flash.py) · [词表分片与归约](https://github.com/vllm-project/vllm/blob/24420b7a341af37a9bc3186e6fc6062512e0aa58/vllm/model_executor/layers/vocab_parallel_embedding.py#L522)

## baseline 完成标准

1. 固定源码 SHA、镜像 digest 或安装前缀、完整依赖版本及模型 manifest，`pip check` 和 `torch_npu` 的 BF16 分配、matmul、同步通过。
2. Clef 主干与独立 head 加载成功，短文本三类型请求返回完整结果并保存原始日志。
3. 同一套固定输入覆盖单字段、多字段、不同候选数量、中文、变长 batch、边界长度和非法请求，记录概率误差及决策一致率。
4. 完成一轮官方或明确标记为开发集的评测，记录样本数、版本、预处理、指标定义与测量方法。
5. 将 `Clef-27B/start.sh`、`inference.py` 接入真实服务，启动脚本可复现对应版本、参数和日志位置。

图模式、TP2、多模态、长上下文与并发吞吐逐项扩展，并为每次变化保留相同数据上的精度与性能对照。公开 PR 中的 0.01645 概率差属于已报告测试结果，项目验收阈值需结合比赛规则和参考精度确定。
