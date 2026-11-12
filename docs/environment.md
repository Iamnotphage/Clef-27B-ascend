# 早期实例与仓库环境调查

本文记录 2026-11-12 14:50 左右的早期实例。当前机器已切换为 `199.88.54.221`，新机器实测 CANN 9.2.1，当前环境和执行安排见新机器计划。[新机器执行计划](new-machine-plan.md)

调查日期：2026-11-12（北京时间）。本轮通过 SSH 执行只读盘点及一次 ACL 初始化/设备枚举，未安装依赖、升级 CANN、加载模型或启动推理服务。

## 1. 实测资源

| 项目 | 本轮结果 | 对 baseline 的影响 |
| --- | --- | --- |
| 系统 | Ubuntu 22.04.5 LTS，aarch64；内核 `5.10.0-216.0.0.115.oe2203sp4.aarch64` | 选择 ARM64 安装包与 wheel |
| 实例形态 | 存在 `/.dockerenv`，根文件系统为 overlay | 当前 SSH 环境已经位于容器内 |
| CPU 配额 | cgroup quota/period = 3200000/100000，即 32 核等价配额 | 控制编译并发与服务 CPU 线程数 |
| 内存配额 | cgroup `memory.limit_in_bytes=137438953472`，即 128 GiB | 模型加载及 TP2 多进程须监测容器内存峰值 |
| 宿主可见资源 | `lscpu` 显示 640 个逻辑 CPU，`free` 显示约 2 TiB RAM | 容器实际预算采用上述 cgroup 配额 |
| 可写盘 | `/`、`/workspace`、`/tmp` 共用约 300 GiB overlay；检查时剩余约 151 GiB | 依赖、编译缓存、数据解压共同消耗此空间 |
| 共享内存 | `/dev/shm` 16 GiB | TP2 和多进程评测时检查共享内存占用 |
| NPU | ACL 返回 `Ascend910_9382`、设备数 2；两个计算单元各 65536 MiB HBM | 按 A3 平台配套；先 TP1 短文本，再 TP2 |
| NPU 状态 | Health OK、AICore 0%、无运行中的 NPU 进程 | 检查时 HBM 分别已用 3118 / 2887 MiB，仍需实测推理峰值 |
| 芯片互联 | 物理 14 与 15 之间为 SIO | TP2 通信还需 HCCL 实测 |

设备编号的对应关系：

| 工具/接口 | 第一个计算单元 | 第二个计算单元 |
| --- | --- | --- |
| ACL / 逻辑设备号 | 0 | 1 |
| `npu-smi` 卡号、Chip ID | 7、0 | 7、1 |
| 物理设备号 | 14 | 15 |
| 设备文件 | `/dev/davinci14` | `/dev/davinci15` |

`npu-smi info -t board -i 0` 返回无效卡号；修正为 `-i 7 -c 0` 后查询成功。后续启动配置和设备挂载应采用各接口对应的编号。

根据 `npu-arch` 的产品映射，`Ascend910_93` 属于 A3 / DAV_2201 系列。上游 Clef 测试使用的 950DT 属于不同架构，A3 算子分支需要独立验证。

## 2. 软件栈和基础健康检查

| 组件 | 本轮结果 |
| --- | --- |
| 驱动 / npu-smi | 26.1.1 |
| 固件 | 9.0.0.9.220（卡 7、芯片 0） |
| CANN toolkit | 9.0.0，`/usr/local/Ascend/cann-9.0.0` |
| CANN 算子包 | `Ascend-cann-A3-ops` 9.0.0 |
| NNAL ATB | 9.0.0，build 9.0.0.B160 |
| Python | `/usr/bin/python3`，3.10.12 |
| Python 包管理 | 系统 Python 缺少 `pip`、`ensurepip`；PATH 未发现 uv、conda |
| 推理依赖 | 系统 Python 未发现 torch、torch_npu、transformers、vllm、vllm_ascend、numpy、safetensors、huggingface_hub、FastAPI、Pillow、ModelScope、pytest |
| 编译工具 | GCC/G++ 11.4.0、CMake 3.22.1、Git 2.34.1；PATH 未发现 Ninja |
| 容器工具 | PATH 未发现 Docker、Podman |

SSH 非交互 shell 的 Ascend 环境变量未自动设置。在子进程中显式加载 `/usr/local/Ascend/cann/set_env.sh` 后，使用 `ctypes` 调用现有 `libascendcl.so` 得到：

```json
{"aclInit":0,"aclrtGetDeviceCount":0,"device_count":2,"soc":"Ascend910_9382","aclFinalize":0}
```

这验证了当前 CANN 的动态库加载和设备枚举。BF16 运算、torch_npu、HCCL 及完整 Clef 推理仍待安装配套环境后验证。

首选采用包含目标版本栈的平台镜像，并先确认 `/workspace/yangchen` 在切换实例/镜像时的持久化方式。另一个候选是在当前容器内使用独立 Python 环境和独立 CANN/NNAL 安装前缀。当前实例无法直接运行 `docker run`，也不宜将已有 CANN 9.0.0 当成上游 9.2.1 构建的已验证依赖。精确版本与构建要求见 [上游适配调查](upstream-npu-baseline.md)。

固定 vLLM 提交的 `pyproject.toml` 要求 Python `>=3.11,<3.15`，与 Ascend 组件支持范围取交集后采用 Python 3.11 / 3.12，优先跟随 A3 镜像的 3.12。其构建依赖还要求 CMake `>=3.26.1` 和 Ninja。因此当前 Python 3.10.12 与 CMake 3.22.1 均需要在新环境中配套更新。

## 3. 模型和工程资产

| 资产 | 实测状态 |
| --- | --- |
| 远端工程 | `/workspace/yangchen/Clef-27B-ascend`，HEAD `3e83ae17df7b559ec2dd6b2a38aa029c2777e061`，工作区干净 |
| 远端模型 | `/workspace/yangchen/models/Clef-27B`，28 个文件，总计 54,989,897,681 字节 |
| 主干权重 | 12 个 safetensors 分片；既有校验记录为 1184 个 BF16 tensor |
| 独立决策头 | `joint_head.safetensors`，既有记录为 122 个 BF16 tensor，约 244.26 MiB |
| 全部 safetensors | 约 51.1946 GiB；驻留显存以实际加载测量为准 |
| 实际模型配置 | `architectures=[Qwen3_5ForConditionalGeneration]`，`model_type=qwen3_5`，BF16，包含视觉编码器 |
| 主干结构 | 64 层；48 层 linear attention、16 层 full attention；hidden size 5120 |
| 决策头配置 | width 1024，routing_layers 2，layers 4，heads 16，feedforward 4096 |
| 模型公开评测记录 | `.eval_results/Cloudflare__clef.yaml` 和 `safeagents-judge-bench.yaml` 存在 |

本轮重新核对了文件数量、大小和配置；完整 SHA-256 校验沿用当天 13:52 已完成的[校验记录](../Clef-27B/logs/download_verification.json)，没有重复读取全部 55 GB 权重。现有下载脚本未固定模型 revision，后续复现应按 [manifest](../Clef-27B/logs/model_manifest.json) 固定逐文件版本和哈希。

TP1 在 64 GiB HBM 上值得先尝试。权重之外还需要 Mamba 状态、Attention 缓存、全序列 hidden states、联合决策头中间结果、算子 workspace 和运行时内存。采用 `--gpu-memory-utilization 0.90` 时名义预算为 57.6 GiB，按完整权重文件大小估算只剩约 6.4 GiB；加载布局、视觉模块裁剪和运行时预留会改变实际余量。首轮应限制文本长度和 batch，记录真实峰值再决定是否进入 TP2。

本地仓库只存脚手架与模型校验记录：

- [`start.sh`](../Clef-27B/start.sh) 转发到 Python。
- [`inference.py`](../Clef-27B/inference.py) 只解析 `--model-dir`，随后退出。`--help` 返回 0，默认启动返回 2。
- `results.csv`、`metrics.csv` 仅有项目自定表头。
- 推理服务、依赖锁定、评测脚本、结果截图和 NPU 调用截图待补齐。
- 本地已有 `README.md` 的表格空格调整，本轮保留该修改。

## 4. 数据和网络

`/workspace/shared_assets/README.md` 明确该目录为只读公共资源。已核对的数据包括 MMLU、MMMU、Daily-Omni、Video-MME、Alpaca / AlpacaEval、LLaVA-Instruct-150K 和 seed-tts-eval。格式、规模、缺项和 baseline 使用建议见 [比赛与数据集](competition-and-datasets.md)。转换结果、索引和缓存应写到 `/workspace/yangchen` 下。

2026-11-12 14:54 左右，从服务器执行小型 HTTP HEAD 探测：

| 目标 | 本轮结果 | 下一步 |
| --- | --- | --- |
| PyPI `torch-npu` 索引 | HTTP 200 | 安装前确认实际 wheel 平台和版本可下载 |
| ModelScope Clef 配置文件 | HTTP 200 | 可作为模型/数据下载渠道候选 |
| GitHub raw 固定提交文件 | 超时 | 可在本地准备固定版本源码包或 Git bundle，再传到服务器 |
| Hugging Face 数据集 API | Connection reset by peer | 公开数据集可在可访问的环境下载、固定 revision/哈希后传入 |

这些结果代表本轮网络观测；各站点的大文件下载能力仍需单独确认。已有模型无需重复下载。Video-MME 共享压缩包约 101 GB，首轮仅取需要的少量样例，避免占满约 151 GiB 可写空间。

## 5. 证据与复查入口

- [环境主快照](evidence/environment-2026-11-12.json)：2026-11-12 14:50，系统、依赖、NPU、模型配置与原始 eval YAML。
- [环境补充快照](evidence/environment-details-2026-11-12.json)：14:52，cgroup、逻辑/物理设备映射、ACL、CANN/固件和共享目录。
- [数据与网络快照](evidence/data-and-network-2026-11-12.json)：数据文件统计、共享目录说明、网络探测。

复查基础状态可在远端执行：

```bash
source /usr/local/Ascend/cann/set_env.sh
npu-smi info
npu-smi info -m
npu-smi info -t board -i 7 -c 0
cat /sys/fs/cgroup/memory/memory.limit_in_bytes
cat /sys/fs/cgroup/cpu/cpu.cfs_quota_us
cat /sys/fs/cgroup/cpu/cpu.cfs_period_us
python3 --version
df -h /workspace /dev/shm
```

SSH 跳板连接命令和密码未写入仓库。调查过程中遇到一次连接凭据过期，更新连接命令后完成了上述服务器检查。
