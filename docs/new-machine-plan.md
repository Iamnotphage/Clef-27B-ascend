# 新机器环境与执行计划

更新日期为 2026-11-13，北京时间。CANN 9.2.1 / PyTorch 2.10 环境与首轮工程 baseline 已完成；本轮端到端 profiling、TP1/TP2 稳定测量、长输入/多模态/batching 对照及服务恢复已验收。[profiling 实测记录](profiling-2026-11-12.md)

主评测现统一为 **Decision Index 0.2.1** 与 **Typesafe WorkflowEvals**，两套均待接入与运行。已有 JevBench、OpenDecisions、Typed Decisions 结果保留为工程回归记录；后续版本、指标与同协议对照按[统一评测口径](evaluation-protocol.md)执行。

## 1. 连接与目录

| 项目 | 值 |
| --- | --- |
| 当前机器 | `root@199.88.54.221` |
| 跳板 | `113.47.8.48:2234` |
| 工程目录 | `/workspace/yangchen/Clef-27B-ascend` |
| 模型目录 | `/workspace/yangchen/models/Clef-27B` |
| 运行日志 | `/workspace/yangchen/logs` |
| 下载和构建临时目录 | `/workspace/yangchen/work` |
| 目标 Python 环境 | `/workspace/yangchen/venvs/clef-py312` |
| profiling 复现工具 | `/workspace/yangchen/work/profiling-tools-20261113`，原工程工作区保留 |

跳板账号和令牌使用平台当前生成的值，目标机密码按终端提示输入。

本机已配置 `Host clef` 与固定主连接 socket。日常在仓库根目录运行 `./connect.sh`，主连接缺失时按提示输入最新平台凭据；终端和本机 agent 统一通过 `ssh clef` 连接。`./connect.sh status` 检查本地主连接进程，`./connect.sh stop` 关闭主连接及所有复用会话。连接中断后重新运行脚本即可重建，具体配置见 [本机 SSH 连接](../README.md#本机-ssh-连接)。

平台原始命令格式如下，脚本支持将整条命令作为隐藏输入解析：

```bash
ssh -J '<跳板账号>:<临时令牌>@113.47.8.48:2234' root@199.88.54.221
```

## 2. 查看环境

以下命令可直接在新机器执行。

```bash
source /usr/local/Ascend/cann/set_env.sh
npu-smi info
npu-smi info -m
cat /usr/local/Ascend/cann/aarch64-linux/ascend_toolkit_install.info
cat /usr/local/Ascend/cann/aarch64-linux/ascend_ops_install.info
cat /usr/local/Ascend/nnal/atb/9.2.1/version.info
cat /usr/local/Ascend/driver/version.info
python3 --version
python3 -m pip show torch torch-npu transformers vllm vllm-ascend
cat /sys/fs/cgroup/memory/memory.limit_in_bytes
cat /sys/fs/cgroup/cpu/cpu.cfs_quota_us
cat /sys/fs/cgroup/cpu/cpu.cfs_period_us
df -h /workspace /dev/shm
```

系统 Python 的 pip 当前待安装，因此包查询会报缺少模块。目标环境建立后，用其解释器查看实际运行版本。

```bash
/workspace/yangchen/venvs/clef-py312/bin/python -m pip show \
  torch torch-npu transformers vllm vllm-ascend
```

| 项目 | 2026-11-12 首次环境调查，安装前快照 |
| --- | --- |
| OS / 架构 | Ubuntu 22.04.5 / aarch64 |
| CANN Toolkit / A3 Ops | 9.2.1，innerversion `V100R001C12SPC002B201` |
| NNAL ATB | 9.2.1，build 9.2.1.B020 |
| 驱动 / npu-smi | 26.1.1 |
| NPU | `Ascend910_9382`，2 个逻辑设备，各 65536 MiB HBM |
| 设备映射 | 逻辑 0、1 对应物理 14、15，`npu-smi` 卡号 7、Chip 0 / 1 |
| 基础健康 | ACL 初始化、设备枚举和结束均返回 0 |
| Python | 系统 3.10.12 |
| 推理依赖 | pip、torch、torch_npu、vLLM 等待配置 |
| 构建工具 | GCC/G++ 11.4.0、CMake 3.22.1、Git 2.34.1，Ninja 待配置 |
| 容器配额 | RAM 128 GiB，CPU quota/period = 32 核等价时间配额 |
| 可写空间 | 模型下载前约 299 GiB，`/dev/shm` 16 GiB |
| 网络 | ModelScope、PyPI 小型请求通过，GitHub raw 返回网络不可达 |

机器和软件版本以本轮快照为准。用户截图所展示的软件组合已由本轮选定的 9.2.1 / 2.10 组合更新。[新机器原始记录](evidence/new-machine-2026-11-12.json)

## 3. 模型和工程准备

本轮已创建工程、模型、日志和工作目录，并按原有模型 manifest 从 ModelScope 下载全部 28 个文件。逐文件 SHA-256 校验全部通过，总计 54,989,897,681 字节。[新机器模型校验报告](evidence/new-machine-model-verification-2026-11-12.json)

| 产物 | 路径与验收标准 |
| --- | --- |
| 模型 | `/workspace/yangchen/models/Clef-27B`，28 个文件，54,989,897,681 字节 |
| 下载日志 | `/workspace/yangchen/logs/clef-download-20261112.log` |
| 校验报告 | `/workspace/yangchen/logs/clef-download-20261112.json`，`all_sha256_verified=true` |
| 工程 | 当前本地工作区同步到 `/workspace/yangchen/Clef-27B-ascend` |
| 模型来源 | `Clef-27B/logs/model_manifest.json` 中的逐文件 revision 和 SHA-256 |

下载工具使用系统 Python 标准库，并行请求固定版本文件的 HTTP Range。每个文件通过哈希校验后写入正式文件名。查看进度和报告可运行以下命令。

```bash
tail -n 10 /workspace/yangchen/logs/clef-download-20261112.log
du -sh /workspace/yangchen/models/Clef-27B
python3 -m json.tool /workspace/yangchen/logs/clef-download-20261112.json
```

工程同步包括 Git 基础历史、当前 README 修改、新增文档、下载工具和入口骨架。同步后比较文件 SHA-256，并确认远端 `git status` 对应当前本地修改。

## 4. 安装目标与材料准备

| 组件 | 固定值 |
| --- | --- |
| CANN / A3 Ops / ATB | 9.2.1 |
| Python | CPython 3.12，安装时记录精确补丁版本 |
| torch / torch_npu | `2.10.0+cpu` / `2.10.0.post7` |
| torchvision / torchaudio | `0.25.0` / `2.10.0` 的 CPU 构建 |
| Transformers | `5.19.0` |
| Triton Ascend | `3.6.0.post2` |
| FLA | `flash-linear-attention-npu-a3==26.9.1` |
| flash_attn_npu | `0.5.0` |
| vLLM | `24420b7a341af37a9bc3186e6fc6062512e0aa58` |
| vLLM-Ascend | `89eb8a5e7a76d085b453a1aeadb6883e12d5ea2d` |
| EvalScope | 实际使用源码 `3d39bab09c1d63095490a8b0f4a6dcbcc5ea843e`，独立 API 评测环境 |

安装前在可访问网络的机器上准备 ARM64 的 Python 3.12 发行包、wheel、固定源码和 submodules，记录来源与 SHA-256，再通过 SSH 数据流传入 `work/`。根据本轮网络结果，GitHub 来源材料优先在本地准备，PyPI 和 ModelScope 材料可由新机器直接获取。Python 3.12 独立环境的路径属于本方案的实现选择。

两个核心 wheel 已由官方索引确认，安装材料应匹配 `cp312` 和 `manylinux_2_28_aarch64`。

| wheel | SHA-256 |
| --- | --- |
| `torch-2.10.0+cpu-cp312-cp312-manylinux_2_28_aarch64.whl` | `179451716487f8cb09b56459667fa1f5c4c0946c1e75fbeae77cfc40a5768d87` |
| `torch_npu-2.10.0.post7-cp312-cp312-manylinux_2_28_aarch64.whl` | `2577ea441f7d21937b74750d432f55acefafda7ccbcfe4b07f71fa00170a6b34` |

索引中 TorchNPU 元数据要求 `torch==2.10.0`。完整 wheelhouse 还需包含所有传递依赖。[PyTorch CPU 索引](https://download.pytorch.org/whl/cpu/torch/) · [TorchNPU 元数据](https://pypi.org/pypi/torch-npu/2.10.0.post7/json)

## 5. 详细实施顺序

### 阶段 A — Python 和构建工具

1. 在工程专用前缀准备 CPython 3.12，创建 `/workspace/yangchen/venvs/clef-py312`，记录解释器路径、完整版本和发行包哈希。
2. 配置该环境的 pip，安装 CMake `>=3.26.1`、Ninja、`packaging>=24.2`、`setuptools>=78.1.1,<81`、`setuptools-scm>=8`、`setuptools-rust>=1.9`、wheel、jinja2、pybind11。
3. 检查 GCC/G++、clang-15 和源码配方需要的系统开发库。采用完整 A3 Dockerfile 配方时一并准备其 Rust、protobuf 工具。
4. 初始编译并发采用 `MAX_JOBS=8`，根据容器内存和 CPU 配额调整。
5. 将激活脚本放入工程 `scripts/`，显式选择 CANN、ATB、Python 环境，并把运行日志与缓存放在 `/workspace/yangchen` 下。

### 阶段 B — PyTorch 与基础 NPU 验收

1. 从校验过的 CPU wheel 安装 PyTorch 2.10.0、torchvision 0.25.0、torchaudio 2.10.0，再安装 TorchNPU 2.10.0.post7。
2. 加载 CANN 和 ATB 的环境脚本，检查包的版本、文件路径与 `torch.npu.is_available()`。
3. 分别在逻辑设备 0、1 上执行 FP16、BF16 的 2048×2048 matmul，并同步读取结果。
4. 用同一输入的 CPU FP32 结果检查数值误差、NaN/Inf 和完成状态，记录判定方法及实际误差。
5. 执行 `python -m pip check`，保存版本清单和动态库加载信息。

基础检查示例在目标环境安装后执行。

```bash
source /usr/local/Ascend/cann/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
source /workspace/yangchen/venvs/clef-py312/bin/activate
python - <<'PY'
import torch
import torch_npu
print('torch', torch.__version__, torch.__file__)
print('torch_npu', torch_npu.__version__, torch_npu.__file__)
print('available', torch.npu.is_available())
print('device_count', torch.npu.device_count())
PY
python -m pip check
```

### 阶段 C — vLLM 与 Ascend 插件

1. 将两个固定提交的源码与 submodules 放到 `/workspace/yangchen/work/src/`，记录 SHA 和工作区差异。
2. 安装固定运行依赖 Transformers、Triton Ascend 及配套组件，用 constraints 保持最终 torch / torch_npu 版本。
3. 在 vLLM 源码目录执行 `VLLM_TARGET_DEVICE=empty python -m pip install --no-build-isolation -c /workspace/yangchen/Clef-27B-ascend/requirements/runtime-constraints.txt -e .`，复用已验收的 PyTorch 2.10，并记录实际构建工具快照。
4. 在已加载 CANN/ATB 的环境构建固定版本 vLLM-Ascend，采用 `COMPILE_CUSTOM_KERNELS=1`。构建目标按实际 `Ascend910_9382` 和构建脚本支持列表选择。
5. 安装 `flash-linear-attention-npu-a3==26.9.1`，再按 A3 配方用 `FLASH_ATTN_BUILD_NPU=910` 及 `--no-build-isolation` 安装 `flash_attn_npu==0.5.0`。
6. 检查 Ascend 插件注册、自定义算子加载和最终依赖一致性，再保存完整 `pip freeze`、源码版本及安装日志。

该 vLLM 提交的 PEP 517 构建声明含 torch 2.14.0，本轮实际采用 `--no-build-isolation`，构建与运行均使用已验收的 torch 2.10.0。OpenCV / NumPy 声明、flash 构建并发、完整加载审计及 TP2 lexical embedding 契约的补丁均有来源与验证记录。[实际安装与补丁](installation-2026-11-12.md) · [固定 vLLM 构建声明](https://github.com/vllm-project/vllm/blob/24420b7a341af37a9bc3186e6fc6062512e0aa58/pyproject.toml) · [固定 A3 构建配方](https://github.com/vllm-project/vllm-ascend/blob/89eb8a5e7a76d085b453a1aeadb6883e12d5ea2d/Dockerfile.a3)

### 阶段 D — Clef 完整服务

1. 使用 BF16、TP1、eager、max model len 1024、max-num-seqs 1 启动，关闭 prefix caching 与 chunked prefill。
2. 检查架构 `ClefFlashForJointSchema`、pooling 路径、独立 `joint_head.safetensors` 和全部主干权重加载日志。
3. 对 `POST /v1/systemone` 分别发送 choice、noul、score，再发送三类混合的多字段请求。
4. 校验概率归一化、候选映射、score 的零起始期望、token/span 对齐和请求截断行为。
5. 接通 `Clef-27B/start.sh`、`inference.py`，统一服务模型名 `clef`，支持现有模型路径和 Python 解释器参数。

启动参数与三类请求示例已按固定源码核对，可直接作为该阶段的执行起点。[上游启动和请求说明](upstream-npu-baseline.md)

### 阶段 E — 精度、性能和比赛产物

1. 固定 Decision Index 0.2.1 的源码、数据来源、子集和哈希，使用 HTTP 引擎接入 `POST /v1/systemone`，保存 BFCL、API-Bank 等子项的原生指标和套件汇总。
2. 固定 Typesafe WorkflowEvals 的代码、数据 revision、业务策略和共识参考标签，适配 Clef 客户端后运行四个工作流，保存 `exact_actions`、`primary_action` 等适用指标。
3. 保留已完成的 JevBench / OpenDecisions / Typed Decisions 评测作为工程回归；模型或服务改动后按需要复测协议、联合问题和候选概率。
4. 在固定输入、长度、问题数、候选数、并发和预热条件下，分别记录 TP1 / TP2 的吞吐、p50/p95、成功率、峰值 HBM 与原始日志，并保留 HCCL、多模态和长输入自检。
5. 填写 `results.csv`、`metrics.csv`、`optimization.md`，标明两套主评测与历史回归的来源、配置和完成状态，按正式赛题协议补齐比赛验收。

发布方同口径主评测的当前状态与接入条件见[统一评测口径](evaluation-protocol.md)。比赛专用正式评测数据包、主办方 baseline 数值及性能聚合公式仍待取得。[比赛与数据说明](competition-and-datasets.md)

## 6. 本轮完成状态

- 新机器环境补查和 ACL 设备枚举已完成。
- 模型固定版本下载与逐文件 SHA-256 校验已完成，共 28 个文件、54,989,897,681 字节。
- `AGENTS.md`、`README.md` 和本执行计划已更新，工程及文档同步到新机器，并按文件哈希核对。
- CPython 3.12.16 环境、CMake 3.31.10 与 Ninja 1.13.0 已安装，发行包哈希已记录。
- 固定 PyTorch / TorchNPU / torchvision / torchaudio、Transformers 与 Triton-Ascend 已安装，激活 CANN / ATB 后 `pip check` 通过。两卡 FP16 / BF16 的 2048×2048 matmul 均通过数值验收。
- 固定 vLLM、Ascend 及递归 submodules 已构建，FLA / flash-attn / A3 通信内核安装完成；最终 `pip check`、自动架构识别和 78 个自定义算子加载通过。
- TP1、TP2 完整主干、视觉塔和联合头实际加载。两种 TP 的固定七个决策均与 CPU BF16 参考一致，token/span/候选映射差异为 0，性能和 1 秒采样 HBM 已保存。
- 双 rank HCCL、两张纯色图片和双向视频请求通过。2026-11-12 首轮公开回归评测 2,983 请求、4,583 决策已完成，执行错误为 0。
- Decision Index 0.2.1 与 Typesafe WorkflowEvals 已确认为后续主评测口径，版本固定、客户端接入和实际评分待完成。[统一评测口径](evaluation-protocol.md)
- 当前运行服务为目标机 `127.0.0.1:8000` 上的 `clef`，TP2、4096、BF16、eager、单序列，图片/视频各限 1。2026-11-13 profiling 结束后已按原 argv/cwd 恢复，API PID 为 `115206`；PID 文件 `/workspace/yangchen/logs/clef-tp2-4096-mm-20261112.pid` 已更新。当前日志为 `/workspace/yangchen/logs/profiling/20261112-223732-systemone/restored-tp2-4096/service.log`，服务指针为同一 run 的 `service-current.json`。原工程、vLLM 与 Ascend 的 diff 与实验前完全对应，协议/参考精度/重新编码/多模态均通过。[恢复证据](evidence/profiling-20261112/restoration.json)
- 两份 CSV、运行报告图、兼容记录和完整评测报告已整理。正式赛题输入、官方 baseline 数值和截图规范继续按主办方协议对齐。[实际安装记录](installation-2026-11-12.md) · [实测 baseline](baseline-2026-11-12.md)
