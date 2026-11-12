# Clef-27B-ascend

Clef-27B 在昇腾 910A3 上的适配与性能调优工程。

## 模型与工作区

| 项目          | 内容                                        |
| ------------- | ------------------------------------------- |
| ModelScope ID | `Cloudflare/clef`                           |
| 当前 SSH 目标 | `root@199.88.54.221`                        |
| SSH 跳板     | `113.47.8.48:2234`，账号与令牌由平台即时提供 |
| 模型目录      | `/workspace/yangchen/models/Clef-27B`       |
| 远端工程      | `/workspace/yangchen/Clef-27B-ascend`       |
| 本地工程      | `~/Documents/code/projects/Clef-27B-ascend` |
| 目标设备      | 两个 NPU 计算单元，每个 64 GiB HBM          |

上游 README 标注基座为 `Qwen/Qwen3.8-27B`。实际配置使用 `Qwen3_5ForConditionalGeneration`，`model_type=qwen3_5`，主干为 BF16，包含视觉编码器。完整模型包含 12 个主干权重分片，以及 `joint_head.safetensors`、`joint_head_config.json` 和 `joint_schema_model.py`。[模型来源](https://www.modelscope.cn/models/Cloudflare/clef)

## 目录

```text
Clef-27B-ascend/
├── AGENTS.md
├── README.md
├── .gitignore
├── docs/
│   └── new-machine-plan.md
├── scripts/
│   ├── download_model.sh
│   └── download_model_manifest.py
└── Clef-27B/
    ├── README.md
    ├── start.sh
    ├── inference.py
    ├── optimization.md
    ├── results.csv
    ├── metrics.csv
    └── logs/
```

`Clef-27B/` 按调优赛道组织提交文件。后续上板运行完成后补充真实的 `result.png`、`npu_device_call.png`、评测数据与原始日志。[赛事说明](https://competition.gitcode.com/competition/2119684851993116674/intro)

## 下载

按仓库 manifest 固定逐文件版本并校验，可直接使用 Python 标准库下载。重复执行会验证已完成文件并续传 `.part` 文件。

```bash
python3 scripts/download_model_manifest.py \
  --manifest Clef-27B/logs/model_manifest.json \
  --destination /workspace/yangchen/models/Clef-27B \
  --report /workspace/yangchen/logs/clef-download-20261112.json \
  --workers 8
```

环境中安装 ModelScope CLI 后，可在仓库根目录运行下载脚本。第一个参数可指定下载目录。

```bash
bash scripts/download_model.sh
```

## 当前阶段

2026-11-12 已切换到 `199.88.54.221`。新机器实测安装 CANN Toolkit、A3 Ops 和 NNAL ATB 9.2.1，驱动为 26.1.1。ACL 初始化成功，返回两个 `Ascend910_9382` 设备。容器资源配额为 128 GiB RAM 和 32 核等价 CPU 时间，可写盘检查时剩余约 299 GiB。

独立 CPython 3.12.16 环境已建立，PyTorch 2.10.0+cpu、torch_npu 2.10.0.post7、torchvision 0.25.0+cpu、torchaudio 2.10.0+cpu、Transformers 5.19.0 和 Triton-Ascend 3.6.0.post2 已安装。激活 CANN / ATB 后 `pip check` 通过，两个设备的 FP16 / BF16 2048×2048 matmul 均通过 CPU FP32 参考误差与 NaN/Inf 检查。[实际安装与验收记录](docs/installation-2026-11-12.md)

固定提交的 vLLM 已按 `empty` 安装，vLLM-Ascend、FLA、flash-attn 和 A3 通信内核构建及完整环境验收均已完成。完整服务自动识别为 `ClefFlashForJointSchema`，主干、视觉塔和联合头实际加载；TP1/TP2 的 7 个固定决策均与 CPU 参考一致，token/span 映射完全对应。[实测 baseline](docs/baseline-2026-11-12.md)

固定 307-token、三字段、并发 1 的 30 次测试，TP1 为 7.10 请求/秒、p95 143.5 ms，TP2 为 6.84 请求/秒、p95 159.8 ms。HCCL 与红蓝图片、双向视频自检均通过。TP2/4096 完整公开评测共 2,983 请求，执行错误为 0：JevBench 公开 231 条的 `public_accuracy=0.8701`，OpenDecisions 2,352 条的 `capability=0.7443`，Typed Decisions 400 条、2,000 决策的准确率为 `0.7165`。

新机器已完成全部 28 个模型文件的下载，总计 54,989,897,681 字节（约 55.0 GB），逐文件 SHA-256 与原 manifest 一致。源码提交、兼容补丁、运行依赖和实际日志均已记录。[新机模型校验报告](docs/evidence/new-machine-model-verification-2026-11-12.json)

入口说明见 [Clef-27B/README.md](Clef-27B/README.md)，环境和优化记录见 [optimization.md](Clef-27B/optimization.md)。

文件快照和校验记录见 [Clef-27B/logs/](Clef-27B/logs/)。本次通过 ModelScope 官方文件接口完成下载，逐文件版本与 SHA-256 保存在 model_manifest.json。

## 查看当前环境

登录后可用以下命令查询 NPU、CANN、驱动、Python 包和容器资源。目标 Python 环境创建后，通过其解释器查询包版本。

```bash
source /usr/local/Ascend/cann/set_env.sh
npu-smi info
npu-smi info -m
cat /usr/local/Ascend/cann/aarch64-linux/ascend_toolkit_install.info
cat /usr/local/Ascend/driver/version.info
python3 --version
python3 -m pip show torch torch-npu transformers vllm vllm-ascend
cat /sys/fs/cgroup/memory/memory.limit_in_bytes
cat /sys/fs/cgroup/cpu/cpu.cfs_quota_us
cat /sys/fs/cgroup/cpu/cpu.cfs_period_us
df -h /workspace /dev/shm
```

系统 Python 保留原有状态，推理环境通过以下方式加载。128 GiB 对应 `137438953472` 字节，CPU quota `3200000` 除以 period `100000` 得到 32 核等价时间配额。

```bash
source scripts/activate_env.sh
python -m pip check
python scripts/verify_npu_environment.py --stage runtime --require-extensions \
  --output /workspace/yangchen/logs/runtime-smoke.json
```
