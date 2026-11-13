# Clef-27B-ascend

Clef-27B 在昇腾 910A3 上的适配与性能调优工程。

2026-11-13 已完成端到端 profiling：同四补丁、固定307 tokens，各900次请求的 TP1/TP2 吞吐为 **7.144 / 7.917 requests/s**。14个窗口、24份rank trace、长输入/字段/候选/多模态/batching矩阵与恢复验收均已保存；当前服务为已恢复的 TP2/4096 多模态配置。[完整分析报告](docs/profiling-2026-11-12.md) · [性能矩阵](docs/evidence/profiling-20261112/all-performance-matrix.csv)

主评测统一采用 Cloudflare 发布成绩所用的 **Decision Index 0.2.1** 与 **Typesafe WorkflowEvals**。BFCL、API-Bank 等为 Decision Index 的子项，WorkflowEvals 覆盖四个业务工作流。2026-11-13 WorkflowEvals 已全量完成，Decision Index 已于北京时间 13:46 启动后台全量评测，结果以原生输出和退出记录为准。[评测口径](docs/evaluation-protocol.md) · [客户端与预检证据](Clef-27B/logs/official-baseline-preparation-20261113.json)

2026-11-13 两套评测数据已下载到 `/workspace/yangchen/datasets/clef-official-20261113/`，共 41 文件、106,784,678 字节（106.78 MB），服务器逐文件 SHA-256 复核通过。Decision Index 为 84.04 MB（含官方校验元数据），第三方镜像的两份数据解压 SHA-256 与行数均匹配官方 0.2.1；WorkflowEvals 四个官方固定版本快照为 22.75 MB，12 个 Parquet 均可读取。下载采用本地 HTTPS 与 SSH 传输，保留完整来源、revision 和校验记录。[逐文件校验报告](Clef-27B/logs/official-datasets-download-20261113.json)

## 本轮主评测与后台运行

复用现有 A3 / TP2 / BF16 / eager / 4096-token / 单序列服务，通过 `http://127.0.0.1:8000/v1/systemone` 调用。客户端、源码和数据均独立固定版本，保留各套件原生业务逻辑与评分器。

WorkflowEvals 使用官方 `questions=one`、`workers=1`、`question_workers=1`：发票 `all` 模式首例的问题 schema 为 28,719 tokens，超出现有服务上限。北京时间 12:17:41—13:45:27，705/705 case 执行成功，1,524 个策略决策全部计分，12,244 次正式 HTTP 请求全部返回 200。[完整汇总](Clef-27B/logs/workflowevals-baseline-20261113.json)

| 工作流 | 原生 consensus 指标 | 本轮结果 |
| --- | --- | ---: |
| 发票处理 | exact_actions | 62.89%（283/450） |
| 客服 | exact_actions | 72.71%（445/612） |
| 安全事件 | exact_actions | 67.92%（163/240） |
| Agent 运行审查 | primary_action | 69.37%（154/222） |

精度可靠性另按固定输出计算了 10,000 次分组 bootstrap：同一 case 的策略结果整组保留，客服按对话分组，安全事件按 activity 分组，Agent 按 group_id 分组。[95% 抽样区间与方法](Clef-27B/logs/workflowevals-uncertainty-20261113.json)用于量化有限样本的不确定性；模型跨次推理波动与参考标签偏差保留为后续验证项。本轮已有一次正式全量推理；独立重复推理应使用新的 `--name`，以独立生成结果及检查点。

原生 WorkflowEvals 正式命令如下，输出与断点均按 `--name` 保存：

```bash
/workspace/yangchen/venvs/workflowevals-py313/bin/python \
  /workspace/yangchen/work/workflowevals-runner-20261113/workflow_baseline.py \
  --questions one --name clef-tp2-4096-one-20261113
```

Decision Index 使用固定 `--edition 0.2.1`、原生串行 HTTP 引擎和全部 150,759 请求。正式命令为 `bash /workspace/yangchen/Clef-27B-ascend/scripts/official_eval/decision_index_baseline.sh`，已由独立 supervisor 托管（启动时 supervisor PID `146512`、评测 PID `146869`），SSH 断开后继续运行，完成后自动生成原生评分并记录退出码。本轮保留现有 26 候选及 4096-token 容量限制和类型转换行为，错误、拒绝及覆盖率随原生结果报告。2026-11-13 用户要求停止持续监控，辅助守护和轮询已停止。

登录服务器后，可查看进程、实时进度与日志：

```bash
ssh clef
cd /workspace/yangchen/logs/official-baseline-20261113/decision-index
ps -p 146512,146869 -o pid,ppid,etime,pcpu,pmem,args
watch -n 5 'cat full/status.json'
tail -f full-native.log
```

`Ctrl+C` 结束当前查看命令。完成后读取 `exit.json`、`full/scores.json` 和 `full/benchmark-summary.json`；逐请求结果在 `full/results.jsonl`，WorkflowEvals 汇总在相邻 `workflowevals/summary.json`。

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
├── connect.sh
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

## 本机 SSH 连接

本机 `~/.ssh/config` 使用固定别名与 socket，平台跳板凭据在运行 `connect.sh` 时提供：

```sshconfig
Host clef
    HostName 199.88.54.221
    User root
    Port 22
    ControlMaster auto
    ControlPath ~/.ssh/clef-master.sock
    ControlPersist yes
    ServerAliveInterval 30
    ServerAliveCountMax 3
    ProxyCommand /usr/bin/false
```

在仓库根目录运行：

```bash
./connect.sh
ssh clef
ssh clef 'hostname'
./connect.sh status
```

`connect.sh` 优先复用现有主连接；连接缺失时，按隐藏输入提示粘贴最新 `jt_跳板账号:临时令牌` 或平台完整 `ssh -J` 命令，再按 SSH 提示输入服务器密码。跳板账号与令牌会一起变化，需要同时更新。也可通过 `CLEF_JUMP_AUTH` 环境变量提供这组凭据。脚本只解析凭据，输入内容始终作为数据处理，凭据仅用于本次连接进程。

主连接建立后在后台保持，本机同一用户下的终端与 agent 统一使用 `ssh clef`。`ControlPersist yes` 表示无限期保留空闲连接；实际连接受休眠、断网、服务器和平台会话策略影响。连接中断时，`ssh clef` 会快速失败，重新运行 `./connect.sh` 并提供最新凭据即可恢复。

`./connect.sh status` 检查本地主连接进程；执行 `ssh clef 'true'` 可验证远端命令。主动关闭使用 `./connect.sh stop`，会同时断开所有复用会话。远端长任务建议使用 `tmux`。凭据轮换周期与已有会话是否到期关闭，遵循平台实际规则。

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

固定 307-token、三字段、并发 1 的 30 次测试，TP1 为 7.10 请求/秒、p95 143.5 ms，TP2 为 6.84 请求/秒、p95 159.8 ms。HCCL 与红蓝图片、双向视频自检均通过。

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
