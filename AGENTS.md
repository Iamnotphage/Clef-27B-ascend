# Clef-27B Ascend 工程约定

## 沟通与目标

- 使用中文直接陈述结论、证据和下一步，遵循用户关于肯定表达的要求。
- 当前任务为在昇腾 A3 上建立可复现的 Clef-27B baseline，包含完整主干、联合决策头和 `POST /v1/systemone`。
- 2026-11-12 用户已选择新机器现有的 CANN 9.2.1，并采用 PyTorch 2.10 配套环境。
- 环境安装、服务运行及精度性能结论使用实际日志和检查结果标注完成状态。

## 阶段提交与推送

- 每完成一个可验证的小阶段，或取得有实际证据的阶段性进展，及时整理相关代码、文档和验证记录，执行 `commit + push`，保持小步、可回溯的提交。
- 小阶段提交与推送按本约定直接执行。提交说明写清本阶段的具体改动和验证结果；提交前完成对应的必要检查。
- 提交内容聚焦当前阶段，保留用户已有修改，凭据及模型权重大文件按仓库现有约定管理。推送使用正常快进流程，遇到远端更新先核对并整合。

## 当前机器与工作目录

| 项目 | 当前值 |
| --- | --- |
| SSH 目标 | `root@199.88.54.221` |
| 跳板 | `113.47.8.48:2234`，使用平台即时发放的跳板账号及令牌 |
| 远端工程 | `/workspace/yangchen/Clef-27B-ascend` |
| 模型目录 | `/workspace/yangchen/models/Clef-27B` |
| 日志目录 | `/workspace/yangchen/logs` |
| 下载及构建中间文件 | `/workspace/yangchen/work` |
| 计划 Python 环境 | `/workspace/yangchen/venvs/clef-py312` |
| 共享数据 | `/workspace/shared_assets/datasets`，按平台说明只读使用 |

SSH 模板如下。凭据通过当前会话或平台提供，仓库保存脱敏连接信息。

```bash
ssh -J '<跳板账号>:<临时令牌>@113.47.8.48:2234' root@199.88.54.221
```

新机实测为 Ubuntu 22.04.5 / aarch64，两个 `Ascend910_9382`，各 64 GiB HBM，逻辑设备 0、1 对应物理 14、15。`npu-smi` 卡号为 7。容器资源为 128 GiB 内存和 32 核等价 CPU 时间配额。设备编号和资源配额在实例变化后重新查询。

## 固定的软件组合

| 组件 | 目标 |
| --- | --- |
| CANN Toolkit / A3 Ops / NNAL ATB | 9.2.1，新机器已经安装 |
| Python | 独立 CPython 3.12 环境，安装时记录补丁版本 |
| torch | `2.10.0+cpu` |
| torch_npu | `2.10.0.post7` |
| torchvision / torchaudio | `0.25.0` / `2.10.0`，采用对应 CPU 构建 |
| Transformers | `5.19.0` |
| Triton Ascend | `3.6.0.post2` |
| FLA NPU A3 | `flash-linear-attention-npu-a3==26.9.1` |
| flash_attn_npu | `0.5.0` |
| vLLM | `24420b7a341af37a9bc3186e6fc6062512e0aa58` |
| vLLM-Ascend | `89eb8a5e7a76d085b453a1aeadb6883e12d5ea2d` |

构建前加载所选 CANN 与 ATB 的环境脚本，安装 CMake 至少 3.26.1、Ninja 及官方 A3 配方要求的编译依赖。按固定源码的 `empty` 构建流程安装 vLLM，随后构建 Ascend 插件与 A3 扩展。运行依赖由 constraints 固定，构建隔离环境的依赖单独记录。验收包含 `pip check`、实际版本、加载路径和 NPU 运算结果。

## 模型与入口

- 模型以 `Clef-27B/logs/model_manifest.json` 中的逐文件 revision、大小和 SHA-256 为准。
- `scripts/download_model_manifest.py` 使用系统 Python 标准库下载，支持 `.part` 续传、并行 HTTP Range 和 SHA-256 校验。成功报告保存到工程或远端日志目录。
- `Clef-27B/start.sh` 和 `inference.py` 是比赛入口。接入固定版本的 vLLM-Ascend 服务，保持 `MODEL_DIR` / `--model-dir` 和 `PYTHON_BIN` 的调用方式。
- 模型加载必须覆盖 `joint_head_config.json`、`joint_head.safetensors` 和完整主干，确认识别为 `ClefFlashForJointSchema`。
- 首轮配置为 BF16、TP1、eager、max model len 1024、batch 1、纯文本，关闭 prefix caching 和 chunked prefill。
- 服务模型名统一为 `clef`。请求中的 `joint` 采用默认值，三种决策由模型联合决策头计算。
- 在相同输入上检查 `choice`、`noul`、`score`、候选概率、多字段和 token/span 对齐，再扩展 TP2 与多模态。

## 数据、验证与记录

- 转换数据与缓存放在 `/workspace/yangchen` 下，保留共享数据来源、revision、split、模板和哈希。
- 功能自检完成后，按 JevBench / OpenDecisions、Typed Decisions 的顺序建立项目指标，正式比赛指标使用主办方指定协议。
- 下载完成以逐文件 SHA-256 校验报告为准。依赖完成以版本与 NPU smoke 日志为准。完整 baseline 以服务、精度和性能记录为准。
- 新机器的当前状态与后续执行计划见 `docs/new-machine-plan.md`，旧机器记录在文档中保留采集时间和来源。
- 修改任务涉及的代码和文档，保留用户已有修改。提交前检查文档链接、命令语法以及本次实际执行产生的结果。
