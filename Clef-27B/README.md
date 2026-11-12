# Clef-27B 调优目录

## 当前状态

完整 Clef-27B 已接入固定提交的 vLLM-Ascend，提供 `POST /v1/systemone`，服务模型名为 `clef`。主干 12 个分片、视觉塔和独立联合头已实际加载，联合头 122/122 个参数张量齐全。TP1、TP2 的三类决策、多字段、候选概率与 token/span 自检均通过。[完整 baseline 记录](../docs/baseline-2026-11-12.md)

## 环境记录

2026-11-12 当前机器为 `199.88.54.221`，Ubuntu 22.04 ARM64、CANN / A3 Ops / ATB 9.2.1、驱动 26.1.1，两个 `Ascend910_9382` 各有 64 GiB HBM。独立环境为 CPython 3.12.16、PyTorch 2.10.0+cpu、torch_npu 2.10.0.post7、Transformers 5.19.0。构建、全部固定包导入、`pip check`、双卡基础运算及 HCCL 均通过。[实际安装记录](../docs/installation-2026-11-12.md)

上游模型说明保留其 H200 测试背景，本工程采用上述 A3 实测组合。[上游模型说明](https://www.modelscope.cn/models/Cloudflare/clef)

## 模型目录

默认目录为 `/workspace/yangchen/models/Clef-27B`，可通过 `MODEL_DIR` 或 `--model-dir` 指定。

```bash
cd /workspace/yangchen/Clef-27B-ascend
source scripts/activate_env.sh
bash Clef-27B/start.sh --dry-run
bash Clef-27B/start.sh --model-dir /workspace/yangchen/models/Clef-27B
```

`start.sh` 使用 `PYTHON_BIN` 指定的解释器转发到 `inference.py`，启动前检查完整分片和联合头文件。默认 BF16、TP1、eager、1024、单序列、`127.0.0.1:8000`，关闭 prefix caching 和 chunked prefill；保留视觉塔并跳过视觉预热，初始请求全部使用纯文本。附加 vLLM 参数放在 `--` 后。

```bash
python scripts/verify_systemone.py \
  --report /workspace/yangchen/logs/systemone-smoke.json
```

## 上游参考入口

`joint_schema_model.py` 提供 `load_release_model(local_model_path, device=...)` 和 `systemone(model, processor, request)`，支持 `noul`、`choice`、`score`。[上游文件](https://www.modelscope.cn/models/Cloudflare/clef/files)

比赛入口已对接 vLLM-Ascend 的原生决策协议，`joint` 使用默认值。[赛事说明](https://competition.gitcode.com/competition/2119684851993116674/intro)

## 结果记录

`results.csv` 保存固定请求的服务性能，`metrics.csv` 保存精度、延迟、采样 HBM 和各来源评测指标；未知官方 baseline 数值保留为空。原始日志在远端 `/workspace/yangchen/logs/`，本地证据在 [docs/evidence/](../docs/evidence/)。运行图为实测报告导出图，保留来源时间和原始数据。[优化与兼容记录](optimization.md)
