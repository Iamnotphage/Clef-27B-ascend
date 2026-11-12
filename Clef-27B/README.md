# Clef-27B 调优目录

## 当前状态

本目录包含启动入口骨架、优化记录、结果表头和日志目录。推理逻辑待实现，默认启动会输出待适配提示并以状态码 2 退出。

## 环境记录

2026-11-12 检查远端环境为 Ubuntu 22.04 ARM64、Python 3.10.12、CANN 9.0.0、驱动 26.1.1。两个 NPU 计算单元各有 64 GiB HBM。PyTorch、torch_npu、vLLM 和 vLLM-Ascend 待配置。

上游测试组合为 PyTorch 2.11、Transformers 5.10.2 和单张 H200。参考代码依赖 `torch`、`transformers`、`huggingface_hub`、`safetensors`，图像与视频输入还需要 Pillow。910A3 配套依赖版本待验证。[上游模型说明](https://www.modelscope.cn/models/Cloudflare/clef)

## 模型目录

默认目录为 `/workspace/yangchen/models/Clef-27B`，可通过 `MODEL_DIR` 或 `--model-dir` 指定。

```bash
bash start.sh --help
bash start.sh --model-dir /workspace/yangchen/models/Clef-27B
```

`start.sh` 转发参数至 `inference.py`，可通过 `PYTHON_BIN` 指定 Python 解释器。当前启动命令用于检查入口骨架，推理服务和 Demo 待适配完成后补充。

## 上游参考入口

`joint_schema_model.py` 提供 `load_release_model(local_model_path, device=...)` 和 `systemone(model, processor, request)`，支持 `noul`、`choice`、`score`。[上游文件](https://www.modelscope.cn/models/Cloudflare/clef/files)

调优赛道目标入口为 `POST /v1/systemone`，完整决策路径通过 vLLM-Ascend 实现，后续适配时对齐服务协议。[赛事说明](https://competition.gitcode.com/competition/2119684851993116674/intro)

## 结果记录

`results.csv` 和 `metrics.csv` 当前仅包含项目自定表头，字段用于后续记录性能、精度和对应日志。正式评测时按官方输入、基线和指标定义填写实际结果，原始运行日志保存到 `logs/`。
