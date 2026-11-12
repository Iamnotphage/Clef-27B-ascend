# Clef-27B 优化记录

## 初始化

- 目标平台为昇腾 910A3，当前可见两个 NPU 计算单元，每个 64 GiB HBM。
- 模型源为 `Cloudflare/clef`，本地模型目录为 `/workspace/yangchen/models/Clef-27B`。
- 2026-11-12 已从 ModelScope 下载完整模型，共 28 个文件，54,989,897,681 字节（约 55.0 GB）。所有文件的 SHA-256、主干权重索引和 safetensors 头部边界检查通过。校验明细见 [logs/download_verification.json](logs/download_verification.json)，权重检查见 [logs/model_inspection.json](logs/model_inspection.json)。
- CANN 9.0.0、驱动 26.1.1、Python 3.10.12，PyTorch、torch_npu、vLLM 和 vLLM-Ascend 待配置。

## 待开展

1. 确认运行环境版本组合和完整决策路径的 Ascend 兼容性。
2. 对齐官方基线，验证输出精度并记录真实性能。
3. 根据采集结果定位瓶颈，记录改动、复现命令、精度和性能对比。

## 结果

实测数据、日志和截图待上板运行后补充。
