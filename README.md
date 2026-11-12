# Clef-27B-ascend

Clef-27B 在昇腾 910A3 上的适配与性能调优工程。

## 模型与工作区

| 项目 | 内容 |
| --- | --- |
| ModelScope ID | `Cloudflare/clef` |
| 模型目录 | `/workspace/yangchen/models/Clef-27B` |
| 远端工程 | `/workspace/yangchen/Clef-27B-ascend` |
| 本地工程 | `~/Documents/code/projects/Clef-27B-ascend` |
| 目标设备 | 两个 NPU 计算单元，每个 64 GiB HBM |

上游 README 标注基座为 `Qwen/Qwen3.8-27B`。实际配置使用 `Qwen3_5ForConditionalGeneration`，`model_type=qwen3_5`，主干为 BF16，包含视觉编码器。完整模型包含 12 个主干权重分片，以及 `joint_head.safetensors`、`joint_head_config.json` 和 `joint_schema_model.py`。[模型来源](https://www.modelscope.cn/models/Cloudflare/clef)

## 目录

```text
Clef-27B-ascend/
├── README.md
├── .gitignore
├── scripts/
│   └── download_model.sh
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

环境中安装 ModelScope CLI 后，可在仓库根目录运行下载脚本。第一个参数可指定下载目录。

```bash
bash scripts/download_model.sh
```

## 当前阶段

仓库目录与入口骨架已准备。2026-11-12 已从 ModelScope 下载完整模型，共 28 个文件，54,989,897,681 字节（约 55.0 GB）。所有文件的 SHA-256、主干权重索引和 safetensors 头部边界检查通过。推理适配、运行环境配置和性能评测待后续执行。

入口说明见 [Clef-27B/README.md](Clef-27B/README.md)，环境和优化记录见 [optimization.md](Clef-27B/optimization.md)。

文件快照和校验记录见 [Clef-27B/logs/](Clef-27B/logs/)。本次通过 ModelScope 官方文件接口完成下载，逐文件版本与 SHA-256 保存在 model_manifest.json。
