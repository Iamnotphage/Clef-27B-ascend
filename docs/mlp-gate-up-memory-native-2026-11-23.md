# BF16 gate/up 搬运和下发复核

## 权重排列与短形状补齐

使用新 A3 卡 6 的逻辑设备 0/1、aa3e 真实 `od-0677` BF16 捕获和同源模型权重。M307 与 M967 取 M1595 父张量前缀。按层 `[31,0,63]` streaming，九轮交错。以下为每轮取双 rank 慢者后，候选减原厂的配对差中位。单位为 Event µs/层。

| M | 试验 | 配对差 | 原厂 A/A 最大 | 候选胜出 |
| ---: | --- | ---: | ---: | ---: |
| 307 | 原厂 GEMM 使用成对列权重 | +7.53 | 26.21 | 3/9 |
| 307 | 自定义 GEMM 使用同一成对列权重 | +58.54 | 26.21 | 0/9 |
| 307 | 预先补到 320，计 GEMM | +2.00 | 26.21 | 4/9 |
| 307 | 在线补到 320，计补齐与 GEMM | +28.47 | 26.21 | 0/9 |
| 307 | 在线补齐、裁切并接原厂 SwiGLU/down | +45.76 | 7.17 | 0/9 |
| 967 | 原厂 GEMM 使用成对列权重 | +0.35 | 26.68 | 4/9 |
| 967 | 自定义 GEMM 使用同一成对列权重 | +23.76 | 26.68 | 1/9 |

M307 补齐前后的短 GEMM 和 MLP 完整链在六组真实输入上逐位一致。原厂成对列在逆排列后六组通过原 BF16 误差阈值。切换列排列后同一原厂算子的延迟差在 A/A 内。在线补齐这条候选结束。

## Level1 计数

`Memory`、`L2Cache`、`PipeUtilization` 在 rank 0 分别采集。每组按实际执行顺序关联 84 条算子记录，每臂 12 条。关联时核对输入形状与算子名。原厂走 `aclnnMatmulWeightNz_MatMulCommon_MatMulV2`，24 个 AI Core block。原型使用同一数量的 Cube block，MIX 启动 48 个向量 sub-block。

| M | 臂 | 纯 task µs，Pipe 组 | task 间隔 µs，Pipe 组 | L2 读命中率 |
| ---: | --- | ---: | ---: | ---: |
| 307 | 原厂 streaming | 233.86 | 0.0 | 76.06% |
| 307 | 原厂成对列 streaming | 232.94 | 0.0 | 76.17% |
| 307 | 自定义 streaming | 224.32 | 40.4 | 75.77% |
| 307 | 原厂同一层热权重 | 207.90 | 0.0 | 87.37% |
| 967 | 原厂 streaming | 595.94 | 0.0 | 90.78% |
| 967 | 原厂成对列 streaming | 596.17 | 0.0 | 90.82% |
| 967 | 自定义 streaming | 573.36 | 2.8 | 90.77% |

task 和间隔均取中位。间隔是同臂相邻设备 task 起止之差，各间隔按原始时间戳计算。L2 命中率用两个读通道的 hit/(hit+miss allocate)。M307 原厂、自定义在 Pipe 组的 MAC 时间均约 167.4 µs，Fixpipe 分别约 4.77/14.37 µs。M967 原型的 MTE2 时间比原厂多约 60.81 µs，Fixpipe 多约 26.45 µs，流水可重叠。这些分项不可直接加到端到端上。Memory 组的 `aic_main_mem_read_bw` 未通过绝对物理 HBM 字节口径校准，本次保留原始计数。现有数据表明权重列排列与 L2 命中率的差异很小，短形状的设备 task 之间出现了显著空档。

权重排列和补齐计时报告 `timing.json` SHA-256 为 `7438bfd0ed694e7252e9baabb59c5f360c7338ab4181dad8f96bab6ea1c3e2ab`。三组 `kernel_details.csv` 摘要如下。

| 计数组 | SHA-256 |
| --- | --- |
| Memory | `738782105d7f28db4ea359da25b99ef2e913ece8d01760907b822c1288ff2fef` |
| L2Cache | `1b2beab43ebb88bcb006765f3d636af9986a36df69e87631ac6006913cc96bd3` |
| PipeUtilization | `ad4978b461cf2fa53e68d936e46fa38fcfd344df8df80dd9e47eca27416276db` |

原始日志为 `/workspace/yangchen/logs/gate-up-memory-20261123.frTSYRRP`。设备窗口 53 秒，随后恢复原厂 aa3e/S1 并通过 22/22 项。

## 原生下发与长形状混合链

新增试验用 ATen/ACLNN 桥接，通过显式绝对路径加载同一私有 op_api。每调用新分配输出和 workspace，重新获取 executor，跟随 TorchNPU 当前设备及流。算子 ELF、权重和 BF16 累加与舍入保持一致。原生融合臂用配对列权重并调用原型的 SwiGLU。混合臂用未重排的原始 NZ 权重，调用同一原生 GEMM 的 raw 模式，接原厂 SwiGLU 和 down。

第一窗的 18/18 组比较中，原生与 Python 下发的 raw/fused 输出分别逐位一致。原生下发较 Python 的完整链 Event 配对中位数减少 61.71、56.95、73.76 µs/层，对应 M307、M967、M1595，三个形状均九轮全部获胜。对原厂完整链，M1595 的配对 Event 为 −39.35 µs/层，9/9 胜，超过原厂 A/A 最大 30.83。M307/M967 的差值分别为 −4.28/+4.41 µs/层，处于 A/A 内。

同窗最小 Level1 复查关联 72/72 条任务。M307 的 Python/原生 task 间隔分别为 2.98/2.92 µs，M967 为 2.82/2.86 µs。第一次采集 Python 的 40.4 µs 间隔本窗未重复。未采集的同 ELF 配对给出直接的下发路径差；采集窗口的逐 task 间隔不用于绝对归因。该窗口设备外层 25 秒，恢复另计。

第二独立窗验证混合路径和邻近长形状；M1536、M1568 同样取真实 M1595 父张量前缀。下表是完整链九轮配对差中位数，负数为候选更快，单位 µs/层。胜出和 A/A 分别给出 Wall / Event。所有 30/30 组通过数值验证，混合链对原厂的最大相对 RMSE 为 0.0002841。

| M | 原生融合 Wall / Event | 混合链 Wall / Event | 混合链胜出 | 原厂 A/A 最大 |
| ---: | ---: | ---: | ---: | ---: |
| 307 | +4.44 / +8.12 | −0.03 / +0.53 | 5/9 / 4/9 | 27.31 / 30.31 |
| 967 | +5.32 / +6.47 | −0.21 / −3.00 | 5/9 / 5/9 | 265.35 / 25.89 |
| 1536 | −9.37 / −24.21 | −28.65 / −26.19 | 8/9 / 8/9 | 18.36 / 24.51 |
| 1568 | −46.31 / −44.39 | −92.01 / −80.97 | 9/9 / 9/9 | 42.28 / 30.29 |
| 1595 | −47.87 / −49.19 | −79.81 / −79.40 | 9/9 / 9/9 | 32.01 / 32.89 |

M1568 和 M1595 的混合链约快 5%，Wall 与 Event 的配对差均超过各自 A/A 最大值。M967 仍在 A/A 内。历史全量输入在 1536–1595 token 仅有 OpenDecisions 1/2352 和 ERQA 2/400，两套数据的长度 P95 均在此范围之外。保留候选供长输入进一步验证，S1 继续走原厂链。完整服务基准未纳入该实验。

| 文件 | SHA-256 |
| --- | --- |
| `native_dispatch.so` | `a7993df88b8d32958065b0432b9be7b447b5e5e09e766d55452c0c77546e436e` |
| 原生窗 `timing.json` | `22b1997dda9c5d7240908a87bff469be72eb90f1acef965b00bed96ae0ffe603` |
| 原生窗 `kernel_details.csv` | `c8524dbe005479544ba292ae9824055b3af6ef1a32e75fc3d86ce5d652982a57` |
| 混合窗 `timing.json` | `6651c3239a22b3845cfa2be7b739f2e9e664fb1bc21f143fa9f082806d0e6d38` |

工作目录为 `/workspace/yangchen/work/gate-up-memory-20261123.frTSYRRP`，原始日志另存于 `/workspace/yangchen/logs/gate-up-native-20261123.frTSYRRP` 与 `/workspace/yangchen/logs/gate-up-hybrid-20261123.frTSYRRP`。使用固定 venv 下的 `runtime/probe_mlp_gate_up_memory.py`，传入既有 `--vendor`、`--model-dir`、`--captures` 和新 `--output`。计数加 `--metric`，原生计时加用 `build_native_dispatch.sh` 构建的 `--native-library`，混合计时再加 `--hybrid`。

三个设备窗口共 93 秒，服务加载和恢复另计。最后一次用原厂 aa3e/S1 重新启动，22/22 项验收通过，18143 返回 `ready`。新 API 进程为 96565，TP2 worker 为 96692/96693。最终验收 `final-aa3e-checks.json` SHA-256 为 `cc92ae8c108dac32c964c301fa36f1fca39ccd1a2bebe7b813751f369ed0d98f`。双端恢复指针 SHA-256 一致，为 `c9ef865b0be60dd5c491c7c1b2eb40a19dbee05330628c5863f9265970ca6a11`。
