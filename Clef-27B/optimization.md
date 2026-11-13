# Clef-27B 优化记录

## 首次调查快照，2026-11-12

- 目标平台为昇腾 910A3，当前可见两个 NPU 计算单元，每个 64 GiB HBM。
- 模型源为 `Cloudflare/clef`，本地模型目录为 `/workspace/yangchen/models/Clef-27B`。
- 2026-11-12 已从 ModelScope 下载完整模型，共 28 个文件，54,989,897,681 字节（约 55.0 GB）。所有文件的 SHA-256、主干权重索引和 safetensors 头部边界检查通过。校验明细见 [logs/download_verification.json](logs/download_verification.json)，权重检查见 [logs/model_inspection.json](logs/model_inspection.json)。
- 早期实例为 CANN 9.0.0、驱动 26.1.1、Python 3.10.12，当时推理依赖待配置；保留该次环境来源与采集记录。[早期环境调查](../docs/environment.md)

## 当前 A3 baseline，2026-11-12

新机器 `199.88.54.221` 使用现有 CANN / A3 Ops / ATB 9.2.1，独立 CPython 3.12.16、torch 2.10.0+cpu、torch_npu 2.10.0.post7。固定 vLLM / Ascend 源码、依赖、构建日志、最终 `pip check` 和 78 个自定义算子导入均已记录。[安装记录](../docs/installation-2026-11-12.md)

完整主干 12 分片、视觉塔和独立联合头已在 TP1/TP2 实际加载。联合头 122/122 参数张量、128,056,324 参数加载齐全，strict tracking 的缺失列表为空。CPU 参考为完整 27,484,784,884 参数 BF16 模型；NPU 的 GDN 派生卷积布局参数额外占 3.75 MiB，有逐项核算证据。

| 固定条件：307 tokens，3 字段，候选 2/2/3，预热 3，正式 30，并发 1，max len 1024 | TP1 | TP2 |
| --- | ---: | ---: |
| 请求/秒 | 7.095650 | 6.838840 |
| p50，ms | 140.301320 | 144.985050 |
| p95，ms | 143.511251 | 159.794070 |
| CPU 参考离散决策一致 | 7/7 | 7/7 |
| 最大候选概率绝对差 | 0.001453638 | 0.000256002 |
| 物理 14 / 15 的采样 HBM，MiB | 59964 / 2885 | 60007 / 59667 |

HBM 为 1 秒采样的设备占用，包含设备基础及缓存占用。TP2 每卡权重约 25.76 GiB；默认 0.90 利用率下，KV 池扩大到约 28.76 / 28.77 GiB。容量、权重和缓存预算应分别比较。上述吞吐与延迟限定为同一组 30 次短请求。

## 兼容与复现改动

1. 保留 Triton-Ascend 的 NumPy 1.26.4 约束，将 vLLM 的 OpenCV 声明固定为 4.11.0.86；图片和视频真实解码已通过。
2. 固定 vLLM 构建声明含 torch 2.14.0；本轮采用 `empty`、`--no-build-isolation` 和 constraints，构建与运行复用验收过的 torch 2.10。
3. flash-attn 编译线程池显式读取 `MAX_JOBS=8`；Arctic 预装 nanobind / grpcio-tools，FLA 补齐其声明的 expecttest。
4. 加载后输出 `CLEF_LOAD_AUDIT`，记录实际模块、张量数量、设备/dtype 和缺失检查。
5. TP2 首请求暴露上游基类 embedding 契约缺失；在 Ascend 构造函数补齐 `use_fused_embedding=False`、`reduce_results=True`、`parallel_group=output_reduce_group`。原有 Ascend forward 的 AST 保持一致，修复后通过真实两卡精度与性能检查。

四份补丁保存在 [scripts/patches/](../scripts/patches/)，源码固定提交和补丁 SHA、最小验证及实际部署日志见安装文档。

## 结果

TP2/4096 的完整公开评测为 2,983 请求、4,583 决策，执行错误为 0：JevBench 公开 231 条 `public_accuracy=0.8701`；OpenDecisions 2,352 条 `capability=0.7443`；Typed Decisions 400 条、2,000 决策的准确率 `0.7165`。HCCL FP32/BF16、红蓝图片和双向视频自检全部通过。

[完整 baseline 与逐题日志路径](../docs/baseline-2026-11-12.md) · [性能 CSV](results.csv) · [指标 CSV](metrics.csv) · [性能报告图](result.png) · [NPU 实测报告图](npu_device_call.png)

两张图由真实报告和 SSH CLI 输出导出，保留来源时间。原始 Terminal 窗口访问受计算机工具安全规则限制。正式比赛输入、官方 baseline 数值及最终图像规范按主办方协议对齐；CSV 中未知 baseline 字段保持为空。

## 后续比较项

按同一精度判据比较固定 KV 池预算、更长输入、更多字段和更高并发，分别记录缓存容量、设备占用和请求性能。公开文本指标、概率分布及已有多模态自检作为回归依据。

## 第二轮性能调优，2026-11-13

A3 单物理卡内两个逻辑 NPU 边界内完成。完整实验报告见 [docs/tuning2-2026-11-13.md](../docs/tuning2-2026-11-13.md)，远端逐点原始记录与 master-results.csv（43 行）在 `/workspace/yangchen/logs/tuning/20261113-085408-perf2a/`。全部数字为关闭 profiler 的正式测量（每点 ≥300 请求、≥30 秒；稳定结论三轮交错，batch 分布由 Prometheus 直方图差分重建并逐点校验）。

| 改动或配置 | 实测收益（fixed307，TP2/4096/eager） | 决定 |
| --- | --- | --- |
| batching seqs4 + 并发 8（真实 B4） | 8.237 → 19.325 rps（CV 0.05%）；dec/s 24.71 → 57.98 | 采纳为 S2 配置 |
| 独立 `--max-num-batched-tokens` 参数 | 长输入 3868 tokens：4096→16384 预算仅 1.964→2.118 rps（MLP 线性封顶） | 参数保留，默认跟随 max-model-len |
| 双 TP1 副本（同卡双实例+入口分流） | C2 14.268 rps @ p95 141 ms（容量假设 14.288 获证实） | 低延迟中并发备选 |
| `weight_nz_mode=2`（BF16 NZ） | 线性层设备时间 −9.3%（46.50→42.17 ms/req/rank）但端到端 −0.8%～−1.4% | 拒绝，保留默认 mode 1 |
| schema token 有界缓存 | encode CPU −89.6%；s4c8 +0.28%（19.380）；质量逐位同界 | 默认关闭；S2 可开启（env 一行） |
| PIECEWISE 编译（去 eager） | 编译 +29.7 s、捕获 3 图；s4c8 19.340 rps 持平 | 不采纳 |
| KV 池 12 GiB（原 28.5） | 307/B4、3868 长输入、冷图全部零回退；回收约 16.5 GiB/rank | 容量可选项 |

质量回归：推荐配置（seqs1）完整 OpenDecisions 1.0.1 2,352 条重跑，Calibration 82.7941149442 / Intelligence 66.0568428085 / Capability 74.4254788763、accuracy 1906/2352、Score MAE 0.401389，与既有 baseline **逐位一致**；B4 混合并发与缓存开启两种条件下 771/771 离散决策一致、概率最大差 0.000851（阈值 0.03）。

S1/S2（results.csv 12 列行，配置来源各自一致）：S1 = seqs1，p95 130.296 ms（三轮均值）；S2 = seqs4，57.98 dec/s @ 并发 8（关缓存）或 58.14（开缓存）。备选低延迟部署：双 TP1 C2 14.27 rps @ p95 141 ms。
