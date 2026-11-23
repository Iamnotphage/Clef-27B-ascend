# 新 A3 上的 MIX MLP workspace 复用实验

## 固定环境与产物

新机为 Ubuntu 22.04.5 aarch64，驱动 26.1.1，CANN 9.2.1，ATB 9.2.1.B020。物理卡 6 的 chip 0、1 对应逻辑设备 0、1 和 Phy 12、13。TorchNPU 在两张逻辑设备上均报告 `Ascend910_9382`。实验入口复查主机、boot、映射、占用和 18143。

标准工作树 `/workspace/yangchen/Clef-27B-ascend` 为 `baseline/aa3e`，提交 `bcb98d9201a8f63e009349d1c2a96898d64064cd`，树 `0c984c82e42564baf06bb611f4ab2570a6de30da`，共 1,528 个跟踪文件。基线头文件 SHA-256 为 `aa3e17585f8bd60a4d05e342abc7e2773c79a0b2d49f0370d15e29bcc6b72503`。正式入口保留 S1、TP2 和原厂 BF16 NZ MLP。

实验目录为 `/workspace/yangchen/work/mix-ws-20261123`，日志目录为 `/workspace/yangchen/logs/mix-ws-20261123`。独立环境为 `/workspace/yangchen/venvs/clef-mix-ws-py312`。Python 3.12.16、Torch `2.10.0+cpu`、TorchNPU `2.10.0.post7`、Triton Ascend `3.6.0.post2` 和 FLA `26.9.1` 的发行文件已验 SHA-256。基础 `pip check` 通过。两 rank 的 512×512 FP16 和 BF16 matmul 均通过数值门槛；仓库 smoke 的整体判定另要求 2048×2048，完整栈使用该尺寸验收。

MIX 源提交为 `a5cf535ee24ed26d3d4be7a23ea029aeb3b91466`。CATLASS 提交为 `eec373fbe601446c72cac422cf63142221d85bf9`，其五个合同头文件已按源码清单校验。构建使用 CMake 3.31.10、CANN 9.2.1 和 `ascend910_93`。生成头声明 `int64_t diagnostic`，并由 `readelf` 验证两个 ACLNN 符号。私有解包路径与共享库依赖记录在日志目录下。

| 产物 | SHA-256 |
| --- | --- |
| `custom_opp_ubuntu_aarch64.run` | `f870df75ef14034c5b4dc1ae060bc9d4bb92fb8171ddb3db9693729ec5f952ab` |
| `libcust_opapi.so` | `30c34c85d0ca2e79c8ac5dc25835cc787faa44279d2580d8242232862a0b9046` |

## 输入与配对口径

在新机从 OpenDecisions 1.0.1 固定 revision `ff0b640be2b72c7d06d9d656a34ad6eed79ec7fe` 的第 677 个零起始样本重建 `od-0677`。请求 SHA-256 为 `e5132ea63a84a823e4fd9056322366b61488f95cd1e36528b5b8855ddc0abff6`，与历史清单一致。权重索引及第 02、07、12 分片也已重新核验 SHA-256。新机 aa3e 捕获清单记录两 rank、三层的父张量原始字节、序列化文件和 M967 前缀的独立摘要。

四臂为 `vendor`、`fused_fresh`、`fused_reuse`、`fused_fresh_repeat`。各臂共享同一组输入和原 BF16 NZ 权重。两条 MIX 使用同一 ELF，每次新分配输出并重新获取 executor，复用臂仅持有同 case、线程、设备和流的临时 workspace。数值阶段记录分配和显存；计时阶段使用交接原探针。

决策使用三层 streaming `[31, 0, 63]` 的五轮交错配对。每个 arm 在同一 pair 取双 rank 慢者，分别报告完整链 wall 和 Event 时间。预热分列，A/A 使用两个 fresh 臂。运行结果须同时核对完整原厂链和本窗 A/A 波动。
