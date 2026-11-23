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

## 首窗实测

完整 `pip check` 和 2048×2048 双 rank smoke 通过。vLLM `24420b7a341af37a9bc3186e6fc6062512e0aa58`、vLLM-Ascend `89eb8a5e7a76d085b453a1aeadb6883e12d5ea2d`、Transformers 5.19.0、FLA 26.9.1、Triton Ascend 3.6.0.post2 和 FlashAttention NPU 0.5.0 均已验证实际导入路径。启动接续使用固定源码入口、`gdn-bsnd.patch` 和 `clef-api-routes.patch`。首启路径失败和首次 `/v1/decide` 404 的日志保留。

aa3e 真实重捕获全部六份 BF16 `[1595,5120]` 父张量，所有臂使用各自父张量前 967 行。按 rank、层、投影逐一核同 18 份 BF16 原权重切片的历史字节摘要。两 rank 的 6 组数值均过门，输出有限值、重复结果、哨兵和 MIX fresh/reuse 逐位一致检查通过。两 rank 都在分配前拒绝了其他流。每 rank 的 workspace 为 39,902,208 字节，数值阶段 fresh 的 3 次调用各分配 scratch，reuse 的 6 次调用共分配 1 次 scratch 和 6 份独立输出。两 rank 观测到的最大 allocator reserved 均为 1,348,468,736 字节，包含当时的输入、权重、输出和 allocator 缓存。

以下为同一 pair 按每臂双 rank 慢者计算的差值中位数，单位 μs/层。A/A 列为五轮绝对差最大值。

| 计时 | 复用相对 fresh 的收益 | 复用较原厂链的额外耗时 | A/A 最大波动 | 复用优于 fresh | 原厂链优于复用 |
| --- | ---: | ---: | ---: | ---: | ---: |
| wall | 5.85 | 15.48 | 99.62 | 3/5 | 3/5 |
| Event | 19.45 | 32.85 | 51.97 | 4/5 | 5/5 |

复用的收益低于同窗 A/A 波动，Event 完整链在五轮中均落后原厂。保留 S1 原厂 BF16 NZ MLP，收口本候选。数值和计时外层分别为 11 秒、10 秒，设备子进程记录分别为 3.907 秒、3.716 秒。真实请求捕获耗时 19.371 秒，服务加载与恢复另计。

新机日志目录内的证据摘要如下。

| 文件 | SHA-256 |
| --- | --- |
| `full-npu-smoke.json` | `47c96b32c7ee15f50249e81c15ee08479d55e234f7e6b547a28d4c1c1ee2fe29` |
| `new-capture-requests.json` | `50fe3de8c72335ac2a0d9476ec1454164d9eddb0a5eafa3f35d0958ed2a23ac0` |
| `paired967-first.json` | `68ee5894a675eaedb21db9aa95ae75bb56b0fe0eb4303c0bc32cc90030f2bc01` |
| `numeric-allocation-observer.json` | `908af2925bf68226a1bd3053dad71712ca690a351be32f2dfa6c3ea5dc16aaf7` |

## 服务恢复

新机起始状态为卡 6 无进程、18143 无监听。实验后保留通过验收的 aa3e/S1，API PID `86622`，启动 tick `432160960`，TP0/TP1 PID 分别为 `86787`、`86788`。22/22 项通过，覆盖实际 argv 和关键环境键、源码、卡与进程归属、文本和图文协议以及 fresh health。最终 18143 返回 `ready`，捕获 hook 已恢复出厂预像，私有 MIX vendor 未进入实际服务 OPP 路径，标准树干净。

`final-aa3e-checks.json` 的 SHA-256 为 `c679e80fea466fa8ac9a212189d5c9cdb806a01ed9ac9376d32402c270cf9c25`。双端 `current-aa3e-44.json` 的 SHA-256 一致，均为 `417c0de3be5c98be7effc23d4c7a68b9c261087dddf5d8d7de9816e56a7ab42c`。
