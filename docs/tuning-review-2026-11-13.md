# Clef-27B A3 profiling 调优建议

分析日期：2026-11-13，北京时间。依据为 [profiling 实测报告](profiling-2026-11-12.md)、其原始证据摘要，以及 `ssh clef` 对当前进程、固定依赖源码和 API 的只读复核。本轮完成分析与建议归档；性能数字沿用已有实验，优化实现与新配置收益仍待 A/B。现场复核没有重启服务、修改依赖或执行压测。

建议按两条路线推进：**吞吐先验证 batching 与双 TP1 副本；单请求延迟先试 BF16 权重 NZ 格式，再处理 GDN 布局转换与下发。** schema token 缓存适合作为独立的小改动；显存池调整以容量为目标。联合头优化放在主干和调度之后。

## 1. 证据支持的判断

| 观察 | 调优含义 |
| --- | --- |
| 固定 307 tokens，TP1/TP2 为 7.144/7.917 rps，p50 为 139.894/124.836 ms | TP2 吞吐提升 10.83%，p50 降低 10.76%；两卡部署方式值得对照 |
| TP2 p95/p99 相对 TP1 分别降低约 3.57%/1.53%；三轮吞吐 CV 为 2.6485%/0.0725% | 短输入尾延迟优势较小，生产选择需用稳定窗口和负载下的 p95/p99 |
| 同并发 4，seqs1→seqs2，吞吐 8.343→14.845 rps，p50 478.472→260.490 ms，p95 485.009→263.949 ms | batching 同时提高该并发下的吞吐并降低请求延迟；吞吐提升 77.93% |
| 最佳 batching 窗口为 60 请求、4.042 秒，p99 314.840 ms | 14.845 rps 是探索值，优先补三轮稳定验证 |
| 3868 tokens，TP1/TP2 平均 871.441/523.015 ms | TP2 对长输入有约 1.67 倍延迟优势 |
| MLP 两类大 MatMul，TP1/TP2 rank0 约 62.47/32.16 ms/request | MLP 缩放接近 1.94 倍，且具有最大的已识别算子优化时间规模 |
| GDN 主核约 13.95/10.80 ms/request；Top20 的 Transpose 合计 TP1 8.56、TP2 rank0/rank1 7.88/8.13 ms/request | GDN 与布局处理值得独立分析，两种配置均有约 8 ms 的可见转换成本 |
| 联合头 active 约 2–3 ms/request | 局部优化适合按毫秒级收益安排投入 |

来源：[正式稳定性](evidence/profiling-20261112/stable-summary.json)、[完整矩阵](evidence/profiling-20261112/all-performance-matrix.csv)、[Top20](evidence/profiling-20261112/top20.csv)。Transpose 数字按各 rank 原生 Top20 的 `cumulative_us` 求和后除以 5 个请求，属于已列算子的设备时长。各组进入 Top20 的 shape 集合不同，全体转换的精确缩放还需完整算子表；逐项 flow 用于确认模块归属。[本轮计算及现场记录](evidence/tuning-review-20261113.json)

分析中保留四项统计边界：

- TP2 原生 profiler 已使平均延迟增加 39.67%；吞吐和 SLA 采用关闭 profiler、观测变量后的结果。
- all-reduce elapsed 包含等待并可与计算重叠；650 次通信的 rank0 等待累计 392.297 ms，无法直接折算为每请求可消除延迟。单一 SIO 通道净 transit 约 14.45 ms/5 请求提供另一口径。
- head-labelled 长等待与主干执行重叠，头的 active 计算继续按 2–3 ms 处理。
- MTE2 高占用支持优先调查数据供给。HBM 带宽饱和度仍需校准计数器范围与带宽分母；独立 workspace 总峰值继续保留 unknown。

## 2. 配置优化：优先形成可部署结论

### 2.1 稳定验证 TP2 batching，并独立控制 token 预算

先比较 `max-num-seqs=1/2`、HTTP 并发 `1/4`，再补 `seqs=4, concurrency=8`。已有 `seqs=2, concurrency=2` 仍形成 B1，`seqs=4, concurrency=4` 形成 B1/B3；参数上限与实际 batch 分布需要一起记录。

seqs2/C4 的每请求 all-reduce 启动次数由 130 降至 67.65，减少 47.96%；逻辑 payload 维持 406,456,320 B/request。batching 的依据包括启动次数摊薄及权重复用，收益需用真实 batch 分布验证。

当前 [启动入口](../Clef-27B/inference.py) 将 `--max-num-batched-tokens` 绑定到 `max_model_len`。建议将其做成独立参数，基于目标 batch 设置并重新检查内存：

| 单请求 tokens | B2 所需最低 token 预算 | B4 所需最低 token 预算 | 可测试预算 |
| ---: | ---: | ---: | --- |
| 307 | 614 | 1228 | 当前 4096 足够 |
| 1024 | 2048 | 4096 | 当前 4096 足够 |
| 2048 | 4096 | 8192 | 4096 / 8192 |
| 3868 | 7736 | 15472 | 8192 / 16384 |

上述是容纳完整输入的必要预算，实际合批仍取决于到达时序、调度与可用状态块。chunked prefill 关闭时，4096 预算限制两个 3868-token 请求在同一批执行。提高预算会改变 dummy profile、激活和 KV/Mamba 分配，应单独作为实验变量。

通过标准：三轮各预热至少 10、正式至少 300 请求且至少 30 秒，成功率 100%，协议和精度通过，实际 B2/B4 有证据，并按业务延迟目标选择配置。已有 p50≈260 ms 不能替代用户尚未指定的 SLA。

### 2.2 补测两卡各运行一个 TP1 副本

这是当前报告之外最值得补的部署对照。按单卡结果简单相加，两个 TP1 的容量假设为 **14.288 rps**，约为 TP2/B1 的 1.80 倍，接近 TP2/B2 探索值 14.845 rps。单卡短输入约 140 ms 也提供较好的低延迟候选。

该数字为待实测估算。两实例需统一使用 4096、相同多模态配置、相同请求和相同总并发，分别绑定一张卡，再测入口分流后的总吞吐、p95/p99 与资源占用。双副本与 TP2 作为独立部署方案比较；两张卡的资源用于其中一种方案。

当前容器 CPU 配额为 32 核等价、内存 128 GiB，现场总内存占用约 77.5 GiB（包含内核计入的缓存等）。双实例的主机内存、加载峰值、CPU 配额与媒体预处理需重新验收。TP1 权重约 51.23 GiB/card，已验证长输入单请求；长输入大 batch 还需独立验证。短请求、高并发工作负载优先考察双 TP1；长请求、较严格单请求延迟优先考察 TP2。

## 3. 算子优化：先用固定版本已有入口

### 3.1 BF16 权重 NZ 格式 A/B

固定 vLLM-Ascend 已提供 **`weight_nz_mode=2`**：`ascend_config.py` 默认值为 1，`utils.py::_should_trans_nz` 对 BF16 的 NZ 转换要求 mode=2，`maybe_trans_nz` 执行转换，`ops/linear.py::process_weights_after_loading` 在加载后调用该路径。原生 MLP trace 为 ND/ND，与当前默认路径对应。

建议先合并到已有 `--additional-config` JSON 中测试 `{"weight_nz_mode":2}`，确认每个目标线性层实际选用的格式和 kernel。该配置涉及符合条件的线性层，收益、启动时间和内存变化均需记录。现场固定源码位置：

- `/workspace/yangchen/work/src/vllm-ascend/vllm_ascend/ascend_config.py`，约 680 行。
- `/workspace/yangchen/work/src/vllm-ascend/vllm_ascend/utils.py`，约 369–425 行。
- `/workspace/yangchen/work/src/vllm-ascend/vllm_ascend/ops/linear.py`，约 108–139 行。

A3 使用可配置权重布局；当前分支跳过 FP32、meta 和 N/K 任一为 1 的权重，目标 MLP 权重的 N/K 均为 128 的整数倍。转换在加载阶段执行一次并替换 `weight.data`，因此需记录加载时的瞬时内存峰值，并检查稳态是否新增格式回转或 `TransData`。

先测 TP2/307/B1 与已确认的 B2，再覆盖 TP1 和 1024/2048/3868 tokens。微复现要保存原位 tensor shape、stride、storage_offset、转置标志、dtype 与实际格式；现有 contiguous 微复现只复用了逻辑签名。重点形状包括：

| 配置 | MLP gate/up 输入、权重 | MLP down 输入、权重 |
| --- | --- | --- |
| TP1、307 tokens | `[307,5120] × [34816,5120]` | `[307,17408] × [5120,17408]` |
| TP2、307 tokens | `[307,5120] × [17408,5120]` | `[307,8704] × [5120,8704]` |

若两类 MLP kernel 总时间降低 20%，设备时间规模约为 TP1 12.49 ms、TP2 6.43 ms/request；在全部收益落入串行关键路径、其余时间固定的假设下，对应短输入吞吐约 +9.8%/+5.4%。**这是条件估算，NZ 的实际收益尚未测量。**

保留 BF16 基线，逐项校验 choice/noul/score、完整概率、候选映射与多模态；确认收益后，再评估定制 MatMul 或融合实现的投入。

### 3.2 GDN 布局转换、状态搬运与融合

固定 A3 FLA 路径 `vllm_ascend/device/device_op.py` 约 1005–1080 行包含 sequence-major contiguous、q/k L2Norm、BNSD transpose/contiguous，以及 state 前后转换。Top20 已表明 Transpose 的设备时间规模约 8 ms/request，适合先锁定其中的重复转换。

建议顺序：

1. 按一层 GDN 的 q/k/v、门控与 state 追踪 layout/stride，给每次转换标注消费方，计算各类转换实际独占区间。
2. 核对固定 FLA NPU A3 26.9.1 的输入约束，尝试在相邻算子之间保留可消费的布局，或融合归一化与布局变换。
3. 再考虑 GDN 主核的 tiling、状态布局与小 shape 调度。TP1/TP2 的头数和 batch 边界分别建模。

收益以消除的真实转换和调用开销为依据。若已列的 8 ms 转换可减半，设备时间规模约 4 ms；端到端收益需实测。改变 state 布局必须覆盖多请求、异构长度、batch 顺序及缓存块复用，避免状态串扰。GDN 的低 MAC ratio 与较高 cube_utilization 字段需要按导出语义分别解释，当前证据还不足以给出具体最优 tiling。

### 3.3 分段图与主机下发列为第二轮可行性实验

固定 vLLM 的 pooling 配置会将 FULL graph 调整为 PIECEWISE，Qwen3.5 主干具有 compile 支持入口，Ascend 平台具备分段图支持。现有启动入口固定 `--enforce-eager`，建议先增加明确的运行模式选项，在独立实验服务中验证分段编译，再比较数千 task/request 的下发成本。

必须覆盖动态 schema、GDN 状态、batch shape、多模态和重编译次数，并把启动/编译成本与稳态收益分开。首次只在主干可支持的范围试验，联合头的动态 Python/schema 逻辑单独检查。

当前 CPU binding 已开启，两个 worker 分别绑定 `2–317`、`322–637`；线程数约 398/397，容器共享 32 核时间配额。下一步采集负载窗口的 CPU 使用、cgroup throttling 差分、NUMA/设备亲和与 rank 提交间隔，再对线程配置和绑定范围做单变量 A/B。空闲现场的线程数和累计 throttling 无法独立证明 CPU 瓶颈。pooling 的 async scheduling 在固定源码中默认关闭，并有性能顾虑说明，保持为后续验证项。

## 4. 低成本应用优化与显存调整

**schema token 有界缓存。** 已有无观测 CPU 消融验证固定307节省 0.963 ms、八字段节省 4.625 ms。将缓存限定到 schema fragment 的 tokenize 调用点，键包含 tokenizer/revision、编码选项与完整片段内容；若缓存完整 schema，还需保留字段/候选顺序及所有影响 span 的模板信息。保存不可变 token tuple，命中时返回独立列表，按条目数和 token/字节预算控制容量。state 路径继续直接编码。当前实验按预知 rendered state 集合排除状态，生产实现应通过调用路径显式区分。校验 IDs、schema、orders、span 完全一致，再测 API 收益。

**KV/Mamba 池按负载缩小。** TP2 每 rank 真实 backing storage 为 28.453 GiB；若 4–8 GiB 池通过目标 workload 验证，潜在回收约 20.45–24.45 GiB/rank。先确定 batch 与 token 预算，再逐步调整缓存预算/内存利用率；覆盖 4096、最大目标 batch、冷媒体、池耗尽与请求取消后的回收。容量回收的时延收益另测，workspace 总峰值继续保持统计边界。

**长短请求与媒体分组测量。** 长短混合测试可检查队首阻塞和公平性；固定 AllPool 有 chunk hidden-state 缓存与 concat 路径，chunked prefill 可作为后续兼容性实验，并核查额外 clone/concat 成本。联合头会读取全 prompt 的归一化/投影 hidden states；启用跨请求 prefix caching 前，需要确认完整 hidden states 的恢复及 span 语义。热媒体缓存、冷媒体编码、实际尺寸与视频帧数分别报告。当前两张小图和两帧视频的证据用于该输入范围。

**联合头保持低优先级。** 可后续研究 schema 索引复用、逐 prompt pooler 合批及小算子融合。当前结果复制已使用异步 D2H 并等待 copy event，`tolist()` 处理 CPU 数据；收益判断沿用实测路径。量化涉及主干与联合决策精度及固定版本算子支持，可在上述 BF16 路线完成后另立验证分支。

## 5. 最小实验顺序与验收

| 顺序 | 实验 | 决策产物 |
| --- | --- | --- |
| 1 | TP2：seqs1/2 × C1/4；补 seqs4/C8，固定307 | 稳定吞吐、实际 batch、p95/p99 与低负载成本 |
| 2 | 双 TP1，统一4096/MM条件，总 C2/4；与相同总负载的 TP2 比较 | 两卡部署下的延迟、吞吐和主机/HBM资源效率 |
| 3 | 当前优胜配置，单变量切换 weight_nz_mode 1/2 | 同精度下的格式收益、kernel变化及启动/内存成本 |
| 4 | 独立 schema token LRU；GDN 转换微复现 | 分别给出 CPU 编码与 NPU 设备耗时收益 |
| 5 | 优胜配置覆盖256/1024/2048/3868、异构schema、冷媒体及长短混合 | 工作负载适用范围；必要时独立扩大 token 预算 |
| 6 | 缩小池、分段图/编译、布局融合 | 每项独立的性能、精度、资源与回退记录 |

每轮使用相同请求 SHA/顺序、相同预热与精度判据；A/B 交错执行以减弱时间漂移。稳定测量关闭 profiler 与逐调用观测，每组至少三轮、每轮至少 300 请求及 30 秒。报告 requests/s、input tokens/s、decisions/s、p50/p95/p99、样本量、实际 batch、CPU 配额差分、HBM 与错误率。

吞吐探索沿用现有闭环并发工具；面向 SLA 再补固定到达率测试，记录请求到达至响应的排队。当前 benchmark 的 `elapsed_ms` 排除了线程池中尚未发送的等待，客户端排队需要保持单独口径。正式精度沿用现有判据（离散结果与 token/span/映射一致，概率 0.03、score 0.05），并报告实际最大差及其样本；边界样本和混合 batch 分别比较。

## 6. 本轮现场复核

2026-11-13 11:15（北京时间）通过 `ssh clef` 确认 API PID `115206`、TP workers `115266/115267` 存续，固定 vLLM/vLLM-Ascend HEAD 与报告一致，两张 Ascend910_9382 健康。当前服务仍使用 TP2/4096、seqs1、eager、内存利用率0.9，监听 `127.0.0.1:8000`。

协议补齐项另行记录：当前 `/health` 返回 HTTP 200、空 body；`/v1/decide/info` 返回404；OpenAPI 中决策相关路由为 `/health` 与 `/v1/systemone`。项目目标要求的 18143 端口、ready/model/revision 响应和 info/decide 路由仍需在比赛入口阶段统一核验。这项现场结果限定了当前验收范围，本轮保持服务配置不变。[只读复核 JSON](evidence/tuning-review-20261113.json)
