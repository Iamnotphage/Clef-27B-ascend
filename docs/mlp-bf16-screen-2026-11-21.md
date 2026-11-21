# BF16 gate/up + SwiGLU 验证 · 2026-11-21

结论：首版大误差定位到 **GEMM→UB 的 ND 输出 stride**，小修复后的六个数值点全部通过局部筛查；当前融合原型的 M967 完整链耗时约为原 NZ 链的 **1.9–2.1 倍**，停止集成。该结果限定于本次实现，BF16 融合机制的最佳可达性能仍未确定。down 诊断同轮收口，保留 aa3e/S1 服务。

基线 `bcb98d9201a8f63e009349d1c2a96898d64064cd`。全程原权重 64 层 dense BF16、无量化、无答案缓存；仅卡7内逻辑 NPU0/1（Phy14/15）。40独占 SSH/设备/服务操作，41只做CPU诊断。无PR、无合并。

## 实现及首个错误阶段

[原型](../runtime/ascendc/clef_mlp_swiglu/op_kernel/clef_mlp_swiglu.cpp) 以独立 vendor 构建，保持部署目录和共享源码原状。一次准备把每rank `[17408,5120]` 的 gate/up 权重按128列配对、转 NZ；固定64×256输出tile，FP32累加结果先写为BF16，再执行FP32 `SiLU(gate)*up`、BF16紧凑输出 `[M,8704]`，最后使用原 NZ down。无服务安装器，探针必须显式 `--execute`，默认仅打印计划。

首版 M967 / layer31 / rank0：mid relRMSE **1.494102**、chain **1.446256**；有限值仍有大误差，立即停止，未计时。

CPU定位证据：

- torch_npu `_op_plugin_docs.py:4390` 与 vLLM `SiluAndMul.forward_native` 均定义 `SiLU(first_half)*second_half`。配对重排为双射，NZ公式 `((k/16)*1088+n/16)*256+(n%16)*16+k%16`；N tile偏移为 `n_start*16`。[CPU地址核对](../runtime/audit_mlp_swiglu_layout.py) 的3332点及两半搬运索引通过。
- CANN9.2.1 `copy_cube_out_fixpipe.h::GetOrgWidth` 在 `orgKc=0` 时使用 `orgN`；`CopyOutNZ2ND` 默认非连续写回。首版四参数 `SetOrgShape(M,17408,5120,5120)` 加默认 `GetTensorC(c)`，与局部C的256列stride冲突。64×256 buffer容量16384元素；旧stride末地址1096959。该错误发生在SwiGLU之前。
- 修复为每tile显式 `orgKc=256`、`GetTensorC(c,0,true)`，增加stage1/2直接输出gate/up。没有扫描tile。CANN `copy_cube_out_utils.h::SetCastMode` 为FP32→BF16；GetTensorC的跨核等待、UB向量barrier及输出队列依赖保留。

## 唯一一次定向复测

修复版先完成两rank×M967/307/1595、layer31的gate/up/mid/down逐阶段筛查及每点两次输出重复检查，再计时。输入来自11-15历史真实捕获；967为1595的显式前缀，307和1595使用完整捕获。数值全部有限、重复输出稳定；局部阈值relRMSE≤0.01且normalized max≤0.02。

六点中最大mid relRMSE **8.372e-5**、max abs **0.01171875**；最大chain relRMSE **3.013e-4**、max abs **0.00390625**。M1595两rank的gate/up/mid/chain均逐位一致。M967、307保留微小BF16差异，比赛四项2PP质量口径未在此局部筛查中验证。

M967完整 `gate GEMM + SwiGLU + down GEMM`：5轮同窗A/B交错；每pair/arm/clock先取慢rank，再取中位数。单位ms，含每次分配、ACLNN/torch下发及epilogue；NZ转换/重排只做一次并单列准备时间，down通信不在本探针内。

| 条件 | 原NZ device | 原型 device | 原NZ wall | 原型 wall |
| --- | ---: | ---: | ---: | ---: |
| warm | 0.938392 | 1.867960 | 0.957534 | 1.884510 |
| 256MiB读刷代理 | 0.961460 | 1.957560 | 1.041420 | 2.044460 |
| 真实0/31/63层权重streaming | 0.967969 | 1.851261 | 0.973645 | 1.857038 |

完整预热各臂/条件5轮单列；warm每轮5次，读刷每轮1次，streaming每轮5×3层、按单层均摊。streaming使用各层历史输入，数值筛查范围为layer31。两rank各条件5/5配对退化，幅度显著超过配对波动，停止307/1595性能护栏、服务集成、S1及OD/ERQA全量测试。当前同步Matmul client分块原型的退化已经证实；更优融合调度的收益保留为未验证。

## down诊断与恢复

41代码固定在 [`92fd985`](https://github.com/Iamnotphage/Clef-27B-ascend/commit/92fd985)，[复现说明](https://github.com/Iamnotphage/Clef-27B-ascend/blob/92fd985/runtime/down_gemm_probe.md)。40单次运行，180个timed样本逐位检查通过，6/6捕获SHA匹配，峰值活跃张量811.06MiB。

| M967条件 | 慢rank device ms | 慢rank wall ms | device A/A最大差 | wall A/A最大差 |
| --- | ---: | ---: | ---: | ---: |
| warm | 0.318730 | 0.389800 | 4.7746% | 6.0488% |
| 256MiB读刷代理 | 0.332570 | 0.406825 | 3.2055% | 1.3512% |
| 真实down权重ring代理 | 0.319740 | 0.401420 | 5.6101% | 4.0512% |

307/1595的ring device为0.129980/0.481680ms。device/wall分别取较慢rank中位数。旧967 CSV的rank均值差0.1034%，同stream SwiGLU→down间隙中位20ns；actual tiling、逐核时长、L1/L0流量及依赖等待仍缺证据。本轮停止专用BF16 down核，不再采集或扫tile。

两个窗口均fresh核进程树/18143/卡归属，绑定新身份后定向停服，退出即恢复原启动argv/env。最终 **2026-11-21 14:17:19（上海，远端记录）** 四检22项全部通过，文本/图像各一次200，实验进程退出；共享源码未写入。恢复启动期曾有一次health连接拒绝，发生在POST之前；ready后四检通过，原记录保留。

当前 `RUNNING_VERIFIED`：API **1641972 / start415870686**，boot `393ecc15-721f-46f7-b119-21fa327a2b3b`；TP worker **1642119/1642120**，卡7 Chip0/1。aa3e/S1/TP2/BF16/eager/4096/seq1/batched4096/util.9/default0、原processor/cacheoff；原env仍 `c18d38bd…c2ba9 / 4678B`。两端 `/tmp/clef27-service-current-state.json` 已更新。

## 复现与原数

先在独立工作目录放置 `runtime/ascendc/clef_mlp_swiglu/`、[探针](../runtime/probe_mlp_swiglu.py) 与 `probe_mlp_ffn.py`。加载现有 `scripts/activate_env.sh` 后构建，包留在独立build_out，完整vendor路径取 `build_out/_CPack_Packages/Linux/External/custom_opp_ubuntu_aarch64.run/packages/vendors/clef_mlp_swiglu`。设备窗口由唯一owner负责fresh定向停服及恢复，探针自行完成计算后退出。

```bash
python3 runtime/audit_mlp_swiglu_layout.py
MAX_JOBS=4 bash runtime/ascendc/clef_mlp_swiglu/build.sh
# 两rank分别运行（rank与device同为0或1），输出必须全新。
timeout --kill-after=5 90 python probe_mlp_swiglu.py --execute --phase numeric \
  --layers 31 --vendor "$task_vendor" --rank "$task_rank" --device "$task_rank" \
  --captures /workspace/yangchen/logs/mlp-20261115-1027/capture-requests.json \
  --output "$task_output"
# 双rank数值通过后才用 --phase timing；该阶段默认5配对并在M967无稳定收益时退出3。
```

[精简原数与哈希](../Clef-27B/logs/mlp-bf16-screen-20261121/summary.json) 包含逐阶段数值、配对数组、down慢rank/噪声、恢复身份和22项检查。大raw保留临时目录：本地 `clef40-mlp-nyklqirq`、`clef40-mlp-retest-wikqcpnu`；远端对应 `/workspace/yangchen/{work,logs}/` 同名目录。最终manifest SHA `91dcc6f9ae3486fc30ea07f7fed1e9fbc904d343bae77f684fe4b0cb4007cd50`。

## v3 CPU准备：先测GEMM本体与预算

本节仅CPU准备；本轮无SSH、SDK编译、NPU、停启、采集或服务源码修改。新增 [gate-only核](../runtime/ascendc/clef_mlp_gate_gemm/op_kernel/clef_mlp_gate_gemm.cpp) 与 [预算探针](../runtime/probe_mlp_gate_only.py)，默认打印计划；新BF16/NZ组合标为 **未编译、未测量**。down支线保持结束。

可用机制的证据层次：上一轮读取的实机CANN9.2.1 `matmul_client.h` 已有异步模板入口；本地固定v9.0.0官方 `matmul_perf` 有A3/DAV2201纯Cube、MDL/unitflag配置，`matmul_async_iterate_all` 明确支持A3。在线[GetTensorC文档](https://www.hiascend.com/doc_center/source/zh/CANNCommunityEdition/910beta2/API/ascendcopapi/atlasascendc_api_07_0639.html)确认A3/BF16/ND与NZ、异步GM缓存；[等待契约](https://asc.gitcode.com/api/SIMD-API/adv_api/cube_compute/Matmul_Kernel/WaitGetTensorC.html)要求Vector消费异步搬入的UB前等待。在线资料分别为9.1 beta及开发分支，当前9.2.1新组合的目标编译/运行仍缺证据。

最小顺序：

1. 原vendor `F.linear(x,W_NZ)` 与纯Cube `MultiCoreMatmulTiling + IterateAll` 分别输出相同canonical `[gate8704,up8704]`、BF16 `[M,17408]`。共享原NZ权重，取消配对重排；候选按完整单核任务提交，保留API内部base块和L1/L0调度，结束整个任务后才End。没有复制原vendor隐藏tiling，也没有预设更大tile。
2. 默认M967、两rank串行，先 `--phase numeric` 检查0/31/63层，再 `--phase timing`。计时分列原/候选gate、vendor gate A/A、独立vendor SwiGLU、两种完整链；布局、分配和下发均在相应区间内，NZ准备单列。warm、256MiB读刷代理、三层真实权重streaming分列；SwiGLU单测仅消费同组预激活，独立结果不等同服务开销。局部参考与候选的实际format/stride均记录。
3. 先检查GEMM差值是否落在A/A波动带，再用实测SwiGLU耗时作为保守预算；额外GM缓存/UB搬运/等待必须有剩余预算覆盖。差值连同波动已耗尽预算就停止。当前原数缺gate/SwiGLU独立计时，预算为UNKNOWN，尚未进入异步融合实现。

后续异步方案的边界：标准 `Iterate<false>` 一次启动整个singleCore任务，C由GM workspace保留，Vector分片取结果；能够减少逐base块client握手，原始C的GM写入/读取仍然存在。`SetWorkspace(GlobalTensor)` 在启动前设置独占区域，覆盖完整singleCore输出，持续到全部消费者排空及End。不能把两个小GM槽当成完整workspace。

| producer / region | consumer / region | 就绪与复用契约 |
| --- | --- | --- |
| Matmul → 独占GM C区域 | API GetTensorC返回的结果片 | API结果就绪后才读；保持C的BF16写回；完成整个任务后释放 |
| GM → `C_ub[slot]` | Vector读取相同slot | API搬入时使用WaitGetTensorC；自行搬入时使用TPipe的MTE2→V队列依赖 |
| Vector读取`C_ub[slot]` | 下一片搬入复用相同slot | 该片gate/up均已消费后释放；两槽按同一个`slot=i%2`管理，预热/尾部单独排空 |
| Vector → 单槽输出队列 | MTE3写最终紧凑结果 | V→MTE3队列就绪、搬出完成后才允许覆盖；不自行占用Matmul内部cross-core flag |

跨片Cube/Vector重叠依靠GM缓存解耦和两槽UB预取；最终仍会在消费尚未就绪的结果时等待。64×256的两槽BF16 C为64KiB；两个可复用FP32缓冲64KiB（SiLU后释放gate临时区用于up），单槽BF16输出16KiB，共144KiB，余48KiB用于API临时区/对齐；实际临时区和in-place限制待目标SDK确认。此UB片段大小不约束GEMM内部base tile，较大的结果片可继续细分给Vector。

具体未解决项：原canonical N自动分核可能将gate与up分到不同producer；须确认同client成对N所有权/就绪顺序，或先验证配对布局对GEMM和恢复canonical输出的完整代价。该映射与9.2.1异步组合未确认前，不写融合流水。官方[MatmulCallBackFunc](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/API/ascendcopapi/docs/en/api/SIMD-API/advanced_api/matrix_compute/Matmul-Kernel/MatmulCallBackFunc.md)属于自编译Matmul实例的搬运回调，不能据此给既有F.linear/vendor内核插入SiLU回调。

CPU验证：Python语法、三个M的默认计划及无输出写入检查、build.sh语法、JSON配置；同步与数据流analyzer扫描新kernel+tiling头均退出0、findings为空。SDK关联头未在本地展开，正则扫描不证明API内部同步或模板实例正确；目标编译、精度、性能和异步生命周期均保留待验证。CPU工作材料在临时目录 `clef40-v3-cpu-b1ywl5ca`。
