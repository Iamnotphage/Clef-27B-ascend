# BF16 gate/up + SwiGLU 验证 · 2026-11-21

最新执行状态：同一入口/依赖字节核同及有效0路径已贯通；vendor唯一BasicInfo尝试由工具明确返回空结果，尚无内部数据。累计133.140s，余106.860s；aa3e/S1恢复22/22通过，详见末节。

02:37阶段采集链修正（历史）：已确认本机msopprof26.2的native ID0可有效，撤回零值失活判断，改验SDK注册/回调及BasicInfo导出。本轮BasicInfo被旧guard在提交前截断，内部原因保持UNKNOWN；累计111.282s，aa3e/S1恢复22/22通过，详见末节。

已取得的task级profile结论：M967三层streaming中，原vendor/v3的kernel task约 **0.592→0.905ms/层**，task增量约覆盖本窗profile配对差距的 **81.48%**。host桥接另有开销且与kernel重叠；当前v3 kernel惩罚已超过可省SwiGLU预算，继续停止该实现的融合路线。原aa3e/S1恢复22/22通过，详见末节。

v3初次实测结论：同步纯Cube gate-only控制完成SDK编译及18组局部筛查；M967三层streaming的gate调用区间从 **0.600360→0.959673ms**，配对增量 **0.359000ms**，A/A最大差 **0.005813ms**，独立SwiGLU仅 **0.055533ms**。当前调用路径预算不足，停止异步融合实现。原aa3e/S1已恢复，最终四检22项通过；本轮没有S1或OD/ERQA候选结论。

此前v1/v2结论：首版大误差定位到 **GEMM→UB 的 ND 输出 stride**，小修复后的六个数值点全部通过局部筛查；当前融合原型的 M967 完整链耗时约为原 NZ 链的 **1.9–2.1 倍**，停止集成。该结果限定于本次实现，BF16 融合机制的最佳可达性能仍未确定。down 诊断同轮收口，保留 aa3e/S1 服务。

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

v2恢复历史记录 `RUNNING_VERIFIED`：API **1641972 / start415870686**，boot `393ecc15-721f-46f7-b119-21fa327a2b3b`；TP worker **1642119/1642120**，卡7 Chip0/1。aa3e/S1/TP2/BF16/eager/4096/seq1/batched4096/util.9/default0、原processor/cacheoff；原env仍 `c18d38bd…c2ba9 / 4678B`。两端 `/tmp/clef27-service-current-state.json` 已更新。

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

以下为3816d8c阶段状态，后续SDK与设备验证见下一节。

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

## v3实测与收口 · 2026-11-21 16:27（上海）

`3816d8c` 的独立vendor首轮SDK编译通过。第一次M967调用在宿主 `GetWorkspaceSize` 的tiling阶段返回561002，尚无候选数值样本，随即恢复aa3e并完成22项检查。恢复后使用**显式命名平台** `PlatformAscendCManager::GetInstance("Ascend910_9382")` 单点CPU诊断：API返回0，但 `singleCoreN=5803`，触发了本地NZ起点16对齐检查。

唯一修复为向上对齐singleCoreN，保持baseM/N/K=128/256/64及base块数量：M967为5803→5808，M307/1595为726→736；CPU列覆盖均为17408。该修正保障 `B_NZ + n_start*16` 的物理块边界，无tile或padding扫描。修正版SDK再次编译通过，fresh绑定新服务身份后做一次定向复测。

两rank×M967/307/1595×layer0/31/63共18组：canonical `[M,17408]`，参考与候选均ND/BF16，所有gate/up/chain有限、局部容差通过，每组两次重复逐位稳定；6/18 gate结果与vendor逐位一致。最大gate relRMSE=3.7365e-5、max abs=0.0625；最大chain relRMSE=3.0133e-4、max abs=1.0（各组normalized max均≤0.02，详见summary）。这些是局部筛查，比赛四项2PP未在当前控制核上评测。

M967同窗5配对，各pair/arm/clock取两rank较慢值，再求中位数。单位ms；gate包含输出分配、ACLNN/torch下发和canonical输出GM写入，原NZ权重共享且一次准备，无新增输出去重排。三层streaming按单层均摊，down维持原实现。完整预热每arm/条件5轮单列。

| 条件 | vendor gate device | v3 gate device | SwiGLU device | 原链 device | v3 gate链 device |
| --- | ---: | ---: | ---: | ---: | ---: |
| warm | 0.599700 | 1.015020 | 0.062840 | 0.957180 | 1.394160 |
| 256MiB读刷代理 | 0.613160 | 1.080800 | 0.071800 | 0.961880 | 1.414840 |
| 三层权重streaming | 0.600360 | 0.959673 | 0.055533 | 0.946860 | 1.299660 |

| 条件 | vendor gate wall | v3 gate wall | SwiGLU wall | 原链 wall | v3 gate链 wall |
| --- | ---: | ---: | ---: | ---: | ---: |
| warm | 0.686220 | 1.143360 | 0.156820 | 1.042540 | 1.487040 |
| 256MiB读刷代理 | 0.697120 | 1.168380 | 0.160320 | 1.043960 | 1.498320 |
| 三层权重streaming | 0.631230 | 0.988443 | 0.083570 | 0.979683 | 1.329667 |

三个条件gate调用耗时配对中位数分别为vendor的约1.70/1.76/1.60倍，device每rank每条件5/5退化；wall的cold/streaming均5/5，warm rank0为4/5、rank1为5/5（rank0首对方向相反，原数保留）。device配对增量中位数0.418280/0.467640/0.359000，A/A最大绝对差0.013840/0.016900/0.005813。streaming wall配对增量0.357647，A/A最大差0.033637；warm wall保留一次A/A尖峰0.963940ms（中位差0.012150），该条件wall最大噪声不支持单独判定，其他条件证据支持收口，无补跑。

以 `median(SwiGLU) - max(0,median(v3_gate-vendor_gate)) - max_abs(A/A)` 作为保守预算，device为 **−0.369280/−0.412740/−0.309280ms**；streaming wall为−0.307713ms。即使给出全部独立SwiGLU时间作为可省上限，当前额外调用成本仍无法覆盖。SwiGLU分项消费同组预激活，缓存条件与链内存在差异；分项不能用于直接拆出down或kernel细分ms。

NPU Event包围的是完整调用区间，包含host下发间隙，尚无独立kernel trace或host tiling占比。本结论限定于当前pure-Cube API调用实现。标准异步的GM workspace成本没有扣除，也没有据此否决其他BF16融合实现。本窗口不写或运行异步融合，不进行S1/OD/ERQA全量，不重开down。

两个服务停启段分别为宿主检查失败后的立即恢复、单一对齐修正后的复测恢复；累计探针耗时 **48.4s**，在本轮累计240s预算内。每段均fresh核对指针、完整身份、18143 inode及卡7/Chip0/1/Phy14/15，保存当时manifest/argv/env。最终 **2026-11-21 16:27:38（上海）** `RUNNING_VERIFIED`，22/22通过，API **1652694 / start416641174 / boot393ecc15-721f-46f7-b119-21fa327a2b3b**，TP **1652851/1652852**仅卡7双芯片；原argv/env匹配，env仍c18d38bd…c2ba9/4678B，两端current-state指针已更新，实验进程退出。manifest SHA `e2bcba916a1c62e21e91e73d071bb27962d1eebc79c9ab8f818cffa450eaf2b9`。

源码与分项数值仍见本报告的探针链接；CPU tiling诊断使用同参数直接调用SDK，命名平台避免初始化NPU。新原数/哈希加入原summary的 `v3_device_validation`，大raw在本地 `clef40-v3-run-2tt4rzwb`、`clef40-v3-aligned-3e_elijo` 及远端同名work/logs目录，无新artifact或PR。

## v3最小关联profile · 2026-11-22 01:19（上海恢复）

固定 `4a51f71` 的v3 gate-only与原vendor NZ，kernel/tiling未改。M967、BF16、canonical ND `[967,17408]`、同一原NZ权重；layer31/0/63历史真实输入均取1595的显式967前缀，每组三层按单层均摊，输出分配/布局/下发保留在区间内。down权重保持resident，无down计算或诊断。

源码循环和原JSON的 `ranks_serial:[0,1]` 均确认原测量为rank0→rank1串行。本窗两子进程完成准备后先并发控制，再rank0串行pre/profile/post、rank1串行pre/profile/post，最后并发控制；另一rank串行期间sleep等待。每段3配对，非profile包含vendor A/A与vendor/v3 A/B，交错顺序；各臂完整预热5×3层单列。另在样本外初始化Event，记录与原探针的协议差异。记录各arm进程CPU时间、上下文切换及affinity，没有设置新绑核策略。

采用 [最小采集脚本](../runtime/profile_mlp_gate.py)：torch_npu CPU+NPU Level1、AiCoreNone，关闭stack/shape/memory/硬件计数，仅粗粒度range，三层后一次同步。只在探针进程中包裹已预热的两个ctypes符号，标记GetWorkspace与launch，原参数/数学实现不变；共享runtime文件未写入。停服前将既有raw复制到私有tmp，由当前导出器成功生成trace。新raw落盘后先恢复aa3e、通过四检，再CPU导出。

以下单位ms/层，每pair/arm/clock独立取较慢rank后求中位，全部原数保留：

| 调度/段 | vendor Event | v3 Event | 配对增量中位 | A/A最大绝对差 |
| --- | ---: | ---: | ---: | ---: |
| 原4a51f71串行 | 0.600360 | 0.959673 | 0.359000 | 0.005813 |
| 本窗串行pre | 0.601107 | 0.969880 | 0.369353 | 0.018147 |
| 本窗串行post | 0.602367 | 0.976287 | 0.374227 | 0.035820 |
| 并发pre | 0.606660 | 0.978680 | 0.370113 | 0.003007 |
| 并发post | 0.604693 | 0.969040 | 0.364847 | 0.012793 |
| 串行profile | 0.643493 | 1.027493 | 0.383087 | 未采profile A/A |

串行pre/post的wall配对增量为0.362293/0.376777ms，A/A最大0.066910/0.123826；并发pre/post为0.365423/0.365760，A/A最大0.020207/0.053454。并发pre的第二对Event增量0.903220、串行post首对0.982467、并发post末对wall增量1.729446等尖峰均保留。串行pre进程总CPU（跨线程、含runtime spin，允许大于wall）vendor/v3中位0.931333/1.397333ms；并发pre为0.894333/1.427000ms，逐arm负载与切换数见summary。

profile两臂Event中位较串行pre增加约7.1%/5.9%，配对增量增加约3.7%；相对post增量增加约2.4%。首profile样本的范围进入开销较大，raw未裁剪。差距在原调度及并发控制均复现，profile支持task级结构归因；这些短样本不用于正式P95，也不把profile比例精确套到历史0.359ms。

[离线解析器](../runtime/analyze_mlp_gate_profile.py) 使用原生flow、精确stream目标和enqueue→dequeue关联，36/36个GEMM任务、24个Event标记均关联成功。两臂各层均一个AI_CORE任务，无额外cast/layout/copy任务；控制Event单列。先用Decimal移除epoch，再计算相对时间，避免浮点吞掉20ns相邻任务间隙。trace Event跨度与独立Event原数最大差0.0677µs/层，时基核对通过。

| profile分项（ms/层） | vendor | v3 |
| --- | ---: | ---: |
| host submit inclusive / self | 0.089350 / 0.031890 | 0.422793 / 0.021420 |
| 非同步native API区间union | 0.018937 | 0.051810 |
| kernel task执行union | 0.592382 | 0.905494 |
| 显式末端sync inclusive | 0.547903 | 0.656273 |
| 最后kernel结束→sync返回 | 0.018972 | 0.018878 |
| Event开始→首kernel（绑定Event慢rank） | 0.051130 | 0.124976 |
| 上项中host范围覆盖 / 未覆盖 | 0.048750 / 0.003160 | 0.121706 / 0.002720 |

表中host、API、kernel与sync会重叠，独立中位数也不具可加性。submit self扣除同主线程嵌套范围；v3 bridge inclusive/self为0.312790/0.181340ms，GetWorkspace 0.059367、launch 0.065343、输出分配0.042597，均为含采集标记的elapsed范围。GetWorkspace内的tiling/executor细分、bridge self内的descriptor/stream查询/销毁和标记开销占比仍为UNKNOWN。

同一Event慢rank绑定后的三对增量：总Event **0.335353/0.383069/0.394242ms**；kernel union **0.312132/0.309223/0.314949ms**；首kernel前空档 **0.023215/0.073846/0.079286ms**。同一样本的这些区间互斥，余下仅约0.000007ms级相邻kernel/末Event间隙。kernel增量中位占总Event增量中位 **81.48%**，约0.071ms剩余量主要落在首kernel前的host下发范围；范围覆盖只证明时间落点，不能唯一拆成CPU指令或可省成本。首profile vendor样本还有0.062862ms首段未覆盖时间，原样保留。

native launch返回到首kernel开始为−6.2～−2.1µs，设备可在CPU调用返回前启动。第二/三个launch到start为vendor约0.536～1.141ms、v3约0.470～1.177ms，与前序kernel占用stream重叠；两臂相邻kernel间隙仅约0.000013～0.000027ms/层。显式sync与kernel大量重叠，末kernel后返回约0.019ms，两臂接近。禁止把这些排队/同步elapsed与kernel相加，也不能把launch到start直接视为可省设备空闲。

**收口依据：** v3 kernel task本体约慢53%，约0.313ms增量已经超过此前同窗SwiGLU 0.055533ms预算约5.6倍。host桥接可减少的部分仍有研究空间，当前kernel惩罚本身已耗尽融合预算。本轮停止该v3融合路线，不实现新核或扫描tile。task时长含内部stall；缺内部pipeline/per-core与实际vendor tiling证据，HBM、Cube调度、分块/权重复用各自占比仍为UNKNOWN。结论限定当前v3实现。

本轮单次计算/采集 **10.512s**，低于240s上限，双rank三层数值有限、gate/up顺序与容差、重复输出检查均通过。数值/采集/四检未发生失败或POST重试；启动ready轮询在POST前执行，轮询次数未留计数。fresh绑定旧树后定向停服；探针退出即恢复。最终 **2026-11-22 01:19:09（上海）RUNNING_VERIFIED**，22/22检查通过：API **1681267 / start419844331 / boot393ecc15-721f-46f7-b119-21fa327a2b3b**，TP **1681440/1681441**仅卡7 Chip0/1/Phy14/15。原argv/env、aa3e源码、OPP9/LD12匹配，env仍c18d38bd…c2ba9/4678B；文本/图像各一次200，无实验进程残留，两端current-state指针更新。manifest SHA `d436d67db9a752e75760e8c0f2de175a92830c187735d6258cd1b93e8467b597`。

复现入口：`python3 runtime/profile_mlp_gate.py` 默认仅打印计划；设备owner另建fresh恢复参考后，在总超时保护内同时启动rank0/1，传入同一全新 `--barrier`、各自全新 `--output` 及固定 `--vendor`。本次实际执行父脚本为远端私有work目录的 `window.py`，包含fresh核/停服、220s子进程总限和finally恢复；历史恢复参考只用于追溯。恢复完成后 `torch_npu.profiler.profiler.analyse(..., max_process_number=2)`，再运行解析器：

```bash
python3 runtime/analyze_mlp_gate_profile.py \
  --trace "$task_raw/trace-r0.json" --trace "$task_raw/trace-r1.json" \
  --samples "$task_raw/rank0.json" --samples "$task_raw/rank1.json" \
  --output "$task_raw/analysis.json"
```

本地raw `clef40-gate-profile-41a26faq`（系统tmp），远端同名`work/logs`目录；原生PROF/FRAMEWORK/trace/API CSV留远端，完整路径、SHA、配对原数和恢复证据写回原summary的 `v3_profile_attribution`。CPU验证包含默认plan/语法、合成flow/区间union、全部36任务显式关联、24Event标记、互斥分区闭合及时基核对；无新artifact、PR或全量质量性能测试。

## 内部PMU补采收口 · 2026-11-22 02:13（上海恢复）

本轮 **INCONCLUSIVE**：请求的内部tiling/逐核/Cube-MTE对照未取得可用数据，当前无法对约0.313ms设备task增量做内部原因排序。保持kernel/tiling固定，不据此调参或否决其他BF16融合实现。

先复用e05ec7a数据，核对结果如下：

| 项目 | 已有/新增CPU证据 | 当前缺口 |
| --- | --- | --- |
| task与关联 | 36/36任务、每层一个GEMM、原时基核对，复用 | 无需重采host关联 |
| 请求blockDim | 两rank旧kernel_details中，两臂均24 | 实际active核数与每核工作分布待测 |
| v3 tiling | SDK静态复算全50字段，结构200+12字节 | 本次runtime TilingData未落盘，核同未完成 |
| vendor tiling | 请求blockDim24 | base/singleCore/L1/L0/step/DB及schema均UNKNOWN |
| 核间负载 | 旧aggregate无法提供逐核数据 | min/median/max、长尾与工作分布UNKNOWN |
| 流水与访存 | 工具help列出所需计数组及逐核接口 | 本次无可用PMU/cycle/访存/等待导出，UNKNOWN |

v3静态复算：singleCoreM/N/K=122/5808/5120，baseM/N/K=128/256/64，depthA1/B1=4/8，stepM/N=1/1，stepKa/Kb=2/4，dbL0A/B/C=2/2/1，shareL1/L0C=327680/131072字节，usedCoreNum24。源代码推算mTasks/nTasks=8/3，M最后113行、N最后5792列。以上仅为**命名平台的CPU复算与源码工作模型**，未作为本次runtime实际tile或性能因果证据；未用相似算子、默认值猜测vendor字段。

本机CANN9.2.1提供msopprof26.2.0，`msprof op --help`支持application replay、MSTX筛选和PipeUtilization/Memory/MemoryL0/ResourceConflictRatio；二进制具有tiling dump接口。选择rank0/layer31，其旧task中位vendor/v3为0.593396/0.902014ms，差0.308619ms，接近三层均摊的0.313ms。新 [补采探针](../runtime/probe_mlp_gate_internal.py) 固定同一BF16/NZ/ND和1595→967历史输入前缀，保留layer31/0/63真实权重ring、输出分配和下发；各臂显式预热15调用、collector warm-up=0，目标仅layer31的MSTX范围。未改部署/原厂库或共享GEV；原vendor opapi库SHA与上一轮相同。

前置rank0三对无prof控制：Event vendor/v3中位 **0.596093/0.970027ms/层**，配对增量 **0.370520ms**，A/A最大 **0.011340ms**；逐对增量0.370520/0.362213/0.373933ms。数值有限、gate/up顺序与容差、重复输出检查均通过。精确wall、预热和输入/权重身份保留原summary。该控制支持差距复现。

原厂臂在85秒子限触发timeout，父wrapper停止了fresh记录的实验身份并恢复服务；v3采集及post控制未启动。工具内部完成7次application replay，每次应用数值通过、Torch MSTX `range_start`均返回 **0**；无可用OPPROF/CSV/tiling bin落盘。当时按SDK `tools/mstx/include/mstx/ms_tools_ext.h:65`注释将0判为失活；后续02:37复核证明本机26.2 native注入可返回合法首ID0，该单值失活诊断撤回。原Torch桥接/范围筛选的具体失败原因仍UNKNOWN；CLI支持列表不能证明实际路径成功。本轮没有可用raw可离线恢复PMU，未执行人工设备重试。

本次探针在range返回0后仍提交了目标，缺少有效性检查，形成无效回放直到超时。d12a4c8阶段曾加range ID0提前报错规则；后续已确认该规则对本机native注入不成立并撤回，改验SDK注册/回调及BasicInfo导出。该阶段CPU测试遗漏“已注册且合法0”，已在02:37阶段补齐。原始执行版本及保护版SHA分别记录。 [CPU解析器](../runtime/analyze_mlp_gate_internal.py) 已准备逐核原数/单位/计数组清单与严格v3结构解码，vendor无对应schema即保留UNKNOWN；已用合成CSV/212字节结构验证，不宣称已在实际PMU导出上验收。

收口依据：目前只证实两臂请求blockDim相同，无法据此推断实际active核数或排除核间不均；tiling差异、Cube/MTE等待、写出瓶颈对0.313ms各占多少均无证据。流水占比不能互补求可省时间，task duration也不能拆成纯乘加与HBM等待。下一步最小方向为修复采集范围的使能/桥接，先以一次BasicInfo确认命中目标，再规划所需PMU；当前没有证据支持调整kernel、tile或buffer参数。本轮不继续占卡。

本轮设备计算/采集累计 **93.274s**（上限240s），collector timeout1次、内部application replay7次、人工设备重试0次；CPU辅助链接首次缺libtiling_api.a，补齐后静态复算成功。恢复优先：最终 **2026-11-22 02:13:04（上海）RUNNING_VERIFIED**，22/22检查通过，文本/图像各一次200、无POST重试；API **1687258 / start420167800 / boot393ecc15-721f-46f7-b119-21fa327a2b3b**，TP **1687428/1687429**仅卡7双芯片。原argv/env、aa3e、OPP9/LD12及卡归属匹配，env仍c18d38bd…c2ba9/4678B，两端指针更新，无实验进程残留。manifest SHA `fee04f48127e12fe46d8665acbc8acf8b036a99e5fde136e3afebf5984f1ffec`。

默认 `python3 runtime/probe_mlp_gate_internal.py` 仅打印计划。实际命令、七次应用记录、超时/清理、静态SDK字段/offset、已执行及未执行范围、恢复证据写回原summary的 `v3_internal_probe`。raw留系统tmp `clef40-gate-internal-api39v66`，远端同名work/logs目录；无新artifact、PR、质量全量或down实验。

## MSTX零值语义修正与BasicInfo退出 · 2026-11-22 02:37

**确定修正：本机msopprof26.2的native range ID 0可为有效首个range，按ID是否为0判断失活的规则已撤回。** 本轮BasicInfo在目标提交前被旧guard截断，未获得内部数据；0.313ms内部原因继续UNKNOWN，未形成kernel/tiling空间结论。

SDK `ms_tools_ext.h:65`注释写明0表示失败，SDK无注入时也返回0；这两点不能覆盖工具注入后的实际ID规则。本机固定 `libmsopprof_injection.so` 复核：`MsTx`构造函数0x1f7958及reset 0x1f75cc将对象+272计数器置0；`MstxRangeAdd`先用当前计数器插入range map，0x1f7068～0x1f7070读取旧值并递增保存，0x1f70a0返回旧值；`MstxRangeEnd`按同一个uint64 key查询map，支持key0。SDK文字约定与26.2注入实现存在差异。此前将Torch返回0直接归为“未激活”的诊断也撤回，旧七次未导出的具体原因仍未定位。

[host桥接](../runtime/mstx_gate_range.cpp) 只调用SDK `mstxRangeStartA/End`。SDK按 `MSTX_INJECTION_PATH` 加载库，或在全局符号中查找 `InitInjectionMstx`，通过函数表绑定回调；本机工具导出该入口，返回协议与SDK一致。uint64 ID完整保留。Torch `MstxMgr`具体使能路径仍UNKNOWN，native桥接避免依赖其开关。将range结束移到原有三层ring末同步之后，覆盖vendor异步工作线程下发，未增加逐op同步或改变计算。

最初CPU测试只覆盖“未注册0/已注册非零”，遗漏“已注册且合法0”，导致本轮canary被错误guard提前结束。CPU修正后，桥接额外提供本地SDK初始化完成且start/end回调已绑定的状态；探针与owner wrapper检查该状态，接受已注册的0。该状态仅证明SDK接入，**最终命中必须由BasicInfo导出验证**。已补三分支CPU注入测试：无注册ID0/ready=false；注册回调返回0且end(0)确实调用；注册回调返回4294967297且64位完整。导入Torch后的mock正例同样通过，未复现全局符号抢占。修正后未再上卡。

[命中检查](../runtime/analyze_mlp_gate_internal.py) 要求每臂一条BasicInfo任务，核名匹配vendor MatMul/v3 ClefMlpGateGemm、BlockDim24；结合该native范围首launch的实时shape/dtype/格式、prefix SHA和原权重rank SHA建立映射。BasicInfo不直接导出的shape/dtype明确来自应用元数据，不冒充工具字段。合成已注册ID0及未注册负例通过CPU检查，真实硬件命中尚未完成。

本轮单次窗口执行：

| 阶段 | 实际结果 |
| --- | --- |
| rank0三层无prof AA/AB | Event vendor/v3中位0.594140/0.966480ms；配对增量0.371480，A/A最大0.017833ms；wall增量0.361434ms |
| 数值/稳定 | pre与应用的3层gate/up容差、有限值、重复检查通过，原BF16/NZ/ND不变 |
| vendor BasicInfo | native ID0，旧guard在目标提交前退出；0个命中、无CSV/tiling dump |
| v3 BasicInfo / Pipe / post | 未启动，未追加设备尝试 |

BasicInfo工具上限24s、后续只允许双命中后单组Pipe，未再跑四组指标。此次实际计算/采集 **18.008s**，与前轮93.274s累计 **111.282s / 240s**，剩余 **128.718s**；不将未命中算为已采。单次验证退出要求已执行。具体SDK实际注册状态在当时未记录，故实际那个0仍不能区分“合法首ID”与“未接入”，修正后的检查待后续真实命中验证。

清理后首次恢复前空闲检查遇到驱动上下文释放延迟；fresh读取显示卡上无进程，无需额外信号，随后恢复。最终 **2026-11-22 02:37:20（上海）RUNNING_VERIFIED**，22/22通过，原argv/env、aa3e、OPP9/LD12和卡归属匹配，文本/图像各一次200，无POST重试。API **1690118 / start420313325 / boot393ecc15-721f-46f7-b119-21fa327a2b3b**，TP **1690285/1690286**仅卡7双芯片；两端指针更新。manifest SHA `571fd2a18ebb3e0aa6ee411243e8d5ecf749dd429ba1c853b14f486086613359`。

代码/复现与判断写回原summary `v3_range_repair`，包含已执行版本与CPU修正版SHA、二进制控制流证据及具体未执行阶段。raw在系统tmp `clef40-gate-hit-54tjmh8q`、远端同名work/logs。独立host桥接构建：

```bash
g++ -std=c++17 -shared -fPIC -fvisibility=hidden \
  -I/usr/local/Ascend/cann-9.2.1/tools/mstx/include \
  runtime/mstx_gate_range.cpp -ldl -o "$task_dir/mstx_gate_range.so"
```

下一步先完成“已注册回调、接受native ID0、BasicInfo实际一任务命中”的验证，再决定补哪一组PMU。当前缺少实际tiling、逐核及Cube/MTE证据，继续保留UNKNOWN，不调整计算kernel/tiling，不据未命中否定融合机制。无PR、新artifact或全量测试。

## 统一执行入口后的真实BasicInfo尝试 · 2026-11-22 03:01

本轮入口一致性与合法0路径已实测通过，**BasicInfo仍未命中**。工具正常返回0，但日志明确报告 `Profiling results are empty. No kernel profiling data was generated.` 和 `Profiling data parse failed.`；输出目录为空，排除单纯CSV命名差异。0.313ms内部原因继续UNKNOWN，未据此判断kernel/tiling空间。

新增 [统一owner入口](../runtime/run_mlp_gate_window.py)，直接导入探针的同一 `capture_failure`；[探针](../runtime/probe_mlp_gate_internal.py) 的CPU `--marker-self-test` 与真实capture复用同一 `NativeRange` 类。所有本轮执行路径均接受已注册的native ID0，并正确调用end(0)；缺少注册观测时保留UNKNOWN，仅明确ready=false或应用失败才提前退出。历史临时window.py保留为执行证据，本轮不使用。

三类CPU用例跑在本轮同一个暂存脚本/桥接：无注册0→ready=false、exit2；注册有效0→ready=true、exit0且mock end(0)计数1；4294967297→64位完整传递。owner入口的实际import路径也检查合法0、UNKNOWN、显式false。实际venv解释器对同一入口补核同样通过。停服前只读核对9项源文件/依赖及桥接二进制SHA，保存精确argv；执行后SHA仍一致，无旧guard/字节版本错配。未增加海量pins，未改计算kernel、tiling或输入语义。

| 阶段 | 结果 |
| --- | --- |
| 三层streaming前控，rank0/3配对 | Event vendor/v3中位0.600947/0.963213ms，配对增量0.363427ms；A/A最大0.007327ms |
| wall前控 | 配对增量0.362257ms，A/A最大0.011060ms |
| vendor应用 | 数值/重复通过，native ID0、ready=true、end_called=true；原BF16/NZ→ND，实时shape/dtype/prefix及权重SHA已记录 |
| vendor BasicInfo | 唯一一次，13.868s，工具返回0但明确空结果/parse失败；实际导出任务0、无tiling或计数raw |
| v3 BasicInfo / Pipe / post | 全部未启动，miss后自动清理恢复，无重试 |

注册和end(0)字段属于SDK辅助观测，不等同工具目标命中。应用元数据为输入 `[967,5120]` BF16、权重 `[17408,5120]` BF16/NZ29、输出 `[967,17408]` BF16/ND2；它们不冒充未取得的工具task字段。现在可排除本轮入口字节错误和数值零值guard拒绝；注册之后的目标捕获/范围筛选/导出环节仍未贯通，具体原因未归因。本轮不扩native调查，不在窗口内换桥接或反复采集。

实验硬限112s，为清理保留余量；实际 **21.858s**，累计 **133.140/240s**，剩余 **106.860s**。owner入口对进程退出后驱动释放做有界fresh空闲观察，未向历史PID或其他模型发信号，恢复无需人工补启动。最终 **2026-11-22 03:01:08（上海）RUNNING_VERIFIED**，22/22检查通过；API **1692910 / start420456310 / boot393ecc15-721f-46f7-b119-21fa327a2b3b**，TP **1693104/1693105**仅卡7双芯片。原argv/env、aa3e、OPP9/LD12、设备归属匹配；文本/图像各一次200，无POST重试，两端指针更新。manifest SHA `11c894f5e8fcf3f642bdeb1219a13ae1ee2828f500494426026e9e465100df19`。

精确执行argv、入口/依赖SHA、执行后核同、三类CPU结果、工具原始错误、前控原数与恢复证据写回原summary `v3_exact_entry_attempt`。raw为系统tmp `clef40-gate-exact-ypdi4odk` 及远端同名work/logs，无新artifact/PR。默认owner入口打印计划，实际执行必须显式 `--execute` 并传入fresh私有work/output、固定vendor、已用/剩余预算；真实BasicInfo任务导出仍为后续采集前提。

## 复用成功collector的单组Pipe补采 · 2026-11-22 03:23（上海恢复）

**18/18目标task与PMU已命中；配对task增量中位0.308772ms，复现此前约0.313ms。** 本次取得MTE2与Fixpipe活动显著增加的证据，内部等待的独占归因、逐核负载及两臂runtime tiling继续UNKNOWN。

CPU复核将msopprof空结果的断点限定在“SDK注册/range结束成功之后，目标捕获/筛选/导出尚未贯通”：应用数值完成且end(0)已调用，工具仍报 `Profiling results are empty`、`Profiling data parse failed`，无目标raw。工具injection库确实导出 `aclrtLaunchKernelWithHostArgsImpl` 等hook；e05 trace中vendor在worker线程launch、v3在main线程launch。现有证据未排他区分hook捕获与range/线程筛选，保留UNKNOWN，不声称ACLNN NZ不受支持，不再重试该空结果路径。

e05成功的torch_npu Level1 raw显式配置 `ACL_AICORE_NONE`，DB只有TASK/COMPUTE_TASK_INFO/AICORE_FREQ等表，无法离线还原未采的PMU/逐核计数。本机torch_npu 2.10.0.post7的ExperimentalConfig允许Level1开启aic_metrics，CANN9.2.1 `acl_prof.h`提供 `ACL_AICORE_PIPE_UTILIZATION`。因此仅在同一成功入口增加 `--solo --aic-metrics pipe`，保留CPU+NPU Level1、关闭stack/shape/memory，原生raw先落盘，恢复后离线导出；无MSTX/preload、无application replay、未改kernel或tiling。

代表rank0，M967使用历史1595输入的显式前缀，层顺序31/0/63与原三层streaming、输出分配/下发/生命期一致；原NZ一次准备，BF16 canonical ND `[967,17408]`，完整预热每臂15次独立记录。三层all/gate/up相对RMSE最大3.494e-5、normalized max最大0.002794，有限值与重复逐位检查通过；prefix与原权重SHA留summary。9项暂存入口/依赖/固定库SHA停服前核同，运行后5项入口/依赖核同。本次仅rank0短采，双rank并发与串行差距代表性沿用e05对照，未把本次数据称为新双rank慢者。

| 段（ms/层，三层均摊） | vendor Event | v3 Event | 配对Event增量中位 | A/A最大绝对差 |
| --- | ---: | ---: | ---: | ---: |
| serial pre | 0.598740 | 0.954827 | 0.356067 | 0.010333 |
| serial post | 0.602793 | 0.965300 | 0.352020 | 0.044187 |
| profile | 0.651960 | 1.056973 | 0.362840 | 未采profile A/A |

pre/post wall配对增量0.351227/0.354630ms，A/A最大0.014623/0.124170ms；profile wall增量0.355797ms。profile Event增量较pre/post约+1.9%/+3.1%；各臂绝对Event中位也受标记扰动。尖峰全部保留，短样本不作正式P95。三对task增量分别307.3718/308.7718/309.0518µs，约0.309ms的task惩罚稳定复现。

task关联沿用原生flow/correlation，再按精确stream/task ID、核名及绝对时基核同CSV；9个vendor `aclnnMatmulWeightNz_MatMulCommon_MatMulV2`、9个v3 `ClefMlpGateGemm`，每层各一个AI_CORE任务，无新增cast/copy/layout compute task。两臂Block Num24、BF16、ND+FRACTAL_NZ输入、ND输出；vendor权重shape导出为物理NZ `320,1088,16,16`，v3为逻辑 `17408,5120`，相同原权重及格式。此描述符表示差异未产生额外布局任务。

下表每臂9个task中位，时间单位µs；pipe时间为官方公式归一的平均活动时间，task duration为设备wall：

| 字段 | vendor | v3 |
| --- | ---: | ---: |
| task duration | 592.896 | 900.394 |
| aicore_time | 577.008 | 888.565 |
| MAC时间 / ratio | 507.611 / 0.880 | 532.679 / 0.600 |
| MTE1时间 / ratio | 427.521 / 0.741 | 398.056 / 0.448 |
| MTE2时间 / ratio | 448.028 / 0.776 | 735.144 / 0.828 |
| Fixpipe时间 / ratio | 11.078 / 0.019 | 126.644 / 0.142 |
| context total cycles | 24,926,749 | 38,385,994 |

请求组 `ACL_AICORE_PIPE_UTILIZATION`，本硬件实际组 `PipeUtilizationExct`、task-based、replay=0。官方 `profiling_bean/stars/ffts_pmu.py` 解码2304B `ffts_profile.data.0.slice_0`：18条context、0条block PMU、无overflow；新DB TASK_PMU_INFO为18×28=504条，无core ID。事件组 `0x416,0x417,0x9,0x302,0xc,0x303,0x54,0x55`，raw cycles与ID映射保留summary。硬件频率1800MHz，当前24-block单wave；官方时间公式为 `event_cycles/(24*1800)`，ratio为 `event_cycles/total_cycle`，MAC分子为r416+r417，MTE1=r302、MTE2=rc、Fixpipe=r303。18条逐项核同CSV，最大时间舍入差0.000491µs；公式来源 `msconfig/ai_core_config.py` 与 `mscalculate/stars/ffts_pmu_calculator.py`，路径/SHA留summary。`cube_utilization(%)`是平均core elapsed相对task wall的归一值，其公式未提供MAC计算占比。

**证据支持的核查顺序：** ①MTE2搬入/复用效率（活动时间+287.116µs，约+64%）；②Fixpipe写出路径（相同逻辑输出，活动时间约11.43倍）；③MAC活动约+25.068µs（+4.9%），task约+52%。以上活动可重叠，不能相加解释0.309ms，不能按100%−MAC估算收益，也不能把MTE2忙直接认作HBM饱和。缺少wait/stall、传输字节数与per-core记录，独占critical-path占比、active核数/长尾、L1/L0等待仍UNKNOWN；runtime v3 TilingData及vendor opaque tiling未导出。

当前值得保留的最小方向是先核runtime C写出分区/burst与搬入复用，再据证据决定实现调整。此前SDK CPU模型singleCoreN=5808时，N起点0/5808/11616对应BF16字节偏移0/11616/23232，mod512为0/352/192；该模型尚未核同runtime，仅为地址核查线索，未把512B对齐当API硬要求或已定位原因。当前v3 gate-only的task惩罚仍超过SwiGLU预算，本轮停止设备实验，保留对其他BF16融合方案的开放结论。

本次外层cap40s，计算/采集 **9.422204s**，累计 **142.562622/240s**；超时/重放/POST重试均0。fresh核指针、完整树、18143、原argv/env/manifest与卡7双芯片后定向停Clef，raw落盘立即恢复，导出随后执行。最终 **2026-11-22 03:23:15（上海）RUNNING_VERIFIED**，22/22检查通过；API **1695411 / start420588963 / boot393ecc15-721f-46f7-b119-21fa327a2b3b**，TP **1695597/1695598**仅卡7 Chip0/1/Phy14/15，原argv/env、aa3e、OPP9/LD12及设备归属匹配，文本/图像各一次200。env仍c18d38bd…c2ba9/4678B，两端当前指针更新，manifest SHA `29f0790b53ed46991e99daeea4986febe704a25f3c239586ff09657279f2719c`。 **03:47:16（上海）末次只读核查**完整树/health/18143/卡归属、原argv/env/manifest与两端指针仍匹配，无实验进程。

raw位于本地系统tmp `clef40-gate-pipe-d42qupyu`，远端同名work/logs；导出目录为logs下 `traces-r0/gate-r0_1695327_20261121192144653_ascend_pt/ASCEND_PROFILER_OUTPUT`。原summary新增 `v3_pipe_profile`，含精确argv、SHA、18条raw cycle、逐样本原数、归一公式、恢复与缺项。离线复现：

```bash
python3 runtime/analyze_mlp_gate_profile.py \
  --trace "$task_raw/trace-r0.json" --kernel-csv "$task_raw/kernel_details.csv" \
  --samples "$task_raw/rank0.json" --output "$task_raw/analysis.json"
python3 "$task_raw/analyze-pmu.py"
```

CPU检查覆盖plan/语法、精确task join正例及重复ID/错误核名/错时基/缺ID负例，实际18任务与原始计数/归一公式核同通过。无新artifact/PR，无kernel修改或全量测试。

实际host tiling观测续接：既有 `/Users/chenclay/.ssh/clef-master.sock` 缺失，master检查与只读SSH均以255退出；未建立远端观测，未调用GetWorkspace/Execute，未取得实际212B，50字段runtime比较仍UNKNOWN。遵守仅复用用户master，未重建连接或修改配置/部署/服务；纯host执行边界待连接恢复后先核。设备预算保持142.562622/240s，最新服务验证仍引用上述03:47记录。精简状态见summary `v3_runtime_host_tiling`。


host观测续接（2026-11-22 09:13，上海）：用户master恢复，通道确认成功，aa3e/S1保持运行。私有host副本仅在所有N/字段修正后增加212B写盘；原host与仓库仅首行注释不同，已核同数学代码，kernel/opapi SHA保持原值。CANN9.2.1的200+12B结构静态断言及host-only入口编译通过。生成GetWorkspace仅走`NnopbaseRunForWorkspace`，Execute另走`NnopbaseRunWithWorkspace`；入口只解析前者，构造同M967/BF16/NZ/ND描述符。CANN加载前安装进程级seccomp限制ioctl/共享mmap/io_uring，不调用ACL init、setDevice、设备分配、stream或Execute，描述符地址来自PROT_NONE的CPU私有映射。

唯一一次受限GetWorkspace返回 **361001**、executor为空，**实际212B未生成**。按生成代码，失败发生在executor创建阶段，尚未进入TilingFunc；当前日志未分离平台初始化与被限制系统调用的具体依赖。此入口未证实可以在严格host-only限制下取得runtime tiling，按范围收口，不放开限制、不初始化设备重试。v3 runtime与静态50字段比较继续UNKNOWN；MAC/MTE2/Fixpipe活动增量25.068/287.116/115.566µs保留，额外burst、实际搬入重读、vendor opaque、逐core及独占wait仍缺证据，活动时间不相加。

09:13:38末次只读确认API **1695411/start420588963/boot393ecc15-721f-46f7-b119-21fa327a2b3b**、原argv与指针保持一致、health ready、aa3e源码匹配；无重启或text/image POST。新增设备计算/采集0s，累计 **142.562622/240s**。私有源码/入口/二进制SHA、实际命令/错误与服务前后身份写回原summary `v3_runtime_host_tiling`；raw留系统tmp及远端work同名 `clef40-host-observe-z4p5wl16`，无新artifact/PR。
