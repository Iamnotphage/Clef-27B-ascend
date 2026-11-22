# CATLASS跨输出tile首K片预取（2026-11-22，当前ON负结果）

固定官方模板在CANN9.2.1/A3可编译并通过真实数值门。跨输出tile预取ON未带来完整链收益，停止当前ON；不进入epilogue、服务集成或S1/OD/ERQA。OFF同源基线的纯task优势单独保留。

base `5086ad4`，分支`perf/mlp-cross-tile-prefetch-20261122`；CPU/编译提交`b0c0634`，执行入口`a95b9d8`。固定CATLASS `eec373fbe601446c72cac422cf63142221d85bf9`（公开v1.2.0），官方头文件保持原字节，五个关键SHA核同。私有vendor未安装进CANN或服务。

## 单变量合同与CPU可行性

同一官方`BlockMmadPreload`、同一ELF、同编译/分配/输出路径，固定L1(128,256,256)/L0(128,256,64)、24核、unit flag开启、K shuffle关闭。两臂保留tile内部下一K片预取；仅以两个关联布尔参数切换跨输出tile首K片：OFF每tile入口加载首片；ON在末K片期间提前加载下一tile首片。末tile仍传有效next shape以避免无效K/地址哨兵。

原BF16 [N17408,K5120] FRACTAL_NZ物理存储直接作为逻辑B[K,N]的nZ；174,080个边界/内点偏移核同。L1双缓冲384KiB、L0A双缓冲32KiB、L0B双缓冲64KiB、FP32 L0C128KiB。K5120按80个K64块保持FP32累加，仅首块初始化，最后一次`F322BF16`/`CFG_ROW_MAJOR`写canonical ND [M,17408]。采用公开K256切片双缓冲；K全载MDL与IBShare-B的前提不在该路径。

M967有544输出tile，24核ON提前520次首K片，OFF/ON各10,880次A/B片读取；M307/1595为204/884 tile、180/860次提前。CPU逐slot模型验证调度覆盖、生产/消费与最终释放；不把相同理论流量的搬入时机调整记为减少bytes。

写出两臂保持相同行距34,816B、N256边界、尾M和Fixpipe调用。unit flag开启分支在block入口省去显式FIX_M等待，OFF首K片载入也具备与前tile异步写出重叠的源码条件；ON进一步前移到末K计算期间。实际重叠比例、内部MTE/Fixpipe等待与GM基址相位效应未隔离。

首两次编译分别遇到MakeCoord<int>转换、MatrixCoord 32/64位重载歧义；定向改为`MatrixCoord(128U,256U)`后通过。数学、tile与官方事件未改。实读生成GetWorkspace签名为x/weight/**int64 prefetch**/out/workspace/executor。

## 真实数值

历史11-15捕获与原权重：layer31/0/63×两rank，M967先过，再M307/1595，共18个唯一case；967明确为od-0677/1595的前缀，307为完整od-0105。父输入SHA、权重来源在原数。NZ转换一次并在计时外，两臂共享同值输入/权重。

18组ON/OFF均逐位、有限且重复稳定，canonical ND与gate/up顺序检查通过。相对vendor最大relRMSE **3.73645e−5**、max_abs **0.0625**；vendor对照存在微小数值差，保持原BF16和FP32累加位置。完整gate+SwiGLU+原NZ down数值门通过。此局部精度门不替代四项全量2pp验收。

## 同窗完整调用

两rank串行、各5对交错，完整预热单列；每pair各arm先取慢rank，再算配对差。主口径layer31→0→63 streaming，以下每层均摊ms。Event含host下发，wall含完成等待；输出分配、布局/cast、生成API桥接/下发和GM写出全部计入。vendor gate/chain与stock gate提供各自A/A。

| 路径 | Event中位ms | wall中位ms |
|---|---:|---:|
| vendor gate | 0.603500 | 0.634877 |
| stock OFF gate | 0.618433 | 0.645927 |
| prefetch ON gate | 0.618133 | 0.649906 |
| vendor完整MLP链 | 0.950827 | 0.985000 |
| stock OFF完整链 | 0.963667 | 0.994970 |
| prefetch ON完整链 | 0.965193 | 0.993753 |
| vendor SwiGLU独立 | 0.061327 | 0.091103 |

配对Event：ON−OFF gate中位 **−1.687µs**，stock A/A最大 **9.520µs**，仅3/5对更快；ON−vendor gate **+12.947µs**，vendor A/A最大 **4.333µs**，未进入vendor带。ON完整链−vendor **+14.567µs**，A/A最大 **6.433µs**，0/5对更快。wall对应ON−OFF **+2.390µs**、ON−vendor gate **+11.846µs**、链 **+9.480µs**；其A/A最大11.593/12.900/21.107µs，未出现稳定收益。

warm单layer31另列：vendor/OFF/ON gate Event **0.606580/0.710380/0.696240ms**，wall **0.698110/0.791661/0.785790ms**；相应完整链Event **0.963020/1.081380/1.124740ms**。不以warm局部差代替streaming结论。两rank逐对原数和完整预热见summary。

## 最小task关联

成功路径CPU+NPU Level1/AiCoreNone，无PMU/MSTX/application replay。代表rank0、相同3层streaming，4个gate arm、3对profile，前后各2对无prof控制。12个range/36个GEMM按原生flow及stream/task/name/start精确核同CSV：每层1kernel、BlockDim24，无额外布局/cast/copy任务。

| 路径 | kernel union中位µs/层 | 核名 |
|---|---:|---|
| vendor | 596.6623 | aclnnMatmulWeightNz_MatMulCommon_MatMulV2 |
| vendor repeat | 595.9957 | 同上 |
| stock OFF | 563.5405 | ClefMlpCrossPrefetch |
| prefetch ON | 585.4285 | 同一个ClefMlpCrossPrefetch |

ON−OFF三对task差 **+22.668/+17.868/+35.109µs**，中位 **+22.668µs**；vendor task A/A最大0.913µs。profile未采stock repeat，stock A/A保留主无prof计时。OFF任务快于vendor，完整wrapper仍落后；当前接口/非kernel区间的具体host、排队、等待占比继续UNKNOWN，不能将不同样本Event减task直接记为纯host耗时。活动重叠与内部等待无新计数证据。

采集扰动与代表性：rank0 streaming vendor Event前/profile/后 **0.598320/0.612940/0.603667ms**，OFF为 **0.618680/0.625120/0.621027ms**，ON为 **0.631400/0.643660/0.630543ms**。该短采4个gate arm，主计时10臂的邻接负载不同；前后控制与短采同4臂。其task结构/差异只代表该点，不能替代两rank完整链结果或正式P95。原v3的约309µs差距属于另一实现/窗口；本模板改变GEMM主循环与tile布局，跨实现差距不能归因于本次预取开关。

## 同步静态边界

正则扫描官方block+回写与私有入口：11项红线/高候选、1项性能候选；flow扫描0项。以下原样保留message，完整detail在tmp sync-audit.json；不将有限CPU模型当硬件同步证明，不据此改官方同步或增加审批链。

|编号|等级/条目|源码行|message|
|---|---|---:|---|
|1|红线/SYNC-04|222|Set/Wait 个数不一致: 方向=MTE1_MTE2 flag=l1AEventList[l1ListIdNext] Set=0 Wait=2|
|2|红线/SYNC-04|222|Wait 无对应 Set（死等风险）: 方向=MTE1_MTE2 flag=l1AEventList[l1ListIdNext]|
|3|红线/SYNC-04|228|Set/Wait 个数不一致: 方向=MTE1_MTE2 flag=l1BEventList[l1ListIdNext] Set=0 Wait=2|
|4|红线/SYNC-04|228|Wait 无对应 Set（死等风险）: 方向=MTE1_MTE2 flag=l1BEventList[l1ListIdNext]|
|5|高/SYNC-01|167|Wait 先于 Set（跨循环，可能为双 buffer 合法序）: 方向=FIX_M flag=EVENT_ID0|
|6|高/SYNC-01|192|Wait 先于 Set（跨循环，可能为双 buffer 合法序）: 方向=MTE1_MTE2 flag=l1AEventList[l1ListId]|
|7|高/SYNC-01|198|Wait 先于 Set（跨循环，可能为双 buffer 合法序）: 方向=MTE1_MTE2 flag=l1BEventList[l1ListId]|
|8|高/SYNC-02|225|循环内 Set 多于 Wait（疑似 buffer 复用缺循环顶部 Wait）: 方向=MTE2_MTE1 flag=l1AEventList[l1ListIdNext]|
|9|高/SYNC-02|231|循环内 Set 多于 Wait（疑似 buffer 复用缺循环顶部 Wait）: 方向=MTE2_MTE1 flag=l1BEventList[l1ListIdNext]|
|10|高/SYNC-01|282|Wait 先于 Set（跨循环，可能为双 buffer 合法序）: 方向=M_MTE1 flag=l0AEventList[l0ABufId]|
|11|高/SYNC-01|305|Wait 先于 Set（同函数消费+生产，可能为 ping-pong 合法序）: 方向=M_MTE1 flag=l0BEventList[l0BBufId]|
|12|性能/SYNC-09|54|PipeBarrier<PIPE_ALL> 粒度过粗|


第1–11项来自官方block_mmad_preload.hpp，第12项来自私有入口clef_mlp_cross_prefetch.cpp。上述静态候选保留，18组局部数值/重复覆盖仅代表已选路径。

## 账目、恢复与复现

新MLP局部计算/采集累计 **40.16165188048035/240s**，包含4个有界probe子进程的完整wall；编译/加载/恢复另设界。GDN **76.87979675224051s**、Head **22.48895183391869s**与旧MLP账均保持冻结。未跑shape/tile矩阵扫描，307/1595性能阶段因M967未过门而未进入。

原aa3e/S1 **RUNNING_VERIFIED**：API **1754374/start424176890**，boot `393ecc15-721f-46f7-b119-21fa327a2b3b`。22项worker/health/text/image等通过，原argv/env核同；离线导出后再次fresh核完整身份、源码、env、health、卡7/chip0/1（逻辑0/1、Phy14/15）归属与双端指针，私有probe树已清理。用户master、共享GEV及生产源码保持原样。

原数本地tmp `clef40-cross-prefetch-at10fi3u`，远端`/workspace/yangchen/work/clef40-cross-prefetch-at10fi3u`。`window.json`为预算/实际命令/返回码；`numeric967.json`、`numeric_guards.json`、`timing967.json`、`profile.json`为完整原数；`task-analysis.json`保存36任务关联，原生trace/CSV路径与SHA见summary。大raw仅留tmp。

CPU复现：`python3 runtime/check_catlass_cross_prefetch.py --catlass-root FIXED_SOURCE_ROOT`；私有编译设置`CATLASS_ROOT=FIXED_SOURCE_ROOT`后运行工程`build.sh`。默认`python3 runtime/probe_mlp_cross_prefetch.py`只打印plan；实际`--execute --phase numeric/timing/profile`的tokens/ranks/vendor/output/timeout见tmp window记录。离线解析复用`runtime/analyze_mlp_gate_profile.py --sample-prefix crosspref/ --divisor 3 --trace ... --kernel-csv ... --output ...`。CPU语法、固定source SHA、3M逐slot模型与实际36任务映射均通过。精简原数在`Clef-27B/logs/mlp-cross-tile-prefetch-20261122/summary.json`。


## CPU补记：OFF调用层拆分（仅既有trace）

同pair、同层31/0/63、同trace时钟；先逐pair差再取中位，单位µs/层。range self是扣除同主线程可见子范围/API后的残差，包含未标记Python/FFI/调度，含义不同于CPU占用。所有host/API与kernel均可重叠。

| 字段 | OFF−vendor配对中位 | vendor A/A最大绝对差 |
|---|---:|---:|
| kernel union | −32.802 | 0.913 |
| Event起点→首kernel | +23.101 | 76.612 |
| kernel之间空档 | ≈0 | 0.007 |
| 首kernel前可见范围未覆盖段 | +43.257 | 1.838 |
| 末kernel→末Event | ≈0 | 0.007 |
| 末kernel→同步返回 | +1.058 | 1.523 |
| 主线程range inclusive / self残差 | +16.397 / +127.360 | 170.480 / 87.940 |

“未覆盖段”扣除主线程可见CPU范围、关联worker dequeue和native API的union；余量含未标记Python/FFI/调度，不等于可省空闲。

可见API（各arm中位incl/self）：OFF输出aten::empty 9.723/2.620，vendor empty_tensor 3.827/3.827（范围层级不同）；OFF GetWorkspace 9.030/8.127、外层launch 23.423/8.493，vendor合并aclnnMatmulWeightNz 9.570/3.127（内部GetWorkspace未拆出，记UNKNOWN）。两臂相同native aclrtLaunchKernelWithHostArgs分别为OFF10.477/10.477、vendor5.317/5.317。以上嵌套/重叠字段不相加拟合33µs。

首层Node@launch的原生HostToDevice源→kernel延迟为vendor18.361/8.946/9.840、OFF21.132/17.197/20.570µs；vendor另有Torch enqueue→首kernel35.701/26.226/24.460µs，OFF为主线程直接launch，没有同类Torch enqueue字段。它们包含API/排队，不能视为可省等待。逐层API与flow关联原数留tmp off-host-breakdown.json。

代表性限制：profile三对OFF−vendor Event为−64.547/−8.633/+13.180µs、wall为−66.530/+10.823/+18.783µs。首对vendor Event::record测量壳146.947µs/层，后两对35.097/36.620；range与Event统计受明显扰动。12个range的trace Event跨度与记录Event.elapsed_time最大差0.04435µs/层。原无prof streaming OFF−vendor Event配对中位rank0/rank1为+4.707/+13.320µs；rank1无task短采。代表rank0短trace不能量化解释双rank慢者的全部缺口。

成本核查：完整链两臂均包含gate输出分配/canonical BF16 ND、同一SwiGLU及其输出分配、同一原NZ down。输入/权重搬运与NZ转换、数值cast/D2H、JSON dump均在样本外；完整链未留逐task trace，GPU内部等价只保留源码/格式层证据。OFF只见3次输出empty，未见额外workspace empty；返回workspace字节数未记录。

结论：定位到OFF主线程适配路径及首kernel前区间，**尚无已证实可安全消除并回收关键预算的单点**。descriptor创建/销毁、重复ctypes签名绑定、device guard等未标独立范围；保存的发布版_runtime.py亦缺已部署单文件SHA核同。GetWorkspace executor可能绑定当前地址，现有记录未覆盖地址、shape/dtype/stride、stream、buffer生命期/并发及executor重用约定，不能直接缓存或挪出必要成本。保持UNKNOWN，无新实现/采集；预算仍40.16165188048035s，服务未操作。
