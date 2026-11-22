# GDN raw FP32 KKT 按 K head 复用

base `b14cd0a8a0608ce10ef6d0ef3ab28511e59d46d3`，分支 `perf/gdn-kkt-reuse-20261122`。本版已**数值校准通过**，Event/wall初筛未超A/A；按用户续接补一次最小task trace，保持原aa3e/S1。GDN预算累计 **61.294507/240s**，编译、服务加载/恢复另行有界；旧MLP的142.562622s保持原记录。无PR/合并，后续候选按用户最新指导暂缓。

部署FLA NPU A3 26.9.1归档SHA为`0e0febfe0b524df54c02dead985b787cf51f276ef904e4c2f88c5d43707a370b`。active stable/opapi/opmaster与归档逐字节核同，另核89个kernel文件，总92项一致。固定发布源码`5cc699069c7f91a35aabd583032c8720a0b626fa`，89文件统一换行后全同。重编译ELF与wheel逐字节等价未建立；本轮通过生产公开边界与最终o校准其功能契约。

## 私有实现与ABI修复

[准备器](../runtime/prepare_gdn_kkt_reuse.py) 默认仅plan。显式prepare核固定源SHA，复制全新私有树，生成独立`ClefGdnKktStock`/`ClefGdnKktReuse`及`clef40_kkt_transformer` vendor。共享源码、GEV、生产wheel未改。构建适配包括日志gawk→awk、独立def发现命名和正常API的op_host_aclnn target。

唯一计算变化为Stage P raw score的任务轴`B*Hv*NT→B*Hk*NT`、score分配及consumer只读索引。ABC任务/g/cumsum/beta/mask、BF16 RINT→FP32、FP32 solve、W/U、H/O保持。首版限A3/DAV_2201、BF16/BSND、raw-g BTH prepared、Hk/Hv8/24、K/V128、BT128、单正长度序列及无状态；未知key/多段/长输入/状态路径保留stock或dispatcher回退，采用前仍须补上卡护栏。

前两个窗口的包装问题已解决：

- SDK生成API的binary metadata把首结构大小584/592B用作raw tiling容量，组合Phase6实际需1032/1040B。仅在私有8份metadata设为SDK注册默认2048B，[容量适配脚本](../runtime/fix_gdn_kkt_package.py)核8份ELF逐字节不变。
- normal ACLNN由OpDef Float生成`float tmp2=static_cast<float>(scale)`；原composite `OP_ATTR(scale)`传double。原H/O host的`GetAttrPointer<double>(ATTR_SCALE)`用于normal路径存在4B/8B错配。仅私有normal host改读float，最终`oTiling.scale`仍为float，CPU证明其位型与原composite的double→float相同。**生产composite保留原double传递/读取**；此私有适配不直接套用生产L2。

旧max_abs=5.6875在最终o检查出现，旧实际oScale未记录。修复后实际序列化O字段为`0x1.6a09e6p-4 = 0.0883883461356163`，旧最终差异消除。源码字节检查与双rank公开边界校准支持本次ABI定位；零填充字节示例仅为CPU说明，未当作旧runtime观测。

## 数值与实际tiling

原OD-0677的1595-token请求只在临时capture实例armed一次；保存两rank首层真实归一化后q/k/v/g/beta。q/k为BF16 `[1,1595,8,128]`，v为BF16 `[1,1595,24,128]`，g FP32/beta BF16 `[1,1595,24]`；各组不同Hv的g/beta均有差异。局部各长度是明确标注的冻结1595前缀。原L2Norm/no-state/no-recompute合同保持，capture与timing分离。

公开wheel只把`disable_recompute`改为True以请求cumsum和solved A，保留`return_intermediate_states=False`及全部其它实参；先核导出与原no-export的o逐位相同，再使用这些公开中间量。生产FP32 solve输入x无公开出口，继续标为UNKNOWN。

| 检查 | 两rank结果 |
| --- | --- |
| M967生产export/no-export o | 逐位同值，全部有限 |
| 生产cumsum、有效下三角solved A、最终o → stock-flat/stock-dense/reuse-dense | 三臂全部逐位同值 |
| descriptor控制 | shape/stride/format/storage offset及storage extent已记录；本样本flat/dense均对齐生产 |
| 127/128/129/307/967/1595，共12组 | raw共享槽→原stock槽、逐Hv FP32 x、最终BF16 o全部通过；o与生产wheel逐位同值 |
| 重复/工作区末端哨兵 | 24次重复同值，12组哨兵通过 |

实际私有key均为3；生产key缺直接host dump，保持UNKNOWN。每rank M967：ABC taskNum=192，producer=64，scoreBytes 12→4MiB，coreNum=24；原全组Fixpipe完成发布、SCORE_READY和prepared SyncAll保留，含零producer组。运行时raw tiling为stock1032/reuse1040B、trailer576/584B，O offset168、trailer offset456；33个标量前缀和完整动态trailer从实际字节解码，并与host观测逐项核同，Cube区200B原件保留。静态同步扫描保留stock/reuse各59候选（30红线/高），规范化行号后集合相同，图analyzer无findings；正则工具缺完整模板/跨文件证明能力，未据此宣称全路径同步证明。

## 同窗局部性能与停止判断

数值全部通过后，M967做5对交错A/A、A/B；完整预热单列，观察/描述符记录不在计时内。原数保留两rank，以下每对先取慢rank，再取中位。配对改善是逐对`1-reuse/stock`的中位，与两列中位数之比不同。

| 口径，ms | wheel | stock | reuse | 配对改善中位 | A/A最大波动 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 完整wrapper wall | 0.618360 | 0.947201 | 0.922930 | +3.65% | 23.39% |
| 完整wrapper Event区间 | 0.477740 | 0.846460 | 0.809740 | +5.57% | 28.75% |
| 预建L0下发Event区间 | — | 0.406300 | 0.420060 | −1.51% | 4.86% |

wrapper包含beta转换/BHT连续化、分配、metadata、下发与输出成本。预建L0区间排除了GetWorkspace及前处理，仍包含host launch/queue间隙；独立kernel task trace未采，task时长及内部Cube/MTE/等待继续UNKNOWN。没有把Event区间当作纯kernel或正式P95。此版完整wrapper收益未超过A/A，且候选wrapper慢于生产wheel；预建L0也未显示稳定正收益。按局部门槛停止本版集成，保留KKT复用数值成立的结论。当前结果不支持服务性能收益，也未给出内部抵消机制的因果归因。

varlen/长输入/fallback上卡、服务集成和正式S1/OD/ERQA/S2未执行；采用条件未满足。MLP、down、head队列均未启动。

## 恢复与复现

最后aa3e/S1为 **API1737460/start423304809/boot393ecc15-721f-46f7-b119-21fa327a2b3b**，TP **1737601/1737602**。worker/health/text/image **22/22**、原argv/env、卡7 chip0/1 Phy14/15及两端当前指针核同；fresh health ready，无实验进程。env SHA仍为`c18d38bd10b4d8d1696cccb3f307f9dd72b28ea44ea4a816cd69852e229c2ba9`。

最小复现：固定发布源码后运行prepare，私有构建`--pkg --soc=ascend910_93 --ops=clef_gdn_kkt_stock,clef_gdn_kkt_reuse -j4`，应用私有容量适配。仅在owner fresh绑定的有界窗口运行[探针](../runtime/probe_gdn_kkt_reuse.py)：先`--phase calibration --logical-storage`，两rank通过后`--phase numeric`，全部护栏通过后`--phase timing`，每次使用全新output。[capture钩子](../runtime/capture_gdn_kkt_inputs.py)仅用于私有实例的一次性armed请求。当前已运行代码/数据SHA与精简原数在[summary](../Clef-27B/logs/gdn-kkt-reuse-20261122/summary.json)。

源码/构建原件：local tmp `clef40-gdn-kkt-dgkgrdjy`，远端`/workspace/yangchen/work/clef40-gdn-kkt-dgkgrdjy`。本轮校准/12组数值/5对原数、36份实际host tiling及解码、恢复记录：local tmp `clef40-gdn-kkt-cal-g6ur8jr3`，远端同名work目录。完整路径、SHA及两次历史失败原件由原位summary定位；大raw保留tmp，无新报告artifact链。

最新task补采准备：沿已成功CPU+NPU Level1/AiCoreNone路径，两rank串行各3对，完整wrapper与预建L0分列，profile前后各2对无prof控制；关闭stack/shape/memory/PMU/MSTX/replay。CPU解析器以原生flow、stream和task ID映射，原MLP的divisor3接口保持，GDN用divisor1；错stream诱饵用例通过。设备总预算仍从61.294507s累计，采集完成优先恢复再离线导出。本段仅记录准备状态，task数值尚待实测。
