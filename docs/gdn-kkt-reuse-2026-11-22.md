# GDN raw FP32 KKT 按 K head 复用

基线 `b14cd0a8a0608ce10ef6d0ef3ab28511e59d46d3`，分支 `perf/gdn-kkt-reuse-20261122`。当前按生产参照未对齐停止，aa3e/S1已恢复并fresh核同。此任务独立设备预算240s，累计记账15.585378s；旧MLP的142.562622s记录保持原值。无PR或合并。

部署wheel为FLA NPU A3 26.9.1，归档SHA `0e0febfe0b524df54c02dead985b787cf51f276ef904e4c2f88c5d43707a370b`。worker实际加载的stable/opapi/opmaster与wheel归档逐字节核同；另核89个GDN kernel源文件，总92项一致。发布源码固定 `5cc699069c7f91a35aabd583032c8720a0b626fa`；89文件统一换行后全部一致，1文件原始字节一致，其他为CRLF差异。此证据建立发布包来源关系，尚未证明重编译二进制逐字节相同，局部测试仍需同源stock对生产wheel校准输出与速度。

实际部署调用为BSND、BF16、Hk/Hv=8/24、K/V=128、chunk128、raw g、无初始/最终状态。公开L2 adapter对此选择raw-g BTH与prepared-cumsum，源码预期tiling key3，实际key待私有stock首调用记录。S1的一条序列仍以cu[2]/chunk_indices表示，不能按isVarlen字段单独拒绝该正常分支。

私有[准备器](../runtime/prepare_gdn_kkt_reuse.py) 默认仅输出计划；显式prepare核固定5个源文件SHA，复制到全新目录，生成独立`ClefGdnKktStock`和`ClefGdnKktReuse`低层算子，不写wheel或共享GEV。省略原L2 adapter，使用SDK生成的独立L0 API；完整wrapper与生产wheel的差异须在测试中单列。构建工具兼容修复只限私有副本：日志时间戳gawk→awk，以及独立算子的def文件发现命名。

数学变化限于四处：Phase6 trailer显式复用标志；score工作区按Hk分配；Stage P局部producer tiling改为B*Hk*NT；AIV将原(b,hv,chunk)映射至(b,hv/3,chunk)只读score。原ABC taskNum及g/cumsum/beta/mask、FP32 solve、W/U保持逐Hv。首版guard仅单序列规范metadata、BF16/BSND/key3、chunk128、T≤1595及无状态；多字段、长形状、多段与未知契约先回退wheel。

| 数据/阶段 | 所有者与生命期 | 发布与消费 |
| --- | --- | --- |
| raw score | 每个Hk/chunk唯一producer；共享槽只读；位于独立追加工作区 | KktCube退出前drain FIX_M；每组仍发送一次SCORE_READY；已有全组prepared-cumsum SyncAll后消费 |
| 门控FP32 A | 每个Hv独立，写入solveFp32Input | 原AIV MTE2/V/MTE3协议保持，随后逐Hv solve |
| W/U及后段 | 原ABC任务、物理组及scratch边界 | 原front批握手和末端全组会合保持 |

无producer的物理组仍走初始化/drain/通知及barrier，不以局部SCORE_READY替代跨组发布。静态M967模型score tile192→64、工作区12→4MiB；整核收益待测。局部先比有效raw score、门控A、最终o，数值与重复通过后做至少5对AA/AB。生产wheel为最终参考，收益未超过AA、同步/搬运抵消或数值失败即止损；局部稳定获益后才集成、做正式S1三轮与完整四项2pp验收。

精简状态与来源SHA见[summary](../Clef-27B/logs/gdn-kkt-reuse-20261122/summary.json)。大源码、构建和原数位于系统tmp及远端work同名`clef40-gdn-kkt-dgkgrdjy`。

CPU索引6长度与dispatcher guard 11项通过，三个Python入口AST/py_compile与默认plan通过。同步扫描覆盖Phase6、KktCube、ScaledDotKkt及FP32 solve pipeline：stock/reuse各59个候选，规范化行号后消息集合相同，含30个红线/高候选；完整候选保留tmp的sync-critical-candidates.txt及sync_audit*.json。图analyzer无findings，正则前端缺完整模板/宏/跨文件证明能力，尚不构成同步通过。保持原同步代码，按已授权边界以stock及重复数值检验；不会凭候选改写原协议。

构建接口已按SDK生成头核对：两个私有OpDef使用op_host_aclnn（排除target会生成空头）；ValueDepend的cu/chunk选择aclIntArray重载，参数顺序与生成头一致。SDK/opbase源码给单工作区基址=传入workspace，DAV_2201 RESERVED_WORKSPACE=16MiB；probe仍以独立CPU KKT数值核验声明偏移，不扫描工作区猜地址。生产输入计划只在私有capture实例用一次性导入钩子保存1595-token首层q/k/v/g/beta，前缀长度明确标注；生产源码/wheel保持原件。

## 本轮实际结果与停止依据

私有包已成功编译，vendor为`clef40_kkt_transformer`，未安装到生产。SDK生成API参数及aclIntArray重载已核对。首窗口在stock GetWorkspace报组合tiling容量不足，执行恢复22/22；NNOPBASE使用binary metadata的opParaSize（584/592B）初始化raw tiling，Phase6还要追加trailer。仅将私有8份metadata容量修正为SDK注册默认2048B，8份ELF逐字节不变；修复脚本为[fix_gdn_kkt_package.py](../runtime/fix_gdn_kkt_package.py)。第二窗口通过该接口，实际stock/reuse key均为3。生产wheel的key缺直接观测，继续标为UNKNOWN。

双rank真实输入取自原OD-0677（1595 token）一次性armed捕获，q/k为BF16 `[1,1595,8,128]`，v为BF16 `[1,1595,24,128]`，g为FP32、beta为BF16 `[1,1595,24]`。两rank的同K head下g/beta均有逐Hv差异；capture合同与原BSND/raw-g/chunk128/无状态契约核同。数值首点用明确标注的967前缀。

| rank0 / M967 | 结果 |
| --- | --- |
| 64个共享raw FP32 tile → 原192槽 | 192次有效区比较逐位相同 |
| 独立CPU FP32 KKT核偏移 | 64次通过；最大绝对差6.557e-7，保留加法顺序容差 |
| 逐Hv门控FP32 A | 192个有效chunk逐位相同 |
| 候选最终o → 私有stock | 逐位相同，全部有限 |
| 候选重复 / 工作区末端哨兵 | 2/2逐位相同，哨兵通过 |
| 私有stock最终o → 生产wheel | **未通过**，max_abs=5.6875，全部有限 |

实际host tiling保留ABC taskNum=192，仅producer为64；scoreBytes从12MiB降至4MiB，后续solveInputOffset相应前移8MiB。该内存变化已观察，净性能未测。首个失败检查为私有stock对生产wheel的最终o；当前证据不足以把差异归因于包装、编译或发布二进制对应性。共享score在已测首点与stock同值，保留机制可行性的边界。

按停止条件结束本实现的设备验证，rank1、其余长度/varlen/长输入、A/A与A/B性能、服务集成、S1/OD/ERQA均未执行。完整wrapper代码与已部署C++ L2不同，未经参照对齐不可形成生产收益结论。下一步继续须先用具体CPU证据解释并修正stock/wheel不等值，禁止凭此失败否决所有K-head复用方案。

两个窗口均先恢复再收口。最后aa3e/S1 API **1734792 / start423193402 / boot393ecc15-721f-46f7-b119-21fa327a2b3b**，TP **1734963/1734964**；22项worker/health/text/image、原argv/env、卡7 chip0/1 Phy14/15及双端当前指针核同。env SHA仍为`c18d38bd10b4d8d1696cccb3f307f9dd72b28ea44ea4a816cd69852e229c2ba9`。fresh检查确认实验进程已退出，生产源码/GEV/wheel未改。

最小复现：固定发布源码后运行prepare；按`--pkg --soc=ascend910_93 --ops=clef_gdn_kkt_stock,clef_gdn_kkt_reuse -j4`私有构建，生成API须使用op_host_aclnn。确认私有目录与ELF SHA后应用容量修复；仅在owner新建的有界窗口执行`probe_gdn_kkt_reuse.py --execute --phase numeric --rank 0 --vendor <private-vendor> --captures <actual-capture-dir> --output <new.json>`。两个rank数值全部通过才允许`--phase timing`。本轮使用[一次性capture钩子](../runtime/capture_gdn_kkt_inputs.py)与[探针](../runtime/probe_gdn_kkt_reuse.py)，完整命令、原数、tiling日志、编译日志和恢复记录均留tmp，可由summary的路径/SHA定位。

串行后续队列保持：MLP先准备C地址/stride/尾块CPU清单，再以固定几何选择一个有机制依据的预取或合法基址相位控制；actual212B、vendor tiling、逐核等待保留UNKNOWN。packed K/V head随后。当前阶段未启动后续候选构建或设备实验，down保持结束状态。
