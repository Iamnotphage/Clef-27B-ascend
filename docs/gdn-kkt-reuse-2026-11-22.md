# GDN raw FP32 KKT 按 K head 复用

基线 `b14cd0a8a0608ce10ef6d0ef3ab28511e59d46d3`，分支 `perf/gdn-kkt-reuse-20261122`。当前为私有源码准备/构建阶段，设备验证与收益待测，aa3e/S1保持运行。此任务独立设备预算240s，已用0s；旧MLP的142.562622s记录保持原值。无PR或合并。

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
