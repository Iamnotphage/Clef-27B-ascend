# CATLASS跨输出tile首K片预取（2026-11-22）

固定CATLASS `eec373fbe601446c72cac422cf63142221d85bf9`，base `5086ad4`，分支`perf/mlp-cross-tile-prefetch-20261122`。私有CANN9.2.1/ascend910_93编译通过，aa3e服务保持运行。设备数值与性能尚未执行。

单变量：同一官方BlockMmadPreload、同一ELF与固定L1(128,256,256)/L0(128,256,64)，FP32整K累加，K shuffle关闭、unit flag开启。OFF每个输出tile入口载首K片；ON沿用末K片预取下一输出tile首片。只改变block调用的isFirstBlock/hasNextBlock两个关联布尔量，官方搬运/MMAD/Fixpipe和事件实现保持原字节。

原BF16 [N17408,K5120] NZ物理存储作为逻辑B[K,N]的nZ，174,080个边界/内点索引核同；L1双缓冲384KiB、L0A双缓冲32KiB、L0B双缓冲64KiB、L0C FP32 128KiB。全K的80次K64累加只在首片清零，最终一次F322BF16、CFG_ROW_MAJOR canonical ND GM写出。编译验证BF16/nZ模板，实际舍入仍待真实数值门。

M967有544输出tile；24核ON提前520次跨tile首K片，OFF/ON的A/B片读取均10,880次。M307/1595对应204/884 tile、180/860次提前。CPU逐slot模型检查供需、调度覆盖与最终释放；该模型保留硬件排序/模板展开的证据边界。没有减少理论流量，不将预取计为收益。

CPU编译初两次分别报MakeCoord<int>不能转换MatrixCoord、MatrixCoord字面量32/64重载歧义；定向改为MatrixCoord(128U,256U)后通过。数学、tile和官方同步未改。生成GetWorkspace参数实读为x/weight/int64 prefetch/out/workspace/executor。

## 同步静态边界

正则扫描官方block+回写与私有入口：11项红线/高候选、1项性能候选；flow扫描0项。以下原样保留message，完整detail在tmp sync-audit.json；不将有限CPU模型当硬件同步证明，不据此改官方同步或增加审批链。

|编号|等级/条目|官方block行|message|
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

## 数值与性能门

先双rank M967真实冻结输入，再M307/1595护栏；核gate/up顺序、BF16/finite、OFF/ON逐位和重复稳定、最终MLP链。通过后5对交错vendor A/A、同源stock A/A、stock/ON，warm单列，layer31/0/63 streaming主口径。包含全部输出分配/布局/下发/写出与原SwiGLU+NZ down。候选须进入vendor A/A带或领先，完整链稳定改善才具备后续依据；慢自编译stock的相对改善单列。失败或噪声内停止，当前无epilogue/服务集成。

新MLP局部预算240s，当前0；GDN76.87979675224051s、Head22.48895183391869s冻结。原MLP账不改。只用卡7逻辑0/1，设备前fresh身份/argv/env/manifest/18143/归属；退出先恢复aa3e/S1并核22项、health、指针与卡归属。

原始材料本地 `/var/folders/s7/qw8pytxx2h947bbrpkrml20h0000gn/T/clef40-cross-prefetch-at10fi3u`，远端 `/workspace/yangchen/work/clef40-cross-prefetch-at10fi3u`；编译包仅私有目录，不安装。build-metadata.json记录实际ELF/JSON的SHA及五个关键官方header核同。

复现CPU：`python3 runtime/check_catlass_cross_prefetch.py --catlass-root FIXED_SOURCE_ROOT`；编译：在私有副本设置`CATLASS_ROOT=FIXED_SOURCE_ROOT`运行`bash build.sh`。仅调用编译，不启动NPU。

局部入口`runtime/probe_mlp_cross_prefetch.py`默认只打印CPU plan；`--execute`后显式调用实际生成的int64 prefetch接口。数值按M967双rank→307/1595推进；5对vendor/stock A/A与ON及完整链，warm/三层streaming分列。每个probe子进程有timeout与内部deadline；远端独立supervisor finally恢复，使用户SSH断开后仍能优先恢复。执行字节SHA在tmp execution-sha.json，单一窗口预算由window.json累计。
