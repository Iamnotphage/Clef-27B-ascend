abi-preflight12-rodata-dump — libccl_kernel 指定 rodata 段的最小 NUL 终止字符串
================================================================================

请求来源与性质
--------------
- 35 的最小 rodata 请求：10 个地址范围 + 14 个引用地址，只补文本，不改结论。
- 只读取字节：无设备动作、无运行、未反汇编整函数、未做全库扫描。
- 库固定：/usr/local/Ascend/cann-9.2.1/aarch64-linux/devlib/device/libccl_kernel.so
  SHA-256 = 2281deca93f4d551965ec71bfb1d23846cc9b63f60826d3d1646a96c321a686c
  （与既定 2281deca 一致；见 section-and-sha.txt，文件大小 26,274,080 字节）

地址映射（readelf -SW，非反汇编）
--------------------------------
  [13] .rodata PROGBITS 00000000014eb160 14eb160 182090
  → VMA 0x14eb160，文件偏移同为 0x14eb160（相等），大小 0x182090，末地址 0x166d1f0。
  → 10 个范围端点与 14 个引用地址全部落在该区间内（脚本断言 "all requested
    addresses inside .rodata: True"），因此按文件偏移直接读取与 VMA 一致。

取值方法（rodata-dump.txt）
---------------------------
- 对每个范围输出逐字节 hexdump（16 字节/行，ASCII 侧栏）。
- 字符串按 NUL 切分：从“包含段首的字符串”开始（若段首落在某字符串中间，该字符串
  完整输出），列出所有起始地址落在段内的字符串；跨段尾的字符串顺延到首个 NUL。
- 单列 REFERENCED ADDRESSES：对 14 个引用地址回扫至前一 NUL，输出其所属完整字符串。
- 全量输出未截断（332 行）；空字符串（填充 NUL）也保留并计入计数。

各段结果（非空字符串；地址为起始地址，len 为字节数）
----------------------------------------------------
R1 0x1613b80-0x1613d40（ReadApiValidMsg gate 不匹配，API）
  0x1613b80 len56  "[%s:%d] [%u][MC2] Check msg equal failed, rMsg:%s msg:%s"
  0x1613bc0 len15  "ReadApiValidMsg"                       （x6 引用）
  0x1613bd0 len41  "[%s:%d] [%u][%s]call trace: hcclRet -> %d"      （fmt）
  0x1613c00 len52  "[%s:%d] [%u]ErrToWarn: [%s]call trace: hcclRet -> %d"（ErrToWarn fmt）
  0x1613c38 len38  "[%s:%d] [%u]RcvMsg by task param:%d/%d"
  0x1613c60 len46  "[%s:%d] [%u]RcvMsg on msg:%p, aivId:%d, pos:%d"
  0x1613c90 len49  "basic_string: construction from null is not valid"
  0x1613cc8 len23  "aicpu_kfc_rpc_server.cc"                （共享 file 串）
  0x1613ce0 len32  "[%s:%d] [%u]reset valid value %u"       （fmt）
  0x1613d08 len9   "Addr&work"
  0x1613d18 len9   "EndNotify"
  0x1613d28 len7   "unknown"
  0x1613d30 len200 "000102...9899"（两位数十位表，原样保留）
  计数：列出 56 个串（含空串）。
R2 0x1613f00-0x1613f60（Init4/RpcSrvLaunch）
  段首 0x1613f00 落在 0x1613e30 len236 的 Msg 转储格式串内部，该串完整输出：
  0x1613e30 len236 "[%s:%d] [%u]%s Msg[commType %u, opType %u, sendBuffer %p,
                    recvBuffer %p, count %lu, strideLen %lu, hcclDataType %s,
                    valid %u, isLast %u, funID %u, sendCnt %u, rcvCnt %u,
                    everyTurnRsp %u, everyTurnWait %u, totalTurnCnt %u, winOffset %lu]"
  0x1613f20 len43  "[%s:%d] [%u]AicpuKfcRpcServer::Init addr:%u"   （fmt）
  0x1613f50 len3   "Snd"
  0x1613f58 len47  "[%s:%d] [%u]Post finishedTurnCnt[%u].cnt = %lu."
  计数：15（含空串）。
R3 0x1614040-0x16140a0（CheckRcvWorkMsg 6s 超时，TwoStage）
  0x1614010 len72  "[%s:%d] [%u]current states %s Msg %p[sendCnt:%d, valid:%d, curTurnCnt %u"
  0x1614060 len50  "[%s:%d] [%u]ErrToWarn: ReadValidMsg timeout 6s... " （fmt）
  0x1614098 len39  "[%s:%d] [%u]ReadValidMsg timeout 6s... "（非 ErrToWarn 变体）
  计数：15。
R4 0x16143e0-0x1614440（ReadAddrMsg 等待超时，API，20s 首触发）
  0x16143d8 len47  "[%s:%d] [%u]current states %s Msg %p, msgPos %u"（段首落在其中）
  0x1614408 len74  "[%s:%d] [%u][AicpuKfcRpcServer][ReadAddrMsg] ReadValidMsg[%u] timeout %lus"
  计数：2。
R5 0x14eb190-0x14eb200（RpcSrvLaunch arg0==NULL）
  0x14eb160 len49  "basic_string: construction from null is not valid"（段首在其中）
  0x14eb198 len23  "hccl_aicpu_interface.cc"
  0x14eb1b0 len36  "[%s:%d] [%u]ErrToWarn: args is null."
  0x14eb1d8 len25  "[%s:%d] [%u]args is null."
  0x14eb1f8 len59  "[%s:%d] [%u]RunAicpuKfcResInitV2 isCustom %u, context %#llx"
  计数：20。
R6 0x14ede80-0x14edf20（共享 file 串；调用点 0x7615a4/0x7620e8/0x762280）
  0x14edda0 len249 "[%s:%d] [%u][AiCpuKernel][RunTransportRoceRx]PostSendFlag Failed..."
                    （段首在其中，原样保留于 rodata-dump.txt）
  0x14edea0 len63  "[%s:%d] [%u][AiCpuKernel][RunTransportRoceRx]Kernel run success"
  0x14edee0 len7   "KfcProf"
  0x14edee8 len22  "aicpu_kfc_interface.cc"（请求指定的共享 file 串）
  0x14edf00 len60  "[%s:%d] [%u]ErrToWarn: [RunAicpuKfcClearOpRes] args is null."
  计数：12。
R7 0x14ee630-0x14ee690（调用点 0x762280；35 标注“OneStage 返回 0x16(22)”）
  0x14ee620 len42  "[%s:%d] [%u][NsRecovery][AICPU] Suspending"（段首在其中）
  0x14ee650 len40  "[%s:%d] [%u][NsRecovery][MC2] Suspending"  （fmt，按原文给出）
  0x14ee680 len86  "[%s:%d] [%u]ErrToWarn: Run rpc error, opIdx:%u, sndCnt:%d %d %d %d, rcvCnt:%d %d %d %d"
  计数：15。注：原文为 NsRecovery/MC2 Suspending，与“OneStage/22”的语境对应关系
  需 33 按原调用点判定；本目录只提供文本，不作语义推断。
R8 0x14ee700-0x14ee790（post-Init 错误/告警路径；module 未全解析，保持未知）
  0x14ee6d8 len75  "[%s:%d] [%u]Run rpc error, opIdx:%u, ..."（段首在其中）
  0x14ee728 len63  "[%s:%d] [%u]ErrToWarn: Failed to run aicpu server, MC2 opIdx:%u"
  0x14ee768 len52  "[%s:%d] [%u]Failed to run aicpu server, MC2 opIdx:%u"
  计数：7。
R9 0x14f7160-0x14f71f0（ForApi 每轮）
  0x14f7160 len51  "[%s:%d] [%u]Query sqe info failed, stream:%u, id:%u"
  0x14f7198 len9   "traceData"
  0x14f71a8 len20  "aicpu_kfc_process.cc"（file 串）
  0x14f71c0 len45  "[%s:%d] [%u]ErrToWarn: Invalid args size %zu."
  计数：19。
R10 0x14f8270-0x14f82e0（ForApi 每轮；fmt 0x14f8290，x6=[ctx+12756]）
  0x14f8240 len62  "[%s:%d] [%u]the op should not be launched in suspending status"（段首在其中）
  0x14f8280 len15  "RunRpcServerApi"
  0x14f8290 len38  "[%s:%d] [%u]start to read the [%u] msg"（fmt）
  0x14f82b8 len54  "[%s:%d] [%u]ErrToWarn: fail to get addr msg, msgPos %u"
  计数：6。

引用地址（完整所属字符串）
--------------------------
0x1613bd0 / 0x1613c00 / 0x1613ce0 / 0x1613cc8 / 0x1613bc0 / 0x1613f20 /
0x1614060 / 0x1614408 / 0x14eb1b0 / 0x14edee8 / 0x14ee650 / 0x14ee728 /
0x14f71a8 / 0x14f8290 均见上文及 rodata-dump.txt 末节，逐条为字符串起始地址。

边界与限制
----------
- 只覆盖请求的 10 段 + 14 个引用地址；未扫描其它 rodata。
- 字符串语义（模块、级别、返回码映射）不在本目录结论内；guard 参数未复核。
- 全部文本原样输出（repr 转义），未做筛选式截断。
