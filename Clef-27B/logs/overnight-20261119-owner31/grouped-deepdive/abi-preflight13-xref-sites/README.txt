abi-preflight13-xref-sites — 三个 rodata 字符串的地址交叉引用（只读）
=====================================================================

请求（33→31）
-------------
1) 0x1613f58 `Post finishedTurnCnt[%u].cnt = %lu.` 的引用 site；
2) 0x14ee680 `Run rpc error, opIdx:%u, ...` 的引用 site(s)；
3) 0x14f7160 `Query sqe info failed...` 的真实 site（35 标 0x7b3a24，但同址另有 #10）。

方法（只读；无设备/无执行/不重复 rodata dump）
--------------------------------------------
- 库固定：/usr/local/Ascend/cann-9.2.1/aarch64-linux/devlib/device/libccl_kernel.so
  SHA-256 = 2281deca93f4d551965ec71bfb1d23846cc9b63f60826d3d1646a96c321a686c。
- .text：VMA=文件偏移=0x29c320，大小 0x124ee24，共 4,799,369 条指令；ADRP 站点 289,738。
- 对每个 ADRP 起跟踪其 Rd 寄存器值，向前最多 256 条指令：ADD imm64(sh=0) 且 Rn=该寄存器
  时计算新值；新值命中目标即记站点；Rd=该寄存器时更新跟踪值；其它指令若 Rd 位等于该
  寄存器则停止跟踪（保守）。最终结果按站点去重。
- 正例校准（已知站点全部命中）：0x1613f20→0x7d67bc；0x1614060→0x7d7440；
  0x1614408→0x7d8d14；0x1613bd0→0x7d6fe0（另有 3 处合法引用）；0x14ee650→
  0x762270/0x7627c4；0x14f8290→0x7b3a14。
- 全量输出 xref-final-scan.txt（48 行），站点窗口 site-windows.txt（5 段最小窗口）。

结论
----
1) 0x1613f58：唯一站点 **0x7d6970**
   - 0x7d6958 `adrp x2, 0x1613000`；0x7d695c `add x2,x2,#0xc90`（基址 0x1613c90）；
     0x7d6970 `add x2,x2,#0x2c8` → 0x1613f58。
   - 所属函数 `_ZN17AicpuKfcRpcServer26WriteFinishWhenAllFinalizeEj`
     （objdump `+0x54` 区段）。
   - 调用：0x7d6984 `bl DlogRecord`；原始参数 w0=3、w1=1、w4=0x4a（line 74）；
     x2=fmt(0x1613f58)、x3=0x1613cc8（aicpu_kfc_rpc_server.cc）、x5=tid、x7=[x7+8]。
2) 0x14ee680：唯一站点 **0x76205c**
   - 0x762058 `adrp x2, 0x14ee000`；0x76205c `add x2,x2,#0x680`。
   - 所属函数 `RunAicpuRpcSrvLaunch`（objdump `+0xcdc`）。
   - 调用：0x762098 `bl DlogRecord`；原始参数 w0=0x01000003、w1=2、w4=0x21b（line 539）；
     x2=fmt、x3=0x14edee8（aicpu_kfc_interface.cc）。
3) 0x14f7160：唯一站点 **0x7aa1f4**
   - 0x7aa1ec `adrp x2, 0x14f7000`；0x7aa1f4 `add x2,x2,#0x160`。
   - 所属函数 `_ZN13MC2TraceUtils18SubmitBatchSqeInfoEv`（objdump `+0x1b4`）。
   - 调用：0x7aa204 `bl DlogRecord`；原始参数 w0=3、w1=2、w4=0x2ea（line 746）；
     x2=fmt、x3=0x14f5b68。
   - **0x7b3a24 不是本串站点**：该地址是另一条 fmt 0x14f8290（`start to read the [%u]
     msg`）的 Dlog 调用点；其实参 fmt 由 0x7b3a0c `adrp x2,0x14f8000` +
     0x7b3a14 `add x2,x2,#0x290` 装入，x3=0x14f71a8（aicpu_kfc_process.cc），
     line 1715。35 表中 #9 与 #10 同址的标注应拆开：#9 的真实站点为 0x7aa1f4。

被排除的旧候选（rejected-backward-only-candidates.txt）
-------------------------------------------------------
- 仅做“ADD 回扫最近 ADRP”的旧扫描（scan2）对 0x14f7160 报过 9 个候选，其中 8 个在
  寄存器存活检查下不成立。例：0x105d79c `add x0,x0,#0x160` 的回扫命中了更早的
  0x105d6fc `adrp x0,0x14f7000`，但两者之间 x0 已被覆盖；该 add 实际由
  0x105d798 `adrp x0,0x14eb000` 提供基址，得到 0x14eb160，与 0x14f7160 无关。
- 只保留通过存活检查的站点，即上表三处。

边界
----
- 覆盖模型为 ADRP(+ADD 链) 的地址物化；未检索 movz/movk 立即数拼接。
- 全文件 8 字节指针出现次数为 0（scan1），LDR-literal 解码 0 命中（mask 0xFF000000==
  0x58000000）；因此未发现数据指针表形式。
- 跟踪在“任意指令 Rd 位等于跟踪寄存器”时保守停止，故“唯一站点”是本模型与已审窗口
  内的结论，不是形式化全库证明。
- 本目录只读；无设备动作，服务保持 aa3e/s1（root 1337814）。
