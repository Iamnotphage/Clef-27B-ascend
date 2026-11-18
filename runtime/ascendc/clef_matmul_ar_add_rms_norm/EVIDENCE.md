# E 诊断证据与契约状态（只读核对，非根因结论）

本文件记录 Group E 当前包在 CANN 9.2.1 / ascend910_93 上的**编译级**证据与契约落点。
它**不是**根因结论：设备侧是否因此消除越界，需由 21 的运行对照判定。

## 1. 已从源码/编译分支确认的事实

kernel 被 ASC 编译器按 pass 编译两次，`AscendC::Matmul` 的别名在两趟里不同。以下两条
是**编译器自己的报错**，不是推断：

| 编译条件 | 报错对象 | 说明 |
| --- | --- | --- |
| 默认（未定义 `ASCENDC_CUBE_ONLY`） | `AscendC::MatmulServiceAux<...>` | 报错内容为 `no member named 'SetSubBlockIdx'`；该别名只在 `SPLIT_CORE_CUBE && !ASCENDC_CUBE_ONLY` 分支出现 |
| 加 `-DASCENDC_CUBE_ONLY=1` | `AscendC::MatmulClient<...>` | 同样报 `SetSubBlockIdx` 缺失，但对象变成 Client；说明报错来自 **AIV pass**，两趟各自选类 |

`REGIST_CUBE_OBJ`（= `REGIST_MATMUL_OBJ`）在非 3510 分支的展开为
`KfcServer server; server.Init(workspace); server.InitObj(tpipe, ...); while (server.isRun()) server.Run(...); return;`
—— 它是 **cube pass 的主体与 server 循环**，不是一条注册语句。
`matmul_client.h:117-146` 中非 `USE_SSBUF` 的 `Init` 会立即 `client->AllocUB`、写 tilingGM、
`PostMessage`，因此 **client 必须在 `Init` 之前接线**。

**由此得到的陈述（限定版）**：原 mixed 路径缺少 client 接线，这一点已由源码与编译分支确认；
修复（改用官方 `REGIST_MATMUL_OBJ`）已编译通过。**该缺陷是否解释并消除当前设备越界，待运行对照。**

曾经观察到"`mm_dim=1` 也崩、`mm_step=1` 只 Init 也崩、`norm_ub_late=1` 也崩"——这些只说明
故障在这些配置下均复现，**不构成**"与核数/UB 池/shape 无关"或"根因已闭合"的结论。

## 2. 两个必要契约点的落点

### (a) 宏所建局部 client 与后续调用同一存活作用域

`op_kernel/clef_matmul_ar_add_rms_norm.cpp` 中 `REGIST_MATMUL_OBJ(...)` 与随后的
`SetOrgShape` / `SetTensorA` / `SetTensorB` / `IterateAll` 位于**同一个复合语句**
（同一个 `if` 块）内。宏在 cube pass 声明 `KfcServer`、在 vector pass 声明 `KfcCommClient`
都是该作用域的局部对象，因此其寿命覆盖后续全部 MatMul 调用。

### (b) 系统 workspace 与用户 workspace 按 2201 SDK 边界分开

- host：`context->GetWorkspaceSizes(1)[0] = platform.GetLibApiWorkSpaceSize() + 用户区`，
  用户区 = `partial[M,N] | reduced[M,N] | flag`；
- kernel：自己的缓冲区一律取自 `AscendC::GetUserWorkspace(workspace)`，**不触碰它前面的系统区**；
- `REGIST_MATMUL_OBJ` 的 workspace 参数传 `GetSysWorkSpacePtr()`，与用户区互不重叠。

判定不依赖"指针非空"或"用户 buffer 足够大"，而是依赖这两块区域被显式分开。

## 3. 本地 tiling 的来源与寿命

- 来源：入口函数内 `GET_TILING_DATA(tilingData, tilingGm)` 复制出的**普通本地对象**；
- 传递：`&tiling_->cubeTiling` 为普通指针，**未做任何地址空间强转**；
- 寿命：`tilingData` 是入口函数的局部对象，覆盖 kernel 全程，满足
  `matmul_constant_tiling_struct.h:24-34` 中 `SetTiling` 保存指针的要求；
- 未使用 `__gm__ TCubeTiling*` 的 `Init` 重载——该重载的 `CopyTiling` 受
  `matmul_utils.h:424-435` 的 `static_assert(IsSameTypeV<CFG_TYPE_TYPE, MatmulApiStaticTiling>)`
  限制，只对 static/partial-const 配置成立。

## 4. ND 三处一致性（当前诊断版）

真实连续 ND 的 `B[K,N]` 在非转置路径上，下列三处必须同时为 ND 且 `isTrans == false`，
本包一致：

- host `SetBType(GM, CubeFormat::ND, DT_BFLOAT16, false)`
- kernel `MatmulType<TPosition::GM, CubeFormat::ND, bfloat16_t, false>`
- kernel `SetTensorB(bG, false)`（`mm_btrans=0` 时）

NZ 是物理格式，不能靠改一个 bool 或逻辑 shape 把 ND 解释成 NZ。真实 NZ 契约在
`README.md` 中记录，待正确性通过后启用。

## 5. 下一步的定位方式

若本轮运行仍失败，沿**混合 client/server 生命周期**继续定位（接线时机、作用域、系统区
大小、`Init` 与 `REGIST` 的先后与配对），不再回扫已排除的参数（核数、UB 池大小、
`SetOrgShape` 参数顺序、transpose 布尔）。
