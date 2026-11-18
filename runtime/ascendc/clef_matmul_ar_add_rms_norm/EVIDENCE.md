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

## 6. 多核切片：三条断言的更正（2026-11-18，只读源码）

本节更正此前关于"`IterateAll` 自动跨核分片"的表述。**该说法不成立**，下面每条都给同机
源码路径与行号。

### 6.1 撤回：「IterateAll 自动按 `GetBlockIdx()` 完成跨核 A/B/C 分片」
- `grep -rl GetBlockIdx` 在 `asc/impl` 全树命中 **31 个文件**，**matmul 目录 0 处**。
- matmul server 只认 **`subBlockID`（AIC/AIV 子块 0/1）**，不是核号：
  `asc/impl/adv_api/detail/matmul/kfc/matmul_server.h:57`（`mul.SetSubBlockIdx(kfcCommSrv->subBlockID)`）、
  同文件 `:69`、`:74`、`:102`。
- `singleCoreM/N` 在 `asc/impl/adv_api/detail/matmul/matmul_impl.h` 里**不出现**。

→ 跨 AIC 的分片**不在随包源码内**，不能据此断言自动化。

### 6.2 否证：「`SetTail` 只属 Iterate/UB 变体」
- `asc/include/adv_api/matmul/matmul_config.h:167`：
  `bool enableSetTail = true;  // true: Call SetTail during Matmul computation (default)`；
  `adv_api/matmul/tiling.h:73`、`:155` 的默认配置同样是 `true`。
- `adv_api/matmul/matmul_client.h:234-245`：`SetSingleShape(singleM,singleN,singleK)`
  的实现**就是** `SetTail(singleM, singleN, singleK)`。
- `matmul_client.h:246-259`：`SetTail` 写 `mIter_ = Ceil(tailM, baseM)`、
  `nIter_ = Ceil(tailN, baseN)`，**直接决定每核迭代次数**。

→ `SetSingleShape`/`SetTail` 是**多核必需的每核形状入口**，不是可选项。

### 6.3 只部分有依据：「所有 block 注册在多 usedCore 格无需变化」
- 有依据：`asc/impl/adv_api/detail/kfc/kernel_kfc.h:60-63`
  `isRun(){ return quitSize < MIX_NUM; }`，同文件 `:54-55`、`:123-128`、`:199-218` 的 quit 计数，
  `kfc_register_obj.h:210/250/297/321` 的 `while (server.isRun())`
  —— **每个参与的 client 必须投 quit**，否则 server 不退出。
- **没有依据**：`MIX_NUM` 与 launch block 数的具体对应，以及 `usedCoreNum < blockNum`
  时是否仍成立。**不宣称无需变化。**

### 6.4 正面证据：同栈生产算子的多核写法
`/workspace/yangchen/work/src/vllm-ascend/csrc/gmm/grouped_matmul_swiglu_quant/op_kernel/grouped_matmul_swiglu_quant.h`
```
110:  blockIdx = GetBlockIdx();
179:  mm.SetOrgShape(mnConfig.m, mnConfig.n, mnConfig.k);
180:  mm.SetSingleShape(curSingleM, curSingleN, mnConfig.k);
183:  mm.template IterateAll<sync>(mmOutGM[mnConfig.workSpaceOffset], 0);
```
`.../grouped_matmul_swiglu_quant_spilit_fusion.h`：`:119-120` 同样
`SetOrgShape`+`SetSingleShape`；`:165` `currentBlockId = GetBlockIdx()`；
`:338` `if (GetBlockIdx() >= realCoreDim)`；`:374` 尾核 `blockDimxTailFactor`；
`:379` `coreDimxOffset = blockDimxFactor * GetBlockIdx()`。

→ **多核是调用方自己算切片**：`GetBlockIdx()` 定 M 区间、显式 `SetSingleShape`、
C 写到每核偏移。

### 6.5 本 kernel 的确切缺口与最小修复（未实施）
`op_kernel/clef_matmul_ar_add_rms_norm.cpp:187-204` 只有 `SetOrgShape(m_,n_,k_)` +
整张 `aG/bG/cG` + `IterateAll`；**无 `GetBlockIdx` 切片、无 `SetSingleShape`、无 C 偏移**。
→ **`mm_dim > 1` 当前不可运行**（各核重复计算同一块）。

最小修复（四处，需先协调包身份）：
1. `blockIdx = GetBlockIdx()`；由 `cubeTiling.singleCoreM/singleCoreN` 推
   `curSingleM/curSingleN` 与 `startM`，末核取尾；
2. `mm_.SetSingleShape(curSingleM, curSingleN, k_)`（`SetOrgShape` 保持全张量）；
3. `aG` 偏移 `startM*k_`、`cG` 指向 `partialGm_ + startM*n_`；
4. `diag` 增 `singleCoreM/singleCoreN/curSingleM` 三槽便于验证。

### 6.6 相关口径
- `mm_dim=1` **不能**验证多核 offset——1 核不触发跨核分片。
- `SetDim` 是**预算**；`usedCoreNum=24` **不能预先保证**。
  `MultiCoreMatmulTiling::GetTiling` 在 `adv_api/matmul/matmul_tiling.h:44/51` **只有声明**，
  实现是预编译库 → **无法从源码判断** Dim 变化是否改变 `baseK`/`step`/累加顺序，
  因此"数值必不变"不作预测。
- `diag[5..7]` 是 `usedCoreNum/baseM/depthA1`；**`diag[7]` 不是 format 码**
  （`op_kernel:114`、`:135` 两处仍写 `depthA1`）。
