# clef_matmul_ar_add_rms_norm_mc — 多核 M 分片包（构建归档）

**未上板，E 无设备授权**。本包只做过纯 CPU 校验与纯 CANN 编译；设备侧多核行为尚未验证。

## 身份

| 项 | 值 |
| --- | --- |
| vendor 名 | `clef_matmul_ar_add_rms_norm_mc`（与归档的 nzguard 包**不同 vendor**） |
| op 名 | `ClefMatmulArAddRmsNorm`（**与归档包同名**，两者不可同时安装） |
| 源提交 | `85b3028` |
| 构建目录 | `/workspace/yangchen/work/wave3-e-build-22-mc`（独立，非 21 共享目录） |

## 构建产物 SHA-256

| 产物 | SHA-256 |
| --- | --- |
| `custom_opp_ubuntu_aarch64.run` | `873ef8fb56df860978802584b3cbdea8ca9b2435d479a43239847abe29ee81d1` |
| `ClefMatmulArAddRmsNorm_b87a0f8f….o`（kernel） | `05cf6bd218ce15ac483a4f7ac338d1ef60edbf03d5bd71bca03c2249c12574c2` |
| `libcust_opmaster_rt2.0.so` | `1ecdb91fb2f12f85dca27c74ad64c314d839d5df90039a8377c2e54c37c5db18` |
| `libcust_opapi.so` | `9d62051f0d7da2efafd741ed68640becd9b9ea88eb8d08e07d050271d56cc227`（与 nzguard 包**逐字节相同**） |

kernel `.o` 的文件名 hash `b87a0f8f…` 与 nzguard 包相同（名字由 op 与配置派生），**内容不同**。

## 相对归档包的改动（最小差异）

1. **kernel**：按 `GetBlockIdx()` 划分 M —— `workers = usedCoreNum`，
   `startM = blockIdx * singleM`，`curSingleM = min(m - startM, singleM)`；
   调 `SetSingleShape(curSingleM, n_, k_)`，`SetTensorA(aG[startM*k_])`，
   `IterateAll(cG[startM*n_])`。**B 不动**。
2. **覆盖守卫**：若 `singleCoreM * workers < m`，改用 `ceil(m / workers)`。
   纯 CPU 扫描（`tests/slicing_cpu_check.py`，140 例）先发现了 `singleCoreM`
   偏小导致漏行的情况，守卫就是针对它。
3. **屏障**：cube 段之后 `SyncAll<true>()`（isAIVOnly 形式，
   `basic_api/kernel_operator_block_sync_intf.h:77-87`）。
4. **diag 输出与属性列表都不变**：仍 8 float、仍 **14 个属性**，
   `diag[5..7]` 保持 `usedCoreNum / baseM / depthA1`。
   （曾试图新增第 15 个属性 `mc_diag` 复用槽位，但 OpDef 未声明该属性，
   生成的 aclnn 签名里没有它，tiling 读 index 14 会越界，已**整条移除**。）
5. **host**：未指定 `mm_dim` 时改为取 `GetCoreNumAic()`（原为 Vector 计数）。
   这一条**独立于**分片修复，不构成充分修复。

## 两条口径限定（2026-11-18 23 复核后收口）

**⓪ 撤回一条错误分析（2026-11-18 23 复核）**：本包调用的是**无参** `SyncAll<true>()`，
经 `impl/basic_api/kernel_operator_block_sync_intf_impl.h:236-240` 转发到
`SyncAllImpl<isAIVOnly>()`，其 c220 实现在
`impl/basic_api/dav_c220/kernel_operator_sync_impl.h:363-403`：
`:384 PipeBarrier<PIPE_ALL>`，`isAIVOnly` 分支为
`ffts_cross_core_sync(PIPE_MTE3, GetffstMsg(0x0, SYNC_AIV_ONLY_ALL))` + `wait_flag_dev(...)` 后返回
——**硬件 FFTS 屏障，不用 GM、没有按 block 的槽位**。
我先前引用的 `:63-83` 是 `SoftSyncAllImpl`，只对应**带 gm/ub/usedCores 参数**的重载（`:217-226`），
**"两个 AIV 子块共用同一 GM 槽 ⇒ 挂死" 的判断作废**，我不判定本包屏障可用或不可用。

该实现的两个事实与两个未证：
- `isAIVOnly` 分支**没有** `ASCEND_IS_AIC` 提前返回，因此依赖 cube pass 不走到该行；
  本包中 AIC 在 `REGIST_MATMUL_OBJ` 内从 `Process()` 返回，这是它成立的前提。
- `GetBlockIdx()` 在 AIV = `get_block_idx() * GetTaskRationImpl() + get_subblockid()`
  （`dav_c220/kernel_operator_sys_var_impl.h:67-68`），ratio 来自 `get_subblockdim()`（`:43-54`）；
  `MIX_NUM` 在 `dav_c220/kfc/kfc_comm.h:93-97`：有 `__MIX_CORE_AIC_RATION__` 则取该宏，否则 2
  ——**取值取决于构建条件，未直接取得**。
- **重载选择已从源内收窄**：`impl/adv_api/detail/matmul/utils/matmul_utils.h:26-38`
  只在 `__NPU_ARCH__ == 3510` 下定义 `USE_SSBUF`（或 `KFC_C310_SSBUF==0` 时 `USE_WORKSPACE`），
  **其他 arch 在 `:36-38` 无条件定义 `USE_WORKSPACE`**。本包 target 为 ascend910_93 / `__NPU_ARCH__ 2201`，
  因此**源内默认走 `USE_WORKSPACE`**，即上文分析的 `matmul_client.h:931-977` 那个重载。
  仍缺：**外部 `-DUSE_SSBUF` 覆写是否有直接证据**（build log 无 `-D` 既不能自动当源内宏未知，也不构成无覆写的证明）。
- **"24 个 worker 落在 12 个 cube" 只是条件推导**：需 `get_subblockdim()` 在 AIV 返回 2
  **且** launch 集合完整（2×AIC 个 AIV 都在）同时成立；任一条不成立，该结论不成立。
- **未证**：`SYNC_AIV_ONLY_ALL` 是否把各 worker 的 fixpipe/GM 数据写序化到其他 AIV 的 GM 读之前；
  屏障参与者集合是否等于实际发起 matmul 的集合。

**① "切片区间独立于 AIV→AIC 映射" 是一个设计假设，不是已证事实。**
每个参与 block 只认领一段独立 M，这是**按不依赖映射**来设计的。但 2201 上下列三点
**都还没有证据**，CPU 140 例也覆盖不到：

- 实际 launch 的参与者集合是否就是 `blockIdx < usedCoreNum`；
- client/server 配对是否对**整个**参与者集合成立；
- KFC quit 集合是否与 launch 集合一致。

因此**不能从"区间设计独立"跳到"设备上必然正确"**，也不能据此直接进入设备消缺。
这三点是设备验证的**前置待定项**。

**② 单核"行为不变" 只到 CPU 层面的区间/偏移等价。**
`workers=1` 时 CPU 校验给出 `startM=0`、`curSingleM=m`，即 A/C 偏移 0、
`SetSingleShape(m,n,k)`——仅此而已。**新增的 `SetSingleShape` 调用本身**与
**`SyncAll<true>` 屏障**都**不由 CPU 覆盖证明**其在数值与生命周期上与旧路径相同。

## NZ 权重的直接依据

不切 N、不切 K ⇒ B 指针偏移为 0。生产 NZ 偏移助手的直接依据：
`vllm-ascend/csrc/gmm/grouped_matmul_swiglu_quant/op_kernel/grouped_matmul_swiglu_quant.h:222-231`
`GetWOffset(tailN,k) = tailN * AlignUp<16>(k)`（NZ 分支），`tailN = 0` 时恒为 0；
其 M 切分调用方只偏移 A 与 C（`:169-172`），仅在 N 切分时才加 `GetWOffset`（`:174`）。

## 生成 schema 证据（只读，构建产物）

`build_out/autogen/aclnn_clef_matmul_ar_add_rms_norm.h` 的
`aclnnClefMatmulArAddRmsNormGetWorkspaceSize` 参数为 4 个输入张量 + **14 个 `int64_t/double/char*` 属性**
（`eps…normFromX`，止于 `normFromX`）+ 4 个输出张量。
在本包产物里 `grep -c mcDiag` = **0**、`grep -c int64_t` = **14**：
**属性个数与归档包一致，没有第 15 个属性**，adapter 调用**无需改动**。
（`libcust_opapi.so` 与 nzguard 包逐字节相同，但这只说明**文件同字节**，
不代表符号解析选中了哪一份；两包 vendor 不同，不可同时安装。）

## 诊断设计的判据修正

- **D1/D2 目前只是设计，无设备授权。**
- **超时不能单独判定 `SyncAll` 因果**：一次 timeout 可能来自屏障、也可能来自
  server/quit 集合或 AR 段；必须与其他信号（`diag`、是否产出 `partial`、RAW_RC）合并判读。
- **固定门仍是 mixed 容差**（`1e-5 + 0.016|ref|` 全元素），不因本次讨论改变。
- "与归档单核数值差 0" 只能作为**独立诊断列**，不作为通过判据。

## 未证 / 需设备验证

- `SyncAll<true>()` 是否真的把某 worker 的 fixpipe/GM 写序化到另一 AIV 的 GM 读之前
  —— 头文件未给内存序契约。
- `usedCoreNum` 超过 AIC 数时，屏障参与者集合是否仍等于实际发起 matmul 的集合。
- 本点是否还需要 AIC 参与（旧注释的“AIC 已退出”因果已撤回，不再作为理由）。
- `usedCoreNum` 由预编译 `GetTiling` 决定，**不预保证等于请求预算**；也不预保证数值不变。
- 新增第 15 个属性 `mc_diag`：**adapter 调用参数个数需相应调整**（14 → 15）。

## 未做

未安装、未上板、未跑任何 NPU 任务；未改 21/24 的脚本或共享安装。
