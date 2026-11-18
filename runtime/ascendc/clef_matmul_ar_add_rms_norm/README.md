# clef_matmul_ar_add_rms_norm（Group E 首版）

单算子包含 **MatMul → TP AllReduce → 全 N Add + RMSNorm**，用于 down_proj + AR + AddRMS 的
融合原型。首版为串行实现，MatMul 在本算子内完成，不用外部 `torch.mm` 拼装。

## 契约

| 项 | 值 |
| --- | --- |
| 输入 | `x[M,K]` bf16、`w[K,N]` bf16、`residual[M,N]` bf16、`gamma[N]` bf16 |
| 输出 | `out[M,N]` bf16、`residual_out[M,N]` bf16、`rstd[M]` fp32、`diag[8]` fp32 |
| 属性 | `eps`(float, required)、`repeat`(int, default 1)、`group`(string, required)、`use_hccl`(int, default 1) |
| 形状 | 目标 M=967 / K=8704 / N=5120（任一合法 2D/1D 形状均可，host 侧只做一致性校验） |
| 精度 | 全 BF16 数据路径；RMS 统计在 FP32 累加 |
| workspace | `2 * M*N*2 + 128` 字节：`partial[M,N]` \| `reduced[M,N]` \| flag |

计算顺序：`partial = x @ w` → `reduced = AllReduce(partial, SUM)` → `z = reduced + residual`；
`residual_out = z`（bf16）；`rstd = rsqrt(mean(z^2) + eps)`；`out = z * rstd * gamma`。
**算法是 RMSNorm，不做均值中心化。**

## 沿用已验证的 AR 契约

AllReduce 与归一化部分直接沿用已通过的 `clef_ar_add_rms_norm` 契约：

- `InitV2(GetHcclContext<HCCL_GROUP_ID_0>(), &tiling->mc2InitTiling)`
- `SetCcTilingV2(offsetof(TilingData, mc2CcTiling))` —— offset 相对 initTiling 指针，
  因此 `mc2InitTiling` 必须是 tiling 结构体的第一个成员
- `AllReduce<true>(partial, reduced, M*N, HCCL_DATA_TYPE_BFP16, HCCL_REDUCE_SUM, repeat)`
  为 **out-of-place**（in-place 曾观测到源不变）
- `Wait()` 返回后先做整段 cache invalidate 再读 `reduced`
- 只有 core 0 写 `hcclMsgArea`，其余核在 GM flag 上自旋
- `algConfig = "AllReduce=level0:ring"`、`opType=2`、`reduceType=0`、`dtype=11`、`commEngine=2`

## 构建（纯 CANN，编译机可无 NPU）

```bash
source /usr/local/Ascend/cann-9.2.1/set_env.sh   # 提供 CMAKE_PREFIX_PATH 与 LD_LIBRARY_PATH
export MAX_JOBS=4
bash build.sh
```

产物：

- `build_out/custom_opp_ubuntu_aarch64.run`
- `build_out/op_kernel/ascendc_kernels/binary/ascend910_93/clef_matmul_ar_add_rms_norm/ClefMatmulArAddRmsNorm_<hash>.o`

### 已踩到并修掉的三类编译阻塞

1. **`find_package(ASC)` 找不到** —— 必须先 `source set_env.sh`（它把
   `aarch64-linux/lib64/cmake` 放进 `CMAKE_PREFIX_PATH`）。
2. **`LD_LIBRARY_PATH` 顺序** —— 不要手工把 `devlib` 前置，否则 `devlib/aarch64/libmetadef.so`
   会盖住 `lib64` 版本并报 `undefined symbol: ge::AscendString::AscendString(char const*)`。
   `set_env.sh` 的默认顺序（lib64 先、devlib 追加）是正确的。
3. **macOS `._*` AppleDouble 文件** —— 它们以 `.cpp` 结尾，会被
   `aux_source_directory` 收进 `ops_srcs` 并当作 C++ 编译，报一堆 `stray '\5' in program`。
   从 macOS 拷贝源码后必须 `find . -name "._*" -delete`。

### 已修掉的四处内核编译错误

- `GM_ADDR` 成员必须**每个一行**声明（宏展开为 `__gm__ uint8_t*`，同一行的后续声明符会报
  `field may not be qualified with an address space`）。
- `Matmul` 对象没有 `SetSubBlockIdx`（该接口在别的封装上），删掉。
- `Matmul::Init` 只接受**非 const** `TCubeTiling*`，所以 tiling 指针全程不能是 `const`。
- aicore 内不允许 `static_cast<float>(uint32_t)`，diag 不再写 m/n/k。

## 属性

| 属性 | 语义 |
| --- | --- |
| `eps` | RMSNorm epsilon |
| `repeat` | AllReduce 的 repeat（1..255） |
| `group` | 真实 HCCL comm 名（opdef 已 `MC2().HcclGroup("group")`） |
| `use_hccl` | **三态**：`0` = 整体跳过 HCCL 段；`1` = 全跑（生命周期 + AllReduce/Wait）；`2` = **只跑生命周期**（InitV2 / SetCcTilingV2 / Finalize，不发布 collective） |
| `block_dim` | 核数：`0` = 自动取 AIV 数；`-1` = 取 AIC 数；`>0` = 显式。**只影响 `SetBlockDim`** |
| `mm_enable` | `1` = 跑 MatMul；`0` = **整段跳过 cube**。这是把故障定位到 matmul 的决定性开关 |
| `mm_dim` | 只驱动 `MultiCoreMatmulTiling::SetDim`（即 `TCubeTiling.usedCoreNum`），`0` = 跟随 `block_dim`，`-1` = AIC 数，`>0` = 显式 |
| `mm_mode` | `2` = 完整（默认）；`1` = **停在 Init/SetOrgShape/SetTensorA/B 之后，不调 `IterateAll`** |
| `mm_bufspace` | `0` = 用 tiling 自带的缓冲模型；`1` = 把 L1/L0C 约束到平台真实尺寸；`2` = 再为归一化段预留 128 KB UB |
| `norm_ub_late` | `1`（默认）= 归一化段约 120 KB 的 UB **推迟到 cube 阶段之后**才分配；`0` = 恢复在 `Init()` 里先分配 |
| `mm_btrans` | `1` = B 物理上是列主序（`isTransposeB`）；**host tiling 的 `SetBType(isTrans)` 与 kernel 的 `SetTensorB(isTrans)` 由同一个 attr 驱动**，不会打架 |

### 已由设备侧定性的两条（21 实测）

1. **KFC 生命周期假设成立**：`mm_enable=0 + use_hccl=2 + block_dim=-1` 两 rank 正常完成
   （RAW_RC=0，diag = [0/1, 2, 0, 2, 0, usedCoreNum 24]）。保留
   `InitV2/SetCcTilingV2/Finalize`、不发布 collective 可以正常结束；此前 `use_hccl=0`
   的挂起是**整段跳过生命周期**造成的，无效 tiling 的 `param invalid` 是另一回事。
   **后续矩阵一律用 `use_hccl=2` 作"无 AR"对照臂，不要用 0。**
2. **越界故障在 matmul 路径内**，不在向量/AddRMS/标量代码。

### matmul 内部二分（下一步）

| # | `mm_mode` | `mm_bufspace` | `mm_dim` | 覆盖 | 判读 |
| ---: | ---: | ---: | ---: | --- | --- |
| 1 | 1 | 0 | 0 | 仅 Init + 绑张量，不 Iterate | 干净 → tiling/init 无问题，故障在 `IterateAll` |
| 2 | 2 | 1 | 0 | 完整 + L1/L0C 约束到真实值 | 干净 → tiling 高估了内部缓冲 |
| 3 | 2 | 2 | 0 | 完整 + 为归一化预留 128 KB UB | 干净 → UB 预算冲突 |
| 4 | 2 | 0 | 1 | 单核 cube | 干净 → 与多核切分有关 |

### 已由设备侧定性的第三、四条

3. `mm_dim=1`（单核）**仍崩**、`blk=0` → 与多核切分无关。
4. `mm_mode=1`（停在 Init/SetOrgShape/SetTensorA/B，从不 IterateAll）**仍崩**、`blk=11`
   → 故障在 **matmul 对象初始化/形状设置阶段**，不在 `IterateAll` 的流式。

因此 `diag[5..7]`（usedCoreNum / baseM / depthA1）改为**在 cube 阶段之前写出**，
`mm_mode=1` 崩溃时也能拿到 tiling 数值（也可以用 `mm_enable=0` 直接读）。

### 当前主假设

`Init()` 里先把归一化段约 120 KB 的 UB 分配掉了，等 `mm_.Init()` 再去取自己的 buffer 时
池子已经不够 —— 这与"故障发生在 Init、且与核数无关"完全吻合。`norm_ub_late` 默认改为 `1`
把归一化段的分配推到 cube 之后，所以**默认配置下一次运行就直接检验这个假设**。

（历史记录：`E v1 (ND, b61e2836)` 与 `block_dim=-1` 复测都是同一故障
`fftsplus aivector error`、`blk=22`、`subErrType:4`、
"The address for the scalar to access the internal buffer of AICore is out of bounds"，
与 block 数无关，故 `block_dim` 扫描已放弃。）


kernel 用 `usedCoreNum` 做门控（只有 `GetBlockIdx() < usedCoreNum` 的核进 matmul），所以
`mm_dim` 与 `block_dim` 解耦是安全的，不会让未分配 cube 内部缓冲的核进入 matmul。

`diag[0..5]` = rankId / rankDim / arRc / hcclMode / mmRc / **cubeTiling.usedCoreNum**
（最后一槽用来核对 tiling 是否真的按 `mm_dim` 设了 `usedCoreNum`）。

## 上板首轮结果与当前诊断状态

`E v1 (ND, b61e2836)` 上板：两 rank 在 synchronization 崩溃，device 报
`fftsplus aivector error, core id 26`，extend info 为
**"The address for the scalar to access the internal buffer of AICore is out of bounds"**
（errcode 0x200000000000000, subErrType:4）。**不是 KFC 参数错误**。

`block_dim=-1`（AIC 数）复测：**同一故障**，`core id 34/26`、`blk=22`、`subErrType:4`，
与 `block_dim=0` 时相同。因此 **"AIV/AIC 核数错配"这一假设已被否证**，
故障在 kernel 内部标量寻址，与 block 数取值无关。

### 二分矩阵（每次只动一个变量）

| # | `mm_enable` | `use_hccl` | 覆盖的路径 | 判读 |
| ---: | ---: | ---: | --- | --- |
| 1 | 0 | 0 | 仅向量 AddRMS（无 cube、无 HCCL） | 若仍崩 → 故障在我的向量/标量代码；若不崩 → 指向 matmul |
| 2 | 1 | 0 | MatMul + AddRMS | 若 1 通过而 2 崩 → matmul 段确认 |
| 3 | 1 | 2 | MatMul + 仅 HCCL 生命周期 | 判 KFC 生命周期配对假设 |
| 4 | 1 (`mm_dim=1`) | 0 | matmul 只切 1 核 | 判是否与多核切分有关 |

单看第 1 行就能定性：它完全没有 cube 侧代码。`diag[5]` 与 `GetBlockNum()` 的对比
用来确认 tiling 是否按 `mm_dim` 生效。


## W 布局：最终契约是 FRACTAL_NZ（本诊断版暂为 ND）

服务里的 down-projection 权重**已经是 FRACTAL_NZ**（`CLEF_MLP_BF16_NZ` 在加载期做一次
`npu_format_cast(..., 29)`；隔离实测该形状 matmul 约 0.2963 ms）。因此最终契约是
"**输入即 NZ**"，`BT = MatmulType<GM, CubeFormat::NZ, ...>`，**op 内不做权重转换**，
预 pack 归 host / 模型加载侧。为把 block-dim 问题单独隔离，当前提交暂用 ND；
`block_dim` 定案后切回 NZ（`d4a4c43` 的 NZ 版本可作参照）。
NZ 要求 N 按 16 元素对齐，N=5120 与 K=8704 均满足。

## 待与 21 确认并记录（v1 未决）

1. **block dim 与 matmul 切分**：见上，正在按 `block_dim` 扫描定位阈值。
2. **数值正确性完全未验证**：AR 可见性、rstd 语义、与
   `npu_add_rms_norm(dist.all_reduce(partial))` 的逐元素比对都需 21 在设备上做。
3. **matmul 写出的 `partial` 对 AICPU 侧的可见性未验证**（目前只有 `SyncAll()` +
   `PipeBarrier<PIPE_ALL>`）。
4. **host 注入路径当前仍被 KFC 参数/生命周期问题阻塞**：本算子声明了
   `MC2().HcclGroup("group")`，走与 `clef_ar_add_rms_norm` 相同的注入路径。
   `use_hccl=2` 就是为验证"只保留生命周期、不发布 collective"这一假设而加的。


## 沿用的已验证细节（不可回退）

除上面的 AR 契约外，归一化段严格保持 21 已验证的写法：

- `gamma` 的 `DataCopy` 与 `Cast` 之间**必须有 `PipeBarrier`**；
- `rstd` 按 **8-float 对齐批次**写出；
- `ReduceSum` 的 work 缓冲 ≥ 2KB，且 dst 与 work 相互独立；
- `residual_out` 写的是 `z = reduced + residual`（服务要接着传的那条流）。


## 集成步骤（交给 21）

1. 装包：把 `custom_opp_ubuntu_aarch64.run` 装到独立 vendor，**前插**
   `ASCEND_CUSTOM_OPP_PATH`，并把 `op_api/lib` 补进 `LD_LIBRARY_PATH`。
2. 调用：`aclnnClefMatmulArAddRmsNorm(x, w, residual, gamma, eps, repeat, group, use_hccl)`，
   返回 `out / residual_out / rstd / diag`。
3. `group` 必须是真实 HCCL comm 名（opdef 里 `MC2().HcclGroup("group")` 已绑定）。
4. `diag[0..4]` 依次是 rankId / rankDim / `arRc`（`Wait*100+setRc`）/ useHccl / mmRc，
   先用它判 AR 是否真的跑起来。
