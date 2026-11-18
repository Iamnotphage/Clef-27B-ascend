# clef_matmul_ar_add_rms_norm_mc — 多核 M 分片包（构建归档）

**未上板，E 无设备授权**。本包只做过纯 CPU 校验与纯 CANN 编译；设备侧多核行为尚未验证。

## 身份

| 项 | 值 |
| --- | --- |
| vendor 名 | `clef_matmul_ar_add_rms_norm_mc`（与归档的 nzguard 包**不同 vendor**） |
| op 名 | `ClefMatmulArAddRmsNorm`（**与归档包同名**，两者不可同时安装） |
| 源提交 | `aacf371` |
| 构建目录 | `/workspace/yangchen/work/wave3-e-build-22-mc`（独立，非 21 共享目录） |

## 构建产物 SHA-256

| 产物 | SHA-256 |
| --- | --- |
| `custom_opp_ubuntu_aarch64.run` | `65260db799d208e3e7c7d0f6d20fb72120a91c14d24d15350d48125976249e79` |
| `ClefMatmulArAddRmsNorm_b87a0f8f….o`（kernel） | `e60aea4fad532d467d7969827c3c2458d48591b92ebfe291d12a08e303761edf` |
| `libcust_opmaster_rt2.0.so` | `290cc31e3ea593b0f11e6c1f24f583938bd8afcf59a6cadc0be470c262ceaaae` |
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

## 未证 / 需设备验证

- `SyncAll<true>()` 是否真的把某 worker 的 fixpipe/GM 写序化到另一 AIV 的 GM 读之前
  —— 头文件未给内存序契约。
- `usedCoreNum` 超过 AIC 数时，屏障参与者集合是否仍等于实际发起 matmul 的集合。
- 本点是否还需要 AIC 参与（旧注释的“AIC 已退出”因果已撤回，不再作为理由）。
- `usedCoreNum` 由预编译 `GetTiling` 决定，**不预保证等于请求预算**；也不预保证数值不变。
- 新增第 15 个属性 `mc_diag`：**adapter 调用参数个数需相应调整**（14 → 15）。

## 未做

未安装、未上板、未跑任何 NPU 任务；未改 21/24 的脚本或共享安装。
