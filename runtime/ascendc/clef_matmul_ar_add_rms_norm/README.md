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

## 待与 21 确认并记录（v1 未决）

1. **W 的布局**：v1 用 **ND**（`w[K,N]` 行主序，`CubeFormat::ND`）。真实服务中 MLP 权重是
   FRACTAL_NZ（`CLEF_MLP_BF16_NZ=1`），若要直接吃已 NZ 的权重，需把 `BT` 改成
   `CubeFormat::NZ` 并确认内轴/外轴顺序与对齐。
2. **预 pack 边界**：权重转换在 host 侧一次完成，还是由外部传入已 NZ 的权重。
3. **block dim 与 matmul 切分**：当前 `SetBlockDim(GetCoreNumAiv())` 且
   `MultiCoreMatmulTiling::SetDim(agent cores)` 取同一数值。Cube:Vector = 1:2，
   该口径下的 matmul 切分与 sub-block 语义**未在设备侧验证**。
4. **数值正确性完全未验证** —— 本次只做到「纯编译通过」。AR 可见性、rstd 语义、
   与 `npu_add_rms_norm(dist.all_reduce(partial))` 的逐元素比对都需 21 在设备上做。
5. `SyncAll()` + `PipeBarrier<PIPE_ALL>` 只保证核内顺序，matmul 写出的 `partial` 对
   AICPU 侧的可见性未验证。

## 集成步骤（交给 21）

1. 装包：把 `custom_opp_ubuntu_aarch64.run` 装到独立 vendor，**前插**
   `ASCEND_CUSTOM_OPP_PATH`，并把 `op_api/lib` 补进 `LD_LIBRARY_PATH`。
2. 调用：`aclnnClefMatmulArAddRmsNorm(x, w, residual, gamma, eps, repeat, group, use_hccl)`，
   返回 `out / residual_out / rstd / diag`。
3. `group` 必须是真实 HCCL comm 名（opdef 里 `MC2().HcclGroup("group")` 已绑定）。
4. `diag[0..4]` 依次是 rankId / rankDim / `arRc`（`Wait*100+setRc`）/ useHccl / mmRc，
   先用它判 AR 是否真的跑起来。
