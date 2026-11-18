# ClefGdnQkvNorm — GDN 前处理融合原型

把 GDN prefill 的卷积输出 `[40, T, 128]`（head-first，已含 SiLU）一次写成连续的
BSND `q`/`k` `[1, T, 8, 128]`（Q/K 做 head_dim=128 的 L2Norm）与 `v`
`[1, T, 24, 128]`，替代服务路径上的 3 次连续拷贝和 2 次 `l2norm_fwd`。

## 版本

- 目标：Ascend910_9382（A3，`ascend910_93`，24 AI cores）
- CANN：9.2.1
- msopgen：随 CANN 9.2.1（`mindstudio` 模板）
- op：`ClefGdnQkvNorm`，输入 bf16 `x [40,T,128]`，输出 bf16
  `q/k [1,T,8,128]`、`v [1,T,24,128]`，attr `eps: float`
- 打包：`custom_opp_ubuntu_aarch64.run`（`vendors/customize`）

## 构建

```bash
source /workspace/yangchen/Clef-27B-ascend/scripts/activate_env.sh
msopgen gen -i clef_gdn_qkv_norm.json -c ai_core-Ascend910_93 -f aclnn -lan cpp \
  -out ClefGdnQkvNorm
# 把本目录的 op_host/op_kernel 源文件覆盖到生成项目后：
cd ClefGdnQkvNorm
bash build.sh
# 产物：build_out/custom_opp_ubuntu_aarch64.run
./build_out/custom_opp_ubuntu_aarch64.run --noexec \
  --extract=build_out/install
# vendor：build_out/install/packages/vendors/customize
```

`build.sh` 的 `ASCEND_AICPU_PATH`/`CCEC` 由 `activate_env.sh` 提供；`-c
ai_core-Ascend910_93` 与 `AddConfig("ascend910_93")` 必须一致。

## 调用

`runtime/clef_qkv_norm_probe.py` 通过 FLA 的通用 ctypes runtime 加载
`libcust_opapi.so`，调用生成的 `aclnnClefGdnQkvNorm`：

```bash
CLEF_QKV_NORM_VENDOR=<vendor> \
  python runtime/clef_qkv_norm_probe.py --output <run>/fusion-probe.json
```

服务集成需要在 `DeviceOperator.fla_gdn_prefill` 的 `preprocess_fusion` 分支加载
该 vendor；当前默认关闭，提交配置保持 BIC（无自定义算子依赖）。

## 核设计

- 每个 tile 处理一个 head 的 `tileTokens=64` 个 token，输入连续、输出按
  `(outHeads-1)*128*2` 字节跨步写到 BSND。
- Q/K tile：bf16→fp32，平方写入独立 fp32 缓冲；逐行两次 Level 2
  `ReduceSum(..., 64)` 相加，`+eps`、`Rsqrt`，逐行 `Muls`，fp32→bf16
  （`CAST_RINT`）。Level 1 mask 版本不可用，见下。
- V tile：同一 cast 路径，不归一化；bf16 往返逐位保持。
- 24 个 block 按连续 tile 编号分配；`tilesPerCore=ceil(40*ceil(T/64)/24)`。
- UB：输入/输出各 16 KiB、fp32 2×32 KiB（值/平方）、reduce 1 KiB、
  ReduceSum 临时 16 KiB。

## 验证结果（2026-11-15）

NPU 0，torch_npu 2.10.0.post7，`runtime/clef_qkv_norm_probe.py --iters 20`：

| tokens | fused (ms) | 现有单拷贝路径 (ms) | 加速 | Q/K 最大差 | V |
| ---: | ---: | ---: | ---: | ---: | --- |
| 307 | 0.1217 | 0.2444 | **2.01×** | 0.001953（1 bf16 ulp） | 逐位 |
| 401 | 0.1232 | 0.2536 | **2.06×** | 1 ulp | 逐位 |
| 512 | 0.1255 | 0.2538 | **2.02×** | 1 ulp | 逐位 |
| 1595 | 0.3054 | 0.2613 | 0.86× | 1 ulp | 逐位 |
| 10294 | 1.9556 | 0.2608 | 0.13× | 1 ulp | 逐位 |

**结论：不接入服务。** 短序列（≤512 token）每层前处理约省 0.12 ms，按 48 层
约 5.8 ms/请求；但真实 S1 分布 p95 969 token、最大 1595，ERQA 可达 10,294，
融合算子在长序列上反而慢 1.2–7.5×。根因是当前可用的 Level 2 逐行
`ReduceSum`（每行两次）与逐行标量 `Muls` 随 token 线性放大；Level 1
`ReduceSum` 的 int mask 触发 UB 对齐异常，`uint64_t mask[2]` 版本返回 NaN，
不能用。下一步需要与 Triton `tl.sum` 同序的 tile 级归约和广播缩放；在此之前
BIC 是更优的提交路径。

- Q/K 的 1 ulp 差异来自归约顺序（64+64 两半相加 vs Triton 单次 128 归约），
  V 的 bf16 往返逐位保持。
- 长序列 10,294 token 覆盖 16 图 ERQA 的 token 上限。
- 构建身份：`libcust_opapi.so` SHA-256 `4d2440a3…`；kernel 对象
  `d1687581…`；`custom_opp_compiler_version=9.2.1`。

## v2（2026-11-16，已接入）

v1 的逐行两次 `ReduceSum(64)` 和逐行标量 `Muls` 改为：

- 每行一次 `ReduceSum(128)`；
- `Broadcast<float,2,1>` 生成 `[rows,128]` 的缩放向量；
- 一次 `Mul(xf, xf, scales, rows*128)`。

单算子：307 2.00×、969 2.01×、1024 2.01×、1595 **1.65×**；969 p95 0.2098 ms vs baseline 0.3244 ms（1.55×）。服务端同窗口 v2 vs v1 p95 **−2.48%**，cap 扩到 **1600**（覆盖 OD 1595 和 ERQA 1200–1595）。Q/K 仍 1 ulp、V 逐位；质量 2PP。详见 `docs/gdn-kernel-v2-2026-11-16.md`。
