# D0 可运行身份（交 21 上板；未安装、未加载、未执行）

## 1) 包与产物 SHA-256
| 产物 | SHA-256 |
| --- | --- |
| `custom_opp_ubuntu_aarch64.run`（**包 b7190e96**） | `b7190e96c6997b460245becc8e4d34fc5d0c33919914762b65223f94ff7fe62c` |
| kernel `.o`（`ClefD0Trace_7cc2cb49….o`） | `5cb2242c94ddcdd819f445f039e000bd30517efa6eb9b4faabd2e2de214ddd20` |
| `libcust_opmaster_rt2.0.so` | `4f4f9fbf8fd4b44ef846245b7a7b10c8e5550f03ac1972eb3655add7c83b1271` |
| `libcust_opapi.so` | `3314922677b7b78164628f1bc22c7339ac63d59cf537c261533cbf740ac948ec` |

## 2) 隔离 vendor 与 loader
- vendor：**`clef_d0_trace_22`**（与 E 包 `clef_matmul_ar_add_rms_norm`、`..._mc` **不同 vendor**，可共存）
- op：`ClefD0Trace`；kernel 入口符号：`clef_d0_trace`；生成头：`aclnn_clef_d0_trace.h`
- 安装：把包 vendor 目录放到独立 `ASCEND_CUSTOM_OPP_PATH`，并**前插** `…/vendors/clef_d0_trace_22/op_api/lib` 到 `LD_LIBRARY_PATH`
- **不要**卸载/覆盖 E 包 vendor

## 3) ABI（生成头实核）
`aclnnClefD0TraceGetWorkspaceSize(...)`：
**4 输入张量**（`x` bf16/ND、`w` bf16/**FRACTAL_NZ**、`residual` bf16/ND、`gamma` bf16/ND）
**+ 14 个属性**（`eps, repeat, group, use_hccl, block_dim, mm_enable, mm_dim, mm_mode, mm_bufspace, norm_ub_late, mm_btrans, mm_stage, norm_from_residual, norm_from_x`）
**+ 5 输出张量**（`out` bf16、`residual_out` bf16、`rstd` fp32、`diag` fp32 **8 float**、**`trace` uint32 ND `[72,8]` = 2304 B**）

**trace 清零由调用侧在同一 stream 于调用前完成**；kernel 只写自己的槽。

## 4) 槽式与字段（只报容量内已观测集合）
`AIC slot = block`（`block<24, sub==0`）；`AIV slot = 24 + block*2 + sub`（`block<24, sub<2`）；`slots=72`
字段：`0 magic=0xD0D0C0DE` / `1 id=(block<<8)|sub` / `2 domain` / `3 claim_start_m` / `4 claim_m` /
`5 stage_bits`（K1,K2,K3,K4,K5进,K5返,K6前,K6后）/ `6 stage_bits2`（K7a,K7b,K8,is_worker,K8'）/ `7 seal=0xC0FFEE00^id`

## 5) adapter 严格链接产物
`d0_strict_fixed.so` SHA-256 **`f80b0e50c50eb2fb0eb25b13a77df11ef65173015b5861903e57315ad821a60a`**
（`g++ -shared -z defs`，**RC=0**，零未解析链接报错；ELF 仍有 288 个正常具名 UND）

**DT_NEEDED（12）**：`libtorch_npu.so`、`libtorch.so`、`libc10.so`、`libcust_opapi.so`、`libnnopbase.so`、`libascendcl.so`、**`libtorch_cpu.so`**、**`libpython3.12.so.1.0`**、`libstdc++`、`libgcc_s`、`libc`、`ld-linux-aarch64`
**无 RPATH / RUNPATH** ⇒ 运行期必须显式给 `LD_LIBRARY_PATH`。

## 6) 固定依赖绝对路径与 SHA
| 库 | 绝对路径 | SHA-256 |
| --- | --- | --- |
| libtorch_cpu | `/workspace/yangchen/venvs/clef-py312/lib/python3.12/site-packages/torch/lib/libtorch_cpu.so` | `d9529119ac18ba154521854dc8da477c25c9a34c3265c95e40086a593d741926` |
| libpython | `/workspace/yangchen/python/cpython-3.12.16+20261110/lib/libpython3.12.so.1.0` | `212da0dac0371ea3d1f87cd24b6750b6db12d145280d9b4480cac66b16c3c169` |
| opapi | `…/vendors/clef_d0_trace_22/op_api/lib/libcust_opapi.so` | `3314922677b7b78164628f1bc22c7339ac63d59cf537c261533cbf740ac948ec` |
| libnnopbase | `/usr/local/Ascend/cann-9.2.1/aarch64-linux/lib64/libnnopbase.so` | `72cd43b4af1c9c451b566ccb7b32071793402949578f838d8345e369b056477c` |
| torch_npu / torch | `…/site-packages/torch_npu/lib`、`…/torch/lib` | 目录级，按 venv 固定 |

## 7) 源码身份
kernel `op_kernel/clef_d0_trace.cpp`、host `op_host/clef_d0_trace.cpp`、tiling `op_kernel/clef_d0_trace_tiling.h`
（对应源提交 `85b3028` 的 E 代码 + D0 trace 插桩；**新增修补一律给新 SHA**）

## 8) 首格建议（mm_dim=1 / AR off）
- 请求 `mm_dim=1`、`use_hccl=2`（AR off）；`trace` 由调用侧清零
- 断言：**容量内已观测集合**的 `id/claim_start_m/claim_m` 覆盖 `Σ claim_m == M` 且两两不交
- **`trace` 不能证明 GM 可见性或 SDK 收发**；数值仍按**固定门**（mixed 容差）判，不以"差 0"替代
- `mm_dim=2` 待首格确认支持且 20/21 排队后再做
