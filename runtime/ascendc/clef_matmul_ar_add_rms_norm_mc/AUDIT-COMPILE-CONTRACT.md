# 编译期契约取证（基线：包 `873ef8fb`，源提交 `85b3028`）

**范围**：本次只用**同一源码提交**在独立目录
`/workspace/yangchen/work/wave3-e-audit-22` 重建一次，取证编译期事实。
现包 `873ef8fb` 与现构建目录**未被覆盖**。本次重建**只证明本次构建**，
**不追证**旧 E 实际装载过哪一份二进制。

## 1. 构建身份

| 项 | 值 |
| --- | --- |
| 源提交 | `85b3028`（包 `873ef8fb` 对应的代码提交；其后提交仅改文档） |
| kernel 源 SHA-256 | `08c412aad089b1f7c7df796a2dbbbd14c3803b7ea8685c2a604c5803d42fb1e1` |
| host 源 SHA-256 | `811b8ea5bf556501320d9dcd1c1fed0b601e0176fdbf8f03dba33d8ca5c75f13` |
| tiling 头 SHA-256 | `61098f514d865a92cae72f0b9eeda47a65a470eaa9851a240c8c64008295bfc6` |
| 编译单元 | `ascend910_93`（`--compute-unit`） |
| 外部编译选项 | `--compile-options=""`（空） |
| cmake / python / clang | `3.22.1` / `3.10.12` / `clang 15.0.5`（ccec 内嵌） |

真实 kernel 编译驱动命令（`cmake --build --verbose` 实捕获）：
```
python3 .../ascendc_compile_kernel.py --op-type=ClefMatmulArAddRmsNorm \
  --src-file=.../op_kernel//clef_matmul_ar_add_rms_norm.cpp \
  --compute-unit=ascend910_93 --compile-options="" --debug-config="" \
  --config-ini=.../autogen/aic-ascend910_93-ops-info.ini \
  --tiling-lib=.../op_host/libcust_opmaster_rt2.0.so --enable-binary="True" ...
```

**关于 pass 级命令的可捕获性（2026-11-18 收窄）**：
- 我只证明了一件事：**本次普通 build 的日志里没有保存 pass 级 argv**；
  **不能**由此推出"工具不支持捕获"。此前"`opc` 进程内调用 ⇒ 无可复放命令"的说法**撤回**。
- 驱动脚本确实**生成并消费**这些文件：
  `compiler/tikcpp/ascendc_kernel_cmake/fwk_modules/util/ascendc_compile_kernel.py`
  `:130-146` 生成 `bin-param *.json` 与同名 `*.sh`（`self.compile_sh`），`:175` 起循环执行；
  `:96-98` 的 `clean()` 只有在 `dump_cce` **不在** `op_debug_config` 时才删除 `working_dir`。
- **本次取证用的是 `--debug-config=""`**，因此生成的 `*.sh`/`bin-param`/工作目录已被清理，
  **本次没有取得**这些文件，也就**没有取得**真实 pass argv 与逐 pass 预处理。
- **明确记缺项**：AIC/AIV 的宏表与预处理输出**本次未取得**。
  §3 的宏结论因此只是**源内条件 × 构建元数据**的交叉，**不能**替代逐 pass 预处理证据；
  也**不把** `taskRation=1:2` 当作宏证据（两者来源不同）。
- 若将来需要，最小目标是**同一 TU**、`--debug-config` 含 `dump_cce` 的**一次**定向重建，
  保留生成的 `*.sh` / `bin-param *.json` / `opc` 配置 / 完整 argv 及工作目录产物；
  这不属于本次交付，也**不**另开无目标重建。

## 2. 生成 JSON 里的真实字段（构建期实证）

`build_out/op_kernel/ascendc_kernels/binary/ascend910_93/clef_matmul_ar_add_rms_norm/ClefMatmulArAddRmsNorm_b87a0f8f….json`：

| 字段 | 实读值 | 来源与含义 |
| --- | --- | --- |
| `coreType` | **`MIX`** | 该 kernel 按 AIC+AIV 混合核编译（双 pass） |
| **`taskRation`** | **`1:2`** | AIC:AIV 任务配比，**构建期写死进二进制元数据** |
| `blockDim` | `-1` | 未在二进制里固定，运行时由 launch 决定 |
| `intercoreSync` | `1` | 核间同步已启用 |
| `magic` | `RT_DEV_BINARY_MAGIC_ELF` | 产物为 ELF |
| `staticKey` / `sha256` | 见 JSON | 二进制/静态 key |

**`taskRation = 1:2` 直接支撑**：`get_subblockdim()` 在 AIV 侧应为 **2**，
即 `GetBlockIdx() = get_block_idx()*2 + get_subblockid()` 的 ratio 为 2。
（这与 `dav_c220/kfc/kfc_comm.h:93-97` 的 `MIX_NUM` 默认 2 一致，但**两者是不同来源**：
 `taskRation` 来自本包产物，`MIX_NUM` 来自源内宏。）

## 3. 宏与分支：可证到什么程度

| 宏/分支 | 结论 | 依据 |
| --- | --- | --- |
| `__NPU_ARCH__` | 非 3510（c220 树 / `ascend910_93`） | 编译单元 + `dav_c220` 实现树 |
| `USE_SSBUF` | **未定义**（源内条件） | `matmul_utils.h:26-38`：仅 `__NPU_ARCH__==3510` 定义 |
| `USE_WORKSPACE` | **定义**（源内 `:36-38` 无条件分支） | 同上 |
| 选中的 `IterateAll` 重载 | **`matmul_client.h:931-977`**（非 SSBUF 的 GM 重载） | 由上两行推出；`--compile-options=""` 无外部 `-D` |
| `__MIX_CORE_AIC_RATION__` | **无法从产物确定** | 源内 `#ifdef`，产物只给 `taskRation=1:2` |
| `ASCEND_IS_AIV` / `ASCEND_IS_AIC` | 由双 pass 决定，`coreType=MIX` 印证 | `coreType` 实读 |
| 选中的 `SyncAll` | `SyncAllImpl<true>` → FFTS `SYNC_AIV_ONLY_ALL` | `kernel_operator_sync_impl.h:363-403`（同机 SHA 见 §4） |
| `End()` 是否投 `MMFUN_END` | **投**（`isSyncGetC = sync = true`，默认模板实参） | `matmul_client.h:963/966/968-975/977`、`:695-701` |
| quit 来源 | `KfcCommClient` 析构，**独立于 `End()`** | `dav_c220/kfc/kfc_comm_client.h:100-118` |

引用头的 SHA-256（同机）：
`matmul_utils.h` `4871798f…e724e`、`matmul_client.h` `98b308a7…7f23`、
`kernel_operator_sys_var_impl.h` `3f3326c7…a624`、`kernel_operator_sync_impl.h` `695bb460…2c1e`。

## 4. 仍需运行时确认（不再静态检索）

1. `get_block_idx()` / `get_subblockid()` / `GetBlockNum()` 的**实际取值**；
2. 实际 launch 的 block 集合是否完整（2×AIC 个 AIV）；
3. 每个 AIC 的 client/quit 计数是否达到 `MIX_NUM`；
4. `SYNC_AIV_ONLY_ALL` 是否把各 worker 的 GM 数据写序化到其他 AIV 读之前；
5. `__MIX_CORE_AIC_RATION__` 是否在构建中被定义（影响 `MIX_NUM` 取值）。

## 5. 未来 D0 诊断草案（**只写草案，不实施**）

一次运行、有界超时、只采字段不扫参数。

| 项 | 内容 |
| --- | --- |
| 采集字段 | 每 block：`GetBlockIdx()`、`GetSubBlockIdx()`、`get_block_idx()`（如可读）、`GetBlockNum()`、`GetTaskRationImpl()`；`diag[5..7]`（`usedCoreNum/baseM/depthA1`）；kernel 是否产出 `partial`（首/末行非空校验和） |
| 规模 | 单次 `usedCoreNum=1` 与单次 `usedCoreNum=2`，各一次窗口 |
| 超时 | **单次有界 180 s**，超时即杀进程并记录 |
| 停止条件 | 出现 `507014/0x25`、超时、或 `partial` 出现未写行 ⇒ 立即停，不继续下一格 |
| 判读 | 只回答"集合/比值/是否漏行"，**不判数值因果**；固定门仍为 mixed 容差 |
| 前置 | 需 20 显式授权设备时间；当前 E 设备授权关闭 |
