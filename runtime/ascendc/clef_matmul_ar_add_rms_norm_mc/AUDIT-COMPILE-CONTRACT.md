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

**效力边界**：`taskRation = 1:2` 仅作为**本次生成元数据**。
它与 `get_subblockdim()` 的**运行返回值之间没有本次取得的直接契约**，也没有设备实读，
因此**不能**写成"ratio 运行值已证明"。`GetBlockIdx()` 的公式
`get_block_idx()*GetTaskRationImpl() + get_subblockid()`（源码事实）中的 ratio 仍是**动态 builtin**。

## 3. 宏与分支：可证到什么程度

| 宏/分支 | 结论 | 依据 |
| --- | --- | --- |
| `__NPU_ARCH__` | 非 3510（c220 树 / `ascend910_93`） | 编译单元 + `dav_c220` 实现树 |
| `USE_SSBUF` | **源内条件推导**：非 3510 不定义 | `matmul_utils.h:26-38`；**未取得逐 pass 宏表** |
| `USE_WORKSPACE` | **源内条件推导**：`:36-38` 无条件分支 | 同上 |
| 选中的 `IterateAll` 重载 | **源内条件推导**为 `matmul_client.h:931-977` | 由上述两行推出；**外层 `--compile-options` 为空不等于下游无 `-D`**（见 §6.2） |
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
| 采集字段 | 每 block：`GetBlockIdx()`、`GetSubBlockIdx()`、`GetBlockNum()`、`GetTaskRationImpl()`；`diag[5..7]`；`partial` 首/末行**仅作采样位置校验**（**不**证明全张量无漏写或覆盖正确；覆盖问题另需每参与者身份+区间记录+全范围检查）；client 发 `End`/quit 与 server 收到**分别记观测可达性**，未采到一侧不可推定 |
| 规模 | 单次**请求的 `mm_dim` 预算 = 1** 与 **= 2**，各一次窗口；`usedCoreNum` 与实际 launch **只记实测**，不预定为 1/2 |
| 超时 | **单次有界 180 s**，超时即杀进程并记录 |
| 停止条件 | 出现 `507014/0x25`、超时、或 `partial` 出现未写行 ⇒ 立即停，不继续下一格 |
| 判读 | 只回答"集合/比值/是否漏行"，**不判数值因果**；固定门仍为 mixed 容差 |
| 前置 | 需 20 显式授权设备时间；当前 E 设备授权关闭 |

## 6. 本次定向捕获结果（dump_cce，已冻结）

用**同一 TU**、`--debug-config=dump_cce` 做了一次定向重跑（RC=0），生成件被保留下来。
保留目录（driver 的 `working_dir = cwd/<opType>_<computeUnit>`，**独立于** `--output-path`）：

`build_out/op_kernel/ClefMatmulArAddRmsNorm_ascend910_93/`（6 个文件）

| 文件 | SHA-256 |
| --- | --- |
| `bin_param/ClefMatmulArAddRmsNorm-clef_matmul_ar_add_rms_norm-0.sh` | `e0fb4e87f4986c389f95883d3293bc787167613042c92e52cff4534f7cef56e6` |
| `bin_param/ClefMatmulArAddRmsNorm_b87a0f8f….json` | `9caf2d18153d57555d0f533bde6fdffbf90caec3cdbc637c80c3b1376de8b059` |
| `dynamic/ClefMatmulArAddRmsNorm.py` | 见目录（`__pycache__` 一并冻结） |
| `clef_matmul_ar_add_rms_norm.make` | 同上 |
| `customize/op_impl/ai_core/tbe/config/ascend910_93/aic-ascend910_93-ops-info.json` | 同上 |

日志：`/tmp/audit_build.log` `150abca952e31bcae55ec1b513b150a8e36b0e8b50a3d9263a32988846919c45`、
`/tmp/dumpcce.log` `eddb484566ad5a43cdf023ff8cf746b0d17dd1bbe64634842e50724220bc5232`。
清单**不含自身**。

### 6.1 生成 `asc_opc` 驱动调用（`.sh` 原文，按值记录）

> 这是**生成脚本里的 `asc_opc` 驱动命令**；**尚未取得** AIC/AIV **pass 级 `ccec` argv**。

```
asc_opc $1 --main_func=clef_matmul_ar_add_rms_norm --input_param=<param.json> \
  --soc_version=Ascend910_9391  --output=$2 \
  --impl_mode=high_performance,optional --op_mode=dynamic --simplified_key_mode=0
```

**三个名字分别列，不作故障推断**：
`--compute-unit=ascend910_93`（构建单元）／`--soc_version=Ascend910_9391`（本次 opc 目标）／
服务实机 `Ascend910_9382`（03 文档记录）。

### 6.2 更正 §3：外层 `--compile-options=""` ≠ 下游无 `-D`

`dynamic/ClefMatmulArAddRmsNorm.py` **显式注入**编译宏：
`:259-302` 补 `DTYPE/ORIG_DTYPE/FORMAT`、`-DHIGH_PERFORMANCE=1`、`DETERMINISTIC_MODE` 等；
`:327 get_code_channel`、`:332 compile_op`。
因此 **"外层 `--compile-options` 为空" 不能推出"下游没有 `-D`"**，我此前那一层推断**撤回**。
`USE_SSBUF`/`USE_WORKSPACE` 与各 `-D` 的**最终取值**须以该 `.py` + `asc_opc` 内部分支为准，
**本次仍未取得**逐 pass 宏表/预处理。

### 6.3 更正 §2：`taskRation` 的效力边界

`taskRation = 1:2` **只作为本次生成元数据报告**。
它与 `get_subblockdim()` 的**运行返回值之间没有本次取得的直接契约**，也没有设备实读；
动态 builtin 仍是缺口。**不得**写成"ratio 运行值已证明"或"只剩一个条件"。

### 6.4 D0 草案再收窄（仍属未授权建议）

- 两格应写**请求的 `mm_dim` 预算**（1 与 2），`usedCoreNum` 与实际 launch 集合**只记实测**，不预定为 1/2；
- 首末行 checksum **只验证采样位置**，不能判全张量无漏写或覆盖正确；
- 若将来要回答覆盖问题，草案须要求**每个参与者的身份 + 区间记录**与**全范围覆盖检查**；
- client 发出 `End`/quit 与 server 收到，**分别记观测可达性**，未采到的一侧不可推定；
- 180 s 与两格均**未获执行授权**。

## 7. 本次交付的缺项（明确）

1. **逐 pass 的 `-D` 宏表与预处理输出**：未取得（今次只拿到 `.sh`/param/`.py`/config，宏在 `asc_opc` 内注入）；
2. `__MIX_CORE_AIC_RATION__` 是否定义：未知；
3. `get_block_idx()`/`get_subblockid()`/`GetBlockNum()` 运行值：未取得；
4. 本次重建**只证明本次构建**，不追证旧 E 装载。

## 8. 历史撤回集中留存

| 日期 | 我曾写的 | 现状 |
| --- | --- | --- |
| 11-18 | "`IterateAll` 自动按 `GetBlockIdx` 跨核分片" | **撤回**（matmul 头无 `GetBlockIdx`） |
| 11-18 | "`SetTail` 只属 Iterate/UB 变体" | **否证**（`enableSetTail` 默认 true；`SetSingleShape`→`SetTail`） |
| 11-18 | "两个 AIV 子块共用同一 GM 槽 ⇒ 挂死" | **撤回**（无参 `SyncAll` 走 FFTS，`SoftSyncAllImpl` 是另一重载） |
| 11-18 | "`opc` 进程内 ⇒ 无 pass 命令 / 不落盘" | **撤回**（只证普通日志未存；`dump_cce` 可保留） |
| 11-18 | "外层 `--compile-options=""` ⇒ 无外部 `-D`" | **撤回**（`dynamic/<Op>.py` 显式注入 `-D`） |
| 11-18 | "`taskRation=1:2` 证明 ratio" | **收窄**为本次生成元数据 |
| 11-18 | "24 worker 落在 12 cube" | **降为条件推导**（需 ratio=2 且 launch 集合完整） |
| 11-18 | "数值必不变 / 没变就排除多核" | **撤回**（`GetTiling` 预编译，可能改 `baseK`/顺序） |
| 11-18 | "让候选逐位匹配原生不可达" | **撤回**（门是 mixed 容差，非逐位） |

## 9. 冻结副本

实际冻结副本在本包 `audit-frozen/`，含 6 个生成件、2 份日志、`liboptiling.so` 的
**链接原值/realpath/目标 SHA** 记录及**目标副本**；
清单 `audit-frozen/MANIFEST.sha256`（**不含自身**，SHA `1e1891449cef9b0e0f4b0ada8a219f08ffc49cd250b825a6b4e1ae9b63a12ddb`）。
两份日志的 `--src-file` 均指向完整 audit TU（源 SHA `08c412aa…2fb1e1` = `85b3028` blob）。
