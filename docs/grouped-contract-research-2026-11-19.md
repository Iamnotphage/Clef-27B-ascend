# GroupedMatMulAllReduce 支持矩阵与通信/stream/生命周期契约研究（2026-11-19）

只读研究：不写 31/35 代码、不做设备/SSH/installer/推理请求。环境固定 A3 `Ascend910_9382`、CANN 9.2.1、torch 2.10.0+cpu、torch_npu 2.10.0.post7。

来源分层（必须区分）：

- **本机 9.2.1 安装证据**：31 只读 API 包 `/tmp/aa3e-grouped-api-pack-20261119`（tar SHA `685d9a33…`）：头文件、op-info、binary_info_config、两个 A3 kernel `.o`、libopapi/libnnopbase/libhccl/libascendcl/libtorch_npu、version.info、platform ini、torch_npu HCCL 头与符号。
- **上游固定参考**：`hicann/ops-transformer` commit `f5703e415b9be16a2af7ba85564c37414592fc03` 的 op_api 源、op_host 源、kernel 源、docs 与 examples。**不是本机安装版同版本真值**。
- **A12 冻结样例**：`/Users/chenclay/.ao/data/artifacts/clef-27b-ascend-33/grouped-a12-sample/`（repo example SHA `6c1a90ab…`；文档参考 `31333c59…`；编译配方 `c591f0e3…`）。
  官方仓库 f5703e4 的 `examples/` 目录只有 `test_aclnn_grouped_mat_mul_all_reduce.cpp`，**不存在 `examples/CMakeLists.txt`**（GitHub 目录实读；raw 404）；op 级 `CMakeLists.txt`（SHA `53888501…`，1230 B）不含 example target。因此 A12 只能标注为“官方样例源 + 派生 CMake 构建”，31 的 stage-manifest `088f05f6…` 已如此记录（source `6c1a90ab…`、glue `22edb89f…`、binary `37929ec4…`、ldd 0 missing、provider delta 已记）。

## 1. 支持矩阵（逐项对应旧 v4r1）

| 维度 | 本机 9.2.1 证据 | 上游契约/样例 | v4r1 实际 | 结论/来源 |
| --- | --- | --- | --- | --- |
| 产品/SoC | `aic-ascend910_93-ops-info-transformer.json`（SHA `a16b41d0…`）含 `GroupedMatMulAllReduce`；ascend910_93 下两个 MIX kernel `.o`：`85517750…`（json `da5e54b3…`/o `a5d7ec03…`）、`90b5b6bb…`（json `f203a523…`/o `8ee67292…`） | op def 同时 `AddConfig("ascend910b")` 与 `AddConfig("ascend910_93")`，并 `MC2().HcclGroup("group")`；文档产品表写“Atlas A3：不支持”，同文 A3 样例却写“必须在 Atlas A3 上运行” | A3/Ascend910_9382 实跑：prepare/launch/handler rc0，sync 未返回 | **注册/文件存在成立；文档自相矛盾；SoC 运行支持未裁决**。文件存在仅排除“缺失”，不外推支持 |
| kernel/dtype 映射 | binary index `.../kernel/config/ascend910_93/ops_transformer/grouped_mat_mul_all_reduce.json`（SHA `c9bd574a…`，5618 B）：FP16 条目（x/w/bias FP16）→ `GroupedMatMulAllReduce_85517750…`；BF16 条目（x/w BF16、bias FP32）→ `GroupedMatMulAllReduce_90b5b6bb…`；均 ND、`format_match_mode=FormatAgnostic`，simplifiedKey `d=0/1,p=0/1`；attrs splitItem/group/reduceOp/commTurn | 与本机 op-info dtype 列表一致 | BF16 使用 `90b5b6bb…` kernel | **BF16 ND kernel 存在且 bias FP32 由 binary index 实证**；不证明运行正确性 |
| 设备/rank | tiling 接受 rankSize 1/2/4/8（source `rankSizeSupported`） | 文档“2/4/8 卡”；样例 `HcclCommInitAll(2, {0,1})` | TP2、两 logical device 0/1（同物理卡） | 2 rank 合法；**同物理卡两 logical 未在文档单列**，按设备数约束处理 |
| group 数量 | op-info `attr_group` required str；`mc2.ctx=mc2_context_0` | source `MAX_GROUP_LIST_SIZE=64`；头注释写“最大长度 128”；文档 ≤64 | group=1、`group_name_0` | group=1 合法；三处上限口径 64/128/64 不一致，group=1 不受影响 |
| x/weight dtype | op-info `bfloat16,float16`、format `ND,ND` | source 只支持 BF16/FP16 且 x/weight 同 dtype | BF16 | 支持 |
| bias dtype | op-info `float32,float16`（BF16 对应 FP32） | source：BF16→FP32、FP16→FP16；头注释却写支持 BF16/FLOAT16 | FP32 zero `[5120]` | **v4r1 合规**；头注释与实现不一致 |
| 输出 y | op-info `bfloat16,float16`、ND | source 校验 y 与 x 同 dtype | BF16 `[967,5120]` | 支持 |
| ND/NZ/转置/连续 | op-info 仅 ND；无 NZ 注册；API 无 transpose 参数 | 源码按 2D weight、M/K/N 语义；无 transpose 入参 | ND、weight `[K,N]`、x `[M,K]`、连续、offset 0 | **仅 ND、无转置接口**；连续为 binding 约束，文档未显式要求 |
| M/K/N 对齐 | tiling 仅要求 32B 对齐后维度 < int32 max | 文档：最后一维 <65536；各维 32B 对齐后 < int32 max；无 128 倍数要求；样例 M=256/1024、K=256、N=256/1024 | M=967、K=8704、N=5120，均满足 | 约束满足；M 非 128 倍数不是文档违规，H7 降为低优先 |
| splitItem/groupList | op-info attrs `splitItem,group,reduceOp,commTurn` | `splitItem=0`：x/y/weight 组数相等、groupList 必须 null、x/y 2–6 维、weight 2 维 | splitItem=0、groupList null、x/y/w list 各 1 | 合规 |
| commTurn | op-info default 0 | source 仅支持 0（否则 161002 类参数错） | 0 | 合规 |
| streamMode | 头注释写 0/1；source 仅 1 | source 仅支持 1（否则参数错） | 1 | **合规**；头注释过期 |
| reduceOp | op-info default sum | source 仅 "sum" | "sum" | 合规 |
| workspace | API 返回大小；kernel `GetUserWorkspace` 用于 GEMM；HCCL 消息区在 `GetHcclContext<0>()->WorkSpace` | A3 样例：`aclrtMalloc` → launch → `aclrtSynchronizeStreamWithTimeout(10s)` → `aclrtFree` | at::empty 16,777,216 B；handler 返回即释放（早于 sync） | **生命周期偏离官方样例（H2）** |
| group 名称生命周期 | 头文件无“需保持到 sync”说明 | A3 样例 `char hcom_name[N]` 栈上保持到 sync 后；tiling 在 prepare 用 `GetRankSize(group_)`；kernel 用 `GetHcclContext<0>` | pybind 临时 `std::string` 在 torch op 返回即析构；handler 异步 launch 在其后 | **原 string 已析构=已证；“executor/运行时仍引用该 c_str”=假设（H1，待 A1/头文件/生成 inner 代码核）** |
| stream/context | API 显式 `aclrtStream` | 样例每 rank 线程 `aclrtSetCurrentContext`，同 stream sync；`aclrtSynchronizeStreamWithTimeout` | prepare 时捕获 stream；handler 异线程 launch；Python final sync | H4/H5 待 instrumentation |
| executor/descriptor | 两段式，第二段不可重复 | 样例不销毁 executor；launch 后仍持有 | executor 泄漏（未销毁）；handler 内 launch 后立即 `lists->close()` | executor 泄漏不致命；descriptor 早销毁是 H3 候选 |
| 通信创建方式 | torch_npu 用 `hcclCommInitRootInfoConfig`（自带 HCCL 头） | 样例单进程 `HcclCommInitAll` + `HcclGetCommName` | 2 进程 torch_npu RootInfo comm + `get_hccl_comm_name` | **H12：多进程 comm 与样例不同，需 A12 对照裁决** |
| 确定性 | — | 文档默认确定性实现 | 未涉及 | 记录 |

## 2. 已证事实 / 可排除 / 待检假设

**已证事实（v4r1 trace/plog/record）**

1. prepare rc0、workspace 16,777,216 B、executor 非空；rank0/1 用时 274.1/283.2 ms。
2. `cmd.Run()` dispatch 27 µs 返回；handler 异步：launch_return 比 final_sync_enter 晚 48.7/28.1 ms；launch/cleanup/handler rc0。
3. final_sync_enter 后约 239 s 未返回；RUNTIME +180 s 报 `three minutes timeout! stream_id=47, task_id=36, pendingNum=2`。
4. probe 日志 0 ERROR/WARN；device 日志止于初始化，无 AICore trap；HCCL AllReduce ring/SIO 资源创建成功。
5. 原 `std::string group` 在 torch op 返回时析构（pybind 临时），handler 在其后执行——**字符串对象已析构是事实**。

**可排除**

- host 描述符/prepare/入队/cleanup API 失败；旧 streamMode=0 负结果；A3 缺 Grouped kernel/op-info 文件；HCCL comm 创建失败（baseline all_reduce 与 nd_native 均成功）；final_sync 前宿主阻塞。
- kernel/op-info 文件存在**不能**外推“当前芯片/形状/comm 组合支持”。

**待检假设（优先级）**

| 编号 | 假设 | 单变量实验 |
| --- | --- | --- |
| H1 | executor/ACL 运行时在 launch 时仍引用组名 c_str → 悬垂 | A1：prepare 前创建静态/堆 string，直接传其 c_str，保活到 sync 后；记录指针/字节/SSO |
| H2 | workspace 在 sync 前释放，MC2 仍在使用 | A2：workspace global 保活到 sync 后，记录 data_ptr |
| H3 | launch 入队后立即销毁 descriptor/list | A3：lists 保活到 sync 后（延后 close） |
| H4 | handler 线程 ACL context 与 prepare/comm 不同 | A4：记录 thread/context/device；调用线程直接 launch + sync 对照 |
| H5 | 捕获 stream 与 sync stream 不一致 | A5：每臂打印 prepare/handler/sync stream id，对照 RUNTIME 47 |
| H6 | workspace 需零初始化 | A6：launch 前 `aclrtMemsetAsync(workspace,0,size,stream)` 诊断 |
| H7 | M=967 对齐/tiling（**低优先**：文档无 128 倍数要求） | A7：M sweep 128/256/1024/967 |
| H9 | enqueue stream vs sync stream | 并入 A5 instrumentation |
| H11 | tiling/executor host 缓冲生命周期（executor 未销毁 → 长期存活，降级） | A11：仅在头文件/内部证明引用调用方缓冲时保留 |
| H12 | 多进程 torch_npu comm vs 样例单进程 HcclCommInitAll | **A12 决定性对照**：官方样例原样 A3 双设备运行 |

**决定性控制实验 A12**：官方 example `test_aclnn_grouped_mat_mul_all_reduce.cpp`（SHA `6c1a90ab…`）原样 FP16、2 设备（`./main 2 1 0,1`）。若通过 → 收敛到多进程/框架桥接/对象生命周期；若 hang → 保留样例版本/编译配置差异，不单因裁决 SoC 不支持。

## 3. 通信与 stream/生命周期契约要点

1. **组名解析**：tiling 在 prepare 阶段 `GetRankSize(group_)`（v4r1 rc0 证明当时有效）；op def `MC2().HcclGroup("group")` 表明框架在 launch 阶段还需按组名绑定 HCCL 上下文；kernel `GetHcclContext<0>()` 取该上下文并在 `context->WorkSpace + notifyOff` 上运行 `HcclServer`。**launch 阶段组名是否仍被使用是 H1 的关键未知**。
2. **HcclServer 等待**：kernel 在 AIC 侧 `PRELOAD(4)`、`CrossCoreSetFlag/WaitFlag` 与 `TurnWait`/notify 上等待 AICPU 消息；若 AICPU 任务或上下文未正确绑定，表现即为设备任务长期 pending（与观测一致）。
3. **workspace 语义**：用户 workspace 仅用于 GEMM（`GetUserWorkspace`）；MC2 消息区位于 HCCL 上下文 workspace。样例在 sync 后才 free 用户 workspace；v4r1 提前释放是契约偏离，但主要影响 GEMM 侧，需 A2 证伪。
4. **stream/context**：样例每 rank 线程显式 `aclrtSetCurrentContext`，并在同一 stream 上 `aclrtSynchronizeStreamWithTimeout`；v4r1 handler 异线程、捕获 prepare 时 stream，需 A5/A4 记录实际值。
5. **executor/descriptor**：两段式第二段不可重复；样例不销毁 executor；v4r1 executor 泄漏但 descriptor 在 launch 后立即销毁——若运行时在设备侧仍解析描述符则 H3。
6. **通信创建**：样例单进程 `HcclCommInitAll` + `HcclGetCommName`；torch_npu 为 RootInfo 多进程 comm。H12 需 A12 对照；不以“静态 skill 无报错”宣布通信安全。

## 4. 文档/头文件不一致清单（供评审）

1. 产品支持表“A3 不支持” vs 同文样例“必须在 A3 运行” vs op def 注册 `ascend910_93`。
2. 头注释 streamMode 0/1 vs 实现仅 1。
3. 头注释 bias 支持 BF16 vs 实现/op-info BF16→FP32。
4. 头注释 group 最大 128 vs source `MAX_GROUP_LIST_SIZE=64` vs 文档 ≤64。
5. API 标记 deprecated（头 `ACL_DEPRECATED_MESSAGE`；文档“后续版本废弃”）。

## 5. 第一批结论（不重复旧 timeout 结论）

- 旧 timeout 的**已证**仅是“host prepare/入队成功、设备完成等待未返回”；本次把可疑面收敛到 MC2 上下文/组名绑定、workspace/descriptor 生命周期、线程/context/stream 与多进程 comm 桥接四类，并给出决定性 A12 对照与 A1/A2/A3/A4/A5 单变量矩阵。
- 不先判 API 不支持：源码/op-info/样例三方证据冲突，需 A12 实测裁决。

## 6. E1：组名在 GetWorkspaceSize 是否被 executor 复制（本机 9.2.1 二进制证据）

调用链（本机库 SHA 已核）：

1. `libopapi.so`（SHA `eb1c788e…`）导出 `aclnnInnerGroupedMatMulAllReduceGetWorkspaceSize`：`0x2644a1c`，size `0x994`。
2. 该函数对字符串入参调用 `strlen@plt`（`0x2644e60`、`0x2644ec4`、`0x2644f30`），随后调用 `NnopbaseAddAttrWithDtype@plt`（`0x2644e7c`、`0x2644ee0`、`0x2644f4c`）；并在 `0x264500c` 调用 `NnopbaseSetMc2@plt`。即 attrs（含 group）以“指针+长度”形式交给 Nnopbase。
3. `libnnopbase.so`（SHA `72cd43b4…`）`NnopbaseAddAttrWithDtype`（`0x2679e8`）转调 `NnopbaseExecutorAddAttr`（`0x2df8d8`，size `0x13a0`）。
4. `NnopbaseExecutorAddAttr` 内存在 `std::string(char const*, allocator const&)` 构造（`0x2dfa04` `_ZNSsC1EPKcRKSaIcE@plt`）、`strlen`+`std::string::append(char const*, size_t)`（`0x2dfb14`/`0x2dfb24`）、`operator new`（`0x2dfc10`）与 `memcpy_s@plt`（`0x2dfd7c`，目标为 executor args 缓冲区）——attr 值被复制进 executor 自己的存储。

结论（置信度高，二进制级）：**group 字符串在 prepare/GetWorkspaceSize 阶段被复制为 executor-owned 存储；H1“launch 时仍引用调用方 c_str 导致悬垂”被强烈削弱**。局限：这是本机 9.2.1 二进制反汇编（非 generated inner 源），个别分支归属依地址上下文判断；A1 实验可保留为低成本运行时确认，但不应作为首批最高优先。该证据不改变“tiling 在 prepare 用 `GetRankSize(group_)`”的事实。

## 7. E2：torch_npu 2.10 异步 handler 的 device/context/stream 传递（上游 tag 源）

版本：`Ascend/pytorch` tag `v26.1.1-pytorch2.10.0`（与本机 torch_npu 2.10.0.post7 / npu-smi 26.1.1 最近；非精确 patch 对应，本机 libtorch_npu SHA `65e6be19…`）。

调用链：

1. `torch_npu/csrc/framework/OpCommand.cpp`：`OpCommand::Run()`（L129–179）在 task-queue 开启且非 sync-launch stream 时，`aclCmd->ExportParams(execParams)`（`ExecuteParas` 含 customHandler）→ `c10_npu::queue::QueueParas(COMPILE_AND_EXECUTE, …)` → `c10_npu::enCurrentNPUStream(&params)`；`RunOpApi`（L183–228）另有 `EXECUTE_OPAPI` 路径，但 binding 走 `SetCustomHandler` + `Run()`。
2. `torch_npu/csrc/core/npu/NPUStream.cpp`：`enCurrentNPUStream`（L739–748）把 `paramStream = current_streams[device_index]->stream` 写入队列参数——入队时 current stream 被记录。
3. `torch_npu/csrc/core/npu/NPUQueue.cpp`：消费线程 `StartConsume`（L795–810）只做 `SetThreadAffinity(device_id)` 与 `c10_npu::SetDevice(device_id)`，**无显式 `SetCurrentContext`**；`ReadQueue`（L425–490）→ `manager().Call`。
4. `torch_npu/csrc/framework/OpParamMaker.cpp`：`ExecFunc`（L331–370）在 `COMPILE_AND_EXECUTE` + `customHandler` 时直接 `ret = cur_paras->customHandler()`——**handler 无参数、不接收 stream**；传入 `ExecFunc` 的 `stream` 仅用于 aclop 路径（L396–408）。`ExecFuncOpApi`（L425–465）同样无参调用。

结论（上游明确事实）：custom handler 在**设备消费线程**执行；该线程通过 `SetDevice` 使用设备 primary context（ACL primary context 为设备级共享）；handler 使用 binding 捕获的 prepare-time stream，Python `torch.npu.synchronize()` 为设备级同步，覆盖该 stream。**H4“handler 使用另一 context”与 H5“捕获 stream 与同步 stream 不同”均被削弱**；仍以 A9 在本机记录 thread/context/stream id 作最终确认。局限：上游 tag 可能与 post7 有 patch 差异；本机二进制未逐行比对。

## 8. 证据对假设优先级的修订

- 降级：H1（组名复制）、H4（primary context）、H5（stream 一致）、H11（executor 泄漏且 attr 复制）。
- 保持/上升：H2（workspace 在 sync 前释放）、H3（descriptor 早销毁）、H6（workspace 初值）、H12（多进程 torch comm vs 样例单进程 `HcclCommInitAll`）与 MC2 上下文/AICPU 绑定；H7 仍低（无 128 倍数约束）。
- 首批实验仍按 A8→A12→A0→A9→A2/A3→A6→A12b 推进；A1 保留但降为确认项。

## 9. A12 决定性结果与 mc2_server 线索（2026-11-19 追加）

A12 官方样例（FP16、2 group、M=256/1024、K=256、N=256/1024，源 `6c1a90ab…`）在 A3 上**复现同一失败**：

- `child.stdout`：两 rank `hcom name` 对称、`gmm_allreduce workspaceSize = 16777216`（与 v4r1 相同）；`test_aclnn_grouped_mat_mul_all_reduce.cpp:383 retcode 507046` + `TestCall execute AICPU failed`。
- `plog-debug`：两 device `Stream synchronize timeout, stream_id=47, timeout=10000ms`；`current task is type_name=KERNEL_AICORE, task_id=1`；`ErrCode=507046 InnerCode=0x7030010`；随后 `HCCL DestroyAicpuComm` 50s 超时两次。
- `plog-run` **决定性线索**：`insert package:mc2_server.tar.gz config`；随后两 device 均 `current device chip:5 does not support package:mc2_server.tar.gz` → `does not need to load`；`cann-tsch-compat.tar.gz` 同被 chip:5 拒绝。AICPU scheduler 日志止于 KFC `RunAicpuKfcResInit/RunAicpuKfcSrvLaunch` API 加载，无任务处理进展。
- 归档核验：`a12-cell` 10 文件、SHA256SUMS 10/10；恢复 `branch=aa3e-noop`、target aa3e、guard `36eaff44…`、pid 1301295、五 PID/四检/env/joint 全过（32 独立核一致）。

意义：A12 保持组名/workspace/descriptor 到 sync 后、显式 per-rank context/stream、单进程 `HcclCommInitAll`，仍复现 stream 47/AICORE pending → **H1/H2/H3/H11 以及 H4/H5 对“挂起”被决定性削弱；H7（对齐）与 H12（单进程 vs 多进程 comm）亦削弱**。`mc2_server.tar.gz` 在 chip:5 被拒绝加载，与 kernel `HcclServer`/KFC 等待、AICPU 销毁超时一致——**MC2 AICPU/KFC 公共路径缺失**成为首要方向。

## 10. A13 控制臂重设计（MatMulAllReduce 不可用于 A3）

- `aclnnMatmulAllReduce`（V1/V2/V3）**文档产品表：Atlas A3 不支持**；本机 9.2.1 A3 op-info 只注册 `AllGatherMatmul`、`AllGatherMatmulV2`、`MatmulReduceScatter`、`MatmulReduceScatterV2`，**无任何 MatMulAllReduce 变体** → A13 原设计在 A3 会 op 查找失败（161001 类），不能区分 MC2 路径。
- 头文件事实（31 pack）：V1 `(x1,x2,bias,group,reduceOp,commTurn,streamMode,output,ws,exec)`；V2 增 `x3`；V3 增 `x3` 与 `commMode`；bias 注释 f16/bf16、streamMode 注释 0/1（实现仍以 docs/二进制为准）。
- **替代控制对**（文档均 A3 支持，且本机 A3 op-info 已注册）：
  - **A13a `aclnnMatmulReduceScatter` V1**：A3 文档“仅支持 AI CPU 通信”（KFC/mc2_server 路径）；示例 `mc2/matmul_reduce_scatter/examples/test_aclnn_matmul_reduce_scatter.cpp`，SHA `0996d6f4…`，8,873 B；A3 上 bias 暂不支持非 0。
  - **A13b `aclnnMatmulReduceScatterV2` commMode="aiv"**：A3 文档仅支持 `aiv` 且通信缓冲区 ≥200MB（plog 默认 `HCCL_BUFFSIZE=200M`）；示例 `mc2/matmul_reduce_scatter_v2/examples/test_aclnn_matmul_reduce_scatter_v2.cpp`，SHA `83aa3286…`，10,461 B。
  - 判定：A13a hang + A13b 成功 → `mc2_server`/KFC 路径在 chip:5 不可用，Grouped（仅 AICPU 路径）为环境阻塞；A13a 成功 → Grouped op/kernel 特定问题。
- 待 31 只读提供：`/usr/local/Ascend/cann-9.2.1/opp/built-in/op_impl/aicpu/kernel/mc2_server.tar.gz` 与其 support 配置/README、同目录清单、`cann-tsch-compat.tar.gz` 配置，以及 A13a/A13b kernel 路由文件；若确认 chip:5 不在 mc2_server 支持列表，则 Grouped 路径记为环境阻塞，停止 H6/H7/H12 设备臂，不改变 aa3e/f810 默认。

## 11. mc2_server chip:5 支持矩阵收尾与结案（2026-11-19）

31 只读证据包 `Clef-27B/logs/overnight-20261119-owner31/grouped-deepdive/mc2-support/`（commit e33d5f7）经 33 独立核：

- 5 文件 + SHA256SUMS **5/5 OK**；SUMS 文件 SHA `1f5e62c8…`。
- `mc2_server.tar.gz` SHA `71263835…`，873,250 B；文件头 `aa55aa55` 私有包格式（非 gzip），标准工具 `file` 判为 data；`strings` 仅见 PKI/CRL URL，包内 chip 支持清单无法用标准解包读取。
- `ascend_package_load.ini`：`mc2_server.tar.gz` 为 `optional:true`、`package_path:opp/built-in/op_impl/aicpu/kernel`、`load_as_per_soc:false`；全部 AICPU/compat 包均 `optional:true`，ini 本身**无 chip/soc 支持字段** → 支持判定在运行时 package loader。
- `libmc2_server.json`：`HcclLaunchAicpuKernel`（`opKernelLib:AICPUKernel`）、`Mc2AicpuTaskCacheClearKernel` 与 `Mc2ServerKernel`（`opKernelLib:KFCKernel`）均映射到 `libmc2_server.so`——即 MC2 KFC 实现位于被跳过的包内。
- `mc2-load-evidence-lines.txt` 与 A12 `plog-run-1301080.log` 逐字一致（两处 `chip:5 does not support package:mc2_server.tar.gz`；`cann-tsch-compat.tar.gz` 同被跳过）。
- **chip:5 ↔ A3 映射（经验）**：A12 实测设备 0/1 为 `Ascend910_9382`（phy 14/15），loader 对这些设备记录 `chip:5`；本机未找到官方 chip-id 映射表，故以“被测设备即 chip:5”为操作事实，不对 chip 编号做外部推断。
- 同段 AICPU 包加载：`Ascend-aicpu_legacy`、`aicpu_hccl`、`aicpu_hcomm` 为“same as device / skip load”，未被拒绝；被拒绝的唯一 MC2 服务包是 `mc2_server.tar.gz`（另有 compat 包 `cann-tsch-compat`）。因此缺口明确为 **MC2 server/KFC 实现**。

**A13 更正**：35 已实现的 A13（`aclnnMatmulAllReduce` v1，commit 365cb95）在本机 A3 **无法执行**——文档产品表 A3 不支持，且本机 A3 op-info 未注册任何 MatMulAllReduce 变体（仅 AllGatherMatmul/V2、MatmulReduceScatter/V2）；运行会停在 op 查找（161001 类），不能作为“MC2 路径”等价控制。A13 代码保留未执行。若将来需要 A3 AICPU 路径控制，应使用 MatmulReduceScatter V1（A3 仅 AI CPU 通信）与 V2 `commMode="aiv"`（A3 仅 aiv、缓冲 ≥200MB）对照；但 loader 直接拒绝证据已足以给出操作结论，按 32/35 决定暂停全部 Grouped 设备臂。
（MatMulAllReduce 头/文档事实备查：x1/x2/bias/output 同为 BF16 或 FP16，bias shape `(n)` 可空；示例 streamMode=1、commTurn=0；V2 增 x3，V3 增 x3+commMode。）

**结案**：GroupedMatMulAllReduce 的 MC2 AICPU/KFC 路径在本机 A3（Ascend910_9382）/CANN 9.2.1 因 `mc2_server.tar.gz` 在 chip:5 不被加载而环境阻塞；A12 官方样例与 v4r1 同因；H1–H12 对“挂起”结案（非 binding/生命周期/comm 创建缺陷）。官方 aa3e/f810 默认、入口、指标与 CSV 不变；无 PR；服务保持 aa3e/s1（root 1301295）健康在线。关联记录：31 mc2-support `e33d5f7`、32 `34ccde7`、35 `98fc132`/`d4b600c`、33 `2616628`/`334c67f`/`049b5ea`/`d052acd`/`8b584d1`。

## 12. 反证式收窄：mc2_server 依赖 supported_but_unproven（2026-11-19，20 要求）

> 第 11 节的操作性结案仅基于 loader 拒绝证据；20/32 已将其收窄为“environment first-cause clue / 服务依赖与合法控制待核”（32 `0be6809`）。本节保留为修订轨迹与新增证据，旧文本不删。

**支持“Grouped 依赖被拒包”的证据**：`Mc2ServerKernel` 实现只存在于 `mc2_server.tar.gz` 内（磁盘无 libmc2_server.so）；Grouped kernel 源码含 `rac_server.h`（HcclServer + AivAicpuOpParam/MC2 buffer 协议），说明其期望一个 AICPU server 对接；libccl_kernel.so 无 `Mc2ServerKernel`/`RacServer` 符号。

**削弱“必要依赖”的证据（新增）**：

- Grouped tiling（f5703e4 `grouped_mat_mul_all_reduce_tiling.cpp`）明确设置 `commType=HCCL_CMD_ALLREDUCE`、`taskType=KfcTaskType::KFC_TASK_HCC_TASK_DELIVER`、`funID=ALL_REDUCE_FUNC_ID`、`commAlg=0`、`commOrder=1`；AICPU 侧对应 KFC 任务。
- A12 device 日志只显示从 `libccl_kernel.so` 成功加载 `RunAicpuKfcResInit`/`RunAicpuKfcSrvLaunch` 并完成 `AicpuRpcResInit`；**无任何 `Mc2ServerKernel`/`HcclLaunchAicpuKernel`/`libmc2_server` 的加载、请求或失败行**。
- 31 磁盘/符号对照：全 CANN 仅 `libmc2_server.json` 引用 `Mc2ServerKernel`，op-info/binary_info/op 配置均无引用；libccl_kernel.so 导出 46 个 KFC/HCCL kernel 与 MC2 机制符号（`HcomMc2AiCpuStreamAllocAndGet`、`HcclMc2ComOpResCtx` 等）。
- 未解释线索：plog `LoadSysOpKernel: cannot find aicpu packages`（device0/1），待 31 核是否正常。

**判定**：`dependency_status = supported_but_unproven`。被拒包是环境首因线索，但“包被拒 → Grouped 必挂”的必要性未证；不能以“包被拒 + op 挂起”单独结案。证明所需：`libccl_kernel.so`/（可解时）libmc2_server 符号映射、Grouped 任务类型到 server 的对应、MatmulReduceScatter V1 kernel `.o` 符号串、A14 控制；不建议改白名单强载。

**A14 MatmulReduceScatter V1 事实（待 31 补本机头/kernel）**：

- 本机 A3 op-info 注册 `MatmulReduceScatter`（x1/x2 f16/bf16、bias optional、y 同、ND、`mc2.ctx=mc2_context_0`）与 `MatmulReduceScatterV2`。
- 上游头（f5703e4 `mc2/matmul_reduce_scatter/op_api/aclnn_matmul_reduce_scatter.h`，SHA `6e49c20f…`，3,158 B）：`GetWorkspaceSize(x1,x2,bias,group,reduceOp,commTurn,streamMode,output,ws,exec)`；bias 可空、A2/A3 非 0 不支持；output `(m/rank_size,n)`；x1 `(m,k)`、x2 `(k,n)`/转置 `(n,k)`；dtype 全链一致；sum/commTurn 0/streamMode 1；**m 必须为 rank_size 整数倍**；2/4/8 卡、hccs all-mesh、同通信域；默认非确定性。
- 样例：`mc2/matmul_reduce_scatter/examples/test_aclnn_matmul_reduce_scatter.cpp`，SHA `0996d6f4…`，8,873 B。
- kernel：arch22 full-mesh，`GetHcclContext<HCCL_GROUP_ID_0>`、mc2InitTiling/mc2CcTiling、KFCNotify/KFCMsgBody → 与 Grouped 同 KFC 家族；是否依赖 mc2_server 未证。
- 建议形状：小合法 M=128/K=256/N=512；v4r1-like 用 **M=968**（967 非 rank2 整数倍）、K=8704、N=5120，输出 (484,5120)；判据同 A12。
- 兼容提示：V1 文档要求驱动/固件与 CANN 配套（≥8.0.RC2）；本机 Software 26.1.1 / Firmware 9.0.0.9.220、CANN 9.2.1 `compatible_version_fw=[7.0.0,9.9.9]`；无本机官方矩阵，待核不下结论。

**A12-H6 workspace 约定**：GWS 仅返回大小、调用方 aclrtMalloc；官方样例不初始化。Grouped kernel `GetUserWorkspace(workspace)` 仅作 GEMM scratch；MC2 消息区在 `GetHcclContext<0>()->WorkSpace`。无证据要求预置/必须未初始化 → memset 清零在契约上安全。建议同 stream `aclrtMemsetAsync(workspaceAddr, workspaceSize, 0, workspaceSize, rtStream)`，紧随 aclrtMalloc、launch 前；**同流顺序足够，勿额外 sync**（第二变量）。API 可用性：31 只读实读确认 `include/acl/acl_rt.h:2563` 声明 `aclrtMemsetAsync`、`:2293` 声明 `aclrtMemset`，`libascendcl.so` 导出两符号（31 证据；33 本机无 acl_rt.h 副本，未独立复核）。签名：`aclrtMemset(void*, size_t maxCount, int32_t value, size_t count)`；`aclrtMemsetAsync(void*, size_t maxCount, int32_t value, size_t count, aclrtStream)`；成功 0。H6 未被 A12 证伪；仅 memset 的派生仍是单变量。

**A14 当前状态**：本机 A3 注册与合法控制参数已确认（见上）；`MatmulReduceScatter` V1 在 A3 仅 AI CPU；与 Grouped 同 KFC 家族但**对 mc2_server 的依赖仍未证**。A14 本身即“合法 AICPU MC2 控制”：通过 → KFC 路径可用、被拒包非普遍阻断；同挂 → 路径阻塞更广，但仍须证明 A14 与 Grouped 同依赖被拒包。待 31 补本机 `aclnn_matmul_reduce_scatter*.h`、MatmulReduceScatter kernel `.o` 符号串与 `libccl_kernel.so` 符号映射；32 ready、20 排程后执行。

## 13. A14 本机证据核验与支持结论（2026-11-19）

31 `a14-preflight` 包（`Clef-27B/logs/overnight-20261119-owner31/grouped-deepdive/a14-preflight/`，6 文件 + SHA256SUMS，SUMS 文件 `21dcae08…`，6/6 OK）经 33 独立核：

- 本机 V1 头 `include/aclnnop/aclnn_matmul_reduce_scatter.h` SHA `ad8f98b4…`（3,158 B）；v2 头 `055d0ec7…`；level2 版 `a377b21b…`/`7b0745e3…`。**本机头与上游 tag `6e49c20f…` 不同，编译以本机头为准**；签名与 35 `run_mrs` 逐项一致：`GetWorkspaceSize(x1,x2,bias,group,reduceOp,commTurn,streamMode,output,ws,exec)`、`launch(workspace,size,executor,stream)`。
- `libccl_kernel.so`（`aarch64-linux/devlib/device/`，26,274,080 B，SHA `2281deca…`）导出 7,626 个 RunAicpu/Hccl/Mc2 定义符号，含 `RunAicpuKfcClearOpRes/ResInit/ResInitV2/SrvLaunch`、`HcclKernelEntrance`、`HcomMc2AiCpuStreamAllocAndGet`、`HcclMc2Com*`；**`Mc2Server`/`RacServer` 计数 0**。
- A3 `MatmulReduceScatter` kernel `.o` 两个变体（`30500ca7…` SHA `b4819757…`、`3d3ce88a…` SHA `da0cf9db…`）符号仅 `AscendC::AssembleHcclMsg`/`AclnnHcclMsg`/`g_hcclContextReserved` 等 HCCL/KFC 家族，**无 Mc2Server/RacServer**——与 Grouped 同族。
- AICPU kernel 目录仅 6 项（`aicpu_hccl` d98036b1、`aicpu_hcomm` 37cc9e63、`cann-hixl-compat` 05a18e5a、`libaicpu_custom.json/so`、`mc2_server.tar.gz` 71263835），**无其它 mc2* 变体**。
- `LoadSysOpKernel: cannot find aicpu packages` 在 A12 pid 1301080 与健康服务 1301295 启动 plog 均出现 → **常见启动行，非失败特异**，从嫌疑列表移除。

**结论**：A14（MatmulReduceScatter V1）的 A3 注册、本机头、kernel 与 libccl_kernel KFC 家族均成立，是合法的 AICPU/ KFC 路径控制；但 Grouped 与 A14 对 `mc2_server` 包的必要依赖均**未被证明**，且两 kernel 符号层面均无 Mc2Server/RacServer。A14 通过 → KFC 路径可用、被拒包非普遍阻断（重心转 Grouped 特定/H6/H12）；A14 同挂 → KFC 路径阻塞更广，但仍需单独证明被拒包为原因。**不得由 A14 单独裁决 mc2_server 依赖**。
执行门（未执行）：35 preflight `351679f`/manifest `7bc0f155`（binding `cb4430de…`、run_probe `331f9929…`、window `2202165e…`、CPU 44/44）；31 编译/plan（小臂 M=128/K=256/N=512，再 M=968/K=8704/N=5120，output 484×5120）→ 32 ready → 20 排程。A12-H6 同样等 31 编译/plan 与 20 GO。A13 停跑；服务 aa3e/s1 root 1301295 在线；驱动/固件兼容仅记录（V1 文档要求配套 ≥8.0.RC2，version.info compatible_fw `[7.0.0,9.9.9]`），不改白名单强载。

## 14. H6 执行结果与 A14 small v2 plan 核验（2026-11-19）

**H6（workspace 初值）已执行并证伪**：`a12-h6-cell` 10 文件 + SUMS 10/10（SUMS `e88d9d23…`）。child stdout 两 rank 均 `h6 workspace memset api=aclrtMemsetAsync rc=0`，随后仍 `507046`（h6.cpp:392）+ `TestCall execute AICPU failed`；plog-debug（pid 1304920）两 device 均 `stream_id=47, KERNEL_AICORE, task_id=2, 10s timeout, InnerCode=0x7030010`，无 costtime；window `ok=false`。恢复 `branch=aa3e-noop`、root **1305145**、五 PID/四检/env/joint/clients 全过。→ **H6 对该挂起削弱/证伪**（非平台否定）；与 mc2_server 依赖分开。

**A14 small v2 plan（plan-only，未执行）**：`a14-preflight/SHA256SUMS` `2e1086eb…`，13 项 0 FAILED；`plan-a14-small-v2.json` SHA `18fdf185…`，绑定 root **1305145**（starttime `398845277`，identity `4ace4a79…`），child argv 为 `torchrun --nproc_per_node=2 --master_port 29531 run_probe.py --arm A14 --k 256 --n 512 --rows 128 --seed 20261119 --arm-timeout 240 --comm-timeout 90 --adapter probe_adapter.so --execute`，status `plan_not_executed`；stage-identity-v3（`50b659a6…`）记录 commit `e0b06df`、manifest `1c8d2682…`、binding `101660f8…`、run_probe `6ae866b1…`、window `2202165e…`、adapter `d181b15a…`、compile log `7ce25dad…`；旧 1301295 的 small/large plan 已标 superseded，large 等 small 结果与 32 复核。待 32 GO 后由 31 在小臂执行。

**32 A14 small GO（2026-11-19）**：语义确认——window 以 `start_new_session=True` 启动 child=torchrun（session/PGID leader），`--nproc_per_node=2` 两 rank 同 PGID，身份校验 `pgid==child.pid`，cleanup 精确 `killpg`，240s watchdog + 240s child timeout，child rc0 需两 rank pass，maps 门含 adapter+libhccl/ascendcl，恢复 aa3e-noop 不变。**唯一新增前置**：固定 `--master_port 29531` 执行前须预检无监听（`ss -ltn`/lsof）；若被占用则停报/记录，未经批准不得改 plan 参数或换端口。证据要求：两 rank result.json（api/sync/finite/both/pass、reduced max_abs/bit_equal、group/world、versions）、trace、child 日志、window.json/per-PID maps、plog、完整恢复（五 PID/双 worker/四检/env/joint/clients/branch aa3e-noop）；成功/失败均保留。解释：A14 通过 → KFC 路径可用、被拒包非普遍阻断；同挂 → 路径阻塞更广；均不单独裁决被拒包依赖（仍 supported_but_unproven）。large 等 small 结果并重建新 root plan；A13 不执行；服务 aa3e root 1305145。

## 15. A14 small 首格无效：harness mkdir 竞争（2026-11-19）

首格（plan `18fdf185`）实际执行后失败，33 只读核 `a14-small-cell`（10 文件，SUMS 10/10）：

- `child.stderr`：rank1 `run_probe.py:243 out.mkdir(parents=True, exist_ok=False)` → `FileExistsError: probe-out-a14-small-v2`（rank0 先创建）；torchrun 对 rank0 发 SIGTERM（exit -15）。
- `child.stdout` 仅 `ERR99999 UNKNOWN application exception`；rank0 result 仅 stub、trace 空；**无 507046/无 stream_id=47、未进入 MC2 调用**。
- window `ok=false`；restore `branch=aa3e-noop`、root 1306695、config s1。该格**不作为 A14 判据**，证据保留为 invalid。

**最小修复（35）**：`run_probe.py` 共享父目录改为 `out.mkdir(parents=True, exist_ok=True)`；每 rank `rank_dir = out/f'rank{rank}'; rank_dir.mkdir(exist_ok=False)` 保持不变作 fresh 门。33 评估：并发 `exist_ok=True` 建父目录安全（EEXIST 被吸收），无需 rank0/rank1 marker/barrier；per-rank `exist_ok=False` 仍能拒绝陈旧 rank 目录。重跑须用新 probe-out 路径（不复用 v2 目录）、新 run_probe SHA、重冻 stage identity/manifest、以恢复后新 root identity 重出 small plan；adapter binding 未变可复用；执行前预检 master_port 29531 空闲。H6 结论与依赖 `supported_but_unproven` 不变。

**修复核验（33，35 分支 `ao/resume-cc/grouped-mmar-research-20261119`）**：commit `d571641`/`f087746`；`run_probe.py` SHA **`b7f255cb…`**（20,278 B），diff 仅新增 `prepare_out(exist_ok=True)` 与保留 `rank_dir.mkdir()`（fresh 门），无其它语义改动；binding `101660f8…`、window `2202165e…`、manifest `02fdd342…` 与 32/35 报一致；`test_probe_cpu.py` SHA `ea514245…`（7,708 B）含新 `test_out_dir_shared_by_two_ranks`，**实际 14 项**。发现 metadata 计数过期：manifest `cpu_tests.suites.test_probe_cpu.py` 仍为 `13/13`、role 文本仍写 `tests (11)`，与实际 14/14（总 45/45=14+9+22）不一致；请 35 在 v3 metadata 修正（不阻塞，32 已独立跑 45/45）。A14 v3 仍待 31 以 root 1306695 重 stage/重出 plan/端口预检，32 复核后 GO。

**A14 small v3 最终 tuple（32 GO，33 只读核一致）**：commit `f087746` / manifest `02fdd342`（与 d571641 代码逐字节相同）、binding `101660f8`、run_probe `b7f255cb`、window `2202165e`、adapter `d181b15a`、compile log `fc8efe47`；root identity **1306695 / starttime 398875478**（`a85ff9b8…`）；plan **`8782537a…`**（M128/K256/N512、KEEP_ALL、`out=probe-out-a14-small-v3`、run/restore v3 fresh、240s、`plan_not_executed`）；stage-identity-v5 `a3b12870…`；`a14-preflight/SHA256SUMS` `4eaf108e…` **17/17 OK**；`master_port 29531` 计划时 FREE。早前 `d88250e4` 为 v4 旧 plan，已 superseded。33 无阻塞 finding；按 32 GO，31 执行单格。

## 16. A14 small v3 执行：MRS V1 KFC 路径可用；release harness 缺陷（2026-11-19）

33 只读核 `a14-small-v3-cell`（16 文件，SUMS 16/16）：

- rank0 trace 阶段：`prepare_return` rc0、workspace 17,170,432、executor_nonnull=True；`handler_enter`/`api_return`/`launch_return`/`handler_return` 全部出现；`sync_return sync_ms=18.66017`；`python_exception` 在 sync 之后。rank1 同（sync_ms 8.16）。**plog/device 无 `stream_id=47`、无 507046**。
- `rank0/1 result.json`：`api_returned=true`、`sync_returned=true`、`finite=false`、`pass=false`；`error` 为 `NotImplementedError: ... no fallback function is registered for schema clef_mmar_probe::release`（sync 后 `run_probe.py:353 sync_and_release → release(run_id)` 零 Tensor 参数 dispatch）。result 已含 inputs/libraries/provider/versions/trace，但 `reduced/reference/bit_equal` 未计算。
- window `ok=false`；restore `branch=aa3e-noop`、新 root **1308632**（恢复中/已启动）。

**结论**：A14（MatmulReduceScatter V1，A3 AI CPU）在小形状 **prepare/launch/sync rc0、无 stream47/507046** → **MC2 AICPU/KFC 公共路径在本机可用；被拒 `mc2_server` 包并非普遍阻断**。Grouped 仍 507046/stream47，差异转向 Grouped-specific（op/kernel/tiling/rac_server 协议/上下文）与 H12/依赖证据；`mc2_server` 依赖仍 `supported_but_unproven`，不因 A14 单独裁决。A14 v3 的 liveness 有效；完整数值（reduced/reference/bit_equal）因 release harness 缺陷未取得，需修复后同格重跑。

**release 最小修复建议（35）**：① 给 `release` schema 增加一个占位 Tensor 参数（如 `release(Tensor token, int run_id)`，实现忽略 token；可传任一 PrivateUse1 tensor），或注册 `Library.fallback`/PrivateUse1 fallback；② 先计算并原子落盘完整 result.json（api/sync/finite/reduced/reference/bit_equal/versions/trace），再调用 release，并将 release 包 try/except 记录 `release_rc/release_error` 后重 flush；③ 重冻 run_probe 新 SHA、stage identity/manifest；31 用恢复后新 root（1308632）重建 identity/plan v4、fresh out（不复用 v3）、预检 29531，重跑同小格取完整结果；large 等完整 small + 32 有限核。

**归档核验（33）**：`a14-small-v3-cell/SHA256SUMS` SHA `88b823a4…`，16/16 OK；commit `064d469`（17 文件，message “A14 small v3 MC2 success with harness release failure”）；window `c291b877…`；rank0 result `33410194…`/trace `1fb6bf87…`、rank1 result `300eebfd…`/trace `7e7b05a1…`、child stdout `cb1aa6ed…`/stderr `37177294…`、restore manifest `5deee436…`/joint `861044f4…`；恢复 `branch=aa3e-noop`、root **1308632**/start 398926792、五 PID、四检全 true、无残留 probe。A14 v3 收口为“liveness 成立 + release harness 待修”；完整数值重跑门见上。

**顺序根因（33 读 `b7f255cb` run_probe.py 约 L346–360）**：`sync_return` 后立即调用 `release`，之后才 `reduced = torch.matmul(x_n, w_n)`、`dist.reduce_scatter_tensor(reference, ...)` 与 `finite`；release 的 zero-Tensor dispatch 异常因此吞掉了全部数值/reference。**最低风险修复只需改 run_probe、不动 binding/adapter**：把 release 移到 reference/finite/gate 之后；先原子写完整 result.json；再 `try/except` 调 release 并记录 `release_rc/release_error`、重写 result；数值门只看 api/sync/finite/reference，release 失败记为 harness_error，子进程退出后由 teardown 回收（单格一次性，服务已 d0_stop）。若需“release 成功”作为门，再选做 schema 占位 Tensor/fallback（需重编 adapter、新 SHA/compile log）。这样 d181b15a 可继续复用，仅 run_probe SHA/stage identity 变化。

## 17. Grouped-specific/H12 对照：plog 反证、provider/KFC/rac_server、H11 合法性与 v4 阻断（2026-11-19）

**A12/A14 plog：同 chip:5 skip `mc2_server`，A14 仍成功**

- A12 `a12-cell/plog-run-1301080.log`：device0 L84–86，`06:24:20.502.378` begin load → `06:24:20.502.380` `[package_loader.cpp:454][SupportLoadPkg] current device chip:5 does not support package:mc2_server.tar.gz` → `.381` does not need to load；device1 L168–170，`06:24:20.927.134/.136/.137` 同文。
- A14 v3 `plog-run-1308495.log` L91–93，device0 `07:11:21.905.175/.177/.178`；`plog-run-1308496.log` L91–93，device1 `07:11:21.987.154/.156/.157`；均同 `chip:5` skip。
- 结论：skip 是 host loader 的同一行为，A14 在同样 skip 下 prepare/launch/sync rc0 → **skip 不是 MC2/KFC 路径阻塞的充分原因**；Grouped 是否唯一依赖被拒包仍 `supported_but_unproven`。
- **v4r1 plog 缺口已补**（31 `grouped-deepdive/v4r1-plog/`，6 文件，SUMS `a471356e…` 6/6）：device0 `plog-run-1255361.log` L42 `00:37:06.899.567` insert config → L91 `00:37:07.016.986` begin load device0 → L92 `.988` `chip:5 does not support mc2_server.tar.gz` → L93 `.990` no load；device1 `plog-run-1255362.log` L42 `00:37:06.821.836` → L91 `.939.528` begin device1 → L92 `.531` chip:5 skip → L93 `.532` no load。**同一 plog**：device0 L409 `2026-11-19-00:40:09.445.383 [stream.cc:1996] report three minutes timeout! stream_id=47, sq_id=2, task_id=36, sqHead=35, flip_num=0, pendingNum=2`；device1 L388 `00:40:09.434.413` 同文。device0 日志 L111 另有 `ATRACE ... create object KFCtaskAndTilingTraceData success`（08:37:09.388），说明 KFC 任务/ tiling 跟踪对象已创建，但 AICORE 任务仍 pending。
- 三路对照：**A12（Grouped FP16，官方）skip + stream47 pending；v4r1（Grouped BF16，自定义）skip + stream47 pending（task_id=36，3 分钟）；A14（MRS BF16）skip + prepare/launch/sync rc0**。同一 host skip 下结果按 op 分化 → **skip 不是充分原因**；Grouped-specific（op/kernel/tiling/rac_server/context）为优先候选，必要依赖仍 `supported_but_unproven`。

**provider/加载路径（事实）**

- A14 `rank0-result.json` provider：prepare/launch=`/usr/local/Ascend/cann-9.2.1/aarch64-linux/lib64/libopapi.so`，metadata=`libnnopbase.so`；adapter `d181b15a`，`clef_mmar_probe::run_mrs` 经 `TORCH_LIBRARY_IMPL(..., PrivateUse1)` 派发。
- H11/A0 用同一 adapter 的 `run(Tensor x, Tensor weight_kn, Tensor bias, str group, int flags)`，仅 aclnn 符号不同 → A14 与 H11 provider/加载路径匹配。A12 是直接 aclnn C++ 样例（同 libopapi/nnopbase，host 启动路径不同）。v4r1 用旧 adapter `23d9ab07`/binding `5db40ef9`/runner `dde68f34`，与当前 harness 在 op 与 harness 两代上不同，不能直接当 matched 对照。

**KFC/rac_server/context（事实/推断）**

- 事实：Grouped tiling 显式 `taskType=KFC_TASK_HCC_TASK_DELIVER`、`funID=ALL_REDUCE_FUNC_ID`、`commType=HCCL_CMD_ALLREDUCE`、`commOrder=1`、`commAlg=0`、`preparePosition=0`；kernel 含 `HcclServer`、`AivAicpuOpParam`、`rac_server.h`、`GetHcclContext<0>()`。
- 事实：A14 tiling 用 `Mc2CcTilingConfig(groupName, opType, rsConfig, 0)` 生成 `mc2InitTiling/mc2CcTiling`；kernel `GetHcclContext<HCCL_GROUP_ID_0>()` + `MatmulReduceScatterFullMesh`；本地 kernel main/base 无 `HcclServer/rac_server/AivAicpuOpParam`。
- 推断：Grouped 走 KFC server/rac_server 任务投递协议，A14 fullmesh 走标准 MC2 client 路径；A14 fullmesh 的 server 依赖未证（fullmesh 头未在本地归档，缺口）。context 由 runtime/HCCL 管理，两 kernel 均读 reserved context；精确 libhccl 初始化调用链未由本地符号证明。

**H11 matched Grouped 小格（32 设计）合法性**

- binding `run` 事实：x BF16 `[M,K]`、weight BF16 `[K,N]`、bias **FP32 `[N]`**、group 非空；aclnn `splitItem=0`、`groupList=null`、`comm_turn=0`、`streamMode=1`；docs 允许 BF16 x/w + FP32 bias、splitItem0 的 2D x/y、groupList 必须为空、32B 对齐上限。**M128/K256/N512 合法**。
- H11 `flags=KEEP_ALL`，同 adapter `d181b15a`/binding `run`、同 torch 多进程 comm、240s → 正确 matched 控制。A8 是 torch matmul+all_reduce（`native=false`），只能作 reference/comm health；A0 是同 Grouped `run` flags=0，可复用其 CLI/参数化，但 KEEP_ALL 匹配选 H11。
- 显式固有差异：A14 null bias vs H11 FP32 zero bias；output reduce_scatter vs all_reduce；op MRS vs Grouped。
- 优先单变量：A14 small vs H11 small，仅 op 变化（其余 adapter/comm/M/K/N/BF16/KEEP_ALL/240s 全同）。H11 同挂 stream47/507046 → Grouped op/kernel/tiling/rac_server/context 为差异源；返回 → 转 M967/K8704/N5120 shape/tiling；prepare 失败 → 记录形状约束并最小调整一维保持两臂匹配。顺序按 32：A14 v4 完整 small → H11 small fresh plan → large。

**v4 release 修复的执行阻断（33 读 `fdbfeb67`/commit `5ada848`）**

- 已正确：result 先 `flush()` 再 release（L389–392）；`call_release` try/except 记录 `release_rc/release_error`；`numeric_pass` 只看 api/sync/finite/reference。
- 阻断：`meta['pass']=numeric_pass and harness_release_ok`（L413）、`return 0 if pass else 1`（L420）、`destroy_process_group()` 仅在 pass 下。release schema 仍零 Tensor，v4 未改 binding/adapter，dispatch 必抛 `NotImplementedError` → `harness_release_ok=False` → `pass=False` → **child rc1、window child_ok false**；v4 会再次“数值完整但 harness 判失败”。
- 二选一：①（推荐，run_probe-only）`return 0 if numeric_pass else 1`，`pass=exit=numeric_pass`，另存 `harness_pass=numeric_pass and harness_release_ok`；`destroy_process_group()` **在 finally 中以 `dist_ready=True`（init 成功后置位）覆盖所有已初始化 rank 的成功/异常路径**，不新增 barrier/其他 collective（32 按 20 修正：不能 `if numeric_pass`，否则单 rank 数值不同会单侧跳过）；②修 schema 占位 Tensor/fallback（需重编 adapter），再让 pass 含 release。32 的“numeric 只看 api/sync/finite/reference”与 35 当前 `pass` 语义冲突，需先统一再给 v4 GO。

**v4 重冻核验（35 `f3e76cc`，33 只读核 PASS）**：commit `f3e76cc`，manifest **5531e104…**，run_probe **0f90fd23…**（21,854 B），test_probe **a8bf075f…**（9,091 B，15 项，总 46/46），binding `101660f8…`、window `2202165e…` 未变，adapter `d181b15a…` 复用不重编；H11 matched 文档 `1294559d…`。代码核：先 `flush()` 完整 result 再 `call_release`，`release_error/harness_release_ok` 记录并重 flush；`both=all_gather_object` 后 `numeric_pass=api+sync+finite+all(both)`（norm 时含 gate）；`harness_pass=numeric_pass and harness_release_ok`；`pass=numeric_pass`；`trace done` 同时写 passed/harness_release_ok；`return 0 if numeric_pass`；`finally` 中 `dist.is_initialized()` 时 destroy，cleanup 异常写 `harness_cleanup_error` 并 flush，无 finally return，不覆盖退出码。→ v4 可执行；31 从 `f3e76cc` stage v7、root 1308632 重建 plan v5/fresh out/port 29531，32 增量核后 GO。

**v5 plan/stage-v7 核验（33）**：`plan-a14-small-v5.json` SHA `b6429191…`，status `plan_not_executed`；内嵌 root `1308632/398926792`，out **`probe-out-a14-small-v5`**、temp v5、adapter 指向 `grouped-deepdive-35-f3e76cc/.../probe_adapter.so`（`d181b15a…`）、restore out `restore-a14-small-v5`、240s；`root-identity-1308632.json` 868 B、SHA `eb3a4b42…` 与 stage-v7 一致（先前 0 字节为写入中间态，已消失）。**stage-v7 已修正并重出**：SHA `06b7c600…`，`preconditions.probe_out_v5="absent at plan time (verified)"`，`execution_plan="plan-a14-small-v5.json b6429191 (20 final); plan v4 6da7309c superseded_not_executed"`；`a14-preflight/SHA256SUMS` `101fff8983…` **22/22 OK**；plan v4 `6da7309c…` 保留未执行。32 已对 v5 重发 GO（唯一执行目标）；31 已启动单格；v4 不执行。20 报告 v5 已完成 numeric（两 rank numeric_pass/both_finite、max_abs 0、bit_equal true），恢复 root `1311514`，待 31 归档路径/SHA 后只读核 raw。

**运行期协议痕迹（33 新发现，Grouped-specific 优先候选）**：
- A12 `device-0-1301080.log` L85/L88（`14:24:23.281.896`/`.282.289`）、`device-1` L85/L88（`14:24:23.286.442`/`.286.859`）创建 `KFCtaskAndTilingTraceData` + `AivAicpuOpParam`。
- v4r1 `device-0-1255361.log` L111/L114（`08:37:09.388.216`/`.388.620`）、`device-1-1255362.log` L111/L114（`08:37:09.380.528`/`.380.931`）同样创建这两个对象。
- A14 `device-0-1308495.log` L88/L102/L103/L107（`15:11:23.398.524`/`.405.414`/`.405.535`/`15:11:24.014.681`）与 device1 对称，加载 `RunAicpuKfcResInitV2`、`AIKernelsLibAiCpuKFC init finish`、`RunAicpuRpcSrvLaunchV2`、`RunAicpuKfcSrvLaunch`；但 A14 全部 device/plog **0 个** `KFCtaskAndTilingTraceData`/`AivAicpuOpParam`。A14 ATRACE 同期有 `RUNTIME_ATRACE_*`/`HCCL_*` 对象，说明 ATRACE 未关闭。
- 事实：三者共享 libccl_kernel KFC/RPC API 初始化；Grouped（A12/v4r1）额外走 `AivAicpuOpParam`/KFC task 投递，A14 不走该对象路径。推断：A14 成功**不验证** Grouped 的 rac_server/AivAicpu 协议；Grouped hang 的优先候选进一步收敛到该协议/kernel。仍非证明：A14 fullmesh 源未归档，无法确认其确切 server 路径；`mc2_server` 必要依赖仍 `supported_but_unproven`。

**H11 文档 metadata 不一致（33）**：35 `h11_matched_control.md` SHA `1670bc9b…`（commit `6ae7521`，manifest `1d1fa265…`）内容合同正确，但仍写 “A14 small v4”、`run_probe fdbfeb67`、`probe-out-a14-small-v4`；最终为 plan v5 `b6429191`、run_probe `0f90fd23`、out/run/restore v5。属 metadata-only，请在 H11 fresh plan 前改为 v5/0f90fd23 字面，避免 H11 证据引用旧 tuple。

## 18. A14 v5 原件、libraries/provider 来源、window maps 缺口与 H11 plan 冲突（2026-11-19）

**A14 v5 原件核（33）**：`a14-small-v5-cell` 16 文件，SUMS `5524942b…` 16/16；rank0 result `bed2e1c6…`/trace `4ce37e08…`、rank1 result `0c81f74a…`/trace `8b62c55f…`，与 31/32 报一致。两 rank `numeric_pass=true`、`pass=true`、`harness_pass=false`、api/sync/finite/both=true、`reduced.max_abs=0.0`、`bit_equal=true`、group `group_name_0`、versions torch 2.10.0+cpu / torch_npu 2.10.0.post7；`release_rc=null`、`release_error=NotImplementedError`；两 rank plog `mc2_server` skip：`plog-run-1311385.log` L91-93 `07:36:44.586.048/.050/.052`（device0）、`plog-run-1311386.log` L91-93 `07:36:44.446.045/.047/.048`（device1）；无 stream47/507046；恢复 `aa3e-noop`、root **1311514**、config s1。window `ok=false` 仅 maps 硬门。

**libraries/provider 实际来源（20 重点）**：`run_probe.py 0f90fd23` 的 `library_evidence` 读取 **rank 进程 `/proc/self/maps`**，筛选已加载库路径后对磁盘文件算 SHA；rank0 result 的 `libraries` 含 `libascendcl 1c06b9c4…`、`libhccl da6dd17f…`、`libhccl_alg 266289dc…`、`libhccl_legacy 741553bb…`、`libhccl_v2 7965ae02…`、`libhccl_plf 1e99f702…`、`libnnopbase 72cd43b4…`、`libopapi eb1c788e…`、`libopapi_transformer 34baeae3…`、`libtorch_npu 65e6be19…` 以及 adapter `d181b15a…`（`adapter` 字段同 SHA）——**是进程已加载映射，不是纯磁盘扫描**。`provider` 由 binding `grouped_mmar_provider()` 返回 `canonical(OPAPI)+canonical(META)`（`realpath` 硬编码路径 `/usr/local/Ascend/cann-9.2.1/aarch64-linux/lib64/libopapi.so`、`libnnopbase.so`）；但 `symbol()` 对每个 aclnn 符号执行 `dladdr` 并校验 `dli_fname` 的 canonical 路径等于期望（否则 TORCH_CHECK 失败）——provider 字符串本身是 canonical 路径，**实际符号归属由 dladdr 校验背书**。

**window maps 缺口根因（33 读 `2202165e`）**：`maps_evidence` 在 `child_started` 步骤（L534）立即采样；此时 torchrun 父 PID 1311355 已存在，rank 1311385/1311386 尚未 spawn/加载 adapter，故 `maps` 仅父、`binary_pids=[]`；`required_map_ok` 要求 adapter 名出现在 maps → maps_ok=false。窗口之后只 `child.wait`，不再重采样。per-rank `result.libraries` 来自 rank 内 `/proc/self/maps`，已覆盖 adapter+libhccl/libascendcl 等，故 v5 `ok=false` 是**采样时序伪影**，不是数值/加载缺失。最小修法：acceptance 以 per-rank `libraries`+provider 作为 maps 证据并标注 `maps_sampling_artifact`；未来若要窗口硬门，需在 rank 运行期轮询 PGID maps（窗口代码改动/新 SHA），本次 H11 不重 plan。A14 未验证 Grouped rac_server/AivAicpuOpParam 协议；日志对象缺失只收窄控制覆盖，不当作未执行证明。

**H11 plan 已收敛（20 确认，不再作冲突）**：31 当前唯一 plan 为 `plan-h11-small-v1.json` SHA **e159c897…**（out `probe-out-h11-small-v1`、run/restore `h11-small-v1`、arm H11、M128/K256/N512、KEEP_ALL、240/90、plan_not_executed）；stage-identity `e45db9da…` 指向 e159c897；`plan-h11.json` **68cc5a79…**（out `probe-out-h11`）保留为 `superseded_not_executed` 修订轨迹，不执行、不再作冲突/metadata 阻断。H11 SUMS `8f781003…` 4/4、root identity `9802c322…`（1311514/399079056）、port 29531 FREE、adapter d181b15a、冻结 f3e76cc/5531e104/0f90fd23/a8bf075f/101660f8/2202165e 不变；32 核最新 plan 后 GO 一次，不重 plan。

**Grouped 运行时 AICPU 协议痕迹计数（33 新增，v5 对齐后）**：A12 与 v4r1 各 device 均 2 个 `AicpuComTraceData`、2 个 `SqeBatchInfo`、2 个 `End AicpuRpcResInit`、2 个 `RunAicpuKfcSrvLaunch`、2 个 `KFCtaskAndTilingTraceData`、2 个 `AivAicpuOpParam`；A14 v5 为 **0/0/0**（AicpuComTraceData/SqeBatchInfo/End AicpuRpcResInit/KFCtask/AivAicpu 均 0），仅 `RunAicpuKfcSrvLaunch` API 加载 2 次，以及 `AicpuRpcResInitV2`（device-0 15:36:45.857.863）。Grouped kernel 源码用 `HcclServer` + `AivAicpuOpParam`（commType/opType/send/recv/count/stride/valid/funID/轮次/响应），`workspaceMsg=context->WorkSpace+notifyOff`；A14 无该对象路径。**事实**：A14 的 ATRACE 活跃且记录 RPC V2，但不创建 Grouped 的 KFC/AivAicpu 对象；**推断**：A14 走不同 MC2/RPC 路径，未验证 Grouped 的 rac_server/AivAicpu 协议。**不把日志对象缺失当作未执行证明**；需 H11 raw 与（若可行）kernel 侧证据进一步判别。`mc2_server` 必要依赖仍 `supported_but_unproven`。

**v5 有限结论（33）**：A14 v5 两 rank numeric 完整（bit_equal、max_abs 0）、provider/libraries 证据充分（rank 内 `/proc/self/maps`）；harness_pass=false 为零 Tensor release 的已知缺陷，window.ok=false 为 maps 采样时序伪影；均不否定 numeric。A14 只证明 MRS V1 控制路径可用，不证明 Grouped rac_server/AivAicpu 路径；Grouped 仍 stream47 pending，Grouped-specific 候选优先。

## 19. 同版源码链：Grouped KFC/AivAicpu vs A14 Hccl 客户端（2026-11-19）

**固定 commit 源码核（f5703e4）**：Grouped kernel `op_kernel/arch22/grouped_mat_mul_all_reduce.cpp` SHA `7f72cde4…`（3,795 B，=本地 `/tmp/g29_kernel.cpp`）、tiling `op_host/op_tiling/grouped_mat_mul_all_reduce_tiling.cpp` `5e87411c…`（33,327 B，=本地）、def `13d62719…`（3,082 B，=本地）、`op_kernel/rac_server.h` `49ad1cf6…`（11,097 B，=本地）、kernel header `d4f0c27e…`（7,745 B）。A14 main `op_kernel/matmul_reduce_scatter.cpp` `16f8edab…`（4,024 B，=本地）、tiling `936f9cb8…`（39,300 B，=本地）、arch22 a2a3 `9b3be18f…`（3,604 B，=本地）、fullmesh `02d06d2e…`（12,302 B）、base `0637bbd9…`（3,407 B，=本地）、`mc2_nd_to_nz.h` `092d9447…`、`mc2_matmul_compute.h` `2f62d849…`。

**A14 路径（事实）**：main `GetHcclContext<HCCL_GROUP_ID_0>()`（L63）取 context，传入 fullmesh；fullmesh 持有 `Hccl<HCCL_SERVER_TYPE_AICPU> hccl_`（L66），`hccl_.Init(contextGM, mc2InitTiling)`（L77）、`hccl_.SetCcTiling(mc2CcTiling)`（L78），`hccl_.ReduceScatter<false>` 下发 tile/尾块任务（L108/L112），`hccl_.Wait(handle)` 等待（L142/L145），`hccl_.Finalize()`（L263）。tiling 用 `Mc2CcTilingConfig(groupName, opType, rsConfig, 0)` 生成 `mc2InitTiling/mc2CcTiling`（L642/643/652），`cmdType=HCCL_CMD_REDUCE_SCATTER`（L537/620/771），arch22 `MMPlusReduceScatter(..., KernelType::REDUCE_SCATTER, DAV_2201)`（a2a3 L34）。**没有 HcclServer/AivAicpuOpParam 手动协议。**

**Grouped 路径（事实）**：kernel `HcclServer hcclServer`（L73）、`context=GetHcclContext<0>()`（L74）、`workspaceMsg=context->WorkSpace+notifyOff`（L76）、`op.Process(hcclServer)`（L94）；tiling `commType=HCCL_CMD_ALLREDUCE`（L412）、`taskType=KFC_TASK_HCC_TASK_DELIVER`（L419）、`commOrder=1`（L420）、`funID=ALL_REDUCE_FUNC_ID`（L421）、`workspaceOff=libApiWorkSpaceSize_`（L452）、`notifyOff=sizeof(KFCMsgBody)`（L454/527）。`rac_server.h` 的 `AivAicpuOpParam`（L100-120）含 commType/opType/send/recv/count/stride/hcclDataType/valid/isLast/funID/sendCnt/rcvCnt/everyTurnRsp/everyTurnWait/totalTurnCnt；`HcclServer` 用 `msgSndWorkArea/msgRcvRspArea`（L129/130）、`TurnNotifyRunSetMsg` 写 valid/sendCnt（L162-168）、`TurnWait` 轮询 `msgRcvRspArea->rcvCnt>=totalTurn`（L186-207）。这是 AIV→AICPU KFC 消息轮次/响应协议。

**RPC 版本对应（事实/推断）**：运行日志中 Grouped（A12/v4r1）为 `RunAicpuKfcResInit`、`End AicpuRpcResInit`、`KFCtaskAndTilingTraceData`、`AivAicpuOpParam`；A14 v5 为 `AicpuRpcResInitV2`、`RunAicpuKfcResInitV2`、`RunAicpuRpcSrvLaunchV2`，且无 KFC/AivAicpu 对象。**事实**：V1/V2 API 名称与日志路径不同；**推断**：Grouped 对应 V1 KFC/AivAicpu 协议、A14 对应 V2 HCCL 客户端路径。**缺口**：`libccl_kernel.so` 的 AICPU RPC 源不在固定上游/本地归档，V1/V2 到具体 server 实现的映射只能由日志/符号推断，未由源码证明。

**约束与缺口**：协议/tiling 必须成对：`KFC_TASK_HCC_TASK_DELIVER`+`ALL_REDUCE_FUNC_ID`+`HcclServer/AivAicpuOpParam`+context WorkSpace+notifyOff 是一套；`HCCL_CMD_REDUCE_SCATTER`+`Mc2CcTilingConfig`+`Hccl<HCCL_SERVER_TYPE_AICPU>` 客户端是另一套。A14 只验证后者；Grouped 的 V1 KFC/AivAicpu 未被 A14 覆盖。日志缺失不作“未执行”证明；固定上游源码是行为参考，不保证与本机 CANN 9.2.1 安装二进制逐字节同源。

**Grouped 任务投递细节（33 新增）**：official `grouped_mat_mul_all_reduce.h`（`d4f0c27e…`）`GMMProcess::Process(HcclServer&)` L103-142：按 group/totalRun 循环，各 core 调 `hcclServer.TurnNotifyRun(coreIdx, numBlocksN*splitM, i+1)`（L134），core0 在所有轮次后 `hcclServer.TurnWait(totalRun)`（L139）；`rac_server.h` `TurnWait` L186-207 轮询 `msgRcvRspArea->rcvCnt >= totalTurn`，满足后重置 `rcvCnt=0/valid=~AC_MSG_VALID_MASK` 并 DCCI。即 Grouped 的正确完成依赖 AICPU 侧对每轮 `AivAicpuOpParam` 的响应计数推进；H11 `task_id=13 pending` 与该等待边界相容，但不是根因证明。**缺口**：`HcclCombinOpParam` 定义在 CANN kernel headers，未在本地/固定上游归档（仅见 `context->WorkSpace` 使用）；`RunAicpuKfcResInit`/`AicpuRpcResInit`（V1）与 `RunAicpuKfcResInitV2`/`AicpuRpcResInitV2`（V2）的 AICPU 实现源码不在 `libccl_kernel.so` 归档/上游，V1/V2 到 server 实现的映射只能由日志与符号推断。

## 21. H5K/H4K 合同核与 Grouped producer/consumer 精确事实（2026-11-19）

**H5K/H4K freeze 合同核（33 PASS）**：commit `0d3f7fb`；manifest `21758ff5…`；run_probe `508e2931…`（22,275 B）；window `d7e73514…`（31,650 B）；binding `101660f8…`（34,645 B）**未变**；test_probe `89acbb63…` 16 项、test_window `0fcdf991…` 23 项、cell 9 项 = **48/48**。diff 仅 run_probe +4 行（`H4K flags=KEEP_ALL|DIRECT_LAUNCH=24`、`H5K flags=KEEP_ALL|REFETCH_STREAM=40`、rows=128）、window `--arm choices` 加两名、测试/manifest/README；无其它语义改动。binding `keep_group/keep_workspace/keep_descriptors` 均为 `flags & (bit | KEEP_ALL)`（L178-180），24/40 均含 KEEP_ALL → 保活与 H11 一致；H5K 对 H11 仅 handler 内 current stream 重取（仍 OpCommand 线程、handler 无同步、Python sync 真实边界）；H4K 调用线程+真实 `aclrtSynchronizeStream`（线程+同步组合，非单变量）。shape/dtype/comm/bias/240s/恢复/maps/release 逻辑未动。

**Producer/consumer 精确事实（f5703e4 源码）**：

- buffer 来源：Grouped kernel L70 `user1=GetUserWorkspace(workspace)`（GEMM scratch）与 L74 `context=(HcclCombinOpParam*)GetHcclContext<0>()`、L76 `workspaceMsg=context->WorkSpace+aicore_tiling_data.notifyOff` 是**不同区域**。`HcclServer::Init(win)` 将 `msgSndWorkArea=(AivAicpuOpParam*)win`、`msgRcvRspArea=(AivAicpuOpParam*)(win+HCCL_PARAM_SIZE=64)`（rac_server L144-149）。H6 的 `aclrtMemsetAsync(workspaceAddr,...)` 清的是调用方 aclnn workspace/user scratch，**不覆盖 `context->WorkSpace+notifyOff` 的 KFC 消息区**；H6 负结果不能证明该区初值无影响。
- producer（AIC）：`TurnNotifyRun(coreIdx, usedCoreNum, curTurn)`（rac_server L171-184）先 CrossCoreSet/WaitFlag，只有 `blockIdx==0 && g_coreType==AIC` 才 `TurnNotifyRunSetMsg(curTurn)`（L162-169）：写 `msgSndWorkArea->sendCnt=curTurn`，`SetMsg(...AC_MSG_VALID_MASK)` 写 `valid`，`Barrier` 后 DCCI `CACHELINE_OUT`（L135-142）。**AivAicpuOpParam 的 commType/opType/sendBuffer/recvBuffer/count/stride/hcclDataType/funID/everyTurnRsp/everyTurnWait/totalTurnCnt 在 official Grouped kernel/header/rac_server 中没有赋值点**；operation 参数应来自 host tiling/KFC 消息（`KFCMsgBody`：commType/funID/workspaceOff/notifyOff，tiling L412/419/421/452/454/527），但 kernel→AICPU 的具体拷贝/派生路径未在源码中证实——**缺口**。
- consumer（AIC core0）：`TurnWait(totalTurn)`（L186-207）取 `msgRcvRspArea`（send 区+64B），循环 `Barrier + DCCI` 后读 `rcvCnt >= totalTurn`；满足后 `rcvCnt=0`、`valid=~AC_MSG_VALID_MASK` 并 DCCI。期望值来自 `GMMProcess` 的 `totalRun=Ceil(m/(splitM*singleM))`（header L122/L139），每 group 一次；`TurnNotifyRun` 每轮传 `i+1`（L134）。**谁初始化 rcvCnt、谁递增 rcvCnt、响应写入的 cache 可见性由谁保证，均在 AICPU/libccl_kernel 侧，源码缺失**；kernel 侧只保证 send 区写后 DCCI、读 response 前 DCCI，不能证明 AICPU 侧初始化/递增时序。

**ABI 对应（已证/未证）**：已证——f5703e4 上游 Grouped kernel/tiling/def/rac_server 与 A14 main/tiling/a2a3/fullmesh/base 的字节 SHA（见 §19），本机 op-info 注册、kernel `.o` 名称、libccl_kernel 符号与运行日志 API 名存在。未证——本机 CANN 9.2.1 安装的 Grouped kernel/tiling 是否由 f5703e4 源构建、`AivAicpuOpParam`/`HcclCombinOpParam` 字段偏移/大小与上游一致、`KFC_TASK_HCC_TASK_DELIVER`/`ALL_REDUCE_FUNC_ID` 数值与上游一致、V1/V2 RPC 到具体 server 实现的映射。**不能把上游字段/协议直接当本机 ABI 事实**；H5K/H4K 只改变调用线程/stream，不验证 ABI。

**最小资料请求（31，只读）**：① `grep -R "HcclCombinOpParam" /usr/local/Ascend/cann-9.2.1/include /usr/local/Ascend/cann-9.2.1/opp` → 头文件与 SHA（若无则记录 nearest `HcclCombin*`）；② `find .../opp/built-in/op_impl -path '*grouped_mat_mul_all_reduce*'` → 本机 kernel `.json`/`.o`、tiling `.so` 路径/SHA，`strings`/`llvm-objdump -t` 查 `AivAicpuOpParam/KFCMsgBody/sendCnt/rcvCnt/TurnWait`；③ `libccl_kernel.so` 已有路径/SHA，补 `RunAicpuKfcResInit`/`RunAicpuKfcResInitV2`/`RunAicpuKfcSrvLaunch`/`RunAicpuRpcSrvLaunchV2` 的符号大小/反汇编或 `strings`（`AivAicpu/rcvCnt/sendCnt`）；④ `find ... -name 'rac_server.h' -o -name '*aicpu*param*'` 本机安装头。已有归档 `/tmp/aa3e-grouped-api-pack-20261119` 有 op-info/kernel `.o`，无上述头/反汇编，故需补。

**可判别候选**：H5K 返回 → handler 重取 stream/context 绑定为候选（不自动唯一归因 context，先看 stream_compare/reduced）；H5K 同挂 + H4K 返回 → 调用线程+显式同步路径可行，OpCommand 线程/无同步候选；H5K/H4K 均挂 → AivAicpu KFC producer/consumer（rcvCnt 初始化/递增/可见性）优先；现有证据只支持候选，不单格定因。`mc2_server` 必要依赖仍 `supported_but_unproven`。

**H5K preflight 合同核（33 PASS）**：`h5k-preflight/`：plan `f10bfa01…`（arm H5K、flags=40、M128/K256/N512、BF16、FP32 zero、单 group、240/90、out/run/restore h5k-small-v1、plan_not_executed、child `--execute`）；stage `5ff8a4aa…`（commit 0d3f7fb、manifest 21758ff5、run_probe 508e2931、window d7e73514、binding 101660f8、adapter d181b15a、root 1313827/399164390）；root identity `467b7c7c…`；SUMS `04204423…` 3/3；stop root 1313827、restore `restore-h5k-small-v1`、success aa3e、port 18143；port 29531 FREE。H4K 未执行、仅备用；合同无阻塞，32 可核增量后一次 GO。**注意**：H5K plan 不含 window 参数，window 由 supervisor 以 `--arm H5K` 调 d7e73514；证据需同时记录 plan/stage/window SHA。

## 22. ABI 资料核：本机 header/kernel/libccl 对应关系（2026-11-19）

**ABI pack 核（31 `abi-preflight/`，10 文件，SUMS `d7da97f6…` 10/10，commit 951141f）**：

- **本机 HcclCombinOpParam 片段**（`hccl-combin-struct.txt`）：`uint64_t workSpace; uint64_t workSpaceSize; uint32_t rankId; uint32_t rankDim;`。**注意字段为小写 `workSpace`**，而上游 f5703e4 Grouped kernel 源写 `context->WorkSpace`（大写 W）；该片段未注明取自 `ops_transformer` 还是 `ops_legacy` 的 `mc2_common_def.h`（两者 SHA 不同：`9c045a97…` / `a8969b7f…`）。**不能把上游 `WorkSpace` 字段/布局直接当本机 ABI 事实**，需补 header 来源与 struct 原文。
- **本机 rac_server.h**（11,105 B，SHA `f6b286d7…`）与上游 f5703e4 `rac_server.h`（11,097 B，SHA `49ad1cf6…`）**`diff -w` 为空**：差异仅是空格/指针星号格式；`AivAicpuOpParam` 字段顺序、`HcclServer` `TurnNotifyRun/TurnWait/SetMsg` 语义一致。故 rac_server 协议**内容层面**与上游一致（ABI 布局仍需本机 kernel 编译确认）。
- **本机 Grouped kernel 源**：`ops_transformer/.../op_kernel/arch22/grouped_mat_mul_all_reduce.cpp` SHA **`1db26125…`**，上游 f5703e4 `7f72cde4…`；31 补全本机源后 **`diff -w` 为空**（0 行差异），仅空格/指针星号格式，3,795 vs 3,799 B——**明确负结果：无源级语义差异**。本机 kernel `.o`/`.json` SHA `a5d7ec03/8ee67292`、json `da5e54b3/f203a523`；`.o` strings 无 `AivAicpuOpParam/KFCMsgBody/sendCnt/rcvCnt/TurnWait`（设备 ELF 不暴露）。本机无独立 tiling `.so`，配置 `kernel/config/ascend910_93/ops_transformer/grouped_mat_mul_all_reduce.json` SHA `c9bd574a…`（内容已核，见 §23）。
- **libccl_kernel.so**（SHA `2281deca…`）符号/strings：`RunAicpuKfcResInit` 反汇编调用 `AicpuKfcProcess::AicpuRpcResInit(HcclCombinOpParam*)`；`RunAicpuKfcSrvLaunch` 使用 `HcclApi::CommKfcParamDesc`/`AicpuComContext`；strings/symbols 含 `AivAicpuOpParam::PrintMsg`、`AicpuKfcProcess::AicpuCcOpExe(AivAicpuOpParam*,AivAicpuOpParam*,AicpuComContext*)`、`AicpuKfcProcess::AddTaskForHcclMsg`、`AicpuKfcRpcServer::ReadWorkMsg/ReadAddrMsg/CheckRcvWorkMsg/CheckRcvAddrMsg/RcvMsg/GenMsgByTaskParam/HcclMsg2AicAicpuOpParam`、`AicpuKfcUtils::PrintBuffer`、`HcclKFCTilingData.sendCnt`、`rcvCnt/sendCnt Msg` 等。符号大小：ResInit 0x98、ResInitV2 0x238、SrvLaunch 0x10ec、RpcSrvLaunch 0x1704、RpcSrvLaunchV2 0x964。**AivAicpuOpParam 的 AICPU 消费实现就在 libccl_kernel 内，不是只在 mc2_server**；因此 mc2_server skip 不等于删除该消费端，`mc2_server` 必要依赖仍未证。
- **aicpu_init_param.h**（本机，1,137 B）：`CommAicpuParam` 含 `hcomId[128]`、deviceLogicId/devicePhyId/deviceType/userRankSize/userRank、`kfcControlTransferH2DParams`、`kfcStatusTransferD2HParams`；说明 KFC 控制/状态有 H2D/D2H 通道，但未给出 `rcvCnt` 初始化/递增实现。
- **ABI 已证/未证**：已证——本机 rac_server 内容与上游 `diff -w` 等价；本机 HcclCombinOpParam 存在（但来源/大小写待明）；libccl 含 AivAicpu 消费符号/strings；本机 kernel/tiling 文件存在及 SHA。未证——本机 Grouped kernel 源（`1db26125…`）与上游差异内容；本机 HcclCombinOpParam 取自哪个 header、字段偏移/大小与 kernel 编译一致；本机 kernel `.o`/tiling 配置的任务类型/funID 数值；`AivAicpuOpParam` 在本机 kernel 中的布局（kernel .o 不暴露）；V1/V2 到具体 server 实现的映射。
- **最小 31 只读请求（在 H5K 后补，不阻塞）**：① 复制本机 `ops_transformer/.../grouped_mat_mul_all_reduce/op_kernel/arch22/grouped_mat_mul_all_reduce.cpp`（`1db26125…`）全文或 `GetHcclContext/HcclServer/WorkSpace/notifyOff/TurnNotifyRun/TurnWait` 行，并与上游 diff；② 分别给出本机 `ops_transformer`/`ops_legacy` 的 `mc2_common_def.h`（`9c045a97…`/`a8969b7f…`）中 `struct HcclCombinOpParam` 原文（确认 `workSpace` vs `WorkSpace`）及本机 Grouped kernel header/rac_server 编译头；③ 复制本机 tiling 配置 `grouped_mat_mul_all_reduce.json`（`c9bd574a…`）与 kernel `.json` 内容；④ 如可行，补 `AicpuKfcProcess::AicpuCcOpExe`/`AddTaskForHcclMsg`/`AicpuKfcRpcServer::ReadWorkMsg` 的符号大小/反汇编（现有仅 ResInit/SrvLaunch 头）。

**可判别候选（更新）**：H5K 返回 → stream/context 绑定候选；H5K 同挂 + H4K 返回 → 调用线程+显式同步；H5K/H4K 均挂 → AivAicpu KFC producer/consumer 的 `rcvCnt` 初始化/递增/可见性优先。libccl 符号证明消费端存在，不证明其在本机 kernel 路径被正确调用/响应；本机 kernel 源与上游不同是新的 ABI 缺口。H6 仍只清 user workspace，不覆盖 context KFC 区。

## 23. 布局/语义 diff 与通知区（2026-11-19，20 口径）

**负结果（可证实）**：本机 `rac_server.h` 与上游 f5703e4 的 `diff -w` **为空**；`AivAicpuOpParam` 与 `HcclServer`（`SetMsg/TurnNotifyRun/TurnWait`）语义一致，差异仅空格/指针星号。源级布局（LP64、无 pragma pack）：`AivAicpuOpParam` 总 **64 B**；offset：commType 0、opType 4、sendBuffer 8、recvBuffer 16、count 24、strideLen 32、hcclDataType 40、valid 44、isLast 48、funID 49、**sendCnt 50、rcvCnt 51**、everyTurnRsp 52、everyTurnWait 53、totalTurnCnt 54、res[9] 55-63。`HCCL_PARAM_SIZE=sizeof=64`，`HcclServer::Init(win)` 的 send 区在 `win`、response 区在 `win+64`；`workspaceMsg=context->WorkSpace+notifyOff`，故 response 地址 = context->WorkSpace + notifyOff + 64。

**HcclCombinOpParam 两结构区分（关键）**：`mc2_common_def.h` 的 `AscendC::HcclCombinOpParam` 只有 `uint64_t workSpace; uint64_t workSpaceSize; uint32_t rankId; uint32_t rankDim;`（24 B，offsets 0/8/16/20）；Grouped kernel 源 L29-43 **另定义本地大结构** `HcclCombinOpParam`（`WorkSpace/WorkSpaceSize/rankId/rankDim/winSize/windowsIn[8]/windowsOut[8]/hcomId[128]/streamInfo[8]/signalInfo/config`），首四字段类型/顺序/offset 与 24B 结构一致（0/8/16/20），尾部按 `rac_server.h` 定义源级计算为 **2248 B**（windowsIn 32、windowsOut 96、hcomId 160、streamInfo 288、signalInfo 352-2239、config 2240，尾部对齐 8）。**`workSpace` vs `WorkSpace` 仅是不同结构/字段名，不是 ABI 冲突证据**；本机 kernel 源 `1db26125…` 与上游不同，完整本地结构仍需 31 源码确认。

**通知区/轮次（源级值，本机 config 未证）**：`KFCMsgBody` = `HcclAicpuOpParam msgSndArea[64][16] + msgRcvArea[64][16]`，每项 64 B → **notifyOff = 2×64×16×64 = 131,072 B**（tiling `set_notifyOff(sizeof(KFCMsgBody))`）；send 区在 +131,072，response 在 +131,136；`workspaceOff=libApiWorkSpaceSize_`（值未归档）。本机 tiling config `c9bd574a…` 与 kernel `.json` 尚未取内容，**不能把上游 notifyOff/结构大小直接当本机 config 事实**；需 31 提供内容后核实际 constants/struct。

**consumer 存在 vs dispatch 版本（分开）**：libccl 符号/strings 证明 `AivAicpuOpParam` 消费代码存在（`AicpuKfcProcess::AicpuCcOpExe`、`AddTaskForHcclMsg`、`AicpuKfcRpcServer::ReadWorkMsg/CheckRcvWorkMsg`、`AivAicpuOpParam::PrintMsg`、`HcclKFCTilingData.sendCnt`、`rcvCnt/sendCnt`），`RunAicpuKfcResInit` 反汇编调用 `AicpuRpcResInit(HcclCombinOpParam*)`；但**未证明**本次 H11 的 KFC task 实际 dispatch 到哪个版本/哪个函数（日志只有 API load 与 V1/V2 名称，非调用证据）。`AivAicpuOpParam` 字段（除 kernel 写的 sendCnt/valid）与 rcvCnt 初始化/递增仍无源。

**实际 config 选择（已证，JSON 级）**：本机 `grouped_mat_mul_all_reduce.json` binList 两项——FP16 x/w/bias→kernel `85517750…`；**BF16 x/w + FP32 bias→kernel `90b5b6bb…`**，与 H11 BF16/FP32 zero bias 对应；kernel json `coreType=MIX`、`opParaSize=9216`、`taskRation=1:0`、`workspace.num=1/size=-1`。这证明 H11 选择 90b5b6bb，但**不证明 tiling 运行时常量/notifyOff**（config 未给 AC_MAX_AIV/AC_MSG_CNT/notifyOff 数值）。

**consumer 反汇编补充**：`AicpuCcOpExe` 读取 `[x20,#24]`（count）与 `[x20,#40]`（hcclDataType），与 `AivAicpuOpParam` 源级 offsets（count 24、hcclDataType 40）一致，进一步支持 64B 布局；函数体未显示 rcvCnt 写入（只见头部，未做全函数分析）。消费端存在与 dispatch 版本仍分开。

**已证/未证**：已证——本机 Grouped kernel 源与上游 **`diff -w` 空**、rac_server 语义/布局一致、AivAicpuOpParam 64B offsets（disasm 读 24/40 相符）、两 HcclCombinOpParam 首四字段 offsets、libccl 消费端存在、KFCMsgBody 源级 131,072B、BF16+FP32 bias→90b5b6bb actual config。未证——本机实际使用的完整 HcclCombinOpParam 编译 sizeof/offsetof（源级 2248B）、tiling 运行时常量/notifyOff/workspaceOff 数值、KFC task dispatch 版本、rcvCnt 初始化/递增、AivAicpuOpParam 字段（除 kernel 写的 sendCnt/valid）由谁填充。**不把候选升级为结案**。

**最小 31 请求（H5K 后）**：本机 Grouped kernel `.cpp`（`1db26125…`）全文/关键行；两版 `mc2_common_def.h` 的 struct 原文；本机 tiling config `c9bd574a…` 与 kernel `.json` 内容；如可行补 `AicpuCcOpExe/AddTaskForHcclMsg/ReadWorkMsg` 反汇编。已有 pack 足够确认 rac_server/64B offsets 与首四字段。

## 24. H5K 结果与 H4K preflight（2026-11-19）

**H5K 原件核（33）**：`h5k-small-v1-cell` 15 文件，SUMS `b89b8fa7…` 15/15，commit 93eac05；flags=40；rank0 prepare stream `0xaaab071605b8`@dev0、handler stream 同值；rank1 `0xaaab344702e8`@dev1、handler 同值；`sync_enter` 无 `sync_return`；plog 仅 `Mc2AiCpuStreamAllocAndGet` stream 12 / `Mc2AiCpuInitStreamAllocAndGet` stream 11，**无 stream47/三分钟超时**；device 仅 `RunAicpuKfcResInitV2`/`AicpuRpcResInitV2`/`RunAicpuRpcSrvLaunchV2`，**无 KFCtaskAndTiling/AivAicpuOpParam**；result stub（api/sync false、libraries []）；恢复 aa3e-noop root **1316176**/399261496。

**H5K 有限结论**：H5K（唯一变量 REFETCH）仍挂起；prepare/handler stream 指针相同，但日志签名与 H11 不同（H11 有 KFC/AivAicpu/V1、stream47；H5K 无这些、仅 V2/stream12/11）。**日志缺失不作未 dispatch 证明**；REFETCH 路径改变了观测行为/挂起位置，**不自动归因 stream/context**；numeric 未产出；`mc2_server` 依赖仍 `supported_but_unproven`。

**H5K 时间线（33 新增，精确边界）**：rank0 trace 11 阶段：`handler_enter` @2.557746s（stream 0xaaab071605b8/context 0xaaab03d47600）→ Python `api_return` @2.557772s → `sync_enter` @2.557821s；**没有** `stream_compare`、`launch_enter`、`launch_return`、`handler_return`。binding `101660f8` 代码事实（35 核）：L335 `handler_enter` 的 fields（含 current_device/current_context）在 **mark 之前**；L335 mark 与 L344 `stream_compare` mark 之间只有 L338 `state->stream`、L339 REFETCH `getCurrentNPUStream()`、L341-343 `hex/snprintf`、L344 `mark("stream_compare")`；**区间内没有 fields()/current_device/current_context**；`mark→emit`（probe_trace.h）只用 clock_gettime/gettid/getpid/getpgrp/write，**无 ACL/runtime 调用**。stream_compare 仅写 prepare/handler 两指针。因此**候选是 L339 `getCurrentNPUStream()` 内部或 trace 写盘**，**未证具体哪一处**，不归因 Grouped/context；32 的 L339 代码级定位与 35 的 trace 写候选并存，需后续核验。`launch_enter`/launch 均未到达；Python sync_enter 只是 OpCommand.Run 已返回后等待，240s watchdog 杀死。plog 仅 AICPU stream 12/11/InitGroupCtx、无 KFC/stream47，与该“未到 launch”相容；`libraries []` 是 watchdog 早于最终 flush，不能推导未加载。**H5K 不是有效的 Grouped device-path 判别格**；H11 才是 matched Grouped hang；不把签名差异当 REFETCH 改变设备路径的证据。

**H4K preflight 合同核（33 PASS）**：`h4k-preflight/`：plan `947fe77a…`（arm H4K、flags=24=KEEP_ALL|DIRECT_LAUNCH、M128/K256/N512、BF16、FP32 zero、单 group、240/90、out/run/restore h4k-small-v1、plan_not_executed）；stage `2dcf4544…`（commit 0d3f7fb、manifest 21758ff5、run_probe 508e2931、window d7e73514、binding 101660f8、adapter d181b15a）；root identity `7ca5d3f2…`（1316176/399261496）；SUMS `3775ecf5…` 3/3；stop root 1316176、restore `restore-h4k-small-v1`、success aa3e、port 18143；port 29531 FREE。H4K 为线程+真实同步组合，非单变量；32 核增量后可 GO（本记录不请求）。验收：trace 区分 `source=cpp` 与 Python sync、result numeric/diff 或 stub、libraries/provider、plog 是否出现 KFC/stream47、window/child/cleanup、restore；返回时按实际 reduced diff 单列。

## 25. Producer/consumer/dispatch 证据边界（已有反汇编，2026-11-19）

**能证**：kernel producer 只写 send 区 `sendCnt=curTurn` 与 `valid=AC_MSG_VALID_MASK`（Barrier+DCCI）；core0 consumer `TurnWait(totalTurn)` 轮询 response 区 `rcvCnt>=totalTurn`，随后清零/失效并 DCCI；`GMMProcess` 的 `totalRun=Ceil(m/(splitM*singleM))` 是期望值；`AivAicpuOpParam` 64B（sendCnt@50/rcvCnt@51）。libccl_kernel 内含 `AicpuKfcProcess::AicpuCcOpExe`（反汇编读 `[x20,#24]` count、`[x20,#40]` hcclDataType，写 AicpuComContext 字段）、`AddTaskForHcclMsg`、`AicpuKfcRpcServer::ReadWorkMsg/CheckRcvWorkMsg`、`AivAicpuOpParam::PrintMsg`、`HcclKFCTilingData.sendCnt` 等符号/strings；`RunAicpuKfcResInit` 反汇编调用 `AicpuRpcResInit(HcclCombinOpParam*)`。本机 tiling config 已证 BF16+FP32 bias→kernel `90b5b6bb`（MIX、opParaSize 9216、taskRation 1:0）。

**不能证**：`rcvCnt` 的初始化值/递增指令与可见性（现有反汇编只覆盖 `AicpuCcOpExe/AddTaskForHcclMsg` 头部，未覆盖 `ReadWorkMsg/CheckRcvWorkMsg` 全文；未见 rcvCnt offset 51 访问）；`AivAicpuOpParam` 除 sendCnt/valid 外的字段由 kernel/AICPU 谁填充；H11 的 KFC task 实际 dispatch 到哪个 KFC/RPC 版本/函数（符号存在≠本次调用）；tiling 运行时 `notifyOff/workspaceOff` 实际值；本机 kernel 编译后完整 `sizeof/offsetof`。H6 仍只清 user workspace，不覆盖 context KFC 区。候选保持“AivAicpu KFC producer/consumer 响应推进”而非结案；后续 H4K 若返回/仍挂分别按 32 口径分层。

## 26. H4K 结果：调用线程真实 sync 仍挂 stream47 task13（2026-11-19）

33 只读核 `h4k-small-v1-cell`（23 文件，SUMS `8fe4986b…` 23/23，commit cd561f6）：plan 947fe77a/stage 2dcf4544/window d7e73514/run_probe 508e2931/binding 101660f8/adapter d181b15a/identity 1316176/399261496；out/run/restore h4k-small-v1 fresh。

- trace（source=cpp，flags=24）：`handler_enter` @2.546568（prepare stream 0xaaaaf0db4448/context 0xaaaaed99b470）→ `stream_compare` @.546576 → `launch_enter` @.546579 → `launch_return launch_rc=0` @2.609939 → **`sync_enter`（调用线程真实 aclrtSynchronizeStream）@2.609959，无 `sync_return`**；**无 Python `api_return`/`sync_enter`**（direct launch body 仍阻塞在 cpp sync，torch.ops 未返回）。rank1 同型（launch_return @2.565001、sync_enter @2.565022）。240s watchdog child rc -9、timed_out=true、survivors []、result stub（libraries []，早 flush）。
- plog 两 device `2026-11-19-08:26:31.680/.677 [stream.cc:1996] three minutes timeout! stream_id=47, sq_id=2, task_id=13, sqHead=13, pendingNum=1`；device 日志创建 `KFCtaskAndTilingTraceData` + `AivAicpuOpParam`（device0 L108/L111、device1 L108/L111）、`End AicpuRpcResInit` L113，并加载 `RunAicpuKfcResInit`（V1）/`ResInitV2`/`RpcSrvLaunchV2`/`SrvLaunch`（L88/L107/L114）。
- 活 rank 快照：rank0 pid 1318300、rank1 1318301，PGID self、starttime 399358649、cmdline=torchrun H4K；保存 `/proc/<pid>/maps` 全文 + 15 项过滤库 SHA；**快照在启动早期采集，probe_adapter.so/libopapi.so 尚未映射，result libraries 未 flush**——adapter 加载门不能由该快照证明，时序限制如实记录。restore aa3e-noop root **1319033**/五 PID/四检全 true。

**有限结论（按 32 措辞修订）**：H4K（调用线程 + 真实 `aclrtSynchronizeStream`）与 H11（OpCommand handler + Python sync）**同样挂 stream47/task_id=13 pending**；H5K 为 launch 前区间阻塞（非 Grouped device 判别）。→ **该线程+显式同步组合未解除挂起**；更广 context/线程内部仍未证，**不能单格断言线程/同步不是差异源**。Grouped device path（KFC task/AivAicpuOpParam 响应推进）候选保持，不单格定因；H4K 无 numeric/reduced diff（stub）；`mc2_server` 依赖仍 `supported_but_unproven`。A14 large 的设备事实条件已满足，但按 32/20 **暂缓**：fac2d948/0cd3531a/bf119527 预检保留、未执行；待 window rank-group 修复+CPU fixtures+新 window plan/32 核后再按原授权继续。A15 无设备授权；Grouped large/A13 暂停。

## 28. H4K rank PGID/清理边界核（2026-11-19，20 优先）

**rank-snapshots 原始事实**：`rank0-pid1318300.stat`/`rank1-pid1318301.stat` 是 JSON 元数据（674 B，keys：boot_id/cmdline/pgid/pid/ppid/starttime/state），**不是原始 `/proc/<pid>/stat`，没有 SID/session 字段**。两 rank：`ppid=1318268`、`pgid=1318300/1318301`（self）、`starttime=399358649`（相同）、boot_id `393ecc15…`、state R/S、cmdline=torchrun run_probe --arm H4K。**“双 rank 同 PGID”不成立**（基于快照）；两 rank 是否为独立 session/SID 未采到，不能确认。

**window d7e73514 对照**：`subprocess.Popen(plan['child_argv'], start_new_session=True)`（L518-519）→ torchrun child pid **1318268**、ppid 1318248、pgid 1318268、starttime 399358408。`survivors = {p | row.pgid == child.pid}`（L551）仅枚举 child PGID；`cleanup_child(pid, table, protected)`（L284-303）`os.killpg(pid, SIGTERM/SIGKILL)` 也只作用于 **child.pid 的 PGID**；`maps_evidence(pgid,names)`（L321-351）同样只筛 `row.pgid == pgid`。H4K window 记录 `survivors=[]`、`cleanup.members=[]` 只证明 **torchrun child PGID 1318268 为空**；两 rank 的 PGID=self 不在该集合内，**不能由 window 证明 rank 组已清理**。`start_restore` 也 `start_new_session=True`（L447-448），与本次 probe 清理边界无关。

**安全含义**：外层 `killpg(child.pid)`/survivors 枚举覆盖不到 rank 自己的 PGID；rank 生命周期依赖 torchrun/elastic agent 的 worker 清理，window 未独立验证。若 torchrun 在 SIGTERM/SIGKILL 下未能回收 worker，rank 组可能残留；当前 H4K 无 post-run rank 组枚举，故“无残留”只能确认 child PGID，不能确认两 rank 组。**A14 large fac2d948 尚未启动**（plan_not_executed），按 20 暂缓其核验；若启动，沿用同一 window 时需补独立 rank 组清理/枚举。此报告不进行进程操作。

**最小修复建议（供 35/32/20）**：window 的 cleanup/survivors 应以 **child.pid 的 PPID 子树**为目标（启动后及超时前重新 scan，记录每个 rank 的 pid/ppid/pgid/starttime/boot_id），对每个独立 PGID 先校验 leader 身份（PID+start+boot 匹配）再 `killpg`；survivors 覆盖这些组；不伤未知/复用 PID 成员。H4K/H11/H5K 的历史 window 记录仍只证明 child PGID；修复需新 window SHA 并由 32/33 复核，A14 large 等安全修复后再按原授权继续。

## 29. Safe-window rank-group 修复审阅清单（2026-11-19）

**当前代码缺口（d7e73514/f05f352f 同型，行号）**：`cleanup_child` L284-303 仅 `pgid==child.pid` 并 `killpg(child.pid)`；survivors L551 仅 `pgid==child.pid`；异常/ finally 清理 L569/L577 仍只传 child.pid；`recovery_gate` L416-417 `residual` 仅 `pgid==child_pid`（`old_pgid` 是服务根，`runners` 是 HTTP clients），**恢复门可看不到独立 rank 组**；`maps_evidence` L321-351 同样只筛 child PGID。H4K window `survivors=[]`/`cleanup.members=[]` 只证明 torchrun PGID 1318268 为空；rank PGID 1318300/1318301 未被枚举/清理。**历史 H4K/H11/H5K 的健康恢复事实保留，但“no_survivors”声明须修订为“child PGID 无 survivor”，不能覆盖 rank 组。**

**原始 stat 事实**：`.stat` 是 JSON（pid/ppid/pgid/starttime/boot_id/state/cmdline，无 SID/session）；child 1318268 pgid self start 399358408；rank0 1318300/rank1 1318301 均 ppid 1318268、pgid self、start 399358649；SID 未落盘且进程已终止，**无法事后补证**；PGID=self 说明独立组，但是否 setsid 未证。

**修复必须满足（审阅门槛）**：
1. **运行期身份登记**：child 启动后及运行期/超时前多次记录每个 rank 的 `pid/ppid/pgid/sid/starttime/boot_id/cmdline`（原始 `/proc/<pid>/stat` 至少含 session 字段），写入 window 证据；不得只依赖 cleanup 时 PPID 树（父异常退出后 reparent 会丢失）。
2. **清理目标**：child PGID + 所有已登记 rank 独立 PGID + 当时仍可归属的 descendants；对每个组先校验 leader PID/start/boot（可加 cmdline）与登记一致，再 `killpg`；SIGTERM→rescan→SIGKILL。
3. **未知/漂移不杀**：leader 身份不匹配、PID 复用、未知 PGID、受保护服务/其它租户成员不得 kill；记录 `unknown/drift` 并阻止恢复，不得报 survivors[] 成功。
4. **survivors/恢复门**：cleanup 后 survivors 覆盖 child PGID + 所有登记 rank PGID + 当时 descendants；`recovery_gate` residual 必须包含这些组，任一存活/未知/drift → 不 `start_restore`；恢复后再用现有“root ppid=1/pgid=self/descendants 同组”门校验服务。
5. **CPU fixtures（Linux，31 提供证据）**：独立 rank PGID、rank reparent 到 1、PID 复用/starttime 漂移、同 PGID 未知成员、rank 早退/采样缺失、child 异常退出/超时、cleanup 逐组 kill、survivors 拦截、restore 被阻断、无任意 kill、恢复门顺序；测试需覆盖“父异常退出后仍能清理已登记 rank 组”。
6. **语义修订**：`no_survivors`/success 文本必须声明覆盖范围（child+rank groups）；旧 H4K/H11/H5K 记录的 `survivors[]` 仅 child PGID，历史证据保留但覆盖声明降级。

**当前状态**：35 尚无新 cleanup freeze（tip 94aa830，window f05f352f 仍只父组）；A14 large `fac2d948` 未启动、暂缓；A15 仅 CPU（94aa830/0bd1fde8），无设备 GO；新 window 审过前设备暂停。本清单不进行进程操作。

## 30. 6def7a0/34f77ea safe-window 代码审 findings（2026-11-19）

- **F1 高：drift 不入 survivors/恢复门**。`_live_verified` 将身份不匹配放 `drift`；`attributed_survivors`（L314-326）只返回 verified+unknown，`cleanup_attributed.members` 不含 drift；`recovery_gate.residual`（L675-677）只用 `attributed_survivors`。因此漂移/PID 复用组不被 signal（正确）但**也不阻断 `start_restore`**。`test_pid_reuse_and_starttime_drift_rejected`（L478-486）只断言 `_attributed_targets` 分类，未断言 gate。与 32 规范“未知/漂移→拒杀+停报、恢复门 blocked”不符；建议 drift 纳入 survivors/residual（单列 drift gate），cleanup 仍不 signal。
- **F2 高：首次登记前父退出的 cleanup 时序**。`registry.start()` 只启动 0.2s 线程，首次 `_record` 不同步执行；`registry.stop()`（L199-209）才最终 `_record`，而 child flow 的 `cleanup_final`（L835）在 `registry.stop()`（L838）之前。父在首次扫描前退出且 window 立即 cleanup 时，cleanup 见空 registry、不杀孤儿；随后 stop 登记，survivors（L839-843）非空并阻断恢复。**最终 gate blocked，但 cleanup 未完成、孤儿存活**。fixture `test_orphan_after_immediate_parent_exit...`（L522-579）父退出后**先等待 registry 快照包含 worker 再 cleanup**，未覆盖“cleanup 立即发生在首次登记前”；建议 start 时先同步 record 或在 cleanup 前强制 record，并新增“不等快照立即 cleanup”fixture。
- **F3 中：subreaper 孤儿子代单次展开不足**。`_record`（L146-170）先按 `tree(root_pid)|known` 展开，之后才加入 `ppid==self` 的孤儿子代，且不重新展开这些新孤儿的下代；`stop()` 只一次最终 record。若 torchrun 退出后由被收养的中间进程持有独立 PGID 的 rank 子进程，最终 record 可能只登记中间进程、漏子代；fixture 只有父→直接 setsid worker，无中间/孙代。建议加入孤儿后循环展开到 fixed point，或 stop 重复 record 至稳定。
- **F4 低：protected_overlap 未显式转 gate**。`cleanup_attributed` 返回 `killed=False/reason=protected_overlap`（L292-296），child flow 未检查该 reason 是否阻断恢复；protected 测试只断言不 signal。建议明确 protected 不杀但记 blocked。
- **已核通过**：plan-only（L131）不调用 enable_subreaper/不执行；Linux subreaper 失败（L143）在 d0_stop 前返回且 `run` 未调用；protected 不 signal；无关进程不杀；Mac 57/57（2 Linux-only skipped）。31 Linux fixtures 仍应执行，并补 F2/F3 用例；设备继续暂缓。

**e75a858 修复复核（33 代码级 PASS，Linux fixtures 待 31）**：window `0ca36426…`；F1 `attributed_survivors` L337-339 加入 drift live PID、cleanup members L320-323 加入 drift，cleanup 仍只 signal verified；F2 `start()` L188-196 同步 `_record`，所有 cleanup 路径 L839/852/889/906 先 `registry.stop()` 再 `cleanup_attributed`；F3 `_record` L151-159 先加 `ppid==self` 孤儿再做 fixed-point，`stop()` L212-225 最多 3 次 record 至稳定；F4 `cleanup_blocks_restore` L343 转 protected_overlap，`cleanup_blocked` L780 初始化并在 L847/866/898/914 置位，L925 gate 拒绝 `start_restore`。测试 33+17+9=59；macOS 59/59（4 Linux-only skip）。**小建议（非阻塞）**：`test_protected_identity_is_not_signaled` 只断言不 signal，未断言 `cleanup_blocks_restore(outcome)`；drift 测试未直接断言 cleanup 不 signal drift；可在 Linux fixture 同跑时补两条直接断言。Linux 真实 orphan/fixture 需 31 通道恢复后执行；设备仍暂缓。

**35b4fbe test-only 复核（33 PASS）**：window `0ca36426` 未变；test_window `33961f36…`（34 个）、manifest `eceef30f…`；新增 protected 测试直接 `assertTrue(cleanup_blocks_restore(outcome))`；新增 Linux-only `test_drift_identity_is_not_signaled` 用错误 starttime，断言 drift、不在 singles/groups、进程仍存活。测试 34+17+9=**60**；macOS 60/60（5 Linux-only skip）。Linux 真实用例（含新增 drift）仍待 31 通道；设备暂缓。

## 32. Linux fixtures 35b4fbe：3 个真实用例失败与修复要求（2026-11-19）

31 归档 `fixtures-35b4fbe/`（unittest.log 10k、3 份 descendants.json、f2_f3_probe.py；SUMS 5/5）。**Ran 34, failures=3**：

1. **收养过宽（高风险，必须先修）**：`grouped-mmar-registry-ffazbstj_descendants.json` 同时登记 pid 1322545（cmdline `python -c "import time; time.sleep(30)"`，无关进程）和 pid 1322547（worker，sleep 60），两者均 `ppid=1322540`（subreaper window）、`pgid=self`。`_record` 把所有 `ppid==self` 的孤儿加入 pids，未验证其是否曾属 child 子树；`cleanup_attributed` 会对这些独立 PGID 执行 `killpg`，**违反“无关进程不杀”**。测试 `test_orphan...` 在 zombie 断言处提前失败，未走到无关进程断言，所以未暴露该风险在用例中的直接失败。**修复**：收养/登记必须带 child 来源证明——如 spawn 时注入唯一 env token 并由 worker 继承，registry 读 `/proc/<pid>/environ` 校验；或至少 cmdline/path 允许列表（run_probe.py/torchrun + 本 cell temp/out、A12 binary）；仅 `ppid==self` 不足。未匹配的 ppid==self 进程不得登记/ kill，另记 unattributed。
2. **Zombie 未 reap（两用例失败）**：`_attributed_targets` 的 `_live_verified` 排除 `state==Z`；SIGTERM 后 rescan 新 `targets` 中 verified 为空，`if reap: _reap(targets['verified'])` 空转；`/proc/<worker>` 以 Z 残留，`test_orphan...` L573、`test_stop_before_cleanup...` L609 `assertFalse(exists)` 失败。**修复**：reap 使用 kill 前的 verified PID 列表，或对所有登记 snapshot PID（含 Z）执行 `waitpid(WNOHANG)`，循环到无子进程；不得依赖 rescan 后的 verified。
3. **Chain fixture 竞态**：`grouped-mmar-registry-chain-8to9sp8__descendants.json` 仅 1 条（pid 1322542，中间进程），无孙代；测试父退出后立即 `registry.start(); registry.stop()`，intermediate 尚未 spawn 孙代。**修复（test）**：intermediate 打印孙代 pid，测试等待 `/proc/<pid>` 出现后再 stop；**产品侧**可考虑 stop 前 settle/重复 record（不能证明未来 spawn）。
4. 现场服务实核：root 1319033/start 399382752/ppid1/pgid1319033/health ready/config s1/TP2/4096/TP0+TP1/card7 chip0/1 phy14/15/boot 393ecc15 未变；fixtures 进程退出后无残留（系统 sleep 300 属既有无关进程未动）。

**结论**：e75a858 代码级 F1-F4 修复仍不满足 Linux 真机安全门槛；P0 收养过宽、P1 zombie reap、P2 chain 竞态需修/补测。设备格继续暂缓；A14 large fac2d948 未执行；A15 仅 CPU；Grouped large/A13 暂停。

## 33. a00edad P0/P1/P2 代码级复核（2026-11-19）

freeze `a00edad`：manifest `778c37af…`、window **`9a31091b…`（46,020 B）**、test_window `5fdc498f…`（34 个，Mac 5 Linux-only skip）、run_probe `0bd1fde8…`、binding `101660f8…`、adapter `d181b15a…`；CPU 60/60。

- **F-A/P0 代码级 PASS**：window 在 spawn 时生成 `secrets.token_hex(16)`，`child_env={**os.environ, CLEF_WINDOW_OWNER: token}` 传入 Popen（L827-831）；`DescendantRegistry(token=...)`；`_record` 的 subreaper 分支仅当 `_has_env_token(pid, token)` 为真（读 `/proc/<pid>/environ`，按 NUL 精确匹配 `CLEF_WINDOW_OWNER=<token>`，L132-142/L165-167）才加入 pids；无 token 的无关进程不登记/不 kill。fixture 现在使用 token 且断言 unrelated 1322545 不在 snapshot、仍存活（L539-561/L578）。
- **F-B/P1 代码级 PASS**：`cleanup_attributed` 在 SIGKILL 后 `_reap(entries)` 遍历**所有登记候选**（含 Z；L307-312/L334-336），再 settle/最终 scan；不再用 rescan 后的 verified。fixture `/proc/<worker>` 不存在断言可验证。
- **F-C/P2 代码级 PASS（产品+测试）**：`stop()` 在 record 新增时 `time.sleep(0.1)` 并最多 3 次（L228-242）；chain fixture 等 intermediate 打印/出现孙代（`_scan_direct` ppid=intermediate）后再 start/stop（L648-662）。
- **未证**：Linux 真机重跑 34 项 + F2/F3 探针尚未执行（31 通道恢复后）；token 在真实 torchrun/worker 环境是否继承、`/proc/environ` 是否可读、worker 被杀后的 Z reap 仍需实测。代码级无新阻塞；设备继续暂缓；A14 large `fac2d948` 未执行；A15 仅 CPU。

## 34. a00edad Linux 真机 fixtures 复核：安全门 PASS（2026-11-19）

31 归档 `fixtures-a00edad/`（3 文件，SUMS `fc780c76…` 3/3，commit 165d5ad）：

- **unittest**：`Ran 34 tests in 9.778s` → **OK**，log SHA `659754bd…`；含 5 个 Linux-only 真实用例（orphan 立即退出/stop-before-cleanup/中间孤儿/无关进程/protected）。
- **F2 探针**：token worker `1323101` 已登记 → cleanup `killed=true` → `worker_exists_after_cleanup=false`、state null（**Z 已 reap**）；无关无 token 进程 `1323102` `unrelated_alive_after_cleanup=true`，registry 仅 `[1323101]`；`survivors_after_stop=[]`。
- **F3 探针**：父→中间→setsid 孙代，登记 `[1323106,1323107]` 两条；cleanup 后两者均不存在；`survivors=[]`。
- 环境：Linux CPU-only，无 NPU/torch_npu；服务未触，root 1319033/start 399382752/ppid1/pgid1319033/health ready/config s1/TP2/4096/TP0+TP1/card7 phy14/15/boot `393ecc15` 未变；fixtures 无残留（系统既有 sleep 未动）。

**结论**：P0 收养过宽（token 过滤）、P1 zombie reap、P2 chain 竞态在真机 fixture 均通过；`no_survivors` 覆盖 token 登记组 + child PGID，无意义外的无关 kill。安全门 PASS；A14 large 可在 31 更新 stage-identity（window 9a31091b、identity 仍 fresh）后由 32 按原授权一次 GO；Grouped large/A13 仍暂停；A15 仅 CPU。

## 35. A14 MRS large 执行：设备路径成立，run_probe/window 两缺陷（2026-11-19）

31 归档 `a14-large-cell/`（19 文件，SUMS `a560be11…` 19/19，commit 4372c7f）；tuple a00edad/778c37af/0bd1fde8/9a31091b/101660f8/d181b15a/plan fac2d948/identity 1319033/399382752（执行前 fresh）。

- **设备事实（两 rank 一致）**：`prepare_rc=0`、executor nonnull、workspace **115,818,496 B（约 110.5 MiB）**、`launch_rc=0`、Python `sync_return` **36.61861 / 9.46084 ms**；无 stream47/507046。→ **MRS V1 large 支持/workspace/同步成立**。
- **缺陷 1（阻断数值）**：`run_probe.py 0bd1fde8` 的 A14 分支只计算 reference reduce_scatter，之后 `if a.arm not in ('A14','A15'): output=y`；A14 未赋值 `output`，`finite = torch.isfinite(output.float())` 抛 `UnboundLocalError`，reduced/reference/bit_equal 未产出。修复：A14 分支同样 `output=y`（在 norm/gate 前）；重冻 run_probe 后重跑同格取 actual reduced diff。
- **缺陷 2（恢复身份误报）**：window `9a31091b` 在 `wait_health` 后立即检查 manifest pid 的 `ppid==1 && pgid==pid`；setsid wrapper（1323588）当时仍是父，root 1323589 已健康但检查抛 `restored root identity mismatch`（branch None）。manual-verification：root 1323589/start 399707126/ppid1/pgid1323589/sid1323589、tree 五 PID、branch aa3e-noop、health ready、info s1、text/image 200、clients 空；无第二棵树。修复：身份检查在 `restore_timeout` 内重试直到 wrapper 退出/reparent，或接受 manifest pid 的父为已登记 restore wrapper 且 tree 同 PGID（之后再复核 ppid1），**不得因此启动第二棵 restore**。
- **判定**：A14 large 设备路径可行、workspace 约 110.5 MiB、同步正常；数值/精度**未取得**（UnboundLocalError）；window 恢复为 false negative，服务已健康 aa3e root 1323589。重跑需 run_probe + window 修复/重冻、新 stage/plan；Grouped large/A13 暂停；A15 仅 CPU。

**a99c93d 修复复核（33 代码级 PASS）**：commit `a99c93d`、manifest `d0676814…`、run_probe **`d54584b7…`（25,772 B）**、window **`52c10dce…`（46,518 B）**、test_probe `c11f4284…`（18）、test_window `56185a93…`（36），合计 63；binding/adapter/H4K tuple 未变。`arm_output` L278-282：A15 返回 `full[:rows]`，其余（含 A14）返回 `y`；A15 分支不再提前赋 output，L432 在 norm/gate 前统一 `output=arm_output(...)`，A14 不再 UnboundLocalError。`wait_restored_root` L701-711：health 后按 0.5s 重试到 `ppid==1 && pgid==pid`，`timeout=min(30,restore_timeout)`，超时返回 last 并由调用方报 identity mismatch；**不启动第二 restore**。window diff 仅新增该函数与调用，a00edad 的 token/reap/survivors/registry 安全代码未改。测试 `test_restored_root_waits_for_wrapper_exit`/`timeout_returns_last`、`test_arm_output_keeps_a14_and_crops_a15` 覆盖。**建议**：新 window 52c10dce 上 Linux 重跑 36 项（或 32 明确豁免）后再执行 A14 large 重跑；A14 重跑用 fresh identity 1323589/399707126、同形状、新 run_probe/window、plan 名保留按 fresh 事实更新；32 一次 GO。Grouped large/A13 暂停；A15 CPU。

**subreaper 作用域缺口（33 新 finding，A14 large v2 暂缓）**：`enable_subreaper()` 只执行 `prctl(36,1)`（L108-116），**没有 `PR_GET_CHILD_SUBREAPER` 读取旧值，也没有在 `start_restore` 前/ finally 还原**。window 保持 subreaper 时，restore wrapper（start_new_session=True）退出后其 service 后代按内核语义会 reparent 到**最近的 subreaper=window**，而非 PID1；`wait_restored_root` 严格 `ppid==1` 在 window 存活时可能永不满足（A14 large v1 的 manual ppid1 只在 window 已退出后观测到）。现有 `test_real_foreground_http_restore...` 用 fake_scan/mock，不覆盖真实 fork/subreaper。**修复**：set 前用 `prctl(37,&old)` 保存原 flag；probe cleanup 后、`start_restore` 前 `prctl(36,old)` 还原（finally 兜底）；加 Linux-only 真实 fork fake_restore fixture（subreaper 时 wrapper 子进程 ppid=window，还原后 ppid=1），而不是放宽身份门。原始旧 flag=1 的场景需明确记录/处理，不得假定 0。
**plan/argv 来源核对（20 要求）**：v1 plan `fac2d948` 的 child argv 指向 `/workspace/yangchen/work/grouped-deepdive-35-0d3f7fb/.../run_probe.py` 与同目录 adapter；v1 stage `0cd3531a` 却写 `a00edad/0bd1fde8/9a31091b`，实际 A14 large cell descendants cmdline 也是 `grouped-deepdive-35-a00edad/run_probe.py` → **plan argv、stage、实际 runner 三者不一致**，fac2d948 不可作为 metadata-only 复用基线。v2 plan `56441483` child argv 指向 `grouped-deepdive-35-a99c93d`、stage-v2 `d54584b7/52c10dce` 一致；但 v2 未启动，须待 subreaper 作用域修复 + Linux 36 项后再 GO。A14 large v2 保持未执行；Grouped large/A13 暂停；A15 CPU。

## 27. A14 MRS large preflight 合同核（2026-11-19）

`a14-large-preflight/`：plan `fac2d948…`（arm A14、flags=KEEP_ALL、run_mrs、M968/K8704/N5120、output `(484,5120)`、240/90、out/run/restore `a14-large-v1`、plan_not_executed）；stage `0cd3531a…`（commit 0d3f7fb、manifest 21758ff5、run_probe 508e2931、window d7e73514、binding 101660f8、adapter d181b15a）；root identity `bf119527…`（1319033/399382752）；SUMS `1833d7bb…` 3/3；port 29531 FREE；stop/restore a14-large-v1、success aa3e、port 18143。形状合法：M968 为 rank2 整数倍（output 484×5120）、K8704/N5120、BF16、bias nullptr；用途仅 MRS V1 large 支持/workspace/数值可行性（非生产 NZ/性能、非 Grouped large）。合同无阻塞；32 核增量后可 GO（本记录不请求）。验收应记录 api/sync/finite、actual reduced max_abs/bit_equal、workspace/executor/资源、240s、plog、window/child、restore；失败/成功均保留原件。

**优先方向修正（32/35 审阅后）**：原 H4（flags=DIRECT_LAUNCH=16）**不是单变量**——它同时改变 ①调用线程 vs OpCommand consumer 线程 ②body 内真实 `aclrtSynchronizeStream` vs handler 默认 sync_rc0 ③失去 KEEP_ALL 保活/cleanup/id。H5（flags=REFETCH_STREAM=32）也不是单变量——同时改 handler 线程内 stream 重取与失去 KEEP_ALL 保活。正确匹配变体需显式组合：**H4K=DIRECT_LAUNCH|KEEP_ALL=24**（仍同时改线程+真实同步，是组合臂）与 **H5K=REFETCH_STREAM|KEEP_ALL=40**（对 H11 仅改 handler 流来源，是单一 stream/context 变量），需在 run_probe `ARMS` 与 window `--arm choices` 加两行（CPU-only，binding/flags 零改），**等 H11 raw + 32 审查后再决定，当前不写码/不排设备**。零代码用原 H4/H5 只能作“线程+同步+保活”/“stream+保活”组合交互臂，不能隔离 context/线程。判别顺序（未来）：H5K 返回→handler 重取 stream/context 绑定为差异；H11 挂 + H5K 挂 + H4K 返回→调用线程+显式同步路径可行，OpCommand consumer 线程/无 sync 为候选（同步与线程仍纠缠）；三者全挂→投递/AivAicpu-KFC 协议候选更强；**不得由 H4K 单独证明 context**。跨进程 stream 指针不可比，只比同进程 prepare/handler 关系与 device；C++ sync trace 与 Python sync 需标 `source=cpp|python`。

## 20. H11 matched small 结果：Grouped 同形状仍挂起（2026-11-19）

33 只读核 `h11-small-v1-cell`（15 文件，SUMS `57932c67…` 15/15；commit f576d23）：plan e159c897/out/run/restore h11-small-v1，identity 1311514/399079056，冻结 f3e76cc/5531e104/0f90fd23/a8bf075f/101660f8/2202165e/d181b15a。

- rank0 trace `6431785c…`/result `78523809…`、rank1 trace `eb4d9097…`/result `fdabc46b…`；两 rank：`api_return` workspace 16,777,216、run_id 1 → `sync_enter` → 无 `sync_return`；`launch_return launch_rc=0`、`handler_return cleanup_rc=0/sync_rc=0` 为 KEEP_ALL/非 DIRECT_LAUNCH 默认值，未调用同步 API。240s watchdog：child rc **-9**、timed_out=true、survivors []；result stub（api/sync false）。
- plog 两 device 对称：`2026-11-19-07:50:08.015/016 [stream.cc:1996] report three minutes timeout! stream_id=47, sq_id=2, task_id=13, sqHead=13, flip_num=0, pendingNum=1`；device 日志创建 `KFCtaskAndTilingTraceData` + `AivAicpuOpParam`（device0 L108/L111、device1 L108/L111）并加载 `RunAicpuKfcResInit`/`RunAicpuKfcResInitV2`/`RunAicpuKfcSrvLaunch`（L88/L107/L114）——与 A12/v4r1 同型，A14 v5 无此二对象。
- 恢复官方 start.sh：branch `aa3e-noop`、root **1313827**、五 PID、双 worker、四检/env/joint/clients 全 true、无残留。

**有限结论**：H11 在 M128/K256/N512、BF16、FP32 zero bias、单 group、KEEP_ALL、同 adapter/comm/240s 下仍挂起（即使 host handler 已返回、Python device sync 未返回）→ **排除“仅大 shape/tiling”解释**，Grouped-specific op/kernel/tiling/`rac_server`/AivAicpu KFC task 投递为优先候选，`task_id=13` pending；不单格定因，`mc2_server` 必要依赖仍 `supported_but_unproven`。H11 KFC/AivAicpu 痕迹与 A14 的 `Hccl<HCCL_SERVER_TYPE_AICPU>` 客户端路径形成对照；日志缺失不作未执行证明，但结合源码链支持协议候选。large 暂停；H4K/H5K 不自行加臂，待 20/32 排程；A13 不执行。

## 39. 120d4e7 begin_restore 全链路与 plan 指纹审计（2026-11-19）

**调用链事实（window b77530bc）**：probe 侧先 `registry.stop()`（最终 record）→ `cleanup_attributed`/`survivors`（reap/token），随后 `recovery_gate`；`cleanup_blocked` 优先拒绝。`needs_restore` 为真后唯一 `begin_restore` 调用在 L998；`start_restore` 只在 `begin_restore` 内（L138）。`begin_restore` 先 `disable_subreaper`（L132），失败写 `blocked_subreaper_restore`/`restore_blocked` 并返回 False，调用方 raise → except 只记录 error，**normal/except/finally 均无第二 restore 旁路**；`wait_restored_root` 的严格 `ppid==1 && pgid==pid` 保留。**F5 仍开放**：`_prctl_subreaper` 只 set 1/0（L108-113），无 `PR_GET_CHILD_SUBREAPER` 旧值读取，`disable_subreaper` 固定清零；docstring “restore pre-cell state”不成立，fixture 只测 enable/disable，不测 original=1。修复需 set 前 `prctl(37,&old)`，restore 前清零，finally `prctl(36,old)` 还原，并加 original=1 fixture。

**plan/stage 指纹审计**：v1 `plan-a14-large-v1.json`（fac2d948）child argv 指向 `grouped-deepdive-35-0d3f7fb/run_probe.py` 与同目录 adapter；v1 `stage-identity.json` 却写 commit a00edad/run_probe 0bd1fde8/window 9a31091b，实际 a14-large-cell descendants cmdline 为 `grouped-deepdive-35-a00edad/run_probe.py` → **plan argv/stage/actual 三者不一致，fac2d948 不可 metadata-only 复用**。v2 `plan-a14-large-v2.json`（56441483）child argv 指向 `grouped-deepdive-35-a99c93d`，而当前 `stage-identity-v2` 已为 commit 120d4e7/window b77530bc/run_probe d54584b7（run_probe/adapter SHA 与 a99c93d 路径相同）；v2 **未启动**，因此应重出 plan 使用当前 stage 路径/argv 并记录实际 SHA，不能沿用旧 path 当不可变执行源。A14 large 数值重跑待 F5 修复、31 Linux 38 项、plan 重出、32 GO；Grouped large/A13 暂停；A15 CPU。

**3d8667a 最终冻复核（33）**：commit `3d8667a`、manifest `4dfb719b…`、window `b77530bc`（未变）、test_window `0c81786a…`（39，7 Linux-only skip）、test_probe **`01fd2f54…`（18，11,437 B）**、run_probe `d54584b7…`、binding/adapter/H4K tuple 未变。manifest 的 test_probe 条目已改实际 SHA `01fd2f54`（32/31 指出的 c11f4284 漏更已修）。新增 Linux-only `test_fake_restore_reaches_pid1_after_disable`：真实 enable → `begin_restore` 内 disable（固定 0）→ fake wrapper fork setsid daemon 写 manifest 后退出 → `wait_restored_root` 得 ppid1/pgid self；断言 `start_restore` 只调用一次、无第二棵树。**F5 未闭合**：window 仍无 `PR_GET_CHILD_SUBREAPER`，`disable_subreaper` 固定写 0，`finally` 只 `enable_subreaper()` 不恢复原值；fixture 不测 original=1。若 window 启动前原 flag=1，最终状态仍被改变。20 要求“还原原 flag，不放宽身份门”；需捕获 `prctl(37,&old)`、restore 前清零、finally `prctl(36,old)`，并补 original=1 fixture；或由 20/32 明确接受固定清零（当前未接受）。A14 v3 plan 待 31 Linux 39 + F5 决策；v2 56441483 superseded/未执行。

## 38. 120d4e7 subreaper lifecycle 复核（2026-11-19）

freeze `120d4e7`：manifest `4beba032…`、window **`b77530bc…`（47,325 B）**、run_probe `d54584b7…`（未变）、test_probe `c11f4284…`（18）、test_window `6f6a3cc9…`（38，6 Linux-only skip）；CPU 65/65。`begin_restore` 在 `start_restore` 前调用 `disable_subreaper`；失败则 `blocked_subreaper_restore`/`restore_blocked`，**不启动第二棵树**（纯逻辑测试在）。新 Linux daemon fixture：enable 时 daemon 化 wrapper 子进程 reparent 到 window；disable 后新 daemon reparent 到 PID1（非 window）。这两点代码级成立。

**F5（中高）原 flag 未读取/未还原**：`_prctl_subreaper(enabled)` 只执行 `prctl(36, 1/0)`（L108-113），没有 `PR_GET_CHILD_SUBREAPER`（37）读取旧值；`disable_subreaper` 固定清零，docstring 的 “Restore the pre-cell subreaper state” 与实现不符。若 window 启动前原本就是 subreaper=1，清零后不会恢复原值；fixture 只测 enable→reparent window / disable→PID1，**未测 original=1**。修复：set 前 `prctl(37,&old)` 保存；restore 前清零以达成 ppid1，finally 再 `prctl(36,old)` 还原原状态；补 original=1 的 Linux fixture。若 original=1，清零期间 service 可 reparent PID1，但结束后必须恢复原状态；不能把“固定清零”称为 restore。

**A14 v2 plan 状态**：`56441483`（window 52c10dce）未启动；新 window 为 b77530bc，故该 plan/stage 已不是当前 window，需 31 重出 stage/plan（run_probe d54584b7、window b77530bc、fresh identity 1323589/399707126），不能用旧 plan metadata-only 复用。A14 large 数值重跑待 F5 修复 + Linux 38 项 + 32/33 核。Grouped large/A13 暂停；A15 CPU。

## 41. 3d8667a Linux 39 项与 fake_restore fixture 缺陷（2026-11-19）

31 归档 `fixtures-3d8667a/`（unittest.log 11k，SHA **`ac816b00…`**；cleanup-note；SUMS 2/2）：`Ran 39 tests` → **38 OK / 1 FAIL**。唯一失败 `test_fake_restore_reaches_pid1_after_disable` L808：`live['pgid']=1325752 != pid=1325753`。

**根因（fixture，不是产品）**：fixture fake_restore.py 在 `os.setsid()` 之后**又 fork 一次**；写 manifest 的是孙代 C，C 的 pgid 是中间进程 B 的 setsid 组（B 是 group leader），所以 `pgid != C.pid`。真实 restore argv 是 `setsid /bin/bash -c ...` → `exec start.sh` → python，root PID 即该新 session/PGID 的 leader；产品 `pgid==pid` 门与真实路径一致。**修复**：fixture 改“单 fork + setsid + 直接写 manifest/sleep”（模拟 setsid→exec），不要二段 fork；产品门不改。期间 fixture 遗留 daemon 1325746/1325749 已按身份 SIGKILL，无其它残留；服务未触 root 1323589 健康。

**F5 仍开放**：原始 subreaper flag 未读/未还原（详见 §38/§40）；fixture 39 项也未测 original=1。最终冻需修 fixture + F5 重冻，31 Linux 39 全过后再出 v3 plan（window b77530bc/新 SHA、fresh identity 1323589、同形状、KEEP_ALL、真实 child SHA）；v2 56441483 superseded/未执行；设备暂停；Grouped large/A13 暂停；A15 CPU。

## 42. 0cea506 F5 修复复核：代码闭合，fixture 仍二段 fork（2026-11-19）

freeze `0cea506`：manifest `300d79e6…`、window **`1f780bf8…`（48,448 B）**、test_window `f45baef5…`（40，7 Linux-only skip）、test_probe `01fd2f54…`（18）、run_probe `d54584b7…`；CPU 67/67。

- **F5 代码级 PASS**：`get_subreaper()` 用 `prctl(37, &value)` 读取（L116-127），execute 在 enable 前保存 `original_subreaper`（L864-865），`begin_restore` 仅 restore 前 `disable_subreaper`（L158-167），主 finally/末尾 `restore_subreaper(original)` 并记录 final（L1087-1089）；原始=1 时 restore 后回 1，不再固定 0。`get_subreaper` 返回 None 时仅不恢复并记录，未阻断；可作为低风险备注。
- **fixture 仍失败（阻断 Linux 40）**：`test_fake_restore_reaches_pid1_after_disable` 的脚本仍是 `os.fork()==0: os.setsid(); if os.fork()==0: write manifest`（L781-792），即 setsid 后**二次 fork**；manifest 的孙代 C 的 pgid 是中间进程 B 的组，`C.pgid != C.pid`。31 在 3d8667a 已见 `1325752 != 1325753`；0cea506 test_window 仍保留该二段 fork，Linux 40 预计同失败。**修复**：fixture 单 fork setsid 后直接写 manifest/sleep（父退出），或使用 `os.setsid()` 后不 fork、同进程写；产品 `pgid==pid` 门不改。F5 本身代码不需要再改 window，仅 fixture/测试。
- **A14 v3**：v2 56441483 superseded/未执行；等 fixture 单 fork 重冻 + Linux 40 全过 + 31 v3 plan（window 新 SHA/run_probe d54584b7/fresh identity 1323589/同形状/KEEP_ALL/实际 child SHA）→ 32/33 核 → 一次 GO。Grouped large/A13 暂停；A15 CPU。

## 43. 6dae03f 单 fork 修复复核：变量遮蔽仍在（2026-11-19）

freeze `6dae03f`：manifest `83dd8e85…`、window `1f780bf8`（未变）、test_window `22f8297d…`（40，7 Linux-only skip）、test_probe `01fd2f54…`、run_probe `d54584b7…`；diff 仅 README/manifest/test_window。

- **已修**：fake_restore 脚本改为**单 fork + setsid 后同进程写 manifest/sleep**（本轮 test L781-792：`if fork==0: setsid(); write manifest; sleep; _exit; exit`），root PID 即新 PGID leader，产品 `pgid==pid` 门正确。
- **未修（阻断 Linux 40）**：测试变量遮蔽仍在：先 `original = win.start_restore`（函数），`counted()` 闭包调用 `original(...)`；随后 `original = win.get_subreaper()` 将其覆盖为 int 1，`begin_restore` 内 `counted()` 触发 `TypeError: 'int' object is not callable`（31 在 0cea506 已见同 ERROR；6dae03f 该段未改）。修复：分离命名，如 `start_restore_fn = win.start_restore`、`subreaper_original = win.get_subreaper()`；`counted` 用 `start_restore_fn`，`restore_subreaper(subreaper_original)`。仅测试改动，不改产品/F5/window。
- **A14 v3**：仍待上述测试修复 + Linux 40 全过 + 31 v3 plan（window 1f780bf8、run_probe d54584b7、fresh identity 1323589/399707126、同 M968/K8704/N5120、KEEP_ALL、实际 child SHA）→ 32/33 核 → 一次 GO；v2 56441483 superseded/未执行；设备暂停；Grouped large/A13 暂停；A15 CPU。

## 45. 6d09515 最终 v3 合同核（2026-11-19）

`a14-large-preflight/SHA256SUMS` `90557c94…` **8/8**；`fixtures-6d09515/SHA256SUMS` `90f20867…` **3/3**。v3 plan `efb6fb19…`：arm A14、M968/K8704/N5120、KEEP_ALL、run_mrs、fresh `run/temp/out/restore-a14-large-v3`、240/90、status plan_not_executed；child argv 指向 **`grouped-deepdive-35-6d09515`** 的 `run_probe.py`（d54584b7）与 adapter（d181b15a）；root identity **1323589/399707126**（ed2e262b）。stage-v4 `47ec314f…`：commit 6d09515、manifest f82f44ad（Linux **29/29** exact）、window **1f780bf8**、run_probe d54584b7、binding 101660f8、adapter d181b15a；tests test_window 228150e4、test_probe 01fd2f54；Linux single fake_restore **OK**（log 6a369a97）、full **40/40 OK**（log 53cb7baf）；port 29531 FREE；旧 fac2d948/56441483 均 superseded。**合同 PASS**：plan/stage/argv/实际 runner SHA 一致；A14 v3 未执行，32 一次 GO；执行验收 actual reduced max_abs/bit_equal、workspace、sync、trace、descendants/cleanup、restore 五 PID/四检/branch；Grouped large/A13 暂停；A15 CPU。

## 46. A14 v3 结果：数值完成但非精度通过（2026-11-19）

31 归档 `a14-large-v3-cell/`（18 文件，SUMS `774bbdf5…` 18/18，commit 691a826）。两 rank：`api/sync/finite/both_finite/numeric_pass/pass=true`；`reduced.max_abs=2.0`、`mean_abs=5.446e-05/4.804e-05`、`bit_equal=false`；workspace **115,818,496 B（110.5 MiB）**；trace sync **9.97/36.81 ms**；无 stream47/错误行。window child rc0、timed_out=false、survivors []、cleanup_final 全空；`ok=false` 仅 maps_ok=false（采样伪影），restore branch aa3e-noop root **1326845**/五 PID/四检全 true、`subreaper_original=0 → cleared=true → final_restored=true → final=0`、无第二棵树；descendants 3 条（torchrun+2 rank），无残留。

**判读**：MRS V1 large 的**支持/workspace/sync/数值链路成立**；**不是精度通过**——`bit_equal=false`，`max_abs=2.0` 不满足 A15 小形状 0.03125 的绝对诊断阈值，但 0.03125 是尺度相关的小形状探针值，不能直接套 large。按 K=8704、N(0,1) 输入估计输出量级 std≈93，BF16 ulp 在 128/256/512 约 1/2/4；`max_abs=2.0` 可能是少量 BF16 reduce 顺序 ulp，mean≈5e-5 远小于 1 ulp，但 result 未存 reference 幅值/相对误差，**不能据此判精度通过**。建议后续 probe 增加 `max_rel/mean_rel/ulp_at_max`（或 reference 幅值）与明确的 large 诊断门；若用于生产替代 down 路径，需模型级 OpenDecisions/ERQA 2PP 验证。A14 large 当前仅作 large 支持/workspace/数值可行性证据，不作生产 NZ/性能/精度结论。Grouped large/A13 暂停；A15 CPU。

## 47. A14 v3 原件数据缺口：不能证实“1 ulp/60–68 元素”（2026-11-19）

逐字段核 `rank0/1-result.json`：keys 仅 `reduced={max_abs,mean_abs,bit_equal}`、`inputs={x,weight}`（shape/dtype/sha256）、plan/libraries/versions/trace/wall 等；**没有 reference/output 张量、参考幅值、逐元素 diff、diff 直方图/分位数、max_rel/mean_rel/ulp_at_max/counts**。因此：
- 可计算：N=(968/2)×5120=**2,478,080**；total_abs=mean_abs×N = **134.958979**（rank0）/ **119.054683**（rank1）；因每元素 |diff|≤max_abs=2.0，非零差元素数**至少 ceil(total_abs/2.0)=68/60**；这只是下界，若存在小于 2.0 的差，元素数更多。不能由此得到“60–68 个元素差 1 ulp”的精确计数。
- 不可证：2.0 是否等于“峰值处 1 个 BF16 ulp”、输出峰值 420–510、相对误差 ~7e-7 都依赖未落盘的 reference/output 幅值与逐元素数据；K=8704、N(0,1) 的 σ≈93.3 只是模型估计，不是本格实测。**不能把 max_abs=2.0 直接归因为 BF16 累加/非缺陷**，也不能据此改门或宣称精度通过。
- 修复/后续：probe 记录 reference/output 的 max/p50/p99 幅值、逐元素 diff 分位数、非零计数、max_rel/mean_rel、ulp_at_max；或做模型级 OD/ERQA 2PP。A14 large v3 仍为 numeric-complete、not precision pass；A14 对照臂不新增绝对门。Grouped 研究不受此格影响；A15 rows127 合同不扩展审核。

## 48. A15 small v1 失败：run_probe A15 pad 分支 NameError（2026-11-19）

31 归档 `a15-small-v1-cell/`（18 文件，SUMS `e185265f…` 18/18，commit 2cb8fdd）；tuple 6d09515/run_probe d54584b7/window 1f780bf8/adapter d181b15a/plan 99db4369/identity 1326845/399871182（执行前实核一致）。

- 两 rank trace：`arm_enter flags=8` 后 `python_exception: NameError: name 'k' is not defined`；result 定位 `run_probe.py:386` `x_mrs = torch.ones((m_pad, k), ...)`，应为 `a.k`；仅 A15 tail pad（rows127→128）触发，未产生 pad_ready/gather/equivalence/diagnostic_gate/数值。api/sync false；child rc1、timed_out=false、survivors []、cleanup_final killed 且各集空。
- 恢复：branch aa3e-noop、root **1328734**/start 399924959、五 PID、四检全 true；`subreaper_original=0 → cleared=true → final_restored=true → final=0`；无第二棵树。
- **判定**：A15 harness 产品缺陷（变量名 `k`→`a.k`），本格 invalid，无数值；Grouped 研究不受影响。修复重冻 run_probe 后需新 A15 plan（a15-small-v2）与一次 GO；A15 large/生产未授权；Grouped large/A13 暂停；设备暂停。

## 49. 22346ec A15 fix 代码级复核（2026-11-19）

freeze `22346ec`：manifest `35bc507a…`、run_probe **`7afd7799…`（27,430 B）**、window `1f780bf8` 未变、test_probe `284974b3…`（21）、test_window `228150e4…`（40）；CPU 70/70（7 Linux-only skip）；binding/adapter/H4K tuple 未变。

- **`a15_flow` L285-309**：参数显式含 `k`；`m_pad=padded_rows(rows,world)`；pad 时 `ones((m_pad,k))` 后仅 `x_mrs[:rows]=x`（原 127 行不变、尾部 pad 为 1）；`run_mrs(x_mrs,w,world,flags)`→`sync`→`full=empty((m_pad,n))`→`all_gather`→`reference=all_reduce(matmul(x,w))`→`final_sync`→`output=full[:rows]`；info 记录 pad/row_block/workspace/run_id。**execute L432-444 实际调用同一 `a15_flow`，并传 `a.k/a.n`**；A15 v1 的未定义 `k` 已消除。
- **测试**：`test_a15_rows127_pad_gather_crop_with_stubs` 验证 pad 行非零、原行不变、run_mrs 128、sync 一次、AG 128、crop 127、reference 127；`F821Test` 用 symtable 检查 run_probe/window 并无负例 `k` 通过；`arm_output` A14/A15 冒烟。Mac 70/70。
- **A15 v2**：v1 invalid；31 用 fresh identity **1328734/399924959** 出 plan（rows127/k256/n512、flags8、bias null、window 1f780bf8、run_probe 7afd7799、adapter d181b15a、fresh a15-small-v2、port 预检）→ 32/33 核 → 一次 GO。A15 large/生产未授权；Grouped large/A13 暂停；设备暂停。

**0146129 test-only 复核（33 PASS）**：commit `0146129`、manifest `47976c76…`、run_probe 7afd7799/window 1f780bf8 未变、test_probe **4418a835…（22）**、test_window 228150e4（40）；CPU 71/71（7 Linux-only skip）。新增 `test_checker_catches_the_old_a15_pad_line` 内嵌旧 `torch.ones((m_pad,k))` 并断言 symtable 报 `('execute','k')`；A15 rows127 桩断言 pad 行=1、原行保留、run_mrs 128、sync 1、AG 128/64、matmul/reference 127、output 127、max_abs=8.0、info pad/row_block/workspace/run_id。产品代码沿用 22346ec PASS；A15 v2 plan 待 31，32/33 核后一次 GO；A15 v1 invalid；设备暂停。

## 50. A15-small-v2 preflight 元数据核（2026-11-19）

`a15-preflight/SHA256SUMS` `9bceab86…` **9/9**；plan `a9ea5fe2…`：A15、rows127/k256/n512、seed 20261119、fresh `run/temp/out/restore-a15-small-v2`、240/90、plan_not_executed；child argv 指向 `grouped-deepdive-35-0146129` 的 run_probe（7afd7799）/adapter（d181b15a）；root identity **1328734/399924959**（8ecd96b6）。stage-v6 `bf21fabb…`：commit 0146129、manifest 47976c76（Linux **29/29** exact）、run_probe 7afd7799、window 1f780bf8（Linux 40 未重跑，test-only）、binding 101660f8、adapter d181b15a；test_probe 4418a835、Linux probe **22/22 OK**（log 3429b5f9）、F821 负例通过；port 29531 FREE；v1 99db4369 invalid、99abe352 superseded。元数据一致；A15 v2 未执行，32 一次 GO；A15 large/生产未授权；Grouped large/A13 暂停；设备暂停。

## 51. A15 v2 结果：pad/gather/sync 成立，equivalence 被 lambda 缺陷跳过（2026-11-19）

31 归档 `a15-small-v2-cell/`（18 文件，SUMS `05b2ccc6…` 18/18，commit 32ef246）：plan a9ea5fe2/identity 1328734/399924959。两 rank result：pad `{rows127,padded128,pad_value1.0}`、row_block rank0 `(0,64)`/rank1 `(64,128)`、gather_ms **1.16421/1.15801**、workspace **17,170,432**、api/sync/finite/both true、numeric_pass/pass true；trace pad_ready/workspace/sync **7.776/9.699 ms**/gather；child rc0、survivors []、cleanup_final 空；window ok=false 仅 maps 采样伪影；restore aa3e-noop root **1330921**/五 PID/四检全 true、subreaper 0→cleared→final0、无第二树。

**缺失字段根因（产品）**：`a15_flow` 的 `reference=all_reduce(matmul(x,w))`，execute 传入 `lambda t: dist.all_reduce(t, op=SUM)`；PyTorch `dist.all_reduce` 原地返回 None → reference=None → 后置 `reduced/equivalence/diagnostic_gate` 分支被跳过（并非零差异）。result 无 `reduced/equivalence/diagnostic_gate`，因此 **numeric_pass 仅覆盖 api/sync/finite，不是等价性通过**。修复：lambda/函数调用 `dist.all_reduce(t)` 后 `return t`，CPU 桩断言 equivalence 出现；重冻后需 a15-small-v3 plan 与一次 GO。A15 v2 无数值等价结论；A15 large/生产未授权；Grouped large/A13 暂停；设备暂停。

## 52. cdc8324 A15 原地/fail-closed 修复复核（2026-11-19）

freeze `cdc8324`：manifest `eedd36c9…`、run_probe **`103b9775…`（28,844 B）**、window `1f780bf8` 未变、test_probe **`f6fe2c99…`（24）**、test_window 228150e4（40）；CPU 73/73（7 Linux-only skip）；binding/adapter/H4K tuple 未变。

- `a15_flow` L285-311：`reference = matmul(x,w)` 后 `all_reduce(reference)` 原地调用，**不再使用返回值**；注释明确 `dist.all_reduce` 返回 None。
- `a15_record` L314-330 纯函数：reference None→None；否则 `reduced/equivalence/diagnostic_gate`（0.03125/1e-4 diagnostic，非 fixed_gate）。execute L512-518 调用，None 时写 `numeric_error=missing_equivalence_record`；L546-553 `record_ok` 要求 A15 三键齐全，缺则 numeric_pass=false（fail-closed，记录完整性，不是误差门）。
- 测试：CPU 桩 all_reduce 返回 None 并原地改 tensor，断言 reference 非 None/值正确、output/reference 127、max_abs=8.0；`A15RecordTest` 正常/None 两态；test_probe 24。代码级 PASS。
- A15 v3 待 31 用 fresh identity **1330921**（restore manifest 无 starttime，需实扫 /proc）出 plan（rows127/k256/n512、flags8、bias null、window 1f780bf8、run_probe 103b9775、adapter d181b15a、fresh a15-small-v3、port 预检）→ 32/33 核 → 一次 GO；A15 large/生产未授权；Grouped large/A13 暂停；设备暂停。

## 53. A15 record shape 校验缺口（2026-11-19）

确认 32 finding：`a15_record` L314-330 只判断 `reference is None`，**未校验 `reference.shape == output.shape == (rows,n)`**；`equivalence.shape` 直接写 `[rows,n]`。若 reference 形状错误但与 output 可广播（如 `[1,n]`），`diff/equal` 会静默广播并生成误导性 `max_abs/mean_abs/bit_equal`，记录仍标 `[rows,n]`；完全不可广播时才抛错（fail-closed）。修复：在 `a15_record`（或调用前）断言两者 `tuple(shape)==(rows,n)`，不匹配返回 None/写 `numeric_error=shape_mismatch`；CPU 用例覆盖可广播错 shape（必须 fail-closed，不得静默记录）。属 record-integrity，不是本轮 v3 的算子精度结论；v3 可先跑，但记录完整性闭合前必须补；设备仍暂停。

## 54. A15-small-v3 preflight 元数据核（2026-11-19）

`a15-preflight/SHA256SUMS` `68c1660f…` **13/13**；plan `2fcd101b…`：A15、rows127/k256/n512、fresh `run/temp/out/restore-a15-small-v3`、240/90、plan_not_executed；child argv 指向 `grouped-deepdive-35-cdc8324` 的 run_probe（103b9775）/adapter（d181b15a）；root identity **1330921/400000296**（78b95dc0，实扫 /proc）。stage-v7 `0e65c9e4…`：commit cdc8324、manifest eedd36c9（Linux **29/29** exact）、run_probe 103b9775、window 1f780bf8、binding 101660f8、adapter d181b15a；test_probe f6fe2c99、Linux **24/24 OK**（log 0b71b25a）；port 29531 FREE；v2 a9ea5fe2 numeric-incomplete、v1 99db4369 invalid。元数据一致；v3 可跑，但 §53 shape validation 仍未修，record 完整性最终必须补；GO 由 32，设备暂停。
