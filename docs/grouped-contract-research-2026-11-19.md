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
