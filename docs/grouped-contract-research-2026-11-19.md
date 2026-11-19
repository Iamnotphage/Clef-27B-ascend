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

**优先方向修正（32/35 审阅后）**：原 H4（flags=DIRECT_LAUNCH=16）**不是单变量**——它同时改变 ①调用线程 vs OpCommand consumer 线程 ②body 内真实 `aclrtSynchronizeStream` vs handler 默认 sync_rc0 ③失去 KEEP_ALL 保活/cleanup/id。H5（flags=REFETCH_STREAM=32）也不是单变量——同时改 handler 线程内 stream 重取与失去 KEEP_ALL 保活。正确匹配变体需显式组合：**H4K=DIRECT_LAUNCH|KEEP_ALL=24**（仍同时改线程+真实同步，是组合臂）与 **H5K=REFETCH_STREAM|KEEP_ALL=40**（对 H11 仅改 handler 流来源，是单一 stream/context 变量），需在 run_probe `ARMS` 与 window `--arm choices` 加两行（CPU-only，binding/flags 零改），**等 H11 raw + 32 审查后再决定，当前不写码/不排设备**。零代码用原 H4/H5 只能作“线程+同步+保活”/“stream+保活”组合交互臂，不能隔离 context/线程。判别顺序（未来）：H5K 返回→handler 重取 stream/context 绑定为差异；H11 挂 + H5K 挂 + H4K 返回→调用线程+显式同步路径可行，OpCommand consumer 线程/无 sync 为候选（同步与线程仍纠缠）；三者全挂→投递/AivAicpu-KFC 协议候选更强；**不得由 H4K 单独证明 context**。跨进程 stream 指针不可比，只比同进程 prepare/handler 关系与 device；C++ sync trace 与 Python sync 需标 `source=cpp|python`。

## 20. H11 matched small 结果：Grouped 同形状仍挂起（2026-11-19）

33 只读核 `h11-small-v1-cell`（15 文件，SUMS `57932c67…` 15/15；commit f576d23）：plan e159c897/out/run/restore h11-small-v1，identity 1311514/399079056，冻结 f3e76cc/5531e104/0f90fd23/a8bf075f/101660f8/2202165e/d181b15a。

- rank0 trace `6431785c…`/result `78523809…`、rank1 trace `eb4d9097…`/result `fdabc46b…`；两 rank：`api_return` workspace 16,777,216、run_id 1 → `sync_enter` → 无 `sync_return`；`launch_return launch_rc=0`、`handler_return cleanup_rc=0/sync_rc=0` 为 KEEP_ALL/非 DIRECT_LAUNCH 默认值，未调用同步 API。240s watchdog：child rc **-9**、timed_out=true、survivors []；result stub（api/sync false）。
- plog 两 device 对称：`2026-11-19-07:50:08.015/016 [stream.cc:1996] report three minutes timeout! stream_id=47, sq_id=2, task_id=13, sqHead=13, flip_num=0, pendingNum=1`；device 日志创建 `KFCtaskAndTilingTraceData` + `AivAicpuOpParam`（device0 L108/L111、device1 L108/L111）并加载 `RunAicpuKfcResInit`/`RunAicpuKfcResInitV2`/`RunAicpuKfcSrvLaunch`（L88/L107/L114）——与 A12/v4r1 同型，A14 v5 无此二对象。
- 恢复官方 start.sh：branch `aa3e-noop`、root **1313827**、五 PID、双 worker、四检/env/joint/clients 全 true、无残留。

**有限结论**：H11 在 M128/K256/N512、BF16、FP32 zero bias、单 group、KEEP_ALL、同 adapter/comm/240s 下仍挂起（即使 host handler 已返回、Python device sync 未返回）→ **排除“仅大 shape/tiling”解释**，Grouped-specific op/kernel/tiling/`rac_server`/AivAicpu KFC task 投递为优先候选，`task_id=13` pending；不单格定因，`mc2_server` 必要依赖仍 `supported_but_unproven`。H11 KFC/AivAicpu 痕迹与 A14 的 `Hccl<HCCL_SERVER_TYPE_AICPU>` 客户端路径形成对照；日志缺失不作未执行证明，但结合源码链支持协议候选。large 暂停；H4K/H5K 不自行加臂，待 20/32 排程；A13 不执行。
