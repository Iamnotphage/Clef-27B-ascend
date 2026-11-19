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

## 55. A15 shape guard 候选 8807597 复核（2026-11-19）

候选冻：commit `8807597`、manifest `d65f0408…`、run_probe **`f7c5f860…`（29,346 B）**、test_probe `d7b6d0c4…`（25）、window 1f780bf8/test_window 228150e4 未变、adapter d181b15a；CPU 74/74（7 Linux-only skip）。**不用于当前 v3**（v3 锁 cdc8324/103b9775/plan 2fcd101b）。

- `a15_record` 现返回 `(record,error)`：reference None→`missing_equivalence_record`；`_shape_of(output)` 与 `_shape_of(reference)` 均须 `(rows,n)` 且互相一致，否则 `invalid_equivalence_shape`；正常才生成 reduced/equivalence/diagnostic_gate。execute L528-535 解包 error，写 `numeric_error`；record_ok 缺三键→numeric_pass=false。
- 测试：`[1,512]` 可广播错 shape 被拒（invalid_equivalence_shape）；正常 `[127,512]` 生成 equivalence/gate；None→missing；三态覆盖。代码级 PASS。
- 该候选只供 v3 后复用/最终记录完整性闭合；v3 结果产出后需评估是否用 shape-guard 重跑或仅 CPU 记录校验。A15 large/生产未授权；Grouped large/A13 暂停；设备暂停。

## 56. A15-small-v3 成功：zero diff 仅本格诊断（2026-11-19）

31 归档 `a15-small-v3-cell/`（17 文件，SUMS `d7e8b90f…` 17/17，commit 1d1daef）；tuple cdc8324/run_probe 103b9775/test f6fe2c99/window 1f780bf8/binding 101660f8/adapter d181b15a/plan 2fcd101b/identity 1330921/400000296。两 rank：pad `{127,128,1.0}`、row_block 0 `(0,64)`/1 `(64,128)`、gather **1.22224/1.21543 ms**、workspace **17,170,432**、sync **13.172/8.525 ms**；`reduced.max_abs=0.0`、`mean_abs=0.0`、`bit_equal=true`；`equivalence.shape=[127,512]` 同为零差；`diagnostic_gate` 0.03125/1e-4 `within_bounds=true`、status diagnostic-only；finite/both/numeric_pass/pass=true。window child rc0、survivors[]、cleanup_final 空、maps_ok=false 仅采样伪影；restore aa3e-noop root **1334752**/五 PID/四检全 true、subreaper 0→cleared→final0、无第二树。

**判读**：A15 rows127/pad128/crop127、k256/n512 的 all-gather 重建与 full-row reference **逐位一致**，是**本格诊断事实**；**不得据此称精度通过或外推 large/生产**。shape guard 8807597 未用于 v3（结果 shape 正确，但 record 完整性最终仍需候选/校验）。A15 large/生产未授权；Grouped large/A13 暂停；official/CSV 未动；服务 aa3e root 1334752 健康。

## 57. Grouped 新证据方向：AICPU kernelType 5 vs 6 与握手可见性（2026-11-19）

**公开对照（非 Grouped 专用，仅类比）**：GitHub `hw-native-sys/simpler` issue **#822**（`https://github.com/hw-native-sys/simpler/issues/822`，closed）维护者调查给出 CANN AICPU kernelType 表：2 `AICPU` 主 `aicpu_scheduler`、4 `AICPU_CUSTOM` cust 子进程、5 `AICPU_KFC` 主 scheduler、6 `CUSTOM_KFC` cust 子进程；**type 4/6 cust 子进程无法与 AICore 通信**；实验 #7 用 `CUSTKFCKernel`（实际映射 type6）在 a2a3 上产生 **507046，AICore kernel execute timeout，AICore 等待 AICPU 握手**。该 issue 最终以把 dispatcher 预部署到主 scheduler 路径（type5/2）解决。**限制**：issue 环境为 CANN 9.0/a2a3、AICPU_CUSTOM 路径、非 Grouped op；只证明 507046 可由 cust KFC 与 AICore 不可通信产生，不证明本机 Grouped 实际用 type6。  
**本地 docs 矛盾（事实/限制）**：f5703e4 `grouped_mat_mul_all_reduce/docs` 产品表写 **Atlas A3 不支持**（`/tmp/g29_docs.md` L10-12），但同文样例注释称 A3 实现、本机 op-info/op-info kernels 在 `ascend910_93` 注册；这是上游文档矛盾，不是 9.2.1 官方裁决；本机安装 CANN 9.2.1 官方文档与公开 Grouped issue 未找到（GitHub Grouped issue 仅 1 条无关 sglang roadmap，Gitee 0）→ **同版官方条件/已知问题缺口**。

**已有日志缺口**：A12/v4r1/H11/H4K/A14 的 device/plog 仅有 `aicpusd_cust_so_manager init`（所有臂都有，不能区分）与 `aicpu_scheduler` 进程名；**没有 `ConstructAICpuSqeForDavinciTask`、`kernel_type=`、`CUSTKFC/CUSTAICPU`、`cust_aicpu_sd` 任务行** → 无法判定 Grouped KFC 实际 kernelType 5 或 6。  
**最小 31 只读请求（不重复已有包）**：  
1. 全量 grep 现有归档 `libccl`/device/plog 的 `ConstructAICpuSqeForDavinciTask|kernel_type=|CUSTKFC|CUSTAICPU|cust_aicpu_sd|KERNEL_TYPE_`，给精确文件/行/时间；若全无，明确记为“运行期 kernelType 未落盘”。  
2. `libccl_kernel.so` SHA 2281deca 的完整反汇编文本（已有仅头部）：ReadWorkMsg `0x7d71cc-0x7d7284`、CheckRcvWorkMsg `0x7d7284-0x7d7624`、AddTaskForHcclMsg `0x7b2800-0x7b3704`（可切 0x7b2800-0x7b3000/0x7b3000-0x7b3704）、AicpuCcOpExe `0x7b21d0-0x7b2800`、RunAicpuKfcSrvLaunch `0x763108-0x7641f4`、RunAicpuRpcSrvLaunch `0x761380-0x762a84`、RpcSrvLaunchV2 `0x7547a0-0x755104`、ResInit `0x760468-0x760500`、ResInitV2 `0x754568-0x7547a0`；并给 `AicpuKfcRpcServer|AicpuKfcProcess|AivAicpuOpParam` 的 nm 起址/size。  
3. 若本机 CANN 9.2.1 安装文档/已知问题可只读检索，给路径/版本/片段；不做外部 issue/消息。  
**结论**：新方向可判别但未证——若 Grouped KFC 是 type6/cust，则 507046/AICore 等握手是强候选；若 type5/2，则依赖/协议路径另找。root cause 仍 **unresolved**，不回到 `confirmed_environment_blocked`；`mc2_server` 依赖仍 `supported_but_unproven`。A15 v3 zero-diff 仅本格；A15 large/生产未授权；设备暂停。

## 58. ed12a12 A15 large 误差字段：生产 tensor 路径未被 CPU 直测（2026-11-19）

32 finding 核：ed12a12（manifest 8e5846de、run_probe 8e308804、test 07e5a5b4）已实现 `relative_l2`（denominator=max(ref_norm,TINY)，0/0→0、zero_reference）和 `tensor_error_stats`（torch tensor：max/mean、ref/out abs max、nonzero/total、max_error index/position+值、max_rel_abs、relative_l2/denominator/zero_reference），但 **CPU 测试仍用 `list_error_stats` 或 `stats=lambda`**；`tensor_error_stats` 的真实 torch 张量路径未被直接执行。故生产字段的数学/边界仅由 list helper 证明，非生产实现证明；需 torch-available 直测（Linux venv 可跑）后重冻。A15 large c482f3f plan d1b02bcd/stage 已 superseded，ed12a12 预检 plan 34bfb3aa/stage39ffb333/identity1334752 元数据待最终冻；GO 应等 torch 直测重冻。A15 bench/生产仍由 32/35 范围；Grouped 研究不受影响；设备暂停。

## 59. consumer 控制流与 sendCnt/rcvCnt 数据流（abi-preflight3+4）

**材料**：`libccl_kernel.so` SHA `2281deca…`（9.2.1 `aarch64-linux/devlib/device`）。包 3：`c1f5a41`、SUMS `20fc3ed8`（ReadWorkMsg 53 行、CheckRcvWorkMsg 239、AddTaskForHcclMsg 968、AicpuCcOpExe 403、SrvLaunch 1090、RpcSrvLaunch 1480、RpcSrvLaunchV2 608 + nm）；包 4：SUMS `d29e80eb`（RunRpcServerApi 967 行、GetSendCnt/GetRecvCnt 27 行、ResetSqBuff 296、WaitTaskFinish 119、DealReturnValue 299、nm 符号 36 行）。31 的日志 grep 结果：**仅有 2026-11-18 旧实验 6 个 plog 命中 `kernel_type=TS_AICPU_KERNEL_AICPU_KFC(5)`**（device 0/1、stream 13、task 6/28/3）；11-19 Grouped 各格无 `kernel_type=`、无 `ConstructAICpuSqeForDavinciTask`、无 `CUSTKFC/CUSTAICPU/cust_aicpu_sd`。这是现有归档中**唯一**观察到的 KFC kernel type，且为 5；但属另一实验，**不能**据此判定 Grouped 也是 5，也不能反证 type6 类比。

### 59.1 已证（静态反汇编事实）

**A. SrvLaunch 顶层分派 `0x763108`**：
- `desc=*(u64*)task`；`(desc&0xf)==2` → `CommKfcDispatcher::Run(task+8,(desc>>4)&0xf)`（0x763564）。
- 否则 `idx=(desc>>9)&0x7f`、`arg=task[idx]`、`sub=*(u32*)arg`：
  - `sub==0`：若 `(desc&~0xffff)==0x170000` 且 `*(u32*)(arg+8196)==0x63` 且 `count=*(u32*)(arg+8192)>0`，逐子 op 调 `RunAicpuRpcSrvLaunch`（0x763368）；否则转局部函数 `0x762a84`。
  - `sub==1` → 同一 `0x762a84`。
  - `sub==2` → MC2 分支：按 `(desc>>4)&0xf` 循环后，`subcount-1<=2` 时 `AicpuRunRpcServerForMC2V2(task_v2,tiling)`（0x763880），否则 `AicpuRunRpcServerForMC2(task_v2)`（0x7639dc），随后调 0x75e3c0 helper。
  - `sub>2`：`sub==0x64` 时按 `AicpuGetInnerDevType()==4` 转向 0x763704，否则 `ClearOpRes` 路径（0x760624+0x124）；其余报错。
- **未证**：Grouped/A12 运行期的 `desc`、`idx`、`sub` 值；因此不能确定走 RpcSrvLaunch、MC2 RPC 还是 CommKfcDispatcher。

**B. RpcSrvLaunch 初始化与分支 `0x761380`**：
- `ctx=AicpuGetComContext()`；要求 `[task+24]`、`[task+40]` 非空；要求 **`ctx+0xf31b2` bit0=1**；`strcmp(ctx, [task+24]+0x220)==0`（组名/上下文匹配），否则错误返回。
- `mode=*(u32*)[task+40]`，`mode>1` 报错；写入 **`ctx+0xf31d0`**（0x761820）；随后清零 **`ctx+0xf31d4`**（0x761880）、**`ctx+0xf31d8`**（0x761884）、**`ctx+0xf32d8`**（0x7618a0）。
- `GetSendCnt(ctx)` → **`ctx+0xf3240`**、`GetRecvCnt(ctx)` → **`ctx+0xf3250`**（0x761888/0x761894）；再 `PrintTilingData`/`PrintMC2AicpuContext`（与 device 日志 `KFCtaskAndTilingTraceData`/`AivAicpuOpParam` 的出现时机一致，但无字符串 xref 证明同一处）。
- `AicpuKfcRpcServer::Init(ctx+184, ctx+192, u16 ctx+196, task)`（0x76192c）。
- 分支（0x761954-0x761994）：`GetPreparePosition()==1` → **`RunRpcServerApi(ctx,server,0)`**（0x761f80）；否则 `GetTaskType()==6` → **`AICPU_RpcServerUnfoldStageWait`**（0x7622e4）；否则 **`ctx+0xf31bc & ~2 == 1`** → `RunRpcServerTwoStageWait`（0x761994），不满足 → `TryRunRpcServerOneStageWait`（0x762228）。
- 返回后再次 `GetSendCnt` → **`ctx+0xf324c`**、`GetRecvCnt` → **`ctx+0xf325c`**（0x7619b0/0x7619bc），随后 `DealReturnValue`。

**C. ReadWorkMsg `0x7d71cc`**：`ctx=*(void**)(this+16)`；校验 **`ctx+44==0x5cdf123a`**，不等则**立即返回 0**；相等则 `memcpy(opParam,ctx,0x306=774)`、`dsb st`，返回 **`opParam[50] >= arg3`**（u8），arg2 未使用。

**D. CheckRcvWorkMsg `0x7d7284`**：同样以 `ctx+44` 魔数 + 774 字节拷贝 + `opParam[50]>=arg3` 轮询；上限 **10001 次迭代或 6,000,000,000 ns（0x165a0bc00）**；**成功与超时都返回 1**（超时仅 verbose 日志），不会返回 0。

**E. AddTaskForHcclMsg `0x7b2800`**：`count=msg[52]`；循环下标在 **`ctx+0xf31d8`**（u8，每轮 +1 后写回）；每轮调 `AicpuCcOpExe(opParam,NULL,ctx)`，再 `TaskOrchestrator::ActiveRecordMain(GetActiveSqId)`。当 `msg[0]==6` 且 `(msg[40]&~8)==3` 时，先调 **`CheckRcvAddrMsg(msg_local, [ctx+0xf31d4]+1)`**（0x7b2e80-0x7b2e98；此处 `ctx+0xf31d4` 是 **turn 计数**，见 59.2）；成功后才 `HcclMsg2AicAicpuOpParam` 并比对 msg/opParam 的 type 与 opcode。`msg[0]==7` 且 `msg[32] < msg[52]*msg[24]` → 错误 1646 并返回 1。调试日志（0x7b2f9c-0x7b3028）分别打印 `opParam` 的 **+44、+48、+49、+50、+51、+52、+53、+54、+56**。

**F. AicpuCcOpExe `0x7b21d0`**：写 `ctx+156`（dtype size）、`ctx+160`（count×size）、`ctx+144/148`（type/subtype）；读取 **`ctx+0xf31d9`** 与 **`ctx+0xf31bc`**；校验字节量 ≤0x3ffff 或 ≤0xfffff、2 的幂；把 `opParam+54/+55/+56` 写入 `ctx+4452/+200/+208`；调 **`TaskOrchestrator::RunConcreteAlgorithm(opParam, workspace, ctx)`**（0x7b247c）；成功后按 `ctx+352` 与 `ctx+4452` 比较，必要时 **`AicpuDispatcher::AddAllEndTaskOnMainStream(GetActiveSqId(ctx))`**（0x7b24c8）释放 AICore；rc=0x14 走特殊日志。

### 59.2 sendCnt/rcvCnt 的真实字段（abi-preflight4 指令级，SUMS d29e80eb）

**`GetSendCnt(ctx)` 0x797450（17 条指令）**：
```
base = *(u64*)(ctx+184) + *(u32*)(ctx+192);
return *(u8*)(base + 50);
```
**`GetRecvCnt(ctx)` 0x797470**：
```
base = *(u64*)(ctx+184) + *(u32*)(ctx+192) + ((u16)*(ctx+196) << 6);
return *(u8*)(base + 51);
```
- **已证**：`+50` 是 **sendCnt**，`+51`（槽位索引 = `ctx[196]`，即 rank 数）是 **rcvCnt**；两者都位于 `ctx[184]`（工作区指针）+`ctx[192]`（偏移）指向的共享区；`ctx[196]`（u16）是 rank 数/64 字节槽位跨度。
- **纠正**：ReadWorkMsg/CheckRcvWorkMsg 比较的 **`opParam[50]` 是 sendCnt，不是 rcvCnt**；`ctx+0xf31d4` **不是接收计数**，而是 **turn 计数器（0..63 环形）**——RunRpcServerApi 入口清零、每处理一条消息 `(turn+1)&0x3f` 写回，AddTask 传给 `CheckRcvAddrMsg` 的是 `turn+1`。
- **仍缺写者**：abi5 时已交付函数没有**独立** `+50/+51` 写指令；abi6 的 `GenMsgByTaskParam` 以 `u16 0x1234` 覆盖 `+50..+51`（常量初始化），整字/整块拷贝也覆盖这些字节。普通递增写者仍可能在未交付函数或 host/runtime/TS 侧写共享工作区（后者不能由 libccl 反汇编证否）；**不能以字面 store 偏移搜索断言全 libccl 唯一写者**。

**`RunRpcServerApi` 0x7b3704（API 路径主循环）**：
- 入口要求 **`ctx+0xf31bc == 2`**，否则返回 1；清零 `ctx+0xf31d4`；`w=ctx+0xf3268+1` 写入所有 context 的 `+0xf3268`（epoch）；置 `ctx+0xf31d1=1`、`ctx+0xf326c=1`。
- 循环 0x7b3918：`ReadAddrMsg(server, msg, turn=ctx+0xf31d4)`；返回 false → `PrintAllHcclMsgArea` + `PrintTimeOutSqInfo`，**返回 9**。
- 返回 true：按类型字节拷贝 40/64 字节；`msg.type==0x12` → `HandleBatchWriteOperation`；否则 `HcclMsg2AicAicpuOpParam` + 校验 + **`AddTaskForHcclMsg(ctx,server,msg,opParam,x24)`**。
- 成功后：`WriteTurnCnt(server,turn)`；`turn=(turn+1)&0x3f` 写回；继续循环。
- 结束（type 0x64）：`AddAllEndTaskOnMainStream` → `ActiveRecordMain` → `ctx+0xf31d1=0` → `WaitTaskFinish(ctx,1)` → `WriteFinishWhenAllFinalize(server,turn)` → 所有 context `+0xf31f0=2` → 返回 0。
- 该函数**不调用 GetSendCnt/GetRecvCnt**，也无 `+50/+51` 直接访问——API 路径不直接读 rcvCnt。

### 59.3 与 Grouped 挂起的可能映射（均未证）

- `ReadWorkMsg/CheckRcvWorkMsg` 属于 **deprecated stage-wait 路径**（OneStageWait/TwoStageWait/UnfoldStageWait），不是 RunRpcServerApi 的循环；把“sendCnt 轮询”直接套到 Grouped 上不成立。
- `RunRpcServerApi` 的超时是 `ReadAddrMsg` 返回 false → 返回 9，不是无限等待；Grouped 240 s 不返回更可能停在 `ReadAddrMsg` 内部长等待、`AddTaskForHcclMsg`/`AicpuCcOpExe` 的算法执行，或 `WaitTaskFinish`。
- 若 `GetTaskType()==6` 或 `ctx+0xf31bc` 模式不同，则走 `UnfoldStageWait/TwoStageWait/OneStageWait`，这些仍可能用 sendCnt 轮询；与 #822 的 cust-KFC 握手只是形式相似，`GetTaskType()`（0..6）≠ TS `kernel_type` 5/6。
- 若 sub==2：走 `AicpuRunRpcServerForMC2(V2)`，同样可能不返回；现有 plog 无法区分。
- A14（MRS V1）未出现 `KFCtaskAndTilingTraceData/AivAicpuOpParam` 且小/大形状均返回，提示其 launch 路径不同，但这不是对 Grouped 分支的证明。

### 59.4 下一项最小可判别只读请求（不重复已交付区间）

1. **`ReadAddrMsg(HcclMsg*,uint)` 0x7d8b6c（0x7f4）**——API 路径真正的读/比较/写链，最可能含 `+50/+51` 写者。
2. `WriteTurnCnt` 0x7d69a0（0xc8）与 `WriteFinishWhenAllFinalize` 0x7d68ec（0xb0）——turn/完成写回位置。
3. `Init(uint64,uint32,uint16,KFCTask*)` 0x7d66f0（0xe4）+ `Init(uint64)` 0x7d67e0（0x14）——确认 `server+16` 与 `ctx[184]/[192]/[196]` 的映射。
4. 仅当 1–3 仍无写者，再补 `RcvMsg` 0x7d8000（0x2e8）、`CheckRcvAddrMsg(HcclMsg)` 0x7d7624（0x144）、`CopyAndCheckApiMsg` 0x7d6ae4（0x438）。
- 明确**不重复**请求已交付的 RunRpcServerApi 0x7b3704、GetSendCnt/GetRecvCnt、ResetSqBuff/WaitTaskFinish/DealReturnValue、ReadWorkMsg/CheckRcvWorkMsg/AddTask/AicpuCcOpExe/SrvLaunch/RpcSrvLaunch/V2。

**结论**：sendCnt/rcvCnt 字段与 getter 已由指令证实；turn 计数与 rcvCnt 已区分；**rcvCnt 的写者仍未定位**，Grouped 运行期实际分支未证；root cause 仍 **unresolved**，不回到 `confirmed_environment_blocked`；`mc2_server` 依赖仍 `supported_but_unproven`。设备暂停。

## 62. abi-preflight5（SUMS ae91a530）consumer 等待链与槽位方程

**材料**：ReadAddrMsg 0x7d8b6c 516 行、CopyAndCheckApiMsg 0x7d6ae4 277 行、RcvMsg 0x7d8000 193 行、CheckRcvAddrMsg(HcclMsg) 0x7d7624 88 行、Init 0x7d66f0/0x7d67e0 74 行、WriteFinish/WriteTurn 0x7d68ec/0x7d69a0 104 行；libccl 2281deca。35 独立表 `abi_turn_fields.md`（ae728984/commit 2d7f1e1）与本节一致。

### 62.1 Init 建立的对象关系（已证）

`Init(y,j,t,KFCTask*)` 0x7d66f0（`y=u64 arg1, j=u32 arg2, t=u16 arg3`）：
- 读 `KFCTask+0x28` 指针及其 `u32[0]`，写 `server+0x330`。
- **零分支**（该 u32==0）：`server+0=y`、`server+0x10=y+j`、`server+0x18=y+j+64*t`；`*(u8*)(y+772)=0`；`server+0x20..0x31f` 清零 768 B；`server+0x320=0`、`server+0x328=KFCTask*`。
- **非零分支**：`server+8=y`、`server+0x320=0`、`server+0x328=KFCTask*`（不写 +0x10/+0x18）。
- `Init(y)` 0x7d67e0：`server+8=y`、`server+0x320=0`、`server+0x328/0x330` 16 B 清零。

与 getter/RpcSrvLaunch 对齐：`C = ctx[184] = y`、`j = ctx[192]`、`t = u16(ctx[196])`；**`server+0x10 = y+j` 正是 ReadWorkMsg 的 `[this+16]`（774 B 拷贝源）**，其 `+50` 即 sendCnt；**`server+0x18 = y+j+64*t` 是 rcvCnt 槽**，其 `+51` 即 rcvCnt；`server+8=y` 是 turn 区基址。若 4 参 Init 前已调用 `Init(y)`（或走非零分支），则 **S(server+8)=C=y**；若只调用 4 参零分支，`server+8` 由其它调用设置（未证）。35 的“S/C 关系未证、别名不能仅凭偏移排除”成立。

### 62.2 等待链（已证静态）

- **ReadAddrMsg(HcclMsg*, turn)** 0x7d8b6c：`slot = server[8] + turn*64`（`turn<<6`）；循环调用 `ReadApiValidMsg(server, out, slot, true)`（0x7d8d78）。返回 true → 打印并返回 1。返回 false → 累计迭代、`GetOpExecCtrlCmd` 检查控制命令。
- **超时/退避（给 35 复核的指令地址）**：
  - 时间单位：`0x7d8bd0 mov x1,#0xca00` + `0x7d8bd8 movk x1,#0x3b9a,lsl#16` = `0x3B9ACA00`（1e9 ns/s）；`0x7d8bf0 madd x0,x2,x1,x0`（start）与 `0x7d8ca8/0x7d8cac/0x7d8cc0 madd x19,x19,x2,x1`（now），`clock_gettime(4)` 在 0x7d8bbc/0x7d8ca0；**clockid 4 = CLOCK_MONOTONIC_RAW**（35 复核）。
  - 20 s：`0x7d8cb0 mov x3,#0xc800` + `0x7d8cb8 movk x3,#0xa817,lsl#16` + `0x7d8cbc movk x3,#0x4,lsl#32` = `0x4A817C800` = 20,000,000,000 ns；`0x7d8cc8 mul x0,w24,x3`、`0x7d8ccc sub x19`、`0x7d8cd0 cmp x0,x19`、`0x7d8cd4 b.cs 7d8d60`（阈值 ≥ elapsed 时继续读）。
  - 更新：`0x7d8bc4 mov w24,#1`；`0x7d8d44 ubfiz w24,w24,#1,#7`（取低 7 位左移 1，1→2→…→64→128→0）。
  - 960 s：`0x7d8d40 mov x0,#0x8000` + `0x7d8d48 movk x0,#0x8475,lsl#16` + `0x7d8d4c movk x0,#0xdf,lsl#32` = `0xDF84758000` = 960,000,000,000 ns；`0x7d8d50 cmp x19,x0`、`0x7d8d58 b.hi 7d9240`（→ `SetErrToWarnSwitch(0)` → 0x7d91c8 → 返回 0）。
  - **单次 vs 累计（按 35 复核修正）**：`w24*20 s` 是**自 start 起的绝对 elapsed 阈值**；到期序列 20、40、80、160、320、640、1280 s（对应 w24=1,2,4,8,16,32,64）；1280 s 到期时 w24 由 64 变 128，随即同一分支检查 `elapsed>960 s` 并返回 0。
  - **可证层级（严格）**：只可说 **ReadAddrMsg 与 ReadApiValidMsg 两层未见显式 sleep/nanosleep/usleep/select/poll**（31/35 已核；4 处 `syscall(178)` 是 gettid）；`CopyAndCheckApiMsg` 的 memcpy/memcmp、`DlogRecord`、`TraceProfSubmit`、`CheckDebugMode` 的深层耗时未知。因此 **1280 s + 一次轮询是“循环+callee 不阻塞”前提下的推导，不是本格实际上界**。
  - **两个“2560”要分开**：1280 s 分支确实把 w24 置为 128（即“若继续则下一阈值 2560 s”的**状态**），但同一分支立即返回，**2560 s 阈值在循环中不可达**；若单次 callee 阻塞使墙钟 elapsed 超过 2560 s，那是 callee 行为，**不能据此证明阈值状态可达**。
  - 180 s runtime 超时落在 160–320 s 日志区间，**仅作假设兼容性**，不是 sleep 或返回点。
  - 另有 10001 次迭代检查 `0x7d8c8c mov w0,#0x2711` / `0x7d8c90 cmp w20,w0` / `0x7d8c94 b.eq 7d8ef8`，到期只重置计数并继续，不构成返回上界。
  - 180 s runtime 超时落在 160 s 与 320 s 两个阈值之间，**仅作假设兼容性**，保留“无本格路径证明”。
- **CheckRcvAddrMsg(HcclMsg*, turn)** 0x7d7624：仅 `ReadApiValidMsg(server, msg, server[8]+turn*64, false)`，无计数器逻辑。
- **CopyAndCheckApiMsg** 0x7d6ae4：`memcpy_s 64 B` → `GenXor` 校验 `+60` → `dsb` → `memcmp(dst,src,64)`；只做撕裂读保护/校验。**按覆盖字节计**：64 B 拷贝覆盖 `+48..+51`（含 +50/+51），但无独立 counter 字节逻辑；不能表述为“不访问 +50/+51”。
- **RcvMsg(opParam, idx, bool)** 0x7d8000：槽位 `server[0] + 776*turn + 49664*idx`（776=0x308，AivAicpuOpParam 大小），`NeedAutoGenMsg` 为真时 `GenMsgByTaskParam` 生成并 `server+800++`；`opParam[48]!=0` 时置 `server[0x220+idx*4]=1`；`turn=(turn+1)&0x3f` 回写 `server[8+idx*4]`。没有**独立** +50/+51 字节操作；生成的 776 B 槽与整块拷贝会覆盖这些字节。
- **WriteTurnCnt(turn)** 0x7d69a0：`[server+8]+0x4008+64*turn` 写 8 B 零；`[server+8]+0x5008+64*turn` 读 u64、+1、写回、`dsb`。
- **WriteFinishWhenAllFinalize(turn)** 0x7d68ec：`[server+8]+0x5008+64*turn` 写 8 B 常量 `0x112210f6ca9997ff`、`dsb`。
- **ReadApiValidMsg 0x7d6f20（0x2ac）未交付**——它是 ReadAddrMsg/CheckRcvAddrMsg 的唯一实际判定函数，也是目前最可能的 `sendCnt` 读/比较与 `rcvCnt` 写回点。

### 62.3 同址/别名判定（地址方程；宽泛可达区间 vs 精确可重叠集合）

- 读：sendCnt 字节 `C + j + 50`（1 B）；rcvCnt 字节 `C + j + 64*T + 51`（1 B，**T = u16[ctx+196]，固定槽索引**，不是任意 t）。
- 写：`S + 0x4008 + 64*τ`（8 B 零）与 `S + 0x5008 + 64*τ`（8 B u64/常量，`τ∈[0,63]`，由 `RunRpcServerApi` 的 `and w0,w0,#0x3f`（0x7b39e4）与 turn 清零/递增证明）。
- 两个 8 B 数组是**离散并集**（每 64 B 只有 8 B 活字，其余 56 B 是空洞）：
  - A = `∪_{t=0..63} [S+0x4008+64t, S+0x4010+64t)`；
  - B = `∪_{t=0..63} [S+0x5008+64t, S+0x5010+64t)`；
  - 其连续包络分别是 `[S+0x4008, S+0x4FD0)`、`[S+0x5008, S+0x5FD0)`，两者之间的 `[0x4FD0,0x5008)` 也是空洞；**包络不等于精确集合**。
- 若 **S==C**：
  - **精确 send 集合**：`j+50 ∈ A∪B` ⇔ `j ∈ ∪_t [0x3FD6+64t, 0x3FDE+64t) ∪ ∪_t [0x4FD6+64t, 0x4FDE+64t)`（每段 8 B、步长 64 B）。
  - **精确 recv 集合**：`j+64T+51 ∈ A∪B` ⇔ `j+64T ∈ ∪_t [0x3FD5+64t, 0x3FDD+64t) ∪ ∪_t [0x4FD5+64t, 0x4FDD+64t)`；T 固定为 u16，只加 64 的倍数，故对给定 j 仅当 `j mod 64` 落入对应 8 B 残类（send：50..57；recv：51..58 的模 64 区间）才可能命中，不是“覆盖所有 j”。
  - 连续大区间 `[0x3FD6,0x4F9E)`/`[0x4FD6,0x5F9E)` 与 `[0x3FD5,0x4F9D)`/`[0x4FD5,0x5F9D)` 只能标为**包络**，不能标精确。
- 若 **S≠C**：无论 j/T 如何都不别名；S/C 依赖 server 对象与 Init 顺序（4 参零分支把 C 存 `server+0`，只有 `Init(y)` 或非零分支写 `server+8`；两个 Init 调用点使用**不同 server 偏移** 0xd10/0xcf0），未证。
- 若 `S==C` 且 `j+776 ≤ 0x4008` 且实际 T 满足 `j+64T+52 ≤ 0x4008`，则 param/slot 区整体低于 turn 区 → 不重叠；该条件依赖 j 与 T，当前均未证。
- 结论：**别名未建立、也未排除**；35 的“字面偏移/宽度不同≠不同址”正确。待 `ctx[192]`（j）与 `ctx[196]`（T）的实际值/构造与 server 对象确认后判定。

### 62.4 静态路径与本格实际支路（分开）

- **静态已证**：API 路径 = `RpcSrvLaunch → RunRpcServerApi → ReadAddrMsg(turn)/AddTaskForHcclMsg → WriteTurnCnt`；等待门槛 = `ReadApiValidMsg(slot[turn])` 的 `+48` 有效性魔数（见 §63）；无消息时循环以 20 s×w24 为**超时日志阈值**（非 sleep）。
- **CFG 确认（按 35 要求）**：1280 s 分支 `0x7d8d44` 写 w24=128 后，`0x7d8d58 b.hi 0x7d9240` → `0x7d91c8` → `0x7d8fcc` 返回 0；**0x7d9240 没有任何回到 0x7d8d60 的边**，因此 w24=128 只被写入、从未作为下一轮 deadline 使用，**2560 s 阈值按 CFG 不可达**。callee 长耗时只抬高墙钟，不改变该结论。
- **条件式** ~1280 s + 一次轮询的推导，仅在“helper 返回、控制回到循环”前提下成立，不是本格实际上界。
- **本格实际**：无任何运行期证据表明 Grouped 走 RunRpcServerApi（kernelType 未落盘、无 Init 重载调用点、无 turn/slot 日志）；180 s plog 超时与 ReadAddrMsg 退避**相容但非证明**。若 Grouped 走 `AicpuRunRpcServerForMC2/V2` 或 stage-wait 路径，等待条件不同。
- **仍缺写者**：+50/+51 在已交付的全部函数中无写指令；可能在 `ReadApiValidMsg`、`GenMsgByTaskParam`，或 host/runtime/TS 侧写共享工作区。

### 62.5 合并后的最小请求（含 35 的三项，不重复已交付）——**已由 abi-preflight6 满足，见 §63/§64**

1. **`ReadApiValidMsg(HcclMsg*,HcclMsg*,bool)` 0x7d6f20（0x2ac）**——第一优先，唯一未交付的判定函数。
2. `GenMsgByTaskParam(AivAicpuOpParam*)` 0x7d7e20（0x1b8）——自动生成消息时是否写 sendCnt/rcvCnt。
3. `ResInit` 0x760468（0x98）+ `ResInitV2` 0x754568（0x238）——`ctx+184/192/196`（y/j/t）的构造链；若不足，再对 `AicpuKfcProcess::AicpuRpcResInit` 0x7b9dc8（0x3074）只做**定向 grep**：`str .*\[x?, #(18[4-9]|19[0-9])\]` 附近的写入，不要求整段反汇编。
4. Init 两重载的调用点与实参（35 项）；`AivAicpuOpParam` 布局（+50/+51 与 u16 slot 索引）确认。
- **不重复**：ReadAddrMsg、CopyAndCheckApiMsg、RcvMsg、CheckRcvAddrMsg、Init、WriteTurnCnt/WriteFinish、RunRpcServerApi、GetSendCnt/GetRecvCnt、ReadWorkMsg/CheckRcvWorkMsg/AddTask/AicpuCcOpExe/SrvLaunch/RpcSrvLaunch/V2。

**结论**：等待链已从“轮询 sendCnt”细化到“ReadAddrMsg 按 turn 槽调用 ReadApiValidMsg”；§63 进一步证明 gate 是 `+48` valid 魔数、`+50/+51` 仅在 GenMsg 常量初始化中出现，普通 rcvCnt 递增仍缺；j/T 写者已定位但取值未证；Grouped 实际支路未证。root cause 仍 **unresolved**，不回 `confirmed_environment_blocked`；`mc2_server` 仍 `supported_but_unproven`。设备暂停。

## 60. A15-large-v1 原件独立复核（有限统计联审）

复核 31 归档 `grouped-deepdive/a15-large-v1-cell/`（commit 0d8cc1a）：`SHA256SUMS` 自身 `2adc7ef64b035b01d820a3523026dc8c4e043e4946b216e4a44a2fb17c0becf3`，`shasum -c` **17/17 OK**。两 rank result 逐字段一致：`pad{967→968,1.0}`、`row_block 0(0,484)/1(484,968)`、`equivalence.shape [967,5120]`、`workspace_bytes 115,818,496`、`gather_ms 1.13913/1.21021`、`sync_returned=true`、`wall_ms 423.578`。统计：`max_abs=2.0`、`mean_abs=5.1305116357980296e-5`、`ref_abs_max=out_abs_max=744.0`、`diff_nonzero=510/4,951,040`、`max_error_index=1040755`、`position=[203,1395]`、`ref=400.0/out=402.0`、`max_rel_abs=1.0`、`relative_l2=4.9197991043609655e-5`（denom 293,496.6875、zero_reference=false）、`bit_equal=false`。**算术自洽**：`203*5120+1395=1040755`；`mean_abs*diff_total=254.0137`，与 510 个非零元素均值 0.498 一致；`max_rel_abs=1.0` 不可能来自 max_error 的 ref=400 点（该点 2/400=0.005）；其样本位置与对应 ref/output **未落盘**，近零参考 `clamp_min(1e-12)` 只是可能解释之一，具体成因**未证**；不得与 max_error 位置混用、不得作门。35 的 `run_probe 8e308804` 第 379-380 行确为 **AND**：`within_bounds = max_abs<=0.03125 and mean_abs<=1e-4`；两 rank 原件均 **`within_bounds=false`**（max 2.0 超限、mean 5.13e-5 在限内），31 此前“true”为转述笔误，原件/SUMS 未回填。plog 两 rank `[ERROR]` 0 条；device 日志 3 条匹配均为 INFO（`Not set aicpu model timeout`、`halGrpAttach timeout:0ms`、`Get enable op execute timeout config ... timeout[547]s`），无 507046/507015。**判读**：仅 large 形状功能/尺度诊断；`max_abs=2.0` 在 ref≈400 处约 1 个 BF16 ulp（7 fraction bits，ulp=2），仅该 max 样本，不代表其余 509 个差异；逐元素分布/ULP、max_rel 位置、其它形状/生产/性能均未证；不称精度通过；A16/benchmark 未授权；official/CSV 未动。

## 61. §15.1 KEEP_ALL 预算复核（条件式 PASS，含两处数字待统一）

核 35 `e9b2c4a`（设计文档 `db6fc7b8`、manifest `59dfd432`）：
- **算术正确**：大 per-call 137,625,600 = 115,818,496+16,850,944+4,956,160；+X=138,674,176（132.25 MiB）；resident 105,962,496（101.05 MiB）；`floor((1024−101.05−99.17)/132.25)=6`；W2/N6=8 → 1,058 MiB retained >1 GiB，确实放不下；W1/N4=5 → 693,370,880 B（661.3 MiB）retained。小 per-call 17,301,504（16.50 MiB）+X=18,350,080（17.5 MiB）；resident 327,168（0.31 MiB）；`floor((1024−1.5)/17.5)=58`；W3/N20=23 成立。X 明确为未验证预留、1 GiB 明确不是进程 RSS 保证、allocator/adapter/runtime 未计、两臂共享需相加、ABBA 进程顺序与 `max_rel_abs` 语义均已写入——32 的条件满足。
- **已修（bd7685a，设计 e91d4df8）**：旧的 676/807/866 MiB 与“~50 MiB transient”残留已删除；统一改用 **94.44 MiB 下界**（4×float32 + BF16 full/ref），W1/N4=5 → 856.75 MiB、6 次 → 989.0 MiB（96.6%，X≤6.83）、5 次 X≤34.45；小格改为 379.5 MiB retained + 23.0 MiB reserve = 402.5 MiB；ABBA A1-B1-B2-A2 写入；`max_rel_abs` 与 `max_error_*` 明确不可互解。
- **已修（c050655，设计 a1df799c）**：①改为 103,987,200 B = **99.17 MiB（about 104 MB）**，并注明该替代口径**含 y 一次、不可与 per-call retained 叠加**；以它替代 94.44 MiB 下界时 W1/N4 ≈ **861.5 MiB**，两种读数替代关系明确。②y 重复计数已澄清为“not additive”。③1 GiB 明确为实验新增预算、非 RSS；allocator `memory_allocated/reserved` 或非计时区 `empty_cache` 留待 benchmark 实现。
- 结论：**条件式 PASS**（32 口径）；仅剩 benchmark 时的 allocator 实测项，不改变“5 次大调用可放、8 次不可放”；A16 未实现、设备未授权。

## 63. abi-preflight6（433b88f，SUMS b8775ff6）：等待门槛是 valid 魔数，不是 rcvCnt

**材料**：ReadApiValidMsg 0x7d6f20（178 行）、GenMsgByTaskParam 0x7d7e20（117）、ResInit 0x760468（45）、ResInitV2 0x754568（149）、init-call-sites、ctx 偏移 grep、rac_server.h（319 行）；libccl 2281deca。

### 63.1 ReadApiValidMsg（0x7d6f20）

- 入参：server、out `HcclMsg*`、slot `HcclMsg*`、consume bool。
- **唯一 gate**：`u32[slot+48] == 0x5CDF123A`（0x7d6f3c-0x7d6f50）；不匹配 → `TraceProfSubmit` 并返回 `rc != 0`（通常 0）。
- 匹配 → `CopyAndCheckApiMsg(server,out,slot)`（0x7d70e4；64 B 拷贝 + GenXor(+60) + memcmp）；失败 → 0。
- consume=true → **`u32[slot+48] = 0xA320EDC5`**（0x7d7118-0x7d7124，消费标记）；随后 `CheckDebugMode(out)`、`dsb st`、返回 1。consume=false 不写消费标记。
- 本函数按 **valid 整字**操作：`0x7d6f40 ldr w1,[slot+48]`（4 B，覆盖 48–51），consume 时 `0x7d7118-0x7d7124 str w1,[slot+48]`（4 B，同覆盖）；**没有独立 +50/+51 字节操作**，但这两个字节被整字覆盖。35 字节表（`abi_turn_fields.md` 34cb1388）与本节一致：GenMsg `+44 str 4B`、`+48/+49 strb`、`+50 strh 2B`；`CopyAndCheckApiMsg` 是 64 B bulk；getter 是 1 B。
- `slot`（gate +48）与 `AivAicpuOpParam`（GenMsg valid +44）的基址是否同一对象、是否恰好相差 4 B **未证**；仅可记“4 B 位移的外观”，不推断布局冲突或同一 buffer。
- 本层无 sleep/显式等待（4 处 syscall 均为 gettid 178）；深层 `CopyAndCheckApiMsg`/`DlogRecord`/`TraceProfSubmit`/`CheckDebugMode` 耗时未知。

### 63.2 GenMsgByTaskParam（0x7d7e20，AivAicpuOpParam 生成）

写入：`+0=task[109]`、`+4=task[110]`、`+24=buffer`、`+40=task[106]`、`+48=GenMsgIsLastMsg 字节`、`+49=task[105]`、**`+44=0x5CDF123A`**、**`+50..51=半字 0x1234`（sendCnt=0x34、rcvCnt=0x12）**、`+52=task[113]`、`+53=task[112]`、`+54=task[80]`、`+55=task[104]`、`+32=task[88]`。
- 结论：+50/+51 在 libccl 内**有写者**，但只是自动生成消息时的常量初始化；**普通递增仍未在已交付函数中出现**。

### 63.3 rac_server.h 的协议语义与布局边界

- AivAicpuOpParam 源码顺序：commType/opType/sendBuffer/recvBuffer/count/strideLen/hcclDataType/valid/isLast/funID/sendCnt/rcvCnt/everyTurnRsp/everyTurnWait/totalTurnCnt/res[9]。
- `TurnNotifyRunSetMsg`/`TurnNotifyRun`：`sendCnt = curTurn`；`TurnWait`：`if (rcvCnt >= totalTurn) { rcvCnt = 0; }`。语义：sendCnt=轮次发布（生产者写），rcvCnt=完成计数（消费者读并归零）。
- **header 不等于已加载二进制**：该头文件描述的结构约 64-72 B，而 libccl 的 ReadWorkMsg 拷贝 774 B、RcvMsg 槽步长 776 B → 已加载 AICPU 结构更大，不能直接套 header 的 `sizeof`/offsetof。当前仅有 GenMsgByTaskParam 指令证实 AivAicpuOpParam 写路径的 +44 valid/+48 isLast/+49 funID/+50 sendCnt/+51 rcvCnt；**64 B HcclMsg 的 +48 magic 布局只有 ReadApiValidMsg 指令证据，无 header 证据**。

### 63.4 Init/ctx 构造链（31 交付）

- 4 参 `Init` 调用点 **0x76192c**（RunAicpuRpcSrvLaunch）：`x0=server r23+0xd10`、`x1=[ctx+184]`、`w2=[ctx+192]`、`w3=ldrh[ctx+196]`、`x4=KFCTask*`。
- `Init(y)` 调用点 **0x7b67c4**（`AicpuKfcProcess::AicpuRunRpcServerForApi`）：`x0=x20+0xcf0`、`x1=[ctx+184]`。**两个调用点的 this 构造式不同（r23+0xd10 vs x20+0xcf0）；r23/x20 的基址来源尚未追，不能仅凭偏移差断言是不同对象**——对象身份未证。
- `ctx+184` 写者 **0x7b9f50**（AicpuRpcResInit 区）；`ctx+192` 写者 **0x76182c**（`*(u32*)(KFCTask+96)`）；`ctx+196` 写者 **0x761834**（`*(u16*)(KFCTask+100)`）；`ctx+198=KFCTask+102`；`ctx+168=KFCTask+72`。
- ResInit 0x760468 仅 `*(HcclCombinOpParam**)arg → AicpuRpcResInit`；ResInitV2 0x754568 走 `AicpuRpcResInitV2(HcclOpResParam*,byte)` 或 `hrtSetWorkModeAicpu(1)`。
- 4 参零分支把 C 存 `server+0`、`y+j` 存 `+0x10`、`y+j+64T` 存 `+0x18`；只有 `Init(y)` 或非零分支写 `server+8`。两个调用点 this 构造式不同（0xd10 vs 0xcf0）但**对象身份未证**，故 **S=C 需同一对象 + Init(y)（或非零分支），未证**。

### 63.5 模型修正

- API 路径等待条件 = **slot+48 的 valid 魔数**，不是 sendCnt/rcvCnt；因此“rcvCnt 永不递增导致 Grouped 挂起”这一具体机制**不成立于 API 路径**。
- rcvCnt 的**普通递增**未在已交付函数中出现；`rac_server.h` 显示 AIV 侧只在 TurnWait 归零。已交付函数中对 `+50..+51` 的覆盖来自 `GenMsgByTaskParam` 的 `u16 0x1234` 常量以及整字/整块拷贝；**字面 store 偏移搜索不能证明全 libccl 唯一写者**。
- 若 Grouped 走 AIV `rac_server` 协议，则需 AIV 侧证据（未交付）；若走 API 路径，则需 HcclMsg 生产者证据（未交付）。Grouped 实际 server 对象/Init 分支/协议路径仍未证；180 s 仅相容。

## 64. 最小路径判别方案（CPU/只读设计，不执行设备）

**目标**：在不新增设备动作的前提下，判定 Grouped 是否走 `RpcSrvLaunch → RunRpcServerApi → ReadAddrMsg → ReadApiValidMsg`，以及 server 对象、Init 分支、j/T、协议与轮询期望/实值；静态不可判时，给出未来最小观测点。

| # | 观测点 | 现有证据 | 尚缺字段 | 可区分假设 | 最低成本取证 |
| --- | --- | --- | --- | --- | --- |
| OP1 | Grouped launch 描述符 | SrvLaunch 0x763108 的 desc nibble/sub 分支；A12 样例入口 `aclnnGroupedMatMulAllReduce` | `CommKfcParamDesc` 位域；Grouped 实际 desc/sub；`KFCTask+96/+100` 来源 | CommKfcDispatcher / RpcSrvLaunch / MC2 V1/V2 | host 侧 aclnn 实现静态符号/源码（安装头文件已检索阴性） |
| OP2 | server 对象 | 两 Init 调用点 this 构造式 0x76192c(r23+0xd10) / 0x7b67c4(x20+0xcf0)；RunRpcServerApi 只用前者；ForApi 0x7b6770/0x430，真正 @plt caller **0x760930**（未导出）；MC2V2/MC2 callers 0x763880/0x7639dc | r23/x20 基址来源与对象身份；0x760930 所属函数及其上游分支 | 同一对象 / 不同对象；Grouped 走 ForApi 还是 RpcSrvLaunch/MC2 | 追 r23/x20 基址来源；0x760930 边界/上游调用点（小窗口） |
| OP3 | Init 分支与 j/T | 零/非零分支读 `u32[*(task+40)]`；j/T 写者 0x76182c/0x761834 | KFCTask 偏移 40/96/100 的类型/语义；host 填充值 | zero 分支（S 未设）/nonzero（S=y）；j 大小；T=rank 数？ | host tiling 代码/二进制站点；**安装头文件无 KFCTask/CommKfcParamDesc，offsetof 不可得** |
| OP4 | HcclMsg valid 生产者 | gate +48=0x5CDF123A、consume 0xA320EDC5；GenMsg 写 AivParam +44；**新候选 0x7d6850 `str w4,[x0,#44]`**（未导出局部函数 0x7d67f4-0x7d68ec） | 0x7d6850 的函数体、x0 对象/调用点；HcclMsg +48 写者；Grouped 是否用 API 区 | AivParam 初始化 / HcclMsg 转码 / API valid 发布；host 是否发布 API 消息 | 0x7d67f4-0x7d68ec 反汇编 + 直接 callers + 各 caller 的 x0 来源（abi8） |
| OP5 | 轮询期望/实值 | turn=ctx+0xf31d4（0..63）；slot=server[8]+64*turn；~1280 s 条件上界 | 运行期 turn/slot/+48 | 无消息 / 错 turn / callee 阻塞 | 先 OP4；仅当仍不可判，未来授权窗口做一次 workspace 只读快照或入口日志 |
| OP6 | AIV rcvCnt 路径 | **PostMsg 0x7d6800 在 TwoStageWait 写 +51（1 B，`ctx+352-1`/`ctx+352`）**；Init 零分支 `server+0x18=y+j+64T` 与 GetRecvCnt 条件式同字节；rac_server.h 的 TurnWait 清 rcvCnt；op-config 阴性 | Grouped 是否进入 TwoStageWait（`GetPreparePosition/GetTaskType/ctx+0xf31bc` 运行值）；`ctx+352` 递增者/上界 | stage-wait ack 路径 / API valid 路径 / MC2 | ①`ctx+352` 递增者小窗口；②GetPreparePosition/GetTaskType/ctx+0xf31bc 来源（Init 字段/host tiling）；不整段扫 |

**当前合并请求（交 31，全部静态、不重复已交付）**：
1. `0x7d67f4-0x7d68ec`（0xf8 B）完整反汇编（若入口非 0x7d67f4，补入口/序言）。
2. 该局部函数的直接 `bl` 调用点与 caller 函数名/边界；每个调用点的 **x0 来源**（server/slot/base+offset）。
3. 与 35 字节覆盖表合并：该函数是否还写 `+48/+50/+51` 及宽度。
4. 追 **r23/x20 基址来源**与 0x760930 所属函数/上游分支（OP2）。
- 说明：头文件检索已阴性（无 `KFCTask`/`CommKfcParamDesc`）；0x5CDF123A 站点、ForApi/MC2 callers、op-info 阴性均已交付，不重复。
- **证据层级**：已加载布局仅以指令 offset/宽度为准（AivAicpuOpParam +44/+48/+49/+50/+51 有 GenMsg/ReadApiValidMsg 指令证据；HcclMsg +48 只有 gate 指令证据；774/776 大小说明不得套 `rac_server.h` 的 `sizeof`；安装头文件无对应结构，offsetof 无来源）。
- **运行期值（turn/slot/+48/j/T）无证据**；“Grouped 走/不走 API 路径”保持未证；不新增 GO/SSH/设备动作，31 仍为唯一远端 owner，A15/benchmark 范围不变。

### 63.6 与 35 的 timing/fields 收口（df1b7a6）

35 最终文档：`readaddrm_msg_timing.md` **2bba59c0**、`abi_turn_fields.md` **9a245cbe**，commit **df1b7a6**、manifest **c0421638**（doc-only）。双方接受以下合并措辞：
- 时间：仅 ReadAddrMsg/ReadApiValidMsg 两层未见显式 sleep/nanosleep/usleep/select/poll（syscall 178=gettid）；CopyAndCheckApiMsg/Dlog/Trace/CheckDebugMode 深层耗时未知；1280 s + 一次轮询是“helper 返回、控制回循环”前提下的推导，**不是本格实际上界**；180 s 仅为 160–320 s 日志区间。
- CFG：1280 s 分支 w24 64→128、cmp、`b.hi 0x7d9240`、`SetErrToWarnSwitch(0)`→`0x7d91c8(w19=0)`→`0x7d8fcc`→ret；`0x7d9240` 无回到 `0x7d8d60` 的边 → **2560 s 阈值状态不可达**；callee 长耗时只抬高墙钟。
- ABI：gate/consume 事实入档；`+50..+51` 的已交付覆盖来自 GenMsgByTaskParam 的 0x1234 常量与整字/整块拷贝，**不称全 libccl 唯一写者**；`rac_server` 的 sendCnt=curTurn、TurnWait 清 rcvCnt；S/C/j/T 未证，离散别名集合未建立/未排除。
- 后续以 §64 的最小实际路径观测设计为主；无新冻结轮/运行代码/设备任务。

**§64 状态**：abi-preflight7 已交付（95e1ddc/SUMS 09178185）：头文件检索阴性、ForApi caller 0x760930、0x5CDF123A 写者候选 0x7d6850、op-info 阴性，已并入 §64/§65；下一最小请求为 `0x7d67f4-0x7d68ec` 函数体 + 调用点 + x0 来源 + r23/x20 基址来源。未执行任何设备/SSH 动作。

## 65. abi-preflight7（95e1ddc，SUMS 09178185）：OP4 producer 候选 0x7d6850

### 65.1 头文件与符号

- 安装的 `include/opp` 中**未找到** `CommKfcParamDesc` 或 `struct KFCTask`（toolkit 9.2.1 `V100R001C12SPC002B201`）→ **无编译 offsetof 可用，只能以二进制站点为准**；这也说明 header 布局与已加载实现的对应关系不能靠该安装头文件建立。
- `AicpuKfcProcess::AicpuRunRpcServerForApi` nm `0x7b6770` size `0x430`；真正 `@plt` 调用者为 **0x760930**（dynsym 无归属的局部函数）；`RunAicpuKfcSrvLaunch` 内 0x763880/0x7639dc 分别是 `AicpuRunRpcServerForMC2V2`/`ForMC2` 的调用。其余列出的 `ForApi+offset` 行是最近符号命名伪影，不是 ForApi 的调用者。

### 65.2 0x5CDF123A 站点

- 字节扫描 libccl/opapi/hccl **0 命中**（常量用 movz/movk 立即数构造，不是数据字）。
- 比较站点：`ReadMsgFromMemory(HcclMsg)` `[x19+48]`、`ReadMsgFromMemory(HcclMsgExt)` `[x19+8240]`、`RetryProcess` `[x1+44]`、`ReadApiValidMsg` `slot+48`、`CheckRcvWorkMsg`、`AicpuKfcRpcServerV2::ReadValidMsgExtArea` `[x19+48]`、`V2::ReadValidMsg` `[x2+48]`。
- 写站点：`GenMsgByTaskParam` `+44`（0x7d7f2c）；以及 **0x7d6850 `str w4,[x0,#44]`**，位于 `Init(y)` 0x7d67e0（止于 0x7d67f4）与 `WriteFinishWhenAllFinalize` 0x7d68ec 之间的**未导出局部函数**（候选边界 `0x7d67f4-0x7d68ec`，0xf8 B）。
- 站点上下文：`ldrh w3,[x2,#704]`、`ldrb w2,[x2,#706]`、`strh w3,[sp,#24]`、`str w4,[x0,#44]`；x0 为目的指针，x2 为带 +704/+706 字段的 task/tiling 源。**函数体未交付**，尚不能判定 x0 指向 `AivAicpuOpParam`、64 B `HcclMsg` 的 base-4，还是别的对象。
- 分类假设：①AivAicpuOpParam 初始化器（+44 valid 且可能写 +48..+55）；②HcclMsg→AivParam 转码（只写部分字段）；③API valid 发布到不同对象。字面 +44 与 gate +48 的关系取决于真实指针差，**暂不认定同一或冲突**。

### 65.3 op-config 阴性

Grouped op-info/kernel json 与 `.o` strings 中**未检出** `rac_server`/`TurnWait`/`TurnNotify`；这只是检索范围内的阴性，不证明 kernel 二进制不含该协议。

### 65.4 最小后续请求（abi-preflight8，不回到全库扫段）

1. `0x7d67f4-0x7d68ec`（0xf8 B）完整反汇编；若入口不是 0x7d67f4，补 0x7d67f4-0x7d6838 的入口/序言。
2. 该局部函数的**直接调用点**（`bl` 目标）与各 caller 的函数名/边界；每个调用点的 x0 来源（server/slot/base+offset）。
3. 与 35 字节覆盖表合并：确认该函数是否还写 `+48/+50/+51`，以及写入宽度。
- 不重复已交付的 ReadApiValidMsg/GenMsg/ReadAddrMsg/Init/Write*/ctx 偏移/常量站点；不做全库扫段。

### 65.5 OP4 状态

生产者候选现为：`0x7d6850`（AivParam +44 写者，函数体未证）、`GenMsgByTaskParam`（+44/+50..51 常量）、host 侧未知写者。**HcclMsg +48 的写者仍未找到**；0x7d6850 是否写 +48 待其函数体。

### 65.6 与 35 的 abi7 字节表合并（7b639dd / b83171ba）

35 的表与 §63 一致：gate/consume 为 `slot+48` 的 4 B 整字（覆盖 48–51）；GenMsg `+44` 4 B、`+48/+49` 1 B、`+50` u16（覆盖 50–51）；`CopyAndCheckApiMsg` 64 B bulk；getter 各 1 B；abi7 写者 `0x7d6850` 在 `+44` 写 4 B。统一表述为“按 valid 整字访问，未见独立字节 counter 操作”，撤回“libccl 唯一写者”。

**对象关系边界**：
- `param = [ctx+184] + u32[ctx+192]` 与 `recv = param + (u16[ctx+196]<<6)` 的**算术关系**已由 Init 零分支（`server+0x10=y+j`、`server+0x18=y+j+64T`）与 getter/ReadWorkMsg 指令证明；这不是头文件 offsetof，而是二进制算术证据。
- 仍未证：j（`KFCTask+96`）的值与 Init 分支；`slot=[server+8]+64*turn` 与 `param=y+j` 是否同一 buffer/固定关系（需 j 与 S/C）。
- AivAicpuOpParam 的**字段级布局**只在 `+0..+64` 有 GenMsg 写证据；`+64..+776` 内部结构无字段级证据（ReadWorkMsg 仅整块拷贝 774、RcvMsg 步长 776）。不把 header 的 64 B 结构外推到 776 B。
- 35 的 (a) 已包含在 abi8 的 `0x7d67f4-0x7d68ec` + 调用点请求中（重点是 0x7d6838-0x7d6850）；(b) 的布局部分已由现有 Init/getter 算术证据覆盖，剩余是 j/分支取值，不另开全量布局反汇编。

## 66. abi-preflight8（最终补窗 SUMS d2a4043b，替代 326a175f）：PostMsg 写 +51；ForApi 属 ClearOpRes

### 66.1 PostMsg 0x7d6800（`AicpuKfcRpcServer::PostMsg(uint)`，const）

- **更正 abi7 归属**：0x7d67f4-0x7d67fc 是 3 条 nop；写者不是未导出局部函数，而是 `PostMsg(uint)` 0x7d6800。
- 函数体（0x7d6800-0x7d68b4）：`0x7d6828 ldr x0,[x0,#24]` → **dest=*(this+0x18)**（entry this 与目的指针分开）；`str w4,[x0,#44]` 写 4 B valid `0x5CDF123A`；**`strb w1,[x0,#51]` 写 1 B = w1（PostMsg 的 uint 参数/turn；w1 在函数内未被改写）**；读 GOT+0xc90 的 +704(u16)/+706(u8) 仅用于 `AivAicpuOpParam::PrintMsg`；返回 1。**不写 +50**。
- 程序序：`+44` 写 →（无屏障）→ `+51` 写 → `PrintMsg`/string dispose → **唯一 `dsb st` 在 0x7d6884**；无 `stlr`。即 ack 字节的可见性屏障在 PrintMsg 之后，二者之间只有程序序。
- 直接调用者：0x7c7f44 与 0x7c7f90，均在 `AicpuKfcDeprecatedProcess::RunRpcServerTwoStageWait`（+0x198/+0x1e4）→ **stage-wait 协议的消息发布者**；参数分别为 `[ctx+352]-1` 与 `[ctx+352]`。
- 与 getter 对齐：`GetRecvCnt` 读 `[ctx+184]+u32[ctx+192]+(u16[ctx+196]<<6)+51`；Init 零分支 `server+0x18=y+j+64T`；PostMsg 写 `dest+51`。**条件式同字节**的 P1–P4 现状：
  - **P1 已证**（abi11）：TwoStageWait 序言 0x7c7dc8 `mov x19,x1`，PostMsg this=x19=x1；RpcSrvLaunch 0x76197c/0x761980 置 x1=`server_A`。
  - **P4 已证**：0x7c7dd0 `mov x20,x0`；RpcSrvLaunch 0x76198c x0=x28(ctx_R)，0x761994 bl。
  - **P3 在本轮已审范围内成立**：方法区 0x7d66f0–0x7da9bc 定向 `str ..,[x?,#24]` 15 命中，除 Init4 零分支 `stp x0,x3,[x19,#16]`（server+0x18）外无 server+0x18 写者；`PrintMsg` 0x7c24e0/0x654 不写 dest。**仅限已审函数/指令形式，不作全库排除**。
  - **P2 仍未证**：Init4 零分支取决于 `[server+816]=*(KFCTask+40)` 的首 u32 是否为 0。
  - 因此同字节结论仍条件于 P2（以及已审范围外的可能写者）；非零分支只写 server+8，不成立。gate `slot+48` 与 valid `+44` 要成同一字还需 `j+64T = 64*turn + 4`（slot=dest−4），**未证**。
- +44 valid 与 API gate 的 `slot+48` 不是同一字节：PostMsg 写 AivAicpuOpParam 的 +44，gate 读 64 B HcclMsg 的 +48；两者对象/协议不同，**不推断同一或冲突**。

### 66.2 ForApi 的调用链与 server 基址

- `AicpuRunRpcServerForApi` 0x7b6770/0x430；真正 `@plt` 调用者 **0x760930**，位于 `RunAicpuKfcClearOpRes`（Base+0x400），传 `x0=x19(ctx)`、`x1=x21` → **ForApi 是 ClearOpRes 清理路径，不是正常 launch 分支**。
- Server 基址（用符号式，避免绝对地址在未定控制流前传播）：
  - `server_A`：`0x7618dc adrp x23,0x191f000`；Init 前 `0x761918 add x0,x23,#0xd00`、`0x761928 add x0,x0,#0x10` → `server_A = ADRP(0x191f000)+0xd00+0x10 = 0x191fd10`。
    *注意*：0x7620b4 还有一条 `add x23,x23,#0xd00`，但它位于 Init 之后的错误/日志分支（0x762008→0x76209c→0x7620b4，随后 `b 0x7614a4` 返回），**没有回到 0x76192c 的边**，因此不参与 Init 的 this 计算；若该 mutation 真的先于 Init，算术会是 0x1920a10——当前控制流不支持这一点。
  - `server_B`：`0x7b678c adrp x20,0x22c7000`；`0x7b6794 add x21,x20,#0xb50`；`x21+0x190=0x22c7ce0` 是 `ldar` 状态字；`Init(y)` 的 this 为 `x20+0xb50+0x1a0 = 0x22c7cf0`；`0x7b67e8 add x20,x20,#0xb50` 之后调用 `RunRpcServerApi(ctx, x20+0x1a0=0x22c7cf0, ...)`，同一值。
- 因此两条路径使用**两个不同静态全局 server 对象**（`server_A` vs `server_B`，按符号记）；但“server 基址 vs ctx 基址”的 S==C 仍未建立（ctx 是另一指针，Init 只建立 y/j/T 映射）。

### 66.3 计数器写者矩阵（更新）

| 字节 | 已证写者 | 宽度/值 | 路径 | 备注 |
| --- | --- | --- | --- | --- |
| +44 valid | PostMsg 0x7d6850 | 4 B 0x5CDF123A | TwoStageWait（recv 槽） | 与 gate +48 不同结构 |
| +44 valid | GenMsgByTaskParam 0x7d7f2c | 4 B 0x5CDF123A | RcvMsg 自动生成 | AivParam |
| +48 isLast | GenMsg 0x7d7f08 | 1 B | 同上 | 非 valid |
| +50 | GenMsg 0x7d7f30 | u16 0x1234（覆盖 50–51） | 同上 | 常量；**无普通 sendCnt 写者** |
| +51 | **PostMsg 0x7d6858** | **1 B = turn 参数** | TwoStageWait | 与 GetRecvCnt 条件式同字节 |
| +51 | GenMsg 0x7d7f30 | u16 高位 0x12 | 生成消息 | 常量 |
| 0–63 | CopyAndCheckApiMsg 0x7d6b24 | 64 B bulk | API 读校验 | 非计数器 |
| +48 gate | ReadApiValidMsg 0x7d6f40 / consume 0x7d7118 | 4 B | API 路径 | valid/consume 标记 |

- **HcclMsg +48 的生产者仍未找到**；**+50 的普通写者仍未找到**；+51 已有 PostMsg 的 turn 写者（stage-wait），但仍无“普通递增”证据。

### 66.4 对 Grouped 的含义与下一步

- 若 Grouped 走 `RunRpcServerTwoStageWait`：PostMsg 会在 recv 槽写 `+44`/`+51=turn`，AIV `TurnWait` 读 `rcvCnt` 并归零；这给出一条完整的 stage-wait 计数器链。
- 若 Grouped 走 `RunRpcServerApi/ReadAddrMsg`：gate 是 HcclMsg `+48` valid，PostMsg 不在该路径；生产者仍在 host 侧。
- Grouped 实际分支仍未证（OP1/OP2）；180 s 仅相容。
- 最小后续：不需新反汇编即可先做 OP1/OP2 的静态分派图；仅当要证 Grouped 的 HcclMsg 生产者时，才对 host 侧库做**指令立即数**扫描（此前 0x5CDF123A 字节扫描只覆盖数据字，libccl/opapi/hccl 数据命中 0）。无设备/SSH。

### 66.5 PostMsg 两处调用条件、ctx+352 与分派关系（最终补窗）

**调用点语义（RunRpcServerTwoStageWait）**：
- 0x7c7f20：`CheckRcvWorkMsg(opParam, 0, [ctx+352])` —— 先等待 `opParam[50]`（sendCnt）≥ `[ctx+352]`；该函数上限 10001 次/6 s，**超时也返回 1**，所以这是“有界等待”，不是硬阻塞。
- 0x7c7f44：`PostMsg(server, [ctx+352]-1)` —— 算法执行后回写 `+44 valid` 与 `+51 = ctx+352-1`（u8 截断）。
- 0x7c7f50：`CheckAivIsEnd(server,0)`；未结束 → 回 0x7c7ed0 继续循环。
- 0x7c7f90：`GetRspPolicy` 非零时，先 `WaitMainStreamFinish(ctx)`、`ClearWorkMsg(server)`，再 `PostMsg(server, [ctx+352])` —— 回写 `+51 = ctx+352`。
- 即 stage-wait 的 ack 链：**wait sendCnt≥turn → 执行 → PostMsg(turn-1) → 可选 PostMsg(turn)**；`ctx+352` 是轮次计数。

**地址等价（条件式）**：PostMsg 的目的 `x0=[this+24]`；Init 零分支 `server+0x18 = y+j+64T`；`GetRecvCnt` 读 `y+j+64T+51`。若同一 server 对象（TwoStageWait 用 RpcSrvLaunch 的 `server_A`（0x191fd10））且走零分支，则 **PostMsg 写的 +51 与 GetRecvCnt 读的 rcvCnt 是同一字节**；非零分支不写 server+0x18，不成立。PostMsg 先写 4 B valid 再 `strb` +51，随后 `dsb st`，字节序/屏障上 ack 在后。

**u8 截断约束**：`PostMsg strb w1` 把 `ctx+352`/`ctx+352-1` 截为 8 位；`CheckRcvWorkMsg` 用 `u8 opParam[50]` 与 **u32 `ctx+352`** 比较。若 `ctx+352 > 255`，比较永不满足 → 每轮 6 s 超时后继续（不是死锁，但轮次协议失配）。`ctx+352` 的写入/递增在已交付段中**只有 RpcSrvLaunch 的 `str wzr,[ctx+352]`（0x76187c）**，未见递增者；可能在 `TaskOrchestrator::ActiveRecordMain` 或未交付函数 → **来源/上界未证**。

**分派关系（静态已知，运行值未知）**：
- `RpcSrvLaunch` 选 TwoStageWait 的条件：`GetPreparePosition()!=1`、`GetTaskType()!=6`、`ctx+0xf31bc & ~2 == 1`。
- `RunRpcServerApi` 条件：`GetPreparePosition()==1`（且入口要求 `ctx+0xf31bc==2`）。
- 因此 **AIV ack 支路（TwoStageWait/PostMsg）与 API valid 支路（HcclMsg+48）是互斥分派**；Grouped 的 `GetPreparePosition/GetTaskType/ctx+0xf31bc` 运行值未证。
- `ForApi` 属 `RunAicpuKfcClearOpRes` 清理路径，server 0x22c7cf0；RpcSrvLaunch 正常路径 server 0x191fd10。静态全局地址不同**只证明对象地址不同，不区分其内部 workspace 值**；两者的 y/j/T 仍由 ctx[184]/[192]/[196] 与 Init 分支决定。

**OP6 更新（静态写者已找到且条件式同址）**：
- 已证：stage-wait 下 PostMsg 是 +51 的 1 B 写者（值 turn-1/turn），与 GetRecvCnt 条件式同字节；sendCnt +50 的等待者是 CheckRcvWorkMsg，但 +50 的写者仍未找到。
- 未证：Grouped 是否进入 TwoStageWait（取决于上述分派运行值）；若走 API 路径则 PostMsg 不参与，gate 是 HcclMsg+48，生产者仍在 host 侧。
- 下一最小证据（不新反汇编整段）：①`ctx+352` 的递增者/上界（TaskOrchestrator 或指定函数的小窗口）；②Grouped 的 `GetPreparePosition/GetTaskType/ctx+0xf31bc` 来源（Init 字段映射/host tiling）。仍无设备/SSH。

## 67. OP1/OP3/OP6 最小运行值字段表（为有限观测设计，不执行）

**原则**：只列能区分“API valid 路径 / TwoStageWait ack 路径 / MC2 / CommKfc”的最小字段；每项注明现有静态证据与缺失来源，避免为未知支路继续全量反汇编。

| # | 字段 | 来源/偏移（静态已知） | 用途（区分什么） | 现有证据 | 尚缺 | 最低成本取证（优先静态） |
| --- | --- | --- | --- | --- | --- | --- |
| F1 | `desc` qword | `*(u64*)KFCTask`；低 4 位、bits4-7、bit8、bits9-15 | SrvLaunch 顶层分支：CommKfcDispatcher / RpcSrvLaunch / MC2 | SrvLaunch 0x763108 分支谓词 | Grouped 实际 desc | host `aclnnGroupedMatMulAllReduce` 实现/结构体填充（静态） |
| F2 | `sub` u32 | `task[(desc>>9)&0x7f]` 的首 u32 | sub=0/1/2/>2 的子分支 | SrvLaunch 0x763285c 起 | Grouped 实际 sub | 同 F1 |
| F3 | `P[0]` u32 | `P=*(void**)(KFCTask+40)`；`P[0]` | Init 零/非零分支（决定 server+0x18 是否建立） | Init 0x7d671c-0x7d6738 | P 的结构体定义与 Grouped 值 | host tiling/KFCTask 填充（静态） |
| F4 | `j` u32 | `*(u32*)(KFCTask+96)` → `ctx+192` | param 区偏移；别名/同址判定 | 写者 0x76182c | Grouped 值/范围 | 同 F3；可用符号 `j` |
| F5 | `T` u16 | `*(u16*)(KFCTask+100)` → `ctx+196` | recv 槽索引；`+64*T` 项 | 写者 0x761834 | Grouped 值（是否 rank 数） | 同 F3 |
| F6 | `y` u64 | `ctx+184`（AicpuRpcResInit 0x7b9f50） | workspace 基址；param/recv 的 base | 写者 0x7b9f50 | Grouped 运行值 | 运行期只读快照（未来授权）或 host 分配日志 |
| F7 | `ctx_R+0x160` turn u32 | `ctx_R = AicpuGetComContext()`（0x7615ac/5b0，x28）；`0x76187c str wzr,[x28,#352]` → `*(u32*)(ctx_R+0x160)=0`；TwoStageWait 以 `x20=ctx_R` 读 `[x20,#352]`（callsite 0x76198c/0x761994 证明同一对象） | TwoStageWait 的期望/ack 轮次；u8 截断上界 | 清零与三次读取（0x7c7f10/38/88） | **递增者/上界** | `TaskOrchestrator::ActiveRecordMain` 等小窗口（已向 31 请求） |
| F8 | `GetPreparePosition` | server 字段（getter 0x7d7b60，0xc） | API 路径 vs TwoStageWait/Unfold | getter 地址；Init 写入字段未映射 | server 偏移与 Grouped 值 | getter 小窗口 + Init 字段映射（已请求） |
| F9 | `GetTaskType` | server 字段（getter 0x7d7b30，0xc） | `==6` 走 UnfoldStageWait；否则 TwoStageWait/OneStageWait | getter 地址 | 同上 | 同上 |
| F10 | `*(u32*)(ctx_R+0xf31bc)` mode | `x26=ctx_R+0xf0000`（0x761654）；`0x761848/0x761978 ldr w0,[x26,#12732]` → `ctx_R+0xf31bc`；`==2` 允许 RunRpcServerApi，`&~2==1` 走 TwoStageWait | 决定 API 是否可达 | 两条 `ldr [x26,#12732]`；`ctx_R` 同 F7 | **写入者/取值** | 定向 grep/小窗口（已请求） |
| F11 | `server_A` vs `server_B` | `server_A = ADRP(0x191f000)+0xd00+0x10`（Init 前 x23 未变；0x7620b4 的 `add x23,x23,#0xd00` 属 Init 后日志分支、无回边）；`server_B = ADRP(0x22c7000)+0xb50+0x1a0`（ForApi/ClearOpRes） | 正常 launch vs 清理路径 | adrp/add 指令与分支边 | 实际进入哪个 | 分派谓词（F1/F2/F8-F10）静态推导 |
| F12 | `HcclMsg+48` valid 生产者 | gate 读 slot+48；PostMsg 写 AivParam+44，非同一结构 | API 路径是否有消息可读 | gate/consume、PostMsg/GenMsg | host 侧写者 | host 库指令立即数扫描（非数据字） |

**观测设计建议（未来、需单独授权）**：若 F1–F10 的静态推导仍不能定支路，最小运行观测只需两组只读值：①launch 时 `desc/sub/P[0]/j/T`（host 或 KFC task 侧）；②server 的 `GetPreparePosition/GetTaskType` 与 `ctx+0xf31bc`、`ctx+352`。两组即可把四条路径区分开；不需要读 +50/+51 或做全量 workspace dump。
**当前不执行**：无 SSH/设备/新 GO；root cause 仍 unresolved。

## 68. ctx+352 / ctx+0xf31bc / server_A 的原始指令与地址式复核（abi9 对齐）

**目的**：回应 31 的“未以字面 store 复现”和 server 地址歧义；所有表达式给出函数内寄存器来源与 callsite，不跨函数拼同偏移。

### 68.1 `ctx_R + 0x160`（offset 352）

- `ctx_R`：`RunAicpuRpcSrvLaunch` 0x7615ac `bl AicpuGetComContext`；0x7615b0 `mov x28,x0`。本函数内 ctx 即 x28。
- 原始写：0x76187c `b901639f str wzr,[x28,#352]` → `*(u32*)(ctx_R + 0x160) = 0`；完整地址 `ctx_R+352`，宽度 4 B，值 0。前驱为 0x761860-0x761878 的 `[x24]`/`[x25]` 字段处理，不改变 x28。
- 第二处清零（31 更正）：0x7608c8 `str wzr,[x19,#352]`（`RunAicpuKfcClearOpRes`），x19 为该函数的 ctx 参数；同样是 4 B 清零。两处均为清零，**递增者仍未定位**。
- 原始读：`RunRpcServerTwoStageWait` 0x7c7f10 `ldr w3,[x20,#352]`、0x7c7f38/0x7c7f88 `ldr w1,[x20,#352]`；`x20` 是该函数的 ctx 参数。
- **同一性证据**：0x76198c `mov x0,x28`；0x761994 `bl RunRpcServerTwoStageWait`。即 TwoStageWait 的 ctx 参数就是 RpcSrvLaunch 的 ctx_R；跨函数同偏移成立有 callsite 依据，不是按偏移拼接。
- 31 的字面检索未命中是**编码/范围问题**：该 store 用 `wzr`（非 wN）且偏移写作 `#352`，不是 `#0xf3,lsl#12` 形式；应能在 0x76187c 复现。**不据此推断另一 DSO**。
- 仍缺：`ctx_R+352` 的递增者/上界；已交付段只有清零与三次读。

### 68.2 `ctx_R + 0xf31bc`（mode）

- 基址：0x761654 `add x26,x28,#0xf0,lsl#12` → `x26 = ctx_R + 0xf0000`。
- 原始读：0x761848 `ldr w0,[x26,#12732]` → `*(u32*)(ctx_R+0xf31bc)`；0x761850 `cmp w0,#0x2`（`!=2` 时取 task 字段覆盖）。
- 原始读 2：0x761978 `ldr w0,[x26,#12732]`；0x761984 `and w0,w0,#0xfffffffd`；0x761988 `cmp w0,#0x1`；0x761990 `b.ne` → OneStageWait，否则 0x761994 TwoStageWait。
- 关系：0x761820 `str w0,[x26,#12752]` 写 `ctx_R+0xf31d0`；0x761880 `str wzr,[x26,#12756]` 清零 `ctx_R+0xf31d4`。这些是同一 `ctx_R` 的不同字段，不与其他函数的同偏移混用。
- **f31bc 的写者（abi9 更正）**：
  - 0x2ea7b8 `str w8,[x1,#12732]`（w8=9）；x1 是静态全局 `0x22ce888+0xf0000`，**不是 ctx_R** → 不能直接作为 ctx_R 的写者。
  - 0x7edfc0 `str w5,[x7,#12732]` 位于 `AicpuHcclProcess::CopyCtxInfo`（起始 0x7edc20），`w5=[x23+12732]`：**从另一个 ctx 复制到目标 ctx**，不是计算/递增。同一函数还复制 +12728/+12744/+12752/+12756 等 KFC 块字段。
  - 因此 f31bc 的来源是 **CopyCtxInfo 的源 ctx 及其取值**，当前未定位；无 +1/递增站点。
- 31 的 `#0xf3,lsl#12` 站点写 `ctx+0xf3270` 等，是 `ctx_R+0xf3000` 基址的另一组；`0xf31bc` 用的是 `ctx_R+0xf0000+12732` 编码，故定向 grep 需按 `#12732` 或 `x26=...+0xf0000` 查。
- 仍缺：`ctx_R+0xf31bc` 的写入者/取值。

### 68.3 `server_A` 算术

- 0x7618dc `adrp x23,0x191f000`（x23=page base）；Init 前 0x761918 `add x0,x23,#0xd00`、0x761928 `add x0,x0,#0x10` → `server_A = 0x191fd10`。
- 0x7620b4 `add x23,x23,#0xd00` 位于 Init 之后的错误/日志分支（0x762008→0x76209c→0x7620b4→`b 0x7614a4`），**无回到 0x76192c 的边**；若它先于 Init，算术会是 0x1920a10，但控制流不支持。
- 为避免绝对地址传播，报告统一用 `server_A` 符号，并在任何新引用处附“Init 前 x23 未变、0x7620b4 无回边”的前提。

### 68.4 对字段表的影响

- F7 改为 `ctx_R+0x160` 表达式，附 callsite 同一性；F10 改为 `x26=ctx_R+0xf0000`、`[x26,#12732]`；F11 用 `server_A` 符号与分支边说明。
- 31 的字面未命中只记检索范围/编码差异；不据此判断其他 DSO。仍无设备/SSH。

## 69. 偏移算术审计：12732=0x31BC，撤回 f31dc 字段名

- **十进制/十六进制核对**：12732 = **0x31BC**；12764 = 0x31DC；434 = 0x1B2；12904 = 0x3268。
- 因此 `[x26,#12732]` 在 `x26=ctx_R+0xf0000` 时是 **`ctx_R+0xf31bc`**，不是 `ctx_R+0xf31dc`；要得到 f31dc 还需额外的 +0x20 基址偏移，而当前没有任何指令把 x26 设为 ctx_R+0xf0020。
- 31 的定向检索也显示：交付段内**没有 `#476`/`#0x1dc` 的访问**；`ctx+0xf31dc` 是此前的转写错误，**撤回该字段名**，统一为 `x26+12732 (=ctx_R+0xf31bc)`。
- 同类修正：`ctx+0xf31d2` → **`ctx+0xf31b2`**（`[x25,#434]`，x25=ctx_R+0xf3000）；epoch `[x19,#12904]` → **`ctx+0xf3268`**，不是 f31f8。`ctx+0xf31d0/f31d4/f31d8/f31d9/f31f0` 等按 12752/12756/472/473/12784 核对无误。
- **CopyCtxInfo 的 base 各自独立**：源 `[x23+0x31bc]`、目标 `[x7+0x31bc]`；x23/x7 是两个 ctx 指针，不能把一个 ctx 的 +0x20 关系套到另一个。
- **静态 global 写 9**：0x2ea7b8 `str w8,[x1,#12732]`，`x1 = ADRP(0x22ce000)+0x888+0xf0000`；地址式为 `x1+0x31bc`（表达式，不传播绝对地址）。它与 ctx_R 是不同对象，**不能把 9 直接赋给 ctx_R+0xf31bc**；只有 CopyCtxInfo 的源/目标与 AicpuGetComContext 返回对象身份连通后，才能推导运行取值。
- 35 有独立地址审计；本节与其结果合并后再作为最终字段名。无新取数/设备动作。

## 70. F1–F12 可执行取证表（现有载体/读取方式/已结束 cell 可得性）

**总原则**：AICPU 设备侧对象（ctx/task/workspace/HcclMsg）在设备内存；host 进程只能看到 plan/argv/env 与自身日志。已结束 cell 没有这些字段的值日志时，明确标“缺测”，**不虚构 read-only 内存读取命令**。

| 字段 | 载体/可见性 | 读取位置/命令（现有） | 已结束 cell 可得？ | 只读静态能证明 | 缺测项 |
| --- | --- | --- | --- | --- | --- |
| F1 `desc` | host 进程（KFCTask 构造）/设备 task | plan JSON 无该字段；device 日志仅 `get api RunAicpuKfcSrvLaunch` | **否** | SrvLaunch 分支谓词与位域位置 | 实际 `desc` 值（需 host 侧 launch 日志或反汇编 host 填充） |
| F2 `sub` | 设备 task 内存 | 无 | **否** | `task[(desc>>9)&0x7f]` 首 u32 的 sub 分支 | 实际 sub |
| F3 `P[0]` | 设备 task+40 | 无 | **否** | Init 零/非零分支判据 | `P` 类型与值 |
| F4 `j` | 设备 KFCTask+96 → ctx+192 | 无；device Dlog 无该字段 | **否** | 写者 0x76182c、`param=y+j` 算术 | 实际 j/范围 |
| F5 `T` | 设备 KFCTask+100 → ctx+196 | 无 | **否** | 写者 0x761834、`+64*T` 算术 | 实际 T |
| F6 `y` | 设备 ctx+184（workspace 指针） | probe trace 记录 workspace_bytes，不记录指针值 | **部分/否** | 写者 0x7b9f50、base 算术 | 实际指针值（无现有日志） |
| F7 `ctx_R+0x160` turn | 设备 ctx 内存 | 无；现有 device 日志无 turn | **否** | 0x7608c8/0x76187c 清零、TwoStageWait 三次读 | 递增者/上界/运行值 |
| F8 `GetPreparePosition` | 设备 server 内存（[server+816]+0） | 无 | **否** | getter 0x7d7b60 偏移；Init 将 `KFCTask+40` 存 server+816 | 实际字段值与映射 |
| F9 `GetTaskType` | 同上（+116） | 无 | **否** | getter 0x7d7b30 偏移 | 实际值 |
| F10 `x26+12732` mode | 设备 ctx 内存 | 无 | **否** | 两条 ldr、cmp/and 分支；写者为 CopyCtxInfo/静态 global | 实际值/源 ctx |
| F11 `server_A/B` | 设备静态全局 | 无 | **否** | adrp/add 地址式与分支前提 | 实际进入哪个对象 |
| F12 `HcclMsg+48` | 设备共享 HcclMsg 区 | 无；现有 device 日志仅 ATRACE 对象创建（`KFCtaskAndTilingTraceData`/`AivAicpuOpParam` create），无字段值 | **否** | gate/consume、PostMsg/GenMsg 写的是 AivParam+44 | 生产者与实际 valid 值 |

**设备 vs host 分离**：
- Host 侧现有可见：plan JSON（rows/k/n/flags/seed/identity）、argv/env、service.log；**不含 KFC `desc/sub/P[0]/j/T`**。
- Device 侧现有可见：device 日志（ATRACE 对象创建、Dlog 配置/错误行）、plog（stream/task/timeout）、probe trace（pad/workspace/sync/gather 时间）；**不含上表字段值**。
- 因此 F1–F5、F7–F12 在已结束 cell 均为**缺测**；F6 仅有 `workspace_bytes`，无指针值。

**最小有限观测设计（未来、需单独授权）**：
1. Host 侧一处 launch 日志：打印 `desc` 的 nibble/sub/idx、`KFCTask+40` 首 u32、`+96/+100`（即 j/T）；只读、无设备格变更。
2. Device 侧一处只读快照或 verbose Dlog：`server+816` 的 u32[0]/u8[116]、`ctx_R+0xf31bc`、`ctx_R+0x160`；这是区分四条路径的最少字段。
- 两组都到位即可判定 API/TwoStageWait/MC2/CommKfc；**当前不执行**，不新增 GO/SSH/设备动作。

## 71. 与 35 的 abi8/abi9 独立交叉核收口（abb446f / 24992616）

**一致项（指令级）**：
- PostMsg 0x7d6800；`0x7d6828 ldr x0,[x0,#24]` → `dest=*(this+0x18)`；仅 `str w4,[x0,#44]`（4 B）与 `strb w1,[x0,#51]`（1 B）；w1 未被改写；调用点 `[ctx+352]-1`/`[ctx+352]`；`strb` 截低 8 位。
- 屏障：`+44`→`+51` 之间无屏障；唯一 `dsb st` 在 0x7d6884（PrintMsg/string dispose 之后）；无 `stlr`。
- 条件同字节 P1–P4（同 server、Init4 零分支、无后续 server+0x18 写、同 ctx）；非零分支不成立。gate `slot+48` 与 valid `+44` 同字需 `j+64T=64*turn+4`（slot=dest−4），未证。
- CheckRcvWorkMsg：`0x7d73b8 ldrb [x19,#50]`（u8 零扩展）vs `0x7d73bc ldr w1,[sp,#36]`（u32，w3=[ctx+352]）；`ctx+352>255` 永不满足；6 s 预算（x24=`0x1_65a0_bc00` ns）后经 0x7d7408/0x7d7414 到 0x7d7450 同一续点，**非死锁**。

**更正项（已并入 §68/§69）**：
- `ctx+0xf31dc` 实为 **`ctx_R+0xF31BC`**：0x761654 `add x26,x28,#0xf0,lsl#12`（x28 全函数只写一次）；0x761848/0x761978 `ldr [x26,#12732]`；12732=0x31BC。0x31DC=12764 在 abi3–abi9 全存档无指令（scope negative）；要 f31dc 需额外 +0x20 基址证据。
- ReadAddrMsg 同基址模式但不同字段：0x7d8bec `add x27,x23,#0xf0,lsl#12`；0x7d8c00 `ldr x0,[x27,#12784]` = `ctx+0xF31F0`（8 B，与 1 比较）——不要把 12784 与 12732 混为同一字段。
- 两个 12732 写者：静态初值 0x2ea7b8 `str w8(=9)`（不同对象）；0x7edfc0 CopyCtxInfo 从另一 ctx 拷贝；无 +1。
- server 算术：0x7618dc adrp 0x191f000；0x761918 +0xd00；0x761928 +0x10；**server_A=0x191fd10**；0x191fd00 只是外层结构基址（x2/x25）。35 对 `disasm-RpcSrvLaunch.txt`（SHA d92fee9d）做 CFG：入口→0x761904 必经 0x7618dc 且不含 0x7620b4；0x7620b4→0x7620ec b 0x7614a4→0x76153c ret，均到不了 0x761904/0x76192c，**0x1920a10 不可达**。
- `0x76187c str wzr,[x28,#352]` 确为 ctx+352 清零；递增者仍未定位。

**合并后的 abi10 最小请求（35 的四项，交 31 一次取；无全库扫）**：
1. `RunRpcServerTwoStageWait` 序言是否 `x19 = arg2`（0x761994 已证 x1=`server_A`；PostMsg 的 this=x19 还需序言）。
2. TwoStageWait 的调用者及传入的 server 对象（预期 0x761994 传 `server_A`，需原文确认）。
3. 除 Init4 零分支外，是否有其它指令写 `server+0x18`（影响 P3）。
4. `AivAicpuOpParam::PrintMsg` 体是否写 `dest`（影响 PostMsg 后 dest 内容）。
- 说明：不做全库 store 扫描/跨 DSO 搜索；唯一写者结论保持撤回；不推断协议冲突；无设备动作。

## 72. abi-preflight10（2de8278/dfcbaa1a）CopyCtxInfo 端点消费

- **偏移更正被接受**：12732=0x31BC，统一 `x26+12732` / `ctx_R+0xf31bc`；f31dc 撤回。
- `CopyCtxInfo`：nm `0x7edc20`/`0xc90`；直接 `bl` 调用者**仅 1 处** 0x7ba9f0，位于 `AicpuKfcProcess::AicpuRpcResInit(HcclCombinOpParam*)+0xc28`；调用点 `x0=x20`（x20 在 AicpuRpcResInit 更早处构造，未在本窗口解析）。
- 内部端点：源侧 `x23=arg+0xf0000`；目标 `x7` 由 `x22`（0x7ede58 madd、0x7edeb0 add 0xf0000）派生；`0x7edfb4 ldr w5,[x23,#12732]` → `0x7edfc0 str w5,[x7,#12732]`。目标是与静态全局表相关的 ctx，**不是 ctx_R**（ctx_R 是 AicpuGetComContext 返回）。
- 静态初值 `0x2ea7b8`：位于未导出静态初始化区域（objdump `DlopenTraceSov+0x24fc0`），调用 `__cxa_atexit`，写静态全局 `0x22ce888+0xf0000` 的 `+0x31bc=9`；`bl` 到该区计数=0、init_array 字节模式命中=0 → **不是 ctx_R 的直接写者**。
- 结论：f31bc 的**直接**写者仍是 CopyCtxInfo 的源 ctx 与静态 global 9；ctx_R 的取值需“源 ctx 身份 + 是否/何时 CopyCtxInfo 到 per-op ctx”两者，当前均未证。仅保留这两个端点，不再扩展 store/helper。

## 73. 有限观测方案（可审核版；不执行设备）

### 73.1 现有可证证据（本地）

- 已结束 cell 的 plog 明确显示日志开关**未设置**：`[TsdClient] get ASCEND_GLOBAL_LOG_LEVEL [] ASCEND_GLOBAL_EVENT_ENABLE [] ASCEND_MODULE_LOG_LEVEL []`（a15-small-v3 plog L29 等）；`HCCL_DEBUG_CONFIG is not set, debugConfig set by default to 0x0`（同 plog L150）。这证明变量名存在且当时为空/默认，**不证明各取值的语义**。
- 设备日志现有内容：`ae_so_manager ... Get api RunAicpuKfcSrvLaunch ... success`、`ATRACE ... create object KFCtaskAndTilingTraceData/AivAicpuOpParam ...`（h11/a12/v4r1 有，a14/a15 无）、Dlog 配置/错误行。**没有任何分支选择、desc、j/T、server 字段、ctx 字段的值**；`RunAicpuRpcSrvLaunch` 字样只出现在 so 管理器取 API 行，不是分支标记。
- Host 可见：plan JSON（rows/k/n/flags/seed/identity）、probe result（arm/flags/api_returned/sync_returned/workspace_bytes/run_id/libraries）、trace（pad/workspace/sync/gather 时间）；**无 KFC `desc/sub/P[0]/j/T`、无 server/ctx 字段**。

### 73.2 标记判别表（类型/前提/窗口可达/支持与排除）

| 标记 | 类型 | 调用者/前提 | 180/240 s 窗内可达？ | 出现支持 | 缺失能否排除 | 证据 |
| --- | --- | --- | --- | --- | --- | --- |
| `ReadAddrMsg` 超时日志（line 360）+ `PrintAllHcclMsgArea`+`PrintTimeOutSqInfo` | **条件失败/超时**，非分支入口 | `RunRpcServerApi→ReadAddrMsg`；无有效 HcclMsg，`elapsed > w24*20 s` | **是**：阈值 20/40/80/160 s；180 s 窗含 20/40/80/160，240 s 同（320 s 未到）；1280 s 条件返回不在窗内 | API 路径被进入**且**在该时刻仍无有效消息（或 turn 不匹配） | **否**：消息到达后算法/`AddTask` 卡住不触发；日志被抑制也不触发；缺席不能排除 API 路径 | §62.2、0x7d8d24-0x7d8d9c |
| `CheckRcvWorkMsg` 超时日志（line 285） | **条件失败/超时**，非分支入口 | `RunRpcServerTwoStageWait` 0x7c7f20；`sendCnt(+50) < ctx+352` 持续 10001 次/6 s | **是**：每 6 s 一次；条件满足则无日志 | TwoStageWait 被进入**且**等待条件未满足（`ctx+352>255` 时必现） | **否**：sendCnt 追平即无日志；缺席不能排除 TwoStageWait | §66.5、0x7d7408-0x7d7450 |
| `PostMsg` 内 `AivAicpuOpParam::PrintMsg` | **分支内必经**（TwoStage 第一次 ack；第二次受 `GetRspPolicy` 条件） | TwoStageWait 0x7c7f44/0x7c7f90；第一次在 `ActiveRecordMain` 成功后无条件 | **通常可达**：CheckRcvWorkMsg 最长 6 s 后继续，随后执行/PostMsg；若算法本身阻塞则不可达 | TwoStageWait 被进入且执行到 ack | **否**：PrintMsg 是否有 Dlog 输出/级别未证；算法阻塞也会缺席 | §66.1、0x7d6800/0x7c24e0 |
| RpcSrvLaunch 的 `PrintTilingData`/`PrintMC2AicpuContext` | **共享 helper/前置必经**（分支前） | RpcSrvLaunch 0x7618a4/0x7618b0，所有子分支共用 | 是 | 说明进入 RpcSrvLaunch（相对 MC2/CommKfc），**不区分子分支** | 否（共享标记） | 0x7618a4/0x7618b0 |
| `ATRACE ... create object KFCtaskAndTilingTraceData/AivAicpuOpParam` | **共享 KFC 路径标记**（对象创建） | KFC task/param 区被使用；Grouped/A12/H11/H4K 有，A14/A15 无 | 是（现有 cell 已见） | KFC 路径（vs MRS） | 否：ATRACE 配置/默认级别与对象创建时机未证；不能分 API/TwoStage/MC2 | 现有 device 日志 |
| `ae_so_manager ... Get api RunAicpuKfcSrvLaunch ... success` | **共享入口**（KFC kernel 取 API） | 任意 KFC 启动 | 是（现有 cell 已见） | KFC kernel 启动，不区分子分支 | 否 | 现有 device 日志 |
| OneStage/Unfold 入口/错误 Dlog | 本轮 10 段内**无唯一归属串**（§78） | 进入对应分支/错误 | 待定 | 无现有标记；§77 #7 的 OneStage 归因已撤回 | 不适用 | abi12 rodata |
| MC2TraceUtils/CommKfc 标记 | 待 35 格式串 | MC2/CommKfc 提交 | 待定 | 指向 MC2/CommKfc | 待定 | 35 rodata |

**缺失归因限缩**：历史 cell 的 plog 只证明 `ASCEND_GLOBAL_LOG_LEVEL`/`ASCEND_GLOBAL_EVENT_ENABLE`/`ASCEND_MODULE_LOG_LEVEL` 为空、`HCCL_DEBUG_CONFIG=0x0`；这**只能证明未采到/按默认配置运行**，默认级别值与标记自身级别尚缺（31 文档检索归档另交）。因此**“日志级别未开”不作为已证事实**；标记缺失不能单独排除任何路径，除非前提（超时/失败）与采集级别都被证明满足。

**已有最小替代**：`KFCtaskAndTilingTraceData`/`AivAicpuOpParam` ATRACE 对象有/无只区分 KFC vs MRS，不能分 API/TwoStage/MC2。

### 73.3 精确开关/命令（同版文档证据，cf9f1d6）

- **ASCEND_GLOBAL_LOG_LEVEL**：安装文本给出 `0:DEBUG、1:INFO、2:WARNING、3:ERROR（默认）、4:NULL（不输出）`（`python/site-packages/msopst/scripts/msopst.ini:16-22`；`asys/config/config_parser.py:44-49` 同映射，asys 默认 INFO）。这是工具自身默认的映射，**不能直接当作 CANN 核心/AICPU 的默认级别或传播效果**。
- **ASCEND_GLOBAL_EVENT_ENABLE**：`asys/config/config_parser.py:50-52` 为 `{"FALSE":"0","TRUE":"1"}`，asys 默认 TRUE；`asc_op_compile_base/common/utils/log_utils.py:151-155` 仅 `int(值)==1` 时做 detail/event 打印。同为工具侧证据。
- **ASCEND_PROCESS_LOG_PATH**：路径型；`asys/drv/env_var.py:26,34-43` 要求存在且为目录并取 abspath；`asys/common/task_common.py:128-137` 作为 CANN/plog 日志目录，缺省回退 `ASCEND_WORK_PATH/<log_type>`。
- **ASCEND_MODULE_LOG_LEVEL**：安装树仅 `pkg_inc/mmpa/sub_inc/mmpa_env_define.h:94` 枚举 ID 6005；**无取值表/模块名格式**，不猜。
- **HCCL_DEBUG_CONFIG**：安装树 3 处（`mmpa_env_define.h:142` 枚举 ID 11025；`aicpu_operator_pub.h:660` 与 `moe_distribute_base.h:240` 仅是 `u64 debugConfig=0` 字段注释）；**无 bit 语义**。
- **边界**：检索为文本级（`grep -rIn -I`，跳过二进制），未检索 devlib 内字符串；`/usr/local/Ascend/driver/{include,tools}` 与 `nnal` 0 命中；安装树无 doc 目录/相关 PDF/MD。故“无”只限本次范围。
- **可用候选命令（未执行）**：`ASCEND_GLOBAL_LOG_LEVEL=0`（DEBUG）+ `ASCEND_PROCESS_LOG_PATH=<已存在目录>`；可选 `ASCEND_GLOBAL_EVENT_ENABLE=1` 仅用于 event/ATRACE。**观测有效性条件**：必须在目标 AICPU 日志中看到已知必经标记（如 Init4 line 27 或 RpcSrvLaunch 的共享标记），否则“配置生效”未证，缺日志仍按缺测。
- 设备侧 AICPU ctx/task/workspace/HcclMsg **无受支持的只读内存接口**证据；不提供 `devmem`/自定义读内存命令。字段值 F1–F12 继续标缺测。

### 73.4 最小有限观测设计（条件式，当前不执行）

1. **静态可完成**：ATRACE 对象集合确认 KFC vs MRS（已完成）；F1/F8-F10 分派谓词做假设表。
2. **若未来授权一次带日志的 cell（非本任务）**：只开现有全局/模块日志变量（取值以 31 同版文档归档为准，当前不猜），按 73.2 的**正向标记**判定：
   - API：20/40/80/160 s 超时日志（180/240 s 窗内可触发）；出现即支持 API 路径且当时无有效消息；
   - TwoStage：`CheckRcvWorkMsg` 6 s 超时日志（等待未满足时）；或 `PostMsg` 的 `PrintMsg`（若格式串/级别证明有输出）；
   - 若两者均无正向标记，**不得排除**任一（前提未满足或未采集）；再考虑 OneStage/Unfold 的格式串。
3. **若字段值必须直接读**：仅能通过新增 host 侧 launch 日志或设备侧代码埋点（均属代码变更，需单独授权），**不是现有接口**；不把“快照”写成可执行命令。设备内存无受支持只读接口的部分继续缺测。
- 当前无新 GO/SSH/设备动作；root cause 仍 unresolved。

## 74. abi-preflight11（88f9151，SUMS 50f82839）：P1/P4 补齐与 P3 范围

- **TwoStageWait** nm `0x7c7dac`/`0x1028`；序言 `0x7c7dc8 mov x19,x1`、`0x7c7dd0 mov x20,x0` → PostMsg 的 this=x19=x1=`server_A`，ctx=x20=x0=`ctx_R`。与 RpcSrvLaunch 调用点（0x76197c/0x761980 置 x1=server_A；0x76198c x0=x28；0x761994 bl）一致，**P1/P4 证毕**。
- **PrintMsg** nm `0x7c24e0`/`0x654`：完整体扫描未发现对入参对象 +24/+44/+51 的写，唯一 `strh w3,[sp,#24]` 是栈槽 → **PrintMsg 不改写 PostMsg dest，P3 不受影响**。
- **server+0x18 写者**：方法区 0x7d66f0–0x7da9bc 定向 `str ..,[x?,#24]` 共 15 命中：
  - Init4 零分支 `0x7d6758 stp x0,x3,[x19,#16]` 写 server+0x10/+0x18（x19=server），是本区唯一 server+0x18 写；
  - `0x7d77f4 str x1,[x19,#24]` 属 `HcclMsg2AicAicpuOpParam`（x19=AivAicpuOpParam 输出参数）；
  - `0x7d7ef0 str x0,[x19,#24]` 属 GenMsgByTaskParam 区（x19=AivAicpuOpParam 入参）；
  - 其余 12 条为 `[sp,#24]` 栈槽。
  - 结论：**在已审函数/指令形式范围内**除 Init4 外无 server+0x18 写者；param 对象的 +0x18 写属另一对象，不并入 server 结论。**P3 不升级为全库排除**，不再追加穷尽扫描/深层 helper。
- 因此 §66.1 的条件式同字节更新为：P1/P4 已证，P3 限已审范围，**P2（KFCTask+40 首 u32==0）仍未证**；gate slot+48 与 valid+44 的 `j+64T=64*turn+4` 亦未证。

## 75. 观测方案优先级的当前状态

- §73 的现有标记仍有效；当前优先“可用观测接口/开关/已有分支标记”，**不再为 P3 或 Write*/helper 追加穷尽反汇编**。
- 35 已定位地址、正优先直接发**最小 rodata 地址清单**（不等文档归档）；收到后 33 立即合并为一次 abi12 交 31，字符串取证与报告并行。
- 31 的同版版本/开关文档检索归档另交，不为此重复请求；在文档到位前不写精确 env/级别/bit 值。
- 历史 cell 运行值缺测口径保持（F1–F12 无现成读取方式）；设备 AICPU 内存无可证只读接口；无新设备 GO/SSH。
- 若 rodata 显示某路径无唯一标记，则明确记“无现有标记”，不把观测愿望写成命令；最小计划只选**已证可触发**的标记。

## 76. abi12 rodata 清单（35 定位，待 31 取文本）

`libccl_kernel.so` 2281deca 固定；每段按 NUL 终止取，跨边界顺延到首个 NUL；不反汇编整函数、不全库扫描。

| # | 地址范围 | 长度 | 调用 site | 路径 | guard（HcclCheckLogLevel） | 备注 |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 0x1613b80–0x1613d40 | 0x1C0 | 0x7d6fec / 0x7d7074 / 0x7d70b8 / 0x7d71bc | ReadApiValidMsg gate 不匹配（API） | (2,3)@0x7d6f8c；(2,0x01000003)@0x7d6ffc + IsErrorToWarn | fmt 0x1613bd0/0x1613c00/0x1613ce0；file 0x1613cc8；x6=0x1613bc0；line 226/241 |
| 2 | 0x1613f00–0x1613f60 | 0x60 | 0x7d67cc | Init4（RpcSrvLaunch） | (0,3)@0x7d6728 → 0x7d67a0 | fmt 0x1613f20；file 0x1613cc8；x6=y；line 27 |
| 3 | 0x1614040–0x16140a0 | 0x60 | 0x7d744c | CheckRcvWorkMsg 6 s 超时（TwoStage 条件） | (3,3)@0x7d7410 + IsErrorToWarn | fmt 0x1614060；file 0x1613cc8；line 285 |
| 4 | 0x16143e0–0x1614440 | 0x60 | 0x7d8d20 | ReadAddrMsg 等待超时（API，20 s 首触发） | (2,0x01000003)@0x7d8ce4 | fmt 0x1614408；file 0x1613cc8；x6=turn、x7=20；line 360 |
| 5 | 0x14eb190–0x14eb200 | 0x70 | 0x7615a4 | RpcSrvLaunch arg0==NULL | (3,3)@0x761568 | fmt 0x14eb1b0；file 0x14edee8；line 433 |
| 6 | 0x14ede80–0x14edf20 | 0xA0 | 共享 file 串（0x7615a4/0x7620e8/0x762280） | 多路径 | — | file 0x14edee8 |
| 7 | 0x14ee630–0x14ee690 | 0x60 | 0x762280 | OneStage 返回 0x16(22) | (1,0x01000003)@0x76224c | fmt 0x14ee650；file 0x14edee8；line 98 |
| 8 | 0x14ee700–0x14ee790 | 0x90 | 0x7620e8 | post-Init 错误/告警路径 | gate@0x7620a4（module 未全解析） | fmt 0x14ee728；file 0x14edee8；x6=[0x191fd04]；line 544 |
| 9 | 0x14f7160–0x14f71f0 | 0x90 | 0x7b3a24 | ForApi 每轮 | (1,3)@0x7b39f0 | file 0x14f71a8 |
| 10 | 0x14f8270–0x14f82e0 | 0x70 | 0x7b3a24 | ForApi 每轮 | (1,3)@0x7b39f0 | fmt 0x14f8290；x6=[ctx+12756]；line 1715 |

**用途**：确认各标记的格式串/级别/是否含字段，决定 73.2 哪些可作正向判别标记。字符串本身不改静态结论。

## 77. 具体有限观测计划（候选命令+有效性条件；不执行）

**候选配置（单次未来 cell，需单独授权）**：
```
ASCEND_GLOBAL_LOG_LEVEL=0            # 0=DEBUG（同版 msopst/asys 文本映射）
ASCEND_PROCESS_LOG_PATH=<已存在目录>  # 目录，abspath
# 可选：ASCEND_GLOBAL_EVENT_ENABLE=1  # 仅当需要 event/ATRACE 细节
```
不设 `ASCEND_MODULE_LOG_LEVEL`/`HCCL_DEBUG_CONFIG`（本范围无语义）。**观测有效性条件**：设备/plog 必须出现至少一个已知必经标记（优先 #2 Init4 line 27；若未进 RpcSrvLaunch 则看 #9/#10 ForApi），否则“日志配置在目标 AICPU 生效”未证，缺日志仍按缺测。

**判别逻辑（正向标记；缺席不排除）**：
| 观测结果 | 支持 | 仍需注意 |
| --- | --- | --- |
| #1 ReadApiValidMsg gate failure（line 226/241） | API 路径，且 gate `+48` 不匹配（可立即出现，不必等 20 s） | 若 gate 匹配则不出现；缺席不排除 API |
| #4 ReadAddrMsg timeout（line 360，20/40/80/160 s） | API 路径且当时无有效消息 | 消息到达后算法卡住不触发 |
| #3 CheckRcvWorkMsg timeout（line 285，每 6 s） | TwoStageWait 且 `sendCnt<ctx+352` | sendCnt 追平即不出现；缺席不排除 |
| #2 Init4 line 27 | 进入 RpcSrvLaunch（共享，不分子分支）；可作“日志生效”证据 | 需 #1/#4/#3 才能分 API/TwoStage |
| #9/#10 ForApi 每轮 | ForApi/ClearOpRes 路径 | 与正常 launch 区分 |
| #7 OneStage 0x16 / #8 post-Init error | OneStage 返回/错误路径 | 需格式串确认语义 |
| ATRACE KFC 对象 | KFC vs MRS | 不分子分支 |

**若 #1/#4/#3 均未出现**：不能排除 API/TwoStage（可能 gate 匹配、消息到达、等待满足或采集未生效）；只能记录“无正向标记”，下一步才是 host launch 日志或代码埋点（需单独授权）。**当前不执行**；无设备 GO。

## 78. abi12 rodata 文本结果（33a2226/dcc6a107）：标记判别更新

**来源**：`libccl_kernel.so` 2281deca；31 按 .rodata VMA/文件偏移直读 10 段（332 行 dump，未截断）；段内 NUL 串完整。

### 78.1 实际字符串（与 §76 地址对应）

| 段 | 地址 | 原文 | 归属/站点 | 类型 |
| --- | --- | --- | --- | --- |
| 1 | 0x1613b80 | `[%s:%d] [%u][MC2] Check msg equal failed, rMsg:%s msg:%s` | **CopyAndCheckApiMsg 0x7d6ae4，Dlog 0x7d6e14**；0x7d6d40 `memcmp` 64 B 不等；非 slot+48 魔数比较 | **API 消息一致性失败（periodic）** |
| 1 | 0x1613bd0 | `[%s:%d] [%u][%s]call trace: hcclRet -> %d` | ReadApiValidMsg 0x7d6fec/0x7d70b8 | API 条件失败（TraceProfSubmit rc） |
| 1 | 0x1613c00 | `[%s:%d] [%u]ErrToWarn: [%s]call trace: hcclRet -> %d` | 0x7d7074 | API 条件失败 |
| 1 | 0x1613ce0 | `[%s:%d] [%u]reset valid value %u` | ReadApiValidMsg 0x7d71bc；x6=[slot+48] | **API 消费成功标记**（打印写后的 valid） |
| 2 | 0x1613f20 | `[%s:%d] [%u]AicpuKfcRpcServer::Init addr:%u` | Init4 0x7d67cc；x6=y | **RpcSrvLaunch Init4 入口**（共享，打印 y 低 32 位） |
| 2 | 0x1613f58 | `[%s:%d] [%u]Post finishedTurnCnt[%u].cnt = %lu.` | 段内相邻；站点未给 | 候选 stage-wait 完成/轮次标记（待站点） |
| 3 | 0x1614060 | `[%s:%d] [%u]ErrToWarn: ReadValidMsg timeout 6s... ` | CheckRcvWorkMsg 0x7d744c | **TwoStage 等待超时**（6 s） |
| 3 | 0x1614098 | `[%s:%d] [%u]ReadValidMsg timeout 6s... ` | 0x7d744c 非 ErrToWarn 变体 | 同上 |
| 4 | 0x1614408 | `[%s:%d] [%u][AicpuKfcRpcServer][ReadAddrMsg] ReadValidMsg[%u] timeout %lus` | ReadAddrMsg 0x7d8d20；x6=turn、x7=20 | **API 无消息超时**（20/40/80/160 s） |
| 5 | 0x14eb1b0 | `[%s:%d] [%u]ErrToWarn: args is null.` | RunAicpuRpcSrvLaunch arg0 NULL | 共享错误 |
| 6 | 0x14edee8 | `aicpu_kfc_interface.cc` | 共享 file 串 | 共享 |
| 7 | 0x14ee650 | `[%s:%d] [%u][NsRecovery][MC2] Suspending` | 0x762280 | **MC2/Suspending**（非 OneStage；§77 #7 归因撤回） |
| 7 | 0x14ee680 | `[%s:%d] [%u]ErrToWarn: Run rpc error, opIdx:%u, sndCnt:%d %d %d %d, rcvCnt:%d %d %d %d` | 站点未给 | 候选 **字段型** stage-wait 错误（sndCnt/rcvCnt 四值）；待站点 |
| 8 | 0x14ee728 | `[%s:%d] [%u]ErrToWarn: Failed to run aicpu server, MC2 opIdx:%u` | 0x7620e8 | **MC2 错误** |
| 9 | 0x14f7160 | `[%s:%d] [%u]Query sqe info failed, stream:%u, id:%u` | site 待核（35 标 0x7b3a24，但同址另有 #10） | 不确定 |
| 10 | 0x14f8280 | `RunRpcServerApi` | 0x7b3a24 区 | 函数名串 |
| 10 | 0x14f8290 | `[%s:%d] [%u]start to read the [%u] msg` | RunRpcServerApi 0x7b3a24；x6=[ctx+12756]=ctx+0xf31d4(turn) | **API 每轮入口**（打印 turn） |
| 10 | 0x14f82b8 | `[%s:%d] [%u]ErrToWarn: fail to get addr msg, msgPos %u` | 0x7b3a24 区 | API 失败标记 |

### 78.2 修正后的正向标记（按原文校正）

- **API 路径（最强）**：`start to read the [%u] msg`（RunRpcServerApi 每轮，打印 `ctx+12756`=turn，按 0x7b3a24 site）→ 直接证明 API 每轮入口；`reset valid value %u`（0x7d71bc，gate 匹配且 consume=true 后的写回值）→ **条件式成功 consume 标记**；`[MC2] Check msg equal failed, rMsg:%s msg:%s`（0x1613b80，CopyAndCheckApiMsg 0x7d6e14）→ **64 B 消息一致性失败**（0x7d6d40 memcmp；非 valid 魔数比较）；**每 10000 次累计不等才打一次**（0x7d6d58-78 `count%10000==0` + HcclCheckLogLevel(2,3)），**不是逐事件标记**；`ReadValidMsg[%u] timeout %lus`（0x7d8d20，turn/20 s）→ 无消息超时（20/40/80/160 s）。
- **关于 `call trace: hcclRet -> %d`**：原文是 **TraceProfSubmit/内部返回码** 的 trace 日志（0x1613bd0/0x1613c00），**不能直接叫“valid mismatch”**；只能记“ReadApiValidMsg 在 gate 不匹配分支调用 TraceProfSubmit 后的返回码”。
- **TwoStage 路径**：`ReadValidMsg timeout 6s...`（CheckRcvWorkMsg **单次调用**的 6 s 超时）；“每 6 s 输出”需 caller 循环/重试证据，否则只记“每次调用最多一条”。`Post finishedTurnCnt[%u].cnt = %lu.` 与 `Run rpc error ... sndCnt ... rcvCnt ...` 仍是候选，站点未确认前不单独判路。
- **`[NsRecovery][MC2] Suspending`**：原文无 OneStage/22 字样，只能靠 site CFG 条件归属，**不作必经入口**。
- **共享入口**：`AicpuKfcRpcServer::Init addr:%u`（Init4 每次，打印 y 低 32 位）可作“日志在目标 AICPU 生效”的必经标记。
- **MC2**：`[NsRecovery][MC2] Suspending`、`Failed to run aicpu server, MC2 opIdx:%u`。
- **OneStage/Unfold**：本轮 10 段内**没有**可唯一归属的入口串；§77 的 #7 OneStage 归因撤回，保持“无现有标记”。
- **ATRACE KFC 对象**仍只分 KFC vs MRS。

### 78.3 级别/守卫边界

35 给出 guard 原始操作数（如 (2,3)、(0,3)、(1,3)、(3,3)）与 DlogRecord w0/w1；**没有证据表明这些操作数直接取 `ASCEND_GLOBAL_LOG_LEVEL` 的 0–4 枚举**（可能是 bitmask/另一枚举）。因此：
- 不把“设 GLOBAL=0”写成“一定能打开这些标记”；
- 观测有效性仍以“看到 `Init addr:%u` 等必经标记”为准；
- `MODULE_LOG_LEVEL`/`HCCL_DEBUG_CONFIG` 语义仍未证。

### 78.4 更新后的最小观测计划

1. 候选配置同 §77（GLOBAL=0 + PROCESS_LOG_PATH；候选，未执行）。
2. 采到日志后先验 `AicpuKfcRpcServer::Init addr:%u`；无它 → 配置未生效/未进 RpcSrvLaunch，缺测。
3. 有 `start to read the [%u] msg` / `reset valid value` → API 路径；有 `ReadValidMsg timeout 6s` → TwoStage 等待失败；两者都无 → 不排除，需看候选 `Post finishedTurnCnt`/`Run rpc error` 站点后再定。
4. 若出现 `MC2` 串 → 该路径非 API/TwoStage（但同一进程可能多阶段，需按时间序列）。
5. 仍无设备 GO/SSH；字段值（turn 等）仅在上述标记打印参数中出现，属日志 payload，不是内存读取。

## 79. 具体有限观测计划（H11 small；可审核，不执行/不自动 GO）

### 79.1 现有冻结 arm（无需改代码）

- **arm H11**：M128/K256/N512、world=2、BF16 x/w、FP32 zero bias、单 group、`KEEP_ALL`、seed 20261119、child 240 s / comm 90 s；reference=local `x@w` + all_reduce。该 arm 在现有 `run_probe`（`--arm H11 --rows 128 --k 256 --n 512`）与 `supervised_probe_window`（choices 含 H11）中可直接表达。
- **冻结 tuple**：commit `e38b541`/manifest `8e324bea`、run_probe `8e308804`（30,686 B）、window `1f780bf8`、binding `101660f8`、adapter `d181b15a`、test_probe `ad257b5f`；H11 文档需从旧 run_probe/window 引用更新到该 tuple（doc-only，由 35/31 负责）；不修改代码/不新增 arm。
- **恢复**：沿用既有 supervised window 的 aa3e-noop 恢复路径；root identity、service-root、official-start、activate、model-dir、restore-output、stop-script 由 31 按 fresh plan 生成（不在本计划中臆造路径/identity）。

### 79.2 候选环境与命令（未执行）

```bash
# 1) fresh 日志目录（必须已存在、可写）
mkdir -p <fresh>/cann-plog
export ASCEND_GLOBAL_LOG_LEVEL=0          # 0=DEBUG（同版 msft/asys 文本；仅候选）
export ASCEND_PROCESS_LOG_PATH=<fresh>/cann-plog
unset ASCEND_MODULE_LOG_LEVEL HCCL_DEBUG_CONFIG
source <activate_env.sh>                   # 现有 probe 运行前提（无 CANN env 会 torch_npu 段错误）

# 2) probe child（必须由 torchrun 2 ranks 启动；不能直接 python run_probe --execute，
#    否则 world/RANK/LOCAL_RANK 未注入会 ValueError。window build_child_argv 生成如下形状：）
torchrun --nproc_per_node=2 --master_port 29531 \
  runtime/grouped_mmar/run_probe.py --arm H11 \
  --rows 128 --k 256 --n 512 --seed 20261119 \
  --temp-root <fresh>/temp --out <fresh>/out \
  --adapter <frozen>/probe_adapter.so \
  --arm-timeout 240 --comm-timeout 90 --execute

# 3) window 形状（其余 restore/identity 参数按既有 H11 plan 模板由 31 填）
python runtime/grouped_mmar/supervised_probe_window.py --arm H11 \
  --rows 128 --k 256 --n 512 --seed 20261119 \
  --probe-temp-root <fresh>/temp --probe-out <fresh>/out \
  --adapter <frozen>/probe_adapter.so \
  --master-port 29531 --child-timeout 240 --comm-timeout 90 \
  --run-dir <fresh>/run --root-identity <fresh identity json> \
  --source-target <aa3e source target> --official-start <official start> \
  --activate <activate_env.sh> --service-root <service root> \
  --model-dir <model dir> --restore-output <fresh>/restore \
  --stop-script /tmp/d0cc/d0_stop.py \
  --text-body <existing cell text body> --image-body <existing cell image body>
```
- 端口 29531 执行前须 `ss -ltn` 预检空闲（既有前置）；不自动 GO。
- `ASCEND_GLOBAL_LOG_LEVEL=0` 只是候选：**不假设 Init4/ReadApiValidMsg 的 guard 会被它满足**；未证模块/bit 不设置。

### 79.3 采集与归属（本 cell）

- **时间窗**：probe child 启动到 child 退出/被 watchdog 杀（≤240 s）；restore 阶段单独按时间/PID 排除。
- **对象（完整 raw 保留）**：`<fresh>/cann-plog`、`/root/ascend/log/device-*/`、window/child/result/trace **原件全部保留，不做删除式过滤**。严格按 probe PID 过滤只作**派生视图**：AICPU/device 日志的 PID 可能是 daemon/设备侧 ID，只按 probe PID 过滤会丢关键行。
- **归属字段**：rank/device（device-0/1 文件）、pid/tid、file:line（`aicpu_kfc_rpc_server.cc:27/226/241/285/360`、`aicpu_kfc_process.cc:1715` 等）、时间；有则记 stream/task/opIdx。
- **关联视图（另存，不替代原件）**：rank/device、pid/tid、file:line、时间、stream/task/opIdx；未知关联保留歧义并标 `unknown`，不强行归属。
- **restore 排除**：只作派生视图（按时间/PID 标注），**不删除原件**；aa3e 服务 root 五 PID 与 probe PID 可能并存，归属不明确时保留歧义。
- **注意**：240 s watchdog 杀进程时日志可能未完全 flush；只采用 kill 前已落盘行，但原件全量保留以便复核。

### 79.4 期望证据与允许判读（正向标记；缺席不排除）

| 观测 | 支持 | 限制 |
| --- | --- | --- |
| `AicpuKfcRpcServer::Init addr:%u`（line 27，Init4） | 该次运行进入 **Init4/RpcSrvLaunch** 且该级别 Dlog 生效 | 只证 Init4 路径；若实际走 `Init(y)`/ForApi 或 MC2，不要求它必现，**缺失不能判日志关闭** |
| `start to read the [%u] msg`（line 1715，0x7b3a24） | RunRpcServerApi 每轮读 msg（API 路径） | 需该 site 实际执行；缺席不排除 |
| `reset valid value %u`（0x7d71bc） | gate 匹配且 consume=true 的成功路径 | 条件标记；gate 失败/未读不出现 |
| `[MC2] Check msg equal failed...`（0x1613b80，CopyAndCheckApiMsg 0x7d6e14） | **64 B 消息一致性失败**（memcmp） | **每 10000 次累计不等才打一次**；出现=持续不一致且计数到万倍；缺席=累计不足/guard 未开，不排除；非逐事件、非 valid 魔数比较 |
| `ReadValidMsg[%u] timeout %lus`（0x7d8d20） | API 无消息超时（20/40/80/160 s） | 非入口；消息到达后算法卡住不触发；缺席不排除 |
| `ReadValidMsg timeout 6s...`（0x7d744c） | CheckRcvWorkMsg **单次调用** 6 s 超时 | **不宣称每 6 s 周期**；多次出现需 caller 循环/重试证据；缺席不排除 |
| `[NsRecovery][MC2] Suspending`（0x762280） | 站点 CFG 条件证据（MC2/Suspending） | 文本无 OneStage/22；非必经入口 |
| `Failed to run aicpu server, MC2 opIdx:%u`（0x7620e8） | post-Init 错误并走向清区 | 非入口 |
| `Post finishedTurnCnt...` / `Run rpc error ... sndCnt/rcvCnt` | 候选（站点未定） | 未确认前不判路；确认后可能提供字段型证据 |
| ATRACE KFCtaskAndTiling/AivAicpuOpParam | KFC vs MRS | 不分子分支 |

**有效性判据（按实际路径）**：任一**适用**的 K-entry/循环标记（Init4 用于 RpcSrvLaunch；`start to read` 用于 RunRpcServerApi/ForApi；若实际为 MC2 则以 MC2 标记为适用入口）出现，才算该路径日志生效；**不能把 Init4 缺失当全局日志关闭**。没有任何适用正向标记 → 本次运行按**缺测**处理。

### 79.5 允许结论与缺测退出

- **可判**：出现 `start to read`/`reset valid` → API 路径；出现 6 s 单次超时 → TwoStage 等待失败（该次调用）；出现 MC2 串 → 该阶段 MC2；出现 Init4 但无任何分支标记 → 只证 RpcSrvLaunch Init4，不判子分支。
- **不可判（缺测退出）**：无任何适用正向标记；或只有共享/候选标记；或日志未落盘/被 kill 截断。此时记录“缺测”，不排除任何路径，下一步才是 host launch 日志或代码埋点（需单独授权）。
- **不执行**：本计划不运行设备、不自动 GO、不新增 arm；plan/identity/GO 由 31/32 按既有流程决定。

## 80. `Check msg equal failed` 站点确认（35 3d26ea1）

- 站点：Dlog **0x7d6e14**，函数 `AicpuKfcRpcServer::CopyAndCheckApiMsg(HcclMsg*,HcclMsg*)` **0x7d6ae4**；由 `ReadApiValidMsg` 在 **0x7d70e4** `bl CopyAndCheckApiMsg@plt` 调用。
- 比较对象：`0x7d6d30 dsb ld`；`0x7d6d34 mov x2,#0x40`；`0x7d6d38/3c` 取两参；**`0x7d6d40 bl memcmp` 比较 64 B**；`0x7d6d48 b.eq 0x7d6e4c`（相等继续）。**不是 slot+48 魔数比较**。
- 日志门：不等后 `0x7d6d4c` 取全局计数（`0x22ca000+0xe64`），`0x7d6d58-78` 判 `count % 10000 == 0`，仅余数为 0 进入；再过 `HcclCheckLogLevel(2,3)`@`0x7d6d88`。Dlog 参数 w0=3/w1=2/x2=0x1613b80/x3=0x1613cc8(line 207)/x5=tid/x6=GetMsgSimpleStr(rMsg)/x7=GetMsgSimpleStr(msg)。
- 判读：出现=累计 64 B 不等达到 10000 的倍数（持续不一致）；缺席=累计不足 10000 或 guard 未开，**不排除**不一致；**非逐事件标记**，也不用于判 valid 魔数。

## 81. 32 H11 草案预审结果（无 GO）

**已确认（32 本机 plan-only）**：
- H11 是真实 arm：`flags=8 KEEP_ALL`、`native=True`、`rows=128`、bias `fp32_zero`；plan-only 已验 H11/flags8/native/128/256/512/240/90。
- argv：torchrun 2 ranks + `run_probe --arm H11 --rows 128 --k 256 --n 512 --seed 20261119 --temp-root/out fresh --adapter d181b15a --arm-timeout 240 --comm-timeout 90 [--execute]`。
- env：window 设 `GLOBAL_LOG_LEVEL=0` + fresh `PROCESS_LOG_PATH`；`child_env` 继承到 torchrun/ranks；restore 用 `env -i`，观察 env 不进入服务；source SHA 检查。
- window `1f780bf8` 含 H11，兼容 plan-only/240 s/cleanup/唯一 restore；runtime 用 `e38b541`。35 已完成 H11 doc-only 更新：`h11_matched_control.md` SHA `fa62db6c`（commit 2b5729e），旧 plan `68cc5a79`/run_probe `0f90fd23`/window `2202165e` 标注为历史、不是当前 tuple。

**采集口径（32 同意）**：fresh `PROCESS_LOG_PATH` + 默认 `/root/ascend/log` 时间窗 + device 日志**全量**；**不按 probe PID 过滤**；按 time/device/file/host 派生；正向标记出现才算采到，否则缺测。级别证据为 cf9f1d6/rodata 33a2226；guard 映射/新 xref 未闭。

**阻断/前置（当前不运行、不发 GO）**：
1. 无设备 GO；
2. **device/AICPU daemon 的 env 传播未证**——window 的 env 只证到 child/torchrun，daemon 是否继承未证；无正向标记时不能区分“未生效”与“未走该路径”；
3. guard 级别映射未证（Dlog w0/w1 不套 GLOBAL 0–4）；
4. fresh log dir 必须预创建；
5. 不得按 PID 过滤，raw 全量保留；
6. H11 hang 未解，本观测仅诊断，不构成根因结论；
7. H11 文档需从旧 run_probe/window 更新到 `e38b541`（doc-only）。

## 82. 32 预审 v2：两个 argv 阻断已修正

- **阻断 1（已修）**：probe child 不能直接 `python run_probe.py --execute`——`execute` 要求 world=2 且 RANK/LOCAL_RANK 由 torchrun 注入，直跑会 ValueError。§79.2 已改为 window `build_child_argv` 的 torchrun 形状：
  `torchrun --nproc_per_node=2 --master_port 29531 runtime/grouped_mmar/run_probe.py --arm H11 --rows 128 --k 256 --n 512 --seed 20261119 --temp-root <fresh>/temp --out <fresh>/out --adapter <frozen>/probe_adapter.so --arm-timeout 240 --comm-timeout 90 --execute`。
- **阻断 2（已修）**：window argparse **required** `--text-body` 与 `--image-body`，且 `build_plan` 检查文件存在；§79.2 已补 `<existing cell text body>` / `<existing cell image body>`（取现有 cell 文件，fresh plan 检查存在性）。
- **可闭项**：H11 flags8 KEEP_ALL/native/rows128/bias fp32_zero；child 由 build_child_argv 生成 torchrun，传 `--arm-timeout 240`/`--comm-timeout 90`/`--adapter`；KEEP_ALL/backend/契约一致。
- **env**：`ASCEND_GLOBAL_LOG_LEVEL=0`、`ASCEND_PROCESS_LOG_PATH=<fresh abs dir>` 必须在**启动 window 前**导出到进程环境（不在 argv）；window `child_env={**os.environ, CLEF_WINDOW_OWNER}` → torchrun→ranks 继承；fresh dir 预创建；执行中只读 `/proc/<rank>/environ` 验证。`GLOBAL=0` 仅候选。
- **restore**：`env -i PATH=... CLEF_LAUNCH_CONFIG=s1 setsid ... start.sh`；观察 env 不进入服务；branch/五 PID/source 三段 SHA 检查保留；aa3e-noop fresh identity 对应现网 root `1337814/400405818`，由 31 预检。
- **前置**：fresh identity/port/dirs、stop-script SHA、text/image body、log dir、31 preflight；**无设备 GO、当前不运行**。修正 2 个 argv 阻断后继续 CPU 预审，仍不发 GO。

### 82.1 预审版本对齐

32 的 §81 补审消息引用的是 **7c2cf0a 之前**的 §79.2 文本；当前版本（7c2cf0a/§82）已：
- probe child 改为 `torchrun --nproc_per_node=2 ... run_probe.py --arm H11 ... --execute`；
- window argv 已补 `--text-body` / `--image-body`；
- env 顺序、restore `env -i`、采集口径不变。
请以当前 §79.2/§82 复核；若仍看到旧文本，说明审阅快照未更新。H11 文档新 SHA 为 `fa62db6c`（35 2b5729e）。
