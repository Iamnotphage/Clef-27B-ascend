# GroupedMatMulAllReduce 支持矩阵与通信/stream/生命周期契约研究（2026-11-19）

只读研究：不写 31/35 代码、不做设备/SSH/installer/推理请求。环境固定 A3 `Ascend910_9382`、CANN 9.2.1、torch 2.10.0+cpu、torch_npu 2.10.0.post7。

来源分层（必须区分）：

- **本机 9.2.1 安装证据**：31 只读 API 包 `/tmp/aa3e-grouped-api-pack-20261119`（tar SHA `685d9a33…`）：头文件、op-info、binary_info_config、两个 A3 kernel `.o`、libopapi/libnnopbase/libhccl/libascendcl/libtorch_npu、version.info、platform ini、torch_npu HCCL 头与符号。
- **上游固定参考**：`hicann/ops-transformer` commit `f5703e415b9be16a2af7ba85564c37414592fc03` 的 op_api 源、op_host 源、kernel 源、docs 与 examples。**不是本机安装版同版本真值**。
- **A12 冻结样例**：`/Users/chenclay/.ao/data/artifacts/clef-27b-ascend-33/grouped-a12-sample/`（repo example SHA `6c1a90ab…`；文档参考 `31333c59…`；编译配方 `c591f0e3…`）。

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
