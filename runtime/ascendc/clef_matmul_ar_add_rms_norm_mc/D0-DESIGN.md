# D0 有限诊断设计（只设计，未实施；不改 kernel、不编译、不上板）

基线源码：**`85b3028`**（包 `873ef8fb`）。**下列行号经 `git show 85b3028:<kernel>` 实测**：
`WriteTilingDiag()` 调用 `:151`、`REGIST_MATMUL_OBJ` `:184`、worker 判定 `:221-222`、`IterateAll` `:249`、`End()` `:255`、**无参屏障 `:295`**。
> 此前文中的 `:308` 取自**后续（仅改注释/文档的）提交**的 HEAD，**已更正**。

## 0. 本轮只回答什么

**预算 1 为第一候选**：请求 `mm_dim = 1` + **AR 关闭**（沿用既有 `use_hccl = 2` 通路）。
**预算 2 仅条件草案**，不是已排定的第二格。

只回答三件事：**(a)** 实际参与者身份集合；**(b)** 每个参与者认领的 M 区间；
**(c)** 认领覆盖是否等于 M。**不**回答数值因果，**不**替代固定数值门。

## 1. trace 出口容量方案（现有 diag 不能复用）

现状（实读）：`diag` 是 **固定 8 float** —— `diag_` 的 `SetGlobalBuffer`/`InitBuffer(diagT_, 32)`/
三处 `DataCopy(diag_[0], d, 8)`；`partial` 位于 **user workspace**，并非算子输出。
因此**不能**把 trace 塞进 8 float，**也不能**与 workspace 复用同一块 (会与
`partial[M,N] | reduced[M,N] | flag` 重叠)。

**数据归属只选一种：独立输出**（不采用"workspace 追加一段"的写法，两者此前混写已删除）。
- **新增独立输出** `trace`：dtype **`uint32`**、format ND、**host 声明 shape `trace[slots][fields]`**、
  **adapter 侧显式分配**并按该 shape 读取；
- **容器与 workspace 完全分离**：`trace` 是算子第 5 个输出，不占 user workspace 的
  `partial | reduced | flag` 任何一段；host 另行校验总 workspace 与 trace 字节数互不相关；
- **容量**：`slots = AIC_CAP * (1 + SUB_CAP)`，`fields` 见 §2 字段表；
  `AIC_CAP` 来自 host 侧平台 AIC 数（tiling 时写入），`SUB_CAP = 2` 为**声明上界**。

## 2. 每参与者唯一槽（分 AIC/AIV 域，用原始值）

**槽位公式（分域紧凑；实测已消除别名与越界）**：
```
// AIC 域
slot = block                                    // 0 .. AIC_CAP-1
// AIV 域
slot = AIC_CAP + block * SUB_CAP + sub           // AIC_CAP .. AIC_CAP*(1+SUB_CAP)-1
slots = AIC_CAP * (1 + SUB_CAP)
```
`block = get_block_idx()`、`sub = get_subblockid()` 均为**原始值**，不使用 `GetBlockIdx()` 的推导式。
以 `AIC_CAP=24, SUB_CAP=2` 校验：AIC(12,0)=**12**；AIV(0,0)=**24**；AIV(12,0)=**48** < 72。
**无别名、无越界**（旧式 `domain*24 + block*2 + sub` 会让 AIC(12) 与 AIV(0) 都落到 24、
且 AIV(12) 落到 48 越界——该写法**已废弃**）。
- **为什么用原始值**：`GetBlockIdx()` 在 AIV 是 `get_block_idx()*ratio + sub`，
  ratio 尚未由运行时确认；若用它推槽，一旦 ratio 假设错就会**把两个参与者映射到同一槽**（别名）。
  用 `domain/block/sub` 三元组则槽位唯一，**与 ratio 无关**。
- **越界处理**：`block`/`sub` 超界或 `slot >= slots` ⇒ **只关闭该 block 的 trace 写**，
  **保留 `REGIST_MATMUL_OBJ`／析构／屏障的原控制流不动**（trace 不得改变主路径）。
- **不设共享 `oob` 字**：多个写者写同一位置会竞态。改为**缺槽即证据**——
  host 事后按 `slots` 全量扫描，**缺哪一槽就报哪一槽未闭合**；若越界者本应占的槽不存在，
  它自然表现为"未闭合集合"的一员。
- **host 不能预知 builtin 越界**：`AIC_CAP/SUB_CAP` 只是**声明上界**；
  真实 builtin 若超出，host 无法先验拦截，**只能靠缺槽诊断暴露**（不假定其不会发生）。

## 2.1 字段表（每参与者 × 每阶段的独占记录）

`fields = 8` 个 `uint32`，每参与者一条，**阶段独占**，不共享：

| idx | 字段 | 写入阶段 | 说明 |
| --- | --- | --- | --- |
| 0 | `magic` | K1 | 固定魔数；**未写即表示该参与者未到达 K1** |
| 1 | `id` | K1 | `(block<<8) | sub`（原始值打包） |
| 2 | `domain` | K1 | 0 = AIC，1 = AIV |
| 3 | `claim_start_m` | K4 | 认领起点（非 worker 写 0 并把 bit0 清零） |
| 4 | `claim_m` | K4 | 认领行数 `curSingleM` |
| 5 | `stage_bits` | K1..K8 逐位或入 | 位图：bit0=K1、bit1=K2、bit2=K3、bit3=K4、bit4=K5、bit5=K6、bit6=K7、bit7=K8 |
| 6 | `reserved` | — | 置 0 |
| 7 | `seal` | 结束处 | `0xC0FFEE00 ^ id`，**写完才置** |

- **dtype/容量**：`uint32`，每条 `8*4 = 32 B`；总字节 `slots * 32`（示例 72 槽 → 2304 B）。
- **初始化**：host 侧分配后**清零**，故"`magic` 为 0"= 从未到达；`stage_bits` 逐位可判读。
- **完整记录判读**：`magic == 常量` **且** `seal` 正确 ⇒ 该参与者记录完整；
  否则按 `stage_bits` 判读到过哪些阶段，**不把不完整记录当成缺失**。

## 3. 插桩表（分"kernel 可达点"与"SDK 内部点"）

| # | 点 | 本包位置 | 能证明什么 | 不能证明什么 |
| --- | --- | --- | --- | --- |
| K1 | kernel 入口 / 域名判定后 | 入口函数（`Process()` 之内） | 本 block 的身份三元组 | 是否被 server 消费 |
| K2 | **`REGIST_MATMUL_OBJ` 之前** | `op_kernel:184` 前 | 进入接线前的身份与时序 | 接线是否成功 |
| K3 | **`REGIST_MATMUL_OBJ` 之后** | `:184` 后 | **仅 AIV 到达**；**AIC 在此从 `Process()` 返回**，其后代码 AIC 不可达 | AIC 之后的任何状态 |
| K4 | worker 判定 | `:221-222` | 本 block 是否认领、认领区间 | 是否真的写入 |
| K5 | **`IterateAll` 返回后** | `:249` 之后 | 调用已返回（客户端视角） | cube 是否完成、GM 是否可见 |
| K6 | **`End()` 调用前/后** | `:255` 前/后 | **仅证明调用与返回** | `MMFUN_END` 是否真的发出 |
| K7 | **无参屏障前/后** | `:308` 前/后 | 本 block 到达/离开屏障 | 屏障语义与参与者集合 |
| K8 | 作用域退出（析构） | 接线作用域结束处 | **仅证明范围退出** | `SERVICE_QUIT` 是否真的发出 |

**SDK 内部点（本包无观测手段，保留缺项）**：
`MMFUN_END` 实际发出、`SERVICE_QUIT` 实际发出、server 收到 quit、server 执行完 cube ——
这些都在 `KfcCommClient`/`KfcServer`/`asc_opc` 产物内部，**没有本包可插的观测点**；
如将来需要，必须另做 SDK 侧观测（非本设计范围）。

**client 发送与 server 接收分列**：K5/K6/K8 属**客户端侧**；server 侧一律标"未观测"。

## 4. 认领区间覆盖 vs 实际写入覆盖（刻意分开）

- **认领覆盖（可 host 检查）**：把每个参与者的 `(startM, curSingleM)` 写进 trace，
  host 侧检查 `Σ curSingleM == M` 且区间两两不交 ⇒ **只证明"声称的划分正确"**。
- **实际写入覆盖（另一件事，本轮保留为缺口）**：`partial` 目前**在 user workspace 内、
  不是算子输出**，**没有 host 读取路径**。首候选**只判认领覆盖**；
  实际写入覆盖**保持缺项**，除非另增 `partial` 输出或安全导出方案（属新 ABI，未实现）。
- **消费点 GM 可见性（第三件事）**：AR/Norm 在设备内读到 `partial` 的时序与可见性，
  **本轮设计无法由 host 侧推断**，单列未观测项。
- **`partial` 全量诊断与固定数值门分开记录**：前者是覆盖诊断，后者仍是 mixed 容差
  （`1e-5 + 0.016|ref|`，全元素）。
- **首末行 checksum 仅作采样范围校验**，不代表全张量。

## 5. 单次有界超时、退出/恢复与判读

| 项 | 值 |
| --- | --- |
| 超时 | **单次有界**（建议 180 s）；到点即杀进程 |
| 退出动作 | 记录 `RAW_RC`、可读到的 trace 槽；**恢复服务到既有配置**并记录 PID/argv |
| 超时/故障后 trace | 可能**无法 copy** ⇒ 标 **`unavailable`**；**不保证槽数可回收** |
| 停止条件 | 出现 `507014/0x25`、超时、或 trace `oob` 置位 ⇒ **停，不进入预算 2** |

**判读表**

| 观测 | 可判读 | **不可**判读 |
| --- | --- | --- |
| trace 出现预期槽数、区间不交且覆盖 M | 认领划分正确、参与者集合符合预期 | 数值正确、GM 可见性 |
| 槽数少于预期 | **只报告"未闭合集合"**（缺槽可能来自插桩未到／丢写／提前失败，三者不可区分） | 具体原因、以及是否 `SyncAll` 所致 |
| 超时 | **仅"本次窗口未完成"** | **不能单独归因 `SyncAll`**（也可能 server/quit/AR 段） |
| `partial` 有未写行 | 漏写 | 是哪个环节漏 |
| `partial` 全写且可读 | 最终可读状态 OK | 消费点可见性 OK |

## 6. 明确未观测项（交付即列出）

1. `MMFUN_END` / `SERVICE_QUIT` 的实际发出与 server 收到；
2. 消费点 GM 可见性；
3. 运行时 `get_subblockdim()` / `get_block_idx()` / `GetBlockNum()` 的真实值；
4. `__MIX_CORE_AIC_RATION__` 与逐 pass `-D` 宏表；
5. 预算 2 的一切结论（未设计为已排格）；
6. `partial` **实际写入覆盖**（无 host 读取路径）；
7. 超时/故障后 trace 的可回收性（标 `unavailable`）。

## 7. 实施前置（未做）

需**新增独立输出 `trace` + host shape + adapter 读取**（新 ABI），
**不复用 8 float `diag`**；需 20 显式授权设备时间。
本文件为**可审查设计**，未改 kernel、未编译、未上板。
