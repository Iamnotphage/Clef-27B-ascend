# D0 有限诊断设计（只设计，未实施；不改 kernel、不编译、不上板）

基线源码：`85b3028`（包 `873ef8fb`）。**行号以该提交为准。**

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

**方案（需新增 ABI，属待实现）**：
- **新增独立输出** `trace`（dtype `int32`/`float`，ND），由 **host 声明 shape**
  `trace[slots][fields]`，**adapter 侧显式分配**并按 shape 读取；
- **容量上界**在 host 侧计算并写入 tiling：`slots = 2 * AIC_max`（AIC、AIV 各一域），
  `fields = 8`；超出即在 host 侧 `GRAPH_FAILED`，**不进 kernel**；
- workspace 划分**追加**一段 `trace_bytes`，与 `partial/reduced/flag` **显式分段**，
  互不覆盖；host 校验总 workspace ≥ 各段之和。

## 2. 每参与者唯一槽（分 AIC/AIV 域，用原始值）

**槽位公式（不依赖待验证 ratio / `GetBlockIdx()` 推导）**：
```
domain  = (ASCEND_IS_AIC ? 0 : 1)          // 分域，避免 AIC/AIV 别名
block   = get_block_idx()                  // 原始 AIC 域块号
sub     = get_subblockid()                 // 原始子块号 0..1（AIC 上为 0）
slot    = domain * AIC_MAX + block * 2 + sub
```
- **为什么用原始值**：`GetBlockIdx()` 在 AIV 是 `get_block_idx()*ratio + sub`，
  ratio 尚未由运行时确认；若用它推槽，一旦 ratio 假设错就会**把两个参与者映射到同一槽**（别名）。
  用 `domain/block/sub` 三元组则槽位唯一，**与 ratio 无关**。
- **越界处理**：`block` 或 `sub` 超界、或 `slot >= slots` ⇒ **该 block 停止写 trace**
  并置一个 `oob` 标记（写域内固定位置），**绝不截断或回绕**。

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
- **实际写入覆盖（另一件事）**：需要 host 在同步后**全量读 `partial`**，
  检查每行是否被写过（例如用行级哨兵值/NaN 预填）。⇒ **只证明"最终可读状态"**。
- **消费点 GM 可见性（第三件事）**：AR/Norm 在设备内读到 `partial` 的时序与可见性，
  **本轮设计无法由 host 侧推断**，单列未观测项。
- **`partial` 全量诊断与固定数值门分开记录**：前者是覆盖诊断，后者仍是 mixed 容差
  （`1e-5 + 0.016|ref|`，全元素）。
- **首末行 checksum 仅作采样范围校验**，不代表全张量。

## 5. 单次有界超时、退出/恢复与判读

| 项 | 值 |
| --- | --- |
| 超时 | **单次有界**（建议 180 s）；到点即杀进程 |
| 退出动作 | 记录 `RAW_RC`、trace 已写槽数、是否 OOB；**恢复服务到既有配置**并记录 PID/argv |
| 停止条件 | 出现 `507014/0x25`、超时、或 trace `oob` 置位 ⇒ **停，不进入预算 2** |

**判读表**

| 观测 | 可判读 | **不可**判读 |
| --- | --- | --- |
| trace 出现预期槽数、区间不交且覆盖 M | 认领划分正确、参与者集合符合预期 | 数值正确、GM 可见性 |
| 槽数少于预期 | 参与者集合不完整（需结合 launch 记录） | 具体原因 |
| 超时 | **仅"本次窗口未完成"** | **不能单独归因 `SyncAll`**（也可能 server/quit/AR 段） |
| `partial` 有未写行 | 漏写 | 是哪个环节漏 |
| `partial` 全写且可读 | 最终可读状态 OK | 消费点可见性 OK |

## 6. 明确未观测项（交付即列出）

1. `MMFUN_END` / `SERVICE_QUIT` 的实际发出与 server 收到；
2. 消费点 GM 可见性；
3. 运行时 `get_subblockdim()` / `get_block_idx()` / `GetBlockNum()` 的真实值；
4. `__MIX_CORE_AIC_RATION__` 与逐 pass `-D` 宏表；
5. 预算 2 的一切结论（未设计为已排格）。

## 7. 实施前置（未做）

需**新增独立输出 `trace` + host shape + adapter 读取**（新 ABI），
**不复用 8 float `diag`**；需 20 显式授权设备时间。
本文件为**可审查设计**，未改 kernel、未编译、未上板。
