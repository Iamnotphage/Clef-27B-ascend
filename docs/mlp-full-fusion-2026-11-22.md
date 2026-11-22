# 单 kernel BF16 MLP 原型

固定基线 `60f7337`。候选为每 TP rank 的 `gate/up → SwiGLU → down`，保留外部 TP2 all-reduce。CANN 9.2.1、Ascend A3、原 NZ29 权重、ND 输入。私有 vendor 仅供局部实验。

AIC 用固定 CATLASS `eec373fb` 的同一 OFF block 实现两段 GEMM。AIC 先写 128×256 的 gate 和同列 up，row stride 512，两份均从 FP32 整 K 结果经 F322BF16 转出。两个对应 AIV 各拥有 64 行，分 32 行转 FP32 算 SiLU(gate)×up，RINT 回 BF16，再写 [M,8704] 激活。AIV 完成全局 MTE3 屏障后通知各组 AIC 开始 down。每个 down 输出 tile 按原 K 轴顺序累加完整 8704 个值，L0C 保留 FP32 部分和，最后一次 F322BF16 写 ND [M,5120]。

| 空间 | 容量 | 寿命 |
|---|---:|---|
| AIC L1 / L0A / L0B | 384 / 32 / 64 KiB | 两段 GEMM 顺序共用 K256/K64 双缓冲 |
| AIC L0C | 128 KiB | 128×256 FP32，跨完整 K 保留 |
| AIV UB | 144 KiB | 每次 32 行，两份 BF16 输入、一份 BF16 输出、三份 FP32；CANN 2201 SiLU 利用独立目标暂存 |
| GM gate/up 环 | 6 MiB | 24 组，各两对 tile；FIX 发布，两个 AIV 经 MTE3 回压后该组复用 |
| GM 激活 | M967 16.054 MiB | 全局发布后供每个 down N tile 读取；每次调用独立 workspace |
| SDK workspace | 运行时 GetWorkspace 一并请求 | 采用官方 GetUserWorkspace 约定，计入完整调用 |

同组 AIC/AIV 使用相同 swizzle、group 和 ordinal。两次 READY 后 AIC 等两个 AIV 的回压，第二 lane 没有有效行也按协议参与。最后一个不足两槽的批次经全局发布屏障结束。Ready/recycle/phase 使用 0/1/4，CATLASS AIV 全局 barrier 使用 8。定长 CPU 检查覆盖 1、64、65、127、128、129、307、967、1595 的逻辑槽位、单写所有权、尾块和 down 读取范围。静态工具在显式协议中仅报性能提示，CATLASS 原头文件的 4 个红线、7 个高候选见私有原始报告。CPU 模型无法验证实际硬件可见性，设备哨兵和阶段比较独立执行。

比较过两种替代数据流。M128 若每个 down N256 tile 独立重算上游，固定 N5120 导致 20 路重复。每个激活 K256 批次都把 down FP32 部分和溢出到 GM，则 M967 的 33 次往返约 1.22 GiB 逻辑流量。也可让多个 AIC 并行持有互异 N tile 并广播激活，但会占据约 20 个 AIC 服务一个 M128 分块，还须另外证明 gate 生产、跨组发布和负载。当前原型选顺序阶段，保留 down 权重排布与缓冲，省显式跨 kernel 交接和完整 gate/up 峰值容量。M967 gate/up 逻辑写读仍约 64.22 MiB，ring 优先改变局部性。激活至少有一写一读约 32.11 MiB，20 个 down N tile 各发起自己的 A 读取，真实 L2/HBM 流量未知。先前 rank0 OFF task 与独立 SwiGLU 给出约 89 µs/层的机会量级，实际开销包含 GM 搬运、回压、全局屏障、down tiling、分配与下发。

CPU 复现执行 `python3 runtime/check_mlp_full_fusion.py`，用固定 CATLASS 源设置 `CATLASS_ROOT` 后执行独立源码目录的 `bash build.sh`。设备入口为 `runtime/probe_mlp_full_fusion.py`，默认只打印计划。使用私有 vendor 和独占卡 7 执行，数字和计时均加载同一份真实 NZ29 权重与归档输入，M967 明标 1595 前缀。诊断 attr 生成 ABI 为 int64，导出 [down、gate、up、激活] 连续区段供数值定位，正常分配 ND [M,5120]。源码编译包为单个 MIX 二进制，实际 task 数以需要时的 Level1 trace 核验。

## 实测结果

本轮在 CANN 9.2.1 编译完成。正式构建先以纯源码归档避开 macOS 元数据文件，再为 CANN 生成的 MIX wrapper 补上 `lib/matmul_intf.h`，实际 GEMM 使用 CATLASS 底层 API。固定源码哈希、生成 ACLNN 头文件和 MIX 二进制 SHA 在私有执行记录中。构建未安装到共享 OPP 或服务。

M967、307、1595 × 两 rank × 三层共 18 个真实用例通过。gate/up 相对原 vendor 最大 relRMSE 为 3.87e−5 / 3.65e−5。同一组 BF16 gate/up 送入原 SwiGLU，激活 18/18 逐位一致。同激活的 down 最大 relRMSE 为 4.21e−5，整链对原 vendor 最大 3.02e−4。输出和 workspace 两端哨兵未变，输出全量有限，普通与诊断路径和重复调用均逐位一致。实际 workspace 为 27.097 / 38.054 / 48.479 MiB（M307/967/1595），其中 SDK 预留 16 MiB。这里的阶段误差上界来自固定输入，未进行服务任务质量验收。

每 arm 取两 rank 中较慢者再做同轮配对。以下单位均为 ms/层。权重布局准备与预热单列，计时含每次分配、workspace、下发及全部 GM staging。

| M967 场景 | 原链 Event | 原型 Event | 配对中位差 | A/A 最大差 | Event 胜出 |
|---|---:|---:|---:|---:|---:|
| 三层 streaming | 0.952733 | 1.009393 | +0.057820 | 0.011207 | 0/5 |
| 单层 warm | 0.967120 | 1.100340 | +0.139480 | 0.022760 | 0/5 |

三层 streaming wall 中位 0.996067→1.037380，配对 +0.041313，1/5 胜出。首轮原链 wall 有离群值，完整 A/A 最大 0.267327。warm wall 1.059840→1.176510，配对 +0.140429，0/5 胜出。307/1595 完成阶段数值护栏，M967 已触发局部停止条件。

代表 rank0 的两对 Level1/AiCoreNone 中，42 个 task 按 stream/task/name/start 与 CSV 核同。原链每层两个 vendor NZ GEMM 和一个 SwiGlu，候选每层一个 `ClefMlpFullFusion` MIX_AIC，BlockDim 24。task union 逐对 941.470→940.010 µs、942.790→940.210 µs，原链 A/A 的绝对差最大 1.227 µs。首 task 前区间在原链两样本分别 113.96、24.39 µs/层，候选为 84.23、81.08 µs/层。短采前／后，候选−原链的 Event 中位差约 +78.8／+58.3 µs。首轮有采集扰动，kernel 内部回压、GM 命中、两段 GEMM 的各自时间及主机适配细分仍缺测。减少下发次数后，task union 节省约 2 µs/层，完整调用没有收益。本实现止于局部实验。

新账 38.659624/240 s，包含两段数值、主计时和短采子进程。旧三账冻结。恢复原 aa3e/S1，API 1763009/start424819444，boot `393ecc15-721f-46f7-b119-21fa327a2b3b`，两个 TP worker 为 1763156/1763157。22 项、原 argv/env、卡 7 chip 0/1 和两端指针 fresh 核对通过。没有进入集成、S1 三轮或 OD/ERQA。

复现资料位于本地 tmp `clef40-full-mlp-bhj4ztar` 和远端 `/workspace/yangchen/work/clef40-full-mlp-bhj4ztar`。其中 `window.json` 记录实际命令、预算和恢复结果，`numeric967.json`、`numeric_guards.json`、`timing967.json`、`profile967.json` 是原数。恢复后导出的 `trace_view.json`、`kernel_details.csv` 用 `runtime/analyze_mlp_gate_profile.py --sample-prefix fullmlp/ --divisor 3` 关联。大 raw 与私有包留在上述目录。
