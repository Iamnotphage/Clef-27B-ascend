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
