# 新 A3 上的 BF16 gate/up 成对列实验

## 方法

新机卡 6 的逻辑设备 0/1 对应 Phy 12/13。实验前核对 aa3e/S1 进程、设备归属、服务健康和请求队列，使用本 worker 自己启动的基线让出独占设备窗口。源码以 MIX `a5cf535ee24ed26d3d4be7a23ea029aeb3b91466` 为依据，使用已验 CATLASS `eec373fbe601446c72cac422cf63142221d85bf9` 和 CANN 9.2.1 构建私有 `ascend910_93` vendor。

实验算子以配对顺序一次性重排各 rank 的原 BF16 权重。通用 Cube tile 是 128×256，输出相邻的 gate/up 各 128 列。短输入变体采用 256×128，输出 gate/up 各 64 列，让一份权重 tile 覆盖更多 token 行。两种均为整 K 的 FP32 累加及 BF16 输出。融合模式用两个 GM 槽做 Cube/Vector 交接，SwiGLU 以 BF16 投影值为输入并写回 BF16 激活。完整链使用原厂 down。原型各模式均在私有进程中调用。

两 rank、三层 `[31,0,63]` 使用既有的六份新机 aa3e 真实 BF16 激活。M307 和 M967 均取 `od-0677` 的真实 M1595 父输入前缀。M307 代表该长度的局部算子形状。原厂和候选每层读取同源权重。权重重排和 NZ 转换计在一次性准备，所有臂的在线分配、调用和必要搬运计入各自阶段。raw 只测 GEMM，候选以成对列顺序输出，数值阶段才做逆排列对照。activation 两臂都输出原顺序 BF16 激活。chain 两臂都经过原厂 down。

## 实测

七轮交错，每轮每臂三层 streaming。以下是每轮先取双 rank 慢者，再取七轮中位数。Wall 和 Event 单位均为 µs/层。配对差以候选减原厂为正。A/A 列是原厂重复臂 Event 配对绝对差的七轮最大值。

| M | 阶段 | 原厂 Wall / Event | 配对候选 Wall / Event | 候选 Event 配对差 | A/A 最大 |
| ---: | --- | ---: | ---: | ---: | ---: |
| 967 | raw | 653.97 / 619.91 | 683.26 / 652.10 | +29.77 | 28.05 |
| 967 | activation | 679.24 / 651.42 | 729.84 / 697.53 | +46.11 | 21.43 |
| 967 | chain | 996.59 / 967.64 | 1057.15 / 1029.07 | +59.21 | 20.35 |
| 307 | raw | 278.40 / 250.62 | 370.03 / 338.83 | +95.94 | 29.19 |
| 307 | activation | 300.11 / 268.36 | 384.54 / 356.55 | +96.08 | 32.10 |
| 307 | chain | 430.41 / 399.40 | 526.78 / 497.82 | +102.20 | 37.90 |
| 307 | short raw | 278.40 / 250.62 | 423.60 / 390.15 | +137.09 | 29.19 |
| 307 | short activation | 300.11 / 268.36 | 440.36 / 412.39 | +141.98 | 32.10 |
| 307 | short chain | 430.41 / 399.40 | 578.40 / 548.87 | +137.55 | 37.90 |
| 1595 | raw | 1029.13 / 1003.44 | 1007.87 / 978.81 | −20.51 | 64.80 |
| 1595 | activation | 1081.10 / 1050.61 | 1084.72 / 1051.61 | +0.41 | 24.34 |
| 1595 | chain | 1577.32 / 1543.15 | 1595.91 / 1565.11 | +21.96 | 19.08 |

24/24 组数值通过。相同投影输入的 SwiGLU 在 24 组均逐位一致。相对原厂完整链的最大相对 RMSE 为 0.0002841。raw 和 fused 的输出、workspace 哨兵及重复调用检查均通过。申请的 workspace 分别为 16,777,216 和 19,922,944 字节，差值为两槽环形缓冲的 3 MiB。设备窗口外层 14 秒，探针记录 6.077 秒。基线加载和恢复另计。

M967 的 raw、activation 和 chain Event 均为 0/7 胜。融合交接进一步扩大差距。M307 的短形状大 M tile 增加调度 tile 数，Event 在三个阶段均为 0/7 胜。实际 HBM 读字节和同步等待本轮未采集。M1595 的 raw 优势落在 A/A 波动内，完整链为 0/7 胜。保留原厂 BF16 NZ MLP，本候选止于局部实验。

## 复现和证据

源码 `runtime/ascendc/clef_mlp_gate_up_pair/`，入口 `runtime/probe_mlp_gate_up_pair.py`。在 CANN 9.2.1 环境设置固定 `CATLASS_ROOT` 后运行 `build.sh`，用 `--noexec --extract=<私有目录>` 解包。ACLNN 的 `mode` 实参为 int64。探针传入新机经核验的 `--vendor`、`--model-dir`、`--captures` 和新 `--output`，默认 M967/307/1595、rank 0/1，每臂同窗七轮。

| 文件 | SHA-256 |
| --- | --- |
| `custom_opp_ubuntu_aarch64.run` | `610bef8524f731bdc9f2f402c43c4e200e0de2a7d4210e863a9594706f8d0732` |
| `libcust_opapi.so` | `da34659128287bd1b2a987dc96764750f12fcf31427587b51ed41fbac3bb640d` |
| `ClefMlpGateUpPair_dbc29d0ee336f4a17954084a1b02b78a.o` | `a20286b22ef12c5311ef1d7b868cb72273224e334af42520db3636b686dfa5a8` |
| `paired.json` | `524afc0affea3417afa22999e393d53d4f99905b5325c319750bc9822081c005` |

新机私有构建路径为 `/workspace/yangchen/work/gate-up-pair-20261123.xCG0x0yq`，原始证据为 `/workspace/yangchen/logs/gate-up-pair-20261123.xCG0x0yq`。

## 服务恢复

停止窗口内的实验进程后，重新启动同一 aa3e/S1 原厂 BF16 NZ 配置。全新 API PID 90267，双 rank worker 90386/90387，卡 6 的两个逻辑设备。18143 实测 `ready`，22/22 项验收通过，涵盖版本和启动配置、设备归属、文本及图文协议。标准工作树保持干净，实验 vendor 留在私有目录。

`final-aa3e-checks.json` 的 SHA-256 为 `2534e2a964d73ee2045d1652f9ac46a0b58acc42087f0f45f01df6d8961d2524`。本机与新机恢复指针 `current-aa3e-44.json` 的 SHA-256 均为 `e7381272122a0dab2fb685cc677f4edd089a3edecb32da5e9b21624ee5772ec2`。
