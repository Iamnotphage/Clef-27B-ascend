# down GEMM 最小诊断（2026-11-21）

CPU 准备基线 `bcb98d9201a8f63e009349d1c2a96898d64064cd`。唯一设备 owner 为
[40](ao://sessions/clef-27b-ascend/clef-27b-ascend-40)，41 仅做 CPU 分析。
原权重 64 层 BF16、共享 GEV 和当前服务配置保持原状。本轮无 PR、无合并。

## 已有证据

旧 f810 head profile 的主干 bf027/ead25/a4de 与 aa3e 对应，绝对数值归属旧版本。
`perf-wave2-20261118/profile/window-967/manifest.json` 记录 10 请求；两 rank CSV
精确筛选 `967,8704;544,320,16,16`、`ND;FRACTAL_NZ`，各 640 条：

| rank | down 累计 ms/请求 | kernel 均值 μs | MAC | MTE1 | MTE2 | Fixpipe |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 20.5361194 | 320.876866 | .839350 | .710103 | .790925 | .011798 |
| 1 | 20.5573616 | 321.208775 | .838949 | .709563 | .789804 | .011741 |

两 rank 均值差 0.103385%；Block Num 全为 24。全部 down 前一同 stream kernel 为
SwiGlu，间隙中位数 0.020 μs、最大 0.040/0.041 μs；该采集窗口的相邻设备 dispatch
间隙很小。跨 rank 时钟与 CPU dispatch 时间没有用于相减。

原采集只有 `ACL_AICORE_PIPE_UTILIZATION`；`l2_cache=false`、`record_op_args=false`。
这些重叠活动计数仅支持上述原数；actual tiling、逐核尾部负载、L1/L0 复用字节与
MTE/Cube/Fixpipe 依赖等待仍缺证据。当前不据此判定 bound 或专用核收益。

CSV SHA-256：rank0 `b755a0fd35e2b49abd5869bdbcad6d9dec15fca5c0da0d992d1ff5743c5203ed`，
rank1 `8cd18efae98751083f5de58a7209e7bd66994eac258500c9a73a65a37ca84e34`。

## owner 单次串行窗口

把本目录 `probe_down_gemm.py`、`analyze_down_gemm.py`、基线 `probe_mlp_ffn.py`
放到同一个独立远端工作目录，如 `/workspace/yangchen/work/down-diagnosis-20261121/`。
加载现有环境后执行，输出路径必须全新：

```bash
timeout 240 python /workspace/yangchen/work/down-diagnosis-20261121/probe_down_gemm.py \
  --phase timing --execute --ranks 0,1 --tokens 307,967,1595 --pairs 5 \
  --output /workspace/yangchen/logs/down-diagnosis-20261121/timing.json
```

该脚本 rank0→rank1 串行，每 rank 真实 layer31 权重。307 复用
`mlp-20261115-1027/capture/od-0105-7b4ab816610eaeda.layer31.rank{0,1}.pt`；
1595 复用 `od-0677-a3153de1d6509904.layer31.rank{0,1}.pt`，967 明确标为其前缀。
原始 BF16 MLP 输入经过 vendor gate+SwiGLU 得到 down 激活；记录来源 metadata 与 SHA。
这属于历史激活回放，与当前 aa3e 全链激活的差异保留在验证范围中。

仅测固定 K8704/N5120，三种条件各 5 对相同函数 A/A，按轮次轮换条件顺序：

- warm：同一权重预热。
- read_flush_proxy：256 MiB 读刷后的冷态代理。
- streamed_down_ring：layer0/32/63 的真实 TP2 down 权重先运行，再测 layer31。
  三份前置权重共 255 MiB，加目标权重总工作集 340 MiB。前置算子只用于缓存条件，
  使用同一层31激活；该轮转仍属于局部缓存代理。

每 rank 每层完整 MLP 权重 255 MiB、64 层约 15.94 GiB；下投影 ring、读刷与单权重
warm 均分列，服务 streaming 结论仍由 owner 的完整链配对支持。
最多 258 次目标 GEMM + 180 次前置 GEMM，额外活跃张量预计约 1 GiB；活跃张量
超过 1.5 GiB 即停，缓存分配器保留与 vendor workspace 另有开销。预计 240 秒内；
超时由 owner 停止本探针并保留部分输出。脚本不调用服务、不停启、不编译、无后台。

计时只包围目标 down `F.linear`，逐次保存 NPU event、异步 enqueue、同步 wall、
缓存条件 wall。权重 NZ 转换和 gate/SwiGLU 输入准备在计时外，范围显式记录。
有效 FLOPs=`2*M*8704*5120`；输出必须 finite 且重复 vendor 调用逐位相同。
本轮没有 pad/crop/选路候选，后续候选的完整链必须计入这些成本。

CPU 解析：

```bash
python runtime/analyze_down_gemm.py --timing-json <timing.json> --output <summary.json>
python runtime/analyze_down_gemm.py \
  --kernel-csv 0=<rank0/kernel_details.csv> --kernel-csv 1=<rank1/kernel_details.csv> \
  --requests 10 --layers 64 --tokens 967 --output <existing-summary.json>
```

解析器分别取双 rank 的较慢 device/wall 中位数，分别保留两类 A/A 最大绝对差作为观测噪声。

## 一次必要补采与停止条件

若 owner 现有工具能直接导出 actual tiling 和逐核指标，最多为 M967 各 rank 补采一次；
工具缺失时记录缺项并停止，不安装工具、不为此重启服务或无限扫 tile。
独立 replay 入口供已有外部 profiler 包裹，计时与 profiling 分开执行：

```bash
python /workspace/yangchen/work/down-diagnosis-20261121/probe_down_gemm.py \
  --phase once --execute --ranks 0 --tokens 967 --output <new-profile-replay.json>
```

此入口包含 3 次同 shape 准备/预热和最后 1 次读刷后的目标；外部采集按精确 down
shape 选最后一次，准备阶段其他算子与转换成本单独保留。rank1 使用另一次串行调用。
所需字段：kernel 名/tiling key、baseM/N/K、singleCoreM/N/K、usedCoreNum/blockDim、
stepM/N/K、双 buffer/iterate order（可导出部分按实保存）、每核时长/有效工作量、
L1/L0 搬运量与命中、MTE1/2/Cube/Fixpipe 活动及依赖等待、原始 CSV/tiling 文本。
只有观测实际 tile 后才挑最多两个必要尾点，本包当前禁止额外 M。

缺输入、dtype/shape 不符、非 finite、重复结果变化、资源/工具错误立即停。
缺具体气泡/不均衡证据则结束诊断；候选改善不超过同窗口 A/A 配对噪声则停止该候选。
不重跑 ND/SP/MC2/N1 已败机制、不研究通信融合、不据 MAC 活动或经验方阵推算剩余空间。

## 单次实测与收口

40 在同一设备窗口串行执行成功，torch `2.10.0+cpu`、torch_npu `2.10.0.post7`。
原数 `/workspace/yangchen/logs/clef40-mlp-nyklqirq/down-timing.json`，SHA-256
`b9201a59a22db86fac6f216739193ac5b0d7ca9b2b63889cdccedd6d3ab679a7`；执行探针 SHA-256
`971118a385d47f329d6a1dc2777b0cc08ce6d5e4e000bf911eb0075fc27bd66a`。
两 rank 各 3 个 M、每条件 5 对，共 180 个计时样本，重复输出逐位检查全部通过；
6/6 捕获输入的完整张量 SHA 与既有 `capture-requests.json` 相符，活跃张量最高 811.06 MiB。

以下分别取两个 rank 的 device/wall 中位数较大者；噪声为两个 rank 所有 A/A 对的
`200*abs(A-B)/(A+B)` 最大值。缓存条件保留独立范围，没有候选收益或服务加速结论。

| M | 条件 | device ms | wall ms | device A/A max % | wall A/A max % |
| ---: | --- | ---: | ---: | ---: | ---: |
| 307 | warm | .114600 | .193485 | 8.248 | 135.288 |
| 307 | read flush | .138530 | .213920 | 18.286 | 17.660 |
| 307 | down ring | .129980 | .209230 | .997 | 2.881 |
| 967 | warm | .318730 | .389800 | 4.775 | 6.049 |
| 967 | read flush | .332570 | .406825 | 3.205 | 1.351 |
| 967 | down ring | .319740 | .401420 | 5.610 | 4.051 |
| 1595 | warm | .478130 | .561825 | 13.092 | 15.252 |
| 1595 | read flush | .498830 | .577085 | 2.432 | 2.335 |
| 1595 | down ring | .481680 | .559595 | 4.399 | 4.237 |

967 ring 的 rank0/1 device 中位数分别为 .315250/.319740 ms；warm 为 .318730/.306220，
read flush 为 .332570/.330070。与旧 profile 约 .321 ms/层处在相近量级，版本、输入、
缓存条件与窗口差异保留，不能将该差值记为优化收益。307 warm 有一次 wall 1.003390 ms
离群值，原数完整保留。

40 确认本轮没有 actual tiling/per-core 导出结果，设备采集到此停止。
**结论：已有证据未发现显著 rank 级失衡或 SwiGlu→down 邻接 dispatch 气泡；
核内尾块、L1/L0 复用与 MTE/Cube/Fixpipe 等待仍无法归因，当前缺少专用 BF16 down
核的立项依据。本轮结束，不启动专用核、不扩展 tile/尾点扫描。**

CPU 验证：5 项单测全部通过，覆盖精确 shape/计数/时间单位、独立选择 device/wall 慢 rank、
A/A 噪声、缺 rank/缺配对/非 finite 拒绝及设备执行参数边界；AST/计划、旧两 rank CSV
与 owner 原数解析通过。41 全程 CPU-only，服务恢复和身份校验由唯一 owner 40 负责。
