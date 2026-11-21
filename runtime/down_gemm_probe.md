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

解析器分别取双 rank 的较慢 device/wall 中位数，保留 A/A 最大绝对差作为观测噪声。

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

CPU 验证：AST/计划入口、精确 shape 筛选与计数、缺 rank/缺配对拒绝、本地两 rank
旧 CSV 复算。NPU 运行与后续判断由 owner 回传原数后补记。
