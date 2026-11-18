# Wave 3 通信探针：载荷曲线与单层串行基线（2026-11-18）

## 0. 结论

1. **T2（allreduce payload/下游 matmul K 切分）关闭**：在 p95 主导的 967 token 上，半载荷只让单次 collective 早回来 **1.5%**（ratio 0.985），1595 为 8.4%（0.916）。固定启动成本约 **30 µs**，带宽约 **110 GB/s**，967 的 9.4 MB 中固定项占约 24%。切半不能把 967 的 p95 往前挪。
2. **真正的 K-reduction 切分（down_proj K=8704 分两块）结构性不成立**：每块仍产出 `[M,5120]` partial，需要 **两次全载荷 allreduce** 加一次 partial add；按实测 0.125 ms/次链式成本，通信量翻倍，最好情况也只能与当前暴露持平，且引入额外同步与精度变化。
3. **N（输出列）切分**同样没有收益：通信总字节不变，每列块产出不相交输出，967 上半载荷 collective 不更早返回。
4. **M（行/token）切分**为已关闭方向（0.29–0.80×，`docs/npu-gemm-overlap-probe-2026-11-16.md`）。
5. **T1 的上限已量化**：单层 `down_proj → allreduce → AddRmsNorm` 串行链中加入 allreduce 的边际成本为 **0.204 / 0.208 / 0.218 ms**（600/967/1595 token），×64 层即 **13.1 / 13.3 / 14.0 ms/请求**；967 的 13.3 ms 与线上 profile 的非重叠通信 14.17 ms 吻合。这是任何融合/重叠方案的天花板（占 p95 8.5%），不是预测收益。

## 1. 窗口与设备

| 项 | 值 |
| --- | --- |
| 设备 | A3 单物理卡 2 逻辑 NPU（`Ascend910_9382`，卡 7 / Chip 0、1），全窗口仅本会话使用 |
| 服务 | 探针前停服务；快照 `service-before-wave3-probe.json`（PID 947988、argv/env/health/NPU 状态） |
| 恢复 | 探针后按默认 `--config s1` 重启，PID **972842**，`/health` ready，文本 choice/score 与图文请求均 200，conv+QKV 标记 2 次 |
| 被测代码 | `perf/wave2-npu-wave3` @ `7757804`（与 main `09b50d3` 树一致） |
| host load | 17.6–18.5 |

## 2. allreduce 延迟-载荷（TP2，BF16）

`torchrun --nproc_per_node=2 runtime/wave3_allreduce_payload_probe.py`，每个形状预热 5、30 次单次计时、32 次链式 ×5 轮。

| tokens | payload | 链式 ms/次 | 有效带宽 | 单次+sync ms |
| ---: | ---: | ---: | ---: | ---: |
| 967 | 9.44 MB | **0.1249** | 73.9 GB/s | 0.397 |
| 1595 | 15.58 MB | **0.1643** | 92.6 GB/s | 0.434 |
| 4800 | 46.88 MB | **0.4531** | 101.0 GB/s | 0.738 |
| 9600 | 93.75 MB | **0.8837** | 103.6 GB/s | 1.187 |

最小二乘（链式 ms = fixed + slope × MB）：**fixed = 0.0304 ms**，**slope = 0.009082 ms/MB ⇒ 110.1 GB/s**。
（证据 JSON 中的 `implied_bw_gbs` 字段由修正前的公式写出、偏小 1024×，以 `implied_bw_gbs_recomputed` 为准。）

## 3. 三种“切分”的严格区分

| 维度 | 定义 | 通信量 | 实测/推演 | 结论 |
| --- | --- | --- | --- | --- |
| **M（行）** | 把 `[M,K]` 按 token 切块分别 allreduce | 块数 × 小 payload | 已实测 0.29–0.80× | 已关闭 |
| **N（输出列）** | 把 `[M,K]×[K,N]` 的 N 切块，各自 allreduce 不相交列 | 总字节不变 | 967 半载荷 ratio 0.985 | 不采用 |
| **payload/下游 K** | 把 allreduce 输出按 hidden 切两半，下游 matmul 先算第一半（T2 原设想） | 总字节不变，调用数×2 | 967 0.985、1595 0.916 | **关闭**（967 无早返回） |
| **K-reduction（down_proj K=8704）** | 沿 GEMM K 切两块，partial Y 相加 | **2 次全载荷** + partial add | 按实测 0.125 ms/次，通信翻倍 | **结构性不成立** |

K-reduction 的算术：现状 `gemm(0.32) → AR(0.125) → AddRmsNorm` 的边际暴露 0.208 ms；切两块后 `gemm_half(0.16) → AR_full(0.125) → gemm_half(0.16) → AR_full(0.125) → add → norm`，即使 AR1 完全藏在 gemm2 后面，仍需再付一次 0.125 ms 加 partial add，最好情况与现状持平，典型情况更差。

### 3.1 N 切分的双流对照（直接墙钟，不外推）

`runtime/wave3_dual_stream_probe.py`：同一 communicator，比较「一次 `[T,5120]`」与「两次 `[T,2560]`」的三种发法，各自一次 sync。

| tokens | 一次整块 | 两次·同流顺序 | 两次·双流并发 |
| ---: | ---: | ---: | ---: |
| 967 | 0.3459 | 0.4210（**1.217×**） | 0.5209（**1.506×**） |
| 1595 | 0.3923 | 0.4472（1.140×） | 0.5296（1.350×） |

- 双流并发比整块慢 **35–51%**：同一进程组内的并发 collective 在 910_93 上并非双向并行，反而叠加了流调度与同步成本。
- 前一个窗口（`allreduce-payload.json` 的 payload_split）曾测得 967 顺序两次/整块 = 0.985、1595 = 0.916；两个窗口的方向一致：**半载荷不会更早返回，双流更差**。窗口间绝对值的差异（整块 0.396 → 0.346 ms）属于负载漂移，不影响判定。
- 因此「N 切分 + 双流」这条 T2 的最后一种形态也关闭。

## 4. 单层串行基线与理想重叠下界

`runtime/wave3_layer_serial_bound_probe.py`，真实形状 `X[M,8704] @ W[8704,5120]` + `all_reduce([M,5120])` + `npu_add_rms_norm`；A/B/C 三项各只做一次 host sync，保证开销可比。

| tokens | A: gemm+AR+norm | B: gemm+norm | C: AR only | A−B（加 AR 的边际成本） | ×64 层 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 600 | 0.6527 | 0.4485 | 0.3957 | **0.2042** | 13.07 ms |
| 967 | 0.6678 | 0.4595 | 0.4068 | **0.2083** | 13.33 ms |
| 1595 | 0.8635 | 0.6455 | 0.4888 | **0.2180** | 13.95 ms |

- A−B 是“把一次 allreduce 插进层链”的真实边际成本，含 launch/同步边界；链式 allreduce 本身只要 0.125 ms，差值是 0.083 ms 的排序/同步开销。
- B 与 C 的单独测量都含一次 sync（0.27 ms 量级），不能相加；因此这里用 A−B 而不是 A−max(B,C) 作为暴露量，避免把测量开销算成收益。
- 967 的 13.33 ms 与线上 profile（`perf-wave2-20261118/profile/window-967`，非重叠通信 14.17 ms）一致，说明该探针可作为 T1 的判定基线。

## 5. 对 T1/T2 的判定

| 方向 | 判定 | 依据 |
| --- | --- | --- |
| T2（payload / 下游 K 切分） | **关闭** | 967 半载荷 0.985；固定成本 30 µs 主导 |
| T2（K-reduction 切分） | **关闭** | 通信翻倍 + partial add，最好情形持平 |
| N 列切分 | 不采用 | 通信不变、无早返回 |
| T1（MatMul+AllReduce+Add+RMSNorm 融合） | **保留为唯一候选**，天花板 13.3 ms/请求（967） | 上述 A−B 实测；下一步是离线融合核微基准，过 0.85× 门槛才接服务 |

T1 的关键前提仍待核实（22 号并行）：A3/CANN 9.2.1 上是否存在可用的 `AllReduce + Add + RMSNorm` KFC/context API（上游 PR #12649 是 row=1 decode 形状，#19033 是 decode graph 路径），以及本形状 `M=967, N=5120, K=8704` 的 kernel 是否支持 TP2 通信域。**本轮未编译、未上板任何新 kernel。**

## 6. 证据

| 内容 | 仓库内 | 远端原始 |
| --- | --- | --- |
| 载荷曲线 | `Clef-27B/logs/perf-wave3-20261118/allreduce-payload.json` | `/workspace/yangchen/logs/perf-wave3-20261118/allreduce-payload.json` |
| 单层串行基线 | `Clef-27B/logs/perf-wave3-20261118/layer-serial-bound-v2.json` | 同目录 `layer-serial-bound-v2.json` |
| 服务快照 | — | `perf-wave2-20261118/snapshots/service-before-wave3-probe.json` |
| 探针源码 | `runtime/wave3_allreduce_payload_probe.py`、`runtime/wave3_layer_serial_bound_probe.py` | 同路径 |

复现：

```bash
# 先停服务并确认 NPU 无他人进程，然后：
torchrun --standalone --nproc_per_node=2 runtime/wave3_allreduce_payload_probe.py \
  --output /workspace/yangchen/logs/perf-wave3-20261118/allreduce-payload.json
torchrun --standalone --nproc_per_node=2 runtime/wave3_layer_serial_bound_probe.py \
  --output /workspace/yangchen/logs/perf-wave3-20261118/layer-serial-bound-v2.json
```
