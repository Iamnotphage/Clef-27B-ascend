# GEMM + AllReduce 分块重叠探针（2026-11-16）

按 GEMM 融合/重叠方案，先实现不改布局的 down_proj 分块：把 `[M, K]` 按 token 切成 C 块，计算下一块 GEMM 的同时，对上一块结果做 async AllReduce。结论：**HCCL 小 AllReduce 固定开销和流同步远大于重叠收益，2/4 块全部慢于一次完整 AllReduce；不采用。**

## 1. 方法

- A3、TP2、BF16、layer31 down_proj：`K=8704`、`N=5120`；
- 所有 rank 使用相同全局 x，按 rank 取本地权重分片；
- 基线：`partial = F.linear(x, W)` → `dist.all_reduce(partial, sum)`；
- 分块：把 x 按行切成 C 块；每块在 compute stream 上 GEMM，record event；comm stream 等待 event 后对块结果做 `dist.all_reduce(async_op=True)`；最后 `wait()` 并让主 stream 等待 comm stream；
- 3 warmup + 10 次，device event 平均；C=2/4；M=308/970/1596。

## 2. 结果

| M | 基线 ms | C=2 ms | 加速 | C=4 ms | 加速 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 308 | 0.247 | 0.591 | **0.42×** | 0.856 | **0.29×** |
| 970 | 0.408 | 0.512 | **0.80×** | 0.883 | **0.46×** |
| 1596 | 0.610 | 0.771 | **0.79×** | 0.883 | **0.69×** |

正确性：与基线 `max_abs=0.03125`（C=2）或 `0.015625`（C=4），mean_abs 5–8e-7，有限值正常。

## 3. 解释

1. **固定开销**：把一个 AllReduce 拆成 2/4 个小 AllReduce，HCCL 每次通信的固定开销重复 2/4 次；308 token 的 down 输出很小，固定开销占主导。
2. **流/事件同步**：compute/comm 之间的 event wait 和 work.wait 没有形成有效的硬件并行，反而增加了串行依赖。
3. **同层内没有足够计算可重叠**：down GEMM 本身只有约 0.2–0.6 ms；AllReduce 即使重叠，能隐藏的时间也有限，且小消息效率低。
4. 这与 MLP-only Dense SP 的结论一致：**单层/同层内做 GEMM+通信融合不划算**；只有跨层 sequence-sharded 布局（Full Dense SP）才可能把多个层的通信重叠起来。

## 4. 结论

- **不采用**分块 AllReduce 方案；不修改模型源码。
- `docs/npu-gemm-fusion-overlap-plan-2026-11-16.md` 中的决策门仍成立：MLP-only SP 0.88–0.98×、同层分块 0.29–0.80×，均未过 G1。
- Full Dense SP 仍是唯一可能利用 13.4 ms 非重叠通信的路线，但需要跨层布局和 attention/GDN 改造；建议优先移植上游 `c6857b4`，而不是继续做同层融合。
- 当前保留 BF16 BIC S1 198.42 ms、S2 B8 C16 16.464 dec/s 候选。

## 5. 证据与恢复

- 探针：`runtime/mlp_allreduce_overlap_probe.py`，`Clef-27B/logs/gdn-data-path-20261115/mlp-overlap/probe.json`、`probe.log`
- 服务恢复：`.../restore-mlp-overlap/`，baseline **PID 702535**，argv/cwd/env 与 phase-0 一致，`/health` 200，冻结 8 例 **8/8**
