# BF16 MatMulReduceScatter 探针（2026-11-16）

此前 `npu_mm_all_reduce_base` 在 `ascend910_93` 缺 `MatmulAllReduce` 内核（161001）。本轮验证另一条接口：`torch_npu.npu_mm_reduce_scatter_base(..., comm_mode="aiv")` 对应的 `aclnnMatmulReduceScatterV2`。结论：**V2 在 A3/TP2/BF16 可用，单算子 reduce-scatter 有 1.10–1.23×，但“reduce-scatter + all-gather”还原完整输出只有 1.04–1.06×（1596 token 0.97×），暂不接入服务。**

## 1. 探针

- 真实 layer31 `down_proj` 分片：`K=8704`、`N=5120`（TP2 每 rank）；
- 基线：`F.linear(x, W)` + `dist.all_reduce(sum)`，输出完整 `[M, 5120]`；
- 融合：`npu_mm_reduce_scatter_base(x, W.T, hcom, 2, comm_mode="aiv")`，输出本地 `[M/2, 5120]`；再用 `all_gather_into_tensor` 还原 `[M, 5120]` 做等价比较；
- `comm_mode=None` 作为默认模式对照；
- 每个点 3 warmup + 10 次，rank-0 device event 平均；
- M 取 308 / 970 / 1596（由 307/969/1595 向上补齐到 2 的倍数）；
- 运行：`torchrun --standalone --nproc_per_node=2`。

## 2. 结果

| M | 模式 | 基线 ms | 融合+AllGather ms | 加速 | 仅本地 RS ms | 本地加速 | max_abs |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 308 | **aiv** | 0.2606 | **0.2451** | **1.06×** | 0.2126 | **1.23×** | 0.03125 |
| 970 | **aiv** | 0.3983 | **0.3773** | **1.06×** | 0.3355 | **1.19×** | 0.03125 |
| 1596 | **aiv** | 0.6017 | 0.6229 | 0.97× | 0.5458 | **1.10×** | 0.03125 |
| 308 | 默认 | 0.2606 | 0.3855 | 0.68× | 0.3103 | 0.84× | 0.03125 |
| 970 | 默认 | 0.3983 | 0.5756 | 0.69× | 0.4854 | 0.82× | 0.03125 |
| 1596 | 默认 | 0.6017 | 0.8701 | 0.69× | 0.7447 | 0.81× | 0.0 |

- **`comm_mode="aiv"` 是必要的**；默认模式反而慢 0.68–0.69×；
- `aiv` 输出与基线不是逐位相等，`max_abs=0.03125`（BF16 舍入），`mean_abs≈7e-7`，有限值正常；
- 单独 reduce-scatter（下游保持 sequence-sharded）有 1.10–1.23×；  
  但“RS + AllGather 还原完整输出”只有 1.04–1.06×（308/970），1596 反而 0.97×。

## 3. 判断

1. **接口可用性已确认**：`npu_mm_reduce_scatter_base(comm_mode="aiv")` 在 A3、TP2、BF16、真实 down_proj 形状上成功执行，不是此前的 161001。
2. **单算子收益不足以直接接入**：现有 MLP 是 `RowParallelLinear` 的 MatMul + AllReduce；替换成 RS+Gather 只在短/中序列约 1.06×，长序列退化；要把 1.10–1.23× 兑现，必须让下游保持 sequence-sharded，做跨层 Dense SP（residual/norm/attention/GDN 都在本地 token 分片），这是完整模型改造，不是换一个算子。
3. **质量风险**：`aiv` 与 BF16 基线有 0.03125 的最大差；若接入必须跑全量 OD/ERQA 四项 2PP，不能只看单算子误差。
4. **暂不接入提交**：保留为后续 Dense SP 的可行性证据；当前 BF16 BIC 仍是主路径。

## 4. 复现

```bash
cd /workspace/yangchen/Clef-27B-ascend
source scripts/activate_env.sh
python -m torch.distributed.run --standalone --nproc_per_node=2 \
  runtime/mc2_reduce_scatter_probe.py \
  --tokens 308,970,1596 --iters 10 --warmup 3 \
  --output /workspace/yangchen/logs/gdn-data-path-20261115/mc2-rs/probe.json
```

## 5. 证据与恢复

- 探针：`Clef-27B/logs/gdn-data-path-20261115/mc2-rs/probe.json`、`probe.log`
- 服务恢复：`.../restore-mc2-rs/`，baseline **PID 678485**，argv/cwd/env 与 phase-0 一致，`/health` 200，冻结 8 例 **8/8**
