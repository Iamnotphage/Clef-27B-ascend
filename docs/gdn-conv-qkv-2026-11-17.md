# conv + QKV norm 融合算子与通用 QKV 归一化算子（2026-11-17）

## 0. 结论

- 新增 Ascend C 候选 `ClefGdnConvQkv`：一次调用完成 causal conv1d 与 q/k L2 norm、v 直拷，
  输出 BSND `q/k/v`，替代原来 conv（3.79 ms / 967 token）与 `ClefGdnQkvNorm`（4.66 ms）两个 kernel。
- 三次探针（v1/v2/v3）共 **177 组** 长度、seed、tile、输入布局组合全部 finite、`max_abs = 0`，
  与部署版 conv + QKV v2 两段流水**逐位一致**（`Clef-27B/logs/perf-next-20261117/conv-qkv-probe-v{1,2,3}.json.gz`）。
- v2 起支持 **packed QKVZ `[T,8192]`** 输入：Qwen3.5 把投影结果拆成 QKV `[T,5120]` 与 Z 两个视图，
  过去要 `contiguous()` 拷贝一次；现在用受存储边界检查的 `as_strided` 把整行 8192 暴露给算子，
  kernel 只读每行前 5120 通道，Z 保持原值，省掉逐层 5120×T×2 B 的 D2D 拷贝。
- v3 把逐行 `ReduceSum` 换成 FLA `08d21f7` arch22 prepare-vector 的 tile 归约：
  先 `Add` 两个 64 列半行，再一条连续 mask `WholeReduceSum<float>` 归约整个 tile。
- 服务级（install-v2）在 307/967/1595 三个长度各 50 次固定请求上，`answers` 与 eager 对照
  **JSON 完全一致、数值最大差 0**，eligibility 日志确认真实走到 `x_stride=(8192,1)`、`x_contiguous=False` 的快路径。
- 另建通用 QKV 归一化算子 `ClefGdnQkvGeneric`（独立 vendor），支持 TP1 H=80 / TP2 H=40 任意 token 数，
  36 组硬件检查通过：TP2 对部署 v2 逐位一致，TP1 对 FLA 最大绝对差 0.001953125。
- **尚未完成**：install-v3（tile 归约）服务级复核、全量 S1 与全量质量；host load 长期 40+ 期间不测。

## 1. 背景：为什么值得融合

最终版 967 token profile（`docs/profile-final-2026-11-17.md`，cache-only、NZ 前）里，
单次前向的设备 compute 144.6 ms 中 conv **3.79 ms**、`ClefGdnQkvNorm` **4.66 ms**，合计 8.45 ms（5.8%）。
两者是同一条数据路径上的相邻步骤：conv 的输出就是 norm 的输入，且都只依赖当前 prefill 的 token。
把它们合成一个 kernel 的收益有两层：

1. 少一次 kernel launch 与一次完整的中间张量写回/读入（`[T,40,128]` BF16 ≈ 9.9 MB @967 token）；
2. 少一次 `[T,8192] → [T,5120]` 的连续化拷贝（QKVZ 视图）。

## 2. 算子设计

输入输出契约（`runtime/ascendc/clef_gdn_conv_qkv/`）：

| 张量 | 形状 | 说明 |
| --- | --- | --- |
| `x` | `[T,5120]` 或 `[T,8192]` BF16 | 连续 QKVZ 投影；8192 布局下只读前 5120 通道 |
| `weight` | `[4,5120]` BF16 | depthwise conv tap，width=4 |
| `q` / `k` | `[1,T,8,128]` BF16 | L2 归一化后的 BSND |
| `v` | `[1,T,24,128]` BF16 | SiLU 后直拷 |

- 单序列、零左历史（`skip_initial_state` / `skip_conv_state`）、无 bias、SiLU、TP2 固定头数，
  其他形状由宿主 guard 返回 `None` 走原始路径。
- 每核处理一个 head 的连续 token tile（tile=32/64），`DataCopyPad` 以 `inputChannels_` 为 GM 行跨距，
  因此 5120 与 8192 两种布局共用同一份 kernel，没有分支。
- Q/K：BF16 → FP32，`x*x`，tile 归约，`rsqrt(sum+eps)`，`Mul`，`CAST_RINT` → BF16；V 逐位搬运。
- tile=32 UB 预算约 88.5 KiB，tile=64 约 152.8 KiB，均含 512 B 对齐余量。

两侧宿主 guard（`runtime/apply_conv_qkv_patch.py` v2）：

```python
if not x.is_contiguous():
    if x.stride(1) != 1 or x.stride(0) != 8192:
        return None
    if (x.storage_offset() + tokens * 8192) * 2 > x.untyped_storage().nbytes():
        return None
    x = x.as_strided((tokens, 8192), (8192, 1))
```

`as_strided` 不复制数据，但会越过 `[T,5120]` 视图的边界读到 Z 区；因此先校验底层存储确实覆盖
`tokens × 8192` 个元素，再交给算子。单元测试覆盖“视图可展开”与“存储不足必须回退”两条路径
（`runtime/test_conv_qkv_patch.py`，共 42 项全部通过）。

## 3. 离线探针证据

工具：`runtime/clef_conv_qkv_probe.py`，参考实现为 NPU 上的 conv + 部署版 `ClefGdnQkvNorm` v2
（`/workspace/yangchen/Clef-27B-ascend/Clef-27B/src/ascendc/vendors/customize`），
另有 CPU FP32 参考。输出预先填 NaN，逐项记录 finite / NaN / Inf / 逐位 / max-abs / mean-abs / RMSE。

| 探针 | 覆盖 | 结果 | kernel `.o` SHA-256 |
| --- | --- | --- | --- |
| v1 | 10 长度 × 3 seed × tile 16/32/64，连续 5120 | 30/30 逐位 | `f668d671607a5f31…` |
| v2 | 307/967/1595 × 3 seed × 3 tile | 27/27 逐位 | `6694bfff8a8097d3…` |
| v3 | 10 长度 × 3 seed × tile 32/64 × 输入 5120/8192 | 120/120 逐位、0 失败 | `19c89bc15eb3c10b…` |

v3 单算子计时（NPU event 中位，ms，`--seed 20261117`）：

| tokens | 输入 | tile | 参考（conv + QKV v2） | 融合 | 加速 |
| --- | --- | --- | --- | --- | --- |
| 307 | 5120 | 32 | 0.2372 | 0.1812 | 1.31× |
| 307 | 8192 | 32 | 0.3055 | 0.1804 | 1.69× |
| 967 | 5120 | 32 | 0.2576 | 0.1934 | 1.33× |
| 967 | 8192 | 32 | 0.3143 | 0.2113 | 1.49× |
| 967 | 8192 | 64 | 0.3331 | 0.1858 | 1.79× |
| 1595 | 5120 | 64 | 0.2894 | 0.1285 | 2.25× |
| 1595 | 8192 | 64 | 0.3357 | 0.1296 | 2.59× |

注意这些是**孤立单算子**计时，已包含 launch 间隙；真实模型里相邻 kernel 的 enqueue 会重叠，
按 profile 的 8.45 ms 上限折算，融合后预期省 **2–4 ms**（166 ms p95 上约 1.5–2.5%），
最终以服务级 S1 为准。

## 4. 服务级验证（install-v2）

配置：TP2 / BF16 / eager / 4096 / seqs1，`CLEF_GDN_CONV_QKV=1`、
`CLEF_CONV_QKV_VENDOR=<install-v2>`、`CLEF_CONV_QKV_TILE=32`，其余与最终提交一致
（`ACLNN_CACHE_LIMIT`、MLP NZ、chunk 64）。

- eligibility 日志（`/workspace/yangchen/logs/perf-next-20261117/conv-diag/service.log`）：

  ```
  clef-conv-qkv eligibility {..., 'prefills': 1, 'decodes': 0, 'tp': 2, 'conv_heads': 40,
    'k_heads': 16, 'v_heads': 48, 'dk': 128, 'dv': 128, 'bias_none': True, 'activation': 'silu',
    'x_shape': (967, 5120), 'x_stride': (8192, 1), 'x_contiguous': False, ...}
  ```

  即真实前向确实走 packed QKVZ 快路径，且 `as_strided` 展开没有回退。
- 固定样本（`conv-v2/fixed-{307,967,1595}.json`）各 50 次 measured 请求，
  与同日 eager 对照（`eager-control/fixed-*.json`）逐条比较 `answers`：
  **3 × 50 条 JSON 完全一致，数值字段最大绝对差 0**。
- eager 对照的全量 S1（12:20–12:24，host load 17.6）：
  mean **121.434** / p50 **113.735** / p95 **164.347** / p99 202.568 ms、**8.223 dec/s**，
  与官方入口历史值（121.705 / 113.656 / 166.002、8.213）一致，可作为同一台机器的对照基线。

## 5. 通用 QKV 归一化算子（独立 vendor）

`runtime/ascendc/clef_gdn_qkv_generic/`，算子名 `ClefGdnQkvGeneric`，vendor `clef_qkv_generic`：

- 输入连续 BF16 `x[H,T,128]`，H=40（TP2）或 H=80（TP1），HK=H/5、HV=3HK，输出 BSND `q/k/v`；
- 每 task 处理一个 head 的连续 token tile，`task = blockIdx; task += blockNum` 轮转，均摊 Q/K 与 V 成本；
- Q/K 用两条连续 mask `WholeReduceSum<float>(sum, square, 64, rows, 1, 1, 16)` + 一次 `Add`，
  srcRepStride=16×32B=512B、dstRepStride=1 float；V 用 `TQueBind<VECIN,VECOUT,1>` 走 MTE2→MTE3，无 FP32 cast；
- 36 组检查（2 × 6 长度 × 3 tile）：TP2 对部署 v2 **逐位一致**；TP1 对 FLA l2norm+BSND 最大绝对差 0.001953125
  （0.5 ulp 级，两组求和树不同所致，CPU FP32 参考同时记录）。

计时（NPU event 中位，ms）：

| tokens | TP2 部署 v2 | TP2 generic | 加速 | TP1 FLA | TP1 generic | 加速 |
| --- | --- | --- | --- | --- | --- | --- |
| 307 | 0.1522 | 0.1487 (t16) | 1.02× | 0.3513 | 0.1613 (t64) | 2.18× |
| 967 | 0.2000 | 0.1433 (t16) | 1.40× | 0.3489 | 0.1620 (t64) | 2.15× |
| 1595 | 0.2590 | 0.1384 (t16) | 1.87× | 0.3080 | 0.1782 (t64) | 1.73× |

它和 conv+QKV 融合是**互斥**的两条路径：融合走的是 conv 边界，generic 只在关闭 conv 融合
（或 token 超过 `CLEF_GDN_QKV_NORM_MAX_TOKENS`）时才有意义，因此当前优先级低于 conv 融合，
仅保留为 TP1 与回退候选。安装器 `runtime/apply_qkv_generic_patch.py` 只在
`CLEF_QKV_GENERIC=1` 时接管，并且可以从已验证的 QKV 或 Conv+QKV postimage 组合升级。

## 6. 复现

```bash
# 构建（远端）
source /workspace/yangchen/Clef-27B-ascend/scripts/activate_env.sh
cd runtime/ascendc/clef_gdn_conv_qkv && bash build.sh
./build_out/custom_opp_ubuntu_aarch64.run --noexec --extract=build_out/install-v3

# 离线探针（含 packed QKVZ）
python runtime/clef_conv_qkv_probe.py \
  --vendor "$PWD/build_out/install-v3/packages/vendors/clef_conv_qkv" \
  --key-heads 8 --lengths 1,2,3,4,31,32,33,307,967,1595 \
  --input-channels 5120,8192 --tiles 32,64 \
  --output /workspace/yangchen/logs/perf-next-20261117/conv-qkv-probe-v3.json.gz

# 宿主补丁
python runtime/apply_conv_qkv_patch.py --apply \
  --backup-dir /workspace/yangchen/logs/gdn-conv-qkv-20261117/phase0
# 服务环境
export CLEF_GDN_CONV_QKV=1
export CLEF_CONV_QKV_VENDOR=<install-v3>/packages/vendors/clef_conv_qkv
export CLEF_CONV_QKV_TILE=32
```

## 7. 未完成与下一步

1. install-v3（tile 归约）的服务级固定样本复核 —— 代码与 kernel 已就绪，
   远端 `build_out/install-v3` 已产出，只差重启服务与比对。
2. 全量 S1：host load 必须回到与 12:20 对照（17.6）可比或更低；当前 40+。
3. 采纳后跑 OpenDecisions 2,352 全量与 ERQA 400 题，确认 2PP 且逐位一致。
4. 每 token ≤1200 的 cap 与 `CLEF_GDN_QKV_NORM_MAX_TOKENS` 一致，超过则回退 BIC + QKV v2，
   长序列（1,595 / 10,294）仍走原路径。

## 8. 证据索引

| 内容 | 路径 |
| --- | --- |
| 探针 v1/v2/v3 | `Clef-27B/logs/perf-next-20261117/conv-qkv-probe-v{1,2,3}.json.gz` |
| 通用 QKV 探针 | `Clef-27B/logs/perf-next-20261117/qkv-generic-probe.json.gz` |
| chunk 扫描（未归一化修正后） | `Clef-27B/logs/perf-next-20261117/gdn-chunk-sweep.json.gz` |
| eager 对照全量 S1 | `Clef-27B/logs/perf-next-20261117/eager-s1-full.json.gz` |
| 服务安装记录 | `/workspace/yangchen/logs/perf-next-20261117/conv-install.json` |
| 服务固定样本 | `/workspace/yangchen/logs/perf-next-20261117/conv-v2/fixed-*.json` |
| eligibility 日志 | `/workspace/yangchen/logs/perf-next-20261117/conv-diag/service.log` |
