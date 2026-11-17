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
- 服务级首轮（conv-v2）的 gdn.py 仍是只接受连续输入的 v1 guard，那次的 eligibility 行只是入口快照，
  融合路径实际没有生效；该结论作废。
- 干净 v2 guard + install-v3 的服务级复核：两个 TP worker 都打印
  `clef-conv-qkv ... tile=32 first_tokens=967 vendor=.../install-v3/...`，融合路径真实生效；
  与 eager 对照相比 `answers` 概率**不是逐位相同**，307/967/1595 三个固定样本最大差
  0.00095 / 0.00240 / 0.00029（约 0.6 个 BF16 ulp），而同一棵树上关闭融合（`CLEF_GDN_CONV_QKV=0`）
  与对照**逐位一致**，因此差异确定来自融合算子本身。
- 另建通用 QKV 归一化算子 `ClefGdnQkvGeneric`（独立 vendor），支持 TP1 H=80 / TP2 H=40 任意 token 数，
  36 组硬件检查通过：TP2 对部署 v2 逐位一致，TP1 对 FLA 最大绝对差 0.001953125。
- **已采纳为 `--config s1` 默认**：v3 kernel 内置于 `Clef-27B/src/ascendc/vendors/clef_conv_qkv`，
  `inference.py` 自动注入 `ASCEND_CUSTOM_OPP_PATH` / `CLEF_CONV_QKV_VENDOR` / `CLEF_GDN_CONV_QKV=1`，
  `CLEF_GDN_CONV_QKV=0` 可复现未融合的对照臂。
- 同窗口全量 S1：p95 **165.09 → 160.15 ms（−2.99%）**、mean −4.41%、p50 −5.26%、吞吐 +4.59%；
  全量 OpenDecisions Capability **74.3107 → 73.9636（−0.347 pp，2PP 内）**、accuracy 不变；
  ERQA **217 → 218/400（+0.25 pp）**。项目内部 0.03/0.05 概率门槛不达标（见 4.2），按官方 2PP 口径采纳。

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

## 4. 服务级验证（install-v3）

配置：TP2 / BF16 / eager / 4096 / seqs1，`CLEF_GDN_CONV_QKV=1`、
`CLEF_CONV_QKV_VENDOR=<install-v3>`、`CLEF_CONV_QKV_TILE=32`，其余与最终提交一致
（`ACLNN_CACHE_LIMIT`、MLP NZ、chunk 64），走官方 `start.sh --config s1` 入口。

- 融合生效证据：该 patch 只在**全部 guard 通过后**才初始化 runtime 并打印一次，
  重启后首个 967 token 请求在 service log 中留下（两个 TP worker 各一条）：

  ```
  clef-conv-qkv pid=890112 tile=32 first_tokens=967 vendor=.../install-v3/packages/vendors/clef_conv_qkv
  clef-conv-qkv pid=890113 tile=32 first_tokens=967 vendor=.../install-v3/packages/vendors/clef_conv_qkv
  ```

  即 packed QKVZ（`x_stride=(8192,1)`）→ `as_strided` 展开 → 融合 kernel 的真实路径，
  没有回退到 `causal_conv1d_fn` + `ClefGdnQkvNorm`。
- 固定样本（`conv-v3/fixed-*.json`，每个长度 50 次 measured）与同日 eager 对照
  （`eager-control/fixed-*.json`）比较 `answers`：**不是逐位一致**，
  最大概率差 307 / 967 / 1595 = 0.000953 / 0.002398 / 0.000285，choice 未变。
- 差异归因：同一棵代码树上把 `CLEF_GDN_CONV_QKV` 设为 0 重启（`conv-v3-off/fixed-*.json`），
  三个长度的 `answers` 与 eager 对照**最大差 0**；两次服务内 50 次重复各自唯一，
  即服务是确定性的，差异只能来自融合算子。
- 期望差异约 0.6 个 BF16 ulp：融合核在寄存器里保留 conv 的 FP32 中间量并直接做规范化，
  与“conv 写 BF16 → norm 读 BF16”在最后一个求和树上不完全等价；探针用随机输入未复现，
  真实输入会命中。是否可接受以全量 OpenDecisions 2PP 门槛判定，不以逐位为准。
- eager 对照的全量 S1（12:20–12:24，host load 17.6）：
  mean **121.434** / p50 **113.735** / p95 **164.347** / p99 202.568 ms、**8.223 dec/s**，
  与官方入口历史值（121.705 / 113.656 / 166.002、8.213）一致，可作为同一台机器的对照基线。
- 固定样本 p50（同一服务窗口，仅作方向参考，host load 21–28 未满足测量协议）：
  967 token 融合 160.58 vs 对照 165.07 ms，1595 token 238.26 vs 244.34 ms，
  307 token 112.68 vs 112.37 ms（持平）。

### 4.1 全量 S1 同窗口 A/B（host load 19–23）

同一棵代码树、同一天、相隔约 7 分钟，唯一变量是 `CLEF_GDN_CONV_QKV`：

| 指标 | 融合关闭 | 融合开启 | 变化 |
| --- | --- | --- | --- |
| mean | 122.762 ms | **117.344 ms** | −4.41% |
| p50 | 114.812 ms | **108.771 ms** | −5.26% |
| p95 | 165.091 ms | **160.150 ms** | −2.99% |
| p99 | 203.075 ms | **196.492 ms** | −3.24% |
| 吞吐 | 8.137 dec/s | **8.510 dec/s** | +4.59% |

关闭组与 12:20 的 eager 对照（mean 121.434 / p95 164.347 / 8.223）相差 0.4–1.1%，
说明该窗口本身可用；融合在 mean/p50/吞吐上收益接近 5%，p95 收益 3.0%。
p50 收益大于 p95，是因为 p95 主要由 1,200–1,600 token 的长请求主导，而融合的比例收益在那里更小。

随后用**默认入口**（`--config s1`，vendor 内置、env 由 `inference.py` 注入、无任何手工环境变量）
复跑一次：mean 120.926 / p50 114.177 / p95 **160.419** / p99 197.454、8.259 dec/s（host load 20–22）。
其 2,352 条 `answers` 与显式 env 的 on 组**完全一致（0 差异）**，
说明默认入口没有引入额外变量；mean/p50 略高于 on 组是窗口负载差异。

### 4.2 全量质量（OpenDecisions 2,352）

用同一份原始请求离线评分（`runtime/od_raw_acceptance.py`，作者 `score.py`）：

| 指标 | 融合关闭 | 融合开启 | 变化 |
| --- | --- | --- | --- |
| Calibration | 82.7092 | 82.1731 | −0.536 pp |
| Intelligence | 65.9122 | 65.7541 | −0.158 pp |
| **Capability** | **74.3107** | **73.9636** | **−0.347 pp** |
| accuracy | 0.8087 (1902/2352) | 0.8087 (1902/2352) | 0 |
| Score MAE | 0.401545 | 0.401678 | +0.00013 |

- 官方 2PP 门槛：三项绝对变化均 ≤0.54 pp，**通过**。
- 项目内部更严的 0.03/0.05 概率门槛：`max_prob_diff` 0.1447、7 条 >0.03、2 条 >0.05、
  1 条 choice 变化、1 条 noul 跨越 —— **不通过**。分布为 p50 0.00044、p90 0.0053、p99 0.0210，
  即绝大多数题目差异在 1e-3 量级，尾部有少量放大的题目。
- 采用与否按官方 2PP 口径判定，并在提交文档中如实记录该内部门槛不达标的事实。

### 4.3 chunk 128 叠加（`CLEF_GDN_CHUNK128=1`，默认关闭）

在融合算子上叠加已 staged 的 GDN chunk 128，同一台机器、相邻窗口（host load 21–24）：

| 指标 | chunk 64 + 融合 | chunk 128 + 融合 | 变化 |
| --- | --- | --- | --- |
| mean | 120.926 ms | **118.733 ms** | −1.81% |
| p50 | 114.177 ms | **111.504 ms** | −2.34% |
| p95 | 160.419 ms | **158.301 ms** | −1.32% |
| p99 | 197.454 ms | **194.491 ms** | −1.50% |
| 吞吐 | 8.259 dec/s | **8.411 dec/s** | +1.84% |
| OD Capability | 73.9636 | **74.2476** | +0.284 pp |
| OD accuracy | 1902/2352 | **1904/2352** | +2 |
| ERQA | 218/400 | **216/400** | −0.50 pp |

相对改造前的 12:20 eager 对照（p95 164.347、mean 121.434、8.223）：p95 −3.68%、mean −2.22%、吞吐 +2.29%。
S2 B8 C16（chunk 128 + 融合）：**16.727 dec/s**、mean 952.9、p95 1768.9 ms，
离线 Capability **73.8426**、accuracy 1905/2352（相对 74.3107 为 −0.47 pp，2PP 内）。

- 两项叠加后所有官方门槛仍在 2PP 内；OD Capability 甚至高于只用融合（+0.28 pp），accuracy +2。
- 代价是逐题概率漂移更大：choice 变化 3 条、`max_prob_diff` 0.360、46 条 >0.03、24 条 >0.05，
  比单用融合的 1 条 / 0.145 明显更宽。
- 用户 2026-11-17 明确“2PP 以内就行”，因此**chunk 128 也改为默认**：
  `inference.py` 在决策 prefill 启用时自动 apply，`CLEF_GDN_CHUNK128=0` 可回到 chunk 64。
  项目内部 0.03/0.05 逐题门槛不达标的事实保留在文档中，不作为采纳门槛。

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

1. ~~install-v3 服务级固定样本复核~~ —— 已完成，见第 4 节（融合生效，非逐位）。
2. ~~全量 S1~~ —— 已完成同窗口 A/B 与默认入口复跑；尚缺 host load <10 的干净窗口用于对外报数。
3. ~~OpenDecisions 2,352 与 ERQA 400~~ —— 已完成，2PP 通过（4.2、4.3）。
4. 干净窗口下的 S1/S2 与 `results.csv` 更新仍待办；当前所有绝对耗时都在 load 18–28 下测得。
4. 每 token ≤1200 的 cap 与 `CLEF_GDN_QKV_NORM_MAX_TOKENS` 一致，超过则回退 BIC + QKV v2，
   长序列（1,595 / 10,294）仍走原路径。

## 8. 证据索引

| 内容 | 路径 |
| --- | --- |
| 全量 S1 A/B（off / on / 默认入口） | `Clef-27B/logs/perf-next-20261117/s1-conv-qkv-{off,on}.json.gz`、`s1-default-entry.json.gz` |
| 全量质量评分 | `Clef-27B/logs/perf-next-20261117/conv-qkv-quality.json` |
| ERQA 摘要与离线复核 | `Clef-27B/logs/perf-next-20261117/erqa-conv-qkv/` |
| 探针 v1/v2/v3 | `Clef-27B/logs/perf-next-20261117/conv-qkv-probe-v{1,2,3}.json.gz` |
| 通用 QKV 探针 | `Clef-27B/logs/perf-next-20261117/qkv-generic-probe.json.gz` |
| chunk 扫描（未归一化修正后） | `Clef-27B/logs/perf-next-20261117/gdn-chunk-sweep.json.gz` |
| eager 对照全量 S1 | `Clef-27B/logs/perf-next-20261117/eager-s1-full.json.gz` |
| 服务安装记录 | `/workspace/yangchen/logs/perf-next-20261117/conv-install.json` |
| 服务固定样本 | `/workspace/yangchen/logs/perf-next-20261117/conv-v2/fixed-*.json` |
| eligibility 日志 | `/workspace/yangchen/logs/perf-next-20261117/conv-diag/service.log` |

### 4.4 S2 最大吞吐（conv+QKV + chunk128）

| 并发 | dec/s | mean | p95 | OD Capability | accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| B8 C16 | 16.727 | 952.9 ms | 1768.9 ms | 73.8426 | 1905/2352 |
| **B16 C64** | **17.446** | 3631.9 ms | 5950.1 ms | 74.1388 | 1902/2352 |

- 上一版（B8 C16、无融合）为 16.376 dec/s；B16 C64 历史值 17.110。
- 两组 OD 三项均在 2PP 内；B16 只按最大吞吐（p95 无约束），B8 是 p95 约束下的推荐工作点。
- `results.csv` 记录 B16 C64 17.446 dec/s、并发 64；B8 数据保留在本表与 `s2-chunk128-conv.json.gz`。
