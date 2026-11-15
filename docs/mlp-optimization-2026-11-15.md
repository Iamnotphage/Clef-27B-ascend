# MLP 优化阶段报告 · 2026-11-15

## 阶段结果

真实激活的 W8A8_DYNAMIC 完整 MLP 链完成 12 点验证，设备事件配对加速比为 **1.354–1.633×**，同步 wall 配对加速比为 **1.284–1.604×**。每点六轮均获得局部加速，输出全部有限。相对 RMSE 为 **1.06%–14.37%**，gate/up 候选的完整 OD2352 与 ERQA400 结果见第8节。 [W8A8 运行清单](../Clef-27B/logs/mlp-20261115/w8a8-nz-v2/driver.json)

当前 `npu_ffn` SwiGLU 接口接受 FP16。四个真实 layer0 输入的 BF16→FP16→FFN→BF16 完整局部链耗时为 BF16 分解实现的 **1.777–2.573 倍**，各点六轮方向一致。后续优先评估 W8A8 的量化范围与完整模型质量。 [FFN 运行清单](../Clef-27B/logs/mlp-20261115/ffn-v1/driver.json)

选择性量化补充 12 点四臂实验。`gate_up_only` 的 device 配对收益为 **1.205–1.376×**，wall 为 **1.176–1.371×**，首个完整模型候选采用全部 64 层 gate/up W8A8。layer31 局部 relative RMSE 为 10.90%–12.50%，任务质量以完整 OD/ERQA 评测继续确认。 [四臂运行清单](../Clef-27B/logs/mlp-20261115/w8a8-partial-v1/driver.json)

本阶段测量输出为 TP2 的单 rank、单层 down projection 部分和，all-reduce 位于测量终点之后。完整请求的调度、注意力、GDN、通信、视觉编码与联合头成本纳入第8节服务实测。当前倍数用于评价该局部计算链。

## 1. 输入、模型和复现范围

捕获通过真实服务请求取得 layers 0、31、63 的 MLP 输入，每层覆盖 rank0、rank1，四条请求共 24 张量。每个张量均为 BF16、`[T,5120]`、stride `[5120,1]`。四条响应的 `answers` 与冻结参考逐位一致，实际输入 token 数与参考一致。离线校验确认身份、shape、dtype、hidden size、finite、token 数和 SHA 全部通过。 [请求与逐张量来源](../Clef-27B/logs/mlp-20261115/capture-requests.json) · [24 张量离线校验](../Clef-27B/logs/mlp-20261115/capture-offline-validation.json)

| 输入 | 任务来源 | 实际 T | 捕获层 | ranks | 张量数 |
| --- | --- | ---: | --- | --- | ---: |
| `od-0105` | OD 文本 | 307 | 0、31、63 | 0、1 | 6 |
| `od-0677` | OD 最长文本 | 1595 | 0、31、63 | 0、1 | 6 |
| `erqa-0089` | ERQA 4 图 | 662 | 0、31、63 | 0、1 | 6 |
| `erqa-0167` | ERQA 16 图 | 10294 | 0、31、63 | 0、1 | 6 |

本轮微基准使用上述 rank0 的完整 T 行，W8A8 覆盖三层共 12 点，FFN 覆盖 layer0 共 4 点。307-token 输入来自 OD `od-0105`。四图输入 `erqa-0089` 实际为 662 tokens，十六图输入 `erqa-0167` 为 10294 tokens。模型维度为 H=5120、完整 I=17408、TP2 局部 I=8704。输入张量文件保留在远端 run 的 `capture/`，仓库归档其元数据及 SHA。

运行环境实际记录为 `Ascend910_9382`、`torch 2.10.0+cpu`、`torch_npu 2.10.0.post7`。插件源码实际 commit 与固定 `89eb8a5e7a76d085b453a1aeadb6883e12d5ea2d` 一致。W8A8 每份 JSON 记录原始权重 key、切片轴、层/rank、权重和 scale 哈希、输入文件哈希及原始计时。远端 run 为 `/workspace/yangchen/logs/mlp-20261115-1027`。 [环境与完整 provenance 示例](../Clef-27B/logs/mlp-20261115/w8a8-nz-v2/od-0105-l00-r0.json)

## 2. W8A8_DYNAMIC 的实现和计时

BF16 链采用 `F.linear → npu_swiglu → F.linear`。W8A8 链采用 `npu_dynamic_quant → npu_quant_matmul(BF16) → npu_swiglu → npu_dynamic_quant → npu_quant_matmul(BF16)`，两次 activation quantization 均计入完整链。权重与输入预先加载到设备，权重转换发生在预热之前。

权重使用对称逐输出通道 absmax/127。scale 先保存为 BF16，再提升为 FP32 参与除法与 ties-to-even rounding，整数截断到 [-127,127]。down 的 scale 沿完整 checkpoint K=17408 计算，随后切 TP 输入列。gate/up 按各自输出行切片并以 `[gate,up]` 拼接。

BF16 权重实际为 ND、format 2。INT8 权重先转置为 `[K,N]`，启用 `torch.npu.config.allow_internal_format=True` 后转换为 FRACTAL_NZ、format 29，运行前逐权重检查格式。gate_up INT8 为 `[5120,17408]`，down INT8 为 `[8704,5120]`。 [本轮冻结探针](../Clef-27B/logs/mlp-20261115/w8a8-nz-v2/probe_mlp_w8a8.py)

每臂预热 10 次，六轮中 BF16/W8A8 先后交替，每臂每轮正式执行 30 次，共 180 个样本。每次执行前同步设备，device 使用包围完整链的 NPU event，wall 使用包含 event 记录、Python 下发和末尾同步的 `perf_counter`。下表耗时为 180 次中位数，加速比为六个同轮中位数之比的中位数。每点 device 与 wall 的加速轮数均为 6/6。

### 12 点 NZ 性能矩阵

| 输入 | T | 层 | device BF16 / W8A8 ms | device 配对倍数 | wall BF16 / W8A8 ms | wall 配对倍数 | 记录 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `od-0105` | 307 | 0 | 0.502630 / 0.367130 | 1.370× | 0.589810 / 0.453195 | 1.302× | [JSON](../Clef-27B/logs/mlp-20261115/w8a8-nz-v2/od-0105-l00-r0.json) |
| `od-0105` | 307 | 31 | 0.512330 / 0.369060 | 1.382× | 0.593620 / 0.447335 | 1.326× | [JSON](../Clef-27B/logs/mlp-20261115/w8a8-nz-v2/od-0105-l31-r0.json) |
| `od-0105` | 307 | 63 | 0.501050 / 0.369990 | 1.354× | 0.587900 / 0.458365 | 1.284× | [JSON](../Clef-27B/logs/mlp-20261115/w8a8-nz-v2/od-0105-l63-r0.json) |
| `od-0677` | 1595 | 0 | 1.536740 / 0.941110 | 1.633× | 1.603890 / 1.007790 | 1.590× | [JSON](../Clef-27B/logs/mlp-20261115/w8a8-nz-v2/od-0677-l00-r0.json) |
| `od-0677` | 1595 | 31 | 1.531080 / 0.956560 | 1.601× | 1.618755 / 1.043955 | 1.550× | [JSON](../Clef-27B/logs/mlp-20261115/w8a8-nz-v2/od-0677-l31-r0.json) |
| `od-0677` | 1595 | 63 | 1.541800 / 0.979350 | 1.575× | 1.633385 / 1.070295 | 1.526× | [JSON](../Clef-27B/logs/mlp-20261115/w8a8-nz-v2/od-0677-l63-r0.json) |
| `erqa-0089` | 662 | 0 | 0.712880 / 0.517730 | 1.376× | 0.801020 / 0.604145 | 1.328× | [JSON](../Clef-27B/logs/mlp-20261115/w8a8-nz-v2/erqa-0089-l00-r0.json) |
| `erqa-0089` | 662 | 31 | 0.714300 / 0.518890 | 1.379× | 0.801405 / 0.604315 | 1.326× | [JSON](../Clef-27B/logs/mlp-20261115/w8a8-nz-v2/erqa-0089-l31-r0.json) |
| `erqa-0089` | 662 | 63 | 0.720830 / 0.512600 | 1.407× | 0.798115 / 0.591995 | 1.348× | [JSON](../Clef-27B/logs/mlp-20261115/w8a8-nz-v2/erqa-0089-l63-r0.json) |
| `erqa-0167` | 10294 | 0 | 9.115330 / 5.670310 | 1.608× | 9.204155 / 5.759325 | 1.599× | [JSON](../Clef-27B/logs/mlp-20261115/w8a8-nz-v2/erqa-0167-l00-r0.json) |
| `erqa-0167` | 10294 | 31 | 9.124790 / 5.683970 | 1.606× | 9.245365 / 5.803940 | 1.594× | [JSON](../Clef-27B/logs/mlp-20261115/w8a8-nz-v2/erqa-0167-l31-r0.json) |
| `erqa-0167` | 10294 | 63 | 9.163910 / 5.676400 | 1.614× | 9.260410 / 5.777060 | 1.604× | [JSON](../Clef-27B/logs/mlp-20261115/w8a8-nz-v2/erqa-0167-l63-r0.json) |

### 同输入局部数值差异

下表比较 BF16 与 W8A8 的本 rank 部分和，relative RMSE 按 BF16 输出 RMS 归一化。每点两端输出均为有限值。layer63 的最大绝对差同时结合相对 RMSE 与 cosine 解读，gate/up 候选完整网络的决策变化和校准分数见第8节。

| 输入 | 层 | max absolute error | RMSE | relative RMSE | cosine |
| --- | ---: | ---: | ---: | ---: | ---: |
| `od-0105` | 0 | 0.238281 | 0.00125754 | 8.7999% | 0.99613915 |
| `od-0105` | 31 | 0.250000 | 0.01834252 | 12.6728% | 0.99206576 |
| `od-0105` | 63 | 24.000000 | 0.52744566 | 7.2614% | 0.99736418 |
| `od-0677` | 0 | 0.281250 | 0.00106001 | 8.6053% | 0.99631740 |
| `od-0677` | 31 | 0.406250 | 0.01826812 | 13.2507% | 0.99132374 |
| `od-0677` | 63 | 24.000000 | 0.25385222 | 7.2220% | 0.99738870 |
| `erqa-0089` | 0 | 0.250000 | 0.00265675 | 1.0644% | 0.99994353 |
| `erqa-0089` | 31 | 0.304688 | 0.01871376 | 11.7905% | 0.99311092 |
| `erqa-0089` | 63 | 40.000000 | 0.47232230 | 8.1566% | 0.99666823 |
| `erqa-0167` | 0 | 0.750000 | 0.00318592 | 1.2618% | 0.99992073 |
| `erqa-0167` | 31 | 0.675781 | 0.01856192 | 12.8845% | 0.99178566 |
| `erqa-0167` | 63 | 24.000000 | 0.37881663 | 14.3675% | 0.98974051 |

三个采样层的误差随输入和层次变化。layer31 的四个输入 relative RMSE 均超过 11%，第 4 节进一步测量 gate/up 与 down 的独立量化。候选排序同时使用任务分数、推理吞吐和延迟。

### 初轮 ND 结果与修正

首轮 `w8a8-v1` 漏设 `allow_internal_format`，调用格式转换后两份 INT8 权重实际仍记录为 format 2。随后在初始化中启用内部格式，并添加 format 29 强校验，重新完成全部 12 点 `w8a8-nz-v2`。本报告矩阵取自这轮 NZ 复测。首轮原始 JSON、日志和 driver 保留在远端 `/workspace/yangchen/logs/mlp-20261115-1027/w8a8-v1`，用途为追溯格式修正。

### 分配器范围

探针同时常驻 BF16 与 INT8 两组权重及 scale，逻辑权重总量为 382.543 MiB。以下记录为该进程暖态计时期间的 Torch 分配器峰值，反映双臂实验的常驻对象和临时分配。完整服务的权重容量、缓存池及独立 workspace 峰值由服务验收单独采集。

| 输入 | allocated 基线 MiB | allocated 峰值 MiB | 峰值增量 MiB | reserved MiB |
| --- | ---: | ---: | ---: | ---: |
| `od-0105` | 388.542 | 433.384 | 44.842 | 456.000 |
| `od-0677` | 401.121 | 539.623 | 138.502 | 592.000 |
| `erqa-0089` | 392.009 | 462.201 | 70.191 | 496.000 |
| `erqa-0167` | 486.072 | 1256.915 | 770.843 | 1440.000 |

## 3. FFN 接口、gate 顺序和四点实测

接口测试对 BF16 返回错误码 161002，运行时消息为 `x only support dtype float16 without expert tokens, but found DT_BFLOAT16`。FP16、无 expert tokens、SwiGLU 的调用成功。实现据此测量 FP16 融合路径，并补齐 BF16 输入转 FP16 与输出转回 BF16 的完整链。 [接口实际结果](../Clef-27B/logs/mlp-20261115/ffn-interface-v2.json)

独立 gate 顺序探针确认 `first_half * silu(second_half)`，relative RMSE 为 0.00041199。将 SiLU 放在第一半时 relative RMSE 为 0.82082858。因此融合权重采用 `[up,gate]`，分解链沿用 `[gate,up]`。四点正式测试全部记录 `gate_order=second_half` 与上述 packing。 [本轮 FFN 冻结实现](../Clef-27B/logs/mlp-20261115/ffn-v1/probe_mlp_ffn.py)

所有 FFN 权重实际为 ND、format 2，内部格式开关为 enable。分解实现权重采用 `[out,in]`，融合实现采用连续 `[in,out]`。每臂每轮预热 10 次后连续执行 30 次，六轮按正序与逆序交替。下表 device 为六个轮内平均耗时的中位数，wall 为六个轮内同步总耗时/30 的中位数。W8A8 探针逐次同步，FFN 探针按轮同步，两种探针各自的同轮 BF16 对照用于计算收益。

### device 耗时与完整转换链

| 输入 | T | BF16 分解 ms | FP16 分解 ms | FP16 FFN ms | 含转换分解 ms | 含转换 FFN ms | BF16 / 含转换 FFN | 记录 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `od-0105` | 307 | 0.497856 | 0.499122 | 0.888090 | 0.509533 | 0.884498 | 0.563× | [JSON](../Clef-27B/logs/mlp-20261115/ffn-v1/od-0105-l00-r0.json) |
| `od-0677` | 1595 | 1.535140 | 1.523952 | 3.425113 | 1.559718 | 3.519922 | 0.436× | [JSON](../Clef-27B/logs/mlp-20261115/ffn-v1/od-0677-l00-r0.json) |
| `erqa-0089` | 662 | 0.709370 | 0.710018 | 1.682781 | 0.717163 | 1.621419 | 0.437× | [JSON](../Clef-27B/logs/mlp-20261115/ffn-v1/erqa-0089-l00-r0.json) |
| `erqa-0167` | 10294 | 9.167607 | 9.182608 | 23.198121 | 9.477469 | 23.588519 | 0.389× | [JSON](../Clef-27B/logs/mlp-20261115/ffn-v1/erqa-0167-l00-r0.json) |

| 输入 | wall BF16 分解 ms | wall FP16 FFN ms | wall 含转换 FFN ms |
| --- | ---: | ---: | ---: |
| `od-0105` | 0.501129 | 0.891131 | 0.887962 |
| `od-0677` | 1.538819 | 3.428907 | 3.523841 |
| `erqa-0089` | 0.712825 | 1.686286 | 1.625294 |
| `erqa-0167` | 9.174905 | 23.202780 | 23.594004 |

`fused_io_cast_overhead_ms` 是两条独立计时路径的中位数之差，307 与 662 tokens 下分别为 -0.003592、-0.061363 ms。该差值保留测量波动和路径差异，转换路径的判断依据为上表完整链耗时。

### FFN 局部精度

| 输入 | FP16 FFN 对 FP16 分解 relative RMSE | 含转换 FFN 对 BF16 relative RMSE | 含转换 max absolute error | BF16 对照 allclose |
| --- | ---: | ---: | ---: | --- |
| `od-0105` | 0.00003531 | 0.00285475 | 0.015625 | true |
| `od-0677` | 0.00000000 | 0.00277449 | 0.015625 | true |
| `erqa-0089` | 0.00001867 | 0.00148171 | 0.125000 | false |
| `erqa-0167` | 0.00000000 | 0.00160726 | 0.250000 | false |

四点均为有限输出。表中 allclose 使用 atol=0.001、rtol=0.01，两条 ERQA 输入记录为 false。探针另记录 relative RMSE 与归一化最大误差诊断，完整模型任务质量仍待候选服务评测。

## 4. gate/up 与 down 选择性量化

补充实验在同一真实输入与权重上同时比较四臂，`bf16` 为完整 BF16 链，`both` 量化 gate/up 和 down，`gate_up_only` 量化合并 gate/up 投影，`down_only` 量化 down 投影。局部链保留完整 SwiGLU 与对应 activation quantization，INT8 格式全部为 NZ29。 [运行清单](../Clef-27B/logs/mlp-20261115/w8a8-partial-v1/driver.json)

本轮覆盖四输入×三层共 12 点，每臂预热 10 次，八轮各 30 次，共 240 个正式样本。八轮内每臂在每个执行位置恰好出现两次，任意两臂的先后次序各出现四次。全部 36 个候选点输出有限，device 与 wall 均为 8/8 轮加速。耗时列为 240 样本中位数，配对倍数为八个同轮中位数之比的中位数。

### 四臂 device 矩阵

| 输入 | 层 | BF16 ms | both ms | gate_up_only ms | down_only ms | 记录 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `od-0105` | 0 | 0.505380 | 0.363650 | 0.389040 | 0.466880 | [JSON](../Clef-27B/logs/mlp-20261115/w8a8-partial-v1/od-0105-l00-r0.json) |
| `od-0105` | 31 | 0.503020 | 0.357390 | 0.385650 | 0.453130 | [JSON](../Clef-27B/logs/mlp-20261115/w8a8-partial-v1/od-0105-l31-r0.json) |
| `od-0105` | 63 | 0.501730 | 0.370690 | 0.395640 | 0.464970 | [JSON](../Clef-27B/logs/mlp-20261115/w8a8-partial-v1/od-0105-l63-r0.json) |
| `od-0677` | 0 | 1.540070 | 0.970060 | 1.150140 | 1.380000 | [JSON](../Clef-27B/logs/mlp-20261115/w8a8-partial-v1/od-0677-l00-r0.json) |
| `od-0677` | 31 | 1.539010 | 0.974020 | 1.152250 | 1.380920 | [JSON](../Clef-27B/logs/mlp-20261115/w8a8-partial-v1/od-0677-l31-r0.json) |
| `od-0677` | 63 | 1.532240 | 0.968090 | 1.149060 | 1.373780 | [JSON](../Clef-27B/logs/mlp-20261115/w8a8-partial-v1/od-0677-l63-r0.json) |
| `erqa-0089` | 0 | 0.719680 | 0.502630 | 0.583140 | 0.646930 | [JSON](../Clef-27B/logs/mlp-20261115/w8a8-partial-v1/erqa-0089-l00-r0.json) |
| `erqa-0089` | 31 | 0.723590 | 0.506650 | 0.593660 | 0.656140 | [JSON](../Clef-27B/logs/mlp-20261115/w8a8-partial-v1/erqa-0089-l31-r0.json) |
| `erqa-0089` | 63 | 0.714620 | 0.493830 | 0.591790 | 0.639340 | [JSON](../Clef-27B/logs/mlp-20261115/w8a8-partial-v1/erqa-0089-l63-r0.json) |
| `erqa-0167` | 0 | 9.175730 | 5.683810 | 6.670270 | 8.173090 | [JSON](../Clef-27B/logs/mlp-20261115/w8a8-partial-v1/erqa-0167-l00-r0.json) |
| `erqa-0167` | 31 | 9.140360 | 5.694070 | 6.655120 | 8.166440 | [JSON](../Clef-27B/logs/mlp-20261115/w8a8-partial-v1/erqa-0167-l31-r0.json) |
| `erqa-0167` | 63 | 9.089120 | 5.672210 | 6.602610 | 8.160370 | [JSON](../Clef-27B/logs/mlp-20261115/w8a8-partial-v1/erqa-0167-l63-r0.json) |

### device / wall 配对加速比

| 输入 | 层 | both | gate_up_only | down_only |
| --- | ---: | ---: | ---: | ---: |
| `od-0105` | 0 | 1.401× / 1.330× | 1.307× / 1.252× | 1.090× / 1.075× |
| `od-0105` | 31 | 1.408× / 1.374× | 1.305× / 1.254× | 1.111× / 1.100× |
| `od-0105` | 63 | 1.356× / 1.283× | 1.278× / 1.214× | 1.076× / 1.063× |
| `od-0677` | 0 | 1.585× / 1.539× | 1.338× / 1.317× | 1.115× / 1.108× |
| `od-0677` | 31 | 1.579× / 1.529× | 1.334× / 1.309× | 1.113× / 1.106× |
| `od-0677` | 63 | 1.583× / 1.539× | 1.335× / 1.307× | 1.115× / 1.106× |
| `erqa-0089` | 0 | 1.430× / 1.385× | 1.239× / 1.210× | 1.113× / 1.101× |
| `erqa-0089` | 31 | 1.429× / 1.326× | 1.221× / 1.176× | 1.104× / 1.100× |
| `erqa-0089` | 63 | 1.451× / 1.376× | 1.205× / 1.182× | 1.116× / 1.103× |
| `erqa-0167` | 0 | 1.614× / 1.605× | 1.375× / 1.371× | 1.123× / 1.121× |
| `erqa-0167` | 31 | 1.604× / 1.596× | 1.373× / 1.368× | 1.119× / 1.117× |
| `erqa-0167` | 63 | 1.602× / 1.591× | 1.376× / 1.370× | 1.114× / 1.112× |

### 局部 relative RMSE

| 输入 | 层 | both | gate_up_only | down_only |
| --- | ---: | ---: | ---: | ---: |
| `od-0105` | 0 | 8.7999% | 1.0845% | 8.7753% |
| `od-0105` | 31 | 12.6728% | 11.8085% | 4.5792% |
| `od-0105` | 63 | 7.2614% | 1.0183% | 7.2045% |
| `od-0677` | 0 | 8.6053% | 0.9917% | 8.5424% |
| `od-0677` | 31 | 13.2507% | 12.5049% | 4.3666% |
| `od-0677` | 63 | 7.2220% | 1.9043% | 6.9694% |
| `erqa-0089` | 0 | 1.0644% | 0.4115% | 1.0024% |
| `erqa-0089` | 31 | 11.7905% | 10.9024% | 4.4937% |
| `erqa-0089` | 63 | 8.1566% | 1.3218% | 8.0392% |
| `erqa-0167` | 0 | 1.2618% | 0.3959% | 1.2137% |
| `erqa-0167` | 31 | 12.8845% | 12.0006% | 4.6908% |
| `erqa-0167` | 63 | 14.3675% | 2.4248% | 14.0911% |

`gate_up_only` 的 device 配对收益为 1.205–1.376×，wall 为 1.176–1.371×。layer0/63 的 relative RMSE 为 0.396%–2.425%，layer31 为 10.902%–12.505%。据此首个完整模型候选采用 gate/up 量化，并通过 OD2352、ERQA400 衡量累计质量变化。

layer31 的 gate/up 误差与其他采样层存在明显差异，后续可按任务得分变化定位需要保留 BF16 的层。完整 64 层的选择需要完整模型评测与更细粒度层诊断共同决定。

## 5. 调优优先级和后续验收

1. 首个完整模型候选采用全部 64 层 gate/up W8A8，先按原生任务分数评价累计质量变化。根据得分和逐题差异，再检验部分层保留 BF16 的方案。
2. 接入完整模型后运行 OD2352，重算作者原生 Calibration、Intelligence、Capability，并记录 choice 变化、noul 跨 0.5、score 期望差与概率差分布。原逐概率 0.03、score 0.05 保留为诊断统计。
3. 运行 ERQA400，按官方 exact-match 统计 accuracy、正确转错误与错误转正确，核对固定请求结构哈希、样本集合和 token 对齐。
4. 对可接受的候选采集同输入服务 A/B，记录实际吞吐、p50/p95/p99、错误率和显存。完成后恢复基线服务并重新执行协议、多模态及模型身份验收。

正常入口验证阶段完成模型转换、原生量化加载和 OD18/ERQA19 冒烟。完整 OD2352/ERQA400 评测、已完成的性能窗口与暂停位置见第8节。

## 6. 复现与归档

采集 manifest、离线校验、12 点双臂 W8A8、12 点四臂 W8A8 与 4 点 FFN 的完整 JSON、日志和 driver 均已归档。执行时探针源码按轮保存。driver 中记录逐点 argv、开始时间、return code 与探针 SHA。下列命令复用已验证的捕获，输出目录每次使用新路径。

```bash
cd /workspace/yangchen/Clef-27B-ascend
source scripts/activate_env.sh
mlp_recheck_dir="/workspace/yangchen/logs/mlp-recheck-$(date -u +%Y%m%d-%H%M%S)"
python runtime/run_mlp_microbench.py \
  --captures /workspace/yangchen/logs/mlp-20261115-1027/capture-requests.json \
  --model-dir /workspace/yangchen/models/Clef-27B \
  --kind w8a8 --layers 0 31 63 --rank 0 --device 0 \
  --output-dir "$mlp_recheck_dir/w8a8"
python runtime/run_mlp_microbench.py \
  --captures /workspace/yangchen/logs/mlp-20261115-1027/capture-requests.json \
  --model-dir /workspace/yangchen/models/Clef-27B \
  --kind ffn --layers 0 --rank 0 --device 0 \
  --output-dir "$mlp_recheck_dir/ffn"
python runtime/run_mlp_microbench.py \
  --captures /workspace/yangchen/logs/mlp-20261115-1027/capture-requests.json \
  --model-dir /workspace/yangchen/models/Clef-27B \
  --kind partial_w8a8 --layers 0 31 63 --rank 0 --device 0 \
  --output-dir "$mlp_recheck_dir/partial-w8a8"
```

本报告表格直接从归档 JSON 生成。复核覆盖双臂 W8A8 的 180 样本中位数与六轮配对倍数、四臂 W8A8 的 240 样本中位数与八轮配对倍数、位置及先后均衡、NZ 格式、完整输入 SHA、driver 与冻结源码 SHA，以及 FFN 六轮计时统计。

| 归档对象 | SHA-256 |
| --- | --- |
| [capture-requests.json](../Clef-27B/logs/mlp-20261115/capture-requests.json) | `dee7b915dccbd55b920fbad6dd00c6028e394c8a835908b35ee9de793e82b02c` |
| [probe_mlp_w8a8.py](../Clef-27B/logs/mlp-20261115/w8a8-nz-v2/probe_mlp_w8a8.py) | `b6372bf9d7a503893157874ce62e854abf20b443f858882d7e250468c7c2cddf` |
| [probe_mlp_ffn.py](../Clef-27B/logs/mlp-20261115/ffn-v1/probe_mlp_ffn.py) | `659b6e24af1e7b6b4979e9e3c1f20ed4f83d052733cb02690dd5d878b923ca8e` |
| [partial/probe_mlp_partial_w8a8.py](../Clef-27B/logs/mlp-20261115/w8a8-partial-v1/probe_mlp_partial_w8a8.py) | `7458b57d3b0b7adef4430830d230590e8720bfd3d6952876db0cd4fcee13a581` |
| [partial/probe_mlp_w8a8.py](../Clef-27B/logs/mlp-20261115/w8a8-partial-v1/probe_mlp_w8a8.py) | `8b2224bc91cf6e63206ec36f31ba7966a0f670f9029fc5d7ab343519b92dc17e` |

## 7. 混合模型转换与正常入口

全 64 层 gate/up 转换已完成，128 份原始权重转为 INT8，增加逐输出通道 BF16 scale 与零 offset。主干索引共 1440 个张量、43,313,863,136 字节，目录为 `/workspace/yangchen/models/Clef-27B-mlp-gateup-w8a8`。CPU 独立复核确认 layers 0/31/63 的两 rank 共 12 个分片与重新量化结果一致，rank0 的权重及 scale 哈希与微基准一致。1056 个保留主干张量逐项字节 SHA 一致，覆盖全部 down、attention、vision，联合头整文件 SHA 一致。[转换清单](../Clef-27B/logs/mlp-20261115/conversion-manifest.json) · [独立核验](../Clef-27B/logs/mlp-20261115/model-conversion-check.json)

正常启动测试发现 `inference.py` 将未知的双 token 透传参数拼成了一个 argv 元素，导致图像处理器覆盖参数被 CLI 拒绝。修复保留参数与值的 token 边界，将 `mm_processor_kwargs` 纳入有效配置和来源记录，并在最终 argv 输出一次。10 项新增启动器回归与 34 项原有 GDN 检查通过。正常 `start.sh` 入口已启动 BF16 对照服务，TP2、seqs1、16384、16 图、像素 65,536～1,048,576，GDN 决策 prefill 开启，冻结回放 8/8 逐位一致。[入口验证](../Clef-27B/logs/mlp-20261115/launch-validation.json) · [实际 manifest](../Clef-27B/logs/mlp-20261115/bf16-entry-manifest.json) · [冻结回放](../Clef-27B/logs/mlp-20261115/bf16-entry-frozen8.json)

候选通过正常入口显式指定 `--quantization ascend`，服务启动 PID 为 517359。加载审计确认每个 rank 的 64 个 gate/up INT8 参数合计 5,704,253,440 个元素，联合头 122 个参数全部加载。首次启动额外启用严格跟踪后，两个 rank 均报告 48 个 `ascend_conv1d_weight` 派生缓存缺失。这些缓存由已加载的 `conv1d.weight` 在权重后处理阶段转置生成，跟踪检查发生在该阶段之前。随后沿用原生量化加载默认值，实际审计记录为 `strict_tracking_enabled=false`。首次检查完整缺失集合与成功启动日志分别保存在远端 run 的 `gateup-entry/service.log` 和 `gateup-entry-v2/service.log`。

候选 OD 冒烟完成 18/18，ERQA 冒烟完成 19/19，均无请求失败。ERQA 子集与 BF16 对照的请求结构哈希及输入 token 全部对齐，正确数从 7/19 变为 6/19，包含 1 条正确转错误。完整质量结论继续以 OD2352 和 ERQA400 为准。实际记录保存在远端 run 的 `gateup-smoke-od.json`、`gateup-smoke-erqa/summary.json` 和 `gateup-smoke-erqa/diff.json`。


## 8. 全量结果与暂停位置

2026-11-15 用户因 token 预算要求暂停后续实验。两配置的完整质量评测已经完成，候选并发扫描、短形状 ND 复测与下一候选优化留待交接执行。

两配置均由正常入口启动，使用 TP2、BF16 主干、seqs1、eager、16384 上下文与 token 预算、GDN 决策 prefill 开启、schema 缓存关闭、16 图上限、像素 65,536～1,048,576。候选将全 64 层 gate/up 设为动态 W8A8。OpenDecisions 每配置先运行完整 2352 题预热，再测量一次完整 2352 题，均无错误。

| 指标 | BF16 对照 | gate/up W8A8 候选 |
| --- | ---: | ---: |
| OD Calibration | 82.794115 | 82.649369 |
| OD Intelligence | 66.056843 | 65.526827 |
| OD Capability | 74.425479 | 74.088098 |
| OD accuracy | 1906/2352 | 1903/2352 |
| OD Score MAE | 0.401389 | 0.401847 |
| OD C1 p50 / p95 / p99 ms | 199.515 / 207.708 / 213.291 | 195.245 / 207.105 / 210.495 |
| OD C1 wall 秒 | 408.704 | 416.646 |
| OD C1 decisions/s | 5.754781 | 5.645078 |
| ERQA accuracy | 219/400 | 219/400 |
| ERQA C1 p95 ms | 331.495 | 292.618 |
| ERQA C1 decisions/s | 3.528630 | 3.870919 |

这些性能值来自每配置一次全量窗口。ERQA 的 BF16 对照在冻结8例回放后运行，候选此前还执行了19题 ERQA 冒烟，视觉形状预热覆盖不同。后续应给两配置统一完整 ERQA 预热与测量协议，再确认性能收益。OD 正式窗口前约38.34～38.69秒缺少资源采样前界，后续2070条完整请求所在区间与整个 ERQA 区间的 CPU throttling 增量均为0。[BF16 核验摘要](../Clef-27B/logs/mlp-20261115/reference-summary.json)

候选 OD 的 choice 变化15条、noul 跨0.5共5条，作者有效性判定下18条正确变错误、15条错误变正确。最大概率差0.396890，319条概率差超过0.03，68条 score 期望差超过0.05，作为数值诊断保留。候选自身预热与正式轮2352条输出、8683个返回概率位模式完全一致。[候选重复性](../Clef-27B/logs/mlp-20261115/candidate-repeatability.json)

正式 OD 吞吐下降1.906%，p95减少0.603ms。小于512 tokens 的1636条请求累计增加10.573秒，至少512 tokens 的716条请求节省2.627秒。部分固定短形状出现约60～90ms的整请求退化，例如203、204、206、252、269、280、448 tokens。下一步优先确认这些形状的 INT8 GEMM 格式与 tiling 行为。[正式配对分析](../Clef-27B/logs/mlp-20261115/formal-paired-latency.json) · [短形状交接记录](../Clef-27B/logs/mlp-20261115/short-shape-followup.md)

BF16 的588题分层探索扫描在 C2/C4/C8/C16 分别测得5.7718/5.8414/5.8357/5.8600请求/秒，最高观测点 C16 的 p95 为3168.125ms。候选的对应扫描尚待执行。正式提交配置与 CSV 的采用结论以各自验收记录为准，本轮候选作为独立实验记录。[扫描结果](../Clef-27B/logs/mlp-20261115/bf16-scan/summary.json)

CPU 诊断还发现 layer31 的输入通道3994几乎总是决定 per-token INT8 scale。该层四种输入的近似输入量化 relative RMSE 为8.845%～9.649%，gate/up 权重重建误差约1%。进一步的激活平滑和选择性 BF16 可沿此证据开展，原生动态路径当前尚未接入 smooth_scales。[激活诊断](../Clef-27B/logs/mlp-20261115/activation-quantization-diagnostics.json)


## 9. 恢复与交接

用户要求暂停后，候选服务已停止。原服务恢复为 PID **520642**，8000端口 `/health` 200，冻结8/8通过，argv、cwd、记录环境与实验前快照完全一致。临时MLP输入采集源码恢复到 preimage SHA，采集处于未武装状态，实验驱动和探针进程全部结束。当前服务指针已更新到实际恢复进程。[恢复核验](../Clef-27B/logs/mlp-20261115/restore/verification.json)

候选全量原始输出、概率、启动manifest、严格追踪首次失败与原生加载日志均已保存，并核对压缩内容SHA与源文件SHA。[候选摘要](../Clef-27B/logs/mlp-20261115/candidate-summary.json) · [候选归档索引](../Clef-27B/logs/mlp-20261115/candidate-archive-index.json)

后续可直接复制独立交接prompt，按短形状ND/NZ、统一ERQA预热与候选并发扫描的顺序接续。[DeepSeek交接prompt](handoff-mlp-deepseek-2026-11-15.md)

## 10. ND 重量格式复核与全量候选，2026-11-15

交接后的第一项实验是原生 `weight_nz_mode=0` 的 ND 路径。探针已增加显式 `--weight-format nd|nz`，默认 NZ，运行前分别断言 INT8 权重为 format 2 或 format 29；驱动按 ND→NZ、NZ→ND 平衡顺序执行，报告保留实际格式、逐轮时序和源码 SHA。证据：[ND 汇总](../Clef-27B/logs/mlp-20261115/nd-format-ab.json)。

### 10.1 局部 MLP 链：ND 更慢

真实捕获输入（od-0677 1595-token 前缀）覆盖 145–255、256–383、384–511 及更长档，3 层（0/31/63）× 21 形状，四臂各 8 轮 × 20 次。`gate_up_only` 的 NZ/ND device 中位比：

| 档位 | 形状数 | NZ/ND 中位比 | ND 更快的点数 |
| --- | ---: | ---: | ---: |
| 145–255 | 24 | 0.768 | 0 |
| 256–383 | 21 | 0.919 | 0 |
| 384–511 | 12 | 0.886 | 0 |
| ≥512 | 6 | 0.905 | 2 |
| 全部 | 63 | — | **2/63** |

长输入同样由 NZ 胜出：T=662 的 gate_up 中位比 0.921，T=10294 为 0.905（>1 表示 ND 更快）。局部完整 MLP 链的结论是 **ND 比 NZ 慢 8%–30%**，与“短形状 NZ 退化”的假设相反。

### 10.2 服务级短形状子集：ND 明显更快

服务参数统一为 TP2/seqs1/eager/16384/16 图/像素 65,536–1,048,576/缓存关/GDN 决策 prefill；候选模型为 gate/up W8A8。选取 63 条集中在 203/204/205/206/252/254/269/270/280/281/307/335/448/449/498 tokens 的 OD 请求，ND 与 NZ 各启动一次服务，warmup 1 后 C1 测量：

| 指标 | ND | NZ |
| --- | ---: | ---: |
| p50 ms | **116.886** | 127.447 |
| p95 ms | **126.773** | 134.144 |
| wall s | **7.48** | 8.12 |
| 成功 | 63/63 | 63/63 |

逐请求配对：ND−NZ 中位 **−11.65 ms**、均值 −10.16 ms，ND 在 **60/63** 请求更快；对 BF16 全量同请求记录，ND 在 55/63 更快（中位 −5.51 ms）。ND/NZ 质量最大概率差 **0.0284**，离散决策变化 **0**。局部 MLP 链 ND 更慢、服务级 ND 更快，说明 NZ 的端到端退化来自**权重格式转换/调度路径**，不是 MLP 算术本身；ND 修复了异常短形状的整请求退化。

### 10.3 ND 全量 OD2352 与 ERQA400

ND 服务按同一 TP2/seqs1/16384 配置完成 2352 题预热 + 一次正式 2352 题：

| 指标 | BF16 对照 | NZ 候选 | **ND 候选** |
| --- | ---: | ---: | ---: |
| OD Calibration | 82.794115 | 82.649369 | 82.775913 |
| OD Intelligence | 66.056843 | 65.526827 | 65.367346 |
| OD Capability | **74.425479** | 74.088098 | 74.071630 |
| OD accuracy | 1906/2352 | 1903/2352 | 1902/2352 |
| OD Score MAE | 0.401389 | 0.401847 | 0.403257 |
| OD C1 p50 / p95 / p99 ms | 199.515 / 207.708 / 213.291 | 195.245 / 207.105 / 210.495 | **196.490 / 205.761 / 208.460** |
| OD C1 decisions/s | 5.754781 | 5.645078 | **5.771124** |

ND 相对 BF16：吞吐 **+0.28%**，P95 **−1.95 ms（−0.94%）**，p50 −3.02 ms；相对 NZ：吞吐 **+2.23%**，P95 −1.34 ms。质量相对 BF16 下降 Capability **−0.354**、accuracy −4 条（18 条正确变错、15 条错误变正确等累计迁移），相对 NZ 仅 −0.016 Capability、−1 条 accuracy。

ERQA400 在 ND 服务上先执行完整 400 题预热，再测量完整 400 题：400/400 成功、**219/400 = 0.5475**，与 BF16/NZ 相同；p50 267.816、P95 **305.716**、p99 507.308 ms，吞吐 **3.7055 req/s**。21 条预测变化（7 正确→错误、7 错误→正确、7 错误互换），最大概率差 0.4754。该窗口使用完整预热；BF16/NZ 的旧窗口预热条件不同，横向性能比较仍需按同一预热协议重测。

### 10.4 采用判断

- **默认提交配置保持 BF16**：ND 仍未消除 W8A8 的质量代价（Capability 74.0716 vs 74.4255，accuracy 1902 vs 1906），赛事默认质量优先。
- **ND 是已验证的性能候选**：它修复了 NZ 的短形状端到端退化，OD S1 吞吐略高于 BF16、P95 更低；若后续接受约 0.35 Capability 的代价，ND 可作为 S1 性能方案。
- **NZ 布局路径被否定**：局部 MLP 链 NZ 更快，但整请求在异常形状退化；ND 的端到端优势来自格式/调度，后续优化应优先检查 NZ 的权重转换、图/调度路径，而非继续改 MLP GEMM tiling。
- **待办**：候选 S2 并发扫描；BF16/NZ 在统一完整 ERQA 预热下的性能复测；ND 质量迁移的逐层/逐字段归因。

证据：`Clef-27B/logs/mlp-20261115/nd-format-ab.json`（局部短/长矩阵、服务子集、ND 全量 OD/ERQA 摘要）；远端原始报告 `/workspace/yangchen/logs/mlp-nd-20261115-042745/`。恢复后的 BF16 服务 PID **529132**，argv/cwd/环境与快照一致，8/8 冻结用例通过。
