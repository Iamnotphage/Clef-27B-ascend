# 多序列 Varlen `ClefGdnConvQkvVarlen`（2026-11-25）

> 分支 `perf/conv-qkv-varlen-20261125`，基线 `origin/main` **512317e**。目标：让 OD S2 与 ERQA S2 的多序列 prefill 也走 fused conv+QKV 预处理（原 v4 只在 `num_prefills == 1` 生效），S1 / ERQA C1 保持 v4 原样逐位一致。阶段 4 同窗 A/B 与全量验收已通过，按计划快进合入 `origin/main`。

## 1. 算子与接入

- **独立算子** `ClefGdnConvQkvVarlen`（源码 `runtime/ascendc/clef_gdn_conv_qkv_varlen/`，vendor `Clef-27B/src/ascendc/vendors/clef_conv_qkv_varlen/`），不改动 v4 vendor 与 `ClefGdnConvQkv`。
- 输入 `x` (`[T,5120]` 或 packed `[T,8192]` 零拷贝)、`weight` (`[4,5120]`)、`cu_seqlens` (`int32, [num_seqs+1]`)；输出 `q/k/v` (`[1,T,8/8/24,128]`, BF16)。
- **Per-sequence tile decomposition**：`Init()` 用 `MTE2_S` 事件把 ≤65 个 `cu_seqlens` 搬入 UB，构造 `tileOff_`；`Process()` 在 48 AIV 上按 `(head, seq, localTile)` 轮询，每个 tile 完全落在单条序列内，`history = min(localT0,3)`、`leading = 3-history`，绝不读取相邻序列字节；内层 conv/SiLU/BF16 round/QK L2Norm/BSND 写出与 v4 逐行一致。
- `TilingFunc` 校验 `1 <= num_seqs <= 64`、`inputChannels ∈ {5120,8192}`、`tile ∈ {16,32,64}`，`blockDim = min(40*(CeilDiv(T,tile)+num_seqs), 48)`。
- `gdn.py` 叠层安装器 `runtime/apply_conv_qkv_varlen_patch.py`：preimage `ead25cb2…`（v4 postimage）→ postimage **`afe6d4e8…`**；`num_prefills == 1` 仍走 v4，`2..64` 且 `CLEF_GDN_CONV_QKV_VARLEN=1` 才分流 varlen；`--check/--apply/--revert`、文件锁、字节级备份、未知源 Fail-Closed。
- `inference.py`：`s1` 默认 `CLEF_GDN_CONV_QKV_VARLEN=1`（显式 `=0` 一键回退）；自动注入 varlen vendor 到 `ASCEND_CUSTOM_OPP_PATH`/`LD_LIBRARY_PATH`；manifest 记录 `patch_state.conv_varlen`、vendor SHA 与 `CLEF_CONV_QKV_VARLEN_TILE=64`。

## 2. 阶段 2 离线探针（4 门全部通过）

脚本 `runtime/clef_conv_qkv_varlen_probe.py`，证据 `Clef-27B/logs/conv-qkv-varlen-20261125/offline-probe.json`。

| 门 | 用例数 | 结果 |
| --- | ---: | --- |
| 1. B=1 vs v4，`T ∈ {1,2,3,4,31,32,33,180,307,967,1595,3559,10294}` × 5120/8192 × tile 16/32/64 | 78 | 全部 `torch.equal=True`（max_abs 0.0） |
| 2. 多序列 vs 逐条 v4，B 2–64、含 `[1,2,3,4,30,31,32,33,34,63,64,65,...]`、packed 8192 毒化尾 | 48 | 全部 `torch.equal=True` |
| 3. 零串扰：单序列 ×100 / ±1e4 / 全 NaN 篡改 | 144 | 其余序列 q/k/v 全部逐位不变，0 NaN 扩散 |
| 4. 单算子 vs 分解路径（`causal_conv1d_fn` + `ClefGdnQkvNorm`） | 20 组合 | **3.53–4.54×**，tile64 最优；锁定 `tile=64` |

单算子示例（tile64, 8192 输入）：3600 (8×450) **4.06×**、7200 (16×450) **4.19×**、7736 (8×967) **4.33×**、15472 (16×967) **4.53×**、irregular `[10294,307,967,1653]` **4.17×**（5120）/ **4.54×**（8192）。

## 3. 阶段 4 同窗 A/B（OFF vs ON，单一常驻 TP2，POST /v1/systemone）

服务均从同一官方入口启动（TP2/BF16/eager/16384/seqs16/util .85/encoder 262144/TASK_QUEUE=2/convQKV-v4/chunk128/joint fused `26d3ad04`/state A2）。OFF 设 `CLEF_GDN_CONV_QKV_VARLEN=0`；ON 为默认。OFF1、ON1 为完整臂，OFF2 为第二个 OFF 样本（host load 19–25）。

| 工作负载 | OFF1 | OFF2 | OFF 均值 | ON1 | ON 相对 OFF |
| --- | ---: | ---: | ---: | ---: | ---: |
| OD S2 C16 dec/s | 16.292 | 16.259 | 16.276 | **16.971** | **+4.27%** |
| OD S2 C16 p95 ms | 1784.85 | 1786.28 | 1785.57 | **1704.39** | **−4.55%** |
| OD S2 C64 dec/s | 17.203 | — | 17.203 | **18.039** | **+4.86%** |
| OD S2 C64 p95 ms | 6081.21 | — | 6081.21 | **5779.14** | **−4.97%** |
| ERQA C16 冷首遍 dec/s | 6.936 | 6.988 | 6.962 | **7.281** | **+4.59%** |
| ERQA C16 冷首遍 p95 ms | 3487.98 | 3511.11 | 3499.55 | **3334.71** | **−4.71%** |
| ERQA C16 暖第二遍 dec/s | 8.142 | — | 8.142 | **8.627** | **+5.96%** |
| ERQA C16 暖第二遍 p95 ms | 3049.15 | — | 3049.15 | **2815.63** | **−7.66%** |

生效核验：ON 服务两个 TP worker 均打印 `clef-conv-qkv-varlen pid=… tile=64 first_num_prefills=15 first_tokens=3689`，同时 `num_prefills==1` 仍有 `clef-conv-qkv`（v4）日志；OFF 服务只打印 v4，无 varlen 日志。

S1（OD C1 全量 2,352）ON：**154.11 ms / 9.244 dec/s**（mean 108.15、p50 97.53），与 512317e 报告逐题 **max_prob_diff=0.0、0 choice flips、2,352/2,352**。

ERQA C1（ON2 干净冷启动，先 C1）：冷 **206.94 ms / p95 289.26 / 216/400**；暖 **165.40 ms / p95 244.66 / 216/400**；冷/暖 `max_prob_diff=0.0`、0 flips（与 512317e 的 205.17/288.40、164.71/244.38 一致）。

## 4. 全量质量（官方评分器）

| 运行 | Cal | Int | Cap | accuracy | Score MAE |
| --- | ---: | ---: | ---: | ---: | ---: |
| ON OD S1 | 82.4113 | 66.0841 | **74.2477** | 1904/2352 | 0.402612 |
| ON OD S2 C16 | 82.2821 | 65.8429 | **74.0625** | 1903/2352 | 0.401356 |
| ON OD S2 C64 | 82.3892 | 66.0753 | **74.2322** | 1904/2352 | 0.402291 |
| OFF OD S2 C16 | 82.3997 | 65.8934 | 74.1466 | 1904/2352 | 0.401677 |
| OFF OD S2 C64 | 82.1744 | 66.0460 | 74.1102 | 1905/2352 | 0.402636 |

- 全部相对 baseline 74.4255 / ERQA 0.5475 的绝对波动 **≤ 0.36 PP（OD）**、**≤ 1.0 PP（ERQA）**，2PP 门通过；0 失败、0 重试。
- ERQA ON：C16 冷/暖均 **218/400 = 0.5450**；C1 冷/暖 **216/400 = 0.5400**；OFF C16 冷 216/217、暖 217。`erqa_score.py --verify-requests` 全部 **0 mismatch**。
- 冒烟：`/v1/decide/info` 正常、恶意 `image_url` 空串返回 **400** 且服务保持 ready；完整 400 题与 OD 2,352 已覆盖。

## 5. 结果与未决项

- **采纳**：`CLEF_GDN_CONV_QKV_VARLEN=1` 与 `CLEF_CONV_QKV_VARLEN_TILE=64` 作为 `s1` 默认；S1 逐位不变，S2 约 **+4.3–5.0% 吞吐、−4.6–7.7% P95**。
- **双缓冲未实现**：当前 UB 深度 1，阶段 2 单算子已达 3.5–4.5×，未再评估深度 2；如后续需要，可在不改变数值的前提下单独实验。
- **未决**：主办方对 S2 有效性（p95 约束）的最终口径仍按规则文档跟踪；本次 A/B 同时记录 C16/C64 与 P95，便于按最终规则取点。

## 6. 证据与入口

- 报告：本文件；`Clef-27B/optimization.md` 追加同名章节；`Clef-27B/README.md` 更新默认配置与最终行。
- 轻量证据：`Clef-27B/logs/conv-qkv-varlen-20261125/`（`offline-probe.json`、`ab-summary.json`、各 ERQA/质量 JSON）。
- 远端原始：`/workspace/yangchen/logs/conv-qkv-varlen-20261125/`、vendor 构建 `runtime/ascendc/clef_gdn_conv_qkv_varlen/build_out/`。
- 安装器：`runtime/apply_conv_qkv_varlen_patch.py`；gdn.py postimage **`afe6d4e87c0eb47bfb325fb630620e47f845fb5a893cb1dada78069ad4e3ca07`**。
