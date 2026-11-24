# 单一常驻 TP2 优化执行记录（2026-11-24）

本页记录按 `p0_execution_plan.md` 在新 A3 机器上执行的阶段 0–3：代码基线 `origin/main`（`9942a3a`），单一常驻 TP2 服务、纯 BF16、无 RAM/JPEG 缓存、无温度校准。所有性能数字来自本机同窗口实测，质量使用服务原始概率。

## 0. 环境与代码状态

| 项 | 值 |
| --- | --- |
| 主机 | Ubuntu aarch64，640 逻辑 CPU / 320 物理核，8 NUMA 节点（每节点 80 CPU） |
| NPU | 卡 6，Chip 0/1（Phy-ID 12/13，Bus 0000:85:00.0 / 87:00.0），`Ascend910_9382` = 24 AIC / 48 AIV |
| 设备 NUMA | 两个芯片的 `numa_node` 均为 `-1`（容器内无设备 NUMA 亲和），绑核按独立 CPU 集处理 |
| CANN | 9.2.1；venv `/workspace/yangchen/venvs/clef-mix-ws-py312` |
| 代码 | 新 worktree `/workspace/yangchen/Clef-27B-ascend-tp2`，`origin/main` `9942a3a`（远端旧仓库为 shallow 快照，已用本地 bundle 重建） |
| 服务 | TP2 / BF16 / eager / 4K→16K / B16 / util .85 / encoder cache 262144 / 无 RAM-JPEG 缓存 |

现场旧服务（K56 激活缓存，PID 278578）已完整快照到 `logs/tp2-alwayson-20261124/phase0/prestop/` 后受控停止；端口 18143 与显存已释放。

## 1. 阶段 0：统一常驻 TP2 基线

同窗口先跑 `od-s1` 独立 profile（v3 aa3e、`TASK_QUEUE_ENABLE=1`），再切到 `UNIFIED_BASE`（16K/B16/util .85/encoder 262144）连续执行 OD S1 两轮、S2 C16/C64、ERQA C1/C16。

| 运行 | mean ms | p50 | p95 | p99 | dec/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| `od-s1` 独立 profile | 114.152 | 105.739 | 156.655 | 193.038 | 8.758 |
| unified S1 r1 | 119.498 | 111.687 | 156.519 | 193.598 | 8.367 |
| unified S1 r2 | 115.195 | 107.017 | 155.406 | 192.444 | 8.679 |

S1 P95 差值 ≤ 0.7 ms（unified 略优），锁定 `UNIFIED_BASE` 为后续统一评测配置。S2：C16 **16.239 dec/s / p95 1747.79 ms**，C64 **17.539 / p95 5935.28**。ERQA：C1 冷首遍 **216/400**、p95 299.44 ms；C16 **216/400**、7.150 dec/s。

## 2. 阶段 1：TP2 系统/绑核/同步消融

### 2.1 `TASK_QUEUE_ENABLE=2`（已采用）

ABBA（A-r1、B-r1、B-r2、A-r2，全量 2,352，C1）：

| 轮次 | mean | p50 | p95 | p99 | dec/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| A1（=1） | 119.450 | 112.657 | 156.743 | 194.416 | 8.370 |
| B1（=2） | 110.127 | 100.002 | 156.004 | 192.912 | 9.078 |
| B2（=2） | 111.578 | 102.176 | 156.408 | 193.448 | 8.960 |
| A2（=1） | 117.614 | 110.118 | 156.372 | 193.881 | 8.501 |
| 中位 A → B | **118.53 → 110.85（−7.68 ms）** | 111.39 → 101.09 | 156.56 → 156.21（−0.35） | 194.15 → 193.18 | **+6.9%** |

质量：B1 对 A1 全量 **2,352/2,352 逐位一致，`max_prob_diff=0.0`，0 choice 翻转**。S2 同窗口中性：C16 16.401 vs 16.379 dec/s，C64 17.652 vs 17.679；P95 差 ≤ 6.5 ms。`service.log` 零告警。
默认写回：`Clef-27B/inference.py` 与 `scripts/activate_env.sh` 的 `TASK_QUEUE_ENABLE` 默认值改为 `2`；`=1` 可复现旧基线。

### 2.2 ATB kernel cache（关闭）

`libatb.so` 确认消费 `ATB_OPSRUNNER_KERNEL_CACHE_GLOABL_COUNT/LOCAL_COUNT`（服务主进程与 EngineCore 均映射 libatb）。设 100/10 单轮：mean 108.40 / p50 97.59 / p95 155.64 / 9.223 dec/s；与紧邻 base（109.55/99.78/156.51/9.126）和 nolog（108.32/97.29/156.69/9.230）不可分离。记为中性关闭，不进入默认。

### 2.3 关闭每请求日志（关闭）

本 vLLM 版本请求日志为 opt-in（`--enable-log-requests` 默认 false），仅 uvicorn access log 可关。加 `--disable-uvicorn-access-log` 单轮与 base 等价（mean 108.32 vs 109.55，p95 +0.18 ms），无收益，关闭。

### 2.4 CPU 绑核（有效，保留为 opt-in）

同进程 A/B/B 全量 S1：未绑 mean 109.967 / p50 100.028 / p95 156.282 / p99 193.311；绑定后两轮 107.951/97.658/155.712/192.469 与 108.019/97.228/155.700/192.887。绑定收益 mean **−1.95 ms**、p50 −2.5 ms、p95 −0.57 ms、p99 −0.6 ms、吞吐 +1.8%。映射：api 160-175、EngineCore 320-335、Worker TP0 240-255、Worker TP1 400-415。
新增 `runtime/clef_bind_supervisor.py`（`CLEF_BIND_CPUS`，支持 `auto` 复现该映射），`Clef-27B/start.sh` 仅在显式设置时启用。未设为默认：CPU 集是机器相关配置，提交入口保持最小变更。

### 2.5 Rank 1 跳过 Joint Head（中性，未采用）

契约核查：multiproc executor 的 `output_rank = world_size - tp_size*pcp = 0`，仅 rank 0 的 `ModelRunnerOutput` 返回 EngineCore；`embed_lexical` 的 vocab-parallel all-reduce 两 rank 必须执行。补丁在 `clef_joint_schema.py` 中先执行 `embed_lexical`，再于 `tp_rank != 0` 追加等长零输出并跳过 `segment_mean/head/softmax`，默认 `CLEF_JOINT_RANK0_ONLY=0`。
单轮：mean 109.666 / p50 99.363 / p95 156.870 / p99 194.058 / 9.117 vs base 109.551/99.780/156.508/193.225/9.126 → 中性（P95 +0.36 ms）。未采用，开关保留默认 0。

## 3. 阶段 2：GDN 自适应 chunk（负结果，默认全 128）

### 3.1 拐点微基准（归一化 q/k，`npu:0`）

单序列（varlen）：

| T | 180 | 256 | 307 | 384 | 450 | 512 | 640 | 768 | 967 | 1200 | 1595 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| c64 / c128 | 1.027 | 1.000 | **0.953** | 0.993 | 1.058 | 1.015 | 1.036 | 1.090 | 1.217 | 1.107 | 1.116 |

多序列打包（总 token）：8×300 1.036、8×450 1.272、4×967 1.373、混合 1.280、16×450 1.298、8×967 1.406。

结论：chunk 64 仅在 307 token 附近（约 −4.7%）占优；≥450 与所有多序列批次均明显劣于 128，`T_th` 实际窄于计划的 384/512。

### 3.2 安装器与全量筛选

新增 `runtime/apply_gdn_adaptive_chunk_patch.py`（preimage = chunk128 postimage，builder 同时预构建 64/128 两套 host chunk 索引 + `max_prefill_seq_len`/`total_prefill_tokens`，device_op 读取 `CLEF_GDN_CHUNK128_MIN_TOKENS`/`CLEF_GDN_CHUNK_RULE` 选择）。chunk128/decision/qkv-norm 安装器均把 adaptive postimage 视为 patched，重启幂等。

阈值 384 全量单轮：mean 109.628 / p50 98.563 / p95 156.581 / p99 193.182 / 9.120 vs base 109.551/99.780/156.508/193.225/9.126 → p50 −1.2 ms，但 mean +0.08、P95 +0.07，未达 “mean/p50 改善且 S2 无退化” 的采纳门。**保留全 128 默认（阈值 0）**，`CLEF_GDN_ADAPTIVE_CHUNK` 默认改为 `0`（opt-in），线上源码树已回退到 chunk128 postimage。
S2 同窗对照：adaptive 15.998（C16）/17.146（C64），回退后 chunk128 16.046/17.162 → 自适应补丁本身无 S2 成本；两者都低于 08:46 窗口（16.379/17.679），属窗口/租户漂移。S2 为多 prefill，融合 kernel 本身不参与。
注意：当前 selector 未加 “仅单序列” 守卫；若启用非零阈值，多序列 S2 会退化（见 3.1），启用前必须补守卫。

## 4. 阶段 3：`ClefGdnConvQkv` 48-AIV 与解除 1,600 上限（已采用）

### 4.1 离线交叉验证

`Ascend910_9382` 的平台配置为 24 AIC / **48 AIV**；v3 只用了 24 个 AIV。v4 把 host tiling 的 `kCores` 改为 48，并按 `min(totalTiles, 48)` 设置 blockDim；kernel 源码未改（kernel `.o` 与 v3 完全相同），仅 tiling 库变化。
- 逐位：v3 vs v4 在 48 个用例（180–10,294 token、5120/8192 输入、2 seeds、tile32）共 144 个输出张量 **`torch.equal` 全部为真，`max_abs=0.0`**。
- 单算子 fused 时间（v4/v3）：≤512 约 0.84–1.11（噪声）；967 0.78–0.84；1595 0.71–0.88；1653 0.67–0.76；2048 0.65–0.80；3559 0.53–0.55；4096 0.54；8192 0.55–0.57；10,294 0.55–0.57。对分解路径的加速比由 v3 的约 1.9× 提升到 **3.2–3.7×**（≥3,559 token）。

### 4.2 服务级验收（v4 vendor + `CLEF_GDN_QKV_NORM_MAX_TOKENS=16384`）

| 指标 | v3/1600 基线 | v4/16384 | 变化 |
| --- | ---: | ---: | ---: |
| OD S1 p95（三轮） | 156.37–156.51 | **155.34 / 154.73 / 154.06（中位 154.73）** | −1.6 ~ −1.8 ms |
| OD S1 mean（三轮） | 109.97（09:30 窗口） | 117.45 / 116.64 / 113.02 | 窗口负载更高，非退化结论 |
| OD 质量 | Cap 74.247710 | **逐位一致（2,352/2,352，0.0，0 翻转）** | 通过 |
| ERQA C1 p95 / 吞吐 | 299.44 ms / 4.633 | **278.08 ms / 4.831** | −7.1% / +4.3% |
| ERQA C1 准确率 | 216/400 | **216/400** | 持平 |
| ERQA C16 p95 / 吞吐 | 3429.44 ms / 7.150 | **2974.53 ms / 8.197** | −13.3% / **+14.6%** |
| ERQA C16 准确率 | 216/400 | **218/400（0.545）** | +0.5 pp，2PP 内 |

S2（同一 v4 服务）：C16 **15.998 dec/s / p95 1795.85 ms**、C64 **17.146 / p95 6095.71**。融合 kernel 只在 `num_prefills == 1` 启用，S2 B16 不命中；同窗口回退 adaptive 后的 chunk128 对照为 16.046 / 17.162，说明 v4 对 S2 无因果影响，当前窗口整体比 08:46 低约 2–3%。

ERQA 长序列（>1,600 token）此前全部回退分解路径；v4 在 1,653–10,294 token 均快于分解路径，因此 C1/C16 的 P95 与吞吐同时改善。

## 5. 阶段 4：官方 profile 复核（单一常驻服务）

最终默认配置（TQ2 + joint R + conv v4 + cap 16384 + chunk128，无 RAM/JPEG 缓存）用官方 `scripts/start_dataset_profile.sh` 逐 profile 冷启动复测：

| profile | 结果 |
| --- | --- |
| `od-s1`（4K/seqs1/B1，3 轮） | p95 **156.117 / 155.014 / 154.480 ms（中位 155.014）**；mean 119.156 / 117.671 / 114.871；吞吐 8.391 / 8.496 / 8.704 dec/s；2,352/2,352 × 3 |
| `od-s2`（4K/B16） | C16 **16.138 dec/s / p95 1778.85 ms**；C64 **17.241 dec/s / p95 6076.61 ms**；2,352/2,352 |
| `erqa-c1`（16K/seqs1/util .85，冷首遍） | **216/400 = 0.540**；p95 **293.893 ms**；4.633 dec/s；400/400、0 重试 |
| `erqa-s2`（16K/seqs8，客户端 C16） | **219/400 = 0.5475**；7.006 dec/s；p95 3670.31 ms；400/400、0 重试 |

对照计划的 2026-11-23 参考：S1 P95 156.182 → **155.014（−1.17 ms）**；S2 C64 17.689 → 17.241（本窗口整体偏慢约 2.5%，融合路径不参与 S2）；ERQA S1 P95 300.538 → **293.893（−6.65 ms）**、S2 7.166 → 7.006（窗口漂移）、准确率 216/400 → **219/400（+0.75 pp）**。OD 质量与基线逐位一致（2,352/2,352、`max_prob_diff=0.0`、0 翻转；Cap 74.2477、accuracy 1904/2352、MAE 0.402612）。全部评估均通过官方 2PP 门、0 失败、0 重试。

## 6. 采纳清单与回退

| 项 | 状态 | 回退 |
| --- | --- | --- |
| `TASK_QUEUE_ENABLE=2` | 采用（默认） | `TASK_QUEUE_ENABLE=1` |
| `ClefGdnConvQkv` v4（48 AIV）+ cap 16384 | 采用（默认） | 用 `vendor-v3-backup` 覆盖 vendor；`CLEF_GDN_QKV_NORM_MAX_TOKENS=1600` |
| Joint Rank0-only | 未采用 | `CLEF_JOINT_RANK0_ONLY=0`（默认） |
| CPU 绑核 | opt-in | `CLEF_BIND_CPUS` 未设置（默认） |
| GDN 自适应 chunk | 未采用 | `CLEF_GDN_CHUNK128_MIN_TOKENS=0`（默认） |
| ATB cache / 关 access log | 关闭 | 不设置 |

## 7. 证据路径

- 阶段 0/1：`/workspace/yangchen/logs/tp2-alwayson-20261124/phase0|phase1/`（status.jsonl、逐轮 summary/report、质量 JSON）。
- 阶段 2：`phase2/chunk-single-20261124.json`、`chunk-multi2-20261124.json`（以及误解析的 `chunk-multi-20261124.json`，仅作短序列参考）。
- 阶段 3：`phase3/probe-v3.json`、`probe-v4.json`、`probe-v3-v4-equal.json`、`v4-s1-r{1,2,3}/`、`v4-quality.json`、`v4-erqa-*/summary.json`；vendor 备份 `/workspace/yangchen/work/tp2-alwayson-20261124/vendor-v3-backup`。
- 阶段 4：`phase4-official/official-s1.json`（官方 od-s1 三轮）、`phase4/smoke-result.json`（冷启动 + 5 协议用例）、`phase5/final-profiles.json`（od-s2 / erqa-c1 / erqa-s2）。
- 汇总：`Clef-27B/logs/tp2-alwayson-20261124/evidence-summary.json`。
