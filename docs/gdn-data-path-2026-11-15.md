# GDN 决策 prefill 数据路径优化（2026-11-15）

本轮在已采用的 GDN 决策专用 prefill（跳过 final state、写回与反向中间量）之上，整理完整 prefill 的数据路径：Q/K/V 布局、初始循环状态、卷积持久状态，并实现 Ascend C 前处理融合原型。所有实验维持 TP2 / BF16 / eager / 固定权重与软件版本，资源限定在同一物理 A3 卡内的两个逻辑 NPU。

> 所有原始报告保存在远端 `/workspace/yangchen/logs/gdn-data-path-20261115/`，仓库内保留精简证据与索引。

## 1. 结论

- **采用 BIC**：`bsnd` + `skip_initial_state` + `skip_conv_state` 三项在完整、无历史的 prefill 上逐位等价。OpenDecisions 全量 2,352 题与 baseline 逐位一致（Capability 74.425479），ERQA 400 题 R/BIC 概率逐位一致（219/400）。
- **性能**：6 对交错 A/B 中 BIC **6/6 更快**，p95 中位 **−7.01%（−13.81 ms）**，吞吐 +0.378 decisions/s。正常入口 S1 P95 **198.42 ms**，较上一提交入口 213.56 ms 降 **7.09%**；seqs4 C8 **14.652 dec/s**（p95 913.93 ms），C16 14.689。
- **负结果**：Ascend C 融合原型短序列约 2×，但 1595/10,294 token 反而慢 0.86×/0.13×，且 Q/K 有 1 ulp 差异；不接入。
- **入口**：`--config s1`/`s2` 默认启用 BIC，manifest 记录三开关与来源；v3 安装器支持从 v2 树升级；冻结 8 例 8/8，原服务按 phase-0 快照恢复（argv/cwd/env 逐项一致）。

## 2. 范围与已固定的优化

所有实验臂显式设置 `decision_prefill=true`、`no_recompute=true`，即上一轮已验收的决策专用路径保持不变。本轮新增三个独立开关：

| 开关 | 作用 |
| --- | --- |
| `bsnd` | 以 sequence-major 的 Q/K/V 调用 `layout="BSND"`，并在 head-first 卷积存储上完成 Q/K L2Norm，去掉 BNSD 路径的两次布局拷贝 |
| `skip_initial_state` | 完整、无历史的 prefill 直接传 `initial_state=None`，跳过 `gather_ssm_states`、状态转置与索引 |
| `skip_conv_state` | prefill 卷积传 `conv_states=None`，跳过持久卷积缓存的读取与尾部写回 |

并形成 R、B、I、C、BI、BC、IC、BIC 八个组合。开关文件带严格递增的 `epoch`，服务按 vLLM forward context 固化一次 forward 的配置；每个 TP rank 在新 epoch 首次命中时各写一行宿主日志，驱动以此证明两 rank 已加载同一配置。

## 3. 代码落点

| 文件 | 改动 |
| --- | --- |
| `runtime/gdn_decision_flags.py` | 六开关 + epoch + 严格布尔校验 + 每 forward 快照 |
| `runtime/apply_gdn_decision_patch.py` | 安装器 v3：A3 包装器可选初始状态、head-first BSND 归一化、conv/state 开关；识别 v2 postimage 并从已验证 preimage 备份升级 |
| `runtime/gdn_data_path_driver.py` | 多臂驱动：原子写、epoch 预热与双 rank 校验、逐臂保存、异常恢复 |
| `runtime/gdn_data_path_probe.py` | NPU 契约探针：conv 无状态、initial_state=None、BSND 连续/跨步、八臂包装器、空 segment |
| `runtime/clef_qkv_norm_probe.py` | Ascend C 融合算子与现有单拷贝路径的正确性/微基准 |
| `Clef-27B/inference.py` | `s1`/`s2` 默认启用 BIC 三开关，manifest 记录请求值、生效值、来源与环境变量 |

## 4. 接口级正确性

`runtime/gdn_data_path_probe.py` 使用安装的 26.9.1 算子与真实 TP2 头布局（Q/K 8 heads、V 24 heads、head_dim 128），在 NPU 上覆盖 chunk 边界（1/3/4/63/64/65/127/128/129）、文本代表形状（203/307/401/512）、异构合批（307/64/129/205）、长序列（662、10,294）和空 segment：

| 检查 | 结果 |
| --- | --- |
| `causal_conv1d_fn(conv_states=None)` vs 零状态基线（真实 1821/2048 行缓存、int32 cache、bool has） | 全部逐位一致；基线写回状态、候选不写 |
| `chunk_gated_delta_rule_fwd(initial_state=None)` vs 显式零状态 | 全部逐位一致，`output_final_state=False` 返回 `None` |
| BSND 连续操作数 vs BNSD 基线 | 逐位一致 |
| BSND 跨步视图（head-first 归一化后直接转置） | aclnn 以 161002 拒绝，必须保留一次连续拷贝 |
| 包装器 R/B/I/C/BI/BC/IC/BIC（真实卷积输出视图） | 8/8 逐位一致，输出 finite，`final_state=None` |
| `keep_meta` 空 segment（`cu=[0,0,64,64,129]`） | 与显式零状态逐位一致 |

结论：`skip_initial_state` 与 `skip_conv_state` 可以按逐位等价接入；BSND 的接口级收益只来自去掉 BNSD 的第二次布局拷贝，剩余的一次连续拷贝需要融合算子才能消除。

## 5. 八臂筛选与受控 A/B

### 5.1 八臂筛选

固定 49 条代表样本（按 2,352 题真实 token 分布的 4 个桶各取 12 条，另加 307-token 探针；`screening.json` SHA-256 `05adec5d…`），同一服务进程内 2 轮、交替/反向顺序。表中为两轮中位：

| 臂 | p95 (ms) | mean (ms) | p50 (ms) | decisions/s |
| --- | ---: | ---: | ---: | ---: |
| R | 201.97 | 143.38 | 126.93 | 6.978 |
| B | 197.40 | 137.13 | 117.49 | 7.291 |
| I | 200.07 | 137.24 | 119.26 | 7.286 |
| C | 200.75 | 138.72 | 119.39 | 7.207 |
| BI | 196.36 | 133.42 | 115.79 | 7.494 |
| BC | 196.42 | 135.67 | 116.13 | 7.369 |
| IC | 199.23 | 135.68 | 117.97 | 7.369 |
| **BIC** | **195.72** | **132.55** | **114.72** | **7.543** |

八臂共 98×8 条响应与 R 的归一化 JSON **逐位一致**（max float diff 0.0）。BIC 进入受控 A/B。

### 5.2 六对交错 A/B（R vs BIC）

588 题等距子集，同进程、每臂相同预热、交替顺序：

| 对 | 顺序 | p95 R | p95 BIC | p95 降幅 | mean 差 | 吞吐差 |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | R→BIC | 198.78 | 184.99 | 6.94% | 9.45 ms | +0.363 |
| 2 | BIC→R | 197.79 | 183.53 | 7.21% | 9.12 ms | +0.357 |
| 3 | R→BIC | 197.19 | 181.68 | 7.87% | 10.07 ms | +0.393 |
| 4 | BIC→R | 193.39 | 183.12 | 5.31% | 10.68 ms | +0.427 |
| 5 | R→BIC | 189.10 | 182.51 | 3.49% | 7.31 ms | +0.297 |
| 6 | BIC→R | 195.33 | 181.50 | 7.08% | 10.74 ms | +0.441 |
| **中位** | | | | **7.01%（13.81 ms）** | **10.07 ms** | **+0.378** |

BIC 在 **6/6** 对更快；降幅算术均值 6.32%。六对共 588×6×2 条响应与同对 R 逐位一致。

## 6. 全量质量与 S1

### 6.1 OpenDecisions 1.0.1 全量 2,352（seqs1，同进程依次 R→BIC）

| 臂 | 成功 | p50 | p95 | p99 | mean | decisions/s | Capability | accuracy | Score MAE | vs 冻结 baseline |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| R | 2,352 | 185.19 | 197.95 | 204.31 | 168.62 | 5.930 | 74.425479 | 1906/2352 | 0.401389 | **2,352/2,352 逐位** |
| **BIC** | 2,352 | 175.27 | **185.62** | 198.93 | 160.96 | **6.212** | **74.425479** | 1906/2352 | 0.401389 | **2,352/2,352 逐位** |

- BIC 对 R：p95 −12.33 ms（−6.23%）、mean −7.66 ms、吞吐 +4.76%，与 6 对 A/B 方向一致（单次全量不作采用判定的唯一依据）。
- 两臂 choice 变化 0、noul 跨 0.5 为 0、概率差 >0.03/0.05 为 0、最大概率差 0.0、Score 期望差 0.0。
- 本次 R 的绝对值（197.95）低于历史 entry S1（213.56，同一 v2 决策路径），属于本机跨运行漂移；采用结论以同进程 A/B 为准。
- 输入 token：mean 446.81、p50 357.5、p95 969.25、max 1595。

**正常入口（无 flag 文件）最终 S1**：BIC p50 179.86、p95 **198.42**、p99 204.70、
mean 160.59、**6.226 decisions/s**，2352/2352 成功；质量与 baseline
**2,352/2,352 逐位一致**，且与上面的 flag 文件 BIC 运行 **2,352/2,352 逐位一致**。
入口运行与 11:00 窗口的 flag 文件运行相差约 6.9%，属于本机跨运行漂移；采用结论
以同进程 A/B 为准。相对上一提交入口 S1（213.56 ms），正常入口 BIC 降幅 **7.09%**，
与 A/B 的 7.01% 一致。

### 6.2 ERQA（官方 400 题，`/v1/decide`）

| 臂 | 成功 | accuracy | p50 | p95 | p99 |
| --- | ---: | ---: | ---: | ---: | ---: |
| R | 400/400 | 219/400 = 0.5475 | 259.50 | 325.84 | 559.08 |
| **BIC** | 400/400 | 219/400 = 0.5475 | **244.69** | **309.84** | **552.08** |

R 与 BIC 的 400 题**概率逐位一致**（exact 400/400、prediction 变化 0、
max 概率差 0.0）；BIC 的 p95 低 16.00 ms（−4.91%）。

### 6.3 seqs4 / S2 兼容性

seqs4 服务（`--config s1 --max-num-seqs 4`，缓存关），全量 2,352、C8：

| 臂 | 吞吐 (dec/s) | p50 (ms) | p95 (ms) | p99 (ms) | mean (ms) |
| --- | ---: | ---: | ---: | ---: | ---: |
| R | 14.201 | 480.13 | 946.54 | 1049.88 | 562.50 |
| **BIC** | **14.652** | **466.40** | **913.93** | 1083.44 | **545.17** |

BIC 对 R：吞吐 +3.17%、p95 −32.61 ms（−3.45%）。

C16 BIC：**14.689 dec/s**、p50 969.36、p95 1841.20、mean 1084.38；相比 C8
只多 0.25% 吞吐、p95 翻倍，推荐工作点仍是 C8。

质量（C8，作者评分器，全量 2,352）：

| 臂 | Capability | accuracy | max 概率差 | >0.03 | >0.05 | choice 变化 | noul 跨 0.5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline C1 | 74.425479 | 1906 | 0 | 0 | 0 | 0 | 0 |
| R C8 | 74.225345 | 1905 | 0.097399 | 23 | 7 | 4 | 2 |
| BIC C8 | 74.194172 | 1904 | 0.097399 | 23 | 6 | 5 | 2 |

C8 的差异来自既有的批内位置数值非确定性（见 `docs/tuning4-2026-11-14.md`），
不是本轮开关：C1 下两臂均与 baseline 逐位一致。

**R-vs-R 对照**：同一 R 臂在 C8 再跑一次全量（14.198 dec/s、p95 950.39），
与第一次 R 比较：2352 题中 **85 题不同、最大概率差 0.0656**；而 R vs BIC
的 C8 差异为 305 题、最大 0.0359。同臂重跑的差异在幅度上不小于跨臂差异，
说明 C8 的跨臂差异由请求到达/批内位置调度决定，不能归因于三个开关。因此
seqs4 的 S2 采用 BIC 的性能数字，质量沿用既有结论：C8 未通过 0.03/0.05
工程门槛，最终提交仍为 seqs1/B1。

## 7. 融合原型

### 7.1 算子与构建

`runtime/ascendc/clef_gdn_qkv_norm/` 是 msopgen 生成的 A3 自定义算子项目，目标
`ascend910_93`（24 AI cores），CANN 9.2.1。输入 head-first 卷积输出
`x [40,T,128]` bf16，输出连续 BSND `q/k [1,T,8,128]`（Q/K L2Norm）与
`v [1,T,24,128]`。

- Q/K tile：bf16→fp32，平方写入独立 fp32 缓冲；逐行两次 Level 2
  `ReduceSum(..., 64)` 相加（最终可用版本）、`+eps`、`Rsqrt`、逐行 `Muls`、
  `CAST_RINT` 回 bf16。Level 1 `ReduceSum` 的 int mask 触发 UB 对齐异常，
  `uint64_t mask[2]` 版本返回 NaN，均保留为负结果。
- V tile：同一 cast 路径，不归一化；bf16 往返逐位保持。
- 24 block 按连续 tile 分配，每 tile 64 token；输出按
  `(outHeads-1)*128*2` 字节跨步写 BSND。
- `bash build.sh` 生成 `custom_opp_ubuntu_aarch64.run`；解包到
  `vendors/customize`，通过 `ASCEND_CUSTOM_OPP_PATH` 注册，aclnn 入口为
  `aclnnClefGdnQkvNorm`。

### 7.2 结果与采用决定

`runtime/clef_qkv_norm_probe.py --iters 20`，NPU 0：

| tokens | fused (ms) | 现有单拷贝路径 (ms) | 加速 | Q/K 最大差 | V |
| ---: | ---: | ---: | ---: | ---: | --- |
| 307 | 0.1217 | 0.2444 | **2.01×** | 0.001953（1 bf16 ulp） | 逐位 |
| 401 | 0.1232 | 0.2536 | **2.06×** | 1 ulp | 逐位 |
| 512 | 0.1255 | 0.2538 | **2.02×** | 1 ulp | 逐位 |
| 1595 | 0.3054 | 0.2613 | 0.86× | 1 ulp | 逐位 |
| 10294 | 1.9556 | 0.2608 | 0.13× | 1 ulp | 逐位 |

**决定：保留原型，不接入服务。** 短序列每层约省 0.12 ms（48 层约 5.8 ms/请求），但 S1 分布 p95 969、max 1595，ERQA 可达 10,294；融合算子在长序列上慢 1.2–7.5×。原因是当前可用的 Level 2 逐行 `ReduceSum`（每行两次）和逐行标量 `Muls` 随 token 线性放大；Level 1 `ReduceSum` 的 int mask 会触发 UB 对齐异常，`uint64_t mask[2]` 版本返回 NaN。下一步需要与 Triton `tl.sum` 同序的 tile 级归约与广播缩放；在此之前 BIC 是更优路径。

构建身份：`libcust_opapi.so` SHA-256 `4d2440a3…`，kernel 对象 `d1687581…`，`custom_opp_compiler_version=9.2.1`。负结果与失败分支均保留在 `runtime/ascendc/clef_gdn_qkv_norm/README.md`。

## 8. 入口与服务恢复

### 8.1 正常入口默认值

`--config s1`/`s2` 在 GDN 决策补丁启用时默认打开 BIC 三开关：
`CLEF_GDN_BSND=1`、`CLEF_GDN_SKIP_INITIAL_STATE=1`、
`CLEF_GDN_SKIP_CONV_STATE=1`；`preprocess_fusion` 默认关闭。显式
`CLEF_GDN_*` 环境变量或实验 flag 文件优先，manifest 的
`gdn_decision_prefill.data_path` 记录 `configured`、`effective`、`source`
和 `env`，并记录 flag 文件路径、SHA-256 与 epoch。

最终 S1/S2 性能测量使用正常入口（不设置实验 flag 文件），入口 manifest、
`/health`、冻结 8 例与全量质量见第 9 节。

2026-11-15 11:38 正常入口实测（`--config s1 --port 18143`）：

- `launch-manifest.json`：`patch_version=3`，三个源码文件 `patched`，
  `flag_module_sha256=3c9cb53a…`；`gdn_decision_prefill.data_path` 的
  `bsnd`/`skip_initial_state`/`skip_conv_state` 均为
  `configured=true, effective=true, source=config:s1`，`preprocess_fusion`
  为 `false`，`flag_file.path=null`；
- 服务环境仅含 `CLEF_GDN_DECISION_PREFILL=1`、`CLEF_GDN_NO_RECOMPUTE=1` 与
  三个数据路径变量，无 `CLEF_GDN_FLAG_FILE`；
- 冻结 8 例 `8/8` 逐位一致；全量 2,352 的 S1 与质量见 9.1/9.2。

### 8.2 服务恢复

实验服务（seqs1、seqs4、ERQA、正常入口）全部停止后，用 phase-0 快照恢复原
baseline 服务：

- 新 PID **587932**，监听 `127.0.0.1:8000`；`argv`、`cwd` 与 phase-0
  （PID 551494）**逐项相等**，白名单环境变量**逐项相等**（无任何 `CLEF_GDN_*`，
  因此走原始 GDN 路径）；
- `GET /health` **200**（原服务未设置 `CLEF_HEALTH_JSON`，body 为空，与
  phase-0 一致）；`GET /v1/decide/info` 返回原服务的默认
  `revision=unknown/config=custom`；冻结 8 例 **8/8** 逐位一致；
- 源码树保留本轮采用的 v3 补丁（`device_op.py 6886fa3f…`、`gdn.py
  7bc9b8f1…`、`patch_qwen3_5.py 1b82b52e…`、`gdn_decision_flags.py
  3c9cb53a…`）；恢复服务不加载决策路径；
- `/workspace/yangchen/logs/current-service.json` 更新为 PID 587932 与实际
  argv/cwd/env；
- `npu-smi` 仅显示恢复服务的两个 TP worker，无实验进程。

## 9. 证据与复现

远端原始目录：`/workspace/yangchen/logs/gdn-data-path-20261115/`。仓库内
`Clef-27B/logs/gdn-data-path-20261115/` 保存精简报告、索引与源文件哈希；
`evidence-index.json` 逐文件记录大小与 SHA-256。

| 内容 | 路径 |
| --- | --- |
| phase-0 进程/设备/版本/冻结 8 例 | `phase0/snapshot.json`、`phase0/versions.json` |
| 接口契约探针（含 10,294 token） | `phase0/probe-final.json`、`phase0/probe-10294.json` |
| v3 安装报告 | `phase0/patch-apply-v3.json` |
| 八臂筛选 | `screen/driver.json`、`screen/screen-summary.json`、`screen/screen-response-diff.json` |
| 六对 A/B | `ab/driver.json`、`ab/r0*-summary.json` |
| 全量 OD R/BIC | `full/R-*.json`、`full/BIC-*.json` |
| seqs4 C8/C16 | `seqs4/c8-*-summary.json`、`seqs4/c16-BIC-summary.json`、`seqs4/seqs4-c8-quality.json` |
| ERQA | `erqa/R-summary.json`、`erqa/BIC-summary.json`、`erqa/erqa-R-vs-BIC.json` |
| 融合原型 | `fusion/fusion-probe.json`、`runtime/ascendc/clef_gdn_qkv_norm/` |
| 正常入口 | `entry/launch-manifest.json`、`entry/frozen8.json`、`entry/quality.json` |
| 服务恢复 | `restore/compare.json`、`restore/verify.json` |

关键源码与算子身份（`source-hashes.json`）：

| 工件 | SHA-256 |
| --- | --- |
| `device_op.py`（v3 postimage） | `6886fa3f…71d3ce` |
| `gdn.py`（v3 postimage） | `7bc9b8f1…77c893` |
| `patch_qwen3_5.py`（v3 postimage） | `1b82b52e…c0b0460` |
| `gdn_decision_flags.py` | `3c9cb53a…947287` |
| 安装器 v3 | `b67d4ed9…cc8f25` |
| 驱动 | `2cd18d17…51c8704` |
| 契约探针 | `2b7572c4…d98128` |
| 融合 kernel 源码 | `85e76cb0…e8943d` |
| 融合 `libcust_opapi.so` | `4d2440a3…1c2c98` |
| 融合 kernel 对象 | `d1687581…b6dc472` |

复现命令：

```bash
cd /workspace/yangchen/Clef-27B-ascend
source scripts/activate_env.sh

# 安装/校验 v3 补丁（源码树固定 vLLM-Ascend 89eb8a5）
python runtime/apply_gdn_decision_patch.py --check
python runtime/apply_gdn_decision_patch.py --apply
python runtime/test_gdn_decision_patch.py
python runtime/test_gdn_data_path_driver.py
python runtime/test_inference_launch.py

# 实验服务（显式 opt-in flag 文件）
CLEF_GDN_FLAG_FILE=/dev/shm/clef-gdn-ab.json bash Clef-27B/start.sh \
  --model-dir /workspace/yangchen/models/Clef-27B \
  --output-dir <run>/service --config s1 --port 18143

# 八臂筛选 / 六对 A/B
python runtime/gdn_data_path_driver.py --sample-file <screening.json> \
  --limit 49 --warmup 4 --arms R,B,I,C,BI,BC,IC,BIC --rounds 2 \
  --flag-file /dev/shm/clef-gdn-ab.json --service-pid <pid> --output-dir <run>/screen
python runtime/gdn_data_path_driver.py --sample-file <ab-subset-stride4.json> \
  --limit 588 --warmup 8 --arms R,BIC --pairs 6 \
  --flag-file /dev/shm/clef-gdn-ab.json --service-pid <pid> --output-dir <run>/ab

# 全量质量（驱动逐臂）
python runtime/gdn_data_path_driver.py --sample-file <all2352.json> --limit 2352 \
  --warmup 8 --arms BIC --rounds 1 --flag-file /dev/shm/clef-gdn-ab.json \
  --service-pid <pid> --output-dir <run>/full-BIC

# 融合原型
cd runtime/ascendc/clef_gdn_qkv_norm
msopgen gen -i clef_gdn_qkv_norm.json -c ai_core-Ascend910_93 -f aclnn -lan cpp -out ClefGdnQkvNorm
# 覆盖 op_host/op_kernel 后 bash build.sh；解包 run 到 vendors/customize
CLEF_QKV_NORM_VENDOR=<vendor> python runtime/clef_qkv_norm_probe.py --output <run>/fusion-probe.json

# 正常入口（不设置 flag 文件；s1 默认 BIC）
bash Clef-27B/start.sh --model-dir /workspace/yangchen/models/Clef-27B \
  --output-dir <run>/entry-s1 --config s1 --port 18143
```
