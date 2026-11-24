# Clef-27B 优化记录

## 首次调查快照，2026-11-12

- 目标平台为昇腾 910A3，当前可见两个 NPU 计算单元，每个 64 GiB HBM。
- 模型源为 `Cloudflare/clef`，本地模型目录为 `/workspace/yangchen/models/Clef-27B`。
- 2026-11-12 已从 ModelScope 下载完整模型，共 28 个文件，54,989,897,681 字节（约 55.0 GB）。所有文件的 SHA-256、主干权重索引和 safetensors 头部边界检查通过。校验明细见 [logs/download_verification.json](logs/download_verification.json)，权重检查见 [logs/model_inspection.json](logs/model_inspection.json)。
- 早期实例为 CANN 9.0.0、驱动 26.1.1、Python 3.10.12，当时推理依赖待配置；保留该次环境来源与采集记录。[早期环境调查](../docs/environment.md)

## 当前 A3 baseline，2026-11-12

新机器 `199.88.54.221` 使用现有 CANN / A3 Ops / ATB 9.2.1，独立 CPython 3.12.16、torch 2.10.0+cpu、torch_npu 2.10.0.post7。固定 vLLM / Ascend 源码、依赖、构建日志、最终 `pip check` 和 78 个自定义算子导入均已记录。[安装记录](../docs/installation-2026-11-12.md)

完整主干 12 分片、视觉塔和独立联合头已在 TP1/TP2 实际加载。联合头 122/122 参数张量、128,056,324 参数加载齐全，strict tracking 的缺失列表为空。CPU 参考为完整 27,484,784,884 参数 BF16 模型；NPU 的 GDN 派生卷积布局参数额外占 3.75 MiB，有逐项核算证据。

| 固定条件：307 tokens，3 字段，候选 2/2/3，预热 3，正式 30，并发 1，max len 1024 | TP1 | TP2 |
| --- | ---: | ---: |
| 请求/秒 | 7.095650 | 6.838840 |
| p50，ms | 140.301320 | 144.985050 |
| p95，ms | 143.511251 | 159.794070 |
| CPU 参考离散决策一致 | 7/7 | 7/7 |
| 最大候选概率绝对差 | 0.001453638 | 0.000256002 |
| 物理 14 / 15 的采样 HBM，MiB | 59964 / 2885 | 60007 / 59667 |

HBM 为 1 秒采样的设备占用，包含设备基础及缓存占用。TP2 每卡权重约 25.76 GiB；默认 0.90 利用率下，KV 池扩大到约 28.76 / 28.77 GiB。容量、权重和缓存预算应分别比较。上述吞吐与延迟限定为同一组 30 次短请求。

## 兼容与复现改动

1. 保留 Triton-Ascend 的 NumPy 1.26.4 约束，将 vLLM 的 OpenCV 声明固定为 4.11.0.86；图片和视频真实解码已通过。
2. 固定 vLLM 构建声明含 torch 2.14.0；本轮采用 `empty`、`--no-build-isolation` 和 constraints，构建与运行复用验收过的 torch 2.10。
3. flash-attn 编译线程池显式读取 `MAX_JOBS=8`；Arctic 预装 nanobind / grpcio-tools，FLA 补齐其声明的 expecttest。
4. 加载后输出 `CLEF_LOAD_AUDIT`，记录实际模块、张量数量、设备/dtype 和缺失检查。
5. TP2 首请求暴露上游基类 embedding 契约缺失；在 Ascend 构造函数补齐 `use_fused_embedding=False`、`reduce_results=True`、`parallel_group=output_reduce_group`。原有 Ascend forward 的 AST 保持一致，修复后通过真实两卡精度与性能检查。

四份补丁保存在 [scripts/patches/](../scripts/patches/)，源码固定提交和补丁 SHA、最小验证及实际部署日志见安装文档。

## 结果

TP2/4096 的完整公开评测为 2,983 请求、4,583 决策，执行错误为 0：JevBench 公开 231 条 `public_accuracy=0.8701`；OpenDecisions 2,352 条 `capability=0.7443`；Typed Decisions 400 条、2,000 决策的准确率 `0.7165`。HCCL FP32/BF16、红蓝图片和双向视频自检全部通过。

[完整 baseline 与逐题日志路径](../docs/baseline-2026-11-12.md) · [性能 CSV](results.csv) · [指标 CSV](metrics.csv) · [性能报告图](result.png) · [NPU 实测报告图](npu_device_call.png)

两张图由真实报告和 SSH CLI 输出导出，保留来源时间。原始 Terminal 窗口访问受计算机工具安全规则限制。正式比赛输入、官方 baseline 数值及最终图像规范按主办方协议对齐；CSV 中未知 baseline 字段保持为空。

## 后续比较项

按同一精度判据比较固定 KV 池预算、更长输入、更多字段和更高并发，分别记录缓存容量、设备占用和请求性能。公开文本指标、概率分布及已有多模态自检作为回归依据。

## 第二轮性能调优，2026-11-13

A3 单物理卡内两个逻辑 NPU 边界内完成。完整实验报告见 [docs/tuning2-2026-11-13.md](../docs/tuning2-2026-11-13.md)，远端逐点原始记录与 master-results.csv（43 行）在 `/workspace/yangchen/logs/tuning/20261113-085408-perf2a/`。全部数字为关闭 profiler 的正式测量（每点 ≥300 请求、≥30 秒；稳定结论三轮交错，batch 分布由 Prometheus 直方图差分重建并逐点校验）。

| 改动或配置 | 实测收益（fixed307，TP2/4096/eager） | 决定 |
| --- | --- | --- |
| batching seqs4 + 并发 8（真实 B4） | 8.237 → 19.325 rps（CV 0.05%）；dec/s 24.71 → 57.98 | 采纳为 S2 配置 |
| 独立 `--max-num-batched-tokens` 参数 | 长输入 3868 tokens：4096→16384 预算仅 1.964→2.118 rps（MLP 线性封顶） | 参数保留，默认跟随 max-model-len |
| 双 TP1 副本（同卡双实例+入口分流） | C2 14.268 rps @ p95 141 ms（容量假设 14.288 获证实） | 低延迟中并发备选 |
| `weight_nz_mode=2`（BF16 NZ） | 线性层设备时间 −9.3%（46.50→42.17 ms/req/rank）但端到端 −0.8%～−1.4% | 拒绝，保留默认 mode 1 |
| schema token 有界缓存 | encode CPU −89.6%；s4c8 +0.28%（19.380）；质量逐位同界 | 默认关闭；S2 可开启（env 一行） |
| PIECEWISE 编译（去 eager） | 编译 +29.7 s、捕获 3 图；s4c8 19.340 rps 持平 | 不采纳 |
| KV 池 12 GiB（原 28.5） | 307/B4、3868 长输入、冷图全部零回退；回收约 16.5 GiB/rank | 容量可选项 |

质量回归：推荐配置（seqs1）完整 OpenDecisions 1.0.1 2,352 条重跑，Calibration 82.7941149442 / Intelligence 66.0568428085 / Capability 74.4254788763、accuracy 1906/2352、Score MAE 0.401389，与既有 baseline **逐位一致**；B4 混合并发与缓存开启两种条件下 771/771 离散决策一致、概率最大差 0.000851（阈值 0.03）。

S1/S2（results.csv 12 列行，配置来源各自一致）：S1 = seqs1，p95 130.296 ms（三轮均值）；S2 = seqs4，57.98 dec/s @ 并发 8（关缓存）或 58.14（开缓存）。备选低延迟部署：双 TP1 C2 14.27 rps @ p95 141 ms。

## 第四轮复核与最终配置，2026-11-14

完整报告见 [docs/tuning4-2026-11-14.md](../docs/tuning4-2026-11-14.md)。

- **S2 合批质量**：seqs4/cache-on C8 的全量 2,352 条重算 Capability **74.200178**，4 条 choice 变化、2 条 noul 跨 0.5、26 条概率差 >0.03（最大 0.097399）、3 条 score 差 >0.05（最大 0.076599），未通过 0.03/0.05 工程门槛。C2 与 S1 逐位一致。根因是跨请求合批的批内位置相关 BF16 数值差（相同输入在不同批位置得到不同结果）；GDN 融合算子、conv1d 均验证为多序列隔离正确，差异首先出现在 `in_proj_ba` 的 1 ulp 行差，按序列执行 BA 仍残留 0.104 最大概率差。
- **最终采用**：seqs1/B1（TP2/4096/eager/cache off）。全量质量 74.425479 与 baseline 逐位一致；S1 P95 **217.036 ms**；S2 **5.715 decisions/s @ C16**（C2/C4/C8 为 5.626/5.673/5.584）。
- **吞吐候选**：seqs4/C8 13.764 dec/s 保留在 `results.csv`，标注 `throughput-candidate-quality-fail`。
- **GDN 候选**：旧 guard 在真实 prefill 因非连续 `[1,T,H,D]` 视图 100% 回退；修正后真实服务命中、0 回退、8/8 冻结用例逐位一致；不再声称端到端收益。
- **启动与补丁**：`start.sh --output-dir` 真实落盘 manifest/log，默认 18143，`--config s1/s2` 显式配置；schema cache v2 安装器支持上游 SHA 校验、版本识别、幂等安装与精确回退。
- **统计修正**：cgroup v1/v2 单位、unknown 有效性、预注册顺序条件、B4 分母（2.762 ms/req）、输入长度 446.809、每并发点独立证据与 SHA。

## ERQA 图文配置与结果，2026-11-14

完整报告见 [docs/erqa-baseline-2026-11-14.md](../docs/erqa-baseline-2026-11-14.md)。

- **数据**：官方 ERQA TFRecord，commit `5bba488`，91,402,921 字节，SHA-256 `1974f5e5…b4c93e2`；400 题、630 图、8 类，单题 1～16 图。
- **服务**：TP2 / BF16 / eager / `max_model_len=16384` / `max_num_seqs=1` / 缓存关；`--limit-mm-per-prompt {"image":16,"video":0}`。
- **图像策略**：`shortest_edge=65536`、`longest_edge=1048576`（`JointSchemaStrategy.IMAGE_SIZE` 参考边界）；全量前逐题 processor 预算最大 10,294 tokens，实测 `usage.input_tokens` 与预算一致。
- **质量**：219/400 = **0.5475**，400/400 成功、0 重试；单图 0.5923、多图 0.4336；8 类分项见报告。
- **C1 计时**：wall 101.099 s、3.9565 decisions/s，p50 / p95 / p99 为 267.849 / 323.476 / 551.516 ms。该窗口为 ERQA 质量运行，不与 OpenDecisions S1/S2 混用。
- **适配**：EvalScope ERQAAdapter 修补 `visual_indices` 图文排列、2～4 候选解析与 SystemOne 图文转换；19 题原生路径与直接 `/v1/decide` 逐题一致。补丁与 SHA 见 [logs/erqa-baseline-20261114/adapter-patch/](logs/erqa-baseline-20261114/adapter-patch/)。
- **边界**：Clef 原生协议将图片放在 `state` 前，未按官方字符索引交错插入；30 条交错样本单独统计。图像上限 1,048,576 像素为 16 图与 16,384 token 的联合约束。

## GDN 决策 prefill 数据路径，2026-11-15

完整报告见 [docs/gdn-data-path-2026-11-15.md](../docs/gdn-data-path-2026-11-15.md)。

- **开关**：在已采用的 `decision_prefill`/`no_recompute` 之上新增 `bsnd`、`skip_initial_state`、`skip_conv_state`，八臂 R/B/I/C/BI/BC/IC/BIC；严格 `epoch`、逐 forward 快照、双 TP rank 日志校验。
- **契约探针**：`conv_states=None`、`initial_state=None`、连续 BSND 与实际包装器八臂在真实 TP2 形状（含 10,294 token、空 segment）全部逐位一致；BSND 跨步视图被 aclnn 161002 拒绝，保留一次连续拷贝。
- **筛选与 A/B**：49 条代表样本八臂筛选 BIC 最优（p95 195.72 ms，R 201.97）；588 题 6 对交错 A/B 中 BIC **6/6 更快**，p95 中位 **−7.01%（−13.81 ms）**，吞吐 +0.378 dec/s，响应逐位一致。
- **质量**：OpenDecisions 全量 2,352 的 R 与 BIC 都 **2,352/2,352 与 baseline 逐位一致**（Capability 74.425479、accuracy 1906、Score MAE 0.401389）。ERQA 400 题 R/BIC 概率逐位一致，accuracy 219/400。
- **性能**：正常入口 S1 P95 **198.42 ms**（较上一提交入口 213.56 −7.09%）；同进程全量窗口 R 197.95 → BIC **185.62 ms**（−6.23%）；seqs4 C8 14.201 → **14.652 dec/s**、p95 946.54 → **913.93 ms**；C16 14.689。C8 的跨臂响应差异与同臂重跑（85 题、max 0.0656）同量级，归因于既有合批调度，不是本轮开关。
- **入口**：`--config s1/s2` 默认启用 BIC，manifest 记录配置/生效值/来源；`preprocess_fusion` 默认关闭。
- **融合原型**：Ascend C `ClefGdnQkvNorm` 短序列 2.0×、Q/K 1 ulp、V 逐位；1595/10,294 token 反而慢 0.86×/0.13×，不接入。

## 2026-11-17 aclnn executor cache（已采用）

- **现象**：S1 按精确 input token 数分快/慢两档，慢档固定多约 55 ms；同长度请求 within-length sd 中位 4.7 ms，快/慢档输出逐位相同。
- **根因**：CANN aclnn executor 缓存默认只覆盖前约 145 个不同 prompt 长度，之后首次出现的新 shape 走慢路径；MLP、GDN chunk、conv、QKV norm 的纯算子时间无跳变。
- **修复**：`ACLNN_CACHE_LIMIT=1000000`，写入 `inference.py` 的 service environment，launch manifest 记录。
- **结果**：官方 `start.sh --config s1` 全量 2,352 条 mean **125.552** / p50 **118.170** / p95 **168.633 ms**、7.960 dec/s；修复前 153.761 / 168.485 / 190.405、6.503。四组独立全量复测 p95 167.28–169.02 ms。
- **质量**：2,352 条响应逐位一致（max 0.0、0 choice 翻转）；ERQA 217/400 不变；S2 B8 C16 16.398 dec/s 不变。
- **证据**：`docs/aclnn-cache-shape-fix-2026-11-17.md`、`Clef-27B/logs/tuning5-20261117/`。

## 2026-11-17 MLP BF16 NZ（已采用）

- **改动**：`Qwen2MoeMLP.forward` 首次调用时把 `gate_up_proj` / `down_proj` 权重转为 FRACTAL_NZ（`torch_npu.npu_format_cast(..., 29)`），由 `CLEF_MLP_BF16_NZ=1` 控制；安装器升级为 GDN v4，`patch_qwen3_5.py` 追加该 hook。
- **正确性**：官方入口全量 2,352 条响应与 cache-only **逐位一致**（max 0.0、0 choice 翻转）；ERQA 217/400、400/400 成功、0 重试。
- **性能**：官方入口全量 S1 **mean 121.705 / p50 113.656 / p95 166.002 / p99 204.324 ms、8.213 dec/s**；cache-only 为 125.552 / 118.170 / 168.633 / 203.452、7.960。相对 cache-only p95 −1.56%、mean −3.06%、吞吐 +3.18%；三组 NZ 复测 p95 165.53–166.00。
- **S2**：B8 C16 + NZ **16.376 dec/s**、p95 1803.2 ms、Capability 74.1572（Δ−0.094，2PP）；pre-NZ 16.398。
- **证据**：`Clef-27B/logs/tuning5-20261117/s1-official-nz-full.json.gz`、`s2/s2-nz-c16-quality.json`、`erqa-nz/`；完整运行目录 `/workspace/yangchen/logs/tuning5-20261117/shape-ab/official-nz/`。

## 2026-11-17 conv+QKV 融合算子（已采用）

完整报告见 [docs/gdn-conv-qkv-2026-11-17.md](../docs/gdn-conv-qkv-2026-11-17.md)。

- **改动**：新增 Ascend C 算子 `ClefGdnConvQkv`，一次调用完成 causal conv1d、q/k L2 norm 与 v 直拷，替代 conv（3.79 ms）+ `ClefGdnQkvNorm`（4.66 ms，均为 967 token 设备时间）两个 kernel。
- **免拷贝**：Qwen3.5 把连续 `[T,8192]` 的 QKVZ 投影拆成 QKV `[T,5120]` 与 Z 两个视图；宿主用受存储边界检查的 `as_strided` 把整行 8192 暴露给算子，kernel 只读每行前 5120 通道，省掉逐层 5120 列连续化拷贝。
- **正确性**：离线探针 v1/v2/v3 共 177 组（10 长度 × 3 seed × tile 32/64 × 输入 5120/8192）与部署版 conv + QKV v2 **逐位一致、0 失败**；单算子 967 token 0.2113 vs 0.3143 ms、1595 0.1296 vs 0.3357。
- **服务级**：官方 `start.sh --config s1` 生效（两个 TP worker 打印 fused 标志）。与 eager 对照**不是逐位一致**：三个固定样本最大概率差 0.00095 / 0.00240 / 0.00029；把 `CLEF_GDN_CONV_QKV` 设为 0 后与对照逐位一致，差异确定来自融合算子（约 0.6 BF16 ulp）。
- **性能**：同窗口全量 A/B，p95 **165.091 → 160.419 ms（−2.83%）**、mean 122.762 → 120.926、吞吐 8.137 → 8.259 dec/s；显式 env 组为 p95 160.150、mean 117.344、8.510 dec/s。
- **质量**：OpenDecisions Capability **73.9636**（−0.347 pp）、accuracy 1902 不变、Score MAE 0.401545 → 0.401678；ERQA **218/400 = 0.5450**（+0.25 pp）。官方 2PP 通过；项目内部 0.03/0.05 逐题门槛不通过（max 0.1447、1 条 choice 变化）。
- **开关**：vendor 内置于 `src/ascendc/vendors/clef_conv_qkv`，`inference.py` 自动注入 `ASCEND_CUSTOM_OPP_PATH` / `CLEF_CONV_QKV_VENDOR` / `CLEF_GDN_CONV_QKV=1`；`CLEF_GDN_CONV_QKV=0` 复现未融合对照臂。

## 2026-11-17 GDN chunk 128（已改为默认）

完整报告见 [docs/gdn-chunk128-2026-11-17.md](../docs/gdn-chunk128-2026-11-17.md)、叠加数据见 [docs/gdn-conv-qkv-2026-11-17.md](../docs/gdn-conv-qkv-2026-11-17.md) 第 4.3 节。

- **改动**：`gdn_attn_builder.py` 的 `_GDN_CHUNK_SIZE=128` 与 `device_op.py` 决策分支 `chunk_size=128`，由独立安装器 `runtime/apply_gdn_chunk128_patch.py` 管理；`inference.py` 在决策 prefill 启用时自动 apply，`CLEF_GDN_CHUNK128=0` 回到 64。
- **性能（叠加融合算子）**：mean 120.926 → **118.733**、p50 114.177 → 111.504、p95 160.419 → **158.301**、p99 197.454 → 194.491、吞吐 8.259 → **8.411 dec/s**；默认入口复跑 **157.745 ms / 8.652 dec/s**。
- **质量**：OD Capability **74.2476**、accuracy 1904/2352、Score MAE 0.402615；ERQA **216/400 = 0.5400**；S2 B8 C16 **16.727 dec/s**、p95 1768.9、Capability 73.8426、accuracy 1905。全部在 2PP 内。
- **逐题漂移**：3 条 choice 变化、`max_prob_diff` 0.360、46 条 >0.03、24 条 >0.05；项目内部 0.03/0.05 门槛不达标，用户确认按官方 2PP 口径采纳。

## 2026-11-18 Wave 2 复核（未新增默认改动）

完整报告见 [docs/perf-wave2-20261118.md](../docs/perf-wave2-20261118.md)。

- **重新 profile（967/1595 token，双 rank）**：设备 compute union **132.69 ms**（conv+QKV 融合与 chunk128 之前为 144.60）；MLP gate/up 41.50、MLP down 20.54（cube 94–97%）、GDN 17.97、QKV 5120→8192 14.88（MTE2 0.93）、`ClefGdnConvQkv` 4.17（原 conv+QKV norm 8.45）。通信非重叠 14.17 ms、无 device kernel 11.84 ms。
- **S2**：B8 C16 **16.738**、B16 C64 **17.462 / 17.475**（两次独立复测）、B16 C64 把预算放大到 24576 仅 17.374 → 17.4–17.5 已饱和，维持 B16 C64 / 16384。
- **GDN chunk**：96/160/192/256 均返回 `161002` 不支持；128 是唯一快于 64 的受支持取值（1595 token device 0.521 vs 0.608 ms），已默认。
- **joint head**：门控打点显示 967 token 下 pooler 5.90–6.25 ms，head 4.78–5.05（decoder 2.23–2.33、routing 1.09–1.18、final 0.62–0.67），全部是小算子下发；CPU 卸载不划算（`memory_projection` 10.1 GFLOP 必须留 NPU），不采用。
- **prefix caching**：Mamba 前缀缓存要求 chunked prefill（关闭态启动即断言失败），且 chunked prefill 会关闭决策快速路径；不启用。
- **最终验证**：S1 五轮 **8.646 / 8.705 / 8.849 / 8.983 / 9.035 dec/s**、p95 **156.81 / 157.50 / 155.85 / 155.86 / 155.72 ms**，中位 **8.849 dec/s / 155.86 ms**（2352/2352）；全量 OpenDecisions 重评 Cap **74.2476**、Int 66.0840、Cal 82.4112、accuracy 1904/2352、MAE 0.40261，与既有证据逐位一致；ERQA 独立复跑 **216/400**（400/400、0 重试、请求哈希校验通过）。

## 2026-11-24 单一常驻 TP2：TASK_QUEUE=2 + 48-AIV Conv+QKV（已采用）

完整报告见 [docs/tp2-alwayson-2026-11-24.md](../docs/tp2-alwayson-2026-11-24.md)。

- **统一评测服务**：单实例 TP2/BF16/eager，16K/seqs16/util .85/encoder cache 262144，无 RAM/JPEG 缓存；OD S1 与 `od-s1` 独立 profile 的 P95 差 ≤0.7 ms。
- **TASK_QUEUE_ENABLE=2**：ABBA 全量 S1 mean **118.53→110.85 ms**、p50 111.39→101.09、p95 156.56→156.21、吞吐 +6.9%；2,352/2,352 逐位一致、0 choice 翻转；S2 中性。已设为入口默认，`=1` 复现旧基线。
- **Conv+QKV v4（48 AIV）**：`Ascend910_9382` 为 24 AIC / 48 AIV，旧 tiling 只用 24；新 tiling 用满 48 并按 `min(totalTiles,48)` 设 blockDim，kernel `.o` 不变。v3/v4 48 例 144 张量 `max_abs=0.0`；长序列单算子 1.8–1.9×，对分解路径 3.2–3.7×。`CLEF_GDN_QKV_NORM_MAX_TOKENS` 1600→16384，使 ERQA 长 prefill 保持融合。
- **官方 profile 实测（2026-11-24）**：OD S1 三轮 p95 **156.117 / 155.014 / 154.480 ms（中位 155.014）**、mean 119.156/117.671/114.871、吞吐 8.391/8.496/8.704；OD S2 profile B16 C64 **17.241 dec/s**、p95 6076.6 ms（C16 16.138 / 1778.8）。
- **ERQA**：C1 冷首遍 **216/400**、p95 **293.893 ms**（基线 300.538）、4.633 dec/s；B8 C16 **219/400 = 0.5475**、7.006 dec/s（基线 216/400、7.166）。全部 400/400、0 失败、0 重试。
- **质量**：OpenDecisions 与基线逐位一致（Cal 82.4113 / Int 66.0841 / Cap 74.2477、accuracy 1904/2352、MAE 0.402612）；ERQA 0.5400（C1）/0.5475（C16），均在 2PP 内。
- **未采用**：ATB kernel cache（中性）、关闭 access log（中性）、Rank1 跳过 joint head（中性，开关默认 0）、GDN 自适应 chunk（64 仅在 307 token 附近占优、多序列全面更慢，默认全 128）、CPU 绑核（mean −1.95 ms，但 CPU 集与机器相关，保留 `CLEF_BIND_CPUS` opt-in）。

## 2026-11-24 `/v1/systemone` 图文 `state` 列表 + 单一常驻 TP2 最终验收（已采用）

完整记录见 [docs/systemone-mm-state-2026-11-24.md](../docs/systemone-mm-state-2026-11-24.md)。

- **接口**：`serving.py` 在 `create_decision` 中归一化 `state` 内容块；文本/JSON 原样透传，命中 `image_url` 列表后 Fail-Closed（400）；图片与原顺序文本进入 `media["image"]` 与 `render_state`。安装器 `runtime/apply_systemone_mm_state_patch.py` 校验 `serving.py` preimage `5a700bb7…`、postimage `99a8e1b7…` 与模块 `a8b87841…`。
- **模式**：A1 `extract_plain` ERQA **213/400**；A2 `extract_question_dict` **216/400**，锁定 A2（等价 2026-11-14 基线 state 包装），未改 `strategies.py`。
- **单一常驻 TP2**：16384 / seqs16 / util .85 / encoder cache **262144** / TASK_QUEUE=2 / conv+QKV v4（48 AIV）/ joint fused `26d3ad04`。官方裸命令 75 s 就绪。
- **ERQA**：C1 冷首遍 **216/400**、P95 **288.400 ms**、4.790 dec/s；自然第二遍 **216/400**、P95 **244.382 ms**、5.942 dec/s，冷/暖 **400/400 逐位一致（0 概率差、0 翻转）**；C16 **217/400**、8.240 dec/s、P95 2953.745 ms。全部 400/400、0 失败、0 重试。
- **OpenDecisions**：S1 2,352/2,352、P95 **154.548 ms**、9.131 dec/s，与已验收 2026-11-24 运行 **max_prob_diff=0.0、0 翻转、2,352/2,352 token 相等**；质量 Cal 82.4113 / Int 66.0841 / Cap **74.2477**、accuracy 1904/2352、MAE 0.402612。S2 C16 **16.345 dec/s**（p95 1783.72 ms，Cap 74.1510）、C64 **17.193 dec/s**（p95 6087.10 ms，Cap 74.0514）。
- **冒烟**：`erqa-0000` 366 tokens、`erqa-0167` 10,294 tokens、交错 200、恶意 400、旧 `/v1/decide` 与 A2 逐位一致、冻结 8 例逐位一致。
- **阻塞**：EvalScope 未安装且无离线包/网络，原生客户端联调未执行；完整 2,352 题已由固定 `/v1/systemone` 客户端覆盖。

## 2026-11-25 多序列 Varlen Conv+QKV（已采用）

完整报告见 [docs/conv-qkv-varlen-2026-11-25.md](../docs/conv-qkv-varlen-2026-11-25.md)。

- **算子**：新增独立 `ClefGdnConvQkvVarlen`（48 AIV，per-sequence tile，`cu_seqlens` 作为 int32 输入，`MTE2_S` 同步，`history = min(localT0,3)`），v4 vendor 与单序列路径零改动。
- **离线 4 门**：B=1 78/78 与 v4 逐位一致；多序列 48/48 与逐条 v4 逐位一致（含 1/2/3/4 边界、64 序列、packed 8192 毒化尾）；144/144 零串扰（×100、±1e4、全 NaN）；单算子对分解路径 **3.53–4.54×**，锁定 tile 64。
- **同窗 A/B**（OFF→ON→OFF）：OD S2 C16 **16.276→16.971 dec/s（+4.27%）、p95 1785.6→1704.4 ms（−4.55%）**；C64 **17.203→18.039（+4.86%）、p95 6081.2→5779.1（−4.97%）**；ERQA C16 冷 **6.962→7.281（+4.59%）、p95 3499.6→3334.7（−4.71%）**，暖 **8.142→8.627（+5.96%）、p95 3049.2→2815.6（−7.66%）**。
- **S1 零回退**：ON1 OD S1 2,352/2,352 与 512317e `max_prob_diff=0.0`、0 flips；ERQA C1 冷/暖逐位一致，216/400，P95 289.26/244.66 ms。
- **质量**：OD ON Cap 74.2477（S1）/74.0625（C16）/74.2322（C64），ERQA C16 **218/400**、C1 216/400，全部 2PP 内、0 失败、0 重试。
- **入口**：`runtime/apply_conv_qkv_varlen_patch.py`（preimage `ead25cb2…` → postimage `afe6d4e8…`），`s1` 默认 `CLEF_GDN_CONV_QKV_VARLEN=1`、`CLEF_CONV_QKV_VARLEN_TILE=64`；显式 `=0` 回退。

## 2026-11-25 分层缓存 Tier 2：实验完成但默认关闭

完整记录见 [docs/tiered-cache-2026-11-25.md](../docs/tiered-cache-2026-11-25.md)。

- **阶段 0–2 已完成**：OD/ERQA 前缀审计（离线 token 2,752/2,752 与服务一致）、A3 pinned H2D 55/D2H 39 GiB/s、生产 GDN 128 对齐续算 18/18 组逐位一致、Tier 1 定容 262144（HBM 61,358/61,009 MiB，余量 ≥4 GiB）、HostStore/LRU/SHM/异步传输与 35 单测、4 GiB 16 线程压测 0 泄漏。
- **阶段 3 阻塞（自动不采纳）**：原生 Mamba prefix caching 强制 `mamba_cache_mode="align"`，而决策快路径要求 `not enable_prefix_caching` 且 `not enable_chunked_prefill`；两者互斥，打开即关闭 BIC/varlen，无法满足冷首遍 ≤1%。自定义 scheduler/KV/mamba per-request 状态恢复未完成。**Tier 2 默认 off，主线保持 29a1bf8。**
- **最终 29a1bf8 指标**：OD S1 154.11 ms / 9.244 dec/s（与 512317e 逐位一致）、S2 C16 16.971 / p95 1704.4、C64 18.039 / p95 5779.1；ERQA C1 冷/暖 216/216、p95 289.26/244.66；C16 冷/暖 218/218、7.281/8.627 dec/s；OD Cap 74.2477/74.0625/74.2322，ERQA 0.5400/0.5450，全部 2PP 内、0 失败/重试。
