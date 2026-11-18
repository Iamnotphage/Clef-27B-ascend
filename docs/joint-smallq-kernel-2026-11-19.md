# Joint head 小 Q 自定义融合核（候选，默认关闭）

2026-11-19 首阶段：已完成实际源码核对、独立 Triton Ascend kernel、hash-lock 安装器、真实权重设备 probe、CPU 检查与 A3 离线编译。首设备窗口现由唯一 owner 27 完成：冻结数值门全部通过，整头三长度全部退化，维持默认关闭；源码回退和恢复四检已闭合。早期协调由 21 完成，交接后设备执行统一归 27，使用 24/23 核验的完整 wrapper。禁止与 24 的 head postimage 混装。

## 实际接口与数值语义

服务 preimage `f8101b98228bbd472a63a1efe6827583e35c6cc9dede4379b925f2b016670863`。真实 `joint_head_config.json`：hidden 5120、width 1024、heads 16、head_dim 64、routing 2 层、decoder 4 层、FF 4096。权重 safetensors header 确认 BF16，routing in_proj `[3072,1024]`。

单字段有 2 个 `[1,O,1024]` routing query 和 4 个 `[1,1,1024]` decoder query。K/V 为 `[1,L,1024]`；全长非因果 attention，scale 0.125、无 mask、dropout 0、无 attention weights。保留 query/key/value 投影、out projection、norm、FFN、残差与 schema/span 全部原计算。

routing 原实现两次 memory LayerNorm 产生不同 K/V 对象，走独立 K/V projection；decoder K/V 共用对象，走 packed KV projection。本候选继续调用实际 PyTorch `F._in_projection_packed`，保留该身份区分与序列优先输入 rank。

安装的 torch_npu 2.10.0.post7 git 为 `3363ea59d49f3e08cfd8d4a1f532b756ccda0682`，其 op-plugin submodule 为 `79a42714ebf7662b5a3f7361f2b86ebb3dc4a1e3`。[固定源码](https://github.com/Ascend/op-plugin/blob/79a42714ebf7662b5a3f7361f2b86ebb3dc4a1e3/op_plugin/ops/opapi/ScaledDotProductAttentionKernelNpuOpApi.cpp) 的 V2R5+ 分支在这些形状、`force_math=false`、A3 下调用 `npu_fusion_attention_v3`，`inner_precise=0`。实际原路径由 probe 记录兼容选项与 SDPA 开关，并以实际 SDPA 输出作为参考。底层 softmax/乘法的中间舍入由后端决定，候选与原输出的差异由设备 gate 衡量。

## kernel 与覆盖域

`runtime/joint_smallq/clef_smallq_kernel.py` 中 `clef_smallq_online` 为实际自定义 Triton kernel：每个 `(head, query)` 一个 program，64 token 分块，FP32 QK 点积、在线 max/exp/sum、FP32 PV 累加，最终写 BF16 BSH。每处 attention core 一次 launch，无全局 score/probability 中间张量。使用当前输入即时计算，未缓存答案、权重或激活。

首阶段后的服务化修正将 L 改成运行时 i32，并设 `do_not_specialize=["L"]`，全部支持长度共用一个 binary，避免 OD 大量长度逐个 JIT。Q 数目只改变 launch grid。该版本离线编译 2.591547 秒成功，kernel SHA `08ad88f6cebfb148aff8d973097d5bcf234e25d144a824385b18b8b078e82dff`，binary SHA `4fbed082933ae2fcb597ad758384fb707f5ce5ff66bd3546a13a4676c682ef44`。下表保留首版 constexpr L 的编译证据；设备首跑统一使用运行时 L 版本。

`CLEF_JOINT_SMALLQ=1` 才启用。域为单字段推理、eval、梯度关闭、B1、BF16、width1024/head16、Q1–8、L1–2048、最后一维连续、同设备、无 bias_k/v/add_zero_attn。其它情况返回原 MHA；默认关闭路径不导入 Triton/新 kernel。多字段原 decoder 路径保持完整。长 ERQA 回退原路径。

收益预算：相对独立 QK/softmax/PV，core 可由至少 3 launches 降为 1，免写读 score/probability；O4、6 层、L967 时，两个中间量共 `192×967` 元素，各一次写读约 1.49 MB（BF16）或 2.97 MB（FP32）。原 SDPA 常用分支已有融合，上述预算仅适用于 math 分支；本窗口主要验证小 Q 的 vector tiling、直接 BSH 写回与通用 attention 调度成本。Q1 仅 16 个 program，QO 重读 K/V，均可能抵消收益。真实 head 约 5 ms，最终采用需稳定同窗口 S1/S2 收益及完整质量通过。

## 已完成检查

- CPU 数学在线归约：Q1/Q4，L=1/63/64/65/307/967/1595/2048，正常及高动态 logits；FP32 参考 `rtol=3e-5, atol=3e-6` 通过。
- CPU 实际 head 定义对照：默认关闭、CPU 域外、单/多字段均与 preimage 逐位一致；未导入 torch_npu/Triton。
- 安装器临时树 apply/check/idempotent apply/revert/idempotent revert/未知哈希拒绝通过。安装器仅标准库，前后核验两个文件 SHA，保留真实 preimage backup。
- Python 语法检查通过。远端仅 CPU 测试使用 `TORCH_DEVICE_BACKEND_AUTOLOAD=0`，3 tests 全通过；本地无 torch，安装器测试通过、两项 torch 测试跳过。
- `joint_smallq_compile.py` 显式 `GPUTarget("npu","Ascend910_9382",0)`，替换编译器硬件查询为固定 metadata，不初始化驱动、不加载设备二进制。使用独立 scratch cache，L307/967/2048 全部离线编译成功。

| L | 离线编译秒数 | npubin SHA-256 |
|---|---:|---|
| 307 | 2.815477 | `3a0d2467a1cf010486369a2a02b93e82eb33083e167703d1453debb9963862eb` |
| 967 | 2.576837 | `2977a8a6f65e2ffee214ee0adcf5ab619bff1d65bca715310961099905651992` |
| 2048 | 2.352388 | `e17adbf3bebc648ec7725700d4b6464e2a589cd85e76fc6dc37c84204c0995b9` |

## 冻结设备预筛与执行接口

安装器 `runtime/apply_joint_smallq.py` 支持 `--tree --backup-dir --check/--apply/--revert --manifest`。payload 必须同时包含 `joint_smallq/clef_joint_schema.py` 和 `joint_smallq/clef_smallq_kernel.py`，保留相对布局。统一 wrapper 负责互斥安装、停服、进程回收、恢复；本候选没有停服/恢复脚本。

21 在 wrapper 控制的候选窗口内执行的单条 probe 命令：

```bash
/workspace/yangchen/venvs/clef-py312/bin/python runtime/joint_smallq_probe.py \
  --tree /workspace/yangchen/work/src/vllm --device 0 \
  --lengths 307,967,1595 --options 4 --seed 20261119 \
  --core-atol 0.0078125 --head-atol 0.03125 --prob-atol 0.005 \
  --warmup 5 --iters 20 --output /ABS/NEW/RUN/head-probe.json
```

这些门在首跑前冻结：core 最大绝对误差 ≤0.0078125、head logits ≤0.03125、softmax 概率 ≤0.005，全部 finite；真实投影后六处形状和触达顺序必须为 `[O,O,1,1,1,1]`。多字段与 L2049 回退要求逐位一致、零 kernel 触达。它们是合成输入、真实权重的工程预筛门。正式准入仍要求完整 OpenDecisions Intelligence/Calibration/Capability 与 ERQA accuracy 相对 baseline 各≤2pp。

所有 gate 先于 timing。通过后测 core 与完整 head，AB/BA/BA/AB，每轮每臂 20 个逐调用 wall/device 样本，inference_mode，预热 5 次；首调用含 JIT 的时间独立列出。报告保存真实权重、输入与源码 hash、dtype/stride、max/mean/RMS/relative 差异、逐位变化数、模块路径和 SDPA 配置。任何编译/执行/精度错误落盘且返回非零。`schema=joint-smallq-probe/1`、`gate_passed=true` 且无 `error` 才通过。

S1/S2 与全量 OD/ERQA 尚未运行，当前整头负收益不进入服务 AB/全量质量。06:30 CST 停止新候选，后续用于最终验证和恢复。


## 2026-11-19 02:34–02:36 CST 首设备窗口：整头负收益

冻结候选 `2e8af5d59fc88de8922276ed4f55db187b7e2fea`，27 使用 5c wrapper 执行，23 与 26 均从原始样本独立重算。原始目录 `/workspace/yangchen/logs/joint-smallq-27-2e8af5d-first/`。仓库保存 [原件 manifest](../Clef-27B/logs/joint-smallq-20261119/first/manifest.json)、[完整 probe](../Clef-27B/logs/joint-smallq-20261119/first/head-probe.json.gz) 和 [配对重算结果](../Clef-27B/logs/joint-smallq-20261119/first/paired-summary.json)。gzip 解压后字节保持原件，probe SHA-256 `25686c9de1f1f5eb20c9a5be5b16eb1b2688a1aa15519be6ef19870f63dc6cad`。

### 数值与触达

- 6 个 core 格全部 finite，max_abs ≤0.001953125；18 个真实投影后 attention 调用均触达，三个长度各 `[4,4,1,1,1,1]`，BF16，q/k/v stride 均 `[1024,1024,1]`。
- 307/967 的完整 head logits 与概率逐位一致；1595 的 4 个 logit 中 1 个变化，max_abs=0.0009765625，概率 max_abs=0.00019699335098266602，全部通过冻结门。
- 多字段 L307/Q2/O8 与长序列 L2049/Q1/O4 均零 kernel 触达、ON/OFF 逐位一致。
- 真实 checkpoint SHA `a010ac04f078e699988e4049cbea5e62c962393f59fec366640b64e8d69a4953`；输入是固定 seed 的合成 hidden/lexical。质量结论限于此次预筛。
- 模块记录 torch `2.10.0+cpu`、torch_npu `2.10.0.post7`、triton `3.6.0`；flash/math/mem-efficient SDPA 开关均 true，COMPATIBLE_IMPL/NPU_COMPATIBLE_IMPL 均未设置。只读 dist-info 另确认 triton-ascend 发行包为 `3.6.0.post2`，固定环境保持原样。
- 首次 JIT wall=3714.304550 ms 单列；后续不同 L 首调用 0.241–0.278 ms，未混入稳态样本。该证据支持运行时 L 未逐长度再次产生秒级编译。

### 配对原始计时

每格四轮 AB/BA/BA/AB，每轮每臂 20 样本、每臂共 80；全部 raw 样本有限且为正。以下每个 delta 先从同轮原始 20 样本求 A/B 各自中位数，再算 B−A，最后报告四个配对 delta 的中位数与 min/max；百分比同样逐轮配对。单位 ms，负数表示加速。此处的 device 是 NPU event 之间的流时间，包含主机未及时提交产生的空隙，不能当成纯 kernel compute。

| 范围 | L | Q | device Δ中位 [min,max] | wall Δ中位 [min,max] | device / wall 配对变化 | 同向轮数 |
|---|---:|---:|---|---|---|---|
| core | 307 | 1 | -0.029190 [-0.034410,-0.026330] | -0.036215 [-0.041300,-0.032210] | -23.46% / -16.19% | 快4/4 |
| core | 307 | 4 | -0.063095 [-0.065840,-0.061790] | -0.059882 [-0.062310,-0.058815] | -41.15% / -24.47% | 快4/4 |
| core | 967 | 1 | -0.042280 [-0.043260,-0.040490] | -0.043498 [-0.045830,-0.041770] | -32.86% / -19.16% | 快4/4 |
| core | 967 | 4 | -0.054865 [-0.056240,-0.053760] | -0.052645 [-0.053840,-0.050850] | -34.24% / -21.24% | 快4/4 |
| core | 1595 | 1 | -0.037155 [-0.037880,-0.036180] | -0.027660 [-0.031420,-0.026415] | -28.42% / -12.78% | 快4/4 |
| core | 1595 | 4 | -0.031550 [-0.033370,-0.030080] | -0.026165 [-0.031195,-0.025590] | -19.43% / -10.45% | 快4/4 |
| head | 307 | O4→Q1 | +0.162455 [+0.145580,+0.175460] | +0.163127 [+0.139740,+0.191010] | +3.15% / +3.10% | 慢4/4 |
| head | 967 | O4→Q1 | +0.140460 [+0.068330,+0.306680] | +0.154033 [+0.052570,+0.294230] | +2.73% / +2.93% | 慢4/4 |
| head | 1595 | O4→Q1 | +0.119930 [+0.099700,+0.170420] | +0.122842 [+0.099050,+0.177340] | +2.36% / +2.37% | 慢4/4 |

完整 head 各轮 wall A→B 中位数依时间顺序为：

- L307 wall：5.262000→5.453010, 5.284630→5.441635, 5.282510→5.422250, 5.251790→5.421040；device：5.158250→5.333710, 5.170860→5.320870, 5.161240→5.306820, 5.131780→5.306680。
- L967 wall：5.275545→5.413725, 5.380230→5.432800, 5.263190→5.557420, 5.242425→5.412310；device：5.168410→5.295130, 5.245640→5.313970, 5.142500→5.449180, 5.136130→5.290330。
- L1595 wall：5.269255→5.391450, 5.081585→5.258925, 5.072920→5.171970, 5.111665→5.235155；device：5.159360→5.278120, 4.968290→5.138710, 4.963520→5.063220, 5.003880→5.124980。

967 的配对范围最宽，wall +0.052570～+0.294230 ms、device +0.068330～+0.306680 ms；四轮仍全部退化。307/1595 也各 4/4 退化。以配对轮次判断，当前结果不支持把差异归为随机正负波动。

### 归因边界与最小可证伪下一步

已证事实：core 六格全部加速，整头三长度全部变慢；首编译已排除，六处触达与回退正确，原始投影/norm/残差计算保留。将 4×Q1+2×Q4 的隔离 core delta 直接相加，会预测约 −0.243/−0.279/−0.212 ms，实际整头为 +0.162/+0.140/+0.120 ms。这种差别说明每调用 synchronize 的独立 core 基准无法直接相加预测完整 head 的关键路径。

源码层面可提出两个待证假设：

1. 当前 helper 每次重复检查 module 配置、三组 tensor metadata、四个投影参数，执行多次视图变换，再进入通用 Triton JIT launcher；完整 head 每请求重复 6 次。额外主机调度可能抵消已经被其它操作提交掩盖的 core 节省。
2. Triton AIV kernel 与前后投影/FFN 的排队、资源使用或缓存交互可能改变整头流时间。当前 event 数据无法把这些因素与 host gap 分开，不能宣称已经证明单一 guard 或 tiling 根因。

有限建议只安排一个诊断改动，冻结 kernel/数值门不动：在独立诊断 probe 增加 C 臂，沿用现有 helper 的所有 guard、投影和输出投影，仅将 smallq core 切回相同 projected Q/K/V 上的原 SDPA。A=原 MHA、B=当前 smallQ、C=当前 adapter+原 SDPA，同一权重/输入、三长度、交错成对计时，先过原精度门。A↔C 估计接入路径的净影响，C↔B 检查同一 adapter 下 kernel/布局替换在整头的净影响。

- 若 C 相对 A 也稳定变慢，才针对已定位部分提出把静态 module 检查前移、每请求域检查合并的候选；保留动态 dtype/device/training/fallback 安全检查。
- 若 C≈A 且 B>C，优先检查 Triton 启动与整头流的交互，再决定是否改变 tile/分工；当前数据没有支持先改 BLOCK 或 split-K 的证据。
- 当前三个完整 head 长度都慢，生产采用全域原路径（默认 off）。Q1-only、routing-only、短 L 子域均没有完整 head 消融实测，保持待验证状态；core 的正收益不用于放行这些域。
- 由 20 排程后仍仅由 27 执行新诊断；当前按计划转 N1，不重跑首格。只有整头同窗口稳定收益及冻结门通过后才考虑服务 AB，S1/S2 与全量 OD/ERQA≤2pp 必须另行实测。

### 与 24 预筛的对照范围

26 使用真实 joint_head.safetensors；24 probe 使用 normal(0,0.02) 随机 head 参数。26 OFF 保持 f810 的 routing 两次 memory LayerNorm、不同 K/V 对象及 unpacked projection；24 aa3e postimage 的 OFF 路径先共享一次 memory norm 的 kv_in，MHA 会选择 packed KV。两套 OFF 的工作量与权重范围不同，绝对 head 耗时和百分比不能跨窗口直接归并。24 当前服务 AB 两臂同为 aa3e，仅切换 flag，仍可判断该源码内的 flag 增量，按原计划继续。最终相对生产 f810 的总收益与质量保留独立来源对照，由 20 排程后续最小验证；本结论范围说明保持当前服务 AB 流程。

### 生命周期闭合

window overall_rc=0，probe rc=0、gate_passed=true、error缺失；cleanup reaped=true，unknown/residual/killed 均空。revert 原件恢复 f810，smallQ kernel 移除；restore 四项 health/workers/text/image 均 rc0、ok=true。最终 label `d0-restore-20261118T183442Z-d3301487`，root1193641/start394386882，workers1193787/1193788 属恢复树且使用卡7两个逻辑设备。

原件 SHA：window `4d4f99f5d5bf0bd0b39c4cb56cc10d1ced8285e4c861e9631fe0270b45ad3f18`；record `b943da38d7025b8ff332fee60308e3694e7f27471ecc13ab30ecd434cad2733d`；revert `d500245a137a48be3ce53cbe12e119e86de38b9703dda8966874b6a9aa3ee64f`；restore `a0b953c2577bc611f3896a4863b5abaebfd775bfa7f32e32a511534e089ebafc`。完整压缩副本与哈希清单位于同一日志目录。

纯 CPU 重算命令：

```bash
python3 runtime/analyze_joint_smallq_probe.py \
  Clef-27B/logs/joint-smallq-20261119/first/head-probe.json.gz \
  --expect-sha 25686c9de1f1f5eb20c9a5be5b16eb1b2688a1aa15519be6ef19870f63dc6cad \
  --output /tmp/smallq-paired-summary.json
```

## A/C/B 真实诊断（2026-11-19 03:12–03:14 CST）

27 执行 `13fd807a2349aad36f3cf972dc82dbc118c01f17` 的独立 profile，B 四冻件仍为 `2e8af5d`。完整原件压缩副本、SHA 清单、逐轮重算统计在 `Clef-27B/logs/joint-smallq-20261119/acb-v1/`。probe SHA256 为 `79d115accb3822425db30c40ccd11775213e45ed5e3ffbecbf5d46c09c203caa`。

A 使用同一 head 实例、同一真实 checkpoint 参数存储，绑定 f810 的三处原方法，运行原 MHA。B 绑定冻结 smallQ 方法与 Triton backend。C 明确在 probe 进程内替换 `clef_smallq_kernel.smallq_attention` 模块属性为原 SDPA，保留 B 的 adapter、guard、projection；BSH↔BNSD 由 probe 完成。切臂、hook 和 hash 检查均在稳态计时外。三臂相同输入/权重，A↔C 衡量接入与投影/布局路径的净差异，C↔B 衡量同一 adapter 下的 backend 替换。

### 完整性与数值

独立分析器从原始 samples 重算 9 格×4 轮×3 臂×20 = **2160** 个正式样本，全部有限且为正。顺序固定 `ACB/BCA/CAB/BAC`，每对正反各两轮；记录内每个 median 与 paired_delta 均逐项精确一致。三长度六 core + 三 head + 两 fallback 共 11 组 gate 完整。A 六个 MHA 的 Q 为 `[4,4,1,1,1,1]`，K/V 对象身份为 `[false,false,true,true,true,true]`；B/C 六个替代 backend、零原 cross-MHA。

- core 最大绝对差 `.001953125`；head 最大 `.0009765625`；prob 最大 `.00019699335`，均通过原冻结门。
- A/C 三个完整 head logits/prob 均逐位一致；B 在 307/967 逐位一致，1595 有上述小差异。
- 两个 fallback 所有臂逐位一致、零替代 backend。same_weight_storage_verified、frozen_files_unchanged 均为 true。
- B 第一调用 wall **217.33546 ms**，包含首次编译或缓存查找路径，单列且未混入正式样本；首格原实验首次 JIT 3.714s 的证据保留。
- 分析器另验证缺样本、错顺序、NaN、改写 delta、fallback 不等、输入 hash 不符、旧双臂 experiment 共 7 类原件变异全部拒绝。

### 同轮配对结果

下表为四个“该轮两臂各20次的中位数之差”的中位数，单位 ms。每轮差值、极值范围、百分比、五类计时字段均保留在 `paired-summary.json`；结论使用同轮配对，未用跨轮 pooled 值。

| scope / L / Q | C−A wall | C−A device | B−C wall | B−C device | B−A wall | B−A device |
|---|---:|---:|---:|---:|---:|---:|
| core / 307 / 1 | +.001673 | +.001070 | −.042703 | −.036600 | −.041233 | −.035530 |
| core / 307 / 4 | −.000578 | −.000760 | −.058655 | −.067270 | −.059185 | −.068030 |
| core / 967 / 1 | −.001028 | −.000025 | −.046688 | −.046430 | −.047015 | −.047100 |
| core / 967 / 4 | −.001463 | −.000005 | −.031813 | −.047050 | −.033558 | −.047185 |
| core / 1595 / 1 | −.000295 | −.002220 | −.031625 | −.041740 | −.030118 | −.044035 |
| core / 1595 / 4 | −.000360 | +.000965 | −.018013 | −.039945 | −.019058 | −.036665 |
| head / 307 / 4 | +.153818 | +.158200 | +.015033 | +.014085 | +.168850 | +.172285 |
| head / 967 / 4 | +.170415 | +.173040 | +.038173 | +.041915 | +.200400 | +.211025 |
| head / 1595 / 4 | +.151345 | +.161305 | +.056908 | +.050525 | +.208253 | +.211830 |

六个 core 的 B−A、B−C wall/device 全部 4/4 更快；A/C core 同函数控制差异为数微秒、总体方向混合。完整 head 的 C−A 与 B−A wall/device 三长度均 4/4 更慢。完整 head 的 B−C 每长度均 3/4 更慢，wall 差值范围依次为 `[-.074600,+.044465]`、`[-.088780,+.048395]`、`[-.083200,+.059785]`，不足以声称 backend 本身在整头稳定拖慢。

完整 head C−A 的 wall 差值范围为 `[+.131450,+.179860]`、`[+.128315,+.286845]`、`[+.116235,+.322190]`；host_submit 中位差为 `+.155313/+.178628/+.163035 ms`，三长度也均 4/4 更慢，与 wall/device 增量同量级。device event 区间包含 stream gap；host_submit 包含 Python/框架调用与可能的内部同步。这些字段支持“接入/投影路径存在净开销”，无法单独归因纯 Python、某条 guard 或独立的设备 gap。

### 唯一后续假设与否证方式

下一最小变量建议为 **guard 元数据读取去重**：每次调用内复用已读取的 shape/device/stride；decoder 的 K/V 为同一对象时只校验该对象一次。每次调用仍读取训练/grad、flag、参数 dtype/device 与完整原域条件；不做跨调用缓存，不改变算子、投影身份、tile、布局、域或数值门。当前 guard 重复读取 query.device、各 tensor.shape，并对共享 K/V 重复做相同检查，这是可从源码确认的工作量。全部损失能否通过这项改动回收仍待设备实测。

以冻结 B 为对照的新候选 D，只替换 guard 实现；先 CPU 逐条边界/动态切换检查，再由 20 排程、27 运行相同真实权重/输入的独立 B/D 整头微基准。若三长度任一方向不稳定，或完整 head 相对 f810 仍无稳定收益，即否证该项足以支撑服务化的假设；所有原门、fallback 和清理恢复条件继续适用。当前只提出该单变量，未创建或运行新设备窗口。

生产全域 smallQ 默认 off；当前数据继续不支持采用或升级服务 AB。完整 OD/ERQA≤2pp 与 S1/S2 均未对 smallQ 实测。24 的随机权重/aa3e OFF 基线与本轮真实权重/f810 A 的绝对耗时保持分开。

### 生命周期与复算

window `d512383a267104546ec9d50d4cf1f35821ef3cc338d267a0988171ed0246b304` overall_rc=0；probe record cleanup rc0/reaped=true，unknown/residual/killed 均空；revert 回 f810 并移除 smallQ kernel。restore `461e13bd74d7b8372f724879e4d3688708d4c972f829c448b70fc48c54f61f4c`，label `d0-restore-20261118T191302Z-5067eaf0`，root1216295/start394616796，workers1216463/1216464 同恢复树且在卡7两chip；health/workers/text/image 四项 rc0、ok=true。

```bash
python3 runtime/analyze_joint_smallq_acb.py \
  Clef-27B/logs/joint-smallq-20261119/acb-v1/head-probe.json.gz \
  --expect-sha 79d115accb3822425db30c40ccd11775213e45ed5e3ffbecbf5d46c09c203caa \
  --output /tmp/smallq-acb-paired-summary.json
```

## B/D 元数据去重实测（2026-11-19 03:31–03:33 CST）

27 按 `9b803fefba4d1aa3beab1125bb4cbc7a3daae12f` / profile `71f7e7a190189ead00df6a6e9d3b5201a7df1c081aef1807a84629958651bf3b` 执行单一新窗口。原件归档 `Clef-27B/logs/joint-smallq-20261119/bd-v1/`；probe SHA `8aea817c7b220a8c681578a39d9ee4bc6bf75cb782cb68c879278fdc9feadfcd`。D 仅每调用元数据读取去重，共享 K/V 校验一次；B 四冻件、projection tail、layout、kernel 和支持域保持。

专用标准库分析器核验 **1440 raw×5字段**全有限且为正、9格×4轮×2臂×20、顺序 `BD/DB/DB/BD`、每轮记录中位数与全部配对 delta 精确匹配。首调用与稳态分列，首个 B wall261.39744ms 属首次编译或缓存查找路径，未计入正式比较。10种损坏原件（缺样本、错顺序、NaN、改写delta、fallback不等、输入hash、sourcehash、D/B不等、defaultoff不等、错误experiment）全部拒绝。

11组原门通过：A参考下 core max_abs≤.001953125、head≤.0009765625、prob≤.00019699335；**所有D/B gate逐位一致**。D defaultoff 对 f810 逐位一致且禁止 backend 调用；两fallback所有臂逐位一致、零backend；A六MHA及K/V身份、B/D六backend、同权重storage、单变量AST、冻件与九组输入哈希全部通过。真实 checkpoint 与前两窗相同，输入为合成BF16。

### 三个完整 head 的同轮配对

单位 ms；四轮各臂20样本取中位数，再计算 D−B。所有下列 wall/device/host_submit 差值均 **4/4为负**。

| L | wall中位差 | wall范围 | wall百分比中位 | device中位差 | device范围 | host_submit中位差 |
|---|---:|---|---:|---:|---|---:|
|307|−.0458875|[−.078400,−.023630]|−.9631%|−.0439849|[−.089070,−.024690]|−.0428275|
|967|−.0369825|[−.204970,−.026035]|−.7922%|−.0390600|[−.156880,−.023360]|−.0388450|
|1595|−.0495500|[−.254780,−.021020]|−1.0329%|−.0471551|[−.219700,−.023100]|−.0464350|

wall四轮原差值：307 `[-.078400,-.052900,-.038875,-.023630]`；967 `[-.204970,-.026165,-.047800,-.026035]`；1595 `[-.066585,-.254780,-.021020,-.032515]`。个别轮次有较大负差，结论使用四轮配对中位数；所有绝对臂中位数、百分比、范围及五字段逐轮差值均在 `paired-summary.json`。

### 相同 core callable 的控制结果

B/D core均直接调用同一kernel；以下差值单位微秒，表示本窗波动。

| L/Q | wall中位差 | device中位差 | wall快轮/device快轮 |
|---|---:|---:|---|
|307/1|−.8725|−1.5050|3/4，3/4|
|307/4|+.3175|+1.6100|2/4，0/4|
|967/1|−.7950|+.2700|2/4，1/4|
|967/4|−.9925|−.5800|3/4，3/4|
|1595/1|+1.2400|+.4750|1/4，1/4|
|1595/4|−.2575|−.6850|2/4，3/4|

整头 metadata 去重相对B稳定回收约 **37–50μs**，host_submit 与 wall/device 同向；event含stream gap、host_submit可能含框架同步，仍保留其测量边界。core控制中的小差异不作为kernel改进证据。

本轮 A 只做数值参照，**D相对生产f810尚无稳态总收益证据**；保留各窗独立结果，禁止用旧A/C/B减本窗D/B推算总收益。生产smallQ继续默认off，不升级服务AB；S1/S2与全量OD/ERQA≤2pp尚未实测。

### 恢复闭合与复算

window `862af0c08338c8032cc98ed0d85f5b27c709aa3598206f1b1d5a1739fd4df136` overall0、六步rc0/oktrue；record `5ff9b005470215128229851aa38c2ce4d46a59e19cada152413b4acf8671c500` cleanup rc0/reaped、unknown/residual/killed均空；revert `616e901a9aeed6e3be32be713ef46fd3039c58eca4e89d1f392f00e42207e2b0` 回f810。restore `d6c33ba56886893d8904e6ce739e4cc4927b1bbd4987dc26730f038bf4cbd7d0` 四检rc0/oktrue；label `d0-restore-20261118T193157Z-1b3839a2`，root1220123/start394730336，workers1220262/1220263。

```bash
python3 runtime/analyze_joint_smallq_bd.py \
  Clef-27B/logs/joint-smallq-20261119/bd-v1/head-probe.json.gz \
  --expect-sha 8aea817c7b220a8c681578a39d9ee4bc6bf75cb782cb68c879278fdc9feadfcd \
  --output /tmp/smallq-bd-paired-summary.json
```

## Packed-KV 路径核验：E 假设关闭

20 要求在 CPU 侧核查 D decoder 是否丢失 f810 的 shared-KV packed 投影。实际固定源码与 CPU 调用跟踪均表明：**D 与 f810 已使用相同 packed-KV 分支**。该差异不存在，关闭“恢复 packed-KV”这一 E 假设，B/D 保持冻结，不创建 E。

实读链路：f810 head 第255–256行将同一个 memory 对象作为 K/V。固定 torch `nn/modules/activation.py` 第1453–1462行在 batch_first 转置后保持 `value=key`，并调用 `F.multi_head_attention_forward`；非 self-attention 在第1361行排除 native fastpath。`nn/functional.py` 第6435行调用 `_in_projection_packed`。D helper 第112–118行同样使用 `v_in=k_in`，调用完全相同的 helper。

`_in_projection_packed` 第5820、5835–5852行：K is V、Q is not K 时，权重拆成 `[W,W]` 与 `[2W,W]`，仅两次 linear：Q 和 packed-KV。routing 的两个 memory_norm 保持独立对象，两源均进入第5853–5860行的三个 linear 分支。

| cross-attention类型 | 每层输入projection linear/GEMM请求 | 每层out projection | 层数 | 两源总数 |
|---|---:|---:|---:|---:|
|routing Q=O、独立K/V|3|1|2|8|
|decoder Q=1、共享K/V|2|1|4|12|

因此两源的六次 cross-attention 都有14次输入projection、6次out projection，共20个linear/GEMM请求。4层单token self-attention另外各有V+out两个linear，两源也保持。上述为前端数学请求计数，不等同NPU内部launch计数。

布局方面，两源每个decoder都在同一packed helper执行 `unflatten/unsqueeze/transpose/squeeze/contiguous`，全头共4处KV拆包contiguous；routing输入projection没有该复制。D前后转置均为view，kernel直接写BSH，随后out projection。f810另有每cross输出 `permute(...).contiguous()` 的6处调用点；是否分配取决于实际SDPA返回stride。本次CPU代表Q1与Q4中该输出contiguous均复用原storage，不能据Python调用点数量宣称NPU发生6次复制。

纯CPU跟踪使用实际固定torch2.10.0+cpu、BF16、width1024、L307、Q1共享KV/Q4独立KV。对照为真实 `nn.MultiheadAttention` 调用与从冻结D提取的投影AST切片；该切片明确跳过guard/backend，用于核对投影分支。两者投影输出CPU逐位一致；Q1输入projection确为Q与 `[2048,1024]` packed-KV两次linear，Q4确为三次；packed-KV contiguous两路径都产生一个新storage。`torch_npu`/Triton均未导入，未产生NPU精度/性能结论。torch_npu安装树的该函数名称检索仅命中testing字符串，未发现Python MHA/projection替换。

源摘录、CPU trace与SHA清单在 `Clef-27B/logs/joint-smallq-20261119/projection-audit/`，复现脚本 `runtime/audit_joint_smallq_projection.py`。实读完整源文件SHA：functional.py `f6335e443ce93ddf21ecf9d94bbb4d4a4d92e128c5bfb59826f35495c9c28703`；activation.py `d2929845da0371704b4fa75d9ef487ac3842525e6a979c526016a740926aa900`。
