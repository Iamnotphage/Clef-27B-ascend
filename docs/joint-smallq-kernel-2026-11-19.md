# Joint head 小 Q 自定义融合核（候选，默认关闭）

2026-11-19 首阶段：已完成实际源码核对、独立 Triton Ascend kernel、hash-lock 安装器、真实权重设备 probe、CPU 检查与 A3 离线编译。当前无设备精度/性能结论；设备执行与恢复统一由 21 使用 24/23 核验的完整 wrapper 完成。禁止与 24 的 head postimage 混装。

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

S1/S2、全量 OD/ERQA 及恢复验证待设备窗口完成后补充。06:30 CST 停止新候选，后续用于最终验证和恢复。
