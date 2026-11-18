# GDN 输出 Norm 的 BLOCK_M 单变量准备（2026-11-18）

状态：**CPU 准备完成，未执行任何设备或服务测试**。四格由 21 在受限 E 采集之后串行排队；本文不报告任何
未实测收益，也不把 48 调用微基准批次写成每请求 48 层链。

## 1. 范围与冻结契约

- 唯一变量：`vllm_ascend/ops/triton/layernorm_gated.py` 的 `BLOCK_M`，取值 **16 / 32 / 64 / 128**，默认 **64**。
- 冻结项：wrapper 函数签名、`out`/`mean`/`rstd` 分配、`rstd` 保存、`BLOCK_N`、
  `grid = (triton.cdiv(M, BLOCK_M), ngroups)`、kernel 主体与启动参数；不新增 `num_warps`/`num_stages` 传参。
- 不重开：默认 chunk128、ConvQkv 吸收 Norm、beta 转置、E 算子与 `wave3_e_test.py`。本准备不修改这些方向。

## 2. 源码核准（只读实测）

远端固定树：`/workspace/yangchen/work/src/vllm-ascend`，HEAD `89eb8a5e7a76d085b453a1aeadb6883e12d5ea2d`。
该树整体为 dirty（其他会话的 GDN 补丁），但目标文件本身 **git clean**。

| 项 | 实测值 |
| --- | --- |
| 目标文件 | `vllm_ascend/ops/triton/layernorm_gated.py` |
| 预镜像 SHA-256 | `68f61d7f13547a11ca38eb33c0cd560dc6a569ad97c7fbc6681d1370db70a774` |
| 后镜像 SHA-256 | `a1093e1290afd48b546830a9ae0639f945910f44f9e22f12111b67aa72f8fe6c` |
| 原 `BLOCK_M` | 第 163 行硬编码 `BLOCK_M = 64`，无环境变量或接口旋钮 |
| `BLOCK_N` | 第 158 行 `min(65536 // element_size, next_power_of_2(group_size))`，本形状为 128 |
| grid | 第 166 行 `(triton.cdiv(M, BLOCK_M), ngroups)`，`ngroups=1` |
| 启动参数 | 第 167-187 行未传 `num_warps` / `num_stages`；Ascend `NPUOptions` 默认 `num_warps=32`、`num_stages=2`，实际值由探针从编译 metadata 读取 |
| kernel | 第 27-119 行 `_layer_norm_fwd_1pass_kernel_npu`；wrapper 第 121-188 行 `layer_norm_fwd_npu` |

生产调用链（逐段读取确认）：

1. `vllm_ascend/ops/gdn.py:654-658`：`core_attn_out.reshape(-1, head_v_dim)`、`z` 同形，`self.norm(core_attn_out, z)`。
2. `RMSNormGated` 由 `vllm_ascend/utils.py:904` 注册为 `AscendRMSNormGated`；Ascend 作为 OOT 平台走
   `forward_oot`（`vllm_ascend/ops/layernorm.py:207-219`）。
3. `LayerNormFn.forward`（`vllm_ascend/ops/layernorm.py:123-167`）reshape 为 2D 并调用 `layer_norm_fwd_npu`。
4. 模型侧 `qwen_gdn_linear_attn.py:696-703`：`RMSNormGated(head_v_dim, eps=layer_norm_epsilon,
   group_size=None, norm_before_gate=True, activation=output_gate_type)`。

模型 config（`/workspace/yangchen/models/Clef-27B/config.json`，`text_config`）：

- `num_hidden_layers=64`，`layer_types` 中 `linear_attention` **48 层**、`full_attention` 16 层。
- `linear_num_value_heads=48`、`linear_value_head_dim=128`、`linear_num_key_heads=16`、
  `linear_key_head_dim=128`、`rms_norm_eps=1e-6`、`output_gate_type=swish`。

真实形状（TP2，与 23 交接一致）：

- `core_attn_out` 与 `z` 展平为 **`[23208, 128]` BF16 contiguous**，`weight[128]` BF16，`group_size=None → N=128`，
  `ngroups=1`，RMS（`is_rms_norm=True`），`norm_before_gate=True`，activation silu，`bias=None`，`eps=1e-6`。
- `23208 = 967 token × 24 value heads`（48 heads / TP2）。**不使用** M=967、N=3072 或 groups24 的旧口径；
  48 是模型 GDN 层数，不能据此把微基准批次写成真实每请求链。

四臂派生量：

| 臂 | grid blocks | padding rows |
| --- | ---: | ---: |
| BM16 | 1451 | 8 |
| BM32 | 726 | 24 |
| BM64 | 363 | 24 |
| BM128 | 182 | 88 |

`grid` 是 wrapper 派生请求；Ascend 后端可能通过 `coalesce_factor/axis` 在 launcher 侧收缩网格，
探针会记录这些 metadata 字段，不用它替代实测，也不把请求 grid 写成实际 launch blocks。

## 3. 交付物

| 文件 | 作用 |
| --- | --- |
| `runtime/apply_gdn_norm_block_m.py` | 哈希门控安装器：`--check/--apply/--revert/--print-diff/--manifest`，preimage/postimage SHA 校验，原子写与 preimage 备份 |
| `runtime/gdn_norm_block_m/layernorm_gated_preimage.py` | 上游文件逐字节副本（SHA-256 `68f61d7f…a774`），供离线测试与审阅 |
| `runtime/wave3_gdn_norm_block_m_probe.py` | 四臂探针：共用输入、48 调用微基准批次、≥3 轮预定交错、BM64 配对差、编译 metadata 与 specialization 校验、逐元素/finite/rstd 比较 |
| `runtime/test_gdn_norm_block_m_patch.py` | 纯 CPU 离线测试：哈希、patch/unpatch、后镜像语法、CLI 状态机、未知内容拒绝 |

精确 patch（唯一行为改动，默认值不变）：

```diff
+import os
+
 import torch
 from vllm.triton_utils import tl, triton
@@
     # Choose BLOCK_M: e.g., 16, 32, 64 — depends on NPU vector core capacity
-    BLOCK_M = 64  # Tune this based on your NPU's register/shared memory
+    # Tune this based on your NPU's register/shared memory
+    BLOCK_M = int(os.environ.get("CLEF_GDN_NORM_BLOCK_M", "64"))
+    if BLOCK_M not in (16, 32, 64, 128):
+        raise ValueError(f"unsupported CLEF_GDN_NORM_BLOCK_M={BLOCK_M}")
```

环境变量未设置时 `BLOCK_M=64`；只允许四臂；非法值在进入 kernel 前报错，不会静默落到别的网格。

## 4. 21 的执行清单（四格）

前置：在 21 自己的受限窗口内执行，服务停止，单逻辑 NPU；先完成 E 采集排队。以下命令中的 `REPO` 指向
`perf/wave3-gdn-block-m-prep` 的检出目录，`TREE` 为固定 vLLM-Ascend 树。

```bash
PY=/workspace/yangchen/venvs/clef-py312/bin/python
TREE=/workspace/yangchen/work/src/vllm-ascend
REPO=<perf/wave3-gdn-block-m-prep 检出目录>
OUT=/workspace/yangchen/logs/gdn-norm-block-m-20261118

# 1) 只读核对：应为 preimage，ready=true，knob_present=false
$PY "$REPO/runtime/apply_gdn_norm_block_m.py" --tree "$TREE" --check

# 2) 安装旋钮：写备份并校验后镜像 SHA
$PY "$REPO/runtime/apply_gdn_norm_block_m.py" --tree "$TREE" --apply \
  --manifest "$OUT/apply.json"

# 3) 四格探针：默认 BM16/32/64/128，参考臂 BM64，3 轮交错
$PY "$REPO/runtime/wave3_gdn_norm_block_m_probe.py" --tree "$TREE" \
  --expect-source-sha a1093e1290afd48b546830a9ae0639f945910f44f9e22f12111b67aa72f8fe6c \
  --output "$OUT/probe.json"

# 4) 回退：恢复 preimage 并校验 SHA
$PY "$REPO/runtime/apply_gdn_norm_block_m.py" --tree "$TREE" --revert \
  --manifest "$OUT/revert.json"
$PY "$REPO/runtime/apply_gdn_norm_block_m.py" --tree "$TREE" --check
```

可选 `--per-call`：额外跑一批逐调用事件采样，只用于报告单次最大值；该批次含事件开销，**不参与四臂比较**。
未加该参数时只报告 `batch_max_device_ms`（批次间最大值），不报告单次最大值。

建议每次运行保持服务停止、同进程只跑该探针；探针在结束时移除 `CLEF_GDN_NORM_BLOCK_M`。

## 5. 计时协议（公平性口径）

- **批次定义**：每批 = `--calls-per-batch`（默认 48）次连续 wrapper 调用，称“**48 调用微基准批次**”；
  它不是一次真实请求的 48 层链，批次差值不能直接换算服务收益。
- **输入**：一份 `x/z/weight`，设备上只分配一次并由四臂共用；实际字节 SHA-256 写入 `inputs.sha256`。
  默认用固定 seed 的 BF16 合成输入与 `weight=ones`；如需真实权重/真实调用点样本，用 `--inputs <pt>`
  （dict：`x`、`z`、`weight`，可选 `eps`），四臂仍共用同一份。
- **buffer 与缓存状态**：`x/z/weight` 常驻复用；wrapper 每次调用按生产行为分配 `out` 与 `rstd`，
  返回元组被立即丢弃，同流上缓存分配器复用块。每个臂的 Triton specialization 在 warmup 阶段编译完成，
  进程内无其他 NPU 工作。
- **warmup / 同步边界**：每臂先跑 `--warmup-calls`（默认 48）次并 `torch.npu.synchronize()`；
  每个计时批次外同步一次、内不同步。环境变量只在臂之间切换，绝不在批次内切换。
- **指标分开记录**：`batch_device_ms` 是批次外一对 event 的差值；`batch_mean_device_ms = batch_device_ms / 48`；
  `batch_wall_ms` 是同步墙钟批次时间（含 host dispatch 与收尾同步），三者不混写。
- **交错与配对**：`round_orders` 在运行前写入 JSON，按臂列表逐轮旋转；每轮保留相对参考臂 BM64 的
  `paired_device_delta_ms` 与 `paired_device_ratio`，不做跨轮绝对中位数相减。
- **metadata**：从该臂实际选中的 Triton 缓存条目读取 `num_warps`、`num_stages`、`coalesce_*`、
  `shared` 与 `BLOCK_M` constexpr；缺字段记 `null` 并列入 `missing_metadata`，不以默认值代替实测。
  `specialization_ok` 断言实际编译的 `BLOCK_M` 与请求臂一致；不一致即失败，避免“切了变量但命中同一 kernel”。

判读顺序：先检查 4 个 specialization 与 finite/逐位记录，再看 ≥3 轮的配对差是否同向；任何单轮异常值不直接
当作结论。四臂都在噪声内则关闭该方向。

## 6. 正确性口径

项目既有 kernel 探针惯例（`runtime/clef_conv_qkv_probe.py`、`runtime/clef_qkv_norm_probe.py`）是
**finite + 逐位 + max_abs/mean_abs/rmse**，没有为“norm 换 BLOCK_M”建立独立的逐元素容差门。本探针：

- 输出 `out` 与 `rstd` 都相对同输入、同进程的 BM64 参考臂比较；
- 记录 `bit_equal`、`equal_elements/total_elements`、`finite_*`、`nan_*`、`inf_*`、`max_abs`、`mean_abs`、`rmse`；
- `mean` 在 RMS 路径应为 `None`，同时记录。

待定项（不临时发明门槛）：

1. 逐位一致是目标而非保证：BLOCK_M 不改变每行沿 N 的归约范围，但不同 tile 调度仍可能改变指令顺序；
2. 本 kernel 没有既有的逐元素容差准入值；是否接受非逐位结果由下游服务 A/B 与官方质量口径决定；
3. 项目内部 0.03/0.05 是**概率级**逐题门槛，不适用于本层输出，不能挪用；
4. 最终准入仍为：全量 S1 p95 不回归 + OpenDecisions/ERQA 三项指标 ≤2 pp；本探针不替代服务 A/B。

## 7. 假设与风险（均未实测）

- **BM128 的 UB/寄存器压力**：每 block 持有 `x/z/xbar/y` 的 FP32 tile，BM128 约 4×64 KiB，可能超过单核 UB；
  BM32/64 更宽裕。这是待验证假设，不作为收益或排除依据。
- **值分布**：本 kernel 无数据相关分支，预计耗时与值分布无关；未单独验证。四臂共用同一输入，配对比较不受影响。
- **合成权重**：默认 `weight=ones`，与训练后权重不同；只影响数值分布，不影响四臂比较。需要真实权重时用 `--inputs`。
- **后端网格变换**：`coalesce_*` metadata 可能表示 launcher 收缩网格；探针记录字段，不推断实际 launch blocks。
- **窗口干扰**：与 E 采集共用 NPU 会污染计时；必须串行，且探针进程内无其他负载。

## 8. 身份、证据与回退

- vLLM-Ascend 固定 `89eb8a5e7a76d085b453a1aeadb6883e12d5ea2d`；Triton Ascend `3.6.0.post2`；
  CANN `9.2.1`；torch_npu `2.10.0.post7`。
- 安装器拒绝未知内容；apply 前将 preimage 备份到
  `/workspace/yangchen/logs/gdn-norm-block-m-20261118/phase0/layernorm_gated.py.preimage`，
  revert 优先用该备份，缺失时按精确逆变换恢复并校验 preimage SHA。
- 探针 JSON 记录 source path/SHA、输入 SHA、协议、四臂 specialization、correctness、逐轮 timing 与 summary；
  每个阶段落盘，异常时保留已完成部分。
- 离线验证：`python3 -I runtime/test_gdn_norm_block_m_patch.py`（20 项检查，含 patch/unpatch、CLI 状态机、
  后镜像语法与未知内容拒绝），以及三个脚本的 `py_compile`。本准备未导入 torch/torch_npu/vllm_ascend。

## 9. 交给 21 的回报清单

请回传：`probe.json`、四个 `specialization_ok`、`num_warps/num_stages` 实测值、每臂 `out/rstd` 的
`bit_equal/finite/max_abs`、三轮 `paired_device_delta_ms` 与 `batch_device_ms/batch_wall_ms`，
以及运行时的 source SHA 与 `CLEF_GDN_NORM_BLOCK_M` 值。收到这些数据后再决定是否进入服务 A/B。
