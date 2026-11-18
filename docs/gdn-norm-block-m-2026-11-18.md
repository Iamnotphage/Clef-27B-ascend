# GDN 输出 Norm 的 BLOCK_M 单变量准备（2026-11-18）

状态：CPU 准备完成。设备侧已有三轮启动尝试，均未产生可用的四 BM 计时：run1 是 runner 未加载
`activate_env.sh`（`libhccl.so` 缺失），run2 是 probe 对 Triton decorator 的解包入口（`Heuristics` 无
`device_caches`），run3 是 **BM128 在固定栈下的 UB 编译容量限制**（BM64/16/32 已成功编译并走到准备阶段，
但结果尚未落盘、计时未进入）。三轮均已归档，**不构成四 BM 的数值或性能证据**。probe 已补按臂失败隔离与
逐阶段落盘，等待 21 在 E 采集和三检之后用新空目录单次重试。本文不报告任何未实测收益，也不把 48 调用
微基准批次写成每请求 48 层链。

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
| `runtime/wave3_gdn_norm_block_m_probe.py` | 四臂探针：共用输入、48 调用微基准批次、≥3 轮预定交错、BM64 配对差、编译 metadata 与 specialization 校验、逐元素/finite/rstd 比较；启动时先解包 Triton decorator，并按臂分阶段落盘与隔离可识别的编译容量失败 |
| `runtime/gdn_norm_block_m_env_check.sh` | CPU-only loader 预检：source `activate_env.sh` 后检查 `libhccl.so`、torch lib 与 `torch_npu` 扩展的依赖解析，不 import torch/torch_npu |
| `runtime/run_gdn_norm_block_m.sh` | 正式 runner 模板：在启动任何 Python 之前 source 环境，拒绝已存在输出目录，依次 check/apply/probe/revert |
| `runtime/test_gdn_norm_block_m_patch.py` | 纯 CPU 离线测试：哈希、patch/unpatch、后镜像语法、CLI 状态机、未知内容拒绝 |
| `runtime/test_gdn_norm_block_m_probe.py` | 纯 CPU 探针 helper 测试：decorator 解包、循环保护、四臂 grid/padding，且 import 不加载 torch |

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

前置：在 21 的受限窗口内、服务停止、单逻辑 NPU、E 采集与三检完成之后执行。**shell 必须先 source
`activate_env.sh`，再启动 Python**；不要用 Python 启动后修改 `LD_LIBRARY_PATH` 来掩盖 loader 问题。推荐直接用
runner（它已包含该 source、CPU 预检、单次新目录防护与无条件回退）：

```bash
REPO=<本分支检出目录>
OUT=/workspace/yangchen/logs/gdn-norm-block-m-20261118-run4   # 必须尚不存在
bash "$REPO/runtime/run_gdn_norm_block_m.sh" "$OUT"
```

等价的直接命令（供审阅）：

```bash
source /workspace/yangchen/Clef-27B-ascend/scripts/activate_env.sh
PY="${PYTHON_BIN:?}"
TREE=/workspace/yangchen/work/src/vllm-ascend
REPO=<本分支检出目录>
OUT=/workspace/yangchen/logs/gdn-norm-block-m-20261118-run4

bash "$REPO/runtime/gdn_norm_block_m_env_check.sh"          # CPU loader 预检
$PY "$REPO/runtime/apply_gdn_norm_block_m.py" --tree "$TREE" --check
$PY "$REPO/runtime/apply_gdn_norm_block_m.py" --tree "$TREE" --apply --manifest "$OUT/apply.json"
$PY "$REPO/runtime/wave3_gdn_norm_block_m_probe.py" --tree "$TREE" \
  --expect-source-sha a1093e1290afd48b546830a9ae0639f945910f44f9e22f12111b67aa72f8fe6c \
  --output "$OUT/probe.json"
$PY "$REPO/runtime/apply_gdn_norm_block_m.py" --tree "$TREE" --revert --manifest "$OUT/revert.json"
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
- **按臂状态与逐阶段落盘**：每臂依次写 `arm_status` 的 start / warmup / metadata / compare 阶段并立即落盘；
  `specializations`/`correctness` 从创建起即挂在 report 上，flush 不会丢掉此前的 metadata/逐元素结果。
  仅当异常本身是 `MLIRCompilationError`、显式 `__cause__` 链上有 `bishengir-compile` 的
  `CalledProcessError`、且文本同时包含 `ub overflow` 与 `requires N bits while M bits available` 时，
  才在 **warmup 阶段**识别为可隔离的编译容量失败；不跟随隐式 `__context__`。该臂记入
  `arm_status`/`arm_failures`（完整 error、完整 traceback、独立 `*.traceback.txt`、cache_dir），
  不进入计时，其余臂继续。其他编译错误、NonFinite、设备运行或同步异常一律 fatal：先置
  `status=fatal`/`ready=false` 并落盘，再抛出，不做宽泛吞错。
  BM64 参考臂失败时立即停止后续臂比较，剩余臂登记为 `not_run`，`paired_comparison_possible=false`，
  不进入计时与配对。
- **指标分开记录**：`batch_device_ms` 是批次外一对 event 的差值；`batch_mean_device_ms = batch_device_ms / 48`；
  `batch_wall_ms` 是同步墙钟批次时间（含 host dispatch 与收尾同步），三者不混写。
- **交错与配对**：预登记顺序写入 `preregistered_round_orders`，再按可执行臂投影为 `executed_round_orders`；
  每轮同时记录原始与实际顺序，并保留相对参考臂 BM64 的 `paired_device_delta_ms` 与
  `paired_device_ratio`，不做跨轮绝对中位数相减。
- **metadata**：从该臂实际选中的 Triton 缓存条目读取 `num_warps`、`num_stages`、`coalesce_*`、
  `shared` 与 `BLOCK_M` constexpr；缺字段记 `null` 并列入 `missing_metadata`，不以默认值代替实测。
  `specialization_ok` 断言实际编译的 `BLOCK_M` 与请求臂一致；不一致即失败，避免“切了变量但命中同一 kernel”。
  编译失败臂的 `specialization_ok` 记 `null`，不伪造为通过。

判读顺序：先检查 `arm_status` 与可执行臂的 specialization/finite/逐位记录，再看 ≥3 轮的配对差是否同向；
任何单轮异常值不直接当作结论。BM128 若仍为 UB 编译容量失败，只对其余可执行臂判读；三臂都在噪声内则关闭该方向。

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

- **BM128 已实测为固定栈编译容量限制**：失败 TTIR 为 `128x128` tile，要求 256.5 KiB 而 UB 可用 192 KiB；
  编译器已自动关闭 code-motion 与 multi-buffer 仍失败。该臂记录为 `compile_failed`，不进入计时；
  不通过改 `num_warps`/`num_stages` 或改 kernel 来强行凑四臂成功。
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
- 探针 JSON 记录 source path/SHA、输入 SHA、协议、`arm_status`、四臂 specialization、correctness、逐轮 timing
  与 summary；每臂 start/warmup/metadata/compare 分阶段落盘，异常时保留已完成部分。
- 离线验证：patch 测试 20 项、probe helper 测试 25 项、四个 Python 脚本 `py_compile`、两个 shell 脚本
  `bash -n`，全部通过。远端 CPU loader 预检在真实环境通过（`hccl_dir=/usr/local/Ascend/cann-9.2.1/lib64`）；
  不含 `libhccl.so` 的 stub 环境按预期失败。本准备未导入 torch/torch_npu/vllm_ascend。

## 9. 交给 21 的回报清单

请回传：`probe.json` 及其同目录 `*.traceback.txt`（如有失败臂）、`arm_status` 与 `arms_ok/arms_failed`
（BM128 预期为 `compile_failed`，参考臂失败时其余为 `not_run`）、四个臂的 specialization（BM128 预期带完整
error）、每臂 `out/rstd` 的 `bit_equal/finite/max_abs`、三轮 `paired_device_delta_ms` 与
`batch_device_ms/batch_wall_ms`，以及运行时的 source SHA 与 `CLEF_GDN_NORM_BLOCK_M` 值。收到这些数据后再决定
是否进入服务 A/B。

## 10. 三轮执行记录与修复（归档）

三轮都没有可用的四 BM 计时结果：run1/run2 失败在启动阶段，run3 失败在 BM128 编译；**没有四 BM 的数值或
性能证据**，失败格保留。

| 轮次 | 日志 | 时间（UTC） | 结果 | 首错 |
| --- | --- | --- | --- | --- |
| run1 | `/workspace/yangchen/logs/perf-wave3-20261118/e-test/run-gdn.log` | 14:07:27 起 | APPLY 0 / PROBE 1 / REVERT 0 | `torch_npu/_C` → `ImportError: libhccl.so: cannot open shared object file` |
| run2 | `/workspace/yangchen/logs/perf-wave3-20261118/e-test/run-gdn2.log`（SHA-256 `c0b0e544…3e21d14`） | 14:09:53 起 | APPLY 0 / PROBE 1 / REVERT 0 | 旧 probe 第 340 行 `AttributeError: 'Heuristics' object has no attribute 'device_caches'` |
| run3 | `/workspace/yangchen/logs/perf-wave3-20261118/e-test/run-gdn3.log`（SHA-256 `361d0a29…40f7b`） | 14:17:07 起 | APPLY 0 / PROBE 1 / REVERT 0 | BM128 `npubin_stage` UB overflow：requires 2101248 bits / available 1572864 bits |

run1 根因与修复：

- `e_run_gdn.sh` 没有 source `activate_env.sh`，进程继承不到 CANN 的 `LD_LIBRARY_PATH`；torch 自动加载
  `torch_npu` 扩展时在 `torch_npu/npu/utils.py:8 import torch_npu._C` 处失败，随后
  `RuntimeError: Failed to load the backend extension: torch_npu`。
- 修复放在 **Python 启动前的 shell**：runner 先 source
  `/workspace/yangchen/Clef-27B-ascend/scripts/activate_env.sh`，由它依次 source CANN `set_env.sh`、
  ATB `set_env.sh` 与 venv。`nohup`/`timeout`/Python 子进程自然继承；不在 Python 内改 `LD_LIBRARY_PATH`。
- CPU 预检脚本 `runtime/gdn_norm_block_m_env_check.sh` 只做 source + `ldd` 解析 + `sys` 版本检查，不 import
  torch/torch_npu；真实环境通过，缺 `libhccl.so` 的 stub 按预期失败。

run2 根因与修复：

- 模块属性 `_layer_norm_fwd_1pass_kernel_npu` 是两层 `@triton.heuristics` 包住的 `Heuristics` 对象。
  固定 Triton Ascend `3.6.0.post2` 源码：`triton/runtime/autotuner.py:449-459` 的 `Heuristics.__init__`
  保存 `.fn`，`run` 委托给 `.fn.run`；`triton/runtime/jit.py:787` 的 `JITFunction.device_caches` 与
  `:797` 的 `arg_names` 只在最内层 JITFunction 上。run2 把外层 `Heuristics` 当 JITFunction 用，因此在
  warmup 之前的 cache 读取处失败（run2 的旧 probe `dae7cd74…` 第 340 行；其 warmup 在第 349 行之后）。
- 修复只在 **metadata 入口**：`unwrap_jit_function()` 沿 `.fn` 逐层解包到有 `device_caches` 的对象，并记录
  `unwrap_chain`；找不到就带 `kernel_object` 信息早退。
- 计算调用继续走原来的 `layer_norm_fwd_npu` → 原装饰器 → `Heuristics.run`，heuristics 语义不变；探针没有
  绕过 wrapper，也没有把期望配置当成观测值。
- 编译后的 `num_warps/num_stages/coalesce_*` 与 `BLOCK_M` constexpr 仍从实际 `CompiledKernel` 读取；
  缺字段记 `null`，`specialization_ok` 仍断言实际编译的 `BLOCK_M` 与请求臂一致，未删除该校验。

run3 根因与修复：

- 失败 cache `SK3IPFDG6YCGOV4NFA2UAWAKERFTFZDKYGOINLZPWMQZHIJFM3GQ` 的
  `_layer_norm_fwd_1pass_kernel_npu.ttir` 第 24/31/32 行为 `c128`、`pid_m * 128`、`make_range ... 128`，
  故失败臂为 **BM128**。同一窗口 14:17:22–28 已依次生成 BM64 `64XTB…`、BM16 `U3WJ…`、BM32 `ZDLD…` 的
  `npubin` + `json`；三者 metadata 均为 `num_warps=32`、`num_stages=2`、`coalesce_factor=1`、`axis=-1`、
  `shared=1`、`mix_mode=aiv`。生产侧独立读取的 BM64 缓存条目 `T4ZH…`（mtime 2026-11-12 09:16，早于 run3）
  同样为 `num_warps=32`、`num_stages=2`、`coalesce_factor=1`、`axis=-1`、`shared=1`、`mix_mode=aiv`、
  `target=Ascend910_9382`。本次只比较了这些 cache metadata 字段，**不能**据此宣称 service 的全部编译选项或
  环境与探针相同；完整服务环境未逐项枚举。
- probe 顺序为 64,16,32,128；每臂 48 次 warmup + 同步 + compare 后才进入下一臂，flush 原在四臂循环之后，
  所以 run3 没有 `probe.json`，但前三臂的准备阶段已实际执行；计时从后续代码才开始，未进入；数值明细未保存，
  不追认任何过门结论。
- 编译器日志明确：先自动关闭 code-motion，再关闭 multi-buffer，仍报
  `requires 2101248 bits while 1572864 bits available`，即 BM128 的 128×128 FP32 tile 需求 256.5 KiB，
  超过 192 KiB UB。该臂在固定栈下属于编译容量限制，不通过改 `num_warps`/`num_stages` 或改 kernel 强行凑过。
- 修复（仅 probe）：每臂 start/warmup/metadata/compare 逐阶段落盘；四臂全部登记在 `arm_status`；仅识别
  `MLIRCompilationError` + `bishengir-compile` + UB 溢出文本为可隔离编译失败，BM128 记录完整 error/traceback/
  cache_dir 后继续 BM16/32/64；其他编译/设备/同步异常先落盘再抛出；BM64 参考失败则
  `paired_comparison_possible=false`；三轮预登记顺序投影到可执行臂并同时记录原始/实际顺序。
  计时调用仍走原 wrapper 与装饰器，`num_warps`/`num_stages`/精度/kernel 均未改。

本次修复后的身份：

| 文件 | SHA-256 |
| --- | --- |
| `runtime/wave3_gdn_norm_block_m_probe.py` | `ae346bed3ce78980b776f1d278afff47bb40a4328a3e70b321b48fc16219a80f` |
| `runtime/apply_gdn_norm_block_m.py` | `140766f26914a26d13c0b25b7082fcb5e5fbdac4c4b1438888a926244ba02bc1`（未变） |
| `runtime/gdn_norm_block_m_env_check.sh` | `805d60b2213f2fe856b5cf777e7ce28b4d015a6cb7c83631a748c60a495daf58` |
| `runtime/run_gdn_norm_block_m.sh` | `716c82154a82c8ce8d03b318d52cf885f586e09503a25674b874e18a0438057a` |
| `runtime/test_gdn_norm_block_m_probe.py` | `66ef11b2a12b5c7916587482a4a10fcd7fdaaad4787282e59694a7e95b59da01` |

重试策略：旧三格归档；21 完成三检后使用**尚不存在的**新目录
`/workspace/yangchen/logs/gdn-norm-block-m-20261118-run4` 单次重试；runner 会拒绝已存在目录，并在探针
失败时仍执行 revert。设备、服务与三检仍由 21 负责。
