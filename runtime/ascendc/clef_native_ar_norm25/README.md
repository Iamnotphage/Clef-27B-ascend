# N1：原生 GEMM + 独立 AllReduce/AddRMSNorm 边界

2026-11-19，worker25。独立 op `ClefNativeArNorm25`，vendor
`clef_native_ar_norm25`，Torch namespace `clef_n1_25`。设备执行、隔离安装、
加载与服务恢复由 worker21 负责。E、D0 b719 与生产文件均保持原身份。

## 已完成与待验

- 完整 host/kernel TU 在实机 CANN 9.2.1 编译成包，真实 Torch adapter 链接成功。
- 生成 metadata：`coreType=MIX`、`intercoreSync=1`、`magic=RT_DEV_BINARY_MAGIC_ELF`。
  入口指定 `KERNEL_TYPE_MIX_AIV_1_0`，仅包含向量计算，host 使用 `SetScheduleMode(1)`。
- CPU 契约 4/4、固定门回归 5/5、Python 语法与 diff 检查通过。
- 同步脚本最终红线/高等级 0，性能提示 15（粗粒度屏障）。图脚本 findings 0；
  正则分析未展开 SDK/HCCL，不能证明设备上的完成顺序或 GM 可见性。
- **设备固定门、实际加载、重复运行与性能均待验证。**本阶段未安装或加载候选。

## 本轮实际新增

旧 AR+Norm 同窗口负结果已核：旧融合路径比 native 慢约 8–11%。本候选提供可独立
证伪的正确性修复与新的运行契约，收益需重新实测。

| 边界 | N1 的真实代码变化 | 可证伪检查 |
| --- | --- | --- |
| GEMM | 使用 `torch.matmul` 的原生 BF16 输出作显式输入；kernel 不含 Matmul/KFC | partial 原始字节与 hash；相同 partial 的 ARoff gate |
| 通信 | HCCL 写显式 reduced 输出；删除旧 GM flag 轮询及复用状态；Wait/Finalize 后所有 AIV 必经 FFTS SyncAll | reduced 与原生 AllReduce 逐位对照；NaN 预填、不同输入连续调用 |
| 数值链 | 平方项逐元素乘 1/N 后按 native 64-lane/80-repeat 归约；向量 Sqrt/Div；归一化值 BF16 roundtrip 后乘 gamma | 相同 partial/r/gamma/eps 的 Y/residual/rstd 固定门 |
| rstd | `[M]` 每行 DataCopyPad 精确写 4 字节 | M=1/7/8/307/967/1595 的尾行与 shape/finite |
| 诊断 | 每核独立 128B 槽，UB 初始化 32 字段并一次 DMA；31 字段 checksum | 每个字段损坏、空槽、额外写入的 CPU 反例；设备读回另验 |

本实现没有用旧 union76 的 `z-r` 派生 partial。D0 首格有效记录中未观察到的
worker/claim 仅作为观测缺失保留，空槽与 seal 均不能证明计算执行与否。

SDK 实读来源：

- `/usr/local/Ascend/cann-9.2.1/opp/built-in/op_impl/ai_core/tbe/impl/ops_nn/ascendc/add_rms_norm/add_rms_norm.h`
  SHA256 `ebe9dcbe41fb116a3b197b9cb33334ab024a38b837dbeafdb9877a548273d02f`。
- 同树 `rms_norm/rms_norm_base.h` SHA256
  `b7650f8a340869239963cd6a6d2ff71d1c17752aa8ed75265dd41516d5ad5e63`。
  归约次序来自其 `ReduceSumFP32`（Huawei CANN Open Software License 2.0）。
- `aarch64-linux/asc/impl/basic_api/dav_c220/kernel_operator_sync_impl.h`：
  `SyncAllImpl<true>` 用 `ffts_cross_core_sync(PIPE_MTE3, flag14)` 与 `wait_flag_dev`。
- `kernel_tensor_impl.h`：dav220 `GlobalTensor::SetValue` 为标量 GM store；
  DCCI 依赖条件宏。D0 trace 写回机制仍需设备证据。

## ABI / tiling

输入：`partial[M,5120]`、`residual[M,5120]`、`gamma[5120]`，均连续 BF16 ND。
M=1..16384。属性顺序：`eps:double, group:char*, useHccl:int64, cores:int64`。
输出顺序：`y BF16[M,5120]`、`residual_out BF16[M,5120]`、`rstd FP32[M]`、
`reduced BF16[M,5120]`、`diag UINT32[48,32]`。

生成 GetWorkspaceSize：3 输入 → 4 属性 → 5 输出 → workspaceSize/executor。
adapter 的 diag 容器为 int32，acl view 为 UINT32。实际 ND 存储描述、shape、dtype、
同设备与连续性均在 launch 前检查。显式绝对路径 `dlopen(RTLD_LOCAL)` + 唯一 op 符号，
避免依赖其他 vendor 的 `libcust_opapi.so` 名称解析。

tiling 固定 N=5120，完整 N 归约；行区间 `[M*i/cores,M*(i+1)/cores)`；
cores=1..48 且不超过 SDK 查询的 AIV 核数；每核 UB=103040 B；通信无分块，repeat=1。
所有 AIV 在行区间选择前参加一次硬屏障。useHccl=0 时逐行复制 partial 到 reduced，
该诊断成本保留；useHccl=1 时 reduced 是 HCCL 的直接目标。

| producer | consumer | 完成/可见性契约 |
| --- | --- | --- |
| native GEMM partial | HCCL/send 或 ARoff norm | 同一 Torch NPU stream 排序 |
| HCCL reduced | 全部 AIV 的 MTE2 | core0 Wait/Finalize → 所有 AIV SyncAll → DCCI → PipeBarrier |
| MTE2 x/r/gamma | vector | PipeBarrier ALL |
| vector residual/Y/rstd | MTE3 | PipeBarrier ALL；下一行复用前再等待 |
| UB 完整诊断记录 | GM diag[core*32] | PipeBarrier ALL → DataCopy 128B → PipeBarrier ALL |

第二行契约需由两 rank 实测验证；编译成功与记录 checksum 均不能替代它。

## 构建与身份

实机独立目录：`/workspace/yangchen/work/clef-n1-25-v1`。
CPU runner：`/workspace/yangchen/work/clef-n1-25-runner`。
构建日志 `clef-n1-25-v1-build4.log`、adapter 日志 `clef-n1-25-v1-adapter3.log`。
所有 vendor 文件的完整哈希见 `runtime/n1_25/build_identity.json`。

| 产物 | SHA256 |
| --- | --- |
| `build_out/custom_opp_ubuntu_aarch64.run` | `0e12ac492d9c02a3a9c84760602841db417ca0729fc62fcaa0ea7cacdfd33180` |
| `n1_adapter.so` | `98f40f3a8cf0c865f0dc9e9d3baed1ed67505cc35a85033486b2dde04b837f3e` |
| 生成 `aclnn_clef_native_ar_norm25.h` | `e56737c99f6f3d67b68d1475b23a6a346275c5ee7642aa903d74642db923d231` |
| kernel `.o` | `2ba0a8bb17928a850da434106ce45b4f374dd4501bbac0a5f7a072415b2f67a4` |
| kernel `.json` | `403b703a5a920911a38700b2dddc47b9d8bfe1e2b3fceb3222847eecf603752c` |

纯编译：加载 CANN 9.2.1 环境、将固定 venv/bin 放在 PATH，然后在独立目录
`MAX_JOBS=4 bash build.sh`；`bash compile_adapter.sh /absolute/path/n1_adapter.so`。
adapter 只需 ATen/Torch library 头，不依赖 Python 扩展头；实测 Torch 2.10.0+cpu
`_GLIBCXX_USE_CXX11_ABI=True`。

CPU 测试：

```bash
python3 -m unittest discover -s runtime/n1_25 -p test_contract.py -v
TORCH_DEVICE_BACKEND_AUTOLOAD=0 /workspace/yangchen/venvs/clef-py312/bin/python \
  -m unittest discover -s runtime/n1_25 -p test_fixed_gate.py -v
```

## 交给 21 的首格

21 在自己的监督器内完成隔离安装、hash/库路径核验与服务恢复。
安装根与生产 vendor 分离，候选进程 `ASCEND_CUSTOM_OPP_PATH` 仅指向隔离
`vendors/clef_native_ar_norm25`，`LD_LIBRARY_PATH` 保留固定 CANN/torch/torch_npu
库，opapi 由 runner 传绝对路径。host 产物直接依赖 `libplatform.so`。

```bash
ASCEND_RT_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 \
  /workspace/yangchen/work/clef-n1-25-runner/run_cell.py \
  --owner clef-27b-ascend-21 \
  --identity /workspace/yangchen/work/clef-n1-25-v1/identity.json \
  --vendor-root /OWNER_ISOLATED_ROOT/vendors/clef_native_ar_norm25 \
  --adapter /workspace/yangchen/work/clef-n1-25-v1/n1_adapter.so \
  --output /OWNER_NEW_EMPTY_RUN_DIR --rows 967 --seed 20261118 --cores 1 --use-hccl 0
```

该命令由 owner 补入具体隔离路径与独占超时/恢复监督。先 ARoff 相同 partial 固定门，
再在该边界通过后由 owner 安排 cores 与 ARon。已有 24 head 候选优先。
runner 不管理服务、不安装、不重试；新建 rank 子目录防止覆写原件。
所有输出预填 NaN，先完整同步再读回；保存输入 hash 与候选/参考原始 `.bin`。
reduced 额外要求逐位相等，用于单独归因通信，三输出固定门完全保持原协议。

固定门：双方 shape/dtype/finite；residual 全元素 abs≤1e-5+0.016abs(ref)；
Y max_abs≤0.02、abs(ref)≥1e-5 域内 max_rel≤0.01；近零仍检查 abs/finite；
rstd max_abs≤1e-3。固定门通过后才进入性能对照。

后续 MC2 配平需要 owner 提供真实 tiling、单次 full-payload AR 与 norm 隔离耗时；
本阶段没有按缺失数据推导通信切块或服务收益。
