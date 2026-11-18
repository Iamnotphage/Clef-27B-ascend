# N1：原生 GEMM + 独立 AllReduce/AddRMSNorm 边界

当前待验产物为文末 **v2 生命周期修复**。下列 v1 包 `0e12…` 保持冻结，暂停上板；
v1 对应源码提交 `eb2e33f`，v2 位于独立分支 `ao/ascend-fusion/native-ar-norm-lifecycle`。

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

首格采用文末的完整监督入口，v1 保持冻结。固定 rows967/seed20261118/cores1/ARoff，
只由21执行；后续设备格由 owner 按实际固定门结果安排。
runner 不管理服务、不安装、不重试；新建 rank 子目录防止覆写原件。
所有输出预填 NaN，先完整同步再读回；保存输入 hash 与候选/参考原始 `.bin`。
reduced 额外要求逐位相等，用于单独归因通信，三输出固定门完全保持原协议。

固定门：双方 shape/dtype/finite；residual 全元素 abs≤1e-5+0.016abs(ref)；
Y max_abs≤0.02、abs(ref)≥1e-5 域内 max_rel≤0.01；近零仍检查 abs/finite；
rstd max_abs≤1e-3。固定门通过后才进入性能对照。

后续 MC2 配平需要 owner 提供真实 tiling、单次 full-payload AR 与 norm 隔离耗时；
本阶段没有按缺失数据推导通信切块或服务收益。

## v2：MC2 生命周期修复（独立编译）

证据：生成 `aclnn_clef_native_ar_norm25.cpp` 第 231–232 行，无条件调用
`NnopbaseSetMc2(*executor)` 与 `NnopbaseSetHcomGroup(*executor, group)`。
SDK `asc/include/adv_api/hccl/hccl.h` 的 `Finalize` 文档：通知服务端无后续任务、
等待最后任务完成、清理 `finishedTurnCnt`。实读 `hccl_aicpu_impl.h:524` 的实现
包含 `SendFinalizeMsg` 路径。因此 v2 为 AR off 也保留完整生命周期；
v1 的设备行为尚未实测，其挂起风险维持待证。

相对 `eb2e33f` 的功能差异仅有以下五项：

1. host：group 在两臂均必填；`Mc2CcTilingConfig.GetTiling` 在两臂均执行，
   构造有效 init/cc tiling。
2. kernel：core0 均执行 InitV2、SetCcTilingV2、GetRankId/Dim、Finalize。
   AllReduce/Wait 仅在 `useHccl=1 && setRc==0` 执行。全核屏障位置保持一致。
3. adapter：group 在 AR off 也必须非空。调用生成 API 的签名和输出分配保持一致。
4. diag/parser：AR off core0 的 rank/ranks 从 `-1/-1` 改为实际 `rank/2`；
   其余核仍为 `-1/-1`。`expected_comm` 与测试同步更新。
5. CPU 回归新增 AR off 生命周期 rank 的正反例。契约 5/5 通过；固定门实现未改，
   既有 5/5 回归仍适用；最终静态同步扫描红线/高等级 0，性能提示 15。

数值归约、舍入、reduced 输出、固定门、五输出 shape/dtype、4 属性顺序及生成头
SHA 均保持 v1 契约。当前 `runtime/n1_25/build_identity.json` 为 v2 的16文件完整清单，
v1 清单保留在 `eb2e33f` 与远端 v1/identity.json。

| v2 产物（根 `/workspace/yangchen/work/clef-n1-25-v2`） | SHA256 |
| --- | --- |
| `build_out/custom_opp_ubuntu_aarch64.run` | `b9c3aa2bf3a5beb7b213e45e6b44fc0bd7abeff72fe982cd6e329bb34bbf9164` |
| `n1_adapter.so` | `3e73937a9df211f0dbbbfd2b6070774ff80a6ef6f838fb5b2f1efee20ef61c0c` |
| `build_out/autogen/aclnn_clef_native_ar_norm25.h` | `e56737c99f6f3d67b68d1475b23a6a346275c5ee7642aa903d74642db923d231` |

编译日志：`/workspace/yangchen/work/clef-n1-25-v2-build.log` 与
`clef-n1-25-v2-adapter.log`。runner 改用独立目录
`/workspace/yangchen/work/clef-n1-25-v2-runner/run_cell.py`。完整监督入口会按SHA复制
原runner、v2 parser、identity、adapter到窗口stage，使用该冻结副本。

v1/v2 各自独立冻结，使用相同 N1 op 名，按互斥安装/进程隔离测试。
v2 尚未安装/加载/执行；23 核新增链及身份后，由21安排 head 后首格。

## v2 完整监督接线（2026-11-19 02:15 CST）

复用24 `5c045eb1427e9cb902867bd7147c0d636ce91e1c` 的真实监督窗口，SHA
`9245e1a6504742dc4706eb466f8053cb3999ff3d80abfac960ed218a37a747d5`。
远端完整 runtime 根：`/workspace/yangchen/work/scratch-24/joint-fused-5c045eb`。
其六个 D0 恢复模块在停服前核SHA并stage；stop、PGID、timeout、失败finally、
唯一label的launch及health/2worker/text/image四检沿用该版本。

N1 bundle：`/workspace/yangchen/work/clef-n1-25-window-v2`。
`runtime/n1_25/window_payload.json` 列出9文件完整SHA；manifest本身为
`79a66589fb6fb20805e19312649b8f6fff2b4365cfb54667a21d600efbcb5003`。
远端 `plan.json` 保存完整wrapper argv和所有SRC:REL:SHA，SHA为
`b1099cc50ed6dd7c5ce306454fbf2e1d42d5b0aa4c1fb29d4596c8019dbbc715`。

21规范执行命令（省略 `--execute` 仅打印计划，25已执行此CPU路径）：

```bash
/workspace/yangchen/venvs/clef-py312/bin/python \
  /workspace/yangchen/work/clef-n1-25-window-v2/supervised_window.py \
  --owner clef-27b-ascend-21 \
  --wrapper /workspace/yangchen/work/scratch-24/joint-fused-5c045eb/runtime/joint_fused_attention_window.py \
  --wrapper-sha 9245e1a6504742dc4706eb466f8053cb3999ff3d80abfac960ed218a37a747d5 \
  --expect-payload-sha 79a66589fb6fb20805e19312649b8f6fff2b4365cfb54667a21d600efbcb5003 \
  --repo /workspace/yangchen/work/scratch-24/joint-fused-5c045eb \
  --run-dir /workspace/yangchen/logs/n1-v2-cell1-20261119T0215 \
  --execute
```

- `window_vendor.py` 实现 check/install/revert：精确核b9c3包，读取其中16个已核
  vendor成员，仅物化到本次 `isolated-opp/vendors/clef_native_ar_norm25`；已知
  liboptiling相对symlink物化为相同SHA字节。原包脚本和全局CANN目录均不执行/修改。
  check纯读；install失败保留ownership状态；revert将目录改名`isolated-opp.retired`，
  保留所有原件、清除活跃路径。revert失败由wrapper阻断restore。
- `window_probe.py` 接受真实`--tree/device/output`，固定首格及240秒外层probe期限。
  内部用`torch.distributed.run --standalone --nnodes=1 --nproc-per-node=2
  --max-restarts=0 --run-path --start-method=spawn`；shim/torchrun都继承wrapper建立的PGID。
  同目录runner/contract/fixed_gate通过torchrun child专用PYTHONPATH导入。
- OPP、opapi前插和`ASCEND_RT_VISIBLE_DEVICES=0,1`由wrapper仅注入probe child。
  adapter为stage绝对路径；opapi再由adapter绝对路径dlopen。外层及restore环境保持原样。
- 聚合必须child rc0、两rank固定门/diag/reduced全true、完成状态与identity一致；
  输入/原始partial绑定、全部12份raw字节SHA及双rank x/r/gamma一致均检查。
  输出`n1-probe/1`，失败保留原件，整体失败。head目标始终要求f8101b98。

CPU验证：接线12/12，既有diag5/5；冻结包16成员只读SHA通过。
真实`Window.probe`的synthetic CPU子链原件：
`/tmp/n1-wrapper-cpu-25-5c045eb/`，另复制到bundle的`cpu-pgid-evidence/`。
`result.json` SHA `ab75ad08103c69ec5277d42173ada9212a2950e63906315002f4893027fb8a46`。
固定torch2.10.0+cpu、autoload0；成功格wrapper PID1187579/PGID1187578，
probe1187580、torchrun1187581、rank1187586/1187587均PGID/SID1187580；
12秒timeout格各子进程均PGID1187599，真实killpg后rc-9、reaped=true、
unknown/residual空、四PID postcleanup均null。记录包含完整CPU命令与每层argv。
此验证范围为CPU probe子链；preflight/stop/finish、设备加载/退出和数值结果仍待21首格。
