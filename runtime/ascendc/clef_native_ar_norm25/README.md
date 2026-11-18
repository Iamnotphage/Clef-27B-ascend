# N1：原生 GEMM + 独立 AllReduce/AddRMSNorm 边界

当前待验产物为文末 **v2 生命周期修复**。下列 v1 包 `0e12…` 保持冻结，暂停上板；
v1 对应源码提交 `eb2e33f`，v2 位于独立分支 `ao/ascend-fusion/native-ar-norm-lifecycle`。

2026-11-19，worker25。独立 op `ClefNativeArNorm25`，vendor
`clef_native_ar_norm25`，Torch namespace `clef_n1_25`。设备执行、隔离安装、
加载与服务恢复在02:27交接至 worker27，须20激活后执行；此前worker21的
命令与证据按历史记录保留。E、D0 b719 与生产文件均保持原身份。

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
v2 的首格设备结果和后续最小格见文末；本节记录独立编译交付身份。

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

## v2 首格实际原件与 ARon/cores1 后续格

21首格目录：`/workspace/yangchen/logs/n1-v2-cell1-20261119T0215`。
ARoff接线源码冻结于`a289a6ba757eaeedf20d39544fe97c550e829243`，原bundle保持不变。
实际torch2.10.0+cpu/torch_npu2.10.0.post7，rows967/seed20261118/cores1/eps1e-6。

25只读CPU复算两rank各4,951,040个BF16 residual/Y、967个FP32 rstd：
shape/dtype/双方finite通过，residual fail_count=0/max_abs=0，Y max_abs/max_rel=0，
rstd max_abs=0；Y近零域分别4,604/4,606个元素，近零max_abs=0。
每rank12份原始文件SHA/长度/声明shape/dtype全匹配。
Y/residual/rstd/reduced候选与参考原字节分别相等，partial与reduced原字节相等。
用记录的reduced+r重建BF16 residual、用记录的rstd和gamma重建Y亦完全相等。
diag slot0覆盖[0,967)，rank0/1、ranks2、status0，前31字段checksum通过，余47槽全0。

| 输入 | rank | SHA256 |
| --- | --- | --- |
| x | 0/1 | `220586f0b0d5edab9e8f2e5cfe6da8a81bf444ecc90f0b5c0ddc8206566d4b35` |
| w | 0 | `1fd0e09d01aa862184076b0313a3566c697d644610e62dae902e09a4e8e6e72b` |
| w | 1 | `12539641fe24794c9e17cac3c83c0147705ff04f8afe93fd58ef1405d0926785` |
| r | 0/1 | `62a80d252e84ffcf82c8ca613e0158dc0bf6b90ca06d55802c61b73c8ebb955f` |
| gamma | 0/1 | `5cadbc9f2438a7db5712c8cc7d3ac9ecfd3b68c90b5d1dac35401d3bbaecddfb` |
| partial | 0 | `86cfcdfa184479daa95287b92a568856acc18175306efa4d63d56cceebff540d` |
| partial | 1 | `f0ace3fd51f869e9c7998a2223c6d4d77775dd319d20ffbee2a844518da80201` |

缺项：x/w有hash/shape/dtype/bytes元数据，缺少x.bin/w.bin；r/gamma/partial及全部输出
均有真实bytes。当前只证此输入链的cores1/ARoff生命周期与数值，ARon、多核、计时、
完整质量与服务收益均待各自验证；旧E的Matmul根因仍缺真实workspace partial证据。
23另已闭合六步骤RC0、vendor16文件退役、active目录消失、head f810和真实恢复四检。

CPU数值原件：`/tmp/n1-cell1-cpu-analysis-rank0.json`
（`c2cbcebfc45047491f995f92a6e229132dc2f0412238c1435ed0e099e188983d`）、
`/tmp/n1-cell1-cpu-analysis-rank1.json`
（`8fae07ca45f87d06f40f1f94730a5d6817e08da5d7df33bdeef006a0acaaf6ba`）。
逐项字节/identity报告：`/tmp/n1-v2-cell1-20261119T0215-byte-identity-audit.json`
（`1cdaca982af8f5af8ba70e483ae80dca125541eb01aeb1b903f623554f9bae2f`）。

下一独立bundle：`/workspace/yangchen/work/clef-n1-25-window-v2-aron-c1`。
仅改变common.CELL、probe argv与plan的use_hccl为1及对应CPU断言（12/12通过）；
runner、输入生成、固定门、v2算子源码/包/adapter/vendor安装器/监督wrapper均保持相同SHA。
本仓最新payload用于ARon；ARoff payload在a289a6b和原bundle保留。
ARon `window_payload.json` SHA
`b2130071e100b5467a0265617c85387e85a6617ca3cb98deb255cf3aa90ec7a0`，
`plan.json` SHA `6ac8fffad6dcaf25bc8d81e1554b8bb803491ee8f4dbecbf80af554b675c39dd`。
运行后须逐项核x/w/r/gamma/partial与上表同输入，继续要求explicit reduced逐位等于native AR参考。
21唯一执行，排程由20统一；通过后再提出单变量cores扩展与计时。

```bash
/workspace/yangchen/venvs/clef-py312/bin/python \
  /workspace/yangchen/work/clef-n1-25-window-v2-aron-c1/supervised_window.py \
  --owner clef-27b-ascend-21 \
  --wrapper /workspace/yangchen/work/scratch-24/joint-fused-5c045eb/runtime/joint_fused_attention_window.py \
  --wrapper-sha 9245e1a6504742dc4706eb466f8053cb3999ff3d80abfac960ed218a37a747d5 \
  --expect-payload-sha b2130071e100b5467a0265617c85387e85a6617ca3cb98deb255cf3aa90ec7a0 \
  --repo /workspace/yangchen/work/scratch-24/joint-fused-5c045eb \
  --run-dir /workspace/yangchen/logs/n1-v2-aron-c1-20261119T0230 \
  --execute
```

## owner27 接手入口（仅owner与来源增量）

执行归属从21交接至27。新bundle独立保留于
`/workspace/yangchen/work/clef-n1-25-window-v2-aron-c1-owner27`；旧21 bundle与manifest
均保持原件。基于`425b5e1abaeedd642a8e7fbe0e94f37b254129e8`，manifest仅入口SHA改变：

- entry：`7e98aba9e854e4e41645216d68dfb3b0642c6a5335b437cec572b8716a50a9cb`
- payload：`f548b3689b91cfa2e81f5ba5282ef022aa050cc8cc58542b214c0d5b32eb409e`
- plan：`59da5843df72fdd1b668302d5bd1d0c711768360beb14b7488e5802910f0f8a6`

入口只接受owner27；plan的owner为27，`runner_owner_compat_tag`明确记为21，
`activation_required_from`为20。固定runner e035及其`--owner21`兼容校验和meta字段
保留原字节；实际执行身份通过新入口、plan和仅probe child的
`CLEF_N1_EXECUTION_OWNER=clef-27b-ascend-27`记录。该区分不改变runner的任何执行逻辑。
其余probe、core、包、adapter、输入生成、门、PGID链均保持上一格SHA，cores1/ARon/240s。
CPU只读入口生成成功，旧owner21在parser以rc2拒绝。23已有限核最小diff及全部字节。

交接期只读单命令如下；20激活且现场恢复闭合后，由27给此入口加`--execute`串行运行：

```bash
/workspace/yangchen/venvs/clef-py312/bin/python \
  /workspace/yangchen/work/clef-n1-25-window-v2-aron-c1-owner27/supervised_window.py \
  --owner clef-27b-ascend-27 \
  --wrapper /workspace/yangchen/work/scratch-24/joint-fused-5c045eb/runtime/joint_fused_attention_window.py \
  --wrapper-sha 9245e1a6504742dc4706eb466f8053cb3999ff3d80abfac960ed218a37a747d5 \
  --expect-payload-sha f548b3689b91cfa2e81f5ba5282ef022aa050cc8cc58542b214c0d5b32eb409e \
  --repo /workspace/yangchen/work/scratch-24/joint-fused-5c045eb \
  --run-dir /workspace/yangchen/logs/n1-v2-aron-c1-owner27-20261119T0230
```

## ARon/cores1 实际闭合与 cores2 待排程格

27实际运行：`/workspace/yangchen/logs/n1-v2-aron-c1-owner27-20261119T0230`。
window SHA `2c968520017f19e14004901e8f3a525387a495e45182275c8631efa69cc2333e`，
overall0、preflight/stop/apply/probe/revert/restore六步骤均RC0。
restore SHA `f97217637f4ee952dcf5856c6fdfaa77371ad993ca25a43c06809cb57eec6399`，
label `d0-restore-20261118T183704Z-b8572d6f`，root1195216，双worker1195493/1195494；
23独立闭合health/workers/text/image四项RC0，卡7 chip0/1，vendor16原件退役且active消失。

两rank x/w/r/gamma/partial元数据逐项同ARoff，r/gamma/partial原bytes相同。
每rank12 raw文件大小/SHA全匹配；Y/residual/rstd/reduced候选与native参考原字节分别相同。
四份reduced（两rank × cand/ref）SHA均为
`02215064e0e393b3ab21a3bf0aca14c51281435ce74b3840e26647810adbfcd1`，
与预先由两份真实partial做FP32相加→BF16的CPU可证伪预测吻合。
固定门shape/dtype/双方finite通过，resfail0、所有max_abs/max_rel为0；Y近零域各4469元素，
近零max_abs0。CPU用各自实际reduced/rstd重建res/Y逐字节等。
diag首槽[0,967)、cores1、ARon、rank0或1/ranks2、set/wait0，31字段FNV过，余47槽全0。
probe rc0/timeoutfalse/reapedtrue、unknown/residual空；记录的actual owner27与legacy tag21分列。
此结论仅覆盖本输入cores1 ARon数值与生命周期，多核和性能另需实测。

25 CPU结果：`/tmp/n1-aron-c1-cpu-analysis-rank0.json`
SHA `c4ed939e6969465827f2c80f02fa90a9b58fc79984e0d6b141b27c4f1f670eb8`；
rank1同前缀，SHA `84a27a6f91e4c213c6a3196b39eb69b48f81657382239b3c551e56a6ea68b55c`。
逐项字节报告`/tmp/n1-v2-aron-c1-owner27-20261119T0230-byte-audit.json`
SHA `fe302c32fdb7b121de925f325c9c9cbccf2bab9cd12f456862ad9e29d673e75c`。

CPU准备的下一最小格仅将cores1→2：两段为[0,483)、[483,967)，用于检验第二核参与
FFTS后的HCCL reduced可见性、不等长行分片及rstd4B行边界。包、核心实现、runner、输入、
ARon、固定门、PGID监督与owner27均保持；接线CPU12/12通过。原c1包与计划保留。
该格尚待20统一排程，25仅生成只读计划，当前无新设备结果或计时。

bundle：`/workspace/yangchen/work/clef-n1-25-window-v2-aron-c2-owner27`；
payload SHA `697a7538b218b6c059b6b6f162fc75266ac53fdb85e4e6ecfb27139e9415b478`；
plan SHA `da35e14f150c84c799ebfa3373abb22a76366ff52d9d54f2e8e289e9b3428609`。

```bash
/workspace/yangchen/venvs/clef-py312/bin/python \
  /workspace/yangchen/work/clef-n1-25-window-v2-aron-c2-owner27/supervised_window.py \
  --owner clef-27b-ascend-27 \
  --wrapper /workspace/yangchen/work/scratch-24/joint-fused-5c045eb/runtime/joint_fused_attention_window.py \
  --wrapper-sha 9245e1a6504742dc4706eb466f8053cb3999ff3d80abfac960ed218a37a747d5 \
  --expect-payload-sha 697a7538b218b6c059b6b6f162fc75266ac53fdb85e4e6ecfb27139e9415b478 \
  --repo /workspace/yangchen/work/scratch-24/joint-fused-5c045eb \
  --run-dir /workspace/yangchen/logs/n1-v2-aron-c2-owner27-20261119T0245
```

## cores2 实际闭合与 cores24 条件计时交付

cores2 run：`/workspace/yangchen/logs/n1-v2-aron-c2-owner27-20261119T0245`。
window SHA `32019923fd4f5400da18fcd87389dfb7674a933ccfe80f1e26e8a9da0fad102c`，
overall0/六步骤RC0；restore SHA
`f736bb3e8b4053595e4e7ca402b64915bedef90c5d1a1ee18a77a97db208a9c3`，
label `d0-restore-20261118T184817Z-ea064140`，root1200025/worker1200187、1200188，
四检通过，active vendor消失，retired16文件全SHA匹配。

两rank各12原bin大小/SHA通过，五输入元数据和所有输出与cores1相同，原固定门误差全0。
实际diag slot0[0,483)、slot1[483,967)，两个槽全31字段FNV过、后46槽全0。
rstd row482/483实际字节分别`edf5153f`/`5cb8183f`，值为0.5857837796/0.5965631008；
row480..485及两段完整输出均与本格参考和cores1相等。
边界报告`/tmp/n1-aron-c2-boundary-analysis.json` SHA
`48ebc3a524fb4486a37c578f6a1ea2989a6b4c54c0a1e960c3d5495ac737d3d6`；
字节审计`/tmp/n1-v2-aron-c2-owner27-20261119T0245-byte-audit.json` SHA
`e28ada7f4bafad47f8e8238edeead687ffd8e7e17bd61928b30248ca968a32fa`。
本格正确性及生命周期闭合，尚无性能结论或普遍同步证明。

20授权下一独立cores24格及其门后计时。原v2包、ABI、adapter、run_cell、输入、
HCCL类型/算法和固定门保持原SHA。CPU分区模型为17段40行、7段41行，互斥覆盖[0,967)；
这是请求参数模型。实际diag非零槽集合和每槽区间由6144B raw解析，单列observed与requested。

同一个240秒监督probe按顺序执行：

1. 原run_cell执行cores24/ARon正确性格；两rank固定门、explicit reduced、diag、
   完整raw SHA全部通过后才允许下一步。
2. 同PGID、`--run-path --start-method=spawn`的新torchrun运行`bench_ready.py`。
   导入设备后端前再次要求**本窗口**cores24结果及全部原bytes；cores1/2均被拒绝。
3. 以相同partial/r/gamma测native all_reduce+AddRMSNorm和N1；每调用先恢复输入、
   barrier/sync及创建事件，均放计时外。N1计时包括adapter分配、diag与实际算子调用。
4. 1对首次调用、3对warmup独列；正式4轮AB/BA/BA/AB，每轮5对。每rank共48条调用记录，
   正式40条；保存每次device event、wall、perf_counter_ns起止。固定AOT包未在窗口内编译，
   runtime编译耗时标为未单独观测，首次调用包含可能的惰性初始化；加载/输入上传setup另列。
5. 每轮结束复核固定门、diag、reduced及本窗口native source参考；最后一轮保存9份原bin。
   任一轮或计时子进程失败，整个probe判失败并沿原finally恢复。
6. 聚合只接受两rank完整同序列、同host/boot及monotonic实现；每调用保留两rank原始样本，
   分列max(rank event)、max(rank wall)、max(end)-min(start)整体wall关键路径。
   median/p95及配对速度比仅取正式轮次，p95线性插值，零分母比值显式null。

CPU验证14项接线/分区＋10项统计协议通过；涵盖拒绝cores2前置启动计时、缺样本/重排、
非finite时长、错wall、跨host/boot/clock等。CPU synthetic数据只验证协议，无设备计时含义。
计时范围是input-ready AR+AddRMSNorm，GEMM及输入恢复在外；设备原件到齐前不报收益。

bundle：`/workspace/yangchen/work/clef-n1-25-window-v2-aron-c24-timing-owner27`。
payload SHA `b782e49087e0bd0c7ccc48790540a3af8804bd19aee690c73372cc3da8614977`；
plan SHA `8a380481cae6d1b105f4a7a2ed39d46010aa5ea91bd1e1a27b83830d11c4a5e3`；
entry SHA `5753d8e43096a876666c1340d5574f5cf97ef7fc6809574a090429198ad5aae8`；
bench SHA `1fd5dcefc43f1b91535c334c077013dd6deca5016abde4077faeea9aeb7ac05e`。
完整payload含新timing模块，原1/2核bundle保持冻结。下面只读命令已由25执行生成plan，
最终增量核准且前置窗口恢复闭合后，由27按20授权加`--execute`。cores48不排。

```bash
/workspace/yangchen/venvs/clef-py312/bin/python \
  /workspace/yangchen/work/clef-n1-25-window-v2-aron-c24-timing-owner27/supervised_window.py \
  --owner clef-27b-ascend-27 \
  --wrapper /workspace/yangchen/work/scratch-24/joint-fused-5c045eb/runtime/joint_fused_attention_window.py \
  --wrapper-sha 9245e1a6504742dc4706eb466f8053cb3999ff3d80abfac960ed218a37a747d5 \
  --expect-payload-sha b782e49087e0bd0c7ccc48790540a3af8804bd19aee690c73372cc3da8614977 \
  --repo /workspace/yangchen/work/scratch-24/joint-fused-5c045eb \
  --run-dir /workspace/yangchen/logs/n1-v2-aron-c24-timing-owner27-20261119T0310
```

## cores24 实际闭合与 input-ready 计时负结果

27执行上述冻结窗口，run 为
`/workspace/yangchen/logs/n1-v2-aron-c24-timing-owner27-20261119T0310`。
window SHA `36f8875517cbede58161d2578e7058638de6eb6c488d11f14ac7df0ba589488a`，
overall0、六步骤RC0；restore SHA
`3f400e4a7ffe8185a47d2c893c2e0c2e7d4a31221c6a5418cb2bc7b1742af9ca`，
label `d0-restore-20261118T191020Z-4516976e`，真实health/workers/text/image四项RC0/oktrue。
probe清理reaped=true，unknown/residual为空；active vendor已消失，退役原件保留。

23与25逐项核验初始24份、计时终态18份原bin，共42份大小/SHA/shape/dtype一致。
两rank reduced/Y/residual/rstd候选与本格参考、cores2对应字节全部相等，原固定门误差全0。
五输入metadata同cores2，r/gamma/partial实际字节也相同；x/w只有哈希，未保存原bin。
四份实际diag（两rank×初始/终态）有效槽集合均为0..23，17段40行与7段41行覆盖967行，
FNV、通信状态、rank字段通过，后24槽全0，初始与终态逐字节相等。rstd的967项有限且为正。
这些证据覆盖本输入c24/ARon正确性及生命周期，实际参与观测为上述24槽。

96条原始样本（48/rank，其中40/rank正式）重新按`timing_contract.aggregate_samples`
聚合，所有计时字段与现场summary精确相等；summary额外封装diag和rank结果哈希。
summary SHA `fc8e8008b857a5ffe5ef6cad8d26713892f2ac26b5e8a0492ec8e27cc80e601b`。
原`probe/timing/rank0/samples.jsonl` SHA
`c37769cca4affeca68aae531f4d8b76869bad414c9dd88741932c5593ef743b5`；rank1 SHA
`5f0a281a4b4aeafacd8a06282c4ce205dfacb7ccb21fb6bd46902fa03bf31850`。

正式20对样本全部保留，单位ms：

| 口径 | baseline中位数 | N1中位数 | 配对(candidate-baseline)中位数 | 配对baseline/N1中位数 |
| --- | ---: | ---: | ---: | ---: |
| 整体critical wall span | 0.567595 | 0.578880 | +0.012930 | 0.977116 |
| slow-rank wall | 0.544860 | 0.571320 | +0.028000 | 0.950803 |
| slow-rank event | 0.449630 | 0.478040 | +0.027530 | 0.941371 |

四轮critical wall分列如下；两臂中位数之差与配对差中位数使用不同统计定义：

| 轮次/顺序 | baseline中位数 | N1中位数 | 两臂中位数之差 | 配对差中位数 |
| --- | ---: | ---: | ---: | ---: |
| 0 / AB | 0.509020 | 0.534600 | +0.025580 | +0.043190 |
| 1 / BA | 0.530990 | 0.568860 | +0.037870 | +0.039780 |
| 2 / BA | 0.577380 | 0.583980 | +0.006600 | +0.011540 |
| 3 / AB | 1.251440 | 0.964910 | -0.286530 | -0.300140 |

前三轮候选偏慢，末轮两臂都升高；保留末轮且不对变化机制归因。
critical/event/slow-wall分别有9/8/7对候选更快。首调用critical为baseline10.57658、
N1 303.83746，setup rank0/1为1435.97522/1660.58330，均单列并排除于正式统计。
四轮末固定门、diag及native source门均通过；最后一轮另保存18份真实tensor原件。

**当前N1 v2没有稳定性能收益证据，保持独立实验产物。**计时范围只覆盖input-ready
AR+AddRMSNorm，包含adapter分配及diag，GEMM/输入恢复在外；完整服务收益与全量质量未测。
当前总event/wall无法分辨MC2通信、归一化、adapter分配或diag各阶段耗时，
后续MC2配平仍需实际tiling及隔离Tcomm/Tcompute。cores48不排，冻结包与原件保持不变。
