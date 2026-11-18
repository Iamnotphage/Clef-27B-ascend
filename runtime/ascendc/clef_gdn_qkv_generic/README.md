# ClefGdnQkvGeneric 候选算子

状态：A3 / CANN 9.2.1 构建通过，36 组 TP1/TP2 硬件检查通过。
TP2 对部署 v2 全部逐位一致，TP1 对 FLA 最大绝对差为 0.001953125。
967 tokens 局部加速：TP2 1.32–1.40×，TP1 最高约 2.15×；服务验收继续单独执行。

输入连续 BF16 `x[H,T,128]`，支持 H=40（TP2）与 H=80（TP1）。
HK=H/5、HV=3HK，输出连续 BSND `q/k[1,T,HK,128]`、`v[1,T,HV,128]`。
Q/K 使用 FP32 `x * rsqrt(sum(x*x)+eps)` 后 CAST_RINT 至 BF16；V 逐位搬运。
属性顺序固定为 `eps: float`、`tile_tokens: int`（16/32/64）。

每个 task 处理一个 head 的连续 token tile，按 `task=blockIdx; task+=blockNum`
轮转分配，均摊 Q/K 与 V 的成本。Q/K 每 tile 使用两条官方连续 mask
`WholeReduceSum<float>(sum, square, 64, rows, 1, 1, 16)` 及一次 Add；
srcRepStride=16 个 32B block=512B，dstRepStride=1 个 float。
所有 FP32 buffer 基址单独补齐 512B；第二个 half-row 偏移为 256B，满足
WholeReduceSum 的 32B 源对齐。V 使用 `TQueBind<VECIN,VECOUT,1>`，
通过 MTE2→MTE3 队列同步直接从输入写 BSND，无 FP32 cast。
DataCopy 的数据大小与 GM 偏移始终为 128 个 BF16 的整数倍（256B）。
输出 DataCopyPad 的 GM stride 以字节计，blockCount 最大 64。

tile=64 的 UB 预算约 147 KiB，包含三块 BF16 队列（48 KiB）、三块
FP32 数据 buffer（96 KiB）及两个归约 buffer、各 FP32 buffer 的 512B 余量。
三个单 buffer 队列，分配/入队/出队/释放成对，无跨核同步。
两段 64 列归约改变求和树，Q/K 与部署 v2 的逐位关系以实机探针为准。

独立 vendor 为 `clef_qkv_generic`，独立算子名为 `ClefGdnQkvGeneric`，
现有 `customize` vendor 与服务配置保持原值。

```bash
source /workspace/yangchen/Clef-27B-ascend/scripts/activate_env.sh
cd runtime/ascendc/clef_gdn_qkv_generic
bash build.sh
./build_out/custom_opp_ubuntu_aarch64.run --noexec --extract=build_out/install
python /workspace/yangchen/Clef-27B-ascend/runtime/clef_qkv_generic_probe.py \
  --vendor "$PWD/build_out/install/packages/vendors/clef_qkv_generic" \
  --key-heads 8,16 --lengths 307,967,1595 --tiles 16,32,64 \
  --output /workspace/yangchen/logs/qkv-generic-probe.json
```

探针默认额外覆盖 tile 前后边界、两个 seed、零/常数/稀疏行，输出预先填 NaN，
记录 finite、NaN、Inf、逐位一致性、最大/平均绝对差、RMSE。
HK=8 对照部署 QKV v2，HK=16 对照 FLA l2norm + BSND 搬运；两者均保存
CPU FP32 参考。`--reference-vendor ''` 对两个 HK 都使用 FLA 对照。
计时交错调用参考与候选，以 NPU event 报告中位、均值、最小值及逐次时间，
同时报告 host enqueue 时间。局部阈值默认 0.00390625 且 V 必须逐位一致；
服务接入仍需完整 OD / ERQA 的 2PP 与 S1/S2 验收。

API 来源：

- [WholeReduceSum（官方，A3 支持、连续 mask 与 repeat stride）](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/900/API/ascendcopapi/atlasascendc_api_07_0081.html)
- [搬运算子复用 VECIN/VECOUT（官方 TQueBind 用法）](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/850/opdevg/Ascendcopdevg/atlas_ascendc_best_practices_10_0027.html)

本地使用 `kernel_operator.h` 已有基础 API，避免高阶 Pattern 临时空间依赖；
实机 CANN 头文件和编译结果决定该候选的最终可用状态。
