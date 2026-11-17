# ClefGdnConvQkv 候选融合核

状态：新建候选，等待 A3/CANN 9.2.1 构建与实机验证。当前服务默认值保持现有配置。

输入为连续 BF16 `x[T,5120]` 或打包 QKVZ `x[T,8192]`，以及 packed BF16 `weight[4,5120]`。
QKVZ 输入读取每行前 5120 通道，Z 分支保持原值。服务通过受存储边界检查的
`as_strided` 暴露完整投影行，省去 QKV 连续化拷贝。
输出为连续 BSND `q/k[1,T,8,128]`、`v[1,T,24,128]`。
仅支持一个完整序列、零左历史、width=4、无 bias、SiLU、TP2 固定头数。
服务接入时必须验证上述条件，其他请求走原始路径。

每个 task 负责一个 head 的 16/32/64 个 token，搬入 3 个历史 token。
输入从 token-major 张量分行搬运，GM stride 使用字节单位。
四 tap FP32 累加使用 oldest→newest 顺序，SiLU 后 `CAST_RINT` 到 BF16。
Q/K 重新转 FP32，计算 `rsqrt(sum(x*x)+eps)`，再写 BF16。
保留卷积 BF16 舍入边界；V 写入 SiLU 后的 BF16 值。
数据搬运使用 TQue 完成 MTE2/V/MTE3 依赖，每个输出单写者，无核间同步。

首版沿用已部署 QKV v2 的逐行 ReduceSum，已完成 57 组硬件逐位一致性检查。
当前候选把每行两段 64 列先相加，再用 WholeReduceSum 归约整个 tile；
源与工作区 512B 对齐，归约版本及 QKVZ 路径继续按探针和全量服务验收。
核内 half-row 算术每次处理 64 FP32，repeat stride=16 个 32B block；
weight repeat stride=0，逐 token 复用同一 head 的 tap 权重。
tile=32 的总 UB 预算 88.5 KiB；tile=64 约 152.8 KiB（包含对齐余量）。
tile 归约和双缓冲可在本版契约验证后独立比较。

构建流程继承 QKV norm 工程，使用独立 vendor `clef_conv_qkv`：

```bash
source /workspace/yangchen/Clef-27B-ascend/scripts/activate_env.sh
cd runtime/ascendc/clef_gdn_conv_qkv
bash build.sh
./build_out/custom_opp_ubuntu_aarch64.run --noexec --extract=build_out/install
python /workspace/yangchen/Clef-27B-ascend/runtime/clef_conv_qkv_probe.py \
  --vendor "$PWD/build_out/install/packages/vendors/clef_conv_qkv" \
  --tiles 16,32,64 --output /workspace/yangchen/logs/conv-qkv-probe.json
```

探针覆盖 T=1/2/3/4/31/32/33/307/967/1595 和三个 seed，报告 finite、NaN、
逐元素误差、CPU FP32/FP64 卷积参考、交错 NPU event 时间。
局部筛选阈值只用于排除错误，赛事验收仍需完整 OD 与 ERQA 的 2PP 结果及 S1/S2。
