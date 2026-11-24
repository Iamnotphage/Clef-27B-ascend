# ClefGdnConvQkvVarlen 多序列融合核

状态：**已验收并作为 `s1` 默认开启**（2026-11-25）。单序列 `num_prefills == 1`
继续走原 v4 `aclnnClefGdnConvQkv`，本算子只处理 `2 <= num_prefills <= 64`。

输入为 BF16 `x[T,5120]` 或打包 QKVZ `x[T,8192]`（零拷贝，读每行前 5120 通道）、
packed BF16 `weight[4,5120]`、int32 `cu_seqlens[num_seqs+1]`。输出为连续 BSND
`q/k[1,T,8,128]`、`v[1,T,24,128]`。支持 width=4、无 bias、SiLU、TP2 固定头数；
不满足的输入由宿主侧返回 `None` 并回退分解路径。

## 调度与数值

- `Init()` 通过 `MTE2_S` 事件把 `cu_seqlens` 搬入 UB，构造每条序列的 tile 前缀
  `tileOff_`；`Process()` 在 48 AIV 上按 `(head, seq, localTile)` 轮询，单核任务
  数 = `40 * sum(ceil(len_s/tile))`。
- 每个 tile 完全落在一条序列内；`history = min(localT0, 3)`、`leading = 3-history`，
  全局读偏移 `(tileStart-history)*inputChannels + head*128`，绝不读取相邻序列。
- 内层逐行算式、BF16 `CAST_RINT`、Q/K L2Norm、BSND 写出与 v4 完全一致；B=1 与
  多序列对比逐条 v4 均 `torch.equal == True`（离线 78 + 48 用例）。
- tile 16/32/64；实测 tile 64 最快（相对分解路径 3.53–4.54×），服务默认
  `CLEF_CONV_QKV_VARLEN_TILE=64`。tile=64 的 UB 预算约 153 KiB。

## 构建、vendor 与探针

```bash
source /usr/local/Ascend/cann/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
cd runtime/ascendc/clef_gdn_conv_qkv_varlen
bash build.sh
# 取 build_out/_CPack_Packages/.../packages/vendors/clef_conv_qkv_varlen
# 安装到 Clef-27B/src/ascendc/vendors/clef_conv_qkv_varlen
python runtime/clef_conv_qkv_varlen_probe.py \
  --varlen-vendor Clef-27B/src/ascendc/vendors/clef_conv_qkv_varlen \
  --v4-vendor Clef-27B/src/ascendc/vendors/clef_conv_qkv \
  --reference-vendor Clef-27B/src/ascendc/vendors/customize \
  --output Clef-27B/logs/conv-qkv-varlen-20261125/offline-probe.json
```

探针 4 门：B=1 对齐 v4、多序列对齐逐条 v4（含 1/2/3/4 边界、64 序列、packed
8192 毒化尾）、单序列篡改零串扰（×100/±1e4/NaN）、融合 vs 分解路径单算子计时。
服务接入由 `runtime/apply_conv_qkv_varlen_patch.py` 管理（preimage `ead25cb2…`，
postimage `afe6d4e8…`），`CLEF_GDN_CONV_QKV_VARLEN=0` 一键回退。
