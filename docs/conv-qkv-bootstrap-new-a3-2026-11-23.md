# 新机 Conv+QKV 启动修复与 C1 复核

## 启动链

新机标准工程 `bcb98d9201a8f63e009349d1c2a96898d64064cd` 的启动入口注入了 Conv+QKV 的 vendor 和开关。实际 `gdn.py` 仍为 `c8c0d4d8a693658e075a2232b6938584760672bacf101bf50ca5fdfab2c0be7e`，因此 C1 走拆分卷积和 QKV norm。原 QKV 安装器把这个预像和融合后像都识别为 `qkv`。

修复在 decision、QKV、chunk128 和 joint 安装之后、启动 worker 之前调用独立 Conv+QKV 安装器。它解析 worker 使用的 vLLM-Ascend 树，使用独立备份目录，对精确预像执行安装，并再次核对后像 `ead25cb2691cb9ab1c322353fc8e168f8bfc9c23b9e4d53de01c762943027e78`。已安装后像可直接复用。未知或缺失源码、预像备份冲突会阻止开启融合。dry-run 记录待安装状态，显式 OFF 走原有 fallback。

CPU 检查覆盖精确安装、第二次检查幂等、未知及缺失源码、冲突备份、只读 dry-run 和 OFF。启动器与独立安装器共 45 项测试通过。

```sh
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest runtime/test_inference_launch.py runtime/test_conv_qkv_patch.py
```

## 新机私有 A/B/A

两个 A 阶段使用 `c8c0…`，B 使用已有融合源码 `ead25…`。三阶段独立加载 worker，使用同一套 BF16 权重、aa3e joint head、TP2、eager、S1 4096 参数及 vendor。各阶段先完整预热 10 次，再对三条固定请求各计时 9 次。A1 和 B 的 Level1 各另外采三条请求，采集时延与主计时分列。下面的收益取每一轮同请求的两组 A 时延均值减去 B 时延后再取中位数。

| 请求 | 任务 | 实际 M | A1 / B / A2 中位数 ms | 夹测中位收益 ms | `abs(A1 − A2)` 中位数 ms | B 快于两组 A 且收益超过该轮 A/A |
| --- | --- | ---: | --- | ---: | ---: | ---: |
| od-2041 | choice | 982 | 160.308 / 155.719 / 161.171 | 5.031 | 0.906 | 9/9 |
| od-1109 | score | 977 | 159.720 / 155.177 / 161.333 | 5.183 | 1.656 | 9/9 |
| od-1474 | noul | 990 | 160.504 / 155.860 / 161.716 | 5.295 | 1.304 | 9/9 |

M982 的 L0 真实输入、卷积权重及 Q/K/V 在各 rank 内重复和跨 A1/B 均逐位一致。三条请求的答案协议和 choice 检查通过，最大候选概率差为 0.005537。

Level1 在 B 的六个请求与 rank 组合中均确认了 48 次 `ClefGdnConvQkv`。每个对应 A1 组合有 48 次 packed 输入切片、48 次 `CausalConv1d`、48 次 `ClefGdnQkvNorm`。同窗卷积与归一化设备区间并集由 A1 的 8.326–8.544 ms 降到 B 的 4.171–4.200 ms。BSND BF16 chunk128 FLA 仍为 48 次，主干 attention 16 次，joint head attention 6 次。MatMul、SwiGLU、FLA 和 attention 的形状及相对 stream 归属一致，130 次通信的形状和相对 stream 归属一致。

现场 L0 的 QKV 输入为 `[M,5120]`、stride `(8192,1)`、offset 0。路由为 TP2、PCP1、单 prefill、无 decode 或 spec，BF16 `[4,5120]` 同设备权重，SiLU 无 bias。cap 为 1600，tile 为 32。B 的两个 worker 均映射已核验的融合 `op_api`。

## 证据与验收边界

- 原始窗口与关联报告位于 `/workspace/yangchen/logs/aa3e-conv-qkv-restore-20261123.Fbq1lPh4`。`conv-qkv-restore-verified.json` 的 SHA-256 为 `d035787d343c11b849a5abde203ebd185c43bdf2538f06d64c0eb31e4809335d`。
- 固定融合 `op_api` SHA-256 为 `2185231c7f78b0a14fdfdd2cbca8cb8482b91655453eea15ce7fdde7ead71321`。`ascend910_93` kernel 对象 SHA-256 为 `19c89bc15eb3c10ba7fb843e97a03ae9f6eba29783729e5262d07a4c3ec2aee7`。源码和 vendor 文件的完整锁在同名私有工作目录的 `source-lock.json`。
- 请求和采集外层新增 26.338 秒，累计 193.671 秒。三阶段加载分别为 81.121、79.435、77.089 秒，原服务恢复 87.820 秒，离线导出约 22.34 秒。采前 `c8c0…` 服务已恢复，18143 ready，22/22，双端状态指针 SHA-256 为 `819e399d794ecd8741ededf30e351c25301527e32a17c9d03f3ddb2829489285`。

下一阶段可以用这一条路径进入正式质量与性能验收。质量须分别核 OpenDecisions Calibration、Intelligence、Capability 和 ERQA 总体准确率的 2pp 门槛。S1 须做同配置全量对照。S2 须先测实际 step prefill 数和融合命中，因为多 prefill 保持原拆分路径。采用融合后的恢复验收需同时记录已加载 postimage、OPP 产物及实际融合命中，和现有 22 项健康与协议校验一起保存。
