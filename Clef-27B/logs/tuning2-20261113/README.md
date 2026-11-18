# 2026-11-13 第二轮性能调优证据索引

远端完整实验目录：`/workspace/yangchen/logs/tuning/20261113-085408-perf2a/`（含每个测量点的 benchmark 逐请求 JSON、metrics 前后差分、cpu.stat、npu-smi、服务日志）。

| 文件 | 内容 |
| --- | --- |
| `master-results.csv` / `.json` | 全轮 43 个测量点汇总（阶段1-6），含逐点 batch 证据 |
| `p1-matrix.csv` / `.json` | 阶段1 探索矩阵 12 点（seqs×并发） |
| `phase0-summary.json` | 现场、快照、回退验证（重启后概率差 0.0） |
| `quality-b4-batched-precision.json` | B4 混合并发 551 响应对 CPU 参考：771/771 离散一致，概率差 0.000851 |
| `p5-quality-batched-precision.json` | schema 缓存开启条件下的同口径质量门（结果同上） |
| `p6-batched-precision-compile.json` | PIECEWISE 编译模式质量门：759/759 离散一致 |
| `p5-protocol-cache-on.json` / `p6-protocol-compile.json` | 协议校验报告 |
| `p6-mm-pool12.json` | 12 GiB 池红/蓝冷图回归 |
| `encoding-evidence.json` | 固定编码证据（5 用例，与 CPU 参考一致） |
| `opendecisions-final-author-score.json` | 推荐配置完整 OpenDecisions 作者评分（与 baseline 逐位一致） |
| `schema-cache.patch` | 固定 vLLM strategies.py 的有界 schema token 缓存补丁 |

## 源码与配置来源

- 固定 vLLM HEAD `24420b7a341af37a9bc3186e6fc6062512e0aa58`；strategies.py 基线 SHA256 `d7e25f1d879cd0dc2d8e38629805a66018a9c63e364572d6cbfee6a9fedd97bb`（补丁前，工作区另有四份既有安装补丁未触碰）。
- 固定 vLLM-Ascend `89eb8a5e7a76d085b453a1aeadb6883e12d5ea2d`（本轮只读，无改动）。
- 数据：OpenDecisions 1.0.1 revision `ff0b640be2b72c7d06d9d656a34ad6eed79ec7fe`，JSONL SHA256 `78614972006c…`；作者评分器 SHA256 `478d14ae…`。
- 资源边界：A3 单物理卡（npu-smi 卡 7，Chip 0/1，逻辑 0/1，Phy-ID 14/15，各 64 GiB HBM）；cgroup 32 核等价 CPU、128 GiB 内存。

## 测量纪律

每个探索点：预热 10 次顺序请求（B1）与正式分离；正式 ≥300 请求且窗口 ≥30 秒；HBM 1 秒采样；Prometheus `/metrics` 前后差分重建实际 batch（预热扣除 + 桶约束整数解，逐点校验误差 0）；cgroup cpu.stat 节流差分（全部为 0）。稳定测量为三轮交错（A/B/C×3）。客户端计时含 HTTP 传输、UTF-8/JSON 解码与协议校验；吞吐按校验通过的有效请求与实际决策数计。
