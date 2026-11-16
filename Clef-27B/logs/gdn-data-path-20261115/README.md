# GDN 数据路径与后续优化证据（2026-11-15/16）

本目录是 `docs/gdn-data-path-2026-11-15.md` 及后续 host/MLP 诊断的精简证据。远端原始日志与 trace 在 `/workspace/yangchen/logs/gdn-data-path-20261115/`；`evidence-index.json` 逐文件记录 202 个原始文件的大小与 SHA-256。

## 最终采用的提交配置

- TP2 / BF16 / eager / max_model_len 4096 / max_num_seqs 1 / prefix cache 关 / chunked prefill 关；
- GDN 决策 prefill 默认开启，数据路径 **BIC**（`bsnd` + `skip_initial_state` + `skip_conv_state`）默认开启；
- 正常入口 S1 P95 **198.42 ms**（2,352/2,352），Capability **74.425479**、accuracy 1906/2352、Score MAE 0.401389；
- 6 对 A/B BIC vs R p95 中位 **−7.01%（−13.81 ms）**；
- ERQA 400 题 R/BIC 概率逐位一致，accuracy 219/400；
- seqs4 C8 **14.652 dec/s**、C16 14.689（S2 候选，C8 质量未过 0.03/0.05 门槛）。

## 目录

| 目录/文件 | 内容 |
| --- | --- |
| `phase0/` | 服务/设备/版本快照、契约探针、v3 安装报告、本地 git 快照 |
| `screen/`、`ab/` | 八臂筛选、6 对 R/BIC A/B |
| `full/`、`entry/` | 全量 OD R/BIC 与正常入口质量 |
| `erqa/` | ERQA R/BIC 摘要与逐题对比 |
| `seqs4/` | C8/C16 与合批质量 |
| `mlp-nz/` | BF16 NZ 局部 cold 矩阵与方向变体 |
| `hbm-bandwidth.json`、`matmul-ceiling.json` | HBM 1.26–1.32 TB/s、纯 matmul 300–320 TFLOPS |
| `fusion/` | Ascend C `ClefGdnQkvNorm` 原型探针（不接入） |
| `mc2/` | `MatmulAllReduce` 161001、Grouped 单组 SIGSEGV/挂起（不采用） |
| `profile-residual/` | 18.8 ms 残余的 CPU scope 与设备窗口 |
| `graph-test/`、`eager-test/` | ACL graph vs eager（graph 不采用） |
| `schema-cache-test/`、`nz-test/` | schema cache、BF16 NZ 服务化复测（均不采用） |
| `restore*/` | 各轮服务恢复核验 |

## 最终服务状态

baseline 服务 **PID 653005**，`127.0.0.1:8000`；argv/cwd/env 与 phase-0 逐项一致，`/health` 200，冻结 8 例 8/8。`current-service.json` 已更新。提交分支 `codex/tuning-perf2a-20261113`，HEAD 与 origin 一致。
