# 2026-11-19 最终提交选定版本（aa3e fused / s1）证据

本目录汇总最终提交版本在本轮同窗口内测得的性能、质量与恢复原件。所有指标只使用官方 JSON benchmark 的
`decisions_per_second`（= successful / wall_seconds）；journal span（首 start→末 finish）重算仅作交叉核对，
末位差单注，不另立指标。

## 版本与配置

- 头：aa3e fused（`aa3e17585f8bd60a4d05e342abc7e2773c79a0b2d49f0370d15e29bcc6b72503`）
- 单一数据路径：BIC + GDN v2 + 单字段（SINGLE_FIELD=1）+ conv+QKV + chunk128 + MLP BF16 NZ + ACLNN cache
- joint 环境：`CLEF_JOINT_FUSED_ATTN=1`、`CLEF_JOINT_ALLOWED_POSTIMAGE_SHA=aa3e1758…`、`CLEF_JOINT_SINGLE_FIELD=1`
- 服务参数：TP2 / BF16 / eager / max len 4096；S1 seqs1 / batched 4096；S2 选点 B16 C64（seqs16 / batched 16384）
- 服务 env 逐字节：OPP 9 项（`7b79c5db…`）、LD 12 项（`ca27dbb7…`），对照冻结 expected
  `perf-wave2-20261118/s1-final-r1/launch-manifest.json`（12933 B，`fcbdcbf1…`）

## 结果

| 点 | p95（ms） | dec/s | accuracy | Capability | MAE |
| --- | ---: | ---: | ---: | ---: | ---: |
| S1 r1 | 156.888 | 8.5184 | 0.8095（1904/2352） | 74.247710 | 0.402612 |
| S1 r2 | 156.442 | 8.8209 | 同上 | 同上 | 同上 |
| S1 r3 | 156.116 | 8.7227 | 同上 | 同上 | 同上 |
| **S1 中位** | **156.4417** | 8.7227 | 0.8095 | 74.247710 | 0.402612 |
| B8 C8 | 937.613 | 14.7832 | 1903/2352 | 74.253107 | 0.402792 |
| B8 C16 | 1764.774 | 16.9037 | 1905/2352 | 73.825010 | 0.402936 |
| B16 C32 | 3117.810 | 17.6572 | 1902/2352 | 74.146850 | 0.402420 |
| **B16 C64（S2 选点）** | **5913.838** | **17.7047** | 1902/2352 | 74.141828 | 0.402338 |
| ERQA400 | — | — | **216/400 = 0.5400** | — | — |

同窗口 f810 对照：S1 中位 156.4051（157.513/156.405/156.144）；B8 14.7148/16.8259；B16 17.5432/17.5996；
ERQA 216/400。所有质量差在 2PP 内（相对 11-13 适配 baseline 最大 −0.6005 pp）。

## 目录结构

- `s1/`：候选 S1 三轮的 acceptance/execution/quality/performance JSON + `c1-r*.jsonl.gz`（逐题 raw）+ supervision
- `b8/`、`b16/`：typed 候选 B8/B16 两点的同结构原件
- `erqa/`：ERQA400 acceptance/summary/erqa-score、`predictions.jsonl.gz`、expected-inputs、supervision
- `restore/`：最终 restore 的 complete/restore/envcompare/actual/expected/restored-identities，以及 B8/B16 中间 restore 的 complete/envcompare/actual/expected/restored-identities
- `entry/`：官方 `start.sh` 入口验证原件——abs/（绝对参数首启：launch-manifest、joint-patch-record、patch-manifest、service.log、patch-backup）、rel/（相对参数重启：aa3e-noop 记录与日志）、run/（旧 f810 stop、首启 stop、wrapper 清理、最终服务身份/env、四检、生产根 dry-run）、backup/（受控部署备份清单）

远端原件路径（执行机）：
- S1：`/workspace/yangchen/logs/owner31-candidate-s1-20261119T0059Z/`
- B8：`/workspace/yangchen/logs/owner31-typed-s2b8-20261119T0150Z/`
- B16：`/workspace/yangchen/logs/owner31-typed-s2b16-20261119T0155Z/`
- ERQA：`/workspace/yangchen/logs/owner31-typed-erqa-20261119T0202Z/`

## 复现命令（要点）

```bash
# S1（aa3e 候选，candidate manifest 31202d27）
python <v5 supervisor 172c2838> --code <cc7> \
  --manifest <e20e128>/runtime/final_acceptance_candidate_manifest.json --manifest-sha 31202d27... \
  --phase s1 --output-dir <new> --records-dir <new> \
  --identities <deploy>/ready-identities.json --launch-manifest <label>/launch-manifest.json \
  --owner-evidence <deploy>/restore.json --deadline-utc <absolute> \
  --helper <5c>/runtime/joint_fused_attention_window.py --execute

# S2/ERQA typed 链：deploy -> 同 phase v5 -> restore（见 docs/official-entry-aa3e-2026-11-19.md）
```

## 证据身份

- 主记录：`docs/overnight-owner27-2026-11-19.md`（采用理由、分列与已关闭项）
- 收口提交：`934fff0`（f810 全链）、`da5a6d1`（typed 链）、`dd2f513`（采用与分列）
- 独立核验：32（生命周期/性能/raw/恢复）、33（逐 index 逐题质量）
- 官方入口：35 `f9e601c`（`inference.py de5b0d1e…`、stage `d7516e8a…`、evidence `1cad8ba1…`），
  入口验证原件在入口实测通过后补入本目录。
