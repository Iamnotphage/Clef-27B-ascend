# 新机 aa3e Conv+QKV 启动恢复验收

## 变化

新机原入口为 Conv+QKV 注入了 vendor 与开关，实际 GDN 源码仍是拆分版。恢复版入口在 worker 导入前对实际 vLLM-Ascend 源树独立验收 Conv+QKV 能力。对已知拆分 SHA 执行 check、apply、check，对目标 SHA 幂等通过，未知预像拒绝。OFF 使用拆分 fallback，dry-run 只读。模型权重和激活保持 BF16，联合头保持 aa3e，FLA 保持 chunk128。

## 正式结果

A 为 c8c0 拆分，B 为 ead25 融合。S1 为 A 三轮后 B 三轮，每臂同一进程，C1、batch 1、4096/4096、util .90，client warmup 0。下表吞吐采用 benchmark wall，P95 采用线性插值。两项三轮中位数各自取值。

| S1 | 拆分 A | 融合 B | 变化 |
| --- | ---: | ---: | ---: |
| P95 r1 / r2 / r3（ms） | 161.203456 / 160.277255 / 160.447190 | 156.967955 / 155.760317 / 156.181845 | 三轮编号对照均改善 |
| P95 中位（ms） | 160.447190 | 156.181845 | −4.265346（−2.66%） |
| 吞吐中位（decisions/s） | 8.417925 | 8.948975 | +6.31% |

A 的三轮 P95 范围为 0.926201 ms。S1 共 14,112/14,112 成功，零失败。分块顺序限制了对时间漂移的控制。

| S2 点位 | A decisions/s | B decisions/s | A P95（ms） | B P95（ms） |
| --- | ---: | ---: | ---: | ---: |
| B8 C8 | 14.400519 | 14.929262 | 941.620 | 936.186 |
| B8 C16 | 16.929487 | 16.920164 | 1763.870 | 1763.581 |
| B16 C32 | 17.589396 | 17.606152 | 3121.227 | 3117.397 |
| B16 C64 | 17.694092 | 17.689352 | 5922.168 | 5918.830 |

S2 每臂每点一轮，同一 batch 的两个并发点沿用同进程。共 18,816/18,816 成功，零失败。最大有效吞吐点 B16 C64 差 −0.02679%，四点未显示一致退化。另有一次 192 请求的 B16 C64 命中观测，每 rank 13 个 step 中 1 个命中，另外 12 个为多 prefill 且 M 超过 1600。该短窗独立于正式计时，覆盖率限于本次观测。

| 质量口径 | A | B | B 对固定基准 |
| --- | ---: | ---: | ---: |
| OD S1 Calibration | 82.705628402 | 82.411343998 | −0.382771 pp |
| OD S1 Intelligence | 66.116927051 | 66.084075602 | +0.027233 pp |
| OD S1 Capability | 74.411277726 | 74.247709800 | −0.177769 pp |
| OD S1 Accuracy | 1905/2352 | 1904/2352 | −2 题 |
| OD S1 Score MAE（等级） | 0.402633979 | 0.402612340 | +0.001223 |
| ERQA 冷遍 | 217/400 | 216/400 | −0.75 pp |
| ERQA 自然第二遍 | 217/400 | 216/400 | −0.75 pp |

每轮 OD 三项与每遍 ERQA 均通过固定适配基准的绝对 2pp 门。Accuracy 和 MAE 单独记录。ERQA 两臂各自复用同一个新进程，GET-only 就绪后依次各跑 400 题。两臂共 1,600/1,600，零失败、零重试。实际 .85/16384/262144 与 GET/POST 顺序由原日志摘要和监督记录复核。已退出进程精确的 `/proc` 启动 ticks 与完整 ERQA 服务日志不在回传包内。全部评测使用原始输出概率。

历史 aa3e 的 156.4417 ms 和 17.7047 decisions/s 属于 11-19 归档。本页效果由新机同窗 A/B 得出。

## 证据和复现

- [验收索引和逐轮原数](manifest.json) 固定输入、工具、原始 report/journal 的压缩前后 SHA、质量和执行统计。
- [OD 逐题证据](od-requests.jsonl.gz) 保留 32,928 条答案、完整概率、请求哈希、起止时间和 HTTP 状态。[ERQA 逐题证据](erqa-predictions.jsonl.gz) 保留 1,600 条预测、标签、概率、wire/图像哈希和计时。两个文件是从原件核对后导出的逐题副本。原 report/journal 各自字节哈希在索引中。
- [部署证据](deployment.json) 记录普通冷请求、预热后 Level1、最终无 profiler 幂等冷启动，以及运行源码和 vendor 哈希。归档保留两次部署验证和回滚边界，最终在线状态随 PR 执行记录更新。Level1 计数只覆盖关联代表请求，历史三个代表样本与正式全量评分分开记录。

正常入口使用 `bash Clef-27B/start.sh --config s1 --model-dir "$MODEL" --output-dir "$NEW"`。安装后需同时检查 launch manifest 的 `conv_qkv_patch.ready=true` 和 `source_sha256=ead25cb…`。在已知已安装源上使用 `CLEF_GDN_CONV_QKV=0` 走拆分 fallback。正式 A 的源为 c8c0，正式 B 为 ead25。S2 同样使用 s1 配置，依次覆盖 `--max-num-seqs 8 --max-num-batched-tokens 8192` 或 `--max-num-seqs 16 --max-num-batched-tokens 16384`，并按表切换客户端并发。ERQA 沿用 s1 配置并覆盖 `--max-model-len 16384 --max-num-batched-tokens 16384 --encoder-cache-tokens 262144 --gpu-memory-utilization .85 --limit-mm-per-prompt '{"image":16,"video":0}'`，媒体像素使用 `--mm-processor-kwargs '{"min_pixels":65536,"max_pixels":1048576}'`。

评测器固定到 `cc7f32842a78786a8f0e5da373f66597752ea9ab`（归档分支 `ao/clef-evaluation/final-acceptance`），先逐字节核对索引中的数据和模板。每次使用新输出目录。以下命令在评测器归档树中执行，`BASE` 指对应新机服务， `C` 取表中并发。

```sh
python runtime/od_full_benchmark.py --sample-file "$OD_SAMPLES" \
  --url "$BASE/v1/systemone" --concurrency "$C" --warmup 0 --timeout 300 \
  --report "$REPORT" --journal "$JOURNAL"
python runtime/od_raw_acceptance.py --data "$OD_DATA" --scorer "$OD_SCORER" \
  --template "$OD_TEMPLATE" --run "candidate=$REPORT" --reference candidate --output "$OD_SCORE"
python scripts/official_eval/erqa_baseline.py --data-dir "$ERQA_DATA" --output-dir "$ERQA_RUN" \
  --api-url "$BASE/v1" --model clef --scope full --retries 1 --retry-interval 5 --timeout 600
python scripts/official_eval/erqa_score.py --data-dir "$ERQA_DATA" --run-dir "$ERQA_RUN" \
  --verify-requests --output "$ERQA_SCORE"
```
