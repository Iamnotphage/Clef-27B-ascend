# `/v1/systemone` 图文 `state` 验收日志（2026-11-24）

分支 `feat/systemone-mm-state-20261124`，基线 `origin/main` `92cd072`。远端干净 checkout 为 `/workspace/yangchen/Clef-27B-ascend-systemone`；vLLM/Ascend 固定树为 `/workspace/yangchen/work/mix-ws-20261123/src/{vllm,vllm-ascend}`。

## 执行命令

```bash
source scripts/activate_env.sh
# 阶段 0：全量离线单测（含 2,352 OD 与 400 ERQA 全量断言）
python3 -m unittest runtime/test_systemone_mm_state.py \
  runtime/test_inference_launch.py scripts/official_eval/test_erqa_state_list.py

# 阶段 4：官方裸命令冷启动（单一常驻 TP2）
bash Clef-27B/start.sh --model-dir /workspace/yangchen/models/Clef-27B \
  --output-dir /workspace/yangchen/logs/systemone-mm-20261124/phase4-final-service
```

服务 manifest 记录 `CLEF_STATE_MM_MODE=extract_question_dict`、`CLEF_ENCODER_CACHE_TOKENS=262144`、`TASK_QUEUE_ENABLE=2`、`serving.py` postimage `99a8e1b7…`、模块 `a8b87841…`、joint `26d3ad04…`。

## 结果

| 项目 | 结果 | 文件 |
| --- | --- | --- |
| 离线单测 | 59/59 通过，0 skipped；OD 0/2,352 误判；ERQA 400/400 text join | 终端输出（本目录 final-summary） |
| 新接口冒烟 | 单图 366 token、16 图 10,294 token、交错 200、恶意 400、旧格式逐位一致、冻结 8 例逐位一致 | `phase4-smoke-final.json` |
| ERQA A/B | A1 213/400；A2 216/400；锁定 A2 | `erqa-A1-c1/`、`erqa-A2-c1/` |
| ERQA C1 冷/暖 | 216/400、P95 288.400/244.382 ms、逐位一致 | `erqa-final-c1-cold/`、`erqa-final-c1-warm/` |
| ERQA C16 | 217/400、8.240 dec/s、P95 2953.745 ms | `erqa-final-c16/` |
| OD S1 | 2,352/2,352、9.131 dec/s、P95 154.548 ms、Cap 74.2477 | `od-final-s1/` |
| OD S2 C16 / C64 | 16.345 / 17.193 dec/s、Cap 74.1510 / 74.0514 | `od-final-s2-c16/`、`od-final-s2-c64/` |
| OD 逐位对照 | 对已验收 `official-s1-r1`：max_prob_diff 0.0、0 flips、2,352/2,352 token 相等 | `final-summary.json` |
| ERQA 结构哈希 | C1/C16 均 400/400、0 mismatch | `erqa-final-c1-cold/offline-score.json` 等 |

ERQA `predictions.jsonl` 保留逐题 prediction、完整 probabilities、usage 与时间；OD 的 5MB 原始 report 留在远端 `/workspace/yangchen/logs/systemone-mm-20261124/od-final-*/report.json`，本目录保存其 summary 与 quality JSON。

## 清场

最终服务 PID 338998 于 2026-11-24 12:52 UTC 收到 SIGTERM，6 s 内退出；18143 连接被拒，vLLM/VLLMWorker 进程为 0，NPU 6 无进程。`phase4-final-service/stop.json` 记录。

## 阻塞

EvalScope 未安装于 `/workspace/yangchen/venvs/clef-mix-ws-py312`，镜像无离线包且 pip index 不可达；原生客户端联调未执行。其余阶段 4 项目均由固定 `/v1/systemone` 客户端覆盖。
