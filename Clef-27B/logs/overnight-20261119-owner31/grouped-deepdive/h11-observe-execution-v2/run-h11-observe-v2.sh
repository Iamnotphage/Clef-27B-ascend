#!/bin/bash
# H11-observe-v2 单一 wrapper（doc-only；仅在 20 明确 v2 GO 后使用）
# 作用：在同一个 shell/进程环境内完成 env -> torchrun 校验 -> preflight(rc0) -> window --execute。
# 严禁在未授权时运行；严禁拆 shell 手抄 argv 绕过本 wrapper。
set -u
CELL=/workspace/yangchen/work/grouped-deepdive-cell-h11-observe-v2
STAGE=/workspace/yangchen/work/grouped-deepdive-35-e38b541/runtime/grouped_mmar
VENV=/workspace/yangchen/venvs/clef-py312

source /workspace/yangchen/Clef-27B-ascend/scripts/activate_env.sh || { echo "activate_env failed" >&2; exit 2; }
export PATH="$VENV/bin:$PATH"
export ASCEND_GLOBAL_LOG_LEVEL=0
export ASCEND_PROCESS_LOG_PATH="$CELL/logs-h11-observe-v2"

t=$(command -v torchrun 2>/dev/null || true)
[ "$t" = "$VENV/bin/torchrun" ] || { echo "caller torchrun mismatch: ${t:-MISSING}" >&2; exit 3; }
torchrun --help >/dev/null 2>&1 || { echo "torchrun --help failed" >&2; exit 4; }
bash "$CELL/execution-v2/preflight-rescan.sh" || { echo "preflight failed; abort before stop" >&2; exit 5; }

export ARCH="$CELL/archive-h11-observe-v2-attempt2"
bash "$CELL/execution-v2/collect-raw.sh" begin || { echo "collect begin failed; abort before stop" >&2; exit 6; }

exec /workspace/yangchen/venvs/clef-py312/bin/python /workspace/yangchen/work/grouped-deepdive-35-e38b541/runtime/grouped_mmar/supervised_probe_window.py --python /workspace/yangchen/venvs/clef-py312/bin/python --run-dir /workspace/yangchen/work/grouped-deepdive-cell-h11-observe-v2/run-h11-observe-v2 --root-identity /workspace/yangchen/work/grouped-deepdive-cell-h11-observe-v2/root-identity-h11-observe-v2.json --stop-script /tmp/d0cc/d0_stop.py --expect-stop-sha d6d49d75012a0789d3bff3e7b522e9cd496844ca0ee0424608ab8e6b43a33886 --arm H11 --adapter /workspace/yangchen/work/grouped-deepdive-35-e38b541/runtime/grouped_mmar/probe_adapter.so --probe-temp-root /workspace/yangchen/work/grouped-deepdive-cell-h11-observe-v2/probe-temp-h11-observe-v2 --probe-out /workspace/yangchen/work/grouped-deepdive-cell-h11-observe-v2/probe-out-h11-observe-v2 --k 256 --n 512 --rows 128 --seed 20261119 --master-port 29531 --comm-timeout 90.0 --child-timeout 240.0 --stop-timeout 120.0 --restore-timeout 900.0 --port 18143 --source-target /workspace/yangchen/work/src/vllm/vllm/model_executor/models/clef_joint_schema.py --expected-source-sha aa3e17585f8bd60a4d05e342abc7e2773c79a0b2d49f0370d15e29bcc6b72503 --official-start /workspace/yangchen/Clef-27B-ascend/Clef-27B/start.sh --activate /workspace/yangchen/Clef-27B-ascend/scripts/activate_env.sh --service-root /workspace/yangchen/Clef-27B-ascend --model-dir /workspace/yangchen/models/Clef-27B --restore-output /workspace/yangchen/work/grouped-deepdive-cell-h11-observe-v2/restore-h11-observe-v2 --clean-env-expect /workspace/yangchen/logs/perf-wave2-20261118/s1-final-r1/launch-manifest.json --text-body /workspace/yangchen/work/grouped-deepdive-cell-a15-large/text-body.json --image-body /workspace/yangchen/work/grouped-deepdive-cell-a15-large/image-body.json --execute
