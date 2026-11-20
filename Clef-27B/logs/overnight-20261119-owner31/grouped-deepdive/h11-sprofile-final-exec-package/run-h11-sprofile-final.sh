#!/bin/bash
# H11-sprofile-final wrapper. Ordered gates: caller_env -> preflight_ALL_PASS -> collect_begin_rc0.
# No shell splitting/eval for the window argv; the builder output is one quoted JSON argument.
set -u
CELL=${H11_CELL:-/workspace/yangchen/work/grouped-deepdive-cell-h11-sprofile-final}
CODE=${H11_CODE:-/workspace/yangchen/work/grouped-deepdive-h11-sprofile-final-code}
VENV=${H11_VENV:-/workspace/yangchen/venvs/clef-py312/bin}
TORCHRUN=${H11_TORCHRUN:-$VENV/torchrun}
PREFLIGHT=${H11_PREFLIGHT:-$CODE/preflight-rescan-final.sh}
COLLECT=${H11_COLLECT:-$CODE/collect-raw-final.sh}
CAPTURE=${H11_CAPTURE:-$CODE/capture-identity-final.py}
SIDECAR=${H11_SIDECAR:-$CODE/write_env_sidecar.py}
BUILDER=${H11_BUILDER:-$CODE/build_window_argv.py}
CHILD=${H11_CHILD:-$CODE/child-argv.json}
WINDOW=${H11_WINDOW:-$CODE/supervised_probe_window.py}
RUN_WINDOW=${H11_RUN_WINDOW:-$CODE/run_window.py}
ROOT_PID=${H11_ROOT_PID:?H11_ROOT_PID required}
ARCH=${H11_ARCH:-$CELL/archive-h11-sprofile-final}
IDENT=$CELL/root-identity-h11-sprofile-final.json
SIDE=$CELL/env-sidecar-h11-sprofile-final.json
RUN=$CELL/run-h11-sprofile-final
TEMP=$CELL/probe-temp-h11-sprofile-final
OUT=$CELL/probe-out-h11-sprofile-final
RESTORE=$CELL/restore-h11-sprofile-final
export ASCEND_GLOBAL_LOG_LEVEL=0
export ASCEND_PROCESS_LOG_PATH="$CELL/logs-h11-sprofile-final"
export CLEF_WINDOW_OWNER=h11-sprofile-final
export CELL CODE
# 1) caller env (before any stop)
t=$(command -v torchrun 2>/dev/null || true)
[ "$t" = "$TORCHRUN" ] || { echo "caller torchrun mismatch: ${t:-MISSING}" >&2; exit 3; }
"$TORCHRUN" --help >/dev/null 2>&1 || { echo "torchrun --help failed" >&2; exit 4; }
# 1a) fresh identity/sidecar (execution-time; preflight input)
"$VENV/python" "$CAPTURE" --pid "$ROOT_PID" --out "$IDENT" >/dev/null || { echo "identity capture failed" >&2; exit 2; }
"$VENV/python" "$SIDECAR" "$IDENT" "$SIDE" || { echo "sidecar write failed" >&2; exit 2; }
# 2) preflight ALL PASS (before stop)
bash "$PREFLIGHT" || { echo "preflight failed; abort before stop" >&2; exit 5; }
# 3) collect begin (exit 6; no window/stop on failure)
cp -p "$CODE/plan-h11-sprofile-final.json" "$CELL/plan-h11-sprofile-final.json" || { echo "plan copy failed" >&2; exit 6; }
CELL="$CELL" ARCH="$ARCH" SRC_PLAN="$CELL/plan-h11-sprofile-final.json" bash "$COLLECT" begin || { echo "collect begin failed; abort before stop" >&2; exit 6; }
mkdir -p "$CELL/logs-h11-sprofile-final"
# 4) canonical child -> builder -> one window invocation (no eval/split)
WINDOW_JSON=$("$VENV/python" "$BUILDER" \
  --window-path "$WINDOW" --python "$VENV/python" --run-dir "$RUN" \
  --root-identity "$IDENT" --stop-script /tmp/d0cc/d0_stop.py \
  --expect-stop-sha d6d49d75012a0789d3bff3e7b522e9cd496844ca0ee0424608ab8e6b43a33886 \
  --source-target /workspace/yangchen/work/src/vllm/vllm/model_executor/models/clef_joint_schema.py \
  --expected-source-sha aa3e17585f8bd60a4d05e342abc7e2773c79a0b2d49f0370d15e29bcc6b72503 \
  --official-start /workspace/yangchen/Clef-27B-ascend/Clef-27B/start.sh \
  --activate /workspace/yangchen/Clef-27B-ascend/scripts/activate_env.sh \
  --service-root /workspace/yangchen/Clef-27B-ascend \
  --model-dir /workspace/yangchen/models/Clef-27B \
  --restore-output "$RESTORE" \
  --clean-env-expect /workspace/yangchen/logs/perf-wave2-20261118/s1-final-r1/launch-manifest.json \
  --text-body /workspace/yangchen/work/grouped-deepdive-cell-a15-large/text-body.json \
  --image-body /workspace/yangchen/work/grouped-deepdive-cell-a15-large/image-body.json \
  --port 18143 --child-timeout 240.0 --stop-timeout 120.0 --restore-timeout 900.0 \
  --comm-timeout 90.0 --master-port 29531 --settle-timeout 300 --execute) || { echo "window argv build failed" >&2; exit 7; }
"$VENV/python" "$RUN_WINDOW" "$WINDOW_JSON"
wrc=$?
# 5) single collect end; preserve window RC; end failure keeps partial and does not auto-retry
CELL="$CELL" ARCH="$ARCH" bash "$COLLECT" end
erc=$?
if [ "$wrc" -ne 0 ]; then exit "$wrc"; fi
exit "$erc"
