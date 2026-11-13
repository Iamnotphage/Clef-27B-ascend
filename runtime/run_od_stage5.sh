#!/bin/bash
# Stage 5 runner, two modes:
#   s1mode: subset regression + ramp pass + S1 full-2352 (c1, P95) + full quality
#   s2mode: subset regression + S2 concurrency scan (c2/4/8/16) on seqs4+cache config
# Usage: run_od_stage5.sh <mode> <tag> <env-json> <seqs-set-json>
set -u
TOOLS=/workspace/yangchen/work/profiling-tools-20261113
RUN=$(cat /workspace/yangchen/logs/tuning/CURRENT_RUN)
MODE="$1"; TAG="$2"; ENVJ="$3"; SETJ="$4"
OD="$RUN/opendecisions/$TAG"
mkdir -p "$OD"
cd "$TOOLS"
source scripts/activate_env.sh >/dev/null 2>&1
LOG="$OD/run.log"
ALL="$RUN/opendecisions/all2352.jsonl"
SUB="$RUN/opendecisions/subset-regression.json"

stop_svc() { python scripts/manage_systemone_profile_service.py --run-dir "$RUN" stop >> "$LOG" 2>&1; sleep 3; }
start_svc() {
  python3 /workspace/yangchen/logs/tuning/start_service_custom.py --run-dir "$RUN" \
    --label "od-$TAG" --overrides "{\"env\":$ENVJ,\"set\":$SETJ,\"wait_health_seconds\":900}" >> "$LOG" 2>&1
  for _w in $(seq 1 120); do
    C=$(curl -s -m 3 -o /dev/null -w "%{http_code}" http://127.0.0.1:8000/health || true)
    [ "$C" = "200" ] && break; sleep 10
  done
  [ "$C" = "200" ] || { echo "HEALTH FAILED" | tee -a "$LOG"; exit 1; }
  echo "$(TZ=Asia/Shanghai date +%T) service up for $TAG ($MODE)" | tee -a "$LOG"
}

stop_svc
start_svc

if [ "$MODE" = "s1mode" ]; then
  # subset regression (response capture for comparison against baselines)
  python scripts/benchmark_systemone.py --warmup 1 --requests 102 --concurrency 1 \
    --sample-file "$SUB" --base-url http://127.0.0.1:8000 \
    --report "$OD/subset.json" > "$OD/subset.log" 2>&1
  echo "subset rc=$?" | tee -a "$LOG"
  # ramp pass: full sequence once (discarded), then formal S1 pass
  python scripts/benchmark_systemone.py --warmup 0 --requests 2352 --concurrency 1 \
    --sample-file "$ALL" --base-url http://127.0.0.1:8000 \
    --report "$OD/s1-ramp.json" > "$OD/s1-ramp.log" 2>&1
  echo "s1-ramp rc=$?" | tee -a "$LOG"
  python scripts/benchmark_systemone.py --warmup 0 --requests 2352 --concurrency 1 \
    --sample-file "$ALL" --base-url http://127.0.0.1:8000 \
    --report "$OD/s1-full.json" > "$OD/s1.log" 2>&1
  echo "s1 rc=$?" | tee -a "$LOG"
  # full quality via official harness (deterministic outputs; ramp irrelevant)
  HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 \
/workspace/yangchen/venvs/clef-eval-py312/bin/python \
    /workspace/yangchen/Clef-27B-ascend/scripts/run_evalscope_offline.py \
    --data-dir /workspace/yangchen/datasets/clef-full-20261112/full-data \
    --work-dir "$OD/evalscope" \
    --api-url http://127.0.0.1:8000/v1 \
    --benchmark open_decisions --scope full > "$OD/quality.log" 2>&1
  echo "quality rc=$?" | tee -a "$LOG"
  # author-score cross check from saved probabilities
  QD="$OD/evalscope"
  PRED=$(ls "$QD"/predictions/clef/*.jsonl 2>/dev/null | head -1)
  if [ -n "$PRED" ]; then
    export CLEF_OD_DATA=/workspace/yangchen/datasets/clef-full-20261112/full-data/open-decisions2352.jsonl
    export CLEF_OD_SCORER=/workspace/yangchen/logs/opendecisions-baseline-20261113-1600/author-score.py
    /workspace/yangchen/venvs/clef-eval-py312/bin/python \
      /workspace/yangchen/Clef-27B-ascend/scripts/official_eval/convert_open_decisions_predictions.py \
      --predictions "$PRED" --data "$CLEF_OD_DATA" --scorer "$CLEF_OD_SCORER" \
      --output "$OD/author-predictions.jsonl" > "$OD/author-conversion.json" 2>&1
    echo "convert rc=$?" | tee -a "$LOG"
    /workspace/yangchen/venvs/clef-eval-py312/bin/python "$CLEF_OD_SCORER" \
      "$OD/author-predictions.jsonl" --data "$CLEF_OD_DATA" --json > "$OD/author-score.json" 2>&1
    echo "author-score rc=$?" | tee -a "$LOG"
  fi
fi

if [ "$MODE" = "s2mode" ]; then
  python scripts/benchmark_systemone.py --warmup 1 --requests 102 --concurrency 1 \
    --sample-file "$SUB" --base-url http://127.0.0.1:8000 \
    --report "$OD/subset.json" > "$OD/subset.log" 2>&1
  echo "subset rc=$?" | tee -a "$LOG"
  for C in 2 4 8 16; do
    python scripts/benchmark_systemone.py --warmup 1 --requests 2352 --concurrency "$C" \
      --sample-file "$ALL" --base-url http://127.0.0.1:8000 \
      --report "$OD/s2-c$C.json" > "$OD/s2-c$C.log" 2>&1
    echo "s2-c$C rc=$?" | tee -a "$LOG"
  done
fi

stop_svc
python3 - <<'EOF'
import os, shutil
p = "/workspace/yangchen/logs/tuning/20261113-120651-perf3/restored-tp2-4096"
if os.path.exists(p):
    shutil.move(p, p + "-old")
EOF
python scripts/manage_systemone_profile_service.py --run-dir "$RUN" restore >> "$LOG" 2>&1
for _w in $(seq 1 120); do C=$(curl -s -m 3 -o /dev/null -w "%{http_code}" http://127.0.0.1:8000/health || true); [ "$C" = "200" ] && break; sleep 10; done
echo "restored health=$C $(TZ=Asia/Shanghai date +%T)" | tee -a "$LOG"
echo "OD stage5 $MODE/$TAG done" | tee -a "$LOG"
