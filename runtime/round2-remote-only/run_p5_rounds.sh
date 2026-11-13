#!/bin/bash
set -u
RUN=$(cat /workspace/yangchen/logs/tuning/CURRENT_RUN)
cd /workspace/yangchen/work/profiling-tools-20261113
source scripts/activate_env.sh >/dev/null 2>&1
for ROUND in 2 3; do
  python scripts/manage_systemone_profile_service.py --run-dir "$RUN" stop >/dev/null 2>&1
  sleep 2
  LABEL="p5-cache-on-s4-r$ROUND"
  rm -rf "$RUN/$LABEL"
  python3 /workspace/yangchen/logs/tuning/start_service_custom.py --run-dir "$RUN" --label "$LABEL" --overrides-file "$RUN/overrides-cache-on-s4.json" >/dev/null 2>&1
  P="$RUN/$LABEL"
  python scripts/benchmark_systemone.py --warmup 10 --requests 300 --concurrency 8 --min-duration-seconds 30 --report "$P/benchmark.json" > "$P/benchmark-stdout.log" 2>&1
  python3 /workspace/yangchen/logs/tuning/summarize_point.py "$P" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d[\"label\"], \"rps=%.3f\"%d[\"rps\"], \"p50=%.1f\"%d[\"p50_ms\"], \"p95=%.1f\"%d[\"p95_ms\"], \"n=%d\"%d[\"successful\"])"
  grep -oE "schema token cache stats after [0-9]+ encodes: .*" "$P/service.log" | tail -1
done
echo "P5 ROUNDS DONE $(TZ=Asia/Shanghai date +%H:%M:%S)"
