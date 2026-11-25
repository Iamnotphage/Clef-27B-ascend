#!/bin/bash
# usage: run_p3_point.sh <total-concurrency> <requests> [per-instance-seqs]
# Dual TP1 deployment: two instances bound to logical NPU 0/1 of the same
# physical card, behind the single dispatch entry on port 8002.
set -u
CONC=$1; REQS=$2; SEQSI=${3:-1}
TOOLS=/workspace/yangchen/work/profiling-tools-20261113
RUN=$(cat /workspace/yangchen/logs/tuning/CURRENT_RUN)
cd "$TOOLS"
source scripts/activate_env.sh >/dev/null 2>&1
LABEL="p3-c${CONC}${LABEL_SUFFIX:-}"
POINT="$RUN/$LABEL"
mkdir -p "$POINT"
echo "=== point $LABEL start $(TZ=Asia/Shanghai date +%H:%M:%S) ==="

# stop any snapshotted single service, then any recorded dual instances
python scripts/manage_systemone_profile_service.py --run-dir "$RUN" stop > "$POINT/stop-single.json" 2>&1
python3 /workspace/yangchen/logs/tuning/stop_dual_tp1.py --run-dir "$RUN" > "$POINT/stop-dual.json" 2>&1
sleep 2

# start dual TP1 (staggered inside the launcher), then the dispatch proxy
DUAL_TP1_SEQS="$SEQSI" python3 /workspace/yangchen/logs/tuning/start_dual_tp1.py \
  --run-dir "$RUN" --label "$LABEL-instances" > "$POINT/dual-launch.json" 2>&1
if [ $? -ne 0 ]; then echo "DUAL LAUNCH FAILED"; cat "$POINT/dual-launch.json" | tail -5; exit 1; fi

python3 /workspace/yangchen/logs/tuning/dispatch_proxy.py --listen-port 8002 \
  --targets 127.0.0.1:8000,127.0.0.1:8001 --stats-file "$POINT/proxy-stats.json" \
  > "$POINT/proxy.log" 2>&1 &
PROXY_PID=$!
echo $PROXY_PID > "$POINT/proxy.pid"
sleep 2
CODE=$(curl -s -m 5 -o "$POINT/proxy-health.json" -w "%{http_code}" http://127.0.0.1:8002/health)
if [ "$CODE" != "200" ]; then echo "PROXY HEALTH FAILED"; cat "$POINT/proxy-health.json"; exit 1; fi
echo "proxy ready at $(TZ=Asia/Shanghai date +%H:%M:%S)"

# device binding evidence: which pid uses which chip
npu-smi info 2>/dev/null > "$POINT/npu-smi-before.txt"
cp /sys/fs/cgroup/cpu/cpu.stat "$POINT/cpu-stat-before.txt"
free -m > "$POINT/hostmem-before.txt"

bash /workspace/yangchen/logs/tuning/scrape_metrics.sh "$POINT/metrics-A-before.txt"
curl -s -m 10 http://127.0.0.1:8001/metrics > "$POINT/metrics-B-before.txt"

python scripts/benchmark_systemone.py --warmup 10 --requests "$REQS" \
  --concurrency "$CONC" --min-duration-seconds 30 --sample-hbm \
  --base-url http://127.0.0.1:8002 \
  --report "$POINT/benchmark.json" > "$POINT/benchmark-stdout.log" 2>&1
BENCH_RC=$?

bash /workspace/yangchen/logs/tuning/scrape_metrics.sh "$POINT/metrics-A-after.txt"
curl -s -m 10 http://127.0.0.1:8001/metrics > "$POINT/metrics-B-after.txt"
cp /sys/fs/cgroup/cpu/cpu.stat "$POINT/cpu-stat-after.txt"
free -m > "$POINT/hostmem-after.txt"
npu-smi info 2>/dev/null > "$POINT/npu-smi-after.txt"

kill -TERM "$PROXY_PID" 2>/dev/null
sleep 3
python3 /workspace/yangchen/logs/tuning/stop_dual_tp1.py --run-dir "$RUN" > "$POINT/stop-dual-end.json" 2>&1

python3 /workspace/yangchen/logs/tuning/summarize_point.py "$POINT"
TPR_HIST=$(python3 -c "
import json
r=json.load(open('$POINT/benchmark.json'))
h=(r.get('summary',{}).get('actual_input_tokens') or {}).get('per_request_histogram') or {}
items=sorted(h.items(), key=lambda kv:-kv[1])
print(items[0][0] if items else 307)
")
python3 /workspace/yangchen/logs/tuning/reconstruct_batch.py \
  --before "$POINT/metrics-A-before.txt" --after "$POINT/metrics-A-after.txt" \
  --warmup 10 --tokens-per-request "$TPR_HIST" \
  --output "$POINT/batch-A.json" > /dev/null 2>&1
python3 /workspace/yangchen/logs/tuning/reconstruct_batch.py \
  --before "$POINT/metrics-B-before.txt" --after "$POINT/metrics-B-after.txt" \
  --warmup 10 --tokens-per-request "$TPR_HIST" \
  --output "$POINT/batch-B.json" > /dev/null 2>&1
echo "=== point $LABEL done rc=$BENCH_RC tpr=$TPR_HIST $(TZ=Asia/Shanghai date +%H:%M:%S) ==="
