#!/bin/bash
# usage: run_p2_point.sh <seqs> <tokens> <concurrency> <requests> <sample-file> <tokens-per-request>
# Starts the service with independent --max-num-batched-tokens via start_service_custom.py,
# then benchmarks the given sample file. Label: p2-s<seqs>-t<tokens>-c<conc>.
set -u
SEQS=$1; TOKENS=$2; CONC=$3; REQS=$4; SAMPLE=$5
TOOLS=/workspace/yangchen/work/profiling-tools-20261113
RUN=$(cat /workspace/yangchen/logs/tuning/CURRENT_RUN)
cd "$TOOLS"
source scripts/activate_env.sh >/dev/null 2>&1
LABEL="p2-s${SEQS}-t${TOKENS}-c${CONC}${LABEL_SUFFIX:-}"
POINT="$RUN/$LABEL"
mkdir -p "$POINT"
echo "=== point $LABEL start $(TZ=Asia/Shanghai date +%H:%M:%S) ==="

python scripts/manage_systemone_profile_service.py --run-dir "$RUN" stop > "$POINT/stop.json" 2>&1
sleep 2

python3 /workspace/yangchen/logs/tuning/start_service_custom.py --run-dir "$RUN" --label "$LABEL" \
  --overrides "{\"set\":{\"--max-num-seqs\":\"$SEQS\",\"--max-num-batched-tokens\":\"$TOKENS\"},\"wait_health_seconds\":600}" \
  > "$POINT/custom-launch.json" 2>&1
if [ $? -ne 0 ]; then echo "LAUNCH FAILED"; cat "$POINT/custom-launch.json"; exit 1; fi
echo "ready at $(TZ=Asia/Shanghai date +%H:%M:%S)"

npu-smi info 2>/dev/null > "$POINT/npu-smi-before.txt"
cp /sys/fs/cgroup/cpu/cpu.stat "$POINT/cpu-stat-before.txt"
bash /workspace/yangchen/logs/tuning/scrape_metrics.sh "$POINT/metrics-before.txt"
grep -iE "available|budget|alloc|KV cache|profile|memory" "$POINT/service.log" > "$POINT/startup-memory.log" 2>/dev/null || true

python scripts/benchmark_systemone.py --warmup 10 --requests "$REQS" \
  --concurrency "$CONC" --min-duration-seconds 30 --sample-hbm \
  --sample-file "$SAMPLE" \
  --report "$POINT/benchmark.json" > "$POINT/benchmark-stdout.log" 2>&1
BENCH_RC=$?

bash /workspace/yangchen/logs/tuning/scrape_metrics.sh "$POINT/metrics-after.txt"
cp /sys/fs/cgroup/cpu/cpu.stat "$POINT/cpu-stat-after.txt"
npu-smi info 2>/dev/null > "$POINT/npu-smi-after.txt"

python3 /workspace/yangchen/logs/tuning/summarize_point.py "$POINT"
# batch reconstruction with the actual dominant tokens-per-request
TPR_HIST=$(python3 -c "
import json
r=json.load(open('$POINT/benchmark.json'))
h=(r.get('summary',{}).get('actual_input_tokens') or {}).get('per_request_histogram') or {}
items=sorted(h.items(), key=lambda kv:-kv[1])
print(items[0][0] if items else 307)
")
python3 /workspace/yangchen/logs/tuning/reconstruct_batch.py \
  --before "$POINT/metrics-before.txt" --after "$POINT/metrics-after.txt" \
  --warmup 10 --tokens-per-request "$TPR_HIST" \
  --output "$POINT/batch-reconstruction.json" > /dev/null 2>&1
echo "=== point $LABEL done rc=$BENCH_RC tokens_per_request=$TPR_HIST $(TZ=Asia/Shanghai date +%H:%M:%S) ==="
