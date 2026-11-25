#!/bin/bash
# usage: run_p1_point.sh <seqs> <concurrency> <requests>
set -u
SEQS=$1; CONC=$2; REQS=$3
TOOLS=/workspace/yangchen/work/profiling-tools-20261113
RUN=$(cat /workspace/yangchen/logs/tuning/CURRENT_RUN)
cd "$TOOLS"
source scripts/activate_env.sh >/dev/null 2>&1
LABEL="${LABEL_PREFIX:-p1}-s${SEQS}-c${CONC}${LABEL_SUFFIX:-}"
POINT="$RUN/p1/$LABEL"
mkdir -p "$POINT"
echo "=== point $LABEL start $(TZ=Asia/Shanghai date +%H:%M:%S) ==="

python scripts/manage_systemone_profile_service.py --run-dir "$RUN" stop > "$POINT/stop.json" 2>&1
sleep 2

python scripts/manage_systemone_profile_service.py --run-dir "$RUN" start \
  --label "$LABEL" --tp 2 --max-model-len 4096 --max-num-seqs "$SEQS" --multimodal \
  > "$POINT/launch.json" 2>&1
if [ $? -ne 0 ]; then echo "LAUNCH FAILED"; cat "$POINT/launch.json"; exit 1; fi

READY=""
for i in $(seq 1 48); do
  CODE=$(curl -s -m 2 -o /dev/null -w "%{http_code}" http://127.0.0.1:8000/health 2>/dev/null)
  if [ "$CODE" = "200" ]; then READY=yes; break; fi
  sleep 10
done
if [ -z "$READY" ]; then echo "HEALTH TIMEOUT"; exit 1; fi
echo "ready at $(TZ=Asia/Shanghai date +%H:%M:%S)"

npu-smi info 2>/dev/null > "$POINT/npu-smi-before.txt"
cp /sys/fs/cgroup/cpu/cpu.stat "$POINT/cpu-stat-before.txt"
bash /workspace/yangchen/logs/tuning/scrape_metrics.sh "$POINT/metrics-before.txt"

python scripts/benchmark_systemone.py --warmup 10 --requests "$REQS" \
  --concurrency "$CONC" --min-duration-seconds 30 --sample-hbm \
  --report "$POINT/benchmark.json" > "$POINT/benchmark-stdout.log" 2>&1
BENCH_RC=$?

bash /workspace/yangchen/logs/tuning/scrape_metrics.sh "$POINT/metrics-after.txt"
cp /sys/fs/cgroup/cpu/cpu.stat "$POINT/cpu-stat-after.txt"
npu-smi info 2>/dev/null > "$POINT/npu-smi-after.txt"

python3 /workspace/yangchen/logs/tuning/summarize_point.py "$POINT"
echo "=== point $LABEL done rc=$BENCH_RC $(TZ=Asia/Shanghai date +%H:%M:%S) ==="
