#!/bin/bash
# Cache v2 interleaved A/B: same source, CLEF_SCHEMA_TOKEN_CACHE 0/1 only.
# S2: s4-c8 fixed307 x3 pairs. S1-latency: s1-c1 fields-8 x3 pairs (with ramp).
# Pair order is balanced (odd pairs OFF->ON, even pairs ON->OFF). The archived
# cache-ab-analysis.json was collected OFF->ON for all three pairs and is
# therefore a fixed-order observation, not a causal estimate.
set -u
TOOLS=/workspace/yangchen/work/profiling-tools-20261113
RUN=$(cat /workspace/yangchen/logs/tuning/CURRENT_RUN)
AB="$RUN/ab-cache"
mkdir -p "$AB"
cd "$TOOLS"
source scripts/activate_env.sh >/dev/null 2>&1
LOG="$AB/driver.log"
FIELDS8=/workspace/yangchen/logs/profiling/20261112-223732-systemone/workloads/fields-8.json

stop_svc() { python scripts/manage_systemone_profile_service.py --run-dir "$RUN" stop >> "$LOG" 2>&1; sleep 3; }
start_svc() { # $1 label $2 extra-set-json $3 env
  python3 /workspace/yangchen/logs/tuning/start_service_custom.py --run-dir "$RUN" \
    --label "$1" --overrides "{\"env\":$3,\"set\":$2,\"wait_health_seconds\":600}" >> "$LOG" 2>&1
  for _w in $(seq 1 60); do C=$(curl -s -m 3 -o /dev/null -w "%{http_code}" http://127.0.0.1:8000/health || true); [ "$C" = "200" ] && break; sleep 10; done
  [ "$C" = "200" ] || { echo "HEALTH FAIL $1" | tee -a "$LOG"; exit 1; }
}

bench() { # $1 tag $2 conc $3 sample-file(or empty)
  SF=""
  [ -n "$3" ] && SF="--sample-file $3"
  python scripts/benchmark_systemone.py --warmup 10 --requests 300 --concurrency "$2" \
    --min-duration-seconds 30 --sample-hbm $SF --base-url http://127.0.0.1:8000 \
    --report "$AB/$1.json" > "$AB/$1.stdout" 2>&1
  echo "$(TZ=Asia/Shanghai date +%T) $1 rc=$?" >> "$LOG"
}

# --- S2 throughput pairs (no ramp needed: s4 saturates) ---
for i in 1 2 3; do
  if [ $((i % 2)) -eq 1 ]; then first=off; second=on; else first=on; second=off; fi
  stop_svc; start_svc "c4-$first-$i" '{"--max-num-seqs":"4"}' "{\"CLEF_SCHEMA_TOKEN_CACHE\":\"$([ $first = on ] && echo 1 || echo 0)\"}"
  bench "c4-$first-$i" 8 ""
  stop_svc; start_svc "c4-$second-$i" '{"--max-num-seqs":"4"}' "{\"CLEF_SCHEMA_TOKEN_CACHE\":\"$([ $second = on ] && echo 1 || echo 0)\"}"
  bench "c4-$second-$i" 8 ""
done

# --- fields-8 single-concurrency pairs (with ramp) ---
for i in 1 2 3; do
  if [ $((i % 2)) -eq 1 ]; then first=off; second=on; else first=on; second=off; fi
  stop_svc; start_svc "c8f-$first-$i" '{}' "{\"CLEF_SCHEMA_TOKEN_CACHE\":\"$([ $first = on ] && echo 1 || echo 0)\"}"
  bench "c8f-$first-$i-ramp" 1 "$FIELDS8"
  bench "c8f-$first-$i" 1 "$FIELDS8"
  stop_svc; start_svc "c8f-$second-$i" '{}' "{\"CLEF_SCHEMA_TOKEN_CACHE\":\"$([ $second = on ] && echo 1 || echo 0)\"}"
  bench "c8f-$second-$i-ramp" 1 "$FIELDS8"
  bench "c8f-$second-$i" 1 "$FIELDS8"
done

stop_svc
python3 - <<'EOF'
import os, shutil
p = "/workspace/yangchen/logs/tuning/20261113-120651-perf3/restored-tp2-4096"
if os.path.exists(p):
    shutil.move(p, p + "-old")
EOF
python scripts/manage_systemone_profile_service.py --run-dir "$RUN" restore >> "$LOG" 2>&1
for i in $(seq 1 60); do C=$(curl -s -m 3 -o /dev/null -w "%{http_code}" http://127.0.0.1:8000/health || true); [ "$C" = "200" ] && break; sleep 10; done
echo "restored ($C)" | tee -a "$LOG"
