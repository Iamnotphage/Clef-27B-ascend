#!/bin/bash
# S1 GDN A/B supplementary pairs 4-6, REVERSED order (ON -> OFF) per user
# instruction. Same warmup/sequence/window as pairs 1-3. Enhanced capture:
# npu-smi around formal windows, service PID/age, cpu.stat per window.
set -u
TOOLS=/workspace/yangchen/work/profiling-tools-20261113
RUN=$(cat /workspace/yangchen/logs/tuning/CURRENT_RUN)
AB="$RUN/ab-gdn-s1-ramped2"
mkdir -p "$AB"
cd "$TOOLS"
source scripts/activate_env.sh >/dev/null 2>&1
LOG="$AB/driver.log"

exec 9>"$RUN/.s1_ab_ramped.lock"
flock -n 9 || { echo "another instance holds the lock; aborting" | tee -a "$LOG"; exit 1; }
echo "$(TZ=Asia/Shanghai date +%T) lock acquired (pairs 4-6, ON->OFF)" >> "$LOG"

stop_svc() { python scripts/manage_systemone_profile_service.py --run-dir "$RUN" stop >> "$LOG" 2>&1; sleep 3; }
start_svc() { # $1 label $2 env
  python3 /workspace/yangchen/logs/tuning/start_service_custom.py --run-dir "$RUN" \
    --label "$1" --overrides "{\"env\":$2,\"wait_health_seconds\":600}" >> "$LOG" 2>&1
  for _w in $(seq 1 60); do C=$(curl -s -m 3 -o /dev/null -w "%{http_code}" http://127.0.0.1:8000/health || true); [ "$C" = "200" ] && break; sleep 10; done
  [ "$C" = "200" ] || { echo "HEALTH FAIL $1" | tee -a "$LOG"; exit 1; }
  # process age evidence: pid + start epoch recorded at launch
  python3 - "$1" "$AB" <<'EOF'
import json, subprocess, sys, time
label, ab = sys.argv[1], sys.argv[2]
cs = json.load(open(f"{ab.rsplit('/',1)[0]}/current-service.json"))
pid = cs["pid"]
age = {"pid": pid, "started_epoch": time.time(), "label": label,
       "proc_start_ticks": open(f"/proc/{pid}/stat").read().split()[21]}
json.dump(age, open(f"{ab}/{label}.age.json", "w"))
EOF
}

npu_snap() { # $1 tag
  npu-smi info 2>/dev/null | grep -E "Temp|Power|AICore|HBM" > "$AB/$1.npu.txt" || true
}

window() { # $1 tag
  local T="$1" P="$AB/$1"
  cp /sys/fs/cgroup/cpu/cpu.stat "$P.cpu-before.txt"
  npu_snap "$T-before"
  python scripts/benchmark_systemone.py --warmup 10 --requests 300 --concurrency 1 \
    --min-duration-seconds 30 --sample-hbm --base-url http://127.0.0.1:8000 \
    --report "$P.json" > "$P.stdout" 2>&1
  local RC=$?
  cp /sys/fs/cgroup/cpu/cpu.stat "$P.cpu-after.txt"
  npu_snap "$T-after"
  python3 - "$P.json" "$LOG" "$T" "$RC" <<'EOF'
import json, sys
p, log, tag, rc = sys.argv[1:5]
try:
    d = json.load(open(p))
    s = d["summary"]; l = s["successful_request_latency_ms"]
    line = f"  {tag}: rc={rc} rps={s['successful_requests_per_second']:.4f} mean={l['mean']:.2f} p50={l['p50']:.1f} p95={l['p95']:.1f} p99={l['p99']:.1f}"
except Exception as e:
    line = f"  {tag}: rc={rc} REPORT ERROR {e}"
print(line)
with open(log, "a") as f:
    f.write(line + "\n")
EOF
  echo "$(TZ=Asia/Shanghai date +%T) $T done" >> "$LOG"
}

for i in 4 5 6; do
  stop_svc; start_svc "s1x-on-$i" '{"CLEF_GDN_BSND":"1"}'
  window "s1x-on-$i-ramp"
  window "s1x-on-$i"
  stop_svc; start_svc "s1x-off-$i" '{"CLEF_GDN_BSND":"0"}'
  window "s1x-off-$i-ramp"
  window "s1x-off-$i"
done

# restore (clear prior restore dir)
stop_svc
python3 - <<'EOF'
import os, shutil
p = "/workspace/yangchen/logs/tuning/20261113-120651-perf3/restored-tp2-4096"
if os.path.exists(p):
    shutil.move(p, p + "-old")
EOF
python scripts/manage_systemone_profile_service.py --run-dir "$RUN" restore >> "$LOG" 2>&1
for _w in $(seq 1 60); do C=$(curl -s -m 3 -o /dev/null -w "%{http_code}" http://127.0.0.1:8000/health || true); [ "$C" = "200" ] && break; sleep 10; done
echo "restored ($C)" | tee -a "$LOG"
