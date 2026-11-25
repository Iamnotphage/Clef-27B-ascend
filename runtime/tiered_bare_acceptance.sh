#!/usr/bin/env bash
# Official bare start: one service process, one warmup pass, two measured
# passes for OD S1 / S2 C16 / S2 C64 and ERQA S1 / S2 C16.
#
# The service is started exactly as the submission does (`bash start.sh
# --config s1`, no restarts) with Tier 2 left at its config default and trace
# enabled for hit/populate evidence.  Every report keeps the raw responses so
# quality can be re-scored offline.
set -euo pipefail

REPO=${REPO:?}
MODEL_DIR=${MODEL_DIR:?}
OUT=${OUT:?}
OD=${OD:?}
ERQA=${ERQA:?}
PORT=${PORT:-18143}
BASE=http://127.0.0.1:$PORT

cd "$REPO"
export CLEF_VENV=${CLEF_VENV:-/workspace/yangchen/venvs/clef-mix-ws-py312}
source scripts/activate_env.sh >/dev/null 2>&1 || true
mkdir -p "$OUT"

CLEF_TIER2_TRACE=1 bash Clef-27B/start.sh --model-dir "$MODEL_DIR" \
  --output-dir "$OUT/service" --config s1 > "$OUT/start.log" 2>&1 &
SERVICE=$!
cleanup() {
  kill "$SERVICE" 2>/dev/null || true
  wait "$SERVICE" 2>/dev/null || true
}
trap cleanup EXIT

for _ in $(seq 1 300); do
  if curl -sf "$BASE/health" >/dev/null; then break; fi
  sleep 2
done
curl -sf "$BASE/health"
curl -s "$BASE/v1/decide/info" > "$OUT/info.json"
python - "$OUT/info.json" <<'PY'
import json, sys
info = json.load(open(sys.argv[1]))
assert info["cache"]["prefix_state_cache"] is True, info
print("prefix_state_cache disclosed: true")
PY

od_run() { # round concurrency
  python runtime/od_full_benchmark.py --sample-file "$OD" \
    --concurrency "$2" --report "$OUT/od-$1.json" > "$OUT/od-$1.log" 2>&1
  python - "$OUT/od-$1.json" <<'PY'
import json, sys
s = json.load(open(sys.argv[1]))["summary"]
assert s["failed"] == 0, s
print(json.dumps({"run": sys.argv[1].split("/")[-1],
                  "dec/s": round(s["decisions_per_second"], 3),
                  "p50": round(s["latency_ms"]["p50"], 1),
                  "p95": round(s["latency_ms"]["p95"], 1)}))
PY
}

erqa_run() { # round tag [concurrency]
  if [[ -n "${2:-}" ]]; then
    PYTHONPATH="$REPO/scripts/official_eval" python runtime/erqa_s2_concurrent.py \
      --data-dir "$ERQA" --output-dir "$OUT/erqa-$1" \
      --api-url "$BASE/v1" --wire-format state-list --endpoint /systemone \
      --concurrency "$2" > "$OUT/erqa-$1.log" 2>&1
  else
    python scripts/official_eval/erqa_baseline.py \
      --data-dir "$ERQA" --output-dir "$OUT/erqa-$1" \
      --api-url "$BASE/v1" --wire-format state-list --endpoint /systemone \
      --scope full > "$OUT/erqa-$1.log" 2>&1
  fi
  echo "erqa-$1 done"
}

echo "== warmup OD"
od_run warmup 1
echo "== measured OD"
od_run s1-r1 1
od_run s1-r2 1
od_run c16-r1 16
od_run c16-r2 16
od_run c64-r1 64
od_run c64-r2 64
# ERQA is warmed right before its own rounds: the 32 GiB store cannot hold
# both datasets (OD ~28 GB + ERQA ~25 GB), and aged_keep_longest keeps the
# longer entries, so a single shared warmup would turn one dataset's measured
# rounds into pure bypass.  Dataset-major order keeps each suite warm for its
# own two measured passes.
echo "== warmup ERQA"
erqa_run warmup
echo "== measured ERQA"
erqa_run s1-r1
erqa_run s1-r2
erqa_run c16-r1 16
erqa_run c16-r2 16

python - "$OUT/service/service.log" "$OUT/counters.json" <<'PY'
import json, re, sys
text = open(sys.argv[1], errors="replace").read()
fields = ("entries", "bytes", "populates", "publications", "evictions",
          "rejections", "hits", "misses")
last = {}
for line in text.splitlines():
    if "batch admit" not in line and "decision=" not in line:
        continue
    for name in fields:
        match = re.search(rf"\b{name}=(\d+)", line)
        if match:
            last[name] = int(match.group(1))
assert last, "no Tier 2 counters in the service log"
json.dump(last, open(sys.argv[2], "w"), indent=2)
print(json.dumps(last))
PY

echo BARE_ACCEPTANCE_DONE
