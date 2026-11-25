#!/usr/bin/env bash
# Scenario D: 4 GiB/rank store, OD C16 mixed load, eviction stress.
#
# Starts the service through the official launcher with the Tier 2 store
# capped at 4 GiB and trace enabled, runs two full C16 passes, then requires
# zero failures, evictions > 0 and a 2PP quality result from the standard
# od_raw_acceptance scorer.
set -euo pipefail

REPO=${REPO:?}
OUT=${OUT:?}
OD=${OD:?}
OD_DATA=${OD_DATA:-}
OD_SCORER=${OD_SCORER:-}
OD_TEMPLATE=${OD_TEMPLATE:-}
RANK_BYTES=${RANK_BYTES:-4294967296}
PORT=${PORT:-18143}
BASE=http://127.0.0.1:$PORT

cd "$REPO"
source scripts/activate_env.sh >/dev/null 2>&1 || true
mkdir -p "$OUT"

CLEF_PREFIX_CACHE_RANK_BYTES=$RANK_BYTES \
CLEF_TIER2_TRACE=1 \
python Clef-27B/inference.py --config s1 --prefix-state-cache on \
  --output-dir "$OUT/service" > "$OUT/service-launch.log" 2>&1 &
LAUNCHER=$!
cleanup() {
  # inference.py execv()s, so the launcher PID is the serving process.
  kill "$LAUNCHER" 2>/dev/null || true
  wait "$LAUNCHER" 2>/dev/null || true
  pkill -P "$LAUNCHER" 2>/dev/null || true
}
trap cleanup EXIT

for _ in $(seq 1 300); do
  if curl -sf "$BASE/health" >/dev/null; then break; fi
  sleep 2
done
curl -sf "$BASE/health"
curl -s "$BASE/v1/decide/info" > "$OUT/info.json"
python - "$OUT/info.json" "$RANK_BYTES" <<'PY'
import json, sys
info = json.load(open(sys.argv[1]))
cache = info.get("cache", {})
assert cache.get("prefix_state_cache") is True, info
print(json.dumps({"prefix_state_cache": cache["prefix_state_cache"],
                  "rank_bytes": int(sys.argv[2])}))
PY

for round in r1 r2; do
  python runtime/od_full_benchmark.py --sample-file "$OD" --concurrency 16 \
    --report "$OUT/od-c16-$round.json" > "$OUT/od-c16-$round.log" 2>&1
  python - "$OUT/od-c16-$round.json" "$round" <<'PY'
import json, sys
report = json.load(open(sys.argv[1]))
summary = report["summary"]
assert summary["failed"] == 0, summary
print(json.dumps({"round": sys.argv[2],
                  "decisions_per_second": summary["decisions_per_second"],
                  "p95": summary["latency_ms"]["p95"]}))
PY
done

python - "$OUT/service/service.log" "$OUT/counters.json" <<'PY'
import json, re, sys
text = open(sys.argv[1], errors="replace").read()
fields = ("entries", "bytes", "populates", "publications", "evictions",
          "rejections", "hits", "misses")
last = {}
for line in text.splitlines():
    if "decision=" not in line:
        continue
    for name in fields:
        match = re.search(rf"\b{name}=(\d+)", line)
        if match:
            last[name] = int(match.group(1))
assert last, "no Tier 2 trace lines in the service log"
assert last.get("evictions", 0) > 0, last
json.dump(last, open(sys.argv[2], "w"), indent=2)
print(json.dumps(last))
PY

if [[ -n "$OD_DATA" && -n "$OD_SCORER" && -n "$OD_TEMPLATE" ]]; then
  for round in r1 r2; do
    python runtime/od_raw_acceptance.py --data "$OD_DATA" --scorer "$OD_SCORER" \
      --template "$OD_TEMPLATE" --run "candidate=$OUT/od-c16-$round.json" \
      --reference candidate --output "$OUT/od-c16-$round-acceptance.json"
  done
fi

echo SCENARIO_D_DONE
