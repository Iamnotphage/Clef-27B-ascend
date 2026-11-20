#!/bin/bash
# H11-sprofile-final preflight. Fail-closed; prints RESULT and exits non-zero on any failure.
set -u
CELL=${CELL:-/workspace/yangchen/work/grouped-deepdive-cell-h11-sprofile-final}
CODE=${CODE:-/workspace/yangchen/work/grouped-deepdive-h11-sprofile-final-code}
STAGE=${STAGE:-/workspace/yangchen/work/grouped-deepdive-35-e38b541/runtime/grouped_mmar}
SRC_ID=${SRC_ID:-$CELL/root-identity-h11-sprofile-final.json}
HEALTH_URL=${HEALTH_URL:-http://127.0.0.1:18143/health}
INFO_URL=${INFO_URL:-http://127.0.0.1:18143/v1/decide/info}
PORT=${PORT:-29531}
RUN=$CELL/run-h11-sprofile-final
TEMP=$CELL/probe-temp-h11-sprofile-final
OUT=$CELL/probe-out-h11-sprofile-final
MSOUT=$CELL/msprof-out-h11-sprofile-final
LOGDIR=$CELL/logs-h11-sprofile-final
RESTORE=$CELL/restore-h11-sprofile-final
ARCH=$CELL/archive-h11-sprofile-final
FAIL=0
ok(){ echo "OK: $*"; }
bad(){ echo "FAIL: $*" >&2; FAIL=1; }
check_sha(){ local f=$1 want=$2 label=$3; [ -f "$f" ] || { bad "$label missing ($f)"; return; }; got=$(sha256sum "$f" | awk '{print $1}'); [ "$got" = "$want" ] && ok "$label sha" || bad "$label sha mismatch ($f)"; }
# 1) fresh identity: file must match live /proc on pid/starttime/boot
if [ -s "$SRC_ID" ]; then
  SRC_ID="$SRC_ID" python3 -I - <<'PY' && ok "identity live match" || bad "identity live mismatch"
import json, os, sys
s = json.load(open(os.environ["SRC_ID"])); pid = int(s["pid"])
st = open(f"/proc/{pid}/stat").read(); r = st[st.rfind(")")+2:].split()
live = {"pid": pid, "ppid": int(r[1]), "pgid": int(r[2]), "sid": int(r[3]),
        "starttime": r[19], "state": r[0],
        "boot_id": open("/proc/sys/kernel/random/boot_id").read().strip()}
for k in ("pid", "ppid", "pgid", "sid", "starttime", "boot_id"):
    if str(s.get(k)) != str(live[k]): sys.exit(2)
if live["state"] == "Z": sys.exit(3)
PY
else bad "identity missing/empty ($SRC_ID)"; fi
# 2) health/info
curl -sS --max-time 5 "$HEALTH_URL" | python3 -I -c 'import json,sys;d=json.load(sys.stdin);sys.exit(0 if d.get("status")=="ready" and d.get("model")=="clef" else 1)' && ok "health" || bad "health"
curl -sS --max-time 5 "$INFO_URL" | python3 -I -c 'import json,sys;d=json.load(sys.stdin);sys.exit(0 if d.get("config")=="s1" else 1)' && ok "info s1" || bad "info s1"
# 3) caller/port/fresh dirs
command -v torchrun >/dev/null 2>&1 && ok "caller torchrun present" || bad "caller torchrun missing"
python3 -I - <<PY && ok "port $PORT free" || bad "port $PORT busy"
import socket,sys
for proto in (socket.SOCK_STREAM, socket.SOCK_DGRAM):
    for host in ("127.0.0.1","0.0.0.0"):
        s=socket.socket(socket.AF_INET,proto)
        try: s.bind((host,int("$PORT")))
        except OSError: sys.exit(1)
        finally: s.close()
PY
for d in "$RUN" "$TEMP" "$OUT" "$MSOUT" "$LOGDIR" "$RESTORE" "$ARCH"; do
  [ -e "$d" ] && bad "must be absent before window: $d" || ok "absent: $d"
done
# 4) frozen/native/source pins
check_sha "$CODE/supervised_probe_window.py" 3178f76375e80c1f0c23525f7d0742766c1c7a531ae1550b75b89f03edd78e28 supervisor
check_sha "$CODE/collect-raw-final.sh" 6077acb4a2e6bde4856c6855a7c3d3ba6e9f07a20b17f012f5957adbca1972e3 collect
check_sha "$CODE/build_window_argv.py" f98cfe183a0ff7c8436ecaca9b1faf21e32fe3d0ac80ea0821260c2943b83b0e builder
check_sha "$CODE/child-argv.json" d3db38a8aed434667a1b78f264dbeb7bf6a1ad4bb5fc1a0acd1a1d2ee040d7aa child
check_sha "$CODE/plan-h11-sprofile-final.json" 5ee2ab5d9bbb42193cb67c2445242d21992bcb45cdeda9e9af55c8278f213a1d plan
check_sha "$STAGE/run_probe.py" 8e3088042ad771c4d9a1b51d94ad1f49ec918c078d0b80ee25e8b3ca99cddf2f run_probe
check_sha "$STAGE/binding_probe.cpp" 101660f812306177f6d09c77644492f8f1b4a7f08ec6f22fc54e3caf7986ce3f binding
check_sha "$STAGE/probe_adapter.so" d181b15ab75e25982e527303de616c76a29e8f162038eb9f7dfe2a0c364110c3 adapter
check_sha /workspace/yangchen/work/src/vllm/vllm/model_executor/models/clef_joint_schema.py aa3e17585f8bd60a4d05e342abc7e2773c79a0b2d49f0370d15e29bcc6b72503 source
check_sha /workspace/yangchen/work/grouped-deepdive-cell-a15-large/text-body.json a7dd3f64133cfc9ecdc0b7784867bc8f3f5529b5465a6f5fa8da254cc28b5b38 text-body
check_sha /workspace/yangchen/work/grouped-deepdive-cell-a15-large/image-body.json f3229f5c120098ac4eebcc6f4025c28254e35f959fd01998cac5db7b8d209aab image-body
check_sha /tmp/d0cc/d0_stop.py d6d49d75012a0789d3bff3e7b522e9cd496844ca0ee0424608ab8e6b43a33886 stop
[ "$FAIL" -eq 0 ] && { echo "RESULT: ALL PASS"; exit 0; } || { echo "RESULT: FAIL"; exit 1; }
