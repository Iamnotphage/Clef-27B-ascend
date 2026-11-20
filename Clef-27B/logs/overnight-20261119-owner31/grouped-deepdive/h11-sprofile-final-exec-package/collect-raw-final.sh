#!/bin/bash
# H11-sprofile-final raw 采集副本（derived from h11-observe-execution-v2/collect-raw.sh 2bedc64a; single end）
# 用法： collect-raw.sh begin    # 执行窗口开始前
#        ... 人工授权后运行 window ...
#        collect-raw-final.sh end   # restore 完成后仅一次；重复 end 被拒绝
#
# 约定：raw 目录只放原始副本；派生视图写 meta/correlation.txt；SUMS 覆盖全部 raw。
#       不按 PID 过滤：默认 plog/device 源目录按时间窗整文件复制。
#       必需输入缺失会显式报错并置 FAIL（不再用 || true 静默）；来源映射写入 meta/sources.txt。
set -u

CELL=${CELL:-/workspace/yangchen/work/grouped-deepdive-cell-h11-sprofile-final}
ARCH=${ARCH:-$CELL/archive-h11-sprofile-final}
BODIES=${BODIES:-/workspace/yangchen/work/grouped-deepdive-cell-a15-large}
# 必需元数据实际来源（已在 CELL 核 SHA；可用环境变量覆盖为其它权威副本）
SRC_ID=${SRC_ID:-$CELL/root-identity-h11-sprofile-final.json}      # 84c9edb2…
SRC_SIDECAR=${SRC_SIDECAR:-$CELL/env-sidecar-h11-sprofile-final.json}  # 2474b487…
SRC_PLAN=${SRC_PLAN:-$CELL/plan-h11-sprofile-final.json}           # 9c2936c2…
SRC_DEV0=/root/ascend/log/run/device-0
SRC_DEV1=/root/ascend/log/run/device-1
SRC_PLOG=/root/ascend/log/run/plog
SRC_VLLM=/root/ascend/log/vllm_ascend
RUN=$CELL/run-h11-sprofile-final
TEMP=$CELL/probe-temp-h11-sprofile-final
OUT=$CELL/probe-out-h11-sprofile-final
RESTORE=$CELL/restore-h11-sprofile-final
LOGDIR=$CELL/logs-h11-sprofile-final

FAIL=0
need_file() {  # path label
  if [ -f "$1" ]; then return 0; fi
  echo "MISSING REQUIRED FILE: $2 ($1)" >&2
  FAIL=1
  return 1
}
need_dir() {
  if [ -d "$1" ]; then return 0; fi
  echo "MISSING REQUIRED DIR: $2 ($1)" >&2
  FAIL=1
  return 1
}
need_json() {  # path label
  if ! python3 -I -c 'import json,sys; json.load(open(sys.argv[1]))' "$1" >/dev/null 2>&1; then
    echo "INVALID REQUIRED JSON: $2 ($1)" >&2
    FAIL=1; return 1
  fi
  return 0
}
need_identity() {  # identity.json 非空、可解析、且与 SRC_ID 的 pid/starttime/boot 一致
  if [ ! -s "$ARCH/meta/identity.json" ]; then
    echo "MISSING REQUIRED FILE: identity.json (empty or absent)" >&2
    FAIL=1; return 1
  fi
  if ! ARCH="$ARCH" SRC_ID="$SRC_ID" python3 -I - <<'PY'
import json, os, sys
a = json.load(open(os.path.join(os.environ["ARCH"], "meta", "identity.json")))
b = json.load(open(os.environ["SRC_ID"]))
for k in ("pid", "starttime", "boot_id"):
    if str(a.get(k)) != str(b.get(k)):
        sys.exit(2)
PY
  then
    echo "MISMATCH REQUIRED: identity.json vs SRC_ID ($SRC_ID)" >&2
    FAIL=1; return 1
  fi
  return 0
}
record_src() { printf '%s -> %s\n' "$1" "$2" >> "$ARCH/meta/sources.txt"; }

phase=${1:-}

case "$phase" in
begin)
  [ -e "$ARCH" ] && { echo "archive must be absent before begin: $ARCH" >&2; exit 5; }
  mkdir -p "$ARCH"/{cann-plog,device,cell,meta}
  mkdir -p "$ARCH"/cann-plog/{plog,vllm_ascend} "$ARCH"/device/{device-0,device-1}
  mkdir -p "$ARCH"/cell/{run,probe-temp,probe-out,restore} "$ARCH"/fresh-logs
  : > "$ARCH/meta/sources.txt"
  date -u +%Y-%m-%dT%H:%M:%SZ > "$ARCH/meta/start-utc.txt"
  date +%s.%N > "$ARCH/meta/start-epoch.txt"
  if SRC_ID="$SRC_ID" python3 -I - > "$ARCH/meta/identity.tmp" <<'PY'
import json, os, sys
src = os.environ["SRC_ID"]
saved = json.load(open(src))
pid = int(saved["pid"])
try:
    st = open(f"/proc/{pid}/stat").read()
    r = st[st.rfind(")")+2:].split()
    live = {"pid": pid, "ppid": int(r[1]), "pgid": int(r[2]), "session": int(r[3]),
            "starttime": r[19], "state": r[0],
            "boot_id": open("/proc/sys/kernel/random/boot_id").read().strip(),
            "cmdline": open(f"/proc/{pid}/cmdline","rb").read().replace(b"\0",b" ").strip().decode()}
except OSError as exc:
    print(f"identity capture failed: {exc}", file=sys.stderr); sys.exit(3)
for k in ("pid", "starttime", "boot_id"):
    if str(saved.get(k)) != str(live.get(k)):
        print(f"identity mismatch on {k}: saved={saved.get(k)} live={live.get(k)}", file=sys.stderr)
        sys.exit(4)
json.dump(live, sys.stdout, indent=1)
PY
  then
    mv "$ARCH/meta/identity.tmp" "$ARCH/meta/identity.json"
  else
    echo "MISSING REQUIRED: identity capture failed (SRC_ID=$SRC_ID)" >&2
    rm -f "$ARCH/meta/identity.tmp"; FAIL=1
  fi
  need_identity || true
  curl -sS --max-time 5 http://127.0.0.1:18143/health > "$ARCH/meta/health.json" 2>&1 || true
  curl -sS --max-time 5 http://127.0.0.1:18143/v1/decide/info > "$ARCH/meta/info.json" 2>&1 || true
  python3 -I - > "$ARCH/meta/port-29531.txt" 2>&1 <<'PY'
import socket
for proto,name in ((socket.SOCK_STREAM,"tcp"),(socket.SOCK_DGRAM,"udp")):
    for host in ("127.0.0.1","0.0.0.0"):
        s=socket.socket(socket.AF_INET,proto)
        try:
            s.bind((host,29531)); print(name,host,"FREE")
        except OSError as e:
            print(name,host,"BUSY",e.errno,e.strerror)
        finally:
            s.close()
PY
  # 必需元数据：实际来源 -> meta（sidecar/plan/root-identity 在 CELL；body 在 A15-large cell）
  for pair in \
      "$SRC_ID" "$SRC_SIDECAR" "$SRC_PLAN" \
      "$BODIES/text-body.json" "$BODIES/image-body.json"; do
    name=$(basename "$pair")
    if need_file "$pair" "meta $name"; then
      cp -p "$pair" "$ARCH/meta/" && record_src "$pair" "meta/$name"
    fi
  done
  # 必需 metadata 的解析/取值校验（失败即 FAIL，begin 非零）
  need_json "$ARCH/meta/health.json" health || true
  python3 -I -c 'import json,sys;d=json.load(open(sys.argv[1]));sys.exit(0 if d.get("status")=="ready" and d.get("model")=="clef" else 1)' \
    "$ARCH/meta/health.json" >/dev/null 2>&1 \
    || { echo "INVALID REQUIRED META: health status/model" >&2; FAIL=1; }
  need_json "$ARCH/meta/info.json" info || true
  python3 -I -c 'import json,sys;d=json.load(open(sys.argv[1]));sys.exit(0 if d.get("config")=="s1" else 1)' \
    "$ARCH/meta/info.json" >/dev/null 2>&1 \
    || { echo "INVALID REQUIRED META: info config!=s1" >&2; FAIL=1; }
  python3 -I -c 'import sys;l=open(sys.argv[1]).read().splitlines();sys.exit(0 if len(l)==4 and all("FREE" in x for x in l) else 1)' \
    "$ARCH/meta/port-29531.txt" >/dev/null 2>&1 \
    || { echo "INVALID REQUIRED META: port-29531 not all FREE" >&2; FAIL=1; }
  need_json "$ARCH/meta/plan-h11-sprofile-final.json" plan || true
  python3 -I -c 'import json,sys;d=json.load(open(sys.argv[1]));sys.exit(0 if d.get("status")=="plan_not_executed" else 1)' \
    "$ARCH/meta/plan-h11-sprofile-final.json" >/dev/null 2>&1 \
    || { echo "INVALID REQUIRED META: plan status" >&2; FAIL=1; }
  need_json "$ARCH/meta/env-sidecar-h11-sprofile-final.json" env-sidecar || true
  need_json "$ARCH/meta/text-body.json" text-body || true
  need_json "$ARCH/meta/image-body.json" image-body || true

  touch "$ARCH/meta/START"
  start_epoch=$(cut -d. -f1 "$ARCH/meta/start-epoch.txt")
  touch -d "@$((start_epoch - 300))" "$ARCH/meta/START_SLACK" 2>/dev/null || touch "$ARCH/meta/START_SLACK"
  echo "archive=$ARCH"
  echo "markers: $ARCH/meta/START $ARCH/meta/START_SLACK"
  [ "$FAIL" -eq 0 ] || { echo "begin: REQUIRED INPUTS MISSING" >&2; exit 1; }
  ;;
end)
  [ -d "$ARCH" ] || { echo "missing archive: $ARCH (run begin first)" >&2; exit 2; }
  [ -e "$ARCH/meta/END" ] && { echo "end already recorded; single end only: $ARCH" >&2; exit 7; }
  date -u +%Y-%m-%dT%H:%M:%SZ > "$ARCH/meta/end-utc.txt"
  date +%s.%N > "$ARCH/meta/end-epoch.txt"
  touch "$ARCH/meta/END"
  end_epoch=$(cut -d. -f1 "$ARCH/meta/end-epoch.txt")
  touch -d "@$((end_epoch + 300))" "$ARCH/meta/END_SLACK" 2>/dev/null || touch "$ARCH/meta/END_SLACK"

  copy_window() {  # $1 src dir, $2 dst dir, $3 label
    src=$1; dst=$2; label=$3
    if need_dir "$src" "$label source"; then
      mkdir -p "$dst"
      find "$src" -maxdepth 1 -type f \
        -newer "$ARCH/meta/START_SLACK" ! -newer "$ARCH/meta/END_SLACK" \
        -exec cp -p {} "$dst"/ \; 2>/dev/null || true
      record_src "$src" "$dst"
    fi
  }
  copy_window "$SRC_DEV0" "$ARCH/device/device-0" "device-0"
  copy_window "$SRC_DEV1" "$ARCH/device/device-1" "device-1"
  copy_window "$SRC_PLOG" "$ARCH/cann-plog/plog" "run/plog"
  copy_window "$SRC_VLLM" "$ARCH/cann-plog/vllm_ascend" "vllm_ascend"

  # fresh LOGDIR 整目录（PROCESS_LOG_PATH 候选）
  if need_dir "$LOGDIR" "fresh logdir"; then
    cp -a "$LOGDIR/." "$ARCH/fresh-logs"/ 2>/dev/null || true
    record_src "$LOGDIR" "fresh-logs"
  fi
  # cell 产物（window/child/descendants/stop/restore + rank traces/results）
  if need_dir "$RUN" "run dir"; then
    cp -a "$RUN/." "$ARCH/cell/run"/ 2>/dev/null || true
    record_src "$RUN" "cell/run"
  fi
  if need_dir "$TEMP" "probe temp"; then
    cp -a "$TEMP/." "$ARCH/cell/probe-temp"/ 2>/dev/null || true
    record_src "$TEMP" "cell/probe-temp"
  fi
  if need_dir "$OUT" "probe out"; then
    cp -a "$OUT/." "$ARCH/cell/probe-out"/ 2>/dev/null || true
    record_src "$OUT" "cell/probe-out"
  fi
  if need_dir "$RESTORE" "restore output"; then
    cp -a "$RESTORE/." "$ARCH/cell/restore"/ 2>/dev/null || true
    record_src "$RESTORE" "cell/restore"
  fi
  # 可选：执行包装日志
  for f in "$CELL"/exec-*.log; do
    [ -f "$f" ] && cp -p "$f" "$ARCH/cell/" && record_src "$f" "cell/$(basename "$f")"
  done
  # plan 与 body：以实际来源为准（CELL plan；A15-large body）
  if need_file "$SRC_PLAN" "plan"; then
    cp -p "$SRC_PLAN" "$ARCH/cell/" && record_src "$SRC_PLAN" "cell/$(basename "$SRC_PLAN")"
  fi
  for f in "$BODIES/text-body.json" "$BODIES/image-body.json"; do
    if need_file "$f" "body $(basename "$f")"; then
      cp -p "$f" "$ARCH/cell/" && record_src "$f" "cell/$(basename "$f")"
    fi
  done

  need_identity || true

  # 派生视图（单独文件，不改 raw）
  find "$ARCH" -type f ! -name SHA256SUMS ! -name correlation.txt \
    -printf '%TY-%Tm-%TdT%TH:%TM:%TS %10s %p\n' 2>/dev/null | sort > "$ARCH/meta/correlation.txt"

  # SUMS（覆盖全部 raw + 派生视图）
  ( cd "$ARCH" && find . -type f ! -name SHA256SUMS -print0 \
      | sort -z | xargs -0 sha256sum > SHA256SUMS )
  echo "SUMS=$ARCH/SHA256SUMS"
  echo "raw files: $(find "$ARCH" -type f ! -name SHA256SUMS | wc -l)"
  if [ "$FAIL" -ne 0 ]; then
    touch "$ARCH/meta/INCOMPLETE"
    echo "end: REQUIRED ARTIFACTS MISSING (see above); INCOMPLETE marker set" >&2
    exit 1
  fi
  ;;
*)
  echo "usage: $0 begin|end" >&2
  exit 2
  ;;
esac
