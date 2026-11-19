#!/bin/bash
# H11-observe-v1 raw 采集模板（doc-only；当前无 GO，未执行）
# 用法： collect-raw.sh begin    # 执行窗口开始前
#        ... 人工授权后运行 window ...
#        collect-raw.sh end      # restore 完成后；可重复 end 以补收未 flush 的文件
#
# 约定：raw 目录只放原始副本；派生视图写 meta/correlation.txt；SUMS 覆盖全部 raw。
#       不按 PID 过滤：默认 plog/device 源目录按时间窗整文件复制。
set -u

CELL=${CELL:-/workspace/yangchen/work/grouped-deepdive-cell-h11-observe}
ARCH=${ARCH:-$CELL/archive-h11-observe-v1}
SRC_DEV0=/root/ascend/log/run/device-0
SRC_DEV1=/root/ascend/log/run/device-1
SRC_PLOG=/root/ascend/log/run/plog
SRC_VLLM=/root/ascend/log/vllm_ascend
RUN=$CELL/run-h11-observe-v1
TEMP=$CELL/probe-temp-h11-observe-v1
OUT=$CELL/probe-out-h11-observe-v1
RESTORE=$CELL/restore-h11-observe-v1
LOGDIR=$CELL/logs-h11-observe-v1

phase=${1:-}

case "$phase" in
begin)
  mkdir -p "$ARCH"/{cann-plog,device,cell,meta}
  mkdir -p "$ARCH"/cann-plog/{plog,vllm_ascend} "$ARCH"/device/{device-0,device-1}
  mkdir -p "$ARCH"/cell/{run,probe-temp,probe-out,restore} "$ARCH"/fresh-logs
  date -u +%Y-%m-%dT%H:%M:%SZ > "$ARCH/meta/start-utc.txt"
  date +%s.%N > "$ARCH/meta/start-epoch.txt"
  python3 -I - > "$ARCH/meta/identity.json" <<'PY'
import json
pid=1337814
st=open(f"/proc/{pid}/stat").read(); r=st[st.rfind(")")+2:].split()
print(json.dumps({
 "pid": pid, "ppid": int(r[1]), "pgid": int(r[2]), "session": int(r[3]),
 "starttime": r[19], "state": r[0],
 "boot_id": open("/proc/sys/kernel/random/boot_id").read().strip(),
 "cmdline": open(f"/proc/{pid}/cmdline","rb").read().replace(b"\0",b" ").strip().decode(),
}, indent=1))
PY
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
  cp -p "$CELL/root-identity-h11-observe-v1.json" "$ARCH/meta/" 2>/dev/null || true
  cp -p "$CELL/env-sidecar-h11-observe-v1.json" "$ARCH/meta/" 2>/dev/null || true
  touch "$ARCH/meta/START"
  start_epoch=$(cut -d. -f1 "$ARCH/meta/start-epoch.txt")
  touch -d "@$((start_epoch - 300))" "$ARCH/meta/START_SLACK" 2>/dev/null || touch "$ARCH/meta/START_SLACK"
  echo "archive=$ARCH"
  echo "markers: $ARCH/meta/START $ARCH/meta/START_SLACK"
  ;;
end)
  [ -d "$ARCH" ] || { echo "missing archive: $ARCH (run begin first)" >&2; exit 2; }
  date -u +%Y-%m-%dT%H:%M:%SZ > "$ARCH/meta/end-utc.txt"
  date +%s.%N > "$ARCH/meta/end-epoch.txt"
  touch "$ARCH/meta/END"
  end_epoch=$(cut -d. -f1 "$ARCH/meta/end-epoch.txt")
  touch -d "@$((end_epoch + 300))" "$ARCH/meta/END_SLACK" 2>/dev/null || touch "$ARCH/meta/END_SLACK"

  copy_window() {  # $1 src dir, $2 dst dir
    src=$1; dst=$2
    mkdir -p "$dst"
    find "$src" -maxdepth 1 -type f \
      -newer "$ARCH/meta/START_SLACK" ! -newer "$ARCH/meta/END_SLACK" \
      -exec cp -p {} "$dst"/ \; 2>/dev/null || true
  }
  copy_window "$SRC_DEV0" "$ARCH/device/device-0"
  copy_window "$SRC_DEV1" "$ARCH/device/device-1"
  copy_window "$SRC_PLOG" "$ARCH/cann-plog/plog"
  copy_window "$SRC_VLLM" "$ARCH/cann-plog/vllm_ascend"

  # fresh LOGDIR 整目录（PROCESS_LOG_PATH 候选）
  [ -d "$LOGDIR" ] && cp -a "$LOGDIR/." "$ARCH/fresh-logs"/ 2>/dev/null || true
  # cell 产物（window/child/descendants/stop/restore + rank traces/results）
  [ -d "$RUN" ] && cp -a "$RUN/." "$ARCH/cell/run"/ 2>/dev/null || true
  [ -d "$TEMP" ] && cp -a "$TEMP/." "$ARCH/cell/probe-temp"/ 2>/dev/null || true
  [ -d "$OUT" ] && cp -a "$OUT/." "$ARCH/cell/probe-out"/ 2>/dev/null || true
  [ -d "$RESTORE" ] && cp -a "$RESTORE/." "$ARCH/cell/restore"/ 2>/dev/null || true
  cp -p "$CELL"/exec-*.log "$ARCH/cell/" 2>/dev/null || true
  cp -p "$CELL/plan-h11-observe-v1.json" "$ARCH/cell/" 2>/dev/null || true
  cp -p "$CELL/text-body.json" "$CELL/image-body.json" "$ARCH/cell/" 2>/dev/null || true

  # 派生视图（单独文件，不改 raw）
  find "$ARCH" -type f ! -name SHA256SUMS ! -name correlation.txt \
    -printf '%TY-%Tm-%TdT%TH:%TM:%TS %10s %p\n' 2>/dev/null | sort > "$ARCH/meta/correlation.txt"

  # SUMS（覆盖全部 raw + 派生视图）
  ( cd "$ARCH" && find . -type f ! -name SHA256SUMS -print0 \
      | sort -z | xargs -0 sha256sum > SHA256SUMS )
  echo "SUMS=$ARCH/SHA256SUMS"
  echo "raw files: $(find "$ARCH" -type f ! -name SHA256SUMS | wc -l)"
  ;;
*)
  echo "usage: $0 begin|end" >&2
  exit 2
  ;;
esac
