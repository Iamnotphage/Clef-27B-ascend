#!/bin/bash
# H11-observe-v1 执行前完整重扫（doc-only；只读，不 stop、不执行 child）
# 用法： preflight-rescan.sh
# 退出码：0 全部通过；1 有任一不符（此时不得执行）。
set -u
CELL=/workspace/yangchen/work/grouped-deepdive-cell-h11-observe
STAGE=/workspace/yangchen/work/grouped-deepdive-35-e38b541/runtime/grouped_mmar
FAIL=0
say() { printf '%-6s %s\n' "$1" "$2"; }
ok()  { say PASS "$1"; }
bad() { say FAIL "$1"; FAIL=1; }

# 1) identity / boot / ppid / pgid / sid / starttime
python3 -I - <<'PY' > /tmp/h11-rescan-identity.txt 2>&1
import json
pid=1337814
st=open(f"/proc/{pid}/stat").read(); r=st[st.rfind(")")+2:].split()
print(json.dumps({"pid":pid,"ppid":int(r[1]),"pgid":int(r[2]),"sid":int(r[3]),
 "starttime":r[19],"state":r[0],
 "boot_id":open("/proc/sys/kernel/random/boot_id").read().strip()},indent=1))
PY
grep -q '"pid": 1337814' /tmp/h11-rescan-identity.txt && ok "identity pid=1337814" || bad "identity pid"
grep -q '"starttime": "400405818"' /tmp/h11-rescan-identity.txt && ok "starttime 400405818" || bad "starttime"
grep -q '"boot_id": "393ecc15-721f-46f7-b119-21fa327a2b3b"' /tmp/h11-rescan-identity.txt && ok "boot_id" || bad "boot_id"
grep -q '"ppid": 1' /tmp/h11-rescan-identity.txt && ok "ppid=1" || bad "ppid"
grep -q '"pgid": 1337814' /tmp/h11-rescan-identity.txt && ok "pgid=1337814" || bad "pgid"
grep -q '"sid": 1337814' /tmp/h11-rescan-identity.txt && ok "sid=1337814" || bad "sid"

# 2) 五 PID（root + 4 descendants）与名字
python3 -I - <<'PY' > /tmp/h11-rescan-pids.txt 2>&1
import os
root=1337814
rows={}
for n in os.listdir("/proc"):
    if not n.isdigit(): continue
    pid=int(n)
    try:
        st=open(f"/proc/{pid}/stat").read(); r=st[st.rfind(")")+2:].split()
        cmd=open(f"/proc/{pid}/cmdline","rb").read().replace(b"\0",b" ").strip().decode("utf-8","replace")
    except Exception:
        continue
    rows[pid]=(int(r[1]),int(r[2]),int(r[3]),r[19],cmd)
kids={}
for p,v in rows.items(): kids.setdefault(v[0],[]).append(p)
sel=[]; stack=[root]
while stack:
    for c in sorted(kids.get(stack.pop(),[])): sel.append(c); stack.append(c)
print("root",root)
for p in [root]+sel:
    ppid,pgid,sid,start,cmd=rows[p]
    print(p,ppid,pgid,sid,start,cmd[:100])
PY
n=$(($(wc -l < /tmp/h11-rescan-pids.txt) - 1))
[ "$n" -eq 5 ] && ok "five PIDs ($n)" || bad "five PIDs (got $n)"
pidset=$(awk 'NR>1{print $1}' /tmp/h11-rescan-pids.txt | sort -n | tr '\n' ' ')
[ "$pidset" = "1337814 1337961 1337962 1337979 1337980 " ] \
  && ok "PID set exact 1337814/1337961/1337962/1337979/1337980" \
  || bad "PID set mismatch ($pidset)"
grep -q 'resource_tracker' /tmp/h11-rescan-pids.txt && ok "resource_tracker present" || bad "resource_tracker"
grep -q 'EngineCore' /tmp/h11-rescan-pids.txt && ok "EngineCore present" || bad "EngineCore"
grep -q 'Worker_TP0' /tmp/h11-rescan-pids.txt && ok "Worker_TP0 present" || bad "Worker_TP0"
grep -q 'Worker_TP1' /tmp/h11-rescan-pids.txt && ok "Worker_TP1 present" || bad "Worker_TP1"

# 3) health / info
h=$(curl -sS --max-time 5 http://127.0.0.1:18143/health 2>/dev/null || true)
i=$(curl -sS --max-time 5 http://127.0.0.1:18143/v1/decide/info 2>/dev/null || true)
case "$h" in *'"status":"ready"'*'"model":"clef"'*) ok "health ready/clef";; *) bad "health";; esac
case "$i" in *'"config":"s1"'*) ok "info config=s1";; *) bad "info config";; esac
case "$i" in *'"tensor_parallel_size":2'*) ok "info TP2";; *) bad "info TP";; esac

# 4) clients：/proc/net/tcp{,6} 中 18143（0x46DF）连接
python3 -I - <<'PY' > /tmp/h11-rescan-clients.txt 2>&1
port=hex(18143)[2:].upper().rjust(4,"0")
listen=estab=0
for path in ("/proc/net/tcp","/proc/net/tcp6"):
    try:
        lines=open(path).read().splitlines()[1:]
    except OSError:
        continue
    for ln in lines:
        f=ln.split()
        if len(f)<4: continue
        lp=f[1].split(":")[-1].upper(); st=f[3].upper()
        if lp==port:
            if st=="0A": listen+=1
            elif st=="01": estab+=1
print("port",18143,"listen",listen,"established",estab)
PY
cat /tmp/h11-rescan-clients.txt
grep -q 'listen 1' /tmp/h11-rescan-clients.txt && ok "service listens 18143" || bad "service listen"
grep -q 'established 0' /tmp/h11-rescan-clients.txt && ok "no established clients" || bad "pending clients"

# 5) 端口 29531 FREE（tcp/udp × 127.0.0.1/0.0.0.0）
python3 -I - <<'PY' > /tmp/h11-rescan-port.txt 2>&1
import socket
bad=0
for proto in (socket.SOCK_STREAM, socket.SOCK_DGRAM):
    for host in ("127.0.0.1","0.0.0.0"):
        s=socket.socket(socket.AF_INET,proto)
        try:
            s.bind((host,29531)); print("FREE", "tcp" if proto==socket.SOCK_STREAM else "udp", host)
        except OSError as e:
            bad+=1; print("BUSY", "tcp" if proto==socket.SOCK_STREAM else "udp", host, e.errno, e.strerror)
        finally:
            s.close()
raise SystemExit(1 if bad else 0)
PY
[ $? -eq 0 ] && ok "port 29531 free (tcp/udp x 127/0.0.0.0)" || bad "port 29531"

# 6) fresh 四路径 absent + logs 可写
for p in "$CELL/run-h11-observe-v1" "$CELL/probe-temp-h11-observe-v1" \
         "$CELL/probe-out-h11-observe-v1" "$CELL/restore-h11-observe-v1"; do
  [ -e "$p" ] && bad "must be absent: $p" || ok "absent $p"
done
[ -w "$CELL/logs-h11-observe-v1" ] && ok "logdir writable" || bad "logdir not writable"

# 7) SHA 清单
check_sha() { # path expected label
  local p=$1 e=$2 l=$3 got
  got=$(sha256sum "$p" 2>/dev/null | cut -d' ' -f1)
  [ "$got" = "$e" ] && ok "sha $l" || bad "sha $l (got ${got:-missing})"
}
check_sha /tmp/d0cc/d0_stop.py d6d49d75012a0789d3bff3e7b522e9cd496844ca0ee0424608ab8e6b43a33886 stop
check_sha /workspace/yangchen/work/src/vllm/vllm/model_executor/models/clef_joint_schema.py aa3e17585f8bd60a4d05e342abc7e2773c79a0b2d49f0370d15e29bcc6b72503 source
check_sha /workspace/yangchen/work/grouped-deepdive-cell-a15-large/text-body.json a7dd3f64133cfc9ecdc0b7784867bc8f3f5529b5465a6f5fa8da254cc28b5b38 text-body
check_sha /workspace/yangchen/work/grouped-deepdive-cell-a15-large/image-body.json f3229f5c120098ac4eebcc6f4025c28254e35f959fd01998cac5db7b8d209aab image-body
check_sha "$STAGE/supervised_probe_window.py" 1f780bf884cd6725574ffc71e89c87011b7d1dc2d8c7f110217ce7dbe3ce85a4 window
check_sha "$STAGE/run_probe.py" 8e3088042ad771c4d9a1b51d94ad1f49ec918c078d0b80ee25e8b3ca99cddf2f run_probe
check_sha "$STAGE/binding_probe.cpp" 101660f812306177f6d09c77644492f8f1b4a7f08ec6f22fc54e3caf7986ce3f binding
check_sha "$STAGE/probe_adapter.so" d181b15ab75e25982e527303de616c76a29e8f162038eb9f7dfe2a0c364110c3 adapter
check_sha "$STAGE/test_probe_cpu.py" ad257b5fd702d0692140dd3ece89362bdba6081f1782d7e439b7094a9f449719 test_probe
check_sha "$STAGE/manifest.json" 8e324beae91cc858ef557fed97c7c60d7dbbef30eeca6709d84355e3f3f9a0f8 manifest
check_sha "$CELL/plan-h11-observe-v1.json" c95ae5c366a370423fdc42c99d5c94d35892c78f98a7b40dacf9b5bf85b21d15 plan
check_sha "$CELL/root-identity-h11-observe-v1.json" de780872a55d49a0db494f0715055f68b7e0dd1d6bdaea358275922781d18752 root-identity

# 8) 无其他设备格 / 无本窗口遗留 child
if pgrep -fa 'run_probe\.py|supervised_probe_window\.py|torchrun' 2>/dev/null | grep -v 'preflight-rescan' | grep -q .; then
  bad "other probe processes present"; pgrep -fa 'run_probe\.py|supervised_probe_window\.py|torchrun' | grep -v preflight-rescan
else
  ok "no other probe/torchrun processes"
fi

echo
if [ "$FAIL" -eq 0 ]; then echo "RESULT: ALL PASS"; else echo "RESULT: FAIL"; fi
exit "$FAIL"
