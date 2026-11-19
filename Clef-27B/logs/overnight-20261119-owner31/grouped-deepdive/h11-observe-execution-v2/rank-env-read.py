#!/usr/bin/env python3
"""H11-observe-v1 rank env 三白名单读取（doc-only；只读 /proc）。

用法： python3 -I rank-env-read.py [CELL] [--watch seconds]
- PID 来源（rank 存活期即可取得）：
  1) CELL/run-h11-observe-v1/descendants.json 的 `entries`（窗口运行中周期写入）；
  2) /proc 扫描：pgid == descendants.root_pid 的进程（含 torchrun rank 子进程）。
- 读取 /proc/<pid>/environ 的 ASCEND_GLOBAL_LOG_LEVEL / ASCEND_PROCESS_LOG_PATH /
  CLEF_WINDOW_OWNER；缺失输出 MISSING，不推断。
- 不触发 NPU、不停止/恢复服务；只读 /proc。
"""
import json
import os
import sys
import time

CELL = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("--") \
    else "/workspace/yangchen/work/grouped-deepdive-cell-h11-observe"
RUN = os.path.join(CELL, "run-h11-observe-v1")
WATCH = 0
if "--watch" in sys.argv:
    WATCH = float(sys.argv[sys.argv.index("--watch") + 1])

KEYS = ("ASCEND_GLOBAL_LOG_LEVEL", "ASCEND_PROCESS_LOG_PATH", "CLEF_WINDOW_OWNER")


def read_entries():
    path = os.path.join(RUN, "descendants.json")
    try:
        with open(path) as f:
            doc = json.load(f)
        return doc.get("entries", []), doc.get("root_pid")
    except (OSError, ValueError):
        return [], None


def scan_pgid(root_pid):
    pids = []
    for name in os.listdir("/proc"):
        if not name.isdigit():
            continue
        pid = int(name)
        try:
            stat = open(f"/proc/{pid}/stat").read()
            r = stat[stat.rfind(")") + 2:].split()
        except (OSError, ValueError):
            continue
        if int(r[2]) == root_pid:
            pids.append(pid)
    return pids


def env3(pid):
    try:
        raw = open(f"/proc/{pid}/environ", "rb").read()
    except OSError as exc:
        return {"error": type(exc).__name__, "errno": getattr(exc, "errno", None)}
    values = {}
    for item in raw.split(b"\0"):
        if b"=" not in item:
            continue
        k, v = item.split(b"=", 1)
        ks = k.decode("utf-8", "replace")
        if ks in KEYS:
            values[ks] = v.decode("utf-8", "replace")
    return {k: values.get(k, "MISSING") for k in KEYS}


def snapshot():
    entries, root_pid = read_entries()
    pids = [e["pid"] for e in entries]
    if root_pid:
        pids += scan_pgid(root_pid)
    seen = []
    out = []
    for pid in sorted(set(pids)):
        if pid in seen:
            continue
        seen.append(pid)
        alive = os.path.isdir(f"/proc/{pid}")
        row = {"pid": pid, "alive": alive, "env": env3(pid) if alive else "MISSING(not_alive)"}
        try:
            stat = open(f"/proc/{pid}/stat").read()
            r = stat[stat.rfind(")") + 2:].split()
            row.update({"ppid": int(r[1]), "pgid": int(r[2]), "starttime": r[19]})
        except (OSError, ValueError):
            pass
        out.append(row)
    return {"root_pid": root_pid, "entries": len(entries), "rows": out}


deadline = time.time() + WATCH
while True:
    doc = snapshot()
    print(json.dumps(doc, indent=1))
    if WATCH <= 0 or time.time() >= deadline:
        break
    time.sleep(1.0)
