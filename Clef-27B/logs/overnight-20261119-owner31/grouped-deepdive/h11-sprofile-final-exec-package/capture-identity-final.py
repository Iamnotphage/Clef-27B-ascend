#!/usr/bin/env python3
"""Capture a fresh /proc identity atomically (CPU-only; no device/service calls)."""
import argparse, json, os, sys
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--pid", type=int, required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
pid = a.pid
try:
    st = Path(f"/proc/{pid}/stat").read_text()
    r = st[st.rfind(")") + 2:].split()
    live = {"pid": pid, "ppid": int(r[1]), "pgid": int(r[2]), "sid": int(r[3]),
            "starttime": r[19], "state": r[0],
            "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
            "cmdline": Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").strip().decode("utf-8", "replace")}
except OSError as e:
    print(f"identity capture failed: {e}", file=sys.stderr)
    sys.exit(3)
out = Path(a.out)
tmp = out.with_suffix(out.suffix + ".tmp")
out.parent.mkdir(parents=True, exist_ok=True)
tmp.write_text(json.dumps(live, indent=1) + "\n")
os.replace(tmp, out)
print(json.dumps(live, indent=1))
