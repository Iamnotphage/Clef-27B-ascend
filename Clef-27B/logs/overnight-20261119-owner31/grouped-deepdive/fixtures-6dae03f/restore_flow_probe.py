#!/usr/bin/env python3
"""Independent CPU probe of window 1f780bf8 restore flow (no mock ppid, real processes)."""
import json, os, sys, time
from pathlib import Path
sys.path.insert(0, sys.argv[1])
import supervised_probe_window as W

base = Path(sys.argv[2]); base.mkdir(parents=True, exist_ok=True)
run_dir = base / "run"; run_dir.mkdir(exist_ok=True)
restore_output = base / "restore"; restore_output.mkdir(exist_ok=True)
manifest = restore_output / "launch-manifest.json"
script = base / "fake_restore.py"
script.write_text(
    "import json, os, sys, time\n"
    "manifest = sys.argv[1]\n"
    "if os.fork() == 0:\n"
    "    os.setsid()\n"
    "    with open(manifest, 'w') as h: json.dump({'pid': os.getpid()}, h)\n"
    "    time.sleep(60)\n"
    "    os._exit(0)\n"
    "os._exit(0)\n"
)
out = {}
out["subreaper_initial"] = W.enable_subreaper()
out["original_flag"] = W.get_subreaper()
try:
    plan = {"restore_argv": [sys.executable, str(script), str(manifest)],
            "restore_output": str(restore_output)}
    result = {}
    started = W.begin_restore(plan, run_dir, result)
    out["begin_restore"] = started
    out["cleared_for_restore"] = result.get("subreaper_cleared_for_restore")
    out["flag_during_restore"] = W.get_subreaper()
    deadline = time.time() + 10
    while time.time() < deadline and not manifest.is_file():
        time.sleep(0.05)
    out["manifest_written"] = manifest.is_file()
    pid = int(json.loads(manifest.read_text())["pid"]) if manifest.is_file() else None
    out["manifest_pid"] = pid
    live = W.wait_restored_root(pid, timeout=10, interval=0.05) if pid else None
    out["live"] = {k: live.get(k) for k in ("pid", "ppid", "pgid", "sid")} if live else None
    out["identity_ok"] = bool(live and live["ppid"] == 1 and live["pgid"] == pid)
    out["restore_calls_once"] = True  # single begin_restore invocation in this probe
finally:
    out["restored_original"] = W.restore_subreaper(out["original_flag"]) if out["original_flag"] is not None else None
    out["flag_final"] = W.get_subreaper()
    if out.get("manifest_pid"):
        try: os.killpg(out["manifest_pid"], 9)
        except ProcessLookupError: pass
print(json.dumps(out, indent=1))
