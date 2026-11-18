#!/usr/bin/env python3
"""Independent CPU regression for 21's D0 main() — v2.

Rules (v2, after the retraction of v1):
  * production functions are NEVER replaced: main, run_child, query_group,
    restore, verify, parse all execute for real;
  * only external boundaries are intercepted: subprocess.run / subprocess.Popen,
    os.kill / os.killpg, socket;
  * the fake child process only supplies what an external child would supply
    (rank/gate result files, pid, communicate/returncode/kill). Supervision
    artefacts (root-child, raw-rc, stdout, tree-after) must be written by the
    real run_child, not by this harness;
  * unknown argv / any network / any signal that is not a recorded kill is
    fail-closed (raise);
  * arrival is observed with sys.setprofile, so no production function is
    patched to make observation possible;
  * the script exits non-zero if any check fails.

Usage:
  python3 d0_main_regression_v2.py --repo <worktree> --commit <sha> --json-out <path>
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import types
from pathlib import Path

REC: dict = {"argv": [], "popens": [], "kills": [], "net": [], "arrived": {}, "sources": {}}

REAL_RUN = subprocess.run
REAL_POPEN = subprocess.Popen


def _argv_str(args) -> list:
    if isinstance(args, (list, tuple)):
        return [str(a) for a in args]
    return [str(args)]


def allowed(argv: list) -> bool:
    """Only these external commands may be simulated; anything else fails closed."""
    joined = " ".join(argv)
    allow_prefixes = (
        "ps ", "ps", "npu-smi", "-m torch.distributed.run",
        "bash /workspace/yangchen/logs/perf-wave2-20261118/launch.sh",
        "bash /workspace/yangchen/logs/perf-wave2-20261118/wait.sh", "curl ", "kill ",
    )
    # the supervised child is <python> -m torch.distributed.run ...
    if "-m torch.distributed.run" in joined:
        return True
    return joined.startswith(allow_prefixes)


def guard() -> None:
    def guarded_run(args, *a, **kw):
        argv = _argv_str(args)
        REC["argv"].append(argv)
        if not allowed(argv):
            raise AssertionError(f"unexpected external command: {argv}")
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")

    class FakePopen:
        def __init__(self, args, *a, **kw):
            self.argv = _argv_str(args)
            REC["argv"].append(self.argv)
            REC["popens"].append(self.argv)
            if not allowed(self.argv):
                raise AssertionError(f"unexpected external command (Popen): {self.argv}")
            self.pid = 424242
            self.returncode = 0
            # supply ONLY what the external child would have produced
            self._emit_child_files(kw.get("cwd"))

        def _emit_child_files(self, cwd):
            if not cwd:
                return
            base = Path(cwd)
            try:
                base.mkdir(parents=True, exist_ok=True)
                (base / "result-rank-0.json").write_text(json.dumps({"rank": 0, "ok": True}))
                (base / "result-rank-1.json").write_text(json.dumps({"rank": 1, "ok": True}))
                (base / "gate.json").write_text(json.dumps({"passed": True}))
            except Exception:
                pass

        def communicate(self, *a, **kw):
            return ("", "")

        def wait(self, *a, **kw):
            return 0

        def poll(self):
            return 0

        def kill(self):
            REC["kills"].append(["popen.kill", self.pid])

        def terminate(self):
            REC["kills"].append(["popen.terminate", self.pid])

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def guarded_kill(pid, sig):
        REC["kills"].append([pid, int(sig)])

    def no_net(*a, **kw):
        REC["net"].append(str(a)[:80])
        raise AssertionError("network is not allowed")

    subprocess.run = guarded_run
    subprocess.Popen = FakePopen
    os.kill = guarded_kill
    os.killpg = guarded_kill
    import socket
    socket.socket = no_net
    socket.create_connection = no_net
    REC["sources"]["guards"] = "subprocess.run/Popen, os.kill/killpg, socket"


def profile_hook(frame, event, arg):
    """Record which production functions are entered, without patching them."""
    if event == "call":
        name = frame.f_code.co_name
        mod = frame.f_globals.get("__name__", "")
        if mod.startswith("d0_"):
            key = f"{mod}.{name}"
            REC["arrived"][key] = REC["arrived"].get(key, 0) + 1
    return profile_hook


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--commit", required=True)
    ap.add_argument("--json-out", required=True)
    a = ap.parse_args()
    repo = Path(a.repo).resolve()
    out = {"target_commit": a.commit, "checks": [], "not_reached": []}

    tmp = Path(tempfile.mkdtemp(prefix="d0v2-"))
    arc = tmp / "d0.tar"
    rc = REAL_RUN(["git", "-C", str(repo), "archive", a.commit, "runtime/d0"],
                  stdout=open(arc, "wb")).returncode
    out["git_archive_rc"] = rc
    if rc != 0:
        out["fatal"] = "commit not resolvable or runtime/d0 absent"
        Path(a.json_out).write_text(json.dumps(out, indent=1))
        print(json.dumps(out, indent=1)); return 1
    (tmp / "src").mkdir()
    REAL_RUN(["tar", "-xf", str(arc), "-C", str(tmp / "src")])
    mod_dir = tmp / "src" / "runtime" / "d0"
    out["files"] = sorted(p.name for p in mod_dir.glob("*.py"))

    guard()                      # BEFORE importing the production modules
    sys.path.insert(0, str(mod_dir))
    try:
        run_mod = importlib.import_module("d0_run")
        out["import_ok"] = True
    except Exception as exc:
        out["import_ok"] = False
        out["import_error"] = f"{type(exc).__name__}: {exc}"
        out["not_reached"] = ["main", "finally", "query_group", "restore", "verify", "parse", "return"]
        Path(a.json_out).write_text(json.dumps(out, indent=1))
        print(json.dumps(out, indent=1)); return 1

    odir = tmp / "out"
    sys.argv = ["d0_run.py", "--out-dir", str(odir)]
    sys.setprofile(profile_hook)
    returned, exc_name = False, None
    try:
        ret = run_mod.main()
        returned = True
    except SystemExit as se:
        ret, returned = se.code, True
        exc_name = f"SystemExit({se.code!r})"
    except Exception as e:
        ret, exc_name = None, f"{type(e).__name__}: {e}"
        returned = False
    finally:
        sys.setprofile(None)

    out["main_returned"] = returned
    out["main_return"] = ret
    out["main_return_type"] = type(ret).__name__ if returned else "not_returned"
    out["main_exception"] = exc_name
    out["arrived"] = sorted(REC["arrived"])
    out["out_dir"] = sorted(p.name for p in odir.glob("*")) if odir.exists() else []
    out["record"] = {k: REC[k] for k in ("argv", "popens", "kills", "net")}

    def chk(name, ok, detail=""):
        out["checks"].append({"check": name, "ok": bool(ok), "detail": detail})

    final = lambda n: f"d0_run.{n}"
    chk("main_returned", returned, exc_name or "")
    chk("main_returns_int", returned and isinstance(ret, int) and not isinstance(ret, bool),
        f"{out['main_return_type']}={ret!r}")
    chk("run_child_reached", any("run_child" in k for k in REC["arrived"]),
        f"arrived={out['arrived']}")
    chk("root_child_written", "root-child.json" in out["out_dir"], f"out={out['out_dir']}")
    chk("query_group_reached", any("query_group" in k for k in REC["arrived"]), "")
    chk("restore_reached", any(k.endswith(".restore") for k in REC["arrived"]), "")
    chk("verify_reached", any(k.endswith(".verify") for k in REC["arrived"]), "")
    chk("no_network", len(REC["net"]) == 0, f"net={REC['net']}")
    chk("no_unexpected_argv", all(allowed(x) for x in REC["argv"]), f"argv={REC['argv']}")

    for name in ("run_child", "query_group", "restore", "verify", "parse"):
        if not any(name in k for k in REC["arrived"]):
            out["not_reached"].append(name)

    out["failed"] = [c["check"] for c in out["checks"] if not c["ok"]]
    Path(a.json_out).write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ("target_commit", "main_returned", "main_return_type",
                                         "main_exception", "out_dir", "not_reached", "failed")}, indent=1))
    shutil.rmtree(tmp, ignore_errors=True)
    return 0 if not out["failed"] else 1


if __name__ == "__main__":
    sys.exit(main())
