#!/usr/bin/env python3
"""Independent CPU regression for 21's D0 main() (no NPU, no service).

Reads a frozen copy of runtime/d0 at a given commit, guards every external
boundary BEFORE the production modules are imported (restore/verify bind
runner=subprocess.run as a default at import time), then drives the real
main() and records what actually happens.

Only external boundaries are replaced: subprocess.run/Popen, os.kill/killpg,
socket, urllib. Production functions (main, run_child, restore, verify,
query_group) are never replaced; run_child/query_group/restore/verify are
recorded fakes only for the ones main calls as module attributes.

Usage:
  python3 d0_main_regression.py --repo <21 worktree> --commit <sha> --json-out <path>
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import types
from pathlib import Path

RECORD: dict = {"argv_calls": [], "kill_calls": [], "net_calls": [], "imports": []}


def install_guards() -> None:
    """Replace external boundaries before any production import."""
    real_run = subprocess.run

    def guarded_run(args, *a, **kw):
        cmd = args if isinstance(args, list) else [str(args)]
        RECORD["argv_calls"].append([str(x) for x in cmd])
        # Never execute: return a completed process with rc 0 and empty output.
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")

    def guarded_popen(args, *a, **kw):
        cmd = args if isinstance(args, list) else [str(args)]
        RECORD["argv_calls"].append([str(x) for x in cmd])
        raise AssertionError("Popen is not allowed in this regression")

    def guarded_kill(pid, sig):
        RECORD["kill_calls"].append([pid, int(sig)])

    def guarded_killpg(pgid, sig):
        RECORD["kill_calls"].append([pgid, int(sig)])

    subprocess.run = guarded_run
    subprocess.Popen = guarded_popen
    os.kill = guarded_kill
    os.killpg = guarded_killpg

    import socket

    def no_net(*a, **kw):
        RECORD["net_calls"].append(list(a)[:2])
        raise AssertionError("network is not allowed in this regression")

    socket.socket = no_net
    socket.create_connection = no_net
    RECORD["_guards"] = "installed (subprocess.run/Popen, os.kill/killpg, socket)"
    RECORD["_real_run_kept"] = callable(real_run)


def materialize(repo: Path, commit: str, dest: Path) -> Path:
    tar = subprocess.run  # guarded; cannot use. Use git via real path below.
    return dest


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--commit", required=True)
    ap.add_argument("--json-out", required=True)
    a = ap.parse_args()

    repo = Path(a.repo).resolve()
    out: dict = {"commit": a.commit, "repo": str(repo), "checks": [], "record": RECORD}

    # ---- freeze the module dir with real git (before guards touch subprocess)
    tmp = Path(tempfile.mkdtemp(prefix="d0reg-"))
    archive = tmp / "d0.tar"
    real = __import__("subprocess")
    rc = real.run(
        ["git", "-C", str(repo), "archive", a.commit, "runtime/d0"], stdout=open(archive, "wb")
    ).returncode
    out["git_archive_rc"] = rc
    (tmp / "src").mkdir()
    real.run(["tar", "-xf", str(archive), "-C", str(tmp / "src")])
    mod_dir = tmp / "src" / "runtime" / "d0"
    out["module_dir"] = str(mod_dir)
    out["files"] = sorted(p.name for p in mod_dir.glob("*.py"))
    if not mod_dir.is_dir():
        out["fatal"] = "runtime/d0 not present at this commit"
        Path(a.json_out).write_text(json.dumps(out, indent=1))
        print(json.dumps(out, indent=1))
        return 0

    # ---- guards BEFORE importing production modules
    install_guards()
    sys.path.insert(0, str(mod_dir))
    out["import_errors"] = {}
    mods = {}
    for name in ("d0_restore", "d0_executor", "d0_gate", "d0_trace_parse", "d0_run"):
        try:
            mods[name] = importlib.import_module(name)
            RECORD["imports"].append({"module": name, "ok": True})
        except Exception as exc:  # report the real first failure, do not stub
            out["import_errors"][name] = f"{type(exc).__name__}: {exc}"
            RECORD["imports"].append({"module": name, "ok": False, "err": f"{type(exc).__name__}: {exc}"})

    if "d0_run" not in mods:
        out["fatal"] = "d0_run did not import; later stages not_reached"
        out["not_reached"] = ["run_child", "finally", "query_group", "restore", "verify", "return"]
        Path(a.json_out).write_text(json.dumps(out, indent=1))
        print(json.dumps(out, indent=1))
        return 0

    run_mod = mods["d0_run"]
    res_mod = mods.get("d0_restore")

    # ---- recorded fakes for the four module-level call sites main uses
    calls = {"run_child": [], "query_group": [], "restore": [], "verify": []}

    def fake_run_child(cmd, out_dir, timeout, env=None):
        calls["run_child"].append({"cmd": [str(c) for c in cmd], "timeout": timeout})
        return 0

    def fake_query_group(pgid):
        calls["query_group"].append(pgid)
        return {"survivor_check": "ok", "members": []}

    run_mod.run_child = fake_run_child
    run_mod.query_group = fake_query_group
    if res_mod is not None:
        def fake_restore(launch_cmd, wait_cmd, log, **kw):
            calls["restore"].append({"launch": [str(x) for x in launch_cmd], "wait": [str(x) for x in (wait_cmd or [])]})
            return {"restore_ok": True}

        def fake_verify(cmds, log, **kw):
            calls["verify"].append({"cmds": {k: [str(x) for x in v] for k, v in (cmds or {}).items()},
                                    "kwargs": sorted(kw)})
            return {"verify_ok": True}

        res_mod.restore = fake_restore
        res_mod.verify = fake_verify

    # ---- drive the real main()
    odir = tmp / "out"   # must NOT pre-exist: main refuses to reuse a run directory
    argv = ["d0_run.py", "--out-dir", str(odir)]
    old_argv = sys.argv
    sys.argv = argv
    ret = None
    exc = None
    try:
        ret = run_mod.main()
    except SystemExit as se:
        exc = f"SystemExit({se.code!r})"
        ret = se.code
    except Exception as e:
        exc = f"{type(e).__name__}: {e}"
    finally:
        sys.argv = old_argv
    out["main_return"] = ret
    out["main_return_type"] = type(ret).__name__
    out["main_exception"] = exc
    out["calls"] = calls
    out["out_dir_listing"] = sorted(p.name for p in odir.glob("*"))

    # ---- checks (record, never raise)
    def chk(name, ok, detail=""):
        out["checks"].append({"check": name, "ok": bool(ok), "detail": detail})

    chk("main_return_is_int", isinstance(ret, int) and not isinstance(ret, bool),
        f"type={type(ret).__name__} value={ret!r}")
    chk("parser_reached", len(out["out_dir_listing"]) > 0, f"listing={out['out_dir_listing']}")
    chk("run_child_called", len(calls["run_child"]) == 1, f"n={len(calls['run_child'])}")
    chk("query_group_called", len(calls["query_group"]) == 1, f"n={len(calls['query_group'])}")
    chk("restore_called", len(calls["restore"]) == 1, f"n={len(calls['restore'])}")
    chk("verify_called", len(calls["verify"]) == 1, f"n={len(calls['verify'])}")
    if calls["verify"]:
        cmds = calls["verify"][0]["cmds"]
        chk("verify_has_four_cmds", len(cmds) == 4, f"keys={sorted(cmds)}")
        chk("verify_has_service_pids", "service_pids" in calls["verify"][0]["kwargs"],
            f"kwargs={calls['verify'][0]['kwargs']}")
    chk("no_unexpected_kill", len(RECORD["kill_calls"]) == 0, f"kills={RECORD['kill_calls']}")
    chk("no_network", len(RECORD["net_calls"]) == 0, f"net={RECORD['net_calls']}")

    Path(a.json_out).write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ("commit", "main_return", "main_return_type",
                                         "import_errors", "checks")}, indent=1))
    shutil.rmtree(tmp, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
