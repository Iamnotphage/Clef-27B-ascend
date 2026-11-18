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
    # generic python -m torch.distributed.run is allowed (the supervised child)
    if len(argv) >= 3 and argv[0].endswith("python3.12") or (len(argv) >= 2 and argv[0].endswith("python3")):
        if "-m" in argv and "torch.distributed.run" in argv:
            return True
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
        joined = " ".join(argv)
        # the launch runs through subprocess.run (restore's default runner),
        # so the label fixture must be emitted here as well
        if len(argv) >= 3 and argv[0] == "bash" and argv[1].endswith("launch.sh"):
            label = argv[2]
            REC["label"] = label
            REC["_spawn_tree"]()
            put, mkdir = REC["_vfs_put"], REC["_vfs_dir"]
            base = f"/workspace/yangchen/logs/perf-wave2-20261118/{label}"
            mkdir(base, ["api.pid"])
            put(f"{base}/api.pid", f"{REC['root_pid']}\n")
            REC["api_pid_written"] = f"{base}/api.pid"
        if "curl" in joined:
            if "/health" in joined:
                return types.SimpleNamespace(returncode=0, stdout='{"status":"ready"}', stderr="")
            if "systemone" in joined or "decide" in joined:
                # text sample -> score, image sample -> choice (both shapes given)
                return types.SimpleNamespace(
                    returncode=0,
                    stdout='{"answers":{"answer":{"type":"score"}},'
                           '"choices":[{"type":"choice"}]}', stderr="")
        if joined.startswith("npu-smi"):
            # header + two worker rows: second column is the pid
            rows = ["NPU Name Chip Process id Process name",
                    f"7  0  0  {REC.get('root_pid', 424242)}  python",
                    f"7  1  0  {REC.get('root_pid', 424242)}  python",
                    "7  2  0  424244  VLLMWorker_TP0",
                    "7  3  0  424245  VLLMWorker_TP1"]
            return types.SimpleNamespace(returncode=0, stdout="\n".join(rows), stderr="")
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
            self._emit_launch_fixture(self.argv)
            self._emit_child_files(self.argv)

        def _emit_launch_fixture(self, argv):
            """label is the third token of: bash <launch.sh> <label>"""
            if len(argv) >= 3 and argv[0] == "bash" and argv[1].endswith("launch.sh"):
                label = argv[2]
                REC["label"] = label
                root = 424242
                REC["root_pid"] = root
                base = Path("/workspace/yangchen/logs/perf-wave2-20261118") / label
                try:
                    base.mkdir(parents=True, exist_ok=True)
                    (base / "api.pid").write_text(f"{root}\n")
                    REC["api_pid_written"] = str(base / "api.pid")
                except Exception as exc:
                    REC["api_pid_error"] = f"{type(exc).__name__}: {exc}"

        def _emit_child_files(self, argv):
            """Write only what the external child would write, into --out-dir."""
            out_dir = None
            for i, tok in enumerate(argv):
                if tok == "--out-dir" and i + 1 < len(argv):
                    out_dir = argv[i + 1]
                elif tok.startswith("--out-dir="):
                    out_dir = tok.split("=", 1)[1]
            if not out_dir:
                return
            base = Path(out_dir)
            try:
                base.mkdir(parents=True, exist_ok=True)
                for rank in (0, 1):
                    (base / f"result-rank{rank}.json").write_text(json.dumps({
                        "rank": rank, "rc": 0,
                        "flat_u32": [0] * (72 * 8),
                    }))
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
        # signals to real processes are a boundary we do not simulate: fail closed
        raise AssertionError(f"signal to pid {pid} is not allowed")

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

    # ---- virtual filesystem for every read-only production source ----------
    # Nothing is written to /workspace: all fixtures live in this mapping, while
    # the temp run directory keeps real IO.
    import pathlib

    VFS_FILES: dict = {}
    VFS_DIRS: dict = {}
    real_read_text = pathlib.Path.read_text
    real_is_file = pathlib.Path.is_file
    real_is_dir = pathlib.Path.is_dir
    real_iterdir = pathlib.Path.iterdir

    def vfs_put(path, text):
        VFS_FILES[str(path)] = text
        parent = str(pathlib.Path(path).parent)
        VFS_DIRS.setdefault(parent, set()).add(str(pathlib.Path(path).name))

    def vfs_dir(path, children):
        VFS_DIRS[str(path)] = set(children)

    def fake_read_text(self, *a, **kw):
        p = str(self)
        if p in VFS_FILES:
            return VFS_FILES[p]
        if p.startswith("/workspace/") or p.startswith("/proc/"):
            REC.setdefault("vfs_miss", []).append(p)
            raise FileNotFoundError(p)
        return real_read_text(self, *a, **kw)

    def fake_is_file(self):
        p = str(self)
        if p in VFS_FILES:
            return True
        if p.startswith("/workspace/") or p.startswith("/proc/"):
            return False
        return real_is_file(self)

    def fake_is_dir(self):
        p = str(self)
        if p in VFS_DIRS:
            return True
        if p.startswith("/proc"):
            return False
        return real_is_dir(self)

    def fake_iterdir(self):
        p = str(self)
        if p in VFS_DIRS:
            base = pathlib.Path(p)
            for name in sorted(VFS_DIRS[p]):
                yield base / name
            return
        if p.startswith("/proc"):
            REC.setdefault("vfs_miss", []).append(p)
            return
        yield from real_iterdir(self)

    pathlib.Path.read_text = fake_read_text
    pathlib.Path.is_file = fake_is_file
    pathlib.Path.is_dir = fake_is_dir
    pathlib.Path.iterdir = fake_iterdir

    real_listdir = os.listdir

    def fake_listdir(path="."):
        p = str(path)
        if p in VFS_DIRS:
            return sorted(VFS_DIRS[p])
        if p.startswith("/proc"):
            REC.setdefault("vfs_miss", []).append(p)
            return []
        return real_listdir(path)

    os.listdir = fake_listdir
    REC["_vfs"] = (VFS_FILES, VFS_DIRS, vfs_put, vfs_dir)
    REC["_vfs_put"] = vfs_put
    REC["_vfs_dir"] = vfs_dir

    # /proc: the root plus two VLLMWorker_TP descendants
    ROOT_PID = 424242
    WORKERS = [424301, 424302]
    vfs_dir("/proc", [])   # service tree must NOT exist before the launch
    def spawn_service_tree():
        """Called from the launch fixture: the tree appears only then."""
        vfs_dir("/proc", [str(ROOT_PID)] + [str(w) for w in WORKERS])
        for pid, start, ppid in [(ROOT_PID, "1000", 1),
                                 (WORKERS[0], "1100", ROOT_PID),
                                 (WORKERS[1], "1200", ROOT_PID)]:
            fields = ["0"] * 20
            fields[1] = str(ppid)
            fields[19] = start
            vfs_put(f"/proc/{pid}/stat", f"{pid} (python3) " + " ".join(fields) + "\n")
    REC["_spawn_tree"] = spawn_service_tree
    vfs_put("/proc/sys/kernel/random/boot_id", "BOOT-TEST-0001\n")
    REC["root_pid"] = ROOT_PID
    # discovery reads SERVICE_LABEL (a module constant), not the dynamic label
    REC["workers"] = WORKERS
    REC["sources"]["fs"] = "Path.read_text for /proc/*/stat, boot_id, api.pid"


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
    ap.add_argument("--scenario", default="happy",
                    choices=("happy", "root_changed", "boot_missing"))
    ap.add_argument("--service-label", default=None,
                    help="pass --service-label to the operator so fixtures can place api.pid")
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
    REC["scenario"] = a.scenario
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
    if a.service_label:
        sys.argv += ["--service-label", a.service_label]
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
    mods_seen = {}
    for m in ("d0_run", "d0_restore"):
        mod = sys.modules.get(m)
        if mod is not None:
            f = getattr(mod, "__file__", None)
            spec = getattr(mod, "__spec__", None)
            import hashlib
            try:
                sha = hashlib.sha256(open(f, "rb").read()).hexdigest() if f else None
            except Exception:
                sha = None
            mods_seen[m] = {"__file__": f, "origin": getattr(spec, "origin", None), "sha256": sha}
    out["module_identity"] = mods_seen
    out["fixtures"] = {k: REC.get(k) for k in ("label", "root_pid", "api_pid_written", "api_pid_error")}

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
    # a false verify_ok does NOT prove verify ran: main may set verified=False
    # directly when discovery reported an error.
    verify_arrived = any(k.endswith(".verify") for k in REC["arrived"])
    chk("verify_reached", verify_arrived,
        f"arrived={verify_arrived}; verify_ok={out.get('verify_ok')!r}; "
        f"verify_blocked={out.get('verify_blocked')!r}")
    chk("no_network", len(REC["net"]) == 0, f"net={REC['net']}")
    chk("no_unexpected_argv", all(allowed(x) for x in REC["argv"]), f"argv={REC['argv']}")

    if True:
        res_mod = sys.modules.get("d0_restore")
        info = res_mod.discover_service_pids(REC.get("label"), before=None) if (res_mod and REC.get("label")) else {}
        out["discovery"] = {k: (sorted(v) if isinstance(v, set) else v) for k, v in info.items()}
        if a.scenario != "happy":
            chk("discovery_reports_error", bool(info.get("error")), f"error={info.get('error')!r}")
        if a.scenario == "root_changed":
            chk("error_with_nonempty_pids", bool(info.get("error")) and bool(info.get("pids")),
                f"pids={out['discovery'].get('pids')}")
        if a.scenario == "boot_missing":
            chk("boot_missing_not_stable", info.get("identity_stable") is not True,
                f"identity_stable={info.get('identity_stable')!r}")
        chk("verify_not_passed_on_bad_identity",
            (not any(k.endswith(".verify") for k in REC["arrived"])) or (not out.get("verify_pids_ok", False)),
            f"verify_arrived={any(k.endswith('.verify') for k in REC['arrived'])}")

    for name in ("run_child", "query_group", "restore", "verify", "parse"):
        if not any(name in k for k in REC["arrived"]):
            out["not_reached"].append(name)

    out["failed"] = [c["check"] for c in out["checks"] if not c["ok"]]
    Path(a.json_out).write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ("target_commit", "main_returned", "main_return_type",
                                         "main_exception", "out_dir", "not_reached", "failed")}, indent=1))
    # preserve the evidence that would otherwise be deleted with the temp dir
    keep = {}
    for name in ("restore.json", "restore-failure.json", "parse-failure.json",
                 "root-child.json", "raw-rc.json", "tree-after.json",
                 "result-rank0.json"):
        f = odir / name
        if f.is_file():
            try:
                keep[name] = json.loads(f.read_text())
            except Exception:
                keep[name] = f.read_text()[:2000]
    out["artifacts"] = keep
    _r = keep.get("restore.json") or {}
    out["verify_ok"] = _r.get("verify_ok")
    out["verify_blocked"] = _r.get("verify_blocked")
    out["restore_ok"] = _r.get("restore_ok")
    Path(a.json_out).write_text(json.dumps(out, indent=1))
    REC["_keepdir"] = str(tmp)
    if os.environ.get("D0REG_KEEP") != "1":
        shutil.rmtree(tmp, ignore_errors=True)
    return 0 if not out["failed"] else 1


if __name__ == "__main__":
    sys.exit(main())
