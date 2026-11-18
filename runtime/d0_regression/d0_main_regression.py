#!/usr/bin/env python3
"""Independent CPU regression for 21's D0 main() and restore wrapper.

This is the 24-owned rewrite of the 22 handoff baseline.  The production
functions (main, run_child, query_group, restore, verify, discover, parse) are
never replaced.  Only external boundaries are intercepted, and they are
installed before the production modules are imported:

  * subprocess.run / subprocess.Popen (exact argv allow-list, fail closed)
  * os.kill / os.killpg / os.getpgid (simulated PIDs only; no real signal)
  * socket / create_connection (fail closed)
  * Path.read_text/read_bytes/is_file/is_dir/iterdir and os.listdir for
    /workspace and /proc (virtual filesystem; no real write to either)

Each scenario runs in its own child process, so sys.modules and module-level
state cannot leak between cases.  Supervision artefacts (root-child.json,
raw-rc.json, stdout.log, stderr.log, tree-after.json) are written by the real
run_child; FakePopen only supplies external rank/gate files and a simulated
process handle.  The real parser decodes the rank files; the fixture values
are validated by the real parser, not copied from a previous result.

Usage:
  python3 d0_main_regression.py --repo <worktree> --commit <sha> \
      --json-out <suite.json> [--freeze-dir <dir>] [--keep-temp]
  python3 d0_main_regression.py --single --scenario <name> ...   # internal
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import traceback
import types
from pathlib import Path

TARGET = "56427329248f49490a87877f6a47d04239418bfe"
MODULES = ("d0_run", "d0_restore", "d0_launch", "d0_gate", "d0_executor", "d0_trace_parse")
SERVICE_LOG_ROOT = "/workspace/yangchen/logs/perf-wave2-20261118"
LAUNCH_SH = f"{SERVICE_LOG_ROOT}/launch.sh"
WAIT_SH = f"{SERVICE_LOG_ROOT}/wait.sh"
HEALTH_URL = "http://127.0.0.1:18143/health"
API_BASE = "http://127.0.0.1:18143"
TEXT_SOURCE = "/workspace/yangchen/logs/tuning/20261114-013000-fix4/opendecisions/all2352.json"
IMAGE_SOURCE = "/workspace/yangchen/logs/perf-wave2-20261118/erqa/samples/erqa-0000/request.json"
TEXT_REQUEST = {"prompt": "x", "candidates": ["a", "b"]}
IMAGE_REQUEST = {"request": {"prompt": "x", "images": ["i0"], "candidates": ["a", "b"]}}
TEXT_BODY = json.dumps(TEXT_REQUEST, ensure_ascii=False)
IMAGE_BODY = json.dumps(IMAGE_REQUEST, ensure_ascii=False)
LABEL_RE = re.compile(r"^d0-restore-\d{8}T\d{6}Z-[0-9a-f]{8}$")
MAGIC = 0xD0D0C0DE
SEAL = 0xC0FFEE00
AIC_CAP, SUB_CAP, SLOTS = 24, 2, 72
M = 967
ROOT_PID = 424242
CHILD_PID = 424200
WORKERS = (424301, 424302)
BOOT = "BOOT-TEST-0001"

HAPPY = "happy"
SCENARIOS = (
    "happy", "root_changed", "boot_missing", "boot_empty", "timeout",
    "child_nonzero", "query_unavailable", "cleanup_residue", "verify_failure",
    "parse_failure",
)
NONZERO_SCENARIOS = {
    "root_changed": None, "boot_missing": None, "boot_empty": None,
    "timeout": 2, "child_nonzero": 7, "query_unavailable": None,
    "cleanup_residue": None, "verify_failure": None, "parse_failure": 5,
}

REAL_RUN = subprocess.run
REAL_POPEN = subprocess.Popen


def sha256_bytes(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def argv_str(args) -> list:
    if isinstance(args, (list, tuple)):
        return [str(a) for a in args]
    return [str(args)]


def valid_trace(rank: int) -> list:
    """A parser-valid two-record trace: one AIC and one AIV worker claim."""
    raw = [0] * (SLOTS * 8)

    def put(slot, domain, block, sub, start, claim, bits, bits2):
        ident = (block << 8) | sub
        raw[slot * 8:(slot + 1) * 8] = [
            MAGIC, ident, domain, start, claim, bits, bits2, SEAL ^ ident]

    put(0, 0, 0, 0, 0, 0, 0b1, 0b100)                 # AIC: K1, K8_final
    block = 0 if rank == 0 else 1
    put(AIC_CAP + block * SUB_CAP, 1, block, 0, 0, M,
        0b1111, 0b1100)                               # AIV worker: K1,K4,K8_final
    return raw


def expected_commands(out_dir: Path, mod_dir: Path) -> dict:
    child = [sys.executable, "-m", "torch.distributed.run", "--standalone",
             "--nproc-per-node=2", str(mod_dir / "d0_launch.py"),
             "--out-dir", str(out_dir), "--rows", str(M), "--seed", "20261118",
             "--mm-dim", "1", "--use-hccl", "2",
             "--adapter", "None", "--adapter-sha", "None"]
    return {
        "child": child,
        "ps": ["ps", "-eo", "pid=,pgid="],
        "wait": ["bash", WAIT_SH],
        "health": ["curl", "-s", "-m", "5", HEALTH_URL],
        "workers": ["npu-smi", "info"],
        "text": ["curl", "-s", "-m", "120", "-X", "POST", f"{API_BASE}/v1/systemone",
                 "-H", "Content-Type: application/json", "-d", TEXT_BODY],
        "image": ["curl", "-s", "-m", "120", "-X", "POST", f"{API_BASE}/v1/decide",
                  "-H", "Content-Type: application/json", "-d", IMAGE_BODY],
    }


def classify(argv: list, expected: dict) -> str:
    for kind, want in expected.items():
        if argv == want:
            return kind
    if len(argv) == 3 and argv[0] == "bash" and argv[1] == LAUNCH_SH and LABEL_RE.match(argv[2]):
        return "launch_service"
    raise AssertionError(f"unexpected argv: {argv!r}")


def install_guard(state: dict) -> None:
    import pathlib
    import socket

    vfs_files: dict[str, str] = state["vfs_files"]
    vfs_dirs: dict[str, set] = state["vfs_dirs"]
    expected = state["expected"]

    def vfs_put(path, text):
        vfs_files[str(path)] = text
        parent = str(pathlib.Path(path).parent)
        vfs_dirs.setdefault(parent, set()).add(str(pathlib.Path(path).name))

    def vfs_dir(path, children):
        vfs_dirs[str(path)] = set(children)

    state["vfs_put"], state["vfs_dir"] = vfs_put, vfs_dir

    def in_virtual(path: str) -> bool:
        return path.startswith("/workspace/") or path.startswith("/proc")

    real_read_text = pathlib.Path.read_text
    real_read_bytes = pathlib.Path.read_bytes
    real_is_file = pathlib.Path.is_file
    real_is_dir = pathlib.Path.is_dir
    real_iterdir = pathlib.Path.iterdir
    real_write_text = pathlib.Path.write_text
    real_mkdir = pathlib.Path.mkdir

    def fake_read_text(self, *a, **kw):
        p = str(self)
        if p in vfs_files:
            return vfs_files[p]
        if in_virtual(p):
            if p == f"/proc/{ROOT_PID}/stat" and not state["spawned"]:
                raise FileNotFoundError(p)
            if p == "/proc/sys/kernel/random/boot_id" and state["scenario"] == "boot_missing":
                raise FileNotFoundError(p)
            state["vfs_miss"].append(p)
            raise FileNotFoundError(p)
        return real_read_text(self, *a, **kw)

    def fake_read_bytes(self, *a, **kw):
        p = str(self)
        if p in vfs_files:
            return vfs_files[p].encode()
        if p.startswith("/proc/"):
            parent = str(pathlib.Path(p).parent)
            if parent in vfs_dirs or p.endswith("/cmdline") or p == root_stat_missing_allowed():
                return b""
        if in_virtual(p):
            state["vfs_miss"].append(p)
            raise FileNotFoundError(p)
        return real_read_bytes(self, *a, **kw)

    def fake_is_file(self):
        p = str(self)
        if p in vfs_files:
            return True
        if in_virtual(p):
            return False
        return real_is_file(self)

    def fake_is_dir(self):
        p = str(self)
        if p in vfs_dirs:
            return True
        if in_virtual(p):
            return False
        return real_is_dir(self)

    def fake_iterdir(self):
        p = str(self)
        if p in vfs_dirs:
            base = pathlib.Path(p)
            for name in sorted(vfs_dirs[p]):
                yield base / name
            return
        if p.startswith("/proc"):
            state["vfs_miss"].append(p)
            return
        yield from real_iterdir(self)

    def fake_write_text(self, *a, **kw):
        if in_virtual(str(self)):
            state["workspace_writes"].append(str(self))
            raise AssertionError(f"real write to virtual path: {self}")
        return real_write_text(self, *a, **kw)

    def fake_mkdir(self, *a, **kw):
        if in_virtual(str(self)):
            state["workspace_writes"].append(str(self))
            raise AssertionError(f"real mkdir to virtual path: {self}")
        return real_mkdir(self, *a, **kw)

    pathlib.Path.read_text = fake_read_text
    pathlib.Path.read_bytes = fake_read_bytes
    pathlib.Path.is_file = fake_is_file
    pathlib.Path.is_dir = fake_is_dir
    pathlib.Path.iterdir = fake_iterdir
    pathlib.Path.write_text = fake_write_text
    pathlib.Path.mkdir = fake_mkdir

    real_listdir = os.listdir

    def fake_listdir(path="."):
        p = str(path)
        if p in vfs_dirs:
            return sorted(vfs_dirs[p])
        if in_virtual(p):
            state["vfs_miss"].append(p)
            return []
        return real_listdir(path)

    os.listdir = fake_listdir

    def put_proc(pid: int, ppid: int, start: str):
        vfs_dirs.setdefault("/proc", set()).add(str(pid))
        fields = ["0"] * 20
        fields[1] = str(ppid)
        fields[19] = start
        vfs_put(f"/proc/{pid}/stat", f"{pid} (python3) " + " ".join(fields) + "\n")

    state["put_proc"] = put_proc
    vfs_dir("/proc", [])
    put_proc(os.getpid(), 1, "900")          # the harness process itself
    if state["scenario"] != "boot_missing":
        vfs_put("/proc/sys/kernel/random/boot_id", "" if state["scenario"] == "boot_empty" else BOOT)
    vfs_put(TEXT_SOURCE, json.dumps({"samples": [{"name": "od-1534", "request": TEXT_REQUEST}]}))
    vfs_put(IMAGE_SOURCE, json.dumps(IMAGE_REQUEST))
    state["proc_before_launch"] = sorted(
        name for name in vfs_dirs.get("/proc", set()) if name.isdigit())

    def spawn_tree():
        state["spawned"] = True
        put_proc(ROOT_PID, 1, "1000")
        put_proc(WORKERS[0], ROOT_PID, "1100")
        put_proc(WORKERS[1], ROOT_PID, "1200")
        state["root_reads"] = 0

    def kill_tree():
        state["tree_killed"] = True
        vfs_dirs["/proc"] = {str(os.getpid())}

    state["spawn_tree"], state["kill_tree"] = spawn_tree, kill_tree

    def root_stat_missing_allowed():
        return f"/proc/{ROOT_PID}/stat"
    root_stat = f"/proc/{ROOT_PID}/stat"
    real_vfs_read = fake_read_text

    def scenario_read(self, *a, **kw):
        p = str(self)
        if p == root_stat and state["scenario"] == "root_changed":
            state["root_reads"] += 1
            if state["root_reads"] > 1:
                vfs_files[p] = vfs_files[p].replace(" 1000\n", " 9999\n")
        return real_vfs_read(self, *a, **kw)

    pathlib.Path.read_text = scenario_read

    def guarded_run(args, *a, **kw):
        argv = argv_str(args)
        try:
            kind = classify(argv, expected)
        except AssertionError:
            state["unexpected"].append(argv)
            raise
        state["argv"].append({"kind": kind, "argv": argv})
        if kind == "launch_service":
            label = argv[2]
            state["label"] = label
            spawn_tree()
            base = f"{SERVICE_LOG_ROOT}/{label}"
            vfs_dir(base, ["api.pid"])
            vfs_put(f"{base}/api.pid", f"{ROOT_PID}\n")
            return types.SimpleNamespace(returncode=0, stdout="", stderr="")
        if kind == "wait":
            return types.SimpleNamespace(returncode=0, stdout="", stderr="")
        if kind == "ps":
            if state["scenario"] == "query_unavailable":
                return types.SimpleNamespace(returncode=1, stdout="", stderr="ps unavailable")
            if state["scenario"] == "cleanup_residue":
                return types.SimpleNamespace(returncode=0, stdout=f"{CHILD_PID} {CHILD_PID}\n", stderr="")
            return types.SimpleNamespace(returncode=0, stdout="", stderr="")
        if kind == "workers":
            rows = ["+---+--------+----------------+",
                    f"| 0 | {ROOT_PID} | python |"]
            workers = state["npu_workers"]
            for index, pid in enumerate(workers):
                rows.append(f"| {index} | {pid} | VLLMWorker_TP |")
            return types.SimpleNamespace(returncode=0, stdout="\n".join(rows), stderr="")
        if kind == "health":
            body = '{"status":"ready"}' if state["scenario"] != "verify_failure" else '{"status":"loading"}'
            return types.SimpleNamespace(returncode=0, stdout=body, stderr="")
        if kind in ("text", "image"):
            expect = "score" if kind == "text" else "choice"
            if state["verify_bad"] and kind == "text":
                expect = "choice"
            body = json.dumps({"answers": {"answer": {"type": expect}}})
            return types.SimpleNamespace(returncode=0, stdout=body, stderr="")
        raise AssertionError(f"run kind not handled: {kind}")

    class FakePopen:
        def __init__(self, args, *a, **kw):
            self.argv = argv_str(args)
            try:
                kind = classify(self.argv, expected)
            except AssertionError:
                state["unexpected"].append(self.argv)
                raise
            if kind != "child":
                raise AssertionError(f"Popen only allowed for the supervised child: {self.argv}")
            state["argv"].append({"kind": "child", "argv": self.argv})
            self.pid = CHILD_PID
            state["put_proc"](CHILD_PID, os.getpid(), "950")
            self.returncode = 0 if state["child_rc"] == 0 else state["child_rc"]
            self._timeout_raised = False
            if self.returncode == 0:
                self._emit_child_files()
                self._emit_gate()

        def _out_dir(self):
            for index, token in enumerate(self.argv):
                if token == "--out-dir" and index + 1 < len(self.argv):
                    return Path(self.argv[index + 1])
            return None

        def _emit_child_files(self):
            base = self._out_dir()
            if base is None:
                return
            base.mkdir(parents=True, exist_ok=True)
            for rank in (0, 1):
                raw = [1] if state["scenario"] == "parse_failure" else valid_trace(rank)
                (base / f"result-rank{rank}.json").write_text(json.dumps(
                    {"rank": rank, "rc": 0, "flat_u32": raw}))

        def _emit_gate(self):
            base = self._out_dir()
            if base is not None:
                (base / "gate.json").write_text(json.dumps({"gate_pass": True}))

        def communicate(self, *a, **kw):
            if state["scenario"] == "timeout" and not self._timeout_raised:
                self._timeout_raised = True
                raise subprocess.TimeoutExpired(self.argv, kw.get("timeout"))
            return ("", "")

        def wait(self, *a, **kw):
            return self.returncode

        def poll(self):
            return self.returncode

        def kill(self):
            state["kills"].append(["popen.kill", self.pid])
            kill_tree()

        def terminate(self):
            state["kills"].append(["popen.terminate", self.pid])
            kill_tree()

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    known = {CHILD_PID, ROOT_PID, WORKERS[0], WORKERS[1]}

    def fake_getpgid(pid):
        if pid == CHILD_PID:
            return CHILD_PID
        if pid in known:
            return ROOT_PID
        raise ProcessLookupError(pid)

    def fake_kill(pid, sig):
        state["kills"].append([pid, int(sig)])
        if pid in known or pid == ROOT_PID:
            state["kill_simulated"].append([pid, int(sig)])
            kill_tree()
            return None
        raise AssertionError(f"signal to unknown pid {pid} is not allowed")

    def fake_killpg(pgid, sig):
        state["kills"].append(["pgid", pgid, int(sig)])
        if pgid in (ROOT_PID, CHILD_PID):
            state["kill_simulated"].append(["pgid", pgid, int(sig)])
            kill_tree()
            return None
        raise AssertionError(f"signal to unknown pgid {pgid} is not allowed")

    def no_net(*a, **kw):
        state["net"].append(str(a)[:80])
        raise AssertionError("network is not allowed")

    subprocess.run = guarded_run
    subprocess.Popen = FakePopen
    os.kill = fake_kill
    os.killpg = fake_killpg
    os.getpgid = fake_getpgid
    socket.socket = no_net
    socket.create_connection = no_net
    state["guards"] = ["subprocess.run", "subprocess.Popen", "os.kill", "os.killpg",
                       "os.getpgid", "socket", "Path VFS"]


def profile_hook(frame, event, arg):
    if event == "call":
        name = frame.f_code.co_name
        mod = frame.f_globals.get("__name__", "")
        if mod.startswith("d0_"):
            state_ref["arrived"][f"{mod}.{name}"] = state_ref["arrived"].get(f"{mod}.{name}", 0) + 1
    return profile_hook


state_ref: dict = {}


def collect_artifacts(out_dir: Path) -> dict:
    artifacts = {}
    for path in sorted(out_dir.iterdir()) if out_dir.exists() else []:
        if not path.is_file():
            continue
        raw = path.read_bytes()
        entry = {"sha256": sha256_bytes(raw), "size": len(raw)}
        try:
            entry["json"] = json.loads(raw)
        except Exception:
            entry["text"] = raw.decode(errors="replace")
        artifacts[path.name] = entry
    return artifacts


def expected_source_hashes(repo: Path, commit: str) -> tuple[dict, list]:
    expected, failures = {}, []
    for name in MODULES:
        source = f"runtime/d0/{name}.py"
        result = REAL_RUN(["git", "-C", str(repo), "show", f"{commit}:{source}"],
                          capture_output=True)
        if result.returncode != 0:
            failures.append(f"git show failed for {source}")
            continue
        expected[name] = sha256_bytes(result.stdout)
    return expected, failures


def module_identity(mod_dir: Path, expected: dict) -> tuple[dict, list]:
    identity, failures = {}, []
    for name in MODULES:
        try:
            mod = importlib.import_module(name)
        except Exception as exc:
            identity[name] = {"import_error": f"{type(exc).__name__}: {exc}"}
            failures.append(f"{name} import failed")
            continue
        file = Path(getattr(mod, "__file__", "")).resolve()
        spec = getattr(mod, "__spec__", None)
        origin = str(getattr(spec, "origin", ""))
        try:
            sha = sha256_file(file)
        except OSError:
            sha = None
        functions = {}
        for fn_name in ("main", "run_child", "query_group", "restore", "verify",
                        "discover_service_pids", "decode", "build_commands"):
            fn = getattr(mod, fn_name, None)
            if fn is not None and hasattr(fn, "__code__"):
                functions[fn_name] = fn.__code__.co_filename
        identity[name] = {"file": str(file), "origin": origin, "sha256": sha,
                          "functions": functions}
        if sha != expected.get(name):
            failures.append(f"{name} source SHA mismatch")
        if file != (mod_dir / f"{name}.py").resolve():
            failures.append(f"{name} file is not the archived target")
        if Path(origin).resolve() != file:
            failures.append(f"{name} spec.origin mismatch")
        for fn_name, co_file in functions.items():
            if Path(co_file).resolve() != file:
                failures.append(f"{name}.{fn_name} co_filename mismatch")
    return identity, failures


def run_single(args) -> int:
    repo = Path(args.repo).resolve()
    out = {"schema": "d0-regression-case/1", "scenario": args.scenario,
           "target_commit": args.commit, "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "checks": [], "not_reached": [], "production_gap": None}
    tmp = Path(tempfile.mkdtemp(prefix="d0reg-"))
    try:
        archive = tmp / "d0.tar"
        rc = REAL_RUN(["git", "-C", str(repo), "archive", args.commit, "runtime/d0"],
                      stdout=open(archive, "wb")).returncode
        out["git_archive_rc"] = rc
        if rc != 0:
            out["fatal"] = "git archive failed"
            return finish_case(out, args, 1)
        (tmp / "src").mkdir()
        REAL_RUN(["tar", "-xf", str(archive), "-C", str(tmp / "src")])
        mod_dir = tmp / "src" / "runtime" / "d0"
        out["files"] = sorted(p.name for p in mod_dir.glob("*.py"))
        out_dir = tmp / "out"
        scenario = args.scenario
        state = {
            "scenario": scenario, "argv": [], "kills": [], "kill_simulated": [],
            "net": [], "unexpected": [], "arrived": {}, "vfs_miss": [],
            "workspace_writes": [], "label": None, "spawned": False,
            "tree_killed": False, "child_rc": 7 if scenario == "child_nonzero" else 0,
            "npu_workers": [WORKERS[0]] if scenario == "verify_failure" else list(WORKERS),
            "verify_bad": scenario == "verify_failure",
            "vfs_files": {}, "vfs_dirs": {}, "expected": {},
        }
        state_ref.clear()
        state_ref.update(state)
        state["expected"] = expected_commands(out_dir, mod_dir)
        expected_sources, source_failures = expected_source_hashes(repo, args.commit)
        out["expected_source_sha256"] = expected_sources
        out["source_hash_failures"] = source_failures
        install_guard(state)
        if args.scenario == "parse_failure":
            pass
        entry = "d0_run"
        out["entry"] = entry
        sys.path.insert(0, str(mod_dir))
        try:
            run_mod = importlib.import_module(entry)
            out["import_ok"] = True
        except Exception as exc:
            out["import_ok"] = False
            out["import_error"] = f"{type(exc).__name__}: {exc}"
            return finish_case(out, args, 1)
        if entry == "d0_run":
            sys.argv = ["d0_run.py", "--out-dir", str(out_dir)]
        else:
            sys.argv = ["d0_restore.py", "--out", str(out_dir / "restore.json"), "--test-rc", "0"]
        sys.setprofile(profile_hook)
        returned, exc_name = False, None
        ret = None
        try:
            ret = run_mod.main()
            returned = True
        except SystemExit as exc:
            ret, returned = exc.code, True
            exc_name = f"SystemExit({exc.code!r})"
        except Exception as exc:
            ret, exc_name = None, f"{type(exc).__name__}: {exc}"
        finally:
            sys.setprofile(None)
        out.update({"main_returned": returned, "main_return": ret,
                    "main_return_type": type(ret).__name__ if returned else "not_returned",
                    "main_exception": exc_name})
        out["arrived"] = sorted(state["arrived"])
        out["record"] = {"argv": state["argv"], "kills": state["kills"],
                         "kill_simulated": state["kill_simulated"],
                         "net": state["net"], "unexpected": state["unexpected"],
                         "label": state["label"], "vfs_miss": state["vfs_miss"],
                         "workspace_writes": state["workspace_writes"],
                         "proc_before_launch": state["proc_before_launch"]}
        out["artifacts"] = collect_artifacts(out_dir)
        out["out_files"] = sorted(out["artifacts"])
        identity, id_failures = module_identity(mod_dir, expected_sources)
        id_failures = source_failures + id_failures
        out["module_identity"] = identity
        out["module_identity_failures"] = id_failures
        evaluate_case(out, state, args, mod_dir)
        out["failed"] = [c["check"] for c in out["checks"] if not c["ok"]]
        out["case_ok"] = not out["failed"]
        if args.freeze_dir:
            freeze = Path(args.freeze_dir) / scenario
            freeze.mkdir(parents=True, exist_ok=True)
            (freeze / "case.json").write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n")
            dst = freeze / "artifacts"
            dst.mkdir(exist_ok=True)
            for name, entry in out["artifacts"].items():
                blob = (out_dir / name).read_bytes()
                (dst / name).write_bytes(blob)
        return finish_case(out, args, 0 if out["case_ok"] else 1)
    finally:
        if not args.keep_temp:
            shutil.rmtree(tmp, ignore_errors=True)
        else:
            out["temp_dir"] = str(tmp)


def finish_case(out: dict, args, code: int) -> int:
    Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json_out).write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n")
    print(json.dumps({k: out.get(k) for k in ("scenario", "main_return", "main_return_type",
                                              "case_ok", "failed", "production_gap")},
                     ensure_ascii=False, indent=1))
    return code


def add(checks: list, name: str, ok, detail="", category="common") -> None:
    checks.append({"check": name, "ok": bool(ok), "detail": detail, "category": category})


def evaluate_case(out: dict, state: dict, args, mod_dir: Path) -> None:
    checks = out.setdefault("checks", [])
    scenario = args.scenario
    artifacts = out.get("artifacts", {})
    raw = (artifacts.get("raw-rc.json") or {}).get("json") or {}
    restore = (artifacts.get("restore.json") or {}).get("json") or {}
    main_return = out.get("main_return")
    out["test_rc"] = restore.get("test_rc")
    out["child_raw_rc"] = raw.get("child_raw_rc")
    out["wrapper_rc"] = raw.get("wrapper_rc")
    out["restore_ok"] = restore.get("restore_ok")
    out["verify_ok"] = restore.get("verify_ok")
    out["restore_blocked"] = restore.get("restore_blocked")
    out["verify_blocked"] = restore.get("verify_blocked")
    out["main_int"] = isinstance(main_return, int) and not isinstance(main_return, bool)
    out["production_gap"] = None

    add(checks, "main_returned", out.get("main_returned"))
    add(checks, "main_returns_int", out.get("main_int"), f"{out.get('main_return_type')}={main_return!r}")
    add(checks, "no_network", not state["net"], str(state["net"]))
    add(checks, "no_unexpected_argv", not state["unexpected"], str(state["unexpected"]))
    add(checks, "no_workspace_write", not state["workspace_writes"], str(state["workspace_writes"]))
    add(checks, "module_identity", not out.get("module_identity_failures"),
        str(out.get("module_identity_failures")))
    add(checks, "target_commit", out.get("target_commit") == args.commit, out.get("target_commit"))
    add(checks, "wrapper_rc_matches_main",
        out.get("main_int") and out["wrapper_rc"] == main_return,
        f"wrapper={out.get('wrapper_rc')!r} main={main_return!r}")
    add(checks, "restore_test_rc_recorded", "test_rc" in restore, repr(out.get("test_rc")))
    blocked_before_launch = scenario in ("query_unavailable", "cleanup_residue")
    add(checks, "run_child_reached", any(k.endswith(".run_child") for k in out["arrived"]))
    add(checks, "restore_reached",
        any(k.endswith(".restore") for k in out["arrived"]) == (not blocked_before_launch),
        f"blocked_before_launch={blocked_before_launch}")
    add(checks, "query_reached", any(k.endswith(".query_group") for k in out["arrived"]))
    add(checks, "root_child_by_run_child",
        "root-child.json" in artifacts and "raw-rc.json" in artifacts and "tree-after.json" in artifacts,
        str(sorted(artifacts)))
    if blocked_before_launch:
        add(checks, "launch_not_attempted",
            not any(item["kind"] in ("launch_service", "wait") for item in state["argv"]),
            str([item["kind"] for item in state["argv"]]))
        add(checks, "no_dynamic_label", state["label"] is None, repr(state["label"]))
    else:
        add(checks, "label_dynamic",
            bool(state["label"] and LABEL_RE.match(state["label"])),
            repr(state["label"]))
    if scenario == HAPPY:
        add(checks, "no_vfs_miss", not state["vfs_miss"], str(state["vfs_miss"]))
    else:
        add(checks, "no_unexpected_vfs_miss",
            all(p == "/proc/sys/kernel/random/boot_id" for p in state["vfs_miss"]),
            str(state["vfs_miss"]))

    child_pid = ((artifacts.get("root-child.json") or {}).get("json") or {}).get("child_pid")
    add(checks, "root_child_pid_separate", child_pid == CHILD_PID
        and child_pid != ROOT_PID and child_pid != os.getpid(),
        f"child={child_pid} service={ROOT_PID} harness={os.getpid()}")
    add(checks, "proc_before_empty",
        state["proc_before_launch"] in ([], [str(os.getpid())]),
        str(state["proc_before_launch"]))

    expected = state["expected"]
    kinds = [item["kind"] for item in state["argv"]]
    if scenario == HAPPY:
        counts = {kind: kinds.count(kind) for kind in expected}
        counts["launch_service"] = kinds.count("launch_service")
        add(checks, "argv_counts", counts.get("child") == 1 and counts.get("ps") == 2
            and all(counts.get(k) == 1 for k in ("launch_service", "wait", "health",
                                                  "workers", "text", "image")), str(counts))
        text_args = next((x["argv"] for x in state["argv"] if x["kind"] == "text"), None)
        image_args = next((x["argv"] for x in state["argv"] if x["kind"] == "image"), None)
        add(checks, "text_body_exact", text_args == expected["text"], repr(text_args))
        add(checks, "image_body_exact", image_args == expected["image"], repr(image_args))
        add(checks, "main_zero", main_return == 0, repr(main_return))
        add(checks, "test_rc_zero", out["test_rc"] == 0, repr(out["test_rc"]))
        add(checks, "child_raw_rc_zero", out["child_raw_rc"] == 0, repr(out["child_raw_rc"]))
        add(checks, "wrapper_rc_zero", out["wrapper_rc"] == 0, repr(out["wrapper_rc"]))
        add(checks, "restore_ok_true", out["restore_ok"] is True, repr(out["restore_ok"]))
        add(checks, "verify_ok_true", out["verify_ok"] is True, repr(out["verify_ok"]))
        for stage in ("health", "workers", "text", "image"):
            entry = restore.get(stage) or {}
            add(checks, f"verify_{stage}_ok", entry.get("ok") is True,
                f"rc={entry.get('rc')!r} workers={entry.get('workers')!r}")
        add(checks, "service_label_match", state["label"] == restore.get("service_label")
            and state["label"] != "final-default-s1", str(restore.get("service_label")))
        for rank in (0, 1):
            parsed = ((artifacts.get(f"result-rank{rank}-parsed.json") or {}).get("json") or {})
            observations = parsed.get("observations") or []
            claims = parsed.get("claim_ranges") or []
            add(checks, f"rank{rank}_parser_semantics",
                len(observations) >= 2 and len(claims) == 1
                and parsed.get("observed_coverage") is True
                and not parsed.get("identity_rejected"),
                f"obs={len(observations)} claims={len(claims)} "
                f"covered={parsed.get('observed_coverage')}")
        add(checks, "raw_rc_happy",
            raw.get("child_raw_rc") == 0 and raw.get("wrapper_rc") == 0
            and raw.get("timed_out") is False, json.dumps(raw))
    else:
        expected_rc = NONZERO_SCENARIOS[scenario]
        dut_nonzero = out["main_int"] and main_return != 0
        exact = expected_rc is None or main_return == expected_rc
        add(checks, "dut_nonzero", dut_nonzero and exact,
            f"main={main_return!r} expected={expected_rc!r}")
        add(checks, "restore_recorded", "restore.json" in artifacts)
        add(checks, "rc_separated",
            "test_rc" in out and "wrapper_rc" in out and "restore_ok" in out
            and "verify_ok" in out, json.dumps({k: out.get(k) for k in
                                                ("test_rc", "wrapper_rc", "restore_ok", "verify_ok")}))
        if not (dut_nonzero and exact):
            out["production_gap"] = {
                "scenario": scenario, "main_return": main_return,
                "expected": expected_rc if expected_rc is not None else "non-zero",
                "restore_ok": out["restore_ok"], "verify_ok": out["verify_ok"],
                "restore_blocked": out["restore_blocked"],
                "verify_blocked": out["verify_blocked"],
            }
        if scenario in ("root_changed", "boot_missing", "boot_empty"):
            add(checks, "identity_rejected", out["restore_ok"] is False
                and bool(((restore.get("service_discovery") or {}).get("error"))),
                str((restore.get("service_discovery") or {}).get("error")))
            add(checks, "verify_not_entered", not any(k.endswith(".verify") for k in out["arrived"]))
        if scenario == "timeout":
            add(checks, "timeout_recorded", raw.get("timed_out") is True
                and raw.get("wrapper_rc") == 2 and out["test_rc"] == 2, json.dumps(raw))
            add(checks, "trace_unavailable", "trace-unavailable.json" in artifacts)
            add(checks, "kill_simulated_only", bool(state["kill_simulated"])
                and all(k[0] in (ROOT_PID, "pgid") for k in state["kill_simulated"]))
        if scenario == "child_nonzero":
            add(checks, "child_rc_recorded", raw.get("child_raw_rc") == 7
                and raw.get("wrapper_rc") == 7 and out["test_rc"] == 7, json.dumps(raw))
        if scenario == "query_unavailable":
            add(checks, "query_unavailable_branch",
                (restore.get("prelaunch_group") or {}).get("survivor_check") == "unavailable"
                or "unavailable" in str(out["restore_blocked"]),
                json.dumps(restore.get("prelaunch_group")))
        if scenario == "cleanup_residue":
            members = (restore.get("prelaunch_group") or {}).get("members")
            add(checks, "residue_branch", isinstance(members, list) and len(members) > 0,
                json.dumps(restore.get("prelaunch_group")))
        if scenario == "verify_failure":
            stage_bad = [s for s in ("health", "workers", "text", "image")
                         if (restore.get(s) or {}).get("ok") is not True]
            add(checks, "verify_failed_branch", out["verify_ok"] is False and bool(stage_bad),
                f"verify_ok={out['verify_ok']} bad={stage_bad}")
        if scenario == "parse_failure":
            add(checks, "parse_failure_recorded", "parse-failure.json" in artifacts
                and raw.get("wrapper_rc") == 5, json.dumps(raw))
            add(checks, "parsed_absent", not any(name.endswith("-parsed.json")
                                                 for name in artifacts))


def run_suite(args) -> int:
    suite_dir = Path(tempfile.mkdtemp(prefix="d0suite-"))
    cases = []
    try:
        for scenario in SCENARIOS:
            case_json = suite_dir / f"{scenario}.json"
            cmd = [sys.executable, str(Path(__file__).resolve()), "--single",
                   "--scenario", scenario, "--repo", str(Path(args.repo).resolve()),
                   "--commit", args.commit, "--json-out", str(case_json)]
            if args.freeze_dir:
                cmd += ["--freeze-dir", str(Path(args.freeze_dir).resolve())]
            if args.keep_temp:
                cmd += ["--keep-temp"]
            rc = REAL_RUN(cmd).returncode
            data = json.loads(case_json.read_text()) if case_json.is_file() else {
                "scenario": scenario, "case_ok": False, "fatal": "case json missing", "rc": rc}
            data["case_rc"] = rc
            cases.append(data)
        summary = {
            "schema": "d0-regression-suite/1",
            "target_commit": args.commit,
            "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "cases": [{k: case.get(k) for k in
                       ("scenario", "case_ok", "case_rc", "main_return", "main_return_type",
                        "test_rc", "wrapper_rc", "restore_ok", "verify_ok",
                        "restore_blocked", "verify_blocked", "production_gap", "failed")}
                      for case in cases],
            "case_ok": all(case.get("case_ok") for case in cases),
            "production_gaps": [case["production_gap"] for case in cases
                                if case.get("production_gap")],
        }
        summary["suite_rc"] = 0 if summary["case_ok"] else 1
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(json.dumps(summary, ensure_ascii=False, indent=1) + "\n")
        if args.freeze_dir:
            Path(args.freeze_dir).mkdir(parents=True, exist_ok=True)
            (Path(args.freeze_dir) / "suite.json").write_text(
                json.dumps(summary, ensure_ascii=False, indent=1) + "\n")
        print(json.dumps(summary, ensure_ascii=False, indent=1))
        return summary["suite_rc"]
    finally:
        shutil.rmtree(suite_dir, ignore_errors=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--commit", default=TARGET)
    ap.add_argument("--json-out", required=True)
    ap.add_argument("--freeze-dir")
    ap.add_argument("--keep-temp", action="store_true")
    ap.add_argument("--single", action="store_true")
    ap.add_argument("--scenario", choices=SCENARIOS, default=HAPPY)
    a = ap.parse_args()
    if a.single:
        return run_single(a)
    return run_suite(a)


if __name__ == "__main__":
    raise SystemExit(main())
