#!/usr/bin/env python3
"""CPU tests for the joint-head window wrapper.

The real Window methods are kept.  Only external boundaries are replaced:
a temporary /proc tree (module PROC_ROOT), subprocess.run/Popen, os.kill/
killpg, the localhost port helper and installer command execution.  No torch,
torch_npu, service or production wrapper function is imported or replaced.
"""

from __future__ import annotations

import importlib.util
import io
import json
import shutil
import socket
import subprocess
import sys
import tempfile
import types
from contextlib import redirect_stdout
from pathlib import Path

HERE = Path(__file__).resolve().parent
WRAPPER = HERE / "joint_fused_attention_window.py"
PREIMAGE = HERE / "joint_single_field" / "clef_joint_schema.py"
POSTIMAGE = HERE / "joint_fused_attention" / "clef_joint_schema.py"


def load():
    spec = importlib.util.spec_from_file_location("joint_fused_window", WRAPPER)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def put_proc(root: Path, pid: int, state="S", ppid=1, pgid=None,
             start="100", cmdline="python -m vllm.entrypoints.cli.main serve --port 18143"):
    pgid = pid if pgid is None else pgid
    directory = root / str(pid)
    directory.mkdir(parents=True, exist_ok=True)
    fields = ["0"] * 20
    fields[0] = state
    fields[1] = str(ppid)
    fields[2] = str(pgid)
    fields[19] = start
    (directory / "stat").write_text(f"{pid} (python) " + " ".join(fields) + "\n")
    (directory / "cmdline").write_text(cmdline.replace(" ", "\0") + "\0")
    (directory / "environ").write_bytes(b"CLEF_CONFIG=s1\0PATH=/bin\0")


def main() -> int:
    mod = load()
    failures = []

    def check(name, condition, detail=""):
        print(("ok   " if condition else "FAIL ") + name + (f" {detail}" if detail else ""))
        if not condition:
            failures.append(name)

    env = mod.parse_environ(b"A=1\0SECRET=x\0CLEF_CONFIG=s1\0EMPTY=\0")
    check("parse_environ", env == {"A": "1", "SECRET": "x", "CLEF_CONFIG": "s1", "EMPTY": ""})
    selected = mod.selected_env({"CLEF_A": "1", "ASCEND_B": "2", "PATH": "p",
                                 "SECRET": "x", "LD_LIBRARY_PATH": "l"})
    check("selected_env filters unrelated", "SECRET" not in selected
          and selected["CLEF_A"] == "1" and selected["PATH"] == "p")

    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind(("127.0.0.1", 0))
    port = listener.getsockname()[1]
    listener.listen(1)
    check("port_open detects listener", mod.port_open(port) == "open", str(port))
    listener.close()
    check("port_open detects refusal", mod.port_open(port) == "closed")

    with tempfile.TemporaryDirectory(prefix="window-test-") as tmp:
        tmp = Path(tmp)
        proc_root = tmp / "proc"
        (proc_root / "sys/kernel/random").mkdir(parents=True)
        (proc_root / "sys/kernel/random/boot_id").write_text("BOOT\n")
        put_proc(proc_root, 321, start="555")
        original_proc = mod.PROC_ROOT
        mod.PROC_ROOT = proc_root
        saved = [mod.proc_identity(321)]
        check("proc_identity reads fake proc", saved[0]["pid"] == 321
              and saved[0]["state"] == "S")

        class FakeProc:
            pid = 321
            returncode = None
            def poll(self):
                return self.returncode
            def wait(self, timeout=None):
                self.returncode = -9
                shutil.rmtree(proc_root / "321", ignore_errors=True)
                return -9

        original_killpg = mod.os.killpg
        original_kill = mod.os.kill
        mod.os.killpg = lambda pgid, sig: shutil.rmtree(proc_root / "321", ignore_errors=True)
        mod.os.kill = lambda pid, sig: shutil.rmtree(proc_root / str(pid), ignore_errors=True)
        window = object.__new__(mod.Window)
        cleanup = window._probe_cleanup(FakeProc(), saved, True)
        check("probe timeout reaps and clears", cleanup["rc"] == -9
              and cleanup["reaped"] and not cleanup["residual"]
              and not cleanup["unknown"], str(cleanup))

        put_proc(proc_root, 322, state="Z", start="556")
        zombie_saved = [mod.proc_identity(322)]
        cleanup = window._probe_cleanup(FakeProc(), zombie_saved, False)
        check("zombie is not residual", cleanup["residual"] == []
              and 322 in cleanup["zombies"], str(cleanup))
        mod.PROC_ROOT = original_proc
        mod.os.killpg = original_killpg
        mod.os.kill = original_kill

        # real probe supervision with a fake Popen and fake proc tree
        probe_root = tmp / "probe-proc"
        (probe_root / "sys/kernel/random").mkdir(parents=True)
        (probe_root / "sys/kernel/random/boot_id").write_text("BOOT\n")
        original_proc = mod.PROC_ROOT
        mod.PROC_ROOT = probe_root
        stage = tmp / "stage"
        (stage / "runtime").mkdir(parents=True)
        (stage / mod.PROBE_REL).write_text("")
        run_dir = tmp / "run"
        run_dir.mkdir()
        window = object.__new__(mod.Window)
        window.args = types.SimpleNamespace(
            python="python", tree=tmp, device=0, probe_timeout=1.0,
            candidate_opp_prepend="", candidate_lib_prepend="")
        window.stage = stage
        window.run_dir = run_dir
        window.prod_env = {"PATH": "/bin"}
        window.state = {"steps": {}}

        class FakePopen:
            def __init__(self, cmd, env=None, stdout=None, stderr=None,
                         start_new_session=False):
                self.pid = 401
                self.returncode = None
                self.timed = False
                output = Path(cmd[cmd.index("--output") + 1])
                output.write_text(json.dumps({
                    "interface": {"ok": True}, "candidate_error": None,
                    "delta": {"device_median_ms": -0.4}}))
                put_proc(probe_root, 401, start="777")
            def poll(self):
                return self.returncode
            def wait(self, timeout=None):
                if not self.timed:
                    self.timed = True
                    raise subprocess.TimeoutExpired("probe", timeout)
                self.returncode = -9
                shutil.rmtree(probe_root / "401", ignore_errors=True)
                return -9

        original_popen = mod.subprocess.Popen
        original_killpg = mod.os.killpg
        original_members = mod.group_members
        mod.subprocess.Popen = FakePopen
        mod.os.killpg = lambda pgid, sig: shutil.rmtree(probe_root / "401", ignore_errors=True)
        try:
            try:
                window.probe()
            except mod.StepError as exc:
                detail = exc.detail
            else:
                detail = {}
            check("real probe timeout keeps raw record", detail["probe_record"]["timed_out"] is True
                  and detail["probe_record"]["cleanup"]["reaped"] is True
                  and (run_dir / "probe/record.json").is_file(), str(detail)[:300])
            check("clean timeout clears residual flag", window.probe_residual is False)

            run_dir2 = tmp / "run2"
            run_dir2.mkdir()
            window2 = object.__new__(mod.Window)
            window2.args = window.args
            window2.stage = stage
            window2.run_dir = run_dir2
            window2.prod_env = {"PATH": "/bin"}
            window2.state = {"steps": {}}
            mod.group_members = lambda pgid: {999: {"pid": 999, "state": "S",
                                                    "starttime": "1", "boot_id": "BOOT",
                                                    "pgid": pgid}}
            try:
                try:
                    window2.probe()
                except mod.StepError as exc:
                    detail = exc.detail
                else:
                    detail = {}
                check("unknown group keeps residual flag",
                      window2.probe_residual is True
                      and detail["probe_record"]["cleanup"]["unknown"],
                      str(detail)[:300])
            finally:
                pass
        finally:
            mod.subprocess.Popen = original_popen
            mod.os.killpg = original_killpg
            mod.group_members = original_members
            mod.PROC_ROOT = original_proc

        # stop verification refuses a live saved root, then accepts gone
        check_root = tmp / "stop-proc"
        (check_root / "sys/kernel/random").mkdir(parents=True)
        (check_root / "sys/kernel/random/boot_id").write_text("BOOT\n")
        put_proc(check_root, 501, start="900")
        original_proc = mod.PROC_ROOT
        mod.PROC_ROOT = check_root
        run_dir = tmp / "stop-run"
        run_dir.mkdir()
        window = object.__new__(mod.Window)
        window.identity = {"root": mod.proc_identity(501), "descendants": []}
        window.run_dir = run_dir
        window.state = {"steps": {}}
        window.flush = lambda: None
        ok, detail = window.service_stopped_verified()
        check("live saved service blocks restore", ok is False and detail["alive"], str(detail))
        shutil.rmtree(check_root / "501", ignore_errors=True)
        ok, detail = window.service_stopped_verified()
        check("gone saved service passes", ok is True, str(detail))
        mod.PROC_ROOT = original_proc

        # pure preflight/read-only failure must not revert an existing postimage
        tree = tmp / "tree"
        target = tree / "vllm/model_executor/models/clef_joint_schema.py"
        target.parent.mkdir(parents=True)
        shutil.copy2(POSTIMAGE, target)
        before = mod.sha256(target)
        window = object.__new__(mod.Window)
        window.args = types.SimpleNamespace(tree=tree, port=18143)
        window.run_dir = tmp / "readonly"
        window.state = {"steps": {}}
        window.candidate_started = False
        window.applied = False
        window.stopped = False
        window.restored = False
        window.probe_residual = False
        window.overall_failed = False
        window.flush()
        window.finish()
        check("read-only preflight never mutates", mod.sha256(target) == before
              and window.state["steps"]["finish"]["action"] == "read_only_noop")

        # applied + residual: real revert (fake installer command), then block
        tree = tmp / "tree2"
        target = tree / "vllm/model_executor/models/clef_joint_schema.py"
        target.parent.mkdir(parents=True)
        shutil.copy2(POSTIMAGE, target)
        stage = tmp / "stage2"
        (stage / "runtime").mkdir(parents=True)
        (stage / mod.INSTALLER_REL).write_text("")
        window = object.__new__(mod.Window)
        window.args = types.SimpleNamespace(tree=tree, port=18143, python="python")
        window.run_dir = tmp / "applied"
        window.run_dir.mkdir()
        window.stage = stage
        window.state = {"steps": {}}
        window.candidate_started = True
        window.applied = True
        window.stopped = True
        window.stop_attempted = True
        window.restored = False
        window.probe_residual = True
        window.overall_failed = False
        window.identity = None
        original_run = mod.subprocess.run
        def fake_run(cmd, **kwargs):
            if "--revert" in cmd:
                shutil.copy2(PREIMAGE, target)
            return types.SimpleNamespace(returncode=0, stdout=json.dumps(
                {"state_after": "preimage"}), stderr="")
        mod.subprocess.run = fake_run
        try:
            window.finish()
        finally:
            mod.subprocess.run = original_run
        check("applied residual reverts and blocks restore",
              mod.sha256(target) == mod.PREIMAGE_SHA
              and window.state["steps"]["restore"]["action"] == "blocked_probe_residual")

    # real main refuses an existing run directory without writing into it
    with tempfile.TemporaryDirectory(prefix="window-main-") as tmp:
        existing = Path(tmp) / "exists"
        existing.mkdir()
        argv = sys.argv
        sys.argv = ["window", "--run-dir", str(existing), "--repo", tmp]
        try:
            with redirect_stdout(io.StringIO()):
                code = mod.main()
        finally:
            sys.argv = argv
        check("main rejects existing run dir", code == 1
              and not (existing / "window.json").exists())

    # source failure must not be hidden by env -0 returning zero
    with tempfile.TemporaryDirectory(prefix="window-source-") as tmp:
        window = object.__new__(mod.Window)
        window.args = types.SimpleNamespace(activate="/missing/activate.sh")
        window.state = {"preflight": {}}
        original_run = mod.subprocess.run
        mod.subprocess.run = lambda *a, **kw: types.SimpleNamespace(
            returncode=3, stdout=b"", stderr=b"")
        try:
            try:
                window._source_env()
            except mod.StepError:
                source_failed = True
            else:
                source_failed = False
        finally:
            mod.subprocess.run = original_run
        check("source failure raises", source_failed)

    # real probe aggregation regression (the wall-shadow first failure)
    probe_spec = importlib.util.spec_from_file_location(
        "joint_probe_test", HERE / "joint_fused_attention_probe.py")
    probe_mod = importlib.util.module_from_spec(probe_spec)
    assert probe_spec.loader is not None
    probe_spec.loader.exec_module(probe_mod)
    arms = probe_mod.summarize_arms({"0": [1.0, 2.0], "1": [3.0, 4.0]},
                                    {"0": [5.0, 6.0], "1": [7.0, 8.0]})
    check("probe round aggregation", arms["0"]["device_samples"] == [1.0, 2.0]
          and arms["0"]["wall_samples"] == [5.0, 6.0]
          and arms["1"]["samples"][1]["wall_ms"] == 8.0, str(arms))
    check("probe writes partial per round",
          "partial.json" in (HERE / "joint_fused_attention_probe.py").read_text())

    # restore modules must be staged/verified before stop
    with tempfile.TemporaryDirectory(prefix="window-d0-") as tmp:
        window = object.__new__(mod.Window)
        window.stage = Path(tmp) / "stage"
        window.state = {"preflight": {}}
        window._stage_d0(HERE.parent)
        staged = sorted(path.name for path in (window.stage / "runtime/d0").iterdir())
        check("stage six restore modules", staged == sorted(mod.D0_MODULES),
              str(staged))
        empty = Path(tmp) / "empty"
        empty.mkdir()
        window2 = object.__new__(mod.Window)
        window2.stage = Path(tmp) / "stage2"
        window2.state = {"preflight": {}}
        try:
            window2._stage_d0(empty)
        except mod.StepError:
            missing_blocked = True
        else:
            missing_blocked = False
        check("missing restore module blocks preflight", missing_blocked)

    print(json.dumps({"checks": 21, "failed": failures, "ok": not failures}))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
