#!/usr/bin/env python3
"""CPU tests for the joint-head window wrapper's non-device logic.

Only external boundaries are exercised: a localhost listening socket, the
environment parser/selector and the finally-state machine with a fake step
callback.  No production wrapper function, torch, torch_npu or service is
imported or replaced.
"""

from __future__ import annotations

import importlib.util
import io
import json
import socket
import sys
import tempfile
import types
from contextlib import redirect_stdout
from pathlib import Path

HERE = Path(__file__).resolve().parent
WRAPPER = HERE / "joint_fused_attention_window.py"


def load():
    spec = importlib.util.spec_from_file_location("joint_fused_window", WRAPPER)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


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

    # step preserves the real RC and detail instead of coercing to 1
    with tempfile.TemporaryDirectory(prefix="window-test-") as tmp:
        window = object.__new__(mod.Window)
        window.state = {"steps": {}}
        window.run_dir = Path(tmp)
        window.overall_failed = False
        window.flush = lambda: None
        def fail():
            raise mod.StepError("boom", rc=7, detail={"probe_record": {"rc": 7}})
        window.fail = fail
        entry = window.step("fail")
        check("step preserves RC", entry["rc"] == 7
              and entry["probe_record"]["rc"] == 7)

        # flush with Path in args must be JSON serializable
        window.state["args"] = {"tree": Path("/tmp/x")}
        mod.Window.flush(window)
        check("flush serializes Path args", (Path(tmp) / "window.json").is_file())

        def make(applied, stopped, restored, port_state, sha_state, health=True,
                 revert_ok=True):
            window = object.__new__(mod.Window)
            tree = Path(tmp) / "tree"
            (tree / "vllm/model_executor/models").mkdir(parents=True, exist_ok=True)
            target = tree / "vllm/model_executor/models/clef_joint_schema.py"
            target.write_text("x")
            window.args = types.SimpleNamespace(port=18143, tree=tree)
            window.run_dir = Path(tmp)
            window.state = {"steps": {}}
            window.applied = applied
            window.stopped = stopped
            window.stop_attempted = stopped
            window.restored = restored
            window.probe_residual = False
            window.overall_failed = False
            window.identity = {"root": {"pid": 1}}
            calls = []
            def step(name, required=True):
                calls.append(name)
                if name == "revert" and revert_ok:
                    sha_state["value"] = mod.PREIMAGE_SHA
                return {"rc": 0, "ok": revert_ok if name == "revert" else True}
            window.step = step
            window.flush = lambda: None
            window.verify_existing = lambda: (health, "health ready")
            return window, calls, sha_state

        original_port = mod.port_open
        original_sha = mod.sha256
        mod.port_open = lambda port: "open"
        state = {"value": mod.PREIMAGE_SHA}
        mod.sha256 = lambda path: state["value"]
        window, calls, state = make(False, True, False, "open", state)
        window.finish()
        check("stop+apply failure restores", calls == ["restore"], str(calls))
        state = {"value": mod.POSTIMAGE_SHA}
        window, calls, state = make(True, True, False, "open", state)
        window.finish()
        check("applied failure reverts then restores", calls == ["revert", "restore"], str(calls))
        state = {"value": mod.PREIMAGE_SHA}
        window, calls, state = make(False, False, False, "open", state)
        window.finish()
        check("stop failure verifies existing service", calls == []
              and window.state["steps"]["restore"]["action"] == "existing_service_verified")
        window, calls, state = make(False, False, False, "open", state,
                                     health=False)
        window.finish()
        check("unhealthy existing service fails", calls == []
              and window.overall_failed is True)
        state = {"value": mod.PREIMAGE_SHA}
        window, calls, state = make(False, False, False, "closed", state)
        mod.port_open = lambda port: "closed"
        window.finish()
        check("stop failure with no service fails", calls == []
              and window.overall_failed is True)
        state = {"value": mod.POSTIMAGE_SHA}
        window, calls, state = make(True, True, False, "open", state)
        window.probe_residual = True
        window.finish()
        check("probe residual blocks restore", calls == ["revert"]
              and window.state["steps"]["restore"]["action"] == "blocked_probe_residual")
        state = {"value": "unknown"}
        window, calls, state = make(False, True, False, "open", state)
        window.finish()
        check("unknown source blocks restore", calls == []
              and window.state["steps"]["restore"]["action"] == "blocked_source_not_preimage")
        state = {"value": mod.POSTIMAGE_SHA}
        window, calls, state = make(True, True, False, "open", state, revert_ok=False)
        window.finish()
        check("failed revert blocks restore", calls == ["revert"]
              and window.state["steps"]["restore"]["action"] == "blocked_source_not_preimage")
        state = {"value": mod.PREIMAGE_SHA}
        window, calls, state = make(False, False, False, "closed", state)
        window.stop_attempted = True
        mod.port_open = lambda port: "closed"
        window.finish()
        check("partial stop with no service restores", calls == ["restore"], str(calls))
        mod.port_open = original_port
        mod.sha256 = original_sha

    # real main refuses an existing run directory without writing into it
    with tempfile.TemporaryDirectory(prefix="window-main-") as tmp:
        existing = Path(tmp) / "exists"
        existing.mkdir()
        argv = sys.argv
        sys.argv = ["window", "--run-dir", str(existing), "--repo", tmp]
        try:
            buffer = io.StringIO()
            with redirect_stdout(buffer):
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

    # probe cleanup compares saved identity and reports unknown group members
    with tempfile.TemporaryDirectory(prefix="window-probe-") as tmp:
        class FakeProc:
            pid = 123
            returncode = -9
            def poll(self):
                return -9
        window = object.__new__(mod.Window)
        saved = [{"pid": 123, "starttime": "1", "boot_id": "b", "pgid": 123,
                  "cmdline": "x"}]
        original_identity = mod.proc_identity
        original_members = mod.group_members
        original_killpg = mod.os.killpg
        original_kill = mod.os.kill
        calls = []
        alive = {"value": True}
        mod.proc_identity = lambda pid: saved[0] if pid == 123 and alive["value"] else None
        mod.group_members = lambda pgid: {}
        def killpg(pgid, sig):
            calls.append(("pgid", pgid))
            alive["value"] = False
        def kill(pid, sig):
            calls.append(("pid", pid))
            alive["value"] = False
        mod.os.killpg = killpg
        mod.os.kill = kill
        cleanup = window._probe_cleanup(FakeProc(), saved, True)
        check("probe timeout killpg exact", ("pgid", 123) in calls
              and not cleanup["unknown"] and not cleanup["residual"], str(cleanup))
        mod.proc_identity = lambda pid: None
        mod.group_members = lambda pgid: {999: {"pid": 999, "starttime": "2",
                                               "boot_id": "b", "pgid": pgid}}
        cleanup = window._probe_cleanup(FakeProc(), [], False)
        check("unknown group member blocks", len(cleanup["unknown"]) == 1, str(cleanup))
        mod.proc_identity = original_identity
        mod.group_members = original_members
        mod.os.killpg = original_killpg
        mod.os.kill = original_kill

    print(json.dumps({"checks": 18, "failed": failures, "ok": not failures}))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
