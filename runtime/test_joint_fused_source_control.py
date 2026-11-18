#!/usr/bin/env python3
"""CPU tests for the independent f810/aa3e source-control runner."""
from __future__ import annotations

import importlib.util
import io
import json
import os
import tempfile
import types
from contextlib import redirect_stdout
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNNER = HERE / "joint_fused_source_control.py"


def load():
    spec = importlib.util.spec_from_file_location("joint_fused_source_control", RUNNER)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def main() -> int:
    mod = load()
    checks = []

    def check(name, ok, detail=""):
        print(("ok   " if ok else "FAIL ") + name + (f" {detail}" if detail else ""))
        checks.append((name, bool(ok)))

    check("ABBA source order", mod.ORDERS == [
        [("f810", 0), ("aa3e", 1)],
        [("aa3e", 1), ("f810", 0)],
        [("aa3e", 1), ("f810", 0)],
        [("f810", 0), ("aa3e", 1)]])

    runner = object.__new__(mod.SourceControl)
    runner.prod_env = {"CLEF_JOINT_FUSED_ATTN": "9",
                       "CLEF_JOINT_ALLOWED_POSTIMAGE_SHA": "stale",
                       "CLEF_JOINT_SMALLQ": "1", "PATH": "/bin"}
    f810_env = runner.arm_environment("f810", 0)
    aa3e_env = runner.arm_environment("aa3e", 1)
    check("f810 arm env: flag0, no allow",
          f810_env["CLEF_JOINT_FUSED_ATTN"] == "0"
          and "CLEF_JOINT_ALLOWED_POSTIMAGE_SHA" not in f810_env
          and "CLEF_JOINT_SMALLQ" not in f810_env
          and f810_env["CLEF_LAUNCH_CONFIG"] == "s1")
    check("aa3e arm env: flag1, exact allow",
          aa3e_env["CLEF_JOINT_FUSED_ATTN"] == "1"
          and aa3e_env["CLEF_JOINT_ALLOWED_POSTIMAGE_SHA"] == mod.AA3E
          and aa3e_env["CLEF_LAUNCH_CONFIG"] == "s1")

    with tempfile.TemporaryDirectory(prefix="source-control-") as tmp:
        tmp = Path(tmp)
        tree = tmp / "tree"
        target = tree / "vllm/model_executor/models/clef_joint_schema.py"
        target.parent.mkdir(parents=True)
        target.write_text("f810")
        calls = []
        manifest_seen = []
        shas = {str(target): mod.F810}

        def fake_call(cmd, timeout=120, env=None):
            calls.append(cmd)
            action = "check" if "--check" in cmd else "apply" if "--apply" in cmd else "revert"
            if action == "check":
                return types.SimpleNamespace(stdout=json.dumps(
                    {"state_after": "preimage" if shas[str(target)] == mod.F810 else "postimage"}))
            manifest = Path(cmd[cmd.index("--manifest") + 1])
            manifest.write_text(json.dumps({"state_after": "postimage" if action == "apply" else "preimage"}))
            manifest_seen.append(str(manifest))
            target.write_text("aa3e" if action == "apply" else "f810")
            shas[str(target)] = mod.AA3E if action == "apply" else mod.F810
            return types.SimpleNamespace(stdout="{}")

        runner.args = types.SimpleNamespace(python="python", tree=str(tree), port=18143)
        runner.run_dir = tmp / "run"
        runner.run_dir.mkdir()
        runner.state = {}
        runner.source_switches = []
        runner.active_source = None
        runner.win = types.SimpleNamespace(
            port_open=lambda port: "closed",
            sha256=lambda path: shas[str(path)])
        runner.call = fake_call
        rec = runner.switch_source("f810", "pre")
        check("f810 no-op source switch", rec["actual_sha256"] == mod.F810
              and rec["manifest"] is None and runner.active_source == "f810")
        rec = runner.switch_source("aa3e", "on")
        check("aa3e apply switch with owned manifest",
              rec["actual_sha256"] == mod.AA3E and rec["manifest"] in manifest_seen
              and Path(rec["manifest"]).is_file())
        rec = runner.switch_source("f810", "back")
        check("f810 revert switch", rec["actual_sha256"] == mod.F810
              and rec["manifest"] in manifest_seen)
        try:
            runner.switch_source("unknown", "bad")
        except mod.base.AbError:
            blocked = True
        else:
            blocked = False
        check("unknown source rejected", blocked)
        runner.win.port_open = lambda port: "open"
        try:
            runner.switch_source("aa3e", "open")
        except mod.base.AbError:
            active = True
        else:
            active = False
        check("source switch requires stopped service", active)

    with tempfile.TemporaryDirectory(prefix="source-main-") as tmp:
        tmp = Path(tmp)
        existing = tmp / "exists"
        existing.mkdir()
        args = ["--run-dir", str(existing), "--repo", str(HERE.parent),
                "--tree", str(tmp / "tree")]
        with redirect_stdout(io.StringIO()):
            rc = mod.main(args)
        check("main rejects existing run dir", rc == 1)

        run = tmp / "run"
        tree = tmp / "tree2"
        target = tree / "vllm/model_executor/models/clef_joint_schema.py"
        target.parent.mkdir(parents=True)
        target.write_text("wrong")
        args = ["--run-dir", str(run), "--repo", str(HERE.parent),
                "--tree", str(tree)]
        with redirect_stdout(io.StringIO()):
            rc = mod.main(args)
        state = json.loads((run / "ab.json").read_text())
        check("main rejects non-f810 initial source",
              rc == 1 and "requires f810" in state.get("error", "")
              and "restore" not in state.get("steps", {}))

    # Real main: target f810 + non-closed port must block before restore.
    def run_main_port(port_state):
        with tempfile.TemporaryDirectory(prefix="source-port-") as tmp:
            tmp = Path(tmp)
            tree = tmp / "tree"
            target = tree / "vllm/model_executor/models/clef_joint_schema.py"
            target.parent.mkdir(parents=True)
            target.write_text("f810")
            service_root = tmp / "service"
            guard = service_root / "runtime" / "apply_joint_single_field_patch.py"
            guard.parent.mkdir(parents=True)
            guard.write_text("guard")
            class FakeWin:
                def sha256(self, path):
                    name = Path(path).name
                    return {target.name: mod.F810, guard.name: mod.base.SERVICE_CHECK_PREIMAGE_SHA,
                            "d0_stop.py": mod.base.STOP_SHA,
                            "activate_env.sh": mod.base.ACTIVATE_SHA,
                            "launch.sh": mod.base.LAUNCH_SHA,
                            "wait.sh": mod.base.WAIT_SHA}.get(name, "0" * 64)
                def port_open(self, port):
                    return port_state
            original_load = mod.base.load_window
            original_root = mod.base.SERVICE_ROOT
            original = {}
            patches = {
                "stage_d0": lambda self: None,
                "stage_candidate": lambda self: None,
                "current_service": lambda self: ({"root": {"pid": 1}, "descendants": []}, {}),
                "stop_tree": lambda self, identity, label: {"stopped": True},
                "apply_service_check": lambda self: setattr(self, "service_check_applied", True),
                "switch_source": lambda self, source, tag: {"target": source},
                "input_bodies": lambda self: {},
                "run_arm": lambda self, flag, label: {
                    "label": label, "provenance": {}, "samples": {}, "error": None,
                    "stop": {}, "stop_error": None},
                "revert_service_check": lambda self: setattr(self, "service_check_applied", False),
                "restore_prod": lambda self: (_ for _ in ()).throw(
                    AssertionError("restore must not launch")),
            }
            for name, value in patches.items():
                original[name] = getattr(mod.SourceControl, name)
                setattr(mod.SourceControl, name, value)
            mod.base.load_window = lambda: FakeWin()
            mod.base.SERVICE_ROOT = str(service_root)
            try:
                run = tmp / "run"
                with redirect_stdout(io.StringIO()):
                    rc = mod.main(["--run-dir", str(run), "--repo", str(HERE.parent),
                                   "--tree", str(tree)])
                state = json.loads((run / "ab.json").read_text())
                return rc, state
            finally:
                for name, value in original.items():
                    setattr(mod.SourceControl, name, value)
                mod.base.load_window = original_load
                mod.base.SERVICE_ROOT = original_root

    for port_state in ("open", "unknown"):
        rc, state = run_main_port(port_state)
        check(f"main blocks restore on {port_state} port",
              rc == 1
              and state["steps"]["port_gate"]["port"] == port_state
              and state["steps"]["port_gate"]["ok"] is False
              and state["steps"]["final_source"]["ok"] is False
              and state["steps"]["restore"]["action"] == "blocked_service_not_proven_stopped"
              and state["steps"].get("restore_port_gate", {}).get("ok") is False)

    failed = [name for name, ok in checks if not ok]
    print(json.dumps({"checks": len(checks), "failed": failed, "ok": not failed}))
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
