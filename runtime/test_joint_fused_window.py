#!/usr/bin/env python3
"""CPU tests for the joint-head window wrapper's non-device logic.

Only external boundaries are exercised: a localhost listening socket, the
environment parser/selector and the finally-state machine with a fake step
callback.  No production wrapper function, torch, torch_npu or service is
imported or replaced.
"""

from __future__ import annotations

import importlib.util
import json
import socket
import tempfile
import types
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

    with tempfile.TemporaryDirectory(prefix="window-test-") as tmp:
        def make(applied, stopped, restored, port_state):
            window = object.__new__(mod.Window)
            window.args = types.SimpleNamespace(port=18143)
            window.run_dir = Path(tmp)
            window.state = {"steps": {}}
            window.applied = applied
            window.stopped = stopped
            window.restored = restored
            window.overall_failed = False
            calls = []
            def step(name, required=True):
                calls.append(name)
                return {"rc": 0, "ok": True}
            window.step = step
            window.flush = lambda: None
            return window, calls

        original = mod.port_open
        mod.port_open = lambda port: "open"
        window, calls = make(False, True, False, "open")
        window.finish()
        check("stop+apply failure restores", calls == ["restore"], str(calls))
        window, calls = make(True, True, False, "open")
        window.finish()
        check("applied failure reverts then restores", calls == ["revert", "restore"], str(calls))
        window, calls = make(False, False, False, "open")
        window.finish()
        check("stop failure keeps existing service", calls == []
              and window.state["steps"]["restore"]["action"] == "existing_service_kept")
        window, calls = make(False, False, False, "closed")
        mod.port_open = lambda port: "closed"
        window.finish()
        check("stop failure with no service fails", calls == []
              and window.overall_failed is True)
        mod.port_open = original

    print(json.dumps({"checks": 8, "failed": failures, "ok": not failures}))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
