#!/usr/bin/env python3
"""CPU entry/failure tests for the service A/B runner.

Only external boundaries are faked: subprocess for stop, a fake window module
for port/process helpers and temporary input files.  The real runner methods
are exercised; no service or NPU is started.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNNER = HERE / "joint_fused_service_ab.py"


def load():
    spec = importlib.util.spec_from_file_location("joint_fused_service_ab", RUNNER)
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

    check("ABBA order", mod.ORDERS == [[0, 1], [1, 0], [1, 0], [0, 1]])
    check("four configured rounds", len(mod.ORDERS) >= 4)
    check("input hashes fixed", mod.TEXT_SHA["od-0105"].startswith("25b8acaf")
          and mod.TEXT_SHA["od-1534"].startswith("568ea158")
          and mod.IMAGE_SHA.startswith("f3229f5c"))

    with tempfile.TemporaryDirectory(prefix="ab-test-") as tmp:
        tmp = Path(tmp)
        existing = tmp / "exists"
        existing.mkdir()
        argv = sys.argv
        sys.argv = ["ab", "--run-dir", str(existing)]
        try:
            code = mod.main()
        finally:
            sys.argv = argv
        check("existing run dir rejected", code == 1)

        runner = object.__new__(mod.ServiceAB)
        runner.args = types.SimpleNamespace(
            python="python", stop_script="/stop.py", stop_timeout=1,
            port=18143)
        runner.run_dir = tmp / "run"
        runner.run_dir.mkdir()
        runner.win = types.SimpleNamespace(
            parse_environ=lambda blob: {},
            port_open=lambda port: "open",
            sha256=lambda path: "",
            proc_identity=lambda pid: None,
            descendants=lambda pid: [],
        )
        identity = {"root": {"pid": 42, "starttime": "1", "boot_id": "b",
                             "pgid": 42, "cmdline": "x"}, "descendants": []}
        original = mod.subprocess.run
        mod.subprocess.run = lambda *a, **kw: types.SimpleNamespace(
            returncode=1, stdout="", stderr="stop failed")
        try:
            try:
                runner.stop_tree(identity, "bad")
            except mod.AbError:
                blocked = True
            else:
                blocked = False
        finally:
            mod.subprocess.run = original
        check("stop failure raises", blocked)

        data = tmp / "data.json"
        data.write_text(json.dumps({"samples": [{"name": "od-0105",
                                                  "request": {"prompt": "x"}}]}))
        image = tmp / "image.json"
        image.write_text("{}")
        runner.args.data = str(data)
        runner.args.short = "od-0105"
        runner.args.long = "od-0105"
        runner.args.image_source = str(image)
        runner.args.text_url = "http://x"
        runner.args.image_url = "http://y"
        runner.state = {}
        try:
            runner.input_bodies()
        except mod.AbError:
            hash_blocked = True
        else:
            hash_blocked = False
        check("input hash mismatch blocked", hash_blocked)

        # real run_inputs: warmup/sample split, raw JSONL and full response
        class FakeResponse:
            status = 200
            def __init__(self, payload):
                self.payload = payload
            def read(self):
                return json.dumps(self.payload).encode()
            def __enter__(self):
                return self
            def __exit__(self, *a):
                return False
        runner = object.__new__(mod.ServiceAB)
        runner.args = types.SimpleNamespace(warmup=1, reps=2, http_timeout=5)
        runner.run_dir = tmp / "rawrun"
        runner.run_dir.mkdir()
        runner.inputs = {"x": {"body": b"{}", "sha256": "abc", "url": "http://x",
                               "expect": "score", "kind": "text"}}
        original_url = mod.urllib.request.urlopen
        mod.urllib.request.urlopen = lambda request, timeout: FakeResponse(
            {"answers": {"answer": {"type": "score", "probabilities": {"a": 0.5}}}})
        try:
            result = runner.run_inputs("L")
        finally:
            mod.urllib.request.urlopen = original_url
        raw = Path(result["x"]["raw"])
        lines = [json.loads(line) for line in raw.read_text().splitlines()]
        check("raw JSONL warmup/sample split", len(lines) == 3
              and sum(row["warmup"] for row in lines) == 1
              and len(result["x"]["samples"]) == 2
              and result["x"]["samples"][0]["response"]["answers"]["answer"]["probabilities"]["a"] == 0.5)

    with tempfile.TemporaryDirectory(prefix="ab-stage-") as tmp:
        runner = object.__new__(mod.ServiceAB)
        runner.args = types.SimpleNamespace(repo=str(HERE.parent), tree="", python="",
                                            stop_script="", port=18143)
        runner.run_dir = Path(tmp) / "run"
        runner.state = {}
        runner.win = mod.load_window()
        staged = runner.stage_candidate()
        ok = (runner.win.sha256(staged / "apply_joint_fused_attention.py") == mod.INSTALLER_SHA
              and runner.win.sha256(staged / "joint_fused_attention" /
                                    "clef_joint_schema.py") == mod.POSTIMAGE_SHA
              and runner.win.sha256(Path(runner.win.__file__)) == mod.WINDOW_SHA
              and runner.win.sha256(staged / "apply_joint_service_check.py")
                  == mod.SERVICE_CHECK_INSTALLER_SHA
              and runner.win.sha256(staged / "joint_service_check" /
                                    "apply_joint_single_field_patch.py")
                  == mod.SERVICE_CHECK_SOURCE_SHA)
        check("stage candidate against approved wrapper", ok)
        checks[-1] = (checks[-1][0], ok)

    failed = [name for name, ok in checks if not ok]
    print(json.dumps({"checks": len(checks), "failed": failed, "ok": not failed}))
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
