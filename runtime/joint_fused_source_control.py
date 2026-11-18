#!/usr/bin/env python3
"""Independent production-f810 vs aa3e source-control runner.

This is a separate entry point; it does not change the frozen two-arm aa3e
A/B runner.  Arm A is the real f810 single-field source with flag0 and no
candidate allow env.  Arm B is aa3e with flag1 and the exact aa3e allow env.
Every arm is started only after the previous arm was proven stopped; source
switches use the existing hash-gated installer check/apply/revert plus an
owned manifest, and never hand-write an unknown source file.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import shlex
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE_PATH = HERE / "joint_fused_service_ab.py"
F810 = "f8101b98228bbd472a63a1efe6827583e35c6cc9dede4379b925f2b016670863"
AA3E = "aa3e17585f8bd60a4d05e342abc7e2773c79a0b2d49f0370d15e29bcc6b72503"
CLEAN_PATH = ("/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:"
              "/usr/games:/usr/local/games:/snap/bin")
CLEAN_ENV_KEYS = ("ASCEND_CUSTOM_OPP_PATH", "LD_LIBRARY_PATH")
CLEAN_ENV_EXPECT_SHA = "fcbdcbf11521e1ae499c1ac5329eed990d63f3e8a75a99b5fbb6b4df1dcb0230"
CLEAN_ENV_EXPECT_PATH = ("/workspace/yangchen/logs/perf-wave2-20261118/"
                         "s1-final-r1/launch-manifest.json")
ORDERS = [
    [("f810", 0), ("aa3e", 1)],
    [("aa3e", 1), ("f810", 0)],
    [("aa3e", 1), ("f810", 0)],
    [("f810", 0), ("aa3e", 1)],
]


def load_base():
    spec = importlib.util.spec_from_file_location("joint_fused_service_ab_base",
                                                  BASE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


base = load_base()


class SourceControl(base.ServiceAB):
    def __init__(self, args):
        super().__init__(args)
        self.active_source = None
        self.source_switches = []

    def source_target(self):
        return Path(self.args.tree) / "vllm/model_executor/models/clef_joint_schema.py"

    def switch_source(self, target, tag):
        if target not in ("f810", "aa3e"):
            raise base.AbError(f"unknown source arm: {target}")
        if self.win.port_open(self.args.port) != "closed":
            raise base.AbError("source switch requires a stopped service")
        desired = F810 if target == "f810" else AA3E
        installer = self.run_dir / "stage" / "runtime" / "apply_joint_fused_attention.py"
        check = self.call([self.args.python, str(installer),
                           "--tree", self.args.tree, "--check"])
        report = json.loads(check.stdout)
        before = report.get("state_after")
        manifest = self.run_dir / f"source-{tag}-{target}.json"
        if target == "f810":
            if before == "postimage":
                self.call([self.args.python, str(installer), "--tree", self.args.tree,
                           "--revert", "--manifest", str(manifest)])
            elif before != "preimage":
                raise base.AbError(f"cannot switch to f810 from {before}")
        else:
            if before == "preimage":
                self.call([self.args.python, str(installer), "--tree", self.args.tree,
                           "--apply", "--manifest", str(manifest)])
            elif before != "postimage":
                raise base.AbError(f"cannot switch to aa3e from {before}")
        actual = self.win.sha256(self.source_target())
        if actual != desired:
            raise base.AbError(f"source switch {target} landed on {actual}")
        if manifest.is_file():
            m = json.loads(manifest.read_text())
            if m.get("state_after") not in (None, "preimage" if target == "f810" else "postimage"):
                raise base.AbError(f"owned manifest disagrees for {target}: {m}")
        record = {"target": target, "actual_sha256": actual,
                  "state_before": before,
                  "manifest": str(manifest) if manifest.is_file() else None,
                  "at": base.now()}
        self.source_switches.append(record)
        self.state["source_switches"] = self.source_switches
        self.active_source = target
        return record

    def arm_environment(self, source, flag):
        env = dict(self.prod_env)
        for key in base.CANDIDATE_FLAGS:
            env.pop(key, None)
        env["CLEF_JOINT_FUSED_ATTN"] = str(flag)
        env["CLEF_LAUNCH_CONFIG"] = "s1"
        if source == "aa3e":
            env[base.ALLOW_ENV] = base.POSTIMAGE_SHA
        return env

    def clean_restore_environment(self):
        return {"PATH": CLEAN_PATH, "CLEF_LAUNCH_CONFIG": "s1"}

    def _read_root_env_bytes(self, label):
        pid_file = Path(self.args.service_log_root) / label / "api.pid"
        if not pid_file.is_file():
            raise base.AbError(f"restored api.pid missing: {pid_file}")
        pid = int(pid_file.read_text().strip())
        raw = (self.win.PROC_ROOT / f"{pid}/environ").read_bytes()
        entries = {}
        for item in raw.split(b"\0"):
            if b"=" in item:
                key, value = item.split(b"=", 1)
                entries[key.decode()] = value
        return pid, entries

    def verify_restored_env(self, result):
        expected_path = Path(self.args.clean_env_expect or CLEAN_ENV_EXPECT_PATH).expanduser().resolve()
        if not expected_path.is_file():
            raise base.AbError(f"clean env expected missing: {expected_path}")
        expected_sha = self.win.sha256(expected_path)
        if expected_sha != CLEAN_ENV_EXPECT_SHA:
            raise base.AbError(f"clean env expected SHA mismatch: {expected_sha}")
        expected_doc = json.loads(expected_path.read_text())
        expected_env = expected_doc.get(
            "service_environment", expected_doc.get("env", expected_doc.get("expected", expected_doc)))
        expected = {}
        for key in CLEAN_ENV_KEYS:
            if key not in expected_env:
                raise base.AbError(f"clean env expected lacks {key}")
            expected[key] = str(expected_env[key]).encode()
        pid, actual_raw = self._read_root_env_bytes(result["label"])
        actual = {}
        for key in CLEAN_ENV_KEYS:
            if key not in actual_raw:
                raise base.AbError(f"restored root lacks {key}")
            actual[key] = actual_raw[key]
        compare = {
            "schema": "restore-env-compare/1",
            "expected_path": str(expected_path),
            "expected_sha256": self.win.sha256(expected_path),
            "root_pid": pid,
            "match": actual == expected,
            "counts": {key: len(actual[key].split(b":")) for key in CLEAN_ENV_KEYS},
            "expected_counts": {key: len(expected[key].split(b":")) for key in CLEAN_ENV_KEYS},
            "expected_sha256_values": {k: __import__("hashlib").sha256(expected[k]).hexdigest() for k in CLEAN_ENV_KEYS},
            "actual_sha256_values": {k: __import__("hashlib").sha256(actual[k]).hexdigest() for k in CLEAN_ENV_KEYS},
        }
        self.run_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(expected_path, self.run_dir / "restore-env-expected.json")
        (self.run_dir / "restore-env-actual.json").write_text(json.dumps(
            {k: v.decode() for k, v in actual.items()}, indent=1) + "\n")
        (self.run_dir / "restore-env-compare.json").write_text(
            json.dumps(compare, indent=1) + "\n")
        self.state.setdefault("steps", {})["restore_env"] = {
            "ok": compare["match"], "compare": compare}
        if not compare["match"]:
            raise base.AbError("restored root OPP/LD mismatch against frozen expected")
        result["restore_env"] = compare
        return compare

    def restore_prod(self):
        label = (f"d0-restore-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}-"
                 f"{uuid.uuid4().hex[:8]}")
        clean = self.clean_restore_environment()
        original = self.prod_env
        self.prod_env = clean
        self.state.setdefault("steps", {})["restore_env_input"] = {
            "source": "28-clean-parent", "keys": sorted(clean),
            "PATH": clean["PATH"], "CLEF_LAUNCH_CONFIG": clean["CLEF_LAUNCH_CONFIG"]}
        try:
            before = {int(p.name) for p in self.win.PROC_ROOT.iterdir() if p.name.isdigit()}
            launch = subprocess.run(["/bin/bash", self.args.launch, label], env=clean,
                                    capture_output=True, text=True, timeout=60)
            (self.run_dir / "restore-launch.log").write_text(launch.stdout + launch.stderr)
            if launch.returncode != 0:
                raise base.AbError("clean restore launch failed")
            wait = subprocess.run(["/bin/bash", self.args.wait], env=clean,
                                  capture_output=True, text=True,
                                  timeout=self.args.wait_timeout)
            (self.run_dir / "restore-wait.log").write_text(wait.stdout + wait.stderr)
            if wait.returncode != 0:
                raise base.AbError("clean restore wait failed")
            sys.path.insert(0, str(self.run_dir / "stage" / "d0"))
            import d0_restore
            import d0_run
            disc = d0_restore.discover_service_pids(label, before)
            if disc.get("error") or not disc.get("pids"):
                raise base.AbError(f"restore discovery failed: {disc.get('error')}")
            log = {"label": label, "service_discovery": disc,
                   "restore_env": dict(clean), "restore_env_source": "28-clean-parent"}
            verified = d0_restore.verify(
                d0_restore.build_commands(d0_run.render_request), log,
                service_pids=set(disc["pids"]))
            (self.run_dir / "restore.json").write_text(
                json.dumps(log, ensure_ascii=False, indent=1) + "\n")
            if not verified:
                raise base.AbError("restore four-check verification failed")
            result = {"label": label, "verify_ok": True,
                      "service_pids": sorted(disc["pids"])}
            self.verify_restored_env(result)
            return result
        finally:
            self.prod_env = original

    def _start_arm(self, flag, label):
        source = self.active_source
        expected = {"f810": 0, "aa3e": 1}
        if source not in expected or flag != expected[source]:
            raise base.AbError(f"arm/source mismatch: {source} flag {flag}")
        env = self.arm_environment(source, flag)
        before = {int(p.name) for p in self.win.PROC_ROOT.iterdir() if p.name.isdigit()}
        self.arm_identity = None
        self.arm_label = label
        launch = subprocess.run(["bash", self.args.launch, label], env=env,
                                capture_output=True, text=True, timeout=60)
        (self.run_dir / f"launch-{label}.log").write_text(launch.stdout + launch.stderr)
        if launch.returncode != 0:
            if self.win.port_open(self.args.port) != "closed":
                self.attribution_unknown = True
            raise base.AbError(f"launch failed: {label}")
        pid_file = Path(self.args.service_log_root) / label / "api.pid"
        if not pid_file.is_file():
            self.attribution_unknown = True
            raise base.AbError(f"api.pid missing immediately after launch: {pid_file}")
        root_pid = int(pid_file.read_text().strip())
        identity = self.win.proc_identity(root_pid)
        after = {int(p.name) for p in self.win.PROC_ROOT.iterdir() if p.name.isdigit()}
        if identity is None or root_pid not in (after - before):
            self.attribution_unknown = True
            raise base.AbError(f"root {root_pid} is not a new label-bound process")
        self.arm_identity = {"root": identity,
                             "descendants": self.win.descendants(root_pid)}
        wait = subprocess.run(["bash", self.args.wait], capture_output=True, text=True,
                              timeout=self.args.wait_timeout)
        (self.run_dir / f"wait-{label}.log").write_text(wait.stdout + wait.stderr)
        if wait.returncode != 0:
            raise base.AbError(f"wait failed: {label}")
        live = self.win.proc_identity(root_pid)
        if not self.same_process(identity, live):
            self.attribution_unknown = True
            raise base.AbError(f"root identity changed during wait: {root_pid}")
        if not self.is_final_vllm(live):
            raise base.AbError("root did not exec to the known vllm argv")
        if self.win.port_open(self.args.port) != "open":
            raise base.AbError("service port is not open after wait")
        self.arm_identity = {"root": live,
                             "descendants": self.win.descendants(root_pid)}
        identity = live
        root_env = self.win.parse_environ(
            (self.win.PROC_ROOT / f"{root_pid}/environ").read_bytes())
        source_sha = self.win.sha256(self.source_target())
        if root_env.get("CLEF_LAUNCH_CONFIG") != "s1":
            raise base.AbError(f"arm {label} launch config mismatch")
        if source == "f810":
            if source_sha != F810 or root_env.get("CLEF_JOINT_FUSED_ATTN") != "0":
                raise base.AbError(f"arm {label} is not real f810 flag0")
            if root_env.get(base.ALLOW_ENV) is not None:
                raise base.AbError(f"arm {label} f810 must not carry allow env")
        else:
            if source_sha != AA3E or root_env.get("CLEF_JOINT_FUSED_ATTN") != "1":
                raise base.AbError(f"arm {label} is not aa3e flag1")
            if root_env.get(base.ALLOW_ENV) != AA3E:
                raise base.AbError(f"arm {label} allow env mismatch")
        npu = subprocess.run(["npu-smi", "info"], capture_output=True, text=True,
                             timeout=30)
        (self.run_dir / f"npu-{label}.txt").write_text(npu.stdout + npu.stderr)
        return {"root": identity, "descendants": self.arm_identity["descendants"],
                "label": label, "source": source, "source_sha256": source_sha,
                "before_count": len(before), "after_new": sorted(after - before),
                "env_flag": root_env.get("CLEF_JOINT_FUSED_ATTN"),
                "launch_config": root_env.get("CLEF_LAUNCH_CONFIG"),
                "service_check_env": root_env.get(base.ALLOW_ENV),
                "npu_file": str(self.run_dir / f"npu-{label}.txt")}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--tree", default="/workspace/yangchen/work/src/vllm")
    ap.add_argument("--python", default="/workspace/yangchen/venvs/clef-py312/bin/python")
    ap.add_argument("--activate", default="/workspace/yangchen/Clef-27B-ascend/scripts/activate_env.sh")
    ap.add_argument("--launch", default="/workspace/yangchen/logs/perf-wave2-20261118/launch.sh")
    ap.add_argument("--wait", default="/workspace/yangchen/logs/perf-wave2-20261118/wait.sh")
    ap.add_argument("--stop-script", default="/tmp/d0cc/d0_stop.py")
    ap.add_argument("--service-log-root", default="/workspace/yangchen/logs/perf-wave2-20261118")
    ap.add_argument("--clean-env-expect", default=CLEAN_ENV_EXPECT_PATH,
                    help="28 frozen launch-manifest; SHA pinned")
    ap.add_argument("--data", default="/workspace/yangchen/logs/tuning/20261114-013000-fix4/opendecisions/all2352.json")
    ap.add_argument("--short", default="od-0105")
    ap.add_argument("--long", default="od-1534")
    ap.add_argument("--image-source", default="/workspace/yangchen/logs/perf-wave2-20261118/erqa/samples/erqa-0000/request.json")
    ap.add_argument("--text-url", default="http://127.0.0.1:18143/v1/systemone")
    ap.add_argument("--image-url", default="http://127.0.0.1:18143/v1/decide")
    ap.add_argument("--port", type=int, default=18143)
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--http-timeout", type=float, default=120.0)
    ap.add_argument("--stop-timeout", type=float, default=90.0)
    ap.add_argument("--wait-timeout", type=float, default=900.0)
    args = ap.parse_args(argv)
    if Path(args.run_dir).expanduser().resolve().exists():
        print(json.dumps({"error": "run dir exists"}))
        return 1
    if args.rounds < 4:
        print(json.dumps({"error": "rounds must be at least 4"}))
        return 1
    ab = SourceControl(args)
    ab.state = {"schema": "joint-fused-source-control/1", "at": base.now(),
                "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                "rounds": [], "steps": {}, "source_switches": [], "overall_rc": 1}
    candidate_started = False
    stop_attempted = False
    initial_stopped = False
    applied = False
    try:
        ab.run_dir.mkdir(parents=True, exist_ok=False)
        ab.stage_d0()
        ab.stage_candidate()
        target = ab.source_target()
        target_sha = ab.win.sha256(target) if target.is_file() else None
        if target_sha != F810:
            raise base.AbError(f"source-control preflight requires f810: {target_sha}")
        guard = Path(base.SERVICE_ROOT) / "runtime" / "apply_joint_single_field_patch.py"
        if not guard.is_file() or ab.win.sha256(guard) != base.SERVICE_CHECK_PREIMAGE_SHA:
            raise base.AbError("service guard is not the original preimage before stop")
        if ab.win.sha256(Path(args.stop_script)) != base.STOP_SHA:
            raise base.AbError("stop script SHA mismatch")
        for path, expected, name in ((args.activate, base.ACTIVATE_SHA, "activate"),
                                     (args.launch, base.LAUNCH_SHA, "launch"),
                                     (args.wait, base.WAIT_SHA, "wait")):
            if ab.win.sha256(Path(path)) != expected:
                raise base.AbError(f"{name} SHA mismatch")
        current, ab.prod_env = ab.current_service()
        initial_identity = current
        ab.state["current_root"] = current["root"]["pid"]
        ab.inputs = ab.input_bodies()
        ab.flush("preflight")
        candidate_started = True
        stop_attempted = True
        ab.stop_tree(current, "initial")
        initial_stopped = True
        ab.state["steps"]["initial_stop"] = {"ok": True}
        applied = True
        ab.apply_service_check()
        ab.state["steps"]["service_check_apply"] = {"ok": True}
        ab.switch_source("f810", "initial")
        ab.state["steps"]["initial_source"] = {"ok": True}
        ab.flush("initial-source")
        for index in range(args.rounds):
            order = ORDERS[index % len(ORDERS)]
            row = {"round": index, "order": [[s, f] for s, f in order], "arms": {}}
            ab.state["rounds"].append(row)
            ab.flush(f"round-{index}-start")
            for source, flag in order:
                ab.active_source = source
                ab.switch_source(source, f"{index}-{source}")
                label = (f"d0-src-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"
                         f"-{index}-{source}-{uuid.uuid4().hex[:6]}")
                ab.last_arm_row = None
                try:
                    arm_row = ab.run_arm(flag, label)
                except Exception as exc:
                    partial = getattr(ab, "last_arm_row", None) or {
                        "label": label, "provenance": None, "samples": None,
                        "error": f"{type(exc).__name__}: {exc}",
                        "stop": None, "stop_error": None}
                    row["arms"][str(flag)] = partial
                    ab.flush(f"round-{index}-{source}-error")
                    raise
                row["arms"][str(flag)] = arm_row
                ab.flush(f"round-{index}-{source}")
    except Exception as exc:
        ab.state["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        target = ab.source_target()
        target_sha = ab.win.sha256(target) if target.is_file() else None
        if not candidate_started:
            ab.state["steps"]["finish"] = {"ok": True, "action": "read_only_noop",
                                           "finished_at": base.now()}
            ab.state["finished_at"] = base.now()
            ab.state["overall_rc"] = 1
            ab.flush("read-only-noop")
            print(json.dumps({k: ab.state.get(k) for k in (
                "schema", "at", "error", "steps", "overall_rc")},
                ensure_ascii=False, indent=1))
            return 1
        stopped = initial_stopped
        if stop_attempted and not stopped:
            try:
                ab.stop_tree(initial_identity, "cleanup-initial")
                stopped = True
            except Exception as exc:
                ab.state["cleanup_stop_error"] = f"{type(exc).__name__}: {exc}"
                stopped = False
        cleanup_identity = getattr(ab, "arm_identity", None)
        if cleanup_identity is not None:
            try:
                ab.stop_tree(cleanup_identity, "cleanup-arm")
                stopped = True
            except Exception as exc:
                ab.state["cleanup_stop_error"] = f"{type(exc).__name__}: {exc}"
                stopped = False
                ab.cleanup_unconfirmed = True
            ab.arm_identity = None
        if ab.attribution_unknown or ab.cleanup_unconfirmed:
            stopped = False
        port_state = ab.win.port_open(args.port)
        ab.state["steps"]["port_gate"] = {"ok": port_state == "closed",
                                          "port": port_state}
        if port_state != "closed":
            stopped = False
            ab.state["service_state"] = {"stopped": False, "port": port_state,
                                         "action": "blocked"}
        source_ok = True
        if stopped:
            try:
                ab.switch_source("f810", "final")
                ab.state["steps"]["final_source"] = {"ok": True}
            except Exception as exc:
                source_ok = False
                ab.state["steps"]["final_source"] = {
                    "ok": False, "error": f"{type(exc).__name__}: {exc}"}
                stopped = False
        else:
            ab.state["steps"]["final_source"] = {
                "ok": False, "action": "blocked_service_or_port_not_clean",
                "port": port_state}
        service_check_ok = True
        if getattr(ab, "service_check_applied", False):
            try:
                ab.revert_service_check()
                ab.state["steps"]["service_check_revert"] = {"ok": True}
            except Exception as exc:
                service_check_ok = False
                ab.state["steps"]["service_check_revert"] = {
                    "ok": False, "error": f"{type(exc).__name__}: {exc}"}
        else:
            service_target = Path(base.SERVICE_ROOT) / "runtime" / "apply_joint_single_field_patch.py"
            service_check_ok = (service_target.is_file()
                                and ab.win.sha256(service_target) == base.SERVICE_CHECK_PREIMAGE_SHA)
        port_state = ab.win.port_open(args.port)
        ab.state["steps"]["restore_port_gate"] = {"ok": port_state == "closed",
                                                  "port": port_state}
        if port_state != "closed":
            stopped = False
        target_sha = ab.win.sha256(target) if target.is_file() else None
        if target_sha != F810 or not source_ok or not service_check_ok:
            ab.state["steps"]["restore"] = {
                "ok": False, "action": "blocked_source_or_guard_not_clean",
                "target_sha256": target_sha, "source_ok": source_ok,
                "service_check_ok": service_check_ok}
        elif not stopped:
            ab.state["steps"]["restore"] = {
                "ok": False, "action": "blocked_service_not_proven_stopped",
                "port": ab.win.port_open(args.port)}
        else:
            try:
                restored = ab.restore_prod()
                ab.state["steps"]["restore"] = {"ok": True, **restored}
            except Exception as exc:
                ab.state["steps"]["restore"] = {"ok": False,
                                                "error": f"{type(exc).__name__}: {exc}"}
        ab.state["finished_at"] = base.now()
        ab.state["overall_rc"] = 0 if (
            not ab.state.get("error")
            and ab.state["steps"].get("final_source", {}).get("ok")
            and ab.state["steps"].get("service_check_revert", {}).get("ok")
            and ab.state["steps"].get("restore", {}).get("ok")
            and len(ab.state["rounds"]) == args.rounds) else 1
        ab.flush("finished")
    print(json.dumps({k: ab.state.get(k) for k in (
        "schema", "at", "error", "steps", "overall_rc")}, ensure_ascii=False, indent=1))
    return ab.state["overall_rc"]


if __name__ == "__main__":
    raise SystemExit(main())
