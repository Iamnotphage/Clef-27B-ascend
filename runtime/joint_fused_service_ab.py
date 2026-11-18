#!/usr/bin/env python3
"""Service-level A/B runner for the joint-head fused-attention candidate.

Execution owner: 21.  Both arms use the same installed postimage (aa3e1758);
only CLEF_JOINT_FUSED_ATTN differs.  Fixed real inputs are od-0105 (short),
od-1534 (long) and erqa-0000 (image).  Each arm gets its own service label;
stop is the identity-checked d0_stop; finally reverts to f8101b98 and restores
the production service with health/two-worker/text/image checks.

The runner never imports torch or torch_npu; it only drives the HTTP service
and the existing exact stop/restore helpers.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
import uuid
from pathlib import Path

HERE = Path(__file__).resolve().parent
WIN_PATH = HERE / "joint_fused_attention_window.py"
PREIMAGE_SHA = "f8101b98228bbd472a63a1efe6827583e35c6cc9dede4379b925f2b016670863"
POSTIMAGE_SHA = "aa3e17585f8bd60a4d05e342abc7e2773c79a0b2d49f0370d15e29bcc6b72503"
STOP_SHA = "d6d49d75012a0789d3bff3e7b522e9cd496844ca0ee0424608ab8e6b43a33886"
ACTIVATE_SHA = "3b363394536fbe2433a84b666170568fcb58848b1d8d24551ff6ae9f53b4044a"
LAUNCH_SHA = "a737386119b89c5bc787e6f69e5017b83391934148528d08d498ff943a958321"
WAIT_SHA = "e3fb08882211aef6ea0fd4f3db3d37bba0a591034cc397ba95559e09d1f1ffa0"
TEXT_SHA = {
    "od-0105": "25b8acaf74a0d2b9fc210e08687e3a08c3d5fff82e6785c20611418e8cc72bf0",
    "od-1534": "568ea1583377dcc6044d3fcd4a099c3beb4cd0e118155f95b595bda72b1c76bf",
}
IMAGE_SHA = "f3229f5c120098ac4eebcc6f4025c28254e35f959fd01998cac5db7b8d209aab"
CANDIDATE_FLAGS = {"CLEF_JOINT_FUSED_ATTN", "CLEF_JOINT_SMALLQ", "CLEF_JOINT_SMALLQ_KERNEL"}
ORDERS = [[0, 1], [1, 0], [1, 0], [0, 1]]


def load_window():
    spec = importlib.util.spec_from_file_location("joint_fused_window", WIN_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


class AbError(RuntimeError):
    pass


class ServiceAB:
    def __init__(self, args):
        self.args = args
        self.win = load_window()
        self.run_dir = Path(args.run_dir).expanduser().resolve()

    def flush(self, phase):
        self.state["phase"] = phase
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "ab.json").write_text(
            json.dumps(self.state, ensure_ascii=False, indent=1, default=str) + "\n")

    def call(self, cmd, timeout=120, env=None):
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                                env=env)
        if result.returncode != 0:
            raise AbError(f"command failed {result.returncode}: {cmd}: "
                          f"{(result.stdout + result.stderr)[-1500:]}")
        return result

    def source_env(self):
        script = 'set -a; source "$1" >/dev/null 2>&1 || exit 3; env -0'
        result = subprocess.run(["bash", "-c", script, "bash", self.args.activate],
                                capture_output=True, timeout=60)
        if result.returncode != 0:
            raise AbError(f"source activate failed: {result.returncode}")
        return self.win.parse_environ(result.stdout)

    def stage_d0(self):
        target = self.run_dir / "stage" / "d0"
        target.mkdir(parents=True, exist_ok=True)
        for name, expected in self.win.D0_MODULES.items():
            source = Path(self.args.repo) / "runtime" / "d0" / name
            if not source.is_file() or self.win.sha256(source) != expected:
                raise AbError(f"restore module missing/SHA mismatch: {name}")
            shutil.copy2(source, target / name)
        return target

    def current_service(self):
        if self.win.port_open(self.args.port) != "open":
            raise AbError("service port is not open at preflight")
        pids = self.win.find_api_pid(self.args.port)
        if len(pids) != 1:
            raise AbError(f"api pid discovery is not unique: {pids}")
        identity = self.win.proc_identity(pids[0])
        if identity is None:
            raise AbError("api identity unavailable")
        root = {"root": identity,
                "descendants": self.win.descendants(identity["pid"])}
        env = self.win.parse_environ(
            (self.win.PROC_ROOT / f"{identity['pid']}/environ").read_bytes())
        for key in CANDIDATE_FLAGS:
            env.pop(key, None)
        return root, env

    def stop_tree(self, identity, label):
        out = self.run_dir / f"stop-{label}.json"
        identity_path = self.run_dir / f"identity-{label}.json"
        identity_path.write_text(json.dumps(identity["root"], ensure_ascii=False) + "\n")
        cmd = [self.args.python, self.args.stop_script,
               "--api-pid", str(identity["root"]["pid"]),
               "--saved-identity", str(identity_path), "--out", str(out)]
        result = subprocess.run(cmd, capture_output=True, text=True,
                                timeout=self.args.stop_timeout)
        (self.run_dir / f"stop-{label}.log").write_text(result.stdout + result.stderr)
        if result.returncode != 0:
            raise AbError(f"stop failed for {label}: {result.returncode}")
        if self.win.port_open(self.args.port) != "closed":
            raise AbError(f"port still open after stop {label}")
        return {"stop_json": str(out)}

    def apply_postimage(self):
        installer = Path(self.args.repo) / "runtime" / "apply_joint_fused_attention.py"
        check = self.call([self.args.python, str(installer), "--tree", self.args.tree,
                           "--check"])
        report = json.loads(check.stdout)
        target = Path(self.args.tree) / "vllm/model_executor/models/clef_joint_schema.py"
        state = self.win.sha256(target) if target.is_file() else None
        if state == POSTIMAGE_SHA:
            return {"state": "postimage"}
        if state != PREIMAGE_SHA:
            raise AbError(f"unknown head source: {state}")
        self.call([self.args.python, str(installer), "--tree", self.args.tree,
                   "--apply", "--manifest", str(self.run_dir / "apply.json")])
        if self.win.sha256(target) != POSTIMAGE_SHA:
            raise AbError("postimage SHA mismatch after apply")
        return {"state": "postimage", "applied": True}

    def revert_postimage(self):
        installer = Path(self.args.repo) / "runtime" / "apply_joint_fused_attention.py"
        self.call([self.args.python, str(installer), "--tree", self.args.tree,
                   "--revert", "--manifest", str(self.run_dir / "revert.json")])
        target = Path(self.args.tree) / "vllm/model_executor/models/clef_joint_schema.py"
        if not target.is_file() or self.win.sha256(target) != PREIMAGE_SHA:
            raise AbError("revert did not restore f8101b98")

    def input_bodies(self):
        doc = json.loads(Path(self.args.data).read_text())
        bodies = {}
        for name in (self.args.short, self.args.long):
            sample = next((item for item in doc["samples"] if item.get("name") == name), None)
            if sample is None:
                raise AbError(f"input sample missing: {name}")
            body = json.dumps(sample["request"], ensure_ascii=False,
                              separators=(",", ":")).encode()
            digest = hashlib.sha256(body).hexdigest()
            if digest != TEXT_SHA[name]:
                raise AbError(f"input hash mismatch: {name}: {digest}")
            bodies[name] = {"body": body, "sha256": digest, "url": self.args.text_url,
                            "expect": "score", "kind": "text"}
        image = Path(self.args.image_source).read_bytes()
        digest = hashlib.sha256(image).hexdigest()
        if digest != IMAGE_SHA:
            raise AbError(f"image input hash mismatch: {digest}")
        bodies["erqa-0000"] = {"body": image, "sha256": digest,
                               "url": self.args.image_url, "expect": "choice",
                               "kind": "image"}
        self.state["inputs"] = {name: {key: value for key, value in item.items()
                                       if key != "body"} for name, item in bodies.items()}
        return bodies

    def run_inputs(self, label):
        rows = {}
        for name, item in self.inputs.items():
            samples = []
            for _ in range(self.args.reps):
                request = urllib.request.Request(
                    item["url"], data=item["body"],
                    headers={"Content-Type": "application/json"})
                begin = time.perf_counter()
                with urllib.request.urlopen(request, timeout=self.args.http_timeout) as response:
                    payload = json.loads(response.read())
                    status = response.status
                elapsed = (time.perf_counter() - begin) * 1000.0
                answer = ((payload.get("answers") or {}).get("answer") or {})
                valid = status == 200 and answer.get("type") == item["expect"]
                samples.append({"ms": elapsed, "status": status,
                                "answer_type": answer.get("type"), "valid": valid})
            if not samples or not all(row["valid"] for row in samples):
                raise AbError(f"invalid response for {name} in {label}")
            rows[name] = samples
        return rows

    def start_arm(self, flag, label):
        env = dict(self.prod_env)
        for key in CANDIDATE_FLAGS:
            env.pop(key, None)
        if flag is not None:
            env["CLEF_JOINT_FUSED_ATTN"] = str(flag)
        launch = subprocess.run(["bash", self.args.launch, label], env=env,
                                capture_output=True, text=True, timeout=60)
        (self.run_dir / f"launch-{label}.log").write_text(launch.stdout + launch.stderr)
        if launch.returncode != 0:
            raise AbError(f"launch failed: {label}")
        wait = subprocess.run(["bash", self.args.wait], capture_output=True, text=True,
                              timeout=self.args.wait_timeout)
        (self.run_dir / f"wait-{label}.log").write_text(wait.stdout + wait.stderr)
        if wait.returncode != 0:
            raise AbError(f"wait failed: {label}")
        pid_file = Path(self.args.service_log_root) / label / "api.pid"
        if not pid_file.is_file():
            raise AbError(f"api.pid missing: {pid_file}")
        root_pid = int(pid_file.read_text().strip())
        identity = self.win.proc_identity(root_pid)
        if identity is None:
            raise AbError(f"root identity missing: {root_pid}")
        return {"root": identity, "descendants": self.win.descendants(root_pid)}

    def restore_prod(self):
        label = f"d0-restore-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}-{uuid.uuid4().hex[:8]}"
        env = dict(self.prod_env)
        for key in CANDIDATE_FLAGS:
            env.pop(key, None)
        env["CLEF_LAUNCH_CONFIG"] = "s1"
        before = {int(p.name) for p in self.win.PROC_ROOT.iterdir() if p.name.isdigit()}
        launch = subprocess.run(["bash", self.args.launch, label], env=env,
                                capture_output=True, text=True, timeout=60)
        (self.run_dir / "restore-launch.log").write_text(launch.stdout + launch.stderr)
        if launch.returncode != 0:
            raise AbError("restore launch failed")
        wait = subprocess.run(["bash", self.args.wait], capture_output=True, text=True,
                              timeout=self.args.wait_timeout)
        (self.run_dir / "restore-wait.log").write_text(wait.stdout + wait.stderr)
        if wait.returncode != 0:
            raise AbError("restore wait failed")
        sys.path.insert(0, str(self.run_dir / "stage" / "d0"))
        import d0_restore
        import d0_run
        disc = d0_restore.discover_service_pids(label, before)
        if disc.get("error") or not disc.get("pids"):
            raise AbError(f"restore discovery failed: {disc.get('error')}")
        log = {"label": label, "service_discovery": disc}
        verified = d0_restore.verify(
            d0_restore.build_commands(d0_run.render_request), log,
            service_pids=set(disc["pids"]))
        (self.run_dir / "restore.json").write_text(
            json.dumps(log, ensure_ascii=False, indent=1) + "\n")
        if not verified:
            raise AbError("restore four-check verification failed")
        return {"label": label, "verify_ok": True, "service_pids": sorted(disc["pids"])}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--repo", default=str(HERE.parent))
    ap.add_argument("--tree", default="/workspace/yangchen/work/src/vllm")
    ap.add_argument("--python", default="/workspace/yangchen/venvs/clef-py312/bin/python")
    ap.add_argument("--activate", default="/workspace/yangchen/Clef-27B-ascend/scripts/activate_env.sh")
    ap.add_argument("--launch", default="/workspace/yangchen/logs/perf-wave2-20261118/launch.sh")
    ap.add_argument("--wait", default="/workspace/yangchen/logs/perf-wave2-20261118/wait.sh")
    ap.add_argument("--stop-script", default="/tmp/d0cc/d0_stop.py")
    ap.add_argument("--stop-timeout", type=float, default=120.0)
    ap.add_argument("--wait-timeout", type=float, default=900.0)
    ap.add_argument("--http-timeout", type=float, default=180.0)
    ap.add_argument("--port", type=int, default=18143)
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--short", default="od-0105")
    ap.add_argument("--long", default="od-1534")
    ap.add_argument("--data", default="/workspace/yangchen/logs/tuning/20261114-013000-fix4/opendecisions/all2352.json")
    ap.add_argument("--image-source", default="/workspace/yangchen/logs/perf-wave2-20261118/erqa/samples/erqa-0000/request.json")
    ap.add_argument("--text-url", default="http://127.0.0.1:18143/v1/systemone")
    ap.add_argument("--image-url", default="http://127.0.0.1:18143/v1/decide")
    ap.add_argument("--service-log-root", default="/workspace/yangchen/logs/perf-wave2-20261118")
    args = ap.parse_args()
    if Path(args.run_dir).expanduser().resolve().exists():
        print(json.dumps({"error": "run dir exists"}))
        return 1
    if args.rounds < 4:
        print(json.dumps({"error": "rounds must be at least 4"}))
        return 1
    ab = ServiceAB(args)
    ab.state = {"schema": "joint-fused-service-ab/1", "at": now(),
                "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                "rounds": [], "steps": {}, "overall_rc": 1}
    original = None
    applied = False
    arm = None
    initial_stopped = False
    try:
        ab.run_dir.mkdir(parents=True, exist_ok=False)
        ab.stage_d0()
        if ab.win.sha256(Path(args.stop_script)) != STOP_SHA:
            raise AbError("stop script SHA mismatch")
        for path, expected, name in ((args.activate, ACTIVATE_SHA, "activate"),
                                     (args.launch, LAUNCH_SHA, "launch"),
                                     (args.wait, WAIT_SHA, "wait")):
            if ab.win.sha256(Path(path)) != expected:
                raise AbError(f"{name} SHA mismatch")
        current, ab.prod_env = ab.current_service()
        ab.state["current_root"] = current["root"]["pid"]
        ab.state["prod_env_keys"] = sorted(ab.win.selected_env(ab.prod_env))
        ab.inputs = ab.input_bodies()
        ab.flush("preflight")
        ab.stop_tree(current, "initial")
        initial_stopped = True
        ab.state["steps"]["initial_stop"] = {"ok": True}
        ab.flush("initial-stopped")
        applied = True
        ab.apply_postimage()
        ab.state["steps"]["apply"] = {"ok": True}
        ab.flush("applied")
        for index in range(args.rounds):
            order = ORDERS[index % len(ORDERS)]
            row = {"round": index, "order": order, "arms": {}}
            for flag in order:
                label = f"d0-ab-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}-{index}-{flag}-{uuid.uuid4().hex[:6]}"
                arm = ab.start_arm(flag, label)
                samples = ab.run_inputs(label)
                ab.stop_tree(arm, label)
                arm = None
                row["arms"][str(flag)] = {"label": label, "samples": samples}
                ab.flush(f"round-{index}-flag-{flag}")
            ab.state["rounds"].append(row)
            ab.flush(f"round-{index}")
    except Exception as exc:
        ab.state["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        stopped = initial_stopped
        if arm is not None:
            try:
                ab.stop_tree(arm, "cleanup-arm")
                stopped = True
            except Exception as exc:
                ab.state["cleanup_stop_error"] = f"{type(exc).__name__}: {exc}"
                stopped = False
        port_state = ab.win.port_open(args.port)
        if stopped and port_state != "closed":
            stopped = False
            ab.state["service_state"] = {"stopped": False, "port": port_state}
        try:
            if applied:
                ab.revert_postimage()
                ab.state["steps"]["revert"] = {"ok": True}
        except Exception as exc:
            ab.state["steps"]["revert"] = {"ok": False,
                                           "error": f"{type(exc).__name__}: {exc}"}
        if stopped:
            try:
                restored = ab.restore_prod()
                ab.state["steps"]["restore"] = {"ok": True, **restored}
            except Exception as exc:
                ab.state["steps"]["restore"] = {"ok": False,
                                                "error": f"{type(exc).__name__}: {exc}"}
        else:
            ab.state["steps"]["restore"] = {
                "ok": False, "action": "blocked_service_not_proven_stopped",
                "port": port_state}
        ab.state["finished_at"] = now()
        ab.state["overall_rc"] = 0 if (
            not ab.state.get("error")
            and ab.state["steps"].get("revert", {}).get("ok")
            and ab.state["steps"].get("restore", {}).get("ok")
            and len(ab.state["rounds"]) == args.rounds) else 1
        ab.flush("finished")
    print(json.dumps({k: ab.state.get(k) for k in ("schema", "at", "error", "steps",
                                                    "overall_rc")},
                     ensure_ascii=False, indent=1))
    return ab.state["overall_rc"]


if __name__ == "__main__":
    raise SystemExit(main())
