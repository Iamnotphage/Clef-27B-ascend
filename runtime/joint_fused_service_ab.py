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
INSTALLER_SHA = "b09cd6025b0aa63ae0410781f3c611e9d512f7ef939cb78dab41b8413a19d406"
WINDOW_SHA = "700cf10cdc3fda18d1ae7f113158342e1ef6ec114569e52b15b134949bfe8473"
SERVICE_CHECK_INSTALLER_SHA = "1ed35f0a24f8b23d8ac099992d703aca20d7f1460c89ff02c30635dfde6a219c"
SERVICE_CHECK_SOURCE_SHA = "36eaff440fd58063a3311b2efabfd1604b4a284cd24f3b7ac3cc4e7fe177fcc7"
SERVICE_CHECK_PREIMAGE_SHA = "76595657d312699b4b1564d1bc054229808cd775c16702fb0e36fcdb1f9706d9"
SERVICE_ROOT = "/workspace/yangchen/Clef-27B-ascend"
ALLOW_ENV = "CLEF_JOINT_ALLOWED_POSTIMAGE_SHA"
ACTIVATE_SHA = "3b363394536fbe2433a84b666170568fcb58848b1d8d24551ff6ae9f53b4044a"
LAUNCH_SHA = "a737386119b89c5bc787e6f69e5017b83391934148528d08d498ff943a958321"
WAIT_SHA = "e3fb08882211aef6ea0fd4f3db3d37bba0a591034cc397ba95559e09d1f1ffa0"
TEXT_SHA = {
    "od-0105": "25b8acaf74a0d2b9fc210e08687e3a08c3d5fff82e6785c20611418e8cc72bf0",
    "od-1534": "568ea1583377dcc6044d3fcd4a099c3beb4cd0e118155f95b595bda72b1c76bf",
}
IMAGE_SHA = "f3229f5c120098ac4eebcc6f4025c28254e35f959fd01998cac5db7b8d209aab"
CANDIDATE_FLAGS = {"CLEF_JOINT_FUSED_ATTN", "CLEF_JOINT_SMALLQ",
                   "CLEF_JOINT_SMALLQ_KERNEL", ALLOW_ENV}
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
        self.attribution_unknown = False
        self.cleanup_unconfirmed = False
        self.service_check_applied = False

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

    def stage_candidate(self):
        root = self.run_dir / "stage" / "runtime"
        root.mkdir(parents=True, exist_ok=True)
        shutil.copy2(Path(self.args.repo) / "runtime" / "apply_joint_fused_attention.py",
                     root / "apply_joint_fused_attention.py")
        (root / "joint_fused_attention").mkdir(parents=True, exist_ok=True)
        shutil.copy2(Path(self.args.repo) / "runtime" / "joint_fused_attention" /
                     "clef_joint_schema.py", root / "joint_fused_attention" /
                     "clef_joint_schema.py")
        if self.win.sha256(root / "apply_joint_fused_attention.py") != INSTALLER_SHA:
            raise AbError("staged installer SHA mismatch")
        if self.win.sha256(root / "joint_fused_attention" / "clef_joint_schema.py") != POSTIMAGE_SHA:
            raise AbError("staged postimage SHA mismatch")
        if self.win.sha256(Path(self.win.__file__)) != WINDOW_SHA:
            raise AbError("helper window SHA mismatch")
        shutil.copy2(Path(self.args.repo) / "runtime" / "apply_joint_service_check.py",
                     root / "apply_joint_service_check.py")
        (root / "joint_service_check").mkdir(parents=True, exist_ok=True)
        shutil.copy2(Path(self.args.repo) / "runtime" / "joint_service_check" /
                     "apply_joint_single_field_patch.py",
                     root / "joint_service_check" /
                     "apply_joint_single_field_patch.py")
        if self.win.sha256(root / "apply_joint_service_check.py") != SERVICE_CHECK_INSTALLER_SHA:
            raise AbError("staged service-check installer SHA mismatch")
        if self.win.sha256(root / "joint_service_check" /
                           "apply_joint_single_field_patch.py") != SERVICE_CHECK_SOURCE_SHA:
            raise AbError("staged service-check source SHA mismatch")
        return root

    def apply_service_check(self):
        installer = self.run_dir / "stage" / "runtime" / "apply_joint_service_check.py"
        target = Path(SERVICE_ROOT) / "runtime" / "apply_joint_single_field_patch.py"
        try:
            check = self.call([self.args.python, str(installer),
                               "--service-root", SERVICE_ROOT, "--check"])
            report = json.loads(check.stdout)
            if report.get("state_after") == "preimage":
                self.call([self.args.python, str(installer),
                           "--service-root", SERVICE_ROOT, "--apply",
                           "--manifest", str(self.run_dir / "service-check-apply.json")])
            elif report.get("state_after") != "postimage":
                raise AbError(f"service-check state: {report.get('state_after')}")
        finally:
            # A successful write followed by a manifest/child failure must still
            # be reverted by finally.
            self.service_check_applied = (
                target.is_file() and self.win.sha256(target) == SERVICE_CHECK_SOURCE_SHA)
        if not self.service_check_applied:
            raise AbError("service-check adapter not installed")

    def revert_service_check(self):
        installer = self.run_dir / "stage" / "runtime" / "apply_joint_service_check.py"
        target = Path(SERVICE_ROOT) / "runtime" / "apply_joint_single_field_patch.py"
        try:
            self.call([self.args.python, str(installer),
                       "--service-root", SERVICE_ROOT, "--revert",
                       "--manifest", str(self.run_dir / "service-check-revert.json")])
        finally:
            self.service_check_applied = (
                target.is_file() and self.win.sha256(target) == SERVICE_CHECK_SOURCE_SHA)
        if not target.is_file() or self.win.sha256(target) != SERVICE_CHECK_PREIMAGE_SHA:
            raise AbError("service-check adapter did not restore original guard")

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

    def stopped_verified(self, identity):
        window = object.__new__(self.win.Window)
        window.identity = identity
        return window.service_stopped_verified()

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
        verified, detail = self.stopped_verified(identity)
        if not verified:
            raise AbError(f"saved service not proven stopped {label}: {detail}")
        if self.win.port_open(self.args.port) != "closed":
            raise AbError(f"port still open after stop {label}")
        return {"stop_json": str(out), "stopped": detail}

    def apply_postimage(self):
        installer = self.run_dir / "stage" / "runtime" / "apply_joint_fused_attention.py"
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
        installer = self.run_dir / "stage" / "runtime" / "apply_joint_fused_attention.py"
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
        raw_dir = self.run_dir / "raw"
        raw_dir.mkdir(exist_ok=True)
        rows = {}
        for name, item in self.inputs.items():
            raw_path = raw_dir / f"{label}-{name}.jsonl"
            warmup_rows, samples = [], []
            with raw_path.open("w") as stream:
                for index in range(self.args.warmup + self.args.reps):
                    started = now()
                    begin = time.perf_counter()
                    status, payload, error = None, None, None
                    try:
                        request = urllib.request.Request(
                            item["url"], data=item["body"],
                            headers={"Content-Type": "application/json"})
                        with urllib.request.urlopen(
                                request, timeout=self.args.http_timeout) as response:
                            status = response.status
                            payload = json.loads(response.read())
                    except Exception as exc:
                        error = f"{type(exc).__name__}: {exc}"
                    elapsed = (time.perf_counter() - begin) * 1000.0
                    answer = ((payload or {}).get("answers") or {}).get("answer") or {}
                    valid = (status == 200 and error is None
                             and answer.get("type") == item["expect"])
                    row = {"index": index, "warmup": index < self.args.warmup,
                           "body_sha256": item["sha256"], "kind": item["kind"],
                           "started_at": started, "finished_at": now(),
                           "ms": elapsed, "status": status,
                           "answer_type": answer.get("type"), "valid": valid,
                           "response": payload, "error": error}
                    stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                    stream.flush()
                    (warmup_rows if row["warmup"] else samples).append(row)
            if not samples or not all(row["valid"] for row in samples):
                raise AbError(f"invalid response for {name} in {label}; raw={raw_path}")
            rows[name] = {"samples": samples, "warmup": warmup_rows,
                          "raw": str(raw_path)}
        return rows

    def start_arm(self, flag, label):
        try:
            return self._start_arm(flag, label)
        except Exception as exc:
            identity = getattr(self, "arm_identity", None)
            if identity is None or self.attribution_unknown:
                self.state["start_cleanup_error"] = (
                    f"{type(exc).__name__}: no label-bound identity; no signal sent")
                self.cleanup_unconfirmed = True
                raise
            try:
                self.stop_tree(identity, f"failed-{label}")
                self.arm_identity = None
            except Exception as stop_exc:
                self.state["start_cleanup_error"] = (
                    f"{type(stop_exc).__name__}: {stop_exc}")
                self.cleanup_unconfirmed = True
            raise

    def _start_arm(self, flag, label):
        env = dict(self.prod_env)
        for key in CANDIDATE_FLAGS:
            env.pop(key, None)
        if flag is not None:
            env["CLEF_JOINT_FUSED_ATTN"] = str(flag)
        env[ALLOW_ENV] = POSTIMAGE_SHA
        env["CLEF_LAUNCH_CONFIG"] = "s1"
        before = {int(p.name) for p in self.win.PROC_ROOT.iterdir() if p.name.isdigit()}
        self.arm_identity = None
        self.arm_label = label
        launch = subprocess.run(["bash", self.args.launch, label], env=env,
                                capture_output=True, text=True, timeout=60)
        (self.run_dir / f"launch-{label}.log").write_text(launch.stdout + launch.stderr)
        if launch.returncode != 0:
            if self.win.port_open(self.args.port) != "closed":
                self.attribution_unknown = True
            raise AbError(f"launch failed: {label}")
        # Bind this label's ownership immediately, before wait/inputs.
        pid_file = Path(self.args.service_log_root) / label / "api.pid"
        if not pid_file.is_file():
            self.attribution_unknown = True
            raise AbError(f"api.pid missing immediately after launch: {pid_file}")
        root_pid = int(pid_file.read_text().strip())
        identity = self.win.proc_identity(root_pid)
        after = {int(p.name) for p in self.win.PROC_ROOT.iterdir() if p.name.isdigit()}
        if identity is None or root_pid not in (after - before):
            self.attribution_unknown = True
            raise AbError(f"root {root_pid} is not a new label-bound process")
        self.arm_identity = {"root": identity,
                             "descendants": self.win.descendants(root_pid)}
        wait = subprocess.run(["bash", self.args.wait], capture_output=True, text=True,
                              timeout=self.args.wait_timeout)
        (self.run_dir / f"wait-{label}.log").write_text(wait.stdout + wait.stderr)
        if wait.returncode != 0:
            raise AbError(f"wait failed: {label}")
        live = self.win.proc_identity(root_pid)
        if (live is None or live["starttime"] != identity["starttime"]
                or live["boot_id"] != identity["boot_id"]):
            self.attribution_unknown = True
            raise AbError(f"root identity changed during wait: {root_pid}")
        root_env = self.win.parse_environ(
            (self.win.PROC_ROOT / f"{root_pid}/environ").read_bytes())
        source_sha = self.win.sha256(Path(self.args.tree) /
                                     "vllm/model_executor/models/clef_joint_schema.py")
        if root_env.get("CLEF_JOINT_FUSED_ATTN") != str(flag):
            raise AbError(f"arm {label} root env flag mismatch: {root_env.get('CLEF_JOINT_FUSED_ATTN')}")
        if root_env.get("CLEF_LAUNCH_CONFIG") != "s1":
            raise AbError(f"arm {label} launch config mismatch: {root_env.get('CLEF_LAUNCH_CONFIG')}")
        if root_env.get(ALLOW_ENV) != POSTIMAGE_SHA:
            raise AbError(f"arm {label} service-check env mismatch: {root_env.get(ALLOW_ENV)}")
        if source_sha != POSTIMAGE_SHA:
            raise AbError(f"arm {label} source SHA mismatch: {source_sha}")
        npu = subprocess.run(["npu-smi", "info"], capture_output=True, text=True,
                             timeout=30)
        (self.run_dir / f"npu-{label}.txt").write_text(npu.stdout + npu.stderr)
        return {"root": identity, "descendants": self.arm_identity["descendants"],
                "label": label, "before_count": len(before),
                "after_new": sorted(after - before),
                "env_flag": root_env.get("CLEF_JOINT_FUSED_ATTN"),
                "launch_config": root_env.get("CLEF_LAUNCH_CONFIG"),
                "service_check_env": root_env.get(ALLOW_ENV),
                "source_sha256": source_sha, "npu_file": str(self.run_dir / f"npu-{label}.txt")}

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
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--short", default="od-0105")
    ap.add_argument("--long", default="od-1534")
    ap.add_argument("--data", default="/workspace/yangchen/logs/tuning/20261114-013000-fix4/opendecisions/all2352.json")
    ap.add_argument("--image-source", default="/workspace/yangchen/logs/perf-wave2-20261118/erqa/samples/erqa-0000/request.json")
    ap.add_argument("--text-url", default="http://127.0.0.1:18143/v1/systemone")
    ap.add_argument("--image-url", default="http://127.0.0.1:18143/v1/decide")
    ap.add_argument("--service-log-root", default="/workspace/yangchen/logs/perf-wave2-20261118")
    ap.add_argument("--restore-only", action="store_true",
                    help="restore baseline f810 service only; no install/AB")
    args = ap.parse_args()
    if Path(args.run_dir).expanduser().resolve().exists():
        print(json.dumps({"error": "run dir exists"}))
        return 1
    if args.rounds < 4:
        print(json.dumps({"error": "rounds must be at least 4"}))
        return 1
    if args.restore_only:
        if Path(args.run_dir).expanduser().resolve().exists():
            print(json.dumps({"error": "run dir exists"}))
            return 1
        ab = ServiceAB(args)
        ab.state = {"schema": "joint-fused-service-ab/1", "at": now(),
                    "mode": "restore-only", "steps": {}, "overall_rc": 1}
        try:
            ab.run_dir.mkdir(parents=True, exist_ok=False)
            ab.stage_d0()
            target = Path(args.tree) / "vllm/model_executor/models/clef_joint_schema.py"
            if not target.is_file() or ab.win.sha256(target) != PREIMAGE_SHA:
                raise AbError("head target is not f8101b98")
            for path, expected, name in ((args.activate, ACTIVATE_SHA, "activate"),
                                         (args.launch, LAUNCH_SHA, "launch"),
                                         (args.wait, WAIT_SHA, "wait")):
                if ab.win.sha256(Path(path)) != expected:
                    raise AbError(f"{name} SHA mismatch")
            if ab.win.port_open(args.port) != "closed":
                raise AbError("service port is open; stop it with the exact identity first")
            ab.prod_env = ab.source_env()
            for key in CANDIDATE_FLAGS:
                ab.prod_env.pop(key, None)
            restored = ab.restore_prod()
            ab.state["steps"]["restore"] = {"ok": True, **restored}
            ab.state["overall_rc"] = 0
        except Exception as exc:
            ab.state["error"] = f"{type(exc).__name__}: {exc}"
            ab.state["overall_rc"] = 1
        ab.state["finished_at"] = now()
        ab.flush("finished")
        print(json.dumps({k: ab.state.get(k) for k in ("mode", "error", "steps", "overall_rc")},
                         ensure_ascii=False, indent=1))
        return ab.state["overall_rc"]
    ab = ServiceAB(args)
    ab.state = {"schema": "joint-fused-service-ab/1", "at": now(),
                "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                "rounds": [], "steps": {}, "overall_rc": 1}
    original = None
    applied = False
    arm = None
    initial_stopped = False
    candidate_started = False
    stop_attempted = False
    try:
        ab.run_dir.mkdir(parents=True, exist_ok=False)
        ab.stage_d0()
        ab.stage_candidate()
        target = Path(args.tree) / "vllm/model_executor/models/clef_joint_schema.py"
        target_sha = ab.win.sha256(target) if target.is_file() else None
        if target_sha != PREIMAGE_SHA:
            raise AbError(f"head target is not preimage at preflight: {target_sha}")
        service_guard = Path(SERVICE_ROOT) / "runtime" / "apply_joint_single_field_patch.py"
        if not service_guard.is_file() or ab.win.sha256(service_guard) != SERVICE_CHECK_PREIMAGE_SHA:
            raise AbError("service start guard is not the original preimage before stop")
        if ab.win.sha256(Path(args.stop_script)) != STOP_SHA:
            raise AbError("stop script SHA mismatch")
        for path, expected, name in ((args.activate, ACTIVATE_SHA, "activate"),
                                     (args.launch, LAUNCH_SHA, "launch"),
                                     (args.wait, WAIT_SHA, "wait")):
            if ab.win.sha256(Path(path)) != expected:
                raise AbError(f"{name} SHA mismatch")
        current, ab.prod_env = ab.current_service()
        initial_identity = current
        ab.state["current_root"] = current["root"]["pid"]
        ab.state["prod_env_keys"] = sorted(ab.win.selected_env(ab.prod_env))
        ab.inputs = ab.input_bodies()
        ab.flush("preflight")
        candidate_started = True
        stop_attempted = True
        ab.stop_tree(current, "initial")
        initial_stopped = True
        ab.state["steps"]["initial_stop"] = {"ok": True}
        ab.flush("initial-stopped")
        applied = True
        ab.apply_postimage()
        ab.apply_service_check()
        ab.state["steps"]["apply"] = {"ok": True}
        ab.state["steps"]["service_check_apply"] = {"ok": True}
        ab.flush("applied")
        for index in range(args.rounds):
            order = ORDERS[index % len(ORDERS)]
            row = {"round": index, "order": order, "arms": {}}
            ab.state["rounds"].append(row)
            ab.flush(f"round-{index}-start")
            for flag in order:
                label = f"d0-ab-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}-{index}-{flag}-{uuid.uuid4().hex[:6]}"
                arm = ab.start_arm(flag, label)
                provenance = arm
                samples = ab.run_inputs(label)
                stop_info = ab.stop_tree(arm, label)
                arm = None
                ab.arm_identity = None
                row["arms"][str(flag)] = {"label": label, "provenance": provenance,
                                          "samples": samples, "stop": stop_info}
                ab.flush(f"round-{index}-flag-{flag}")
    except Exception as exc:
        ab.state["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        target = Path(args.tree) / "vllm/model_executor/models/clef_joint_schema.py"
        target_sha = ab.win.sha256(target) if target.is_file() else None
        revert_ok = True
        if not candidate_started:
            ab.state["steps"]["finish"] = {"ok": True, "action": "read_only_noop",
                                           "finished_at": now()}
            ab.state["finished_at"] = now()
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
        cleanup_identity = arm or getattr(ab, "arm_identity", None)
        if cleanup_identity is not None:
            try:
                ab.stop_tree(cleanup_identity, "cleanup-arm")
                stopped = True
            except Exception as exc:
                ab.state["cleanup_stop_error"] = f"{type(exc).__name__}: {exc}"
                stopped = False
                ab.cleanup_unconfirmed = True
            arm = None
            ab.arm_identity = None
        if ab.attribution_unknown or ab.cleanup_unconfirmed:
            stopped = False
            ab.state["service_attribution"] = {
                "unknown": ab.attribution_unknown,
                "cleanup_unconfirmed": ab.cleanup_unconfirmed,
            }
        if applied:
            try:
                ab.revert_postimage()
                revert_ok = True
                ab.state["steps"]["revert"] = {"ok": True}
            except Exception as exc:
                revert_ok = False
                ab.state["steps"]["revert"] = {"ok": False,
                                               "error": f"{type(exc).__name__}: {exc}"}
            target_sha = ab.win.sha256(target) if target.is_file() else None
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
            service_target = Path(SERVICE_ROOT) / "runtime" / "apply_joint_single_field_patch.py"
            service_check_ok = (service_target.is_file()
                                and ab.win.sha256(service_target) == SERVICE_CHECK_PREIMAGE_SHA)
        ab.state["target_sha256"] = target_sha
        if stopped and ab.win.port_open(args.port) != "closed":
            stopped = False
        if target_sha != PREIMAGE_SHA or not revert_ok or not service_check_ok:
            ab.state["steps"]["restore"] = {
                "ok": False, "action": "blocked_source_or_revert_not_clean",
                "target_sha256": target_sha, "revert_ok": revert_ok,
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
