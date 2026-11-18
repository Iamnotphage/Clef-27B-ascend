#!/usr/bin/env python3
"""One bounded real-execution window for the joint-head fused-attention candidate.

Execution owner: 21.  This wrapper performs the whole window so the device
operator does not have to stitch shell commands:

  preflight (env/loader/presence/hash/preimage/LISTEN/identity)
  -> exact stop (21's d0_stop.py)
  -> apply candidate postimage
  -> probe (own process group, bounded timeout, raw logs)
  -> revert to f8101b98
  -> restore with a fresh label and fixed s1, then health/2-worker/text/image

Every step records its own rc and writes raw logs into a new empty run directory.
A failed step stops the candidate path; the finally block still reverts and
restores.  The production wrapper functions are not mocked.
"""

from __future__ import annotations

import argparse
import errno
import hashlib
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.request
import uuid
from pathlib import Path

POSTIMAGE_REL = "runtime/joint_fused_attention/clef_joint_schema.py"
INSTALLER_REL = "runtime/apply_joint_fused_attention.py"
PROBE_REL = "runtime/joint_fused_attention_probe.py"
CLIENT_REL = "runtime/joint_fused_attention_client.py"
POSTIMAGE_SHA = "aa3e17585f8bd60a4d05e342abc7e2773c79a0b2d49f0370d15e29bcc6b72503"
INSTALLER_SHA = "b09cd6025b0aa63ae0410781f3c611e9d512f7ef939cb78dab41b8413a19d406"
PROBE_SHA = "441764eb13866f9cdfc90247c1e4ec9992592d63bffd72e3f359e4a0526d80aa"
CLIENT_SHA = "9f7f700c9b0c16c62a5951e1e0b0a1954e82a3f55a4683b616ab417bb2630777"
PREIMAGE_SHA = "f8101b98228bbd472a63a1efe6827583e35c6cc9dede4379b925f2b016670863"
ACTIVATE_DEFAULT = "/workspace/yangchen/Clef-27B-ascend/scripts/activate_env.sh"
ACTIVATE_SHA = "3b363394536fbe2433a84b666170568fcb58848b1d8d24551ff6ae9f53b4044a"
LAUNCH_DEFAULT = "/workspace/yangchen/logs/perf-wave2-20261118/launch.sh"
LAUNCH_SHA = "a737386119b89c5bc787e6f69e5017b83391934148528d08d498ff943a958321"
WAIT_DEFAULT = "/workspace/yangchen/logs/perf-wave2-20261118/wait.sh"
WAIT_SHA = "e3fb08882211aef6ea0fd4f3db3d37bba0a591034cc397ba95559e09d1f1ffa0"
STOP_DEFAULT = "/tmp/d0cc/d0_stop.py"
DEFAULT_TREE = Path("/workspace/yangchen/work/src/vllm")
DEFAULT_PYTHON = "/workspace/yangchen/venvs/clef-py312/bin/python"
SERVICE_LOG_ROOT = "/workspace/yangchen/logs/perf-wave2-20261118"
PROC_ROOT = Path("/proc")
CANDIDATE_KEYS = {"CLEF_JOINT_FUSED_ATTN", "CLEF_JOINT_SMALLQ",
                  "CLEF_JOINT_SMALLQ_KERNEL"}
CLEANUP_KEYS = ("CLEF_PROFILE_MARKERS", "CLEF_PROFILE_OBSERVE_DIR",
                "CLEF_PROFILE_SCOPE_FILTER", "CLEF_GDN_NORM_BLOCK_M")
ENV_PREFIXES = ("CLEF_", "ASCEND_", "ATB_", "HCCL_", "FLA_")
ENV_KEYS = {"PATH", "LD_LIBRARY_PATH", "PYTHONPATH", "PYTHONUNBUFFERED",
            "VIRTUAL_ENV", "OMP_NUM_THREADS", "TASK_QUEUE_ENABLE", "MODEL_DIR",
            "HF_HOME", "TORCH_HOME", "TRITON_CACHE_DIR", "XDG_CACHE_HOME",
            "ACLNN_CACHE_LIMIT"}


class StepError(RuntimeError):
    def __init__(self, message: str, rc: int = 1, detail: dict | None = None):
        super().__init__(message)
        self.rc = rc
        self.detail = detail or {}


def json_safe(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def parse_environ(blob: bytes) -> dict:
    out = {}
    for item in blob.split(b"\0"):
        if not item or b"=" not in item:
            continue
        key, value = item.split(b"=", 1)
        out[key.decode(errors="replace")] = value.decode(errors="replace")
    return out


def proc_identity(pid: int) -> dict | None:
    try:
        raw = (PROC_ROOT / f"{pid}/stat").read_text()
        fields = raw.split(") ", 1)[1].split()
        ppid, pgrp, start = int(fields[1]), int(fields[2]), fields[19]
    except (OSError, IndexError, ValueError):
        return None
    try:
        boot = (PROC_ROOT / "sys/kernel/random/boot_id").read_text().strip()
    except OSError:
        boot = ""
    try:
        cmdline = (PROC_ROOT / f"{pid}/cmdline").read_bytes().replace(
            b"\0", b" ").decode(errors="replace").strip()
    except OSError:
        cmdline = ""
    return {"pid": pid, "ppid": ppid, "pgid": pgrp, "starttime": start,
            "boot_id": boot, "cmdline": cmdline, "state": fields[0]}


def selected_env(env: dict) -> dict:
    return {key: value for key, value in env.items()
            if key in ENV_KEYS or key.startswith(ENV_PREFIXES)}


def descendants(root: int) -> list[dict]:
    children: dict[int, list[int]] = {}
    identities: dict[int, dict] = {}
    for entry in PROC_ROOT.iterdir():
        if not entry.name.isdigit():
            continue
        ident = proc_identity(int(entry.name))
        if ident is None:
            continue
        identities[int(entry.name)] = ident
        children.setdefault(ident["ppid"], []).append(int(entry.name))
    out, stack = [], [root]
    while stack:
        current = stack.pop()
        for child in children.get(current, []):
            if child in {item["pid"] for item in out}:
                continue
            out.append(identities[child])
            stack.append(child)
    return out


def identity_matches(saved: dict, live: dict | None) -> bool:
    return bool(saved and live and saved["pid"] == live["pid"]
                and saved["starttime"] == live["starttime"]
                and saved["boot_id"] == live["boot_id"])


def group_members(pgid: int) -> dict[int, dict]:
    out = {}
    for entry in PROC_ROOT.iterdir():
        if not entry.name.isdigit():
            continue
        ident = proc_identity(int(entry.name))
        if ident is not None and ident["pgid"] == pgid:
            out[ident["pid"]] = ident
    return out


def port_open(port: int) -> str:
    """Return 'open', 'closed' (ECONNREFUSED), or 'unknown'."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(2)
    try:
        result = sock.connect_ex(("127.0.0.1", port))
    except OSError as exc:
        return f"unknown:{type(exc).__name__}"
    finally:
        sock.close()
    if result == 0:
        return "open"
    if result == errno.ECONNREFUSED:
        return "closed"
    return f"unknown:errno{result}"


def find_api_pid(port: int) -> list[int]:
    needle = f"--port {port}"
    found = []
    for entry in PROC_ROOT.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            cmd = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode(
                errors="replace")
        except OSError:
            continue
        if "vllm.entrypoints.cli.main" in cmd and needle in cmd:
            found.append(int(entry.name))
    return found


class Window:
    def cfg(self, name, default):
        return getattr(self.args, name, default)

    @property
    def external(self) -> bool:
        return self.cfg("candidate_kind", "head-postimage") == "external-vendor"

    def __init__(self, args):
        self.args = args
        self.run_dir = Path(args.run_dir).expanduser().resolve()
        self.stage = self.run_dir / "stage"
        self.state = {"schema": "joint-fused-window/1", "run_dir": str(self.run_dir),
                      "started_at": now(), "steps": {}, "args": json_safe(vars(args))}
        self.applied = False
        self.stopped = False
        self.candidate_started = False
        self.stop_attempted = False
        self.restored = False
        self.probe_residual = False
        self.vendor_applied = False
        self.identity = None
        self.prod_env = {}
        self.source_env = {}
        self.overall_failed = False

    def flush(self):
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "window.json").write_text(
            json.dumps(self.state, ensure_ascii=False, indent=1, default=str) + "\n")

    def step(self, name: str, required: bool = True):
        started = now()
        try:
            result = getattr(self, name)()
            entry = {"rc": 0, "ok": True, "started_at": started,
                     "finished_at": now(), **(result or {})}
        except StepError as exc:
            entry = {"rc": exc.rc, "ok": False, "started_at": started,
                     "finished_at": now(), "error": str(exc), **exc.detail}
            if required:
                self.overall_failed = True
        except Exception as exc:
            entry = {"rc": 1, "ok": False, "started_at": started,
                     "finished_at": now(),
                     "error": f"{type(exc).__name__}: {exc}"}
            if required:
                self.overall_failed = True
        self.state["steps"][name] = entry
        self.flush()
        return entry

    # ------------------------------------------------------------------
    def preflight(self):
        if self.run_dir.exists():
            raise StepError(f"run dir already exists: {self.run_dir}")
        self.run_dir.mkdir(parents=True)
        self.stage.mkdir()
        repo = Path(self.args.repo)
        def source_arg(name, default):
            value = getattr(self.args, name, None)
            return Path(value) if value else default
        if self.external:
            vendor_rel = self.cfg("vendor_installer_rel", "")
            if not vendor_rel:
                raise StepError("external-vendor requires --vendor-installer-rel")
            vendor_sha = self.cfg("vendor_installer_sha", None)
            if not vendor_sha:
                raise StepError("external-vendor requires --vendor-installer-sha")
            spec = {
                vendor_rel: (source_arg("vendor_installer_source", repo / vendor_rel),
                             vendor_sha),
                self.cfg("probe_rel", PROBE_REL):
                    (source_arg("probe_source", repo / PROBE_REL),
                     self.cfg("probe_sha", PROBE_SHA)),
            }
        else:
            spec = {
                self.cfg("postimage_rel", POSTIMAGE_REL):
                    (source_arg("postimage_source", repo / POSTIMAGE_REL),
                     self.cfg("postimage_sha", POSTIMAGE_SHA)),
                self.cfg("installer_rel", INSTALLER_REL):
                    (source_arg("installer_source", repo / INSTALLER_REL),
                     self.cfg("installer_sha", INSTALLER_SHA)),
                self.cfg("probe_rel", PROBE_REL):
                    (source_arg("probe_source", repo / PROBE_REL),
                     self.cfg("probe_sha", PROBE_SHA)),
            }
        client_rel = self.cfg("client_rel", CLIENT_REL)
        if client_rel and not self.external:
            spec[client_rel] = (source_arg("client_source", repo / client_rel),
                                self.cfg("client_sha", CLIENT_SHA))
        for payload in self.cfg("payload", []) or []:
            try:
                source, rel, digest = payload.split(":", 2)
            except ValueError:
                raise StepError(f"invalid --payload (SRC:REL:SHA): {payload}")
            spec[rel] = (Path(source), digest)
        staged = {}
        for rel, (source, expected) in spec.items():
            if not source.is_file():
                raise StepError(f"missing candidate file: {source}")
            actual = sha256(source)
            if actual != expected:
                raise StepError(f"candidate SHA mismatch: {rel}: {actual}")
            target = self.stage / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            if sha256(target) != expected:
                raise StepError(f"staged SHA mismatch: {rel}")
            staged[rel] = expected
        for path, expected, label in (
                (self.args.activate, ACTIVATE_SHA, "activate_env"),
                (self.args.launch, LAUNCH_SHA, "launch"),
                (self.args.wait, WAIT_SHA, "wait")):
            actual = sha256(Path(path))
            if actual != expected:
                raise StepError(f"{label} SHA mismatch: {actual}")
        stop_path = Path(self.args.stop_script)
        if not stop_path.is_file():
            raise StepError(f"stop script missing: {stop_path}")
        if not self.args.expect_stop_sha:
            raise StepError("--expect-stop-sha is required for this window")
        stop_sha = sha256(stop_path)
        if stop_sha != self.args.expect_stop_sha:
            raise StepError(f"stop script SHA mismatch: {stop_sha}")
        stop_stage = self.stage / "runtime" / "stop.py"
        stop_stage.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(stop_path, stop_stage)
        if sha256(stop_stage) != self.args.expect_stop_sha:
            raise StepError("staged stop script SHA mismatch")
        self.state["preflight"] = {"staged": staged, "stop_sha256": stop_sha,
                                   "activate_sha256": ACTIVATE_SHA,
                                   "launch_sha256": LAUNCH_SHA,
                                   "wait_sha256": WAIT_SHA}
        self.source_env = self._source_env()
        self._loader_check()
        target_path = self.args.tree / "vllm/model_executor/models/clef_joint_schema.py"
        actual_pre = sha256(target_path) if target_path.is_file() else None
        if actual_pre != self.cfg("preimage_sha", PREIMAGE_SHA):
            raise StepError(f"target is not preimage: {actual_pre}")
        if self.external:
            self.state["preflight"]["mode"] = "external-vendor"
            self.state["preflight"]["vendor_check"] = self._vendor_installer(["--check"])
        else:
            state = self._installer(["--check"])
            self.state["preflight"]["mode"] = "head-postimage"
        self.state["preflight"]["preimage"] = self.cfg("preimage_sha", PREIMAGE_SHA)
        port_state = port_open(self.args.port)
        self.state["preflight"]["port"] = port_state
        if port_state != "open":
            raise StepError(f"service port is not open: {port_state}")
        pids = find_api_pid(self.args.port)
        if len(pids) != 1:
            raise StepError(f"api pid discovery is not unique: {pids}")
        root = pids[0]
        identity = proc_identity(root)
        if identity is None or "vllm.entrypoints.cli.main" not in identity["cmdline"]:
            raise StepError(f"api identity invalid: {identity}")
        self.identity = {"root": identity, "descendants": descendants(root)}
        self.prod_env = parse_environ((PROC_ROOT / f"{root}/environ").read_bytes())
        for key in CANDIDATE_KEYS:
            self.prod_env.pop(key, None)
        for key in CLEANUP_KEYS:
            self.prod_env.pop(key, None)
        (self.run_dir / "identity-pre.json").write_text(
            json.dumps(self.identity, ensure_ascii=False, indent=1) + "\n")
        (self.run_dir / "root-identity.json").write_text(
            json.dumps(identity, ensure_ascii=False, indent=1) + "\n")
        selected = selected_env(self.prod_env)
        activated = selected_env(self.source_env)
        mismatch = sorted(key for key in set(selected) | set(activated)
                          if selected.get(key) != activated.get(key))
        self.state["preflight"]["env_compare"] = {
            "selected_keys": len(selected),
            "mismatch_keys": mismatch,
            "candidate_keys_removed": sorted(CANDIDATE_KEYS),
            "note": "values are not written to public logs",
        }
        (self.run_dir / "env-prod-selected.json").write_text(
            json.dumps(selected, ensure_ascii=False, indent=1) + "\n")
        self._loader_check(self.prod_env, "prod")
        npu = subprocess.run(["npu-smi", "info"], capture_output=True, text=True,
                             timeout=30)
        (self.run_dir / "npu-before.txt").write_text(npu.stdout + npu.stderr)
        self.state["preflight"]["api_pid"] = root
        self.state["preflight"]["descendant_count"] = len(self.identity["descendants"])
        return {"api_pid": root}

    def _source_env(self) -> dict:
        script = 'set -a; source "$1" >/dev/null 2>&1 || exit 3; env -0'
        result = subprocess.run(["bash", "-c", script, "bash", self.args.activate],
                                capture_output=True, timeout=60)
        if result.returncode != 0:
            raise StepError(f"source activate_env failed: {result.returncode}",
                            rc=result.returncode,
                            detail={"cmd": ["bash", "-c", script, self.args.activate],
                                    "rc": result.returncode,
                                    "stderr": result.stderr.decode(errors="replace")[-1000:]})
        env = parse_environ(result.stdout)
        for key in ("PYTHON_BIN", "ASCEND_HOME_PATH", "LD_LIBRARY_PATH"):
            if not env.get(key):
                raise StepError(f"activate_env did not set {key}")
        self.state["preflight"]["activate_env"] = {
            key: env.get(key) for key in
            ("PYTHON_BIN", "ASCEND_HOME_PATH", "CLEF_CANN_ENV", "CLEF_ATB_ENV",
             "CLEF_VENV", "LD_LIBRARY_PATH")}
        return env

    def _loader_check(self, env: dict | None = None, label: str = "activate"):
        env = dict(self.source_env if env is None else env)
        paths = [p for p in env.get("LD_LIBRARY_PATH", "").split(":") if p]
        hccl = next((p for p in paths if Path(p, "libhccl.so").is_file()), None)
        if not hccl:
            raise StepError("libhccl.so not on LD_LIBRARY_PATH")
        venv = env.get("VIRTUAL_ENV")
        if not venv:
            raise StepError("VIRTUAL_ENV is not set after activate_env")
        site = Path(venv) / "lib" / "python3.12" / "site-packages"
        if not site.is_dir():
            raise StepError(f"venv site-packages not found: {site}")
        ext = next(iter(sorted(site.glob("torch_npu/_C*.so"))), None)
        if ext is None:
            raise StepError("torch_npu extension not found")
        torch_lib = site / "torch" / "lib"
        if not torch_lib.is_dir():
            raise StepError("torch lib directory not found")
        env["LD_LIBRARY_PATH"] = f"{torch_lib}:{env['LD_LIBRARY_PATH']}"
        result = subprocess.run(["ldd", str(ext)], capture_output=True, text=True,
                                env=env, timeout=60)
        if result.returncode != 0 or "not found" in result.stdout:
            raise StepError("torch_npu loader check failed: " +
                            (result.stdout + result.stderr)[-1000:])
        if "libhccl.so" not in result.stdout:
            raise StepError("libhccl.so is not resolved")
        self.state["preflight"].setdefault("loader", {})[label] = {
            "hccl_dir": hccl, "extension": str(ext), "venv": venv, "ok": True}

    def _installer(self, action: list) -> dict:
        cmd = [self.args.python, str(self.stage / self.cfg("installer_rel", INSTALLER_REL)),
               "--tree", str(self.args.tree), *action]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        log = self.run_dir / (action[0].lstrip("-") + ".log")
        log.write_text(result.stdout + result.stderr)
        if result.returncode != 0:
            raise StepError(f"installer {action} failed: {result.returncode}",
                            rc=result.returncode,
                            detail={"cmd": cmd, "rc": result.returncode,
                                    "stdout": result.stdout[-1000:],
                                    "stderr": result.stderr[-1000:]})
        return json.loads(result.stdout)

    def _vendor_installer(self, action: list) -> dict:
        cmd = [self.args.python, str(self.stage / self.cfg("vendor_installer_rel", "")),
               *action, *self.cfg("vendor_extra_arg", [])]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        log = self.run_dir / ("vendor-" + action[0].lstrip("-") + ".log")
        log.write_text(result.stdout + result.stderr)
        if result.returncode != 0:
            raise StepError(f"vendor installer {action} failed: {result.returncode}",
                            rc=result.returncode,
                            detail={"cmd": cmd, "rc": result.returncode,
                                    "stdout": result.stdout[-1000:],
                                    "stderr": result.stderr[-1000:]})
        try:
            return json.loads(result.stdout)
        except Exception:
            return {"raw": result.stdout[-1000:]}

    # ------------------------------------------------------------------
    def stop(self):
        self.stop_attempted = True
        self.candidate_started = True
        root = self.identity["root"]["pid"]
        out = self.run_dir / "stop.json"
        cmd = [self.args.python, str(self.stage / "runtime" / "stop.py"),
               "--api-pid", str(root),
               "--saved-identity", str(self.run_dir / "root-identity.json"),
               "--out", str(out)]
        result = subprocess.run(cmd, capture_output=True, text=True,
                                timeout=self.args.stop_timeout)
        (self.run_dir / "stop.log").write_text(result.stdout + result.stderr)
        if result.returncode != 0:
            detail = {}
            if out.is_file():
                detail["stop_json"] = str(out)
                try:
                    detail["stop"] = json.loads(out.read_text()).get("stop")
                except Exception:
                    pass
            raise StepError(f"stop returned {result.returncode}",
                            rc=result.returncode, detail=detail)
        record = json.loads(out.read_text())
        survivors = (record.get("stop") or {}).get("survivors") or []
        if survivors:
            raise StepError(f"stop survivors: {survivors}")
        for item in [self.identity["root"], *self.identity["descendants"]]:
            live = proc_identity(item["pid"])
            if live is not None and live["starttime"] == item["starttime"] \
                    and live["boot_id"] == item["boot_id"]:
                raise StepError(f"saved pid still alive: {item['pid']}")
        self.stopped = True
        port_state = port_open(self.args.port)
        if port_state != "closed":
            raise StepError(f"port after stop is {port_state}")
        return {"stop_json": str(out), "port": port_state}

    def apply(self):
        if self.external:
            self.vendor_applied = True
            report = self._vendor_installer([self.cfg("vendor_apply_flag", "--install")])
            return {"vendor_apply": report}
        target = self.args.tree / "vllm/model_executor/models/clef_joint_schema.py"
        try:
            report = self._installer(["--apply", "--manifest", str(self.run_dir / "apply.json")])
        finally:
            # The file may already be the postimage when manifest parsing fails;
            # finish() must still revert it.
            self.applied = (target.is_file()
                            and sha256(target) == self.cfg("postimage_sha", POSTIMAGE_SHA))
        if not self.applied:
            raise StepError(f"apply did not install postimage: {report.get('state_after')}",
                            detail={"postimage": self.applied})
        return {"postimage": self.cfg("postimage_sha", POSTIMAGE_SHA)}

    def _probe_cleanup(self, proc, saved: list[dict], timed_out: bool) -> dict:
        killed, unknown, zombies = [], [], []
        leader = saved[0] if saved else None
        live_leader = proc_identity(proc.pid)
        if (timed_out and leader and identity_matches(leader, live_leader)
                and live_leader.get("state") != "Z"
                and live_leader["pgid"] == proc.pid and leader["pgid"] == proc.pid):
            try:
                os.killpg(proc.pid, signal.SIGKILL)
                killed.append({"pid": proc.pid, "signal": "pgid"})
            except OSError:
                pass
        for target in saved:
            live = proc_identity(target["pid"])
            if live is None:
                continue
            if live.get("state") == "Z":
                zombies.append(target["pid"])
                continue
            if not identity_matches(target, live):
                unknown.append({"pid": target["pid"], "reason": "identity changed"})
                continue
            if timed_out or proc.poll() is not None:
                try:
                    os.kill(target["pid"], signal.SIGKILL)
                    killed.append({"pid": target["pid"], "signal": "pid"})
                except OSError:
                    pass
        reaped = False
        try:
            proc.wait(timeout=10)
            reaped = True
        except subprocess.TimeoutExpired:
            pass
        def gone(pid: int) -> bool:
            live = proc_identity(pid)
            return live is None or live.get("state") == "Z"

        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            if all(gone(item["pid"]) for item in saved):
                break
            time.sleep(0.2)
        residual = []
        for item in saved:
            live = proc_identity(item["pid"])
            if live is not None and live.get("state") != "Z":
                residual.append(item["pid"])
        saved_pids = {item["pid"] for item in saved}
        for pid, ident in group_members(proc.pid).items():
            if ident.get("state") == "Z":
                continue
            if pid not in saved_pids:
                unknown.append({"pid": pid, "reason": "unattributed group member"})
            elif pid not in residual:
                residual.append(pid)
        return {"rc": proc.returncode if proc.returncode is not None else -9,
                "timed_out": timed_out, "killed": killed, "reaped": reaped,
                "zombies": sorted(set(zombies)), "unknown": unknown,
                "residual": sorted(set(residual))}

    def probe(self):
        probe_dir = self.run_dir / "probe"
        probe_dir.mkdir()
        output = probe_dir / "head-probe.json"
        env = dict(self.prod_env)
        for key in CANDIDATE_KEYS:
            env.pop(key, None)
        if self.args.candidate_opp_prepend:
            existing = env.get("ASCEND_CUSTOM_OPP_PATH", "")
            env["ASCEND_CUSTOM_OPP_PATH"] = (
                self.args.candidate_opp_prepend + (":" + existing if existing else ""))
        if self.args.candidate_lib_prepend:
            existing = env.get("LD_LIBRARY_PATH", "")
            env["LD_LIBRARY_PATH"] = (
                self.args.candidate_lib_prepend + (":" + existing if existing else ""))
        self.state["steps"].setdefault("probe", {})["env_prepend"] = {
            "ASCEND_CUSTOM_OPP_PATH": self.args.candidate_opp_prepend,
            "LD_LIBRARY_PATH": self.args.candidate_lib_prepend,
        }
        for flag in ("CLEF_JOINT_FUSED_ATTN", "CLEF_JOINT_SMALLQ",
                     "CLEF_JOINT_SMALLQ_KERNEL"):
            env.pop(flag, None)
        env["CLEF_JOINT_SINGLE_FIELD"] = "1"
        flag = self.cfg("env_flag", "CLEF_JOINT_FUSED_ATTN")
        if flag:
            env[flag] = "1"
        for item in self.cfg("child_env", []) or []:
            if "=" not in item:
                raise StepError(f"invalid --child-env KEY=VALUE: {item}")
            key, value = item.split("=", 1)
            env[key] = value
        env["PYTHONUNBUFFERED"] = "1"
        cmd = [self.args.python, str(self.stage / self.cfg("probe_rel", PROBE_REL)),
               "--tree", str(self.args.tree), "--device", str(self.args.device),
               "--output", str(output)]
        stdout = (probe_dir / "probe.log").open("wb")
        stderr = (probe_dir / "probe.err").open("wb")
        started = now()
        saved = []
        rc = 1
        timed_out = False
        try:
            proc = subprocess.Popen(cmd, env=env, stdout=stdout, stderr=stderr,
                                    start_new_session=True)
            # Until the bounded cleanup proves otherwise, restore must not run.
            self.probe_residual = True
            saved = [proc_identity(proc.pid)]
            saved.extend(descendants(proc.pid))
            saved = [item for item in saved if item]
            try:
                rc = proc.wait(timeout=self.args.probe_timeout)
            except subprocess.TimeoutExpired:
                timed_out = True
                rc = None
            cleanup = self._probe_cleanup(proc, saved, timed_out)
        except Exception as exc:
            record = {"rc": rc, "timed_out": timed_out, "pid": getattr(locals().get("proc"), "pid", None),
                      "saved": saved, "cleanup_error": f"{type(exc).__name__}: {exc}",
                      "started_at": started, "finished_at": now(),
                      "probe_json": str(output)}
            (probe_dir / "record.json").write_text(
                json.dumps(record, ensure_ascii=False, indent=1) + "\n")
            raise StepError("probe cleanup failed", rc=rc,
                            detail={"probe_record": record})
        finally:
            stdout.close()
            stderr.close()
        data = None
        record = {"rc": cleanup["rc"], "timed_out": timed_out,
                  "pid": proc.pid, "saved": saved, "cleanup": cleanup,
                  "started_at": started, "finished_at": now(),
                  "probe_json": str(output)}
        if output.is_file():
            try:
                data = json.loads(output.read_text())
                record["interface_ok"] = (data.get("interface") or {}).get("ok")
                record["candidate_error"] = data.get("candidate_error")
                record["delta"] = data.get("delta")
            except Exception as exc:
                record["probe_parse_error"] = f"{type(exc).__name__}: {exc}"
        (probe_dir / "record.json").write_text(
            json.dumps(record, ensure_ascii=False, indent=1) + "\n")
        if cleanup["unknown"] or cleanup["residual"]:
            self.probe_residual = True
            raise StepError("probe left unknown/residual processes",
                            rc=record["rc"], detail={"probe_record": record})
        self.probe_residual = False
        self.vendor_applied = False
        if timed_out:
            raise StepError("probe timed out", rc=record["rc"],
                            detail={"probe_record": record})
        if rc != 0:
            raise StepError(f"probe failed rc={rc}", rc=rc,
                            detail={"probe_record": record})
        schema = self.cfg("probe_schema", "joint-fused-attention-probe/2")
        if schema in ("joint-smallq-probe/1", "n1-probe/1"):
            if (not isinstance(data, dict) or data.get("schema") != schema
                    or data.get("gate_passed") is not True
                    or data.get("error") is not None):
                raise StepError("smallq probe gate failed", rc=rc,
                                detail={"probe_record": record})
            record["probe_timing_path"] = "timing[].rounds[].arms[].samples"
        elif not record.get("interface_ok") or record.get("candidate_error") is not None:
            raise StepError("probe interface/candidate failure", rc=rc,
                            detail={"probe_record": record})
        return record

    def revert(self):
        if self.external:
            report = self._vendor_installer([self.cfg("vendor_revert_flag", "--revert")])
            self.vendor_applied = False
            return {"vendor_revert": report}
        report = self._installer(["--revert", "--manifest", str(self.run_dir / "revert.json")])
        target = self.args.tree / "vllm/model_executor/models/clef_joint_schema.py"
        actual = sha256(target) if target.is_file() else None
        if actual != self.cfg("preimage_sha", PREIMAGE_SHA):
            raise StepError("revert did not restore preimage",
                            detail={"target_sha256": actual})
        self.applied = False
        return {"preimage": self.cfg("preimage_sha", PREIMAGE_SHA)}

    def restore(self):
        sys.path.insert(0, str(Path(self.args.repo) / "runtime" / "d0"))
        import d0_restore
        import d0_run

        label = f"d0-restore-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}-{uuid.uuid4().hex[:8]}"
        label_dir = Path(SERVICE_LOG_ROOT) / label
        if label_dir.exists():
            raise StepError(f"fresh label already exists: {label_dir}")
        before = {int(p.name) for p in PROC_ROOT.iterdir() if p.name.isdigit()}
        env = dict(self.prod_env)
        env["CLEF_LAUNCH_CONFIG"] = "s1"
        for key in CANDIDATE_KEYS:
            env.pop(key, None)
        launch = subprocess.run(["bash", self.args.launch, label], env=env,
                               capture_output=True, text=True, timeout=60)
        (self.run_dir / "restore-launch.log").write_text(launch.stdout + launch.stderr)
        if launch.returncode != 0:
            raise StepError(f"launch.sh failed: {launch.returncode}",
                            rc=launch.returncode,
                            detail={"cmd": ["bash", self.args.launch, label],
                                    "rc": launch.returncode,
                                    "stderr": launch.stderr[-1000:]})
        wait = subprocess.run(["bash", self.args.wait], capture_output=True, text=True,
                              timeout=self.args.wait_timeout)
        (self.run_dir / "restore-wait.log").write_text(wait.stdout + wait.stderr)
        if wait.returncode != 0:
            raise StepError(f"wait.sh failed: {wait.returncode}",
                            rc=wait.returncode,
                            detail={"cmd": ["bash", self.args.wait],
                                    "rc": wait.returncode,
                                    "stderr": wait.stderr[-1000:]})
        disc = d0_restore.discover_service_pids(label, before)
        if disc.get("error") or not disc.get("pids"):
            raise StepError(f"discovery failed: {disc.get('error')}")
        service_pids = set(disc["pids"])
        rlog = {"label": label, "service_discovery": disc}
        cmds = d0_restore.build_commands(d0_run.render_request)
        verified = d0_restore.verify(cmds, rlog, service_pids=service_pids)
        (self.run_dir / "restore.json").write_text(
            json.dumps(rlog, ensure_ascii=False, indent=1) + "\n")
        if not verified:
            raise StepError("restore text/image/health/worker verification failed")
        self.restored = True
        return {"label": label, "service_pids": sorted(service_pids),
                "verify_ok": True, "restore_json": str(self.run_dir / "restore.json")}

    # ------------------------------------------------------------------
    def service_stopped_verified(self) -> tuple[bool, dict]:
        if not self.identity:
            return False, {"error": "no saved identity"}
        targets = [self.identity["root"], *self.identity["descendants"]]
        alive, unknown, zombies = [], [], []
        for item in targets:
            live = proc_identity(item["pid"])
            if live is None:
                continue
            if live.get("state") == "Z":
                zombies.append(item["pid"])
                continue
            if identity_matches(item, live):
                alive.append(item["pid"])
            else:
                unknown.append({"pid": item["pid"], "reason": "identity changed"})
        root = self.identity["root"]
        for pid, ident in group_members(root["pgid"]).items():
            if ident.get("state") == "Z":
                continue
            if pid not in {item["pid"] for item in targets}:
                unknown.append({"pid": pid, "reason": "unattributed group member"})
        return (not alive and not unknown,
                {"alive": alive, "unknown": unknown, "zombies": zombies})

    def verify_existing(self) -> tuple[bool, dict]:
        if not self.identity:
            return False, {"error": "no saved identity"}
        saved = self.identity["root"]
        live = proc_identity(saved["pid"])
        if not identity_matches(saved, live) or live.get("state") == "Z":
            return False, {"error": "saved root identity no longer matches"}
        if live["pgid"] != saved["pgid"] or live["cmdline"] != saved["cmdline"]:
            return False, {"error": "saved root pgid/cmdline changed"}
        if port_open(self.args.port) != "open":
            return False, {"error": "port is not open"}
        sys.path.insert(0, str(Path(self.args.repo) / "runtime" / "d0"))
        import d0_restore
        import d0_run
        pids = {saved["pid"]} | {item["pid"] for item in self.identity["descendants"]}
        log = {"source": "existing-service"}
        try:
            verified = d0_restore.verify(
                d0_restore.build_commands(d0_run.render_request), log,
                service_pids=pids)
        except Exception as exc:
            return False, {"error": f"{type(exc).__name__}: {exc}", "stages": log}
        (self.run_dir / "existing-verify.json").write_text(
            json.dumps(log, ensure_ascii=False, indent=1) + "\n")
        return verified, {"stages": log}

    def finish(self):
        if not self.candidate_started:
            self.state["steps"]["finish"] = {
                "rc": 0, "ok": True, "action": "read_only_noop",
                "finished_at": now()}
            self.flush()
            return
        target = self.args.tree / "vllm/model_executor/models/clef_joint_schema.py"
        target_sha = sha256(target) if target.is_file() else None
        if getattr(self, "vendor_applied", False):
            entry = self.step("revert", required=True)
            if not entry["ok"]:
                self.overall_failed = True
        if self.applied or (not self.external
                            and target_sha == self.cfg("postimage_sha", POSTIMAGE_SHA)):
            entry = self.step("revert", required=True)
            if not entry["ok"]:
                self.overall_failed = True
            target_sha = sha256(target) if target.is_file() else None
        if target_sha != self.cfg("preimage_sha", PREIMAGE_SHA):
            self.state["steps"]["restore"] = {
                "rc": 1, "ok": False, "action": "blocked_source_not_preimage",
                "target_sha256": target_sha, "finished_at": now()}
            self.overall_failed = True
            self.flush()
            return
        if self.probe_residual:
            self.state["steps"]["restore"] = {
                "rc": 1, "ok": False, "action": "blocked_probe_residual",
                "finished_at": now()}
            self.overall_failed = True
            self.flush()
            return
        if self.restored:
            self.flush()
            return
        stopped_ok, detail = self.service_stopped_verified()
        self.state["service_stopped_check"] = detail
        port_state = port_open(self.args.port)
        if stopped_ok and port_state == "closed":
            entry = self.step("restore", required=True)
            if not entry["ok"]:
                self.overall_failed = True
        elif not stopped_ok and port_state == "open":
            healthy, verify_detail = self.verify_existing()
            self.state["steps"]["restore"] = {
                "rc": 0 if healthy else 1, "ok": healthy,
                "action": "existing_service_verified" if healthy else "existing_service_invalid",
                "detail": verify_detail, "finished_at": now()}
            if not healthy:
                self.overall_failed = True
            self.flush()
        else:
            self.state["steps"]["restore"] = {
                "rc": 1, "ok": False, "action": "blocked_service_not_proven_stopped",
                "stopped": detail, "port": port_state, "finished_at": now()}
            self.overall_failed = True
            self.flush()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--repo", default=str(Path(__file__).resolve().parent.parent))
    ap.add_argument("--tree", type=Path, default=DEFAULT_TREE)
    ap.add_argument("--python", default=DEFAULT_PYTHON)
    ap.add_argument("--activate", default=ACTIVATE_DEFAULT)
    ap.add_argument("--launch", default=LAUNCH_DEFAULT)
    ap.add_argument("--wait", default=WAIT_DEFAULT)
    ap.add_argument("--stop-script", default=STOP_DEFAULT)
    ap.add_argument("--expect-stop-sha", default=None)
    ap.add_argument("--port", type=int, default=18143)
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--stop-timeout", type=float, default=120.0)
    ap.add_argument("--probe-timeout", type=float, default=900.0)
    ap.add_argument("--candidate-opp-prepend", default="",
                    help="optional D0/candidate vendor OPP path, child only")
    ap.add_argument("--candidate-lib-prepend", default="",
                    help="optional D0/candidate op_api/lib path, child only")
    ap.add_argument("--candidate-kind", choices=("head-postimage", "external-vendor"),
                    default="head-postimage")
    ap.add_argument("--vendor-installer-rel", default="")
    ap.add_argument("--vendor-installer-source", default=None)
    ap.add_argument("--vendor-installer-sha", default=None)
    ap.add_argument("--vendor-apply-flag", default="--install")
    ap.add_argument("--vendor-revert-flag", default="--revert")
    ap.add_argument("--vendor-extra-arg", action="append", default=[])
    ap.add_argument("--child-env", action="append", default=[],
                    help="KEY=VALUE injected only into the child process")
    ap.add_argument("--postimage-rel", default=POSTIMAGE_REL)
    ap.add_argument("--postimage-source", default=None)
    ap.add_argument("--postimage-sha", default=POSTIMAGE_SHA)
    ap.add_argument("--installer-rel", default=INSTALLER_REL)
    ap.add_argument("--installer-source", default=None)
    ap.add_argument("--installer-sha", default=INSTALLER_SHA)
    ap.add_argument("--probe-rel", default=PROBE_REL)
    ap.add_argument("--probe-source", default=None)
    ap.add_argument("--probe-sha", default=PROBE_SHA)
    ap.add_argument("--client-rel", default=CLIENT_REL)
    ap.add_argument("--client-source", default=None)
    ap.add_argument("--client-sha", default=CLIENT_SHA)
    ap.add_argument("--payload", action="append", default=[],
                    help="extra staged file SRC:REL:SHA (repeatable)")
    ap.add_argument("--preimage-sha", default=PREIMAGE_SHA)
    ap.add_argument("--env-flag", default="CLEF_JOINT_FUSED_ATTN")
    ap.add_argument("--probe-schema", default="joint-fused-attention-probe/2")
    ap.add_argument("--wait-timeout", type=float, default=900.0)
    ap.add_argument("--preflight-only", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    requested = Path(args.run_dir).expanduser().resolve()
    if requested.exists():
        print(json.dumps({"error": f"run dir already exists: {requested}"},
                         ensure_ascii=False))
        return 1
    window = Window(args)
    code = 0
    if args.dry_run:
        try:
            window.preflight()
            window.state["dry_run"] = True
            window.flush()
            print(json.dumps({"dry_run": True, "run_dir": str(window.run_dir),
                              "preflight": window.state.get("preflight")},
                             ensure_ascii=False, indent=1))
            return 0
        except StepError as exc:
            print(json.dumps({"dry_run": False, "error": str(exc)}, ensure_ascii=False))
            return 1
    try:
        window.step("preflight", required=True)
        if args.preflight_only:
            window.flush()
            return 0 if window.state["steps"]["preflight"]["ok"] else 1
        if window.state["steps"]["preflight"]["ok"]:
            stop = window.step("stop", required=True)
            if stop["ok"]:
                apply = window.step("apply", required=True)
                if apply["ok"]:
                    window.step("probe", required=True)
    except Exception as exc:
        window.state["fatal"] = f"{type(exc).__name__}: {exc}"
        window.overall_failed = True
    finally:
        try:
            window.finish()
        except Exception as exc:
            window.state["finish_error"] = f"{type(exc).__name__}: {exc}"
            window.overall_failed = True
        window.state["finished_at"] = now()
        window.state["overall_rc"] = 1 if window.overall_failed else 0
        window.flush()
    code = window.state["overall_rc"]
    print(json.dumps(window.state, ensure_ascii=False, indent=1))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
