#!/usr/bin/env python3
"""CPU test for the service-check adapter: exact aa3e allowance only."""
from __future__ import annotations
import hashlib, json, shutil, subprocess, sys, tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
PREIMAGE = HERE / "joint_service_check" / "apply_joint_single_field_patch_preimage.py"
POSTIMAGE = HERE / "joint_service_check" / "apply_joint_single_field_patch.py"
AA3E_SOURCE = HERE / "joint_fused_attention" / "clef_joint_schema.py"
AA3E = "aa3e17585f8bd60a4d05e342abc7e2773c79a0b2d49f0370d15e29bcc6b72503"
POST_SHA = "36eaff440fd58063a3311b2efabfd1604b4a284cd24f3b7ac3cc4e7fe177fcc7"

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main() -> int:
    failures = []
    def check(name, ok, detail=""):
        print(("ok   " if ok else "FAIL ") + name + (f" {detail}" if detail else ""))
        if not ok: failures.append(name)
    check("preimage hash", sha(PREIMAGE) == "76595657d312699b4b1564d1bc054229808cd775c16702fb0e36fcdb1f9706d9")
    check("postimage hash", sha(POSTIMAGE) == POST_SHA)
    with tempfile.TemporaryDirectory(prefix="service-check-") as tmp:
        tree = Path(tmp) / "vllm"
        target = tree / "vllm/model_executor/models/clef_joint_schema.py"
        target.parent.mkdir(parents=True)
        target.write_bytes(AA3E_SOURCE.read_bytes())  # aa3e is the service target
        check("target is aa3e", sha(target) == AA3E)
        def run(env_value):
            cmd = [sys.executable, str(POSTIMAGE), "--tree", str(tree), "--check"]
            env = dict(__import__("os").environ)
            env.pop("CLEF_JOINT_ALLOWED_POSTIMAGE_SHA", None)
            if env_value is not None:
                env["CLEF_JOINT_ALLOWED_POSTIMAGE_SHA"] = env_value
            p = subprocess.run(cmd, capture_output=True, text=True, env=env)
            return p.returncode, json.loads(p.stdout)
        rc, rep = run(None)
        check("aa3e rejected without env", rc == 1 and rep.get("error", "").find(AA3E) >= 0)
        rc, rep = run(AA3E)
        check("aa3e accepted with exact env", rc == 0 and rep.get("ready") is True
              and rep.get("status_after") == "single-field")
        rc, rep = run("f" * 64)
        check("other env value rejected", rc == 1)
        for flag in ("--apply", "--revert"):
            cmd = [sys.executable, str(POSTIMAGE), "--tree", str(tree), flag]
            env = dict(__import__("os").environ)
            env["CLEF_JOINT_ALLOWED_POSTIMAGE_SHA"] = AA3E
            p = subprocess.run(cmd, capture_output=True, text=True, env=env)
            check(f"aa3e not allowed for {flag}", p.returncode == 1)
    print(json.dumps({"checks": 8, "failed": failures, "ok": not failures}))
    return 0 if not failures else 1

if __name__ == "__main__":
    raise SystemExit(main())
