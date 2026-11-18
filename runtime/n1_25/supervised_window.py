#!/usr/bin/env python3
"""Print the exact N1 supervision plan; --execute is reserved for worker21.

No candidate environment is exported in this process. The existing worker24
wrapper owns stop, installation, timeout, cleanup, finally and four real checks.
"""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys

from window_common import IDENTITY_SHA, PREIMAGE_SHA, STOP_SHA, VENDOR, require_sha

PYTHON = "/workspace/yangchen/venvs/clef-py312/bin/python"
PAYLOAD = ("window_common.py", "window_vendor.py", "window_probe.py", "run_cell.py",
           "contract.py", "fixed_gate.py", "build_identity.json")


def plan(a):
    bundle = Path(__file__).resolve().parent
    manifest_path = bundle / "window_payload.json"
    require_sha(manifest_path, a.expect_payload_sha)
    manifest = json.loads(manifest_path.read_text())
    for name, expected in manifest["files"].items():
        require_sha(bundle / name, expected)
    require_sha(a.wrapper, a.wrapper_sha)
    require_sha(a.stop_script, STOP_SHA)
    require_sha(bundle / "build_identity.json", IDENTITY_SHA)
    identity = json.loads((bundle / "build_identity.json").read_text())
    candidate = a.candidate_root.resolve()
    package = candidate / "build_out/custom_opp_ubuntu_aarch64.run"
    adapter = candidate / "n1_adapter.so"
    require_sha(package, identity["package_sha256"])
    require_sha(adapter, identity["adapter_sha256"])
    run = a.run_dir.resolve()
    if run.exists():
        raise ValueError("supervised run directory must be new")
    if not run.is_relative_to(Path("/workspace/yangchen/logs")):
        raise ValueError("run directory must be under /workspace/yangchen/logs")
    vendor = run / "isolated-opp/vendors" / VENDOR
    rel = "runtime/n1_25/"
    cmd = [a.python, str(a.wrapper.resolve()), "--run-dir", str(run),
           "--repo", str(a.repo.resolve()), "--tree", str(a.tree.resolve()),
           "--python", a.python, "--device", "0", "--port", "18143",
           "--stop-script", str(a.stop_script.resolve()), "--expect-stop-sha", STOP_SHA,
           "--stop-timeout", "120", "--probe-timeout", "240", "--wait-timeout", "900",
           "--preimage-sha", PREIMAGE_SHA, "--candidate-kind", "external-vendor",
           "--vendor-installer-rel", rel + "window_vendor.py",
           "--vendor-installer-source", str(bundle / "window_vendor.py"),
           "--vendor-installer-sha", manifest["files"]["window_vendor.py"],
           "--vendor-apply-flag=--install", "--vendor-revert-flag=--revert",
           "--vendor-extra-arg=--run-dir=" + str(run),
           "--vendor-extra-arg=--tree=" + str(a.tree.resolve()),
           "--probe-rel", rel + "window_probe.py", "--probe-source", str(bundle / "window_probe.py"),
           "--probe-sha", manifest["files"]["window_probe.py"], "--probe-schema", "n1-probe/1",
           "--env-flag=", "--client-rel=", "--candidate-opp-prepend", str(vendor),
           "--candidate-lib-prepend", str(vendor / "op_api/lib"),
           "--child-env", "ASCEND_RT_VISIBLE_DEVICES=0,1"]
    for name in PAYLOAD:
        if name in ("window_vendor.py", "window_probe.py"):
            continue
        cmd.extend(["--payload", f"{bundle / name}:{rel + name}:{manifest['files'][name]}"])
    for source, name, expected in ((package, "package.run", identity["package_sha256"]),
                                    (adapter, "n1_adapter.so", identity["adapter_sha256"])):
        cmd.extend(["--payload", f"{source}:{rel + name}:{expected}"])
    return {"schema": "n1-window-plan/1", "owner": a.owner, "argv": cmd,
            "shell": shlex.join(cmd), "payload_sha256": a.expect_payload_sha,
            "wrapper_sha256": a.wrapper_sha, "identity_sha256": IDENTITY_SHA,
            "env_scope": "OPP/opapi/devices on probe child; PYTHONPATH on torchrun child only",
            "revert": "rename isolated-opp to isolated-opp.retired; preserve all bytes",
            "cell": {"rows": 967, "seed": 20261118, "cores": 1, "use_hccl": 1}}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--owner", required=True, choices=["clef-27b-ascend-21"])
    p.add_argument("--wrapper", type=Path, required=True)
    p.add_argument("--wrapper-sha", required=True)
    p.add_argument("--expect-payload-sha", required=True)
    p.add_argument("--repo", type=Path, required=True)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--tree", type=Path, default=Path("/workspace/yangchen/work/src/vllm"))
    p.add_argument("--stop-script", type=Path, default=Path("/tmp/d0cc/d0_stop.py"))
    p.add_argument("--candidate-root", type=Path, default=Path("/workspace/yangchen/work/clef-n1-25-v2"))
    p.add_argument("--python", default=PYTHON)
    p.add_argument("--execute", action="store_true")
    a = p.parse_args()
    result = plan(a)
    print(json.dumps(result, indent=2), flush=True)
    return subprocess.call(result["argv"]) if a.execute else 0


if __name__ == "__main__":
    raise SystemExit(main())
