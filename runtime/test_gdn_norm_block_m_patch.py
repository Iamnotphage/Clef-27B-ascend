#!/usr/bin/env python3
"""Offline tests for the GDN output-norm BLOCK_M installer.

Pure CPU: this script imports only the standard library and the installer
module.  It never imports torch, torch_npu or vllm_ascend, so it cannot
initialise a device.  It checks the bundled preimage hash, the exact
patch/unpatch round-trip, and the --check/--apply/--revert CLI states in a
temporary tree.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
INSTALLER_PATH = HERE / "apply_gdn_norm_block_m.py"
BUNDLED_PREIMAGE = HERE / "gdn_norm_block_m" / "layernorm_gated_preimage.py"
TARGET = "vllm_ascend/ops/triton/layernorm_gated.py"


def load_installer():
    spec = importlib.util.spec_from_file_location("apply_gdn_norm_block_m", INSTALLER_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def run_cli(installer, argv):
    stream = io.StringIO()
    with contextlib.redirect_stdout(stream):
        code = installer.main(argv)
    return code, json.loads(stream.getvalue())


def main() -> int:
    installer = load_installer()
    failures = []

    def check(name, condition):
        if condition:
            print(f"ok   {name}")
        else:
            print(f"FAIL {name}")
            failures.append(name)

    preimage_text = BUNDLED_PREIMAGE.read_text()
    check("bundled preimage hash", installer.sha256_file(BUNDLED_PREIMAGE) == installer.PREIMAGE_SHA)

    postimage_text = installer.patch_text(preimage_text)
    check("patched hash", installer.sha256_bytes(postimage_text.encode()) == installer.POSTIMAGE_SHA)
    check("unpatched hash", installer.sha256_bytes(installer.unpatch_text(postimage_text).encode())
          == installer.PREIMAGE_SHA)
    check("knob name in postimage", installer.ENV_VAR in postimage_text)
    check("default stays 64", '"CLEF_GDN_NORM_BLOCK_M", "64"' in postimage_text)
    try:
        compile(postimage_text, TARGET, "exec")
    except SyntaxError as exc:
        check(f"postimage compiles ({exc})", False)
    else:
        check("postimage compiles", True)

    try:
        installer.patch_text(postimage_text)
    except ValueError:
        check("double patch rejected", True)
    else:
        check("double patch rejected", False)

    with tempfile.TemporaryDirectory(prefix="gdn-bm-test-") as temporary:
        tree = Path(temporary) / "tree"
        target = tree / TARGET
        target.parent.mkdir(parents=True)
        target.write_text(preimage_text)
        backup = Path(temporary) / "backup"

        code, report = run_cli(installer, ["--tree", str(tree), "--backup-dir", str(backup), "--check"])
        check("check preimage ready", code == 0 and report["ready"] and report["state_after"] == "preimage")
        check("check has no knob", report["knob_present"] is False)

        code, report = run_cli(installer, ["--tree", str(tree), "--backup-dir", str(backup), "--apply"])
        check("apply returns patched", code == 0 and report["state_after"] == "patched")
        check("apply writes postimage", installer.sha256_file(target) == installer.POSTIMAGE_SHA)
        backup_file = backup / "layernorm_gated.py.preimage"
        check("apply keeps preimage backup", installer.sha256_file(backup_file) == installer.PREIMAGE_SHA)
        check("apply manifest env", report["env"]["name"] == installer.ENV_VAR)

        code, report = run_cli(installer, ["--tree", str(tree), "--backup-dir", str(backup), "--check"])
        check("check patched ready", code == 0 and report["ready"] and report["knob_present"] is True)

        code, report = run_cli(installer, ["--tree", str(tree), "--backup-dir", str(backup), "--apply"])
        check("apply is idempotent", code == 0 and report["state_after"] == "patched")

        code, report = run_cli(installer, ["--tree", str(tree), "--backup-dir", str(backup), "--print-diff",
                                           "--check"])
        check("print-diff shows env line", code == 0 and installer.ENV_VAR in report.get("diff", ""))

        code, report = run_cli(installer, ["--tree", str(tree), "--backup-dir", str(backup), "--revert"])
        check("revert returns preimage", code == 0 and report["state_after"] == "preimage")
        check("revert writes preimage", installer.sha256_file(target) == installer.PREIMAGE_SHA)

        code, report = run_cli(installer, ["--tree", str(tree), "--backup-dir", str(backup), "--revert"])
        check("revert is idempotent", code == 0 and report["state_after"] == "preimage")

        target.write_text("# unknown content\n")
        code, report = run_cli(installer, ["--tree", str(tree), "--backup-dir", str(backup), "--check"])
        check("unknown content rejected", code == 1 and report["ready"] is False)

    if failures:
        print(f"{len(failures)} failure(s): {', '.join(failures)}")
        return 1
    print("all installer checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
