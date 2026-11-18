#!/usr/bin/env python3
"""21-only isolated vendor lifecycle for the existing supervised window."""
import argparse
import json
from pathlib import Path

from window_common import VENDOR, IDENTITY_SHA, archive_bytes, check_head, frozen, staged_paths


def lifecycle(action, stage, run, tree):
    check_head(tree)
    root, retired = run / "isolated-opp", run / "isolated-opp.retired"
    marker = run / "n1-vendor-state.json"
    expected = {"schema": "n1-vendor/1", "identity_sha256": IDENTITY_SHA,
                "root": str(root), "retired": str(retired)}
    if root.is_symlink() or retired.is_symlink():
        raise ValueError("vendor roots must be ordinary directories")
    if action == "revert":
        if not marker.exists():
            if root.exists() or retired.exists():
                raise ValueError("unowned vendor directory")
            return {**expected, "state": "absent", "pass": True}
        state = json.loads(marker.read_text())
        if any(state.get(k) != v for k, v in expected.items()):
            raise ValueError("vendor ownership mismatch")
        if root.exists():
            if retired.exists():
                raise ValueError("archive already exists alongside active vendor")
            root.rename(retired)
        state.update({"state": "retired", "active_path_absent": not root.exists(), "pass": True})
        marker.write_text(json.dumps(state, indent=2) + "\n")
        return state
    identity = frozen(stage)
    contents = archive_bytes(stage / "package.run", identity)
    if root.exists() or retired.exists() or marker.exists():
        raise ValueError("N1 window installation requires fresh isolated paths")
    if action == "check":
        return {**expected, "state": "checked", "vendor_files": len(contents), "pass": True}
    marker.write_text(json.dumps({**expected, "state": "installing"}, indent=2) + "\n")
    root.mkdir()
    vendor = root / "vendors" / VENDOR
    for rel, data in contents.items():
        target = vendor / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        target.chmod(0o640)
    from contract import verify_identity
    verify_identity(identity, vendor, stage / "n1_adapter.so")
    state = {**expected, "state": "installed", "vendor_files": len(contents), "pass": True}
    marker.write_text(json.dumps(state, indent=2) + "\n")
    return state


def main():
    p = argparse.ArgumentParser(description=__doc__)
    modes = p.add_mutually_exclusive_group(required=True)
    for mode in ("check", "install", "revert"):
        modes.add_argument("--" + mode, action="store_true")
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--tree", type=Path, required=True)
    a = p.parse_args()
    stage, run = staged_paths(__file__, a.run_dir)
    action = next(k for k in ("check", "install", "revert") if getattr(a, k))
    try:
        result = lifecycle(action, stage, run, a.tree)
    except Exception as exc:
        print(json.dumps({"schema": "n1-vendor/1", "pass": False, "error": str(exc)}))
        return 1
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
