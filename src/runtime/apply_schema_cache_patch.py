#!/usr/bin/env python3
"""Install, identify and revert the schema-token cache v2 patch.

The fixed vLLM tree is identified by the upstream SHA256 of
``vllm/entrypoints/openai/decisions/strategies.py``. The installer is
idempotent: applying on v2 is a no-op, applying on v1 is refused (restore the
upstream file first), and every apply stores a byte-exact preimage that
``--revert`` restores and verifies.

Usage:
  python apply_schema_cache_patch.py --check
  python apply_schema_cache_patch.py --apply
  python apply_schema_cache_patch.py --revert
"""

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

UPSTREAM_SHA256 = "d7e25f1d879cd0dc2d8e38629805a66018a9c63e364572d6cbfee6a9fedd97bb"
TARGET_REL = "vllm/entrypoints/openai/decisions/strategies.py"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def detect_version(text):
    match = re.search(r"^SCHEMA_CACHE_VERSION\s*=\s*(\d+)$", text, re.MULTILINE)
    if match:
        return int(match.group(1))
    return 1 if "_SchemaTokenCache" in text else 0


def default_source():
    try:
        import importlib.util
        spec = importlib.util.find_spec("vllm")
        if spec and spec.submodule_search_locations:
            return Path(list(spec.submodule_search_locations)[0]) / "entrypoints/openai/decisions/strategies.py"
    except ImportError:
        pass
    return Path("/workspace/yangchen/work/src/vllm") / TARGET_REL


def repo_root(source):
    source = Path(source).resolve()
    if source.name != "strategies.py":
        raise SystemExit(f"unexpected source path: {source}")
    for parent in source.parents:
        candidate = parent / TARGET_REL
        if candidate.is_file() and candidate.resolve() == source:
            return parent
    raise SystemExit(f"cannot locate the vllm package root for {source}")


def state_dir(repo):
    git_dir = repo / ".git"
    if git_dir.is_dir():
        return git_dir / "clef-schema-cache-v2"
    return repo / ".clef_schema_cache_state"


def status(source):
    text = Path(source).read_text()
    current = sha256(source)
    version = detect_version(text)
    if current == UPSTREAM_SHA256:
        state = "upstream"
    elif version == 2:
        state = "v2"
    elif version == 1:
        state = "v1"
    else:
        state = "unknown"
    return {
        "source": str(source),
        "sha256": current,
        "upstream_sha256": UPSTREAM_SHA256,
        "version": version,
        "state": state,
    }


def apply_patch(repo, patch):
    if not patch.is_file():
        raise SystemExit(f"patch not found: {patch}")
    if (repo / ".git").exists():
        check = subprocess.run(
            ["git", "-C", str(repo), "apply", "--check", str(patch)],
            capture_output=True, text=True)
        if check.returncode != 0:
            raise SystemExit(f"patch does not apply cleanly:\n{check.stderr}")
        done = subprocess.run(
            ["git", "-C", str(repo), "apply", "--whitespace=nowarn", str(patch)],
            capture_output=True, text=True)
        if done.returncode != 0:
            raise SystemExit(f"git apply failed:\n{done.stderr}")
        return
    check = subprocess.run(
        ["patch", "-p1", "--dry-run", "-i", str(patch)],
        cwd=str(repo), capture_output=True, text=True)
    if check.returncode != 0:
        raise SystemExit(f"patch does not apply cleanly:\n{check.stdout}{check.stderr}")
    done = subprocess.run(
        ["patch", "-p1", "-i", str(patch)],
        cwd=str(repo), capture_output=True, text=True)
    if done.returncode != 0:
        raise SystemExit(f"patch failed:\n{done.stdout}{done.stderr}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=None,
                        help="strategies.py path (default: installed vllm package)")
    parser.add_argument("--patch", type=Path,
                        default=Path(__file__).resolve().with_name("schema-cache-v2.patch"))
    parser.add_argument("--check", action="store_true", help="identify the installed version")
    parser.add_argument("--apply", action="store_true", help="install v2 (idempotent)")
    parser.add_argument("--revert", action="store_true", help="restore the byte-exact preimage")
    args = parser.parse_args()
    if sum((args.check, args.apply, args.revert)) != 1:
        parser.error("select exactly one of --check, --apply, --revert")

    source = (args.source or default_source()).resolve()
    repo = repo_root(source)
    state = status(source)
    state_path = state_dir(repo) / "state.json"

    if args.check:
        print(json.dumps(state, ensure_ascii=False, indent=1))
        return 0 if state["state"] in ("upstream", "v2") else 1

    if args.apply:
        if state["state"] == "v2":
            print(json.dumps({"result": "already-v2", **state}, ensure_ascii=False))
            return 0
        if state["state"] != "upstream":
            raise SystemExit(
                f"source is {state['state']} (sha256 {state['sha256']}), not the pinned "
                f"upstream preimage {UPSTREAM_SHA256}; restore the upstream file first")
        directory = state_dir(repo)
        directory.mkdir(parents=True, exist_ok=True)
        backup = directory / "strategies.py.pre-v2"
        shutil.copyfile(source, backup)
        apply_patch(repo, args.patch)
        after = status(source)
        if after["state"] != "v2" or after["version"] != 2:
            shutil.copyfile(backup, source)
            raise SystemExit(f"post-apply verification failed: {after}; restored preimage")
        record = {
            "applied_at": datetime.now(timezone.utc).isoformat(),
            "source": str(source),
            "upstream_sha256": UPSTREAM_SHA256,
            "applied_sha256": after["sha256"],
            "version": after["version"],
            "backup": str(backup),
            "patch": str(args.patch.resolve()),
            "patch_sha256": sha256(args.patch),
        }
        state_path.write_text(json.dumps(record, ensure_ascii=False, indent=1) + "\n")
        print(json.dumps({"result": "applied", **record}, ensure_ascii=False))
        return 0

    # --revert
    record = json.loads(state_path.read_text()) if state_path.is_file() else {}
    backup = Path(record.get("backup", ""))
    if backup.is_file() and sha256(backup) == UPSTREAM_SHA256:
        shutil.copyfile(backup, source)
        restored = status(source)
        if restored["sha256"] != UPSTREAM_SHA256:
            raise SystemExit(f"revert verification failed: {restored}")
        state_path.unlink()
        print(json.dumps({"result": "reverted-from-backup", **restored}, ensure_ascii=False))
        return 0
    if (repo / ".git").exists():
        done = subprocess.run(
            ["git", "-C", str(repo), "checkout", "--", TARGET_REL],
            capture_output=True, text=True)
        if done.returncode == 0:
            restored = status(source)
            if restored["sha256"] == UPSTREAM_SHA256:
                print(json.dumps({"result": "reverted-from-git", **restored}, ensure_ascii=False))
                return 0
        raise SystemExit(f"git checkout did not restore the preimage: {done.stderr}")
    raise SystemExit("no byte-exact backup and no git checkout available; refusing to guess")


if __name__ == "__main__":
    sys.exit(main())
