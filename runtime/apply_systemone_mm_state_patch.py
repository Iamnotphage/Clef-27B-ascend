#!/usr/bin/env python3
"""Install, identify and revert the /v1/systemone multimodal-state patch.

The fixed vLLM tree is identified by the upstream SHA256 of
``vllm/entrypoints/generate/structured_decisions/serving.py``.  Applying the
patch also installs a byte-identical copy of ``runtime/systemone_mm_state.py``
as ``_clef_mm_state.py`` next to ``serving.py``, so the vLLM worker needs no
extra ``PYTHONPATH`` entry.  The installer is idempotent: applying on the
postimage is a no-op, applying on an unknown source is refused, and every
apply stores byte-exact preimages that ``--revert`` restores and verifies.

Usage:
  python apply_systemone_mm_state_patch.py --check
  python apply_systemone_mm_state_patch.py --apply
  python apply_systemone_mm_state_patch.py --revert
"""

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

PATCH_VERSION = 1
UPSTREAM_SHA256 = "5a700bb79c5db36f46d096d9153bb000357644bc77938fbbf667c3b4ef4c400a"
POSTIMAGE_SHA256 = "99a8e1b73dd55800736de4c2b29cf300aa7f819e4e5495f59805733dbe62994a"
TARGET_REL = "vllm/entrypoints/generate/structured_decisions/serving.py"
MODULE_NAME = "_clef_mm_state.py"
MODULE_SOURCE = Path(__file__).resolve().with_name("systemone_mm_state.py")
MODULE_SHA256 = (
    hashlib.sha256(MODULE_SOURCE.read_bytes()).hexdigest()
    if MODULE_SOURCE.is_file() else None
)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def default_source():
    try:
        import importlib.util
        spec = importlib.util.find_spec("vllm")
        if spec and spec.submodule_search_locations:
            return Path(list(spec.submodule_search_locations)[0]) / TARGET_REL[len("vllm/"):]
    except ImportError:
        pass
    return Path("/workspace/yangchen/work/src/vllm") / TARGET_REL


def repo_root(source):
    source = Path(source).resolve()
    if source.name != "serving.py":
        raise SystemExit(f"unexpected source path: {source}")
    for parent in source.parents:
        candidate = parent / TARGET_REL
        if candidate.is_file() and candidate.resolve() == source:
            return parent
    raise SystemExit(f"cannot locate the vllm package root for {source}")


def state_dir(repo):
    git_dir = repo / ".git"
    if git_dir.is_dir():
        return git_dir / "clef-systemone-mm-state"
    return repo / ".clef_systemone_mm_state"


def status(source):
    source = Path(source)
    if not source.is_file():
        raise SystemExit(f"serving.py not found: {source}")
    current = sha256(source)
    if current == UPSTREAM_SHA256:
        state = "preimage"
    elif current == POSTIMAGE_SHA256:
        state = "postimage"
    else:
        state = "unknown"
    module = source.parent / MODULE_NAME
    if not module.is_file():
        module_state = "missing"
        module_sha = None
    else:
        module_sha = sha256(module)
        module_state = "current" if module_sha == MODULE_SHA256 else "unknown"
    return {
        "patch_version": PATCH_VERSION,
        "source": str(source),
        "sha256": current,
        "upstream_sha256": UPSTREAM_SHA256,
        "postimage_sha256": POSTIMAGE_SHA256,
        "state": state,
        "module": str(module),
        "module_sha256": module_sha,
        "module_expected_sha256": MODULE_SHA256,
        "module_state": module_state,
        "ready": state == "postimage" and module_state == "current",
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


def install_module(source, backup_dir, record):
    module = source.parent / MODULE_NAME
    if module.is_file():
        if sha256(module) != MODULE_SHA256:
            raise SystemExit(
                f"{module} exists with unknown content; refusing to overwrite")
        record["module_action"] = "existing"
        return module
    backup_dir.mkdir(parents=True, exist_ok=True)
    module.write_bytes(MODULE_SOURCE.read_bytes())
    record["module_action"] = "installed"
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=None,
                        help="serving.py path (default: installed vllm package)")
    parser.add_argument("--patch", type=Path,
                        default=Path(__file__).resolve().with_name(
                            "systemone-mm-state.patch"))
    parser.add_argument("--check", action="store_true", help="identify the installed version")
    parser.add_argument("--apply", action="store_true", help="install (idempotent)")
    parser.add_argument("--revert", action="store_true", help="restore the byte-exact preimage")
    args = parser.parse_args()
    if sum((args.check, args.apply, args.revert)) != 1:
        parser.error("select exactly one of --check, --apply, --revert")
    if MODULE_SHA256 is None:
        raise SystemExit(f"module source missing: {MODULE_SOURCE}")

    source = (args.source or default_source()).resolve()
    repo = repo_root(source)
    state = status(source)
    directory = state_dir(repo)
    state_path = directory / "state.json"

    if args.check:
        print(json.dumps(state, ensure_ascii=False, indent=1))
        known = (
            state["state"] in ("preimage", "postimage")
            and state["module_state"] in ("missing", "current")
        )
        return 0 if known else 1

    if args.apply:
        if state["ready"]:
            print(json.dumps({"result": "already-applied", **state},
                             ensure_ascii=False))
            return 0
        if state["state"] not in ("preimage", "postimage"):
            raise SystemExit(
                f"source is {state['state']} (sha256 {state['sha256']}), not the "
                f"pinned preimage {UPSTREAM_SHA256} or postimage "
                f"{POSTIMAGE_SHA256}; restore the pinned file first")
        if state["module_state"] == "unknown":
            raise SystemExit(
                f"module {state['module']} has unknown content; refusing to replace it")
        directory.mkdir(parents=True, exist_ok=True)
        backup = directory / "serving.py.pre-systemone-mm"
        if state["state"] == "preimage":
            # The current file is the pinned preimage: refresh the backup.
            shutil.copyfile(source, backup)
        elif not (backup.is_file() and sha256(backup) == UPSTREAM_SHA256):
            # Already patched (partial install): never overwrite the stored
            # preimage with the postimage.  Revert can still use git.
            backup = None
        record = {
            "applied_at": datetime.now(timezone.utc).isoformat(),
            "source": str(source),
            "upstream_sha256": UPSTREAM_SHA256,
            "postimage_sha256": POSTIMAGE_SHA256,
            "patch_version": PATCH_VERSION,
            "backup": str(backup) if backup is not None else None,
            "patch": str(args.patch.resolve()),
            "patch_sha256": sha256(args.patch) if args.patch.is_file() else None,
            "module_source": str(MODULE_SOURCE),
            "module_sha256": MODULE_SHA256,
        }
        module = source.parent / MODULE_NAME
        module_preexisting = module.is_file()
        try:
            if state["state"] == "preimage":
                apply_patch(repo, args.patch)
            install_module(source, directory, record)
            after = status(source)
            if not after["ready"]:
                raise SystemExit(f"post-apply verification failed: {after}")
        except BaseException:
            if state["state"] == "preimage" and backup is not None:
                shutil.copyfile(backup, source)
            if record.get("module_action") == "installed":
                module.unlink(missing_ok=True)
            raise
        record.update(
            result="applied",
            installed_state=after["state"],
            installed_sha256=after["sha256"],
            module=str(module),
            module_preexisting=module_preexisting,
        )
        state_path.write_text(json.dumps(record, ensure_ascii=False, indent=1) + "\n")
        print(json.dumps(record, ensure_ascii=False))
        return 0

    # --revert
    record = json.loads(state_path.read_text()) if state_path.is_file() else {}
    backup = Path(record["backup"]) if record.get("backup") else None
    restored_from = None
    if backup is not None and backup.is_file() and sha256(backup) == UPSTREAM_SHA256:
        shutil.copyfile(backup, source)
        restored_from = "backup"
    elif (repo / ".git").exists():
        done = subprocess.run(
            ["git", "-C", str(repo), "checkout", "--", TARGET_REL],
            capture_output=True, text=True)
        if done.returncode != 0:
            raise SystemExit(f"git checkout did not restore the preimage: {done.stderr}")
        restored_from = "git"
    else:
        raise SystemExit("no byte-exact backup and no git checkout available; refusing to guess")
    restored = status(source)
    if restored["state"] != "preimage":
        raise SystemExit(f"revert verification failed: {restored}")
    module = source.parent / MODULE_NAME
    if module.is_file():
        if sha256(module) != MODULE_SHA256:
            raise SystemExit(
                f"{module} changed after installation; refusing to remove it")
        module.unlink()
    restored["module_state"] = "missing"
    restored["module_sha256"] = None
    state_path.unlink(missing_ok=True)
    print(json.dumps({"result": "reverted", "from": restored_from,
                      "module_removed": str(module), **restored},
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
