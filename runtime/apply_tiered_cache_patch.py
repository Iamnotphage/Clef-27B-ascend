#!/usr/bin/env python3
"""Install/revert the Tier 2 prefix-state cache entry points.

Two byte-exact targets, both verified against their accepted preimage and
Fail-Closed on any unknown content:

1. ``vllm_ascend/worker/v2/model_runner.py`` — appends a lazy installer that
   runs ``runtime.tiered_prefix_cache.engine_hooks.install()`` before the
   first ``execute_model`` call (never at import: the hooks patch classes that
   import this module).  ``CLEF_PREFIX_STATE_CACHE=0`` skips installation;
   ``CLEF_TIER2_REPO`` names the tree that contains ``runtime/``.
2. ``vllm/entrypoints/generate/structured_decisions/api_router.py`` — inserts
   ``prefix_state_cache`` into the ``/v1/decide/info`` cache block so the
   served configuration is disclosed on the API.

The patched state is detected by a round-trip: removing the injected text
must reproduce the recorded preimage SHA-256 exactly.
"""

from __future__ import annotations

import argparse
import ast
import fcntl
import hashlib
import json
import os
from pathlib import Path
import stat
import tempfile

DEFAULT_VLLM_TREE = Path("/workspace/yangchen/work/mix-ws-20261123/src/vllm")
DEFAULT_ASCEND_TREE = Path("/workspace/yangchen/work/mix-ws-20261123/src/vllm-ascend")
DEFAULT_BACKUP = Path("/workspace/yangchen/logs/tiered-cache-20261124/patch-backup")

MODEL_RUNNER = "vllm_ascend/worker/v2/model_runner.py"
MODEL_RUNNER_PREIMAGE = "e866aeb27d04afe2141850122c03343c38d2a9b14cf1f9f136866003b35968a6"

API_ROUTER = "vllm/entrypoints/generate/structured_decisions/api_router.py"
API_ROUTER_PREIMAGE = "bdf11bee92c92a7caf6070321eb2294b833f0e19be6ca9f3e90b7790c34ddc9f"

API_ANCHOR = '"chunked_prefill": False,'
API_MARKER = '"prefix_state_cache":'
API_INSERT = (
    '{indent}"prefix_state_cache": __import__("os").environ.get('
    '"CLEF_PREFIX_STATE_CACHE", "1").strip().lower() not in '
    '("0", "false", "no", "off"),'
)

ENGINE_BLOCK = '''
# >>> Clef Tier 2 prefix-state cache (managed by runtime/apply_tiered_cache_patch.py)
def _clef_tier2_install():
    import os as _os
    value = _os.environ.get("CLEF_PREFIX_STATE_CACHE", "1").strip().lower()
    if value in ("0", "false", "no", "off"):
        print("clef-tier2 disabled by CLEF_PREFIX_STATE_CACHE", flush=True)
        return
    import sys as _sys
    root = _os.environ.get("CLEF_TIER2_REPO", "")
    if root and root not in _sys.path:
        _sys.path.insert(0, root)
    from runtime.tiered_prefix_cache.engine_hooks import install
    install()
    print("clef-tier2 hooks installed pid=%d repo=%s" % (_os.getpid(), root),
          flush=True)


_clef_tier2_installed = False
_clef_tier2_execute = NPUModelRunner.execute_model


def _clef_tier2_execute_wrapper(self, *args, **kwargs):
    global _clef_tier2_installed
    if not _clef_tier2_installed:
        _clef_tier2_installed = True
        _clef_tier2_install()
    return _clef_tier2_execute(self, *args, **kwargs)


NPUModelRunner.execute_model = _clef_tier2_execute_wrapper
# <<< Clef Tier 2 prefix-state cache
'''


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def patch_model_runner(text: str) -> str:
    if digest(text.encode()) != MODEL_RUNNER_PREIMAGE:
        raise ValueError("model_runner.py: verified preimage required")
    if "_clef_tier2_install" in text:
        raise ValueError("model_runner.py: Tier 2 block already present")
    postimage = text + ENGINE_BLOCK
    ast.parse(postimage)
    return postimage


def patch_api_router(text: str) -> str:
    if digest(text.encode()) != API_ROUTER_PREIMAGE:
        raise ValueError("api_router.py: verified preimage required")
    lines = text.splitlines(keepends=True)
    matches = [index for index, line in enumerate(lines)
               if line.strip() == API_ANCHOR]
    if len(matches) != 1:
        raise ValueError(
            f"api_router.py: cache anchor is not unique ({len(matches)} hits)")
    index = matches[0]
    body = lines[index].rstrip("\n")
    indent = body[:len(body) - len(body.lstrip())]
    if not body.endswith(API_ANCHOR):
        raise ValueError("api_router.py: cache anchor has trailing content")
    lines.insert(index + 1, API_INSERT.format(indent=indent) + "\n")
    postimage = "".join(lines)
    ast.parse(postimage)
    return postimage


def strip_model_runner(text: str) -> str | None:
    if not text.endswith(ENGINE_BLOCK):
        return None
    return text[: -len(ENGINE_BLOCK)]


def strip_api_router(text: str) -> str | None:
    lines = text.splitlines(keepends=True)
    matches = [index for index, line in enumerate(lines)
               if line.strip().startswith(API_MARKER)]
    if len(matches) != 1:
        return None
    del lines[matches[0]]
    return "".join(lines)


def status(path: Path, preimage: str, strip) -> tuple[str, str | None]:
    if not path.is_file():
        return "missing", None
    data = path.read_bytes()
    sha = digest(data)
    if sha == preimage:
        return "preimage", sha
    try:
        stripped = strip(data.decode())
    except UnicodeDecodeError:
        return "unknown", sha
    if stripped is not None and digest(stripped.encode()) == preimage:
        return "patched", sha
    return "unknown", sha


def write_atomic(path: Path, data: bytes, mode: int = 0o644) -> None:
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".tiered-", dir=path.parent)
    temporary_path = Path(temporary)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary_path, mode)
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


TARGETS = (
    ("model_runner", MODEL_RUNNER, MODEL_RUNNER_PREIMAGE,
     patch_model_runner, strip_model_runner),
    ("api_router", API_ROUTER, API_ROUTER_PREIMAGE,
     patch_api_router, strip_api_router),
)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vllm-tree", type=Path, default=DEFAULT_VLLM_TREE)
    parser.add_argument("--ascend-tree", type=Path, default=DEFAULT_ASCEND_TREE)
    parser.add_argument("--backup-dir", type=Path, default=DEFAULT_BACKUP)
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--check", action="store_true")
    actions.add_argument("--apply", action="store_true")
    actions.add_argument("--revert", action="store_true")
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args(argv)
    vllm_tree = args.vllm_tree.expanduser().resolve()
    ascend_tree = args.ascend_tree.expanduser().resolve()
    backup_dir = args.backup_dir.expanduser().resolve()
    report = {
        "patch_version": 1,
        "vllm_tree": str(vllm_tree),
        "ascend_tree": str(ascend_tree),
        "action": "check" if args.check else "apply" if args.apply else "revert",
        "targets": {},
        "ready": False,
    }
    try:
        fd = os.open(ascend_tree, os.O_RDONLY)
        try:
            fcntl.flock(fd, fcntl.LOCK_SH if args.check else fcntl.LOCK_EX)
            statuses = {}
            for name, relative, preimage, patch, strip in TARGETS:
                tree = ascend_tree if relative.startswith("vllm_ascend") else vllm_tree
                target = tree / relative
                backup = backup_dir / (relative.replace("/", "__") + ".preimage")
                before, sha_before = status(target, preimage, strip)
                entry = {"target": str(target), "preimage_sha256": preimage,
                         "status_before": before, "sha256_before": sha_before}
                if before in ("missing", "unknown"):
                    raise ValueError(f"{relative}: refused {before} source content")
                if args.apply and before == "preimage":
                    original = target.read_bytes()
                    if not backup.exists():
                        backup.parent.mkdir(parents=True, exist_ok=True)
                        write_atomic(
                            backup, original, stat.S_IMODE(target.stat().st_mode))
                    elif digest(backup.read_bytes()) != preimage:
                        raise ValueError(f"{relative}: preimage backup SHA mismatch")
                    postimage = patch(original.decode()).encode()
                    if digest(strip(postimage.decode()).encode()) != preimage:
                        raise ValueError(f"{relative}: generated postimage round-trip failed")
                    write_atomic(
                        target, postimage, stat.S_IMODE(target.stat().st_mode))
                elif args.revert and before == "patched":
                    if not backup.is_file() or digest(backup.read_bytes()) != preimage:
                        raise ValueError(f"{relative}: verified preimage backup required")
                    write_atomic(
                        target, backup.read_bytes(),
                        stat.S_IMODE(target.stat().st_mode))
                after, sha_after = status(target, preimage, strip)
                entry.update(status_after=after, sha256=sha_after)
                statuses[name] = after
                report["targets"][name] = entry
            report["status"] = statuses
            report["ready"] = all(value == "patched" for value in statuses.values())
        finally:
            os.close(fd)
    except (OSError, ValueError, SyntaxError) as exc:
        report["error"] = str(exc)
    if args.manifest:
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return int("error" in report)


if __name__ == "__main__":
    raise SystemExit(main())
