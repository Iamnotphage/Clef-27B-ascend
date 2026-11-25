#!/usr/bin/env python3
"""Install or revert an armed, one-shot Qwen2MoeMLP input capture hook.

Apply before starting the experimental service, with CLEF_MLP_CAPTURE_DIR set
to the capture directory. After warmup, atomically create arm.json in that
directory before sending each request. Use a unique request_name each time.

Example arm.json
  {"request_name": "text-307-run1", "token_count": 307, "layers": [0, 31, 63]}

token_count is optional request metadata. The actual MLP input token count is
recorded separately because padding or sequence parallelism can change it.
An absent arm.json disables capture. The default layer list is [0, 31, 63].
Each request/layer/TP-rank writes one .pt file containing input and metadata.
CPU copies synchronize the captured input, so use separate timing runs.

The helper is embedded in qwen2_moe.py. --output-dir stores the verified source
backup and may also serve as CLEF_MLP_CAPTURE_DIR. All patch operations require
the pinned preimage or the exact postimage generated from its verified backup.
"""

from __future__ import annotations

import argparse
import ast
import fcntl
import hashlib
import json
import os
from pathlib import Path
import tempfile


TARGET_RELATIVE = "vllm/model_executor/models/qwen2_moe.py"
PREIMAGE_SHA256 = "efd3c5a3ac286c48bf9a1ed14574a6d5a30c61557d387036148452ca318a9fdb"
CLASS_ANCHOR = "class Qwen2MoeMLP(nn.Module):\n"
INIT_ANCHOR = (
    "        super().__init__()\n"
    "        disable_tp = disable_tp or is_sequence_parallel\n"
)
FORWARD_ANCHOR = (
    "    def forward(self, x):\n"
    "        gate_up, _ = self.gate_up_proj(x)\n"
)

CAPTURE_HELPER = '''# Clef armed MLP input capture, version 1.
_CLEF_MLP_CAPTURED = set()
_CLEF_MLP_CAPTURE_WARNED = set()


def _clef_capture_mlp_input(prefix, x):
    import os

    capture_directory = os.environ.get("CLEF_MLP_CAPTURE_DIR")
    if not capture_directory:
        return

    import json
    from pathlib import Path
    import re

    directory = Path(capture_directory).expanduser().resolve()
    arm_path = directory / "arm.json"
    if not arm_path.is_file():
        return

    try:
        try:
            arm = json.loads(arm_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return
        if not isinstance(arm, dict):
            raise ValueError("arm.json must contain an object")
        request_name = arm.get("request_name")
        if not isinstance(request_name, str) or not request_name.strip():
            raise ValueError("arm.json requires a nonempty request_name")
        layers = arm.get("layers", [0, 31, 63])
        if not isinstance(layers, list) or any(type(layer) is not int for layer in layers):
            raise ValueError("arm.json layers must be a list of integers")
        match = re.search(r"(?:^|\\.)layers\\.(\\d+)(?:\\.|$)", prefix)
        if match is None:
            return
        layer = int(match.group(1))
        if layer not in layers:
            return

        # Resolve the initialized TP group only for an armed, selected layer.
        from vllm.distributed import get_tensor_model_parallel_rank

        rank = get_tensor_model_parallel_rank()
        key = (str(directory), request_name, layer, rank)
        if key in _CLEF_MLP_CAPTURED:
            return

        import fcntl
        import hashlib
        import math
        import tempfile
        import time

        label = re.sub(r"[^a-zA-Z0-9_.-]", "_", request_name)[:80]
        digest = hashlib.sha256(request_name.encode("utf-8")).hexdigest()[:16]
        destination = directory / f"{label}-{digest}.layer{layer:02d}.rank{rank}.pt"
        # Persistent lock files allow retries after worker failure and prevent
        # concurrent writers with the same request/layer/rank from duplicating.
        with destination.with_suffix(".lock").open("a") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            if destination.is_file():
                _CLEF_MLP_CAPTURED.add(key)
                return
            metadata = {
                "request_name": request_name,
                "prefix": prefix,
                "layer": layer,
                "rank": rank,
                "shape": list(x.shape),
                "stride": list(x.stride()),
                "dtype": str(x.dtype),
                "device": str(x.device),
                "token_count": arm.get("token_count"),
                "input_token_count": math.prod(x.shape[:-1]),
                "time_ns": time.time_ns(),
                "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "pid": os.getpid(),
            }
            payload = {
                "input": x.detach().to(device="cpu", copy=True),
                "metadata": metadata,
            }
            temporary = None
            try:
                with tempfile.NamedTemporaryFile(
                    mode="wb", dir=directory, prefix=f".{destination.name}.",
                    suffix=".tmp", delete=False,
                ) as stream:
                    temporary = Path(stream.name)
                    torch.save(payload, stream)
                    stream.flush()
                    os.fsync(stream.fileno())
                temporary.replace(destination)
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
            _CLEF_MLP_CAPTURED.add(key)
    except Exception as exc:
        warning_key = (str(arm_path), str(exc))
        if warning_key not in _CLEF_MLP_CAPTURE_WARNED:
            _CLEF_MLP_CAPTURE_WARNED.add(warning_key)
            logger.warning("Clef MLP input capture failed for %s: %s", prefix, exc)


'''


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def patched_source(preimage: bytes) -> bytes:
    text = preimage.decode("utf-8")
    if text.count(CLASS_ANCHOR) != 1:
        raise ValueError("Qwen2MoeMLP class anchor must occur exactly once")
    start = text.index(CLASS_ANCHOR)
    end = text.find("\nclass ", start + len(CLASS_ANCHOR))
    if end < 0:
        end = len(text)
    body = text[start:end]
    for name, anchor in (("initialization", INIT_ANCHOR), ("forward", FORWARD_ANCHOR)):
        if body.count(anchor) != 1:
            raise ValueError(f"Qwen2MoeMLP {name} anchor must occur exactly once")
    body = body.replace(
        INIT_ANCHOR,
        "        super().__init__()\n"
        "        self._clef_mlp_prefix = prefix\n"
        "        disable_tp = disable_tp or is_sequence_parallel\n",
        1,
    )
    body = body.replace(
        FORWARD_ANCHOR,
        "    def forward(self, x):\n"
        "        _clef_capture_mlp_input(self._clef_mlp_prefix, x)\n"
        "        gate_up, _ = self.gate_up_proj(x)\n",
        1,
    )
    result = text[:start] + CAPTURE_HELPER + body + text[end:]
    ast.parse(result, filename=TARGET_RELATIVE)
    return result.encode("utf-8")


def write_atomic(path: Path, data: bytes, mode: int) -> None:
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=path.parent, prefix=f".{path.name}.", delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.chmod(mode)
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--apply", action="store_true")
    action.add_argument("--revert", action="store_true")
    parser.add_argument("--vllm-tree", type=Path, default=Path("/workspace/yangchen/work/src/vllm"))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    tree = args.vllm_tree.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    target = tree / TARGET_RELATIVE
    backup = output / "qwen2_moe.py.preimage"
    tree_fd = os.open(tree, os.O_RDONLY)
    try:
        fcntl.flock(tree_fd, fcntl.LOCK_EX)
        current = target.read_bytes()
        before_sha = sha256(current)
        mode = target.stat().st_mode & 0o777
        preimage = None
        if backup.exists():
            preimage = backup.read_bytes()
            if sha256(preimage) != PREIMAGE_SHA256:
                raise ValueError(f"preimage backup SHA-256 mismatch at {backup}")
        if before_sha == PREIMAGE_SHA256:
            preimage = current
        if preimage is None:
            raise ValueError(
                f"source SHA-256 {before_sha} requires a verified preimage backup at {backup}")
        patched = patched_source(preimage)
        patched_sha = sha256(patched)
        if before_sha not in (PREIMAGE_SHA256, patched_sha):
            raise ValueError(f"source SHA-256 {before_sha} matches neither supported image")
        desired = patched if args.apply else preimage
        if args.apply and not backup.exists():
            output.mkdir(parents=True, exist_ok=True)
            write_atomic(backup, preimage, mode)
            if sha256(backup.read_bytes()) != PREIMAGE_SHA256:
                raise ValueError(f"preimage backup verification failed at {backup}")
        changed = current != desired
        if changed:
            write_atomic(target, desired, mode)
        after_sha = sha256(target.read_bytes())
        if after_sha != sha256(desired):
            raise ValueError("source verification failed after atomic replacement")
        print(json.dumps({
            "action": "apply" if args.apply else "revert",
            "changed": changed,
            "target": str(target),
            "backup": str(backup),
            "preimage_sha256": PREIMAGE_SHA256,
            "postimage_sha256": patched_sha,
            "sha256_before": before_sha,
            "sha256_after": after_sha,
            "capture_environment": "CLEF_MLP_CAPTURE_DIR",
            "arm_file": "arm.json",
        }, ensure_ascii=False, indent=2))
    finally:
        os.close(tree_fd)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError) as error:
        raise SystemExit(str(error)) from error
