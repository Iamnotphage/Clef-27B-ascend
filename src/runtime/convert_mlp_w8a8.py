#!/usr/bin/env python3
"""Convert selected Clef MLP projections to native Ascend W8A8_DYNAMIC on CPU.

Requires torch and safetensors. Select --layers all or comma-separated indices,
and --projections gate_up/down/both. At most one source shard, its output shard,
and a row-block FP32 workspace are held in memory. All unselected tensors retain
their original dtype and values. Unchanged files use hardlinks with a copy
fallback, or --unchanged-files copy for independent files.

The final output path becomes available by atomic directory rename after
conversion-manifest.json has status=complete. Interrupted work resumes in the
sibling .<output-name>.w8a8-inprogress directory when source hashes and options
match. Repeating a completed conversion verifies every output file. Existing
files with mismatched metadata or hashes are refused.

Example
  python runtime/convert_mlp_w8a8.py --source-dir /models/Clef-27B \
    --output-dir /models/Clef-27B-mlp-w8a8 --layers 0,31,63 --projections both
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import errno
import fcntl
import gc
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import shutil
import sys


VERSION = 1
INDEX = "model.safetensors.index.json"
DESCRIPTION = "quant_model_description.json"
MANIFEST = "conversion-manifest.json"
STATE = ".conversion-state.json"
WORKING = ".conversion-working.safetensors"
CHUNK_BYTES = 8 * 1024 * 1024
DTYPE_BYTES = {"BOOL": 1, "U8": 1, "I8": 1, "I16": 2, "F16": 2,
               "BF16": 2, "I32": 4, "F32": 4, "I64": 8, "F64": 8}


def json_bytes(value) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode()


def file_info(path: Path) -> dict:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(CHUNK_BYTES), b""):
            digest.update(chunk)
            size += len(chunk)
    return {"bytes": size, "sha256": digest.hexdigest()}


def checked_path(root: Path, name: str) -> Path:
    relative = PurePosixPath(name)
    if relative.is_absolute() or ".." in relative.parts or str(relative) != name:
        raise ValueError(f"invalid relative model path: {name!r}")
    path = root.joinpath(*relative.parts)
    if not path.resolve().is_relative_to(root):
        raise ValueError(f"model path resolves outside its directory: {path}")
    return path


def write_atomic(path: Path, data: bytes) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def quantize_weight(weight, block_rows: int, torch):
    """Use the saved BF16 scale for division, matching probe_mlp_w8a8.py."""
    quantized = torch.empty(weight.shape, dtype=torch.int8, device="cpu")
    scales = torch.empty((weight.shape[0], 1), dtype=torch.bfloat16, device="cpu")
    for start in range(0, weight.shape[0], block_rows):
        end = min(start + block_rows, weight.shape[0])
        block = weight[start:end].to(dtype=torch.float32, device="cpu", copy=True)
        if not torch.isfinite(block).all():
            raise ValueError(f"non-finite weight values in rows {start}:{end}")
        scale = block.abs().amax(dim=1, keepdim=True) / 127.0
        scale.masked_fill_(scale == 0, 1.0)
        scale = scale.to(torch.bfloat16)
        if not (torch.isfinite(scale).all() and (scale > 0).all()):
            raise ValueError(f"unrepresentable BF16 weight scale in rows {start}:{end}")
        block.div_(scale.float()).round_().clamp_(-127, 127)
        quantized[start:end].copy_(block.to(torch.int8))
        scales[start:end].copy_(scale)
    return quantized, scales, torch.zeros_like(scales)


def convert_shard(source: Path, target: Path, selected: set[str], block_rows: int, torch, safe_open, save_file):
    tensors = {}
    with safe_open(source, framework="pt", device="cpu") as reader:
        metadata = reader.metadata()
        for name in reader.keys():
            weight = reader.get_tensor(name)
            if name in selected:
                quantized, scale, offset = quantize_weight(weight, block_rows, torch)
                prefix = name.removesuffix(".weight")
                tensors[name] = quantized
                tensors[prefix + ".weight_scale"] = scale
                tensors[prefix + ".weight_offset"] = offset
            else:
                tensors[name] = weight
        save_file(tensors, target, metadata=metadata)
    del tensors
    gc.collect()


def link_or_copy(source: Path, target: Path, mode: str) -> str:
    if mode == "hardlink":
        try:
            os.link(source, target)
            return "hardlink"
        except OSError as error:
            if error.errno not in (errno.EXDEV, errno.EPERM, errno.EACCES, errno.ENOTSUP):
                raise
    shutil.copyfile(source, target)
    return "copy"


def prepare_plan(args, safe_open) -> tuple[dict, dict, dict, set[str]]:
    source = args.source_dir
    if (source / DESCRIPTION).exists() or (source / MANIFEST).exists():
        raise ValueError("source already contains quantization or conversion metadata")
    config = json.loads((source / "config.json").read_text())
    if config.get("quantization_config"):
        raise ValueError("source config already declares quantization")
    index = json.loads((source / INDEX).read_text())
    weight_map = index["weight_map"]
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError("source weight_map must be a nonempty object")
    layer_count = config.get("text_config", config)["num_hidden_layers"]
    if args.layers == "all":
        layers = list(range(layer_count))
    else:
        layers = sorted({int(value.strip()) for value in args.layers.split(",")})
        if not layers or any(layer < 0 or layer >= layer_count for layer in layers):
            raise ValueError(f"--layers must select indices in [0, {layer_count - 1}]")
    projections = {"gate_up": ["gate_proj", "up_proj"], "down": ["down_proj"],
                   "both": ["gate_proj", "up_proj", "down_proj"]}[args.projections]
    selected = {f"model.language_model.layers.{layer}.mlp.{projection}.weight"
                for layer in layers for projection in projections}
    missing = selected - set(weight_map)
    if missing:
        raise ValueError(f"selected tensors missing from source index: {sorted(missing)}")
    parameters = {}
    for shard in sorted(set(weight_map.values())):
        with safe_open(checked_path(source, shard), framework="pt", device="cpu") as reader:
            indexed_names = {name for name, filename in weight_map.items() if filename == shard}
            if set(reader.keys()) != indexed_names:
                raise ValueError(f"source index and shard tensor names disagree: {shard}")
            for name in reader.keys():
                view = reader.get_slice(name)
                shape, dtype = view.get_shape(), view.get_dtype()
                if dtype not in DTYPE_BYTES:
                    raise ValueError(f"unsupported source dtype {dtype}: {name}")
                quantized = name in selected
                if quantized:
                    if dtype not in ("BF16", "F16", "F32") or len(shape) != 2 or min(shape) < 1:
                        raise ValueError(f"selected weight must be a nonempty floating matrix: {name}")
                    prefix = name.removesuffix(".weight")
                    if any(prefix + suffix in weight_map for suffix in (".weight_scale", ".weight_offset")):
                        raise ValueError(f"source already contains quantization parameters for {name}")
                parameters[name] = {
                    "source_shard": shard, "output_shard": shard, "shape": shape,
                    "source_dtype": dtype, "output_dtype": "I8" if quantized else dtype,
                    "quantization": "W8A8_DYNAMIC" if quantized else "FLOAT",
                }
                if quantized:
                    for suffix, role in ((".weight_scale", "per_output_channel_scale"), (".weight_offset", "zero_offset")):
                        parameters[prefix + suffix] = {
                            "source_weight": name, "output_shard": shard, "shape": [shape[0], 1],
                            "output_dtype": "BF16", "role": role,
                        }
    names = set(weight_map.values()) | {path.name for path in source.iterdir() if path.is_file()}
    required = {"config.json", INDEX, "tokenizer_config.json", "joint_head_config.json", "joint_head.safetensors"}
    if not required.issubset(names):
        raise ValueError(f"required model artifacts missing: {sorted(required - names)}")
    source_files = {}
    for name in sorted(names):
        source_files[name] = file_info(checked_path(source, name))
        print(json.dumps({"source_verified": name, **source_files[name]}), flush=True)
    with safe_open(source / "joint_head.safetensors", framework="pt", device="cpu") as reader:
        secondary = {name: {"shape": reader.get_slice(name).get_shape(), "dtype": reader.get_slice(name).get_dtype(),
                            "file": "joint_head.safetensors"} for name in reader.keys()}
    plan = {
        "converter_version": VERSION, "converter_sha256": file_info(Path(__file__).resolve())["sha256"],
        "source_directory": str(source), "layers": layers, "projections": projections,
        "row_block_size": args.row_block_size, "unchanged_files": args.unchanged_files,
        "quantization": "symmetric_int8_per_output_channel_bf16_scale_then_ties_to_even",
        "integer_range": [-127, 127], "zero_row_scale": 1.0,
        "source_files": source_files, "parameters": parameters, "secondary_parameters": secondary,
    }
    return plan, config, index, selected


def verify_file(path: Path, expected: dict) -> None:
    if file_info(path) != {key: expected[key] for key in ("bytes", "sha256")}:
        raise ValueError(f"existing file does not match its recorded hash: {path}")


def verify_files(directory: Path, records: dict) -> None:
    for name, expected in records.items():
        verify_file(checked_path(directory, name), expected)


def publish(staging: Path, destination: Path) -> None:
    (staging / STATE).unlink(missing_ok=True)
    if destination.exists():
        destination.rmdir()  # An existing empty destination is supported.
    staging.rename(destination)


def resume_state(staging: Path, fingerprint: str) -> dict:
    state = json.loads((staging / STATE).read_text())
    if state.get("plan_sha256") != fingerprint:
        raise ValueError("existing staging belongs to a different source or conversion plan")
    working = staging / WORKING
    pending = state.get("pending")
    if pending:
        target = checked_path(staging, pending["name"])
        if target.exists():
            verify_file(target, pending)
        elif working.exists():
            verify_file(working, pending)
            target.parent.mkdir(parents=True, exist_ok=True)
            working.replace(target)
        else:
            raise ValueError("journaled pending output is missing")
        state["output_files"][pending["name"]] = {key: pending[key] for key in ("bytes", "sha256", "method")}
        state.pop("pending")
        write_atomic(staging / STATE, json_bytes(state))
    # These paths belong to this verified conversion and contain unfinished
    # artifacts only. A crash before journaling may leave one of them behind.
    for name in (WORKING, STATE + ".tmp", MANIFEST + ".tmp"):
        (staging / name).unlink(missing_ok=True)
    verify_files(staging, state["output_files"])
    allowed = set(state["output_files"]) | {STATE, MANIFEST}
    unexpected = {str(path.relative_to(staging)) for path in staging.rglob("*") if path.is_file()} - allowed
    if unexpected:
        raise ValueError(f"unrecorded staging files require inspection: {sorted(unexpected)}")
    return state


def convert(args, torch, safe_open, save_file) -> dict:
    plan, config, index, selected = prepare_plan(args, safe_open)
    fingerprint = hashlib.sha256(json_bytes(plan)).hexdigest()
    destination = args.output_dir
    staging = destination.parent / f".{destination.name}.w8a8-inprogress"
    expected_files = set(plan["source_files"]) | {DESCRIPTION}
    for directory in (destination, staging):
        if (directory / MANIFEST).is_file():
            manifest = json.loads((directory / MANIFEST).read_text())
            if manifest.get("status") != "complete" or manifest.get("plan_sha256") != fingerprint:
                raise ValueError(f"existing output belongs to a different conversion plan: {directory}")
            if any(manifest.get(key) != value for key, value in plan.items()):
                raise ValueError(f"existing conversion manifest metadata differs from the plan: {directory}")
            if set(manifest["output_files"]) != expected_files:
                raise ValueError(f"completed manifest has an incomplete output file set: {directory}")
            verify_files(directory, manifest["output_files"])
            if directory == staging:
                publish(staging, destination)
            return {"status": "verified_existing", "output_dir": str(destination), "plan_sha256": fingerprint}
    if destination.exists() and any(destination.iterdir()):
        raise ValueError(f"existing output lacks a completed conversion manifest: {destination}")
    if staging.exists() and not (staging / STATE).exists():
        entries = {path.name for path in staging.iterdir()}
        if entries - {STATE + ".tmp"}:
            raise ValueError(f"staging lacks a valid conversion state: {staging}")
        # No artifacts were published if initial state creation was interrupted.
        (staging / (STATE + ".tmp")).unlink(missing_ok=True)
        state = {"status": "in_progress", "plan_sha256": fingerprint, "output_files": {}}
        write_atomic(staging / STATE, json_bytes(state))
    if staging.exists():
        state = resume_state(staging, fingerprint)
    else:
        staging.mkdir()
        state = {"status": "in_progress", "plan_sha256": fingerprint, "output_files": {}}
        write_atomic(staging / STATE, json_bytes(state))
    by_shard = defaultdict(set)
    for name in selected:
        by_shard[index["weight_map"][name]].add(name)
    parameters = plan["parameters"]
    output_index = {**index, "weight_map": {name: item["output_shard"] for name, item in parameters.items()}}
    metadata = dict(index.get("metadata", {}))
    metadata["total_size"] = sum(math.prod(item["shape"]) * DTYPE_BYTES[item["output_dtype"]] for item in parameters.values())
    metadata["total_parameters"] = sum(math.prod(item["shape"]) for item in parameters.values())
    output_index["metadata"] = metadata
    config = {**config, "quantization_config": {"quant_method": "ascend"}}
    description = {name: "W8A8_DYNAMIC" if name in selected else "FLOAT" for name in index["weight_map"]}
    generated = {"config.json": json_bytes(config), INDEX: json_bytes(output_index), DESCRIPTION: json_bytes(description)}
    for name in sorted(set(plan["source_files"]) | set(generated)):
        if name in state["output_files"]:
            continue
        target = checked_path(staging, name)
        if target.exists():
            raise ValueError(f"refusing to overwrite unrecorded destination: {target}")
        working = staging / WORKING
        if name in generated:
            working.write_bytes(generated[name])
            method = "generated"
        elif name in by_shard:
            convert_shard(args.source_dir / name, working, by_shard[name], args.row_block_size, torch, safe_open, save_file)
            method = "converted"
        else:
            method = link_or_copy(args.source_dir / name, working, args.unchanged_files)
        with working.open("rb") as stream:
            os.fsync(stream.fileno())
        info = {**file_info(working), "method": method}
        state["pending"] = {"name": name, **info}
        write_atomic(staging / STATE, json_bytes(state))
        target.parent.mkdir(parents=True, exist_ok=True)
        working.replace(target)
        state["output_files"][name] = info
        state.pop("pending")
        write_atomic(staging / STATE, json_bytes(state))
        print(json.dumps({"output_verified": name, **info}), flush=True)
    verify_files(staging, state["output_files"])
    verify_files(args.source_dir, plan["source_files"])
    manifest = {"status": "complete", "completed_at": datetime.now(timezone.utc).isoformat(),
                "plan_sha256": fingerprint, **plan, "output_files": state["output_files"]}
    write_atomic(staging / MANIFEST, json_bytes(manifest))
    publish(staging, destination)
    return {"status": "complete", "output_dir": str(destination), "plan_sha256": fingerprint,
            "quantized_weights": len(selected), "output_tensor_bytes": metadata["total_size"]}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--layers", required=True, help="all or comma-separated layer indices")
    parser.add_argument("--projections", choices=("gate_up", "down", "both"), default="both")
    parser.add_argument("--row-block-size", type=int, default=128)
    parser.add_argument("--unchanged-files", choices=("hardlink", "copy"), default="hardlink")
    args = parser.parse_args(argv)
    if args.row_block_size < 1:
        parser.error("--row-block-size must be positive")
    args.source_dir = args.source_dir.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    if args.source_dir.is_relative_to(args.output_dir) or args.output_dir.is_relative_to(args.source_dir):
        parser.error("source and output must be separate, non-nested directories")
    # Disable automatic accelerator-backend imports before importing torch.
    os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    lock_path = args.output_dir.parent / f".{args.output_dir.name}.w8a8.lock"
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = convert(args, torch, safe_open, save_file)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError, ImportError) as error:
        print(f"conversion failed: {error}", file=sys.stderr)
        raise SystemExit(1) from error
