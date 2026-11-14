#!/usr/bin/env python3
"""Independent CPU BF16 reference forward for the frozen Clef baselines.

The publisher's fixed ``joint_schema_model.py`` is loaded from the model
directory with the same weights as the NPU service, then one batch-1 eager
forward runs per sample.  Every sample record stores the exact prompt token ids,
the question/option spans, the raw logits and the unrounded probabilities, plus
the execution identity that a later resume must match.

The runner never contacts the service: the comparison inputs are the frozen wire
requests saved by the NPU runs.  Writes are atomic per sample so a long run can
be resumed after a restart.

Media layout is the publisher reference's own: ``PLACEHOLDER * n + "\\n"``
(``_encode_media``).  The fixed serving strategy used by this service,
``ClefFlashStrategy._render_media``, overrides the base class placeholders with
the same no-trailing-newline form and appends a single ``\\n``, so the two are
identical for any image count.  ``--encode-check`` proves this per sample by
comparing the encoded token count with the saved NPU ``usage.input_tokens``.
"""

from __future__ import annotations

import argparse
import base64
from collections import Counter
import hashlib
import importlib.metadata
import importlib.util
import io
import json
import os
import resource
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

REFERENCE_SHA256 = "0e304cf7c6500e8bb59bef7e2afd2c6373f82596dfb3b57d1aa93c175e2dc3a3"
RUN_SCHEMA = 1
OPENDECISIONS_DATA_SHA256 = (
    "78614972006c5e5efbbfc121ce31e15cf576ef6c0cdf24a229b33d34d2bfcda6"
)
OPENDECISIONS_RECORDS = 2352
ERQA_TFRECORD_SHA256 = (
    "1974f5e587eac144cc3ba325477477b8ad6a56c22ce4e419c34229520b4c93e2"
)
ERQA_RECORDS = 400
IMAGE_SIZE = {"shortest_edge": 65536, "longest_edge": 1048576}
DEFAULT_MAX_LENGTH = {"opendecisions": 4096, "erqa": 16384}


# ---------------------------------------------------------------------------
# small helpers


def now():
    return datetime.now(timezone.utc).isoformat()


def sha256_bytes(blob):
    return hashlib.sha256(blob).hexdigest()


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(value):
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )


def digest(value):
    return sha256_bytes(canonical_json(value).encode("utf-8"))


def atomic_write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def append_jsonl(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def rss_bytes():
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return usage if sys.platform == "darwin" else usage * 1024


def decode_data_uri(uri):
    """Decode a ``data:`` URI exactly as the service connector would."""
    if not isinstance(uri, str) or not uri.startswith("data:"):
        raise ValueError("image entry is not a data URI")
    header, _, payload = uri.partition(",")
    if ";base64" not in header:
        raise ValueError("image data URI is not base64 encoded")
    return base64.b64decode(payload)


def load_pil_images(uris):
    from PIL import Image

    images = []
    for uri in uris:
        blob = decode_data_uri(uri)
        with Image.open(io.BytesIO(blob)) as image:
            images.append(image.convert("RGB"))
    return images


# ---------------------------------------------------------------------------
# dataset loading and the frozen request for each sample


def read_jsonl(path):
    """Read JSONL by file line boundaries only.

    ``str.splitlines`` also splits on Unicode separators such as U+2028, which
    appear verbatim inside JSON strings in both frozen datasets.
    """
    rows = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_opendecisions(args):
    data_path = Path(args.data_dir) / "open-decisions2352.jsonl"
    if not data_path.is_file():
        raise SystemExit(f"missing OpenDecisions file: {data_path}")
    checksum = sha256_file(data_path)
    if checksum != OPENDECISIONS_DATA_SHA256:
        raise SystemExit(f"OpenDecisions SHA256 mismatch: {checksum}")
    tasks = read_jsonl(data_path)
    if len(tasks) != OPENDECISIONS_RECORDS:
        raise SystemExit(f"expected {OPENDECISIONS_RECORDS} records, got {len(tasks)}")
    if not args.npu_run:
        raise SystemExit("OpenDecisions needs --npu-run (the frozen /v1/systemone benchmark)")
    run = json.loads(Path(args.npu_run).read_text(encoding="utf-8"))
    saved = run["samples"]
    measured = {item["sample_name"]: item for item in run["requests"]}
    if len(saved) != len(tasks):
        raise SystemExit(f"saved sample count {len(saved)} != data {len(tasks)}")
    samples = []
    for index, (task, item) in enumerate(zip(tasks, saved)):
        name = item["name"]
        request = item["request"]
        if request.get("state") != task["state"]:
            raise SystemExit(f"{name}: saved state differs from the fixed record")
        expected_question = dict(task["question"])
        actual_question = request["questions"]["answer"]
        for key, value in expected_question.items():
            if actual_question.get(key) != value:
                raise SystemExit(f"{name}: saved question {key} differs from the record")
        record = measured.get(name)
        if record is None:
            raise SystemExit(f"{name}: no measured request in the NPU benchmark")
        samples.append(
            {
                "index": index,
                "id": task["id"],
                "name": name,
                "request": request,
                "request_sha256": digest(request),
                "npu_input_tokens": record["actual_input_tokens"],
                "npu_phase": record["phase"],
                "question_type": task["question"]["type"],
                "gold": task.get("expected"),
                "labels": task.get("labels"),
                "images": [],
                "image_sha256": [],
            }
        )
    return samples, {"path": str(data_path), "sha256": checksum, "records": len(tasks)}


def load_erqa(args):
    data_dir = Path(args.data_dir)
    manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    source = manifest["source"]
    if source["sha256"] != ERQA_TFRECORD_SHA256:
        raise SystemExit(f"unexpected ERQA tfrecord SHA256: {source['sha256']}")
    records = read_jsonl(data_dir / "samples.jsonl")
    if len(records) != ERQA_RECORDS:
        raise SystemExit(f"expected {ERQA_RECORDS} ERQA records, got {len(records)}")
    if not args.npu_run:
        raise SystemExit("ERQA needs --npu-run (the frozen /v1/decide run directory)")
    run_dir = Path(args.npu_run)
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    predictions = {}
    with (run_dir / "predictions.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                predictions[row["id"]] = row
    samples = []
    for index, record in enumerate(records):
        sample_id = record["id"]
        request_path = run_dir / "samples" / sample_id / "request.json"
        if not request_path.is_file():
            raise SystemExit(f"{sample_id}: missing frozen wire request {request_path}")
        request = json.loads(request_path.read_text(encoding="utf-8"))
        prediction = predictions.get(sample_id)
        if prediction is None:
            raise SystemExit(f"{sample_id}: missing NPU prediction row")
        blobs = [decode_data_uri(uri) for uri in request.get("images") or []]
        hashes = [sha256_bytes(blob) for blob in blobs]
        if len(hashes) != record["image_count"]:
            raise SystemExit(
                f"{sample_id}: wire image count {len(hashes)} != record {record['image_count']}"
            )
        expected_hashes = [record["images"][i]["sha256"] for i in prediction["image_order"]]
        if hashes != expected_hashes:
            raise SystemExit(f"{sample_id}: wire image order differs from the NPU prediction row")
        criteria = request["questions"]["answer"]["criteria"]
        samples.append(
            {
                "index": index,
                "id": sample_id,
                "name": sample_id,
                "request": request,
                "request_sha256": digest(request),
                "npu_input_tokens": prediction["input_tokens"],
                "npu_phase": "measured",
                "question_type": record["question_type"],
                "gold": record["answer"],
                "image_count": record["image_count"],
                "option_count": len(criteria),
                "images": blobs,
                "image_sha256": hashes,
                "image_order": prediction["image_order"],
                "visual_index_pattern": record.get("visual_index_pattern"),
                "wire_request_sha256": prediction["request_wire_sha256"],
                "wire_structural_sha256": prediction["request_structural_sha256"],
            }
        )
    return samples, {
        "path": str(data_dir),
        "sha256": sha256_file(data_dir / "samples.jsonl"),
        "manifest_sha256": sha256_file(data_dir / "manifest.json"),
        "npu_run": str(run_dir),
        "npu_summary_accuracy": summary.get("accuracy"),
        "records": len(records),
    }


# ---------------------------------------------------------------------------
# reference loading, including the documented media-layout switch


def load_reference(model_dir, threads):
    os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
    # activate_env.sh defaults OMP_NUM_THREADS to 1 for NPU serving; without
    # this the CPU forward silently runs on a single core.
    os.environ["OMP_NUM_THREADS"] = str(threads)
    os.environ["MKL_NUM_THREADS"] = str(threads)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import torch

    torch.set_num_threads(threads)
    torch.manual_seed(0)
    source_path = Path(model_dir) / "joint_schema_model.py"
    checksum = sha256_file(source_path)
    if checksum != REFERENCE_SHA256:
        raise SystemExit(f"reference source SHA256 mismatch: {checksum}")
    spec = importlib.util.spec_from_file_location("clef_cpu_reference", source_path)
    release = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = release
    spec.loader.exec_module(release)
    model, processor = release.load_release_model(
        model_dir,
        device="cpu",
        dtype=torch.bfloat16,
        local_files_only=True,
        use_kernels=False,
        attn_implementation="eager",
    )
    return torch, release, (model, processor)


def weight_manifest(model_dir):
    index_path = Path(model_dir) / "model.safetensors.index.json"
    if not index_path.is_file():
        raise SystemExit(f"missing weight index: {index_path}")
    index = json.loads(index_path.read_text(encoding="utf-8"))
    names = sorted(set(index["weight_map"].values()) | {"joint_head.safetensors"})
    entries = []
    for name in names:
        path = (Path(model_dir) / name).resolve()
        if not path.is_file():
            raise SystemExit(f"missing weight file: {name}")
        entries.append(
            {
                "name": name,
                "bytes": path.stat().st_size,
                "sha256_prefix": sha256_file(path)[:16],
            }
        )
    return entries, index_path


# ---------------------------------------------------------------------------
# per-sample execution


def build_inputs(release, processor, sample, max_length, media_kwargs):
    import torch

    request = dict(sample["request"])
    if sample["images"]:
        request["images"] = load_pil_images(sample["request"]["images"])
        request["media_kwargs"] = media_kwargs
    encoded = release.encode_record(
        processor.tokenizer, request, max_length=max_length, processor=processor
    )
    questions = []
    for question in encoded.questions:
        questions.append(
            {
                "id": question.question_id,
                "type": question.question_type,
                "option_ids": list(question.option_ids),
                "question_span": list(question.question_span),
                "option_spans": [list(span) for span in question.option_spans],
            }
        )
    media = {}
    if encoded.media:
        for key, value in encoded.media.items():
            if key in ("pixel_values", "pixel_values_videos"):
                # The pixel tensors are tens of megabytes per image; keep their
                # identity (shape, dtype, byte hash) instead of the payload.
                tensor = value.detach().cpu().contiguous()
                media[f"{key}_shape"] = list(tensor.shape)
                media[f"{key}_dtype"] = str(tensor.dtype).replace("torch.", "")
                media[f"{key}_sha256"] = sha256_bytes(
                    tensor.view(torch.uint8).numpy().tobytes())
            elif key == "mm_token_type_ids":
                values = list(value)
                media["mm_token_type_ids_count"] = len(values)
                media["mm_token_type_ids_nonzero"] = sum(1 for item in values if item)
                media["mm_token_type_ids_sha256"] = digest(values)
            elif hasattr(value, "tolist"):
                media[key] = value.tolist()
            else:
                media[key] = value
    return encoded, {
        "input_tokens": len(encoded.input_ids),
        "input_ids": list(encoded.input_ids),
        "input_ids_sha256": digest(list(encoded.input_ids)),
        "questions": questions,
        "media": media,
    }


def run_sample(release, model_bundle, processor, sample, max_length, media_kwargs):
    torch, release_module = release
    model, _ = model_bundle
    started = time.perf_counter()
    encoded, inputs = build_inputs(
        release_module, processor, sample, max_length, media_kwargs
    )
    batch = release_module.collate_records(
        [encoded], processor.tokenizer.pad_token_id, torch.device("cpu")
    )
    with torch.inference_mode():
        logits = model(batch)[0]
    values = {}
    probabilities = {}
    for question, question_logits in zip(encoded.questions, logits):
        tensor = question_logits.float()
        if not torch.isfinite(tensor).all().item():
            raise ValueError(f"{sample['id']}/{question.question_id}: non-finite logits")
        values[question.question_id] = [float(value) for value in tensor.tolist()]
        probs = tensor.softmax(-1)
        probabilities[question.question_id] = {
            option_id: float(probability)
            for option_id, probability in zip(question.option_ids, probs.tolist())
        }
    elapsed = time.perf_counter() - started
    del batch, logits
    return inputs, values, probabilities, elapsed


def encode_check(args, samples, max_length, media_kwargs, model_dir, output_dir,
                 run_fingerprint, fingerprint_payload):
    """Compare the reference encoding against the saved NPU token counts.

    Runs the processor only (no weights), so it is cheap enough to cover every
    sample before a long forward.
    """
    os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import torch  # noqa: F401 - the reference module imports torch at module level

    source_path = Path(model_dir) / "joint_schema_model.py"
    spec = importlib.util.spec_from_file_location("clef_cpu_reference_encode", source_path)
    release = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = release
    spec.loader.exec_module(release)
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(model_dir, local_files_only=True)
    rows = []
    started = time.perf_counter()
    for position, sample in enumerate(samples, start=1):
        _, inputs = build_inputs(release, processor, sample, max_length, media_kwargs)
        media = inputs["media"]
        image_tokens = None
        if "image_grid_thw" in media:
            grid = media["image_grid_thw"]
            image_tokens = sum(
                int(part[0]) * int(part[1]) * int(part[2]) // 4 for part in grid
            )
        row = {
            "id": sample["id"],
            "cpu_input_tokens": inputs["input_tokens"],
            "npu_input_tokens": sample["npu_input_tokens"],
            "match": inputs["input_tokens"] == sample["npu_input_tokens"],
            "input_ids_sha256": inputs["input_ids_sha256"],
            "questions": inputs["questions"],
            "image_count": len(sample["image_sha256"]),
            "image_tokens": image_tokens,
        }
        rows.append(row)
        if position % 100 == 0 or position == len(samples):
            print(json.dumps({"encoded": position, "of": len(samples)}), flush=True)
    mismatches = [row for row in rows if not row["match"]]
    report = {
        "created_at": now(),
        "dataset": args.dataset,
        "run_fingerprint": run_fingerprint,
        "fingerprint_payload": fingerprint_payload,
        "max_length": max_length,
        "media_kwargs": media_kwargs,
        "samples": len(rows),
        "matches": len(rows) - len(mismatches),
        "mismatches": mismatches,
        "elapsed_seconds": time.perf_counter() - started,
        "token_counts": rows,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output_dir / "encode-check.json", report)
    print(json.dumps({"status": "ok" if not mismatches else "mismatch",
                      "samples": len(rows), "matches": report["matches"],
                      "mismatches": len(mismatches), "output": str(output_dir)}))
    return 0 if not mismatches else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=sorted(DEFAULT_MAX_LENGTH), required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--npu-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path,
                        default=Path(os.environ.get("MODEL_DIR", "/workspace/yangchen/models/Clef-27B")))
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--sample-ids", type=Path, default=None,
                        help="JSON list of sample ids to run (frozen pilot selection)")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed-from", type=Path, default=None,
                        help="import matching per-sample records from an earlier run")
    parser.add_argument("--max-failures", type=int, default=20)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--encode-check", action="store_true",
                        help="encode every selected request and compare token counts "
                             "with the saved NPU usage.input_tokens; no model forward")
    args = parser.parse_args()

    if args.threads <= 0:
        parser.error("--threads must be positive")
    max_length = args.max_length or DEFAULT_MAX_LENGTH[args.dataset]
    model_dir = args.model_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    if args.dataset == "opendecisions":
        samples, data_info = load_opendecisions(args)
    else:
        samples, data_info = load_erqa(args)

    if args.sample_ids:
        wanted = json.loads(Path(args.sample_ids).read_text(encoding="utf-8"))
        if not isinstance(wanted, list):
            raise SystemExit("--sample-ids must be a JSON list")
        by_id = {sample["id"]: sample for sample in samples}
        missing = [sample_id for sample_id in wanted if sample_id not in by_id]
        if missing:
            raise SystemExit(f"unknown sample ids: {missing[:5]} ({len(missing)} total)")
        samples = [by_id[sample_id] for sample_id in wanted]
    if args.offset:
        samples = samples[args.offset:]
    if args.limit is not None:
        samples = samples[: args.limit]
    if not samples:
        raise SystemExit("no samples selected")

    weights, index_path = weight_manifest(model_dir)
    media_kwargs = {"images_kwargs": {"size": IMAGE_SIZE}} if args.dataset == "erqa" else None

    versions = {}
    os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
    for name in ("torch", "transformers", "accelerate", "safetensors", "pillow", "numpy"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None

    fingerprint_payload = {
        "schema": RUN_SCHEMA,
        "dataset": args.dataset,
        "model_dir": str(model_dir),
        "weights": weights,
        "weight_index_sha256": sha256_file(index_path),
        "reference_sha256": REFERENCE_SHA256,
        "runner_sha256": sha256_file(Path(__file__)),
        "media_kwargs": media_kwargs,
        "dtype": "bfloat16",
        "device": "cpu",
        "attn_implementation": "eager",
        "use_kernels": False,
        "max_length": max_length,
        "threads": args.threads,
        "versions": versions,
        "data": data_info,
    }
    run_fingerprint = digest(fingerprint_payload)

    if args.encode_check:
        return encode_check(
            args, samples, max_length, media_kwargs, model_dir, output_dir,
            run_fingerprint, fingerprint_payload,
        )

    identity_path = output_dir / "run-identity.json"
    identity = {
        "created_at": now(),
        "schema": RUN_SCHEMA,
        "run_fingerprint": run_fingerprint,
        "fingerprint_payload": fingerprint_payload,
        "selected_samples": len(samples),
        "sample_ids_sha256": digest([sample["id"] for sample in samples]),
        "selection": {
            "sample_ids": str(args.sample_ids.resolve()) if args.sample_ids else None,
            "offset": args.offset,
            "limit": args.limit,
        },
        "command": sys.argv,
    }
    if identity_path.is_file():
        existing = json.loads(identity_path.read_text(encoding="utf-8"))
        if not args.resume:
            raise SystemExit(
                f"{output_dir} already has a run identity; pass --resume or use a new directory"
            )
        if existing.get("run_fingerprint") != run_fingerprint:
            raise SystemExit(
                "existing run identity does not match this configuration; "
                "refusing to resume into it"
            )
        identity = existing
        identity["resumed_at"] = now()
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(identity_path, identity)

    summary = {
        "schema": RUN_SCHEMA,
        "dataset": args.dataset,
        "run_fingerprint": run_fingerprint,
        "started_at": now(),
        "planned_samples": len(samples),
        "completed": 0,
        "failed": 0,
        "skipped": 0,
        "imported": 0,
        "token_count_matches": 0,
        "token_count_mismatches": [],
        "elapsed_seconds": 0.0,
        "peak_rss_bytes": rss_bytes(),
        "status": "running" if not args.dry_run else "planned",
    }
    atomic_write_json(output_dir / "summary.json", summary)

    if args.dry_run:
        planned = [
            {
                "id": sample["id"],
                "name": sample["name"],
                "request_sha256": sample["request_sha256"],
                "npu_input_tokens": sample["npu_input_tokens"],
                "image_count": len(sample["image_sha256"]),
            }
            for sample in samples
        ]
        atomic_write_json(
            output_dir / "plan.json",
            {"run_fingerprint": run_fingerprint, "samples": planned},
        )
        print(json.dumps({"status": "planned", "samples": len(planned)}, ensure_ascii=False))
        return 0

    torch, release, bundle = load_reference(model_dir, args.threads)
    processor = bundle[1]

    samples_dir = output_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / "progress.jsonl"

    parameters = Counter()
    for parameter in bundle[0].parameters():
        parameters[f"{parameter.device}/{parameter.dtype}"] += parameter.numel()
    summary["parameter_counts"] = dict(parameters)
    summary["torch_num_threads"] = torch.get_num_threads()
    summary["load_complete_at"] = now()
    atomic_write_json(output_dir / "summary.json", summary)

    if args.seed_from:
        source_dir = Path(args.seed_from).expanduser().resolve()
        imported = 0
        for sample in samples:
            target = samples_dir / f"{sample['id']}.json"
            if target.exists():
                continue
            candidate = source_dir / "samples" / f"{sample['id']}.json"
            if not candidate.is_file():
                continue
            record = json.loads(candidate.read_text(encoding="utf-8"))
            if (
                record.get("run_fingerprint") == run_fingerprint
                and record.get("sample_identity", {}).get("request_sha256")
                == sample["request_sha256"]
                and record.get("status") == "ok"
            ):
                atomic_write_json(target, record)
                imported += 1
        summary["imported"] = imported
        atomic_write_json(output_dir / "summary.json", summary)

    started_all = time.perf_counter()
    for position, sample in enumerate(samples, start=1):
        target = samples_dir / f"{sample['id']}.json"
        sample_identity = {
            "id": sample["id"],
            "dataset": args.dataset,
            "request_sha256": sample["request_sha256"],
            "image_sha256": sample["image_sha256"],
            "image_order": sample.get("image_order"),
            "npu_input_tokens": sample["npu_input_tokens"],
            "max_length": max_length,
            }
        if target.is_file():
            existing = json.loads(target.read_text(encoding="utf-8"))
            if (
                existing.get("run_fingerprint") == run_fingerprint
                and existing.get("sample_identity") == sample_identity
                and existing.get("status") == "ok"
            ):
                summary["skipped"] += 1
                if existing["input"]["input_tokens"] == sample["npu_input_tokens"]:
                    summary["token_count_matches"] += 1
                else:
                    summary["token_count_mismatches"].append(
                        {
                            "id": sample["id"],
                            "cpu_input_tokens": existing["input"]["input_tokens"],
                            "npu_input_tokens": sample["npu_input_tokens"],
                        }
                    )
                continue
        record = {
            "schema": RUN_SCHEMA,
            "id": sample["id"],
            "dataset": args.dataset,
            "run_fingerprint": run_fingerprint,
            "sample_identity": sample_identity,
            "status": "error",
            "attempts": [],
            "started_at": now(),
        }
        attempt = {"started_at": now()}
        try:
            inputs, logits, probabilities, elapsed = run_sample(
                (torch, release), bundle, processor, sample, max_length, media_kwargs
            )
            attempt["status"] = "ok"
            attempt["finished_at"] = now()
            record.update(
                {
                    "status": "ok",
                    "finished_at": attempt["finished_at"],
                    "elapsed_seconds": elapsed,
                    "input": inputs,
                    "logits": logits,
                    "probabilities": probabilities,
                }
            )
            record["encoding_matches_npu"] = (
                inputs["input_tokens"] == sample["npu_input_tokens"]
            )
            if record["encoding_matches_npu"]:
                summary["token_count_matches"] += 1
            else:
                summary["token_count_mismatches"].append(
                    {
                        "id": sample["id"],
                        "cpu_input_tokens": inputs["input_tokens"],
                        "npu_input_tokens": sample["npu_input_tokens"],
                    }
                )
            summary["completed"] += 1
        except Exception as exc:  # noqa: BLE001 - the record keeps every failure
            attempt["status"] = "error"
            attempt["finished_at"] = now()
            attempt["error"] = f"{type(exc).__name__}: {exc}"
            attempt["traceback"] = traceback.format_exc()
            record["status"] = "error"
            record["error"] = attempt["error"]
            summary["failed"] += 1
            if summary["failed"] > args.max_failures:
                record["attempts"].append(attempt)
                atomic_write_json(target, record)
                summary["status"] = "aborted-too-many-failures"
                break
        record["attempts"].append(attempt)
        record["peak_rss_bytes"] = rss_bytes()
        atomic_write_json(target, record)
        summary["elapsed_seconds"] = time.perf_counter() - started_all
        summary["peak_rss_bytes"] = rss_bytes()
        if record["status"] == "ok":
            done = summary["completed"]
            average = summary["elapsed_seconds"] / max(done, 1)
            summary["average_sample_seconds"] = average
            summary["eta_seconds"] = average * max(summary["planned_samples"] - position, 0)
        summary["last_sample"] = sample["id"]
        atomic_write_json(output_dir / "summary.json", summary)
        append_jsonl(
            progress_path,
            {
                "at": now(),
                "position": position,
                "id": sample["id"],
                "status": record["status"],
                "elapsed_seconds": record.get("elapsed_seconds"),
                "input_tokens": record.get("input", {}).get("input_tokens"),
                "npu_input_tokens": sample["npu_input_tokens"],
                "peak_rss_bytes": summary["peak_rss_bytes"],
                "error": record.get("error"),
            },
        )
        print(
            json.dumps(
                {
                    "position": position,
                    "of": len(samples),
                    "id": sample["id"],
                    "status": record["status"],
                    "input_tokens": record.get("input", {}).get("input_tokens"),
                    "seconds": round(record.get("elapsed_seconds") or 0.0, 2),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    summary["finished_at"] = now()
    summary["elapsed_seconds"] = time.perf_counter() - started_all
    summary["peak_rss_bytes"] = rss_bytes()
    if summary["status"] == "running":
        summary["status"] = (
            "complete"
            if summary["completed"] + summary["skipped"] == len(samples)
            else "incomplete"
        )
    atomic_write_json(output_dir / "summary.json", summary)
    print(json.dumps({"status": summary["status"], "output_dir": str(output_dir)}))
    return 0 if summary["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
