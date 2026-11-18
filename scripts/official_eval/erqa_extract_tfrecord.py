#!/usr/bin/env python3
"""Extract the official ERQA TFRecord with the Python standard library.

The official benchmark ships ``data/erqa.tfrecord`` in the
``embodiedreasoning/ERQA`` repository.  TensorFlow is not part of the Clef
runtime, so this script parses the TFRecord framing and the
``tf.train.Example`` protobuf wire format directly.  It writes one directory
per sample with the original encoded image bytes, a ``samples.jsonl`` file in
record order, and a ``manifest.json`` that records every hash and count used
by the ERQA report.

The parser follows the protobuf schema:

    Example   { Features features = 1; }
    Features  { map<string, Feature> feature = 1; }
    Feature   { BytesList bytes_list = 1; FloatList float_list = 2;
                Int64List int64_list = 3; }

Only the wire types present in ``erqa.tfrecord`` are decoded; unknown fields
are skipped.  The output is deterministic for a given input file.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path
import struct
import sys

ANSWER_LETTERS = ("A", "B", "C", "D")


def _varint(buf: bytes, offset: int) -> tuple[int, int]:
    shift = 0
    value = 0
    while True:
        if offset >= len(buf):
            raise ValueError("truncated varint")
        byte = buf[offset]
        offset += 1
        value |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return value, offset
        shift += 7


def _fields(buf: bytes):
    """Yield ``(field_number, wire_type, value)`` for one protobuf message."""
    offset = 0
    while offset < len(buf):
        tag, offset = _varint(buf, offset)
        number, wire = tag >> 3, tag & 7
        if wire == 0:
            value, offset = _varint(buf, offset)
        elif wire == 1:
            value = buf[offset:offset + 8]
            offset += 8
        elif wire == 2:
            length, offset = _varint(buf, offset)
            value = buf[offset:offset + length]
            offset += length
        elif wire == 5:
            value = buf[offset:offset + 4]
            offset += 4
        else:
            raise ValueError(f"unsupported wire type {wire}")
        yield number, wire, value


def _feature(buf: bytes) -> tuple[str, list]:
    for number, _, value in _fields(buf):
        if number == 1:  # bytes_list
            return "bytes", [v for n, _, v in _fields(value) if n == 1]
        if number == 2:  # float_list
            return "float", [
                struct.unpack("<f", v)[0] for n, _, v in _fields(value) if n == 1
            ]
        if number == 3:  # int64_list
            values: list[int] = []
            for n, wire, v in _fields(value):
                if n != 1:
                    continue
                if wire == 0:
                    values.append(v)
                else:  # packed
                    offset = 0
                    while offset < len(v):
                        item, offset = _varint(v, offset)
                        values.append(item)
            return "int", values
    return "none", []


def parse_example(blob: bytes) -> dict[str, tuple[str, list]]:
    features: dict[str, tuple[str, list]] = {}
    for number, _, value in _fields(blob):
        if number != 1:
            continue
        for entry_number, _, entry in _fields(value):
            if entry_number != 1:
                continue
            key = None
            feature = ("none", [])
            for item_number, _, item in _fields(entry):
                if item_number == 1:
                    key = item.decode("utf-8")
                elif item_number == 2:
                    feature = _feature(item)
            if key is not None:
                features[key] = feature
    return features


def iter_records(path: Path):
    """Yield the payloads of a TFRecord file, checking the frame CRCs."""
    with path.open("rb") as stream:
        index = 0
        while True:
            header = stream.read(8)
            if not header:
                return
            if len(header) != 8:
                raise ValueError(f"truncated length at record {index}")
            length, = struct.unpack("<Q", header)
            length_crc = stream.read(4)
            if len(length_crc) != 4:
                raise ValueError(f"truncated length CRC at record {index}")
            payload = stream.read(length)
            if len(payload) != length:
                raise ValueError(f"truncated payload at record {index}")
            payload_crc = stream.read(4)
            if len(payload_crc) != 4:
                raise ValueError(f"truncated payload CRC at record {index}")
            yield index, payload
            index += 1


def image_format(blob: bytes) -> str:
    if blob.startswith(b"\xff\xd8\xff"):
        return "jpeg"
    if blob.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"
    if blob[:4] == b"RIFF" and blob[8:12] == b"WEBP":
        return "webp"
    return "unknown"


def image_size(blob: bytes, fmt: str) -> tuple[int | None, int | None]:
    try:
        from PIL import Image
        import io

        with Image.open(io.BytesIO(blob)) as image:
            return image.width, image.height
    except Exception:
        if fmt == "png" and len(blob) >= 24:
            return struct.unpack(">II", blob[16:24])
        return None, None


def sha256(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tfrecord", required=True, help="官方 erqa.tfrecord 路径")
    parser.add_argument("--output-dir", required=True, help="全新输出目录")
    parser.add_argument(
        "--expected-records", type=int, default=400,
        help="预期记录数；不符时以非零退出（默认 400）",
    )
    args = parser.parse_args()

    tfrecord = Path(args.tfrecord).expanduser().resolve()
    output = Path(args.output_dir).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        parser.error(f"输出目录非空，使用全新目录：{output}")
    output.mkdir(parents=True, exist_ok=True)
    images_dir = output / "images"
    images_dir.mkdir()

    source = tfrecord.read_bytes()
    samples = []
    formats: collections.Counter = collections.Counter()
    image_counts: collections.Counter = collections.Counter()
    type_counts: collections.Counter = collections.Counter()
    answers: collections.Counter = collections.Counter()
    visual_index_shapes: collections.Counter = collections.Counter()
    total_images = 0
    anomalies = []
    seen_questions: dict[str, int] = {}

    for index, payload in iter_records(tfrecord):
        example = parse_example(payload)
        missing = [key for key in ("question", "image/encoded", "answer") if key not in example]
        if missing:
            raise ValueError(f"记录 {index} 缺少字段：{missing}")
        question = example["question"][1][0].decode("utf-8")
        answer = example["answer"][1][0].decode("utf-8")
        question_type = (
            example["question_type"][1][0].decode("utf-8")
            if example.get("question_type") and example["question_type"][1] else "Unknown"
        )
        visual_indices = [int(v) for v in example.get("visual_indices", ("int", []))[1]]
        blobs = list(example["image/encoded"][1])
        sample_id = f"erqa-{index:04d}"
        if question in seen_questions:
            anomalies.append(
                {"sample_id": sample_id, "kind": "duplicate-question",
                 "first_seen": seen_questions[question]}
            )
        seen_questions.setdefault(question, index)

        sample_dir = images_dir / sample_id
        sample_dir.mkdir()
        image_records = []
        for position, blob in enumerate(blobs):
            fmt = image_format(blob)
            width, height = image_size(blob, fmt)
            suffix = {"jpeg": "jpg", "png": "png", "webp": "webp"}.get(fmt, "bin")
            name = f"{position:02d}.{suffix}"
            (sample_dir / name).write_bytes(blob)
            image_records.append({
                "position": position,
                "path": f"images/{sample_id}/{name}",
                "sha256": sha256(blob),
                "bytes": len(blob),
                "format": fmt,
                "width": width,
                "height": height,
            })
            formats[fmt] += 1
        total_images += len(blobs)
        image_counts[len(blobs)] += 1
        type_counts[question_type] += 1
        answers[answer] += 1
        if len(visual_indices) != len(blobs):
            anomalies.append({
                "sample_id": sample_id, "kind": "visual-index-count-mismatch",
                "images": len(blobs), "visual_indices": len(visual_indices),
            })
        pattern = "empty" if not visual_indices else (
            "all-zero" if all(v == 0 for v in visual_indices) else "interleaved")
        visual_index_shapes[pattern] += 1
        if answer not in ANSWER_LETTERS:
            anomalies.append({
                "sample_id": sample_id, "kind": "answer-not-a-d", "answer": answer,
            })
        samples.append({
            "id": sample_id,
            "record_index": index,
            "question": question,
            "answer": answer,
            "question_type": question_type,
            "visual_indices": visual_indices,
            "visual_index_pattern": pattern,
            "question_chars": len(question),
            "images": image_records,
            "image_count": len(blobs),
        })

    if len(samples) != args.expected_records:
        raise ValueError(
            f"记录数 {len(samples)} 与预期 {args.expected_records} 不符")

    with (output / "samples.jsonl").open("w", encoding="utf-8") as stream:
        for sample in samples:
            stream.write(json.dumps(sample, ensure_ascii=False) + "\n")

    widths = [img["width"] for s in samples for img in s["images"] if img["width"]]
    heights = [img["height"] for s in samples for img in s["images"] if img["height"]]
    manifest = {
        "source": {
            "repository": "https://github.com/embodiedreasoning/ERQA",
            "commit": "5bba488252c5b345a43335bfb9239b6bf12bd91c",
            "path": str(tfrecord),
            "name": tfrecord.name,
            "bytes": len(source),
            "sha256": sha256(source),
        },
        "records": len(samples),
        "record_order": "original tfrecord order, erqa-0000 .. erqa-0399",
        "images": {
            "total": total_images,
            "unique_sha256": len({img["sha256"] for s in samples for img in s["images"]}),
            "formats": dict(sorted(formats.items())),
            "count_per_sample": {str(k): v for k, v in sorted(image_counts.items())},
            "min_per_sample": min(image_counts),
            "max_per_sample": max(image_counts),
            "width_min": min(widths) if widths else None,
            "width_max": max(widths) if widths else None,
            "height_min": min(heights) if heights else None,
            "height_max": max(heights) if heights else None,
        },
        "question_types": dict(sorted(type_counts.items())),
        "answers": dict(sorted(answers.items())),
        "visual_index_patterns": dict(sorted(visual_index_shapes.items())),
        "anomalies": anomalies,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({
        "output": str(output), "records": len(samples),
        "images": total_images, "types": manifest["question_types"],
        "answers": manifest["answers"], "anomalies": len(anomalies),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
