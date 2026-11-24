#!/usr/bin/env python3
"""ERQA protocol helpers shared by the runner, budget check and scorer.

The official ERQA TFRecord is extracted by ``erqa_extract_tfrecord.py`` into
``samples.jsonl`` plus ``images/``.  This module parses the multiple-choice
options out of the question text, orders the images the way the official
``eval_harness.py`` does, builds the native Clef ``/v1/decide`` request, and
applies the official exact-match answer rule.
"""

from __future__ import annotations

import base64
import hashlib
import json
import re
from pathlib import Path

OFFICIAL_REPOSITORY = "https://github.com/embodiedreasoning/ERQA"
OFFICIAL_COMMIT = "5bba488252c5b345a43335bfb9239b6bf12bd91c"
OFFICIAL_TFRECORD_SHA256 = "1974f5e587eac144cc3ba325477477b8ad6a56c22ce4e419c34229520b4c93e2"
OFFICIAL_RECORDS = 400
MODEL_NAME = "clef"
DEFAULT_INSTRUCTIONS = "Which option correctly answers `question`?"
SYSTEM_PROMPT = (
    "Read the complete state and schema. Decide every field jointly. Each "
    "answer must be exactly one of that field's allowed options."
)
IMAGE_PLACEHOLDER = "<|vision_start|><|image_pad|><|vision_end|>"
#: ``JointSchemaStrategy.IMAGE_SIZE``: the reference service's pixel bound.
REFERENCE_IMAGE_SIZE = {"shortest_edge": 65536, "longest_edge": 1048576}
ANSWER_INSTRUCTION = re.compile(r"Please answer directly", re.IGNORECASE)
WIRE_FORMATS = ("top-level-images", "state-list")
DEFAULT_WIRE_FORMAT = "top-level-images"


def parse_options(question: str) -> list[tuple[str, str]]:
    """Return ``[(letter, text), ...]`` for the question's embedded options.

    Handles the formats present in the official snapshot: ``A.`` / ``A:``
    / ``A)``, two to four options, and empty option texts (kept as ``""`` so
    the candidate count and mapping stay identical to the source).
    """
    heading = re.search(r"Choices:", question, re.IGNORECASE)
    body = question[heading.end():] if heading else question
    body = ANSWER_INSTRUCTION.split(body, 1)[0].rstrip()
    markers = []
    position = 0
    for expected in range(4):
        letter = chr(ord("A") + expected)
        match = re.search(rf"(?<![A-Za-z0-9]){letter}\s*[.):]\s*", body[position:])
        if match is None:
            break
        markers.append((letter, position + match.start(), position + match.end()))
        position = position + match.end()
    if len(markers) < 2:
        raise ValueError(f"no multiple-choice options found: {question[:120]!r}")
    letters = [letter for letter, _, _ in markers]
    if letters != [chr(ord("A") + i) for i in range(len(markers))]:
        raise ValueError(f"non-contiguous option labels {letters}: {question[:120]!r}")
    options = []
    for index, (letter, _, text_start) in enumerate(markers):
        end = markers[index + 1][1] if index + 1 < len(markers) else len(body)
        options.append((letter, body[text_start:end].strip()))
    return options


def image_order(sample: dict) -> list[int]:
    """Official harness ordering: stable sort by ``visual_indices``.

    Empty indices place every image before the question in the original order;
    all-zero indices do the same.  The stable sort keeps that behavior.
    """
    indices = sample.get("visual_indices") or []
    if not indices:
        return list(range(sample["image_count"]))
    return sorted(range(sample["image_count"]), key=lambda i: indices[i])


def data_url(blob: bytes, fmt: str) -> str:
    media = {"jpeg": "image/jpeg", "png": "image/png", "webp": "image/webp"}.get(
        fmt, "application/octet-stream")
    return f"data:{media};base64,{base64.b64encode(blob).decode('ascii')}"


def content_segments(sample: dict, images: list[str]) -> list[tuple[str, str]]:
    """Official ``eval_harness.py`` placement as ``[(kind, value), ...]``.

    ``images`` must already be in ``image_order``.  Empty or all-zero
    ``visual_indices`` put every image before the question; otherwise the
    question is sliced by character index, exactly like the official loop
    (each image keeps its own source index, and the stable sort matches
    ``image_order``).
    """
    question = sample["question"]
    indices = sample.get("visual_indices") or []
    if not indices:
        return [*[("image", url) for url in images], ("text", question)]
    order = image_order(sample)
    pairs = [(images[position], indices[image_index])
             for position, image_index in enumerate(order)]
    if all(index == 0 for _, index in pairs):
        return [*[("image", url) for url, _ in pairs], ("text", question)]
    segments: list[tuple[str, str]] = []
    last = 0
    for url, index in pairs:
        if index == 0:
            segments.append(("image", url))
        elif index <= len(question):
            if index > last:
                segments.append(("text", question[last:index]))
            segments.append(("image", url))
            last = index
        else:
            segments.append(("image", url))
    if last < len(question):
        segments.append(("text", question[last:]))
    if not segments:
        segments = [("text", question), *[("image", url) for url, _ in pairs]]
    return segments


def state_content_parts(segments: list[tuple[str, str]]) -> list[dict]:
    """Build the 2026-11-24 ``state`` content-part list."""
    parts = []
    for kind, value in segments:
        if kind == "image":
            parts.append({"type": "image_url", "image_url": {"url": value}})
        else:
            parts.append({"type": "text", "text": value})
    return parts


def state_text(state) -> str:
    """Join a content-part list's text parts; ``str`` passes through."""
    if isinstance(state, str):
        return state
    return "".join(
        part["text"] for part in state
        if isinstance(part, dict) and part.get("type") == "text"
    )


def build_request(sample: dict, data_dir: Path, model: str = MODEL_NAME,
                  instructions: str = DEFAULT_INSTRUCTIONS,
                  wire_format: str = DEFAULT_WIRE_FORMAT) -> dict:
    """Build the complete Clef request, images in official order.

    ``top-level-images`` keeps the historical ``/v1/decide`` body byte for
    byte.  ``state-list`` sends the same images as ``state`` content parts in
    the official interleaved order and omits the top-level ``images`` field.
    """
    if wire_format not in WIRE_FORMATS:
        raise ValueError(f"unknown wire format: {wire_format!r}")
    options = parse_options(sample["question"])
    criteria = {letter: text for letter, text in options}
    order = image_order(sample)
    images = []
    for index in order:
        record = sample["images"][index]
        blob = (data_dir / record["path"]).read_bytes()
        if hashlib.sha256(blob).hexdigest() != record["sha256"]:
            raise ValueError(f"image hash mismatch: {record['path']}")
        images.append(data_url(blob, record["format"]))
    questions = {
        "answer": {
            "type": "choice",
            "instructions": instructions,
            "criteria": criteria,
        }
    }
    if wire_format == "state-list":
        return {
            "model": model,
            "state": state_content_parts(content_segments(sample, images)),
            "questions": questions,
        }
    # Keep the historical key order so the default body stays byte-identical.
    return {
        "model": model,
        "state": {"question": sample["question"]},
        "questions": questions,
        "images": images,
    }


def request_summary(request: dict, sample: dict) -> dict:
    """Lightweight request identity without the base64 payload."""
    criteria = request["questions"]["answer"]["criteria"]
    summary = {
        "model": request["model"],
        "state_sha256": hashlib.sha256(json.dumps(
            request["state"], ensure_ascii=False, sort_keys=True,
            separators=(",", ":")).encode()).hexdigest(),
        "criteria": criteria,
        "option_count": len(criteria),
        "image_order": image_order(sample),
        "image_sha256": [sample["images"][i]["sha256"] for i in image_order(sample)],
    }
    if isinstance(request["state"], list):
        summary["wire_format"] = "state-list"
    return summary


def wire_body(request: dict) -> bytes:
    return json.dumps(
        request, ensure_ascii=False, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def structural_sha256(request: dict, sample: dict) -> str:
    """Hash of the request with image bytes replaced by their SHA-256.

    Reconstructs the same value from the saved sample manifest, so the hash is
    verifiable without shipping the base64 payload in the repository.
    """
    summary = request_summary(request, sample)
    summary["images"] = summary.pop("image_sha256")
    return hashlib.sha256(json.dumps(
        summary, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest()


def score(prediction: str | None, gold: str) -> bool:
    """Official ERQA rule: exact match after removing periods and case."""
    if prediction is None:
        return False
    return prediction.replace(".", "").strip().lower() == gold.strip().lower()


def load_samples(data_dir: Path) -> tuple[list[dict], dict]:
    data_dir = data_dir.expanduser().resolve()
    manifest = json.loads((data_dir / "manifest.json").read_text())
    source = manifest["source"]
    if source["sha256"] != OFFICIAL_TFRECORD_SHA256:
        raise ValueError(f"unexpected tfrecord SHA256: {source['sha256']}")
    samples = [
        json.loads(line) for line in
        (data_dir / "samples.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(samples) != OFFICIAL_RECORDS:
        raise ValueError(f"expected {OFFICIAL_RECORDS} samples, got {len(samples)}")
    return samples, manifest
