#!/usr/bin/env python3
"""Stage 0.1 audit: prefix split and reuse of the frozen evaluation sets.

The Tier 2 hybrid prefix-state cache splits every prompt at the fixed
``"\\n\\nSCHEMA FIELDS:\\n"`` delimiter and caches the aligned prefix
``[0 : L_aligned]``.  This script replays the pinned strategy's tokenization
offline (same fixed prefix/suffix, same per-piece schema encoding as
``JointSchemaStrategy.encode``) and reports, for OpenDecisions (2,352) and
ERQA (400):

- ``L_prefix`` / ``L_aligned`` / ``L_suffix`` / total token distributions;
- the share of samples whose aligned prefix is reusable (``L_aligned >= 128``);
- unique full prompts and unique state-prefix keys, including "same STATE,
  different SCHEMA" groups;
- ERQA unique-image count and the visual-token total of the unique working
  set versus the configured encoder-cache capacity;
- the joint head width, so the per-token ``prefix_memory`` size is exact.

The script also diffs every reconstructed token count against the frozen
service reports (``usage.input_tokens``) when they are supplied, which is the
strongest available proof that the offline split matches the real service.

Usage (remote A3, pinned venv)::

    source /workspace/yangchen/venvs/clef-mix-ws-py312/bin/activate
    python runtime/tiered_prefix_cache/audit_dataset_prefixes.py \
        --output docs/evidence/tiered-cache-20261124/stage0-dataset-audit.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
import statistics
import sys
import time

OD_FILE = "open-decisions2352.jsonl"
ERQA_FILE = "samples.jsonl"
SYSTEM_PROMPT = (
    "Read the complete state and schema. Decide every field jointly. Each "
    "answer must be exactly one of that field's allowed options."
)
SUFFIX = (
    "\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    "JOINT SCHEMA DECISIONS:"
)
DELIMITER = "\n\nSCHEMA FIELDS:\n"
IMAGE_PLACEHOLDER = "<|vision_start|><|image_pad|><|vision_end|>"
ERQA_INSTRUCTIONS = "Which option correctly answers `question`?"
NOUL_CRITERIA = {
    "true": "The proposition is true or the answer is yes.",
    "false": "The proposition is false or the answer is no.",
}
REFERENCE_IMAGE_SIZE = {"shortest_edge": 65536, "longest_edge": 1048576}
ANSWER_INSTRUCTION = re.compile(r"Please answer directly", re.IGNORECASE)
ALIGN = 128
MIN_PREFIX = 128


def percentile(values, fraction):
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * fraction
    lower, upper = math.floor(index), math.ceil(index)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)


def distribution(values):
    values = list(values)
    return {
        "count": len(values),
        "min": min(values) if values else None,
        "p50": percentile(values, 0.50),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "max": max(values) if values else None,
        "mean": statistics.fmean(values) if values else None,
    }


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_ids(ids) -> str:
    digest = hashlib.sha256()
    for token in ids:
        digest.update(int(token).to_bytes(4, "little", signed=True))
    return digest.hexdigest()


def parse_options(question: str) -> list[tuple[str, str]]:
    """The ERQA option parser from ``scripts/official_eval/erqa_protocol.py``.

    Kept in sync by a unit test; the audit must not import the evaluation
    script because the remote tree only ships ``runtime/``.
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
    indices = sample.get("visual_indices") or []
    if not indices:
        return list(range(sample["image_count"]))
    return sorted(range(sample["image_count"]), key=lambda i: indices[i])


class Encoder:
    """Tokenization and schema encoding of the pinned ClefFlashStrategy."""

    def __init__(self, model_dir: str):
        from transformers import AutoProcessor  # noqa: PLC0415 - remote venv

        self.processor = AutoProcessor.from_pretrained(
            model_dir, local_files_only=True)
        self.tokenizer = self.processor.tokenizer
        self.fixed_prefix = self.tokens(
            f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
            "<|im_start|>user\nSTATE:\n")
        self.suffix = self.tokens(SUFFIX)
        self.delimiter = self.tokens(DELIMITER)
        self.vision_const = len(self.tokens("<|vision_start|><|vision_end|>"))
        self.newline_tokens = len(self.tokens("\n"))

    def tokens(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    @staticmethod
    def render_state(value) -> str:
        if isinstance(value, str):
            return value
        return json.dumps(
            value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)

    def schema(self, questions: dict) -> list[int]:
        ids: list[int] = list(self.delimiter)
        for number, (field_id, question) in enumerate(questions.items(), start=1):
            ids += self.tokens(
                f"\nFIELD {number}\nID: {field_id}\n"
                f"TYPE: {question['type']}\nINSTRUCTION: ")
            ids += self.tokens(question.get("instructions") or field_id)
            ids += self.tokens("\nALLOWED OPTIONS:\n")
            if question["type"] == "choice":
                options = sorted(question["criteria"].items(), key=lambda kv: kv[0])
                for slot, (name, description) in enumerate(options, start=1):
                    ids += self.tokens(f"OPTION {slot}: ")
                    semantics = {"option_id": name}
                    if description is not None:
                        semantics["description"] = description
                    ids += self.tokens(json.dumps(
                        semantics, ensure_ascii=False,
                        separators=(",", ":"), sort_keys=True))
                    ids += self.tokens("\n")
            elif question["type"] == "noul":
                # Mirrors _options(): noul true then false, with the strategy's
                # default criteria when the request omits a side.
                criteria = question.get("criteria") or {}
                for slot, key in enumerate(("true", "false"), start=1):
                    description = criteria.get(key)
                    if description is None:
                        description = NOUL_CRITERIA[key]
                    ids += self.tokens(f"OPTION {slot}: ")
                    semantics = {"option_id": key, "description": description}
                    ids += self.tokens(json.dumps(
                        semantics, ensure_ascii=False,
                        separators=(",", ":"), sort_keys=True))
                    ids += self.tokens("\n")
            else:  # score: an ordered list; labels are the level indices
                for slot, level in enumerate(question["criteria"], start=1):
                    ids += self.tokens(f"OPTION {slot}: ")
                    semantics = {"option_id": str(slot - 1), "description": str(level)}
                    ids += self.tokens(json.dumps(
                        semantics, ensure_ascii=False,
                        separators=(",", ":"), sort_keys=True))
                    ids += self.tokens("\n")
            ids += self.tokens("END FIELD\n")
        return ids

    def media_tokens(self, images: list) -> int:
        """The processor's expanded length for the image placeholder block."""
        if not images:
            return 0
        text = [IMAGE_PLACEHOLDER * len(images) + "\n"]
        encoded = self.processor(
            text=text, images=images, return_tensors="pt",
            images_kwargs={"size": REFERENCE_IMAGE_SIZE})
        return int(encoded["input_ids"].shape[1])

    def single_image_tokens(self, image) -> int:
        """The merged visual tokens of one image, excluding text tokens."""
        encoded = self.processor(
            text=[IMAGE_PLACEHOLDER + "\n"], images=[image], return_tensors="pt",
            images_kwargs={"size": REFERENCE_IMAGE_SIZE})
        expanded = int(encoded["input_ids"].shape[1])
        return expanded - self.vision_const - self.newline_tokens


def split(prompt_len: int, state_len: int, prefix_len: int,
          suffix_len: int) -> dict:
    l_prefix = prefix_len + state_len
    l_aligned = (l_prefix // ALIGN) * ALIGN
    return {
        "l_prefix": l_prefix,
        "l_aligned": l_aligned,
        "l_suffix": prompt_len - l_aligned,
        "total": prompt_len,
    }


def summary_for(rows: list[dict]) -> dict:
    l_prefix = [row["l_prefix"] for row in rows]
    l_aligned = [row["l_aligned"] for row in rows]
    l_suffix = [row["l_suffix"] for row in rows]
    totals = [row["total"] for row in rows]
    return {
        "samples": len(rows),
        "L_prefix": distribution(l_prefix),
        "L_aligned_128": distribution(l_aligned),
        "L_suffix": distribution(l_suffix),
        "total_tokens": distribution(totals),
        "L_aligned_ge_128": sum(1 for value in l_aligned if value >= MIN_PREFIX),
        "L_aligned_ge_128_ratio": (
            sum(1 for value in l_aligned if value >= MIN_PREFIX) / len(rows)
            if rows else None),
        "L_aligned_eq_0": sum(1 for value in l_aligned if value == 0),
    }


def reuse_stats(keys: list[str]) -> dict:
    counts: dict[str, int] = {}
    for key in keys:
        counts[key] = counts.get(key, 0) + 1
    reused_groups = [count for count in counts.values() if count > 1]
    return {
        "unique_keys": len(counts),
        "reused_groups": len(reused_groups),
        "reused_samples": sum(reused_groups),
        "second_plus_requests": sum(count - 1 for count in counts.values()),
        "max_group": max(counts.values()) if counts else 0,
        "group_sizes": {
            str(size): sum(1 for count in counts.values() if count == size)
            for size in sorted(set(counts.values()))
        },
    }


def od_rows(encoder: Encoder, data_dir: Path, report: Path | None) -> tuple:
    source = data_dir / OD_FILE
    samples = [
        json.loads(line) for line in
        source.read_text(encoding="utf-8").split("\n") if line.strip()]
    rows = []
    prompt_hashes = {}
    state_keys: dict[str, list[int]] = {}
    for index, sample in enumerate(samples):
        questions = {"answer": sample["question"]}
        state = encoder.render_state(sample["state"])
        state_ids = encoder.tokens(state)
        schema_ids = encoder.schema(questions)
        prompt = encoder.fixed_prefix + state_ids + schema_ids + encoder.suffix
        split_info = split(
            len(prompt), len(state_ids), len(encoder.fixed_prefix),
            len(encoder.suffix))
        prefix_hash = sha256_ids(prompt[:split_info["l_aligned"]])
        prompt_hash = sha256_text(state + "\x1f" + json.dumps(
            questions, ensure_ascii=False, sort_keys=True))
        prompt_hashes.setdefault(prompt_hash, []).append(index)
        state_keys.setdefault(state, []).append(index)
        rows.append({
            "name": f"od-{index:04d}",
            "id": sample.get("id"),
            "state_sha256": sha256_text(state),
            "prompt_sha256": prompt_hash,
            "prefix_sha256": prefix_hash,
            "state_tokens": len(state_ids),
            "schema_tokens": len(schema_ids),
            "suffix_tokens": len(encoder.suffix),
            **split_info,
        })
    same_state_multi_schema = [
        indices for group, indices in state_keys.items()
        if len(group) and len({
            json.dumps(samples[i]["question"], ensure_ascii=False, sort_keys=True)
            for i in indices}) > 1
    ]
    tokens_match = validate_tokens(
        rows, report, "od") if report else None
    eligible = [row for row in rows if row["l_aligned"] >= MIN_PREFIX]
    return rows, {
        "data_file": str(source),
        "data_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "unique_state_prefixes": len(state_keys),
        "unique_prompts": len(prompt_hashes),
        "state_reuse": reuse_stats([row["state_sha256"] for row in rows]),
        "tier2_eligible_samples": len(eligible),
        "tier2_eligible_ratio": len(eligible) / len(rows) if rows else None,
        "prefix_reuse": reuse_stats([row["prefix_sha256"] for row in eligible]),
        "same_state_multi_schema_groups": len(same_state_multi_schema),
        "same_state_multi_schema_samples": sum(
            len(indices) for indices in same_state_multi_schema),
        "max_schema_variants_per_state": max(
            (len({
                json.dumps(samples[i]["question"], ensure_ascii=False, sort_keys=True)
                for i in indices}) for indices in state_keys.values()),
            default=0),
        "tokens_match_service": tokens_match,
    }, rows


def erqa_rows(encoder: Encoder, data_dir: Path,
              report: Path | None) -> tuple:
    source = data_dir / ERQA_FILE
    samples = [
        json.loads(line) for line in
        source.read_text(encoding="utf-8").split("\n") if line.strip()]
    from PIL import Image  # noqa: PLC0415 - remote venv

    rows = []
    state_keys: dict[str, list[int]] = {}
    unique_images: dict[str, dict] = {}
    for index, sample in enumerate(samples):
        state = encoder.render_state({"question": sample["question"]})
        state_ids = encoder.tokens(state)
        options = parse_options(sample["question"])
        questions = {"answer": {
            "type": "choice",
            "instructions": ERQA_INSTRUCTIONS,
            "criteria": {letter: text for letter, text in options},
        }}
        schema_ids = encoder.schema(questions)
        ordered = [sample["images"][i] for i in image_order(sample)]
        images = []
        for record in ordered:
            image = Image.open(data_dir / record["path"]).convert("RGB")
            images.append(image)
            unique_images.setdefault(record["sha256"], {
                "path": record["path"],
                "format": record["format"],
            })
        media_tokens = encoder.media_tokens(images)
        prompt_len = (
            len(encoder.fixed_prefix) + media_tokens + len(state_ids)
            + len(schema_ids) + len(encoder.suffix))
        split_info = split(
            prompt_len, len(state_ids) + media_tokens,
            len(encoder.fixed_prefix), len(encoder.suffix))
        state_key = sha256_text(state + "\x1f" + json.dumps(
            [record["sha256"] for record in ordered]))
        state_keys.setdefault(state_key, []).append(index)
        rows.append({
            "name": sample["id"],
            "question_type": sample.get("question_type"),
            "image_count": sample["image_count"],
            "media_tokens": media_tokens,
            "state_tokens": len(state_ids),
            "schema_tokens": len(schema_ids),
            "suffix_tokens": len(encoder.suffix),
            "state_key": state_key,
            "prefix_sha256": sha256_text(
                state_key + f":{split_info['l_aligned']}"),
            **split_info,
        })
    per_image_tokens = {}
    for sha, record in unique_images.items():
        image = Image.open(data_dir / record["path"]).convert("RGB")
        per_image_tokens[sha] = encoder.single_image_tokens(image)
    unique_total = sum(per_image_tokens.values())
    total_visual = sum(
        per_image_tokens[record["sha256"]]
        for sample in samples for record in
        [sample["images"][i] for i in image_order(sample)])
    # Cross-check the per-image sum against the directly processed length.
    residual = [abs(
        row["media_tokens"]
        - (sum(per_image_tokens[record["sha256"]]
               for record in [samples[i]["images"][k]
                              for k in image_order(samples[i])])
           + 2 * row["image_count"] + encoder.newline_tokens))
        for i, row in enumerate(rows)]
    tokens_match = validate_tokens(rows, report, "erqa") if report else None
    return rows, {
        "data_dir": str(data_dir),
        "samples": len(samples),
        "unique_images": len(unique_images),
        "image_occurrences": sum(sample["image_count"] for sample in samples),
        "unique_image_visual_tokens": unique_total,
        "all_image_occurrence_visual_tokens": total_visual,
        "per_image_visual_tokens": distribution(per_image_tokens.values()),
        "media_token_residual_max": max(residual) if residual else 0,
        "encoder_cache_capacity_tokens": 262144,
        "unique_fits_262144": unique_total <= 262144,
        "state_reuse": reuse_stats([row["state_key"] for row in rows]),
        "tier2_eligible_samples": sum(
            1 for row in rows if row["l_aligned"] >= MIN_PREFIX),
        "prefix_reuse": reuse_stats([
            row["prefix_sha256"] for row in rows
            if row["l_aligned"] >= MIN_PREFIX]),
        "tokens_match_service": tokens_match,
    }, rows


def load_service_tokens(report: Path, kind: str) -> list[int | None]:
    if report.suffix == ".jsonl":
        tokens = []
        for line in report.read_text().split("\n"):
            if not line.strip():
                continue
            record = json.loads(line)
            tokens.append(record.get("input_tokens"))
        return tokens
    document = json.loads(report.read_text())
    requests = document.get("requests", [])
    return [item.get("input_tokens") for item in requests]


def validate_tokens(rows: list[dict], report: Path, kind: str) -> dict:
    service = load_service_tokens(report, kind)
    mismatches = []
    compared = 0
    for row, actual in zip(rows, service):
        if actual is None:
            continue
        compared += 1
        if row["total"] != actual:
            mismatches.append({
                "name": row["name"],
                "offline": row["total"],
                "service": actual,
                "delta": row["total"] - actual,
            })
    return {
        "report": str(report),
        "compared": compared,
        "mismatches": len(mismatches),
        "examples": mismatches[:10],
        "exact": not mismatches,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--od-data-dir", type=Path,
        default=Path("/workspace/yangchen/datasets/clef-fixed-20261123/"
                     "opendecisions-1.0.1"))
    parser.add_argument(
        "--erqa-data-dir", type=Path,
        default=Path("/workspace/yangchen/work/aa3e-conv-qkv-formal-20261123."
                     "Q3ozHxgd/erqa-fixed"),
        help="Directory with samples.jsonl and images/ extracted from the "
             "official tfrecord.")
    parser.add_argument("--model-dir",
                        default="/workspace/yangchen/models/Clef-27B")
    parser.add_argument(
        "--od-tokens-report", type=Path, default=None,
        help="Frozen service report with RAW input_tokens (OD).")
    parser.add_argument(
        "--erqa-tokens-report", type=Path, default=None,
        help="Frozen service report or predictions.jsonl (ERQA).")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    od_dir = args.od_data_dir.expanduser().resolve()
    erqa_dir = args.erqa_data_dir.expanduser().resolve()
    for path in (od_dir / OD_FILE, erqa_dir / ERQA_FILE):
        if not path.exists():
            raise SystemExit(f"missing fixed data: {path}")

    encoder = Encoder(args.model_dir)
    od_samples, od_summary, od_rows_ = od_rows(
        encoder, od_dir, args.od_tokens_report)
    erqa_samples, erqa_summary, erqa_rows_ = erqa_rows(
        encoder, erqa_dir, args.erqa_tokens_report)

    head_config = json.loads(
        (Path(args.model_dir) / "joint_head_config.json").read_text())
    width = int(head_config["width"])
    prefix_bytes = {
        "kv16_tp2_total": 64 * 1024,
        "prefix_memory": 2 * width,
        "gdn48_bf16_total": 74.8 * 1024 * 1024,
        "gdn48_fp32_total": 146.8 * 1024 * 1024,
    }

    document = {
        "schema": "tiered-prefix-cache-dataset-audit/1",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model_dir": str(Path(args.model_dir).resolve()),
        "alignment": ALIGN,
        "min_prefix_tokens": MIN_PREFIX,
        "joint_head_config": head_config,
        "prefix_memory_bytes_per_token": 2 * width,
        "kv_bytes_per_token_tp2": 64 * 1024,
        "opendecisions": {
            "split": summary_for(od_samples),
            "prefix_bytes": prefix_bytes,
            **od_summary,
        },
        "erqa": {
            "split": summary_for(erqa_samples),
            **erqa_summary,
        },
        "rows": {
            "opendecisions": od_rows_,
            "erqa": erqa_rows_,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp")
    temporary.write_text(
        json.dumps(document, ensure_ascii=False, allow_nan=False, indent=1) + "\n",
        encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps({
        "output": str(args.output),
        "od_split": document["opendecisions"]["split"],
        "od_reuse": document["opendecisions"]["prefix_reuse"],
        "erqa_split": document["erqa"]["split"],
        "erqa_unique_images": document["erqa"]["unique_images"],
        "erqa_unique_visual_tokens": document["erqa"]["unique_image_visual_tokens"],
        "od_tokens_match": document["opendecisions"].get("tokens_match_service"),
        "erqa_tokens_match": document["erqa"].get("tokens_match_service"),
    }, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
