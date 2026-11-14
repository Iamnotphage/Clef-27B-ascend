#!/usr/bin/env python3
"""Measure the native Clef prompt length for every ERQA sample.

Replicates ``JointSchemaStrategy.encode`` on the CPU: the fixed system prefix,
the media placeholders expanded by the model processor with the pinned image
size bound, the JSON state, the schema block and the fixed suffix.  The result
sets the service ``max_model_len`` and ``--limit-mm-per-prompt`` budget before
the formal run.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys

from erqa_protocol import (
    DEFAULT_INSTRUCTIONS, IMAGE_PLACEHOLDER, REFERENCE_IMAGE_SIZE, SYSTEM_PROMPT,
    load_samples, parse_options,
)

SUFFIX = (
    "\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    "JOINT SCHEMA DECISIONS:"
)


def tokens(tokenizer, text):
    return tokenizer.encode(text, add_special_tokens=False)


def schema_tokens(tokenizer, sample):
    ids = tokens(tokenizer, "\n\nSCHEMA FIELDS:\n")
    ids += tokens(
        tokenizer,
        "\nFIELD 1\nID: answer\nTYPE: choice\nINSTRUCTION: ",
    )
    ids += tokens(tokenizer, DEFAULT_INSTRUCTIONS)
    ids += tokens(tokenizer, "\nALLOWED OPTIONS:\n")
    for number, (letter, text) in enumerate(parse_options(sample["question"]), start=1):
        ids += tokens(tokenizer, f"OPTION {number}: ")
        semantics = {"option_id": letter}
        if text:
            semantics["description"] = text
        ids += tokens(tokenizer, json.dumps(
            semantics, ensure_ascii=False, separators=(",", ":"), sort_keys=True))
        ids += tokens(tokenizer, "\n")
    ids += tokens(tokenizer, "END FIELD\n")
    return ids


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--model-dir", default="/workspace/yangchen/models/Clef-27B")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0,
                        help="只测前 N 条（0 为全部）")
    parser.add_argument("--image-size", default=json.dumps(REFERENCE_IMAGE_SIZE))
    args = parser.parse_args()

    from PIL import Image
    from transformers import AutoProcessor

    image_size = json.loads(args.image_size)
    samples, manifest = load_samples(args.data_dir)
    if args.limit:
        samples = samples[: args.limit]
    processor = AutoProcessor.from_pretrained(args.model_dir, local_files_only=True)
    tokenizer = processor.tokenizer
    prefix = tokens(
        tokenizer,
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\nSTATE:\n",
    )
    suffix = tokens(tokenizer, SUFFIX)

    rows = []
    for sample in samples:
        state = json.dumps(
            {"question": sample["question"]}, ensure_ascii=False,
            sort_keys=True, separators=(",", ":"))
        state_ids = tokens(tokenizer, state)
        schema_ids = schema_tokens(tokenizer, sample)
        text = IMAGE_PLACEHOLDER * sample["image_count"] + "\n"
        media_ids = 0
        if sample["image_count"]:
            images = [
                Image.open(args.data_dir / record["path"]).convert("RGB")
                for record in sample["images"]
            ]
            encoded = processor(
                text=[text], images=images, return_tensors="pt",
                images_kwargs={"size": image_size})
            media_ids = int(encoded["input_ids"].shape[1])
        total = len(prefix) + media_ids + len(state_ids) + len(schema_ids) + len(suffix)
        rows.append({
            "id": sample["id"],
            "question_type": sample["question_type"],
            "images": sample["image_count"],
            "visual_index_pattern": sample["visual_index_pattern"],
            "prefix_tokens": len(prefix),
            "media_tokens": media_ids,
            "state_tokens": len(state_ids),
            "schema_tokens": len(schema_ids),
            "suffix_tokens": len(suffix),
            "total_tokens": total,
        })

    totals = [row["total_tokens"] for row in rows]
    media = [row["media_tokens"] for row in rows if row["images"]]
    document = {
        "data_dir": str(args.data_dir),
        "tfrecord_sha256": manifest["source"]["sha256"],
        "model_dir": args.model_dir,
        "image_size": image_size,
        "samples": len(rows),
        "tokens": {
            "min": min(totals), "max": max(totals),
            "mean": statistics.mean(totals), "p50": statistics.median(totals),
            "p95": sorted(totals)[int(0.95 * (len(totals) - 1))],
        },
        "media_tokens": {
            "min": min(media), "max": max(media), "mean": statistics.mean(media),
        },
        "longest": sorted(rows, key=lambda row: row["total_tokens"], reverse=True)[:10],
        "over_16384": [row["id"] for row in rows if row["total_tokens"] > 16384],
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, ensure_ascii=False, indent=1) + "\n")
    print(json.dumps({
        "samples": len(rows), "max_tokens": max(totals), "mean_tokens": round(statistics.mean(totals), 1),
        "max_media_tokens": max(media) if media else 0, "over_16384": document["over_16384"],
        "longest": [(row["id"], row["total_tokens"], row["images"]) for row in document["longest"][:5]],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
