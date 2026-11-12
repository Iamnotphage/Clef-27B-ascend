#!/usr/bin/env python3
"""Prepare reproducible profiling requests with the pinned serving encoder.

This loads configuration/tokenizer/processor code only, and creates no model or
engine. Original dataset requests are retained for natural length examples;
controlled length probes vary prefixes of one real state under a fixed schema.
Field/candidate scans are explicitly derived performance probes, with the source
of every state, question and candidate recorded.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace


VLLM_COMMIT = "24420b7a341af37a9bc3186e6fc6062512e0aa58"


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def json_bytes(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def git(source, *arguments):
    return subprocess.run(["git", "-C", str(source), *arguments], capture_output=True, check=True, timeout=20).stdout


def read_cases(path, model):
    raw = path.read_bytes()
    document = json.loads(raw)
    rows = document.get("cases") if isinstance(document, dict) else document
    if not isinstance(rows, list) or not rows:
        raise ValueError("--cases must contain a nonempty cases array")
    selected = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or not isinstance(row.get("request"), dict):
            raise ValueError(f"Dataset row {index} needs a request object")
        request = dict(row["request"], model=model)
        if any(request.get(key) for key in ("images", "videos", "audio")):
            continue
        if not isinstance(request.get("questions"), dict) or not request["questions"] or "state" not in request:
            raise ValueError(f"Dataset row {index} needs state and questions")
        selected.append({
            "name": row.get("name", f"record_{index}"), "request": request,
            "source": row.get("source"), "source_row_index": index,
            "source_request_sha256": sha256(json_bytes(row["request"])),
        })
    if not selected:
        raise ValueError("Dataset has no pure-text requests")
    return selected, {"path": str(path.resolve()), "sha256": sha256(raw), "source_rows": len(rows), "pure_text_rows": len(selected)}


class ServingEncoder:
    def __init__(self, model_dir, vllm_src, max_length):
        head = git(vllm_src, "rev-parse", "HEAD").decode().strip()
        if head != VLLM_COMMIT:
            raise ValueError(f"vLLM HEAD must be {VLLM_COMMIT}; got {head}")
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        from vllm.config import ModelConfig
        from vllm.tokenizers import cached_tokenizer_from_config
        from vllm.entrypoints.openai.decisions import strategies
        from vllm.entrypoints.generate.structured_decisions import protocol, serving

        self.request_class = protocol.StructuredDecisionRequest
        self.parse_questions = serving.parse_questions
        self.model_dir = model_dir
        self.mm_renderer = None
        self.tokenizer = None
        self.strategies = {}
        for limit in dict.fromkeys((max_length, 262144)):
            config = ModelConfig(model=str(model_dir), tokenizer=str(model_dir), runner="pooling", dtype="bfloat16", max_model_len=limit, enforce_eager=True, limit_mm_per_prompt={"image": 0, "video": 0})
            if self.tokenizer is None:
                self.tokenizer = cached_tokenizer_from_config(config, local_files_only=True)
            context = strategies.ReadContext(
                engine_client=SimpleNamespace(model_config=config),
                online_renderer=SimpleNamespace(renderer=SimpleNamespace(get_tokenizer=lambda: self.tokenizer)),
                chat_template=None, chat_template_content_format="auto", default_chat_template_kwargs={},
            )
            strategy_class = strategies.select_read_strategy(config, ("token_classify",))
            if strategy_class is not strategies.ClefFlashStrategy:
                raise ValueError(f"Expected ClefFlashStrategy; got {strategy_class.__name__}")
            self.strategies[limit] = strategy_class(context)
        self.max_length = max_length
        source_modules = {"strategies": strategies, "protocol": protocol, "serving": serving}
        self.provenance = {
            "vllm_head": head, "vllm_source_diff_sha256": sha256(git(vllm_src, "diff", "HEAD")),
            "encoding_source_files": {name: {"path": inspect.getfile(module), "sha256": sha256(Path(inspect.getfile(module)).read_bytes())} for name, module in source_modules.items()},
            "tokenizer_class": f"{type(self.tokenizer).__module__}.{type(self.tokenizer).__name__}",
            "versions": {name: importlib.metadata.version(name) for name in ("vllm", "transformers", "torch")},
            "model_metadata_sha256": {name: sha256((model_dir / name).read_bytes()) for name in ("config.json", "joint_head_config.json", "tokenizer.json", "tokenizer_config.json", "processor_config.json", "preprocessor_config.json") if (model_dir / name).is_file()},
            "weights_loaded": False, "engine_constructed": False,
        }

    def encode(self, request_data, full=False, prefix=None):
        if prefix is None and any(request_data.get(key) for key in ("images", "videos", "audio")):
            raise ValueError("encode() accepts pure text; media expansion requires the real processor path")
        strategy = self.strategies[262144 if full else self.max_length]
        request = self.request_class.model_validate(request_data)
        questions = self.parse_questions(request, strategy.limits())
        input_ids, schema, orders = strategy.encode(questions, strategy.render_state(request.state), prefix=prefix)
        input_ids = [int(value) for value in input_ids]
        length, offset, encoded_questions = len(input_ids), 0, []
        for index, question in enumerate(questions):
            option_count = schema["option_counts"][index]
            question_span = [int(value) + length for value in schema["question_spans"][index]]
            spans_from_end = schema["option_spans"][offset:offset + option_count]
            option_spans = [[int(value) + length for value in span] for span in spans_from_end]
            if any(start < 0 or end < start or end > length for start, end in [question_span, *option_spans]):
                raise ValueError("Encoded schema span falls outside actual input IDs")
            option_ids = [option_id for _, option_id, _ in strategy._options(question)]
            encoded_questions.append({
                "id": question.id, "option_ids": option_ids,
                "question_span": question_span, "question_span_from_end": schema["question_spans"][index],
                "option_spans": option_spans, "option_spans_from_end": spans_from_end,
                "candidate_span_input_ids": [input_ids[start:end] for start, end in option_spans],
                "question_span_input_ids": input_ids[question_span[0]:question_span[1]],
            })
            offset += option_count
        if offset != len(schema["option_spans"]):
            raise ValueError("Schema option counts and flattened option spans differ")
        return {"input_ids": input_ids, "input_ids_sha256": sha256(json_bytes(input_ids)), "input_tokens": length, "joint_schema": schema, "option_output_orders": orders, "questions": encoded_questions, "max_length": strategy.max_length, "encoding_scope": "actual pinned serving strategy with real expanded media prefix" if prefix is not None else "actual pinned pure-text serving strategy"}

    def media_prefix(self, request_data):
        """Run the same decoder/renderer/processor as serving, without an engine."""
        if self.mm_renderer is None:
            from vllm.config import ModelConfig, VllmConfig
            from vllm.renderers.registry import renderer_from_config
            from vllm.entrypoints.openai.decisions import strategies
            from verify_systemone_multimodal import MM_PROCESSOR_KWARGS

            config = ModelConfig(model=str(self.model_dir), tokenizer=str(self.model_dir), runner="pooling", dtype="bfloat16", max_model_len=self.max_length, enforce_eager=True, limit_mm_per_prompt={"image": 1, "video": 1}, mm_processor_kwargs=MM_PROCESSOR_KWARGS)
            self.mm_renderer = renderer_from_config(VllmConfig(model_config=config))
            context = strategies.ReadContext(engine_client=SimpleNamespace(model_config=config), online_renderer=SimpleNamespace(renderer=self.mm_renderer), chat_template=None, chat_template_content_format="auto", default_chat_template_kwargs={})
            self.mm_strategy = strategies.ClefFlashStrategy(context)
            self.provenance["multimodal_encoder"] = {
                "renderer_class": f"{type(self.mm_renderer).__module__}.{type(self.mm_renderer).__name__}",
                "processor_kwargs": MM_PROCESSOR_KWARGS,
                "video_loader_backend": os.environ.get("VLLM_VIDEO_LOADER_BACKEND"),
                "source_path": inspect.getfile(type(self.mm_renderer)),
                "source_sha256": sha256(Path(inspect.getfile(type(self.mm_renderer))).read_bytes()),
                "execution": "Real ClefFlashStrategy._render_media followed by real BaseRenderer.process_for_engine_async; no model or engine construction",
            }
        media = {kind: request_data[key] for kind, key in (("image", "images"), ("video", "videos")) if request_data.get(key)}
        started = time.perf_counter()
        processed = asyncio.run(self.mm_strategy._render_media(media, None))
        elapsed_ms = (time.perf_counter() - started) * 1000
        placeholders, fields = {}, {}
        for kind, ranges in processed.get("mm_placeholders", {}).items():
            placeholders[kind] = [{"offset": item.offset, "length": item.length, "is_embed": item.is_embed.tolist() if item.is_embed is not None else None, "vision_tokens": int(item.is_embed.sum().item()) if item.is_embed is not None else item.length} for item in ranges]
        for kind, items in processed.get("mm_kwargs", {}).items():
            fields[kind] = []
            for item in items:
                description = {}
                for key, value in item.items():
                    data = value.data
                    if hasattr(data, "shape") and hasattr(data, "dtype"):
                        if data.device.type != "cpu":
                            raise ValueError("Workload preparation requires CPU-only multimodal processor output")
                        if data.numel() <= 64:
                            description[key] = {"shape": list(data.shape), "dtype": str(data.dtype), "values": data.tolist()}
                        else:
                            description[key] = {"shape": list(data.shape), "dtype": str(data.dtype), "sha256": sha256(data.contiguous().view(__import__("torch").uint8).numpy().tobytes())}
                    else:
                        description[key] = data
                fields[kind].append(description)
        prefix = list(processed["prompt_token_ids"])
        return prefix, {"expanded_prefix_input_ids": prefix, "expanded_prefix_tokens": len(prefix), "mm_hashes": processed.get("mm_hashes"), "mm_placeholders": placeholders, "multimodal_processor_fields": fields, "vision_tokens": sum(item["vision_tokens"] for ranges in placeholders.values() for item in ranges), "decode_and_processor_elapsed_ms": elapsed_ms, "timing_scope": "One offline preparation call through the actual media decoder and renderer; includes initialization/cache effects and is excluded from service throughput"}

    def close(self):
        if self.mm_renderer is not None:
            self.mm_renderer.shutdown()


def source_index(row):
    return {key: row.get(key) for key in ("name", "source", "source_row_index", "source_request_sha256")}


def select_natural_lengths(rows, lengths, targets, limit):
    eligible = [(index, length) for index, length in enumerate(lengths) if length <= limit]
    if not eligible:
        raise ValueError("No original request fits the service model length")
    result = []
    for target in targets:
        index, length = min(eligible, key=lambda value: (abs(value[1] - target), value[0]))
        result.append({"name": f"natural-length-{target}", "request": copy.deepcopy(rows[index]["request"]), "metadata": {"group": "natural_length_examples", "target_input_tokens": target, "selected_full_input_tokens": length, "source": source_index(rows[index]), "variable": "Original real request selection by actual full encoded length", "controlled_variables": [], "confounders": "Original state, schema, field count and candidate counts can differ; use controlled_length for a length-only comparison", "derived_probe": False}})
    return result


def prefix_for_target(request, target, encode, radius=32):
    if not isinstance(request.get("state"), str):
        raise ValueError("Controlled length probes require a string state")
    state = request["state"]
    cache = {}

    def evaluate(length):
        if length not in cache:
            candidate = dict(request, state=state[:length])
            cache[length] = encode(candidate, full=True)["input_tokens"]
        return cache[length]

    low, high = 0, len(state)
    while low < high:
        middle = (low + high) // 2
        if evaluate(middle) < target:
            low = middle + 1
        else:
            high = middle
    # BPE merges can perturb local monotonicity; choose from actual encodings.
    for length in range(max(0, low - radius), min(len(state), low + radius) + 1):
        evaluate(length)
    evaluate(0)
    evaluate(len(state))
    end = min(cache, key=lambda length: (abs(cache[length] - target), cache[length] > target, length))
    return dict(request, state=state[:end]), {"state_prefix_characters": end, "source_state_characters": len(state), "state_prefix_truncated": end < len(state), "actual_full_input_tokens": cache[end], "target_delta_tokens": cache[end] - target, "prefix_search_encoded_candidates": len(cache)}


def make_controlled_scans(rows, lengths, encode, targets, field_counts, candidate_counts):
    choices = [(row, qid, question) for row in rows for qid, question in row["request"]["questions"].items() if question.get("type") == "choice" and isinstance(question.get("criteria"), dict) and len(question["criteria"]) >= 2 and isinstance(row["request"].get("state"), str)]
    if not choices:
        raise ValueError("Controlled probes need a real text choice request with at least two options")
    anchor, qid, question = choices[0]
    base = {"model": anchor["request"]["model"], "state": anchor["request"]["state"], "questions": {"field_00": copy.deepcopy(question)}}
    state_sha = sha256(json_bytes(base["state"]))
    source = source_index(anchor)
    output = []
    longest = max((index for index, row in enumerate(rows) if isinstance(row["request"]["state"], str)), key=lambda index: lengths[index])
    length_base = dict(base, state=rows[longest]["request"]["state"])
    for target in targets:
        request, prefix = prefix_for_target(length_base, target, encode)
        output.append({"name": f"controlled-length-{target}", "request": request, "metadata": {"group": "controlled_length", "target_input_tokens": target, "variable": "state text prefix length", "controlled_variables": ["model", "question schema", "field count", "candidate descriptions and order"], "source": source, "state_source": source_index(rows[longest]), "derived_probe": True, **prefix}})
    for count in field_counts:
        request = copy.deepcopy(base)
        request["questions"] = {f"field_{index:02d}": copy.deepcopy(question) for index in range(count)}
        output.append({"name": f"fields-{count}", "request": request, "metadata": {"group": "field_count", "variable": "number of repeated real choice fields with distinct IDs", "field_count": count, "controlled_variables": ["state", "question instructions", "per-field candidate descriptions and order"], "state_sha256": state_sha, "source": source, "source_question_id": qid, "derived_probe": True}})
    bank, seen = [], set()
    for row, candidate_qid, candidate_question in choices:
        for candidate_id, description in candidate_question["criteria"].items():
            description_hash = sha256(json_bytes(description))
            if description_hash not in seen:
                seen.add(description_hash)
                bank.append({"description": description, "source": source_index(row), "source_question_id": candidate_qid, "source_candidate_id": candidate_id})
        if len(bank) >= max(candidate_counts):
            break
    if len(bank) < max(candidate_counts):
        raise ValueError(f"Real candidate bank has only {len(bank)} distinct descriptions; requested {max(candidate_counts)}")
    for count in candidate_counts:
        request = copy.deepcopy(base)
        request["questions"]["field_00"]["criteria"] = {f"option_{index:02d}": item["description"] for index, item in enumerate(bank[:count])}
        output.append({"name": f"candidates-{count}", "request": request, "metadata": {"group": "candidate_count", "variable": "ordered candidate-bank prefix size", "candidate_count": count, "controlled_variables": ["state", "field count=1", "question ID and instructions", "candidate prefix descriptions and IDs"], "state_sha256": state_sha, "source": source, "source_question_id": qid, "candidate_sources": bank[:count], "derived_probe": True, "semantic_scope": "Performance probe built from real candidate strings; original dataset labels do not apply"}})
    return output


def encode_sample(sample, encoder, prefix=None, media_evidence=None):
    request = sample["request"]
    full = encoder.encode(request, full=True, prefix=prefix)
    bounded = encoder.encode(request, full=False, prefix=prefix)
    empty_request = dict(request, state="")
    empty = encoder.encode(empty_request, full=True, prefix=prefix)
    sample["metadata"].update({
        "request_sha256": sha256(json_bytes(request)), "actual_input_tokens": bounded["input_tokens"],
        "full_input_tokens": full["input_tokens"], "service_truncated": full["input_ids"] != bounded["input_ids"],
        "fixed_prompt_and_empty_state_schema_tokens": empty["input_tokens"],
        "candidate_span_tokens": sum(len(ids) for question in bounded["questions"] for ids in question["candidate_span_input_ids"]),
        "field_count": len(request["questions"]),
        "candidate_counts": {qid: 2 if question["type"] == "noul" else len(question["criteria"]) for qid, question in request["questions"].items()},
        "encoding_complete": True,
        "encoding_scope": bounded["encoding_scope"],
    })
    if media_evidence:
        sample["metadata"]["vision_tokens"] = media_evidence["vision_tokens"]
    return {"name": sample["name"], "request_sha256": sample["metadata"]["request_sha256"], "full_encoding": full, "service_encoding": bounded, "empty_state_encoding": empty, "media_evidence": media_evidence, "metadata": sample["metadata"]}


def existing_mm_samples(manifest_path, model):
    from verify_systemone_multimodal import MM_PROCESSOR_KWARGS, planned_cases

    raw = manifest_path.read_bytes()
    manifest = json.loads(raw)
    for name, media in manifest["files"].items():
        path = Path(media["path"])
        if sha256(path.read_bytes()) != media["sha256"]:
            raise ValueError(f"Existing media fixture SHA-256 mismatch: {name}")
    samples = []
    for case in planned_cases(model, manifest, "all"):
        media = manifest["files"][case["name"]]
        samples.append({"name": "mm-" + case["name"].replace(".", "-"), "request": case["request"], "metadata": {"group": "existing_multimodal", "source_fixture": media, "fixture_manifest_path": str(manifest_path.resolve()), "fixture_manifest_sha256": sha256(raw), "processor_kwargs": MM_PROCESSOR_KWARGS, "expected_choice": case["expected_choice"], "encoding_complete": False, "actual_input_tokens": None, "encoding_scope": "Request bytes and media fixture hashes verified; full expanded IDs require the serving multimodal processor path", "variable": "the existing image/video fixture", "controlled_variables": ["per-modality resolution", "video frame count and fps", "processor kwargs"], "derived_probe": False}})
    return samples


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--dataset-manifest", type=Path)
    parser.add_argument("--model-dir", type=Path, default=Path("/workspace/yangchen/models/Clef-27B"))
    parser.add_argument("--vllm-src", type=Path, default=Path("/workspace/yangchen/work/src/vllm"))
    parser.add_argument("--model", default="clef")
    parser.add_argument("--fixtures-manifest", type=Path)
    parser.add_argument("--video-loader-backend", default="opencv")
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--targets", nargs="+", type=int, default=[256, 1024, 2048, 4032])
    parser.add_argument("--field-counts", nargs="+", type=int, default=[1, 2, 4, 8])
    parser.add_argument("--candidate-counts", nargs="+", type=int, default=[2, 4, 8, 16])
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if min(args.max_model_len, *args.targets, *args.field_counts, *args.candidate_counts) <= 0 or max(args.targets) > args.max_model_len:
        parser.error("All counts must be positive and targets must fit --max-model-len")
    started = time.perf_counter()
    os.environ["VLLM_VIDEO_LOADER_BACKEND"] = args.video_loader_backend
    report = {"schema_version": 1, "created_at": datetime.now(timezone.utc).isoformat(), "status": "preparing", "passed": False, "weights_loaded": False, "engine_constructed": False, "max_model_len": args.max_model_len, "script_sha256": sha256(Path(__file__).read_bytes()), "performance_environment": {key: os.environ.get(key) for key in ("OMP_NUM_THREADS", "TOKENIZERS_PARALLELISM", "TASK_QUEUE_ENABLE", "ASCEND_RT_VISIBLE_DEVICES")}, "samples": [], "errors": []}
    try:
        rows, provenance = read_cases(args.cases, args.model)
        report["dataset"] = provenance
        manifest_path = args.dataset_manifest or args.cases.with_name("manifest.json")
        if manifest_path.is_file():
            report["dataset_manifest"] = {"path": str(manifest_path.resolve()), "sha256": sha256(manifest_path.read_bytes()), "content": json.loads(manifest_path.read_bytes())}
        encoder = ServingEncoder(args.model_dir.resolve(), args.vllm_src.resolve(), args.max_model_len)
        report["encoder"] = encoder.provenance
        lengths = []
        report["dataset_length_scan"] = []
        for row in rows:
            length = encoder.encode(row["request"], full=True)["input_tokens"]
            lengths.append(length)
            report["dataset_length_scan"].append({**source_index(row), "full_input_tokens": length})
        samples = select_natural_lengths(rows, lengths, args.targets, args.max_model_len)
        samples += make_controlled_scans(rows, lengths, encoder.encode, args.targets, args.field_counts, args.candidate_counts)
        evidence = []
        for sample in samples:
            encoded = encode_sample(sample, encoder)
            evidence.append(encoded)
            if sample["metadata"]["service_truncated"]:
                report["errors"].append({"sample": sample["name"], "error": "Selected request is truncated by the configured serving encoder"})
        if args.fixtures_manifest:
            mm_samples = existing_mm_samples(args.fixtures_manifest, args.model)
            for sample in mm_samples:
                prefix, media_evidence = encoder.media_prefix(sample["request"])
                evidence.append(encode_sample(sample, encoder, prefix=prefix, media_evidence=media_evidence))
                if sample["metadata"]["service_truncated"]:
                    report["errors"].append({"sample": sample["name"], "error": "Multimodal request is truncated by the configured serving encoder"})
            samples += mm_samples
        encoder.close()
        args.output_dir.mkdir(parents=True, exist_ok=True)
        for sample in samples:
            path = args.output_dir / (sample["name"] + ".json")
            save(path, {"samples": [sample]})
            report["samples"].append({"name": sample["name"], "sample_file": str(path.resolve()), "sample_file_sha256": sha256(path.read_bytes()), "request_sha256": sha256(json_bytes(sample["request"])), "metadata": sample["metadata"]})
        encoding_path = args.output_dir / "encoding-evidence.json"
        save(encoding_path, {"weights_loaded": False, "encoder": encoder.provenance, "cases": evidence})
        report["encoding_evidence"] = {"path": str(encoding_path.resolve()), "sha256": sha256(encoding_path.read_bytes())}
        matrix_plan = {
            "schema_version": 1, "scope": "Sequential per-shape unprofiled service measurements; each sample independently prewarmed",
            "recommended_measurement": {"warmup": 10, "requests": 30, "concurrency": 1, "sample_hbm": True},
            "samples": [{
                "name": sample["name"], "group": sample["metadata"]["group"],
                "sample_file": sample["name"] + ".json", "request_sha256": sha256(json_bytes(sample["request"])),
                "variable": sample["metadata"]["variable"], "controlled_variables": sample["metadata"]["controlled_variables"],
                "actual_input_tokens": sample["metadata"]["actual_input_tokens"], "vision_tokens": sample["metadata"].get("vision_tokens", 0),
                "minimum_required_model_len": sample["metadata"]["actual_input_tokens"],
                "fits_model_len_1024": sample["metadata"]["actual_input_tokens"] <= 1024,
                "fits_model_len_4096": sample["metadata"]["actual_input_tokens"] <= 4096,
            } for sample in samples],
            "concurrency_and_batching_scan": {"request": "Use benchmark_systemone.py default historical mixed 307-token request", "client_concurrency": [1, 2, 4], "max_num_seqs": [1, 2, 4], "order": "First vary client concurrency with max_num_seqs=1; then vary max_num_seqs, while recording actual batch and memory"},
        }
        matrix_path = args.output_dir / "matrix-plan.json"
        save(matrix_path, matrix_plan)
        report["matrix_plan"] = {"path": str(matrix_path.resolve()), "sha256": sha256(matrix_path.read_bytes())}
        report["passed"] = not report["errors"]
        report["status"] = "complete" if report["passed"] else "failed"
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
    report["finished_at"] = datetime.now(timezone.utc).isoformat()
    usage = resource.getrusage(resource.RUSAGE_SELF)
    report["resource_usage"] = {"elapsed_seconds": time.perf_counter() - started, "user_cpu_seconds": usage.ru_utime, "system_cpu_seconds": usage.ru_stime, "max_rss_bytes": usage.ru_maxrss if sys.platform == "darwin" else usage.ru_maxrss * 1024}
    save(args.output_dir / "manifest.json", report)
    print(json.dumps({"passed": report["passed"], "status": report["status"], "manifest": str((args.output_dir / "manifest.json").resolve()), "sample_count": len(report["samples"]), "error": report.get("error"), "errors": report["errors"]}, ensure_ascii=False), flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
