#!/usr/bin/env python3
"""Compare the real pinned vLLM encoding path with saved CPU reference encoding.

This constructs ModelConfig and the vLLM tokenizer, parses each request through
the structured-decisions protocol, and invokes the selected ClefFlashStrategy.
Context adapters supply configuration and tokenizer access; no engine or model
weights are created. Only pure-text requests are supported by this check.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import inspect
from itertools import zip_longest
import json
import os
from pathlib import Path
import subprocess
import traceback
from types import SimpleNamespace


VLLM_COMMIT = "24420b7a341af37a9bc3186e6fc6062512e0aa58"
REFERENCE_SOURCE_SHA256 = "0e304cf7c6500e8bb59bef7e2afd2c6373f82596dfb3b57d1aa93c175e2dc3a3"


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def canonical_digest(value):
    return sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode())


def git(source, *arguments):
    return subprocess.run(
        ["git", "-C", str(source), *arguments], check=True,
        capture_output=True, timeout=20,
    ).stdout


def save(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def differences(reference, actual, path=""):
    """Return every differing JSON field, preserving token/span indices."""
    if isinstance(reference, dict) and isinstance(actual, dict):
        result = []
        for key in sorted(reference.keys() | actual.keys()):
            item_path = path + "/" + str(key)
            if key not in reference or key not in actual:
                result.append({"path": item_path, "reference": reference.get(key), "vllm": actual.get(key), "reason": "missing field"})
            else:
                result.extend(differences(reference[key], actual[key], item_path))
        return result
    if isinstance(reference, list) and isinstance(actual, list):
        result = []
        missing = object()
        for index, (left, right) in enumerate(zip_longest(reference, actual, fillvalue=missing)):
            item_path = path + "/" + str(index)
            if left is missing or right is missing:
                result.append({"path": item_path, "reference": None if left is missing else left, "vllm": None if right is missing else right, "reason": "missing item"})
            else:
                result.extend(differences(left, right, item_path))
        return result
    return [] if reference == actual else [{"path": path, "reference": reference, "vllm": actual}]


def compare_case(case, strategy, request_class, parse_questions):
    request_data = case["request"]
    if any(request_data.get(key) for key in ("images", "videos", "audio")):
        raise ValueError("Encoding comparison accepts pure-text requests")
    if request_data.get("joint") or request_data.get("instructions"):
        raise ValueError("Encoding comparison requires the default joint and fixed top-level prompt")
    expected_hash = canonical_digest(request_data)
    if case.get("request_sha256") != expected_hash:
        raise ValueError(f"{case['name']}: saved reference request hash differs")
    # These are the real serving protocol and strategy functions, without
    # reproducing prefix, schema rendering, tokenization or span calculation.
    request = request_class.model_validate(request_data)
    questions = parse_questions(request, strategy.limits())
    input_ids, schema, orders = strategy.encode(questions, strategy.render_state(request.state))
    length = len(input_ids)
    actual_questions = []
    offset = 0
    for index, question in enumerate(questions):
        count = schema["option_counts"][index]
        question_span = schema["question_spans"][index]
        option_spans = schema["option_spans"][offset:offset + count]
        actual_questions.append({
            "id": question.id,
            "option_ids": [option_id for _, option_id, _ in strategy._options(question)],
            "question_span": [value + length for value in question_span],
            "option_spans": [[value + length for value in span] for span in option_spans],
            "question_span_from_end": question_span,
            "option_spans_from_end": option_spans,
        })
        offset += count
    if offset != len(schema["option_spans"]):
        raise ValueError("vLLM option counts differ from the flat span count")
    reference_encoding = {
        "input_ids": case["encoding"]["input_ids"],
        "input_tokens": case["encoding"]["input_tokens"],
        "questions": case["encoding"]["questions"],
    }
    actual_encoding = {"input_ids": input_ids, "input_tokens": length, "questions": actual_questions}
    # Validate the returned mapping from canonical model option order back to
    # request order against the reference's semantic option IDs.
    expected_orders = []
    for question in case["encoding"]["questions"]:
        specification = request_data["questions"][question["id"]]
        if specification["type"] == "choice":
            request_ids = [str(key) for key in specification["criteria"]]
        elif specification["type"] == "noul":
            request_ids = ["true", "false"]
        else:
            request_ids = [str(i) for i in range(len(specification["criteria"]))]
        expected_orders.append([request_ids.index(option_id) for option_id in question["option_ids"]])
    delta = differences(reference_encoding, actual_encoding)
    delta.extend(differences(expected_orders, orders, "/option_output_orders"))
    return {
        "name": case["name"], "request": request_data,
        "request_sha256": expected_hash, "passed": not delta,
        "reference_input_tokens": reference_encoding["input_tokens"],
        "vllm_input_tokens": length,
        "reference_input_ids_sha256": canonical_digest(reference_encoding["input_ids"]),
        "vllm_input_ids_sha256": canonical_digest(input_ids),
        "input_ids_equal": reference_encoding["input_ids"] == input_ids,
        "reference_encoding": reference_encoding, "vllm_encoding": actual_encoding,
        "vllm_joint_schema": schema,
        "reference_option_output_orders": expected_orders,
        "vllm_option_output_orders": orders,
        "differences": delta, "difference_count": len(delta),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reference-report", type=Path, default=Path("/workspace/yangchen/logs/systemone-reference-cpu-20261112.json"))
    parser.add_argument("--model-dir", type=Path, default=Path(os.environ.get("MODEL_DIR", "/workspace/yangchen/models/Clef-27B")))
    parser.add_argument("--vllm-src", type=Path, default=Path("/workspace/yangchen/work/src/vllm"))
    parser.add_argument("--report", "--output", dest="report", type=Path, required=True)
    args = parser.parse_args()
    report = {
        "schema_version": 1, "created_at": datetime.now(timezone.utc).isoformat(),
        "passed": False, "status": "preparing", "weights_loaded": False,
        "reference_report": str(args.reference_report.resolve()),
        "model_dir": str(args.model_dir.resolve()),
        "vllm_source": str(args.vllm_src.resolve()),
        "required_vllm_commit": VLLM_COMMIT, "cases": [],
        "scope": "Exact pure-text input IDs, absolute/from-end question and option spans, semantic option IDs and output-to-request mapping",
    }
    try:
        reference_bytes = args.reference_report.read_bytes()
        report["reference_report_sha256"] = sha256(reference_bytes)
        reference = json.loads(reference_bytes)
        if reference.get("status") != "complete" or not reference.get("cases"):
            raise ValueError("CPU reference report must be complete and contain encoded cases")
        if reference.get("reference_source_sha256") != REFERENCE_SOURCE_SHA256:
            raise ValueError("CPU reference implementation hash differs from the fixed release")
        model_dir, source = args.model_dir.resolve(), args.vllm_src.resolve()
        if sha256((model_dir / "joint_schema_model.py").read_bytes()) != REFERENCE_SOURCE_SHA256:
            raise ValueError("Current model reference implementation hash differs")
        report["vllm_commit"] = git(source, "rev-parse", "HEAD").decode().strip()
        if report["vllm_commit"] != VLLM_COMMIT:
            raise ValueError("vLLM checkout differs from the fixed commit")
        report["working_tree_status"] = git(source, "status", "--porcelain", "--untracked-files=no").decode().splitlines()
        report["model_metadata_sha256"] = {
            name: sha256((model_dir / name).read_bytes())
            for name in ("config.json", "joint_head_config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "added_tokens.json", "vocab.json", "merges.txt")
            if (model_dir / name).is_file()
        }
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        from vllm.config import ModelConfig
        from vllm.tokenizers import cached_tokenizer_from_config
        from vllm.entrypoints.openai.decisions import strategies
        from vllm.entrypoints.generate.structured_decisions import protocol, serving

        source_files = (
            (strategies, "vllm/entrypoints/openai/decisions/strategies.py"),
            (serving, "vllm/entrypoints/generate/structured_decisions/serving.py"),
            (protocol, "vllm/entrypoints/generate/structured_decisions/protocol.py"),
            (inspect.getmodule(serving.parse_questions.__globals__["build_question"]), "vllm/entrypoints/openai/decisions/question_types.py"),
            (inspect.getmodule(cached_tokenizer_from_config), "vllm/tokenizers/registry.py"),
        )
        report["encoding_source_files"] = []
        for module, relative_path in source_files:
            loaded_path = Path(inspect.getfile(module)).resolve()
            actual_sha = sha256(loaded_path.read_bytes())
            expected_sha = sha256(git(source, "show", f"{VLLM_COMMIT}:{relative_path}"))
            item = {"module_path": str(loaded_path), "git_path": relative_path, "sha256": actual_sha, "fixed_commit_sha256": expected_sha, "passed": actual_sha == expected_sha}
            report["encoding_source_files"].append(item)
            if not item["passed"]:
                raise ValueError(f"Loaded encoding source differs from fixed commit: {relative_path}")
        config = ModelConfig(
            model=str(model_dir), tokenizer=str(model_dir), runner="pooling",
            dtype="bfloat16", max_model_len=reference["max_length"],
            enforce_eager=True, limit_mm_per_prompt={"image": 0, "video": 0},
        )
        tokenizer = cached_tokenizer_from_config(config, local_files_only=True)
        context = strategies.ReadContext(
            engine_client=SimpleNamespace(model_config=config),
            online_renderer=SimpleNamespace(renderer=SimpleNamespace(get_tokenizer=lambda: tokenizer)),
            chat_template=None, chat_template_content_format="auto",
            default_chat_template_kwargs={},
        )
        strategy_class = strategies.select_read_strategy(config, ("token_classify",))
        if strategy_class is not strategies.ClefFlashStrategy:
            raise ValueError(f"Expected ClefFlashStrategy, selected {strategy_class.__name__}")
        strategy = strategy_class(context)
        report["runtime"] = {
            "vllm_version": importlib.metadata.version("vllm"),
            "transformers_version": importlib.metadata.version("transformers"),
            "architecture": config.architecture,
            "strategy_class": f"{strategy_class.__module__}.{strategy_class.__name__}",
            "tokenizer_class": f"{type(tokenizer).__module__}.{type(tokenizer).__name__}",
            "tokenizer_mode": config.tokenizer_mode, "max_model_len": config.max_model_len,
            "max_length": strategy.max_length,
            "context_adapter": "Supplies the real ModelConfig and real vLLM tokenizer; no engine methods are invoked",
        }
        report["status"] = "comparing"
        for case in reference["cases"]:
            try:
                result = compare_case(case, strategy, protocol.StructuredDecisionRequest, serving.parse_questions)
            except Exception as exc:
                result = {"name": case.get("name"), "passed": False, "error": f"{type(exc).__name__}: {exc}", "differences": []}
            report["cases"].append(result)
            print(json.dumps({key: result.get(key) for key in ("name", "passed", "vllm_input_tokens", "difference_count", "error")}), flush=True)
        report["passed"] = bool(report["cases"]) and all(case["passed"] for case in report["cases"])
        report["status"] = "complete" if report["passed"] else "failed"
        report["summary"] = {
            "reference_cases": len(reference["cases"]), "compared_cases": len(report["cases"]),
            "passed_cases": sum(case["passed"] for case in report["cases"]),
            "total_differences": sum(case.get("difference_count", 0) for case in report["cases"]),
        }
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
    report["finished_at"] = datetime.now(timezone.utc).isoformat()
    save(args.report, report)
    print(json.dumps({"passed": report["passed"], "status": report["status"], "report": str(args.report.resolve())}), flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
