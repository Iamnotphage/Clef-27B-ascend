#!/usr/bin/env python3
"""Run the verified release implementation on CPU BF16 and compare API output.

Use --dry-run to inspect inputs and weight sizes without importing PyTorch or
loading weights. Actual inference loads the complete backbone and joint head.
"""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import resource
import sys
import time
import traceback

from verify_systemone import cases


SOURCE_SHA256 = "0e304cf7c6500e8bb59bef7e2afd2c6373f82596dfb3b57d1aa93c175e2dc3a3"


def digest(value):
    data = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(data).hexdigest()


def rss_bytes():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value if sys.platform == "darwin" else value * 1024


def write_report(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def common_answer(question, probabilities):
    kind = question["type"]
    if kind == "noul":
        return {
            "type": kind, "noul": probabilities["true"],
            "probabilities": {"yes": probabilities["true"], "no": probabilities["false"]},
            "confidence": max(probabilities.values()),
        }
    if kind == "choice":
        ordered = {str(key): probabilities[str(key)] for key in question["criteria"]}
        return {
            "type": kind, "choice": max(ordered, key=ordered.get),
            "probabilities": ordered, "confidence": max(ordered.values()),
        }
    return {
        "type": kind,
        "score": sum(i * probabilities[str(i)] for i in range(len(question["criteria"]))),
        "probabilities": probabilities,
        "legend": {str(i): label for i, label in enumerate(question["criteria"])},
        "confidence": max(probabilities.values()),
    }


def compare(reference, api):
    fields = {}
    for qid, expected in reference.items():
        actual = api["answers"][qid]
        if actual["type"] != expected["type"] or set(actual["probabilities"]) != set(expected["probabilities"]):
            raise ValueError(f"{qid}: API 题型或候选键与参考结果对应失败")
        difference = {
            key: abs(float(actual["probabilities"][key]) - probability)
            for key, probability in expected["probabilities"].items()
        }
        if not all(math.isfinite(value) for value in difference.values()):
            raise ValueError(f"{qid}: API 概率包含非有限数值")
        item = {"probability_absolute_differences": difference, "max_probability_absolute_difference": max(difference.values())}
        kind = expected["type"]
        if kind == "choice":
            item["same_decision"] = actual["choice"] == expected["choice"]
        elif kind == "noul":
            item["same_decision"] = (actual["noul"] >= 0.5) == (expected["noul"] >= 0.5)
            item["noul_absolute_difference"] = abs(actual["noul"] - expected["noul"])
        else:
            item["same_argmax_level"] = max(actual["probabilities"], key=actual["probabilities"].get) == max(expected["probabilities"], key=expected["probabilities"].get)
            item["score_absolute_difference"] = abs(actual["score"] - expected["score"])
        fields[qid] = item
    return fields


def load_cases(args):
    if args.api_report:
        document = json.loads(args.api_report.read_text())
        selected = document.get("cases", [])
        if not selected:
            raise ValueError("API 报告需要包含已保存请求的 cases")
        return selected
    return [{"name": name, "request": request} for name, request in cases(args.model)]


def main():
    parser = argparse.ArgumentParser(description="CPU BF16 完整 Clef 参考前向与 API 共同语义比较。")
    parser.add_argument("--model-dir", type=Path, default=Path(os.environ.get("MODEL_DIR", "/workspace/yangchen/models/Clef-27B")))
    parser.add_argument("--model", default="clef")
    parser.add_argument("--api-report", type=Path, help="verify_systemone.py 报告；复用其中的原始请求和响应。")
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--threads", type=int, default=32)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if min(args.max_length, args.threads) <= 0:
        parser.error("上下文长度和线程数应为正数")
    model_dir = args.model_dir.expanduser().resolve()
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_dir": str(model_dir), "device": "cpu", "dtype": "bfloat16",
        "max_length": args.max_length, "threads": args.threads,
        "attention_implementation": "eager", "use_kernels": False,
        "noul_comparison_rule": "yes iff P(yes) >= 0.5; benchmark-specific scoring is separate",
        "api_report": str(args.api_report.resolve()) if args.api_report else None,
        "status": "preparing", "cases": [],
    }
    try:
        selected = load_cases(args)
        source_path = model_dir / "joint_schema_model.py"
        source_sha = hashlib.sha256(source_path.read_bytes()).hexdigest()
        report["reference_source_sha256"] = source_sha
        if source_sha != SOURCE_SHA256:
            raise ValueError("参考实现 SHA-256 与模型 manifest 固定值对应失败")
        index = json.loads((model_dir / "model.safetensors.index.json").read_text())
        names = sorted(set(index["weight_map"].values()) | {"joint_head.safetensors"})
        weights = []
        for name in names:
            path = (model_dir / name).resolve()
            if not path.is_relative_to(model_dir) or not path.is_file():
                raise ValueError(f"权重文件路径无效：{name}")
            weights.append({"name": name, "file_bytes": path.stat().st_size})
        report["weights"] = weights
        report["weight_file_gib"] = sum(item["file_bytes"] for item in weights) / 2**30
        report["planned_cases"] = [{"name": case["name"], "request_sha256": digest(case["request"])} for case in selected]
        for case in selected:
            if any(case["request"].get(key) for key in ("images", "videos", "audio")):
                raise ValueError("本参考脚本接受纯文本请求")
            if case["request"].get("joint") or case["request"].get("instructions"):
                raise ValueError("Clef 参考请求采用默认 joint 和固定顶层 prompt")
        if args.dry_run:
            report["status"] = "planned"
            write_report(args.report, report)
            print(json.dumps(report, ensure_ascii=False, indent=2))
            return 0

        # Keep the CPU reference independent of backend auto-registration and downloads.
        os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        import torch

        torch.set_num_threads(args.threads)
        torch.manual_seed(0)
        report["versions"] = {}
        for name in ("torch", "transformers", "accelerate", "safetensors"):
            report["versions"][name] = importlib.metadata.version(name)
        spec = importlib.util.spec_from_file_location("clef_verified_reference", source_path)
        release = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = release
        spec.loader.exec_module(release)
        report["status"] = "loading"
        write_report(args.report, report)
        started = time.perf_counter()
        model, processor = release.load_release_model(
            model_dir, device="cpu", dtype=torch.bfloat16,
            local_files_only=True, use_kernels=False, attn_implementation="eager",
        )
        report["load_seconds"] = time.perf_counter() - started
        parameters = Counter()
        for parameter in model.parameters():
            parameters[f"{parameter.device}/{parameter.dtype}"] += parameter.numel()
            if parameter.device.type != "cpu" or parameter.dtype != torch.bfloat16:
                raise ValueError(f"参考参数设备或 dtype 与 CPU BF16 配置对应失败：{parameter.device}/{parameter.dtype}")
        report["parameter_counts"] = dict(parameters)
        report["status"] = "running"
        write_report(args.report, report)
        for case in selected:
            result = {"name": case["name"], "request": case["request"], "request_sha256": digest(case["request"])}
            started = time.perf_counter()
            encoded = release.encode_record(
                processor.tokenizer, case["request"], max_length=args.max_length, processor=processor,
            )
            length = len(encoded.input_ids)
            result["encoding"] = {
                "input_ids": list(encoded.input_ids),
                "input_ids_sha256": digest(list(encoded.input_ids)),
                "input_tokens": length,
                "questions": [
                    {
                        "id": question.question_id,
                        "option_ids": list(question.option_ids),
                        "question_span": list(question.question_span),
                        "option_spans": [list(span) for span in question.option_spans],
                        "question_span_from_end": [value - length for value in question.question_span],
                        "option_spans_from_end": [[value - length for value in span] for span in question.option_spans],
                    }
                    for question in encoded.questions
                ],
            }
            batch = release.collate_records([encoded], processor.tokenizer.pad_token_id, torch.device("cpu"))
            with torch.inference_mode():
                logits = model(batch)[0]
            semantic, official = {}, {}
            for question, values in zip(encoded.questions, logits):
                if not torch.isfinite(values).all().item():
                    raise ValueError(f"{case['name']}/{question.question_id}: 参考 logits 包含 NaN/Inf")
                probabilities = dict(zip(question.option_ids, values.float().softmax(-1).tolist()))
                requested = case["request"]["questions"][question.question_id]
                semantic[question.question_id] = common_answer(requested, probabilities)
                official[question.question_id] = release.systemone_answer(requested, probabilities)
            result["semantic_answers"] = semantic
            result["official_response"] = {
                "model": case["request"]["model"], "answers": official,
                "usage": {"input_tokens": length, "output_tokens": 0},
            }
            result["elapsed_seconds"] = time.perf_counter() - started
            if "response" in case:
                result["api_comparison"] = compare(semantic, case["response"])
                result["same_input_token_count"] = case["response"]["usage"]["input_tokens"] == length
            report["cases"].append(result)
            report["peak_rss_bytes"] = rss_bytes()
            write_report(args.report, report)
            print(json.dumps({"name": result["name"], "input_tokens": length, "elapsed_seconds": result["elapsed_seconds"]}), flush=True)
            del batch, logits
        report["status"] = "complete"
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        report["traceback"] = traceback.format_exc()
    report["peak_rss_bytes"] = rss_bytes()
    write_report(args.report, report)
    print(json.dumps({"status": report["status"], "report": str(args.report.resolve())}, ensure_ascii=False))
    return 0 if report["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
