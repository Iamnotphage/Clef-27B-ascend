#!/usr/bin/env python3
"""Run the fixed development subset against /v1/systemone and save evidence.

Accuracy is a micro average over the selected decisions; this report records a
project smoke evaluation. Full benchmark aggregate scores use pinned EvalScope
and each benchmark's complete official protocol.
"""

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from verify_systemone import finite_float, get_health, reject_constant, validate


BENCHMARKS = ("jevbench", "open_decisions", "typed_decisions")


def grade(question, answer, gold, benchmark):
    kind = question["type"]
    probabilities = answer["probabilities"]
    if benchmark == "typed_decisions":
        # Match the pinned dataset scorer's option order and yes/no conversion.
        options = list(gold["probabilities"])
        raw = {({"yes": "true", "no": "false"}.get(key, key) if kind == "noul" else key): value for key, value in probabilities.items()}
        total = sum(raw.values())
        predicted_probs = {key: raw[key] / total for key in options}
        target_probs = {key: float(gold["probabilities"][key]) for key in options}
        predicted = max(options, key=predicted_probs.get)
        target = max(options, key=target_probs.get)
        result = {
            "type": kind, "predicted_label": predicted, "gold_label": target,
            "correct": predicted == target, "probabilities_for_scoring": predicted_probs,
            "kl_from_gold": sum(value * math.log(value / max(predicted_probs[key], 1e-9)) for key, value in target_probs.items() if value > 0),
            "brier_from_gold": sum((predicted_probs[key] - target_probs[key]) ** 2 for key in options),
        }
        if kind == "score":
            expected = sum(int(key) * value for key, value in predicted_probs.items())
            target_score = float(gold.get("score", sum(int(key) * value for key, value in target_probs.items())))
            result.update(predicted_expected_score=expected, gold_expected_score=target_score, score_absolute_error=abs(expected - target_score))
        return result

    options = [str(key) for key in gold["labels"]]
    total = sum(probabilities.values())
    # Preserve each source scorer's rounding and normalization rules.
    tolerance = 0.02 if benchmark == "jevbench" else 0.01 + 1e-12
    if abs(total - 1) > tolerance:
        raise ValueError("候选概率总和超出来源评分协议容差")
    normalized = benchmark == "jevbench" and abs(total - 1) > 0.001
    probs = {key: probabilities[key] / (total if normalized else 1) for key in options}
    if kind == "noul" and benchmark == "open_decisions":
        predicted = "yes" if probs["yes"] >= 0.5 else "no"
    else:
        # EvalScope decision_score uses sorted labels to break an argmax tie;
        # this makes a JevBench Noul tie select "no".
        predicted = max(sorted(options), key=probs.get)
    target = gold["expected"]
    if kind == "noul":
        target = "yes" if target is True or str(target).lower() in ("true", "yes") else "no"
    result = {
        "type": kind, "predicted_label": predicted, "gold_label": str(target),
        "correct": predicted == str(target), "probabilities_for_scoring": probs,
        "probabilities_normalized_for_scoring": normalized,
    }
    if kind == "score":
        expected = sum(index * probs[label] for index, label in enumerate(options))
        result.update(predicted_expected_score=expected, gold_expected_score=float(target), score_absolute_error=abs(expected - float(target)))
    return result


def summarize(cases):
    grouped = defaultdict(list)
    for case in cases:
        benchmark = case["source"]["benchmark"]
        for question in case["question_results"]:
            grouped["all"].append(question)
            grouped[benchmark].append(question)
            grouped[benchmark + "/" + question["type"]].append(question)
    summary = {}
    for group, questions in grouped.items():
        valid = [question for question in questions if question["status"] == "valid"]
        score_errors = [question["score_absolute_error"] for question in valid if "score_absolute_error" in question]
        correct = sum(question["correct"] for question in questions)
        result = {
            "requested_decisions": len(questions), "valid_decisions": len(valid),
            "correct_decisions": correct, "accuracy": correct / len(questions),
            "valid_prediction_accuracy": sum(question["correct"] for question in valid) / len(valid) if valid else None,
            "score_mae": sum(score_errors) / len(score_errors) if score_errors else None,
            "score_error_count": len(score_errors),
        }
        for metric in ("kl_from_gold", "brier_from_gold"):
            values = [question[metric] for question in valid if metric in question]
            if values:
                result[metric] = sum(values) / len(values)
                result[metric + "_count"] = len(values)
        summary[group] = result
    return summary


def save(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description="固定小样本项目自检，保存逐题请求、响应和评分。")
    parser.add_argument("--cases", type=Path, required=True, help="api-smoke28-cases.json")
    parser.add_argument("--benchmark", action="append", choices=BENCHMARKS, help="可重复指定；默认评测全部三种来源。")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="clef")
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--wait-seconds", type=float, default=0)
    parser.add_argument("--atol", type=float, default=1e-5)
    args = parser.parse_args()
    if not all(math.isfinite(value) for value in (args.timeout, args.wait_seconds, args.atol)) or args.timeout <= 0 or min(args.wait_seconds, args.atol) < 0:
        parser.error("超时应为正数，等待时间和概率容差应为非负数")
    content = args.cases.read_bytes()
    selected = [case for case in json.loads(content)["cases"] if not args.benchmark or case["source"]["benchmark"] in args.benchmark]
    if not selected:
        parser.error("所选来源需要包含至少一个 case")
    if any(case["source"]["benchmark"] not in BENCHMARKS for case in selected):
        parser.error("cases 包含尚未定义评分语义的来源")
    base_url = args.base_url.rstrip("/")
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scope": "固定小样本项目自检；准确率为所选决策的 micro average，失败计入分母。",
        "cases_file": str(args.cases.resolve()), "cases_sha256": hashlib.sha256(content).hexdigest(),
        "base_url": base_url, "model": args.model, "probability_atol": args.atol,
        "grading": {
            "jevbench": "Argmax over sorted source labels; Noul ties select no; Score MAE compares expected level with discrete gold.",
            "open_decisions": "Noul P(yes)>=0.5; other types use argmax over sorted source labels; Score MAE compares expected level with discrete gold.",
            "typed_decisions": "Preserve each gold distribution's option order; normalize predictions; map yes/no to true/false; accuracy compares distribution argmax; Score MAE compares with gold expected score.",
        },
        "health": get_health(base_url, args.wait_seconds, args.timeout),
        "cases": [], "status": "running",
    }
    save(args.report, report)
    for case in selected:
        request = {**case["request"], "model": args.model}
        result = {"name": case["name"], "source": case["source"], "request": request, "gold": case["gold"], "passed": False, "question_results": []}
        started = time.perf_counter()
        try:
            if not report["health"]["passed"]:
                raise ValueError("服务健康检查尚未通过")
            message = Request(base_url + "/v1/systemone", data=json.dumps(request, ensure_ascii=False).encode(), headers={"Content-Type": "application/json"}, method="POST")
            with urlopen(message, timeout=args.timeout) as response:
                result["http_status"] = response.status
                result["response_text"] = response.read().decode("utf-8")
            result["response"] = json.loads(result["response_text"], parse_float=finite_float, parse_constant=reject_constant)
            if not isinstance(result["response"], dict):
                raise ValueError("响应 JSON 应为对象")
            result["validation_errors"] = validate(request, result["response"], args.atol)
            if result["http_status"] != 200 or result["validation_errors"]:
                raise ValueError("HTTP 响应协议检查失败")
            for qid, question in request["questions"].items():
                target = case["gold"][qid] if case["source"]["benchmark"] == "typed_decisions" else case["gold"]
                score = grade(question, result["response"]["answers"][qid], target, case["source"]["benchmark"])
                result["question_results"].append({"id": qid, "status": "valid", **score})
            result["passed"] = True
        except HTTPError as exc:
            result.update(http_status=exc.code, response_text=exc.read().decode("utf-8", errors="replace"), error=str(exc))
        except (OSError, URLError, ValueError, KeyError, TypeError, ZeroDivisionError) as exc:
            result["error"] = f"{type(exc).__name__}: {exc}"
        completed = {question["id"] for question in result["question_results"]}
        for qid, question in request["questions"].items():
            if qid not in completed:
                result["question_results"].append({"id": qid, "type": question["type"], "status": "failed", "correct": False})
        result["elapsed_seconds"] = time.perf_counter() - started
        report["cases"].append(result)
        report["summary"] = summarize(report["cases"])
        save(args.report, report)
        print(json.dumps({"name": result["name"], "passed": result["passed"], "elapsed_seconds": result["elapsed_seconds"], "error": result.get("error")}, ensure_ascii=False), flush=True)
    report["status"] = "complete" if all(case["passed"] for case in report["cases"]) else "failed"
    report["requested_cases"] = len(selected)
    report["successful_cases"] = sum(case["passed"] for case in report["cases"])
    save(args.report, report)
    print(json.dumps({"status": report["status"], "report": str(args.report.resolve()), "summary": report["summary"]}, ensure_ascii=False))
    return 0 if report["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
