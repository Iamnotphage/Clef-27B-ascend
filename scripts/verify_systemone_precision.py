#!/usr/bin/env python3
"""Combine saved CPU, API and exact-encoding evidence without running a model."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import traceback

from reference_systemone import SOURCE_SHA256, compare, digest
from verify_systemone import finite_float, reject_constant, validate


DEFAULT_CRITERIA = Path(__file__).resolve().parents[1] / "docs/evidence/precision-criteria-2026-11-12.json"
CASE_NAMES = {"choice", "noul", "score", "mixed", "choice_reordered"}


def read_report(path):
    content = path.read_bytes()
    value = json.loads(content, parse_float=finite_float, parse_constant=reject_constant)
    if not isinstance(value, dict):
        raise ValueError(f"报告应为 JSON 对象：{path}")
    return value, hashlib.sha256(content).hexdigest()


def index_cases(report, description):
    rows = report.get("cases")
    if not isinstance(rows, list):
        raise ValueError(f"{description}: cases 应为列表")
    indexed = {}
    for case in rows:
        name = case.get("name")
        if not isinstance(name, str) or not name or name in indexed:
            raise ValueError(f"{description}: case 名称需要非空且唯一")
        indexed[name] = case
    if set(indexed) != CASE_NAMES:
        raise ValueError(f"{description}: 需要五种固定 case，实际为 {sorted(indexed)}")
    return indexed


def require(condition, message):
    if not condition:
        raise ValueError(message)


def request_order(request):
    # The shared canonical digest sorts object keys. Preserve the two object
    # orders that carry field positions and candidate-output mappings as well.
    return {
        "question_ids": list(request["questions"]),
        "choice_candidate_ids": {
            qid: list(question["criteria"])
            for qid, question in request["questions"].items()
            if question["type"] == "choice"
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-report", type=Path, required=True)
    parser.add_argument("--api-report", type=Path, required=True)
    parser.add_argument("--encoding-report", type=Path, required=True)
    parser.add_argument("--criteria", type=Path, default=DEFAULT_CRITERIA)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scope": "Five fixed SystemOne requests, seven decisions; saved complete CPU BF16 reference, API protocol and exact token/span/mapping evidence",
        "passed": False, "status": "preparing", "model_executed": False,
        "inputs": {}, "cases": [],
    }
    try:
        documents = {}
        for name in ("reference_report", "api_report", "encoding_report", "criteria"):
            path = getattr(args, name)
            document, sha = read_report(path)
            documents[name] = document
            report["inputs"][name] = {"path": str(path.resolve()), "sha256": sha}
        reference, api, encoding, criteria = (documents[name] for name in ("reference_report", "api_report", "encoding_report", "criteria"))
        report["criteria"] = criteria
        probability_limit = criteria["candidate_probability_max_absolute_difference"]
        score_limit = criteria["score_max_absolute_difference"]
        require(all(isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0 for value in (probability_limit, score_limit)), "概率与 score 容差应为有限非负数")
        require(criteria.get("noul_threshold") == 0.5, "本比较函数采用固定 P(yes)>=0.5 的离散判定")
        require(criteria.get("established_before_npu_service") is True, "验收标准需要记录为服务执行前确定")
        require(all(criteria.get(key) is True for key in (
            "require_same_choice", "require_same_noul_threshold_decision", "require_same_score_argmax_level",
            "require_identical_token_ids", "require_identical_question_option_spans", "require_identical_candidate_mapping",
        )), "本轮完整验收需要启用全部决策与编码一致性条件")
        require(reference.get("status") == "complete", "CPU 参考报告需要完成全部前向")
        require(reference.get("device") == "cpu" and reference.get("dtype") == "bfloat16", "参考报告需要记录 CPU BF16")
        require(reference.get("reference_source_sha256") == SOURCE_SHA256, "CPU 参考实现 SHA-256 需要与模型 manifest 对应")
        reference_cases = index_cases(reference, "CPU")
        api_cases = index_cases(api, "API")
        encoding_cases = index_cases(encoding, "编码")
        report["api_protocol_passed"] = api.get("passed") is True
        report["encoding_passed"] = encoding.get("passed") is True
        require(encoding.get("reference_report_sha256") == report["inputs"]["reference_report"]["sha256"], "编码报告需要引用本次 CPU 参考文件哈希")
        require(api.get("passed") is True, "API 协议报告需要 passed=true")
        require(encoding.get("passed") is True, "token/span/候选映射编码报告需要 passed=true")

        max_probability, max_score = 0.0, 0.0
        decisions, matching_decisions = 0, 0
        for name, expected in reference_cases.items():
            actual, encoded = api_cases[name], encoding_cases[name]
            result = {"name": name, "passed": False, "fields": {}}
            try:
                request = expected["request"]
                request_sha = digest(request)
                result["request_sha256"] = request_sha
                require(expected.get("request_sha256") == request_sha, f"{name}: CPU 原始请求哈希需要对应")
                require(digest(actual["request"]) == request_sha, f"{name}: API 原始请求需要与 CPU 对应")
                require(digest(encoded["request"]) == request_sha and encoded.get("request_sha256") == request_sha, f"{name}: 编码原始请求哈希需要与 CPU 对应")
                order = request_order(request)
                require(request_order(actual["request"]) == order and request_order(encoded["request"]) == order, f"{name}: 问题顺序和 choice 候选顺序需要与 CPU 对应")
                result["request_order"] = order
                if "request_sha256" in actual:
                    require(actual["request_sha256"] == request_sha, f"{name}: API 保存的请求哈希需要对应")
                require(actual.get("passed") is True and encoded.get("passed") is True, f"{name}: API 与编码 case 均需通过")
                response = actual["response"]
                protocol_errors = validate(request, response, api.get("atol", 1e-5))
                result["api_protocol_errors"] = protocol_errors
                require(not protocol_errors, f"{name}: 保存的 API 响应需要通过协议复查")
                require(set(expected["semantic_answers"]) == set(request["questions"]), f"{name}: CPU 共同语义字段需要完整")
                input_tokens = expected["encoding"]["input_tokens"]
                result["token_counts"] = {
                    "reference": input_tokens, "api": response["usage"]["input_tokens"],
                    "encoding_reference": encoded["reference_input_tokens"], "encoding_vllm": encoded["vllm_input_tokens"],
                }
                require(all(value == input_tokens for value in result["token_counts"].values()), f"{name}: CPU/API/编码 token 数需要完全对应")
                result["same_input_token_count"] = True
                fields = compare(expected["semantic_answers"], response)
                for qid, difference in fields.items():
                    left = expected["semantic_answers"][qid]
                    right = response["answers"][qid]
                    kind = request["questions"][qid]["type"]
                    same = difference["same_argmax_level"] if kind == "score" else difference["same_decision"]
                    difference["type"] = kind
                    difference["same_discrete_decision"] = same
                    difference["probability_within_tolerance"] = difference["max_probability_absolute_difference"] <= probability_limit
                    difference["score_within_tolerance"] = difference.get("score_absolute_difference", 0) <= score_limit
                    if "confidence" in left and "confidence" in right:
                        difference["confidence_absolute_difference"] = abs(left["confidence"] - right["confidence"])
                    difference["passed"] = same and difference["probability_within_tolerance"] and difference["score_within_tolerance"]
                    decisions += 1
                    matching_decisions += bool(same)
                    max_probability = max(max_probability, difference["max_probability_absolute_difference"])
                    max_score = max(max_score, difference.get("score_absolute_difference", 0))
                result["fields"] = fields
                result["passed"] = all(field["passed"] for field in fields.values())
            except Exception as exc:
                result["error"] = f"{type(exc).__name__}: {exc}"
            report["cases"].append(result)
        report["summary"] = {
            "case_count": len(report["cases"]), "passed_cases": sum(case["passed"] for case in report["cases"]),
            "decision_count": decisions, "matching_discrete_decisions": matching_decisions,
            "max_probability_absolute_difference": max_probability,
            "max_score_absolute_difference": max_score,
            "probability_limit": probability_limit, "score_limit": score_limit,
        }
        report["passed"] = decisions == 7 and matching_decisions == 7 and all(case["passed"] for case in report["cases"])
        report["status"] = "complete" if report["passed"] else "failed"
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
    report["finished_at"] = datetime.now(timezone.utc).isoformat()
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"passed": report["passed"], "status": report["status"], "report": str(args.report.resolve()), "summary": report.get("summary"), "error": report.get("error")}, ensure_ascii=False))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
