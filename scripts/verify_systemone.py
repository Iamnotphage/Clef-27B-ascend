#!/usr/bin/env python3
"""Check the pinned /v1/systemone protocol and save every request/response.

This checks probabilities and candidate mappings. Model accuracy and internal
token/span alignment require separate reference-model evidence.
"""

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


STATE = "The customer was charged twice for one order and requests a refund."
QUESTIONS = {
    "department": {
        "type": "choice",
        "instructions": "Which team should handle this ticket?",
        "criteria": {"billing": "Payments and refunds", "shipping": "Deliveries"},
    },
    "refund_requested": {
        "type": "noul",
        "instructions": "Does the customer request a refund?",
    },
    "severity": {
        "type": "score",
        "instructions": "How severe is the payment problem?",
        "criteria": ["none", "minor", "major"],
    },
}


def cases(model):
    for qid, question in QUESTIONS.items():
        yield question["type"], {
            "model": model, "state": STATE, "questions": {qid: question},
        }
    yield "mixed", {"model": model, "state": STATE, "questions": QUESTIONS}
    reordered = dict(QUESTIONS["department"])
    reordered["criteria"] = dict(reversed(list(reordered["criteria"].items())))
    yield "choice_reordered", {
        "model": model, "state": STATE, "questions": {"department": reordered},
    }


def number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def finite_float(text):
    value = float(text)
    if not math.isfinite(value):
        raise ValueError(f"响应包含非有限数值：{text}")
    return value


def reject_constant(text):
    raise ValueError(f"响应包含非 JSON 数值：{text}")


def validate(request, response, atol):
    errors = []

    def check(ok, message):
        if not ok:
            errors.append(message)

    check(response.get("model") == request["model"], "服务模型名匹配")
    check(response.get("object") == "structured_decision", "响应 object 匹配")
    answers = response.get("answers")
    if not isinstance(answers, dict):
        return errors + ["answers 应为对象"]
    check(set(answers) == set(request["questions"]), "问题 ID 完整对应")
    for qid, question in request["questions"].items():
        answer = answers.get(qid)
        if not isinstance(answer, dict):
            errors.append(f"{qid}: 答案应为对象")
            continue
        kind = question["type"]
        check(answer.get("type") == kind, f"{qid}: 答案类型匹配")
        probabilities = answer.get("probabilities")
        if not isinstance(probabilities, dict):
            errors.append(f"{qid}: probabilities 应为对象")
            continue
        expected = (
            set(question["criteria"]) if kind == "choice"
            else {"yes", "no"} if kind == "noul"
            else {str(i) for i in range(len(question["criteria"]))}
        )
        check(set(probabilities) == expected, f"{qid}: 候选概率键完整对应")
        valid_probs = all(number(p) and 0 <= p <= 1 for p in probabilities.values())
        check(valid_probs, f"{qid}: 候选概率有限且位于 [0, 1]")
        if not valid_probs or set(probabilities) != expected:
            continue
        check(abs(sum(probabilities.values()) - 1) <= atol, f"{qid}: 候选概率归一化")
        confidence = answer.get("confidence")
        check(number(confidence) and 0 <= confidence <= 1, f"{qid}: confidence 有效")
        if kind == "choice":
            chosen = answer.get("choice")
            check(chosen in expected, f"{qid}: choice 属于请求候选")
            if chosen in probabilities:
                check(
                    max(probabilities.values()) - probabilities[chosen] <= atol,
                    f"{qid}: choice 对应最大概率",
                )
        elif kind == "noul":
            value = answer.get("noul")
            check(
                number(value) and 0 <= value <= 1 and abs(value - probabilities["yes"]) <= atol,
                f"{qid}: noul 等于 yes 概率",
            )
        else:
            value = answer.get("score")
            expectation = sum(int(i) * p for i, p in probabilities.items())
            check(
                number(value) and 0 <= value <= len(expected) - 1 and abs(value - expectation) <= atol,
                f"{qid}: score 等于零起始等级期望",
            )
            legend = {str(i): name for i, name in enumerate(question["criteria"])}
            check(answer.get("legend") == legend, f"{qid}: score legend 对应等级顺序")
    usage = response.get("usage", {})
    check(isinstance(usage, dict), "usage 应为对象")
    if isinstance(usage, dict):
        check(usage.get("output_tokens") == 0, "pooling output_tokens 等于零")
        count = usage.get("input_tokens")
        check(isinstance(count, int) and not isinstance(count, bool) and count > 0, "input_tokens 为正整数")
    diagnostics = response.get("diagnostics", {})
    check(isinstance(diagnostics, dict) and set(diagnostics) == set(request["questions"]), "diagnostics 问题 ID 对应")
    return errors


def get_health(base_url, wait_seconds, timeout):
    deadline = time.monotonic() + wait_seconds
    while True:
        try:
            with urlopen(base_url + "/health", timeout=min(timeout, 10)) as response:
                if response.status == 200:
                    return {"passed": True, "status": 200}
        except (OSError, URLError) as exc:
            error = str(exc)
        else:
            error = "健康检查返回非 200 状态"
        if time.monotonic() >= deadline:
            return {"passed": False, "error": error}
        time.sleep(min(2, max(0, deadline - time.monotonic())))


def main():
    parser = argparse.ArgumentParser(description="验收 /v1/systemone 并保存原始请求和响应。")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="clef")
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--wait-seconds", type=float, default=0)
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--reorder-atol", type=float, default=1e-5)
    args = parser.parse_args()
    numeric_args = (args.timeout, args.wait_seconds, args.atol, args.reorder_atol)
    if not all(math.isfinite(value) for value in numeric_args) or args.timeout <= 0 or args.wait_seconds < 0 or min(args.atol, args.reorder_atol) < 0:
        parser.error("超时应为正数，等待时间和容差应为非负数")
    base_url = args.base_url.rstrip("/")
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "base_url": base_url,
        "model": args.model,
        "scope": "HTTP 协议、三类答案、混合字段、概率归一化和候选重排映射",
        "atol": args.atol,
        "reorder_atol": args.reorder_atol,
        "health": get_health(base_url, args.wait_seconds, args.timeout),
        "cases": [],
        "passed": False,
    }
    if report["health"]["passed"]:
        for name, request in cases(args.model):
            started = time.perf_counter()
            result = {"name": name, "request": request, "passed": False}
            try:
                message = Request(
                    base_url + "/v1/systemone",
                    data=json.dumps(request, ensure_ascii=False).encode(),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urlopen(message, timeout=args.timeout) as response:
                    result["http_status"] = response.status
                    result["response_text"] = response.read().decode("utf-8")
                result["response"] = json.loads(
                    result["response_text"], parse_float=finite_float,
                    parse_constant=reject_constant,
                )
                if not isinstance(result["response"], dict):
                    raise ValueError("响应 JSON 应为对象")
                result["errors"] = validate(request, result["response"], args.atol)
                result["passed"] = result["http_status"] == 200 and not result["errors"]
            except HTTPError as exc:
                result.update(http_status=exc.code, response_text=exc.read().decode("utf-8", errors="replace"), error=str(exc))
            except (OSError, URLError, ValueError, TypeError, KeyError) as exc:
                result["error"] = str(exc)
            result["elapsed_seconds"] = time.perf_counter() - started
            report["cases"].append(result)
            print(json.dumps({key: result.get(key) for key in ("name", "passed", "elapsed_seconds", "errors", "error")}, ensure_ascii=False), flush=True)
        by_name = {case["name"]: case for case in report["cases"]}
        original, reordered = by_name["choice"], by_name["choice_reordered"]
        if original["passed"] and reordered["passed"]:
            a = original["response"]["answers"]["department"]
            b = reordered["response"]["answers"]["department"]
            delta = max(abs(a["probabilities"][key] - b["probabilities"][key]) for key in a["probabilities"])
            report["candidate_reordering"] = {
                "max_probability_absolute_difference": delta,
                "same_choice": a["choice"] == b["choice"],
                "passed": delta <= args.reorder_atol and a["choice"] == b["choice"],
            }
        else:
            report["candidate_reordering"] = {"passed": False, "error": "候选重排比较需要两个 choice 请求通过"}
        report["passed"] = all(case["passed"] for case in report["cases"]) and report["candidate_reordering"]["passed"]
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"passed": report["passed"], "report": str(args.report.resolve())}, ensure_ascii=False))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
