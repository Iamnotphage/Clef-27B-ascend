#!/usr/bin/env python3
"""Replay the frozen 8-case baseline against /v1/systemone and compare.

Response fields that legitimately change between runs (id, created,
elapsed_ms) are ignored; every answer field, probability, usage count and
diagnostic must match bit for bit.
"""

import argparse
import json
import urllib.request

IGNORED = {"id", "created"}


def normalize(value):
    if isinstance(value, dict):
        return {key: normalize(item) for key, item in value.items() if key not in IGNORED}
    if isinstance(value, list):
        return [normalize(item) for item in value]
    return value


def post(url, payload, timeout):
    body = json.dumps(payload, ensure_ascii=False).encode()
    request = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.status, json.loads(response.read())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--url", default="http://127.0.0.1:8000/v1/systemone")
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    baseline = json.loads(open(args.baseline, encoding="utf-8").read())
    report = {"url": args.url, "cases": {}, "passed": True}
    for name, case in baseline.items():
        status, response = post(args.url, case["request"], args.timeout)
        expected = normalize(case["response"])
        actual = normalize(response)
        equal = expected == actual
        report["cases"][name] = {
            "status": status,
            "expected_status": case["status"],
            "equal": equal,
            "expected_sha256": None,
            "actual": actual if not equal else None,
        }
        if status != case["status"] or not equal:
            report["passed"] = False
        print(f"{name}: status={status} equal={equal}")
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=1)
        handle.write("\n")
    print(json.dumps({"passed": report["passed"], "cases": len(report["cases"])}))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
