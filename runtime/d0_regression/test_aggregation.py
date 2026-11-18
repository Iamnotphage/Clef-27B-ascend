#!/usr/bin/env python3
"""CPU regression for the D0 suite aggregation rule.

The suite must fail when a case JSON says ``case_ok=true`` but the child
process exits non-zero.  No production D0 function is imported or replaced.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRIPT = HERE / "d0_main_regression.py"


def load_module():
    spec = importlib.util.spec_from_file_location("d0_main_regression", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def main() -> int:
    module = load_module()
    cases = {
        "all_zero": ([{"case_ok": True, "case_rc": 0}] * 14, True, 0),
        "ok_json_rc1": ([{"case_ok": True, "case_rc": 1}], False, 1),
        "bad_json_rc0": ([{"case_ok": False, "case_rc": 0}], False, 1),
        "missing_case": ([{"scenario": "missing", "case_ok": False}], False, 1),
        "mixed": ([{"case_ok": True, "case_rc": 0},
                   {"case_ok": True, "case_rc": 1}], False, 1),
    }
    failures = []
    for name, (items, want_ok, want_rc) in cases.items():
        got = module.aggregate_cases(items)
        ok = got["case_ok"] is want_ok and got["suite_rc"] == want_rc
        print(f"{'ok  ' if ok else 'FAIL'} {name}: {got}")
        if not ok:
            failures.append(name)
    result = {"schema": "d0-aggregation-self-test/1",
              "checks": len(cases), "failed": failures,
              "ok": not failures}
    print(json.dumps(result, ensure_ascii=False))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
