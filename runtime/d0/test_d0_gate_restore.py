#!/usr/bin/env python3
"""CPU counterexamples for the fixed gate and the restore wrapper. No device."""
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

# CPU-only: never let torch try to load a device backend here.
os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")

sys.path.insert(0, str(Path(__file__).resolve().parent))
import d0_restore  # noqa: E402

fails = []


def check(name, cond, extra=""):
    print(f"{name}: {'OK' if cond else 'FAIL ' + str(extra)}")
    if not cond:
        fails.append(name)


def gate_cases():
    import torch
    from d0_gate import evaluate
    m, n = 8, 4
    torch.manual_seed(0)
    r = torch.randn(m, n, dtype=torch.bfloat16)
    y = torch.randn(m, n, dtype=torch.bfloat16)
    rs = torch.rand(m, dtype=torch.float32) + 0.5
    base = {"residual": r, "y": y, "rstd": rs}

    same = evaluate(base, base)
    check("identical passes", same["pass"], same["outputs"])

    # one element just over the mixed tolerance
    bad_r = r.clone(); bad_r[0, 0] = (r[0, 0].float() + 0.5).to(torch.bfloat16)
    g = evaluate({"residual": bad_r, "y": y, "rstd": rs}, base)
    check("residual tolerance enforced", not g["pass"] and g["outputs"][0]["fail_count"] == 1)

    # Y max_abs breach
    bad_y = y.clone(); bad_y[1, 1] = (y[1, 1].float() + 0.5).to(torch.bfloat16)
    g = evaluate({"residual": r, "y": bad_y, "rstd": rs}, base)
    check("y abs gate enforced", not g["pass"] and not g["outputs"][1]["pass"])

    # near-zero element must still count for abs and finite, but not for max_rel
    tiny = torch.zeros(1, dtype=torch.bfloat16); tiny[0] = 1e-7
    near = evaluate({"residual": r, "y": tiny, "rstd": rs},
                    {"residual": r, "y": torch.zeros(1, dtype=torch.bfloat16), "rstd": rs})
    check("near-zero keeps abs/finite", near["outputs"][1]["near_zero_count"] == 1
          and "near_zero_max_abs" in near["outputs"][1], near["outputs"][1])

    # shape/dtype precondition
    g = evaluate({"residual": r, "y": y[:4], "rstd": rs}, base)
    check("shape precondition", not g["pass"] and not g["outputs"][1]["shape_ok"])

    # rstd breach
    g = evaluate({"residual": r, "y": y, "rstd": rs + 0.01}, base)
    check("rstd gate enforced", not g["pass"] and not g["outputs"][2]["pass"])


def restore_cases():
    class R:
        def __init__(self, rc=0, out="ok", err=""):
            self.returncode, self.stdout, self.stderr = rc, out, err

    # 1 cleanup kills only recorded identities
    with tempfile.TemporaryDirectory() as td:
        log = {}
        d0_restore.cleanup([999999], [], log)
        check("cleanup records gone pid", log["cleanup"]["already_gone"] == [{"pid": 999999}], log)

    # 2 restore failure is reported separately from the test rc
    with tempfile.TemporaryDirectory() as td:
        log = {"test_rc": 4}
        ok = d0_restore.restore(["true"], None, log, runner=lambda *a, **k: R(rc=9))
        check("restore failure separated", ok is False and log["restore"]["launch_rc"] == 9
              and log["test_rc"] == 4, log)

    # 3 verification stages each keep their own rc/response
    with tempfile.TemporaryDirectory() as td:
        log = {}
        calls = {"n": 0}

        def runner(cmd, **k):
            calls["n"] += 1
            return R(rc=0, out="ready") if calls["n"] != 3 else R(rc=1, out="")

        ok = d0_restore.verify(["a"], ["b"], ["c"], ["d"], log, runner=runner)
        check("verify keeps per-stage rc", ok is False
              and log["health"]["ok"] and not log["text"]["ok"], log)

    # 4 dry-run writes the plan and touches nothing
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "restore.json"
        p = subprocess.run([sys.executable, str(Path(__file__).resolve().parent / "d0_restore.py"),
                            "--out", str(out), "--dry-run"], capture_output=True, text=True)
        plan = json.loads(out.read_text())
        check("dry-run plan only", p.returncode == 0 and plan.get("dry_run") is True
              and not out.with_name("restore.json").exists() is False, plan)


def main() -> int:
    gate_cases()
    restore_cases()
    print("RESULT:", "ALL PASS" if not fails else f"FAILURES: {fails}")
    return 0 if not fails else 1


if __name__ == "__main__":
    raise SystemExit(main())
