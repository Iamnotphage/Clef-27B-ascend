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

    # NaN / Inf on either side
    nan_y = y.clone(); nan_y[0, 0] = float("nan")
    g = evaluate({"residual": r, "y": nan_y, "rstd": rs}, base)
    check("candidate NaN rejected", not g["pass"] and not g["outputs"][1]["cand_finite"])
    inf_r = r.clone(); inf_r[0, 0] = float("inf")
    g = evaluate({"residual": inf_r, "y": y, "rstd": rs},
                 {"residual": r, "y": y, "rstd": rs})
    check("candidate Inf rejected", not g["pass"] and not g["outputs"][0]["cand_finite"])
    bad_ref = {"residual": r.clone(), "y": y.clone(), "rstd": rs.clone()}
    bad_ref["y"][0, 0] = float("nan")
    g = evaluate(base, bad_ref)
    check("reference NaN rejected", not g["pass"] and not g["outputs"][1]["ref_finite"])

    # dtype mismatch
    g = evaluate({"residual": r.float(), "y": y, "rstd": rs}, base)
    check("dtype mismatch rejected", not g["pass"] and not g["outputs"][0]["dtype_ok"])

    # Y relative fails while abs passes
    r0 = torch.tensor([[0.2]], dtype=torch.bfloat16)     # >= near-zero threshold
    c0 = torch.tensor([[0.20390625]], dtype=torch.bfloat16)  # +1 bf16 lsb
    g = evaluate({"residual": r[:1, :1], "y": c0, "rstd": rs[:1]},
                 {"residual": r[:1, :1], "y": r0, "rstd": rs[:1]})
    o1 = g["outputs"][1]
    check("y relative-only failure", (not o1["pass"]) and o1["max_abs"] <= 0.02
          and o1["max_rel"] > 0.01, o1)

    # near-zero abs failure still counts
    g = evaluate({"residual": r[:1, :1], "y": torch.tensor([[5e-6]], dtype=torch.bfloat16),
                  "rstd": rs[:1]},
                 {"residual": r[:1, :1], "y": torch.tensor([[0.0]], dtype=torch.bfloat16),
                  "rstd": rs[:1]})
    out1 = g["outputs"][1]
    check("near-zero abs counted", out1["near_zero_count"] == 1
          and out1["near_zero_max_abs"] > 0, out1)


def restore_cases():
    class R:
        def __init__(self, rc=0, out="ok", err=""):
            self.returncode, self.stdout, self.stderr = rc, out, err

    # 1 an absent target is unavailable, never signalled and never "cleaned"
    killed = []
    with tempfile.TemporaryDirectory() as td:
        log = {}
        d0_restore.cleanup([{"pid": 999999}], log, killer=lambda k, v: killed.append((k, v)))
        check("absent target never signalled",
              not killed and len(log["cleanup"]["unavailable"]) == 1
              and not log["cleanup"]["killed"], log["cleanup"])

    # 1b pid <= 1 and self are refused
    killed.clear()
    log = {}
    d0_restore.cleanup([{"pid": 1, "starttime": "1"}, {"pid": os.getpid(), "starttime": "1"}],
                       log, killer=lambda k, v: killed.append((k, v)))
    check("cleanup refuses pid<=1 and self",
          not killed and len(log["cleanup"]["refused"]) == 2, log["cleanup"])

    # 1c identity mismatch is refused (starttime checked via a stubbed reader)
    killed.clear()
    orig_identity = d0_restore._identity
    d0_restore._identity = lambda pid: {"pid": pid, "starttime": "111", "boot_id": "b"}
    try:
        ok, why = d0_restore._verify_identity({"pid": 424242, "starttime": "222"})
    finally:
        d0_restore._identity = orig_identity
    check("starttime mismatch refused", not ok and "starttime" in why, why)

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

        good = {
            "a": '{"status":"ready"}',
            "b": "x VLLMWorker_TP\ny VLLMWorker_TP",
            "d": '{"answers":{"answer":{"type":"choice"}}}',
        }

        def runner(cmd, **k):
            calls["n"] += 1
            if calls["n"] == 3:
                return R(rc=1, out="")
            return R(rc=0, out=good[cmd[0]])

        ok = d0_restore.verify({"health": ["a"], "workers": ["b"], "text": ["c"],
                                "image": ["d"]}, log, runner=runner)
        check("verify keeps per-stage rc", ok is False
              and log["health"]["ok"] and not log["text"]["ok"], log)

    # 3b rc0 with wrong payload must not pass
    log = {}
    payloads = {"health": '{"status":"starting"}', "workers": "one VLLMWorker_TP",
                "text": '{"answers":{"answer":{"type":"choice"}}}',
                "image": '{"answers":{"answer":{"type":"score"}}}'}
    ok = d0_restore.verify({"health": ["a"], "workers": ["b"], "text": ["c"], "image": ["d"]},
                           log, runner=lambda c, **k: R(rc=0, out=payloads[c[0]]))
    check("rc0 with wrong payload fails", ok is False
          and not any(log[s]["ok"] for s in ("health", "workers", "text", "image")), log)

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
