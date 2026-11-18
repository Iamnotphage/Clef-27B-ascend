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


class R:
    """Minimal completed-process stub shared by the restore cases."""

    def __init__(self, rc=0, out="", err=""):
        self.returncode, self.stdout, self.stderr = rc, out, err


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

    # near-zero element breaching the absolute bound must fail the gate
    g = evaluate({"residual": r[:1, :1], "y": torch.tensor([[0.03]], dtype=torch.bfloat16),
                  "rstd": rs[:1]},
                 {"residual": r[:1, :1], "y": torch.tensor([[0.0]], dtype=torch.bfloat16),
                  "rstd": rs[:1]})
    out1 = g["outputs"][1]
    check("near-zero abs failure rejected", (not out1["pass"]) and (not g["pass"])
          and out1["near_zero_count"] == 1 and out1["max_abs"] > 0.02, out1)

    # reference Inf on the rstd output
    bad_rstd = rs.clone(); bad_rstd[0] = float("inf")
    g = evaluate(base, {"residual": r, "y": y, "rstd": bad_rstd})
    check("reference Inf rejected", not g["pass"] and not g["outputs"][2]["ref_finite"])

    # candidate rstd with a wrong (2-D) shape must be caught, not reshaped away
    g = evaluate({"residual": r, "y": y, "rstd": rs.reshape(1, -1)}, base)
    check("candidate rstd shape enforced", not g["pass"] and not g["outputs"][2]["shape_ok"],
          g["outputs"][2])

    # mask threshold: a reference below 1e-5 is near-zero, at/above is not.
    # bf16 cannot represent values this small precisely, so the check is driven
    # through float32 refs and the mask counter only.
    import torch as _t
    ref_zero = _t.zeros(1, 1, dtype=_t.bfloat16)
    cand_tiny = _t.tensor([[0.03]], dtype=_t.bfloat16)
    g_low = evaluate({"residual": r[:1, :1], "y": cand_tiny, "rstd": rs[:1]},
                     {"residual": r[:1, :1], "y": ref_zero, "rstd": rs[:1]})
    check("ref below threshold is near-zero",
          g_low["outputs"][1]["near_zero_count"] == 1, g_low["outputs"][1])

    # a reference well above the threshold must not be counted near-zero
    ref_big = _t.tensor([[0.5]], dtype=_t.bfloat16)
    cand_big = _t.tensor([[0.50390625]], dtype=_t.bfloat16)
    g_high = evaluate({"residual": r[:1, :1], "y": cand_big, "rstd": rs[:1]},
                      {"residual": r[:1, :1], "y": ref_big, "rstd": rs[:1]})
    check("ref above threshold not near-zero",
          g_high["outputs"][1]["near_zero_count"] == 0, g_high["outputs"][1])


def restore_cases():

    # 1 an absent target is unavailable, never signalled and never "cleaned"
    killed = []
    with tempfile.TemporaryDirectory() as td:
        log = {}
        d0_restore.cleanup([{"pid": 999999, "starttime": "1", "boot_id": "b"}], log,
                           killer=lambda k, v: killed.append((k, v)))
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

    # 1b2 a target without starttime/boot_id is refused even if the pid exists
    killed.clear()
    log = {}
    d0_restore.cleanup([{"pid": os.getppid()}], log,
                       killer=lambda k, v: killed.append((k, v)))
    check("cleanup requires identity fields",
          not killed and len(log["cleanup"]["refused"]) == 1, log["cleanup"])

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
    # 5 worker counting is scoped to the restored service pids
    log = {}
    rows = "| 7 0 | 111 | VLLMWorker_TP | x |\n| 7 1 | 222 | VLLMWorker_TP | y |\n| 7 0 | 999 | other |"
    ok = d0_restore.verify({"health": ["h"], "workers": ["w"], "text": ["t"], "image": ["i"]},
                           log, runner=lambda c, **k: R(rc=0, out={
                               "h": '{"status":"ready"}', "w": rows,
                               "t": '{"answers":{"answer":{"type":"score"}}}',
                               "i": '{"answers":{"answer":{"type":"choice"}}}'}[c[0]]),
                           service_pids={111, 222})

    check("two workers accepted", ok and log["workers"].get("workers") == 2, log["workers"])
    check("worker count scoped helper",
          d0_restore._count_workers(rows, {111}) == 1
          and d0_restore._count_workers(rows, {111, 222}) == 2, "")

    # 6 finally: restore failure does not overwrite the test rc
    with tempfile.TemporaryDirectory() as td:
        d = Path(td) / "run"
        code = f"""
import sys, json
sys.path.insert(0, {str(Path(__file__).resolve().parent)!r})
from pathlib import Path
import d0_run, d0_restore

def fake_child(cmd, out_dir, timeout, env=None):
    (Path(out_dir) / "raw-rc.json").write_text(json.dumps(
        {{"wrapper_rc": 7, "child_raw_rc": 7, "timed_out": False, "pgid": 4242}}))
    (Path(out_dir) / "root-child.json").write_text(json.dumps(
        {{"child_pid": 4242, "pgid": 4242, "starttime": "1", "boot_id": "b"}}))
    return 7
d0_run.run_child = fake_child                   # test fails with rc 7
d0_run.query_group = lambda pgid: {{"survivor_check": "ok", "members": []}}
calls = {{}}
def fake_restore(launch, wait, log, runner=None):
    calls["launch"] = launch
    log["restore"] = {{"launch_rc": 1}}       # restore also fails
    return False
d0_restore.restore = fake_restore
d0_run.restore = fake_restore
argv = ["x", "--out-dir", {str(d)!r}, "--adapter", "/tmp/a.so", "--adapter-sha", "0"*64]
sys.argv = argv
try:
    rc = d0_run.main()
except SystemExit as e:
    rc = e.code
log = json.loads((Path({str(d)!r}) / "restore.json").read_text())
raw = json.loads((Path({str(d)!r}) / "raw-rc.json").read_text())
pf = (Path({str(d)!r}) / "parse-failure.json")
print("RC", rc, "TEST", log["test_rc"], "CHILD", raw.get("child_raw_rc"),
      "OK", log["restore_ok"],
      "PARSE", (json.loads(pf.read_text())["failure"] if pf.exists() else None),
      "LAUNCH", calls.get("launch"))
"""
        p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        out = p.stdout.strip()
        check("restore failure keeps test rc",
              "TEST 7" in out and "CHILD 7" in out and "OK False" in out
              and "PARSE None" in out and "launch.sh" in out,
              out + p.stderr[-300:])

    print("RESULT:", "ALL PASS" if not fails else f"FAILURES: {fails}")
    return 0 if not fails else 1


if __name__ == "__main__":
    raise SystemExit(main())
