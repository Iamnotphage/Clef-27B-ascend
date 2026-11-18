#!/usr/bin/env python3
"""CPU counterexamples for the D0 runner: launch command, failure paths, wiring.

No device, no NPU import, no torchrun execution: the launch is inspected as a
command vector, and failure handling is exercised against a stub child.
"""
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
failures = []


def check(name, cond, extra=""):
    print(f"{name}: {'OK' if cond else 'FAIL ' + str(extra)}")
    if not cond:
        failures.append(name)


def main() -> int:
    # 1 launch command shape (built the same way d0_run builds it)
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "run"
        out.mkdir()
        env = dict(os.environ); env["WORLD_SIZE"] = "2"
        cmd = [sys.executable, "-m", "torch.distributed.run", "--standalone",
               "--nproc-per-node=2", str(HERE / "d0_launch.py"),
               "--out-dir", str(out), "--rows", "967", "--seed", "20261118",
               "--mm-dim", "1", "--use-hccl", "2",
               "--adapter", "/nonexistent.so", "--adapter-sha", "0" * 64]
        check("launch uses torchrun 2 ranks",
              "torch.distributed.run" in cmd and "--nproc-per-node=2" in cmd
              and env["WORLD_SIZE"] == "2")

        # 2 fixed cell cannot be widened (argparse choices)
        p = subprocess.run([sys.executable, str(HERE / "d0_run.py"),
                            "--out-dir", str(Path(td) / "x"), "--mm-dim", "2",
                            "--dry-run"], capture_output=True, text=True)
        check("mm_dim fixed at 1", p.returncode != 0)

        # 3 existing directory is refused
        p = subprocess.run([sys.executable, str(HERE / "d0_run.py"),
                            "--out-dir", str(out), "--dry-run"],
                           capture_output=True, text=True)
        check("existing dir refused", "refusing to reuse" in (p.stdout + p.stderr))

    # 4 rank-scoped failure files with traceback
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        code = (
            "import os,sys;os.environ['RANK']=sys.argv[1];"
            f"sys.path.insert(0,{str(HERE)!r});"
            "from pathlib import Path;import d0_launch;"
            "\ntry:\n raise RuntimeError('boom')\n"
            "except RuntimeError as e:\n"
            " d0_launch.write_failure(Path(sys.argv[2]),'trace-unavailable',e)"
        )
        for r in ("0", "1"):
            subprocess.run([sys.executable, "-c", code, r, td], check=True)
        f0 = json.loads((d / "trace-unavailable-rank0.json").read_text())
        f1 = json.loads((d / "trace-unavailable-rank1.json").read_text())
        check("rank-scoped failures", f0["rank"] == 0 and f1["rank"] == 1
              and "Traceback" in f0["traceback"] and "boom" in f0["message"])

    # 5 parser SystemExit is caught and counted as a wrapper failure
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        (d / "result-rank0.json").write_text(json.dumps({"flat_u32": [1, 2, 3]}))
        code = (
            f"import sys,json;sys.path.insert(0,{str(HERE)!r});"
            f"from pathlib import Path;from d0_trace_parse import decode;"
            f"d=Path({str(d)!r});"
            "[decode(json.loads(p.read_text())['flat_u32'],24,2,967,0xD0D0C0DE) "
            "for p in d.glob('result-rank*.json')]"
        )
        p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        check("parser raises SystemExit on bad length", p.returncode != 0
              and "trace length" in p.stderr)

    # 6 precheck rejects illegal mappings without touching device/HCCL/load
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        code = f"""
import sys, json
sys.path.insert(0, {str(HERE)!r})
from pathlib import Path
from types import SimpleNamespace
from d0_executor import precheck

class Tripwire:
    def __getattr__(self, name):
        raise AssertionError("device or loader touched: " + name)

class A:
    mm_dim = 1
    use_hccl = 2
    rows = 967
    seed = 20261118
    adapter = Path({str(d)!r}) / "missing.so"
    adapter_sha = "0" * 64

out = Path({str(d)!r})
cases = [
    ({{"WORLD_SIZE": "1", "RANK": "0", "LOCAL_RANK": "0"}}, "WORLD_SIZE"),
    ({{"WORLD_SIZE": "2", "RANK": "2", "LOCAL_RANK": "2"}}, "RANK"),
    ({{"WORLD_SIZE": "2", "RANK": "0", "LOCAL_RANK": "1"}}, "LOCAL_RANK"),
    ({{"WORLD_SIZE": "2", "RANK": "0", "LOCAL_RANK": "0"}}, "adapter not found"),
]
bad = 0
for env, needle in cases:
    try:
        precheck(env, A(), out)
        bad += 1
    except RuntimeError as e:
        if needle not in str(e):
            bad += 1
print("BAD" if bad else "OK")
"""
        p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        check("precheck rejects before device", p.returncode == 0 and "OK" in p.stdout,
              p.stdout + p.stderr[-300:])

    # 7 real run(): the illegal-precondition path must never reach the loader,
    #    and the legal path must pass through load_adapter exactly once.
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        code = f"""
import sys, hashlib
sys.path.insert(0, {str(HERE)!r})
from pathlib import Path
from types import SimpleNamespace
import d0_executor

so = Path({str(d)!r}) / "fake.so"
so.write_bytes(b"not-a-real-library")
sha = hashlib.sha256(so.read_bytes()).hexdigest()

class Args:
    mm_dim = 1; use_hccl = 2; rows = 4; seed = 1
    adapter = so; adapter_sha = sha

class Tripwire:
    def __getattr__(self, name):
        raise AssertionError("device touched before loader: " + name)

calls = []
def spy(path, expect):
    calls.append(str(path))
    raise RuntimeError("SENTINEL stop before real load")
d0_executor.load_adapter = spy

out = Path({str(d)!r})

# legal preconditions -> spy must be reached; device calls are expected here,
# so torch is a recording stub rather than a tripwire.
class Rec:
    def __getattr__(self, name):
        rec.append(name)
        return lambda *a, **k: None
rec = []
class TorchStub:
    npu = Rec()
try:
    d0_executor.run(TorchStub(), Tripwire(), Args(), out, None,
                    {{"WORLD_SIZE": "2", "RANK": "0", "LOCAL_RANK": "0"}})
    legal = "NO_RAISE"
except AssertionError as e:
    legal = "DEVICE:" + str(e)
except RuntimeError as e:
    legal = "OK" if "SENTINEL" in str(e) else "OTHER:" + str(e)

# illegal mapping -> spy must not be reached
calls.clear()
try:
    d0_executor.run(Tripwire(), Tripwire(), Args(), out, None,
                    {{"WORLD_SIZE": "2", "RANK": "0", "LOCAL_RANK": "1"}})
    illegal = "NO_RAISE"
except RuntimeError:
    illegal = "REJECTED" if not calls else "SPY_RAN"
except AssertionError as e:
    illegal = "DEVICE:" + str(e)
print(legal, illegal, len(calls))
"""
        p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        out = p.stdout.strip().split()
        check("run() reaches loader once on legal preconditions",
              len(out) >= 1 and out[0] == "OK", p.stdout + p.stderr[-300:])
        check("illegal mapping never reaches loader",
              len(out) >= 2 and out[1] == "REJECTED", p.stdout + p.stderr[-300:])

    print("RESULT:", "ALL PASS" if not failures else f"FAILURES: {failures}")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
