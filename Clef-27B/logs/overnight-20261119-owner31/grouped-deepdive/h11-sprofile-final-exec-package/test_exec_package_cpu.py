#!/usr/bin/env python3
"""CPU substitute tests for the final wrapper. No real preflight/torch/msprof/device/service."""
import json, os, shutil, subprocess, sys, tempfile, textwrap, unittest
from pathlib import Path

PKG = Path(__file__).resolve().parent
WRAPPER = PKG / "run-h11-sprofile-final.sh"
CHILD = json.loads((PKG / "child-argv.json").read_text())

FAKE_CAPTURE = '''#!/usr/bin/env python3
import argparse, json, os
from pathlib import Path
a=argparse.ArgumentParser(); a.add_argument("--pid",type=int); a.add_argument("--out")
x=a.parse_args(); out=Path(x.out)
out.parent.mkdir(parents=True,exist_ok=True)
out.write_text(json.dumps({"pid":x.pid,"ppid":1,"pgid":x.pid,"sid":x.pid,"starttime":"1","state":"S","boot_id":"fake","cmdline":"fake"})+"\\n")
open(os.environ["TRACE"],"a").write("capture\\n")
'''
FAKE_SIDECAR = '''#!/usr/bin/env python3
import json,os,sys
open(sys.argv[2],"w").write("{}\\n")
open(os.environ["TRACE"],"a").write("sidecar\\n")
'''
FAKE_PREFLIGHT = '''#!/bin/bash
echo "preflight" >> "$TRACE"
echo "RESULT: ${PREFLIGHT_RESULT:-ALL PASS}"
exit ${PREFLIGHT_RC:-0}
'''
FAKE_COLLECT = '''#!/bin/bash
phase=${1:-}
echo "$phase" >> "$TRACE"
if [ "$phase" = begin ]; then exit ${BEGIN_RC:-0}; fi
if [ "$phase" = end ]; then
  if [ "${END_RC:-0}" -eq 0 ]; then echo COMPLETE > "$ARCH_COMPLETE"; else echo INCOMPLETE > "$ARCH_INCOMPLETE"; fi
  exit ${END_RC:-0}
fi
exit 9
'''
FAKE_WINDOW = '''#!/usr/bin/env python3
import json,os,sys
open(os.environ["TRACE"],"a").write("window\\n")
open(os.environ["WINDOW_ARGV"],"w").write(json.dumps(sys.argv))
sys.exit(int(os.environ.get("WINDOW_RC","0")))
'''
FAKE_TORCHRUN = '''#!/bin/bash
echo "torchrun $*" >> "$TRACE"
exit 0
'''

class Base(unittest.TestCase):
    def setUp(self):
        self.td = Path(tempfile.mkdtemp(prefix="h11-exec-cpu-"))
        self.addCleanup(shutil.rmtree, self.td, True)
        self.cell = self.td / "cell"; self.cell.mkdir()
        self.fake = self.td / "fake"; self.fake.mkdir()
        self.trace = self.td / "trace"
        self.window_argv = self.td / "window-argv.json"
        self.venv = self.td / "venv"; self.venv.mkdir()
        os.symlink(sys.executable, self.venv / "python")
        sys.path.insert(0, str(self.fake))
        self.env = {
            "PATH": f"{self.fake}:/usr/bin:/bin", "HOME": str(self.td),
            "H11_CELL": str(self.cell), "H11_CODE": str(PKG),
            "H11_VENV": str(self.venv),
            "H11_TORCHRUN": str(self.fake / "torchrun"),
            "H11_CAPTURE": str(self.fake / "capture.py"),
            "H11_SIDECAR": str(self.fake / "sidecar.py"),
            "H11_PREFLIGHT": str(self.fake / "preflight.sh"),
            "H11_COLLECT": str(self.fake / "collect.sh"),
            "H11_BUILDER": str(PKG / "build_window_argv.py"),
            "H11_CHILD": str(PKG / "child-argv.json"),
            "H11_WINDOW": str(self.fake / "window.py"),
            "H11_RUN_WINDOW": str(PKG / "run_window.py"),
            "H11_ROOT_PID": "12345", "TRACE": str(self.trace),
            "WINDOW_ARGV": str(self.window_argv),
            "ARCH_COMPLETE": str(self.cell / "COMPLETE"),
            "ARCH_INCOMPLETE": str(self.cell / "INCOMPLETE"),
        }
        for name, body in (("capture.py", FAKE_CAPTURE), ("sidecar.py", FAKE_SIDECAR),
                           ("preflight.sh", FAKE_PREFLIGHT), ("collect.sh", FAKE_COLLECT),
                           ("window.py", FAKE_WINDOW), ("torchrun", FAKE_TORCHRUN)):
            p = self.fake / name; p.write_text(body); p.chmod(0o755)

    def run_wrapper(self, **over):
        env = dict(self.env); env.update({k: str(v) for k, v in over.items()})
        return subprocess.run(["bash", str(WRAPPER)], env=env, capture_output=True, text=True, timeout=30)

    def marks(self):
        return self.trace.read_text().split() if self.trace.exists() else []

class Gates(Base):
    def test_caller_failure_blocks_all_gates(self):
        r = self.run_wrapper(H11_TORCHRUN=str(self.fake / "missing-torchrun"))
        self.assertEqual(r.returncode, 3, r.stderr)
        self.assertNotIn("preflight", self.marks()); self.assertNotIn("begin", self.marks()); self.assertNotIn("window", self.marks())

    def test_preflight_failure_blocks_begin_window(self):
        r = self.run_wrapper(PREFLIGHT_RC="1")
        self.assertEqual(r.returncode, 5, r.stderr)
        self.assertIn("preflight", self.marks()); self.assertNotIn("begin", self.marks()); self.assertNotIn("window", self.marks())

    def test_collect_begin_failure_exits6_without_window_or_end(self):
        r = self.run_wrapper(BEGIN_RC="1")
        self.assertEqual(r.returncode, 6, r.stderr)
        self.assertIn("begin", self.marks()); self.assertNotIn("window", self.marks()); self.assertNotIn("end", self.marks())

class Lifecycle(Base):
    def test_success_one_window_one_end_and_canonical_argv(self):
        r = self.run_wrapper()
        self.assertEqual(r.returncode, 0, r.stderr)
        m = self.marks()
        self.assertEqual(m.count("window"), 1); self.assertEqual(m.count("end"), 1)
        self.assertLess(m.index("preflight"), m.index("begin")); self.assertLess(m.index("begin"), m.index("window"))
        argv = json.loads(self.window_argv.read_text())
        i = argv.index("--child-argv")
        self.assertEqual(json.loads(argv[i + 1]), CHILD)
        j = argv.index("--settle-timeout"); self.assertEqual(argv[j + 1], "300")
        self.assertEqual(argv[-1], "--execute")
        self.assertTrue((self.cell / "COMPLETE").exists())

    def test_window_failure_still_one_end_and_partial(self):
        r = self.run_wrapper(WINDOW_RC="7")
        self.assertEqual(r.returncode, 7, r.stderr)
        m = self.marks(); self.assertEqual(m.count("window"), 1); self.assertEqual(m.count("end"), 1)
        self.assertTrue((self.cell / "COMPLETE").exists())

    def test_end_failure_preserves_window_rc_and_marks_incomplete(self):
        r = self.run_wrapper(END_RC="1")
        self.assertEqual(r.returncode, 1, r.stderr)
        m = self.marks(); self.assertEqual(m.count("window"), 1); self.assertEqual(m.count("end"), 1)
        self.assertTrue((self.cell / "INCOMPLETE").exists()); self.assertFalse((self.cell / "COMPLETE").exists())

class CollectCopy(Base):
    def test_derived_collect_rejects_second_end(self):
        arch = self.cell / "archive"; (arch / "meta").mkdir(parents=True); (arch / "meta" / "END").write_text("x")
        env = dict(os.environ); env.update({"CELL": str(self.cell), "ARCH": str(arch)})
        r = subprocess.run(["bash", str(PKG / "collect-raw-final.sh"), "end"], env=env, capture_output=True, text=True, timeout=10)
        self.assertEqual(r.returncode, 7, r.stderr)

class PathsAndArchive(Base):
    def test_child_and_plan_paths_are_final_cell(self):
        plan = json.loads((PKG / "plan-h11-sprofile-final.json").read_text())
        self.assertEqual(CHILD, plan["child_argv"])
        cell = "/workspace/yangchen/work/grouped-deepdive-cell-h11-sprofile-final"
        for group in (CHILD, plan["child_argv"]):
            for p in group:
                if "grouped-deepdive-cell-h11-sprofile" in p:
                    self.assertTrue(p.startswith(cell + "/"))
                    self.assertNotIn("h11-sprofile-v1", p)
        self.assertEqual(CHILD[CHILD.index("--output") + 1], cell + "/msprof-out-h11-sprofile-final")
        self.assertEqual(CHILD[CHILD.index("--temp-root") + 1], cell + "/probe-temp-h11-sprofile-final")
        self.assertEqual(CHILD[CHILD.index("--out") + 1], cell + "/probe-out-h11-sprofile-final")

    def _archive_fixture(self, with_msprof=True):
        cell = self.cell; arch = cell / "archive"; meta = arch / "meta"; meta.mkdir(parents=True)
        ident = {"pid": 123, "starttime": "1", "boot_id": "b"}
        (cell / "src-id.json").write_text(json.dumps(ident))
        (meta / "identity.json").write_text(json.dumps(ident))
        for d in (arch / "cell" / n for n in ("run", "probe-temp", "probe-out", "restore", "msprof-out")):
            d.mkdir(parents=True, exist_ok=True)
        dirs = {"run": cell / "run-h11-sprofile-final", "temp": cell / "probe-temp-h11-sprofile-final",
                "out": cell / "probe-out-h11-sprofile-final", "restore": cell / "restore-h11-sprofile-final",
                "logs": cell / "logs-h11-sprofile-final"}
        if with_msprof:
            dirs["msprof"] = cell / "msprof-out-h11-sprofile-final"
        for d in dirs.values():
            d.mkdir(parents=True, exist_ok=True)
            (d / "marker.txt").write_text("x")
        srcs = {}
        for n in ("dev0", "dev1", "plog", "vllm"):
            d = cell / ("src-" + n); d.mkdir(); (d / "m.txt").write_text("x"); srcs[n] = d
        bodies = cell / "bodies"; bodies.mkdir()
        (bodies / "text-body.json").write_text("{}"); (bodies / "image-body.json").write_text("{}")
        (cell / "sidecar.json").write_text("{}"); (cell / "plan.json").write_text(json.dumps({"status": "plan_not_executed"}))
        start = meta / "START_SLACK"; start.write_text(""); os.utime(start, (1000, 1000))
        end = meta / "END_SLACK"; end.write_text(""); os.utime(end, (3000, 3000))
        for f in list(cell.rglob("marker.txt")) + list(cell.rglob("m.txt")):
            os.utime(f, (2000, 2000))
        env = dict(os.environ)
        env.update({"CELL": str(cell), "ARCH": str(arch), "SRC_ID": str(cell / "src-id.json"),
                    "SRC_SIDECAR": str(cell / "sidecar.json"), "SRC_PLAN": str(cell / "plan.json"),
                    "BODIES": str(bodies), "SRC_DEV0": str(srcs["dev0"]), "SRC_DEV1": str(srcs["dev1"]),
                    "SRC_PLOG": str(srcs["plog"]), "SRC_VLLM": str(srcs["vllm"])})
        return env, arch

    def test_collect_archives_msprof_out(self):
        env, arch = self._archive_fixture(True)
        r = subprocess.run(["bash", str(PKG / "collect-raw-final.sh"), "end"], env=env, capture_output=True, text=True, timeout=20)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertTrue((arch / "cell" / "msprof-out" / "marker.txt").exists())
        self.assertIn("cell/msprof-out", (arch / "meta" / "sources.txt").read_text())
        self.assertIn("cell/msprof-out/marker.txt", (arch / "SHA256SUMS").read_text())

    def test_collect_missing_msprof_out_marks_incomplete(self):
        env, arch = self._archive_fixture(False)
        r = subprocess.run(["bash", str(PKG / "collect-raw-final.sh"), "end"], env=env, capture_output=True, text=True, timeout=20)
        self.assertEqual(r.returncode, 1, r.stderr)
        self.assertTrue((arch / "meta" / "INCOMPLETE").exists())
        self.assertTrue((arch / "cell" / "run" / "marker.txt").exists())
        self.assertTrue((arch / "cell" / "probe-temp" / "marker.txt").exists())

if __name__ == '__main__':
    unittest.main(verbosity=2)
