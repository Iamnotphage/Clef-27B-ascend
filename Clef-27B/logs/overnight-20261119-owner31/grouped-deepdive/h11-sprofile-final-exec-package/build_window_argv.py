#!/usr/bin/env python3
"""Generate supervisor argv from canonical child-argv.json (no shell, no split).

Doc-only skeleton helper; does not execute anything.
If --check-argv is supplied, asserts its --child-argv payload equals child-argv.json exactly.
"""
import argparse, json, sys
from pathlib import Path

here = Path(__file__).resolve().parent
child = json.loads((here / "child-argv.json").read_text())
ap = argparse.ArgumentParser()
ap.add_argument("--window-path", required=True)
ap.add_argument("--python", required=True)
ap.add_argument("--run-dir", required=True)
ap.add_argument("--root-identity", required=True)
ap.add_argument("--stop-script", required=True)
ap.add_argument("--expect-stop-sha", required=True)
ap.add_argument("--source-target", required=True)
ap.add_argument("--expected-source-sha", required=True)
ap.add_argument("--official-start", required=True)
ap.add_argument("--activate", required=True)
ap.add_argument("--service-root", required=True)
ap.add_argument("--model-dir", required=True)
ap.add_argument("--restore-output", required=True)
ap.add_argument("--clean-env-expect", required=True)
ap.add_argument("--text-body", required=True)
ap.add_argument("--image-body", required=True)
ap.add_argument("--port", default="18143")
ap.add_argument("--child-timeout", default="240.0")
ap.add_argument("--stop-timeout", default="120.0")
ap.add_argument("--restore-timeout", default="900.0")
ap.add_argument("--comm-timeout", default="90.0")
ap.add_argument("--master-port", default="29531")
ap.add_argument("--settle-timeout", default="300", help="builder argv choice for the supervisor (plan 300); supervisor CLI default remains 30")
ap.add_argument("--execute", action="store_true", help="append top-level --execute (execution mode)")
ap.add_argument("--check-argv", help="existing window argv JSON to equality-check")
a = ap.parse_args()

if a.check_argv:
    old = json.loads(Path(a.check_argv).read_text())
    i = old.index("--child-argv")
    assert old[i + 1] == json.dumps(child), "child-argv drift between window argv and child-argv.json"
    if "--settle-timeout" in old:
        j = old.index("--settle-timeout")
        assert old[j + 1] == a.settle_timeout, "settle-timeout drift"
    else:
        raise AssertionError("checked argv lacks --settle-timeout")
    if a.execute:
        assert old[-1] == "--execute", "checked argv lacks --execute"

argv = [a.window_path, "--python", a.python, "--run-dir", a.run_dir,
        "--root-identity", a.root_identity, "--stop-script", a.stop_script,
        "--expect-stop-sha", a.expect_stop_sha, "--child-argv", json.dumps(child),
        "--source-target", a.source_target, "--expected-source-sha", a.expected_source_sha,
        "--official-start", a.official_start, "--activate", a.activate,
        "--service-root", a.service_root, "--model-dir", a.model_dir,
        "--restore-output", a.restore_output, "--clean-env-expect", a.clean_env_expect,
        "--text-body", a.text_body, "--image-body", a.image_body,
        "--port", a.port, "--child-timeout", a.child_timeout,
        "--stop-timeout", a.stop_timeout, "--restore-timeout", a.restore_timeout,
        "--comm-timeout", a.comm_timeout, "--master-port", a.master_port,
        "--settle-timeout", a.settle_timeout]
if a.execute:
    argv.append("--execute")
print(json.dumps(argv, indent=1))
