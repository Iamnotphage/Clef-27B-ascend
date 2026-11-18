# D0 CPU regression harness

`d0_main_regression.py` is the 24-owned independent CPU regression for 21's D0
Python supervision/restore path.  It is an orchestration script plus a
per-scenario child mode; each case runs in its own process so `sys.modules` and
module-level state cannot leak between cases.

## Run

```bash
python3 -I runtime/d0_regression/d0_main_regression.py \
  --repo <21 worktree> \
  --commit 5e85681ec09b37ec9ba951bde4ca4a38615ab34e \
  --json-out /tmp/d0-suite.json \
  --freeze-dir /tmp/d0-freeze
```

The suite returns 0 only when every case passes.  `--single --scenario <name>`
runs one case; `--keep-temp` preserves the temporary archive/run directory.

## Boundaries

- Production `main`, `run_child`, `query_group`, `restore`, `verify`,
  `discover_service_pids` and `decode` are never replaced.
- Only external boundaries are mocked: `subprocess.run/Popen`, `os.kill`,
  `os.killpg`, `os.getpgid`, `socket`, and `Path`/`os.listdir` reads for
  `/workspace` and `/proc`.  Unknown argv, signals and network fail closed.
- Supervision artefacts are written by the real `run_child`.  `FakePopen`
  supplies only external rank/gate files and a simulated process handle.
- The `/proc` tree and `api.pid` are virtual; service root and the supervised
  child use different simulated PIDs.  Real writes happen only in the temporary
  run directory; writes to `/workspace` or `/proc` raise.
- The six production modules are hashed against the requested commit, and
  `spec.origin`/`__file__`/`co_filename` are checked for every loaded module.

## Evidence

`evidence/run-5642732/` and `evidence/run-5e85681/` hold the before/after suites,
per-case JSON and raw artefacts.  `evidence/MANIFEST.json` records the script
SHA, production SHAs, matrix, happy argv, module identity and failure-injection
list; `evidence/SHA256SUMS` hashes every frozen file.
