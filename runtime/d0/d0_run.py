#!/usr/bin/env python3
"""D0 run harness: bounded window, precise process tree, fresh output dir.

NOT executed against the device from this repository state. The sequence it
performs is fixed and must not be reordered:

    clear trace on the stream -> launch -> synchronize -> read back

The first candidate is exactly one cell: mm_dim=1 with use_hccl=2 (AR off).
Budget 2 stays a conditional item and is not encoded here.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

TRACE_SLOTS = 72
TRACE_FIELDS = 8
TRACE_BYTES = TRACE_SLOTS * TRACE_FIELDS * 4


def refuse_existing(path: Path) -> None:
    if path.exists():
        raise SystemExit(f"refusing to reuse an existing run directory: {path}")


def _read_stat(pid: str):
    try:
        raw = Path(f"/proc/{pid}/stat").read_text()
    except OSError:
        return None
    head, rest = raw.split(") ", 1)
    fields = rest.split()
    return {"comm": head.split("(")[-1], "state": fields[0], "ppid": int(fields[1]),
            "pgrp": int(fields[2]), "session": int(fields[3])}


def snapshot_tree(root_pid: int) -> dict:
    """Capture root identity plus its full descendant set.

    Each entry keeps PID, PPID, PGID and session so exclusivity claims rest on
    the tree rather than on a flat command list.
    """
    if not Path("/proc").is_dir():
        return {"root_pid": root_pid, "unavailable": "/proc not present on this host",
                "processes": {}, "descendants": []}
    procs = {}
    for pid_dir in Path("/proc").iterdir():
        if not pid_dir.name.isdigit():
            continue
        st = _read_stat(pid_dir.name)
        if st is None:
            continue
        try:
            cmd = (pid_dir / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace").strip()
        except OSError:
            cmd = ""
        st["cmdline"] = cmd
        procs[pid_dir.name] = st
    kids = {}
    for pid, st in procs.items():
        kids.setdefault(str(st["ppid"]), []).append(pid)
    desc, stack = [], [str(root_pid)]
    while stack:
        cur = stack.pop()
        for child in kids.get(cur, []):
            desc.append({"pid": int(child), **procs[child]})
            stack.append(child)
    return {"root_pid": root_pid,
            "root": procs.get(str(root_pid)),
            "captured_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "processes": procs, "descendants": desc}


def launch_cmd(a) -> list:
    """Two ranks under torchrun: env:// needs RANK, WORLD_SIZE and MASTER_ADDR,
    and the cell is defined for exactly two ranks on one card."""
    return [sys.executable, "-m", "torch.distributed.run", "--standalone",
            "--nproc-per-node=2", str(Path(__file__).with_name("d0_launch.py")),
            "--out-dir", str(a.out_dir), "--rows", str(a.rows), "--seed", str(a.seed),
            "--mm-dim", str(a.mm_dim), "--use-hccl", str(a.use_hccl),
            "--adapter", str(a.adapter), "--adapter-sha", str(a.adapter_sha)]


def _live_pids() -> set:
    """Pids alive before the restore, so new descendants can be attributed."""
    out = set()
    if not Path("/proc").is_dir():
        return out
    for d in Path("/proc").iterdir():
        if d.name.isdigit():
            out.add(int(d.name))
    return out


def render_request(kind: str) -> str:
    """Serialise the recorded request body for the text or image probe.

    Both bodies come from files already on disk; nothing is synthesised.
    """
    from d0_restore import IMAGE_SOURCE, TEXT_SAMPLE, TEXT_SOURCE
    if kind == "text":
        doc = json.loads(Path(TEXT_SOURCE).read_text())
        for sample in doc["samples"]:
            if sample["name"] == TEXT_SAMPLE:
                return json.dumps(sample["request"], ensure_ascii=False)
        raise RuntimeError(f"sample {TEXT_SAMPLE} not found in {TEXT_SOURCE}")
    if kind == "image":
        return Path(IMAGE_SOURCE).read_text()
    raise ValueError(kind)


def query_group(pgid: int) -> dict:
    """List members of a process group, or say why the listing is unavailable."""
    try:
        # Full listing then filter on the captured pgid: ps -g has different
        # meanings across implementations.
        p = subprocess.run(["ps", "-eo", "pid=,pgid="], capture_output=True,
                           text=True, timeout=10)
    except Exception as exc:                              # noqa: BLE001
        return {"survivor_check": "unavailable", "members": None,
                "query_exception": f"{type(exc).__name__}: {exc}"}
    if p.returncode != 0:
        return {"survivor_check": "unavailable", "members": None,
                "query_returncode": p.returncode, "query_stderr": p.stderr[-500:]}
    members = []
    for line in p.stdout.splitlines():
        parts = line.split()
        if len(parts) == 2 and all(x.isdigit() for x in parts) and int(parts[1]) == pgid:
            members.append(int(parts[0]))
    return {"survivor_check": "ok", "members": members, "query_returncode": 0}


def run_child(cmd: list, out_dir: Path, timeout: int, env: dict | None = None) -> int:
    """Supervise one child tree. Reused by the cpu fixture, so it takes cmd."""
    env = dict(os.environ if env is None else env)
    env.setdefault("WORLD_SIZE", "2")
    child = subprocess.Popen(cmd, start_new_session=True, env=env,
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    pgid = child.pid                                   # leader of its own session
    from d0_restore import _identity
    (out_dir / "root-child.json").write_text(json.dumps(
        {"child_pid": child.pid, "pgid": pgid,
         "starttime": (_identity(child.pid) or {}).get("starttime"),
         "boot_id": (_identity(child.pid) or {}).get("boot_id"),
         "launch_source": " ".join(cmd[:3]),
         "tree_at_start": snapshot_tree(os.getpid())}, ensure_ascii=False) + "\n")
    timed_out = False
    try:
        out, err = child.communicate(timeout=timeout)
        child_raw_rc = child.returncode
    except subprocess.TimeoutExpired:
        timed_out = True
        # The leader may already be gone while grandchildren still hold the pipe,
        # so signal the whole group and then reap with a bound of its own.
        try:
            os.killpg(pgid, signal.SIGKILL)
        except OSError:
            try:
                child.kill()
            except OSError:
                pass
        try:
            out, err = child.communicate(timeout=30)
        except subprocess.TimeoutExpired:
            out, err = "", "reap timed out after 30s; descendants may survive"
        child_raw_rc = child.returncode
        (out_dir / "trace-unavailable.json").write_text(json.dumps(
            {"reason": "timeout", "timeout_s": timeout, "pgid": pgid,
             "note": "trace could not be copied back; slot count is not recoverable"},
            ensure_ascii=False) + "\n")
    (out_dir / "raw-rc.json").write_text(json.dumps(
        {"wrapper_rc": 2 if timed_out else child_raw_rc,
         "child_raw_rc": child_raw_rc, "timed_out": timed_out, "pgid": pgid},
        ensure_ascii=False) + "\n")
    (out_dir / "stdout.log").write_text(out or "")
    (out_dir / "stderr.log").write_text(err or "")
    # Survivors are judged against the PGID actually created. A failed query is
    # reported as unavailable with members null; only a complete, successful
    # listing may yield an empty list.
    q = query_group(pgid)
    (out_dir / "tree-after.json").write_text(json.dumps(
        {"tree": snapshot_tree(os.getpid()), "pgid": pgid, **q},
        ensure_ascii=False) + "\n")
    return 2 if timed_out else child_raw_rc


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--timeout", type=int, default=180, help="bounded window in seconds")
    ap.add_argument("--rows", type=int, default=967)
    ap.add_argument("--seed", type=int, default=20261118)
    # First candidate is fixed; the CLI cannot widen it.
    ap.add_argument("--mm-dim", type=int, default=1, choices=(1,),
                    help="first candidate is fixed at 1")
    ap.add_argument("--use-hccl", type=int, default=2, choices=(2,),
                    help="first candidate is fixed at 2 (AR off)")
    ap.add_argument("--adapter", type=Path, help="prebuilt extension to load")
    ap.add_argument("--adapter-sha", help="expected sha256 of that extension")
    ap.add_argument("--dry-run", action="store_true", help="print the plan only; touches no device")
    a = ap.parse_args()

    refuse_existing(a.out_dir)
    a.out_dir.mkdir(parents=True)

    run_id = f"d0-{datetime.now().strftime('%Y%m%dT%H%M%S')}-{a.out_dir.name}"
    plan = {"run_id": run_id,
            "out_dir": str(a.out_dir),
        "cell": {"mm_dim": a.mm_dim, "use_hccl": a.use_hccl, "rows": a.rows, "seed": a.seed},
        "trace": {"slots": TRACE_SLOTS, "fields": TRACE_FIELDS, "bytes": TRACE_BYTES,
                  "container_dtype": "int32", "acl_view": "ACL_UINT32",
                  "normalisation": "readback & 0xFFFFFFFF",
                  "format": "ND", "allocations": 1},
        "launch": {"tool": "torch.distributed.run", "standalone": True,
                   "nproc_per_node": 2, "world_size": 2, "ranks": [0, 1]},
        "sequence": ["clear trace on stream", "launch", "synchronize stream", "read trace back"],
        "timeout_s": a.timeout,
        "scope": ("capacity-internal observations only; absent slot is not a missing "
                  "participant; launch set, SDK receipt, partial writes and consumer "
                  "visibility are outside this trace"),
    }
    (a.out_dir / "plan.json").write_text(json.dumps(plan, ensure_ascii=False, indent=1) + "\n")
    (a.out_dir / "process-tree.json").write_text(
        json.dumps(snapshot_tree(os.getpid()), ensure_ascii=False) + "\n")

    if a.dry_run:
        print(json.dumps({"dry_run": True, "plan": plan}, ensure_ascii=False))
        return 0

    # Real execution path. Kept explicit so the failure modes are visible:
    # a timeout kills the child and still writes a trace-unavailable record.
    # Real executor lives in its own module; this process only supervises it.
    env = dict(os.environ)
    env.update({"WORLD_SIZE": "2", "OMP_NUM_THREADS": "1"})
    test_rc = None                      # null until the child actually reports
    test_exc = None
    rc = None
    try:
        test_rc = run_child(launch_cmd(a), a.out_dir, a.timeout, env)
    except Exception as exc:                                  # noqa: BLE001
        test_exc = f"{type(exc).__name__}: {exc}"
    finally:
        # Restore runs on every exit path and keeps its own record; a restore
        # failure never overwrites the test rc.
        wrapper_rc = test_rc if isinstance(test_rc, int) else 4
        try:
            from d0_restore import (SERVICE_CONFIG, SERVICE_LAUNCH, SERVICE_WAIT,
                                    build_commands, discover_service_pids,
                                    new_service_label, restore, verify)
            label = new_service_label()          # always generated, never pinned
            launch = ["bash", SERVICE_LAUNCH[1], label]
            rlog = {"test_rc": test_rc, "test_exception": test_exc,
                    "run_id": run_id, "service_label": label,
                    "service_config": SERVICE_CONFIG}

            # The test group must be gone before a restore can start; a failed
            # or incomplete query blocks rather than launching over it.
            root = a.out_dir / "root-child.json"
            blocked = None
            pgid = None
            if root.exists():
                info = json.loads(root.read_text())
                pgid = info.get("pgid")
                if not isinstance(pgid, int) or pgid <= 1:
                    blocked = "root-child pgid missing or invalid"
                else:
                    try:
                        q = query_group(pgid)       # defined in this module
                    except Exception as exc:                 # noqa: BLE001
                        q = {"survivor_check": "unavailable",
                             "members": None, "query_exception": str(exc)}
                    rlog["prelaunch_group"] = q
                    members = q.get("members")
                    if q.get("survivor_check") != "ok":
                        blocked = "survivor query unavailable"
                    elif not isinstance(members, list):
                        blocked = "survivor members not an explicit list"
                    elif len(members) != 0:
                        blocked = f"{len(members)} members still alive"
            else:
                blocked = "missing root-child identity"
            rlog["restore_blocked"] = blocked
            if blocked:
                restored = verified = False
            else:
                before = _live_pids()
                restored = restore(launch, SERVICE_WAIT, rlog)
                disc = discover_service_pids(label, before)
                rlog["service_discovery"] = disc
                service_pids = set(disc.get("pids") or [])
                rlog["service_pids"] = sorted(service_pids)
                if disc.get("error") or not service_pids:
                    # Without a provable tree the restore cannot be verified,
                    # so it fails rather than passing on an empty set.
                    rlog["verify_blocked"] = disc.get("error") or "no attributable service pids"
                    restored = False
                verified = verify(build_commands(render_request), rlog,
                                  service_pids=service_pids) if restored else False
                rlog["verify_ok"] = verified
                rlog["restore_ok"] = restored
            rlog["restore_ok"], rlog["verify_ok"] = restored, verified
            (a.out_dir / "restore.json").write_text(
                json.dumps(rlog, ensure_ascii=False, indent=1) + "\n")
            rlog_written = True
        except Exception as exc:                              # noqa: BLE001
            (a.out_dir / "restore.json").write_text(json.dumps(
                {"test_rc": test_rc, "test_exception": test_exc,
                 "restore_ok": False, "verify_ok": False,
                 "restore_error": f"{type(exc).__name__}: {exc}"},
                ensure_ascii=False) + "\n")
            (a.out_dir / "restore-failure.json").write_text(json.dumps(
                {"failure": type(exc).__name__, "message": str(exc),
                 "traceback": __import__("traceback").format_exc(),
                 "test_rc": test_rc, "test_exception": test_exc},
                ensure_ascii=False) + "\n")


    # Wire the parser: normalise, decode, and keep raw plus parsed side by side.
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from d0_trace_parse import decode
        for res in a.out_dir.glob("result-rank*.json"):
            rec = json.loads(res.read_text())
            parsed = decode(rec["flat_u32"], aic_cap=24, sub_cap=2, m=a.rows,
                            magic=0xD0D0C0DE)
            res.with_name(res.stem + "-parsed.json").write_text(
                json.dumps(parsed, ensure_ascii=False, indent=1) + "\n")
    except (Exception, SystemExit) as exc:                    # noqa: BLE001
        # A parser refusal is a wrapper failure, not a silent pass.
        (a.out_dir / "parse-failure.json").write_text(json.dumps(
            {"failure": type(exc).__name__, "message": str(exc)},
            ensure_ascii=False) + "\n")
        rc_file = a.out_dir / "raw-rc.json"
        prev = json.loads(rc_file.read_text()) if rc_file.exists() else {}
        prev["wrapper_rc"] = 5                 # child_raw_rc is left untouched
        rc_file.write_text(json.dumps(prev, ensure_ascii=False) + "\n")
        wrapper_rc = 5
    return wrapper_rc if isinstance(wrapper_rc, int) else (test_rc or 0)


if __name__ == "__main__":
    raise SystemExit(main())
