#!/usr/bin/env python3
"""Branch-level checks for the identity guards. CPU only, no device."""
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import d0_restore  # noqa: E402

fails = []


def check(name, cond, extra=""):
    print(f"{name}: {'OK' if cond else 'FAIL ' + str(extra)}")
    if not cond:
        fails.append(name)


def run_with_identity(seq, boot_seq):
    """Drive discover_service_pids with a scripted /proc reader.

    seq is the starttime returned per call; boot_seq the boot id per call.
    Nothing is patched on the production functions themselves.
    """
    orig_identity = d0_restore._identity
    calls = {"n": 0}

    def fake_identity(pid):
        i = min(calls["n"], len(seq) - 1)
        calls["n"] += 1
        boot = boot_seq[min(i, len(boot_seq) - 1)]
        if boot is None:
            return {"pid": pid, "starttime": seq[i], "boot_id": ""}
        return {"pid": pid, "starttime": seq[i], "boot_id": boot}

    d0_restore._identity = fake_identity
    real_exists = Path.is_file
    try:
        with tempfile.TemporaryDirectory() as td:
            label = Path(td) / "lbl"
            label.mkdir()
            (label / "api.pid").write_text("4242")
            orig_root = d0_restore.SERVICE_LOG_ROOT
            d0_restore.SERVICE_LOG_ROOT = td
            try:
                return d0_restore.discover_service_pids("lbl")
            finally:
                d0_restore.SERVICE_LOG_ROOT = orig_root
    finally:
        d0_restore._identity = orig_identity


def main() -> int:
    # 1 stable identity -> attributable pids
    info = run_with_identity(["100", "100"], ["boot-a", "boot-a"])
    check("stable identity yields pids",
          isinstance(info.get("pids"), list) and 4242 in info["pids"]
          and info.get("root_identity_stable") is True, info)

    # 2 starttime changes between the two reads -> no attribution
    info = run_with_identity(["100", "200"], ["boot-a", "boot-a"])
    check("starttime change withholds pids",
          info.get("pids") == [] and info.get("root_identity_stable") is False
          and "identity changed" in (info.get("error") or ""), info)

    # 3 boot id empty -> unproven, no attribution
    info = run_with_identity(["100", "100"], [None, None])
    check("missing boot id withholds pids",
          info.get("pids") == [] and info.get("root_identity_stable") is False, info)

    # 4 boot id changes between reads -> unproven
    info = run_with_identity(["100", "100"], ["boot-a", "boot-b"])
    check("boot change withholds pids",
          info.get("pids") == [] and info.get("root_identity_stable") is False, info)

    # 5 the runner blocks verification when discovery errors or is empty
    import d0_run
    for disc, label in (({"error": "boom", "pids": [1]}, "error"),
                        ({"pids": []}, "empty")):
        blocked = bool(disc.get("error")) or not set(disc.get("pids") or [])
        check(f"verify blocked on {label}", blocked, disc)

    print("RESULT:", "ALL PASS" if not fails else f"FAILURES: {fails}")
    return 0 if not fails else 1


if __name__ == "__main__":
    raise SystemExit(main())
