#!/usr/bin/env python3
"""Save a bounded, credential-filtered inventory before a profiling experiment.

Run on the A3 host after sourcing scripts/activate_env.sh. No service is stopped.
Command exit codes and unavailable probes remain visible in manifest.json.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys


PERF_ENV = {
    "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
    "TASK_QUEUE_ENABLE", "ASCEND_RT_VISIBLE_DEVICES", "ASCEND_VISIBLE_DEVICES",
    "CUDA_VISIBLE_DEVICES", "ASCEND_LAUNCH_BLOCKING", "ASCEND_GLOBAL_LOG_LEVEL",
    "HCCL_CONNECT_TIMEOUT", "HCCL_EXEC_TIMEOUT", "HCCL_OP_EXPANSION_MODE",
    "HCCL_BUFFSIZE", "HCCL_DETERMINISTIC", "VLLM_USE_V1", "VLLM_PLUGINS",
    "VLLM_WORKER_MULTIPROC_METHOD", "VLLM_ENABLE_V1_MULTIPROCESSING",
    "PYTHONHASHSEED", "PYTHONPATH", "LD_LIBRARY_PATH", "PATH",
    "ASCEND_HOME_PATH", "ASCEND_TOOLKIT_HOME", "ASCEND_OPP_PATH",
    "ASCEND_CUSTOM_OPP_PATH", "ATB_HOME_PATH", "PYTORCH_NPU_ALLOC_CONF",
    "PYTORCH_ALLOC_CONF", "TORCH_DEVICE_BACKEND_AUTOLOAD", "VIRTUAL_ENV",
    "MODEL_DIR", "XDG_CACHE_HOME", "HF_HOME", "TORCH_HOME", "TRITON_CACHE_DIR",
}


def redact(text):
    text = re.sub(r"(?i)(https?://)[^\s/@]+:[^\s/@]+@", r"\1<redacted>@", text)
    text = re.sub(r"(?i)(--(?:api-key|token|password|secret)(?:=|\s+))\S+", r"\1<redacted>", text)
    return text


def redact_argv(argv):
    result, hidden = [], False
    for arg in argv:
        if hidden:
            result.append("<redacted>")
            hidden = False
            continue
        result.append(redact(arg))
        hidden = arg.lower() in {"--api-key", "--token", "--password", "--secret"}
    return result


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def command(output, name, argv, cwd=None, timeout=120):
    record = {"argv": argv, "cwd": str(cwd) if cwd else None,
              "started_at": datetime.now(timezone.utc).isoformat()}
    try:
        result = subprocess.run(argv, cwd=cwd, capture_output=True, text=True, timeout=timeout)
        record.update(returncode=result.returncode)
        content = result.stdout + ("\nSTDERR:\n" + result.stderr if result.stderr else "")
    except (OSError, subprocess.TimeoutExpired) as exc:
        record.update(returncode=None, error=f"{type(exc).__name__}: {exc}")
        content = record["error"] + "\n"
    path = output / (name + ".txt")
    path.write_text(redact(content))
    record.update(path=path.name, sha256=sha256(path), finished_at=datetime.now(timezone.utc).isoformat())
    return record


def processes():
    result = []
    for directory in sorted(Path("/proc").glob("[0-9]*")):
        try:
            if int(directory.name) == os.getpid():
                continue
            argv = directory.joinpath("cmdline").read_bytes().decode(errors="replace").strip("\0").split("\0")
            if not any("vllm" in arg.lower() or "enginecore" in arg.lower() for arg in argv):
                continue
            environment = dict(item.split("=", 1) for item in directory.joinpath("environ").read_bytes().decode(errors="replace").split("\0") if "=" in item)
            status = directory.joinpath("status").read_text()
            result.append({"pid": int(directory.name),
                           "ppid": int(re.search(r"^PPid:\s+(\d+)", status, re.M).group(1)),
                           "argv": redact_argv(argv),
                           "cwd": str(directory.joinpath("cwd").resolve()),
                           "performance_environment": {key: redact(value) for key, value in environment.items() if key in PERF_ENV},
                           "status": status})
        except (OSError, ValueError, AttributeError):
            continue
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--vllm-source", type=Path, default=Path("/workspace/yangchen/work/src/vllm"))
    parser.add_argument("--ascend-source", type=Path, default=Path("/workspace/yangchen/work/src/vllm-ascend"))
    parser.add_argument("--sample-file", type=Path, action="append", default=[])
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    output.chmod(0o700)
    project = Path(__file__).resolve().parents[1]
    manifest = {"created_at": datetime.now(timezone.utc).isoformat(), "hostname": os.uname().nodename,
                "python": sys.executable, "project": str(project), "commands": {}, "files": {},
                "performance_environment": {key: redact(os.environ[key]) for key in sorted(PERF_ENV) if key in os.environ},
                "scope": "Pre-experiment inventory; command success and hardware availability are recorded individually."}
    for label, directory in (("project", project), ("vllm", args.vllm_source), ("vllm_ascend", args.ascend_source)):
        for suffix, git_args in (("head", ["rev-parse", "HEAD"]), ("status", ["status", "--short", "--branch"]),
                                 ("diff", ["diff", "HEAD", "--"]), ("submodules", ["submodule", "status", "--recursive"])):
            name = f"{label}-{suffix}"
            manifest["commands"][name] = command(output, name, ["git", "-C", str(directory), *git_args])
    probes = {
        "pip-freeze": [sys.executable, "-m", "pip", "freeze", "--all"],
        "pip-check": [sys.executable, "-m", "pip", "check"],
        "npu-smi": ["npu-smi", "info"], "npu-smi-mapping": ["npu-smi", "info", "-m"],
        "npu-smi-help": ["npu-smi", "info", "-h"],
        "msprof-version": ["msprof", "--version"], "msprof-help": ["msprof", "--help"],
        "disk": ["df", "-h", "/workspace/yangchen", "/tmp"], "kernel": ["uname", "-a"],
        "process-tree": ["ps", "-eo", "pid,ppid,psr,pcpu,pmem,rss,etimes,args", "--forest"],
    }
    for name, argv in probes.items():
        manifest["commands"][name] = command(output, name, argv)
    sources = {
        "vllm-profiler-config.py": args.vllm_source / "vllm/config/profiler.py",
        "vllm-profile-api-router.py": args.vllm_source / "vllm/entrypoints/serve/profile/api_router.py",
        "ascend-torch-npu-profiler.py": args.ascend_source / "vllm_ascend/profiler/torch_npu_profiler.py",
    }
    for name, path in sources.items():
        if path.is_file():
            shutil.copyfile(path, output / name)
            manifest["files"][str(path)] = {"sha256": sha256(path), "copy": name}
        else:
            manifest["files"][str(path)] = {"missing": True}
    files = [project / "Clef-27B/logs/model_manifest.json", *sorted((project / "scripts/patches").glob("*.patch")), *args.sample_file]
    files += [Path(p) for p in ("/sys/fs/cgroup/cpu.max", "/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/cpuset.cpus.effective", "/proc/meminfo", "/etc/os-release", "/usr/local/Ascend/driver/version.info")]
    files += [Path(p) for p in ("/sys/fs/cgroup/cpu,cpuacct/cpu.cfs_quota_us",
                               "/sys/fs/cgroup/cpu,cpuacct/cpu.cfs_period_us",
                               "/sys/fs/cgroup/cpu,cpuacct/cpu.stat",
                               "/sys/fs/cgroup/memory/memory.limit_in_bytes",
                               "/sys/fs/cgroup/memory/memory.max_usage_in_bytes",
                               "/sys/fs/cgroup/cpuset/cpuset.cpus")]
    msprof = shutil.which("msprof")
    if msprof:
        files.append(Path(msprof).resolve())
    for root in (Path("/usr/local/Ascend/cann"), Path("/usr/local/Ascend/nnal/atb")):
        files.extend(root.glob("**/version.cfg"))
    for path in files:
        try:
            content = path.read_bytes()
            manifest["files"][str(path)] = {"sha256": hashlib.sha256(content).hexdigest(), "bytes": len(content)}
            if len(content) < 16384 and path not in args.sample_file:
                manifest["files"][str(path)]["content"] = redact(content.decode(errors="replace"))
        except OSError as exc:
            manifest["files"][str(path)] = {"error": str(exc)}
    probe = """import importlib.metadata as m, inspect, json, torch, torch_npu
from torch_npu.profiler.profiler import analyse
print(json.dumps({'torch':torch.__version__,'torch_npu':m.version('torch-npu'),
 'torch_path':torch.__file__,'torch_npu_path':torch_npu.__file__,
 'profile_signature':str(inspect.signature(torch_npu.profiler.profile)),
 'experimental_signature':str(inspect.signature(torch_npu.profiler._ExperimentalConfig)),
 'analyse_signature':str(inspect.signature(analyse)),
 'npu_available':torch.npu.is_available(),'device_count':torch.npu.device_count()},indent=2))
"""
    manifest["commands"]["torch-profiler"] = command(output, "torch-profiler", [sys.executable, "-c", probe])
    manifest["service_processes"] = processes()
    manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
    write_json(output / "manifest.json", manifest)
    print(json.dumps({"snapshot": str(output / "manifest.json"), "process_count": len(manifest["service_processes"]),
                      "failed_probes": [name for name, row in manifest["commands"].items() if row["returncode"] != 0]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
