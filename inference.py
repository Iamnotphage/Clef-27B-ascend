"""Launch the pinned vLLM-Ascend structured-decisions service.

The competition entry is ``bash start.sh --model-dir <dir> --output-dir <dir>``.
``--output-dir`` receives the launch manifest, the service log and runtime
artifacts. Extra vLLM arguments go after ``--``; both ``--flag value`` and
``--flag=value`` are accepted, and a passthrough value overrides the launcher
default for the same flag. The manifest records the effective value and its
source for every configurable flag.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import types

REPO_ROOT = Path(__file__).resolve().parent
# 运行根默认在项目内（source scripts/activate_env.sh 会设置 MODEL_DIR）；
# 不再隐式依赖 /workspace/yangchen。
DEFAULT_MODEL_DIR = str(REPO_ROOT / ".build" / "models" / "Clef-27B")
SERVED_MODEL_NAME = "clef"
PROTOCOL_VERSION = "2026-11-13"


_LEVEL_COLORS = {
    "DEBUG": "\033[36m",
    "INFO": "\033[32m",
    "WARNING": "\033[33m",
    "ERROR": "\033[31m",
}
_RESET = "\033[0m"


def _startup_color_enabled():
    """ANSI color only on an interactive terminal unless explicitly overridden."""
    if os.environ.get("NO_COLOR") is not None:
        return False
    override = os.environ.get("CLEF_LOG_COLOR")
    if override is not None:
        return override.strip().lower() not in ("0", "false", "no", "off")
    return sys.stdout.isatty() and os.environ.get("TERM", "") != "dumb"


def _start_startup_progress(log_path, host, port):
    """Spawn the terminal-only model-load progress watchdog.

    The helper inherits the current stdout (a TTY), tails the vLLM shard
    progress in service.log, and exits when /health is ready.  It is skipped
    for non-interactive runs so redirected logs stay clean.
    """
    if not sys.stdout.isatty():
        return
    setting = os.environ.get("CLEF_STARTUP_PROGRESS", "1").strip().lower()
    if setting in ("0", "false", "no", "off"):
        return
    script = RUNTIME_DIR / "startup_progress.py"
    if not script.is_file():
        return
    try:
        subprocess.Popen(
            [sys.executable, str(script), "--log", str(log_path),
             "--host", host, "--port", str(port), "--pid", str(os.getpid())],
            stdout=None, stderr=subprocess.DEVNULL, close_fds=True)
    except OSError:
        pass


def startup_log(message, level="INFO"):
    """Write one ISO-8601 UTC timestamped startup line to the terminal.

    Format: ``<timestamp> <level> clef.launcher: <message>`` (RFC 3339 time,
    syslog-like level and logger name).  The service later redirects stdout
    and stderr to <output-dir>/service.log, so these lines are the operator's
    terminal-side confirmation that startup has begun.  Colors are emitted
    only for a TTY; set NO_COLOR=1 (or CLEF_LOG_COLOR=0) to force plain text.
    """
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
    if not _startup_color_enabled():
        print(f"{stamp} {level:<5} clef.launcher: {message}", flush=True)
        return
    color = _LEVEL_COLORS.get(level.upper(), "")
    print(
        f"\033[2m{stamp}{_RESET} {color}{level:<5}{_RESET} "
        f"\033[36mclef.launcher:{_RESET} {message}",
        flush=True,
    )
RUNTIME_DIR = REPO_ROOT / "src" / "runtime"
GDN_INSTALLER = RUNTIME_DIR / "apply_gdn_decision_patch.py"
QKV_INSTALLER = RUNTIME_DIR / "apply_qkv_norm_patch.py"
CONV_QKV_INSTALLER = RUNTIME_DIR / "apply_conv_qkv_patch.py"
CONV_QKV_VARLEN_INSTALLER = RUNTIME_DIR / "apply_conv_qkv_varlen_patch.py"
CONV_QKV_TARGET = Path("vllm_ascend/ops/gdn.py")
CONV_QKV_PREIMAGE = "c8c0d4d8a693658e075a2232b6938584760672bacf101bf50ca5fdfab2c0be7e"
CONV_QKV_POSTIMAGE = "ead25cb2691cb9ab1c322353fc8e168f8bfc9c23b9e4d53de01c762943027e78"
CONV_QKV_VARLEN_POSTIMAGE = "afe6d4e87c0eb47bfb325fb630620e47f845fb5a893cb1dada78069ad4e3ca07"
JOINT_INSTALLER = RUNTIME_DIR / "apply_joint_single_field_patch.py"
JOINT_FUSED_INSTALLER = RUNTIME_DIR / "apply_joint_fused_attention.py"
JOINT_FUSED_GUARD = RUNTIME_DIR / "apply_joint_single_field_patch_fused.py"
JOINT_SERVICE_CHECK_INSTALLER = RUNTIME_DIR / "apply_joint_service_check.py"
CHUNK128_INSTALLER = RUNTIME_DIR / "apply_gdn_chunk128_patch.py"
ADAPTIVE_CHUNK_INSTALLER = RUNTIME_DIR / "apply_gdn_adaptive_chunk_patch.py"
SYSTEMONE_MM_INSTALLER = RUNTIME_DIR / "apply_systemone_mm_state_patch.py"
SYSTEMONE_MM_MODULE = RUNTIME_DIR / "systemone_mm_state.py"
SYSTEMONE_MM_MODE_ENV = "CLEF_STATE_MM_MODE"
TIERED_INSTALLER = RUNTIME_DIR / "apply_tiered_cache_patch.py"
TIERED_MODEL_RUNNER_TARGET = Path("vllm_ascend/worker/v2/model_runner.py")
TIERED_MODEL_RUNNER_PREIMAGE = "e866aeb27d04afe2141850122c03343c38d2a9b14cf1f9f136866003b35968a6"
TIERED_MODEL_RUNNER_POSTIMAGE = "a772213f322392608d9204245f301d37e700b799475377f0dade206f887c5f73"
TIERED_API_ROUTER_TARGET = Path(
    "vllm/entrypoints/generate/structured_decisions/api_router.py")
TIERED_API_ROUTER_PREIMAGE = "bdf11bee92c92a7caf6070321eb2294b833f0e19be6ca9f3e90b7790c34ddc9f"
TIERED_API_ROUTER_POSTIMAGE = "1875f0e1ff4ffa5e587b82e5755e743a72c148d53581a576b34ee1fc5cfd1b42"
TIERED_RANK_BYTES_DEFAULT = str(10 << 30)
TIERED2A_RANK_BYTES_DEFAULT = str(28 << 30)
TIERED_MULTI_RANK_BYTES = 38 << 30
TIERED_RING_SLOTS_DEFAULT = 8
TIERED_RAM_FULL = 112 << 30
TIERED_RAM_DISABLE = 48 << 30
#: Phase-4 ERQA A/B locked arm: A1 (extract_plain) 213/400 vs
#: A2 (extract_question_dict) 216/400, so the official entry defaults to A2.
SYSTEMONE_MM_DEFAULT_MODE = "extract_question_dict"
DEFAULT_CONFIG = "s1"
JOINT_ORIGINAL_SHA = "197863d5019f8b5326421f7616fbda13b030c27023248bf130628af7722bf4be"
JOINT_SINGLE_FIELD_SHA = "f8101b98228bbd472a63a1efe6827583e35c6cc9dede4379b925f2b016670863"
JOINT_FUSED_SHA = "26d3ad047c42f47a7fa22e00d6bac53a8a063ed010e753cb79a673660f60d7c4"
JOINT_LEGACY_FUSED_SHA = "aa3e17585f8bd60a4d05e342abc7e2773c79a0b2d49f0370d15e29bcc6b72503"
JOINT_GUARD_PRE_SHA = "76595657d312699b4b1564d1bc054229808cd775c16702fb0e36fcdb1f9706d9"
JOINT_GUARD_POST_SHA = "fdc9dde620a7995faede928fd97edeb3a37cae5de55ca28eea5b17ef4fd6da05"
JOINT_FLAG_ENV = "CLEF_JOINT_FUSED_ATTN"
JOINT_ALLOW_ENV = "CLEF_JOINT_ALLOWED_POSTIMAGE_SHA"
MM_PROCESSOR_KWARGS = (
    '{"images_kwargs":{"size":{"shortest_edge":65536,"longest_edge":1048576}},'
    '"videos_kwargs":{"size":{"shortest_edge":4096,"longest_edge":65536},'
    '"do_sample_frames":false}}'
)

CONFIGS = {
    "custom": {},
    "s1": {
        "tensor_parallel_size": 2,
        # One resident service serves OpenDecisions (1,595 tokens max) and
        # ERQA (16 images / 10,294 tokens max): 16K context, an image bound of
        # 16, and the reference image pixel range.  Tier 1 encoder retention
        # (262,144 embeddings) is the Worker 1 phase-0 verified setting.
        "max_model_len": 16384,
        "max_num_seqs": 16,
        "max_num_batched_tokens": 16384,
        "gpu_memory_utilization": 0.85,
        "limit_mm_per_prompt": '{"image":16,"video":0}',
        "encoder_cache_tokens": 262144,
        "schema_token_cache": "off",
        "prefix_state_cache": True,
        "joint_single_field": True,
        "joint_fused_attention": True,
        "mm_processor_kwargs": MM_PROCESSOR_KWARGS,
        "gdn_data_path": {
            "bsnd": True,
            "skip_initial_state": True,
            "skip_conv_state": True,
            "preprocess_fusion": True,
        },
    },
}

DATA_PATH_ENV = {
    "bsnd": "CLEF_GDN_BSND",
    "skip_initial_state": "CLEF_GDN_SKIP_INITIAL_STATE",
    "skip_conv_state": "CLEF_GDN_SKIP_CONV_STATE",
    "preprocess_fusion": "CLEF_GDN_PREPROCESS_FUSION",
}

# Launcher-owned value flags that a passthrough argument may override.
OVERRIDABLE = {
    "--host": "host",
    "--port": "port",
    "--tensor-parallel-size": "tensor_parallel_size",
    "--max-model-len": "max_model_len",
    "--max-num-seqs": "max_num_seqs",
    "--max-num-batched-tokens": "max_num_batched_tokens",
    "--gpu-memory-utilization": "gpu_memory_utilization",
    "--limit-mm-per-prompt": "limit_mm_per_prompt",
    "--mm-processor-kwargs": "mm_processor_kwargs",
    "--distributed-executor-backend": "distributed_executor_backend",
}
# Fixed flags: passing the same flag through is harmless only with the same
# value; a conflicting value is rejected.
FIXED_FLAGS = {
    "--served-model-name": SERVED_MODEL_NAME,
    "--runner": "pooling",
    "--dtype": "bfloat16",
    "--enforce-eager": None,
    "--no-enable-prefix-caching": None,
    "--no-enable-chunked-prefill": None,
    "--skip-mm-profiling": None,
}
ENV_WHITELIST_PREFIXES = ("CLEF_", "ASCEND_", "ATB_", "HCCL_")
ENV_WHITELIST_KEYS = (
    "PATH", "LD_LIBRARY_PATH", "PYTHONPATH", "PYTHONUNBUFFERED", "VIRTUAL_ENV",
    "OMP_NUM_THREADS", "TASK_QUEUE_ENABLE", "MODEL_DIR", "HF_HOME",
    "TORCH_HOME", "TRITON_CACHE_DIR", "XDG_CACHE_HOME",
)


def sha256_bytes(blob):
    return hashlib.sha256(blob).hexdigest()


def sha256_file(path):
    return sha256_bytes(Path(path).read_bytes())


def load_python_contract(path, name):
    """Load a checked-in module's constants without writing __pycache__."""
    module = types.ModuleType(name)
    module.__file__ = str(path)
    sys.modules[name] = module
    exec(compile(Path(path).read_bytes(), str(path), "exec"), module.__dict__)
    return module


def systemone_mm_mode():
    """Validate and return the effective multimodal-state mode."""
    module = load_python_contract(SYSTEMONE_MM_MODULE, "clef_systemone_mm_state")
    raw = os.environ.get(SYSTEMONE_MM_MODE_ENV)
    return module.resolve_mode(
        raw if raw is not None and raw.strip() else SYSTEMONE_MM_DEFAULT_MODE)


def write_json_atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=1) + "\n")
    os.replace(temporary, path)


def installer_call(argv, env=None):
    """Run one audited installer and keep its report even on a nonzero rc."""
    result = subprocess.run(argv, capture_output=True, text=True, env=env)
    try:
        report = json.loads(result.stdout)
    except (TypeError, ValueError):
        report = {"raw_stdout": result.stdout[-2000:], "raw_stderr": result.stderr[-2000:]}
    return {"rc": result.returncode, "report": report,
            "stdout": result.stdout[-2000:], "stderr": result.stderr[-2000:]}


def resolve_vllm_tree(timeout=60):
    """Resolve the writable vllm source tree in an isolated subprocess.

    The launcher process must not import vllm/torch before ``execv``; the probe
    uses the same interpreter and environment, has a short timeout, writes
    nothing, and fails closed on any unusable layout (editable, zip, namespace,
    missing target or non-writable target).
    """
    probe = [sys.executable, "-c",
             "import vllm, pathlib; print(pathlib.Path(vllm.__file__).resolve())"]
    try:
        result = subprocess.run(probe, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ValueError(f"vllm tree probe failed: {type(exc).__name__}: {exc}")
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if result.returncode != 0 or not lines:
        raise ValueError(
            f"vllm tree probe rc={result.returncode} "
            f"stdout={result.stdout[-500:]!r} stderr={result.stderr[-500:]!r}")
    init = Path(lines[-1]).expanduser()
    if init.name != "__init__.py" or init.parent.name != "vllm":
        raise ValueError(f"vllm probe did not name the package __init__: {init}")
    tree = init.parent.parent.resolve()
    target = tree / "vllm/model_executor/models/clef_joint_schema.py"
    if not target.is_file():
        raise ValueError(f"vllm joint schema target missing: {target}")
    # root 会绕过 0444 权限，os.access 不能反映“声明为只读”的意图；
    # 同时看 mode 位与 os.access（后者覆盖只读挂载/ACL）。
    def _writable(path):
        try:
            mode = path.stat().st_mode
        except OSError:
            return False
        return bool(mode & 0o222) and os.access(path, os.W_OK)

    if not _writable(target) or not _writable(target.parent):
        raise ValueError(f"vllm joint schema target is not writable: {target}")
    return {"tree": str(tree), "target": str(target), "probe_rc": result.returncode,
            "probe_stdout": result.stdout[-500:], "probe_stderr": result.stderr[-500:]}


def resolve_package_tree(package_name):
    """Return the parent directory of an installed package, or None if unavailable."""
    spec = importlib.util.find_spec(package_name)
    init = Path(spec.origin).resolve() if spec and spec.origin else None
    if init is None or init.name != "__init__.py" or init.parent.name != package_name:
        return None
    return init.parent.parent


def model_files(model_dir):
    """Check both the joint head and every shard named by the backbone index."""
    names = {
        "config.json",
        "tokenizer_config.json",
        "joint_head_config.json",
        "joint_head.safetensors",
        "model.safetensors.index.json",
    }
    index = json.loads((model_dir / "model.safetensors.index.json").read_text())
    shards = set(index["weight_map"].values())
    if not shards:
        raise ValueError("主干权重索引为空")
    for shard in shards:
        path = (model_dir / shard).resolve()
        if not path.is_relative_to(model_dir):
            raise ValueError(f"权重索引包含模型目录外的路径：{shard}")
    names.update(shards)
    missing = sorted(
        name for name in names
        if not (model_dir / name).is_file() or (model_dir / name).stat().st_size == 0
    )
    if missing:
        raise ValueError(f"模型文件缺失或为空：{', '.join(missing)}")
    return sorted(names)


def model_revision(model_dir):
    """Deterministic local revision from the files that define the model."""
    parts = []
    for name in ("config.json", "joint_head_config.json", "model.safetensors.index.json"):
        path = model_dir / name
        if path.is_file():
            parts.append(f"{name}:{sha256_bytes(path.read_bytes())}")
    return "local-" + sha256_bytes("\n".join(parts).encode())[:12]


def software_provenance():
    """Best-effort source-tree identity of the pinned vLLM / vLLM-Ascend."""
    provenance = {}
    for name in ("vllm", "vllm_ascend"):
        entry = {"origin": "unavailable"}
        try:
            spec = importlib.util.find_spec(name)
            if spec and spec.submodule_search_locations:
                pkg_dir = Path(list(spec.submodule_search_locations)[0])
                entry["package_dir"] = str(pkg_dir)
                source_root = pkg_dir.parent
                marker = source_root / ".clef-source-commit"
                try:
                    if (source_root / ".git").exists():
                        head = subprocess.run(
                            ["git", "-C", str(source_root), "rev-parse", "HEAD"],
                            capture_output=True, text=True, timeout=10)
                        dirty = subprocess.run(
                            ["git", "-C", str(source_root), "status", "--porcelain"],
                            capture_output=True, text=True, timeout=10)
                        if head.returncode == 0:
                            entry["git_head"] = head.stdout.strip()
                            entry["worktree_dirty"] = bool(dirty.stdout.strip())
                            entry["dirty_files"] = len(dirty.stdout.splitlines())
                    elif marker.is_file():
                        # scripts/setup_env.sh archive 回退：固定提交 tarball 无 .git。
                        entry["git_head"] = marker.read_text(encoding="utf-8").strip()
                        entry["head_source"] = "marker"
                except OSError:
                    pass
                if name == "vllm":
                    strategies = pkg_dir / "entrypoints" / "openai" / "decisions" / "strategies.py"
                    if strategies.is_file():
                        entry["strategies_sha256"] = sha256_bytes(strategies.read_bytes())
                        entry["schema_cache_version"] = schema_cache_version(strategies)
        except (ImportError, AttributeError, OSError) as exc:
            entry["error"] = repr(exc)
        provenance[name] = entry
    return provenance


def gdn_decision_prefill(config, apply=True, output_dir=None):
    """Install and verify the decision-only GDN prefill patch.

    Enabled by default for the submission config ``s1``; set
    ``CLEF_GDN_DECISION_PREFILL=0`` to turn it off.  The installer refuses a
    tree whose preimage does not match, which is how the existing BSND patch
    dependency is enforced: the BSND patch must be applied first.
    """
    requested_value = os.environ.get("CLEF_GDN_DECISION_PREFILL")
    if requested_value is not None:
        requested = requested_value.strip().lower() not in ("", "0", "false", "no", "off")
        source = "env"
    else:
        requested = config == "s1"
        source = f"config:{config}" if requested else "config"
    record = {
        "enabled": False,
        "requested": requested,
        "source": source,
        "installer": str(GDN_INSTALLER),
        "installer_sha256": (
            sha256_bytes(GDN_INSTALLER.read_bytes()) if GDN_INSTALLER.is_file() else None),
    }
    if not requested:
        return record
    if not GDN_INSTALLER.is_file():
        raise SystemExit(f"GDN decision patch installer missing: {GDN_INSTALLER}")
    spec = importlib.util.spec_from_file_location("clef_gdn_installer", GDN_INSTALLER)
    contract = importlib.util.module_from_spec(spec)
    # Load the version contract without producing __pycache__ during dry-run.
    exec(compile(GDN_INSTALLER.read_bytes(), str(GDN_INSTALLER), "exec"), contract.__dict__)

    def run(flag):
        argv = [sys.executable, str(GDN_INSTALLER)]
        tree = resolve_package_tree("vllm_ascend")
        if tree is not None:
            argv += ["--tree", str(tree)]
        if output_dir is not None:
            argv += ["--backup-dir",
                     str(Path(output_dir).expanduser().resolve() / "patch-backup" / "gdn-decision")]
        argv.append(flag)
        result = subprocess.run(argv, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            raise SystemExit(
                f"GDN decision patch {flag} failed:\n"
                f"{result.stdout[-2000:]}{result.stderr[-2000:]}")
        return json.loads(result.stdout)

    def verify(report):
        files = report.get("files", {})
        if (report.get("patch_version") != contract.PATCH_VERSION
                or set(files) != set(contract.POSTIMAGE_SHA)):
            raise SystemExit("GDN decision patch report has an unsupported version or file set")
        statuses = {}
        for path, expected in contract.POSTIMAGE_SHA.items():
            item = files[path]
            state = item.get("status_before")
            allowed = {
                "patched": expected,
                "preimage": contract.PREIMAGE_SHA[path],
            }
            if path in contract.QKV_POSTIMAGE_SHA:
                allowed["qkv"] = contract.QKV_POSTIMAGE_SHA[path]
            if path in contract.CONV_QKV_POSTIMAGE_SHA:
                allowed["conv"] = contract.CONV_QKV_POSTIMAGE_SHA[path]
                if (path == str(CONV_QKV_TARGET)
                        and item.get("sha256") == CONV_QKV_VARLEN_POSTIMAGE):
                    allowed["conv"] = CONV_QKV_VARLEN_POSTIMAGE
            if path in contract.CHUNK128_POSTIMAGE_SHA:
                allowed["chunk128"] = contract.CHUNK128_POSTIMAGE_SHA[path]
            if path in getattr(contract, "ADAPTIVE_POSTIMAGE_SHA", {}):
                allowed["adaptive"] = contract.ADAPTIVE_POSTIMAGE_SHA[path]
            if path in contract.V2_POSTIMAGE_SHA:
                allowed["v2"] = contract.V2_POSTIMAGE_SHA[path]
            if path in contract.LEGACY_POSTIMAGE_SHA:
                allowed["legacy"] = contract.LEGACY_POSTIMAGE_SHA[path]
            if (state not in allowed or item.get("sha256") != allowed[state]
                    or item.get("postimage_sha256") != expected):
                raise SystemExit(
                    f"GDN decision patch source SHA-256 mismatch: {path} ({state}); "
                    "verify the pinned tree and BSND dependency before retrying")
            statuses[path] = state
        flag = report.get("flag_module", {})
        expected_flag = contract.FLAG_SHA256
        if (flag.get("source_sha256") != expected_flag
                or flag.get("expected_sha256") != expected_flag):
            raise SystemExit("GDN decision flag source SHA-256 mismatch")
        flag_state = flag.get("status")
        flag_hashes = {
            "current": expected_flag,
            "legacy": contract.LEGACY_FLAG_SHA256,
            "legacy-v1": contract.LEGACY_FLAG_SHA256_V1,
            "missing": None,
        }
        if (flag_state not in flag_hashes
                or flag.get("sha256") != flag_hashes[flag_state]
                or flag.get("present") != (flag_state != "missing")):
            raise SystemExit("GDN decision flag module is missing version metadata or has unknown content")
        complete = all(
            state in ("patched", "qkv", "conv", "chunk128", "adaptive")
            for state in statuses.values()
        ) and flag_state == "current"
        if report.get("ready") is not complete:
            raise SystemExit("GDN decision patch readiness disagrees with the verified file hashes")
        return statuses, flag_state, complete

    report = run("--check")
    statuses, flag_state, complete = verify(report)
    if not complete:
        if not apply:
            record.update(pending_install=True, patch_state=statuses, flag_module_state=flag_state)
            return record
        run("--apply")
        report = run("--check")
        statuses, flag_state, complete = verify(report)
    if not complete:
        raise SystemExit(f"GDN decision patch incomplete: {statuses}")
    record.update(
        enabled=True,
        patch_version=report["patch_version"],
        patch_state=statuses,
        files={path: item.get("sha256") for path, item in report["files"].items()},
        flag_module_sha256=report.get("flag_module", {}).get("sha256"),
    )
    return record


def systemone_mm_state_patch(apply=True):
    """Install/verify the ``/v1/systemone`` multimodal ``state`` frontend.

    The patch is inert for text requests: a ``state`` that is not a
    well-formed content-part list reaches ``render_state`` unchanged, so the
    OpenDecisions path stays byte-identical.  Dry-run is strictly read-only
    and never probes the vLLM tree.
    """
    mode = systemone_mm_mode()
    record = {
        "requested": True,
        "enabled": False,
        "mode": mode,
        "mode_env": SYSTEMONE_MM_MODE_ENV,
        "installer": str(SYSTEMONE_MM_INSTALLER),
        "installer_sha256": (
            sha256_bytes(SYSTEMONE_MM_INSTALLER.read_bytes())
            if SYSTEMONE_MM_INSTALLER.is_file() else None),
        "module": str(SYSTEMONE_MM_MODULE),
        "module_sha256": (
            sha256_bytes(SYSTEMONE_MM_MODULE.read_bytes())
            if SYSTEMONE_MM_MODULE.is_file() else None),
    }
    if not SYSTEMONE_MM_INSTALLER.is_file():
        raise SystemExit(f"systemone multimodal installer missing: {SYSTEMONE_MM_INSTALLER}")
    if not SYSTEMONE_MM_MODULE.is_file():
        raise SystemExit(f"systemone multimodal module missing: {SYSTEMONE_MM_MODULE}")
    if not apply:
        # Dry-run has no side effects and no vLLM dependency.
        record.update(pending_install=True)
        return record
    contract = load_python_contract(
        SYSTEMONE_MM_INSTALLER, "clef_systemone_mm_installer")

    def run(flag):
        result = subprocess.run(
            [sys.executable, str(SYSTEMONE_MM_INSTALLER), flag],
            capture_output=True, text=True, check=False)
        try:
            report = json.loads(result.stdout)
        except ValueError:
            report = {}
        if result.returncode != 0 and not (flag == "--check" and report):
            raise SystemExit(
                f"systemone multimodal patch {flag} failed:\n"
                f"{result.stdout[-2000:]}{result.stderr[-2000:]}")
        return report

    def verify(report):
        state = report.get("state")
        module_state = report.get("module_state")
        postimage = report.get("sha256") == contract.POSTIMAGE_SHA256
        if (report.get("patch_version") != contract.PATCH_VERSION
                or report.get("upstream_sha256") != contract.UPSTREAM_SHA256
                or report.get("postimage_sha256") != contract.POSTIMAGE_SHA256
                or report.get("module_expected_sha256") != contract.MODULE_SHA256
                or state not in ("preimage", "postimage")
                or module_state not in ("missing", "current")
                or report.get("ready") is not (postimage and module_state == "current")):
            raise SystemExit(
                "systemone multimodal patch source SHA-256 mismatch: "
                f"{report}; verify the pinned vLLM tree before retrying")
        return state, module_state

    report = run("--check")
    status, module_status = verify(report)
    if not report.get("ready"):
        run("--apply")
        report = run("--check")
        status, module_status = verify(report)
    if not report.get("ready"):
        raise SystemExit(
            f"systemone multimodal patch incomplete: {status}/{module_status}")
    record.update(
        enabled=True,
        patch_version=report.get("patch_version"),
        patch_state=status,
        installed_sha256=report.get("sha256"),
        installed_module_sha256=report.get("module_sha256"),
    )
    return record


def qkv_norm_patch(config, apply=True, output_dir=None):
    """Install/verify the checked-in GDN QKV fusion postimage."""
    requested = bool(
        CONFIGS[config].get("gdn_data_path", {}).get("preprocess_fusion", False)
    )
    record = {
        "enabled": False,
        "requested": requested,
        "installer": str(QKV_INSTALLER),
        "installer_sha256": (
            sha256_bytes(QKV_INSTALLER.read_bytes()) if QKV_INSTALLER.is_file() else None
        ),
    }
    if not requested:
        return record
    if not QKV_INSTALLER.is_file():
        raise SystemExit(f"QKV norm patch installer missing: {QKV_INSTALLER}")
    if not apply:
        # Dry-run must not touch the source tree (and unit tests do not have
        # the pinned vLLM-Ascend tree on the host).
        record.update(pending_install=True)
        return record

    def run(flag):
        argv = [sys.executable, str(QKV_INSTALLER)]
        tree = resolve_package_tree("vllm_ascend")
        if tree is not None:
            argv += ["--tree", str(tree)]
        if output_dir is not None:
            argv += ["--backup-dir",
                     str(Path(output_dir).expanduser().resolve() / "patch-backup" / "qkv-norm")]
        argv.append(flag)
        result = subprocess.run(argv, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            raise SystemExit(
                f"QKV norm patch {flag} failed:\n"
                f"{result.stdout[-2000:]}{result.stderr[-2000:]}")
        return json.loads(result.stdout)

    def verify(report):
        statuses = {
            path: item.get("status_after") for path, item in report.get("files", {}).items()
        }
        if set(statuses) != {
            "vllm_ascend/device/device_op.py", "vllm_ascend/ops/gdn.py"
        } or not all(state == "qkv" for state in statuses.values()):
            raise SystemExit(f"QKV norm patch incomplete: {statuses}")
        return statuses

    report = run("--check")
    statuses = {
        path: item.get("status_after") for path, item in report.get("files", {}).items()
    }
    complete = bool(report.get("ready")) and all(
        state == "qkv" for state in statuses.values()
    )
    if not complete:
        if not apply:
            record.update(pending_install=True, patch_state=statuses)
            return record
        run("--apply")
        report = run("--check")
        statuses = verify(report)
    record.update(enabled=True, patch_state=statuses)
    return record


def conv_qkv_patch(requested, vendor, output_dir, *, apply):
    """Install and verify the Conv+QKV source on the worker's import tree."""
    record = {"requested": requested, "ready": False, "installed": False,
              "installer": str(CONV_QKV_INSTALLER), "vendor": vendor}
    if not requested:
        return record
    if not apply:
        record["pending_install"] = True
        return record
    if vendor is None:
        raise SystemExit("Conv+QKV requested but the bundled op_api vendor is unavailable")
    if not CONV_QKV_INSTALLER.is_file():
        raise SystemExit(f"Conv+QKV installer missing: {CONV_QKV_INSTALLER}")

    spec = importlib.util.find_spec("vllm_ascend")
    init = Path(spec.origin).resolve() if spec and spec.origin else None
    if init is None or init.name != "__init__.py" or init.parent.name != "vllm_ascend":
        raise SystemExit("Could not resolve the vLLM-Ascend worker import tree for Conv+QKV")
    tree = init.parent.parent
    target = tree / CONV_QKV_TARGET
    backup = output_dir / "conv-qkv-source-backup"
    command = [sys.executable, str(CONV_QKV_INSTALLER), "--tree", str(tree),
               "--backup-dir", str(backup)]

    def check():
        result = installer_call([*command, "--check"])
        report = result["report"]
        sha = report.get("sha256")
        known = {CONV_QKV_PREIMAGE: "preimage", CONV_QKV_POSTIMAGE: "patched",
                 CONV_QKV_VARLEN_POSTIMAGE: "patched"}
        if (result["rc"] or report.get("tree") != str(tree)
                or report.get("target") != str(CONV_QKV_TARGET)
                or sha not in known or report.get("status_after") != known[sha]
                or report.get("ready") is not (sha in (
                    CONV_QKV_POSTIMAGE, CONV_QKV_VARLEN_POSTIMAGE))
                or not target.is_file() or sha256_file(target) != sha):
            raise SystemExit(f"Conv+QKV source state could not be verified at {target}")
        return sha

    before = check()
    if before == CONV_QKV_PREIMAGE:
        result = installer_call([*command, "--apply"])
        if result["rc"]:
            raise SystemExit(f"Conv+QKV installation failed at {target}: "
                             f"{result['report'].get('error', 'installer failed')}")
        record["installed"] = True
    after = check() if record["installed"] else before
    if after not in (CONV_QKV_POSTIMAGE, CONV_QKV_VARLEN_POSTIMAGE):
        raise SystemExit(f"Conv+QKV postimage SHA-256 mismatch at {target}")
    record.update(ready=True, tree=str(tree), target=str(target), backup_dir=str(backup),
                  source_sha256_before=before, source_sha256=after,
                  installer_sha256=sha256_file(CONV_QKV_INSTALLER))
    return record


def conv_qkv_varlen_patch(requested, tree, output_dir, *, apply):
    """Install and verify the multi-sequence Conv+QKV varlen postimage."""
    record = {"requested": requested, "ready": False, "installed": False,
              "installer": str(CONV_QKV_VARLEN_INSTALLER)}
    if not requested:
        return record
    if not CONV_QKV_VARLEN_INSTALLER.is_file():
        raise SystemExit(f"Conv+QKV varlen installer missing: {CONV_QKV_VARLEN_INSTALLER}")
    if not apply or tree is None:
        record["pending_install"] = True
        return record
    tree = Path(tree)
    target = tree / CONV_QKV_TARGET
    backup = output_dir / "conv-qkv-varlen-source-backup"
    command = [sys.executable, str(CONV_QKV_VARLEN_INSTALLER), "--tree", str(tree),
               "--backup-dir", str(backup)]

    def check():
        result = installer_call([*command, "--check"])
        report = result["report"]
        sha = report.get("sha256")
        known = {CONV_QKV_POSTIMAGE: "preimage", CONV_QKV_VARLEN_POSTIMAGE: "patched"}
        if (result["rc"] or report.get("tree") != str(tree)
                or report.get("target") != str(CONV_QKV_TARGET)
                or sha not in known or report.get("status_after") != known[sha]
                or report.get("ready") is not (sha == CONV_QKV_VARLEN_POSTIMAGE)
                or not target.is_file() or sha256_file(target) != sha):
            raise SystemExit(
                f"Conv+QKV varlen source state could not be verified at {target}")
        return sha

    before = check()
    if before == CONV_QKV_POSTIMAGE:
        result = installer_call([*command, "--apply"])
        if result["rc"]:
            raise SystemExit(
                f"Conv+QKV varlen installation failed at {target}: "
                f"{result['report'].get('error', 'installer failed')}")
        record["installed"] = True
    after = check() if record["installed"] else before
    if after != CONV_QKV_VARLEN_POSTIMAGE:
        raise SystemExit(f"Conv+QKV varlen postimage SHA-256 mismatch at {target}")
    record.update(ready=True, tree=str(tree), target=str(target),
                  backup_dir=str(backup), source_sha256_before=before,
                  source_sha256=after,
                  installer_sha256=sha256_file(CONV_QKV_VARLEN_INSTALLER))
    return record


def tiered_cache_patch(requested, output_dir, *, apply):
    """Install and verify the Tier 2 prefix-state cache entry points.

    Two hash-gated targets: the worker's ``model_runner.py`` (lazy engine
    hook install before the first ``execute_model``) and the API's
    ``api_router.py`` (``/v1/decide/info`` disclosure).  ``ready`` means both
    targets verify as patched in the trees the service will import.
    """
    record = {"requested": requested, "ready": False, "installed": False,
              "installer": str(TIERED_INSTALLER)}
    if not requested:
        return record
    if not TIERED_INSTALLER.is_file():
        raise SystemExit(f"Tier 2 installer missing: {TIERED_INSTALLER}")
    if not apply:
        record["pending_install"] = True
        return record

    trees = {}
    for package, key in (("vllm", "vllm_tree"), ("vllm_ascend", "ascend_tree")):
        spec = importlib.util.find_spec(package)
        init = Path(spec.origin).resolve() if spec and spec.origin else None
        if init is None or init.name != "__init__.py" or init.parent.name != package:
            raise SystemExit(f"Could not resolve the vLLM import tree for {package}")
        trees[key] = init.parent.parent
    targets = {
        "model_runner": trees["ascend_tree"] / TIERED_MODEL_RUNNER_TARGET,
        "api_router": trees["vllm_tree"] / TIERED_API_ROUTER_TARGET,
    }
    known = {
        "model_runner": {"preimage": TIERED_MODEL_RUNNER_PREIMAGE,
                         "patched": TIERED_MODEL_RUNNER_POSTIMAGE},
        "api_router": {"preimage": TIERED_API_ROUTER_PREIMAGE,
                       "patched": TIERED_API_ROUTER_POSTIMAGE},
    }
    backup = output_dir / "tiered-cache-source-backup"
    command = [sys.executable, str(TIERED_INSTALLER),
               "--vllm-tree", str(trees["vllm_tree"]),
               "--ascend-tree", str(trees["ascend_tree"]),
               "--backup-dir", str(backup)]

    def check():
        result = installer_call([*command, "--check"])
        report = result["report"]
        entries = report.get("targets") or {}
        valid = (not result["rc"]
                 and report.get("vllm_tree") == str(trees["vllm_tree"])
                 and report.get("ascend_tree") == str(trees["ascend_tree"])
                 and set(entries) == set(targets))
        if valid:
            for name, target in targets.items():
                entry = entries[name]
                status = entry.get("status_after")
                if (entry.get("target") != str(target)
                        or status not in known[name]
                        or entry.get("sha256") != known[name][status]
                        or not target.is_file()
                        or sha256_file(target) != entry.get("sha256")):
                    valid = False
                    break
        if not valid:
            raise SystemExit(
                "Tier 2 source state could not be verified: "
                f"{report.get('error', 'installer report mismatch')}")
        return {name: entries[name]["status_after"] for name in targets}

    before = check()
    if any(state == "preimage" for state in before.values()):
        result = installer_call([*command, "--apply"])
        if result["rc"]:
            raise SystemExit(f"Tier 2 installation failed: "
                             f"{result['report'].get('error', 'installer failed')}")
        record["installed"] = True
    after = check()
    if any(state != "patched" for state in after.values()):
        raise SystemExit("Tier 2 postimage SHA-256 mismatch")
    record.update(
        ready=True,
        vllm_tree=str(trees["vllm_tree"]),
        ascend_tree=str(trees["ascend_tree"]),
        backup_dir=str(backup),
        targets={name: {"path": str(target),
                        "sha256": sha256_file(target),
                        "status_before": before[name]}
                 for name, target in targets.items()},
        installer_sha256=sha256_file(TIERED_INSTALLER))
    return record


def static_ram_limit_bytes():
    """The container's static memory ceiling, or ``None`` when unreadable.

    ``min(MemTotal, cgroup memory.max)`` — static totals only, never
    ``MemAvailable``, so both TP workers compute the same clamp.
    """
    limits = []
    try:
        with open("/proc/meminfo", encoding="ascii") as handle:
            for line in handle:
                if line.startswith("MemTotal:"):
                    limits.append(int(line.split()[1]) * 1024)
                    break
    except (OSError, ValueError, IndexError):
        pass
    for candidate in ("/sys/fs/cgroup/memory.max",
                      "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        try:
            value = Path(candidate).read_text().strip()
        except OSError:
            continue
        if value in ("", "max"):
            continue
        try:
            number = int(value)
        except ValueError:
            continue
        if number < (1 << 60):  # cgroup v1 "unlimited" sentinel
            limits.append(number)
    return min(limits) if limits else None


def _tier2b_only_policy(policy, *, ram_limit, rank_env, slots_env):
    """Resolve the pure Tier 2B mode triggered by an explicit CLEF_TIER2A=0.

    This branch is only reachable when the operator explicitly disables
    Tier 2A.  It restores the validated 32 GiB/rank Tier 2B pool on large
    containers and keeps the small-container disable rule unchanged.
    """
    policy["representation_bytes"] = 0
    if rank_env is not None:
        policy["rank_bytes"] = int(rank_env)
        if slots_env is not None:
            policy["ring_slots"] = int(slots_env)
        policy["reason"] = "tier2b_only_explicit_rank_bytes"
        return policy
    if ram_limit is None:
        ram_limit = static_ram_limit_bytes()
    policy["ram_limit_bytes"] = ram_limit
    if ram_limit is None:
        policy["rank_bytes"] = 32 << 30
        policy["ring_slots"] = int(slots_env) if slots_env is not None else 8
        policy["reason"] = "tier2b_only_ram_limit_unreadable"
        return policy
    if ram_limit >= TIERED_RAM_FULL:
        policy["rank_bytes"] = 32 << 30
        policy["ring_slots"] = int(slots_env) if slots_env is not None else 8
        policy["reason"] = "tier2b_only_ram>=112GiB"
        return policy
    if ram_limit <= TIERED_RAM_DISABLE:
        policy["enabled"] = False
        policy["reason"] = "ram<=48GiB"
        return policy
    policy["rank_bytes"] = min(
        32 << 30, max(4 << 30, (ram_limit - (36 << 30)) // 2))
    policy["ring_slots"] = int(slots_env) if slots_env is not None else 2
    policy["reason"] = f"tier2b_only_clamped:{ram_limit >> 30}GiB"
    return policy


def tiered_memory_policy(requested, *, ram_limit=None, rank_env=None,
                         slots_env=None, repr_env=None, tier2a_env=None):
    """Clamp the multi-granularity Tier 2 budgets to static container RAM.

    The cache has two host-resident tiers per TP rank:

    * **Tier 2A** (continuous layer-56 representation): 28 GiB pageable,
      enough for the full 2,291-prompt OD set (~19.5 GiB) plus the 400-question
      ERQA set (~7.1 GiB);
    * **Tier 2B** (aligned recurrent/KV prefix state): 10 GiB with
      ``aged_keep_longest`` and the 500-sighting active window.

    Policy:

    * unreadable or >= 112 GiB: keep the measured 28 + 10 GiB defaults;
    * <= 48 GiB: disable the cache so no ring or pool is allocated;
    * 48..112 GiB: split ``min(38 GiB, max(4 GiB, (ram-36 GiB)//2))`` per rank
      in the 28:10 ratio (a 64 GiB container gets 10 GiB + 4 GiB per rank).

    An explicit ``CLEF_PREFIX_CACHE_RANK_BYTES`` (Tier 2B) or
    ``CLEF_TIER2A_RANK_BYTES`` (Tier 2A) always wins and skips the clamp.
    An explicit ``CLEF_TIER2A=0`` switches to the pure Tier 2B profile and
    restores the validated 32 GiB/rank Tier 2B pool on large containers.
    """
    if rank_env is None:
        rank_env = os.environ.get("CLEF_PREFIX_CACHE_RANK_BYTES")
    if repr_env is None:
        repr_env = os.environ.get("CLEF_TIER2A_RANK_BYTES")
    if slots_env is None:
        slots_env = os.environ.get("CLEF_TIER2_RING_SLOTS")
    if tier2a_env is None:
        tier2a_env = os.environ.get("CLEF_TIER2A")
    policy = {
        "requested": bool(requested),
        "enabled": bool(requested),
        "ram_limit_bytes": None,
        "rank_bytes": None,
        "representation_bytes": None,
        "ring_slots": None,
        "reason": "requested" if requested else "requested-off",
    }
    if not requested:
        return policy
    tier2a_off = (tier2a_env is not None
                  and tier2a_env.strip().lower() in
                  ("0", "false", "no", "off"))
    if tier2a_off:
        return _tier2b_only_policy(
            policy, ram_limit=ram_limit, rank_env=rank_env,
            slots_env=slots_env)
    if rank_env is not None or repr_env is not None:
        policy["rank_bytes"] = (int(rank_env) if rank_env is not None
                                else int(TIERED_RANK_BYTES_DEFAULT))
        policy["representation_bytes"] = (
            int(repr_env) if repr_env is not None
            else int(TIERED2A_RANK_BYTES_DEFAULT))
        if slots_env is not None:
            policy["ring_slots"] = int(slots_env)
        policy["reason"] = "explicit_rank_bytes"
        return policy
    if ram_limit is None:
        ram_limit = static_ram_limit_bytes()
    policy["ram_limit_bytes"] = ram_limit
    if ram_limit is None:
        policy["reason"] = "ram_limit_unreadable"
        return policy
    if ram_limit >= TIERED_RAM_FULL:
        policy["reason"] = "ram>=112GiB"
        return policy
    if ram_limit <= TIERED_RAM_DISABLE:
        policy["enabled"] = False
        policy["reason"] = "ram<=48GiB"
        return policy
    total = min(TIERED_MULTI_RANK_BYTES,
                max(4 << 30, (ram_limit - (36 << 30)) // 2))
    # Keep both shares whole GiB so the manifest and the two TP workers see
    # exactly the same budgets.
    representation = (total * 28 // 38) & ~((1 << 30) - 1)
    policy["representation_bytes"] = representation
    policy["rank_bytes"] = total - representation
    policy["ring_slots"] = int(slots_env) if slots_env is not None else 2
    policy["reason"] = f"clamped:{ram_limit >> 30}GiB"
    return policy


def gdn_chunk128_patch(config, apply=True, decision_enabled=False, output_dir=None):
    """Install/verify the GDN chunk-128 decision-path patch.

    Measured default on the decision path: another 1.3-2.3% on S1 over the
    fused conv+QKV kernel while every official 2PP gate stays satisfied.
    ``CLEF_GDN_CHUNK128=0`` restores chunk 64.
    """
    requested = decision_enabled and os.environ.get(
        "CLEF_GDN_CHUNK128", "").strip().lower() not in ("0", "false", "no", "off")
    record = {
        "enabled": False,
        "requested": requested,
        "installer": str(CHUNK128_INSTALLER),
        "installer_sha256": (
            sha256_bytes(CHUNK128_INSTALLER.read_bytes())
            if CHUNK128_INSTALLER.is_file() else None
        ),
    }
    if not requested:
        return record
    if not CHUNK128_INSTALLER.is_file():
        raise SystemExit(f"GDN chunk128 installer missing: {CHUNK128_INSTALLER}")
    if not apply:
        record.update(pending_install=True)
        return record

    def run(flag):
        argv = [sys.executable, str(CHUNK128_INSTALLER)]
        tree = resolve_package_tree("vllm_ascend")
        if tree is not None:
            argv += ["--tree", str(tree)]
        if output_dir is not None:
            argv += ["--backup-dir",
                     str(Path(output_dir).expanduser().resolve() / "patch-backup" / "gdn-chunk128")]
        argv.append(flag)
        result = subprocess.run(argv, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            raise SystemExit(
                f"GDN chunk128 patch {flag} failed:\n"
                f"{result.stdout[-2000:]}{result.stderr[-2000:]}")
        return json.loads(result.stdout)

    def verify(report):
        statuses = {
            path: item.get("status_after") for path, item in report.get("files", {}).items()
        }
        if set(statuses) != {
            "vllm_ascend/device/device_op.py",
            "vllm_ascend/ops/gdn_attn_builder.py",
        } or not all(state == "patched" for state in statuses.values()):
            raise SystemExit(f"GDN chunk128 patch incomplete: {statuses}")
        return statuses

    report = run("--check")
    statuses = {
        path: item.get("status_after") for path, item in report.get("files", {}).items()
    }
    complete = bool(report.get("ready")) and all(
        state == "patched" for state in statuses.values()
    )
    if not complete:
        run("--apply")
        report = run("--check")
    statuses = verify(report)
    record.update(enabled=True, patch_state=statuses)
    return record


def gdn_adaptive_chunk_patch(config, output_dir, apply=True, decision_enabled=False):
    """Install/verify the stacked GDN adaptive-chunk selector.

    The default ``CLEF_GDN_CHUNK128_MIN_TOKENS=0`` keeps the measured all-128
    path; a positive threshold routes short requests to chunk 64.
    """
    requested = decision_enabled and os.environ.get(
        "CLEF_GDN_ADAPTIVE_CHUNK", "0").strip().lower() not in (
            "0", "false", "no", "off")
    record = {
        "enabled": False,
        "requested": requested,
        "installer": str(ADAPTIVE_CHUNK_INSTALLER),
        "installer_sha256": (
            sha256_bytes(ADAPTIVE_CHUNK_INSTALLER.read_bytes())
            if ADAPTIVE_CHUNK_INSTALLER.is_file() else None),
    }
    if not requested:
        return record
    if not ADAPTIVE_CHUNK_INSTALLER.is_file():
        raise SystemExit(f"GDN adaptive-chunk installer missing: {ADAPTIVE_CHUNK_INSTALLER}")
    if not apply:
        record.update(pending_install=True)
        return record
    spec = importlib.util.find_spec("vllm_ascend")
    init = Path(spec.origin).resolve() if spec and spec.origin else None
    if init is None or init.name != "__init__.py" or init.parent.name != "vllm_ascend":
        raise SystemExit("Could not resolve the vLLM-Ascend worker import tree")
    tree = init.parent.parent
    backup = Path(output_dir).expanduser().resolve() / "patch-backup" / "gdn-adaptive-chunk"

    def run(flag):
        result = subprocess.run(
            [sys.executable, str(ADAPTIVE_CHUNK_INSTALLER), "--tree", str(tree),
             "--backup-dir", str(backup), flag],
            capture_output=True, text=True, check=False)
        if result.returncode != 0:
            raise SystemExit(
                f"GDN adaptive-chunk patch {flag} failed:\n"
                f"{result.stdout[-2000:]}{result.stderr[-2000:]}")
        return json.loads(result.stdout)

    def verify(report):
        statuses = {
            path: item.get("status_after") for path, item in report.get("files", {}).items()
        }
        expected = {"vllm_ascend/device/device_op.py",
                    "vllm_ascend/ops/gdn_attn_builder.py"}
        if set(statuses) != expected or not all(
                state == "adaptive" for state in statuses.values()):
            raise SystemExit(f"GDN adaptive-chunk patch incomplete: {statuses}")
        return statuses

    report = run("--check")
    complete = bool(report.get("ready"))
    if not complete:
        run("--apply")
        report = run("--check")
    statuses = verify(report)
    record.update(enabled=True, tree=str(tree), patch_state=statuses)
    return record


def joint_single_field_patch(config, apply=True, output_dir=None):
    """Install/verify the Q=1 joint-head postimage."""
    requested = bool(CONFIGS[config].get("joint_single_field", False))
    record = {
        "enabled": False,
        "requested": requested,
        "installer": str(JOINT_INSTALLER),
        "installer_sha256": (
            sha256_bytes(JOINT_INSTALLER.read_bytes()) if JOINT_INSTALLER.is_file() else None
        ),
    }
    if not requested:
        return record
    if not JOINT_INSTALLER.is_file():
        raise SystemExit(f"joint single-field installer missing: {JOINT_INSTALLER}")
    if not apply:
        record.update(pending_install=True)
        return record

    def run(flag):
        argv = [sys.executable, str(JOINT_INSTALLER)]
        tree = resolve_package_tree("vllm")
        if tree is not None:
            argv += ["--tree", str(tree)]
        if output_dir is not None:
            argv += ["--backup-dir",
                     str(Path(output_dir).expanduser().resolve() / "patch-backup" / "joint-single-field")]
        argv.append(flag)
        result = subprocess.run(argv, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            raise SystemExit(
                f"joint single-field patch {flag} failed:\n"
                f"{result.stdout[-2000:]}{result.stderr[-2000:]}")
        return json.loads(result.stdout)

    def verify(report):
        if report.get("status_after") != "single-field":
            raise SystemExit(f"joint single-field patch incomplete: {report}")
        return report["status_after"]

    report = run("--check")
    complete = bool(report.get("ready")) and report.get("status_after") == "single-field"
    if not complete:
        run("--apply")
        report = run("--check")
        verify(report)
    record.update(enabled=True, patch_state=report.get("status_after"))
    return record


def fused_joint_patch(config, output_dir, apply=True):
    """Install/verify the aa3e fused-attention joint head for the default config.

    The guard (36) and fused (b09) installers are check-first.  The guard's
    ALLOW exemption exists only in ``--check``, so apply is called only when its
    own report says ``status_before=preimage``; f810 vs aa3e is distinguished
    solely by the fused report's ``state_before``.  Every start writes an atomic
    ``joint-patch-record.json`` (including no-op) before any service launch; any
    failure raises SystemExit so vLLM is never started in a mixed state.
    """
    requested = bool(CONFIGS[config].get("joint_fused_attention", False))
    record = {
        "schema": "joint-fused-entry/1",
        "enabled": False,
        "requested": requested,
        "mode": "joint-fused-attention",
        "default_config": DEFAULT_CONFIG,
        "config": config,
        "installer": str(JOINT_FUSED_INSTALLER),
        "installer_sha256": (sha256_file(JOINT_FUSED_INSTALLER)
                             if JOINT_FUSED_INSTALLER.is_file() else None),
        "guard_installer": str(JOINT_FUSED_GUARD),
        "guard_installer_sha256": (sha256_file(JOINT_FUSED_GUARD)
                                   if JOINT_FUSED_GUARD.is_file() else None),
        "schema_sha256": JOINT_FUSED_SHA,
        "flag_env": JOINT_FLAG_ENV,
        "allow_env": JOINT_ALLOW_ENV,
    }
    if not requested:
        os.environ.pop(JOINT_FLAG_ENV, None)
        os.environ.pop(JOINT_ALLOW_ENV, None)
        return record
    # The guard's check exemption reads this env from its own process; setting
    # it before the guard check also keeps the launched service consistent.
    os.environ[JOINT_FLAG_ENV] = "1"
    os.environ[JOINT_ALLOW_ENV] = JOINT_FUSED_SHA
    record.update(enabled=True, pending_install=not apply,
                  flag_value=os.environ[JOINT_FLAG_ENV],
                  allow_value=os.environ[JOINT_ALLOW_ENV])
    if not apply:
        return record

    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    record_path = output / "joint-patch-record.json"
    guard_backup = output / "patch-backup" / "joint-single-field"
    fused_backup = output / "patch-backup" / "joint-fused"
    guard_manifest = output / "patch-manifest-joint-single-field.json"
    fused_manifest = output / "patch-manifest-joint-fused.json"
    record.update(
        at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        output_dir=str(output), record=str(record_path),
        guard_backup=str(guard_backup), fused_backup=str(fused_backup),
        guard_manifest=str(guard_manifest), fused_manifest=str(fused_manifest),
        env_present={JOINT_FLAG_ENV: os.environ.get(JOINT_FLAG_ENV),
                     JOINT_ALLOW_ENV: os.environ.get(JOINT_ALLOW_ENV)},
    )
    try:
        guard_backup.mkdir(parents=True, exist_ok=True)
        fused_backup.mkdir(parents=True, exist_ok=True)
        probe = resolve_vllm_tree()
        record.update(tree=probe["tree"], target=probe["target"], probe=probe)
        target = Path(probe["target"])
        before = sha256_file(target)
        record["target_sha256_before"] = before
        if before not in (JOINT_ORIGINAL_SHA, JOINT_SINGLE_FIELD_SHA, JOINT_FUSED_SHA,
                          JOINT_LEGACY_FUSED_SHA):
            raise ValueError(f"unknown joint schema SHA-256: {before}")

        check_env = dict(os.environ)
        guard_check = installer_call(
            [sys.executable, str(JOINT_FUSED_GUARD), "--tree", probe["tree"],
             "--backup-dir", str(guard_backup), "--check"], env=check_env)
        record["guard_check"] = guard_check
        guard_before = guard_check["report"].get("status_before")
        if guard_before == "preimage":
            guard_apply = installer_call(
                [sys.executable, str(JOINT_FUSED_GUARD), "--tree", probe["tree"],
                 "--backup-dir", str(guard_backup), "--apply",
                 "--manifest", str(guard_manifest)], env=check_env)
            record["guard_apply"] = guard_apply
            report = guard_apply["report"]
            if (guard_apply["rc"] != 0 or report.get("status_after") != "single-field"
                    or report.get("ready") is not True):
                raise ValueError(f"joint guard apply did not reach single-field: {report}")
            guard_action = "apply"
        elif guard_before == "single-field":
            if guard_check["report"].get("ready") is not True:
                raise ValueError(f"joint guard check is not ready: {guard_check['report']}")
            guard_action = "noop"
        else:
            raise ValueError(
                f"joint guard check returned status_before={guard_before!r}: "
                f"{guard_check['report']}")

        fused_check = installer_call(
            [sys.executable, str(JOINT_FUSED_INSTALLER), "--tree", probe["tree"],
             "--backup-dir", str(fused_backup), "--check"])
        record["fused_check"] = fused_check
        fused_before = fused_check["report"].get("state_before")
        if fused_before == "preimage":
            fused_apply = installer_call(
                [sys.executable, str(JOINT_FUSED_INSTALLER), "--tree", probe["tree"],
                 "--backup-dir", str(fused_backup), "--apply",
                 "--manifest", str(fused_manifest)])
            record["fused_apply"] = fused_apply
            report = fused_apply["report"]
            if (fused_apply["rc"] != 0 or report.get("state_after") != "postimage"
                    or report.get("ready") is not True):
                raise ValueError(f"joint fused apply did not reach postimage: {report}")
            fused_action = "apply"
        elif fused_before == "postimage":
            if fused_check["report"].get("ready") is not True:
                raise ValueError(f"joint fused check is not ready: {fused_check['report']}")
            fused_action = "noop"
        else:
            raise ValueError(
                f"joint fused check returned state_before={fused_before!r}: "
                f"{fused_check['report']}")

        after = sha256_file(target)
        record["target_sha256_after"] = after
        if after != JOINT_FUSED_SHA:
            raise ValueError(f"joint fused target is {after}, expected {JOINT_FUSED_SHA}")
        guard_after = sha256_file(JOINT_FUSED_GUARD)
        record["guard_sha256_after"] = guard_after
        if guard_after != JOINT_GUARD_POST_SHA:
            raise ValueError(f"joint guard file is {guard_after}, expected {JOINT_GUARD_POST_SHA}")
        record.update(ok=True, guard_action=guard_action, fused_action=fused_action,
                      branch=("original->f810->fused"
                              if guard_action == "apply" and fused_action == "apply"
                              else "legacy->fused"
                              if fused_action == "apply"
                              and fused_check["report"].get("legacy_postimage")
                              else "f810->fused" if fused_action == "apply" else "aa3e-noop"))
        write_json_atomic(record_path, record)
        return record
    except BaseException as exc:
        record.update(ok=False, error=f"{type(exc).__name__}: {exc}")
        try:
            write_json_atomic(record_path, record)
        except OSError:
            pass
        raise SystemExit(f"joint fused attention setup failed: {exc}") from exc


def gdn_data_path(config, enabled):
    """Resolve the three data-path switches and their service environment.

    ``s1``/``s2`` adopt the measured BIC combination (BSND operands, no
    initial-state gather, no convolution cache) when the decision patch is
    enabled.  An explicit ``CLEF_GDN_*`` value always wins, so an operator can
    reproduce a single-switch arm without editing the launcher.  The fused
    preprocessing prototype stays off until its operator is bundled.
    """
    defaults = CONFIGS[config].get("gdn_data_path", {})
    flag_file = os.environ.get("CLEF_GDN_FLAG_FILE")
    file_values = {}
    if flag_file:
        try:
            raw = Path(flag_file).read_text(encoding="utf-8")
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                file_values = parsed
        except (OSError, ValueError):
            file_values = {}
    record = {}
    environment = {}
    for key, env_name in DATA_PATH_ENV.items():
        configured = bool(defaults.get(key))
        file_value = file_values.get(key)
        if isinstance(file_value, bool):
            value = file_value
            source = "flag_file"
        elif env_name in os.environ:
            value = os.environ[env_name].strip().lower() not in (
                "", "0", "false", "no", "off")
            source = "env"
        elif enabled and configured:
            value = True
            source = f"config:{config}"
        else:
            value = False
            source = "default"
        record[key] = {
            "configured": configured,
            "effective": value,
            "source": source,
            "env": env_name,
        }
        if value and source != "flag_file":
            environment[env_name] = "1"
    record["flag_file"] = {
        "path": flag_file,
        "sha256": sha256_bytes(Path(flag_file).read_bytes()) if flag_file and Path(flag_file).is_file() else None,
        "epoch": file_values.get("epoch"),
    }
    return record, environment


def schema_cache_version(path):
    """Read the schema cache version marker from strategies.py; 0 if absent."""
    try:
        text = Path(path).read_text()
    except OSError:
        return 0
    match = re.search(r"^SCHEMA_CACHE_VERSION\s*=\s*(\d+)$", text, re.MULTILINE)
    if match:
        return int(match.group(1))
    return 1 if "_SchemaTokenCache" in text else 0


def resolve_schema_cache(config, explicit):
    """Return (enabled, version, strategies_path)."""
    requested = explicit if explicit is not None else CONFIGS[config].get("schema_token_cache", "off")
    enabled = requested == "on"
    spec = importlib.util.find_spec("vllm")
    strategies = None
    if spec and spec.submodule_search_locations:
        strategies = (
            Path(list(spec.submodule_search_locations)[0])
            / "entrypoints" / "openai" / "decisions" / "strategies.py"
        )
    version = schema_cache_version(strategies) if strategies else 0
    if enabled and version != 2:
        raise ValueError(
            "schema cache v2 未安装：固定源码中的版本为 "
            f"{version}（{strategies}）。先运行 src/runtime/apply_schema_cache_patch.py --apply。"
        )
    return enabled, version, strategies


def qkv_norm_environment(conv_qkv=False, conv_qkv_varlen=False):
    """Expose the bundled GDN preprocessing custom ops to the vLLM workers.

    The Ascend C vendors are optional: if one is missing, the data-path code
    falls back to the accepted BIC path.  ``conv_qkv`` opts into the fused
    convolution + QKV normalization kernel, which replaces both the conv and
    the ``ClefGdnQkvNorm`` launch on the decision prefill path.
    ``conv_qkv_varlen`` additionally exposes the multi-sequence kernel; the
    single-sequence v4 op stays loaded for every ``num_prefills == 1`` call.
    """
    vendor = Path(__file__).resolve().parent / "src" / "ascendc" / "vendors" / "customize"
    library = vendor / "op_api" / "lib" / "libcust_opapi.so"
    if not library.is_file():
        return {}
    conv_vendor = vendor.parent / "clef_conv_qkv"
    explicit_conv = os.environ.get("CLEF_CONV_QKV_VENDOR")
    if explicit_conv:
        # Opt-in candidate vendor for A/B builds outside the bundled tree.
        conv_vendor = Path(explicit_conv).expanduser().resolve()
    conv_library = conv_vendor / "op_api" / "lib" / "libcust_opapi.so"
    conv_active = bool(conv_qkv) and conv_library.is_file()
    varlen_vendor = vendor.parent / "clef_conv_qkv_varlen"
    explicit_varlen = os.environ.get("CLEF_CONV_QKV_VARLEN_VENDOR")
    if explicit_varlen:
        varlen_vendor = Path(explicit_varlen).expanduser().resolve()
    varlen_library = varlen_vendor / "op_api" / "lib" / "libcust_opapi.so"
    varlen_active = bool(conv_qkv_varlen) and varlen_library.is_file()
    parts = [vendor, vendor / "op_api" / "lib", vendor.parent.parent]
    if conv_active:
        # Distinct op names, so ordering only decides which vendor wins if a
        # future build ever ships the same kernel name twice.
        parts = [conv_vendor, conv_vendor / "op_api" / "lib"] + parts
    if varlen_active:
        parts = [varlen_vendor, varlen_vendor / "op_api" / "lib"] + parts
    spec = importlib.util.find_spec("fla_npu")
    if spec and spec.submodule_search_locations:
        fla = Path(list(spec.submodule_search_locations)[0]) / "opp" / "vendors" / "fla_npu_transformer"
        if fla.is_dir():
            parts.extend([fla, fla / "op_api" / "lib"])
    existing_opp = os.environ.get("ASCEND_CUSTOM_OPP_PATH")
    if existing_opp:
        parts.append(Path(existing_opp))
    existing_ld = os.environ.get("LD_LIBRARY_PATH")
    ld_parts = [str(vendor / "op_api" / "lib")]
    if conv_active:
        ld_parts.insert(0, str(conv_vendor / "op_api" / "lib"))
    if varlen_active:
        ld_parts.insert(0, str(varlen_vendor / "op_api" / "lib"))
    if existing_ld:
        ld_parts.append(existing_ld)
    environment = {
        "CLEF_QKV_NORM_VENDOR": str(vendor),
        # The 48-AIV kernel is bit-identical to v3 and stays faster than the
        # decomposed path through 10,294 tokens; keep the fused path for ERQA.
        "CLEF_GDN_QKV_NORM_MAX_TOKENS": os.environ.get(
            "CLEF_GDN_QKV_NORM_MAX_TOKENS", "16384"),
        "ASCEND_CUSTOM_OPP_PATH": os.pathsep.join(str(part) for part in parts),
        "LD_LIBRARY_PATH": os.pathsep.join(ld_parts),
    }
    if conv_active:
        environment["CLEF_CONV_QKV_VENDOR"] = str(conv_vendor)
        environment["CLEF_CONV_QKV_TILE"] = os.environ.get("CLEF_CONV_QKV_TILE", "32")
        environment["CLEF_GDN_CONV_QKV"] = "1"
    if varlen_active:
        environment["CLEF_CONV_QKV_VARLEN_VENDOR"] = str(varlen_vendor)
        environment["CLEF_CONV_QKV_VARLEN_TILE"] = os.environ.get(
            "CLEF_CONV_QKV_VARLEN_TILE", "64")
        environment["CLEF_GDN_CONV_QKV_VARLEN"] = "1"
    return environment


def effective_environment():
    env = {}
    for key in sorted(os.environ):
        if key in ENV_WHITELIST_KEYS or key.startswith(ENV_WHITELIST_PREFIXES):
            env[key] = os.environ[key]
    return env


def parse_limit_mm(value):
    try:
        spec = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"JSON 解析失败：{exc}") from exc
    if not isinstance(spec, dict) or set(spec) != {"image", "video"}:
        raise ValueError("应形如 {\"image\":<int>,\"video\":<int>} 且只含这两个键")
    for key in ("image", "video"):
        if not isinstance(spec[key], int) or isinstance(spec[key], bool) or spec[key] < 0:
            raise ValueError(f"{key} 应为非负整数")
    return spec


def parse_passthrough(tokens):
    """Parse ``--`` arguments into (flag, value, original) triples.

    Supports ``--flag value`` and ``--flag=value``; a bare ``--flag`` has
    value ``None``. Positional arguments are rejected.
    """
    parsed = []
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if not token.startswith("--"):
            raise ValueError(f"-- 之后只接受 vLLM 选项，收到位置参数：{token}")
        if "=" in token:
            flag, value = token.split("=", 1)
            parsed.append((flag, value, token))
            index += 1
            continue
        flag = token
        if index + 1 < len(tokens) and not tokens[index + 1].startswith("--"):
            parsed.append((flag, tokens[index + 1], f"{flag} {tokens[index + 1]}"))
            index += 2
        else:
            parsed.append((flag, None, flag))
            index += 1
    return parsed


def apply_passthrough(parsed, effective, sources, passthrough):
    seen = {}
    for flag, value, original in parsed:
        if flag in seen:
            raise ValueError(f"重复的透传参数：{flag}")
        seen[flag] = True
        if flag in FIXED_FLAGS:
            expected = FIXED_FLAGS[flag]
            if expected is None:
                if value is not None:
                    raise ValueError(f"{flag} 是布尔开关，不接受值：{original}")
            elif value != expected:
                raise ValueError(
                    f"{flag} 固定为 {expected!r}，透传值 {value!r} 冲突")
            sources[flag] = "passthrough-fixed"
            continue
        if flag in OVERRIDABLE:
            if value is None:
                raise ValueError(f"{flag} 需要值：{original}")
            key = OVERRIDABLE[flag]
            cast = {
                "port": int,
                "tensor_parallel_size": int,
                "max_model_len": int,
                "max_num_seqs": int,
                "max_num_batched_tokens": int,
                "gpu_memory_utilization": float,
            }.get(key, str)
            try:
                effective[key] = cast(value)
            except ValueError as exc:
                raise ValueError(f"{flag} 的值无效：{value!r}") from exc
            sources[key] = "passthrough"
            continue
        if value is None or original.startswith(flag + "="):
            passthrough.append(original)
        else:
            # Keep a separate argv element for the value, including JSON and
            # strings containing spaces. The display string is diagnostic-only.
            passthrough.extend([flag, value])


def choose(config, explicit, key, cast):
    if explicit is not None:
        return cast(explicit), "explicit"
    if key in CONFIGS[config]:
        return cast(CONFIGS[config][key]), f"config:{config}"
    return None, "default"


def build_launch(args):
    model_dir = Path(args.model_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    config = args.config

    effective, sources = {}, {}
    effective["host"], sources["host"] = args.host, "default" if args.host == "127.0.0.1" else "explicit"
    effective["port"], sources["port"] = args.port, "default" if args.port == 18143 else "explicit"

    for key, explicit_value, cast in (
        ("tensor_parallel_size", args.tensor_parallel_size, int),
        ("max_model_len", args.max_model_len, int),
        ("max_num_seqs", args.max_num_seqs, int),
        ("max_num_batched_tokens", args.max_num_batched_tokens, int),
        ("gpu_memory_utilization", args.gpu_memory_utilization, float),
        ("limit_mm_per_prompt", args.limit_mm_per_prompt, str),
        ("mm_processor_kwargs", getattr(args, "mm_processor_kwargs", None), str),
        ("distributed_executor_backend", args.distributed_executor_backend, str),
    ):
        value, source = choose(config, explicit_value, key, cast)
        if value is None:
            if key == "tensor_parallel_size":
                value = 1
            elif key == "max_model_len":
                value = 1024
            elif key == "max_num_seqs":
                value = 1
            elif key == "gpu_memory_utilization":
                value = 0.90
            elif key == "limit_mm_per_prompt":
                value = '{"image":1,"video":0}'
            elif key == "distributed_executor_backend":
                value = "mp" if effective.get("tensor_parallel_size", 1) > 1 else None
            source = "default"
        effective[key] = value
        sources[key] = source

    schema_enabled, schema_version, strategies = resolve_schema_cache(
        config, args.schema_token_cache)
    effective["schema_token_cache"] = "on" if schema_enabled else "off"
    sources["schema_token_cache"] = (
        "explicit" if args.schema_token_cache is not None else f"config:{config}")

    # Tier 2 prefix-state cache: explicit flag, then CLEF_PREFIX_STATE_CACHE,
    # then the config default (s1 on).
    prefix_state_explicit = getattr(args, "prefix_state_cache", None)
    prefix_state_env = os.environ.get("CLEF_PREFIX_STATE_CACHE")
    if prefix_state_explicit is not None:
        prefix_state_requested = prefix_state_explicit == "on"
        prefix_state_source = "explicit"
    elif prefix_state_env is not None:
        prefix_state_requested = prefix_state_env.strip().lower() not in (
            "0", "false", "no", "off")
        prefix_state_source = "environment"
    else:
        prefix_state_requested = bool(CONFIGS[config].get("prefix_state_cache", False))
        prefix_state_source = (f"config:{config}" if "prefix_state_cache" in CONFIGS[config]
                               else "default")
    # Small containers must never allocate the 32 GiB/rank pool; an explicit
    # CLEF_PREFIX_CACHE_RANK_BYTES skips the clamp (see tiered_memory_policy).
    tiered_memory = tiered_memory_policy(prefix_state_requested)
    if prefix_state_requested and not tiered_memory["enabled"]:
        prefix_state_source = "memory_policy"
    tiered_requested = prefix_state_requested and tiered_memory["enabled"]

    # Extra vLLM arguments.
    extra = args.vllm_args
    if extra[:1] == ["--"]:
        extra = extra[1:]
    parsed = parse_passthrough(extra)
    passthrough = []
    apply_passthrough(parsed, effective, sources, passthrough)

    if effective["tensor_parallel_size"] < 1 or effective["max_model_len"] < 1 or effective["max_num_seqs"] < 1:
        raise ValueError("并行数、上下文长度和序列数应为正数")
    if not 1 <= effective["port"] <= 65535:
        raise ValueError("--port 应在 1 到 65535 之间")
    if not 0 < effective["gpu_memory_utilization"] <= 1:
        raise ValueError("--gpu-memory-utilization 应在 (0, 1] 之间")
    parse_limit_mm(effective["limit_mm_per_prompt"])
    batched = effective["max_num_batched_tokens"]
    if batched is None:
        batched = effective["max_model_len"]
        sources["max_num_batched_tokens"] = (
            "passthrough" if sources.get("max_num_batched_tokens") == "passthrough"
            else "default_follows_max_model_len")
    if batched < 1:
        raise ValueError("--max-num-batched-tokens 应为正数")
    if batched < effective["max_model_len"]:
        raise ValueError(
            "--max-num-batched-tokens 不应小于 --max-model-len（chunked prefill 已关闭，"
            f"单请求最多 {effective['max_model_len']} tokens）")
    effective["max_num_batched_tokens"] = batched

    # Encoder-output retention budget.  ``s1`` adopts the 262,144-embedding
    # Tier 1 cache that Worker 1 phase 0 measured as OD S1 neutral (<= 0.7 ms);
    # an explicit 0 disables it for ablation.
    encoder_cache_tokens, encoder_cache_source = choose(
        config, getattr(args, "encoder_cache_tokens", None),
        "encoder_cache_tokens", int)
    encoder_cache_tokens = encoder_cache_tokens or 0
    if encoder_cache_tokens < 0:
        raise ValueError("--encoder-cache-tokens 应为 0（关闭）或正整数")
    if encoder_cache_tokens and encoder_cache_tokens < batched:
        raise ValueError("--encoder-cache-tokens 应至少等于 --max-num-batched-tokens")
    if encoder_cache_tokens:
        for flag, value, _ in parsed:
            if flag in ("--scheduler-cls", "--async-scheduling", "--scheduler-config"):
                raise ValueError(f"encoder cache 候选与 {flag} 透传冲突")
        encoder_scheduler = RUNTIME_DIR / "clef_encoder_cache_scheduler.py"
        if not encoder_scheduler.is_file():
            raise ValueError(f"encoder cache 候选模块缺失：{encoder_scheduler}")

    command = [
        sys.executable, "-m", "vllm.entrypoints.cli.main", "serve", str(model_dir),
        "--served-model-name", SERVED_MODEL_NAME,
        "--runner", "pooling",
        "--dtype", "bfloat16",
        "--tensor-parallel-size", str(effective["tensor_parallel_size"]),
        "--enforce-eager",
        "--max-model-len", str(effective["max_model_len"]),
        "--max-num-seqs", str(effective["max_num_seqs"]),
        "--max-num-batched-tokens", str(batched),
        "--gpu-memory-utilization", str(effective["gpu_memory_utilization"]),
        "--no-enable-prefix-caching",
        "--no-enable-chunked-prefill",
        "--skip-mm-profiling",
        "--host", effective["host"],
        "--port", str(effective["port"]),
        "--limit-mm-per-prompt", effective["limit_mm_per_prompt"],
    ]
    if effective.get("distributed_executor_backend"):
        command += ["--distributed-executor-backend", effective["distributed_executor_backend"]]
    if effective["mm_processor_kwargs"] is not None:
        command += ["--mm-processor-kwargs", effective["mm_processor_kwargs"]]
    command += passthrough
    if encoder_cache_tokens:
        command += ["--scheduler-cls",
                    "clef_encoder_cache_scheduler.EncoderCacheBudgetScheduler"]

    gdn_decision = gdn_decision_prefill(
        config, apply=not getattr(args, "dry_run", False), output_dir=output_dir)

    environment = {
        "CLEF_MODEL_NAME": SERVED_MODEL_NAME,
        "CLEF_MODEL_DIR": str(model_dir),
        "CLEF_MODEL_REVISION": model_revision(model_dir),
        "CLEF_PROTOCOL_VERSION": PROTOCOL_VERSION,
        "CLEF_CONFIG": config,
        "CLEF_MAX_MODEL_LEN": str(effective["max_model_len"]),
        "CLEF_TENSOR_PARALLEL_SIZE": str(effective["tensor_parallel_size"]),
        "CLEF_HEALTH_JSON": "1",
        "CLEF_SCHEMA_TOKEN_CACHE": "1" if schema_enabled else "0",
        "CLEF_SCHEMA_TOKEN_CACHE_VERSION": str(schema_version if schema_enabled else 0),
        # Aclnn executor cache: the default is exhausted after about 145
        # distinct prompt lengths, after which new shapes take a ~55 ms slow
        # path.  A large cache keeps every measured length on the fast path.
        "ACLNN_CACHE_LIMIT": os.environ.get("ACLNN_CACHE_LIMIT", "1000000"),
    }
    if encoder_cache_tokens:
        environment["CLEF_ENCODER_CACHE_TOKENS"] = str(encoder_cache_tokens)
        module_dir = str(encoder_scheduler.parent)
        existing_path = os.environ.get("PYTHONPATH")
        environment["PYTHONPATH"] = (
            module_dir + os.pathsep + existing_path if existing_path else module_dir)
        effective["encoder_cache_scheduler_sha256"] = sha256_file(encoder_scheduler)
        effective["encoder_cache_tokens"] = encoder_cache_tokens
        sources["encoder_cache_tokens"] = encoder_cache_source
    if gdn_decision["enabled"]:
        environment["CLEF_GDN_DECISION_PREFILL"] = "1"
        environment["CLEF_GDN_NO_RECOMPUTE"] = os.environ.get(
            "CLEF_GDN_NO_RECOMPUTE", "1")
        # MLP BF16 NZ is installed by the v4 GDN patch and gated by this var.
        environment["CLEF_MLP_BF16_NZ"] = os.environ.get("CLEF_MLP_BF16_NZ", "1")
    data_path, data_env = gdn_data_path(config, gdn_decision["enabled"])
    gdn_decision["data_path"] = data_path
    environment.update(data_env)
    qkv_patch = qkv_norm_patch(
        config, apply=not getattr(args, "dry_run", False), output_dir=output_dir)
    gdn_decision["qkv_norm_patch"] = qkv_patch
    chunk128_patch = gdn_chunk128_patch(
        config, apply=not getattr(args, "dry_run", False),
        decision_enabled=gdn_decision["enabled"], output_dir=output_dir)
    gdn_decision["gdn_chunk128_patch"] = chunk128_patch
    adaptive_patch = gdn_adaptive_chunk_patch(
        config, output_dir, apply=not getattr(args, "dry_run", False),
        decision_enabled=chunk128_patch["enabled"])
    gdn_decision["gdn_adaptive_chunk_patch"] = adaptive_patch
    if adaptive_patch["enabled"]:
        environment["CLEF_GDN_CHUNK_RULE"] = os.environ.get("CLEF_GDN_CHUNK_RULE", "max_seq")
        environment["CLEF_GDN_CHUNK128_MIN_TOKENS"] = os.environ.get(
            "CLEF_GDN_CHUNK128_MIN_TOKENS", "0")
    if CONFIGS[config].get("joint_fused_attention"):
        joint_patch = fused_joint_patch(
            config, output_dir, apply=not getattr(args, "dry_run", False))
    else:
        joint_patch = joint_single_field_patch(
            config, apply=not getattr(args, "dry_run", False), output_dir=output_dir)
    gdn_decision["joint_patch"] = joint_patch
    if joint_patch["enabled"]:
        environment["CLEF_JOINT_SINGLE_FIELD"] = "1"
    if CONFIGS[config].get("joint_fused_attention"):
        environment[JOINT_FLAG_ENV] = "1"
        environment[JOINT_ALLOW_ENV] = JOINT_FUSED_SHA
    else:
        # Never inherit a candidate flag into the frozen f810 arm.
        os.environ.pop(JOINT_FLAG_ENV, None)
        os.environ.pop(JOINT_ALLOW_ENV, None)
    # The fused conv + QKV normalization kernel is the measured default on the
    # decision prefill path; CLEF_GDN_CONV_QKV=0 reproduces the unfused arm.
    dry_run = getattr(args, "dry_run", False)
    systemone_mm = systemone_mm_state_patch(apply=not dry_run)
    environment["CLEF_STATE_MM_MODE"] = systemone_mm["mode"]
    conv_qkv_requested = (
        (gdn_decision["enabled"] or dry_run and gdn_decision["requested"])
        and os.environ.get("CLEF_GDN_CONV_QKV", "").strip().lower()
        not in ("0", "false", "no", "off")
    )
    conv_qkv_varlen_requested = (
        conv_qkv_requested
        and os.environ.get("CLEF_GDN_CONV_QKV_VARLEN", "1").strip().lower()
        not in ("0", "false", "no", "off")
    )
    qkv_env = qkv_norm_environment(
        conv_qkv=conv_qkv_requested and not dry_run,
        conv_qkv_varlen=conv_qkv_varlen_requested and not dry_run)
    conv_patch = conv_qkv_patch(conv_qkv_requested, qkv_env.get("CLEF_CONV_QKV_VENDOR"),
                                output_dir, apply=not dry_run)
    gdn_decision["conv_qkv_patch"] = conv_patch
    if conv_patch["ready"]:
        gdn_decision["files"][str(CONV_QKV_TARGET)] = conv_patch["source_sha256"]
        gdn_decision["patch_state"][str(CONV_QKV_TARGET)] = "conv"
    varlen_patch = conv_qkv_varlen_patch(
        conv_qkv_varlen_requested and (dry_run or conv_patch["ready"]),
        conv_patch.get("tree"), output_dir, apply=not dry_run)
    gdn_decision["conv_qkv_varlen_patch"] = varlen_patch
    if varlen_patch["ready"]:
        gdn_decision["files"][str(CONV_QKV_TARGET)] = varlen_patch["source_sha256"]
        gdn_decision["patch_state"][str(CONV_QKV_TARGET)] = "conv_varlen"
    tiered_patch = tiered_cache_patch(
        tiered_requested, output_dir, apply=not dry_run)
    tiered_patch["memory"] = {
        "ram_limit_bytes": tiered_memory["ram_limit_bytes"],
        "rank_bytes": tiered_memory["rank_bytes"],
        "representation_bytes": tiered_memory["representation_bytes"],
        "ring_slots": tiered_memory["ring_slots"],
        "enabled": tiered_memory["enabled"],
        "reason": tiered_memory["reason"],
    }
    effective["prefix_state_cache"] = "on" if tiered_patch["ready"] else "off"
    sources["prefix_state_cache"] = prefix_state_source
    if qkv_env:
        environment.update(qkv_env)
        os.environ.update(qkv_env)
        gdn_decision["qkv_norm_vendor"] = qkv_env["CLEF_QKV_NORM_VENDOR"]
        gdn_decision["conv_qkv_vendor"] = qkv_env.get("CLEF_CONV_QKV_VENDOR")
        gdn_decision["conv_qkv_varlen_vendor"] = qkv_env.get(
            "CLEF_CONV_QKV_VARLEN_VENDOR")
    else:
        gdn_decision["qkv_norm_vendor"] = None
        gdn_decision["conv_qkv_vendor"] = None
        gdn_decision["conv_qkv_varlen_vendor"] = None
    environment["CLEF_GDN_CONV_QKV"] = "1" if conv_patch["ready"] else "0"
    # Tier 2 engine hooks are installed by the hash-gated model_runner block;
    # CLEF_PREFIX_STATE_CACHE=0 keeps the hooks out of the service at runtime.
    environment["CLEF_PREFIX_STATE_CACHE"] = "1" if tiered_patch["ready"] else "0"
    environment["CLEF_TIER2_REPO"] = str(REPO_ROOT / "src")
    rank_bytes = tiered_memory["rank_bytes"]
    if rank_bytes is None:
        rank_bytes = int(os.environ.get(
            "CLEF_PREFIX_CACHE_RANK_BYTES", TIERED_RANK_BYTES_DEFAULT))
    repr_bytes = tiered_memory["representation_bytes"]
    if repr_bytes is None:
        repr_bytes = int(os.environ.get(
            "CLEF_TIER2A_RANK_BYTES", TIERED2A_RANK_BYTES_DEFAULT))
    environment["CLEF_PREFIX_CACHE_RANK_BYTES"] = str(rank_bytes)
    environment["CLEF_TIER2A_RANK_BYTES"] = str(repr_bytes)
    tier2a_env = os.environ.get("CLEF_TIER2A")
    if tier2a_env is not None:
        # An explicit environment switch wins for A/B runs.
        environment["CLEF_TIER2A"] = (
            "1" if tier2a_env.strip().lower() not in
            ("0", "false", "no", "off") else "0")
    else:
        environment["CLEF_TIER2A"] = (
            "1" if tiered_patch["ready"] and repr_bytes > 0 else "0")
    if tiered_memory["ring_slots"] is not None:
        environment["CLEF_TIER2_RING_SLOTS"] = str(tiered_memory["ring_slots"])
    tiered_patch["memory"]["effective_rank_bytes"] = rank_bytes
    tiered_patch["memory"]["effective_representation_bytes"] = repr_bytes
    tiered_patch["memory"]["effective_ring_slots"] = (
        tiered_memory["ring_slots"] if tiered_memory["ring_slots"] is not None
        else TIERED_RING_SLOTS_DEFAULT)
    if "CLEF_TIER2_TRACE" in os.environ:
        environment["CLEF_TIER2_TRACE"] = os.environ["CLEF_TIER2_TRACE"]
    environment["CLEF_GDN_CONV_QKV_VARLEN"] = (
        "1" if varlen_patch["ready"] else "0")
    if os.environ.get("CLEF_GDN_FLAG_FILE"):
        # Experiment-only override, recorded so the manifest shows it.
        gdn_decision["flag_file"] = os.environ["CLEF_GDN_FLAG_FILE"]
        environment["CLEF_GDN_FLAG_FILE"] = gdn_decision["flag_file"]
    return {
        "python": sys.executable,
        "model_dir": str(model_dir),
        "output_dir": str(output_dir),
        "config": config,
        "revision": environment["CLEF_MODEL_REVISION"],
        "argv": command,
        "effective": effective,
        "sources": sources,
        "environment": effective_environment(),
        "service_environment": environment,
        "gdn_decision_prefill": gdn_decision,
        "joint_patch": joint_patch,
        "systemone_mm_state": systemone_mm,
        "tiered_cache": tiered_patch,
        "software": software_provenance(),
        "schema_cache": {
            "enabled": schema_enabled,
            "version": schema_version if schema_enabled else 0,
            "strategies": str(strategies) if strategies else None,
            "strategies_sha256": (
                sha256_bytes(strategies.read_bytes()) if strategies else None),
        },
    }


def main():
    parser = argparse.ArgumentParser(
        description="启动 Clef-27B 的 vLLM-Ascend 服务（/health、/v1/decide/info、"
                    "/v1/systemone、/v1/decide）。",
        epilog="先加载 scripts/activate_env.sh；附加 vLLM 参数放在 -- 之后。",
    )
    parser.add_argument(
        "--model-dir",
        default=os.environ.get("MODEL_DIR", DEFAULT_MODEL_DIR),
        help="完整模型目录，可通过 MODEL_DIR 设置默认值。",
    )
    parser.add_argument(
        "--output-dir",
        default=os.environ.get("OUTPUT_DIR"),
        help="日志、manifest 与运行产物目录；默认 <model-dir>/output。",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18143,
                        help="默认监听 127.0.0.1:18143（比赛接口约定）。")
    parser.add_argument("--config", choices=sorted(CONFIGS), default=DEFAULT_CONFIG,
                        help="s1（默认，单服务常驻）：TP2/16384/seqs16/image16/缓存关；"
                             "custom：TP1/1024 保守默认。")
    parser.add_argument("--tensor-parallel-size", type=int, default=None)
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--max-num-seqs", "--seqs", dest="max_num_seqs", type=int, default=None)
    parser.add_argument("--max-num-batched-tokens", type=int, default=None)
    parser.add_argument("--encoder-cache-tokens", type=int, default=None,
                        help="encoder 输出缓存 embedding 容量；0 关闭（s1 默认），"
                             "正整数按 clef_encoder_cache_scheduler 启用。")
    parser.add_argument("--gpu-memory-utilization", type=float, default=None)
    parser.add_argument("--limit-mm-per-prompt", default=None,
                        help='每请求多模态上限 JSON，如 \'{"image":1,"video":0}\'。')
    parser.add_argument("--mm-processor-kwargs", default=None,
                        help="多模态处理器参数 JSON，可覆盖 s1 默认值。")
    parser.add_argument("--schema-token-cache", choices=("off", "on"), default=None,
                        help="显式覆盖 schema token 缓存；on 要求已安装 v2 补丁。")
    parser.add_argument("--prefix-state-cache", choices=("on", "off"), default=None,
                        help="多粒度 Tier 2 缓存（Tier 2A 第 56 层表征 + Tier 2B 前缀状态）；"
                             "on 要求已安装 src/runtime/apply_tiered_cache_patch.py，"
                             "CLEF_PREFIX_STATE_CACHE=0 可运行时关闭，"
                             "CLEF_TIER2A=0 切纯 Tier 2B 并自动恢复 32 GiB/rank。")
    parser.add_argument("--distributed-executor-backend", default=None)
    parser.add_argument("--dry-run", action="store_true", help="输出启动参数 JSON。")
    parser.add_argument("vllm_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    startup_log(
        f"Starting Clef-27B: model_dir={args.model_dir} config={args.config} "
        f"output_dir={args.output_dir or '<model-dir>/output'}"
    )
    startup_log("Validating model files and runtime patches (may take tens of seconds)...")

    if args.output_dir is None:
        args.output_dir = str(Path(args.model_dir).expanduser().resolve() / "output")

    try:
        launch = build_launch(args)
    except ValueError as exc:
        parser.error(str(exc))

    if args.dry_run:
        print(json.dumps(launch, ensure_ascii=False, indent=2))
        return

    startup_log("Runtime patches ready.")

    model_dir = Path(launch["model_dir"])
    try:
        launch["model_files"] = model_files(model_dir)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.error(f"完整模型检查失败：{exc}")

    output_dir = Path(launch["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    launch["pid"] = os.getpid()
    launch["started_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    launch["log"] = str(output_dir / "service.log")
    manifest_path = output_dir / "launch-manifest.json"
    manifest_path.write_text(json.dumps(launch, ensure_ascii=False, indent=1) + "\n")

    effective = launch["effective"]
    health_host = effective["host"]
    if health_host in ("0.0.0.0", "::"):
        health_host = "127.0.0.1"
    startup_log(
        f"Effective config: config={launch['config']} "
        f"tensor_parallel_size={effective['tensor_parallel_size']} "
        f"max_model_len={effective['max_model_len']} port={effective['port']}"
    )
    startup_log(f"Launch manifest: {manifest_path}")
    startup_log(f"Service log: {launch['log']} (run 'tail -f' in another terminal)")
    startup_log(
        f"Readiness check: curl -sf http://{health_host}:{effective['port']}/health"
    )
    startup_log(
        f"Loading model and starting vLLM (pid={os.getpid()}); "
        "progress stays on this terminal, full logs go to the service log."
    )

    os.environ.update(launch["service_environment"])
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("TASK_QUEUE_ENABLE", "2")

    log_fd = os.open(launch["log"], os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    _start_startup_progress(launch["log"], health_host, effective["port"])
    os.dup2(log_fd, 1)
    os.dup2(log_fd, 2)
    print(json.dumps({"launch_manifest": str(manifest_path), **launch},
                     ensure_ascii=False), flush=True)
    os.execv(sys.executable, launch["argv"])


if __name__ == "__main__":
    main()
