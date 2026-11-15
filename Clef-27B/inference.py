"""Launch the pinned vLLM-Ascend structured-decisions service.

The competition entry is ``bash start.sh --model-dir <dir> --output-dir <dir>``.
``--output-dir`` receives the launch manifest, the service log and runtime
artifacts. Extra vLLM arguments go after ``--``; both ``--flag value`` and
``--flag=value`` are accepted, and a passthrough value overrides the launcher
default for the same flag. The manifest records the effective value and its
source for every configurable flag.
"""

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

DEFAULT_MODEL_DIR = "/workspace/yangchen/models/Clef-27B"
SERVED_MODEL_NAME = "clef"
PROTOCOL_VERSION = "2026-11-13"
REPO_ROOT = Path(__file__).resolve().parent.parent
GDN_INSTALLER = REPO_ROOT / "runtime" / "apply_gdn_decision_patch.py"
MM_PROCESSOR_KWARGS = (
    '{"images_kwargs":{"size":{"shortest_edge":65536,"longest_edge":65536}},'
    '"videos_kwargs":{"size":{"shortest_edge":4096,"longest_edge":65536},'
    '"do_sample_frames":false}}'
)

CONFIGS = {
    "custom": {},
    "s1": {
        "tensor_parallel_size": 2,
        "max_model_len": 4096,
        "max_num_seqs": 1,
        "max_num_batched_tokens": 4096,
        "limit_mm_per_prompt": '{"image":1,"video":1}',
        "schema_token_cache": "off",
        "mm_processor_kwargs": MM_PROCESSOR_KWARGS,
        "gdn_data_path": {
            "bsnd": True,
            "skip_initial_state": True,
            "skip_conv_state": True,
            "preprocess_fusion": False,
        },
    },
    "s2": {
        "tensor_parallel_size": 2,
        "max_model_len": 4096,
        "max_num_seqs": 4,
        "max_num_batched_tokens": 4096,
        "limit_mm_per_prompt": '{"image":1,"video":1}',
        "schema_token_cache": "on",
        "mm_processor_kwargs": MM_PROCESSOR_KWARGS,
        "gdn_data_path": {
            "bsnd": True,
            "skip_initial_state": True,
            "skip_conv_state": True,
            "preprocess_fusion": False,
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
                try:
                    head = subprocess.run(
                        ["git", "-C", str(pkg_dir.parent), "rev-parse", "HEAD"],
                        capture_output=True, text=True, timeout=10)
                    dirty = subprocess.run(
                        ["git", "-C", str(pkg_dir.parent), "status", "--porcelain"],
                        capture_output=True, text=True, timeout=10)
                    if head.returncode == 0:
                        entry["git_head"] = head.stdout.strip()
                        entry["worktree_dirty"] = bool(dirty.stdout.strip())
                        entry["dirty_files"] = len(dirty.stdout.splitlines())
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


def gdn_decision_prefill(config, apply=True):
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
        result = subprocess.run(
            [sys.executable, str(GDN_INSTALLER), flag],
            capture_output=True, text=True, check=False)
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
        complete = all(state == "patched" for state in statuses.values()) and flag_state == "current"
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
            "S2 配置要求 schema cache v2，但固定源码中的版本为 "
            f"{version}（{strategies}）。先运行 runtime/apply_schema_cache_patch.py --apply。"
        )
    return enabled, version, strategies


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

    gdn_decision = gdn_decision_prefill(config, apply=not getattr(args, "dry_run", False))

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
    }
    if gdn_decision["enabled"]:
        environment["CLEF_GDN_DECISION_PREFILL"] = "1"
        environment["CLEF_GDN_NO_RECOMPUTE"] = os.environ.get(
            "CLEF_GDN_NO_RECOMPUTE", "1")
    data_path, data_env = gdn_data_path(config, gdn_decision["enabled"])
    gdn_decision["data_path"] = data_path
    environment.update(data_env)
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
    parser.add_argument("--config", choices=sorted(CONFIGS), default="s1",
                        help="s1（默认，最终采用）：TP2/4096/seqs1/缓存关；"
                             "s2：TP2/4096/seqs4/缓存 v2 开；custom：TP1/1024 保守默认。")
    parser.add_argument("--tensor-parallel-size", type=int, default=None)
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--max-num-seqs", "--seqs", dest="max_num_seqs", type=int, default=None)
    parser.add_argument("--max-num-batched-tokens", type=int, default=None)
    parser.add_argument("--gpu-memory-utilization", type=float, default=None)
    parser.add_argument("--limit-mm-per-prompt", default=None,
                        help='每请求多模态上限 JSON，如 \'{"image":1,"video":0}\'。')
    parser.add_argument("--mm-processor-kwargs", default=None,
                        help="多模态处理器参数 JSON，可覆盖 s1/s2 默认值。")
    parser.add_argument("--schema-token-cache", choices=("off", "on"), default=None,
                        help="显式覆盖 schema token 缓存；on 要求已安装 v2 补丁。")
    parser.add_argument("--distributed-executor-backend", default=None)
    parser.add_argument("--dry-run", action="store_true", help="输出启动参数 JSON。")
    parser.add_argument("vllm_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = str(Path(args.model_dir).expanduser().resolve() / "output")

    try:
        launch = build_launch(args)
    except ValueError as exc:
        parser.error(str(exc))

    if args.dry_run:
        print(json.dumps(launch, ensure_ascii=False, indent=2))
        return

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

    os.environ.update(launch["service_environment"])
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("TASK_QUEUE_ENABLE", "1")

    log_fd = os.open(launch["log"], os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    os.dup2(log_fd, 1)
    os.dup2(log_fd, 2)
    print(json.dumps({"launch_manifest": str(manifest_path), **launch},
                     ensure_ascii=False), flush=True)
    os.execv(sys.executable, launch["argv"])


if __name__ == "__main__":
    main()
