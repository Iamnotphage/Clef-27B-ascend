#!/usr/bin/env python3
"""Record fixed-version package checks and real dual-NPU matmul evidence.

Run with the selected Python after sourcing CANN and ATB, for example:
  python scripts/verify_npu_environment.py --stage torch --output /path/smoke.json
  python scripts/verify_npu_environment.py --stage runtime --require-extensions \
      --output /path/final-environment.json

The checks use identical, dtype-quantized inputs on CPU and each NPU. Timings
describe a single synchronized operation and serve as smoke-test evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone


# CPU aarch64 wheels can report either a bare version or a +cpu local version.
# An actual CUDA/ROCm build is rejected separately through torch.version.
BASE_PACKAGES = (
    ("torch", "torch", "2.10.0", True),
    ("torch-npu", "torch_npu", "2.10.0.post7", False),
    ("torchvision", "torchvision", "0.25.0", True),
    ("torchaudio", "torchaudio", "2.10.0", True),
)
RUNTIME_PACKAGES = (
    ("transformers", "transformers", "5.19.0", False),
    ("triton-ascend", "triton", "3.6.0.post2", False),
)
EXTENSION_PACKAGES = (
    ("flash-linear-attention-npu-a3", "fla_npu", "26.9.1", False),
    ("flash_attn_npu", "flash_attn_npu", "0.5.0", False),
    ("vllm", "vllm", None, False),
    ("vllm-ascend", "vllm_ascend", None, False),
)
TOLERANCES = {
    "float16": {"atol": 5e-4, "rtol": 1e-3, "relative_l2_max": 1e-3},
    "bfloat16": {"atol": 4e-3, "rtol": 8e-3, "relative_l2_max": 8e-3},
}


def inspect_package(distribution, module_name, expected, cpu_local, required):
    result = {
        "distribution": distribution,
        "module": module_name,
        "required": required,
        "expected_version": expected,
    }
    try:
        dist = importlib.metadata.distribution(distribution)
        actual = dist.version
        result.update(version=actual, distribution_path=str(dist.locate_file("")))
        result["version_passed"] = expected is None or actual == expected or (
            cpu_local and actual == expected + "+cpu"
        )
    except Exception as exc:
        result.update(version_passed=False, metadata_error=f"{type(exc).__name__}: {exc}")
    # Import the actual runtime module to discover missing shared libraries.
    # Extension imports are deferred until explicitly required by the final check.
    if required:
        try:
            module = importlib.import_module(module_name)
            result.update(
                import_passed=True,
                module_path=getattr(module, "__file__", None),
                module_version=str(getattr(module, "__version__", "unreported")),
            )
        except Exception as exc:
            result.update(import_passed=False, import_error=f"{type(exc).__name__}: {exc}")
    else:
        result["import_passed"] = None
    result["passed"] = bool(result["version_passed"] and result.get("import_passed", False))
    result["validated"] = bool(required)
    return result


def pip_check():
    try:
        completed = subprocess.run(
            [sys.executable, "-m", "pip", "check"],
            text=True,
            capture_output=True,
            timeout=120,
            check=False,
        )
        return {
            "passed": completed.returncode == 0,
            "returncode": completed.returncode,
            "stdout": completed.stdout.strip(),
            "stderr": completed.stderr.strip(),
        }
    except Exception as exc:
        return {"passed": False, "error": f"{type(exc).__name__}: {exc}"}


def matmul_check(torch, device_index, dtype_name, inputs, reference):
    result = {
        "device": device_index,
        "dtype": dtype_name,
        "shape": list(reference.shape),
        "tolerances": TOLERANCES[dtype_name],
        "passed": False,
        "synchronization_completed": False,
    }
    try:
        torch.npu.set_device(device_index)
        result["device_name"] = torch.npu.get_device_name(device_index)
        left = inputs[0].to(f"npu:{device_index}")
        right = inputs[1].to(f"npu:{device_index}")
        torch.npu.synchronize(device_index)
        started = time.perf_counter()
        output_npu = torch.mm(left, right)
        torch.npu.synchronize(device_index)
        result.update(
            synchronized_matmul_ms=(time.perf_counter() - started) * 1000,
            synchronization_completed=True,
            output_device=str(output_npu.device),
            output_dtype=str(output_npu.dtype),
        )
        output_placement_passed = output_npu.device.type == "npu" and output_npu.device.index == device_index
        result["output_placement_passed"] = output_placement_passed
        output = output_npu.float().cpu()
        finite = bool(torch.isfinite(output).all().item())
        reference_finite = bool(torch.isfinite(reference).all().item())
        result.update(
            output_finite=finite,
            output_nan_count=int(torch.isnan(output).sum().item()),
            output_inf_count=int(torch.isinf(output).sum().item()),
            reference_finite=reference_finite,
        )
        if finite and reference_finite:
            error = output - reference
            abs_error = error.abs()
            reference_rms = float(reference.square().mean().sqrt().item())
            rmse = float(error.square().mean().sqrt().item())
            relative_l2 = rmse / max(reference_rms, 1e-30)
            tolerances = TOLERANCES[dtype_name]
            element_pass = abs_error <= tolerances["atol"] + tolerances["rtol"] * reference.abs()
            mismatch_count = int((~element_pass).sum().item())
            result.update(
                max_abs_error=float(abs_error.max().item()),
                mean_abs_error=float(abs_error.mean().item()),
                rmse=rmse,
                reference_rms=reference_rms,
                relative_l2=relative_l2,
                elementwise_mismatch_count=mismatch_count,
                elementwise_passed=mismatch_count == 0,
                passed=output_placement_passed and mismatch_count == 0 and relative_l2 <= tolerances["relative_l2_max"],
            )
        del left, right, output_npu, output
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def check_npu(report, args):
    import torch
    import torch_npu  # noqa: F401; registers torch.npu and its runtime

    report["torch_build"] = {
        "cuda": torch.version.cuda,
        "hip": torch.version.hip,
        "cpu_build_passed": torch.version.cuda is None and torch.version.hip is None,
    }
    report["npu"].update(
        available=bool(torch.npu.is_available()),
        device_count=int(torch.npu.device_count()),
    )
    if not report["npu"]["available"]:
        raise RuntimeError("torch.npu.is_available() returned False")
    if report["npu"]["device_count"] != 2:
        raise RuntimeError(f"Expected 2 devices; found {report['npu']['device_count']}")
    torch.set_num_threads(args.cpu_threads)
    torch.set_float32_matmul_precision("highest")
    if hasattr(torch.npu, "matmul") and hasattr(torch.npu.matmul, "allow_hf32"):
        torch.npu.matmul.allow_hf32 = False
        report["npu"]["matmul_allow_hf32"] = bool(torch.npu.matmul.allow_hf32)
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    # Scaling keeps the output RMS near 1, making absolute tolerances explicit.
    left_fp32 = torch.randn(args.size, args.size, generator=generator) / math.sqrt(args.size)
    right_fp32 = torch.randn(args.size, args.size, generator=generator)
    with torch.inference_mode():
        for dtype_name in ("float16", "bfloat16"):
            dtype = getattr(torch, dtype_name)
            inputs = (left_fp32.to(dtype), right_fp32.to(dtype))
            # Quantize first so CPU and NPU receive the same representable values.
            reference = torch.mm(inputs[0].float(), inputs[1].float())
            for device_index in (0, 1):
                print(f"Checking npu:{device_index} {dtype_name} {args.size}x{args.size}", file=sys.stderr)
                report["matmul"].append(matmul_check(torch, device_index, dtype_name, inputs, reference))
    report["npu"]["passed"] = len(report["matmul"]) == 4 and all(x["passed"] for x in report["matmul"])


def loaded_libraries():
    maps = Path("/proc/self/maps")
    if not maps.exists():
        return []
    paths = set()
    for line in maps.read_text().splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) == 6:
            path = fields[-1]
            if path.startswith("/") and any(word in path.lower() for word in ("ascend", "torch", "libc10", "libatb")):
                paths.add(path)
    return sorted(paths)


def check_extensions(model_dir):
    import torch
    import vllm
    import vllm_ascend
    from vllm.platforms import current_platform
    from vllm.config import ModelConfig
    from vllm.model_executor.models import ModelRegistry
    from vllm_ascend.utils import enable_custom_op

    vllm_ascend.register_model()
    model_config = ModelConfig(model=str(model_dir), runner="pooling", dtype="bfloat16", max_model_len=1024)
    model_class, architecture = ModelRegistry.resolve_model_cls(model_config.architectures, model_config)
    enabled = bool(enable_custom_op())
    operators = sorted(name for name in torch._C._dispatch_get_all_op_names() if name.startswith("_C_ascend::"))
    sources = {}
    for module, expected in (
        (vllm, "24420b7a341af37a9bc3186e6fc6062512e0aa58"),
        (vllm_ascend, "89eb8a5e7a76d085b453a1aeadb6883e12d5ea2d"),
    ):
        root = Path(module.__file__).resolve().parents[1]
        marker = root / ".clef-source-commit"
        if (root / ".git").exists():
            head = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
            head_source = "git"
        elif marker.is_file():
            # scripts/setup_env.sh archive 回退：固定提交 tarball 无 .git，
            # 以提取时写入的 marker 为准；不能向父仓库误读 HEAD。
            head = marker.read_text(encoding="utf-8").strip()
            head_source = "marker"
        else:
            head = None
            head_source = "missing"
        sources[module.__name__] = {
            "path": str(root), "head": head, "head_source": head_source,
            "expected_head": expected, "passed": head == expected,
        }
    return {
        "platform": f"{type(current_platform).__module__}.{type(current_platform).__name__}",
        "architecture": architecture,
        "configuration_architectures": model_config.architectures,
        "model_class": f"{model_class.__module__}.{model_class.__name__}",
        "custom_ops_enabled": enabled,
        "custom_operator_count": len(operators),
        "custom_operators": operators,
        "ascend_custom_opp_path": os.environ.get("ASCEND_CUSTOM_OPP_PATH"),
        "sources": sources,
        "passed": current_platform.device_type == "npu" and enabled and bool(operators)
        and architecture == "ClefFlashForJointSchema"
        and all(source["passed"] for source in sources.values()),
    }


def write_report(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", choices=("torch", "runtime"), default="torch")
    parser.add_argument("--require-extensions", action="store_true", help="Also import and require FLA, vLLM and vLLM-Ascend; flash_attn_npu is recorded but optional")
    parser.add_argument("--require-flash-attn", action="store_true", help="Require flash_attn_npu as well (it is not built by default; use setup_env.sh --with-flash-attn)")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--size", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=20261112)
    parser.add_argument("--cpu-threads", type=int, default=8)
    parser.add_argument("--model-dir", type=Path, default=Path(os.environ.get("MODEL_DIR", "/workspace/yangchen/models/Clef-27B")))
    args = parser.parse_args()
    if args.size < 1 or args.cpu_threads < 1:
        parser.error("--size and --cpu-threads must be positive")
    if args.require_extensions and args.stage != "runtime":
        parser.error("--require-extensions requires --stage runtime")
    if args.require_flash_attn and not args.require_extensions:
        parser.error("--require-flash-attn requires --require-extensions")
    report = {
        "schema_version": 1,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "stage": args.stage,
        "require_extensions": args.require_extensions,
        "require_flash_attn": args.require_flash_attn,
        "script": {"path": str(Path(__file__).resolve()), "sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
        "passed": False,
        "python": {
            "executable": sys.executable,
            "version": sys.version,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "prefix": sys.prefix,
            "base_prefix": sys.base_prefix,
            "passed": sys.version_info[:2] == (3, 12) and platform.machine() == "aarch64" and sys.prefix != sys.base_prefix,
        },
        "input": {
            "size": args.size,
            "baseline_shape_passed": args.size == 2048,
            "seed": args.seed,
            "cpu_threads": args.cpu_threads,
            "reference": "CPU FP32 matmul of dtype-quantized inputs, identical on both NPUs",
            "distribution": "A~N(0,1)/sqrt(size), B~N(0,1)",
        },
        "npu": {"expected_device_count": 2, "passed": False},
        "packages": [],
        "matmul": [],
    }
    try:
        required = BASE_PACKAGES + (RUNTIME_PACKAGES if args.stage == "runtime" else ())
        for package in required:
            report["packages"].append(inspect_package(*package, required=True))
        for package in EXTENSION_PACKAGES:
            required = args.require_extensions
            if package[0] == "flash_attn_npu" and not args.require_flash_attn:
                # 默认不构建 flash_attn_npu；--require-extensions 只记录状态。
                required = False
            report["packages"].append(inspect_package(*package, required=required))
        report["pip_check"] = pip_check()
        try:
            check_npu(report, args)
        except Exception as exc:
            report["npu"]["error"] = f"{type(exc).__name__}: {exc}"
        if args.require_extensions:
            try:
                report["extensions"] = check_extensions(args.model_dir)
            except Exception as exc:
                report["extensions"] = {"passed": False, "error": f"{type(exc).__name__}: {exc}"}
        report["loaded_libraries"] = loaded_libraries()
        report["passed"] = bool(
            report["python"]["passed"]
            and report["input"]["baseline_shape_passed"]
            and report["pip_check"]["passed"]
            and report["npu"]["passed"]
            and report.get("torch_build", {}).get("cpu_build_passed", False)
            and all(p["passed"] for p in report["packages"] if p["required"])
            and (not args.require_extensions or report["extensions"]["passed"])
        )
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
    report["finished_at"] = datetime.now(timezone.utc).isoformat()
    write_report(args.output, report)
    print(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
