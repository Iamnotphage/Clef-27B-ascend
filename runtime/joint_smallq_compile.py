#!/usr/bin/env python3
"""CPU-only offline compile: explicit A3 target, no driver or extension load.

Compiler device queries are replaced by fixed compile-target metadata. This
checks lowering only; it supplies no device correctness/performance evidence.
Use a scratch TRITON_CACHE_DIR, separate from the service/runtime cache.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import time


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--kernel", type=Path,
                    default=Path(__file__).parent / "joint_smallq/clef_smallq_kernel.py")
    args = ap.parse_args()
    os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
    import triton
    from triton.backends.compiler import GPUTarget
    from triton.backends.ascend import compiler as ascend_compiler
    from triton.compiler import ASTSource

    class OfflineTarget:
        def get_arch(self):
            return "Ascend910_9382"

        def has_device_limit(self):
            return False

    ascend_compiler.NPUUtils = OfflineTarget
    spec = importlib.util.spec_from_file_location("clef_smallq_offline", args.kernel)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    target = GPUTarget("npu", "Ascend910_9382", 0)
    start = time.perf_counter()
    signature = {key: "*bf16" for key in ("Q", "K", "V", "Out")} | {"L": "i32"}
    source = ASTSource(module.clef_smallq_online, signature,
                       {"QS": 1024, "KS": 1024, "VS": 1024,
                        "WIDTH": 1024, "D": 64, "BLOCK": 64})
    compiled = triton.compile(source, target=target)
    print(json.dumps({"length": "runtime i32, no per-length specialization",
                      "seconds": time.perf_counter() - start,
                      "binary_sha256": hashlib.sha256(compiled.asm["npubin"]).hexdigest(),
                      "kernel_sha256": hashlib.sha256(args.kernel.read_bytes()).hexdigest(),
                      "mode": "offline_compile_only_no_device"}), flush=True)


if __name__ == "__main__":
    main()
