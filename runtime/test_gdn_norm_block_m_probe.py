#!/usr/bin/env python3
"""Pure-CPU tests for the GDN norm BLOCK_M probe helpers.

Imports the probe module only (its top level uses the standard library), then
checks the Triton decorator unwrapping and the four grid rows.  No torch,
torch_npu or vllm_ascend import happens here.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROBE_PATH = HERE / "wave3_gdn_norm_block_m_probe.py"


def load_probe():
    spec = importlib.util.spec_from_file_location("wave3_gdn_norm_block_m_probe", PROBE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class Core:
    device_caches = {}


class Wrapper:
    def __init__(self, fn):
        self.fn = fn


class NoFn:
    pass


class MLIRCompilationError(Exception):
    pass


def compile_failure(message, cause):
    error = MLIRCompilationError(message)
    error.__cause__ = cause
    return error


def main() -> int:
    probe = load_probe()
    failures = []

    def check(name, condition):
        if condition:
            print(f"ok   {name}")
        else:
            print(f"FAIL {name}")
            failures.append(name)

    check("import does not load torch", "torch" not in sys.modules)
    check("import does not load torch_npu", "torch_npu" not in sys.modules)

    core = Core()
    fn, chain = probe.unwrap_jit_function(Wrapper(Wrapper(core)))
    check("unwraps two heuristic layers", fn is core)
    check("records the wrapper chain", chain == ["Wrapper", "Wrapper", "Core"])
    fn, chain = probe.unwrap_jit_function(core)
    check("plain JITFunction is unchanged", fn is core and chain == ["Core"])
    fn, _ = probe.unwrap_jit_function(NoFn())
    check("object without fn returns None", fn is None)
    first, second = Wrapper(None), Wrapper(None)
    first.fn, second.fn = second, first
    fn, _ = probe.unwrap_jit_function(first)
    check("cycle returns None", fn is None)

    expected = {16: (1451, 8), 32: (726, 24), 64: (363, 24), 128: (182, 88)}
    for arm, (blocks, padding) in expected.items():
        row = probe.grid_row(arm)
        check(f"grid BM{arm}", row["blocks"] == blocks and row["padding_rows"] == padding
              and row["requested_grid"] == [blocks, 1] and row["ngroups"] == 1)

    overflow = "ub overflow, requires 2101248 bits while 1572864 bits available"
    bishengir = subprocess.CalledProcessError(
        1, ["/x/bishengir-compile", "/tmp/kernel.mlir", "-o", "/tmp/kernel"])
    error = compile_failure(overflow + "\nThe compiled kernel cache is in /cache/SK3", bishengir)
    check("recognises bishengir UB overflow", probe.is_compile_capacity_failure(error) is True)
    check("extracts cache dir", probe.extract_cache_dir(str(error)) == "/cache/SK3")
    check("rejects device error", probe.is_compile_capacity_failure(RuntimeError("sync failed")) is False)
    check("rejects unrelated compile error",
          probe.is_compile_capacity_failure(compile_failure("other compile error", bishengir)) is False)
    check("rejects overflow text without compiler",
          probe.is_compile_capacity_failure(RuntimeError(overflow)) is False)

    preregistered = [[16, 32, 64, 128], [32, 64, 128, 16], [64, 128, 16, 32]]
    check("projects orders onto OK arms",
          probe.project_orders(preregistered, [64, 16, 32])
          == [[16, 32, 64], [32, 64, 16], [64, 16, 32]])
    check("projects single reference arm",
          probe.project_orders(preregistered, [64]) == [[64], [64], [64]])

    if failures:
        print(f"{len(failures)} failure(s): {', '.join(failures)}")
        return 1
    print("all probe helper checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
