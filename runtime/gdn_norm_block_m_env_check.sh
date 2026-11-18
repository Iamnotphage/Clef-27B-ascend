#!/usr/bin/env bash
# CPU-only loader precheck for the GDN output-norm BLOCK_M probe.
#
# It sources the project environment before any Python starts (CANN set_env,
# ATB set_env and the project venv), then verifies that the torch_npu
# extension's shared-library dependencies resolve.  It never imports torch or
# torch_npu and never touches a device: ``ldd`` only resolves the loader graph,
# and the Python check imports ``sys`` alone.
#
# Usage: bash runtime/gdn_norm_block_m_env_check.sh
# Exit 0 only when libhccl.so resolves from the expected CANN lib64 and every
# dependency of the torch_npu extension is found.
set -u

ENV_FILE="${CLEF_ENV_FILE:-/workspace/yangchen/Clef-27B-ascend/scripts/activate_env.sh}"
EXT_GLOB="${CLEF_TORCH_NPU_EXT_GLOB:-/workspace/yangchen/venvs/clef-py312/lib/python3.12/site-packages/torch_npu/_C*.so}"

fail() { printf 'FAIL: %s\n' "$*" >&2; exit 1; }

[[ -r "$ENV_FILE" ]] || fail "env file is not readable: $ENV_FILE"
# shellcheck disable=SC1090
source "$ENV_FILE" >/dev/null 2>&1 || fail "sourcing failed: $ENV_FILE"

PY="${PYTHON_BIN:-}"
[[ -n "$PY" && -x "$PY" ]] || fail "PYTHON_BIN is missing or not executable: ${PY:-<unset>}"

LD_PATH="${LD_LIBRARY_PATH:-}"
[[ -n "$LD_PATH" ]] || fail "LD_LIBRARY_PATH is empty after sourcing $ENV_FILE"
hccl_dir=""
IFS=':' read -r -a ld_dirs <<< "$LD_PATH"
for dir in "${ld_dirs[@]}"; do
  if [[ -n "$dir" && -r "$dir/libhccl.so" ]]; then
    hccl_dir="$dir"
    break
  fi
done
[[ -n "$hccl_dir" ]] || fail "libhccl.so is not found on any LD_LIBRARY_PATH entry"

ext="$(compgen -G "$EXT_GLOB" | head -n 1 || true)"
[[ -n "$ext" && -r "$ext" ]] || fail "torch_npu extension not found: $EXT_GLOB"
command -v ldd >/dev/null 2>&1 || fail "ldd is not available"
site_packages="$(cd "$(dirname "$ext")/.." && pwd)" || fail "cannot resolve $ext"
torch_lib="$site_packages/torch/lib"
[[ -d "$torch_lib" ]] || fail "torch lib directory not found: $torch_lib"

# torch adds its own lib directory before loading torch_npu; mirror that here
# so the check sees the same loader graph without importing torch.
loader="$(LD_LIBRARY_PATH="$torch_lib:${LD_PATH}" ldd "$ext" 2>&1)" || fail "ldd failed on $ext"
missing="$(printf '%s\n' "$loader" | grep 'not found' || true)"
[[ -z "$missing" ]] || fail "unresolved shared libraries in $ext:
$missing"
hccl_line="$(printf '%s\n' "$loader" | grep 'libhccl.so =>' || true)"
loaded="$(printf '%s\n' "$hccl_line" | awk '{print $3; exit}')"
[[ -n "$loaded" ]] || fail "libhccl.so is not resolved by the loader"
[[ "$(readlink -f "$loaded")" == "$(readlink -f "$hccl_dir/libhccl.so")" ]] \
  || fail "libhccl.so resolves to $loaded, not $hccl_dir/libhccl.so"

"$PY" - <<'PY'
import sys
if sys.version_info[:2] != (3, 12):
    raise SystemExit(f"unexpected Python: {sys.version.split()[0]}")
print(f"python={sys.version.split()[0]}")
PY

printf 'OK env_file=%s\n' "$ENV_FILE"
printf 'OK python=%s\n' "$PY"
printf 'OK hccl_dir=%s\n' "$hccl_dir"
printf 'OK torch_lib=%s\n' "$torch_lib"
printf 'OK torch_npu_extension=%s\n' "$ext"
printf 'OK hccl=%s\n' "$hccl_line"
printf 'OK loader=all dependencies resolved (no torch/torch_npu import)\n'
