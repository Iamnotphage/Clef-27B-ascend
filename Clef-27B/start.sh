#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
python_bin="${PYTHON_BIN:-python3}"
if ! command -v "$python_bin" >/dev/null 2>&1; then
  printf 'Python 解释器不可用：%s\n请加载 scripts/activate_env.sh 或设置 PYTHON_BIN。\n' "$python_bin" >&2
  exit 127
fi
export PYTHONUNBUFFERED=1
exec "$python_bin" "$script_dir/inference.py" "$@"
