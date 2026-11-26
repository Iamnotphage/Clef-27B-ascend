#!/usr/bin/env bash
set -euo pipefail

# 固定运行环境（如需调整，直接改这里的值）
export OMP_NUM_THREADS=1
export TASK_QUEUE_ENABLE=2
export ACLNN_CACHE_LIMIT=1000000
# 单数据集 S1 可改 3000；同一进程 OD→ERQA 交接建议保持 500。
export CLEF_TIER2_ACTIVE_WINDOW=500
export PYTHONUNBUFFERED=1

# CLEF_GDN_* / CLEF_JOINT_* / CLEF_PREFIX_STATE_CACHE 等运行期补丁开关
# 由 inference.py 按 --config 写入；Tier 2 容量按容器 RAM 自动钳制。

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
python_bin="${PYTHON_BIN:-}"
if [[ -z "$python_bin" ]]; then
  if [[ -x "$script_dir/.venv/bin/python" ]]; then
    python_bin="$script_dir/.venv/bin/python"
  else
    python_bin=python3
  fi
fi
if ! command -v "$python_bin" >/dev/null 2>&1; then
  printf 'Python 解释器不可用：%s\n请先运行 bash scripts/setup_env.sh（项目内 .venv），或 source scripts/activate_env.sh / 设置 PYTHON_BIN。\n' "$python_bin" >&2
  exit 127
fi
if [[ -n "${CLEF_BIND_CPUS:-}" ]]; then
  exec "$python_bin" "$script_dir/src/runtime/clef_bind_supervisor.py" \
    "$python_bin" "$script_dir/inference.py" "$@"
fi
exec "$python_bin" "$script_dir/inference.py" "$@"
