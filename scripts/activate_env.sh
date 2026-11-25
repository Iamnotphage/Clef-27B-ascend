#!/usr/bin/env bash
# Source this file in Bash before building or running the A3 baseline.
#
# 运行根默认在项目内（<repo>/.build），不再隐式依赖 /workspace/yangchen：
#   CLEF_WORKSPACE  运行根      [<repo>/.build]
#   CLEF_VENV       虚拟环境    [<repo>/.venv，存在时]
#   CLEF_LOG_DIR    日志目录    [$CLEF_WORKSPACE/logs]
#   XDG_CACHE_HOME  缓存根      [$CLEF_WORKSPACE/cache]
#   MODEL_DIR       模型目录    [$CLEF_WORKSPACE/models/Clef-27B]
# 需要旧布局时显式 export CLEF_WORKSPACE=/workspace/yangchen。
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo '请使用 source scripts/activate_env.sh 加载环境。' >&2
  exit 2
fi

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export CLEF_WORKSPACE="${CLEF_WORKSPACE:-$repo_root/.build}"
# A project-local .venv (scripts/setup_env.sh) wins unless CLEF_VENV is set.
if [[ -z "${CLEF_VENV:-}" && -x "$repo_root/.venv/bin/python" ]]; then
  CLEF_VENV="$repo_root/.venv"
fi
export CLEF_VENV="${CLEF_VENV:-$CLEF_WORKSPACE/venvs/clef-py312}"

# CANN / ATB 环境脚本自动探测：兼容 cann、ascend-toolkit、cann-9.x 布局。
if [[ -z "${CLEF_CANN_ENV:-}" ]]; then
  for candidate in \
    /usr/local/Ascend/cann/set_env.sh \
    /usr/local/Ascend/ascend-toolkit/set_env.sh \
    /usr/local/Ascend/ascend-toolkit/latest/set_env.sh \
    /usr/local/Ascend/cann-*/set_env.sh; do
    if [[ -r "$candidate" ]]; then
      CLEF_CANN_ENV="$candidate"
      break
    fi
  done
fi
if [[ -z "${CLEF_ATB_ENV:-}" ]]; then
  for candidate in \
    /usr/local/Ascend/nnal/atb/set_env.sh \
    /usr/local/Ascend/nnal/atb/latest/set_env.sh; do
    if [[ -r "$candidate" ]]; then
      CLEF_ATB_ENV="$candidate"
      break
    fi
  done
fi
export CLEF_CANN_ENV="${CLEF_CANN_ENV:-/usr/local/Ascend/cann/set_env.sh}"
export CLEF_ATB_ENV="${CLEF_ATB_ENV:-/usr/local/Ascend/nnal/atb/set_env.sh}"

_clef_activate() {
  local env_file had_nounset=0 rc=0
  for env_file in "$CLEF_CANN_ENV" "$CLEF_ATB_ENV" "$CLEF_VENV/bin/activate"; do
    if [[ ! -r "$env_file" ]]; then
      printf '环境文件缺失：%s\n' "$env_file" >&2
      return 1
    fi
  done
  [[ $- == *u* ]] && had_nounset=1
  set +u
  for env_file in "$CLEF_CANN_ENV" "$CLEF_ATB_ENV" "$CLEF_VENV/bin/activate"; do
    source "$env_file" || { rc=$?; break; }
  done
  (( had_nounset )) && set -u
  return "$rc"
}
_clef_activate || { unset -f _clef_activate; return 1; }
unset -f _clef_activate

export MODEL_DIR="${MODEL_DIR:-$CLEF_WORKSPACE/models/Clef-27B}"
export PYTHON_BIN="$CLEF_VENV/bin/python"
export CLEF_LOG_DIR="${CLEF_LOG_DIR:-$CLEF_WORKSPACE/logs}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$CLEF_WORKSPACE/cache}"
export HF_HOME="${HF_HOME:-$XDG_CACHE_HOME/huggingface}"
export TORCH_HOME="${TORCH_HOME:-$XDG_CACHE_HOME/torch}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$XDG_CACHE_HOME/triton}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$XDG_CACHE_HOME/pip}"
export MAX_JOBS="${MAX_JOBS:-8}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
# 2026-11-24 ABBA: mode 2 narrows S1 mean/p50 (~-7 ms) and keeps S2
# neutral; 1 reproduces the earlier baseline.
export TASK_QUEUE_ENABLE="${TASK_QUEUE_ENABLE:-2}"
