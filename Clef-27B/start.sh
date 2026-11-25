#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------------------------------------
# 默认生效的核心开关（: "${VAR:=default}" 只填空值，显式设置永远优先）
# CANN / 主机调度
: "${OMP_NUM_THREADS:=1}"; export OMP_NUM_THREADS
: "${TASK_QUEUE_ENABLE:=2}"; export TASK_QUEUE_ENABLE
: "${ACLNN_CACHE_LIMIT:=1000000}"; export ACLNN_CACHE_LIMIT
# Tier 2 前缀状态缓存老化窗口（观测次数；单数据集每轮 ≤418 次观测）
: "${CLEF_TIER2_ACTIVE_WINDOW:=500}"; export CLEF_TIER2_ACTIVE_WINDOW
# 以下默认值由 inference.py 在对应安装器校验通过后按 --config 写入
# （s1 的有效值；custom 不启用这些优化）：
#   CLEF_PREFIX_STATE_CACHE=1              Tier 2 前缀状态缓存（s1/s2 默认 on，
#                                          custom 默认 off；=0 可运行时关闭）
#   CLEF_MLP_BF16_NZ=1                     MLP BF16 NZ（GDN v4 补丁内）
#   CLEF_GDN_DECISION_PREFILL=1 / CLEF_GDN_NO_RECOMPUTE=1
#   CLEF_GDN_CONV_QKV=1 / CLEF_GDN_CONV_QKV_VARLEN=1 / CLEF_GDN_CHUNK128=1
#   CLEF_JOINT_SINGLE_FIELD=1 / CLEF_JOINT_FUSED_ATTN=1
# CLEF_PREFIX_CACHE_RANK_BYTES 与 CLEF_TIER2_RING_SLOTS 不在此写死：
#   未显式设置时由 inference.py 按容器静态 RAM 自动钳制——128 GiB 主机保持
#   32 GiB/rank、8 slots；48–112 GiB 降为 2 slots 并按 (RAM-36 GiB)/2 收紧；
#   ≤48 GiB 自动关闭 Tier 2（不分配 PinnedRing）。结果见 launch-manifest.json
#   的 tiered_cache.memory。
# ---------------------------------------------------------------------------

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
python_bin="${PYTHON_BIN:-python3}"
if ! command -v "$python_bin" >/dev/null 2>&1; then
  printf 'Python 解释器不可用：%s\n请加载 scripts/activate_env.sh 或设置 PYTHON_BIN。\n' "$python_bin" >&2
  exit 127
fi
export PYTHONUNBUFFERED=1
if [[ -n "${CLEF_BIND_CPUS:-}" ]]; then
  exec "$python_bin" "$script_dir/../runtime/clef_bind_supervisor.py" \
    "$python_bin" "$script_dir/inference.py" "$@"
fi
exec "$python_bin" "$script_dir/inference.py" "$@"
