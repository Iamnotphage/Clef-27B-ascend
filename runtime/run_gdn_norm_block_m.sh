#!/usr/bin/env bash
# One-shot runner for the GDN output-norm BLOCK_M probe.
#
# Execution owner: 21.  This template preserves the runner fix from the second
# attempt: activate_env.sh is sourced in this shell *before* any Python starts,
# so nohup/timeout children inherit CANN's LD_LIBRARY_PATH.  It refuses an
# existing output directory and always attempts the revert.
#
# Usage: bash runtime/run_gdn_norm_block_m.sh <new-empty-output-dir>
#        (optionally launched through nohup/timeout; the environment is sourced
#         inside this script before Python starts)
set -u

REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
TREE="${TREE:-/workspace/yangchen/work/src/vllm-ascend}"
ENV_FILE="${CLEF_ENV_FILE:-/workspace/yangchen/Clef-27B-ascend/scripts/activate_env.sh}"
POSTIMAGE_SHA=a1093e1290afd48b546830a9ae0639f945910f44f9e22f12111b67aa72f8fe6c
OUT="${1:-}"

[[ -n "$OUT" ]] || { echo "usage: $0 <new-empty-output-dir>" >&2; exit 2; }
[[ -e "$OUT" ]] && { echo "refusing existing output dir: $OUT" >&2; exit 2; }

source "$ENV_FILE" >/dev/null 2>&1 || { echo "source failed: $ENV_FILE" >&2; exit 3; }
PY="${PYTHON_BIN:-}"
[[ -x "$PY" ]] || { echo "PYTHON_BIN is missing: ${PY:-<unset>}" >&2; exit 3; }

APPLY="$REPO/runtime/apply_gdn_norm_block_m.py"
PROBE="$REPO/runtime/wave3_gdn_norm_block_m_probe.py"
ENVCHECK="$REPO/runtime/gdn_norm_block_m_env_check.sh"
mkdir -p "$OUT" || exit 2

echo "utc=$(date -Is) repo=$REPO tree=$TREE out=$OUT"
echo "apply_sha=$(sha256sum "$APPLY" | cut -d' ' -f1)"
echo "probe_sha=$(sha256sum "$PROBE" | cut -d' ' -f1)"
echo "env_file=$ENV_FILE python=$PY"

bash "$ENVCHECK"
env_rc=$?
echo "ENV_CHECK_RC=$env_rc"
[[ "$env_rc" -eq 0 ]] || exit 3

"$PY" "$APPLY" --tree "$TREE" --check
check_rc=$?
echo "CHECK_RC=$check_rc"
[[ "$check_rc" -eq 0 ]] || exit 3

"$PY" "$APPLY" --tree "$TREE" --apply --manifest "$OUT/apply.json"
apply_rc=$?
echo "APPLY_RC=$apply_rc"
if [[ "$apply_rc" -ne 0 ]]; then
  "$PY" "$APPLY" --tree "$TREE" --revert --manifest "$OUT/revert.json"
  echo "REVERT_RC=$?"
  exit "$apply_rc"
fi

"$PY" "$PROBE" --tree "$TREE" --expect-source-sha "$POSTIMAGE_SHA" --output "$OUT/probe.json"
probe_rc=$?
echo "PROBE_RC=$probe_rc"

"$PY" "$APPLY" --tree "$TREE" --revert --manifest "$OUT/revert.json"
revert_rc=$?
echo "REVERT_RC=$revert_rc"

[[ "$probe_rc" -eq 0 && "$revert_rc" -eq 0 ]] || exit 1
exit 0
