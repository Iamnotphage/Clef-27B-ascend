#!/bin/bash
# Recommended Clef-27B launcher for the A3 single-physical-card boundary.
#
# All argument construction lives in inference.py. This wrapper only
# selects the S1 (quality/latency) default; pass --config s2 for the throughput
# candidate, which requires schema-cache v2 and enables it explicitly.
#
# Usage:
#   bash start_recommended.sh --model-dir /path/to/Clef-27B --output-dir /path/to/out \
#     [--config s1|s2] [--port 8000] [--dry-run] [-- <extra vLLM args>]
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [ -n "${CLEF_INFERENCE_PY:-}" ] && [ -f "${CLEF_INFERENCE_PY}" ]; then
  INFER_PY="${CLEF_INFERENCE_PY}"
elif [ -f "$SCRIPT_DIR/../../inference.py" ]; then
  INFER_PY="$SCRIPT_DIR/../../inference.py"
elif [ -f "$SCRIPT_DIR/inference.py" ]; then
  INFER_PY="$SCRIPT_DIR/inference.py"
else
  echo "inference.py not found (set CLEF_INFERENCE_PY or deploy beside the repo root)" >&2
  exit 1
fi
PYTHON_BIN="${PYTHON_BIN:-python3}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Python 解释器不可用：$PYTHON_BIN" >&2
  exit 127
fi

# s1 is the measured quality/latency config; s2 selects seqs4 + schema cache v2.
exec "$PYTHON_BIN" "$INFER_PY" --config "${CLEF_RECOMMENDED_CONFIG:-s1}" "$@"
