#!/usr/bin/env bash
# Run the frozen Decision Index 0.2.1 against the existing Clef service.
# Preparation installs the pinned upstream harness in an isolated client venv.
set -euo pipefail

CLEF_DI_CLIENT_PYTHON="${CLEF_DI_CLIENT_PYTHON:-/workspace/yangchen/venvs/clef-decision-index-py312/bin/python}"
CLEF_DI_DATA="${CLEF_DI_DATA:-/workspace/yangchen/datasets/clef-official-20261113/decision-index}"
CLEF_DI_OUTPUT="${CLEF_DI_OUTPUT:-/workspace/yangchen/logs/official-baseline-20261113/decision-index/full}"
CLEF_DI_BASE_URL="${CLEF_DI_BASE_URL:-http://127.0.0.1:8000}"

export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1
exec "$CLEF_DI_CLIENT_PYTHON" -m decision_index pipeline \
  --edition 0.2.1 \
  --suite-dir "$CLEF_DI_DATA" \
  --engine http \
  --option "base_url=$CLEF_DI_BASE_URL" \
  --option model=clef \
  --option timeout=600 \
  --out "$CLEF_DI_OUTPUT" \
  "$@"
