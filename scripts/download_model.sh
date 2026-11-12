#!/usr/bin/env bash
set -euo pipefail

model_dir="${1:-${MODEL_DIR:-/workspace/yangchen/models/Clef-27B}}"
exec modelscope download \
    --model Cloudflare/clef \
    --local_dir "$model_dir"
