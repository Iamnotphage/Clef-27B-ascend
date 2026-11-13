#!/bin/bash
# Recommended Clef-27B service launcher for the A3 single-physical-card
# (two logical NPU) boundary. Config: TP2, BF16, eager, max_model_len 4096,
# max_num_seqs configurable (1 for S1/quality, 4 for S2 throughput),
# token budget follows max_model_len, prefix caching and chunked prefill off.
#
# Usage:
#   bash start_recommended.sh --model-dir /path/to/Clef-27B --output-dir /path/to/out [--seqs 1|4] [--port 8000]
set -euo pipefail

MODEL_DIR=""
OUTPUT_DIR=""
SEQS="${CLEF_RECOMMENDED_SEQS:-1}"
PORT="${CLEF_RECOMMENDED_PORT:-8000}"
while [ $# -gt 0 ]; do
  case "$1" in
    --model-dir) MODEL_DIR="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --seqs) SEQS="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
[ -n "$MODEL_DIR" ] || { echo "--model-dir is required" >&2; exit 2; }
MODEL_DIR="$(cd "$MODEL_DIR" && pwd)"
if [ -z "$OUTPUT_DIR" ]; then
  OUTPUT_DIR="$MODEL_DIR/output"
fi
mkdir -p "$OUTPUT_DIR"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

# Perf env preserved from the validated baseline service.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export TASK_QUEUE_ENABLE="${TASK_QUEUE_ENABLE:-1}"

echo "launching clef: tp2 seqs=$SEQS port=$PORT model=$MODEL_DIR output=$OUTPUT_DIR"
exec "$PYTHON_BIN" "$SCRIPT_DIR/inference.py" \
  --model-dir "$MODEL_DIR" \
  --port "$PORT" \
  --tensor-parallel-size 2 \
  --max-model-len 4096 \
  --max-num-seqs "$SEQS" \
  --limit-mm-per-prompt '{"image":1,"video":1}' \
  -- \
  --mm-processor-kwargs '{"images_kwargs":{"size":{"shortest_edge":65536,"longest_edge":65536}},"videos_kwargs":{"size":{"shortest_edge":4096,"longest_edge":65536},"do_sample_frames":false}}' \
  --distributed-executor-backend mp
