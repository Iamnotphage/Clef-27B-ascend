#!/bin/bash
# Recommended Clef-27B service launcher for the A3 single-physical-card
# (two logical NPU) boundary. Config: TP2, BF16, eager, max_model_len 4096,
# max_num_seqs configurable (1 for S1/quality, 4 for S2 throughput),
# token budget follows max_model_len, prefix caching and chunked prefill off.
#
# Usage:
#   bash start_recommended.sh --model-dir /path/to/Clef-27B [--output-dir /path/to/out] \
#     [--seqs 1|4] [--port 8000] [--dry-run] [-- <extra vLLM args>]
#
# --dry-run prints the resolved launch JSON and starts nothing.
# Without --dry-run the launch manifest and service log are written into
# --output-dir (default <model-dir>/output).
set -euo pipefail

MODEL_DIR=""
OUTPUT_DIR=""
SEQS="${CLEF_RECOMMENDED_SEQS:-1}"
PORT="${CLEF_RECOMMENDED_PORT:-8000}"
DRY_RUN=false
ORIGINAL_ARGS=("$@")
while [ $# -gt 0 ]; do
  case "$1" in
    --model-dir) MODEL_DIR="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --seqs) SEQS="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --dry-run) DRY_RUN=true; shift ;;
    --) shift; break ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
[ -n "$MODEL_DIR" ] || { echo "--model-dir is required" >&2; exit 2; }
MODEL_DIR="$(cd "$MODEL_DIR" && pwd)"
if [ -z "$OUTPUT_DIR" ]; then
  OUTPUT_DIR="$MODEL_DIR/output"
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# Locate Clef-27B/inference.py: CLEF_INFERENCE_PY override, a copy/symlink next
# to this script (flat deployment dir), or the sibling directory of a repo
# checkout (searching a few levels up).
locate_infer() {
  if [ -n "${CLEF_INFERENCE_PY:-}" ] && [ -f "$CLEF_INFERENCE_PY" ]; then
    echo "$CLEF_INFERENCE_PY"; return
  fi
  if [ -f "$SCRIPT_DIR/inference.py" ]; then
    echo "$SCRIPT_DIR/inference.py"; return
  fi
  local d="$SCRIPT_DIR"
  for _ in 1 2 3 4; do
    if [ -f "$d/Clef-27B/inference.py" ]; then echo "$d/Clef-27B/inference.py"; return; fi
    d="$(dirname "$d")"
  done
  echo ""
}
INFER_PY="$(locate_infer)"
if [ -z "$INFER_PY" ]; then
  echo "inference.py not found (set CLEF_INFERENCE_PY or deploy beside Clef-27B/)" >&2
  exit 1
fi
INFER_PY="$(cd "$(dirname "$INFER_PY")" && pwd)/$(basename "$INFER_PY")"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if $DRY_RUN; then
  exec "$PYTHON_BIN" "$INFER_PY" \
    --model-dir "$MODEL_DIR" \
    --port "$PORT" \
    --tensor-parallel-size 2 \
    --max-model-len 4096 \
    --max-num-seqs "$SEQS" \
    --limit-mm-per-prompt '{"image":1,"video":1}' \
    --dry-run \
    -- \
    --mm-processor-kwargs '{"images_kwargs":{"size":{"shortest_edge":65536,"longest_edge":65536}},"videos_kwargs":{"size":{"shortest_edge":4096,"longest_edge":65536},"do_sample_frames":false}}' \
    --distributed-executor-backend mp
fi

mkdir -p "$OUTPUT_DIR"
"$PYTHON_BIN" - "$OUTPUT_DIR/launch-manifest.json" \
  "$INFER_PY" "$MODEL_DIR" "$OUTPUT_DIR" "$SEQS" "$PORT" \
  ${ORIGINAL_ARGS[@]+"${ORIGINAL_ARGS[@]}"} <<'PYEOF'
import hashlib, json, os, subprocess, sys, time
out, infer_py, model_dir, output_dir, seqs, port = sys.argv[1:7]
launcher_argv = sys.argv[7:]
manifest = {
    "launcher": "runtime/start_recommended.sh",
    "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    "inference_py": infer_py,
    "inference_py_sha256": hashlib.sha256(open(infer_py, "rb").read()).hexdigest(),
    "model_dir": model_dir, "output_dir": output_dir,
    "seqs": seqs, "port": port, "launcher_argv": launcher_argv,
    "environment": {k: os.environ[k] for k in sorted(os.environ)
                    if k in ("PATH", "LD_LIBRARY_PATH", "PYTHONPATH", "PYTHONUNBUFFERED",
                             "VIRTUAL_ENV", "OMP_NUM_THREADS", "TASK_QUEUE_ENABLE",
                             "MODEL_DIR", "HF_HOME", "TORCH_HOME", "TRITON_CACHE_DIR",
                             "XDG_CACHE_HOME") or k.startswith(("CLEF_", "ASCEND_", "ATB_", "HCCL_"))},
}
for name in ("vllm", "vllm_ascend"):
    try:
        import importlib.util
        spec = importlib.util.find_spec(name)
        if spec and spec.submodule_search_locations:
            pkg = list(spec.submodule_search_locations)[0]
            head = subprocess.run(["git", "-C", os.path.dirname(pkg), "rev-parse", "HEAD"],
                                  capture_output=True, text=True, timeout=10)
            if head.returncode == 0:
                manifest.setdefault("software", {})[name] = head.stdout.strip()
    except Exception:
        pass
with open(out, "w") as f:
    json.dump(manifest, f, ensure_ascii=False, indent=1)
    f.write("\n")
PYEOF

# Perf env preserved from the validated baseline service.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export TASK_QUEUE_ENABLE="${TASK_QUEUE_ENABLE:-1}"

echo "launching clef: tp2 seqs=$SEQS port=$PORT model=$MODEL_DIR output=$OUTPUT_DIR"
exec >> "$OUTPUT_DIR/service.log" 2>&1
exec "$PYTHON_BIN" "$INFER_PY" \
  --model-dir "$MODEL_DIR" \
  --port "$PORT" \
  --tensor-parallel-size 2 \
  --max-model-len 4096 \
  --max-num-seqs "$SEQS" \
  --limit-mm-per-prompt '{"image":1,"video":1}' \
  -- \
  --mm-processor-kwargs '{"images_kwargs":{"size":{"shortest_edge":65536,"longest_edge":65536}},"videos_kwargs":{"size":{"shortest_edge":4096,"longest_edge":65536},"do_sample_frames":false}}' \
  --distributed-executor-backend mp
