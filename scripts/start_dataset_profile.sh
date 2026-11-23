#!/usr/bin/env bash
set -euo pipefail

repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
profile="${1:-}"
if [[ $# -gt 0 ]]; then
    shift
fi

case "$profile" in
    od-s1|od-s1-k32|od-s1-k56)
        options=(--config s1)
        ;;
    od-s2)
        options=(--config s1 --max-num-seqs 16 --max-num-batched-tokens 16384)
        ;;
    erqa-c1|erqa-s2)
        batch=1
        if [[ "$profile" == "erqa-s2" ]]; then
            batch=8
        fi
        options=(
            --config s1
            --max-model-len 16384
            --max-num-seqs "$batch"
            --max-num-batched-tokens 16384
            --gpu-memory-utilization 0.85
            --encoder-cache-tokens 262144
            --limit-mm-per-prompt '{"image":16,"video":0}'
            --mm-processor-kwargs '{"images_kwargs":{"size":{"shortest_edge":65536,"longest_edge":1048576}},"videos_kwargs":{"size":{"shortest_edge":4096,"longest_edge":65536},"do_sample_frames":false}}'
        )
        ;;
    *)
        printf '%s\n' 'Usage: scripts/start_dataset_profile.sh {od-s1|od-s1-k32|od-s1-k56|od-s2|erqa-c1|erqa-s2} [start.sh arguments]' >&2
        exit 2
        ;;
esac

case "${CLEF_PRIVATE_JPEG_CACHE_GIB:-0}" in
    0)
        ;;
    2)
        if [[ "$profile" != "erqa-c1" ]]; then
            printf '%s\n' 'The JPEG cache requires the erqa-c1 profile.' >&2
            exit 2
        fi
        export PYTHONPATH="$repo/runtime/jpeg_decode_cache${PYTHONPATH:+:$PYTHONPATH}"
        ;;
    *)
        printf '%s\n' 'CLEF_PRIVATE_JPEG_CACHE_GIB accepts 0 or 2.' >&2
        exit 2
        ;;
esac

if [[ "$profile" == "od-s1-k32" || "$profile" == "od-s1-k56" ]]; then
    export CLEF_PRIVATE_ACTIVATION_BOUNDARY="${profile##*-k}"
    python_bin="${PYTHON_BIN:-python3}"
    rank_bytes="${CLEF_PRIVATE_K32_RANK_BYTES:-22011707392}"
    if [[ ! "$rank_bytes" =~ ^[1-9][0-9]{0,10}$ ]] || (( rank_bytes > 22011707392 )); then
        printf '%s\n' 'Activation cache rank capacity must be 1–22011707392 bytes.' >&2
        exit 2
    fi
    namespace="$("$python_bin" -I -S -B -c 'import secrets; print(secrets.token_hex(32))')"
    receipts="$(mktemp -d "${TMPDIR:-/tmp}/clef-k32-receipts.XXXXXXXX")"
    export CLEF_PRIVATE_K32_DIAG_DIR="$receipts"
    export CLEF_PRIVATE_K32_NAMESPACE="$namespace"
    export CLEF_PRIVATE_K32_RANK_BYTES="$rank_bytes"
    export CLEF_PRIVATE_K32_FORMAL=1 CLEF_PRIVATE_K32_AUDIT=0 CLEF_PRIVATE_K32=cache
    unset CLEF_PRIVATE_K32_CONTROL
    export PYTHONPATH="$repo/runtime/k32_activation_cache${PYTHONPATH:+:$PYTHONPATH}"
    printf 'Activation cache receipts: %s\n' "$receipts" >&2
fi

exec bash "$repo/Clef-27B/start.sh" "${options[@]}" "$@"
