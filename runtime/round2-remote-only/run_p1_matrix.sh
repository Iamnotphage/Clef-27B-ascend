#!/bin/bash
# Phase 1 exploration matrix: seqs 1/2/4 x concurrency 1/2/4/8, 300 requests each, >=30s window
set -u
for SEQS in 1 2 4; do
  for CONC in 1 2 4 8; do
    bash /workspace/yangchen/logs/tuning/run_p1_point.sh "$SEQS" "$CONC" 300 || echo "POINT FAILED s=$SEQS c=$CONC"
  done
done
echo "MATRIX DONE $(TZ=Asia/Shanghai date +%H:%M:%S)"
