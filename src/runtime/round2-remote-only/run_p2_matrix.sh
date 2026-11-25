#!/bin/bash
set -u
LONG=/workspace/yangchen/logs/tuning/sample-long3868.json
# long-input A/B: budget x concurrency at winner seqs4
for TOK in 4096 8192 16384; do
  for CONC in 4 8; do
    bash /workspace/yangchen/logs/tuning/run_p2_point.sh 4 "$TOK" "$CONC" 300 "$LONG" || echo "P2 FAILED s4 t$TOK c$CONC"
  done
done
# fixed307 regression under the largest budget (compare with p1-s4-c8 19.33 rps)
bash /workspace/yangchen/logs/tuning/run_p2_point.sh 4 16384 8 300 none || echo "P2 FAILED 307-reg"
echo "P2 DONE $(TZ=Asia/Shanghai date +%H:%M:%S)"
