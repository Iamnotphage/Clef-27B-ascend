#!/bin/bash
set -u
for CONC in 2 4 8; do
  bash /workspace/yangchen/logs/tuning/run_p3_point.sh "$CONC" 300 1 || echo "P3 FAILED c=$CONC"
done
echo "P3 DONE $(TZ=Asia/Shanghai date +%H:%M:%S)"
