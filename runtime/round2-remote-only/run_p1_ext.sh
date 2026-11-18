#!/bin/bash
set -u
for PT in "4 16" "4 32" "2 16"; do
  set -- $PT
  bash /workspace/yangchen/logs/tuning/run_p1_point.sh "$1" "$2" 300 || echo "POINT FAILED s=$1 c=$2"
done
echo "EXT DONE $(TZ=Asia/Shanghai date +%H:%M:%S)"
