#!/bin/bash
# 3 interleaved stable rounds: baseline s1-c1, candidates s2-c4 and s4-c8
set -u
for ROUND in 1 2 3; do
  for PT in "1 1" "2 4" "4 8"; do
    set -- $PT
    LABEL_SUFFIX="-r${ROUND}" LABEL_PREFIX=p1s bash /workspace/yangchen/logs/tuning/run_p1_point.sh "$1" "$2" 300 || echo "STABLE FAILED s=$1 c=$2 r=$ROUND"
  done
done
echo "STABLE DONE $(TZ=Asia/Shanghai date +%H:%M:%S)"
