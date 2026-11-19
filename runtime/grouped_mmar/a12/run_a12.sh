#!/usr/bin/env bash
# Run the frozen official A12 sample exactly as its usage string: 2 devices, 1
# loop, devices 0,1.  The supervisor owns the outer timeout; this wrapper adds
# a hard cap so a hang cannot outlive the cell.
set -euo pipefail
cd "$(dirname "$0")"
bin=${1:?path to built bin/main required}
seconds=${A12_TIMEOUT:-240}
exec timeout "$seconds" "$bin" 2 1 0,1
