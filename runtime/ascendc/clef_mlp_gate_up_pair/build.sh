#!/usr/bin/env bash
# Build the isolated experimental package; never installs it into CANN/service.
set -eo pipefail
src=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$src"
test -n "${CATLASS_ROOT:?fixed eec373fb private source is required}"
cmake --preset=default -DCATLASS_INCLUDE_DIR="${CATLASS_ROOT}/include"
cmake --build build_out --target binary package -j"${MAX_JOBS:-4}"
