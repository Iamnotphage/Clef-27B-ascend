#!/usr/bin/env bash
# Build the isolated experimental package; never installs it into CANN/service.
set -eo pipefail
src=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$src"
cmake --preset=default
cmake --build build_out --target binary package -j"${MAX_JOBS:-4}"
