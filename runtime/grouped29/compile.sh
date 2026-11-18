#!/usr/bin/env bash
# AOT compilation/link only. Does not import/load this extension or use a device.
set -euo pipefail
cd "$(dirname "$0")"
site=/workspace/yangchen/venvs/clef-py312/lib/python3.12/site-packages
cann=/usr/local/Ascend/cann-9.2.1/aarch64-linux
out=${1:?absolute output .so path required}
[[ "$out" == /* && "$out" != "$site"/* && "$out" != /usr/local/Ascend/* ]]
g++ -std=c++17 -O2 -fPIC -shared -D_GLIBCXX_USE_CXX11_ABI=1 -Wno-deprecated-declarations \
  -I"$site/torch/include" -I"$site/torch/include/torch/csrc/api/include" \
  -I"$site/torch_npu/include" -I"$site" -I"$cann/include" \
  binding.cpp -L"$site/torch/lib" -L"$site/torch_npu/lib" -L"$cann/lib64" \
  -ltorch -ltorch_cpu -lc10 -ltorch_npu -lascendcl -ldl \
  -Wl,-z,defs -Wl,-rpath,"$site/torch/lib:$site/torch_npu/lib:$cann/lib64" -o "$out"
sha256sum "$out"
