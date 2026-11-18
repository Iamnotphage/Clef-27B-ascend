#!/usr/bin/env bash
# Compilation/link only: does not load the adapter, install a vendor, or use a device.
set -euo pipefail
cd "$(dirname "$0")"
venv=/workspace/yangchen/venvs/clef-py312
site="$venv/lib/python3.12/site-packages"
cann=/usr/local/Ascend/cann-9.2.1
out=${1:?pass an absolute output .so path}
[[ "$out" == /* ]]
g++ -std=c++17 -O2 -fPIC -shared -D_GLIBCXX_USE_CXX11_ABI=1 \
  -I"$site/torch/include" -I"$site/torch/include/torch/csrc/api/include" \
  -I"$site/torch_npu/include" -I"$site" \
  -I"$cann/include" -I"$cann/aarch64-linux/include" -Ibuild_out/autogen \
  torch_adapter/n1_adapter.cpp \
  -L"$site/torch/lib" -L"$site/torch_npu/lib" -L"$cann/lib64" \
  -ltorch -ltorch_cpu -lc10 -ltorch_npu -lascendcl -ldl \
  -Wl,-rpath,"$site/torch/lib:$site/torch_npu/lib:$cann/lib64" \
  -o "$out"
sha256sum "$out"
