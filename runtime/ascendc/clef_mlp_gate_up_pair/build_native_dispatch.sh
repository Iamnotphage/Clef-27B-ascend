#!/usr/bin/env bash
# Build the optional experiment-only ATen dispatch bridge.
set -eo pipefail
src=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
py=${PYTHON_BIN:?set Python in the pinned Torch and TorchNPU environment}
out=${1:?provide an output path for the private shared library}
cann=${ASCEND_HOME_PATH:?source the pinned CANN toolkit}
test ! -e "$out"
readarray -t paths < <("$py" - <<'PY'
from importlib.util import find_spec
from pathlib import Path
import torch

print(Path(torch.__file__).parent)
print(Path(find_spec("torch_npu").origin).parent)
print(int(torch._C._GLIBCXX_USE_CXX11_ABI))
PY
)
torch_dir=${paths[0]}
npu_dir=${paths[1]}
"${CXX:-g++}" -std=c++17 -O2 -fPIC -shared "-D_GLIBCXX_USE_CXX11_ABI=${paths[2]}" \
    -I"$torch_dir/include" -I"$npu_dir/include" -I"$cann/include" \
    "$src/native_dispatch.cpp" -o "$out" \
    -L"$torch_dir/lib" -L"$npu_dir/lib" -Wl,-rpath,"$torch_dir/lib:$npu_dir/lib" \
    -ltorch_npu -ltorch_cpu -ltorch -lc10 -ldl
