#!/usr/bin/env bash
# Build the frozen official A12 sample unchanged.  AOT only; no device calls.
set -euo pipefail
cd "$(dirname "$0")"
src="$(pwd)/test_aclnn_grouped_mat_mul_all_reduce_h6.cpp"
cann=${ASCEND_HOME_PATH:-${ASCEND_PATH:-/usr/local/Ascend/cann-9.2.1/aarch64-linux}}
build=${1:?absolute build directory required}
[[ "$build" == /* && "$build" != /usr/local/Ascend/* ]]
[[ "$(sha256sum "$src" | cut -d' ' -f1)" == "4b8e6e2a483d285394b3e8136f4be493462154d09f05b4c225a6716273615181" ]]
mkdir -p "$build"
cat > "$build/CMakeLists.txt" <<CMAKE
cmake_minimum_required(VERSION 3.18.4)
project(GROUPED_MMAR_A12_H6)
add_compile_options(-std=c++17)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "\${CMAKE_BINARY_DIR}/bin")
if(NOT DEFINED ASCEND_PATH OR "\${ASCEND_PATH}" STREQUAL "")
  set(ASCEND_PATH "$cann")
endif()
add_executable(a12_grouped_mmar_h6 "$src")
target_include_directories(a12_grouped_mmar_h6 PRIVATE
  "\${ASCEND_PATH}/include" "\${ASCEND_PATH}/include/aclnn"
  "\${ASCEND_PATH}/include/aclnnop" "\${ASCEND_PATH}/include/hccl")
find_package(Threads REQUIRED)
target_link_libraries(a12_grouped_mmar_h6 PRIVATE
  "\${ASCEND_PATH}/lib64/libascendcl.so"
  "\${ASCEND_PATH}/lib64/libnnopbase.so"
  "\${ASCEND_PATH}/lib64/libopapi_math.so"
  "\${ASCEND_PATH}/lib64/libopapi_transformer.so"
  "\${ASCEND_PATH}/lib64/libc_sec.so"
  -Wl,--no-as-needed
  "\${ASCEND_PATH}/lib64/libhccl.so"
  "\${ASCEND_PATH}/lib64/libhccl_fwk.so"
  \${CMAKE_THREAD_LIBS_INIT})
CMAKE
cmake -S "$build" -B "$build/build" -DCMAKE_CXX_COMPILER=g++ -DCMAKE_SKIP_RPATH=TRUE \
  -DASCEND_PATH="$cann"
cmake --build "$build/build" -j"${A12_JOBS:-4}"
sha256sum "$build/build/bin/a12_grouped_mmar_h6"
