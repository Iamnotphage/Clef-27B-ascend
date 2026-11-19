#!/usr/bin/env bash
# Build the frozen official A12 sample unchanged.  AOT only; no device calls.
set -euo pipefail
cd "$(dirname "$0")"
src="$(pwd)/test_aclnn_grouped_mat_mul_all_reduce.cpp"
cann=${ASCEND_HOME_PATH:-${ASCEND_PATH:-/usr/local/Ascend/cann-9.2.1/aarch64-linux}}
build=${1:?absolute build directory required}
[[ "$build" == /* && "$build" != /usr/local/Ascend/* ]]
[[ "$(sha256sum "$src" | cut -d' ' -f1)" == "6c1a90aba84501a901c413eaa5391889d60beff48b766bed9865ab5bf26eeb9f" ]]
mkdir -p "$build"
cat > "$build/CMakeLists.txt" <<CMAKE
cmake_minimum_required(VERSION 3.18.4)
project(GROUPED_MMAR_A12)
add_compile_options(-std=c++17)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "\${CMAKE_BINARY_DIR}/bin")
if(NOT DEFINED ASCEND_PATH OR "\${ASCEND_PATH}" STREQUAL "")
  set(ASCEND_PATH "$cann")
endif()
add_executable(main "$src")
target_include_directories(main PRIVATE
  "\${ASCEND_PATH}/include" "\${ASCEND_PATH}/include/aclnn"
  "\${ASCEND_PATH}/include/aclnnop" "\${ASCEND_PATH}/include/hccl")
find_package(Threads REQUIRED)
target_link_libraries(main PRIVATE
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
sha256sum "$build/build/bin/main"
