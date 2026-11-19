cmake_minimum_required(VERSION 3.18.4)
project(ACLNN_GROUPED_MMAR_EXAMPLE)
add_compile_options(-std=c++11)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "./bin")
set(CMAKE_CXX_FLAGS_RELEASE "-fPIC -O2 -Wall")
add_executable(a12_grouped_mmar test_aclnn_grouped_mat_mul_all_reduce.cpp)
if(NOT "$ENV{ASCEND_HOME_PATH}" STREQUAL "")
    set(ASCEND_PATH $ENV{ASCEND_HOME_PATH})
else()
    set(ASCEND_PATH "/usr/local/Ascend/cann-9.2.1")
endif()
set(INCLUDE_BASE_DIR "${ASCEND_PATH}/include")
include_directories(${INCLUDE_BASE_DIR} ${INCLUDE_BASE_DIR}/aclnn ${INCLUDE_BASE_DIR}/aclnnop ${INCLUDE_BASE_DIR}/aclnnop/level2)
find_package(Threads REQUIRED)
target_link_libraries(a12_grouped_mmar PRIVATE
    ${ASCEND_PATH}/lib64/libascendcl.so
    ${ASCEND_PATH}/lib64/libnnopbase.so
    ${ASCEND_PATH}/lib64/libopapi_math.so
    ${ASCEND_PATH}/lib64/libopapi_transformer.so
    ${ASCEND_PATH}/lib64/libc_sec.so
    -Wl,--no-as-needed
    ${ASCEND_PATH}/lib64/libhccl.so
    ${ASCEND_PATH}/lib64/libhccl_fwk.so
    ${CMAKE_THREAD_LIBS_INIT})
install(TARGETS a12_grouped_mmar DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
