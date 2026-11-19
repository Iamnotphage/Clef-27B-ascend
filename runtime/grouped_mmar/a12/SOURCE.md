# A12 官方 A3 样例冻结包（GroupedMatMulAllReduce）

用途：35 的 A12 正对照（官方样例原样运行于 A3）；**不修改样例字节**。本包由 33 只读提取，供 31 执行、32 独立核。

## 来源（固定 commit，勿替换）

- 仓库：`hicann/ops-transformer`（镜像）
- commit：`f5703e415b9be16a2af7ba85564c37414592fc03`
- 权威样例（A12 冻结源）：
  - path：`mc2/grouped_mat_mul_all_reduce/examples/test_aclnn_grouped_mat_mul_all_reduce.cpp`
  - raw：`https://raw.githubusercontent.com/hicann/ops-transformer/f5703e415b9be16a2af7ba85564c37414592fc03/mc2/grouped_mat_mul_all_reduce/examples/test_aclnn_grouped_mat_mul_all_reduce.cpp`
  - 本地字节：`test_aclnn_grouped_mat_mul_all_reduce.cpp`，SHA-256 `6c1a90aba84501a901c413eaa5391889d60beff48b766bed9865ab5bf26eeb9f`，21,723 B
- 文档（含 A3 样例与约束，**参考**，非 A12 编译源）：
  - path：`mc2/grouped_mat_mul_all_reduce/docs/aclnnGroupedMatMulAllReduce.md`
  - 本地字节：`aclnnGroupedMatMulAllReduce.docs.md`，SHA-256 `31333c59bf5b5bd733a0d08520fa17859def52784d4e1176232a374b8cb8078f`
  - 文档内嵌样例提取版：`aclnnGroupedMatMulAllReduce.sample.cpp`，SHA-256 `d59a49e5277d577d294b2e8fc2324f3b49199a5389042db9acb1c03d2154610a`（与 repo 版有版权头/格式/个别字符差异，不用于 A12）
- 编译配方参考：`docs/zh/context/compile_and_run_sample.md`，本地 `compile_and_run_sample.md`，SHA-256 `c591f0e34f223c4ee4acea0dc3aaa9b90ebdb59f19da61090de14108236e979d`

版本关系声明：f5703e4 是上游固定参考 commit，**不是**本机 CANN 9.2.1 安装版同版本真值；本机头/库/op-info 以 31 API 包为准。文档产品表写“Atlas A3 系列：不支持”，但同文样例注释写“本样例基于 Atlas A3 实现，必须在 Atlas A3 上运行”，且 `op_host/grouped_mat_mul_all_reduce_def.cpp` 同时注册 `ascend910b` 与 `ascend910_93`——记录为文档自相矛盾，不作支持性裁决。

## 样例实际调用序列（repo 样例行号）

1. `aclInit(nullptr)`（L163；main 内另有 `aclInit(NULL)` L603）
2. 每 rank：`aclrtSetDevice(devices[rankId])`（L550）、`aclrtGetCurrentContext(&context[rankId])`（L551）、`aclrtCreateStream(&rtStream[rankId])`（L552）
3. rank 线程内：`aclrtSetDevice(args.logicDeviceId)`（L423）、`aclrtSetCurrentContext(args.resources.context)`（L424）
4. `HcclCommInitAll(ndev, devices, comms)`（L609）
5. 每 rank：`HcclGetCommName(hcclComm, hcom_name)`（L356），`hcom_name` 为栈上 `char[N]`，保持到 sync 后
6. `aclnnGroupedMatMulAllReduceGetWorkspaceSize(... split_item=0, groupList=nullptr, "sum", comm_turn=0, streamMode=1, ...)`（L370；inputs[3]=nullptr）
7. `aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST)`（L378）
8. `aclnnGroupedMatMulAllReduce(workspaceAddr, workspaceSize, executor, rtStream)`（L380）
9. **`aclrtSynchronizeStreamWithTimeout(rtStream, 10000)`**（L383）
10. `aclrtFree(workspaceAddr)`（L385）
11. finally：`aclrtSynchronizeStreamWithTimeout`（L452/L461）、`aclrtDestroyStreamForce`（L462）、`HcclCommDestroy`（L465）

## 样例 buffer 配置（FP16 默认）

| 张量 | shape | dtype |
| --- | --- | --- |
| x0 / x1 | [256,256] / [1024,256] | float16 |
| weight0 / weight1 | [256,256] / [256,1024] | float16 |
| b0 / b1 | [256] / [1024] | float16 |
| y0 / y1 | [256,256] / [1024,1024] | float16 |
| group_list | 空（nullptr） | int64 |

- `data_size = 10,485,760`、`type_size = 2`；用法 `./main 2 1 0,1`（2 设备、1 循环、设备 0/1）。
- `DtypeMap` 支持 `float16`/`bfloat16`（L59），但默认入口写死 `input_args.dtype = "float16"`（L483）。**BF16 需同时把 bias 改 FP32**（文档约束），将不再是原字节样例；建议 A12 先原样 FP16 跑通，再以“dtype+bias”为明确 delta 的派生臂。

## 编译/链接要点（官方文档 MC2 段）

- 链接：`libascendcl.so`、`libnnopbase.so`、`libopapi_math.so`、`libopapi_transformer.so`、`libc_sec.so`、`-Wl,--no-as-needed`、`libhccl.so`、`libhccl_fwk.so`、`Threads`。
- CMake：`cmake ../ -DCMAKE_CXX_COMPILER=g++ -DCMAKE_SKIP_RPATH=TRUE && make`；运行前 `source ${INSTALL_DIR}/set_env.sh`。

## 校验

`SHA256SUMS` 覆盖本目录 4 个文件（不含自身）；`shasum -a 256 -c SHA256SUMS` 应 0 FAILED。
