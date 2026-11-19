# A12-H6 workspace 初值最小派生（未执行）

- 父源：`../a12/test_aclnn_grouped_mat_mul_all_reduce.cpp`，官方冻结源 SHA `6c1a90ab…`。
- H6 派生源：`test_aclnn_grouped_mat_mul_all_reduce_h6.cpp`，SHA `4b8e6e2a…`；**唯一差异**见 `h6.diff`（SHA `754e2f24…`）：在 `GetWorkspaceSize`/`aclrtMalloc` 之后、`aclnnGroupedMatMulAllReduce` launch 之前，对同一 workspace 做 `aclrtMemsetAsync(workspaceAddr, workspaceSize, 0, workspaceSize, rtStream)` 并打印/校验 rc。
- 其余不变：同 provider/libopapi、同 FP16 默认形状（x0[256,256]/x1[1024,256] 等）、同 2 设备 1 loop 0,1、同 buffer 生命周期、无其它改动。
- 约定待 33/32 确认：workspace 为调用方分配的 scratch，launch 前同 stream 清零不破坏 GetWorkspaceSize 约定；若 33 认为必须同步 `aclrtMemset` 或 memset 后 sync，仅改这一行（当前未执行）。
- 编译：`bash compile_a12_h6.sh <absolute build dir>`（同 A12 CMake 链接配方，输出 `build/bin/a12_grouped_mmar_h6`）。
- 运行：`bash ../a12/run_a12.sh <build>/build/bin/a12_grouped_mmar_h6`（2 1 0,1）；或用窗口 `--arm A12 --a12-binary <该 binary>`，A12 日志门同样适用。
- 成功日志门：launch/sync rc0、stdout/stderr 无 `[ERROR]`/`retcode:`、≥2 行 `mc2 costtime`、per-PID maps 硬门 libhccl+opapi_math/transformer。若仍 `507046`/stream47 → H6（workspace 初值）对该挂起被证伪，不是平台否定证据。
- 状态：**不执行**；等 20 排程、33 契约确认、32 审核。
