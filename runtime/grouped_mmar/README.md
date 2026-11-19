# Grouped MMAR 深挖首包（CPU 冻结，设备未执行）

状态：**Grouped 深挖结案 = mc2_server 在 chip:5 不支持（环境阻塞）；H1–H12 对挂起结案；官方 aa3e/f810 默认、入口、指标不变。** 首包/窗口/A12 证据保留；A13 代码保留未执行。
本包只研究 `aclnnGroupedMatMulAllReduce` 挂起；不改官方 aa3e 入口/默认/指标，不写服务 tree/guard/vendor，
不含服务生命周期与 f810 回退。旧 v4r1 包与 raw 原样保留，本目录为独立研究包。

## 1. 首包范围（按 20 收敛）

| 顺序 | 臂 | 内容 | 成功判据 |
| --- | --- | --- | --- |
| 1 | A8 | 普通 `matmul + dist.all_reduce + sync` | 两 rank sync 返回 + finite |
| 2 | A12 | 官方样例源原字节 FP16（`6c1a90ab…`），**派生 CMake 构建**（官方 CMakeLists 未随包），`./main 2 1 0,1` | 进程 0、两 rank 线程 sync 返回、无挂起（无数值参考答案） |
| 3 | A0 | 旧桥接风格 + 最小 M=128（局部 group string、handler 后立即 close） | API 返回 + 两 rank sync 返回 + finite |
| 4 | H1 | **prepare 前**创建同一 `std::string` 对象、直接传其 `c_str`、保活到 sync 后 | 同上；trace 记录 group object/data 指针 |
| 5 | H2 | workspace tensor 保活到 sync 后，release 仅在 sync 返回后 | 同上 |

33 的本机 9.2.1 二进制证据（E1：prepare 阶段 group 被 executor 复制；E2：handler 在设备消费线程、primary context，使用 prepare-time stream）把 H1/H4/H5/H11 降为确认项；优先 H2 workspace 提前释放、H3 descriptor 早销毁、H6 workspace 初值、H12 多进程 comm/MC2 上下文。首批顺序保持 A8→A12→A0(M128)→A9→H2/H3；H1 保留为低成本运行时确认。

A12 是通路正对照：源字节官方、构建为派生（`a12/BUILD_NOTE.md`）；它与旧桥接有 C1–C5 多个差异，**不能单凭其通过宣布单因根因**；成功仅是 liveness（API rc0 + sync 返回）。
第二批（按结果增量）：H3 描述符保活、H4/H5 线程/context/stream 直接 launch、H6 零初始化、H7 M sweep；
H11（executor/tiling）已降级——旧 binding 从未销毁 executor，tiling 缓冲不是“被释放”，除非头文件证明
executor 引用调用方描述符（则并入 H3）。

## 2. 已证事实与可证伪假设（33 v2）

- 已证：prepare rc0、workspace 16,777,216 B、executor 非空；`cmd.Run()` dispatch 仅 ~27 µs，
  handler 在 Python 返回后执行；launch_return 晚于 final_sync_enter 28–49 ms；final sync 后约 239 s
  未返回，RUNTIME `three minutes timeout! stream_id=47, task_id=36, pendingNum=2`；无 ERROR/WARN、
  无 AICore trap；baseline `F.linear+all_reduce` 与 `nd_native` 均成功，仅 candidate 挂起。
- H1（最高优先）：pybind 的临时 `std::string` 在 torch op 返回后析构，而 handler 异步执行；
  若 executor 未拷贝 group 名则悬垂。**A1 修正**：保活必须是 prepare 实际接收 `c_str` 的同一 string
  对象/存储，且在 prepare 之前建立；prepare 后复制无效（SSO 搬移会改变地址）。记录 prepare/handler
  的 group object/data 指针、size/capacity、`data - object` 偏移。
- H2：workspace 为 C++ 局部、lambda 按值捕获，handler 返回后释放，早于 Python sync；A2 用 global
  保活到 sync 后，记录 data_ptr 与释放时刻。
- H3（强候选）：handler 在 launch 入队后立即 `lists->close()`，销毁描述符/成员；若设备解析仍引用则挂起。
- H4/H5：handler 线程/context 与 prepare 不同；stream 在 prepare 捕获、Python sync 用当前 stream；
  A9 记录 thread id、`aclrtGetCurrentContext`/device、prepare/handler stream，并对照 RUNTIME 47。
- C1–C6（33）：官方样例在栈上保持 `hcom_name` 到 timeout sync 后、先 sync 再 free workspace、
  每 rank 显式 `aclrtSetCurrentContext`、自建 stream、单进程 `HcclCommInitAll`。A12 原样复现该路径。

## 3. 通用埋点 A9（每臂固定）

- prepare/handler：thread id、device、current context、stream 指针/id、group object/data 指针与字节、
  workspace ptr/bytes、descriptor list ptr、executor ptr；CLOCK_MONOTONIC 时间线。
- Python：rank/world/local、torch/torch_npu 版本、`get_hccl_comm_name` 两 rank 对称性、
  `/proc/self/maps` 实际加载的 `libhccl/libopapi/libnnopbase/libascendcl/libtorch_npu/adapter`
  路径 + SONAME + 文件 SHA（不是仅磁盘候选）、API 返回、sync 是否返回、释放 rc、finite。
- 同一进程内 C++ 与 Python 均用 CLOCK_MONOTONIC；Python `time.monotonic` 与其同域。
- 旧 trace 的“C++ launch_return 晚于 Python final_sync_enter”由 handler 异步于 op 返回解释；
  不假定同线程顺序。

## 4. 文件与对象

| 文件 | 作用 | SHA-256 |
| --- | --- | --- |
| `binding_probe.cpp` | 原生 MMAR 单变量实验绑定（flags 见下） | 见 manifest |
| `probe_trace.h/.py` | C++/Python 同 fd 的 CLOCK_MONOTONIC trace | 见 manifest |
| `run_probe.py` | 双 rank runner；默认 plan-only；A8/A0/H1/H2 + A9 | 见 manifest |
| `compile_probe.sh` | AOT 编译 adapter（不改服务） | 见 manifest |
| `fixed_gate.py` | 原冻结门 `041a66aa…`，未改写（stage 3 用） | `041a66aa…` |
| `a12/` | 33 冻结官方样例 5 文件（原字节）+ 编译/运行包装 | 源 `6c1a90ab…` |
| `snapshot_contract.json`、`validate_snapshot.py` | 31/32 snapshot 门 CPU 清单 | 见 manifest |
| `test_probe_cpu.py` | 控制流/路径/释放顺序/watchdog/snapshot 测试 | 见 manifest |
| `cell_supervisor.py` | source-aware 纯监督切片：单 child PGID/硬超时/保护 PID/源 SHA/可选 restore | 见 manifest |
| `test_cell_supervisor_cpu.py` | 监督切片 CPU 测试（8/8） | 见 manifest |
| `manifest.json` | 冻结 tuple/CLI/预期/失败分支/恢复 | 自描述 |

`binding_probe.cpp` flags：`1 keep_group`、`2 keep_workspace`、`4 keep_descriptors`、`8 keep_all`、
`16 direct_launch`、`32 refetch_stream`、`64 memset_workspace`。首包只用 0/1/2；H3+ 用 4/8/16/32/64。
`release(run_id)` 只由 Python 在 `torch.npu.synchronize()` 返回后调用；不销毁 executor，避免双释放；
sync 未返回时 watchdog `os._exit(124)`，不执行 release。

## A12 官方样例结果（2026-11-19，决定性）

- window `ok=false`（正确）；child rc0 是假成功（样例 `std::thread` 忽略 `ACL_CHECK` 返回值）。
- stdout：`[ERROR] acl interface return err ...test_aclnn_grouped_mat_mul_all_reduce.cpp:383, retcode: 507046`、`TestCall execute AICPU failed`。
- plog：两 device `stream_id=47` 10 s sync timeout，current task `KERNEL_AICORE`，`InnerCode=0x7030010`；`DestroyAicpuComm` 50 s timeout。
- **首因线索**：`package_loader` 两次记录 `current device chip:5 does not support package:mc2_server.tar.gz`，该 MC2 server 包未加载；随后 AICORE 任务在 stream 47 上等待永不返回。
- 其他：workspace 16,777,216（与 v4r1 同）；两 rank hcom name 已创建；per-PID maps 实际加载 libhccl/opapi_math/opapi_transformer/hcomm/ascendcl/nnopbase，hccl_fwk absent；binary_pids [1301080]。
- 恢复：branch `aa3e-noop`、tree aa3e、root 1301295、五 PID、health/info/text/image/env/joint/source/clients 全 true。
- **结论**：官方样例与 v4r1 同一 stream 47/AICORE pending。H1/H2/H3/H11（绑定生命周期/线程/stream/comm 创建方式）不能解释挂起；`mc2_server` 在当前 chip:5 未加载是平台层首因线索，与公开文档“A3 不支持”一致。后续仅在 33/31 确认 `mc2_server` 支持矩阵/可启用后再考虑 A13 MatMulAllReduce 控制；否则 Grouped 路径记为**环境阻塞**，不再投入 H6/H7/H12。

## A14 MatmulReduceScatter V1 控制（v3 liveness 通过；v4 取完整数值，待 31 执行）

- 33 契约：上游头 `aclnn_matmul_reduce_scatter.h` SHA `6e49c20f…`（3,158 B），`GetWorkspaceSize(x1,x2,bias,group,reduceOp,commTurn,streamMode,output,ws,exec)`；bias 可空（A2/A3 非 0 bias 不支持，用 nullptr）；output `(m/rank_size,n)`；x1 `(m,k)`、x2 `(k,n)`；reduceOp 仅 sum、commTurn 0、streamMode 1；**m 必须 rank_size 整数倍**；样例 `test_aclnn_matmul_reduce_scatter.cpp` SHA `0996d6f4…`（8,873 B）。A3 ops-info 已注册 `MatmulReduceScatter`/`V2`；符号在 libopapi.so；是否依赖 mc2_server 未证。
- 本包：binding 新增 `run_mrs(x,w,group,rank_size,flags)`（bias nullptr，keep/workspace/descriptor/release 同 grouped）；run_probe `A14`（默认 flags=KEEP_ALL，group/workspace/descriptors/tensors 全部保活到 sync 后 release，避免早释放混淆；M=128/K=256/N=512 小臂，`--rows 968 --k 8704 --n 5120` v4r1-like；output 484×5120）；window choices 增加 A14。
- 待 31 提供本机 `include/aclnnop/aclnn_matmul_reduce_scatter.h` 本地副本/SHA 与 libccl_kernel/MatmulReduceScatter kernel 符号；编译前不得执行。
- CLI：`--arm A14 --adapter <so> --rows 128 --k 256 --n 512 --seed 20261119`（先 plan 后 --execute）；rank 共享 out 父目录由 `prepare_out` 以 `exist_ok=True` 创建（rank0/1 子目录独立），避免双 rank mkdir 竞争；成功=A12 同门（rc0+两 rank sync+finite+maps+无错误行+完成标记）。
- small v2（plan `18fdf185`、identity 1305145）在启动前因双 rank `out.mkdir(exist_ok=False)` 竞争中止：无 MC2 调用、无 507046，**不作 A14 判据**。修复 commit `d571641`/`f087746`，run_probe `b7f255cb`；CPU 45/45（14+9+22）。
- small v3（已执行）：冻结 commit `f087746`、manifest `02fdd342`、binding `101660f8`、window `2202165e`、adapter `d181b15a`；identity 1306695/start 398875478；plan `8782537a`、out `probe-out-a14-small-v3`、M128/K256/N512、KEEP_ALL、240s、master_port 29531 预检空闲；31 stage SUMS 17/17（commit `9600e2a`）。plan `d88250e4` 为过渡版。
- small v3 结果（32/33 核 PASS，31 commit `064d469`，SUMS 16/16 `88b823a4`）：两 rank prepare_rc0、executor 非空、workspace 17,170,432 B、launch_rc0、handler cleanup_rc0/sync_rc0、sync_ms **18.66017/8.16125**，**无 stream47、无 507046** → **MRS V1 A3 AICPU/KFC 路径可用**，被拒 `mc2_server` 非普遍阻断；Grouped 仍 507046/stream47，归因转 Grouped-specific/H12，依赖仍 supported_but_unproven。sync 后 `run_probe.py:353 release` 因零 Tensor 自定义算子无 fallback 抛 `NotImplementedError`，result.json 仅 stub、reduced/bit_equal 未落盘；恢复 aa3e-noop root 1308632/start 398926792、五 PID、四检/env/joint/source/clients 全 true。
- v4 修复（run_probe-only，33/32/20 口径，binding/adapter 不变、`d181b15a` 复用、无需重编）：先算完 reduced/reference/bit_equal/finite/versions 并原子落盘 result，再 `try/except` release 并将 `release_rc`/`release_error`/`harness_release_ok` 写回；`numeric_pass` 只看 api/sync/finite/reference，`pass`/退出码按 `numeric_pass`，另存 `harness_pass = numeric_pass and harness_release_ok`；进程组清理移入 `finally` 且用 `dist.is_initialized()` 保护，两 rank 成功/异常路径都执行既有 teardown、不新增 collective，清理异常写 `harness_cleanup_error` 并保留已落盘数值，240s 窗口兜底。**release 无 fallback 时预期 numeric_pass=true、harness_pass=false、child rc0、window 可 ok**，数值不丢。新 run_probe `0f90fd23`、`test_probe_cpu` `a8bf075f`，CPU 46/46（15+9+22）。31 待以 root 1308632 重建 identity/plan v4、fresh out `probe-out-a14-small-v4`、预检 29531 后重跑同小格。
- matched Grouped 设计（CPU-only，未排设备）：复用 **H11**（KEEP_ALL，无其它 toggle），`--rows 128 --k 256 --n 512`、BF16 x/w、FP32 zero bias、单 group、同 adapter `d181b15a`/同 HCCL comm/240s/同 window，与 A14 small 仅 op 与 bias/output 形态不同；reference=local `x@w` + all_reduce。详见 `h11_matched_control.md`（`1670bc9b`，含单格结果解释表）。顺序：A14 v4 完整 numeric → 32 gate → H11 fresh identity/plan；同挂→Grouped op/协议/context 候选，返回→转 large shape/tiling；不单格定因、不新增臂。

## A12-H6 workspace 初值最小派生（已准备，未执行）

- 父源官方冻结 `a12/test_aclnn_grouped_mat_mul_all_reduce.cpp` `6c1a90ab…`；派生 `a12_h6/test_aclnn_grouped_mat_mul_all_reduce_h6.cpp` SHA `4b8e6e2a…`，唯一差异 `a12_h6/h6.diff` SHA `754e2f24…`：`GetWorkspaceSize`/`aclrtMalloc` 之后、launch 之前对同一 workspace 执行 `aclrtMemsetAsync(workspaceAddr, workspaceSize, 0, workspaceSize, rtStream)` 并打印/校验 rc。
- 其余不变：同 provider/FP16 形状/2 设备 1 loop/buffer 生命周期。编译 `bash a12_h6/compile_a12_h6.sh <abs build>`；运行 `bash a12/run_a12.sh <build>/bin/a12_grouped_mmar_h6`，窗口 `--arm A12 --a12-binary <bin>` 同日志门。
- 约定待 33/32 确认：workspace 为调用方 scratch，同 stream 清零不破坏 GetWorkspace 约定；若需同步 memset/额外 sync 仅改一行。**未执行**；成功=无 `[ERROR]`/`retcode:`、≥2 `mc2 costtime`、rc0；仍 507046 → H6 对该挂起证伪，不作平台否定证据。

## 状态收窄（2026-11-19，32 §8.24；旧“最终结案”仅保留为轨迹）

32/31 独立核验 mc2-support 证据包（5/5 SHA 通过）：`mc2_server.tar.gz` SHA `71263835…`（873,250 B，私有头 `aa55aa55`）；`ascend_package_load.ini` 中 `mc2_server` `optional:true`、`load_as_per_soc:false`；`libmc2_server.json` 明确 `HcclLaunchAicpuKernel/Mc2AicpuTaskCacheClearKernel/Mc2ServerKernel` 均来自 `libmc2_server.so`。A12 plog 两 device 均在 TDT 启动阶段 `current device chip:5 does not support package:mc2_server.tar.gz`，包未加载；AICPU 仅从 `libccl_kernel.so` 取 KFC ResInit/SrvLaunch，MC2 KFC 任务无实现 → AICORE 等待 → stream 47 pending/507046（样例）或 239 s 挂起（v4r1）→ `DestroyAicpuComm` 50 s×2。
**当前口径（收窄，32 §8.24；31 conclusion 8133e16 同口径）**：已确立 A12 失败（stream47/AICORE，507046@10s）与 `mc2_server`/`tsch-compat` 在 TDT chip:5 被 skip；**未确立** Grouped 是否实际依赖被拒的 `mc2_server`（包 optional；AICPU scheduler 已从 `libccl_kernel.so` 加载 RunAicpuKfcResInit/RunAicpuKfcSrvLaunch 并完成 AicpuRpcResInit），H6 workspace 初值、H12 context/AICPU 绑定亦未证伪。旧“confirmed_environment_blocked/所有 MC2 不可运行”表述不再作为最终结论；A13 未注册不作反证。暂停 H6/H7/H12/A8/A13 及所有 Grouped 设备臂，待 A12-H6 与 A14（MatmulReduceScatter V1，A3 已注册）控制；A13 代码保留作未来平台支持时的控制；不改官方默认/指标/CSV，不新建 PR。A12 监督/恢复链已在真实失败下 PASS（rc0 假成功被日志门拦、aa3e-noop 恢复、五 PID/四检全过）。证据：`a12-cell/`（SUMS 10/10）、31 `mc2-support/`（SUMS 5/5，SUMS SHA `1f5e62c8…`）、32 commit `34ccde7`、31 commit `e33d5f7`、33 commit `87ffb96`、35 commit `d4b600c`/`98fc132`。A13 更正：MatMulAllReduce v1 在 A3 未注册，保留未执行；未来控制改用 MatmulReduceScatter V1/V2。

## A13 MatMulAllReduce 控制（下一格）

- `binding_probe.cpp` 新增 `run_mmar`：`aclnnMatmulAllReduce` v1（x1/x2/`bias` BF16、group、`"sum"`、commTurn 0、streamMode 1、output），无 groupList；torch 多进程 comm，与 v4r1 相同的窗口/清理/保活语义。
- 形状 `M=967/K=8704/N=5120` BF16，bias 按 v1 头为 BF16（Grouped 要求 FP32 bias，这是 op 差异，记录不改门）。
- **33/32 复核更正**：`aclnnMatmulAllReduce` v1 在 A3 文档不支持、本机 A3 op-info 未注册（仅 AllGatherMatmul/V2、MatmulReduceScatter/V2）；A13 运行会停在 op 查找（161001 类），**不能作 MC2 路径控制**。A13 代码保留但**未执行**；未来若需 A3 AICPU 控制，应改用 MatmulReduceScatter V1（A3 仅 AI CPU）vs V2 `commMode=aiv`（A3 仅 aiv、≥200MB），且须先确认 A3 注册/平台支持。loader 拒绝证据已足够，不再跑 A13。
- 窗口 `--arm` choices 已加入 A13/H3/H6/H7/H4/H5/H11；A8 仍无 adapter；A13/H6 都需 adapter。
- CLI：`... --arm A13 --adapter <so> --rows 967 --k 8704 --n 5120 --seed 20261119`（先 plan 后 --execute）；A8 用 `--arm A8` 无 adapter。

## 5. 31 执行 CLI（先 plan-only，再 execute；每臂 fresh）

```bash
# 1) 编译（只产出 .so，不加载）
bash runtime/grouped_mmar/compile_probe.sh /abs/run/clef_mmar_probe.so
# 2) A12 编译/运行（原字节样例；独立通路控制）
bash runtime/grouped_mmar/a12/compile_a12.sh /abs/run/a12-build
bash runtime/grouped_mmar/a12/run_a12.sh /abs/run/a12-build/build/bin/main
# 3) 每个 arm 先 plan
torchrun --nproc_per_node=2 --master_port=<free> runtime/grouped_mmar/run_probe.py \
  --arm A0 --temp-root /abs/run/tmp --out /abs/run/A0 --adapter /abs/run/clef_mmar_probe.so \
  --k 8704 --n 5120 --rows 128 --seed 20261119 --arm-timeout 240 --comm-timeout 90
# 4) 同一命令加 --execute；A8 不需要 --adapter；H1/H2 用 --arm H1/H2
```

- `--arm-timeout` 默认 60 s（33 的 240 s 可显式传入；首轮短超时避免无区别挂满）。
- 建议用 `cell_supervisor.py` 包住每格：它在独立 PGID 运行命令、硬超时后只清该 PGID、拒绝 signal
  任何 `--protected-pid`、前后校验 source SHA 恒 aa3e，并可选执行 31 提供的官方 start.sh restore argv
  与 post-check argv。它**不**停/启服务；服务生命周期仍由 31 控制。
- 每臂新 `--out`（必须不存在）；`--temp-root` 显式；输出 `rank{0,1}/result.json`、`trace.jsonl`、
  失败时 `timeout.json`。A12 输出由 31 的 supervisor 捕获 stdout/stderr/rc。
- 固定 seed/形状；967 行真实 MLP 臂在通路成立后另行提供真实 weight bin + SHA（本包支持
  `--weight-bin/--weight-sha`，首包不预置生产权重）。

### 5.1 监督切片 CLI（31 执行；先 plan 后 execute）

```bash
python3 runtime/grouped_mmar/cell_supervisor.py   --source-target <aa3e head> --expected-source-sha aa3e1758...   --timeout 240 --run-dir <fresh> --protected-pid <service root> --protected-pid <member>...   --restore-argv '<official start.sh argv JSON>' --post-check-argv '<31 check argv JSON>'   --execute -- <probe command argv>
```

A12 臂可只用 `--` 加 `bash runtime/grouped_mmar/a12/run_a12.sh <bin>`，不带 restore（服务由 31 决定）；
A8/A0/H1/H2 臂用 `torchrun ... run_probe.py ... --execute`。正常 restore 后必须 `branch=aa3e-noop`、
tree 恒 aa3e；`f810->fused` 仅异常救援且判隔离失败。监督切片只记录 rc/survivors/source SHA/restore rc，
不解释数值。

### 5.2 完整服务窗口 CLI（31 执行；先 plan 后 execute）

`supervised_probe_window.py` 是 31 要求的完整切片：preflight identity/端口/client → 原 d0_stop →
断言端口 111/旧整 PGID 空/卡7 空 → child 独立 PGID/240s 硬超时/只杀精确 PGID/unknown 停报 →
finally 官方 start.sh clean 两键+activate 恢复 aa3e/s1（fresh output）→ health、branch 必须
`aa3e-noop`、tree 恒 aa3e、五 PID 同 PGID、卡7 双 worker、health/info/text/image 四检、OPP9/LD12、
joint 三键、clients 空；`f810->fused` 仅保留漂移并判隔离失败。它不写 tree/guard/vendor，也不复用
5c/v4r1 的 f810 revert 窗口；`window.json` 记录每步 rc/argv/identity/child PGID/cleanup。

```bash
python3 runtime/grouped_mmar/supervised_probe_window.py \
  --run-dir <fresh> --root-identity <fresh-id.json> \
  --stop-script /tmp/d0cc/d0_stop.py --expect-stop-sha d6d49d75... \
  --arm A12 --a12-binary <a12-build>/build/bin/a12_grouped_mmar \
  --source-target <aa3e head> --expected-source-sha aa3e1758... \
  --official-start <service-root>/Clef-27B/start.sh \
  --activate <service-root>/scripts/activate_env.sh \
  --service-root <service-root> --model-dir <model> \
  --restore-output <fresh> --text-body <text request> --image-body <image request> \
  --port 18143 --execute
```

A8/A0/H1/H2 把 `--arm A12 --a12-binary ...` 换成 `--arm A0 --adapter <clef_mmar_probe.so>
--probe-temp-root <fresh> --probe-out <fresh> --rows 128 --k 8704 --n 5120 --seed 20261119`；
也可用 `--child-argv '<JSON argv>'` 直接给完整 child 命令。`--text-body/--image-body` 必须显式提供，
窗口不伪造请求体。plan-only 输出完整 stop/child/restore argv 与成功门。

## 6. 预期观测 / 失败分支

- A8 成功：两 rank sync 返回，`both_finite=true`。
- A12 成功：`main` 退出 0、stdout/stderr 无 `[ERROR]/retcode:/TestCall execute AICPU failed`、
  ≥2 行 `mc2 costtime`、child PGID 子树 per-PID maps 含 libhccl+opapi_math/transformer；挂起则说明
  环境/SoC/通信路径问题，先于 binding 归因。maps 记录 opapi_math/transformer/hccl_fwk/hcomm/
  ascendcl/nnopbase 实际 loaded/absent；hccl_fwk 不作硬门。
- A0 若挂起：与 33 现象一致；进入 H1/H2 单变量。A0 返回而旧 run_probe 挂起：差异在 norm/包装/输入，
  按增量臂定位。
- H1 返回：支持“group 名生命周期”为候选根因，A1 可作修复形态；H1 仍挂起：排除该假设。
- H2 返回：支持 workspace 生命周期；否则排除。
- 失败分支：prepare rc≠0、launch rc≠0、sync 超时、rank group 名不一致、输出非 finite、release rc≠0。
  只 rc0 而无 sync 返回一律判失败。
- 不改写/放宽 `fixed_gate.py`；BF16 NZ 生产参考门在 stage 3 使用。

## 7. 清理与 aa3e 恢复交接（31 唯一执行）

- 每臂 fresh 进程；前置 snapshot 用 31 的 `snapshot-and-restore-spec.json`
  （`39e52eb5…`，恢复 spec 新 SHA `224b0036…`；正常成功仅 `branch=aa3e-noop`）；32 已核 snapshot PASS。
- routine finally 必须用**官方 start.sh** 恢复 **aa3e/s1**：`record branch=aa3e-noop`、tree 前后
  `aa3e1758…`；不得调用旧 5c/`restore_prod` 自动 revert f810。恢复是带门状态机：old root 仍活时
  **绝不 Popen**（health ready+端口 open 记 skip，否则记 `old_root_alive_partial` blocked）；old root
  不在时 Popen 前再门 port==111、旧 PGID 非 Z 空、无 child/probe/runner 残留、source==aa3e；任一
  不满足 blocked+停报。skip/blocked 都写 `window.json` 并返回非 0。
- `f810->fused` 只允许作为已知源异常的救援路径：必须记录漂移前后 SHA/trace/plog/时间戳，该格判隔离
  失败并即报 20/32，再按已授权 aa3e 方案救回；不得静默计成功。unknown 源状态先停报，不猜不改。
- 保留所有 run-dir/raw/trace/plog；不写服务 tree/guard/vendor；服务当前在线不改。
- 31 stage 后交 32 preflight；设备格按 A8 → A12 → A0 → H1 → H2 顺序，其他假设按结果增量。

## 8. 输入与来源

- 33 契约：C1–C6、A12 冻结样例 `6c1a90ab…`（commit `f5703e4`）、矩阵 v2、支持矩阵研究
  （repo commit `2616628`）；A12 样例文档自相矛盾已记录，不作支持性裁决。
- 31：snapshot `39e52eb5…`、restore-spec `224b0036…`、API pack `/tmp/aa3e-grouped-api-pack-20261119`
  （权威 v3 清单 `cfa70c1d…`，64 payload 64/64 OK；binary index `c9bd574a…`；tar `8a288b8c…` 仅派生）；A12 设备构建参考
  glue `22edb89f…`、binary `37929ec4…`、stage manifest `1d529ca8…`。
- 32：首包判据、snapshot/restore 收紧、A12 不可单独作为 C1 单因证据。
- 本包不 import torch/NPU 做 CPU 测试；CPU 测试只验证控制流/资源生命周期/超时/元数据。
