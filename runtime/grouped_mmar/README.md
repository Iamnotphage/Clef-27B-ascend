# Grouped MMAR 深挖首包（CPU 冻结，设备未执行）

状态：**首包源码/计划/CPU 控制流测试完成，待 31 stage/编译/设备执行、32 preflight**。
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
  --k 8704 --n 5120 --rows 128 --seed 20261119 --arm-timeout 60 --comm-timeout 90
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
- A12 成功：`main` 退出 0、两 rank 线程 sync 返回、样例自检通过；挂起则说明环境/SoC/通信路径问题，
  先于 binding 归因。
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
  `aa3e1758…`；不得调用旧 5c/`restore_prod` 自动 revert f810。
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
