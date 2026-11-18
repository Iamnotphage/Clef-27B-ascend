# D0 CPU 回归闭合（2026-11-19）

独立 CPU 回归脚本：`runtime/d0_regression/d0_main_regression.py`
（SHA-256 `9dacd1f978605e36d725675971158c5a017aab6341903a966dad04e38ef66817`）。
本报告只覆盖 `d0_run.main` 的 Python 监督/恢复路径；**独立 `d0_restore.main` 不在本 14 例中**，
其动态 label 由生产代码与模块身份检查覆盖，未单列运行 case。未安装、加载或调用任何 NPU/设备/真实服务命令。

## 1. 目标与结论

| 项 | 值 |
| --- | --- |
| 修复前生产 | `56427329248f49490a87877f6a47d04239418bfe`（suite RC **1**） |
| 中间生产 | `5e85681ec09b37ec9ba951bde4ca4a38615ab34e`（suite RC **1**，仅 restore_exception 的 raw-rc 不一致） |
| 最终生产 | `0bd6f6a4789970728c341c282a88f39ce990cf76`（suite RC **0**，14/14） |
| 生产改动 | restore/verify 未成功时 `wrapper_rc=3`；外层恢复异常同样回写 `raw-rc.wrapper_rc`；`test_rc`/`child_raw_rc` 保留 |
| 测试范围 | 1 happy + 13 固定反例；每 case 独立子进程，隔离 `sys.modules`/模块状态 |
| 汇总聚合 | suite 仅在全部 `case_ok=true` **且** `case_rc==0` 时通过；`test_aggregation.py` 5 项反例覆盖 `case_ok=true`+rc1 |
| 边界 | 只 mock `subprocess.run/Popen`、`os.kill/killpg/getpgid`、`socket` 与 `/workspace`、`/proc` VFS；生产函数全部真实执行 |

## 2. 矩阵

`main/test_rc/child_raw_rc/wrapper_rc/restore_ok/verify_ok` 分列：

| case | main | test_rc | child_raw | wrapper | restore | verify | case |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| happy | 0 | 0 | 0 | 0 | true | true | PASS |
| root_changed | 3 | 0 | 0 | 3 | false | false | PASS |
| boot_missing | 3 | 0 | 0 | 3 | false | false | PASS |
| boot_empty | 3 | 0 | 0 | 3 | false | false | PASS |
| timeout | 2 | 2 | 0 | 2 | true | true | PASS |
| child_nonzero | 7 | 7 | 7 | 7 | true | true | PASS |
| query_unavailable | 3 | 0 | 0 | 3 | false | false | PASS |
| cleanup_residue | 3 | 0 | 0 | 3 | false | false | PASS |
| verify_failure | 3 | 0 | 0 | 3 | true | false | PASS |
| parse_failure | 5 | 0 | 0 | 5 | true | true | PASS |
| restore_exception | 3 | 0 | 0 | 3 | false | false | PASS |
| child7_restore_exception | 7 | 7 | 7 | 7 | false | false | PASS |
| workers_unrelated | 3 | 0 | 0 | 3 | true | false | PASS |
| workers_duplicate | 3 | 0 | 0 | 3 | true | false | PASS |

修复前：前 10 例中 6 条 `dut_nonzero` 失败，restore_exception/workers 两例也为 0；中间 `5e85681` 仅
`restore_exception` 因 `raw-rc.wrapper_rc=0` 而 `wrapper_rc_matches_main` 失败；最终 `0bd6f6a` 全绿。

## 3. happy 路径证据

- 两 rank 的 `result-rank{0,1}-parsed.json` 由真实 `d0_trace_parse.decode` 解析：每 rank 2 条
  observation、1 条 claim（rank0 slot24、rank1 slot26）、`observed_coverage=true`、无 identity rejected。
- `root-child.json` / `raw-rc.json` / `stdout.log` / `stderr.log` / `tree-after.json` 均由真实
  `run_child` 写出；`FakePopen` 只写外部 `result-rank*.json` 与 `gate.json`。
- 精确 argv 顺序（动态 label）：`child -> ps -> ps -> launch_service -> wait -> health -> workers -> text -> image`；
  `launch.sh` 的 `CLEF_LAUNCH_CONFIG=s1` 被捕获并断言；text/image 的 URL、`Content-Type`、`-d` 与
  **虚拟 fixture** 逐字一致。fixture 的来源路径保留，但**未读取实际 request payload**；该断言只覆盖虚拟
  fixture 的逐字传递，不声称与实际服务请求相同。
- 测试 child `PID=424200`、服务 root `PID=424242`、workers `424301/424302`；启动前 `/proc` 只有
  harness 进程，服务树只在 `launch.sh` fixture 出现。
- 已加载模块只有 `d0_run`、`d0_restore`、`d0_trace_parse`；三者的 `__file__`、`spec.origin`、源码 SHA
  与 `0bd6f6a:runtime/d0/*` 一致，关键函数 `co_filename` 与模块文件一致。`d0_launch`/`d0_gate`/`d0_executor`
  未由 main 加载，只列出冻结源 SHA，不强制 import。

## 4. workers 表格来源

fixture 使用 21 仓库的真实原件
`clef-27b-ascend-21/docs/evidence/npu-smi-snapshot-2026-11-12.txt`
（SHA-256 `308468003162edeb57bea1b532fed4d3371c2e4260fe5ae6bc25f7e4b0bf1ceb`，
started `2026-11-12T10:10:51.017948+00:00`，returncode 0）。冻结副本：
`runtime/d0_regression/evidence/npu-smi-snapshot-2026-11-12.{txt,json}`。

fixture 保留原件的完整 5 列表头（NPU Chip / Process id / Process name / Process memory / Process id in
container），只把 root/worker PID 映射为模拟值；`_worker_pids` 从 cell[1] 读取。反例
`workers_unrelated`（一个非服务 PID 的 VLLMWorker 行）与 `workers_duplicate`（同一服务 PID 重复两行）
都只得到 workers=1，`verify_ok=false`，不会被凑成 2。

## 5. 恢复异常与 rc 分列

- `restore_exception`：正常 child rc0，删除 VFS TEXT_SOURCE 触发真实 `render_request` FileNotFoundError；
  生产写出 `restore-failure.json`，main/wrapper=3，`test_rc=0`、`child_raw_rc=0`，`raw-rc.wrapper_rc`
  必须与 main 一致。
- `child7_restore_exception`：child rc7 加同一异常；main/wrapper/test/child 均为 7，3 不得覆盖 7。
- 中间 `5e85681` 的 `restore_exception` 暴露了 raw-rc 仍为 0，最终 `0bd6f6a` 已修复。

## 6. 复现与证据

```bash
python3 -I runtime/d0_regression/d0_main_regression.py \
  --repo /Users/chenclay/.ao/data/worktrees/clef-27b-ascend/clef-27b-ascend-21 \
  --commit 0bd6f6a4789970728c341c282a88f39ce990cf76 \
  --json-out /tmp/d0-suite.json \
  --freeze-dir /tmp/d0-freeze
```

把 `--commit` 换成 `5e85681…`/`5642732…` 可得中间/修复前对照。

冻结路径：

- `runtime/d0_regression/evidence/run-5642732/`（修复前 14 case）
- `runtime/d0_regression/evidence/run-5e85681/`（中间 14 case）
- `runtime/d0_regression/evidence/run-0bd6f6a/`（最终 14 case）
- `runtime/d0_regression/evidence/npu-smi-snapshot-2026-11-12.{txt,json}`（workers 原件）
- `runtime/d0_regression/evidence/MANIFEST.json`（脚本 SHA、三版生产 fullSHA、矩阵、happy argv、
  loaded module identity、六份冻结源 SHA、npu 源、失败注入列表、排除项）
- `runtime/d0_regression/evidence/aggregation-self-test.json`（5 项聚合反例结果）
- `runtime/d0_regression/evidence/SHA256SUMS`（全部证据文件哈希）

## 7. 边界与未做项

- 未安装/加载扩展、未 dlopen、未 import torch_npu、未跑 NPU/真实服务；固定软件组合与数值门未改。
- 所有未知 argv/目标/signal/network fail closed；已知模拟 PID/PGID 信号只改 VFS 状态，不发真实信号。
- 真实写入只发生在临时 run 目录；`/workspace` 与 `/proc` 的写入被断言拒绝。
- 未新增服务采集；npu-smi 使用上述既有真实原件，text/image 仅用虚拟 fixture。
- 独立 `d0_restore.main` 不在 14 例中；本报告不把它计为已运行 case。
- 汇总聚合反例：`ok_json_rc1` 等 5 项由 `runtime/d0_regression/test_aggregation.py` 验证；结果冻结在
  `evidence/aggregation-self-test.json`。
