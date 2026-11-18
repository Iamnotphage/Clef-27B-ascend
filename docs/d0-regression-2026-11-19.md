# D0 CPU 回归闭合（2026-11-19）

独立 CPU 回归脚本：`runtime/d0_regression/d0_main_regression.py`
（SHA-256 `d7f2e9c41e73b27a81ab0acdab330730b7e64f51a135df0760f16e71c5f00460`）。
本报告只覆盖 Python 监督/恢复路径；未安装、加载或调用任何 NPU/设备/真实服务命令。

## 1. 目标与结论

| 项 | 值 |
| --- | --- |
| 修复前生产 | `56427329248f49490a87877f6a47d04239418bfe`（suite RC **1**） |
| 修复后生产 | `5e85681ec09b37ec9ba951bde4ca4a38615ab34e`（suite RC **0**） |
| 生产改动 | `d0_run.main` 在 restore/verify 未成功（含 blocked-before-launch）时把 `wrapper_rc` 置 3，保留 `test_rc`/`child_raw_rc` 并把合成值回写 `raw-rc.json` |
| 测试范围 | 1 个 happy + 9 个固定反例，共 10 case；每 case 独立子进程，隔离 `sys.modules`/模块状态 |
| 边界 | 只 mock `subprocess.run/Popen`、`os.kill/killpg/getpgid`、`socket` 与 `Path`/`os.listdir` 的 `/workspace`、`/proc` VFS；生产 `main/run_child/query_group/restore/verify/discover/parse` 全部真实执行 |

## 2. 矩阵

`main/test_rc/child_raw_rc/wrapper_rc/restore_ok/verify_ok` 分列；修复前仅差异项列在括号内。

| case | main | test_rc | child_raw_rc | wrapper_rc | restore | verify | case |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| happy | 0 | 0 | 0 | 0 | true | true | PASS |
| root_changed | 3（前 0） | 0 | 0 | 3（前 0） | false | false | PASS |
| boot_missing | 3（前 0） | 0 | 0 | 3（前 0） | false | false | PASS |
| boot_empty | 3（前 0） | 0 | 0 | 3（前 0） | false | false | PASS |
| timeout | 2 | 2 | 0 | 2 | true | true | PASS |
| child_nonzero | 7 | 7 | 7 | 7 | true | true | PASS |
| query_unavailable | 3（前 0） | 0 | 0 | 3（前 0） | false | false | PASS |
| cleanup_residue | 3（前 0） | 0 | 0 | 3（前 0） | false | false | PASS |
| verify_failure | 3（前 0） | 0 | 0 | 3（前 0） | true | false | PASS |
| parse_failure | 5 | 0 | 0 | 5 | true | true | PASS |

修复前 suite RC 1（6 条 `dut_nonzero` 失败），修复后 suite RC 0。

## 3. happy 路径证据

- 两 rank 的 `result-rank{0,1}-parsed.json` 由真实 `d0_trace_parse.decode` 解析：每 rank 2 条
  observation、1 条 claim（rank0 slot24、rank1 slot26）、`observed_coverage=true`、无 identity rejected。
- `root-child.json` / `raw-rc.json` / `stdout.log` / `stderr.log` / `tree-after.json` 均由真实
  `run_child` 写出；`FakePopen` 只写外部 `result-rank*.json` 与 `gate.json`。
- 精确 argv 顺序（动态 label）：
  `child -> ps -> ps -> launch_service -> wait -> health -> workers -> text -> image`；
  text/image 的 `-d` 与真实 request 文件逐字相等，URL/超时/`Content-Type` 精确匹配。
- 测试 child `PID=424200`、服务 root `PID=424242`、workers `424301/424302`；启动前 `/proc` 只有
  harness 进程，服务树只在 `launch.sh` fixture 出现。
- `d0_run`/`d0_restore`/`d0_launch`/`d0_gate`/`d0_executor`/`d0_trace_parse` 六个模块的
  `__file__`、`spec.origin`、源码 SHA 与 `5e85681:runtime/d0/*` 一致；关键函数 `co_filename` 与模块文件一致。

## 4. 证据与复现

复现（CPU，无设备）：

```bash
python3 -I runtime/d0_regression/d0_main_regression.py \
  --repo /Users/chenclay/.ao/data/worktrees/clef-27b-ascend/clef-27b-ascend-21 \
  --commit 5e85681ec09b37ec9ba951bde4ca4a38615ab34e \
  --json-out /tmp/d0-suite.json \
  --freeze-dir /tmp/d0-freeze
```

冻结路径：

- `runtime/d0_regression/evidence/run-5642732/`：修复前 suite + 10 case 原始 JSON/artifacts；
- `runtime/d0_regression/evidence/run-5e85681/`：修复后 suite + 10 case 原始 JSON/artifacts；
- `runtime/d0_regression/evidence/MANIFEST.json`：脚本 SHA、前后生产 fullSHA、矩阵、happy argv、
  六模块 identity、失败注入列表；
- `runtime/d0_regression/evidence/SHA256SUMS`：全部证据文件哈希；
- 22 交接原件：`runtime/d0_regression/evidence/{FINAL-22.json,t7-22.json,ev6-22.json,x_f032801_happy-22.json}`。

## 5. 边界与未做项

- 未安装/加载扩展、未 dlopen、未 import torch_npu、未跑 NPU/真实服务；固定软件组合与数值门未改。
- 所有未知 argv/目标/signal/network fail closed；已知模拟 PID/PGID 信号只改 VFS 状态，不发真实信号。
- 真实写入只发生在临时 run 目录；`/workspace` 与 `/proc` 的写入被断言拒绝。
- 未新增服务采集；npu-smi 与 request fixture 按既有真实快照/文件格式构造。
- 本报告等待 23 一次合核；生产 `5e85681` 由 21 维护。
