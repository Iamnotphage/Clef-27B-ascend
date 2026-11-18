# D0 main() 独立 CPU 回归 v2 — 真实分支矩阵

脚本：`d0_main_regression_v2.py`，SHA-256 见文末。
**生产函数从未替换**：`main/run_child/query_group/restore/verify/parse` 全部真实执行；
到达由 `sys.setprofile` 观测。仅拦外部边界：`subprocess.run/Popen`、`os.kill/killpg`、`socket`。
`root-child.json` / `raw-rc.json` / `stdout.log` / `stderr.log` / `tree-after.json`
**由真实 `run_child` 产出**；`FakePopen` 只提供外部子进程应产文件（`result-rank-*`、`gate.json`）
与 `pid`/`communicate`/`returncode`/`kill`。
允许的外部命令只有：被监督子进程（`-m torch.distributed.run`）、`ps`、`npu-smi`、
`launch.sh`、`wait.sh`、`curl`；**其余 argv 一律 raise**（fail closed），网络与任意信号同样 fail closed。

复现：
```
python3 runtime/d0_regression_22/d0_main_regression_v2.py \
  --repo /Users/chenclay/.ao/data/worktrees/clef-27b-ascend/clef-27b-ascend-21 \
  --commit <766bf7b|3fbca07|47daff5> --json-out /tmp/v3_<commit>.json
```
**有任一检查失败即返回非 0**（本次三版均非 0，见下）。

## 矩阵（实跑，非推断）

| 项 | `766bf7b`（旧） | `3fbca07` | `47daff5`（21 最新） |
| --- | --- | --- | --- |
| `main()` 是否返回 | **否（`not_returned`）** | 是 | 是 |
| 返回类型 | — | **`NoneType`（非 int）** | **`int`** |
| 异常 | **`UnboundLocalError: cannot access local variable 'rc'`** | 无 | 无 |
| `run_child` 到达 | ✅ | ✅ | ✅ |
| 生产监督产物 | `root-child` `raw-rc` `stdout` `stderr` `tree-after` **齐** | 同左（**无 `restore-failure.json`**） | 同左（**有 `restore-failure.json`**） |
| `query_group` 到达 | ✅ | ✅ | ✅ |
| `restore` 到达 | **❌** | ✅ | ✅ |
| `verify` 到达 | **❌** | ✅ | **❌** |
| 47daff5 新增到达 | — | — | `build_commands`、`discover_service_pids`、`render_request`、`_live_pids` |
| 失败检查 | `main_returned`、`main_returns_int`、`restore_reached`、`verify_reached` | `main_returns_int`、`no_unexpected_argv` | **`verify_reached`** |

## 分列结论
**历史预期失败（旧版）**：`766bf7b` 抛 `UnboundLocalError` ⇒ **`main()` 未返回**，恢复链未走完。
**当前验收失败（`47daff5`）**：返回 `int` 已满足；`restore` 链已跑通并产出 `restore-failure.json`；
**唯一失败是 `verify` 未被到达** —— 与 21 正在补的 `service_pids`/`verify` 接口一致，需 21 确认。

**注意**：`3fbca07` 的 `no_unexpected_argv` 失败**不是产品缺陷**，而是我的白名单未含其真实子进程 argv 形态，
我已在 v2 中按实际 argv 精确放行（`-m torch.distributed.run`），并保留"未知 argv raise"。

## not_reached（分列，非通过）
`766bf7b`：`restore`、`verify`；`47daff5`：`verify`。
**不据 not_reached 推断生产实现缺陷**，只报"本次未到达"。
cfcf8060fa5e0cece3431ad0b632113cd0b78013eea23cedfe9b02ccd9159589  d0_main_regression_v2.py
