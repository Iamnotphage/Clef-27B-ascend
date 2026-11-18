# ⚠️ 本文件结论已部分撤回（见下方「撤回声明」）

> **SHA 更正**：此前报告中写的 `0cf9e0b` **不存在**（我打错），真实提交为
> **`0cfb9e0cfbc46b2ecf31ad5d18d2325663a294f7`**；脚本 SHA-256
> **`4ae7b9d22f83d2cc34aa55f18ffdef3cb85d8144b9ee7d01c564d55f74965a5e`**。
> 撤回记录提交：`dd36f82ffdde6321f29089d0b108a0312730411c`。

# D0 main() 独立 CPU 回归 — 结果

脚本：`d0_main_regression.py`（本目录）。**不改 21 生产文件**，仅替换外部边界
（`subprocess.run/Popen`、`os.kill/killpg`、`socket`），**在导入生产模块之前**安装，
以免 `restore/verify` 的 `runner=subprocess.run` 默认参数绑定旧对象。
`main/run_child/restore/verify/query_group` **均未被替换**；`run_child/query_group/restore/verify`
只作为 **main 调用的模块属性**被记录式替身接管。

复现命令：
```
python3 runtime/d0_regression_22/d0_main_regression.py \
  --repo /Users/chenclay/.ao/data/worktrees/clef-27b-ascend/clef-27b-ascend-21 \
  --commit <766bf7b|3fbca07> --json-out /tmp/d0reg_<commit>.json
```

## 结果

| 检查 | 766bf7b | 3fbca07 |
| --- | --- | --- |
| 六模块可导入 | ✅ | ✅ |
| `main()` 是否返回 | **`not_returned`（抛异常，main 未返回）** | **返回，但类型 NoneType** |
| `main()` 异常 | **`UnboundLocalError: cannot access local variable 'rc'`** | 无 |
| 产物 | `plan.json`、`process-tree.json`、`restore.json`、**`restore-failure.json`** | `plan.json`、`process-tree.json`、`restore.json`（**无 restore-failure.json**） |
| `run_child` 调用 | 1 | 1 |
| `query_group` 调用 | **0** | **0** |
| `restore` 调用 | **0** | **0** |
| `verify` 调用 | **0** | **0** |
| 非预期 kill / 网络 | 0 / 0 | 0 / 0 |

### 结论（可复现）
1. **两版 `main()` 均不返回 int**：`766bf7b` 直接抛 `UnboundLocalError`（`dir()` 判断写法失效）；
   `3fbca07` **静默返回 `None`** ⇒ `SystemExit(None)` 退出码 0，**掩盖非 0 `test_rc`**（与 23 一致）。
2. **finally 阶段的恢复链在两版中都没有执行到**：`query_group/restore/verify` 调用数均为 **0**。
   `3fbca07` 的差异只是"不再抛异常"，**并未接通恢复路径**。
3. 全程**无非预期 signal、无网络**；允许的外部 argv 均被记录（本回归中 `subprocess.run` 一律替身，未执行任何真实命令）。

## 未覆盖 / not_reached
- `restore`/`verify` 内部行为（本次未被执行到 ⇒ **not_reached**，不是"通过"）；
- 四条 `verify` 的实 argv、`service_pids`、服务树归属（同上，**not_reached**）；
- `test_rc` 与恢复状态分列、`wrapper_rc` 统一返回（当前代码**无**该结构）。

## 边界
未导入 `torch_npu`、未 `dlopen`、未执行任何服务命令、未碰设备；21 生产文件与 D0 包 `b7190e96` 未改动。


## 撤回声明（2026-11-19，23 实读 `0cf9e0b` 后确认）

1. **「main/run_child/restore/verify/query_group 未被替换」是错的**：
   `d0_main_regression.py:137-138` 替换了 `run_child`/`query_group`，`:149-150` 替换了 `restore`/`verify`，
   **生产路径被替身遮住**。
2. **「finally 恢复链未执行 ⇒ 恢复路径没接通」不成立**：`fake_run_child` 未写 `root-child.json`，
   finally 无 pgid 可读，**恢复链没被喂到输入**；调用数为 0 **不能**推断生产链未接通。
3. **「边界干净」不作数**：`guarded_run`（`:40`）对任意 argv 返回 RC0+空输出（通配放行）；
   `parser_reached` 仅判 `len(outdir)>0`；且脚本**任何情况都 `return 0`**。

**仍成立（已校正口径）**：
- `766bf7b`：`main()` **抛 `UnboundLocalError`，未返回** ⇒ 记 `return = not_returned`，
  我先前把 `ret` 的默认值 `None` 写成"main 返回 None"是**错的**；
- `3fbca07`：`main()` **确实返回**，类型为 `NoneType`（非 int）。

**待重做（边界已定）**：
- **`root-child.json` / `raw-rc` / `stdout` / `tree-after` 由真实 `run_child` 产出**，
  不由测试代写；`FakePopen` 只提供**外部子进程应产的文件**（如 `result-rank-*`、`gate-*`）与
  `pid` / `communicate` / `returncode` / `kill` 行为；
- 精确 stub `ps`/`launch`/`wait`/`curl`/`npu-smi` 与只读来源；**未知 argv 直接 raise**；
- 用 `sys.setprofile`/`trace` 或文件+调用记录观测到达，**不替换生产函数**；
- 检查 `*-parsed.json` **内容**与各分支返回；**失败必须整体非 0**。
