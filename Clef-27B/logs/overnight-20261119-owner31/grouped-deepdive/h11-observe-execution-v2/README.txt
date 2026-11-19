h11-observe-execution-v2 — H11 观测 v2 执行命令包（CPU-only 准备；无 v2 GO）
================================================================================

定位与边界
----------
- 本包是 v1（953eb2d/754859dc）失败后的 **v2 CPU 准备**，不覆盖 v1 包与 attempt-1 证据。
- 新 CELL：`/workspace/yangchen/work/grouped-deepdive-cell-h11-observe-v2`；新 run/temp/out/restore
  /logs 均为 h11-observe-v2；旧 v1 目录未改动。
- 当前**不是 v2 设备 GO**：未停服、未启动 torchrun payload、未跑 rank/NPU；仅 CPU/只读准备。
- 冻结产品未变：stage e38b541 / run_probe 8e308804 / window 1f780bf8 / binding 101660f8 /
  adapter d181b15a；未重 stage、未重跑 Linux40。rank helper 与 v1 同 blob（a4b76676）。

新身份与 plan
-------------
- identity 从当前 /proc 读取：pid **1347086** / starttime **401212026** / ppid 1 / pgid=sid 1347086 /
  boot 393ecc15-721f-46f7-b119-21fa327a2b3b；五 PID=[1347086,1347229,1347230,1347258,1347259]。
- plan：`plan-h11-observe-v2.json`，status=plan_not_executed，SHA **9c2936c2…3137**；
  plan-only 同 argv 复核逐字节一致（planonly-rerun.txt + planonly-rerun-output.json）。

基础环境（修复 v1 的 PATH 缺口）
--------------------------------
启动 shell 固定顺序：
1) `source /workspace/yangchen/Clef-27B-ascend/scripts/activate_env.sh`
2) `export PATH=/workspace/yangchen/venvs/clef-py312/bin:$PATH`
3) 仅再导出两项观测 env：`ASCEND_GLOBAL_LOG_LEVEL=0`、`ASCEND_PROCESS_LOG_PATH=<CELL>/logs-h11-observe-v2`

实测记录（cpu-prep-*）：
- 基线（restore 同款 PATH，`env -i`）：`command -v torchrun` 为空，rc=1（复现 v1 缺口）。
- activate + PATH 前置：which/resolved = `/workspace/yangchen/venvs/clef-py312/bin/torchrun`；
  executable=yes；SHA fd801bde…0903；解释器 `#!/workspace/yangchen/venvs/clef-py312/bin/python`；
  `torchrun --help` rc=0（CPU-only，无 probe/rank/NPU payload）。
- 注意：activate_env.sh 实际会 source `$CLEF_VENV/bin/activate`，本身已加入 venv bin；显式前置为幂等
  加固（最终 PATH 中 venv bin 出现两次，无害）。v1 的根因是启动 shell **从未 source activate**。

同 shell 执行要求（v2 GO 后，硬性）
-----------------------------------
- window 的 child env = 运行 window 进程的 `os.environ` + OWNER token（window 1f780bf8 line 894/896）：
  PATH 必须在**同一个 shell** 中导出才会传给 child；`--activate` 只用于 restore（line 546-548）。
- `--execute` 内部先 stop 再 spawn child（line 876/896）：预扫必须在调用 `--execute` 之前完成并
  fail-closed。预扫现在**强制检查调用方 shell**（`command -v torchrun` 必须解析到 venv，
  并执行 caller `torchrun --help` rc0）；从裸 shell 运行预扫会 FAIL，防止“预扫通过但 window
  在另一个裸 shell 启动”的假阳性。
- 因此下列步骤必须在同一 shell / 同一 SSH 会话内顺序执行，不得拆开：
  source activate_env.sh → export PATH=venv/bin:$PATH → `command -v torchrun`（记录路径/SHA）→
  `torchrun --help` rc0 → `preflight-rescan.sh`（rc0）→ `supervised_probe_window.py ... --execute`。
- 若改用绝对路径 torchrun 写入 child_argv，需通过 `--child-argv` 完整重建 argv（与 `--arm` 冲突），
  并在预扫中校验该路径 SHA 与 `--help` rc0；本包采用同 shell PATH 方案。

rank env 观测命令（v2，ranks 存活期运行）
------------------------------------------
必须显式给 v2 CELL（helper 默认 CELL 已是 v2，但仍建议显式）：
  /workspace/yangchen/venvs/clef-py312/bin/python -I \
    /workspace/yangchen/work/grouped-deepdive-cell-h11-observe-v2/execution-v2/rank-env-read.py \
    /workspace/yangchen/work/grouped-deepdive-cell-h11-observe-v2 --watch 180 \
    > /workspace/yangchen/work/grouped-deepdive-cell-h11-observe-v2/rank-env-h11-observe-v2.jsonl
读取 `<CELL>/run-h11-observe-v2/descendants.json` 的 `entries` + /proc pgid 扫描；空/缺失 entries =
缺测（不是无 rank/排除路径），死 PID 记 alive=false/MISSING(not_alive)。可用 `--run <dir>` 覆盖 run 目录。

预扫（preflight-rescan.sh）
---------------------------
执行前只读检查：identity/boot/ppid/pgid/sid/starttime、精确五 PID、health/info、clients、29531
tcp+udp×127/0.0.0.0、四路径 ABSENT、logs 可写、全部冻结 SHA（stop/source/bodies/window/run_probe/
binding/adapter/test/manifest/plan v2/root-identity v2/env-sidecar v2）、无其他 probe/torchrun；
并新增 **effective-env torchrun 段**：基线缺失复现、effective 解析到 venv、可执行、`torchrun --help`
rc=0（帮助日志写入 `<CELL>/cpu-prep/rescan-torchrun-help.log`，与观测日志目录分开）。

文件索引
--------
- window-planonly-argv.json / window-execute-argv.json：v2 完整绝对 argv（60/61，execute 仅多顶层
  `--execute`）。
- window-launch-template.txt：含 env 三步、A（plan-only）与 B（v2 GO 后 execute；唯一支持路径为
  run-h11-observe-v2.sh wrapper）。
- run-h11-observe-v2.sh：单一 wrapper，同一 shell 内完成 env→caller torchrun 校验→preflight(rc0)
  →exec window --execute；禁止未授权运行。
- validation-caller-bare-fail.txt / validation-prepared-pass.txt：caller 强制检查的负/正证据
  （裸 shell 预扫必须 FAIL；activate+PATH 后必须 ALL PASS）。
- helper-v2-fixture-check.txt：v2 helper 的 CPU fixture 证据（默认 run-h11-observe-v2、--run 覆盖、
  死 PID MISSING(not_alive)、活进程三键缺测）。
- plan-h11-observe-v2.json、planonly-rerun.txt、planonly-rerun-output.json：plan 与复核证据。
- root-identity-h11-observe-v2.json / env-sidecar-h11-observe-v2.json：新身份与两项 env sidecar。
- collect-raw.sh：v2 raw 采集（CELL/ARCHIVE 绝对、markers、fresh logs + /root/ascend/log 时间窗、
  不按 PID、sources.txt、SUMS、correlation）。
- rank-env-read.py（v2 绑定版）：默认 CELL=v2、RUN=<CELL>/run-h11-observe-v2；支持 --run 覆盖；
  descendants `entries` + /proc pgid；ranks 存活期读三键，空/缺失=缺测。
- preflight-rescan.sh：上述预扫（含 torchrun）。
- restore-and-failure.txt：唯一 restore argv（v2 restore 输出）与失败留现场条件。
- stage-identity-h11-observe-v2.json：冻结 SHA、identity、activation/torchrun 元数据与 plan 指纹。
- cpu-prep-baseline-which.txt / cpu-prep-effective-torchrun-help.log /
  cpu-prep-rescan-torchrun-help.log / cpu-prep-torchrun-metadata.txt：CPU 准备证据；
  帮助日志不混入观测日志（远端 rescan 版本在 <CELL>/cpu-prep/rescan-torchrun-help.log）。

后续
----
- 交 32 复审；新的单次设备执行必须等 20 另发 **v2 明确 GO**。当前不执行、不重跑 v1、不启动 NPU。
