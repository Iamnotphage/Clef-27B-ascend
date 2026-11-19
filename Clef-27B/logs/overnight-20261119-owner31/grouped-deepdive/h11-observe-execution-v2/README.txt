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
- window-launch-template.txt：含 env 三步与 A（plan-only）/B（v2 GO 后 execute）命令。
- plan-h11-observe-v2.json、planonly-rerun.txt、planonly-rerun-output.json：plan 与复核证据。
- root-identity-h11-observe-v2.json / env-sidecar-h11-observe-v2.json：新身份与两项 env sidecar。
- collect-raw.sh：v2 raw 采集（CELL/ARCHIVE 绝对、markers、fresh logs + /root/ascend/log 时间窗、
  不按 PID、sources.txt、SUMS、correlation）。
- rank-env-read.py：与 v1 同 blob（a4b76676），descendants `entries` + /proc pgid，ranks 存活期读三键。
- preflight-rescan.sh：上述预扫（含 torchrun）。
- restore-and-failure.txt：唯一 restore argv（v2 restore 输出）与失败留现场条件。
- cpu-prep-baseline-which.txt / cpu-prep-effective-torchrun-help.log /
  cpu-prep-rescan-torchrun-help.log / cpu-prep-torchrun-metadata.txt：CPU 准备证据；
  帮助日志不混入观测日志（远端 rescan 版本在 <CELL>/cpu-prep/rescan-torchrun-help.log）。

后续
----
- 交 32 复审；新的单次设备执行必须等 20 另发 **v2 明确 GO**。当前不执行、不重跑 v1、不启动 NPU。
