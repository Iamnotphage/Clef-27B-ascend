h11-observe-execution-v1 — H11 观测执行命令包（doc-only；无 GO，不执行）
=====================================================================

定位
----
- 绑定 plan `c95ae5c3…`（commit 01e8631 的 preflight 产物，status=plan_not_executed）。
- 本包是独立的执行准备/命令包，不覆盖原 preflight；当前没有设备许可，任何命令都不应执行。
- 冻结 Stage：`/workspace/yangchen/work/grouped-deepdive-35-e38b541/runtime/grouped_mmar`
  的 window `1f780bf8` + run_probe `8e308804` + adapter `d181b15a` + binding `101660f8`。
- 服务保持 aa3e/s1（root 1337814 / starttime 400405818 / boot 393ecc15…）。

环境口径（仅两变量 sidecar）
---------------------------
- 只导出：`ASCEND_GLOBAL_LOG_LEVEL=0`、`ASCEND_PROCESS_LOG_PATH=<cell>/logs-h11-observe-v1`。
- 其余环境按已核基线继承；**不再添加** unset `ASCEND_MODULE_LOG_LEVEL`/`HCCL_DEBUG_CONFIG`
  之类的额外改动。历史 plog 中这两个变量为空/0x0 仅作为基线事实记录在
  `../h11-observe-preflight/env-sidecar-h11-observe-v1.json`，本包不改动它们。

文件索引
--------
- window-planonly-argv.json / window-execute-argv.json：完整绝对路径 argv 数组（60 项；
  execute 版本仅多顶层 `--execute`）。由实际生成 c95ae5c3 的同一条 argv 派生。
- window-launch-template.txt：可直接复制的 shell 模板（plan-only 与授权后 execute 两段）。
- planonly-rerun.txt + planonly-rerun-output.json：同模板去执行 flag 的复核结果与原始输出
  （输出与 plan 逐字节相同，SHA `c95ae5c3…`；附 sha256sum/diff 证据）。
- collect-raw.sh：raw 采集（fresh LOGDIR + 默认 plog + device-0/1 + vllm_ascend + cell +
  meta；START/END marker 时间窗；完整文件、不按 PID 过滤；SUMS；派生 correlation.txt）。
- rank-env-read.py：rank 存活期读取三 env 白名单（GLOBAL_LOG_LEVEL / PROCESS_LOG_PATH /
  CLEF_WINDOW_OWNER）；PID 来源为 descendants.json `entries` + /proc pgid 扫描；不触发
  NPU/停止/恢复；缺失记缺测。
- preflight-rescan.sh：执行前完整重扫（identity/boot/五 PID/health/info/clients、端口、
  四路径 absent、logdir 可写、全部冻结 SHA、无其他格）。
- restore-and-failure.txt：唯一 restore argv 与失败留现场条件。
- validation-readonly.txt：本轮只读验证输出（preflight-rescan ALL PASS；rank-env-read 空结果）。

33 问题的直接回答
-----------------
① device 日志既有路径/采集方式：
   - 源头：`/root/ascend/log/run/device-0/device-<pid>_<timestamp>.log`、
     `/root/ascend/log/run/device-1/device-<pid>_<timestamp>.log`；
     plog：`/root/ascend/log/run/plog/plog-<pid>_<timestamp>.log`；
     vllm_ascend：`/root/ascend/log/vllm_ascend/vllm_ascend_*.log`。
   - 既有 cell（a14/a15/h11）证据里的 `device-0-<pid>.log`、`plog-run-<pid>.log` 是上述文件
     的改名副本；仓库内没有独立采集脚本。collect-raw.sh 给出按时间窗复制完整文件的命令，
     源头目录已确认存在且可写（device-0/1、plog、vllm_ascend 均为 root 可写）。
② official/activate/service-root/model-dir 实路径（由 restore_argv 推得并已 test/ls）：
   - `--official-start /workspace/yangchen/Clef-27B-ascend/Clef-27B/start.sh`
   - `--activate /workspace/yangchen/Clef-27B-ascend/scripts/activate_env.sh`
   - `--service-root /workspace/yangchen/Clef-27B-ascend`
   - `--model-dir /workspace/yangchen/models/Clef-27B`（目录存在）
③ `--execute` 模板与 build_plan 一致性：完全一致（含显式 `--python`、`--expect-stop-sha`、
   `--stop-timeout 120.0`、`--restore-timeout 900.0`、`--port 18143`、`--clean-env-expect`）。
   build_plan 的 required 项均已提供；未使用 `--child-argv`（它会覆盖 `--arm`）；`--a12-binary`
   仅 A12 需要；`--prior-client-record` 与 `--cwd` 为可选（本模板不传，保持默认 preflight 扫描）。

状态
----
- plan-only 复核通过（c95ae5c3，逐字节一致）；execute 模板仅存档，未执行。
- 无设备动作、未停服、未启动 torchrun；服务 root 1337814 健康。
