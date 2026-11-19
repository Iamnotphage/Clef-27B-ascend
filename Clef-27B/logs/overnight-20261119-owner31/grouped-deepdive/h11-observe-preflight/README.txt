h11-observe-preflight — H11 observe v1 只读 plan-only 预检（无执行）
=====================================================================

状态
----
- plan status = `plan_not_executed`；无服务停止、无 torchrun、无设备动作、无 GO。
- 计划由冻结窗口的既有 plan-only 入口生成（`supervised_probe_window.py`，未加
  `--execute`），stage 未重建、未重编译、未重跑 Linux 40。
- 冻结字节未因文档更新而变化：run_probe `8e308804`、window `1f780bf8`、
  binding `101660f8`、adapter `d181b15a`、test `ad257b5f`、manifest `8e324bea`。

计划参数（plan-h11-observe-v1.json）
------------------------------------
- arm H11；rows 128 / k 256 / n 512（M128/K256/N512），world 2。
- H11 臂契约（见 h11_matched_control.md）：flags=KEEP_ALL、native=True、FP32 zero bias、
  Grouped op、单元素 tensor lists、同 HCCL comm group；沿用适配器 d181b15a。
- seed 20261119；arm-timeout 240.0；comm-timeout 90.0；master_port 29531。
- child argv：torchrun 2 ranks + e38b541 stage 的 run_probe.py + `--adapter` + `--execute`
  （`--execute` 属于未来执行步，本次没有运行）。
- restore argv：`env -i` + 干净 PATH + `CLEF_LAUNCH_CONFIG=s1` + `setsid bash -c
  "cd .../Clef-27B-ascend && source activate_env.sh && exec bash Clef-27B/start.sh ..."`；
  restore 输出目录为 fresh `restore-h11-observe-v1`。
- 成功门与失败分支沿用窗口既有契约（branch aa3e-noop、four_checks、env、joint_env、
  two_workers、clients_empty 等）。

现场只读核验（preflight-live-checks.txt）
------------------------------------------
- fresh identity：pid 1337814、ppid 1、pgid/sid 1337814、starttime 400405818、
  boot_id 393ecc15-721f-46f7-b119-21fa327a2b3b、state S；identity 文件 SHA `de780872`。
- 服务健康：/health ready（model clef、revision local-65e5a3c5ffb6）；
  /v1/decide/info config=s1、TP2、max_model_len 4096、cache 全关。
- 五 PID：1337814 root、1337961 resource_tracker、1337962 EngineCore、
  1337979 Worker_TP0、1337980 Worker_TP1；全部 pgid/sid=1337814。
- 端口 29531：TCP/UDP × 127.0.0.1/0.0.0.0 全部 FREE（host 无 ss，用 python
  socket bind 探测；四次绑定均成功）。
- fresh 目录：cell 与 logs-h11-observe-v1 已创建；run / probe-temp / probe-out /
  restore 四个路径均 ABSENT（生成 plan 后复查仍为 ABSENT）。
- stop 脚本：/tmp/d0cc/d0_stop.py SHA `d6d49d75`；source `clef_joint_schema.py`
  SHA `aa3e1758`（aa3e）；text/image body 取自 A15-large cell：
  `a7dd3f64` / `f3229f5c`。
- 入口脚本：Clef-27B/start.sh 与 scripts/activate_env.sh 均存在。

环境 sidecar（env-sidecar-h11-observe-v1.json）
-----------------------------------------------
- 候选：`ASCEND_GLOBAL_LOG_LEVEL=0`、`ASCEND_PROCESS_LOG_PATH=<cell>/logs-h11-observe-v1`。
- 仅记录、未应用；未来若 GO 只作用于 child，restore 仍为 `env -i`，不污染服务。
- 条件项（未证）：AICPU env 向 device 侧 libccl_kernel 的传播；HcclCheckLogLevel/
  DlogRecord 的 w0/w1 原始操作数与任何级别枚举的映射；模块级格式与
  HCCL_DEBUG_CONFIG bit 语义。现有 env 空/HCCL_DEBUG_CONFIG=0x0 只作默认配置事实。

文件
----
- plan-h11-observe-v1.json：窗口 plan-only 入口原始输出（status plan_not_executed）。
- stage-identity-h11-observe-v1.json：冻结 SHA、identity、端口、目录、body、sidecar 索引。
- root-identity-h11-observe-v1.json：从 /proc 实时抓取的 fresh identity 副本。
- preflight-live-checks.txt：本轮只读命令原始输出（SHA/identity/health/port/进程/目录）。
- env-sidecar-h11-observe-v1.json：候选 env 与未证条件。

边界
----
- 本目录不含任何执行产物；plan 中的 `--execute` 仅在明确 GO 后才可被使用。
- 服务保持 aa3e/s1（root 1337814），无额外写入（除 fresh cell/logs 目录与 identity/
  sidecar 文件外）。
