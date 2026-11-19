h11-observe-v1-cell-attempt1 — 唯一一次授权执行（child 启动前失败；已 restore）
================================================================================

授权与包
--------
- 20 明确一次 GO；固定包 953eb2d / SUMS 754859dc（11/11），plan c95ae5c3。
- 冻结 stage：e38b541 / run_probe 8e308804 / window 1f780bf8 / binding 101660f8 /
  adapter d181b15a；H11 M128/K256/N512/world2/BF16/KEEP_ALL/native/FP32 zero bias/seed20261119。
- 启动前 mandatory rescan：ALL PASS（rc=0，含包 SUMS 11/11），见
  ../h11-observe-execution-v1/validation-readonly.txt 的上一轮与本次现场输出。

执行结果：失败（child 未启动），已按唯一 restore 恢复
-----------------------------------------------------
- window rc=1；`window.json.error = FileNotFoundError: [Errno 2] No such file or directory:
  'torchrun'`；`steps.child = {started: false, error: ...}`。
- 根因：启动 shell 的 PATH 为
  `/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin`
  （不含 venv bin）；`command -v torchrun` 为空，torchrun 仅存在于
  `/workspace/yangchen/venvs/clef-py312/bin/torchrun`。window 按 plan 的 child_argv 以裸名
  `torchrun` 启动，`subprocess.Popen` 在 fork 前即抛 FileNotFoundError。
- 无 rank 进程产生：descendants.json 不存在，probe-temp / probe-out 目录不存在；
  `rank-env-h11-observe-v1.jsonl` 共 26 次采样，全部为空结果（root_pid null/entries 0/
  rows []），按“空/缺失=缺测”记录，不视为无 rank 或排除路径。
- stop：started 13:32:16Z，finished 13:32:19Z，rc=0；targets=[1337814,1337961,1337962,
  1337979,1337980]，termed 全部五项，killed=[]，refused=[]，survivors=[]；unavailable 五项
  均为 `already gone`（term 时已退出）。
- restore：唯一 env -i + 官方 start.sh，wrapper pid 1347085；health/info/workers/text/image/
  env/joint_env/source/clients 九项检查全 true；branch=aa3e-noop；isolation_failed=false；
  subreaper original 0 → cleared true → final 0（final_restored 0）。
- 当前服务：root 1347086 / starttime 401212026 / ppid 1 / pgid=sid 1347086 / boot
  393ecc15…；五 PID=[1347086,1347229,1347230,1347258,1347259]；health ready（clef，
  revision local-65e5a3c5ffb6）；29531 tcp/udp × 127.0.0.1+0.0.0.0 仍 FREE。
- source 三段 SHA（before / after child / after restore）均为
  aa3e17585f8bd60a4d05e342abc7e2773c79a0b2d49f0370d15e29bcc6b72503。

采集（现场保留）
----------------
- 远端 archive：/workspace/yangchen/work/grouped-deepdive-cell-h11-observe/archive-h11-observe-v1/
  （51 文件，含 device-0/1、run/plog、vllm_ascend、fresh-logs、cell/run、cell/restore、meta、
  sources.txt、correlation.txt；probe-temp/probe-out 缺失被显式标记，end rc=1）。
- archive SUMS：`4fbf9653de5d62f80436f41246c7c7ef884d13b1acca2dd057a7d59c4ce90000`。
- rank env 采样：`<CELL>/rank-env-h11-observe-v1.jsonl`（26 次，全空=缺测）。

规则遵循
--------
- 仅一次 `--execute`，未重试；未启动第二棵 restore；child 失败未自动重启。
- env 仅导出 sidecar 两项（GLOBAL_LOG_LEVEL=0、PROCESS_LOG_PATH），未 unset 其它变量。
- 不重置设备、不改 SSH、不全局 kill；stop 只处理登记的五 PID 与 PGID。
- 本目录为事后证据副本；完整 raw 在远端 archive（按 SUMS 校验）。

后续（待 20/32/33 决定，未执行）
--------------------------------
- 包模板缺 PATH 处理：child_argv 的 `torchrun` 为裸名，执行 shell 不含 venv bin。若再次授权，
  需要（a）在启动 shell 的 PATH 前置 venv bin，或（b）用绝对路径 torchrun 覆盖 child argv；
  两者属于授权前需批准的包修订。本轮不做任何修改、不重试。
