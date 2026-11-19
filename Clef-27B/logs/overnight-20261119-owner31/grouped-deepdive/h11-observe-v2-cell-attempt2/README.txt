h11-observe-v2 attempt2 cell evidence (provisional causal note)
attempt2 执行摘要（H11-observe-v2 唯一 window；handle -2）
- child 240s 超时：api_returned=false/sync_returned=false（两 rank）；cpp launch_return rc0、handler_return rc0，python sync 未返回。
- window cleanup：两独立 PGID 均在 groups=[1354310,1354311] 且 killed=true；drift/unknown 空。
- window restore：allow=false/blocked_probe_residual（gate 时刻两 PID 仍存活），restore 未发起、无 restore.log。
- 20 授权补发第一次唯一 restore：plan restore_argv 原样，14:56Z 启动成功；新 root 1355077 / starttime 401718579 / 五 PID / health ready / info s1 / branch aa3e-noop / source aa3e；clients 0；29531 free。
- collect end：92 文件、SUMS 52d145ee；因 probe-temp 目录 ABSENT，end rc1（如实记录，不补造）。
- 因果标注：cleanup 覆盖两 PGID 但 gate 仍判残留；保留事实，不作最终因果结论。
- 人工唯一 restore 验收：见 manual-restore-record.txt（argv/来源/启动时刻/root 1355077 ppid 偏差/
  五 PID/9 项检查全 true/source 三段/subreaper/clients/29531/残留）。
- 缺失事实：probe-temp 目录不存在；collect end rc1 原样保留，92 文件 SUMS 52d145ee。
- cleanup 边界：groups 两 PGID killed=true、unknown/drift 空；gate 仍判两 PID 残留并阻断；
  signal 与 gate residual 集合不一致原因保留待查，不作最终因果判定。
- marker grep：marker-grep-report.txt（18 个 marker 计数/命中文件/时间/rank）、marker-grep-full.txt（1264 行
  完整命中原文）；命中文件在 archive-marker-hits/（12 文件；>5MB 的 6 个 debug plog 存 .gz，
  原始 SHA256 与解压校验见 archive-marker-hits/MANIFEST.txt，12/12 OK）。
- 恢复复核原始 JSON：recheck-9checks-current-20261119T151015Z.json（sampled_at 2026-11-19T15:09:22Z；
  明确为“当前时刻复核”，不是恢复瞬间历史原件；此前两次终端 9-check 未落盘）。
- outer restore argv 追溯：outer-restore-argv-evidence.txt（/proc/1355076 cmdline 为可用原件；
  剥离 shell 包装/重定向后 head[0:7] 与 inner -c 均与 plan restore_argv 相同，semantic match=true；
  无独立历史启动脚本文件，不补造）。
- 语义结论（API 返回/初始快照）待 32 统一；本目录不做最终定论。
