h11-observe-v2 attempt2 cell evidence (provisional causal note)
attempt2 执行摘要（H11-observe-v2 唯一 window；handle -2）
- child 240s 超时：api_returned=false/sync_returned=false（两 rank）；cpp launch_return rc0、handler_return rc0，python sync 未返回。
- window cleanup：两独立 PGID 均在 groups=[1354310,1354311] 且 killed=true；drift/unknown 空。
- window restore：allow=false/blocked_probe_residual（gate 时刻两 PID 仍存活），restore 未发起、无 restore.log。
- 20 授权补发第一次唯一 restore：plan restore_argv 原样，14:56Z 启动成功；新 root 1355077 / starttime 401718579 / 五 PID / health ready / info s1 / branch aa3e-noop / source aa3e；clients 0；29531 free。
- collect end：92 文件、SUMS 52d145ee；因 probe-temp 目录 ABSENT，end rc1（如实记录，不补造）。
- 因果标注：cleanup 覆盖两 PGID 但 gate 仍判残留；保留事实，不作最终因果结论。
