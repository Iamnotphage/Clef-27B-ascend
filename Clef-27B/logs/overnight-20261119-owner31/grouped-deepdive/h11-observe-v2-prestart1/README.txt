h11-observe-v2-prestart1 — 第一次 v2 GO 在 collect begin 阶段中止的封存证据
=========================================================================

事实
----
- GO 包 991c21c / SUMS 6075c4c0（22 文件）；GO handle h11-observe-v2-go-20261119-1。
- 在 window 启动前执行 collect-raw.sh begin 时发现 identity 硬编码 v1 pid 1337814：
  /proc/1337814 已不存在 → FileNotFoundError；identity.json 0 字节；begin 仍 rc=0。
- 未进入 window：无 stop、无 torchrun、无 probe/restore；四路径 run/probe-temp/probe-out/restore 均 ABSENT。
- 服务未受影响：root 1347086 / starttime 401212026 / boot 393ecc15 / health ready。
- 已修复（commit 4cbe23a，后续 8f4bca74 SUMS）：identity 改为从 SRC_ID 读取并校验 pid/starttime/boot，
  begin/end fail-closed；本轮另加 wrapper 的 collect-begin 硬门与 metadata 校验（见 execution-v2 包）。

封存
----
- 远端原件保留：`archive-h11-observe-v2`（未删除，未改动）。
- 独立快照：`archive-h11-observe-v2-prestart1`（cp -a 副本；内含自己的 SHA256SUMS，
  自哈希 4c370713812b9cc22a46648d3ff2a2fa697734ffc2202b4e5fbdd280d8b43b5f）。
- 本目录：prestart1-live-facts.txt（只读现场）、identity.json.zero（0B，SHA e3b0c442…）、
  START.empty / START_SLACK.empty、prestart1-remote-SHA256SUMS.txt（远端副本 SUMS）。
