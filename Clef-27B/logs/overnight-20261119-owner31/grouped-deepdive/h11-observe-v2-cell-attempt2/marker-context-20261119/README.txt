attempt2 KFC/task13 上下文只读证据包（2026-11-19）
================================================

来源与完整性
------------
- archive：/workspace/yangchen/work/grouped-deepdive-cell-h11-observe-v2/archive-h11-observe-v2-attempt2
- 文件数 92；archive SHA256SUMS=52d145ee6d21ef3b093c3b6d61becb6f8f0f4aa2b56af39ea4bcbaea641005a3
- 命中文件的原始 SHA256/大小/存储校验：见子目录 ../archive-marker-hits/MANIFEST.txt（12/12 OK；>5MB 的 6 个 debug plog 为 .gz）。
- 本包所有内容均由上述 archive 只读 grep/扫描生成；未重跑 probe/window，未改原件。

方法（可复核）
--------------
- 文本 grep：`grep -r -a -F` / `-c/-l/-n`，全 archive 92 文件；marker 计数与命中原文见 marker-grep-full.txt（1264 行）。
- 时间窗上下文：UTC 14:49:03–14:49:08 与 14:51:50–14:52:20（覆盖 child 启动与 14:52:06 三分钟超时），
  对命中行取同文件 ±2 行上下文；输出 kfc-window-extract-2456.txt（2456 行，未截断）。
  注：device 日志时间戳为本地 +8h（如 22:49），plog 为 UTC（14:49），两者同刻。
- 聚焦 grep：RunAicpuKfcSrvLaunch / task_type=1(KERNEL_AICPU) / TaskFinished(stream11/47) /
  notify_record/notify_wait / GetTaskType / AicpuKfcRpcServer / AicpuCcOpExe 等，见 kfc-focused-greps.txt。
- stream47 TaskFinished 与 notify 简表：stream47-notify-and-taskfinished.txt。

文件索引
--------
- kfc-window-extract-2456.txt：18 个 marker 的计数 + 两个时间窗 ±2 行上下文 + task_id=13 行（2456 行）。
- kfc-focused-greps.txt：KFC/task 完成/notify/Init 相关聚焦行（98 行）。
- stream47-notify-and-taskfinished.txt：stream47 全部 TaskFinished（尾 40 行）、task13 recycle 查询、
  notify_record/notify_wait 紧凑表。
- marker-grep-full.txt：18 个 marker 的全量命中原文（1264 行）。
- static-launch-direct-callers.txt：RunAicpuRpcSrvLaunch / V2 / RunAicpuKfcSrvLaunch 的
  **直接 BL/B 绝对目标**扫描结果（0 个）。
- static-init-direct-callers.txt：Init4(y,j,t,KFCTask) / Init(y) 的直接 BL/B 扫描结果（0 个）。
- static-mc2-function-calls.txt：MC2V2/MC2/Api/Launch/LaunchV2 函数体内 BL/B 目标与符号解析清单（222 行）。

静态核对的覆盖范围与限制（必须与结果一起阅读）
----------------------------------------------
- 上述“0 callers”仅指：.text 内 **直接 BL/B 到绝对符号地址** 的扫描无命中。
- 未枚举经 .plt（区间 0x27fdf0 + 0x1c520）转发、函数指针、dlsym/按名解析的调用；
  device 日志已出现 `Get api RunAicpuKfcSrvLaunch from so libccl_kernel.so success`
  与按名取 KFC kernelname 的事实，说明存在按名/间接路径。
- 因此 **0 直接命中不能外推为“MC2/MC2V2 不调用 RunAicpuRpcSrvLaunch”**，也不能外推 Init4 的归属；
  只能记录为“直接 BL/B 形式未覆盖”。具体归属判断交 33。
- MC2/MC2V2 函数体内已符号解析的直接调用包含 `AicpuKfcProcess::RunRpcServerApi+0x161c`、
  `AicpuRunRpcServerForApi+0x1630/0x215c`、`AicpuRpcResInit+0x3ae4` 等（见清单），
  但同样不代表其它路径不存在。

不做的事
--------
- 不重跑 probe/window、不访问服务、不改 archive/window.json/80c73d9。
- 不对“API 是否返回”“D-state/不可杀”“MC2 调用归属”下最终结论；这些交 32/33/20 统一。
