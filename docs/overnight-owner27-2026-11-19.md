# 2026-11-19 夜间优化：owner27 实测记录

用户授权持续优化至 08:00 CST，06:30 停止新候选，随后执行至少三轮最终 S1、S2 和完整 OpenDecisions / ERQA，07:40 前恢复最佳服务。质量约束为 OpenDecisions 三项指标及 ERQA accuracy 相对 baseline 的绝对波动均不超过 2pp；固定 BF16 与既有软件组合，禁止答案缓存。全部设备操作由 owner27 串行执行，限 A3 物理卡 7 的逻辑 NPU 0/1。

## 交接恢复

20 明确确认 21 已退出并激活 27 后，27 执行经 23 审核的单次恢复脚本。旧恢复 shell PID1190564 已自然退出，没有发送信号。成功读取 `/proc`、端口查询和 `npu-smi` 后确认 18143 closed、卡7无运行进程。旧候选源码 `aa3e1758…` 通过原 apply manifest、installer 和已核 f810 备份回退。

恢复脚本 SHA-256 `f0b679f65b9fcaef0a59aa9fa0a84d9ed978393c55738c9e5716a74aa4decff5`。原件：[handoff](../Clef-27B/logs/overnight-20261119-owner27/handoff/complete.json)；远端 `/workspace/yangchen/logs/handoff27-restore-20261118T183212Z-dc78478b/`。归档脚本保留此次具体身份和路径，仅用于审计复现依据。

远端记录 02:32:12–02:33:45 CST 完成恢复，外层 SSH rc0。唯一标签 `d0-restore-20261118T183212Z-36d81642`，root1192060，双 worker1192211/1192212。health ready、服务树内双 worker、文本 score、图文 choice 四项均 rc0/ok=true。源码 SHA-256 `f8101b98228bbd472a63a1efe6827583e35c6cc9dede4379b925f2b016670863`，smallQ kernel 不存在。启动为 TP2/BF16/eager/maxlen4096/maxseqs1/tokens4096，关闭 prefix caching 与 chunked prefill。

## smallQ 首格：数值通过，整头负收益

候选 commit `2e8af5d59fc88de8922276ed4f55db187b7e2fea`，监管 wrapper `5c045eb1427e9cb902867bd7147c0d636ce91e1c`。完整参数与逐文件 SHA 见 [冻结命令](../Clef-27B/logs/overnight-20261119-owner27/smallq-first/run-frozen.sh) 和 [window](../Clef-27B/logs/overnight-20261119-owner27/smallq-first/window.json)。远端 `/workspace/yangchen/logs/joint-smallq-27-2e8af5d-first/`。

真实 joint head 权重、合成输入。六组 core、三组完整 head（每组调用顺序 Q=[4,4,1,1,1,1]）和两组 fallback 冻结门全部通过；有限值检查通过。core 最大绝对误差 ≤0.001953125，head 最大 0.0009765625，概率最大 0.00019699335，fallback 逐位一致。首调用/JIT 单列，未计入稳态。

各形状四轮 AB/BA/BA/AB，每臂每轮20个原始 wall/device 样本。下表为逐轮中位数之差的中位数，candidate−reference，单位 ms。

| 范围 | L | Q/候选数 | device Δ | wall Δ | 四轮方向 |
| --- | ---: | ---: | ---: | ---: | --- |
| core | 307 | 1 | -0.029190 | -0.036215 | 4/4 更快 |
| core | 307 | 4 | -0.063095 | -0.059882 | 4/4 更快 |
| core | 967 | 1 | -0.042280 | -0.043498 | 4/4 更快 |
| core | 967 | 4 | -0.054865 | -0.052645 | 4/4 更快 |
| core | 1595 | 1 | -0.037155 | -0.027660 | 4/4 更快 |
| core | 1595 | 4 | -0.031550 | -0.026165 | 4/4 更快 |
| head | 307 | 4 | +0.162455 | +0.163128 | 4/4 更慢 |
| head | 967 | 4 | +0.140460 | +0.154033 | 4/4 更慢 |
| head | 1595 | 4 | +0.119930 | +0.122842 | 4/4 更慢 |

结论：该版本保持未采纳；完整 head 的稳定负收益支持先做 CPU 归因，暂不进入服务 A/B 或全量质量验证。

probe rc0、gate_passed=true，清理 reaped=true/unknown=[]/residual=[]，随后 revert f810 并删除 kernel。窗口 overall_rc0；02:36:15 CST 恢复四检闭合，标签 `d0-restore-20261118T183442Z-d3301487`，root1193641，worker1193787/1193788。完整 [原始 probe（gzip）](../Clef-27B/logs/overnight-20261119-owner27/smallq-first/probe/head-probe.json.gz)、[配对汇总](../Clef-27B/logs/overnight-20261119-owner27/smallq-first/paired-summary.json)、[恢复原件](../Clef-27B/logs/overnight-20261119-owner27/smallq-first/restore.json) 已归档；23 已独立从 raw samples 重算并核验。

## 后续执行

按已核顺序继续 N1 ARon/cores1 owner27 bundle，再执行经启动 guard 有限修复的 head 服务 A/B。后续结果在本文件续记。当前两项记录仅证明交接恢复与 smallQ 局部探针，最终 S1/S2/完整质量尚待执行。
