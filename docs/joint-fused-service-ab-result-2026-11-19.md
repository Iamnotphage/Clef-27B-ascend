# head 服务 A/B 结果（aa3e 内部 flag 0/1，2026-11-19）

执行 owner 27；runner `c0a176d` / `127e47ae…3caa648`；run
`/workspace/yangchen/logs/joint-fused-service-ab-27-c0a176d-first`。
两臂同源 aa3e，仅 `CLEF_JOINT_FUSED_ATTN=0/1`；固定 od-0105/od-1534/erqa-0000，
4 轮 ABBA × 5 reps + 3 warmup。overall RC0，四轮全部 valid，8 次 stop 均
`alive/unknown/zombies` 为空。

## 配对延迟（每轮 5 样本中位，flag1−flag0）

| 输入 | 四轮 Δms | 配对中位 Δms | 中位 % | 方向 |
| --- | --- | ---: | ---: | --- |
| od-0105 短文本 | +2.375 / −2.723 / +6.034 / −2.239 | +0.068 | +0.066% | 2+/2−，持平 |
| od-1534 长文本 | −1.112 / −2.668 / −0.608 / −0.968 | **−1.040** | **−0.668%** | 4/4 负 |
| erqa-0000 图文 | −1.088 / −0.944 / −0.697 / +9.198 | **−0.820** | **−0.751%** | 3/4 负 |

短文本无可测收益；长文本稳定小幅下降；图文中位下降但第 3 轮 +9.198 ms 为离群。

## 固定输入质量

- 每输入/臂 20 个非 warmup 样本，全部 `valid=true`，期望类型 choice/score/choice。
- 每输入/臂只有 1 个唯一 answer 数值签名；两臂签名集合相等，`max_numeric_abs_diff=0.0`。
- 选择不变：od-0105 `professional_writing`、od-1534 score、erqa `B`。

## 恢复

`steps.revert/service_check_revert/restore` 全 ok；restore label
`d0-restore-20261118T190530Z-150b9178`，`verify_ok=true`，service pids
`1212699/1212822/1212823/1212859/1212860`；health ready；source `f8101b98…`；
guard `76595657…`。24 只读核对，未操作设备。

## 结论边界

本结果只证明 **aa3e 内部 flag0→flag1** 的固定输入增量和数值一致性；不能作为生产
f810 的总收益。最终采用仍需真 f810 来源对照与完整 S1/S2、OD/ERQA 质量门；当前不预设赢家。
