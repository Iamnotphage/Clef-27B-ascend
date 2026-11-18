# Norm 验证证据（2026-11-18）

## 1. 接受准则（先明示，再对照）

| 输出 | 准则 | 依据 |
| --- | --- | --- |
| `residual_sum` | `max_abs ≤ 0.01` | 与原生逐元素比较；BF16 输出粒度在 |z|≈1 时为 0.0039（2⁻⁸ 相对） |
| `rstd` | `max_abs ≤ 1e-3` | FP32 输出，参考量级 1–2；1e-3 为 0.1% 相对 |
| `Y` | `max_abs ≤ 0.02` 且 `max_rel ≤ 1%` | BF16 相对粒度 2⁻⁸=0.39%，2 lsb≈0.78%；1% 覆盖 2 lsb |

- 这是**冒烟级数值准则**，不是质量门槛；正式质量仍以全量 OpenDecisions + ERQA 的 2 pp 为唯一验收。
- 撤回此前「逐位一致 / 约 2 ulp 已验证」的措辞：只保留实测的 `max_abs`；本轮未做逐位与逐对 ULP 计数。

## 2. 受控复测（2026-11-18 12:17 CST，唯一变量：gamma 描述符改为 `[N]`）

- 命令：`./wave3_norm_launcher 967 5120 /workspace/yangchen/logs/perf-wave3-20261118/norm-native`
- 身份：launcher SHA-256 `6c027992…`（本轮），kernel `.o` SHA-256 `71e16663…`，实际加载 vendor
  `build_out/install-norm6/packages/vendors/clef_ar_add_rms_norm`（经 `ASCEND_CUSTOM_OPP_PATH`）；
  原始退出码 **RAW_RC=0**（native 判据）。
- 描述符（launcher 自报）：`x[967,5120] str[5120,1] bf16`；`gamma[5120] str[1] bf16`；`out/residual_out[967,5120]`；`rstd[967] fp32 str[1]`。
- 结果：

| 统计 | 值 |
| --- | ---: |
| `residual_sum` max_abs | **0** |
| `rstd` max_abs | 4.77e-7 |
| `Y` max_abs | 0.015625 |
| `Y` 差异分位 p50 / p90 / p99 / p99.9 / max | 0 / 0.00195 / 0.00781 / 0.00781 / 0.015625 |
| `Y` max_rel | 0.00781（0.78%） |
| `Y` 绝对差 >0.01 的元素 | 403 / 4,951,040（0.0081%） |

判读：三项均在上述冒烟准则内；`Y` 的 p90 以下为 0 或 1 个 BF16 lsb，尾部 0.78% 相对差对应 2 lsb 量级，来源是
`z*rstd*gamma` 的运算顺序/舍入不同，属可解释的数值差异而非布局错误。

## 3. 历史三段记录（缺项如实标注）

| 阶段 | 时间（CST） | 身份 | 结果 | 记录状态 |
| --- | --- | --- | --- | --- |
| 早期 TBuf 无屏障 | 11:44 前后 | kernel 无 `PipeBarrier`，install-norm3/4 | `max_abs` 1e33 级垃圾、部分 block 未写 | 原始 stdout 未保存 |
| 旧 launcher 混淆 | 11:59 前后 | 重编前后的 launcher 混用 | 曾读出 `max_abs_Y=2.17`、首元素为 0 | 原始 stdout 未保存；重编后同输入恢复 0.0156 |
| 重建成功（本记录） | 12:17 | launcher `6c027992…` + kernel `71e16663…` | 见 §2 | 完整日志 `/workspace/yangchen/logs/perf-wave3-20261118/norm-native-run/run2.log` |

- 三段之间不合并归因：第一段是 kernel 同步/写回问题（Barrier + 8-float 对齐批次 + 独立 reduce work 修复）；
  第二段是二进制混用；第三段是当前锁定身份。
- 早期 CPU 参考口径：**BF16 输入的 CPU 高精度参考**（输入量化到 BF16 后以 double 累加 sumsq）；它与 native
  的输入字节是否跨阶段相同**不能默认成立**，仅在同一受控目录（`norm-native/`）下成立。

## 4. 证据路径

| 内容 | 路径 |
| --- | --- |
| native 输入与三输出 | `/workspace/yangchen/logs/perf-wave3-20261118/norm-native/`（x/r/g/y/res 各 9,902,080 B、gamma 10,240 B、rstd 3,868 B、meta.txt） |
| 受控 stdout + 哈希 | `/workspace/yangchen/logs/perf-wave3-20261118/norm-native-run/run2.log` |
| launcher 源码 | `runtime/wave3_norm_launcher.cpp` |
| native dump 脚本 | `runtime/wave3_norm_native_dump.py` |
| kernel | `runtime/ascendc/clef_ar_add_rms_norm/op_kernel/clef_ar_add_rms_norm.cpp` |

注意：gate-B 的 `d8851ed7…` 是 11:30 的**脚手架包**，不能代表本次数值构建；本次 kernel 身份以 `71e16663…` 为准。
