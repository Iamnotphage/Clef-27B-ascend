# Head 融合候选设备预筛（2026-11-19 02:05 CST）

## 身份

| 项 | 值 |
| --- | --- |
| 部署 commit | `89ebd3f02b17aef100d4fc3cae036138fa94b40e`（`git archive runtime` 完整部署，105 文件） |
| wrapper | `700cf10cdc3fda18d1ae7f113158342e1ef6ec114569e52b15b134949bfe8473` |
| probe | `fda7cc8dcce719291e312345ab9f28fe122ff2b2e8b784bce285d4b37b3e15f3` |
| stop | `d6d49d75012a0789d3bff3e7b522e9cd496844ca0ee0424608ab8e6b43a33886`（commit `96ed1b4`） |
| preimage | `f8101b98` |
| 六 d0 模块 | `7cdd2cfe` / `4c4afeae` / `779542c6` / `5dae02c2` / `1c91e0c4` / `13413bf0`（与 23 联检一致） |

## 窗口结果

- `overall_rc = 0`；preflight / stop / apply / probe / revert / restore **全部 RC0**；
- stop：`termed=5 / killed=0 / refused=0`；
- probe `reaped=true`，unknown / residual 空；
- restore label **`d0-restore-20261118T180615Z-b1b3354b`**，root **1186263** 身份前后稳定，
  workers **1186403 / 1186404** 属其树，物理卡 7 chip0/1；health / text(score) / image(choice) 全过；
- 回退身份：`preimage_sha256 = f8101b98`，revert 已生效。

## 配对计时（合成输入、随机 head 权重、L967 / O4）

| 项 | 值 |
| --- | --- |
| 协议 | AB/BA/BA/AB，每臂每轮 30，总各 120 |
| **paired device Δ（4 轮）** | −0.701820 / −0.417740 / −0.438980 / −0.426990 ms |
| **配对中位 Δ** | **−0.432985 ms**，**4 轮同向** |
| pooled device | 4.707900 → **4.278670 ms** |
| pooled wall | 4.828365 → **4.392170 ms** |
| 输出一致性 | **bit_equal = true，max_abs = 0，finite** |

**SDPA 事实**：`COMPATIBLE_IMPL = None`、`NPU_COMPATIBLE_IMPL = None`、flash / math / mem_efficient **均 true**、torch_npu git `3363ea59d49f3e08`。

## 原件

`/workspace/yangchen/logs/head89-20261118T180555Z/`：
`window.json`（`65dcced4…cc5ee`）、`restore.json`（`0c9cc408…43d3f`）、`probe/head-probe.json`（`f5bd3d70…f1676b`）、`probe/partial.json`、`probe/probe.err`、`stop.json`、`apply/revert`、`root-identity.json`。

## 范围限定

**仅合成输入 / 随机 head 权重 / L967 / O4 的预筛**。达到 ≥0.3 ms 预算，**可推进真实服务 A/B 与完整 OD/ERQA**；
**尚无 S1 / S2 / 质量采用结论**。
