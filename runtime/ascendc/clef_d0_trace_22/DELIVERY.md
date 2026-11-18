# D0 诊断包交付（路线 C，**纯编译**；未安装、未加载、未上板**）

## 身份（产物实证）
- vendor `clef_d0_trace_22`；op `ClefD0Trace`；kernel 入口 `clef_d0_trace`
- 生成 JSON `kernelName` = **`ClefD0Trace_7cc2cb4971aae629b21eab47b33a828f`**（非 E）
- 生成头 `aclnn_clef_d0_trace.h`：`mcDiag`/`ClefMatmul` 出现 **0** 次（无 E 残留）

## 构建产物 SHA-256
| 产物 | SHA-256 |
| --- | --- |
| `custom_opp_ubuntu_aarch64.run` | `7cc64277d2fa20a0b5b8f3d0dc496bed5f1138bbe59498703b7f2995b1db4c31` |
| kernel `.o`（`ClefD0Trace_7cc2cb49….o`） | `27030ab4b9ee710e097718dcf0b64a57daee10760875b885922b837c78ba7926` |
| `libcust_opmaster_rt2.0.so` | `505090b192d5756fd6bb5331f6d0df9b93f24d658686cb5825b894dc3be1a249` |
| `libcust_opapi.so` | `3314922677b7b78164628f1bc22c7339ac63d59cf537c261533cbf740ac948ec` |

## ABI（供 21 对接）
- **输出**：`out`(bf16/ND)、`residual_out`(bf16/ND)、`rstd`(fp32/ND)、`diag`(fp32/ND)、
  **`trace`(uint32/ND)** —— 第 5 个输出，**独立分配**，不复用 `diag`
- `trace` shape **`[72, 8]`**，共 **2304 B**；host 侧按此声明，adapter 按此读取
- **属性保持 14 个**（与原 E 包同集合；新增输出不改属性表）
- **清零**：调用侧在**同一 stream** 于调用前完成；kernel **只写自己的槽**
- 槽式：`AIC slot = block`（`block<24, sub==0`）；`AIV slot = 24 + block*2 + sub`（`block<24, sub<2`）
- 字段：`0 magic=0xD0D0C0DE` / `1 id=(block<<8)|sub` / `2 domain` / `3 claim_start_m` /
  `4 claim_m` / `5 stage_bits` / `6 stage_bits2` / `7 seal=0xC0FFEE00 ^ id`

## 路线 C 守卫（已实现）
`linear=GetBlockIdx()`、`sub=GetSubBlockIdx()`、`ratio=GetTaskRatio()`；
校验 `ratio>0`、`0<=sub<ratio`、`linear>=sub`、`(linear-sub)%ratio==0`；
`block=(linear-sub)/ratio`（**不省略 sub**）；再 `block<24`；**AIC `sub==0`、AIV `sub<2` 分别守卫**；
**槽分支写**（非 `(isAIC?0:24)+block*2+sub`）。任一失败**仅禁该参与者 trace 写**，
REGIST/计算/End/屏障/退出**原控制流继续**。

## 证据分层（三层分列）
| 层 | 状态 |
| --- | --- |
| 源内同函数证据 | `sys_var_intf_impl.h:66` `GetTaskRatio→GetTaskRationImpl`；c220 `GetBlockIdxImpl` 用同一函数 |
| **编译可用性** | **本次完整 TU 编译通过**（BUILD_EXIT=0） |
| **运行一致性** | **未验证**（ratio 实值、`(linear-sub)%ratio==0` 是否恒成立） |

## 明确未测
- 8 字段**不含独立 ratio 读回** ⇒ **不宣称 D0 直接测得 ratio**，也**不宣称完整 launch**
- 双 pass 实际写回/顺序、`partial` 实际写入覆盖、GM 消费可见性：**均缺**
- 插桩改变代码与时序，**只证明本包自身**
