# A/C/B 独立诊断 profile v1

20 已批准本诊断；由 23 审核，27 在 N1c2 与服务 AB 的恢复闭合后按排程执行。设备执行继续复用已核 5c wrapper，全程保持该 wrapper 的互斥、cleanup、revert、restore。26 只完成 CPU 实现与 staging，当前没有 A/C/B 设备结果。

## 实验变量与调用路径

同一个 `JointSchemaHead` 实例、同一组真实 checkpoint 参数与存储地址、相同输入供三臂使用。probe 明确进行**进程内方法及 backend 绑定替换**，全部发生在计时区间外；B 的四个源文件保持 `2e8af5d` 冻结哈希。

| 臂 | forward 路径 | attention core |
|---|---|---|
| A | 绑定 SHA=f810 原始 `JointSchemaHead.forward`、`_single_field_decoder` 和两层 `EvidenceRoutingLayer.forward` | 原 `nn.MultiheadAttention`；routing 两次 norm / 独立 K/V，decoder 共享 KV |
| B | 绑定冻结 2e8 的相同三类方法，`CLEF_JOINT_SMALLQ=1` | 冻结 `clef_smallq_kernel.smallq_attention` → 自定义 online-softmax Triton |
| C | 与 B 相同的方法、guard、projection、out projection，`CLEF_JOINT_SMALLQ=1` | **实际替换**模块属性 `clef_smallq_kernel.smallq_attention` 为 probe 内 `sdpa`：BSH→BNSD view→`F.scaled_dot_product_attention`→BSH contiguous |

C 的调用仍经过生产 postimage 的 `_smallq_cross_attention`。A 直接执行 f810 方法，避免沿用首格 OFF 的提前返回 helper 路径。切臂设置、环境写入、Python 方法/函数绑定均不计入样本，各臂每轮预热 5 次。所有绑定在异常及正常退出时恢复；原生产文件保持相同 SHA。

core 独立格的 A/C 都直接使用同一个原 SDPA 函数，B 使用冻结 kernel；A/C 的 core 结果也是调度波动的控制项。完整 head 才包含 adapter 的净差异。A−C 估计完整接入路径影响，C−B 比较同 adapter 下 backend/布局替换的净影响，均需查看带符号的逐轮数据。

## 输入、门与测量

- 固定 L307/967/1595、Q1/Q4，CPU generator seed20261119 和首格随机生成顺序一致。`inputs.json` 中九格 BF16 tensor SHA 来自首格原件；CPU 重生成全部逐字节匹配。实际 probe 再验证这些哈希。
- checkpoint SHA=`a010ac04f078e699988e4049cbea5e62c962393f59fec366640b64e8d69a4953`，strict load，BF16、inference_mode、同一参数存储。
- 数值门保持 core max_abs≤0.0078125、head≤0.03125、prob≤0.005、全部 finite。A 需命中六次原 MHA，Q 顺序 `[4,4,1,1,1,1]`、K/V 对象共享标记 `[false,false,true,true,true,true]`；B/C 各需六次替代 backend 且零原 cross-MHA forward。
- 多字段 L307/Q2/O8 与 L2049/Q1/O4 仍要求 B/C 相对 A 逐位一致、零替代 backend。支持域与 fallback 未改。
- 所有门通过后才计时。每格四轮 `ACB / BCA / CAB / BAC`，每臂每轮 20 样本；每对臂两个正序、两个反序，每臂位置索引总和均为 4。9 格共 2,160 个稳态逐调用样本。
- raw 样本分列 `wall_ms`、`device_ms`、`host_submit_ms`、`sync_wait_ms`、`event_record_host_ms`。wall 为 event record+函数提交+最终同步的总时长；host_submit 为函数调用时长，可包含框架内部同步。device event 包含流空隙，流空隙无法从这些计数器独立拆出；任何列都不会宣称是纯 kernel compute 或纯 Python 开销。
- 首次 core 调用另列 `first_calls`，B 首次 JIT/cache lookup 保留。head 精度检查带 hook/对照计算，全部排除在稳态计时之外。切臂选择与数据 hash/拷回均不在 timed fn 内。
- 报告为兼容 wrapper 使用 `schema=joint-smallq-probe/1`，另有明确 `experiment=joint-smallq-acb/1`、`arm_paths`、`patch_site`。新报告有 A/C/B 三臂，首格的双臂分析器不直接适用。

## 完整 source / rel / SHA 与执行

[profile.json](profile.json) 包含完整 `argv`、七个 source→stage-relative 目标及 SHA、固定 wrapper SHA、结果验收项。远端已 stage：

`/workspace/yangchen/work/scratch-26/joint-smallq-acb-v1/`

profile 路径为该目录下 `runtime/joint_smallq_acb/profile.json`，SHA：`06ede1a10230cc3b83d66180e99f51faf3fbb3dd6a0a6544cfa954cf68d4043c`。

关键文件：

- 新 probe `runtime/joint_smallq_acb_probe.py` SHA=`0b08260957fd869deee46ac2276555e4ffcda9808164f8189e9e30cc31e9390c`。
- 输入哈希 manifest `runtime/joint_smallq_acb/inputs.json` SHA=`189fd36285b762273c0ef7d1dd99cb2a43a7db50f68b5ff7ab0036b320b20429`。
- A 原始源从 `runtime/joint_single_field/clef_joint_schema.py` stage 成 `runtime/joint_smallq_acb/f810_reference.py`，SHA=f810 完整值见 profile。
- B head、installer、kernel 与原 probe 的四个完整 SHA 全部见 profile。原 probe 仅作为冻结来源载荷；实际执行入口为新 A/C/B probe。
- 默认新日志目录：`/workspace/yangchen/logs/joint-smallq-acb-27-v1-first`。目录必须尚不存在，生产源码必须已恢复 f810。若需改 run-dir，27 记录实际 argv 与变动；其它字段保持审核版。
- wrapper 源 `/workspace/yangchen/work/scratch-24/joint-fused-5c045eb/runtime/joint_fused_attention_window.py`，SHA=`9245e1a6504742dc4706eb466f8053cb3999ff3d80abfac960ed218a37a747d5`。`--repo` 同目录根，restore 六模块由现有 wrapper 核验并 stage。`--client-rel` 显式为空，采用 `head-postimage`，不配置 external vendor 或 prepend。

27 排程时可通过标准库读取 profile 的完整 argv 调用现有 wrapper；不另写停止/恢复流程。结束要求 probe rc0、`gate_passed=true`、无 error、`experiment` 正确、`same_weight_storage_verified=true`、`frozen_files_unchanged=true`、9 timing×4轮×3臂×20 raw 完整。随后同一 wrapper 必须完成 residual/unknown 空、revert f810/kernel移除、restore health/workers/text/image 四检 rc0/oktrue。

## CPU 验证（已完成）

远端固定 Python，`TORCH_DEVICE_BACKEND_AUTOLOAD=0 OMP_NUM_THREADS=1`：4 tests 全通过，覆盖均衡顺序、同存储/精确方法绑定/CPU域外数学不变、异常后方法/backend/env 恢复、九格 BF16 hash 与首格一致。未导入 torch_npu/Triton。

七文件 profile 的相邻布局和 SHA 已在临时树验证，实际冻结 installer check/apply/check/revert 全通过，源码恢复且 kernel 移除；远端七载荷与 wrapper hash 再核全部匹配。Python 语法检查通过。Triton 源码及 binary 语义未变，沿用已有编译/真实设备证据；本轮无需新增 kernel 编译。

诊断结果出来前维持生产默认 off。后续仅依据 C 的实测差异选择一个接入/dispatch/布局修复，再走相同数值门；没有证据支持先改变 tile、支持域或数值门。
