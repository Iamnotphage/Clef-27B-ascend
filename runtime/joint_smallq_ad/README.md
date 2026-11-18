# 直接 f810 A / 冻结 D 同窗口诊断

20 授权该直接对照，23 核验，27 在24/rstd恢复闭合后按20排程唯一执行。当前只有 CPU 验证结果。D postimage、installer、kernel与B全部冻件保持字节不变；本实验不增加kernel/tiling候选。

正式 A：同一个真实权重 head 实例上，精确绑定 f810 的 `JointSchemaHead.forward`、`_single_field_decoder`、routing `EvidenceRoutingLayer.forward`。正式 D：同一实例绑定冻结 `086357bd…` 的对应方法。切臂在计时外，参数存储不变。A head 原 MHA 与 D smallQ/adapter 的总差异由本窗口直接测量。

`probe-from-bd.diff` 给出相对已核B/D骨架的完整最小diff：正式顺序 `AD/DA/DA/AD`，配对统计 `D_minus_A`，实验标识/路径/说明相应更新。controller、输入生成、AST验证器、gate、measure及fallback逻辑沿用。B保留为**计时外额外参照**，继续校验D/B逐位一致；`single_variable_verified`字段指D相对B的guard单变量检查，不把A/D差异描述为仅guard。

约束保持：真实checkpoint SHA `a010ac04f078e699988e4049cbea5e62c962393f59fec366640b64e8d69a4953`，seed20261119，9个固定输入哈希；BF16、原域与动态guard。先11组原gate、D/B bitwise、defaultoff bitwise/零backend、两fallback bitwise/零backend、六cross-call和共享storage检查；原门core .0078125/head .03125/prob .005与finite不变。随后9×4×2×20=**1440正式raw**，每臂每轮warmup5。core A为projected SDPA、D为冻结smallQ，和三个完整head分开报告。

每样本仍保存wall/device/host_submit/sync_wait/event_record_host五字段；首调用单列。device事件含stream gap，host_submit可能含框架内部同步。只据同窗口D−A逐轮差值/范围/方向判断原head净收益；不跨窗相减，不自动升级服务AB，S1/S2和完整OD/ERQA≤2pp另行保留。

## 冻结执行 tuple

远端自足包 `/workspace/yangchen/work/scratch-26/joint-smallq-ad-v1/`。`profile.json` 内为完整66元素argv、七个source→stage_rel→SHA、所有验收与恢复约束。

- profile SHA `a7c418af865a3482bec1a60be0173355ddc2bc7c478e68dc65995f8ff2a51a6d`
- 新probe `runtime/joint_smallq_ad_probe.py` SHA `265af902ad3fe39ba51796319035c082fd9dd6c133fc142835a6ec9b295dfbcb`
- D postimage `086357bddbe4fbfbda1630e5ac70503c44f06113d582089c5c54669720d3260a`
- D installer `13dff04404e664b77c61e1eb2dd85df9610224a80fd0899cf3ef0e4d63e084c7`
- 相邻kernel `08ad88f6cebfb148aff8d973097d5bcf234e25d144a824385b18b8b078e82dff`
- 5c完整wrapper `9245e1a6504742dc4706eb466f8053cb3999ff3d80abfac960ed218a37a747d5`，路径仍为scratch-24/joint-fused-5c045eb。

新run `/workspace/yangchen/logs/joint-smallq-ad-27-v1-first`。由27按20排程执行：

```bash
/workspace/yangchen/venvs/clef-py312/bin/python -c 'import hashlib,json,pathlib,subprocess,sys; p=pathlib.Path("/workspace/yangchen/work/scratch-26/joint-smallq-ad-v1/runtime/joint_smallq_ad/profile.json"); assert hashlib.sha256(p.read_bytes()).hexdigest()=="a7c418af865a3482bec1a60be0173355ddc2bc7c478e68dc65995f8ff2a51a6d"; sys.exit(subprocess.run(json.loads(p.read_text())["argv"]).returncode)'
```

probe schema为`joint-smallq-probe/1`，experiment必须`joint-smallq-ad/1`。复用原wrapper stop/apply/check/probe/cleanup/revert/restore；前后核f810、kernel移除、unknown/residual空及health/workers/text/image rc0/oktrue。保持候选互斥。

## CPU 核验

controller、输入生成及AST验证器与已核BD逐段AST相等；四轮顺序正反各两次。固定远端Python以`TORCH_DEVICE_BACKEND_AUTOLOAD=0`执行新AD controller绑定的完整head CPU回退/同storage/异常恢复检查与九组输入hash检查，**2/2通过**，torch_npu/Triton均未导入。实际七payload哈希、冻结wrapper哈希、66argv与新stage相邻布局全部匹配；在标准库临时树执行stage后的installer check/apply/check/revert通过。B/D生产冻件与BD probe/profile无diff。尚无A/D设备或服务结果。
