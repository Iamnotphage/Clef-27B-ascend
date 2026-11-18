# SmallQ B/D：每调用元数据读取去重

执行边界：20 排程、23 核验、27 唯一设备 owner。26 已完成 CPU 实现和检查；当前没有 B/D 设备结果或服务采用结论。

## 冻结的单变量

- B：`2e8af5d59fc88de8922276ed4f55db187b7e2fea` 原 helper，四冻件保持原字节。
- D：独立 `runtime/joint_smallq_dispatch/clef_joint_schema.py`，仅 helper 的 tensor/parameter 元数据检查改成每调用读取后复用；同对象 K/V 检查一次。
- env、enabled、training、grad、module 配置、输入和参数 dtype/device、shape/stride、B1/Q1–8/L1–2048/width1024 均继续每调用检查。没有跨调用缓存。
- projection tail、投影对象身份、routing 两次 norm、decoder 共享 K/V、布局、残差、FFN、single-field/多字段分支均保持原代码。
- kernel 字节保持 `08ad88f6cebfb148aff8d973097d5bcf234e25d144a824385b18b8b078e82dff`；默认 `CLEF_JOINT_SMALLQ=0`、lazy import、域外原 MHA。

`frozen-guard.diff` 是 B→D 的完整 postimage diff。probe 启动时还用 AST 比较除 helper 以外的整个模块、helper 的前四条语句、参数签名及 projection tail，差异越界即拒绝。

控制流元数据 stand-in 计数：有效调用的 tensor/parameter 属性访问 B=47、D=23，共享 K/V 时 D=18。该计数仅验证去重工作量；实际 wall/device 收益由设备 raw 决定。

## 同输入诊断

同一个 head 实例、同一个真实 checkpoint 参数存储；B/D 按完整源方法绑定，绑定和 hook 均在正式计时外。A 绑定经 SHA 核验的 f810 方法，仅作为数值 gate 参照；A 的 core 首次调用计时单列，不参与正式 B/D 比较。

- checkpoint `a010ac04f078e699988e4049cbea5e62c962393f59fec366640b64e8d69a4953`，seed20261119，9 组输入逐字节哈希与首格相同。
- 原门保持：core max_abs≤.0078125、head≤.03125、prob≤.005，全部 finite。D 额外必须对 B **逐位一致**。
- 六 core + 三 head + 两 fallback 共11组 gate；head A 六 MHA 与 K/V 身份必须正确，B/D 六 backend、零原 cross-MHA；fallback 两例逐位一致、零 backend。
- 另核 D flag0 与 A 逐位一致，backend 调用会立即报错。
- 所有门通过后才正式计时：9格×4轮×2臂×20 = **1440 raw**；每臂预热5次，顺序 `BD/DB/DB/BD`。
- 六 core 格的 B/D 使用同一个 kernel callable，作为波动对照；真正改动在三个完整 head 格。
- 每样本分列 wall/device/host_submit/sync_wait/event_record_host。device event 包含 stream gap，host_submit 可包含框架内部同步，均不可直接解释为纯 kernel/纯 Python。
- 切臂上下文退出（包括异常）恢复方法、backend 属性与环境。生产源文件保持不动，末尾复核 D/head、kernel、B reference SHA。

wrapper schema 保持 `joint-smallq-probe/1`；本实验必须为 `experiment=joint-smallq-bd/1`。原双臂0/1分析器与 A/C/B 分析器均不直接套用本结构。

## 完整执行 profile

`profile.json` 提供完整66元素 argv、7个 source→stage_rel→SHA 和所有冻结门。远端自足 package：

`/workspace/yangchen/work/scratch-26/joint-smallq-bd-v1/`

profile SHA：`71f7e7a190189ead00df6a6e9d3b5201a7df1c081aef1807a84629958651bf3b`

| 文件 | SHA256 |
|---|---|
| D postimage | `086357bddbe4fbfbda1630e5ac70503c44f06113d582089c5c54669720d3260a` |
| D installer | `13dff04404e664b77c61e1eb2dd85df9610224a80fd0899cf3ef0e4d63e084c7` |
| B/D probe | `94ca8428cd44b8130bdb1915b93069a7f687e2ccc957f8218ec9c10fc59030b7` |
| kernel（与 B 相同） | `08ad88f6cebfb148aff8d973097d5bcf234e25d144a824385b18b8b078e82dff` |

沿用 5c 完整 wrapper，SHA `9245e1a6504742dc4706eb466f8053cb3999ff3d80abfac960ed218a37a747d5`。预定新目录 `/workspace/yangchen/logs/joint-smallq-bd-27-v1-first`。由 20 排程并获核验后，27 使用以下单条命令按 profile 原样执行：

```bash
/workspace/yangchen/venvs/clef-py312/bin/python -c 'import hashlib,json,pathlib,subprocess,sys; p=pathlib.Path("/workspace/yangchen/work/scratch-26/joint-smallq-bd-v1/runtime/joint_smallq_bd/profile.json"); assert hashlib.sha256(p.read_bytes()).hexdigest()=="71f7e7a190189ead00df6a6e9d3b5201a7df1c081aef1807a84629958651bf3b"; sys.exit(subprocess.run(json.loads(p.read_text())["argv"]).returncode)'
```

installer 的 `SOURCE=runtime/joint_smallq_dispatch` 必须与 postimage/kernel 相邻。B reference 与 f810 reference 分别 stage 到 `runtime/joint_smallq_bd/b_reference.py`、`f810_reference.py`；输入 manifest stage 到同目录 `inputs.json`。所有 source、rel 均在 profile 内，无占位符。

生命周期全部复用 wrapper：先核 f810 且无 smallQ；精确 stop、apply/check、probe、cleanup、revert、恢复四检。installer 拒绝 B/24/未知 postimage 混装。每步失败即按 wrapper 闭合，unknown/residual 必须空；恢复要求 f810、kernel 移除以及 health/workers/text/image rc0/oktrue。没有新增 stop/restore 脚本。

## CPU 验证与判据

远端 scratch 用固定 Python、`TORCH_DEVICE_BACKEND_AUTOLOAD=0`、`OMP_NUM_THREADS=2` 纯 CPU 检查 **9/9 通过**：100组 Q/L/共享对象边界，逐条 tensor/module/parameter 拒绝条件与恢复，env/grad 变化、默认关闭、无跨调用缓存、完整 head CPU 回退、同 storage、异常恢复、输入哈希、installer 往返与混装拒绝。测试结束明确检查 `torch_npu`/Triton 均未导入。另在本地按 profile 实际 stage_rel 建临时树，7 payload SHA 与 installer check/apply/check/revert 均通过。

首次远端 CPU 检查遗漏关闭 torch backend autoload，在导入 torch 时以 `libhccl.so` 缺失退出；测试入口现显式设置 `TORCH_DEVICE_BACKEND_AUTOLOAD=0`，重跑上述9项通过。该首错发生于 CPU 测试导入阶段，未产生设备精度或性能样本。

核验命令：

```bash
TORCH_DEVICE_BACKEND_AUTOLOAD=0 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  /workspace/yangchen/venvs/clef-py312/bin/python runtime/test_joint_smallq_dispatch.py -v
```

只按同窗口 B/D 三个完整 head 的四轮配对 raw 判断该单变量。若方向不稳定，则该方案缺少稳定收益证据；若稳定更快，仍需之后同窗口 f810 对照，不能用旧 A/C/B 的绝对耗时推算生产收益。服务 S1/S2、全量 OD/ERQA≤2pp 独立保留，生产默认 off。
