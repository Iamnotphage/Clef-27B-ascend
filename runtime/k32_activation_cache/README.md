# Clef 文本 S1 主机激活缓存

`od-s1-k32` 是显式配置，沿用 `od-s1` 的 TP2、BF16、eager、B1、4096 上下文与 4096 batched tokens。默认 `od-s1` 使用已采用的 aa3e/ead25 服务。启用前按项目固定软件栈加载运行环境。

```bash
bash scripts/start_dataset_profile.sh od-s1-k32 \
    --model-dir "$MODEL" --output-dir "$NEW"
```

启动脚本输出本次 worker 回执目录。每 rank 默认上限 22,011,707,392 字节，两个 rank 合计最高 41 GiB。可用 `CLEF_PRIVATE_K32_RANK_BYTES` 指定更低上限。动态预算取宿主可用内存、有效 cgroup v1/v2 限制和请求额度的最小值，至有效 cgroup 阈值保留 10 GiB，并按两个参与 rank 平分可新增余额。缓存充满时按字节逐出，内存压力触发限制收缩和普通计算回退。

## 执行与边界

私有 worker 启动检查固定 vLLM、Ascend、GDN 和联合头源码。模型身份需为已验 Ascend Clef 及 Qwen3.5 语言栈，64 层完整分区，PP1/TP2、禁用真实 prefix/chunked prefill、speculation、跨 rank KV/跨层 KV 共享。只有一个 fresh 文本请求的完整 prefill 会走候选路径。图文、合批、continuation 或任何守卫失败均沿原 forward。

缓存内容为 layer 31 返回的完整提示序列 `hidden_states`、`residual` 两张 BF16 张量。键覆盖完整 token IDs、语言栈的实际位置和布局，以及当前启动世代和执行域。每 rank 持有自己的一份运行态缓存，模型重载使旧缓存失效。真实请求 miss 执行 64 层，由同步 D2H 惰性填充。两 rank 对键及实际 H2D 就绪达成一致后，hit 执行 layer 32–63、原 final norm、全部 prompt 的 ALL pooling 和当前请求的 aa3e 联合头。缓存持有的对象只参与只读复制，后半段在请求私有张量中运行。

固定调度器保留 512-token 管理页和 128-token kernel 页的原映射。准入保留 `PrefillCacheHit` 后端标签，核实际组、BF16 NBHD K/V 几何、无共享绑定和 fresh host metadata。在验收诊断窗，两 rank 实际写槽与已展开读表逐 token 验证，三条冻结任务的原始 64 层、K32 fill 和 32 层 hit 最终 hidden 与完整答案逐位相同。

## 测量口径

沿用冻结的 OpenDecisions 2,352 条文本请求、C1、`warmup=0`、一次尝试与 300 秒单请求超时。启动后只做 GET 就绪检查。同进程首轮从空缓存自然填充，随后两轮自然复用。所有真实缓存拷贝、键计算、TP 共识均计入客户端耗时。每轮独立计算 linear P95，并保留首轮及自然复用轮次。三轮主统计取三次 P95 的中位数。正式轮次关闭 tap、逐槽设备取证、张量比较与资源采样，缓存端点记账在空闲边界。

`OD_FROZEN_RUNNER` 指向索引中的原 runner，`OD_FROZEN_INPUT` 指向对应的 `all2352.json`。在同一服务就绪后依次运行。

```bash
for round in 1 2 3; do
    out="$NEW/OD-S1-r$round"
    mkdir "$out"
    "${PYTHON_BIN:-python3}" -IB "$OD_FROZEN_RUNNER" \
        --sample-file "$OD_FROZEN_INPUT" --url http://127.0.0.1:18143/v1/systemone \
        --concurrency 1 --warmup 0 --timeout 300 \
        --report "$out/report.json" --journal "$out/requests.jsonl"
done
```

CPU 回归可从该目录运行 `python3 -B -m unittest`，包括惰性填充、失效、逐出、压力、父层额度和两 rank 一致回退。

## K32 新机正式结果

| 轮次 | 起始状态 | HTTP 成功 | P95 ms | decisions/s | 每 rank 命中 |
| --- | --- | ---: | ---: | ---: | ---: |
| 1 | 空缓存，自然填充 | 2,352 | 169.061668 | 7.852983 | 61 |
| 2 | 同进程自然复用 | 2,352 | 86.236415 | 14.910476 | 2,352 |
| 3 | 同进程自然复用 | 2,352 | 85.985907 | 15.124823 | 2,352 |

三轮 P95 中位 86.236415 ms，吞吐中位 14.910476 decisions/s。已采用的默认服务正式对照为 156.181845 ms、8.948975 decisions/s。默认服务原首轮 156.967955 ms，K32 冷首轮较其多 12.094 ms。性能差异随请求内容复用和状态变化。

三轮作者评分均为 Calibration 82.411344、Intelligence 66.084076、Capability 74.247710，相对适配固定基准为 −0.382771、+0.027233、−0.177769 pp。每轮 1904/2352 正确，Score MAE 0.402612。三轮完整 typed answers 均与原融合 ON 一致。

每 rank 首轮发布 2,291 个实际 token 和 position 键，保留 20,960,185,344 字节。两个 rank 的逻辑计费合计 39.041 GiB。三轮逐出和真实调度 bypass 均为零。共享 cgroup 限制 128 GiB，生命周期高水约 117.790 GiB，含进程与页缓存及更早阶段，高水期间 failcnt 维持 0。各轮端点独立记录 file/cache、RSS 和 SHM。 [测量索引](../../Clef-27B/logs/dataset-profiles-20261123/manifest.json)

## 可选 K56 配置

```bash
bash scripts/start_dataset_profile.sh od-s1-k56 \
    --model-dir "$MODEL" --output-dir "$NEW"
```

K56 复用同一 full-prompt 协议和内存预算，边界移动到 layer 55 输出。实际 BF16 hidden/residual 的形状和计费同 K32。命中依次执行 56–63、final norm、ALL pooling 和 aa3e 联合头。边界参与启动世代和完整内容键。`od-s1-k32` 可显式选择。回到默认服务时使用原入口。

```bash
bash scripts/start_dataset_profile.sh od-s1 \
    --model-dir "$MODEL" --output-dir "$RESTORE"
```

| 轮次 | 起始状态 | HTTP 成功 | P95 ms | decisions/s | 每 rank 命中 |
| --- | --- | ---: | ---: | ---: | ---: |
| 1 | 空缓存，自然填充 | 2,352 | 169.297990 | 8.244777 | 61 |
| 2 | 同进程自然复用 | 2,352 | 32.574693 | 35.697736 | 2,352 |
| 3 | 同进程自然复用 | 2,352 | 32.161995 | 36.196760 | 2,352 |

K56 三轮 P95 中位 **32.574693 ms**、吞吐中位 **35.697736 decisions/s**。同机 K32 的三轮 P95 中位为 86.236415 ms，两条记录来自独立窗口。冷首轮单列，前后原服务无投票锚点见正式证据。三题诊断窗关闭审计后的 K56 完整调用中位为 31.923150 ms，探针 stock A/A 含 TP 投票。已测命中各 rank 实际调用后 8 层与一次 final norm，完整 typed answers 与原 ON 逐位一致。

正式三轮共 7,056 条，原始答案与融合 ON 相同。作者分数仍为 Calibration 82.411344、Intelligence 66.084076、Capability 74.247710，逐轮通过 2pp；每轮 1904/2352、MAE 0.402612。两 rank 保留共 39.041 GiB 逻辑字节，均零逐出。内存端点和生命周期高水见正式索引。[K56 测量索引](../../Clef-27B/logs/dataset-profiles-20261123/k56-s1-evidence.json)

## K56 异步冷填探针

隔离分支的私有开关为 `CLEF_PRIVATE_K56_ASYNC_FILL=1`，配合 `od-s1-k56` 使用。每 rank 最多一项在途和一个 64 MiB pinned 槽，另从上限扣除 40 MiB 设备快照额度。支持最多 2,048 行的边界张量，超过时按完整前向返回。两次 D2H 等待独立 BF16 快照就绪。尾栈和 norm 提交后，主线程等待拷贝完成并生成独立 pageable 缓存值。两 rank 均确认容量和条目后才开放可见性。

双 rank 小窗完成 45 个请求。三形状同步和异步边界、末层 hidden 与完整 typed answers 逐位一致。单 rank 否决后的重试可以重新填充并命中。两代表形状的 PCIe DMA 与尾栈实际任务重叠 0.380–0.681 ms。已建槽、每阶段冷键的五形状无审计中位，稳定同步 161.636–161.985 ms，异步 159.642 ms。端点 cgroup usage 为 74.304–75.480 GiB，failcnt 为零。正式冷首轮目标仍待验证。

独立轻诊断 `od-s1-k56-async-cold-drift-v1-all2352-pinned64-stage3-diag` 只执行一轮。2,352 条请求都成功，完整答案逐题等于原融合 ON，作者三项均过 2pp。含宿主采样的 P95 为 165.862577 ms。896–1050 token 的 109 组配对里，前半程 pageable CPU copy 中位约 1.3 ms，每 rank 第三段为 15.172 / 11.200 ms，末段为 23.382 / 12.773 ms。第二次 TP 共识等待随准备到达差扩大。物理页分配、页回收与线程调度各自的影响仍待验证。

共享 cgroup 128 GiB，窗口高水从累计 117.827316 GiB 增至 118.013378 GiB，最小余量 9.986622 GiB，低于 10 GiB 采用门。两个 rank 分别逐出 21 和 26 项。诊断 tar SHA-256 为 `b6b4a7ddd72c78f57db563a54812d01e7c5ef48f80acb26f4fdb323e6bf47446`。诊断成绩与正式三轮分开。

固定 pageable 小池在每 rank 256 MiB 下跑完 46 条功能及计时请求。三形状填充与命中逐位，单 rank 否决后可重试发布，压力注入后两 rank 共同回退并完成整池 unmap。每 rank 启动触页 8.83 / 9.04 ms，fork 至 GET-ready 78.36 s。五形状无审计冷键夹测，普通异步 A1 / A2 中位 159.528 / 159.407 ms，池方案 159.703 ms。普通异步逻辑上限 48 MiB，池物理容量 256 MiB，结果只比较整体路径。热命中通过逐位检查，独立无审计热态 P95 待测。

小池窗口最高端点 current 75.537235 GiB，failcnt 为零。118.013378 GiB 仍是进入窗口前的累计高水。容器 memory cgroup 只读，现有 exporter 提供宿主和设备指标，本窗口缺整 scope 峰。小池仍作实验，暂不进入正式三轮。原包 SHA-256 为 `4b596c4a5ceffe32f45b24013e0100662363aa10392fca71ce4989f53505a96e`。
