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

## K64 末态缓存候选

`od-s1-k64` 显式选择 post-norm full-T 隐藏态。在锁定的 TP2、BF16、eager、B1 文本 pooling 路径中，冷请求调用原主干并只留存末段 final norm 的一张隐藏张量。命中后，两 rank 在同步恢复各自私有张量并达成就绪共识后返回隐藏态，原 runner 仍执行 ALL pooling 和完整联合决策头。保留同一组 fresh prefill、位置、KV/GDN、模型和来源守卫，并增加有效 embedding 生产缓冲区、pooling runner 及 post-forward 状态消费者检查。三题正确性窗口共 49 条 HTTP 200。982、977、990 token 三种输入的 stock、fill 和 hit 全 T 末态均为独立 ND BF16，分别占每 rank 10,055,680、10,004,480、10,137,600 字节。主干观测为 stock/fill 64 层和一次 final norm，hit 零层、零主干 final norm。两 rank 对应的 ALL pooling 输入与联合头调用均保留，完整概率与固定 ON 逐题相同。单 rank 恢复否决双方回原主干，后续另外一题复用每 rank 977 个槽位后仍通过逐位验证。

关闭审计的三题 A0/A1/B/A2 完整调用中位分别为 157.063、157.095、16.669、156.909 ms。原包 SHA-256 为 `6178e0b9b7a13660788cc1a7fcbd6cc92ac070eadc739ff5fba29c758e1166b7`。吞吐与全量作者评分见后续独立窗口，现有 K56 固定序和乱序结果沿用各自的测量索引。
