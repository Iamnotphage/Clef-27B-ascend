# Clef 数据集入口及可选 JPEG 解码缓存

`scripts/start_dataset_profile.sh` 提供独立数据集入口，均使用正式 BF16、TP2、eager 和 aa3e。

| Profile | 服务预算 | 客户端计量 |
| --- | --- | --- |
| `od-s1` | B1、4096 上下文、4096 batched tokens、util .90 | C1 |
| `od-s2` | B16、4096 上下文、16384 batched tokens、util .90 | 当前正式 C64 |
| `erqa-c1` | B1、16384 上下文和 batched tokens、util .85、encoder 262144 | C1，冷遍和自然第二遍分别记录 |
| `erqa-s2` | B8，同 16K 和 .85 预算、encoder 262144 | C8 或 C16 |

```bash
bash scripts/start_dataset_profile.sh od-s1 --model-dir "$MODEL" --output-dir "$NEW"
bash scripts/start_dataset_profile.sh od-s2 --model-dir "$MODEL" --output-dir "$NEW"
CLEF_PRIVATE_JPEG_CACHE_GIB=0 bash scripts/start_dataset_profile.sh erqa-c1 \
    --model-dir "$MODEL" --output-dir "$NEW" -- --mm-processor-cache-gb 4
CLEF_PRIVATE_JPEG_CACHE_GIB=2 bash scripts/start_dataset_profile.sh erqa-c1 \
    --model-dir "$MODEL" --output-dir "$NEW" -- --mm-processor-cache-gb 4
```

## 生效范围

JPEG 缓存默认关闭。显式设置 `CLEF_PRIVATE_JPEG_CACHE_GIB=2`，为相同软件栈的单 API 进程请求最高 2 GiB 解码缓存。根据宿主可用内存以及有效 cgroup v1/v2 当前用量、父层额度和压力事件缩减目标，给加载及并发保留余量。达到限制会按字节逐出，压力升高时在后续请求中清空并暂缓填充。图文栈的处理器 LRU 为 4 GiB，NPU encoder 输出缓存为 262144 tokens。

仅在原 data URL 校验和有效参数通过后处理 JPEG。内容、Pillow 版本、解码代码、颜色模式、背景和像素限制参与私有键，EXIF 方向由原图 bytes 与同版本归一化确定。缓存保留不可变 RGB bytes；调用方得到独立 PIL 对象和自己的 `MediaWithBytes`，并按原始大小再次接受本请求的解码限额检查。解码失败、超大对象和不支持的元数据走原路径。回收对已有调用使用引用保活，重启进程释放整个缓存。vLLM 图文处理器继续用原输入 bytes、`io_config` 和 EXIF ImageID 路径计算原键。缓存只随真实请求填充。

该可选模块锁定已验证的 `ImageMediaIO` 和多模态 hasher 源码。兼容机需加载相同运行栈；独立入口可不带实验变量运行。

## 新机验证

文本轻筛的四点各 640 请求，B16/16384 的暖态 C64、C96、C128 为 17.392、17.396、17.390 decisions/s，P95 分别为 4.994、7.151、8.925 秒。四点均为 41 个引擎 iteration。正式 2352 题结果继续使用 17.689352 decisions/s @ C64。

图文现有 4 GiB 逻辑 LRU 在实际 uint8 路径上每级保留 878,192,975 字节和 530 个对象。冷遍 100/630 命中，暖遍 630/630，两遍零逐出和零可复用容量 miss。归档 FP32 的 5.208 GiB 估算适用于不同载荷。容量门未触发 8 GiB 档。

独立新进程依次为 A0、A1、B、A2。每进程先跑 400 道冷遍，再跑 400 道自然第二遍，GET 就绪、不额外预热。B 打开 JPEG 缓存；各阶段使用相同冻结数据、4 GiB processor LRU 和 262144 encoder 条件，服务未开 profiler。

| ERQA C1 指标 | A0 | A1 | B | A2 |
| --- | ---: | ---: | ---: | ---: |
| 冷遍 P95（ms） | 298.242 | 295.895 | 297.573 | 297.038 |
| 冷遍 wall（s） | 85.693 | 85.498 | 86.266 | 85.949 |
| 暖遍 P95（ms） | 250.706 | 250.624 | 246.054 | 250.784 |
| 暖遍 wall（s） | 67.903 | 68.693 | 67.186 | 68.932 |

暖遍相对 A 中位快 4.652 ms（1.86%），超过 A/A 的 0.160 ms 范围，整轮净省 1.507 s。冷遍 P95 在 A/A 内，整轮比 A 中位多 0.574 s。八遍共 3,200/3,200，零失败、每遍均为 216/400，逐题原始概率和选择一致。B 两遍合计 JPEG 725/1,260 命中，最高计费权重 1,173,257,380 字节。B 暖遍端点 cgroup 76.017 GiB、同期 cgroup 累计高水 76.037 GiB、四进程合计 PSS 13.726 GiB。A 暖遍端点 cgroup 范围 74.881–75.497 GiB，合计 PSS 范围 12.971–13.099 GiB。累计高水跨阶段保留，PSS 为端点读数。

这组成绩支持可选暖态缓存。JPEG-on 的 **246.053547 ms** 对应 B 臂自然第二遍。选入性能汇总的 JPEG-off C1 来自 aa3e 融合正式验收，冷首遍 **300.537708 ms**、自然第二遍 **252.024530 ms**，两遍各 216/400。[测量索引](../../Clef-27B/logs/dataset-profiles-20261123/manifest.json)

验收冻结输入 SHA-256 为 `35bd36560dcc228c02a8e336cc57b7b8bb0f1c6752521dc63fc482f48a4168af`。私有八遍原数与执行索引的压缩归档 SHA-256 为 `fdee1841796d2d220e3ae39efed03b91fb84e0c185a91c4ae81ad36f76a1e318`。



## 图文 S2 的固定点

使用 `erqa-s2` 启动 B8、16K、util .85、encoder 262144、processor LRU 4 GiB 的图文服务。JPEG 缓存关闭。C 从客户端传入。B16 复现可以在同一可选入口附 `--max-num-seqs 16`，batched tokens 仍为 16384。每个 B/C 点用独立进程，GET 就绪后首遍固定 400 题，紧接同进程自然第二遍。首遍与第二遍独立报数；C1 S1 沿用原独立入口及冻结成绩。

| 实测图文 S2 | 冷 decisions/s | 冷 P95 ms | 冷准确数 | 自然第二遍 decisions/s | 自然第二遍 P95 ms | 自然第二遍准确数 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| B8 C8 | 6.771695 | 2025.446 | 218/400 | 8.024087 | 1694.174 | 218/400 |
| B8 C16 | 7.166461 | 3606.181 | 217/400 | 8.378042 | 3074.490 | 217/400 |
| B16 C16 | 7.138460 | 3436.466 | 218/400 | 8.416542 | 2899.933 | 216/400 |

冷遍最高实测点为 B8 C16，纳入 [`results.csv`](../../Clef-27B/results.csv)。暖遍最高实测点为 B16 C16。B16 暖遍相对 B8 C16 多 0.46%，冷遍少 0.39%，同状态每点一遍。B8 C8 两态 P95 均最低。独立复核核实六遍共 2,400/2,400，零失败、每题一次 HTTP 200，均通过相对 219/400 的绝对 2pp。

实际最大 HTTP 在途数各自达到 C。整窗 checkpoint 的 `vllm:iteration_tokens_total_count` 增量显示 C8 每遍 101 个迭代，C16 两档每遍 51 个。各遍 `_sum` 为 372,671 tokens，与原 HTTP 输入 token 总和吻合。每迭代请求数 3.96 和 7.84 由聚合值计算。逐 step batch、预算 guard 和融合路由未采。服务计时关闭 profiler 和私有观测 hook，独立进程读取原生指标和 OS/HBM。

下表 HBM 峰值来自 checkpoint 合并边界读数与轮询后的取峰。归档的 OS 序列在整窗结束前停止，冷态三点的 HBM 峰值均低于 checkpoint，C16 序列覆盖 50 个迭代。暖态三点 HBM 峰值及六遍主机、等待队列峰值均与 checkpoint 一致。现有归档保留整窗差值和取峰，终点 HBM 读数与精确 Prometheus 两端绝对值未单独留存。

| 配置 | 冷 host GiB | 冷 HBM MiB rank 0/1 | 暖 host GiB | 暖 HBM MiB rank 0/1 |
| --- | ---: | ---: | ---: | ---: |
| B8 C8 | 75.719 | 59695 / 59351 | 75.973 | 60136 / 59787 |
| B8 C16 | 75.966 | 60180 / 59831 | 76.233 | 61220 / 60871 |
| B16 C16 | 76.320 | 59910 / 59564 | 76.527 | 61371 / 61024 |

复用归档评测器 `cc7f32842a78786a8f0e5da373f66597752ea9ab` 的 `erqa_baseline.py`、`erqa_protocol.py` 和原评分器。并发层锁定单题 runner SHA-256 `8d0632825ca5bf58e33427510f5d297b3080b00cce84959f736c72e974b2755a`，调用原 `run_one` 和原 wire 逻辑。墙钟从并发提交开始到最后一条 journal 落盘，计入读图、请求准备、HTTP 和原件写入。每题保留 `request.meta.json`、`attempts.jsonl`、原始 HTTP 响应和请求结构及 wire SHA。按 `record_index` 输出预测，journal 另保留完成顺序及单调时钟 HTTP 起止。

`ERQA_DATA` 指向冻结的 400 题素材。准备固定版评测器后，S1 在独立 `erqa-c1` 进程运行单题入口，S2 使用以下并发入口。两者均先 GET 就绪，再以新输出目录依次运行冷首遍和同进程自然第二遍。

```bash
ERQA_RUNNER_REF=cc7f32842a78786a8f0e5da373f66597752ea9ab
ERQA_RUNNER_DIR="$(mktemp -d)"
for file in erqa_baseline.py erqa_protocol.py erqa_score.py; do
    git show "$ERQA_RUNNER_REF:scripts/official_eval/$file" > "$ERQA_RUNNER_DIR/$file"
done
python -B "$ERQA_RUNNER_DIR/erqa_baseline.py" --data-dir "$ERQA_DATA" \
    --output-dir "$C1_RUN" --api-url "$BASE/v1" --model clef --scope full \
    --retries 1 --retry-interval 5 --timeout 600
python -B "$ERQA_RUNNER_DIR/erqa_score.py" --data-dir "$ERQA_DATA" \
    --run-dir "$C1_RUN" --verify-requests --output "$C1_SCORE"
```

```bash
CLEF_PRIVATE_JPEG_CACHE_GIB=0 bash scripts/start_dataset_profile.sh erqa-s2 \
    --model-dir "$MODEL" --output-dir "$SERVICE" -- --mm-processor-cache-gb 4
PYTHONPATH="$ERQA_RUNNER_DIR${PYTHONPATH:+:$PYTHONPATH}" \
    python -B runtime/erqa_s2_concurrent.py --data-dir "$ERQA_DATA" \
    --output-dir "$RUN" --api-url "$BASE/v1" --concurrency 16 --timeout 600
python -B "$ERQA_RUNNER_DIR/erqa_score.py" --data-dir "$ERQA_DATA" \
    --run-dir "$RUN" --verify-requests --output "$SCORE"
```

自然第二遍在同进程同 C 下重用该命令并换全新的输出目录。C8 与 C16 各自用新进程进行同状态比较。

两份图文 S2 原数包分别归档 B8 与 B16。压缩 SHA-256 为 `0127d45a4f7bbc7c11b59e8904514364d08f2f793488f73138fb0832afd18f28` 和 `6984f962a1daa75cf500f703e654eb8723df3d676e279330a52edeeb290bb066`。

## CPU 合同复现

```bash
PYTHONPATH="$PWD/runtime/jpeg_decode_cache${PYTHONPATH:+:$PYTHONPATH}" \
  python -B runtime/jpeg_decode_cache/cpu_contract.py "$FROZEN_WORKSPACE"
```

在已激活的固定 vLLM/CANN 环境运行。`FROZEN_WORKSPACE` 下的 `erqa-fixed/samples.jsonl` 和图像来自验收冻结数据。检查同时覆盖真实及合成 JPEG、EXIF 与原哈希、坏图原错误、重复计费、独立副本、并发、超大旁路和模拟父层内存压力。
