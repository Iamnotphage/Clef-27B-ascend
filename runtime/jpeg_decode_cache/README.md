# Clef 数据集入口及可选 JPEG 解码缓存

`scripts/start_dataset_profile.sh` 提供三个独立入口。三个入口均使用正式 BF16、TP2、eager 和 aa3e。

| Profile | 服务预算 | 客户端计量 |
| --- | --- | --- |
| `od-s1` | B1、4096 上下文、4096 batched tokens、util .90 | C1 |
| `od-s2` | B16、4096 上下文、16384 batched tokens、util .90 | 当前正式 C64 |
| `erqa-c1` | B1、16384 上下文和 batched tokens、util .85、encoder 262144 | C1，冷遍和自然第二遍分别记录 |

```bash
bash scripts/start_dataset_profile.sh od-s1 --model-dir "$MODEL" --output-dir "$NEW"
bash scripts/start_dataset_profile.sh od-s2 --model-dir "$MODEL" --output-dir "$NEW"
bash scripts/start_dataset_profile.sh erqa-c1 --model-dir "$MODEL" --output-dir "$NEW"
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

这组成绩验证了可选暖态缓存。正式 `results.csv` 和 `metrics.csv` 继续指向既有验收，图文 S2 仍需独立正式实测。

验收冻结输入 SHA-256 为 `35bd36560dcc228c02a8e336cc57b7b8bb0f1c6752521dc63fc482f48a4168af`。私有八遍原数与执行索引的压缩归档 SHA-256 为 `fdee1841796d2d220e3ae39efed03b91fb84e0c185a91c4ae81ad36f76a1e318`。

## CPU 合同复现

```bash
PYTHONPATH="$PWD/runtime/jpeg_decode_cache${PYTHONPATH:+:$PYTHONPATH}" \
  python -B runtime/jpeg_decode_cache/cpu_contract.py "$FROZEN_WORKSPACE"
```

在已激活的固定 vLLM/CANN 环境运行。`FROZEN_WORKSPACE` 下的 `erqa-fixed/samples.jsonl` 和图像来自验收冻结数据。检查同时覆盖真实及合成 JPEG、EXIF 与原哈希、坏图原错误、重复计费、独立副本、并发、超大旁路和模拟父层内存压力。
