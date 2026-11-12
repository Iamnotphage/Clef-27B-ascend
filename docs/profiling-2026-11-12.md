# Clef-27B 端到端 profiling：采集准备与执行记录

## 本轮状态

2026-11-12 已完成仓库核对、测量工具扩展、采集控制、可恢复观测补丁生成器、双 rank 解析和本地验收。起始 HEAD 为 `869a44292b72325dcb95f8af449cf252d35cd943`，起始工作区干净。远端实验等待当前有效的跳板凭据；旧复用 socket 连接被拒绝，目标机直连 22 端口超时。[本轮连接检查](evidence/profiling-preflight-2026-11-12.json)

本轮尚未在 A3 上执行性能请求、创建远端 run 目录、采集 NPU trace 或切换服务。原服务未被本轮修改；当前健康状态待连接后现场验收。本页提供已验证的本地工具与后续执行入口，硬件实测结论随远端证据补充。

| 交付项 | 当前证据与状态 |
| --- | --- |
| 固定源码、补丁和历史条件核对 | 已读 AGENTS、baseline、installation、constraints、入口、测量/协议/精度脚本及判据；起始 HEAD 与用户给定基线相同 |
| 可复用测量、采集、解析工具 | 已完成，见下表；本地 HTTP、故障注入与 synthetic trace 检查通过 |
| 现场软件、设备映射和原服务快照 | 待 SSH 恢复后执行 |
| TP1/TP2 各三轮稳定性能 | 待实测；当前无本轮 requests/s、延迟或轮间波动结果 |
| TP2 双 rank 原始 trace 与导出 | 待实测；本地 synthetic 数据仅用于验证解析逻辑 |
| HTTP/CPU、调度、主干、联合头、HCCL 分解 | 待实际标记、请求窗口与设备依赖关联 |
| 长度、字段/候选、并发/batching、多模态矩阵 | 待实测与实际 HBM 检查 |
| 权重、缓存池、激活与 workspace 分解 | 待现场日志、分配器和内存短窗口采集 |
| 本轮精度回归与服务恢复 | 待现场实验；继续采用既定判据 |
| 瓶颈确认与优化收益排序 | 待关键路径证据；当前无已证实的本轮性能瓶颈 |

## 工具与统计范围

| 工具 | 功能与边界 |
| --- | --- |
| [benchmark_systemone.py](../scripts/benchmark_systemone.py) | 保留原 mixed 请求字节与 `elapsed_ms` 定义；增加 `--sample-file`、最短窗口、逐请求 token/字段/候选/校验、客户端队列、HTTP/JSON/协议计时、请求 ID、p99 样本量及离散程度 |
| [snapshot_systemone_profile.py](../scripts/snapshot_systemone_profile.py) | 记录 Git HEAD/diff、patch/manifest 哈希、freeze/check、CANN/驱动、NPU、配额、磁盘、进程和白名单性能环境；记录每项探测退出码及实际 profiler 源码 |
| [profile_systemone.py](../scripts/profile_systemone.py) | 采集外同 shape 预热，根路径 start/stop，finally 停采，逐 rank 普通独立 Python 解析，再严格验收原始数据与设备时间线 |
| [profile_instrumentation.py](../scripts/profile_instrumentation.py) | 校验固定 HEAD 和 AST 锚点，生成可恢复的源码观测补丁与备份/SHA；转发 record_shapes/with_flops，按需启用 CPU marker 和请求计时；保留已有应用补丁 |
| [analyze_systemone_profile.py](../scripts/analyze_systemone_profile.py) | 按 rank 输出 Top20、通信、kernel 区间 union/计算通信重叠、CPU scope、SHA 证据索引；支持每窗口指定原始目录与前端 JSONL |
| [summarize_systemone_benchmarks.py](../scripts/summarize_systemone_benchmarks.py) | 汇总逐轮 CSV/JSON、请求分布及轮间波动，检查固定 307-token 基准的采样协议；源码/配置一致性继续引用独立现场 manifest |

`elapsed_ms` 包含 HTTP、UTF-8/JSON 解码及协议校验；客户端线程池排队单独记录。HTTP 时间覆盖网络、服务端排队、处理与序列化。客户端观测无法独立识别服务端队列，相关值保留为 `null`，通过请求 ID 与服务端日志补充。p50/p95/p99 使用 `(n−1)×q` 线性插值，报告同时给出样本量。

正式性能取自关闭采集、关闭 CPU 日志与 marker 的窗口。CPU scope 为包含内部调用的 host wall/enqueue 时间，异步范围包含 await；嵌套 scope 和并发协程具有重叠。kernel 累计时间、区间 union 和请求关键路径分别记录；解析器保留 `critical_path_contribution_us=null`，等待请求依赖证据。两个 rank 的时钟校齐由实际 profiler 元数据确认。

## 已核对的源码行为

本地固定源码为 vLLM `24420b7a341af37a9bc3186e6fc6062512e0aa58`、Ascend `89eb8a5e7a76d085b453a1aeadb6883e12d5ea2d`，后续现场再次核对 HEAD/diff 与加载路径。

- `vllm/config/profiler.py` 声明 record_shapes、with_flops、内存、stack 和 schedule 选项。
- 固定 Ascend wrapper 将 memory 转为 `profile_memory`，stack 选项转为 `with_modules`，底层 `with_stack=False`；原实现未转发 record_shapes、with_flops 和 torch schedule。
- `JointSchemaPooler.forward` 按 prompt 取得候选词向量并调用联合头；`_embed_lexical` 通过标准词表 TP 路径归约候选向量。字段循环分布于 schema 编码和响应构造；实际调用次数由 trace 统计。
- `/start_profile` 与 `/stop_profile` 在根路径；注册依赖服务启动时的 profiler config。
- 观测补丁改变编码相关源码 SHA，现有严格源码审计需要在恢复补丁后执行。观测窗口的数值比较可引用本轮开始前已保存的精确编码证据，结束时恢复源码并重新执行编码审计。

这些结论用于确定采集范围。实际 kernel、模块覆盖、耗时和收益上限均待 A3 trace 验收。

## 本地验证记录

[验证记录](evidence/profiling-local-validation-2026-11-12.json)保存本地 Python/平台、命令、完整输出和脚本 SHA-256。`python3 -m unittest discover -s scripts -p 'test_*.py' -v` 的 27 项检查通过，覆盖 HTTP 请求、固定样本哈希、最短测量窗口、吞吐汇总、异常停采、凭据过滤、观测恢复与并发修改保护、双 rank/空原始数据/缺失时间线验收及区间重叠统计。

针对本地两份固定源码材料完成观测生成器 dry-run，涉及十个文件，`applied=false`；生成 Python 语法检查通过，源码材料保持原始状态。`git diff --check` 和本页链接检查通过。本地验证证据明确记录 `hardware_executed=false`。

## 远端复现入口

以下命令为待执行模板。每轮使用新目录，原始大文件保留远端，仓库保存摘要、原始数据索引和必要补丁。执行前从快照确认原服务 PID、进程树和完整命令；单服务顺序切换，核对两卡占用。

```bash
cd /workspace/yangchen/Clef-27B-ascend
source scripts/activate_env.sh
run_id="$(TZ=Asia/Shanghai date +%Y%m%d-%H%M%S)-systemone"
run="/workspace/yangchen/logs/profiling/$run_id"
mkdir -p "$run"
python scripts/snapshot_systemone_profile.py --output-dir "$run/before"
```

先在原 TP2/4096 服务保存五组协议输出与图片/视频输出，并沿用 baseline 中 CPU 参考、encoding 与 criteria 执行精度比较。切换配置时记录原始启动命令和环境、服务日志、实际 PID、健康状态；等待原 worker 退出后再启动下一配置。

TP1/1024 与 TP2/1024 使用相同四份应用补丁、0.90 内存预算、媒体配置和下列参数，分别将 `--tensor-parallel-size` 设为 1、2：

```bash
bash Clef-27B/start.sh --model-dir "$MODEL_DIR" \
  --tensor-parallel-size 2 --max-model-len 1024 --max-num-seqs 1 \
  --gpu-memory-utilization 0.90 --host 127.0.0.1 --port 8000 -- \
  --distributed-executor-backend mp
```

在已通过健康检查的**未开启采集**服务上，分别执行三轮：

```bash
mkdir -p "$run/tp2-1024"
for round in 1 2 3; do
  python scripts/benchmark_systemone.py \
    --base-url http://127.0.0.1:8000 --warmup 10 --requests 300 \
    --concurrency 1 --min-duration-seconds 30 --sample-hbm \
    --report "$run/tp2-1024/round-$round.json"
done
```

TP1 使用独立 `tp1-1024` 输出目录。每轮核对 307 tokens、三个字段、2/2/3 候选、样本数、成功率和实际窗口。三轮汇总示例：

```bash
python scripts/summarize_systemone_benchmarks.py \
  --report "tp2-1024=$run/tp2-1024/round-1.json" \
  --report "tp2-1024=$run/tp2-1024/round-2.json" \
  --report "tp2-1024=$run/tp2-1024/round-3.json" \
  --output-dir "$run/matrix" --require-stable-baseline
```

首次 timeline 从 TP2/1024 开始。在该服务启动参数尾部增加实际绝对路径的 profiler config：

```bash
--profiler-config "{\"profiler\":\"torch\",\"torch_profiler_dir\":\"$run/tp2-1024-traces\",\"torch_profiler_with_stack\":false,\"torch_profiler_with_memory\":false,\"ignore_frontend\":true}"
```

健康检查通过后运行短窗口。控制器在采集外发送十次预热，采集内发送五次请求，stop 超时 600 秒，随后逐 rank 独立解析及严格检查：

```bash
python scripts/profile_systemone.py \
  --base-url http://127.0.0.1:8000 \
  --output-dir "$run/tp2-1024-window-1" \
  --trace-root "$run/tp2-1024-traces" --expected-ranks 2 \
  --warmup 10 --requests 5 --stop-timeout 600
```

对应 TP1 使用独立 trace root、窗口目录和 `--expected-ranks 1`。当前 TP2/4096 代表请求使用独立窗口；自定义完整 POST JSON 通过 `--sample-file` 传入。多样本性能文件格式为 `{"samples":[{"name":"case-id","request":{...},"metadata":{...}}]}`；每种新 shape 单独预热。

原生窗口验收后按需补 shape/内存/CPU 观测。先停止实验服务，再生成并应用观测补丁：

```bash
python scripts/profile_instrumentation.py \
  --vllm-root /workspace/yangchen/work/src/vllm \
  --ascend-root /workspace/yangchen/work/src/vllm-ascend \
  --output-dir "$run/instrumentation" --apply
export CLEF_PROFILE_MARKERS=1
export CLEF_PROFILE_OBSERVE_DIR="$run/cpu-timings"
```

形状短窗口另外设置 `torch_profiler_record_shapes=true`；内存短窗口另外设置 `torch_profiler_with_memory=true`，分别记录配置、开销和实际导出内容。marker 的目标函数与实际 Ascend override 覆盖通过 trace 检查。调度批量、队列等待、主干通信和候选 embedding 通信进一步通过请求/模块标记与 kernel 依赖对应。需要硬件指标时，针对已确认热点按 ops-profiling 的 msprof 流程采集，保留所有相关 worker 和设备。

```bash
python scripts/analyze_systemone_profile.py "$run/tp2-1024-traces" \
  --expected-ranks 2 --observations-root "$run/cpu-timings" \
  --output-dir "$run/tp2-1024-analysis" --strict
```

同一 trace root 包含多个窗口时，为解析器重复传入该窗口 manifest 中的 `--trace-dir`，每 rank 一项。

## 后续采集与验收边界

1. 先完成固定请求三轮对照与 TP2/TP1 timeline；核实双 rank 原始目录、kernel、事件范围、HTTP 和校验结果。
2. 在 4096 上限下选择约 256/1024/2048/接近 4096 的真实输入，保存完整 token IDs、schema/候选 token 数、截断与来源哈希；TP1 范围按实际 HBM 判定。
3. 固定 state 改字段数、候选数；逐次记录唯一变量。max-num-seqs=1 时比较并发 1/2/4，再比较 max-num-seqs=2/4 的实际 batch、token 上限和 HBM。
4. 复现两图两视频，记录分辨率、帧、processor、解码、视觉编码和 token。设备级 HBM、权重、缓存池、allocated/reserved、激活峰值和 workspace 分别保留统计范围。
5. 结束观测时停止实验服务，移除观测环境变量，执行 `profile_instrumentation.py --restore "$run/instrumentation/instrumentation-manifest.json"`。按现场快照恢复 TP2/4096 多模态命令与环境，验收健康、五组协议、token/span、参考精度和 MM 4/4；保存恢复后的快照。

精度沿用[既定判据](evidence/precision-criteria-2026-11-12.json)：候选概率最大绝对差 ≤0.03、score 最大绝对差 ≤0.05，离散结果、token IDs、span 和候选映射对应。观测前后 API 输出均需保存。

## 瓶颈与优化建议的证据门槛

当前本轮已确认的限制为远端连接不可用。性能瓶颈与收益排序等待实际采集，后续每项建议填写 trace/CSV 行或时间区间、受影响关键路径、可优化比例、收益上限、实现成本、精度验证和最小实验。

优先采集候选 TP 归约、联合头同步/搬运、小 kernel 下发间隙、主干 collectives 与 CPU/schema 路径。它们目前属于待验证的测量方向。历史 TP1/TP2 的 30 次短窗口及缓存池占用继续引用[baseline](baseline-2026-11-12.md)，本轮稳定性和优化收益由新增证据建立。
