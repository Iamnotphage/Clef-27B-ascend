# 同请求 head packed KV（2026-11-22，局部负结果）

M967真实冻结输入下，两rank的40项数值检查全部逐位一致、有限且重复稳定。
完整decoder0的改善落在A/A内；本实现停止于单处门，生产保持aa3e/S1。
BSH strided view能进入原fusion attention，实际触发两次输入Slice搬运。
投影合并减少一次MatMul与Cast，整decoder计算任务数仍为24，task时间没有可辨改善。
六处扩展、完整head覆盖、S1/OD/ERQA未进入；本结果只覆盖本机当前库路径与本实现。

基线 `b9580788f0729c74d38d0f8119f5a4011aa132b4`，分支
`perf/head-packed-kv-20261122`，执行代码 `52a8161`。
aa3e源码SHA `aa3e17585f8bd60a4d05e342abc7e2773c79a0b2d49f0370d15e29bcc6b72503`；
真实joint_head.safetensors SHA
`a010ac04f078e699988e4049cbea5e62c962393f59fec366640b64e8d69a4953`。
122项参数全部BF16，两rank实际参数逐项SHA/dtype/ND(format2)与checkpoint核同。

## 单变量和真实输入

只把同一memory的K/V两次BF16 F.linear合成原packed权重尾部的一次F.linear。
Q/out/norm/residual与两routing各自memory_norm保持原调用。
三路径为stock、packed view、packed后显式contiguous；stock_repeat提供A/A。
私有helper以单字段/B1/BF16/W1024/H16/L≤1595/Q≤7/可用packed权重作guard，
其它情况回stock。共享aa3e源码、安装器、allowed SHA和部署备份保持原样。

已有捕获缺真实head输入，本轮在私有stock服务armed保存实际
hidden/lexical/schema/logits与M967六处query/memory、decoder0前后fields。
D2H、参数SHA、捕获导出均在计时样本外。五条原冻结请求为
od-1534(L967/O4)、od-0105(L307/O6)、od-0677(L1595/O4)、od-1455(O2)、od-0250(O7)；
候选筛查只执行M967，其他捕获留tmp且不宣称候选验证通过。
两rank分别使用各自真实capture；M967 capture SHA与原数保存在summary。
stock离线logits、attention与完整decoder均先对在线捕获逐位校准。
K/V、attention、完整decoder、单处替换后的完整head logits/概率及重复输出全部逐位。

packed K/V shape均[1,967,1024]，stride=[1980416,2048,1]，offset=0/1024。
完整decoder包含self-attention、cross-attention、norm、FFN、residual及分配/布局/下发。
预热各8调用单列；两rank并发，5对交错，每样本16调用，pair各arm先取慢rank。
Event含host下发，wall含完成等待；以下均为完整decoder，单位ms。

| 路径 | Event中位数 | wall中位数 | 配对Event变化 | 配对wall变化 | Event更快对数 |
|---|---:|---:|---:|---:|---:|
| stock | 0.476166 | 0.484263 | — | — | — |
| stock_repeat | 0.477585 | 0.485744 | A/A最大7.162% | A/A最大6.564% | — |
| packed view | 0.480289 | 0.489149 | −2.367% | −2.733% | 3/5 |
| packed contiguous | 0.506355 | 0.514036 | +6.013% | +5.755% | 0/5 |

各arm中位数与配对差中位数分别计算；时序波动会使两者方向不同。
rank0/1 view Event配对改善2.367/1.365%，各自A/A最大7.990/3.328%，均未过门。

## 最小task证据

仅代表rank0串行短采，M967/同真实输入/同权重/同模块，3个profile配对。
CPU+NPU Level1/AiCoreNone，关闭stack/shape/memory/opargs，无PMU/MSTX重放。
数值导出在样本外，无逐op同步。各arm预热8调用，前后各2对无prof控制；
trace样本1调用，与原并发batch16口径分列。原生flow匹配12个range、288个kernel，
按stream/task/name/start time精确核同CSV；EVENT WAIT控制保留在raw并从kernel统计排除。

| 路径 | 每decoder MatMul/Cast/Slice数 | 全kernel数 | kernel union中位µs | 与stock配对中位差µs |
|---|---:|---:|---:|---:|
| stock | 8/8/0 | 24 | 138.5105 | — |
| stock_repeat | 8/8/0 | 24 | 138.1090 | −0.4015 |
| view | 7/7/2 | 24 | 138.9685 | +0.6790 |
| contiguous | 7/7/2 | 24 | 139.0685 | +0.5580 |

task A/A最大绝对差1.400µs。view新增的两任务为
`aclnnFlashAttentionScore_SliceAiCore_Slice`，合计中位8.5605µs；显式控制为
`aclnnInplaceCopy_SliceAiCore_Slice`，8.5005µs。证据支持当前BSH后端物化strided K/V。
完整MatMul组中位88.8255→83.3245µs，Cast组7.4200→6.3405µs；这些分组中位数
单独展示，避免相加去拟合整体中位差。所有计算task在同stream顺序执行，union与sum核同。

采集扰动明显：串行单调用stock Event前/profile/后为0.62448/0.89144/0.69815ms，
wall为0.72982/1.08748/0.86014ms。完整前后A/A、A/B原数保留summary，profile不作正式P95。
该trace提供任务结构与copy证据；原并发5对完整模块结果决定停止。
后续重开此变量需要后端免Slice消费现有stride/offset的具体支持依据，本轮不追加实现。

## 账目、恢复与复现

head新局部计算/采集保守按子进程wall累计 **22.48895183391869/240s**；
捕获请求、双rank并发probe和代表ranktrace分别记账，加载/定向停启/恢复另设有界。
GDN冻结 **76.87979675224051s**，旧MLP账保持原值。
首次私有实例加载时用户master socket消失，未发head请求/配对；通道恢复后先fresh
清理并完成aa3e22项恢复，再用新唯一目录续测。没有重建/关闭master或改SSH配置。

最终原aa3e/S1为 **RUNNING_VERIFIED**：
API **1749892/start423969384**，boot `393ecc15-721f-46f7-b119-21fa327a2b3b`；
TP **1750048/1750049**，卡7/chip0/1（逻辑0/1、Phy14/15）。
worker/health/text/image等22项通过，原argv/env匹配，私有实验树已清理，
双端`/tmp/clef27-service-current-state.json`核同。离线导出后再次fresh核身份/health/
源码/env与卡归属；额外NPU行是同一Clef API根进程，全部设备行归当前Clef树。

大raw本地tmp `clef40-head-kv-7spn0kdz`，远端
`/workspace/yangchen/work/clef40-head-kv-7spn0kdz`。
`window.json`记录时序/超时/预算，`single-rank{0,1}.json`为逐对与数值原数，
`profile-rank0.json`为扰动控制，`task-analysis.json`为完整关联；原生trace与CSV的路径/SHA见summary。
中断原件仍留tmp `clef40-head-kv-hi3k0qpk`。

CPU默认plan：`python3 runtime/probe_head_packed_kv.py`，不初始化设备。
设备执行命令由已fresh绑定的有界窗口串行控制；两个rank子进程并发，各有timeout55s，
入口为`probe_head_packed_kv.py --execute --phase single --rank {0,1}`，
其`--captures/--source/--output/--limit-s`实际值见tmp `run_window.py`。
离线解析用`runtime/analyze_mlp_gate_profile.py --sample-prefix headkv/ --divisor 1`
配合`--trace/--kernel-csv/--output`；当前12 range/288 tasks全核同。
必要检查：四脚本py_compile、五请求SHA、40项真实数值、旧21-range parser与修改前精确回归、
新parser CLI与直接API输出完全一致。精简原数在
`Clef-27B/logs/head-packed-kv-20261122/summary.json`。
