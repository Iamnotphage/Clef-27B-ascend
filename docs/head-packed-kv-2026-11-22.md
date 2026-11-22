# 同请求 head packed KV（2026-11-22）

基线 `b9580788f0729c74d38d0f8119f5a4011aa132b4`，分支
`perf/head-packed-kv-20261122`。aa3e 原件 SHA-256
`aa3e17585f8bd60a4d05e342abc7e2773c79a0b2d49f0370d15e29bcc6b72503`。
仅将同一 memory 的 K/V 两次 BF16 F.linear 改为原 packed 权重尾部的一次
F.linear。Q/out/norm/residual、两 routing 独立 memory_norm 保持原调用。
私有 helper 独立于生产安装器；三路径为 stock、strided view、显式 contiguous。
单字段/B1/BF16/W1024/H16/L≤1595/Q≤7/可用原packed权重以外回 stock。

真实 joint_head.safetensors SHA-256
`a010ac04f078e699988e4049cbea5e62c962393f59fec366640b64e8d69a4953`。
私有 stock capture 服务 armed 一次性保存实际 head hidden/lexical/schema/logits，
M967 另存六处 query/memory 及完整 decoder0 输入/输出；D2H 与捕获均在计时外。
实际参数逐项 SHA/dtype/format 与 checkpoint 核同后才复放。
只报告真实来源设备；同一冻结输入在另一逻辑 NPU 复放时保留来源标记。

先 M967 单 decoder 数值与 ≥5 对交错 A/A、A/B，完整模块包含
self-attn、cross-attn、norm、FFN、residual 及布局/分配/下发。
Event 含 host 下发，wall 含完成等待，均与纯 task 时长分开。
单处可辨收益后才放六处并测完整 head（含原概率边界）；再覆盖
L307/1595、候选2/7。多字段和长图文保持回退，采用前补必要守护。
完整 head 改善须超过 A/A；通过后才进入 S1 三轮及完整 OD/ERQA 四项2pp。

本轮新局部计算/采集上限240s；GDN冻结76.87979675224051s，MLP账不变。
控制脚本和大raw：本地 tmp `clef40-head-kv-hi3k0qpk`，远端
`/workspace/yangchen/work/clef40-head-kv-hi3k0qpk`。
所有退出先恢复aa3e/S1，核22项、原argv/env、设备归属和双端状态指针。

CPU准备验证：三个脚本 py_compile、默认plan（不初始化设备）、五请求SHA。
执行状态和关键原数原位更新至 `Clef-27B/logs/head-packed-kv-20261122/summary.json`。

阶段阻点：私有stock捕获服务加载期间，用户master socket消失。启动等待与finally清理均返回SSH 255；无head请求/数值/配对结果，局部计算账0s。共享aa3e源码未改。当前远端服务状态UNKNOWN、恢复待通道；已准备fresh绑定私有实例的恢复入口，用户通道恢复后优先执行22项恢复。
