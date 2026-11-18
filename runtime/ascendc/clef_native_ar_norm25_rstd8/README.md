# N1 v3：仅聚合 rstd DMA

源于v2核心 `d3a27ddaf26c7497761110091b4ebc92475ea6b1`，v2 c24正确性、恢复和计时原件保留。
v2正式20对样本没有稳定收益；实测总结见相邻v2 README的“cores24实际闭合”。
2026-11-19用户授权本独立最小优化。设备、隔离安装、加载和服务只交owner27，20统一排程。

## 可证伪来源与唯一功能差异

v2 `Row()` 每行 `DataCopyPad(rs_[row], stat, {1,4,0,0,0})`；967行共967次rstd写出请求。
同文件 `Process()` 在行循环前每核搬运/转换一次gamma。adapter没有显式stream同步；
ARon的reduced是HCCL直接目标。当前总event/wall没有分阶段耗时，以上均为源码事实，
每行4B DMA的耗时占比仍是假设。

v3增加32B UB缓冲，把原 `stat.GetValue(0)` 得到的同一个FP32 inv放入该核的槽0..7。
槽7或该核尾行触发一次DataCopyPad，长度严格为有效项数×4B。块起点为各核start，
不向全局8行边界取整；UB DMA源始终从32B对齐基址出发，GM允许DataCopyPad非对齐目的地址。
同核下一块复用前完成MTE3→S；每次发布前完成S→MTE3。v2全部26处PipeBarrier保持原序列。
浮点算术/参数/归约顺序、BF16 roundtrip、AR算法与生命周期、gamma、diag和adapter分配均保持。
名称归一化后host、tiling头、adapter和构建文件与v2逐字节相同。

| producer / region | consumer / region | 同步 |
| --- | --- | --- |
| 原vector stat[0] | 原scalar inv | 原PIPE_ALL，保持 |
| scalar rstdTile[0..slot] | MTE3到rs[row-slot..row] | 新S_MTE3 Set/Wait |
| MTE3读取rstdTile | 下一批scalar重用同32B | 新MTE3_S Set/Wait，原PIPE_ALL保持 |

CANN9.2.1真实SDK `compiler/tikcpp/tikcfw/impl/kernel_tensor_impl.h:643,673`：
LocalTensor::SetValue标注S流水，直接写UB；同树
`dav_c220/kernel_operator_sync_impl.h:89-93`存在scalar写→S_MTE3→UB/GM DMA实例。
同步扫描红线/高等级0，图findings0；15条粗屏障性能提示保留。正则扫描未展开SDK/HCCL，
完整TU编译和CPU地址模型均不证明设备完成/可见性。

## 独立身份与完整编译

- op：`ClefNativeArNorm25Rstd8`；vendor：`clef_native_ar_norm25_rstd8`
- Torch：`clef_n1_25_rstd8.run`；`dlopen(RTLD_LOCAL)` 使用独立opapi绝对路径。
- build：`/workspace/yangchen/work/clef-n1-25-v3-rstd8`
- 完整host/kernel TU通过CANN9.2.1 `MAX_JOBS=4 bash build.sh`；真实adapter完整纯编译/链接通过。
- 生成API cpp231-232仍无条件SetMc2/SetHcomGroup；核内Init/SetCc/Finalize原样保留。
- 本阶段没有安装、加载adapter、launch或占用NPU。

| 文件（相对build） | SHA256 |
| --- | --- |
| build_out/custom_opp_ubuntu_aarch64.run | 6d5a371073102b8de096e5227ea36aac630cc32f897655ff65327fee1a1db9f1 |
| n1_adapter.so | 7c17b601d5b1998f180437c4eab9840e865d294503ceeb783d45c1565782dfff |
| build_out/autogen/aclnn_clef_native_ar_norm25_rstd8.h | 84064148a5ec0e9796d68010ec391e5cd81c1c288fdb40fd15b5023ac9bafa88 |
| identity.json | 3cf0b12d8e76a50bd31bb34a5befe97c6548e39568cf611c9a67ef29a1558a63 |

16个vendor文件及6个源码的fullSHA见`runtime/n1_25_rstd8/build_identity.json`。
adapter RUNPATH为固定venv的torch/lib、torch_npu/lib及CANN9.2.1/lib64。
编译日志 `/workspace/yangchen/work/clef-n1-25-v3-rstd8-build.log` 与同前缀`-adapter.log`。

ABI保持3输入、4属性、5输出，输入partial/residual BF16 ND[M,5120]、gamma BF16[5120]，
属性eps(double)/group(char*)/useHccl(int64)/cores(int64)，输出依序Y BF16[M,5120]、
residual BF16[M,5120]、rstd FP32[M]、reduced BF16[M,5120]、diag UINT32[48,32]。
Torch diag容器int32不变。UB从103040增至103072B；tiling布局和字段不变。

## CPU验证与待设备判据

`runtime/n1_25_rstd8/rstd8_model.py --output /tmp/n1-rstd8-model-result.json`：
786432个M1..16384/cores1..48组合、19267584核区间的闭式算术全部通过；
另3552组逐DMA枚举及错误32B尾块反例通过。967/c24为127次DMA=120×32B+7×4B，
总3868B，每一块严格在所属核内；7个尾块属于core3/6/10/13/17/20/23。
该CPU模型覆盖空核、短核、未对齐GM起点、尾与跨核边界；设备只排cores24。
结果SHA `03de46a4f82aab11d39526ca3c06d72ab3fb0d4ff903eea02f1c5f4f8e7a7299`。

监督接线/分区14项、diag契约5项、统计协议10项CPU测试通过。
固定门文件041a66…、统计协议53fe91…保持v2字节；namespace/vendor/stage路径与身份独立。
新窗口仍rows967/seed20261118/ARon/cores24，先两rank固定门、原raw及真实diag集合，
通过后按原4轮AB/BA/BA/AB、5pair/轮、3warmup、1first-call协议计时。
浮点门、shape/dtype/finite、explicit reduced逐位门保持。rstd须完整967项并特别核查各核尾/边界。
新旧输出逐字节对照与计时结论均待设备原件；无稳定收益就关闭本优化假设。
同PGID240秒监督与失败finally恢复复用wrapper5c045eb；cores48不排。
