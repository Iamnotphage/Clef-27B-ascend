# 2026-11-12 A3 安装与完整环境验收记录

本轮已完成独立 Python 3.12.16、构建工具、固定 PyTorch / TorchNPU、vLLM `empty`、vLLM-Ascend 自定义扩展、Arctic、MemFabric、FLA 和 flash_attn_npu 安装。[完整环境验收](evidence/full-environment-smoke-2026-11-12.json)为 `passed=true`：`pip check`、实际版本与导入、双卡 FP16 / BF16 运算、78 个自定义算子注册及 Clef 自动识别均通过。双卡 HCCL、TP1 / TP2 服务、CPU 参考对齐以及 TP2 / 4096 图片和视频验证也已完成，详细验证与性能记录见[baseline](baseline-2026-11-12.md)。

执行机器为 `root@199.88.54.221`，工程为 `/workspace/yangchen/Clef-27B-ascend`。现有 CANN / A3 Ops / ATB 为 9.2.1，驱动为 26.1.1。两个 `Ascend910_9382` 逻辑设备为 0、1，物理设备为 14、15；资源调查见[新机器证据](evidence/new-machine-2026-11-12.json)。模型下载与逐文件哈希验收见[模型报告](evidence/new-machine-model-verification-2026-11-12.json)。

## Python、工具和安装材料

Python 使用 Astral 发布的 [python-build-standalone 20261110](https://github.com/astral-sh/python-build-standalone/releases/tag/20261110)，目标为 `aarch64-unknown-linux-gnu`。

| 项目 | 实际值 |
| --- | --- |
| 发行包 | `cpython-3.12.16+20261110-aarch64-unknown-linux-gnu-install_only.tar.gz` |
| 大小 | 53,264,815 字节 |
| SHA-256 | `4792f81750282597fb2a3389fe18b945ed2ba6aad5e45ad406d122c58cb33081` |
| 完整版本 | `3.12.16 (main, Nov 11 2026, 00:17:04) [Clang 23.1.3 ]` |
| Python 前缀 | `/workspace/yangchen/python/cpython-3.12.16+20261110` |
| 独立环境 | `/workspace/yangchen/venvs/clef-py312` |
| 实际解释器 | `/workspace/yangchen/venvs/clef-py312/bin/python` |

[发行包来源与哈希](evidence/python-material-2026-11-12.json)由 [bootstrap_python.sh](../scripts/bootstrap_python.sh) 固定并校验。该脚本还检查版本及 `ssl`、`sqlite3`、`lzma`、`bz2`、`ctypes` 导入。[activate_env.sh](../scripts/activate_env.sh) 加载 CANN、ATB 和独立环境，将模型、日志及缓存放在 `/workspace/yangchen` 下，默认 `MAX_JOBS=8`。

工具分发版本以[构建工具快照](evidence/build-tools-freeze-2026-11-12.txt)和[最终完整快照](evidence/full-environment-freeze-2026-11-12.txt)为据：CMake 3.31.10、Ninja 1.13.0、packaging 26.3、setuptools 80.10.2、setuptools-scm 10.3.4、setuptools-rust 1.13.0、wheel 0.48.0、Jinja2 3.1.6、pybind11 3.1.0；pip 安装固定为 26.3。Arctic 构建另补齐 nanobind 2.9.2、grpcio-tools / grpcio 1.85.1 和 protobuf 7.36.2，当前 [build-tools.txt](../requirements/build-tools.txt)已包含这些版本。系统 GCC/G++ 11.4.0、clang-15 `1:15.0.7-0ubuntu0.22.04.3` 已具备。开发依赖检查确认 libprotobuf-dev 3.12.4、numactl 2.0.14、libibverbs-dev 39.0、libhiredis-dev 0.14.1 已满足，本轮补装 pkg-config 0.29.2 和 libglib2.0-0 2.72.4。

四个 CPU / NPU wheel 共 188,875,463 字节，大小和 SHA-256 均校验通过，详见[逐包报告](evidence/base-torch-wheels-2026-11-12.json)。Triton Ascend 完整 wheel 为 322,887,778 字节，SHA-256 为 `66d8c315fcd1ab619f38c6250b3d28868254c606b653c098d796f707da4b5400`，来源见[材料记录](evidence/triton-material-2026-11-12.json)。

网络处理采用以下已执行路径：新机器访问 GitHub raw 受限，Python 发行包、固定源码、submodules 和完整 Triton wheel 在本地准备并上传。PyTorch 官方远端单流下载较慢，三个 CPU wheel 经本地无代理直连下载后上传；TorchNPU 通过华为 PyPI 镜像下载，并核对官方 wheel SHA-256。本地 `urllib` 使用 `ProxyHandler({})` 下载材料。GitHub 源码传输发生截断后，按固定 Git tree 补齐并逐 blob 核验，再打包完整源码。

## 固定运行依赖与实际加载路径

北京时间 16:19:26—16:19:31 的[基础报告](evidence/torch-npu-smoke-2026-11-12.json)和 16:20:15—16:20:22 的[runtime 报告](evidence/runtime-smoke-2026-11-12.json)保留了扩展构建前的验收结果。17:13:28—17:13:52 的[完整环境报告](evidence/full-environment-smoke-2026-11-12.json)使用 `--require-extensions`，最终 `passed=true`，`pip check` 返回 0，输出 `No broken requirements found.`。该快照记录候选 embedding 兼容补丁应用前的环境安装结果，后续标准 TP2 补丁另由真实协议与参考精度报告验收。

以下模块均已实际导入，公共目录为 `/workspace/yangchen/venvs/clef-py312/lib/python3.12/site-packages/`。

| 分发包 | 实际分发版本 | 该目录下的实际模块文件 |
| --- | --- | --- |
| torch | `2.10.0+cpu` | `torch/__init__.py` |
| torch-npu | `2.10.0.post7` | `torch_npu/__init__.py` |
| torchvision | `0.25.0+cpu` | `torchvision/__init__.py` |
| torchaudio | `2.10.0+cpu` | `torchaudio/__init__.py` |
| transformers | `5.19.0` | `transformers/__init__.py` |
| triton-ascend | `3.6.0.post2` | `triton/__init__.py`，模块 `__version__=3.6.0` |
| flash-linear-attention-npu-a3 | `26.9.1` | `fla_npu/__init__.py` |
| flash_attn_npu | `0.5.0` | `flash_attn_npu/__init__.py` |

vLLM 实际分发版本为 `0.1.dev1+g24420b7a3.d20261112.empty`，模块位于 `/workspace/yangchen/work/src/vllm/vllm/__init__.py`；Ascend 分发版本为 `0.1.dev1+g89eb8a5e7`，模块位于 `/workspace/yangchen/work/src/vllm-ascend/vllm_ascend/__init__.py`。这两个版本来自固定提交的源码安装，完整验收同时核对了实际 Git HEAD。Arctic 0.1.1、memfabric-hybrid 1.3.0、memcache-hybrid 1.3.0 和 fastokens 0.2.0 已安装，版本保存在最终快照中。

Triton Ascend 的 wheel 元数据固定 NumPy 1.26.4、SciPy 1.13.1、psutil 6.0.0、attrs 24.2.0、decorator 5.1.1、pytest-xdist 3.6.1 和 triton 3.6.0。[runtime-constraints.txt](../requirements/runtime-constraints.txt)保留这些约束；[cann-tools.txt](../requirements/cann-tools.txt)补齐 CANN 经 `PYTHONPATH` 暴露的工具依赖。扩展构建前的快照保存在[runtime freeze](evidence/runtime-freeze-2026-11-12.txt)，最终传递依赖以[完整环境 freeze](evidence/full-environment-freeze-2026-11-12.txt)为准。其中的 CANN `file://` 来源属于预装工具的审计记录，重建入口为仓库 requirements 和已校验材料。

实际动态库映射包含 `/usr/local/Ascend/cann-9.2.1/aarch64-linux/lib64/`、`/usr/local/Ascend/driver/lib64/driver/`，以及 `/usr/local/Ascend/nnal/atb/9.2.1/atb/cxx_abi_1/lib/libatb.so`、`libasdops.so`、`libatb_mixops.so`；完整路径保存在 smoke JSON 的 `loaded_libraries` 中。`torch.version.cuda` / `hip` 均为 `null`，`torch.npu.is_available()` 为 `true`，设备数为 2。

完整扩展检查确认平台为 `vllm_ascend.platform.NPUPlatform`，模型配置自动转为 `ClefFlashForJointSchema`，解析类为 `vllm_ascend.models.clef_flash.AscendClefFlashForJointSchema`，并枚举出 78 个已注册的 `_C_ascend` 自定义算子。该计数对应扩展加载与 dispatcher 注册；模型端到端执行结果记录在 baseline 中。

## 双卡运算结果

[verify_npu_environment.py](../scripts/verify_npu_environment.py)在每卡执行 FP16、BF16 的 2048×2048 matmul，随机种子为 `20261112`，CPU 线程数为 8。输入 `A~N(0,1)/sqrt(2048)`、`B~N(0,1)`，先量化到目标 dtype，再用同一组可表示值计算 CPU FP32 参考结果；两个 NPU 使用相同输入。测试关闭 `matmul.allow_hf32`，并显式同步、拷回 CPU 检查。

| 逻辑设备 | dtype | 最大绝对误差 | relative L2 | NaN / Inf | 超容差元素 | 结果 |
| --- | --- | ---: | ---: | --- | ---: | --- |
| 0 | FP16 | 0.0019483566 | 0.0002076348 | 0 / 0 | 0 | 通过 |
| 1 | FP16 | 0.0019483566 | 0.0002076348 | 0 / 0 | 0 | 通过 |
| 0 | BF16 | 0.0155997276 | 0.0016619213 | 0 / 0 | 0 | 通过 |
| 1 | BF16 | 0.0155997276 | 0.0016619213 | 0 / 0 | 0 | 通过 |

FP16 判据为逐元素 `abs(error) <= 5e-4 + 1e-3*abs(reference)`，且 relative L2 ≤ 1e-3；BF16 使用 `4e-3 + 8e-3*abs(reference)`，且 relative L2 ≤ 8e-3。基础、runtime 和完整扩展验收得到相同误差。输出设备分别为 `npu:0`、`npu:1`，同步和有限值检查均通过。报告中的单次 `synchronized_matmul_ms` 用于运算 smoke 记录，服务性能采用 baseline 中的预热与固定并发测试。

## 源码、补丁和构建记录

| 源码 | 固定提交 |
| --- | --- |
| vLLM | `24420b7a341af37a9bc3186e6fc6062512e0aa58` |
| vLLM-Ascend | `89eb8a5e7a76d085b453a1aeadb6883e12d5ea2d` |
| catlass submodule | `41bf90da655bba3c66d0acd7e00abe33960ecfd6` |
| catlass/googletest submodule | `f8d7d77c06936315286eb55f8de22cd23c188571` |

vLLM 的完整 Git 包为 100,566,986 字节，SHA-256 `6644c8c3e7ae72032da668d8a121c7736e881458b5708d9027c9bc3f26efacdc`，8,665 个 blob 已核对，打包时工作区干净，无 submodule。[vLLM 材料清单](evidence/vllm-source-2026-11-12.json)保留传输恢复过程。Ascend 完整包为 66,764,549 字节，SHA-256 `34f02faff3c9e20faba385729c5c94267e7d911088ca12997705bb688a0318de`，包含上述递归 submodules，打包时均干净，见[Ascend 材料清单](evidence/vllm-ascend-source-2026-11-12.json)。

Mac 打包的源码解压使用 `tar --no-same-owner`，让远端 root 拥有文件。先期解压的 Ascend / flash 源码已修正所有权，解决 Git `dubious ownership` 检查。

vLLM 安装应用了 [vllm-a3-numpy126.patch](../scripts/patches/vllm-a3-numpy126.patch)：将 `opencv-python-headless>=4.13.0` 改为 `==4.11.0.86`，满足 A3 Triton wheel 的 NumPy 1.26.4 约束。该改动作用于视频 IO 依赖声明；主干和联合决策头源码保持固定提交版本。补丁 SHA-256 为 `412cf3f64c23853f8496fc526ca0b0fc701b5525d909c36785b2908ca99d0da0`。

[vllm-clef-load-audit.patch](../scripts/patches/vllm-clef-load-audit.patch)已应用，在权重加载和原有 tracking 完成后输出 `CLEF_LOAD_AUDIT` 日志。补丁读取参数名、形状、dtype、设备和加载集合，保留原有加载与前向逻辑；SHA-256 为 `f6ddcafacdfe8b6a21a31c1afd8e07110c44ebb35c58c3f862163551a73419d9`。TP1 启动已实际输出审计记录，见下节。

[vllm-ascend-clef-tp-embedding.patch](../scripts/patches/vllm-ascend-clef-tp-embedding.patch)已应用，补齐候选词向量读取所需的三个构造属性：`use_fused_embedding=False`、`reduce_results=True` 和 `parallel_group=output_reduce_group`。Clef 通过上游 `VocabParallelEmbedding.forward` 读取 LM head 词表向量；TP2 路径需要这些属性，初次请求实际报告了 `AscendParallelLMHead` 缺少 `use_fused_embedding`。该补丁将 Ascend 构造器与标准 TP 接口对应，采用可移植的 mask / gather 和 TP 归约来还原候选词向量。原有 `AscendVocabParallelEmbedding.forward` 的 AST 经补丁前后核对一致。补丁 SHA-256 为 `862f4f0dc8ebf25b31ab534bd45fa098ec751b8bf31ff37a2f23310414c75e07`；应用后的标准 TP2 协议及参考精度检查均通过，TP1 对照保留补丁应用前的已完成报告。

[flash-attn-npu-max-jobs.patch](../scripts/patches/flash-attn-npu-max-jobs.patch)已应用，让 flash_attn_npu 0.5.0 的编译线程池读取 `MAX_JOBS`，默认上限为 8。原实现依据宿主 `os.cpu_count()` 选择并发；此实例采用 32 核等价 CPU 时间和 128 GiB 内存配额，本轮显式使用 `MAX_JOBS=8`。补丁 SHA-256 为 `bb8bb4ea1c2d93e054065f5df61626bf5c04f35eaa64911f16ba282f84df63c0`。`FLASH_ATTN_BUILD_NPU=910` 选择的 v2 / v3 / v4 共 122 个 Ascend C 编译单元已完成，wheel 构建及安装成功。

固定 vLLM 的 `pyproject.toml` 在 `[build-system].requires` 声明 `torch==2.14.0`。本轮对 `empty` 构建使用 `--no-build-isolation`，由现有 setuptools 后端复用目标环境中的 `torch==2.10.0+cpu`，并使用 runtime constraints 约束依赖解析。Ascend 插件也复用该运行环境，采用 `SOC_VERSION=ascend910_9382 COMPILE_CUSTOM_KERNELS=1` 构建；最终报告核对了运行版本与源码 HEAD。

Arctic 0.1.1 的隔离构建声明包含 torch 2.7.0 和 protobuf 5.29.5。本轮在目标环境补齐 `nanobind==2.9.2`、`grpcio-tools==1.85.1`、`protobuf==7.36.2`，随后以 `--no-build-isolation` 重试，Ascend 与 Arctic 均成功构建并安装。grpcio-tools 1.85.1 要求 `protobuf>=7.35.1,<8`，与实际版本相容。首次 Arctic 构建缺少 nanobind 的失败记录保留在原日志，成功结果保存在 retry 日志。

`TORCH_DEVICE_BACKEND_AUTOLOAD=0 mfcli kernel install --soc-version A3` 已成功完成，在环境内生成 `memfabric_hybrid/lib/libmf_hybm_accoffload.so`。A3 分支使用 wheel 内的 Ascend C 源码与本机 CANN 编译，目标为 `ascend910_9382`。FLA 安装使用已校验的 26.9.1 A3 wheel，随后补齐 `expecttest==0.3.0`；最终 `pip check` 通过。FLA 模块名为 `fla_npu`，其 wheel 自带 `fla_npu_opp_env.pth`，在解释器 site 初始化时把包内 OPP 路径加入 `ASCEND_CUSTOM_OPP_PATH`，早于用户代码导入 `torch_npu`。

## 完整模型加载审计

`clef-tp1-1024-20261112.log` 记录主干 safetensors 的 12/12 分片和独立联合头的 1/1 文件加载完成。实际模型类为 `AscendClefFlashForJointSchema`，语言模型共 64 层，视觉分支为 `Qwen3_VisionTransformer`、460,730,096 个参数元素；联合头 122/122 个参数张量加载完成，`missing_names=[]`，`strict_tracking_enabled=true`。

NPU 审计统计 1,162 个已注册参数张量、27,486,750,964 个元素。与 CPU 参考的 27,484,784,884 相比，多出的 1,966,080 个元素由 48 层 GDN 的 `ascend_conv1d_weight` 派生布局参数贡献：模型配置给出 `D=2×16×128+48×128=10240`、卷积宽度 `W=4`，每层注册 `[4,10240]`、`requires_grad=False` 的参数，总计 `48×4×10240=1,966,080`，BF16 数据量为 3.75 MiB。

该注册与从原权重转置、连续化复制的实现见固定 Ascend [gdn.py:79–115](https://github.com/vllm-project/vllm-ascend/blob/89eb8a5e7a76d085b453a1aeadb6883e12d5ea2d/vllm_ascend/ops/gdn.py#L79)，每层初始化入口见 [patch_qwen3_5.py:218–223](https://github.com/vllm-project/vllm-ascend/blob/89eb8a5e7a76d085b453a1aeadb6883e12d5ea2d/vllm_ascend/patch/worker/patch_qwen3_5.py#L218)，通道数与原始卷积权重形状见固定 vLLM [qwen_gdn_linear_attn.py:583–618](https://github.com/vllm-project/vllm/blob/24420b7a341af37a9bc3186e6fc6062512e0aa58/vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py#L583)。本轮按远端相关源码哈希、实际配置和现有报告完成静态核算。

后续 TP2 的两 rank 也完成主干与联合头加载审计：各有 64 层、联合头 122/122 张量且 `missing_names=[]`，LM head 的本地形状为 `[124160,5120]`。TP2 / 1024 修复后以及 TP2 / 4096 均完成五组协议请求和七个 CPU 参考决策对齐；4096 配置的红、蓝图片与两种视频帧顺序共 4/4 请求通过。双卡 HCCL 的 FP32 / BF16 `all_reduce_SUM`、barrier 与进程组销毁均通过，详见 baseline 与 [rank 0](evidence/hccl-rank0-2026-11-12.json) / [rank 1](evidence/hccl-rank1-2026-11-12.json) 报告。

## 复现命令与远端日志

以下顺序使用已上传到 `/workspace/yangchen/work/materials/` 的已校验材料和当前 requirements。执行目录为工程，解释器由激活脚本选定。

```bash
cd /workspace/yangchen/Clef-27B-ascend
bash scripts/bootstrap_python.sh /workspace/yangchen/work/materials/cpython-3.12.16+20261110-aarch64-unknown-linux-gnu-install_only.tar.gz
source scripts/activate_env.sh
project=/workspace/yangchen/Clef-27B-ascend
materials=/workspace/yangchen/work/materials
constraints="$project/requirements/runtime-constraints.txt"
package_index=https://mirrors.huaweicloud.com/repository/pypi/simple

python -m pip install --index-url "$package_index" --constraint "$constraints" --timeout 60 -r requirements/build-tools.txt
python -m pip install --index-url "$package_index" --constraint "$constraints" --timeout 60 \
  "$materials/wheels/torch-2.10.0+cpu-cp312-cp312-manylinux_2_28_aarch64.whl" \
  "$materials/wheels/torchvision-0.25.0+cpu-cp312-cp312-manylinux_2_28_aarch64.whl" \
  "$materials/wheels/torchaudio-2.10.0+cpu-cp312-cp312-manylinux_2_28_aarch64.whl" \
  "$materials/wheels/torch_npu-2.10.0.post7-cp312-cp312-manylinux_2_28_aarch64.whl" \
  -r requirements/cann-tools.txt
python scripts/verify_npu_environment.py --stage torch --output /workspace/yangchen/logs/torch-npu-smoke-20261112.json

python -m pip install --index-url "$package_index" --constraint "$constraints" --timeout 60 \
  "$materials/triton_ascend-3.6.0.post2-cp312-cp312-manylinux_2_27_aarch64.manylinux_2_28_aarch64.whl" \
  transformers==5.19.0
python scripts/verify_npu_environment.py --stage runtime --output /workspace/yangchen/logs/runtime-smoke-20261112.json
```

全新源码目录按以下方式解压、应用已记录补丁，再构建 `empty` 和 Ascend。重复使用现有源码目录时先核对 `git rev-parse HEAD` 和 `git diff`。当前 `requirements/build-tools.txt` 已包含 Arctic 所需的 nanobind、grpcio-tools 和 protobuf；前一节的工具安装步骤会一并准备它们。

```bash
mkdir -p /workspace/yangchen/work/src
tar --no-same-owner -xzf "$materials/vllm-24420b7-with-git.tar.gz" -C /workspace/yangchen/work/src
tar --no-same-owner -xzf "$materials/vllm-ascend-89eb8a5-with-submodules.tar.gz" -C /workspace/yangchen/work/src
cd /workspace/yangchen/work/src/vllm
patch -p1 < "$project/scripts/patches/vllm-a3-numpy126.patch"
patch -p1 < "$project/scripts/patches/vllm-clef-load-audit.patch"
VLLM_TARGET_DEVICE=empty python -m pip install --no-build-isolation \
  --constraint "$constraints" --index-url "$package_index" --timeout 60 \
  --report /workspace/yangchen/logs/vllm-install-20261112.json -e .

cd /workspace/yangchen/work/src/vllm-ascend
SOC_VERSION=ascend910_9382 COMPILE_CUSTOM_KERNELS=1 MAX_JOBS=8 \
  python -m pip install --no-build-isolation --constraint "$constraints" \
  --index-url "$package_index" --timeout 60 \
  --extra-index-url https://mirrors.huaweicloud.com/ascend/repos/pypi \
  --extra-index-url https://mirrors.huaweicloud.com/ascend/repos/pypi/nightly \
  --report /workspace/yangchen/logs/vllm-ascend-install-20261112.json -e .
patch -p1 < "$project/scripts/patches/vllm-ascend-clef-tp-embedding.patch"

TORCH_DEVICE_BACKEND_AUTOLOAD=0 mfcli kernel install --soc-version A3

python -m pip install --no-deps \
  --report /workspace/yangchen/logs/fla-install-20261112.json \
  "$materials/flash_linear_attention_npu_a3-26.9.1-py3-none-manylinux_2_34_aarch64.whl"
python -m pip install --constraint "$constraints" --index-url "$package_index" \
  --report /workspace/yangchen/logs/fla-deps-completion-20261112.json \
  expecttest==0.3.0

tar --no-same-owner -xzf "$materials/flash_attn_npu-0.5.0.tar.gz" -C /workspace/yangchen/work/src
cd /workspace/yangchen/work/src/flash_attn_npu-0.5.0
patch -p1 < "$project/scripts/patches/flash-attn-npu-max-jobs.patch"
FLASH_ATTN_BUILD_NPU=910 FLASH_ATTENTION_FORCE_BUILD=TRUE MAX_JOBS=8 \
  python -m pip install --no-build-isolation --constraint "$constraints" \
  --index-url "$package_index" --timeout 60 \
  --report /workspace/yangchen/logs/flash-attn-install-20261112.json .

python -m pip check
python "$project/scripts/verify_npu_environment.py" --stage runtime --require-extensions \
  --model-dir "$MODEL_DIR" --output /workspace/yangchen/logs/full-environment-smoke-20261112.json
python -m pip freeze --all > /workspace/yangchen/logs/full-environment-freeze-20261112.txt
```

vLLM、Ascend 和 flash 的 pip 命令保留运行依赖解析。FLA 使用本地 wheel 的 `--no-deps` 安装，传递依赖由已有固定环境与 `expecttest` 补齐，随后通过完整 `pip check` 验收。已有环境补齐 Arctic 构建依赖时，本轮执行的版本组合如下；完整重建由前述 build-tools 步骤覆盖。

```bash
python -m pip install --constraint "$constraints" --index-url "$package_index" \
  --report /workspace/yangchen/logs/arctic-build-deps-install-20261112.json \
  nanobind==2.9.2 grpcio-tools==1.85.1 protobuf==7.36.2
```

实际文本日志和 pip 报告均位于 `/workspace/yangchen/logs/`。成功配置对应下表；首次失败的原日志保留用于核查补依赖过程。

| 阶段 | 日志文件 |
| --- | --- |
| Python | `python-bootstrap-20261112.log` |
| 系统依赖 | `apt-update-20261112.log`、`apt-build-deps-20261112.log` |
| 构建工具 | `build-tools-install-mirror-20261112.log` |
| 基础 wheel | `base-torch-install-local-20261112.log` |
| CANN 工具依赖补齐 | `cann-tools-completion-20261112.log` |
| Transformers / Triton | `runtime-install-20261112.log` |
| vLLM empty | `vllm-build-20261112.log` |
| Ascend 首次构建及 Arctic 缺依赖记录 | `vllm-ascend-build-20261112.log` |
| Arctic 构建依赖补齐 | `arctic-build-deps-install-20261112.log` |
| Ascend / Arctic 成功重试 | `vllm-ascend-build-retry-20261112.log` |
| MemFabric AccOffload | `memfabric-kernel-install-20261112.log` |
| FLA wheel / expecttest 补齐 | `fla-install-20261112.log`、`fla-deps-completion-20261112.log` |
| flash_attn_npu 构建安装 | `flash-attn-build-20261112.log` |
| 完整环境最终验收 | `full-environment-smoke-final-20261112.log` |
| Clef TP1 启动及加载审计 | `clef-tp1-1024-20261112.log` |
| 双卡 HCCL | `hccl-20261112.log`、`hccl-20261112/` |
| TP2 首次候选词向量错误 / 修复后启动 | `clef-tp2-1024-20261112.log`、`clef-tp2-1024-retry-20261112.log` |
| TP2 / 4096 多模态服务 | `clef-tp2-4096-mm-20261112.log` |

pip `--report` 的实际文件名为 `vllm-install-20261112.json`、`vllm-ascend-install-20261112.json`、`arctic-build-deps-install-20261112.json`、`fla-install-20261112.json`、`fla-deps-completion-20261112.json` 和 `flash-attn-install-20261112.json`；构建工具报告为 `build-tools-install-20261112.json`。最终完整环境证据已保存到仓库的 [smoke JSON](evidence/full-environment-smoke-2026-11-12.json)和[依赖快照](evidence/full-environment-freeze-2026-11-12.txt)。

`Clef-27B/start.sh` 与 `inference.py` 已接通真实 TP1 / TP2 服务。TP1 和 TP2 的 CPU BF16 参考决策均达到 7/7；TP2 候选概率最大绝对差约为 0.000256，4096 配置的图片和视频请求共 4/4 通过。三来源固定公开资料的 2,983 请求、4,583 决策已全部完成，执行错误为 0，当前 TP2 / 4096 多模态服务继续运行。启动参数、缓存池与设备级 HBM 的统计范围、原生与项目指标、完整报告路径和复现命令统一见[baseline](baseline-2026-11-12.md)。[性能与精度图](../Clef-27B/result.png)和[设备调用图](../Clef-27B/npu_device_call.png)均为真实 JSON / CSV 与 `npu-smi` 数据生成的“实测报告导出图”，分别记录 1024 固定测量与 4096 单次设备快照。
