# 环境构建与评测镜像 Handoff（2026-11-25）

## 0. 一句话现状

仓库已能在项目内一键建 `.venv`（`scripts/setup_env.sh`），纯 PyPI 层已跑通，但 **vLLM-Ascend 的 Ascend C 扩展编译在 `/workspace/final-test` 容器上失败（`cmake --build ... --target=vllm_ascend_C` exit 2，真实错误未取得）**；同时出现一个更关键的问题：**如果评测机固定使用 `quay.io/ascend/vllm-ascend:v0.30.0-a3` 镜像，本仓库的运行时补丁（`src/runtime/apply_*.py`）会因 SHA-256 不匹配而拒绝启动**。下一步必须先确认评测镜像归属，再决定“移植补丁 / 自带镜像 / wheelhouse 安装”。

## 1. 仓库状态

- 分支 `main`，HEAD `4662f84`，已推送 `origin/main`。
- 本轮相关提交（全部已在 main）：

| 提交 | 内容 |
|---|---|
| `f7d0918` | `scripts/setup_env.sh`：项目内 `.venv` 一键安装/构建 |
| `2eb68e9` | 材料默认改到 `<repo>/.build/materials`，缺失自动下载；CPython 前缀移到 `.build` |
| `89e7aca` | `pip download flash-attn-npu` 加 `--no-build-isolation` |
| `33ac857` | CPython 包优先 NJU GitHub-release 镜像，GitHub 兜底，SHA 校验 |
| `a55b8d6` | torch/torchvision/torchaudio 走 NJU/阿里云镜像；torch-npu 走华为 PyPI 索引；vLLM clone 走 gh-proxy/ghproxy 兜底 |
| `ef46a9d` | 构建加 `-v` 并 tee 到 `.build/logs/`，失败自动打印 `error:` 上下文；构建前检查 `ASCEND_HOME_PATH` |
| `4662f84` | `scripts/probe_runtime_image.py`：评测镜像兼容性探针 |

- 工作区只剩用户自己的 `README.md` 表格对齐改动，**不要提交**。
- 上游 PR：#7（扁平化重构）`f8afbb6`、#8（项目 `.venv`）`30d1716` 已合入 main。

## 2. 当前环境方案（`scripts/setup_env.sh`）

### 2.1 用法

```bash
cd <repo>
bash scripts/setup_env.sh              # 完整：.venv + 依赖 + vLLM/vLLM-Ascend/flash_attn 源码构建
bash scripts/setup_env.sh --pip-only   # 只建 .venv + 平台 wheel + requirements.txt，验证 pip 锁
bash scripts/setup_env.sh --force      # 已完成也重建
source scripts/activate_env.sh         # 加载 CANN/ATB + .venv（自动优先 <repo>/.venv）
bash start.sh                          # 自动使用 <repo>/.venv/bin/python
```

### 2.2 四层结构（关键认知）

`requirements.txt` 只是第 2 层，**不是完整环境**：

| 层 | 内容 | 来源 |
|---|---|---|
| 1 平台 wheel | torch 2.10.0+cpu / torchvision / torchaudio / torch-npu 2.10.0.post7 | `scripts/requirements/base-torch.txt`（sha256 直链）或国内镜像 |
| 2 纯 PyPI 锁 | 199 个包（`requirements.txt`） | 华为 PyPI 镜像 + `scripts/requirements/runtime-constraints.txt` |
| 3 源码构建 | vLLM `24420b7a`、vLLM-Ascend `89eb8a5e`、flash_attn_npu 0.5.0、FLA 26.9.1、triton-ascend 3.6.0.post2 | git clone / Ascend 索引 / PyPI sdist，补丁在 `src/patches/` |
| 4 运行时补丁 | 启动链上的约 10 个 `src/runtime/apply_*.py`（GDN、conv+QKV、varlen、joint fused、chunk128、systemone、Tier 2 等） | `start.sh` 启动时按 SHA-256 门控安装 |

### 2.3 默认值与可选项

| 变量 | 默认 | 说明 |
|---|---|---|
| `CLEF_VENV` | `<repo>/.venv` | venv 路径 |
| `CLEF_MATERIALS` | `<repo>/.build/materials` | 已校验材料目录（可选缓存） |
| `PIP_INDEX_URL` | 华为云 PyPI | 换镜像用 |
| `CLEF_PYTHON_URL` | NJU → GitHub | CPython 3.12.16 包 |
| `CLEF_GIT_MIRROR` | `https://gh-proxy.com` | vLLM clone 代理前缀 |
| `MAX_JOBS` | 8 | 编译并发 |

### 2.4 产物位置

- `<repo>/.venv/`（gitignore）
- `<repo>/.build/src/{vllm,vllm-ascend,flash_attn_npu-0.5.0}`
- `<repo>/.build/materials/`（CPython 包、flash sdist）
- `<repo>/.build/logs/{vllm-build.log,vllm-ascend-build.log,flash-attn-build.log,pip-check.txt,freeze-*.txt,versions.txt}`
- 完成标记 `<repo>/.venv/.clef-env-complete`

## 3. 当前阻塞：`vllm_ascend_C` 编译失败

### 3.1 现象

在 `/workspace/final-test/Clef-27B-ascend`（容器 `root@6bd7712f9124`）执行：

```
cmake --build . -j=8 --target=vllm_ascend_C
→ subprocess.CalledProcessError: exit status 2
→ ERROR: Failed building editable for vllm_ascend
```

用户贴出的只是 pip 的最后一行，**真正的 `error:` 文本尚未取得**。`ef46a9d` 之后重跑会把日志写到 `.build/logs/vllm-ascend-build.log` 并打印关键错误。

### 3.2 复现/取错误（服务器上）

```bash
cd /workspace/final-test/Clef-27B-ascend
git pull --ff-only origin main          # 确保含 ef46a9d
source scripts/activate_env.sh
cd .build/src/vllm-ascend
SOC_VERSION=ascend910_9382 COMPILE_CUSTOM_KERNELS=1 MAX_JOBS=8 \
  python -m pip install --no-build-isolation --no-deps -v -e . 2>&1 | tee /tmp/va-build.log
grep -nE "error:|fatal error|Killed|out of memory|Error [0-9]+" /tmp/va-build.log | head -40
```

同时收集：

```bash
echo "ASCEND_HOME_PATH=$ASCEND_HOME_PATH"
ls -d /usr/local/Ascend/* 2>/dev/null
npu-smi info -l | head -20
nproc; free -g; cat /sys/fs/cgroup/memory.max
command -v cmake gcc g++ ccec bishengir-compile
```

### 3.3 已知可能原因（按概率）

1. **`ASCEND_HOME_PATH` 指向不存在的路径**。`vllm-ascend` 的 `setup.py` 默认 `/usr/local/Ascend/ascend-toolkit/latest`；容器里 CANN 可能在 `/usr/local/Ascend/cann*`。`setup_env.sh` 现在会自动尝试 `/usr/local/Ascend/ascend-toolkit/latest`、`/usr/local/Ascend/cann`、`/usr/local/Ascend/cann-9.2.1`，无效则直接报错。
2. **内存不足**：`-j=8` 编译大模板时 `Killed`。用 `MAX_JOBS=4` 重试。
3. **CANN 编译器不在 PATH**：`ccec` / `bishengir-compile` 缺失，通常是 `scripts/activate_env.sh` 没加载到真实 `set_env.sh`。
4. 缺系统库：GCC 11.4 / clang-15、`libprotobuf-dev`、`numactl`、`libibverbs-dev`、`libhiredis-dev`、`pkg-config`、`libglib2.0-0`（见 `docs/installation-2026-11-12.md`）。

### 3.4 时间预期

- 纯 pip 层：3–10 分钟
- vLLM empty 构建：1–3 分钟
- vLLM-Ascend 自定义算子：20–60 分钟
- flash_attn_npu：10–30 分钟

慢是 Ascend C 编译，不是下载；不是脚本 bug。

## 4. 更关键的问题：评测镜像

用户信息：**评测机预计使用 `quay.io/ascend/vllm-ascend:v0.30.0-a3`，且不会存在 `/workspace/yangchen/venvs`。**

### 4.1 兼容性结论（重要）

本仓库的运行时补丁全部按文件 SHA-256 门控，钉的是：

- vLLM `24420b7a341af37a9bc3186e6fc6062512e0aa58`
- vLLM-Ascend `89eb8a5e7a76d085b453a1aeadb6883e12d5ea2d`

在 `v0.30.0-a3` 镜像上，这些文件的版本不同，启动链上的 `apply_*.py` 会 **fail-closed**，`start.sh` 无法启动。**这不是编译慢的问题，是版本不匹配。**

### 4.2 探针

`scripts/probe_runtime_image.py`（stdlib）在镜像内运行：

```bash
git clone <repo> /tmp/clef && cd /tmp/clef
python3 scripts/probe_runtime_image.py
```

输出：Python 版本、vllm/vllm-ascend/torch/torch-npu/triton 版本、安装路径、我们补丁涉及的 9 个文件的 SHA-256[:16]、是否命中已知 preimage/postimage、`ClefFlashForJointSchema` 是否存在，最后给 `MATCHES_THE_PINNED_BUILD` 或差异数。

已知哈希（用于人工核对）：

- joint schema：pre `197863d5`、f810 `f8101b98`、fused `26d3ad04`
- api_router：pre `bdf11bee`、post `1875f0e1`
- model_runner：pre `e866aeb2`、post `a772213f`
- gdn.py：pre `c8c0d4d8`、qkv `ead25cb2`、varlen `afe6d4e8`

### 4.3 三条路线

| 探针结果 | 路线 | 代价 |
|---|---|---|
| 全部匹配 | 直接 `bash start.sh` | 无 |
| 版本接近/部分匹配 | 把 `src/patches` 和启动链上的安装器移植到镜像版本，重生成哈希 | 大：移植 + 全量 OD/ERQA 重测 |
| 完全不匹配（最可能） | A. 自带镜像：把现有构建打成 image；B. wheelhouse：把 vllm/vLLM-Ascend/flash 的 wheel 放进仓库或材料目录，`start.sh` 首次运行装入镜像 | A 需镜像仓库；B 需镜像可写且 Python/CANN 一致 |

### 4.4 需要向主办方确认

规则文档（`docs/submission-rules-2026-11-13.md`）里 `image` 只是 `results.csv` 要填的字段，示例才是 `v0.30.0-a3`，**没有写死评测镜像**。必须确认：

> 评测是“固定镜像，只收代码”，还是“可以自带镜像 / 自己准备环境”？

这决定后面所有工作方向。

## 5. 已完成验证

- `setup_env.sh` 在 Linux aarch64 语法/参数/平台守卫正常（macOS 上只做了 `bash -n` 与 `--help`）。
- 国内镜像实测：
  - NJU CPython 包：53,264,815 字节，SHA-256 `4792f817…` 与固定值一致，约 9 秒。
  - NJU torch wheel：147 MB，SHA-256 `1794517…` 与固定值一致，约 25 秒。
  - 华为 PyPI `torch_npu-2.10.0.post7-cp312-aarch64`：sha256 `2577ea44…` 与固定值一致。
  - 华为 Ascend 索引有 `triton_ascend-3.6.0.post2-cp312-aarch64` 与 `flash_linear_attention_npu_a3-26.9.1-py3-none-aarch64`。
  - `flash-attn-npu==0.5.0` 在 PyPI 有 sdist（1,331,476 字节）。
- 单元测试：9 个文件 176 tests OK（2 个 host-data skip）。
- 探针脚本自测通过（macOS 上输出 MISSING）。

## 6. 不变量与禁止事项

- **不要修改 11 个字节冻结文件**（清单与 SHA 见 `docs/repo-restructure-plan-2026-11-25.md` 6.4）：`apply_joint_fused_attention.py`、`apply_joint_single_field_patch_fused.py`（其自身 SHA `fdc9dde6…` 被 `inference.py` 硬校验）、两个 `clef_joint_schema.py`、`apply_joint_service_check.py`、`apply_tiered_cache_patch.py`（`from runtime.tiered_prefix_cache.engine_hooks import install` 不能改）、`gdn_decision_flags.py`、`qkv_norm/qkv_device_op.py`、`qkv_norm/qkv_gdn.py`、`systemone_mm_state.py`、`systemone-mm-state.patch`。
- 量化被赛方禁止；只能 BF16。
- `--config s2` 已删除，只有 `s1`（默认）与 `custom`。
- `.venv/`、`.build/` 已 gitignore，不进提交。
- 不要 `git add -A`；`README.md` 是用户 WIP，不提交。

## 7. 其他已知 TODO / 风险

1. `scripts/activate_env.sh` 仍默认 `CLEF_WORKSPACE=/workspace/yangchen`（模型/缓存/logs 默认值）；在 final-test 容器上会在 `/workspace/yangchen` 下建目录。要彻底项目内化，可把缓存默认改到 `<repo>/.build/cache`，模型仍由 `--model-dir` 指定。
2. `requirements.txt` 从未用于完整全新构建（它是 11-12 freeze 的产物），本次是第一次真实验证，可能暴露遗漏包。
3. `triton==3.6.0`（PyPI）与 `triton_ascend` 都提供 `triton/` 包，安装顺序可能互相覆盖；`--pip-only` 后要检查 `triton.__file__` 是否指向 Ascend 的那份。
4. vLLM/vLLM-Ascend 的 git clone 走 gh-proxy/ghproxy；子模块 URL 通过 `url.<proxy>https://github.com/.insteadOf` 改写。若代理不可用会回退直连。
5. `flash_attn_npu` 从 PyPI 拉 sdist 时用了 `--no-build-isolation`，依赖环境里已有 torch；如果未来镜像里没有 torch，这条会失败。

## 8. 建议的下一步顺序

1. 在 `v0.30.0-a3` 镜像里跑 `scripts/probe_runtime_image.py`，把输出发回。
2. 向主办方确认镜像归属（固定 vs 自带）。
3. 若镜像固定：根据探针结果决定“移植补丁”还是“wheelhouse 安装”；移植需要重跑 OD/ERQA 全量质量与 S1/S2 性能。
4. 若允许自带镜像：在现有 A3 机器上把 `.venv` + 源码树打成镜像，`results.csv` 的 `image` 填自己的 tag。
5. 无论哪条路，先把 `/workspace/final-test` 上 `vllm_ascend_C` 的真实编译错误取出来；`ef46a9d` 之后的脚本会自动打印。

## 9. 关键路径与命令速查

```bash
# 服务器
cd /workspace/final-test/Clef-27B-ascend
git pull --ff-only origin main

# 只验证 pip / requirements.txt
bash scripts/setup_env.sh --pip-only

# 全量（含编译）
bash scripts/setup_env.sh

# 启动
source scripts/activate_env.sh
bash start.sh --model-dir /workspace/final-test/models/Clef-27B --output-dir /tmp/clef-out

# 镜像探针
python3 scripts/probe_runtime_image.py
```

## 10. 参考文档

- `docs/installation-2026-11-12.md`：首次完整 A3 安装与验收记录（含所有原始命令、材料 SHA、日志名）。
- `docs/repo-restructure-plan-2026-11-25.md`：目录结构与冻结文件不变量。
- `requirements.txt` 顶部注释：四层安装顺序。
- `AGENTS.md`：固定软件组合、资源限制、提交与沟通约定。
