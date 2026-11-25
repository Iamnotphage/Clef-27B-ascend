# Clef-27B 仓库扁平化重构执行计划（2026-11-25）

> **执行对象**：Worker（`deepseek-v4.1-flash`）  
> **目标**：拆除 `Clef-27B/` 嵌套目录与根目录 `runtime/` / `requirements/` 分散结构，将比赛交付物拉平至仓库根目录、全部补丁/运行时 hooks/Ascend C 自定义算子收口至 `./src/`、解除 `connect.sh` 追踪并加入 `.gitignore`、修复 `inference.py` 中残留的硬编码路径以支持“直接 `git clone` 裸启动”，并补齐根目录 `requirements.txt`。  
> **明确排期在后（本轮不重写）**：`README.md`、`AGENTS.md`、`optimization.md` 的内容重写、A3 真机评测执行与 `result.png` / `npu_device_call.png` 手动截图均由用户在重构验收后手工完成。

---

## 0. 硬性不变量与禁止事项（必读）

执行全过程必须严格遵守以下红线，违背任何一条均视为失败：

1. **禁止修改任何字节冻结的补丁安装器或 postimage 文件内容**（仅允许 `git mv` 移动路径，字节 SHA-256 必须与第 6.4 节逐字一致）：
   - `apply_joint_fused_attention.py`
   - `apply_joint_single_field_patch_fused.py`（`inference.py` 运行时硬校验其自身 SHA-256 `fdc9dde6...`）
   - `joint_fused_attention/clef_joint_schema.py`
   - `joint_single_field/clef_joint_schema.py`
   - `apply_joint_service_check.py`
   - `apply_tiered_cache_patch.py`（**绝不能改 `ENGINE_BLOCK` 中的 `from runtime.tiered_prefix_cache.engine_hooks import install`**！通过将 `CLEF_TIER2_REPO` 设为 `<REPO_ROOT>/src` 即可零改动命中 `src/runtime/tiered_prefix_cache/engine_hooks.py`，从而保住 `TIERED_MODEL_RUNNER_POSTIMAGE = a772213f...` 100% 有效！）
   - `gdn_decision_flags.py`
   - `qkv_norm/qkv_device_op.py`、`qkv_norm/qkv_gdn.py`
   - `systemone_mm_state.py`、`systemone-mm-state.patch`
2. **禁止删除本地磁盘上的 `connect.sh`**：必须使用 `git rm --cached connect.sh`，严禁 `git rm connect.sh` 或 `rm connect.sh`。
3. **禁止重写 `README.md`、`AGENTS.md`、`optimization.md` 正文**：本轮只做文件迁位（`Clef-27B/optimization.md -> optimization.md`、`Clef-27B/README.md -> docs/archive/clef-27b-readme-pre-refactor.md` 备份保留），不改写其业务正文。
4. **禁止丢失 `Clef-27B/README.md` 中的 2026-11-25 最新指标**：原仓库存在根目录 `README.md`（约 11-18）和 `Clef-27B/README.md`（11-25 最新 Tier 2 验收数据）两份文档，严禁用其中一个覆盖删除另一个，必须保留到 `docs/archive/clef-27b-readme-pre-refactor.md` 供用户后续合并重写。
5. **禁止使用 `git add -A` 或 `git add .` 盲目暂存**；禁止 `git push` 或修改远程分支；重构在独立分支 `refactor/flatten-root-20261125` 上完成并提交，由用户审查后合入 `main`。
6. **禁止在 `requirements.txt` 中写入任何本机路径**（如 `file:///root/selfgz...`、`file:///usr/local/Ascend/...`、`file:///workspace/yangchen/...`）。
7. **禁止全局批量 `sed` 替换 `Clef-27B`**：模型权重目录名（如 `/workspace/yangchen/models/Clef-27B`）、模型标题字符串（`"Clef-27B"`）、仓库名（`Clef-27B-ascend`）均须原样保留，只修改第 4 节明确列出的代码路径。

---

## 1. 目标目录结构

重构完成后，仓库根目录结构必须精确对齐以下清单（根目录无 `Clef-27B/`、无 `runtime/`、无 `requirements/`、无多余散落 `.py`）：

```text
Clef-27B-ascend/
├── .gitattributes
├── .gitignore
├── AGENTS.md                       # 保留原位置（后续由用户重写）
├── README.md                       # 保留原位置（后续由用户重写；原 Clef-27B/README.md 归档到 docs/archive/）
├── optimization.md                 # <- Clef-27B/optimization.md（后续由用户重写）
├── start.sh                        # <- Clef-27B/start.sh（必选统一启动入口）
├── inference.py                    # <- Clef-27B/inference.py（必选推理启动与补丁编排脚本）
├── results.csv                     # <- Clef-27B/results.csv（必选性能汇总）
├── metrics.csv                     # <- Clef-27B/metrics.csv（必选精度汇总）
├── result.png                      # <- Clef-27B/result.png（占位，后续由用户手动截图替换）
├── npu_device_call.png             # <- Clef-27B/npu_device_call.png（占位，后续由用户手动截图替换）
├── requirements.txt                # 新增：完整可复现纯 PyPI 依赖锁 + A3 特殊 wheel/源码构建说明
├── imgs/                           # 新增：放 .gitkeep，供 README / optimization 配图使用
├── docs/                           # 保留，新增 docs/archive/clef-27b-readme-pre-refactor.md
├── logs/                           # <- Clef-27B/logs/（含 logs/archive/results-v1-legacy.csv）
├── scripts/                        # 原 scripts/（移除 scripts/patches/，新增 scripts/requirements/）
│   ├── requirements/               # <- 原根目录 requirements/（6 个分阶段依赖/约束文件）
│   ├── official_eval/
│   └── ...
└── src/                            # 所有框架补丁、运行时优化 hooks、Ascend C 自定义融合算子
    ├── ascendc/
    │   ├── vendors/                # <- Clef-27B/src/ascendc/vendors/{clef_conv_qkv,clef_conv_qkv_varlen,customize}
    │   └── kernels/                # <- runtime/ascendc/{clef_gdn_conv_qkv,clef_gdn_conv_qkv_varlen,clef_gdn_qkv_norm,clef_gdn_qkv_generic}
    ├── patches/                    # <- scripts/patches/*.patch + runtime/clef-api-routes.patch + runtime/gdn-bsnd.patch
    └── runtime/                    # <- runtime/ 其余全部内容 + Clef-27B/clef_encoder_cache_scheduler.py
```

### 为什么把 `runtime/` 放在 `src/runtime/` 且保持 `CLEF_TIER2_REPO=<repo>/src`？

- `vllm_ascend/worker/v2/model_runner.py` 的哈希门控 postimage SHA-256（`a772213f...`）内嵌了 `from runtime.tiered_prefix_cache.engine_hooks import install`。
- 仓库本地并不存放上游 `model_runner.py` 原文件，修改该行会导致 `TIERED_MODEL_RUNNER_POSTIMAGE` 无法在本地离线重算且破坏现有 A3 环境已打补丁的源码树。
- 将 `runtime/` 整体移至 `src/runtime/` 并在 `inference.py` 中把 `CLEF_TIER2_REPO` 改为 `str(REPO_ROOT / "src")`，可令 `sys.path.insert(0, "<repo>/src")` 后 `from runtime.tiered_prefix_cache.engine_hooks import install` 零改动生效、所有 `tiered_prefix_cache` 内部 import 零改动、所有 `apply_joint_*.py` 字节 SHA-256 零变动。

---

## 2. Phase 0：执行前 Git 状态检查与分支隔离

按顺序执行：

```bash
# 1. 确认当前在 main 分支且查看未提交变更
git status --short
git branch --show-current
```

> 当前本地工作区已知存在 3 个未提交修改文件（`README.md`、`connect.sh`、`docs/profiling-2026-11-12.md`）与 1 个未追踪目录（`.vscode/`）。处理规则如下：
> - **`connect.sh`**：绝不丢弃、绝不 stash、绝不删除，留在磁盘上，Phase 1 通过 `git rm --cached connect.sh` 解除追踪。
> - **`README.md` 与 `docs/profiling-2026-11-12.md`**：如果 `git status` 显示这两个文件有未提交改动，先将这两个文件单独 stash，避免混入重构提交：

```bash
# 2. 仅 stash 与本轮重构无关的两个文档改动（不要包含 connect.sh！）
git stash push -m "pre-refactor-wip-docs" -- README.md docs/profiling-2026-11-12.md

# 3. 从当前 main 创建重构专用分支
git checkout -b refactor/flatten-root-20261125

# 4. 清理 __pycache__，防止未追踪字节码阻塞 git mv 或残留空目录
find Clef-27B runtime -type d -name '__pycache__' -exec rm -rf {} +
```

---

## 3. Phase 1：目录拉平与文件迁位（完整精确命令）

按顺序执行以下全部命令，不可调换顺序：

```bash
# 1. 准备目标父目录
mkdir -p src/ascendc src/patches docs/archive imgs

# 2. 迁入 Ascend C 预编译 vendor 与算子源码
git mv Clef-27B/src/ascendc/vendors src/ascendc/vendors
git mv runtime/ascendc src/ascendc/kernels

# 3. 统一收口构建期/前置源码补丁到 src/patches/
git mv scripts/patches/flash-attn-npu-max-jobs.patch src/patches/flash-attn-npu-max-jobs.patch
git mv scripts/patches/vllm-a3-numpy126.patch src/patches/vllm-a3-numpy126.patch
git mv scripts/patches/vllm-ascend-clef-tp-embedding.patch src/patches/vllm-ascend-clef-tp-embedding.patch
git mv scripts/patches/vllm-clef-load-audit.patch src/patches/vllm-clef-load-audit.patch
git mv runtime/clef-api-routes.patch src/patches/clef-api-routes.patch
git mv runtime/gdn-bsnd.patch src/patches/gdn-bsnd.patch

# 4. 迁入运行时安装器、Tier 2 引擎、调度器与单测到 src/runtime/
git mv runtime src/runtime
git mv Clef-27B/clef_encoder_cache_scheduler.py src/runtime/clef_encoder_cache_scheduler.py

# 5. 将 Clef-27B/ 交付核心文件拉平至仓库根目录
git mv Clef-27B/start.sh start.sh
git mv Clef-27B/inference.py inference.py
git mv Clef-27B/metrics.csv metrics.csv
git mv Clef-27B/results.csv results.csv
git mv Clef-27B/optimization.md optimization.md
git mv Clef-27B/result.png result.png
git mv Clef-27B/npu_device_call.png npu_device_call.png
git mv Clef-27B/logs logs

# 6. 归档 Clef-27B/ 历史副产物（不丢失任何内容）
git mv Clef-27B/results-v1-legacy.csv logs/archive/results-v1-legacy.csv
git mv Clef-27B/README.md docs/archive/clef-27b-readme-pre-refactor.md

# 7. 收拢原 requirements/ 到 scripts/requirements/，给根目录 requirements.txt 腾位
git mv requirements scripts/requirements

# 8. 确认 Clef-27B/ 与 scripts/patches/ 已完全清空后删除空目录
find Clef-27B -type f
rmdir Clef-27B/src/ascendc Clef-27B/src Clef-27B 2>/dev/null || true
rmdir scripts/patches 2>/dev/null || true

# 9. 初始化 imgs/ 目录占位
touch imgs/.gitkeep

# 10. 从 Git 索引移除 connect.sh（严禁加 -f 删除工作区文件！）
git rm --cached connect.sh
```

**Phase 1 立刻自检**：
- `ls -la connect.sh` 必须仍存在且大小非 0。
- `test ! -e Clef-27B && test ! -e runtime && test ! -e requirements` 必须返回 0。

---

## 4. Phase 2：精确代码与配置修改清单

逐文件按以下规范修改，**未列出的文件不得擅自修改**。

### 4.1 `.gitignore`

在文件末尾追加以下内容（保留原有 26 行不动）：

```gitignore

# Local SSH helper and editor/agent metadata (never commit to main).
connect.sh
.vscode/
.claude/
```

### 4.2 `.gitattributes`

将原内容替换为：

```gitattributes
# Unified-diff blank context lines contain their required leading space.
src/patches/*.patch whitespace=-blank-at-eol
src/runtime/*.patch whitespace=-blank-at-eol
```

### 4.3 `start.sh`

修改第 35 行（将 `../runtime/` 改为 `src/runtime/`）：

- **修改前**：
  ```bash
  if [[ -n "${CLEF_BIND_CPUS:-}" ]]; then
    exec "$python_bin" "$script_dir/../runtime/clef_bind_supervisor.py" \
      "$python_bin" "$script_dir/inference.py" "$@"
  fi
  ```
- **修改后**：
  ```bash
  if [[ -n "${CLEF_BIND_CPUS:-}" ]]; then
    exec "$python_bin" "$script_dir/src/runtime/clef_bind_supervisor.py" \
      "$python_bin" "$script_dir/inference.py" "$@"
  fi
  ```

确认 `start.sh` 保持可执行权限（`chmod +x start.sh`）。

### 4.4 `inference.py`（核心路径更新 + 裸 clone 动态 `--tree` 移植性修复）

严格按以下 6 处改动修改根目录 `inference.py`：

#### (a) 路径常量（原第 26–45 行）

- **修改前**：
  ```python
  REPO_ROOT = Path(__file__).resolve().parent.parent
  GDN_INSTALLER = REPO_ROOT / "runtime" / "apply_gdn_decision_patch.py"
  QKV_INSTALLER = REPO_ROOT / "runtime" / "apply_qkv_norm_patch.py"
  CONV_QKV_INSTALLER = REPO_ROOT / "runtime" / "apply_conv_qkv_patch.py"
  CONV_QKV_VARLEN_INSTALLER = REPO_ROOT / "runtime" / "apply_conv_qkv_varlen_patch.py"
  ...
  JOINT_INSTALLER = REPO_ROOT / "runtime" / "apply_joint_single_field_patch.py"
  JOINT_FUSED_INSTALLER = REPO_ROOT / "runtime" / "apply_joint_fused_attention.py"
  JOINT_FUSED_GUARD = REPO_ROOT / "runtime" / "apply_joint_single_field_patch_fused.py"
  JOINT_SERVICE_CHECK_INSTALLER = REPO_ROOT / "runtime" / "apply_joint_service_check.py"
  CHUNK128_INSTALLER = REPO_ROOT / "runtime" / "apply_gdn_chunk128_patch.py"
  ADAPTIVE_CHUNK_INSTALLER = REPO_ROOT / "runtime" / "apply_gdn_adaptive_chunk_patch.py"
  SYSTEMONE_MM_INSTALLER = (
      REPO_ROOT / "runtime" / "apply_systemone_mm_state_patch.py")
  SYSTEMONE_MM_MODULE = REPO_ROOT / "runtime" / "systemone_mm_state.py"
  SYSTEMONE_MM_MODE_ENV = "CLEF_STATE_MM_MODE"
  TIERED_INSTALLER = REPO_ROOT / "runtime" / "apply_tiered_cache_patch.py"
  ```
- **修改后**：
  ```python
  REPO_ROOT = Path(__file__).resolve().parent
  RUNTIME_DIR = REPO_ROOT / "src" / "runtime"
  GDN_INSTALLER = RUNTIME_DIR / "apply_gdn_decision_patch.py"
  QKV_INSTALLER = RUNTIME_DIR / "apply_qkv_norm_patch.py"
  CONV_QKV_INSTALLER = RUNTIME_DIR / "apply_conv_qkv_patch.py"
  CONV_QKV_VARLEN_INSTALLER = RUNTIME_DIR / "apply_conv_qkv_varlen_patch.py"
  ...
  JOINT_INSTALLER = RUNTIME_DIR / "apply_joint_single_field_patch.py"
  JOINT_FUSED_INSTALLER = RUNTIME_DIR / "apply_joint_fused_attention.py"
  JOINT_FUSED_GUARD = RUNTIME_DIR / "apply_joint_single_field_patch_fused.py"
  JOINT_SERVICE_CHECK_INSTALLER = RUNTIME_DIR / "apply_joint_service_check.py"
  CHUNK128_INSTALLER = RUNTIME_DIR / "apply_gdn_chunk128_patch.py"
  ADAPTIVE_CHUNK_INSTALLER = RUNTIME_DIR / "apply_gdn_adaptive_chunk_patch.py"
  SYSTEMONE_MM_INSTALLER = RUNTIME_DIR / "apply_systemone_mm_state_patch.py"
  SYSTEMONE_MM_MODULE = RUNTIME_DIR / "systemone_mm_state.py"
  SYSTEMONE_MM_MODE_ENV = "CLEF_STATE_MM_MODE"
  TIERED_INSTALLER = RUNTIME_DIR / "apply_tiered_cache_patch.py"
  ```

#### (b) 新增辅助函数 `resolve_package_tree`（放在 `resolve_vllm_tree` 函数之后）

```python
def resolve_package_tree(package_name):
    """Return the parent directory of an installed package, or None if unavailable."""
    spec = importlib.util.find_spec(package_name)
    init = Path(spec.origin).resolve() if spec and spec.origin else None
    if init is None or init.name != "__init__.py" or init.parent.name != package_name:
        return None
    return init.parent.parent
```

#### (c) 修复 4 个安装器调用的硬编码 `/workspace/yangchen/...` 默认树问题

以下 4 个函数增加可选关键字参数 `output_dir=None`（保持单测向后兼容），在构造 `argv` 时自动附带 `--tree`（如果 `resolve_package_tree(...)` 非 `None`）与 `--backup-dir`（如果 `output_dir` 非 `None`），并**始终把 `flag`（`--check` / `--apply`）放在 `argv` 末尾**（否则会破坏 `test_gdn_decision_patch.py` 第 657 行 `call.args[0][-1]` 断言）：

1. **`gdn_decision_prefill(config, apply=True, output_dir=None)`**：
   ```python
   def run(flag):
       argv = [sys.executable, str(GDN_INSTALLER)]
       tree = resolve_package_tree("vllm_ascend")
       if tree is not None:
           argv += ["--tree", str(tree)]
       if output_dir is not None:
           argv += ["--backup-dir",
                    str(Path(output_dir).expanduser().resolve() / "patch-backup" / "gdn-decision")]
       argv.append(flag)
       result = subprocess.run(argv, capture_output=True, text=True, check=False)
       if result.returncode != 0:
           raise SystemExit(
               f"GDN decision patch {flag} failed:\n"
               f"{result.stdout[-2000:]}{result.stderr[-2000:]}")
       return json.loads(result.stdout)
   ```
2. **`qkv_norm_patch(config, apply=True, output_dir=None)`**：
   ```python
   def run(flag):
       argv = [sys.executable, str(QKV_INSTALLER)]
       tree = resolve_package_tree("vllm_ascend")
       if tree is not None:
           argv += ["--tree", str(tree)]
       if output_dir is not None:
           argv += ["--backup-dir",
                    str(Path(output_dir).expanduser().resolve() / "patch-backup" / "qkv-norm")]
       argv.append(flag)
       result = subprocess.run(argv, capture_output=True, text=True, check=False)
       ...
   ```
3. **`gdn_chunk128_patch(config, apply=True, decision_enabled=False, output_dir=None)`**：
   ```python
   def run(flag):
       argv = [sys.executable, str(CHUNK128_INSTALLER)]
       tree = resolve_package_tree("vllm_ascend")
       if tree is not None:
           argv += ["--tree", str(tree)]
       if output_dir is not None:
           argv += ["--backup-dir",
                    str(Path(output_dir).expanduser().resolve() / "patch-backup" / "gdn-chunk128")]
       argv.append(flag)
       result = subprocess.run(argv, capture_output=True, text=True, check=False)
       ...
   ```
4. **`joint_single_field_patch(config, apply=True, output_dir=None)`**：
   ```python
   def run(flag):
       argv = [sys.executable, str(JOINT_INSTALLER)]
       tree = resolve_package_tree("vllm")
       if tree is not None:
           argv += ["--tree", str(tree)]
       if output_dir is not None:
           argv += ["--backup-dir",
                    str(Path(output_dir).expanduser().resolve() / "patch-backup" / "joint-single-field")]
       argv.append(flag)
       result = subprocess.run(argv, capture_output=True, text=True, check=False)
       ...
   ```
5. 在 `build_launch()` 中对应的 4 个调用点传入 `output_dir=output_dir`：
   - `gdn_decision = gdn_decision_prefill(config, apply=not getattr(args, "dry_run", False), output_dir=output_dir)`
   - `qkv_patch = qkv_norm_patch(config, apply=not getattr(args, "dry_run", False), output_dir=output_dir)`
   - `chunk128_patch = gdn_chunk128_patch(config, apply=not getattr(args, "dry_run", False), decision_enabled=gdn_decision["enabled"], output_dir=output_dir)`
   - `joint_patch = joint_single_field_patch(config, apply=not getattr(args, "dry_run", False), output_dir=output_dir)`

#### (d) 核对 `qkv_norm_environment`（原第 1261 行）

`vendor = Path(__file__).resolve().parent / "src" / "ascendc" / "vendors" / "customize"`  
因为 `inference.py` 已在根目录且 `src/ascendc/vendors/customize` 已在根目录，**该行天然正确，无需改动**（可核对确认）。

#### (e) 更新 `encoder_scheduler` 与 `CLEF_TIER2_REPO`

- 原第 1529 行：
  - **修改前**：`encoder_scheduler = Path(__file__).resolve().with_name("clef_encoder_cache_scheduler.py")`
  - **修改后**：`encoder_scheduler = RUNTIME_DIR / "clef_encoder_cache_scheduler.py"`
- 原第 1683 行：
  - **修改前**：`environment["CLEF_TIER2_REPO"] = str(REPO_ROOT)`
  - **修改后**：`environment["CLEF_TIER2_REPO"] = str(REPO_ROOT / "src")`

#### (f) 更新两处提示文本中的旧路径

- 原第 1246 行：`runtime/apply_schema_cache_patch.py` -> `src/runtime/apply_schema_cache_patch.py`
- 原第 1766 行：`runtime/apply_tiered_cache_patch.py` -> `src/runtime/apply_tiered_cache_patch.py`

---

### 4.5 单元测试路径同步（仅改路径，不弱化任何断言）

逐文件修改以下 5 个单测文件：

| 文件 | 原代码 | 修改后 |
| --- | --- | --- |
| `src/runtime/test_inference_launch.py` L17 | `ENTRY = Path(__file__).resolve().parent.parent / "Clef-27B" / "inference.py"` | `ENTRY = Path(__file__).resolve().parents[2] / "inference.py"` |
| `src/runtime/test_inference_launch.py` L47, L78 | `original = (ENTRY.parent.parent / "runtime/qkv_norm/qkv_gdn.py").read_bytes()` | `original = (Path(__file__).resolve().parent / "qkv_norm/qkv_gdn.py").read_bytes()` |
| `src/runtime/test_inference_launch.py` L307 | `str(ENTRY.parent) + os.pathsep + "/existing/path"` | `str(Path(__file__).resolve().parent) + os.pathsep + "/existing/path"` |
| `src/runtime/test_joint_fused_entry.py` L21-22 | `REPO = Path(__file__).resolve().parent.parent`<br>`ENTRY = REPO / "Clef-27B" / "inference.py"` | `REPO = Path(__file__).resolve().parents[2]`<br>`ENTRY = REPO / "inference.py"` |
| `src/runtime/test_joint_fused_entry.py` L27-28 | `F810 = REPO / "runtime" / "joint_single_field" / ...`<br>`AA3E = REPO / "runtime" / "joint_fused_attention" / ...` | `F810 = REPO / "src" / "runtime" / "joint_single_field" / ...`<br>`AA3E = REPO / "src" / "runtime" / "joint_fused_attention" / ...` |
| `src/runtime/test_joint_fused_entry.py` L239-252 | `REPO / "runtime/apply_joint_..."` 等 9 处路径 | `REPO / "src/runtime/apply_joint_..."`（**4 个 SHA-256 常量一字不改！**） |
| `src/runtime/test_gdn_decision_patch.py` L37 | `launcher = load("clef_inference_test", HERE.parent / "Clef-27B" / "inference.py")` | `launcher = load("clef_inference_test", HERE.parents[1] / "inference.py")` |
| `src/runtime/test_gdn_decision_patch.py` L365 | `evidence = HERE.parent / "Clef-27B/logs/gdn-decision-20261115"` | `evidence = HERE.parents[1] / "logs/gdn-decision-20261115"` |
| `src/runtime/test_encoder_cache_scheduler.py` L13 | `ENTRY = Path(__file__).resolve().parents[1] / "Clef-27B" / "clef_encoder_cache_scheduler.py"` | `ENTRY = Path(__file__).resolve().parent / "clef_encoder_cache_scheduler.py"` |
| `src/runtime/test_systemone_mm_state.py` L191 | `Path(__file__).resolve().parent.parent / "datasets" /` | `Path(__file__).resolve().parents[2] / "datasets" /` |

另外在 `src/runtime/test_inference_launch.py` 的 `LaunchTests` 中**补充 2 个回归单测**：
1. `test_tiered_repo_points_to_src_for_runtime_namespace`：验证 `self.launch("--config", "s1")["service_environment"]["CLEF_TIER2_REPO"] == str(ENTRY.parent / "src")` 且 `(Path(repo) / "runtime" / "tiered_prefix_cache" / "engine_hooks.py").is_file()` 为 `True`。
2. `test_gdn_decision_prefill_passes_resolved_tree_and_output_backup`：mock `resolve_package_tree` 返回临时目录 `self.root / "ascend_site"`，mock `subprocess.run`，验证调用 `launcher.gdn_decision_prefill("s1", apply=True, output_dir=self.root / "out")` 时命令中包含 `["--tree", str(self.root / "ascend_site"), "--backup-dir", str((self.root / "out").resolve() / "patch-backup" / "gdn-decision")]` 且末尾仍为 `"--check"`。

---

### 4.6 评测/验收/辅助脚本路径同步

逐文件更新以下脚本中的旧路径：

| 文件 | 修改内容 |
| --- | --- |
| `src/runtime/tiered_bare_acceptance.sh` | L24: `bash Clef-27B/start.sh` -> `bash start.sh`<br>L47: `python runtime/od_full_benchmark.py` -> `python src/runtime/od_full_benchmark.py`<br>L62: `python runtime/erqa_s2_concurrent.py` -> `python src/runtime/erqa_s2_concurrent.py` |
| `src/runtime/tiered_scenario_d.sh` | L26: `python Clef-27B/inference.py` -> `python inference.py`<br>L53: `python runtime/od_full_benchmark.py` -> `python src/runtime/od_full_benchmark.py`<br>L87: `python runtime/od_raw_acceptance.py` -> `python src/runtime/od_raw_acceptance.py` |
| `src/runtime/start_recommended.sh` | L16-17: `$SCRIPT_DIR/../Clef-27B/inference.py` -> `$SCRIPT_DIR/../../inference.py`（并同步注释） |
| `src/runtime/clef_conv_qkv_probe.py` | L35: `Path(__file__).resolve().parents[1] / "Clef-27B/src/ascendc/vendors/customize"` -> `Path(__file__).resolve().parents[2] / "src/ascendc/vendors/customize"` |
| `src/runtime/clef_qkv_generic_probe.py` | L32: `Path(__file__).resolve().parents[1] / "Clef-27B/src/ascendc/vendors/customize"` -> `Path(__file__).resolve().parents[2] / "src/ascendc/vendors/customize"` |
| `scripts/start_dataset_profile.sh` | L47: `$repo/runtime/jpeg_decode_cache` -> `$repo/src/runtime/jpeg_decode_cache`<br>L70: `$repo/runtime/k32_activation_cache` -> `$repo/src/runtime/k32_activation_cache`<br>L74: `$repo/Clef-27B/start.sh` -> `$repo/start.sh` |
| `scripts/analyze_input_distribution.py` | L186, L190, L194: `Path("Clef-27B/logs/...")` -> `Path("logs/...")` |
| `scripts/snapshot_systemone_profile.py` | L147: `Clef-27B/logs/model_manifest.json` -> `logs/model_manifest.json`；`scripts/patches` -> `src/patches` |
| `scripts/summarize_profile_memory.py` | L87: `"/scripts/patches/"` -> `"/src/patches/"` |

---

## 5. Phase 3：生成根目录 `requirements.txt`

在仓库根目录创建 `requirements.txt`，以 `docs/evidence/full-environment-freeze-2026-11-12.txt` 和 `scripts/requirements/*.txt` 为唯一事实来源，遵循以下规则：

1. **顶部注释块明确写出完整 A3 复现前置条件**（禁止误导读者以为一条 `pip install -r requirements.txt` 能绕过 CANN 或编译期补丁）：
   - OS / 硬件：Ubuntu 22.04 aarch64、Ascend A3（`Ascend910_9382`）、CANN 9.2.1 + NNAL ATB 9.2.1 + Driver 26.1.1、CPython 3.12.16（见 `scripts/bootstrap_python.sh`）。
   - Step 1：先安装平台专用 wheel（详见 `scripts/requirements/base-torch.txt`、`scripts/requirements/runtime-constraints.txt`）：
     - `torch==2.10.0+cpu`、`torchvision==0.25.0+cpu`、`torchaudio==2.10.0+cpu`（PyTorch CPU aarch64 wheel）
     - `torch-npu==2.10.0.post7`
     - `triton-ascend==3.6.0.post2`
     - `flash-linear-attention-npu-a3==26.9.1`
   - Step 2：安装本文件锁定的全部 PyPI 依赖：`pip install --constraint scripts/requirements/runtime-constraints.txt -r requirements.txt`。
   - Step 3：按固定提交构建并应用 `src/patches/` 基础补丁（`vllm@24420b7a341af37a9bc3186e6fc6062512e0aa58` 应用 `vllm-a3-numpy126.patch` + `vllm-clef-load-audit.patch` + `clef-api-routes.patch`；`vllm-ascend@89eb8a5e7a76d085b453a1aeadb6883e12d5ea2d` 应用 `vllm-ascend-clef-tp-embedding.patch` + `gdn-bsnd.patch`；`flash_attn_npu==0.5.0` 应用 `flash-attn-npu-max-jobs.patch`；以及 `TORCH_DEVICE_BACKEND_AUTOLOAD=0 mfcli kernel install --soc-version A3`）。
   - Step 4：启动时 `bash start.sh` 自动完成 `src/runtime/apply_*.py` 在线哈希校验与安装。
   - EvalScope 客户端依赖另见 `scripts/requirements/evalscope-api.txt`。
2. **包清单生成规则**（可用一段确定性 Python 脚本从 `docs/evidence/full-environment-freeze-2026-11-12.txt` 过滤生成，确保零手工抄错）：
   - 保留所有 `pkg==ver` 行（约 207 个纯 PyPI 固定版本包，含 `transformers==5.19.0`、`numpy==1.26.4`、`scipy==1.13.1`、`opencv-python-headless==4.11.0.86`、`fastapi==0.136.3`、`pydantic==2.14.3` 等）。
   - **过滤剔除**所有含 ` @ file://` 的行（剔除 `/root/selfgz*`、`/usr/local/Ascend/*` 的 CANN 宿主工具包，以及 `/workspace/yangchen/*` 本机绝对路径 wheel）与 `-e git+...` 行（这些在顶部注释块里以准确版本/commit 说明）。
   - **额外确认** `requirements.txt` 中无 `file://`、无 `/workspace/yangchen`、无 `/root/`。

参考生成命令（worker 可直接执行）：

```bash
python3 - <<'PY'
from pathlib import Path

src = Path("docs/evidence/full-environment-freeze-2026-11-12.txt").read_text(encoding="utf-8").splitlines()
pinned = [
    line.strip() for line in src
    if line.strip()
    and not line.startswith("#")
    and " @ file://" not in line
    and not line.startswith("-e ")
]
header = """# Clef-27B Ascend A3 固定 Python 运行依赖锁（纯 PyPI 传递依赖精确版本）
# 基准环境：Ubuntu 22.04.5 aarch64 | CPython 3.12.16 | CANN 9.2.1 | NNAL ATB 9.2.1 | Driver 26.1.1
#
# 完整复现安装顺序（先加载 /usr/local/Ascend/cann/set_env.sh 与 atb/set_env.sh）：
#   1. 平台专用 wheel（见 scripts/requirements/base-torch.txt 与 runtime-constraints.txt）：
#      - torch==2.10.0+cpu, torchvision==0.25.0+cpu, torchaudio==2.10.0+cpu
#      - torch-npu==2.10.0.post7
#      - triton-ascend==3.6.0.post2
#      - flash-linear-attention-npu-a3==26.9.1
#   2. 安装本文件锁定的全部 PyPI 依赖：
#      python -m pip install --constraint scripts/requirements/runtime-constraints.txt -r requirements.txt
#   3. 固定提交源码构建与前置补丁（补丁位于 src/patches/）：
#      - vllm @ 24420b7a341af37a9bc3186e6fc6062512e0aa58 (VLLM_TARGET_DEVICE=empty, -e .)
#        应用：src/patches/vllm-a3-numpy126.patch, vllm-clef-load-audit.patch, clef-api-routes.patch
#      - vllm-ascend @ 89eb8a5e7a76d085b453a1aeadb6883e12d5ea2d (SOC_VERSION=ascend910_9382 COMPILE_CUSTOM_KERNELS=1, -e .)
#        应用：src/patches/vllm-ascend-clef-tp-embedding.patch, gdn-bsnd.patch
#      - flash_attn_npu == 0.5.0（应用 src/patches/flash-attn-npu-max-jobs.patch，FLASH_ATTN_BUILD_NPU=910）
#      - TORCH_DEVICE_BACKEND_AUTOLOAD=0 mfcli kernel install --soc-version A3
#   4. 评测客户端 EvalScope 依赖见 scripts/requirements/evalscope-api.txt
#   5. CANN 9.2.1 自带工具包（asc_opc_tool / cann_ops_* / te / mspti 等）由宿主机 set_env.sh 注入，不在此安装。
#
# 来源快照：docs/evidence/full-environment-freeze-2026-11-12.txt
# （用户后续可在 A3 实机 clef-mix-ws-py312 中对照 pip freeze 复核更新）

"""
Path("requirements.txt").write_text(header + "\n".join(pinned) + "\n", encoding="utf-8")
print(f"Wrote requirements.txt with {len(pinned)} pinned PyPI packages.")
PY
```

---

## 6. Phase 4：强制验收门禁（全部通过后才允许提交）

Worker 必须逐项运行以下 6 道检查，并将完整结果贴入完工报告。**任何一项失败必须当场修复后复验**。

### 6.1 目录结构与 `connect.sh` 隔离检查

```bash
# 根目录不应再有 Clef-27B / runtime / requirements
test ! -e Clef-27B && test ! -e runtime && test ! -e requirements && echo "TOP_DIRS_CLEAN_OK"

# 必选交付物全部在根目录
for f in AGENTS.md README.md optimization.md start.sh inference.py results.csv metrics.csv result.png npu_device_call.png requirements.txt imgs/.gitkeep; do
  test -e "$f" || { echo "MISSING: $f"; exit 1; }
done && echo "REQUIRED_FILES_OK"

# connect.sh 本地文件完好、已从 git 索引移除且命中 .gitignore
test -f connect.sh && echo "CONNECT_LOCAL_EXISTS_OK"
git check-ignore -v connect.sh
test -z "$(git ls-files connect.sh)" && echo "CONNECT_UNTRACKED_OK"

# src/ 子结构完整
test -f src/ascendc/vendors/clef_conv_qkv/op_api/lib/libcust_opapi.so && \
test -f src/ascendc/vendors/clef_conv_qkv_varlen/op_api/lib/libcust_opapi.so && \
test -f src/ascendc/vendors/customize/op_api/lib/libcust_opapi.so && \
test -f src/runtime/clef_encoder_cache_scheduler.py && \
test -f src/runtime/tiered_prefix_cache/engine_hooks.py && \
test -f src/patches/clef-api-routes.patch && \
test -f src/patches/gdn-bsnd.patch && \
echo "SRC_LAYOUT_OK"
```

### 6.2 `start.sh --dry-run` 端到端启动参数检查

> 注意：在无 `vllm_ascend` 安装的本地 Mac 上运行 `--dry-run` 时需设置 `CLEF_GDN_DECISION_PREFILL=0`（避免 `gdn_decision_prefill` 去 `--check` 远端 A3 树）；在 A3 真机上则不设该变量直接运行。

```bash
CLEF_GDN_DECISION_PREFILL=0 bash start.sh --dry-run \
  --model-dir /tmp/clef-dry-model \
  --output-dir /tmp/clef-dry-out | python3 -c '
import json, sys
from pathlib import Path
data = json.load(sys.stdin)
repo = Path.cwd().resolve()
assert data["config"] == "s1", data["config"]
assert data["service_environment"]["CLEF_TIER2_REPO"] == str(repo / "src"), data["service_environment"]["CLEF_TIER2_REPO"]
assert data["service_environment"]["CLEF_ENCODER_CACHE_TOKENS"] == "262144"
assert str(repo / "src" / "runtime") in data["service_environment"]["PYTHONPATH"]
assert data["tiered_cache"]["installer"] == str(repo / "src" / "runtime" / "apply_tiered_cache_patch.py")
assert data["systemone_mm_state"]["installer"] == str(repo / "src" / "runtime" / "apply_systemone_mm_state_patch.py")
assert data["joint_patch"]["installer"] == str(repo / "src" / "runtime" / "apply_joint_fused_attention.py")
assert data["gdn_decision_prefill"]["conv_qkv_patch"]["installer"] == str(repo / "src" / "runtime" / "apply_conv_qkv_patch.py")
assert data["gdn_decision_prefill"]["conv_qkv_varlen_patch"]["installer"] == str(repo / "src" / "runtime" / "apply_conv_qkv_varlen_patch.py")
print("DRY_RUN_CONTRACT_OK")
'
```

### 6.3 离线 CPU 全量单元测试（预期 0 failure / 0 error）

```bash
python3 -m unittest \
  src/runtime/test_inference_launch.py \
  src/runtime/test_joint_fused_entry.py \
  src/runtime/test_encoder_cache_scheduler.py \
  src/runtime/test_conv_qkv_patch.py \
  src/runtime/test_conv_qkv_varlen_patch.py \
  src/runtime/test_qkv_generic_patch.py \
  src/runtime/test_systemone_mm_state.py \
  src/runtime/test_gdn_decision_patch.py \
  src/runtime/test_gdn_data_path_driver.py

PYTHONPATH=src python3 -m unittest runtime.tiered_prefix_cache.test_tiered_prefix_cache

python3 -m unittest scripts/official_eval/test_erqa_state_list.py

# 验证 apply_tiered_cache_patch 构造器在 src/runtime 下仍可正常 import 并做 round-trip
python3 - <<'PY'
import hashlib, sys
sys.path.insert(0, "src/runtime")
import apply_tiered_cache_patch as inst
fixture = "class NPUModelRunner:\n    def execute_model(self, s):\n        return s\n"
inst.MODEL_RUNNER_PREIMAGE = hashlib.sha256(fixture.encode()).hexdigest()
post = inst.patch_model_runner(fixture)
assert inst.strip_model_runner(post) == fixture
assert "from runtime.tiered_prefix_cache.engine_hooks import install" in post
print("TIERED_INSTALLER_ROUNDTRIP_OK")
PY
```

### 6.4 字节冻结资产 SHA-256 完整性核对

以下命令必须全部输出 `OK`（与重构前逐字节一致）：

```bash
shasum -a 256 -c <<'EOF'
2a659d07154768a5e2a4a51001aff05aa966359bdf1deef3b6b1d2f115e7e7c5  src/runtime/apply_joint_fused_attention.py
fdc9dde620a7995faede928fd97edeb3a37cae5de55ca28eea5b17ef4fd6da05  src/runtime/apply_joint_single_field_patch_fused.py
26d3ad047c42f47a7fa22e00d6bac53a8a063ed010e753cb79a673660f60d7c4  src/runtime/joint_fused_attention/clef_joint_schema.py
f8101b98228bbd472a63a1efe6827583e35c6cc9dede4379b925f2b016670863  src/runtime/joint_single_field/clef_joint_schema.py
1ed35f0a24f8b23d8ac099992d703aca20d7f1460c89ff02c30635dfde6a219c  src/runtime/apply_joint_service_check.py
ebd34967c6113c91ef981633e218d90a54a99d14c90956ee8feda8e47f36a0f0  src/runtime/apply_tiered_cache_patch.py
3c9cb53a34a4ef4c2b88ce0e283a1343a9268c578aeab1c506645f3d79147287  src/runtime/gdn_decision_flags.py
19cf5d0f34015cc1e4f5f7639a084d3ad64ed24e0a7a1a3b1946b475eec14383  src/runtime/qkv_norm/qkv_device_op.py
c8c0d4d8a693658e075a2232b6938584760672bacf101bf50ca5fdfab2c0be7e  src/runtime/qkv_norm/qkv_gdn.py
a8b87841cc777a73dd682fd165739dc047ce2925a80879dbb3bf5dd60753e7f5  src/runtime/systemone_mm_state.py
f92cc769a1b5d6c0c951aaada1963d302f86de719af515f3aba36903dc7e16d0  src/runtime/systemone-mm-state.patch
EOF
```

### 6.5 残留路径与 `requirements.txt` 污染扫描

```bash
# 1. 代码与脚本中不应再残留 Clef-27B/ 目录路径（允许 MODEL_DIR 默认值 /models/Clef-27B 以及历史 build_*.py 注释）
git grep -n "Clef-27B/" -- start.sh inference.py scripts/ src/runtime/test_*.py src/runtime/tiered_bare_acceptance.sh src/runtime/tiered_scenario_d.sh src/runtime/start_recommended.sh src/runtime/clef_*_probe.py || echo "NO_STALE_CLEF27B_REFS"

# 2. requirements.txt 不得含任何 file:// 或本机绝对路径
grep -nE "file://|/workspace/|/root/" requirements.txt && exit 1 || echo "REQUIREMENTS_CLEAN_OK"
```

### 6.6 提交 + 模拟“比赛方裸 clone”本地干跑验收

```bash
# 1. 仅暂存本次重构涉及的文件，严禁 git add -A
git status --short
# 确认无 connect.sh、无 .vscode/、无 README.md / docs/profiling-2026-11-12.md 的无关修改
git add .gitignore .gitattributes start.sh inference.py requirements.txt imgs/.gitkeep src/ scripts/ docs/archive/ docs/repo-restructure-plan-2026-11-25.md logs/ metrics.csv results.csv optimization.md result.png npu_device_call.png
git commit -m "refactor: flatten Clef-27B root layout, consolidate src/, and untrack connect.sh"

# 2. 用 git clone --local 模拟比赛方全新 clone，验证零未追踪文件依赖
rm -rf /tmp/clef-clone-smoke
git clone --local . /tmp/clef-clone-smoke
(
  cd /tmp/clef-clone-smoke
  test ! -e connect.sh && test ! -e Clef-27B && test ! -e runtime && echo "CLONE_CLEAN_OK"
  CLEF_GDN_DECISION_PREFILL=0 bash start.sh --dry-run --model-dir /tmp/mock-model --output-dir /tmp/mock-out >/dev/null && echo "CLONE_DRY_RUN_OK"
  PYTHONPATH=src python3 -m unittest runtime.tiered_prefix_cache.test_tiered_prefix_cache >/dev/null && echo "CLONE_TIER2_TESTS_OK"
  python3 -m unittest src/runtime/test_inference_launch.py src/runtime/test_joint_fused_entry.py >/dev/null && echo "CLONE_LAUNCH_TESTS_OK"
)
rm -rf /tmp/clef-clone-smoke
```

> 完成后向用户汇报：`refactor/flatten-root-20261125` 分支 commit hash、上述 6 项门禁实测输出，以及提醒用户此前 stash 的 `pre-refactor-wip-docs`（如适用）。**严禁自行 `git push` 或合并 `main`**。

---

## 7. Phase 5（用户后续手动执行 — Worker 本轮不执行）

重构分支合入并推送后，由用户按以下顺序在 A3 真机上完成比赛模拟验收与收尾：

1. **模拟比赛方全新克隆与启动**：
   ```bash
   git clone <repo-url> /workspace/yangchen/clef-judge-test
   cd /workspace/yangchen/clef-judge-test
   source scripts/activate_env.sh   # 或激活目标 clef-mix-ws-py312 环境
   bash start.sh --model-dir /workspace/yangchen/models/Clef-27B --output-dir /tmp/judge-out
   ```
   - 如在**全新上游 `vllm 24420b7a` + `vllm-ascend 89eb8a5`** 环境中复现，启动前先应用 `src/patches/` 中的基础补丁（见 `requirements.txt` 顶部 Step 3）。
   - 如复用现有已配好的 A3 环境（`clef-mix-ws-py312`），`bash start.sh` 会自动探测到导入树并对全部在线补丁执行 idempotent SHA-256 校验，无需手工干预。
2. **核对四接口与 Tier 2 就绪**：
   - `curl -s http://127.0.0.1:18143/health` -> `status=ready`
   - `curl -s http://127.0.0.1:18143/v1/decide/info` -> `"prefix_state_cache": true`
3. **手动跑评测并手动截图**：
   - 跑 OpenDecisions + ERQA（可参考 `src/runtime/tiered_bare_acceptance.sh`）核对 `metrics.csv` / `results.csv`；
   - 评测运行期间手动截真实终端与 `npu-smi info`，替换根目录 `result.png` 与 `npu_device_call.png`；
   - （可选）在 A3 活跃 venv 中执行 `pip freeze` 与根目录 `requirements.txt` 做 diff，按需同步最新小版本。
4. **重写三份顶层文档**：
   - 重写根目录 `README.md`（参考 `docs/archive/clef-27b-readme-pre-refactor.md` 中的 11-25 最终指标与旧根目录 `README.md`，并在 `imgs/` 放入需要展示的 profile/架构图）；
   - 重写 `optimization.md`；
   - 重写 `AGENTS.md`。
5. **（可选）二次瘦身 `src/runtime/`**：
   - 本轮为了零风险保住全部历史脚本与单测引用，将 `runtime/` 整体收口到了 `src/runtime/`；如果后续希望提交包更极简，可再另开一轮清理 `src/runtime/` 中早期负结果探针（如 `probe_mlp_w8a8.py`、`round2-remote-only/`、`mc2_*_probe.py`、`k32_activation_cache/` 等）。
