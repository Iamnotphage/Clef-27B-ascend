#!/usr/bin/env bash
# 在仓库根目录构建项目内 .venv（A3 / Linux aarch64）。
#
#   bash scripts/setup_env.sh              # 完整安装（含 vLLM / vLLM-Ascend / flash_attn 源码构建）
#   bash scripts/setup_env.sh --pip-only   # 只建 .venv + 装 requirements.txt，验证 pip 锁
#   bash scripts/setup_env.sh --force      # 已完成也重新执行
#
# 可选覆盖（都有默认值，正常不需要设置）：
#   CLEF_MATERIALS       已校验材料目录   [<repo>/.build/materials，缺失时自动下载]
#   PIP_INDEX_URL        PyPI 镜像        [华为云镜像]
#   CLEF_VENV            venv 路径        [<repo>/.venv]
#
# 完成后：source scripts/activate_env.sh && bash start.sh
set -euo pipefail

repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
mode=full
force=0
for arg in "$@"; do
  case "$arg" in
    --pip-only) mode=pip ;;
    --force) force=1 ;;
    -h|--help) sed -n '2,13p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) printf '未知参数：%s\n' "$arg" >&2; exit 2 ;;
  esac
done

[[ "$(uname -sm)" == 'Linux aarch64' ]] || {
  printf 'setup_env.sh 只支持 Linux aarch64（当前 %s）。\n' "$(uname -sm)" >&2
  exit 1
}

venv="${CLEF_VENV:-$repo/.venv}"
materials="${CLEF_MATERIALS:-$repo/.build/materials}"
index="${PIP_INDEX_URL:-https://mirrors.huaweicloud.com/repository/pypi/simple}"
ascend_index="https://mirrors.huaweicloud.com/ascend/repos/pypi"
python_archive="${CLEF_PYTHON_ARCHIVE:-$materials/cpython-3.12.16+20261110-aarch64-unknown-linux-gnu-install_only.tar.gz}"
python_sha=4792f81750282597fb2a3389fe18b945ed2ba6aad5e45ad406d122c58cb33081
python_urls=(
  "https://mirror.nju.edu.cn/github-release/astral-sh/python-build-standalone/20261110/cpython-3.12.16%2B20261110-aarch64-unknown-linux-gnu-install_only.tar.gz"
  "https://github.com/astral-sh/python-build-standalone/releases/download/20261110/cpython-3.12.16%2B20261110-aarch64-unknown-linux-gnu-install_only.tar.gz"
)
[[ -n "${CLEF_PYTHON_URL:-}" ]] && python_urls=("$CLEF_PYTHON_URL" "${python_urls[@]}")
build="$repo/.build"
src="$build/src"
logs="$build/logs"
constraints="$repo/scripts/requirements/runtime-constraints.txt"
marker="$venv/.clef-env-complete"

vllm_commit=24420b7a341af37a9bc3186e6fc6062512e0aa58
ascend_commit=89eb8a5e7a76d085b453a1aeadb6883e12d5ea2d

step() { printf '\n=== %s ===\n' "$*"; }
die() { printf '错误：%s\n' "$*" >&2; exit 1; }
first_glob() { compgen -G "$1" | head -n1 || true; }

download_python() { # dest
  local dest="$1" url digest
  for url in "${python_urls[@]}"; do
    printf '  下载 %s\n' "$url"
    rm -f "$dest.part"
    if curl -fL --retry 2 --connect-timeout 20 -o "$dest.part" "$url"; then
      digest="$(sha256sum "$dest.part" | awk '{print $1}')"
      if [[ "$digest" == "$python_sha" ]]; then
        mv "$dest.part" "$dest"
        return 0
      fi
      printf '  SHA-256 不匹配（%s），换下一个源\n' "$digest"
    else
      printf '  下载失败，换下一个源\n'
    fi
    rm -f "$dest.part"
  done
  return 1
}

if [[ "$mode" == full && "$force" -eq 0 && -f "$marker" ]]; then
  printf '环境已完整：%s\n如需重建请加 --force；只验证 pip 请用 --pip-only。\n' "$marker"
  exit 0
fi

mkdir -p "$logs" "$materials"

if [[ ! -x "$venv/bin/python" ]]; then
  step "创建 $venv（CPython 3.12.16）"
  if [[ ! -f "$python_archive" ]]; then
    [[ -n "${CLEF_PYTHON_ARCHIVE:-}" ]] && die "CLEF_PYTHON_ARCHIVE 不存在：$python_archive"
    download_python "$python_archive" \
      || die "CPython 下载失败；请手动下载后放到 $python_archive（SHA-256 $python_sha）"
  fi
  CLEF_VENV="$venv" bash "$repo/scripts/bootstrap_python.sh" "$python_archive"
else
  step "复用已有 venv：$venv"
fi

export CLEF_VENV="$venv"
# shellcheck source=scripts/activate_env.sh
source "$repo/scripts/activate_env.sh"

step "基础校验"
python - <<'PY'
import sys
assert sys.version_info[:3] == (3, 12, 16), sys.version
print(sys.executable, sys.version.split()[0])
PY

step "1/4 构建工具与平台 wheel"
python -m pip install --index-url "$index" --constraint "$constraints" --timeout 60 \
  -r "$repo/scripts/requirements/build-tools.txt"

wheels="$materials/wheels"
if [[ -f "$wheels/torch-2.10.0+cpu-cp312-cp312-manylinux_2_28_aarch64.whl" ]]; then
  torch_inputs=("$wheels"/torch-*.whl "$wheels"/torchvision-*.whl
                "$wheels"/torchaudio-*.whl "$wheels"/torch_npu-*.whl)
else
  torch_inputs=(-r "$repo/scripts/requirements/base-torch.txt")
fi
python -m pip install --index-url "$index" --constraint "$constraints" --timeout 60 \
  "${torch_inputs[@]}" -r "$repo/scripts/requirements/cann-tools.txt"

triton_wheel="$(first_glob "$materials/triton_ascend-*.whl")"
if [[ -n "$triton_wheel" ]]; then
  python -m pip install --constraint "$constraints" "$triton_wheel" transformers==5.19.0
else
  python -m pip install --index-url "$index" --extra-index-url "$ascend_index" \
    --constraint "$constraints" --timeout 60 triton-ascend==3.6.0.post2 transformers==5.19.0
fi

step "2/4 纯 PyPI 依赖锁（requirements.txt）"
python -m pip install --index-url "$index" --constraint "$constraints" --timeout 60 \
  -r "$repo/requirements.txt"
python -m pip check | tee "$logs/pip-check.txt"
python -m pip freeze --all > "$logs/freeze-after-pip.txt"
python - "$logs/versions.txt" <<'PY'
import importlib.metadata as md
import sys
names = ("torch", "torch-npu", "torchvision", "torchaudio", "transformers",
         "numpy", "scipy", "triton", "triton-ascend",
         "flash-linear-attention-npu-a3")
lines = []
for name in names:
    try:
        lines.append(f"{name}=={md.version(name)}")
    except md.PackageNotFoundError:
        lines.append(f"{name}: MISSING")
text = "\n".join(lines)
print(text)
open(sys.argv[1], "w").write(text + "\n")
PY

if [[ "$mode" == pip ]]; then
  printf '\nPIP_ONLY_OK  （pip check / 版本见 %s）\n' "$logs"
  exit 0
fi

step "3/4 固定源码构建"
ensure_source() { # dir tarball url commit recursive(0|1)
  local dir="$1" tarball="$2" url="$3" commit="$4" recursive="$5"
  if [[ -d "$dir/.git" ]]; then
    local head
    head="$(git -C "$dir" rev-parse HEAD)"
    [[ "$head" == "$commit" ]] || die "$dir HEAD=$head，期望 $commit"
    if [[ "$recursive" == 1 ]]; then
      git -C "$dir" submodule update --init --recursive >/dev/null
    fi
    return
  fi
  mkdir -p "$src"
  if [[ -f "$tarball" ]]; then
    tar --no-same-owner -xzf "$tarball" -C "$src"
    return
  fi
  command -v git >/dev/null || die "缺少 $tarball 且未安装 git"
  rm -rf "$dir"
  if [[ "$recursive" == 1 ]]; then
    git clone --recursive "$url" "$dir"
  else
    git clone "$url" "$dir"
  fi
  git -C "$dir" checkout "$commit"
  if [[ "$recursive" == 1 ]]; then
    git -C "$dir" submodule update --init --recursive
  fi
}

apply_patch() { # dir patch
  local dir="$1" patch_file="$2"
  ( cd "$dir"
    if patch -p1 --dry-run --silent < "$patch_file" >/dev/null 2>&1; then
      patch -p1 --silent < "$patch_file"
      printf '  applied %s\n' "$(basename "$patch_file")"
    elif patch -p1 -R --dry-run --silent < "$patch_file" >/dev/null 2>&1; then
      printf '  already applied %s\n' "$(basename "$patch_file")"
    else
      printf '  patch 无法应用：%s\n' "$patch_file" >&2
      exit 1
    fi )
}

ensure_source "$src/vllm" "$materials/vllm-24420b7-with-git.tar.gz" \
  https://github.com/vllm-project/vllm.git "$vllm_commit" 0
apply_patch "$src/vllm" "$repo/src/patches/vllm-a3-numpy126.patch"
apply_patch "$src/vllm" "$repo/src/patches/vllm-clef-load-audit.patch"
apply_patch "$src/vllm" "$repo/src/patches/clef-api-routes.patch"
VLLM_TARGET_DEVICE=empty python -m pip install --no-build-isolation \
  --constraint "$constraints" --index-url "$index" --timeout 60 -e "$src/vllm"

ensure_source "$src/vllm-ascend" "$materials/vllm-ascend-89eb8a5-with-submodules.tar.gz" \
  https://github.com/vllm-project/vllm-ascend.git "$ascend_commit" 1
apply_patch "$src/vllm-ascend" "$repo/src/patches/vllm-ascend-clef-tp-embedding.patch"
apply_patch "$src/vllm-ascend" "$repo/src/patches/gdn-bsnd.patch"
SOC_VERSION=ascend910_9382 COMPILE_CUSTOM_KERNELS=1 MAX_JOBS="${MAX_JOBS:-8}" \
  python -m pip install --no-build-isolation --constraint "$constraints" \
  --index-url "$index" --timeout 60 \
  --extra-index-url "$ascend_index" \
  --extra-index-url https://mirrors.huaweicloud.com/ascend/repos/pypi/nightly \
  -e "$src/vllm-ascend"

step "MemFabric / FLA / flash_attn_npu"
command -v mfcli >/dev/null || die "mfcli 不在 PATH；检查 memfabric_hybrid 是否装进 $venv"
( cd "$repo" && TORCH_DEVICE_BACKEND_AUTOLOAD=0 mfcli kernel install --soc-version A3 )

fla_wheel="$(first_glob "$materials/flash_linear_attention_npu_a3-*.whl")"
if [[ -n "$fla_wheel" ]]; then
  python -m pip install --no-deps "$fla_wheel"
else
  python -m pip install --no-deps --index-url "$index" --extra-index-url "$ascend_index" \
    --timeout 60 flash-linear-attention-npu-a3==26.9.1
fi
python -m pip install --index-url "$index" --constraint "$constraints" --timeout 60 \
  expecttest==0.3.0

flash_dir="$src/flash_attn_npu-0.5.0"
if [[ ! -d "$flash_dir" ]]; then
  flash_tar="$(first_glob "$materials/flash_attn_npu-0.5.0.tar.gz")"
  if [[ -z "$flash_tar" ]]; then
    printf '  从 PyPI 获取 flash_attn_npu 0.5.0 源码包\n'
    python -m pip download --no-deps --no-binary :all: --no-build-isolation \
      --dest "$materials" --index-url "$index" --timeout 60 flash-attn-npu==0.5.0 \
      || die "无法下载 flash_attn_npu 0.5.0（把源码包放到 $materials）"
    flash_tar="$(first_glob "$materials"/flash*attn*npu-0.5.0.tar.gz)"
    [[ -n "$flash_tar" ]] || die "$materials 下未找到 flash_attn_npu 0.5.0 源码包"
  fi
  tar --no-same-owner -xzf "$flash_tar" -C "$src"
fi
apply_patch "$flash_dir" "$repo/src/patches/flash-attn-npu-max-jobs.patch"
FLASH_ATTN_BUILD_NPU=910 FLASH_ATTENTION_FORCE_BUILD=TRUE MAX_JOBS="${MAX_JOBS:-8}" \
  python -m pip install --no-build-isolation --constraint "$constraints" \
  --index-url "$index" --timeout 60 "$flash_dir"

step "4/4 验收"
python -m pip check | tee "$logs/pip-check.txt"
python -m pip freeze --all > "$logs/freeze-complete.txt"
python - "$logs/versions.txt" <<'PY'
import importlib.metadata as md
import sys
names = ("torch", "torch-npu", "torchvision", "torchaudio", "transformers",
         "numpy", "scipy", "triton", "triton-ascend", "flash-attn-npu",
         "flash-linear-attention-npu-a3", "vllm", "vllm-ascend")
lines = []
for name in names:
    try:
        lines.append(f"{name}=={md.version(name)}")
    except md.PackageNotFoundError:
        lines.append(f"{name}: MISSING")
text = "\n".join(lines)
print(text)
open(sys.argv[1], "w").write(text + "\n")
PY

smoke="${CLEF_SMOKE_OUTPUT:-$logs/smoke-$(date +%Y%m%d-%H%M%S).json}"
if [[ -d "${MODEL_DIR:-}" ]]; then
  python "$repo/scripts/verify_npu_environment.py" --stage runtime --require-extensions \
    --model-dir "$MODEL_DIR" --output "$smoke" \
    || die "NPU smoke 失败，见 $smoke"
else
  printf '跳过模型 smoke：MODEL_DIR 不存在（%s）。\n' "${MODEL_DIR:-未设置}"
fi

touch "$marker"
printf '\n环境完成：%s\n' "$venv"
printf '启动：source scripts/activate_env.sh && bash start.sh\n'
printf '日志与 freeze：%s\n' "$logs"
