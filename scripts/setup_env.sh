#!/usr/bin/env bash
# 在仓库根目录构建项目内 .venv（A3 / Linux aarch64）。
#
#   bash scripts/setup_env.sh              # 完整：服务 + 质量/性能评测依赖 + 源码构建
#   bash scripts/setup_env.sh --pip-only   # 只建 .venv 并安装全部 pip 依赖（不编译源码）
#   bash scripts/setup_env.sh --no-eval    # 完整服务环境，跳过 EvalScope 与评测依赖
#   bash scripts/setup_env.sh --force      # 已完成也重新执行
#
# 目录约定（可用环境变量覆盖）：
#   CLEF_VENV       虚拟环境          [<repo>/.venv]
#   CLEF_BUILD      日志/材料/缓存    [<repo>/.build]
#   CLEF_SRC_ROOT   源码树根          [<repo>/build]（必须非隐藏目录，见下）
#   CLEF_MATERIALS  已校验材料        [<repo>/.build/materials]
#   PIP_INDEX_URL   PyPI 镜像         [华为云 PyPI]
#   CLEF_GIT_MIRROR GitHub 代理前缀   [https://gh-proxy.com]
#   CLEF_FORCE_TARBALL=1 跳过 git，直接用固定提交 archive（SHA-256 校验）
#   MAX_JOBS        编译并发          [自动：min(配额 CPU 核, 内存 GiB/4, 16)]
#
# 为什么源码树放在 <repo>/build 而不是 <repo>/.build：
#   vllm-ascend 的 Ascend C 构建用 CANN 的 recompile_binary.py 在
#   `glob(..., recursive=True)` 下收集 host 对象文件；Python glob 会跳过
#   隐藏目录（以 . 开头），而 CMake 安装 OBJECT 库时保留源码绝对路径。
#   源码树若位于 .build/src，安装路径含 .build，glob 永远找不到 .o，
#   构建以 TooFewObj 失败。build/src 非隐藏，可稳定通过。
#
# 完成后：source scripts/activate_env.sh && bash start.sh
set -euo pipefail

repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
mode=full
force=0
eval_enabled=1
for arg in "$@"; do
  case "$arg" in
    --pip-only) mode=pip ;;
    --no-eval) eval_enabled=0 ;;
    --force) force=1 ;;
    -h|--help) sed -n '2,/^set -euo pipefail$/p' "${BASH_SOURCE[0]}" | sed '$d'; exit 0 ;;
    *) printf '未知参数：%s\n' "$arg" >&2; exit 2 ;;
  esac
done

[[ "$(uname -sm)" == 'Linux aarch64' ]] || {
  printf 'setup_env.sh 只支持 Linux aarch64（当前 %s）。\n' "$(uname -sm)" >&2
  exit 1
}

venv="${CLEF_VENV:-$repo/.venv}"
build="${CLEF_BUILD:-$repo/.build}"
export CLEF_WORKSPACE="${CLEF_WORKSPACE:-$build}"
src_root="${CLEF_SRC_ROOT:-$repo/build}"
src="$src_root/src"
materials="${CLEF_MATERIALS:-$build/materials}"
index="${PIP_INDEX_URL:-https://mirrors.huaweicloud.com/repository/pypi/simple}"
pypi_fallback="https://pypi.tuna.tsinghua.edu.cn/simple"
ascend_index="https://mirrors.huaweicloud.com/ascend/repos/pypi"
ascend_nightly="https://mirrors.huaweicloud.com/ascend/repos/pypi/nightly"
python_archive="${CLEF_PYTHON_ARCHIVE:-$materials/cpython-3.12.16+20261110-aarch64-unknown-linux-gnu-install_only.tar.gz}"
python_sha=4792f81750282597fb2a3389fe18b945ed2ba6aad5e45ad406d122c58cb33081
python_urls=(
  "https://mirror.nju.edu.cn/github-release/astral-sh/python-build-standalone/20261110/cpython-3.12.16%2B20261110-aarch64-unknown-linux-gnu-install_only.tar.gz"
  "https://github.com/astral-sh/python-build-standalone/releases/download/20261110/cpython-3.12.16%2B20261110-aarch64-unknown-linux-gnu-install_only.tar.gz"
)
[[ -n "${CLEF_PYTHON_URL:-}" ]] && python_urls=("$CLEF_PYTHON_URL" "${python_urls[@]}")
# vLLM / vLLM-Ascend / EvalScope clones: domestic GitHub proxies first, direct last.
git_mirrors=("${CLEF_GIT_MIRROR:-https://gh-proxy.com}" "https://ghproxy.net" "")
logs="$build/logs"
constraints="$repo/scripts/requirements/runtime-constraints.txt"
marker="$venv/.clef-env-complete"
marker_version="clef-env-v2"

vllm_commit=24420b7a341af37a9bc3186e6fc6062512e0aa58
ascend_commit=89eb8a5e7a76d085b453a1aeadb6883e12d5ea2d
evalscope_commit=3d39bab09c1d63095490a8b0f4a6dcbcc5ea843e
# 固定提交的 GitHub archive sha256（内容寻址；git 代理不可用时的快速回退）。
vllm_archive_sha=e933f065c5bed13a75e866ca642dd4257eb8e366a6d82b11b8a69cc0dca3e9e2
ascend_archive_sha=9b59a63495b586569dc64ed3b97e1e0fc3f9d3b4343606588ffde0bd2edad7e6
evalscope_archive_sha=61c596f4a2c2ad9164075cf54e51cbf0c0f6865fd1f2a208f363cce7fbdc2cc8
vllm_version_hint=0.33.0rc1.dev370+g24420b7a3
ascend_version_hint=0.1.dev1+g89eb8a5e7
# 网络到 GitHub git 智能协议极慢时，可强制走 archive 回退。
force_tarball="${CLEF_FORCE_TARBALL:-0}"
source_submodules=()

step() { printf '\n=== %s ===\n' "$*"; }
die() { printf '错误：%s\n' "$*" >&2; exit 1; }
have() { command -v "$1" >/dev/null 2>&1; }
first_glob() { compgen -G "$1" | head -n1 || true; }

# ---------------------------------------------------------------- 计时
script_start_ms=$(date +%s%3N)
timing_file="$logs/setup-timing.json"
timing_tsv="$logs/setup-timing.tsv"
stage_name=""
stage_start_ms=0
timing_python="/usr/bin/python3"
[[ -x "$venv/bin/python" ]] && timing_python="$venv/bin/python"

begin_stage() { stage_name="$1"; stage_start_ms=$(date +%s%3N); step "$1"; }
end_stage() {
  local status="${1:-ok}" ms
  [[ -z "$stage_name" ]] && return 0
  ms=$(( $(date +%s%3N) - stage_start_ms ))
  printf '%s\t%s\t%s\n' "$stage_name" "$ms" "$status" >> "$timing_tsv"
  stage_name=""
}

write_timing() { # status
  local status="${1:-running}"
  [[ -d "$logs" ]] || return 0
  "$timing_python" - "$timing_file" "$timing_tsv" "$status" "$script_start_ms" \
    "$venv" "${MAX_JOBS:-0}" <<'PY' || true
import json, os, sys, time
path, tsv, status, start_ms, venv, jobs = sys.argv[1:7]
stages = []
if os.path.exists(tsv):
    with open(tsv, encoding="utf-8") as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) == 3:
                stages.append({"name": parts[0], "seconds": round(int(parts[1]) / 1000, 3), "status": parts[2]})
doc = {
    "status": status,
    "total_seconds": round((int(time.time() * 1000) - int(start_ms)) / 1000, 3),
    "stages": stages,
    "venv": venv,
    "max_jobs": int(jobs),
    "python": sys.version.split()[0],
}
with open(path, "w", encoding="utf-8") as fh:
    json.dump(doc, fh, ensure_ascii=False, indent=2)
    fh.write("\n")
PY
}

on_exit() {
  local rc=$?
  if [[ -n "$stage_name" ]]; then
    end_stage $([[ "$rc" -eq 0 ]] && echo ok || echo failed)
  fi
  write_timing $([[ "$rc" -eq 0 ]] && echo complete || echo failed)
  if [[ "$rc" -ne 0 ]]; then
    printf '\n构建未完成（exit %s）。分阶段计时：%s\n' "$rc" "$timing_file" >&2
  fi
}
trap on_exit EXIT

# ---------------------------------------------------------------- 资源探测
cpu_quota_cores() {
  local quota period
  if [[ -r /sys/fs/cgroup/cpu.max ]]; then
    read -r quota period < /sys/fs/cgroup/cpu.max
    if [[ "$quota" != max && -n "${period:-}" && "$period" -gt 0 ]]; then
      printf '%s' $(( quota / period ))
      return
    fi
  fi
  if [[ -r /sys/fs/cgroup/cpu/cpu.cfs_quota_us && -r /sys/fs/cgroup/cpu/cpu.cfs_period_us ]]; then
    quota=$(cat /sys/fs/cgroup/cpu/cpu.cfs_quota_us)
    period=$(cat /sys/fs/cgroup/cpu/cpu.cfs_period_us)
    if [[ "$quota" -gt 0 && "$period" -gt 0 ]]; then
      printf '%s' $(( quota / period ))
      return
    fi
  fi
  nproc --all 2>/dev/null || nproc
}

mem_gib() {
  local bytes=""
  if [[ -r /sys/fs/cgroup/memory.max ]]; then
    bytes=$(cat /sys/fs/cgroup/memory.max)
    [[ "$bytes" == max ]] && bytes=""
  elif [[ -r /sys/fs/cgroup/memory/memory.limit_in_bytes ]]; then
    bytes=$(cat /sys/fs/cgroup/memory/memory.limit_in_bytes)
    # cgroup v1 在无限制时给出接近 2^63 的值。
    [[ -n "$bytes" && "$bytes" -gt $((1 << 50)) ]] && bytes=""
  fi
  if [[ -z "$bytes" ]]; then
    bytes=$(( $(getconf _PHYS_PAGES) * $(getconf PAGE_SIZE) ))
  fi
  printf '%s' $(( bytes / 1024 / 1024 / 1024 ))
}

cores=$(cpu_quota_cores)
mem=$(mem_gib)
max_jobs="${MAX_JOBS:-$(( cores < 16 ? cores : 16 ))}"
mem_cap=$(( mem / 4 ))
[[ "$mem_cap" -lt 1 ]] && mem_cap=1
[[ "$max_jobs" -gt "$mem_cap" ]] && max_jobs="$mem_cap"
[[ "$max_jobs" -lt 1 ]] && max_jobs=1
export MAX_JOBS="$max_jobs"
printf '资源：CPU 配额 %s 核，内存上限 %s GiB，MAX_JOBS=%s\n' "$cores" "$mem" "$MAX_JOBS"

# ---------------------------------------------------------------- 前置检查
if [[ "$mode" == full && "$force" -eq 0 && -f "$marker" ]]; then
  if [[ "$(head -n1 "$marker")" == "$marker_version" ]]; then
    printf '环境已完整：%s\n如需重建请加 --force；只验证 pip 请用 --pip-only。\n' "$marker"
    trap - EXIT
    exit 0
  fi
  printf '检测到旧版完成标记，继续增量校验：%s\n' "$marker"
fi

mkdir -p "$logs" "$materials" "$src_root"
: > "$timing_tsv"

# 旧布局迁移：.build/src -> build/src（避开 glob 隐藏目录问题）。
if [[ -d "$build/src" && ! -e "$src" ]]; then
  printf '迁移源码树：%s -> %s（隐藏目录 glob 兼容）\n' "$build/src" "$src"
  mkdir -p "$src_root"
  mv "$build/src" "$src"
fi

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

# ---------------------------------------------------------------- venv
begin_stage "venv"
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
end_stage

export CLEF_VENV="$venv"
# shellcheck source=scripts/activate_env.sh
source "$repo/scripts/activate_env.sh"

begin_stage "base-check"
step "基础校验"
python - <<'PY'
import sys
assert sys.version_info[:3] == (3, 12, 16), sys.version
print(sys.executable, sys.version.split()[0])
PY
end_stage

# ---------------------------------------------------------------- 安装工具
use_uv=0
uv_version="0.13.10"

py_install() { # 与 pip 兼容的参数；优先 uv（失败自动回退 pip）
  if [[ "$use_uv" == 1 ]]; then
    # unsafe-best-match: 像 pip 一样跨索引选版本（memcache-hybrid 等只存在于
    # 华为 Ascend 索引；uv 默认 first-index 不会回退到 extra index）。
    local uv_args=(--index-strategy unsafe-best-match) skip_next=0 arg
    for arg in "$@"; do
      if [[ "$skip_next" == 1 ]]; then
        skip_next=0
        continue
      fi
      case "$arg" in
        --prefer-binary) continue ;;
        --constraint) uv_args+=(--constraints) ;;
        --force-reinstall) uv_args+=(--reinstall) ;;
        --timeout) skip_next=1 ;;
        *) uv_args+=("$arg") ;;
      esac
    done
    if uv pip install --python "$venv/bin/python" "${uv_args[@]}"; then
      return 0
    fi
    printf '  uv 安装失败，回退 pip\n'
  fi
  python -m pip install "$@"
}

begin_stage "build-tools"
step "1/7 构建工具"
py_install --index-url "$index" --constraint "$constraints" -r "$repo/scripts/requirements/build-tools.txt"
if ! have uv; then
  step "安装 uv（加速后续纯 wheel 安装）"
  py_install --index-url "$index" "uv==$uv_version" || printf '  uv 安装失败，后续使用 pip\n'
fi
if have uv; then
  use_uv=1
  printf '  uv %s（%s）\n' "$(uv --version | awk '{print $2}')" "$(command -v uv)"
fi
end_stage

# ---------------------------------------------------------------- 平台 wheel
begin_stage "platform-wheels"
step "2/7 平台 wheel（torch / torchvision / torchaudio / torch-npu）"
wheels="$materials/wheels"
torch_filenames=(
  "torch-2.10.0%2Bcpu-cp312-cp312-manylinux_2_28_aarch64.whl#sha256=179451716487f8cb09b56459667fa1f5c4c0946c1e75fbeae77cfc40a5768d87"
  "torchvision-0.25.0%2Bcpu-cp312-cp312-manylinux_2_28_aarch64.whl#sha256=727334e9a721cfc1ac296ce0bf9e69d9486821bfa5b1e75a8feb6f78041db481"
  "torchaudio-2.10.0%2Bcpu-cp312-cp312-manylinux_2_28_aarch64.whl#sha256=2ef1f8d0f0feba5a9364620925c864890b90abc98a65f35895256dd6aa1710be"
)
torch_mirrors=(
  "https://mirror.nju.edu.cn/pytorch/whl/cpu"
  "https://mirrors.aliyun.com/pytorch-wheels/cpu"
)
if [[ -f "$wheels/torch-2.10.0+cpu-cp312-cp312-manylinux_2_28_aarch64.whl" ]]; then
  py_install --index-url "$index" --constraint "$constraints" \
    "$wheels"/torch-*.whl "$wheels"/torchvision-*.whl \
    "$wheels"/torchaudio-*.whl "$wheels"/torch_npu-*.whl \
    -r "$repo/scripts/requirements/cann-tools.txt"
else
  torch_ok=0
  for mirror in "${torch_mirrors[@]}"; do
    printf '  尝试 PyTorch 镜像 %s\n' "$mirror"
    if py_install --index-url "$index" --constraint "$constraints" \
         "${torch_filenames[@]/#/$mirror/}" \
         "torch-npu==2.10.0.post7" \
         -r "$repo/scripts/requirements/cann-tools.txt"; then
      torch_ok=1
      break
    fi
    printf '  镜像失败，换下一个\n'
  done
  if [[ "$torch_ok" -eq 0 ]]; then
    printf '  镜像不可用，回退官方 wheel URL\n'
    py_install --index-url "$index" --constraint "$constraints" \
      -r "$repo/scripts/requirements/base-torch.txt" \
      -r "$repo/scripts/requirements/cann-tools.txt"
  fi
fi
end_stage

# ---------------------------------------------------------------- 纯 PyPI 锁
begin_stage "pypi-requirements"
step "3/7 纯 PyPI 依赖锁（requirements.txt）"
# requirements.txt 是完整传递闭包（199 包全量冻结），--no-deps 避免 uv 去
# 索引里找 +cpu 本地版本标签；完整性由最后的 pip check 验收。
py_install --index-url "$index" --extra-index-url "$ascend_index" \
  --constraint "$constraints" --timeout 60 --prefer-binary --no-deps \
  -r "$repo/requirements.txt"
end_stage

# ---------------------------------------------------------------- 评测依赖
begin_stage "eval-deps"
if [[ "$eval_enabled" == 1 ]]; then
  step "4/7 评测依赖（EvalScope API）"
  py_install --index-url "$index" --extra-index-url "$ascend_index" \
    --constraint "$constraints" --constraint "$repo/requirements.txt" \
    --timeout 60 --prefer-binary -r "$repo/scripts/requirements/evalscope-api.txt"
  end_stage
else
  printf '\n跳过评测依赖（--no-eval）。\n'
  end_stage
fi

# ---------------------------------------------------------------- triton 覆盖
begin_stage "triton-ascend"
step "5/7 triton-ascend 覆盖（必须晚于 requirements.txt 的 triton==3.6.0）"
triton_wheel="$(first_glob "$materials/triton_ascend-*.whl")"
if [[ -n "$triton_wheel" ]]; then
  python -m pip install --force-reinstall --no-deps "$triton_wheel"
else
  python -m pip install --force-reinstall --no-deps \
    --index-url "$ascend_index" triton-ascend==3.6.0.post2
fi
python - "$logs/triton-check.txt" <<'PY' | tee "$logs/triton-check.txt"
import sys
import triton
from triton.backends import backends
assert "ascend" in backends, sorted(backends)
print("triton.__file__ =", triton.__file__)
print("triton.__version__ =", triton.__version__)
print("triton.backends =", sorted(backends))
print("TRITON_ASCEND_OK")
PY
grep -q TRITON_ASCEND_OK "$logs/triton-check.txt" \
  || die "triton 未指向 Ascend 后端，见 $logs/triton-check.txt"
end_stage

# ---------------------------------------------------------------- 源码构建
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

verify_source() { # dir commit recursive
  local dir="$1" commit="$2" recursive="$3" head status spec path
  if [[ -d "$dir/.git" ]]; then
    head="$(git -C "$dir" rev-parse HEAD 2>/dev/null || true)"
    [[ "$head" == "$commit" ]] || die "$dir HEAD=${head:-<none>}，期望 $commit"
    if [[ "$recursive" == 1 ]]; then
      status="$(git -C "$dir" submodule status --recursive 2>/dev/null || true)"
      if grep -qE '^[-+]' <<<"$status"; then
        die "$dir 子模块状态异常：$(grep -E '^[-+]' <<<"$status" | head -3)"
      fi
    fi
    return
  fi
  # archive（无 .git）模式：用 marker 校验固定提交。
  [[ -f "$dir/.clef-source-commit" ]] || die "$dir 缺少 .clef-source-commit（archive 模式）"
  [[ "$(cat "$dir/.clef-source-commit")" == "$commit" ]] \
    || die "$dir archive commit=$(cat "$dir/.clef-source-commit")，期望 $commit"
  if [[ "$recursive" == 1 ]]; then
    for spec in "${source_submodules[@]}"; do
      path="${spec%%|*}"
      [[ -f "$dir/$path/.clef-source-commit" ]] \
        || die "缺少子模块 $path（archive 模式）"
    done
  fi
}

fetch_commit_git() { # url commit target（用于 gitcode 子模块，固定提交浅拉取）
  local url="$1" commit="$2" target="$3"
  rm -rf "$target"
  mkdir -p "$target"
  if git -C "$target" init -q \
     && git -C "$target" remote add origin "$url" \
     && run_capped 300 git -C "$target" fetch --depth 1 origin "$commit" \
     && git -C "$target" checkout -q --detach FETCH_HEAD; then
    printf '%s\n' "$commit" > "$target/.clef-source-commit"
    return 0
  fi
  return 1
}

download_archive() { # repo_path commit sha dest
  local repo_path="$1" commit="$2" sha="$3" dest="$4"
  local mirror prefix url digest attempt resume
  for mirror in "${git_mirrors[@]}"; do
    prefix=""
    [[ -n "$mirror" ]] && prefix="${mirror%/}/"
    url="${prefix}https://github.com/${repo_path}/archive/${commit}.tar.gz"
    printf '  archive %s\n' "$url"
    rm -f "$dest.part"
    local use_resume=0 rc=0
    for attempt in 1 2 3 4 5; do
      resume=()
      (( use_resume )) && resume=(-C -)
      if curl -fL "${resume[@]}" --connect-timeout 20 --max-time 1800 -o "$dest.part" "$url"; then
        break
      fi
      rc=$?
      # 默认续传；仅当代理不支持 Range（33/36）时改回重下。
      if [[ "$rc" -eq 33 || "$rc" -eq 36 ]]; then
        use_resume=0
      else
        use_resume=1
      fi
      sleep 2
    done
    if [[ -f "$dest.part" ]]; then
      digest="$(sha256sum "$dest.part" | awk '{print $1}')"
      if [[ "$digest" == "$sha" ]]; then
        mv "$dest.part" "$dest"
        return 0
      fi
      printf '  archive SHA-256 不匹配（%s），换下一个源\n' "$digest"
      rm -f "$dest.part"
    fi
  done
  return 1
}

install_source_archive() { # dir repo_path commit sha recursive
  local dir="$1" repo_path="$2" commit="$3" sha="$4" recursive="$5"
  local dest="$materials/$(basename "$repo_path")-${commit}.tar.gz" spec path url sub_commit
  download_archive "$repo_path" "$commit" "$sha" "$dest" \
    || return 1
  rm -rf "$dir"
  mkdir -p "$dir"
  tar --no-same-owner -xzf "$dest" -C "$dir" --strip-components=1
  printf '%s\n' "$commit" > "$dir/.clef-source-commit"
  if [[ "$recursive" == 1 ]]; then
    for spec in "${source_submodules[@]}"; do
      path="${spec%%|*}"
      url="${spec#*|}"; url="${url%%|*}"
      sub_commit="${spec##*|}"
      printf '  子模块 %s @ %s\n' "$path" "$sub_commit"
      fetch_commit_git "$url" "$sub_commit" "$dir/$path" \
        || die "子模块拉取失败：$url @ $sub_commit"
    done
  fi
  return 0
}

run_capped() { # seconds command...
  local secs="$1"; shift
  if have timeout; then
    timeout "$secs" "$@"
  else
    "$@"
  fi
}

git_submodule_update() { # dir mirror-prefix
  local dir="$1" prefix="$2" cfg=()
  [[ -n "$prefix" ]] && cfg=(-c "url.${prefix}https://github.com/.insteadOf=https://github.com/")
  run_capped 1800 git "${cfg[@]}" -C "$dir" submodule update --init --recursive --depth 1 \
    || run_capped 1800 git "${cfg[@]}" -C "$dir" submodule update --init --recursive
}

ensure_source() { # dir tarball url commit recursive [archive_sha]
  local dir="$1" tarball="$2" url="$3" commit="$4" recursive="$5" archive_sha="${6:-}"
  local repo_path="${url#https://github.com/}"
  repo_path="${repo_path%.git}"
  if [[ -d "$dir/.git" ]]; then
    local head
    head="$(git -C "$dir" rev-parse HEAD 2>/dev/null || true)"
    if [[ -z "$head" ]]; then
      printf '  删除不完整源码树：%s\n' "$dir"
      rm -rf "$dir"
    else
      [[ "$head" == "$commit" ]] || die "$dir HEAD=$head，期望 $commit"
      if [[ "$recursive" == 1 ]]; then
        git_submodule_update "$dir" ""
      fi
      verify_source "$dir" "$commit" "$recursive"
      return
    fi
  elif [[ -f "$dir/.clef-source-commit" ]]; then
    verify_source "$dir" "$commit" "$recursive"
    return
  fi
  mkdir -p "$src_root"
  if [[ -n "$tarball" && -f "$tarball" ]]; then
    tar --no-same-owner -xzf "$tarball" -C "$src_root"
    if [[ ! -d "$dir/.git" && ! -f "$dir/.clef-source-commit" ]]; then
      printf '%s\n' "$commit" > "$dir/.clef-source-commit"
    fi
    verify_source "$dir" "$commit" "$recursive"
    return
  fi
  have git || die "缺少 $tarball 且未安装 git"
  local mirror prefix cfg cloned=0
  if [[ "$force_tarball" != 1 ]]; then
    for mirror in "${git_mirrors[@]}"; do
      prefix=""
      [[ -n "$mirror" ]] && prefix="${mirror%/}/"
      cfg=()
      [[ -n "$prefix" ]] && cfg=(-c "url.${prefix}https://github.com/.insteadOf=https://github.com/")
      if ! run_capped 15 git ls-remote "${prefix}${url}" HEAD >/dev/null 2>&1; then
        printf '  源不可用或超时，跳过：%s%s\n' "$prefix" "$url"
        continue
      fi
      printf '  git fetch --depth 1 %s%s @ %s\n' "$prefix" "$url" "$commit"
      rm -rf "$dir"
      mkdir -p "$dir"
      if git -C "$dir" init -q \
         && git -C "$dir" remote add origin "${prefix}${url}" \
         && run_capped 60 git "${cfg[@]}" -C "$dir" fetch --depth 1 origin "$commit" \
         && git -C "$dir" checkout -q --detach FETCH_HEAD; then
        cloned=1
        break
      fi
      printf '  浅拉取超时/失败，换下一个源\n'
      rm -rf "$dir"
    done
  fi
  if [[ "$cloned" -eq 1 ]]; then
    if [[ "$recursive" == 1 ]]; then
      git_submodule_update "$dir" "$prefix"
    fi
    verify_source "$dir" "$commit" "$recursive"
    return
  fi
  if [[ "$cloned" -eq 0 && -n "$archive_sha" ]]; then
    printf '  git 通道不可用，回退固定提交 archive（SHA-256 校验）\n'
    if install_source_archive "$dir" "$repo_path" "$commit" "$archive_sha" "$recursive"; then
      verify_source "$dir" "$commit" "$recursive"
      return
    fi
    printf '  archive 回退失败\n'
  fi
  if [[ "$cloned" -eq 0 && "$force_tarball" != 1 ]]; then
    printf '  尝试完整 clone（最后回退）\n'
    for mirror in "${git_mirrors[@]}"; do
      prefix=""
      [[ -n "$mirror" ]] && prefix="${mirror%/}/"
      cfg=()
      [[ -n "$prefix" ]] && cfg=(-c "url.${prefix}https://github.com/.insteadOf=https://github.com/")
      rm -rf "$dir"
      if run_capped 1800 git "${cfg[@]}" clone "${prefix}${url}" "$dir" \
         && git -C "$dir" checkout -q --detach "$commit"; then
        cloned=1
        break
      fi
      printf '  clone 失败，换下一个源\n'
    done
    if [[ "$cloned" -eq 1 ]]; then
      if [[ "$recursive" == 1 ]]; then
        git_submodule_update "$dir" "$prefix"
      fi
      verify_source "$dir" "$commit" "$recursive"
      return
    fi
  fi
  die "无法获取 $url @ $commit（可把源码包放到 $materials，或设置 CLEF_FORCE_TARBALL=1）"
}

run_build() { # log command...
  local log="$1"; shift
  if ! "$@" 2>&1 | tee "$log"; then
    printf '\n--- 构建失败，关键错误（%s） ---\n' "$log" >&2
    grep -n -A5 -E "error:|fatal error|CMake Error|Killed|TooFewObj|TooFewLinkCmd|undefined reference|No space left" "$log" \
      | tail -80 >&2 || true
    die "构建失败；完整日志：$log"
  fi
}

if [[ "$eval_enabled" == 1 ]]; then
  begin_stage "evalscope-source"
  step "6/7 EvalScope 固定提交"
  source_submodules=()
  ensure_source "$src/evalscope" "" https://github.com/modelscope/evalscope.git \
    "$evalscope_commit" 0 "$evalscope_archive_sha"
  py_install --index-url "$index" --extra-index-url "$ascend_index" \
    --no-deps -e "$src/evalscope"
  end_stage
fi

if [[ "$mode" == pip ]]; then
  begin_stage "verify"
  step "pip-only 验收"
  python -m pip check | tee "$logs/pip-check.txt"
  python -m pip freeze --all > "$logs/freeze-after-pip.txt"
  end_stage
  printf '\nPIP_ONLY_OK  （pip check / 版本见 %s）\n' "$logs"
  exit 0
fi

begin_stage "vllm-build"
step "7/7 固定源码构建：vLLM"
source_submodules=()
ensure_source "$src/vllm" "$materials/vllm-24420b7-with-git.tar.gz" \
  https://github.com/vllm-project/vllm.git "$vllm_commit" 0 "$vllm_archive_sha"
apply_patch "$src/vllm" "$repo/src/patches/vllm-a3-numpy126.patch"
apply_patch "$src/vllm" "$repo/src/patches/vllm-clef-load-audit.patch"
apply_patch "$src/vllm" "$repo/src/patches/clef-api-routes.patch"
vllm_scm_env=()
[[ -f "$src/vllm/.clef-source-commit" && ! -d "$src/vllm/.git" ]] \
  && vllm_scm_env=(SETUPTOOLS_SCM_PRETEND_VERSION="$vllm_version_hint")
run_build "$logs/vllm-build.log" env "${vllm_scm_env[@]}" VLLM_TARGET_DEVICE=empty \
  python -m pip install --no-build-isolation -v \
  --constraint "$constraints" --index-url "$index" --timeout 60 -e "$src/vllm"
end_stage

begin_stage "vllm-ascend-build"
step "固定源码构建：vLLM-Ascend"
source_submodules=(
  "csrc/third_party/catlass|https://gitcode.com/cann/catlass.git|41bf90da655bba3c66d0acd7e00abe33960ecfd6"
  "csrc/third_party/catlass/3rdparty/googletest|https://gitcode.com/GitHub_Trending/go/googletest.git|f8d7d77c06936315286eb55f8de22cd23c188571"
)
ensure_source "$src/vllm-ascend" "$materials/vllm-ascend-89eb8a5-with-submodules.tar.gz" \
  https://github.com/vllm-project/vllm-ascend.git "$ascend_commit" 1 "$ascend_archive_sha"
apply_patch "$src/vllm-ascend" "$repo/src/patches/vllm-ascend-clef-tp-embedding.patch"
apply_patch "$src/vllm-ascend" "$repo/src/patches/gdn-bsnd.patch"
if [[ -z "${ASCEND_HOME_PATH:-}" || ! -d "${ASCEND_HOME_PATH:-/nonexistent}" ]]; then
  for candidate in /usr/local/Ascend/ascend-toolkit/latest /usr/local/Ascend/cann \
                   /usr/local/Ascend/cann-9.2.1; do
    if [[ -d "$candidate" ]]; then
      export ASCEND_HOME_PATH="$candidate"
      printf '  ASCEND_HOME_PATH=%s\n' "$candidate"
      break
    fi
  done
fi
[[ -d "${ASCEND_HOME_PATH:-/nonexistent}" ]] \
  || die "ASCEND_HOME_PATH 无效；请先 source CANN 的 set_env.sh 或 export ASCEND_HOME_PATH"
for tool in ccec bishengir-compile cmake ninja; do
  have "$tool" || die "缺少构建工具 $tool（检查 CANN set_env.sh 与 PATH）"
done
printf '  MAX_JOBS=%s ASCEND_HOME_PATH=%s\n' "$MAX_JOBS" "$ASCEND_HOME_PATH"
ascend_scm_env=()
[[ -f "$src/vllm-ascend/.clef-source-commit" && ! -d "$src/vllm-ascend/.git" ]] \
  && ascend_scm_env=(SETUPTOOLS_SCM_PRETEND_VERSION="$ascend_version_hint")
run_build "$logs/vllm-ascend-build.log" env "${ascend_scm_env[@]}" SOC_VERSION=ascend910_9382 \
  COMPILE_CUSTOM_KERNELS=1 MAX_JOBS="$MAX_JOBS" \
  python -m pip install --no-build-isolation -v --constraint "$constraints" \
  --index-url "$index" --timeout 60 \
  --extra-index-url "$ascend_index" \
  --extra-index-url "$ascend_nightly" \
  -e "$src/vllm-ascend"
end_stage

begin_stage "memfabric-fla"
step "MemFabric / FLA"
have mfcli || die "mfcli 不在 PATH；检查 memfabric_hybrid 是否装进 $venv"
# MemFabric 的 AccOffload kernel 源码在隐藏的 .venv 下；CANN 的
# recompile_binary.py 用 glob 收集 .o 时会跳过隐藏目录。仅在 mfcli 进程树中
# 注入 include_hidden=True 的兼容钩子（见 scripts/compat/glob_hidden）。
mfcli_pythonpath="$repo/scripts/compat/glob_hidden${PYTHONPATH:+:$PYTHONPATH}"
( cd "$repo" && PYTHONPATH="$mfcli_pythonpath" TORCH_DEVICE_BACKEND_AUTOLOAD=0 \
    mfcli kernel install --soc-version A3 ) \
  > "$logs/memfabric-kernel.log" 2>&1 \
  || die "mfcli kernel install 失败，见 $logs/memfabric-kernel.log"

fla_wheel="$(first_glob "$materials/flash_linear_attention_npu_a3-*.whl")"
if [[ -n "$fla_wheel" ]]; then
  python -m pip install --no-deps "$fla_wheel"
else
  python -m pip install --no-deps --index-url "$index" --extra-index-url "$ascend_index" \
    --timeout 60 flash-linear-attention-npu-a3==26.9.1
fi
python -m pip install --index-url "$index" --constraint "$constraints" --timeout 60 \
  expecttest==0.3.0
end_stage

begin_stage "flash-attn-build"
step "flash_attn_npu 0.5.0（无预编译 wheel，源码构建）"
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
run_build "$logs/flash-attn-build.log" env FLASH_ATTN_BUILD_NPU=910 \
  FLASH_ATTENTION_FORCE_BUILD=TRUE MAX_JOBS="$MAX_JOBS" \
  python -m pip install --no-build-isolation -v --constraint "$constraints" \
  --index-url "$index" --timeout 60 "$flash_dir"
end_stage

# ---------------------------------------------------------------- 验收
begin_stage "verify"
step "验收"
python -m pip check | tee "$logs/pip-check.txt"
python -m pip freeze --all > "$logs/freeze-complete.txt"
python - "$logs/versions.txt" <<'PY'
import importlib.metadata as md
import sys
names = ("torch", "torch-npu", "torchvision", "torchaudio", "transformers",
         "numpy", "scipy", "triton", "triton-ascend", "flash-attn-npu",
         "flash-linear-attention-npu-a3", "vllm", "vllm-ascend", "evalscope")
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

CLEF_CHECK_EVAL="$eval_enabled" python - "$logs/env-checks.txt" <<'PY' | tee "$logs/env-checks.txt"
import os
import sys

import triton
from triton.backends import backends
assert "ascend" in backends, sorted(backends)
print("triton:", triton.__version__, triton.__file__)
print("triton backends:", sorted(backends))

import fla_npu  # noqa: F401
import flash_attn_npu  # noqa: F401
import torch
import torch_npu  # noqa: F401
import vllm
import vllm_ascend

print("torch:", torch.__version__, "npu:", torch.npu.is_available(), torch.npu.device_count())
print("vllm:", vllm.__version__)
print("vllm_ascend:", os.path.dirname(vllm_ascend.__file__))

if os.environ.get("CLEF_CHECK_EVAL") == "1":
    import evalscope
    from evalscope.benchmarks.erqa.erqa_adapter import ERQAAdapter  # noqa: F401
    from evalscope.benchmarks.open_decisions.open_decisions_adapter import OpenDecisionsAdapter  # noqa: F401
    print("evalscope:", getattr(evalscope, "__version__", "unknown"))
    print("eval adapters: OpenDecisionsAdapter, ERQAAdapter")

print("ENV_CHECKS_OK")
PY
grep -q ENV_CHECKS_OK "$logs/env-checks.txt" || die "环境导入检查失败，见 $logs/env-checks.txt"

smoke="${CLEF_SMOKE_OUTPUT:-$logs/smoke-$(date +%Y%m%d-%H%M%S).json}"
if [[ -d "${MODEL_DIR:-}" ]]; then
  python "$repo/scripts/verify_npu_environment.py" --stage runtime --require-extensions \
    --model-dir "$MODEL_DIR" --output "$smoke" \
    || die "NPU smoke 失败，见 $smoke"
else
  printf '跳过模型 smoke：MODEL_DIR 不存在（%s）。\n' "${MODEL_DIR:-未设置}"
fi
end_stage

printf '%s\n%s\n' "$marker_version" "$(date -Is)" > "$marker"
printf '\n环境完成：%s\n' "$venv"
printf '源码树：%s（非隐藏目录，规避 Ascend C glob 隐藏目录缺陷）\n' "$src"
printf '启动：source scripts/activate_env.sh && bash start.sh\n'
printf '日志、freeze 与分阶段计时：%s\n' "$logs"
