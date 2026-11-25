#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo '用法：bash scripts/bootstrap_python.sh <CPython 发行包路径>' >&2
  exit 2
fi
[[ "$(uname -sm)" == 'Linux aarch64' ]] || {
  echo '该发行包要求 Linux aarch64。' >&2
  exit 1
}

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
prefix="${CLEF_PYTHON_PREFIX:-$repo_root/.build/python-3.12.16+20261110}"
venv="${CLEF_VENV:-$repo_root/.venv}"
archive="$1"
expected='4792f81750282597fb2a3389fe18b945ed2ba6aad5e45ad406d122c58cb33081'
printf '%s  %s\n' "$expected" "$archive" | sha256sum --check --strict

if [[ -e "$prefix" && ! -x "$prefix/bin/python3.12" ]]; then
  printf '请检查未完成的 Python 安装目录：%s\n' "$prefix" >&2
  exit 1
fi
if [[ ! -e "$prefix" ]]; then
  mkdir -p "$prefix"
  tar -xzf "$archive" --strip-components=1 -C "$prefix"
fi
"$prefix/bin/python3.12" -c 'import sys,ssl,sqlite3,lzma,bz2,ctypes; assert sys.version_info[:3] == (3,12,16); print(sys.version); print(ssl.OPENSSL_VERSION)'
if [[ ! -e "$venv" ]]; then
  "$prefix/bin/python3.12" -m venv "$venv"
fi
"$venv/bin/python" -c 'import sys; assert sys.version_info[:3] == (3,12,16); print(sys.executable); print(sys.version)'
"$venv/bin/python" -m pip --version
