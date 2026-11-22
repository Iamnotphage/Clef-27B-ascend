#!/usr/bin/env bash
# Manage the local SSH master used by `ssh clef`.
set +x
set -euo pipefail

usage() {
  cat <<'EOF'
用法：
  ./connect.sh          复用主连接，或输入最新平台凭据建立后台主连接
  ./connect.sh status   检查本机主连接进程
  ./connect.sh stop     关闭主连接及所有复用会话
  ./connect.sh new      隐藏输入新 A3 的平台 SSH 命令，建立独立前台连接

连接凭据可交互输入，也可通过 CLEF_JUMP_AUTH 提供。
输入格式：jt_跳板账号:临时令牌，或平台提供的完整 ssh -J 命令。
主连接建立后，终端和本机 agent 使用 ssh clef。
EOF
}

fail() {
  printf '%s\n' "$*" >&2
  exit 1
}

auth_pattern='jt_[[:xdigit:]]+:[[:xdigit:]]+'
legacy_target_host=199.88.54.221

new_a3_login() (
  [[ -t 0 ]] || fail '请在终端运行 ./connect.sh new。'
  local input ipv4 pattern jump_auth jump_host target_host private_dir ssh_config
  ipv4='[0-9]{1,3}[.][0-9]{1,3}[.][0-9]{1,3}[.][0-9]{1,3}'
  pattern="^[[:space:]]*ssh[[:space:]]+-J[[:space:]]+['\"]?(${auth_pattern})@(${ipv4}):2234['\"]?[[:space:]]+root@(${ipv4})[[:space:]]*$"
  if ! IFS= read -r -s -p '粘贴新 A3 的平台完整 SSH 命令（输入隐藏）：' input; then
    printf '\n' >&2
    fail '未读取到新 A3 的平台 SSH 命令。'
  fi
  printf '\n' >&2
  [[ "$input" =~ $pattern ]] || fail '请提供新 A3 的完整 ssh -J 命令。'
  jump_auth="${BASH_REMATCH[1]}"
  jump_host="${BASH_REMATCH[2]}"
  target_host="${BASH_REMATCH[3]}"
  unset input
  [[ "$target_host" != "$legacy_target_host" && "$target_host" != "$jump_host" ]] ||
    fail '请核对新 A3 的目标地址。'

  umask 077
  private_dir="$(mktemp -d /tmp/clef-a3-new.XXXXXXXX)" || fail '无法创建临时 SSH 配置目录。'
  ssh_config="$private_dir/config"
  trap 'rm -f -- "$ssh_config"; rmdir -- "$private_dir"' EXIT
  trap 'exit 129' HUP
  trap 'exit 130' INT
  trap 'exit 143' TERM
  cat > "$ssh_config" <<EOF
Host clef-a3-new
    HostName $target_host
    User root
    Port 22
    ProxyJump clef-a3-new-gateway
    HostKeyAlias clef-a3-new
    UserKnownHostsFile ~/.ssh/known_hosts_clef_a3_new
    StrictHostKeyChecking ask
    PreferredAuthentications password,keyboard-interactive
    PubkeyAuthentication no
    NumberOfPasswordPrompts 1
Host clef-a3-new-gateway
    HostName $jump_host
    User $jump_auth
    Port 2234
    StrictHostKeyChecking yes
    BatchMode yes
Host *
    ControlMaster no
    ControlPath none
    ControlPersist no
    ForwardAgent no
    ForwardX11 no
    ConnectTimeout 10
    ConnectionAttempts 1
EOF
  unset jump_auth
  SSH_ASKPASS_REQUIRE=never ssh -F "$ssh_config" clef-a3-new \
    2> >(sed -E 's/jt_[[:xdigit:]]+(:[[:xdigit:]]+)?/jt_REDACTED/g' >&2)
)

master_running() {
  ssh -O check clef >/dev/null 2>&1
}

action="${1:-start}"
if [[ $# -gt 1 ]]; then
  usage >&2
  exit 2
fi
case "$action" in
  -h|--help|help) usage; exit 0 ;;
  new) new_a3_login; exit $? ;;
  start|status|stop) ;;
  *) usage >&2; exit 2 ;;
esac

jump_input="${CLEF_JUMP_AUTH:-}"
unset CLEF_JUMP_AUTH

if [[ "$action" == status ]]; then
  if master_running; then
    printf '本机 SSH 主连接正在运行。终端或 agent 可使用 ssh clef。\n'
    exit 0
  fi
  fail '主连接当前不可用。请运行 ./connect.sh，并提供最新平台凭据。'
fi

if [[ "$action" == start ]] && master_running; then
  printf '已复用现有主连接。终端或 agent 可使用 ssh clef。\n'
  exit 0
fi

effective_config="$(ssh -G clef)" || fail '无法读取 clef 配置，请先按 README 配置 SSH。'
socket_path=''
target_host=''
target_user=''
target_port=''
while IFS=' ' read -r key value; do
  case "$key" in
    controlpath) socket_path="$value" ;;
    hostname) target_host="$value" ;;
    user) target_user="$value" ;;
    port) target_port="$value" ;;
  esac
done <<< "$effective_config"
[[ "$target_host" == "$legacy_target_host" && "$target_user" == root && "$target_port" == 22 ]] ||
  fail "请先按 README 配置 Host clef，目标为 root@$legacy_target_host:22。"
[[ "$socket_path" == /* && -d "${socket_path%/*}" ]] ||
  fail 'clef 的 ControlPath 需要指向已有本机目录下的绝对路径。'

lock_dir="${socket_path}.connect-lock"
mkdir -m 700 "$lock_dir" 2>/dev/null ||
  fail "连接管理正在进行。请稍后重试；若上次进程异常退出，请检查锁目录：$lock_dir"
trap 'rmdir "$lock_dir" 2>/dev/null || true' EXIT
trap 'exit 129' HUP
trap 'exit 130' INT
trap 'exit 143' TERM

if [[ "$action" == stop ]]; then
  if master_running; then
    ssh -O exit clef
    printf '已关闭主连接及其复用会话。重建时运行 ./connect.sh。\n'
  else
    printf '主连接当前已停止。\n'
  fi
  exit 0
fi

# Another invocation may have connected while this process acquired the lock.
if master_running; then
  printf '已复用现有主连接。终端或 agent 可使用 ssh clef。\n'
  exit 0
fi

if [[ -z "$jump_input" ]]; then
  [[ -t 0 ]] || fail '请交互运行 ./connect.sh，或通过 CLEF_JUMP_AUTH 提供最新的 jt_账号:令牌。'
  IFS= read -r -s -p '粘贴最新 jt_账号:令牌，或平台完整 SSH 命令（输入隐藏）：' jump_input || {
    printf '\n' >&2
    fail '未读取到平台凭据。'
  }
  printf '\n' >&2
fi

# Parse only the known credential format; pasted commands are never executed.
raw_pattern="^[[:space:]]*(${auth_pattern})[[:space:]]*$"
command_pattern="^[[:space:]]*ssh[[:space:]]+-J[[:space:]]+['\"]?(${auth_pattern})@113[.]47[.]8[.]48:2234['\"]?[[:space:]]+root@199[.]88[.]54[.]221[[:space:]]*$"
if [[ "$jump_input" =~ $raw_pattern || "$jump_input" =~ $command_pattern ]]; then
  jump_auth="${BASH_REMATCH[1]}"
else
  fail '凭据格式无效。请提供完整 jt_账号:令牌，或该实例的原始 ssh -J 命令。'
fi
unset jump_input

if [[ -e "$socket_path" || -L "$socket_path" ]]; then
  [[ -S "$socket_path" && -O "$socket_path" && ! -L "$socket_path" ]] ||
    fail 'ControlPath 已被其他文件占用，请检查 SSH 配置。'
  rm -- "$socket_path"
fi

printf '正在建立后台 SSH 主连接，请按 SSH 提示完成服务器认证。\n'
# The explicit -J overrides the config's fallback ProxyCommand for this startup.
if ! ssh -MNf \
  -J "$jump_auth@113.47.8.48:2234" \
  -o ControlPersist=yes \
  -o ServerAliveInterval=30 \
  -o ServerAliveCountMax=3 \
  -o ConnectTimeout=15 \
  -o ConnectionAttempts=1 \
  clef 2> >(sed -E 's/jt_[[:xdigit:]]+:[[:xdigit:]]+/jt_REDACTED:REDACTED/g' >&2); then
  fail '主连接建立失败。请获取最新平台账号和令牌后重新运行 ./connect.sh。'
fi
unset jump_auth
master_running || fail '主连接未就绪。请检查 SSH 输出后重新运行 ./connect.sh。'
printf '后台主连接已建立，终端窗口可以关闭。之后使用 ssh clef。\n'
