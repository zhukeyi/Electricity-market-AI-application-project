#!/usr/bin/env bash
# 本地 ↔ Oracle 双向同步。
#
# 架构约定（为什么必须分居两处）：
#   * 本机出口无法访问 oasis.caiso.com（HTTP 000 / DNS SERVFAIL），
#     因此**抓数只能在 Oracle 上做**，Oracle 是数据文件的唯一写者；
#   * 本机是开发与分析环境，也是提交文件最终要拿走的地方（手工上传到平台）。
#
# 用法：
#   scripts/sync.sh up                 # 代码上行（不动远端数据）
#   scripts/sync.sh up --with-data     # 首次播种：连 data/raw 一起上行
#   scripts/sync.sh down               # 结果 + 数据下行（覆盖本地同名文件）
#   scripts/sync.sh status             # 远端定时器状态与最近一次运行记录
#
# 环境变量：CSG_HOST / CSG_KEY / CSG_REMOTE_DIR
set -uo pipefail

HOST="${CSG_HOST:-ubuntu@217.142.246.70}"
KEY="${CSG_KEY:-$HOME/.ssh/id_rsa_oracle}"
RDIR="${CSG_REMOTE_DIR:-csg-ai-market}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 2

SSH=(ssh -o BatchMode=yes -o StrictHostKeyChecking=no -o ConnectTimeout=12 -i "$KEY" "$HOST")
RSYNC=(rsync -az --human-readable -e "ssh -o BatchMode=yes -o StrictHostKeyChecking=no -i $KEY")

up() {
  local with_data=0
  [ "${1:-}" = "--with-data" ] && with_data=1
  echo "→ 上行代码到 ${HOST}:${RDIR}"
  "${RSYNC[@]}" --delete \
    --exclude '__pycache__' --exclude '*.pyc' \
    src scripts deploy tests "$HOST:$RDIR/"
  for f in README.md requirements.txt .pipeline.env.example; do
    [ -f "$f" ] && "${RSYNC[@]}" "$f" "$HOST:$RDIR/"
  done
  # 真实配置只在本地存在时上行一次；远端已存在的不要被覆盖（--ignore-existing）
  if [ -f .pipeline.env ]; then
    "${RSYNC[@]}" --ignore-existing .pipeline.env "$HOST:$RDIR/"
  fi
  if [ "$with_data" -eq 1 ]; then
    echo "→ 播种数据 data/raw"
    "${RSYNC[@]}" --exclude '*.bak_*' data/raw "$HOST:$RDIR/data/"
  fi
  echo "完成"
}

down() {
  echo "← 下行结果与数据"
  mkdir -p data/out
  "${RSYNC[@]}" "$HOST:$RDIR/data/out/submission" data/out/ 2>/dev/null || true
  "${RSYNC[@]}" "$HOST:$RDIR/data/out/runs" data/out/ 2>/dev/null || true
  "${RSYNC[@]}" "$HOST:$RDIR/data/raw/" data/raw/ 2>/dev/null || true
  echo "完成"
}

status() {
  echo "== 定时器 =="
  "${SSH[@]}" 'systemctl is-active csg-pipeline.timer; systemctl list-timers csg-pipeline.timer --no-pager 2>/dev/null | head -3'
  echo "== 最近运行 =="
  "${SSH[@]}" "cd ~/$RDIR 2>/dev/null && ls -1t data/out/runs/*.json 2>/dev/null | head -1 | xargs -r python3 -c \"import json,sys;d=json.load(open(sys.argv[1]));print(d['trade_date'],'ok='+str(d['ok']),'revenue=',d.get('expected_revenue'));[print(' ',s['name'],s['ok'],s['detail'][:90]) for s in d['steps']]\""
}

# 本机定时拉取：每天 23:20 / 23:50 把 Oracle 的结果取回，省掉"记得同步"这件事
PLIST="$HOME/Library/LaunchAgents/com.mike.csg.pull.plist"

install_local() {
  mkdir -p "$(dirname "${PLIST}")" data/out/runs
  sed "s#__PROJECT__#$ROOT#g" deploy/com.mike.csg.pull.plist > "${PLIST}"
  echo "已写入 ${PLIST}"
  # 优先用现代 API（错误信息更有用），退回老式 load
  if ! launchctl bootstrap "gui/$(id -u)" "${PLIST}" 2>/dev/null; then
    launchctl load "${PLIST}" 2>/dev/null || true
  fi
  sleep 1
  if launchctl list 2>/dev/null | grep -q com.mike.csg.pull; then
    echo "已加载（每日 23:20 / 23:50 各拉取一次）"
  else
    echo "!! 自动加载失败（多见于受沙箱限制的终端）。请在你自己的终端里执行一次："
    echo "   launchctl bootstrap gui/\$(id -u) ${PLIST}"
  fi
}

uninstall_local() {
  launchctl bootout "gui/$(id -u)/com.mike.csg.pull" 2>/dev/null || launchctl unload "${PLIST}" 2>/dev/null || true
  rm -f "${PLIST}"
  echo "已卸载本地定时拉取"
}

case "${1:-}" in
  up) shift; up "${1:-}" ;;
  down) down ;;
  status) status ;;
  install-local) install_local ;;
  uninstall-local) uninstall_local ;;
  *) sed -n '2,24p' "$0"; exit 2 ;;
esac
