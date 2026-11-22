#!/usr/bin/env bash
# Oracle 端一次性安装脚本（幂等，可重复执行）。
#
#   ssh -i ~/.ssh/id_rsa_oracle ubuntu@217.142.246.70 'bash ~/csg-ai-market/deploy/install.sh'
#
# 做四件事：校验虚拟环境 → 装 systemd 单元 → 启用定时器 → 打印状态。
# 不负责代码同步（由本地 scripts/sync.sh up 完成）。
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 2

SYSTEMD_DIR="${SYSTEMD_DIR:-/etc/systemd/system}"
SUDO=""
[ "$(id -u)" -ne 0 ] && SUDO="sudo"

echo "== 1/4 校验环境 =="
if [ ! -x "$ROOT/.venv/bin/python" ]; then
  echo "缺少 .venv，请先运行：python3 -m venv .venv && .venv/bin/pip install numpy pandas scipy" >&2
  exit 2
fi
"$ROOT/.venv/bin/python" - <<'PY' || exit 2
import sys
sys.path.insert(0, ".")
import numpy, pandas, scipy  # noqa: F401
from src.pipeline.clock import next_submittable
print(f"  numpy/pandas/scipy ok")
d, roll = next_submittable()
print(f"  当前可提交运行日：{d}" + (f"（已前滚 {roll} 天）" if roll else ""))
PY

echo "== 2/4 安装 systemd 单元 =="
$SUDO install -m 644 "$ROOT/deploy/csg-pipeline.service" "$SYSTEMD_DIR/csg-pipeline.service"
$SUDO install -m 644 "$ROOT/deploy/csg-pipeline.timer" "$SYSTEMD_DIR/csg-pipeline.timer"
$SUDO systemctl daemon-reload
echo "  已写入 $SYSTEMD_DIR"

echo "== 3/4 启用定时器 =="
$SUDO systemctl enable --now csg-pipeline.timer
echo "  已启用 csg-pipeline.timer"

echo "== 4/4 状态 =="
systemctl list-timers csg-pipeline.timer --no-pager 2>/dev/null | head -3
echo
echo "提醒：队名与成员请写入 $ROOT/.pipeline.env，例如"
echo "  CSG_TEAM=你的队名"
echo "  CSG_MEMBERS=张三,李四"
echo "  CSG_WEBHOOK=https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=..."
