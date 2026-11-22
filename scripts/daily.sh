#!/usr/bin/env bash
# 每日提交任务的守护壳 —— systemd/cron 只调这个，不直接调 pipeline.py。
#
# 为什么要多这一层：
# 1. **把运行日钉死**。pipeline.py 缺省按"LA 明天"推算，若在 BJ 00:00 之后重试，
#    推算结果会翻到再下一天。这里在入口处算一次，后续重试全部复用同一个日期。
# 2. **重试有界**。OASIS 偶发 502、SSH 偶发抖动都不该让当天申报作废；
#    最多 3 次、间隔 5/10 分钟，全部发生在 23:00~23:20（BJ），离 02:00 截止仍有余量。
# 3. **退出码语义化**。0=正常 / 1=有告警但文件已落地 / 2=致命。
#    守护层把 0 和 1 都当作"今天完事"，只有 2 才重试。
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 2

PY="$ROOT/.venv/bin/python"
[ -x "$PY" ] || PY="$(command -v python3)"

if [ -f "$ROOT/.pipeline.env" ]; then
  # shellcheck disable=SC1091
  set -a; . "$ROOT/.pipeline.env"; set +a
fi

TRADE_DATE="${1:-}"
if [ -z "$TRADE_DATE" ]; then
  # 取「仍可提交的最近运行日」而非「LA 明天」：后者在 BJ 白天手工触发时已过截止
  TRADE_DATE="$("$PY" -c 'import sys; sys.path.insert(0, "."); from src.pipeline.clock import next_submittable; print(next_submittable()[0])')" || {
    echo "无法推算运行日" >&2; exit 2; }
fi
shift || true

MAX_ATTEMPTS=3
rc=2
for attempt in $(seq 1 "$MAX_ATTEMPTS"); do
  echo "===== $(date -u '+%F %T UTC') 第 ${attempt}/${MAX_ATTEMPTS} 次｜运行日 ${TRADE_DATE} ====="
  "$PY" scripts/pipeline.py --date "$TRADE_DATE" "$@"
  rc=$?
  case "$rc" in
    0) echo "结果：正常"; exit 0 ;;
    # rc=1 表示文件已落地但有告警（如模板待核实项）。对守护层而言今天完事，
    # 返回 0 以免 systemd 反复重启；告警已写进 run json 与日志。
    1) echo "结果：完成但有告警（文件已落地）"; exit 0 ;;
    *) echo "结果：失败 rc=$rc" ;;
  esac
  if [ "$attempt" -lt "$MAX_ATTEMPTS" ]; then
    wait_s=$(( attempt * 300 ))
    echo "…${wait_s}s 后重试" >&2
    sleep "$wait_s"
  fi
done

echo "结果：${MAX_ATTEMPTS} 次均失败，需人工介入" >&2
exit "$rc"
