#!/usr/bin/env python3
"""每日提交流水线 · 命令行入口。

正常跑（自动判定运行日 = LA 明天，对应 BJ 明天 02:00 截止）：

    python scripts/pipeline.py --team "队名" --members "张三,李四"

指定运行日 / 只看不写 / 强制重算：

    python scripts/pipeline.py --date 2026-11-24 --dry-run
    python scripts/pipeline.py --date 2026-11-24 --force

跳过抓数（数据已就绪，纯复算）：

    python scripts/pipeline.py --no-fetch --date 2026-11-24

退出码：0 正常；1 完成但有告警；2 致命失败；3 已有实例在跑。
"""
from __future__ import annotations

import argparse
import datetime as dt
import logging
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.pipeline import envfile  # noqa: E402
from src.pipeline.clock import schedule as make_schedule  # noqa: E402
from src.pipeline.data import health_report  # noqa: E402
from src.pipeline.runner import DEFAULT_CSV, DEFAULT_OUTDIR, DEFAULT_RUNDIR, Config, run  # noqa: E402

LOG_FMT = "%(asctime)s %(levelname)-7s %(name)s | %(message)s"


def setup_logging(rundir: Path, verbose: bool) -> None:
    rundir.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter(LOG_FMT, datefmt="%Y-%m-%d %H:%M:%S")
    root = logging.getLogger()
    root.setLevel(logging.DEBUG if verbose else logging.INFO)
    root.handlers.clear()

    con = logging.StreamHandler(sys.stdout)
    con.setLevel(logging.DEBUG if verbose else logging.INFO)
    con.setFormatter(fmt)
    root.addHandler(con)

    fh = logging.FileHandler(rundir / "pipeline.log", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    root.addHandler(fh)


def parse_date(s: str | None) -> dt.date | None:
    return dt.date.fromisoformat(s) if s else None


def main(argv: list[str] | None = None) -> int:
    # 先读配置，argparse 的 default 才会取到正确的值（环境变量优先级高于文件）
    envfile.load(ROOT / ".pipeline.env")

    ap = argparse.ArgumentParser(description="每日提交流水线")
    ap.add_argument("--date", help="加州运行日 YYYY-MM-DD；缺省为 LA 明天")
    ap.add_argument("--team", default=os.getenv("CSG_TEAM", "xxxx队"))
    ap.add_argument("--members", default=os.getenv("CSG_MEMBERS", ""), help="逗号分隔")
    ap.add_argument("--method", default=os.getenv("CSG_METHOD", "blend"),
                    help="blend/persistence/median7/median14/dow_median/shape_level")

    ap.add_argument("--csv", default=str(DEFAULT_CSV))
    ap.add_argument("--outdir", default=str(DEFAULT_OUTDIR))
    ap.add_argument("--rundir", default=str(DEFAULT_RUNDIR))

    ap.add_argument("--no-fetch", action="store_true", help="跳过抓数")
    ap.add_argument("--transport", default="auto", choices=["auto", "direct", "ssh"])
    ap.add_argument("--history-start", default="2024-01-01",
                    help="首次抓数的起始日（本地无数据时生效）")

    ap.add_argument("--dry-run", action="store_true", help="不写文件")
    ap.add_argument("--force", action="store_true", help="覆盖已存在的文件")
    ap.add_argument("--allow-late", action="store_true", help="已过截止时间仍执行")
    ap.add_argument("--soc-start", type=float, default=None, help="覆盖初始 SOC（MWh）")
    ap.add_argument("--use-actual", action="store_true", help="用真实价格（彩排，勿用于比赛）")
    ap.add_argument("--webhook", default=os.getenv("CSG_WEBHOOK", ""),
                    help="通知 webhook；也可用环境变量 CSG_WEBHOOK")
    ap.add_argument("--check-data", action="store_true", help="只做数据体检后退出")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    rundir = Path(args.rundir)
    setup_logging(rundir, args.verbose)

    if args.check_data:
        rep = health_report(args.csv)
        print(f"数据 {args.csv}")
        print(f"  {rep['rows']} 行 / {rep['days']} 天 / {rep.get('first_date')} ~ {rep.get('last_date')}")
        for p in rep["problems"]:
            print(f"  ! {p}")
        if not rep["problems"]:
            print("  无异常")
        return 0

    members = tuple(x.strip() for x in args.members.split(",") if x.strip())
    cfg = Config(
        trade_date=parse_date(args.date),
        team=args.team,
        members=members,
        method=args.method,
        csv_path=Path(args.csv),
        outdir=Path(args.outdir),
        rundir=rundir,
        do_fetch=not args.no_fetch,
        transport=args.transport,
        history_start=parse_date(args.history_start) or dt.date(2024, 1, 1),
        dry_run=args.dry_run,
        force=args.force,
        allow_late=args.allow_late,
        soc_start=args.soc_start,
        use_actual=args.use_actual,
        webhook=args.webhook or None,
    )

    sch = make_schedule(cfg.trade_date)
    logging.getLogger("pipeline").info(
        "触发：运行日 %s｜BJ 现在 %s｜LA 现在 %s",
        sch.trade_date, sch.now_bj.strftime("%F %T"), sch.now_la.strftime("%F %T"),
    )

    try:
        rep = run(cfg)
    except Exception as exc:  # noqa: BLE001
        logging.getLogger("pipeline").error("未捕获异常：%s", exc, exc_info=True)
        return 2

    print()
    print(rep.summary())

    if rep.fatal or not rep.ok:
        return 2
    return 0 if rep.clean else 1


if __name__ == "__main__":
    raise SystemExit(main())
