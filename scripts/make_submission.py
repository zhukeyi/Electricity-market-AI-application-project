#!/usr/bin/env python3
"""线上组交易决策 · 单日提交文件生成（抓数 → 预测 → 优化 → 写盘 → 自检）。

比赛用法（每个运行日一次，北京时间 02:00 前完成）：

    python scripts/make_submission.py --date 2026-11-24 \
        --team "队名" --members "张三,李四" \
        --soc-from data/out/submission/2026-11-23.csv

`--soc-from` 读取前一日的提交文件，链式推出当日初始 SOC。比赛首日缺省 5%。

链路自检（用真实价格回放，验证端到端正确性，比赛时勿用）：

    python scripts/make_submission.py --date 2026-11-13 --actual --dry-run
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.params import DEFAULT_MARKET, DEFAULT_STORAGE  # noqa: E402
from src.forecast.baseline import PriceForecaster  # noqa: E402
from src.optimize.storage_lp import SubmissionDay, prepare_submission  # noqa: E402
from src.submit.writer import (  # noqa: E402
    TEMPLATE_NOTES,
    check_submission,
    read_submission,
    write_submission,
)
from src.validate.feasibility import apply_deadband, replay_soc, revenue  # noqa: E402

CSV = Path("data/raw/sp15_dam_lmp.csv")
OUTDIR = Path("data/out/submission")


def load_matrix(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    m = (
        df.pivot_table(index="date", columns="hour", values="lmp", aggfunc="last")
        .reindex(columns=range(1, 25))
        .dropna()
    )
    m.index = pd.Index(pd.to_datetime(m.index))
    return m.sort_index()


def soc_from_submission(path: Path, soc_start: float) -> float:
    """读昨日提交文件，重放到末时段，返回末态 SOC（MWh）。"""
    _, data, _ = read_submission(path)
    if not data:
        raise ValueError(f"{path} 无数据行")
    soc = soc_start
    for _, q in data:
        soc = replay_soc(q, soc)[-1]
    return soc


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True, help="加州运行日 YYYY-MM-DD")
    ap.add_argument("--team", default="xxxx队")
    ap.add_argument("--members", default="", help="逗号分隔，如 张三,李四")
    ap.add_argument("--method", default="blend", help="blend/persistence/median7/dow_median/shape_level")
    ap.add_argument("--soc-start", type=float, default=None, help="当日初始 SOC（MWh）")
    ap.add_argument("--soc-from", default=None, help="从前一日提交文件链式推算初始 SOC")
    ap.add_argument("--actual", action="store_true", help="用真实价格（仅链路自检，比赛禁用）")
    ap.add_argument("--dry-run", action="store_true", help="不写文件")
    ap.add_argument("--outdir", default=str(OUTDIR))
    ap.add_argument("--csv", default=str(CSV))
    args = ap.parse_args()

    target = pd.Timestamp(args.date)
    m = load_matrix(Path(args.csv))
    if target not in m.index:
        print(f"警告：{args.date} 不在历史数据中（最后一天 {m.index[-1].date()}）", file=sys.stderr)

    hist = m[m.index < target]
    if len(hist) == 0:
        print(f"错误：{args.date} 之前无历史数据", file=sys.stderr)
        return 2

    if args.actual:
        prices = m.loc[target].to_numpy(dtype=float)
        src = "实际价格（链路自检）"
    else:
        prices = PriceForecaster(method=args.method).predict(hist)
        src = f"预测 · {args.method}"

    # --- 初始 SOC ---
    if args.soc_start is not None:
        soc0 = args.soc_start
    elif args.soc_from:
        soc0 = soc_from_submission(Path(args.soc_from), DEFAULT_STORAGE.soc_init_mwh)
        print(f"初始 SOC 由 {args.soc_from} 链式推出：{soc0:.4f} MWh")
    else:
        soc0 = DEFAULT_STORAGE.soc_init_mwh

    day = prepare_submission(prices, soc0, DEFAULT_STORAGE, date=args.date)

    # 死区清理：把 < 0.05 MW 的数值残量归零，保持曲线可读且仍可行
    q_clean, rep_clean = apply_deadband(day.q_mw, soc0, DEFAULT_STORAGE)
    if q_clean != day.q_mw:
        day = SubmissionDay(
            date=day.date,
            q_mw=q_clean,
            prices=day.prices,
            expected_revenue=revenue(q_clean, day.prices),
            report=rep_clean,
            soc_start_mwh=rep_clean.soc_start_mwh,
            note=f"{day.note}; deadband",
        )

    print(f"\n运行日 {args.date}（加州）｜价格来源：{src}")
    print(f"  价格  最低 {min(prices):7.2f}  最高 {max(prices):7.2f}  均值 {np.mean(prices):7.2f}  $/MWh")
    print(f"  SOC   起 {day.soc_start_mwh:6.3f} → 末 {day.report.soc_end_mwh:6.3f} MWh"
          f"（区间 {day.report.min_soc_mwh:.3f}~{day.report.max_soc_mwh:.3f}）")
    print(f"  动作  充 {day.report.charge_mwh:6.2f} / 放 {day.report.discharge_mwh:6.2f} MWh"
          f"｜循环 {day.report.discharge_mwh / DEFAULT_STORAGE.capacity_mwh:.2f}")
    print(f"  收益  预期 ${day.expected_revenue:8.2f}  ({day.note})")
    print(f"  可行  {'是' if day.report.feasible else '否 —— 将归零！'}")
    if not day.report.feasible:
        for v in day.report.violations[:5]:
            print(f"        - {v}")

    nonzero = [(t, q) for t, q in enumerate(day.q_mw) if abs(q) > 1e-9]
    print(f"  曲线  {len(nonzero)}/24 时段有动作："
          + " ".join(f"{t}:00={q:+.2f}" for t, q in nonzero[:8])
          + (" …" if len(nonzero) > 8 else ""))

    if args.dry_run:
        print("\n[dry-run] 未写文件")
        return 0 if day.report.feasible else 1

    members = tuple(x.strip() for x in args.members.split(",") if x.strip())
    outdir = Path(args.outdir)
    fname = f"（{args.team}_线上组_交易决策）AI+电力市场应用大赛结果提交.csv"
    out = outdir / args.date / fname
    write_submission([day], out, team=args.team, members=members)

    rep = check_submission(out, soc_start_mwh=soc0)
    print(f"\n已生成 {out}")
    print(rep.summary())
    print("\n模板待核实项：")
    for n in TEMPLATE_NOTES:
        print(f"  · {n}")
    return 0 if rep.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
