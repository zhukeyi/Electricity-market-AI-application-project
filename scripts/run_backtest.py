#!/usr/bin/env python3
"""效果测算主脚本：多策略回测 + 相对评分估算。

用法：
    PYTHONPATH=. python scripts/run_backtest.py --start 2025-01-01 --end 2025-12-31
    PYTHONPATH=. python scripts/run_backtest.py --start 2024-11-01 --end 2025-12-07 \
        --eval-start 2025-11-23 --eval-end 2025-12-06
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.backtest.engine import (  # noqa: E402
    DoNothing,
    NaiveExtremes,
    OptimizeOnForecast,
    run_backtest,
    score_table,
)
from src.core.params import DEFAULT_STORAGE  # noqa: E402
from src.forecast.baseline import PriceForecaster  # noqa: E402

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 40)


def build_planners() -> list:
    methods = ["persistence", "median7", "median14", "dow_median", "blend", "shape_level"]
    planners = [OptimizeOnForecast(PriceForecaster(method=m)) for m in methods]
    planners.append(NaiveExtremes(PriceForecaster(method="persistence"), k=2, name="naive-TB2"))
    planners.append(NaiveExtremes(PriceForecaster(method="blend"), k=3, name="naive-TB3"))
    planners.append(DoNothing())
    return planners


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix", default="data/raw/sp15_dam_matrix.csv")
    ap.add_argument("--start", default="2025-01-01")
    ap.add_argument("--end", default="2025-12-31")
    ap.add_argument("--eval-start", default=None, help="统计窗口起点（回测仍从 --start 开始）")
    ap.add_argument("--eval-end", default=None)
    ap.add_argument("--out", default="data/out/backtest")
    args = ap.parse_args()

    m = pd.read_csv(ROOT / args.matrix, index_col=0)
    m.columns = [int(c) for c in m.columns]
    m.index = m.index.astype(str)

    planners = build_planners()
    res = run_backtest(m, planners, DEFAULT_STORAGE, start=args.start, end=args.end)

    daily = res.daily
    if args.eval_start:
        daily = daily[daily["date"] >= args.eval_start]
    if args.eval_end:
        daily = daily[daily["date"] <= args.eval_end]

    print("=" * 110)
    print(f"回测区间 {res.meta['start']} ~ {res.meta['end']}（{res.meta['n_days']} 天）"
          f"　统计窗口 {daily['date'].min()} ~ {daily['date'].max()}（{daily['date'].nunique()} 天）")
    print("=" * 110)
    print(res.__class__.__name__ and "")
    print(BacktestResult_print(daily))
    print()

    # 捕获率 → 评分估算
    pf_sum = daily.groupby("strategy")["pf_revenue"].sum()
    rev_sum = daily.groupby("strategy")["revenue"].sum()
    cap = (rev_sum / pf_sum).dropna().to_dict()
    sc = score_table({k: v for k, v in cap.items() if k != "PF(完美预见)"})
    print("相对评分估算（捕获率 → 得分；s₂=15；得分与绝对收益水平无关，只看相对分布位置）")
    print(sc.to_string())
    print()

    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    daily.to_csv(out / "daily.csv", index=False)
    sc.to_csv(out / "score_estimate.csv")
    print(f"明细已写入 {out}")
    return 0


def BacktestResult_print(daily: pd.DataFrame) -> str:
    g = daily.groupby("strategy", sort=False)
    pf = daily.groupby("strategy")["pf_revenue"].sum()
    tbl = pd.DataFrame({
        "天数": g.size(),
        "总收益$": g["revenue"].sum().round(1),
        "日均$": g["revenue"].mean().round(2),
        "中位$": g["revenue"].median().round(2),
        "日标准差$": g["revenue"].std().round(2),
        "最差日$": g["revenue"].min().round(1),
        "可行率%": (g["feasible"].mean() * 100).round(1),
        "捕获率%": (g["revenue"].sum() / pf * 100).round(2),
    })
    return tbl.sort_values("总收益$", ascending=False).to_string()


if __name__ == "__main__":
    raise SystemExit(main())
