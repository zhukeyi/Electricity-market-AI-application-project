#!/usr/bin/env python3
"""系统级实验：跨年窗口验证 + 价差过滤网格 + 稳健性扫描。

产出用于技术方案的数据支撑，不写死任何结论。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
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

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 40)


def load_matrix() -> pd.DataFrame:
    m = pd.read_csv(ROOT / "data/raw/sp15_dam_matrix.csv", index_col=0)
    m.columns = [int(c) for c in m.columns]
    m.index = m.index.astype(str)
    return m.sort_index()


def tab(daily: pd.DataFrame) -> pd.DataFrame:
    g = daily.groupby("strategy", sort=False)
    pf = g["pf_revenue"].sum()
    out = pd.DataFrame({
        "总收益$": g["revenue"].sum().round(1),
        "日均$": g["revenue"].mean().round(2),
        "日标准差$": g["revenue"].std().round(2),
        "最差日$": g["revenue"].min().round(1),
        "负收益日": (g["revenue"].apply(lambda s: int((s < 0).sum()))),
        "可行率%": (g["feasible"].mean() * 100).round(1),
        "捕获率%": (g["revenue"].sum() / pf * 100).round(2),
    })
    return out.sort_values("捕获率%", ascending=False)


def window(m, start_bt, eval_s, eval_e, planners, params=DEFAULT_STORAGE):
    res = run_backtest(m, planners, params, start=start_bt, end=eval_e)
    d = res.daily[(res.daily.date >= eval_s) & (res.daily.date <= eval_e)]
    return d


def main() -> int:
    m = load_matrix()
    params = DEFAULT_STORAGE

    # ---------- 实验 1：跨年窗口稳健性 ----------
    print("=" * 120)
    print("实验 1　跨年同期验证（比赛期 11-23 ~ 12-06，用 2023/2024/2025 三个历史同期）")
    print("=" * 120)

    planners = [
        OptimizeOnForecast(PriceForecaster("blend")),
        OptimizeOnForecast(PriceForecaster("median7")),
        OptimizeOnForecast(PriceForecaster("median14")),
        OptimizeOnForecast(PriceForecaster("persistence")),
        NaiveExtremes(PriceForecaster("blend"), k=3, name="naive-TB3"),
        DoNothing(),
    ]

    rows = []
    for year in (2024, 2025):
        s_eval = f"{year}-11-23"
        e_eval = f"{year}-12-06"
        d = window(m, f"{year}-10-01", s_eval, e_eval, planners, params)
        t = tab(d).reset_index()
        t.insert(0, "窗口", f"{year}-11-23~12-06")
        rows.append(t)
    cross = pd.concat(rows, ignore_index=True)
    print(cross.to_string(index=False))
    print()

    # ---------- 实验 2：价差过滤网格 ----------
    print("=" * 120)
    print("实验 2　价差过滤阈值网格（opt-blend，阈值 $/MWh；k=2 对应 2h 电池）")
    print("=" * 120)

    grid_rows = []
    for thr in (0, 10, 20, 30, 40, 50):
        pl = OptimizeOnForecast(PriceForecaster("blend"), min_spread=float(thr),
                                name=f"sp{thr}")
        # 全年
        r_full = run_backtest(m, [pl], params, start="2025-01-01", end="2025-12-31")
        t_full = tab(r_full.daily)
        cap_full = float(t_full.loc[f"sp{thr}", "捕获率%"])
        rev_full = float(t_full.loc[f"sp{thr}", "总收益$"])
        neg_full = int(t_full.loc[f"sp{thr}", "负收益日"])
        # 比赛期同期
        d_s = window(m, "2025-10-01", "2025-11-23", "2025-12-06", [pl], params)
        ts = tab(d_s)
        grid_rows.append({
            "阈值": thr,
            "全年总收益$": rev_full,
            "全年捕获率%": cap_full,
            "全年负收益日": neg_full,
            "比赛期收益$": float(ts.loc[f"sp{thr}", "总收益$"]),
            "比赛期捕获率%": float(ts.loc[f"sp{thr}", "捕获率%"]),
        })
    print(pd.DataFrame(grid_rows).to_string(index=False))
    print()

    # ---------- 实验 3：最优阈值在两年比赛期的稳定性 ----------
    print("=" * 120)
    print("实验 3　阈值在 2024 与 2025 比赛期同期的最优性是否一致")
    print("=" * 120)
    stab = []
    for year in (2024, 2025):
        for thr in (0, 20, 30, 40):
            pl = OptimizeOnForecast(PriceForecaster("blend"), min_spread=float(thr), name=f"sp{thr}")
            d = window(m, f"{year}-10-01", f"{year}-11-23", f"{year}-12-06", [pl], params)
            t = tab(d)
            stab.append({
                "年份": year, "阈值": thr,
                "收益$": float(t.loc[f"sp{thr}", "总收益$"]),
                "捕获率%": float(t.loc[f"sp{thr}", "捕获率%"]),
                "最差日$": float(t.loc[f"sp{thr}", "最差日$"]),
            })
    print(pd.DataFrame(stab).to_string(index=False))
    print()

    # ---------- 实验 4：评分情景 ----------
    print("=" * 120)
    print("实验 4　相对评分估算（基于 2025 比赛期同期捕获率）")
    print("=" * 120)
    d = window(m, "2025-10-01", "2025-11-23", "2025-12-06", planners, params)
    g = d.groupby("strategy")
    cap = (g["revenue"].sum() / g["pf_revenue"].sum()).dropna().to_dict()
    cap.pop("不动作(0)", None)
    print(score_table(cap).to_string())
    print()

    out = ROOT / "data/out"
    out.mkdir(parents=True, exist_ok=True)
    cross.to_csv(out / "exp1_cross_year.csv", index=False)
    pd.DataFrame(grid_rows).to_csv(out / "exp2_spread_grid.csv", index=False)
    pd.DataFrame(stab).to_csv(out / "exp3_threshold_stability.csv", index=False)
    print(f"结果已写入 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
