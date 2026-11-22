#!/usr/bin/env python3
"""补充实验：多日滚动优化、价格收缩鲁棒化、SOC 传递、2026 临场校准。"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.backtest.engine import DoNothing, OptimizeOnForecast, run_backtest  # noqa: E402
from src.core.params import DEFAULT_STORAGE  # noqa: E402
from src.forecast.baseline import PriceForecaster  # noqa: E402
from src.optimize.storage_lp import optimize  # noqa: E402
from src.validate.feasibility import quantize_feasible, validate_dispatch  # noqa: E402

pd.set_option("display.width", 220)


@dataclass
class HorizonPlanner:
    """多日滚动优化：一次规划 H 天，只执行第 1 天，次日重新规划。

    依据手册 P16「不约束最终日储能的荷电状态」，跨日联合优化在理论上优于
    逐日贪心；但收益大小需实测，且依赖未来日预测精度。
    """

    forecaster: PriceForecaster
    horizon: int = 2
    name: str = ""

    def __post_init__(self):
        if not self.name:
            self.name = f"opt-{self.forecaster.method}-h{self.horizon}"

    def plan(self, history, target_date, soc_start_mwh, params):
        p_hat = self.forecaster.predict(history)
        pm = np.tile(p_hat, (self.horizon, 1))
        sol = optimize(pm, soc_start_mwh, params, use_milp=True,
                       conservative_final_soc=True)
        return np.asarray(sol.q_mw[:24], dtype=float)


@dataclass
class ShrunkForecast:
    """价格收缩：p' = mean + α·(p − mean)，降低对极端值的误判。"""

    forecaster: PriceForecaster
    alpha: float = 1.0
    name: str = ""

    def __post_init__(self):
        if not self.name:
            self.name = f"opt-blend-shrink{self.alpha:g}"

    def plan(self, history, target_date, soc_start_mwh, params):
        p = self.forecaster.predict(history)
        mu = float(np.mean(p))
        pr = mu + self.alpha * (p - mu)
        sol = optimize(pr, soc_start_mwh, params, use_milp=True)
        return np.asarray(sol.q_mw, dtype=float)


def tab(daily: pd.DataFrame) -> pd.DataFrame:
    g = daily.groupby("strategy", sort=False)
    pf = g["pf_revenue"].sum()
    return pd.DataFrame({
        "总收益$": g["revenue"].sum().round(1),
        "日均$": g["revenue"].mean().round(2),
        "最差日$": g["revenue"].min().round(1),
        "负收益日": g["revenue"].apply(lambda s: int((s < 0).sum())),
        "可行率%": (g["feasible"].mean() * 100).round(1),
        "捕获率%": (g["revenue"].sum() / pf * 100).round(2),
    }).sort_values("捕获率%", ascending=False)


def main() -> int:
    m = pd.read_csv(ROOT / "data/raw/sp15_dam_matrix.csv", index_col=0)
    m.columns = [int(c) for c in m.columns]
    m.index = m.index.astype(str)
    m = m.sort_index()
    P = DEFAULT_STORAGE

    print("=" * 110)
    print("实验 5　多日滚动优化 vs 逐日（opt-blend，2025 比赛期同期 11-23~12-06）")
    print("=" * 110)
    pls = [
        OptimizeOnForecast(PriceForecaster("blend"), name="逐日(h1)"),
        HorizonPlanner(PriceForecaster("blend"), 2),
        HorizonPlanner(PriceForecaster("blend"), 3),
    ]
    r = run_backtest(m, pls, P, start="2025-10-01", end="2025-12-06")
    d = r.daily[r.daily.date >= "2025-11-23"]
    print(tab(d).to_string())
    print()

    print("=" * 110)
    print("实验 6　价格收缩 α 扫描（2024 与 2025 比赛期同期）")
    print("=" * 110)
    rows = []
    for year in (2024, 2025):
        for a in (0.7, 0.85, 1.0, 1.15):
            pl = ShrunkForecast(PriceForecaster("blend"), alpha=a)
            r = run_backtest(m, [pl], P, start=f"{year}-10-01", end=f"{year}-12-06")
            d = r.daily[r.daily.date >= f"{year}-11-23"]
            t = tab(d)
            rows.append({"年份": year, "alpha": a,
                         "收益$": float(t.loc[pl.name, "总收益$"]),
                         "捕获率%": float(t.loc[pl.name, "捕获率%"]),
                         "最差日$": float(t.loc[pl.name, "最差日$"])})
    print(pd.DataFrame(rows).to_string(index=False))
    print()

    print("=" * 110)
    print("实验 7　2026 年最新市场状态（临场校准：比赛期临近，价差水平是否变化）")
    print("=" * 110)
    for label, s, e in [
        ("2024-10-01~11-07", "2024-10-01", "2024-11-07"),
        ("2025-10-01~11-07", "2025-10-01", "2025-11-07"),
        ("2026-10-01~11-07", "2026-10-01", "2026-11-07"),
    ]:
        sub = m[(m.index >= s) & (m.index <= e)]
        if len(sub) == 0:
            continue
        v = sub.to_numpy(float)
        srt = np.sort(v, axis=1)
        tb2 = srt[:, -2:].mean(1) - srt[:, :2].mean(1)
        pl = OptimizeOnForecast(PriceForecaster("blend"), name="opt-blend")
        r = run_backtest(m, [pl], P, start=s, end=e)
        t = tab(r.daily)
        row = t.loc["opt-blend"]
        print(f"{label}: {len(sub)}天  均价 ${v.mean():.2f}  负价 {(v<0).mean()*100:.1f}%  "
              f"TB2中位 ${np.median(tb2):.1f}  |  opt-blend 捕获率 {row['捕获率%']:.1f}%  "
              f"日均 ${row['日均$']:.0f}")
    print()

    print("=" * 110)
    print("实验 8　SOC 逐日传递稳定性（opt-blend 2025 全年，SOC 是否漂移）")
    print("=" * 110)
    r = run_backtest(m, [OptimizeOnForecast(PriceForecaster("blend"), name="opt-blend")],
                     P, start="2025-01-01", end="2025-12-31")
    d = r.daily
    print(f"SOC 起点范围 [{d.soc_start.min():.3f}, {d.soc_start.max():.3f}] MWh "
          f"（合法区间 [0.70, 13.30]）")
    print(f"SOC 终点范围 [{d.soc_end.min():.3f}, {d.soc_end.max():.3f}] MWh")
    print(f"越界次数: {(d.soc_end < 0.6999).sum() + (d.soc_end > 13.3001).sum()}  "
          f"不可行次数: {(~d.feasible).sum()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
