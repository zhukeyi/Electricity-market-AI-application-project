"""回测引擎：多策略逐日滚动模拟 + 相对评分估算。

模拟口径严格对齐手册：
- 逐日独立申报，当日初始 SOC 来自前一日末状态（手册 P17）；
- 每条申报曲线先经 SOC 安全量化，再过校验器；**不可行当日收益记 0**（手册 P17）；
- 收益 = Σ p_t^DA · Q_t（手册 P24），不含实时市场与偏差回收（线上组无此二者）。

为什么要做「完美预见」与「诚实基准」两条线：
完美预见是任何策略的收益上界，但它永远无法实现；persistence（用昨日价格
优化、按今日价格结算）是可实现的诚实基准。两者之比即**捕获率**，才是
策略好坏的真正度量。同业在 CAISO SP15 上测得该捕获率为 95.1%。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Protocol

import numpy as np
import pandas as pd

from ..core.params import StorageParams, DEFAULT_STORAGE
from ..forecast.baseline import PriceForecaster
from ..optimize.storage_lp import optimize
from ..validate.feasibility import (
    quantize_feasible,
    validate_dispatch,
    revenue as calc_revenue,
)


class Planner(Protocol):
    """策略接口：给定历史价格与当前 SOC，产出目标日 24 时段申报曲线。"""

    name: str

    def plan(
        self,
        history: pd.DataFrame,
        target_date: str,
        soc_start_mwh: float,
        params: StorageParams,
    ) -> np.ndarray: ...


# ---------------------------------------------------------------- 策略实现


@dataclass
class OptimizeOnForecast:
    """用预测价格求 MILP 最优解。所有实用策略的主体。

    min_spread 为「价差过滤」：若预测曲线的 top-k 与 bottom-k 均价之差
    低于阈值，则不动作（全零曲线）。依据是 2025 全年回测显示，低价值日
    （PF<50）的期望收益为负（日均 −$27.5），不动作的 0 反而更优。
    k 取电池时长对应的时段数（14MWh/7MW ≈ 2h → k=2）。
    """

    forecaster: PriceForecaster = field(default_factory=PriceForecaster)
    min_spread: float = 0.0
    """价差过滤阈值（$/MWh）。0 表示不过滤。"""
    spread_k: int = 2
    name: str = ""

    def plan(self, history, target_date, soc_start_mwh, params):
        p_hat = self.forecaster.predict(history)
        if self.min_spread > 0 and self._spread(p_hat) < self.min_spread:
            return np.zeros(24)
        sol = optimize(p_hat, soc_start_mwh, params, use_milp=True)
        return np.asarray(sol.q_mw, dtype=float)

    def _spread(self, p: np.ndarray) -> float:
        s = np.sort(np.asarray(p, dtype=float))
        k = max(1, min(self.spread_k, len(s) // 2))
        return float(s[-k:].mean() - s[:k].mean())

    def __post_init__(self):
        if not self.name:
            base = f"opt-{self.forecaster.method}"
            self.name = base if self.min_spread <= 0 else f"{base}+sp{self.min_spread:g}"


@dataclass
class PerfectForesight:
    """用当日真实价格优化 —— 收益上界，不可实现，仅作基准。"""

    name: str = "PF(完美预见)"

    def plan(self, history, target_date, soc_start_mwh, params):
        raise NotImplementedError("PF 需要当日真实价格，由引擎直接注入")


@dataclass
class NaiveExtremes:
    """朴素规则：在预测最便宜的 k 小时满充、最贵的 k 小时满放。

    这是「不建模、只排序」的对照，用来量化优化器本身的增量价值。
    """

    forecaster: PriceForecaster = field(default_factory=PriceForecaster)
    k: int = 2
    name: str = ""

    def plan(self, history, target_date, soc_start_mwh, params):
        p = self.forecaster.predict(history)
        q = np.zeros(24)
        order = np.argsort(p)
        q[order[: self.k]] = -params.power_mw
        q[order[-self.k :]] = params.power_mw
        return q

    def __post_init__(self):
        if not self.name:
            self.name = f"naive-TB{self.k}"


@dataclass
class DoNothing:
    name: str = "不动作(0)"

    def plan(self, history, target_date, soc_start_mwh, params):
        return np.zeros(24)


# ---------------------------------------------------------------- 模拟


@dataclass
class BacktestResult:
    daily: pd.DataFrame
    """列：date, strategy, revenue, feasible, soc_start, soc_end, charge, discharge, pf_revenue"""

    meta: dict = field(default_factory=dict)

    def summary(self) -> pd.DataFrame:
        d = self.daily
        g = d.groupby("strategy", sort=False)
        out = pd.DataFrame({
            "天数": g.size(),
            "总收益$": g["revenue"].sum().round(2),
            "日均$": g["revenue"].mean().round(2),
            "中位$": g["revenue"].median().round(2),
            "标准差$": g["revenue"].std().round(2),
            "最差日$": g["revenue"].min().round(2),
            "最好日$": g["revenue"].max().round(2),
            "可行率%": (g["feasible"].mean() * 100).round(1),
            "捕获率%": (g.apply(
                lambda x: x["revenue"].sum() / x["pf_revenue"].sum() * 100
                if x["pf_revenue"].sum() > 0 else np.nan, include_groups=False
            )).round(1),
        })
        return out.sort_values("总收益$", ascending=False)


def _simulate_one(
    matrix: pd.DataFrame,
    dates: list[str],
    planner: Planner | None,
    params: StorageParams,
    *,
    perfect_foresight: bool = False,
    name: str = "",
    soc_init_mwh: float | None = None,
) -> pd.DataFrame:
    """单策略逐日模拟。"""
    soc = params.soc_init_mwh if soc_init_mwh is None else soc_init_mwh
    rows = []

    for i, d in enumerate(dates):
        actual = matrix.loc[d].to_numpy(dtype=float)
        history = matrix.loc[dates[:i]] if i > 0 else matrix.iloc[:0]

        if perfect_foresight:
            sol = optimize(actual, soc, params, use_milp=True)
            raw = np.asarray(sol.q_mw, dtype=float)
            tag = name or "PF(完美预见)"
        else:
            raw = np.asarray(planner.plan(history, d, soc, params), dtype=float)
            tag = name or planner.name

        # 统一后处理：SOC 安全量化 → 校验
        q = np.asarray(quantize_feasible(raw, soc, params, ndigits=2), dtype=float)
        rep = validate_dispatch(q, soc, params)

        if rep.feasible:
            rev = float(calc_revenue(q, actual))
            soc_next = rep.soc_end_mwh
        else:
            # 手册 P17：任一时段越界 → 全天不充不放，收益 0，SOC 不变
            rev = 0.0
            soc_next = soc

        rows.append({
            "date": d,
            "strategy": tag,
            "revenue": rev,
            "feasible": bool(rep.feasible),
            "soc_start": soc,
            "soc_end": soc_next,
            "charge": rep.charge_mwh if rep.feasible else 0.0,
            "discharge": rep.discharge_mwh if rep.feasible else 0.0,
        })
        soc = soc_next

    return pd.DataFrame(rows)


def run_backtest(
    matrix: pd.DataFrame,
    planners: list[Planner],
    params: StorageParams = DEFAULT_STORAGE,
    *,
    start: str | None = None,
    end: str | None = None,
    include_pf: bool = True,
) -> BacktestResult:
    """在指定日期区间上逐日滚动回测全部策略。

    Args:
        matrix: index=日期字符串（升序），columns=1..24 的价格宽表。
        planners: 策略列表。
        start / end: 闭区间；None 表示不限制。
    """
    m = matrix.copy()
    m.index = m.index.astype(str)
    m = m.sort_index()
    if start:
        m = m[m.index >= start]
    if end:
        m = m[m.index <= end]
    if len(m) == 0:
        raise ValueError("回测区间为空")

    dates = list(m.index)

    # PF 基准（每日单独求上界，SOC 按同一递推推进，保证可比）
    pf = _simulate_one(m, dates, None, params, perfect_foresight=True, name="PF(完美预见)")

    frames = [pf] if include_pf else []
    for pl in planners:
        frames.append(_simulate_one(m, dates, pl, params))

    daily = pd.concat(frames, ignore_index=True)
    pf_map = pf.set_index("date")["revenue"]
    daily["pf_revenue"] = daily["date"].map(pf_map).fillna(0.0)

    return BacktestResult(
        daily=daily,
        meta={"start": dates[0], "end": dates[-1], "n_days": len(dates)},
    )


# ---------------------------------------------------------------- 评分估算


@dataclass
class ScoreScenario:
    """参赛者捕获率分布假设。"""

    label: str
    mu_capture: float
    sigma_capture: float
    s2: float = 15.0
    """手册未公开 s₂；取 15 为中性假设，并做敏感性。"""


SCENARIOS = [
    ScoreScenario("保守（对手强）", mu_capture=0.85, sigma_capture=0.08),
    ScoreScenario("中性", mu_capture=0.75, sigma_capture=0.12),
    ScoreScenario("乐观（对手弱）", mu_capture=0.65, sigma_capture=0.15),
]


def estimate_score(capture_rate: float, sc: ScoreScenario) -> float:
    """相对评分下的单日/期望得分估算。

    手册：s_d = min{100, max[0, 60 + s₂·(μ − metric)/σ]}
    交易决策指标是收益（越高越好），故符号取反：
        s = 60 + s₂·(R_us − μ_R)/σ_R
    若全体参赛者收益 ≈ PF × c（c 为各自捕获率），则
        (R_us − μ_R)/σ_R = (c_us − μ_c)/σ_c
    即得分只取决于**捕获率相对分布的位置**，与当日绝对收益水平无关。
    这也解释了为什么低价值日（12 月）与高价值日（3 月）的得分机制相同。
    """
    z = (capture_rate - sc.mu_capture) / sc.sigma_capture if sc.sigma_capture > 0 else 0.0
    return float(min(100.0, max(0.0, 60.0 + sc.s2 * z)))


def score_table(capture_rates: dict[str, float], scenarios=None) -> pd.DataFrame:
    scenarios = scenarios or SCENARIOS
    data = {
        sc.label: {name: round(estimate_score(c, sc), 1) for name, c in capture_rates.items()}
        for sc in scenarios
    }
    return pd.DataFrame(data)
