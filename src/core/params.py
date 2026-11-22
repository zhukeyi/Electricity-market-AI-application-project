"""手册参数的单一事实来源（Single Source of Truth）。

所有数值均逐条对应《全国"AI+电力市场"应用大赛参赛手册》原文页码，
禁止在其它模块中硬编码储能参数。修改参数只改这里。

手册依据：
- P16  线下组交易决策：储能参数（容量/功率/效率/SOC 区间/首日 SOC/末日不约束）
- P17  功率约束、SOC 递推、SOC 约束、越限 → 全天归零规则
- P24  线上组交易决策：参数同线下组；收益 = Σ p_t^DA · Q_t^DA,battery
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass(frozen=True)
class StorageParams:
    """新型储能物理参数（手册 P16 / P24）。"""

    # --- 容量与功率 ---
    capacity_mwh: float = 14.0
    """储能容量 Q_rated，单位 MWh。手册 P16。"""

    power_mw: float = 7.0
    """每小时最大充放电功率 P_max，单位 MW。手册 P17。"""

    efficiency: float = 0.85
    """充放电单程效率 η。手册 P16/P17。"""

    # --- SOC 区间（以容量占比表示，手册给的是百分比） ---
    soc_min_frac: float = 0.05
    """SOC 下限 5%。手册 P17。"""

    soc_max_frac: float = 0.95
    """SOC 上限 95%。手册 P17。"""

    soc_init_frac: float = 0.05
    """比赛首日初始 SOC0 = 5%。手册 P16。"""

    allow_final_soc_free: bool = True
    """不约束最终日储能的荷电状态。手册 P16 —— 该条允许跨日联合优化。"""

    # --- 时间结构 ---
    periods_per_day: int = 24
    """每运行日 24 个时段，每小时一个。手册 P17/P24。"""

    # --- 派生量 ---
    @property
    def soc_min_mwh(self) -> float:
        return self.capacity_mwh * self.soc_min_frac

    @property
    def soc_max_mwh(self) -> float:
        return self.capacity_mwh * self.soc_max_frac

    @property
    def soc_init_mwh(self) -> float:
        return self.capacity_mwh * self.soc_init_frac

    @property
    def usable_mwh(self) -> float:
        return self.soc_max_mwh - self.soc_min_mwh

    @property
    def roundtrip_efficiency(self) -> float:
        return self.efficiency ** 2

    @property
    def min_profitable_spread_ratio(self) -> float:
        """保本峰谷价比：p_high / p_low 需大于该值才有正套利收益。"""
        return 1.0 / self.roundtrip_efficiency

    def to_dict(self) -> dict:
        d = asdict(self)
        d.update(
            soc_min_mwh=self.soc_min_mwh,
            soc_max_mwh=self.soc_max_mwh,
            soc_init_mwh=self.soc_init_mwh,
            usable_mwh=self.usable_mwh,
            roundtrip_efficiency=self.roundtrip_efficiency,
            min_profitable_spread_ratio=self.min_profitable_spread_ratio,
        )
        return d


DEFAULT_STORAGE = StorageParams()


@dataclass(frozen=True)
class MarketParams:
    """线上组市场与结算参数（手册 P24）。"""

    hub_node: str = "TH_SP15_GEN-APND"
    """CAISO SP15 交易枢纽节点名（手册 P21/P24 指定，OASIS 中的 RESOURCE_NAME）。"""

    market_run_id: str = "DAM"
    """日前市场。"""

    timezone: str = "America/Los_Angeles"
    """CAISO 报告时区为 PPT（Pacific Prevailing Time），即本地时间含夏令时。"""

    # --- 比赛期（加州运行日） ---
    contest_start: str = "2026-11-23"
    contest_end: str = "2026-12-06"

    # --- 提交窗口 ---
    submission_deadline_bj: str = "02:00"
    """北京时间 D 日 02:00 = 加州 D-1 日 10:00 前提交运行日 D 的申报。手册 P25。"""


DEFAULT_MARKET = MarketParams()


@dataclass
class DispatchResult:
    """单日调度结果（申报曲线 + 结算指标）。"""

    date: str
    prices: list[float]
    """24 时段日前价格 λ_t（$/MWh）。"""

    q_mw: list[float]
    """24 时段净出力申报，正数放电、负数充电（MW）。"""

    soc_start_mwh: float
    soc_end_mwh: float
    soc_trajectory_mwh: list[float]
    """24 个时段末 SOC。"""

    revenue: float
    """日前收益 Σ p_t·Q_t·1h（$）。"""

    charge_mwh: float = 0.0
    discharge_mwh: float = 0.0
    feasible: bool = True
    violations: list[str] = field(default_factory=list)
    solver_status: str = ""

    @property
    def cycles(self) -> float:
        """等效循环次数（按放电量/容量计）。"""
        return self.discharge_mwh / DEFAULT_STORAGE.capacity_mwh if DEFAULT_STORAGE.capacity_mwh else 0.0
