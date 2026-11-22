"""24 时段储能日前套利优化器（线上组交易决策核心）。

问题（手册 P24）：
    max  Σ_t p_t · Q_t · 1h        （Q_t 正数放电、负数充电）
    s.t. −P_max ≤ Q_t ≤ P_max
         SOC_{t+1} = SOC_t + g(Q_t),  g(Q) = −Q·η (Q≤0) / −Q/η (Q>0)
         SOC_min ≤ SOC_{t+1} ≤ SOC_max
         SOC_0 = 给定

建模要点（**这是全项目最关键的实现细节**）：

1. **必须用 MILP 强制充放互斥**。
   把 Q 拆成 Q = Qd − Qc（Qd, Qc ≥ 0）后，模型内的 SOC 变化量为
   `y_lp = Qc·η − Qd/η`；而平台按**净曲线**计算的是 `y_true = g(Qd − Qc)`。
   可证若 Qd, Qc 同时为正：
       y_lp − y_true = −min(Qd, Qc)·(1/η − η) = −min(Qd,Qc)·0.3256  < 0
   即 LP 的 SOC 轨迹**系统性低于真实值**。因 y<0 表示 SOC 下降，偏差方向
   在充电侧（y>0）意味着 **LP 低估了 SOC 的上升幅度**，会排出
   「模型认为贴在上限内、实际冲过 SOC 上限」的曲线 —— 提交即全天归零。
   数值实证：Qd=1, Qc=3, SOC₀=11.8 → LP 末端 13.1735（≤13.3，判可行），
   真实末端 13.5000（>13.3，越界）。独立校验器确认该净曲线不可行。
   加入二元变量 z_t 并约束 Qd_t ≤ P_max·z_t、Qc_t ≤ P_max·(1−z_t) 后，
   Qd·Qc = 0 恒成立，此时 `y_lp ≡ y_true`，建模**精确无偏**。
   （纯 LP 模式保留用于快速探索，但**禁止直接用于提交**，除非事后互斥检查通过。）

2. **安全裕量**：SOC 上下界与功率上限在求解时各内缩一个小量，
   避免最优解恰好压在边界、被平台严格不等号判为越界。

3. **末日 SOC**：手册 P16 称"不约束最终日储能的荷电状态"，但 P17 的逐时段
   SOC 约束并未豁免末时段。为消除歧义风险，默认对**所有**时段施加 SOC 区间
   约束（conservative_final_soc=True）。放弃的收益仅约 0.7 MWh×价差，
   与"归零一天"的代价完全不成比例。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
from scipy.optimize import milp, linprog, LinearConstraint, Bounds

from ..core.params import StorageParams, DEFAULT_STORAGE
from ..validate.feasibility import (
    DEFAULT_POWER_MARGIN_MW,
    DEFAULT_SOC_MARGIN_MWH,
    FeasibilityReport,
    quantize_feasible,
    validate_dispatch,
)


@dataclass
class Solution:
    q_mw: np.ndarray                  # 净出力曲线（T*24 或 24），正放负充
    soc_mwh: np.ndarray               # 各时段末 SOC（T*24）
    objective: float                  # 目标值 Σp·Q（$）
    status: str
    n_days: int = 1
    used_milp: bool = True
    has_simultaneous: bool = False     # 是否出现同时充放（纯 LP 模式的隐患标记）
    reports: list[FeasibilityReport] = field(default_factory=list)
    soc_start_mwh: float = 0.0

    def day_q(self, day: int) -> np.ndarray:
        return self.q_mw[day * 24 : (day + 1) * 24]

    def day_soc(self, day: int) -> np.ndarray:
        return self.soc_mwh[day * 24 : (day + 1) * 24]

    def day_soc_start(self, day: int) -> float:
        return self.soc_start_mwh if day == 0 else float(self.soc_mwh[day * 24 - 1])


def _as_price_matrix(prices) -> np.ndarray:
    arr = np.asarray(prices, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2:
        raise ValueError(f"价格需为 (T,24) 或 (24,)，收到 shape={arr.shape}")
    if arr.shape[1] != 24:
        raise ValueError(f"每日期望 24 个时段，收到 {arr.shape[1]}")
    if not np.all(np.isfinite(arr)):
        raise ValueError("价格含 NaN/Inf，请先清洗数据")
    return arr


def _build_soc_matrix(n: int, eta: float) -> sp.csr_matrix:
    """下三角全 1 矩阵 L，用于把逐时段出力累加成 SOC 偏移。"""
    # 传入稀疏矩阵而非 numpy 数组，避免 scipy 的 sparray 迁移弃用警告
    return sp.csr_matrix(sp.tril(sp.csr_matrix(np.ones((n, n), dtype=float))))


def optimize(
    prices,
    soc_start_mwh: float | None = None,
    params: StorageParams = DEFAULT_STORAGE,
    *,
    use_milp: bool = True,
    soc_margin_mwh: float = DEFAULT_SOC_MARGIN_MWH,
    power_margin_mw: float = DEFAULT_POWER_MARGIN_MW,
    conservative_final_soc: bool = True,
    time_limit_s: float = 60.0,
) -> Solution:
    """求解储能日前套利最优申报曲线。

    Args:
        prices: (T,24) 或 (24,) 的日前价格，单位 $/MWh。
        soc_start_mwh: 首时段前 SOC（MWh），默认手册 5%。
        use_milp: True 用 MILP 强制互斥（**提交必须用这个**）；
                  False 用 LP（快，但需事后检查 has_simultaneous）。
        soc_margin_mwh: SOC 边界内缩量，防浮点压线。
        power_margin_mw: 功率上限内缩量。
        conservative_final_soc: True 对末时段也施加 SOC 区间约束（推荐）。
    """
    P = _as_price_matrix(prices)
    T, H = P.shape
    n = T * H
    eta = params.efficiency

    soc0 = params.soc_init_mwh if soc_start_mwh is None else float(soc_start_mwh)
    lo = params.soc_min_mwh + soc_margin_mwh
    hi = params.soc_max_mwh - soc_margin_mwh
    if lo >= hi:
        raise ValueError(f"SOC 裕量过大导致区间为空: lo={lo}, hi={hi}")

    p_flat = P.reshape(-1)
    p_cap = max(params.power_mw - power_margin_mw, 0.0)

    # --- 变量布局: [Qd(n), Qc(n), z(n)] ---
    nv = 3 * n if use_milp else 2 * n
    c = np.concatenate([-p_flat, p_flat, np.zeros(n if use_milp else 0)])

    # --- 目标：min Σ(−p·Qd + p·Qc) ⟺ max Σ p·(Qd−Qc) ---
    L = _build_soc_matrix(n, eta)

    # SOC_{t+1} = soc0 − Σ_{s≤t}Qd_s/η + Σ_{s≤t}Qc_s·η
    # 上限:  −ΣQd/η + ΣQc·η ≤ hi − soc0
    # 下限:   ΣQd/η − ΣQc·η ≤ soc0 − lo
    zero_block = sp.csr_matrix((n, n))
    blocks = [-L / eta, L * eta]
    if use_milp:
        blocks.append(zero_block)

    A_hi = sp.hstack(blocks, format="csr")
    A_lo = sp.hstack([L / eta, -L * eta] + ([zero_block] if use_milp else []), format="csr")
    b_hi = np.full(n, hi - soc0)
    b_lo = np.full(n, soc0 - lo)

    if not conservative_final_soc:
        # 放宽最后一个时段的 SOC 约束（手册"末日不约束"存在歧义，默认不启用）
        A_hi = A_hi.tolil(); A_hi[-1, :] = 0; A_hi = A_hi.tocsr()
        A_lo = A_lo.tolil(); A_lo[-1, :] = 0; A_lo = A_lo.tocsr()

    A_ub = sp.vstack([A_hi, A_lo], format="csr")
    b_ub = np.concatenate([b_hi, b_lo])

    # --- 变量边界 ---
    if use_milp:
        # Qd ≤ p_cap·z  →  Qd − p_cap·z ≤ 0
        # Qc ≤ p_cap·(1−z) →  Qc + p_cap·z ≤ p_cap
        I = sp.identity(n, format="csr")
        A_mx = sp.vstack([
            sp.hstack([I, sp.csr_matrix((n, n)), -p_cap * I]),
            sp.hstack([sp.csr_matrix((n, n)), I, p_cap * I]),
        ], format="csr")
        A_ub = sp.vstack([A_ub, A_mx], format="csr")
        b_ub = np.concatenate([b_ub, np.zeros(n), np.full(n, p_cap)])
        lb = np.concatenate([np.zeros(2 * n), np.zeros(n)])
        ub = np.concatenate([np.full(2 * n, p_cap), np.ones(n)])
        integrality = np.concatenate([np.zeros(2 * n), np.ones(n)])
    else:
        lb = np.zeros(2 * n)
        ub = np.full(2 * n, p_cap)
        integrality = np.zeros(2 * n)

    bounds = Bounds(lb, ub)
    cons = [LinearConstraint(A_ub, -np.inf, b_ub)]

    if use_milp:
        res = milp(
            c=c, constraints=cons, integrality=integrality, bounds=bounds,
            options={"time_limit": time_limit_s, "disp": False, "mip_rel_gap": 1e-6},
        )
        status = {0: "optimal", 1: "time_limit", 2: "infeasible", 3: "unbounded", 4: "error"}.get(
            int(res.status), f"status_{res.status}"
        )
        x = res.x
    else:
        res = linprog(c=c, A_ub=A_ub, b_ub=b_ub, bounds=list(zip(lb, ub)), method="highs")
        status = "optimal" if res.status == 0 else f"lp_status_{res.status}"
        x = res.x

    if x is None:
        zero = np.zeros(n)
        return Solution(
            q_mw=zero, soc_mwh=np.full(n, soc0), objective=0.0,
            status=status, n_days=T, used_milp=use_milp, soc_start_mwh=soc0,
        )

    qd = np.maximum(x[:n], 0.0)
    qc = np.maximum(x[n : 2 * n], 0.0)
    has_simul = bool(np.any((qd > 1e-9) & (qc > 1e-9)))
    q = qd - qc

    # SOC 轨迹：优先用与平台一致的"净曲线重放"，保证与提交内容严格一致
    soc_traj = np.empty(n)
    soc = soc0
    for t in range(n):
        soc = soc - q[t] * eta if q[t] <= 0 else soc - q[t] / eta
        soc_traj[t] = soc

    # 逐日校验
    reports: list[FeasibilityReport] = []
    for d in range(T):
        s0 = soc0 if d == 0 else float(soc_traj[d * H - 1])
        reports.append(validate_dispatch(q[d * H : (d + 1) * H], s0, params))

    return Solution(
        q_mw=q, soc_mwh=soc_traj,
        objective=float(np.dot(p_flat, q)),
        status=status, n_days=T, used_milp=use_milp,
        has_simultaneous=has_simul, reports=reports, soc_start_mwh=soc0,
    )


@dataclass
class SubmissionDay:
    """可直接提交的单日结果（已取整、已复核）。"""

    date: str
    q_mw: list[float]
    prices: list[float]
    expected_revenue: float
    report: FeasibilityReport
    soc_start_mwh: float
    note: str = ""


def prepare_submission(
    prices_day,
    soc_start_mwh: float | None = None,
    params: StorageParams = DEFAULT_STORAGE,
    *,
    ndigits: int = 2,
    date: str = "",
) -> SubmissionDay:
    """求解 → SOC 空间安全量化 → 重放校验 → 必要时收缩修复，产出可提交曲线。

    这条流水线是提交前的唯一入口：**任何绕过它的路径都可能导致归零**。
    """
    from ..validate.feasibility import repair_to_feasible, revenue

    sol = optimize(prices_day, soc_start_mwh, params, use_milp=True)
    q = [float(v) for v in sol.q_mw]
    note = sol.status

    qr = quantize_feasible(q, soc_start_mwh, params, ndigits=ndigits)
    rep = validate_dispatch(qr, soc_start_mwh, params)
    if not rep.feasible:
        qr, rep = repair_to_feasible(qr, soc_start_mwh, params)
        note = f"{note}; quantized-infeasible→repaired"

    return SubmissionDay(
        date=date,
        q_mw=qr,
        prices=[float(v) for v in np.asarray(prices_day, dtype=float).reshape(-1)],
        expected_revenue=revenue(qr, np.asarray(prices_day, dtype=float).reshape(-1)),
        report=rep,
        soc_start_mwh=rep.soc_start_mwh,
        note=note,
    )
