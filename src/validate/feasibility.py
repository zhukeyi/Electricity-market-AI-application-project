"""提交前可行性校验器 —— 独立于优化器的第二实现。

设计原则（为什么需要它）：
1. **手册惩罚极重**：任一时段越界 → 当日全天储能调度归零（手册 P17）。
   因此校验逻辑不能与优化器共用同一段代码，否则优化器的建模错误会被
   自身"验证"通过。本模块按手册公式**逐字重写**，只依赖 params.py。
2. **浮点边界必须留裕量**：若平台用严格不等号（`0.05 < SOC < 0.95`），
   LP 解恰好取到边界值 13.3 MWh 时，`13.300000000000001 > 13.3` 即可判越界。
   故校验器默认启用 eps 容差，并对输出曲线做"保守取整"后重放校验。
3. **量纲**：手册 P17 的 SOC 递推以占比给出，等价换算到 MWh 域为
     - 充电（Q ≤ 0）：dSOC = −Q·η      （充电入库 |Q|·η）
     - 放电（Q > 0）：dSOC = −Q/η      （放电出库 Q/η）
   与手册式 `SOC_{t+1} = SOC_t − (Q/Q_rated)·η` / `SOC_t − Q/(Q_rated·η)`
   乘以 Q_rated 后完全一致。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

from ..core.params import StorageParams, DEFAULT_STORAGE


# 默认容差：宁可判"不可行"再收敛，也不放过一个会归零的解。
DEFAULT_POWER_EPS_MW = 1e-9
DEFAULT_SOC_EPS_MWH = 1e-9

# 优化与输出时预留的安全裕量（远大于浮点噪声，又远小于经济影响）
DEFAULT_SOC_MARGIN_MWH = 0.01      # 相对 12.6 MWh 可用容量 ≈ 0.08%
DEFAULT_POWER_MARGIN_MW = 0.001    # 相对 7 MW ≈ 0.014%


@dataclass
class FeasibilityReport:
    feasible: bool
    violations: list[str] = field(default_factory=list)
    soc_trajectory_mwh: list[float] = field(default_factory=list)
    """24 个时段末 SOC（MWh），索引 0 对应 t=0。"""

    soc_start_mwh: float = 0.0
    soc_end_mwh: float = 0.0
    min_soc_mwh: float = 0.0
    max_soc_mwh: float = 0.0
    charge_mwh: float = 0.0
    discharge_mwh: float = 0.0

    def summary(self) -> str:
        if self.feasible:
            return (
                f"可行 | SOC [{self.min_soc_mwh:.3f}, {self.max_soc_mwh:.3f}] MWh "
                f"| 充 {self.charge_mwh:.2f} / 放 {self.discharge_mwh:.2f} MWh"
            )
        return "不可行 → 当日将归零:\n  - " + "\n  - ".join(self.violations)


def replay_soc(
    q_mw: Sequence[float],
    soc_start_mwh: float,
    params: StorageParams = DEFAULT_STORAGE,
) -> list[float]:
    """按手册 SOC 递推式重放，返回各时段末 SOC（MWh）。

    不抛异常、不做判断，纯状态推进，便于单独测试。
    """
    eta = params.efficiency
    soc = float(soc_start_mwh)
    traj: list[float] = []
    for q in q_mw:
        q = float(q)
        if q <= 0.0:      # 充电：入库 |q|·η
            soc -= q * eta
        else:             # 放电：出库 q/η
            soc -= q / eta
        traj.append(soc)
    return traj


def validate_dispatch(
    q_mw: Sequence[float],
    soc_start_mwh: float | None = None,
    params: StorageParams = DEFAULT_STORAGE,
    *,
    power_eps: float = DEFAULT_POWER_EPS_MW,
    soc_eps: float = DEFAULT_SOC_EPS_MWH,
    require_periods: bool = True,
) -> FeasibilityReport:
    """校验一条 24 时段申报曲线是否满足手册全部物理约束。

    Args:
        q_mw: 24 个时段净出力（MW），正数放电、负数充电。
        soc_start_mwh: 当日初始 SOC（MWh）；默认取手册首日 5%。
        power_eps / soc_eps: 容差，用于吸收浮点噪声。
        require_periods: 是否强制要求恰好 24 个时段。

    Returns:
        FeasibilityReport。`feasible=False` 意味着该曲线提交后当日必然归零。
    """
    p = params
    soc0 = p.soc_init_mwh if soc_start_mwh is None else float(soc_start_mwh)
    violations: list[str] = []

    q = [float(x) for x in q_mw]

    # --- 结构性检查 ---
    if require_periods and len(q) != p.periods_per_day:
        violations.append(f"时段数不符：{len(q)} ≠ {p.periods_per_day}")
    for t, val in enumerate(q):
        if not math.isfinite(val):
            violations.append(f"t={t}: 出力非有限值 ({val})")

    if violations:
        return FeasibilityReport(
            feasible=False, violations=violations, soc_start_mwh=soc0
        )

    # --- 逐时段：功率约束 + SOC 递推 + SOC 约束 ---
    eta = p.efficiency
    p_max = p.power_mw
    soc = soc0
    traj: list[float] = []
    charge_sum = 0.0
    discharge_sum = 0.0

    for t, val in enumerate(q):
        # (1) 功率约束 −P_max ≤ Q_t ≤ P_max
        if val > p_max + power_eps:
            violations.append(f"t={t}: 放电功率 {val:.6f} MW 超过上限 {p_max}")
        elif val < -p_max - power_eps:
            violations.append(f"t={t}: 充电功率 {val:.6f} MW 超过下限 {-p_max}")

        # (2) SOC 递推
        soc = soc - val * eta if val <= 0.0 else soc - val / eta
        traj.append(soc)

        if val <= 0.0:
            charge_sum += -val
        else:
            discharge_sum += val

        # (3) SOC 约束 SOC_min ≤ SOC_t ≤ SOC_max
        if soc > p.soc_max_mwh + soc_eps:
            violations.append(
                f"t={t}: SOC {soc:.6f} MWh 超过上限 {p.soc_max_mwh}"
            )
        elif soc < p.soc_min_mwh - soc_eps:
            violations.append(
                f"t={t}: SOC {soc:.6f} MWh 低于下限 {p.soc_min_mwh}"
            )

    return FeasibilityReport(
        feasible=not violations,
        violations=violations,
        soc_trajectory_mwh=traj,
        soc_start_mwh=soc0,
        soc_end_mwh=traj[-1] if traj else soc0,
        min_soc_mwh=min(traj) if traj else soc0,
        max_soc_mwh=max(traj) if traj else soc0,
        charge_mwh=charge_sum,
        discharge_mwh=discharge_sum,
    )


def _quantize_floor(x: float, step: float) -> float:
    """向零方向量化到 step 的整数倍（放电少放、充电少充）。"""
    n = math.floor(abs(x) / step + 1e-12)
    return math.copysign(n * step, x) if x else 0.0


def quantize_feasible(
    q_mw: Sequence[float],
    soc_start_mwh: float | None = None,
    params: StorageParams = DEFAULT_STORAGE,
    *,
    ndigits: int = 2,
) -> list[float]:
    """把连续解量化为可提交曲线，并**逐时段保证 SOC 落在合法区间内**。

    为什么不能简单逐时段截断：SOC 是累积量，各时段独立截断的误差会叠加。
    反例：LP 解为「先充 a 再放 b」恰好压在 SOC 下限，取整误差 δa、δb 满足
    δb < δa·η² 时，量化后 SOC 末端会低于下限 → 全天归零。
    故本函数在量化后按 SOC 空间**前向修复**：若该时段末 SOC 越界，直接
    反解需要回退的功率量并向上取整到步长，确保每一步都严格可行。

    Args:
        q_mw: 连续解（MW），正数放电、负数充电。
        ndigits: 提交精度，默认 2 位小数（0.01 MW）。
    """
    p = params
    step = 10.0 ** (-ndigits)
    eta = p.efficiency
    lo, hi = p.soc_min_mwh, p.soc_max_mwh
    tol = 1e-12

    soc = p.soc_init_mwh if soc_start_mwh is None else float(soc_start_mwh)
    if soc < lo - 1e-9 or soc > hi + 1e-9:
        raise ValueError(
            f"初始 SOC {soc:.6f} MWh 不在 [{lo}, {hi}] 内，无法构造可行曲线"
        )

    out: list[float] = []
    for t, x in enumerate(q_mw):
        x = float(x)
        q = _quantize_floor(x, step) if x else 0.0

        for _ in range(64):  # 收敛保护：正常情况下 1–2 轮
            nxt = soc - q * eta if q <= 0.0 else soc - q / eta
            if nxt > hi + tol:
                # 充电过多导致 SOC 越上限 → 减少充电量（q 向 0 靠）
                need = (nxt - hi) / eta
                q = min(0.0, q + math.ceil(need / step - 1e-12) * step)
                continue
            if nxt < lo - tol:
                # 放电过多导致 SOC 越下限 → 减少放电量（q 向 0 靠）
                need = (lo - nxt) * eta
                q = max(0.0, q - math.ceil(need / step - 1e-12) * step)
                continue
            break

        soc = soc - q * eta if q <= 0.0 else soc - q / eta
        out.append(round(q, ndigits))

    return out


def round_conservative(q_mw: Sequence[float], ndigits: int = 2) -> list[float]:
    """兼容旧接口：仅做向零量化，**不保证 SOC 可行**，请优先用 quantize_feasible。"""
    step = 10.0 ** (-ndigits)
    return [_quantize_floor(float(x), step) for x in q_mw]


def apply_deadband(
    q_mw: Sequence[float],
    soc_start_mwh: float | None = None,
    params: StorageParams = DEFAULT_STORAGE,
    *,
    deadband_mw: float = 0.05,
) -> tuple[list[float], FeasibilityReport]:
    """把微量动作归零，使提交曲线干净且仍可行。

    为什么需要：`quantize_feasible` 会把 LP 的极小残量（如 −0.005 MW）取整成
    `−0.01`，这类动作对收益贡献不足 $0.01，却让申报曲线出现无意义的小数，
    人工复核时难以判断是模型意图还是数值噪声。

    归零会改变 SOC 轨迹，故必须重放校验；若因此不可行（原解恰好贴边），
    按绝对值从大到小逐个恢复，直到可行，最差回退到原曲线。
    """
    p = params
    q = [float(x) for x in q_mw]
    soc0 = p.soc_init_mwh if soc_start_mwh is None else float(soc_start_mwh)

    cand = [0.0 if abs(x) < deadband_mw else x for x in q]
    rep = validate_dispatch(cand, soc0, p)
    if rep.feasible:
        return cand, rep

    order = sorted(
        (i for i, x in enumerate(q) if x and abs(x) < deadband_mw),
        key=lambda i: -abs(q[i]),
    )
    for i in order:
        cand[i] = q[i]
        rep = validate_dispatch(cand, soc0, p)
        if rep.feasible:
            return cand, rep

    return q, validate_dispatch(q, soc0, p)


def repair_to_feasible(
    q_mw: Sequence[float],
    soc_start_mwh: float | None = None,
    params: StorageParams = DEFAULT_STORAGE,
    *,
    max_rounds: int = 40,
    shrink: float = 0.9,
) -> tuple[list[float], FeasibilityReport]:
    """若曲线不可行，按比例收缩全部动作直至可行。

    这是最后一道保险：宁可少赚，不可归零。收缩系数 0.9 迭代，
    一般 1–3 轮即可收敛；若 40 轮仍不可行，返回全零曲线（保底可行）。
    """
    p = params
    q = [float(x) for x in q_mw]
    rep = validate_dispatch(q, soc_start_mwh, p)
    if rep.feasible:
        return q, rep

    for _ in range(max_rounds):
        q = [x * shrink for x in q]
        rep = validate_dispatch(q, soc_start_mwh, p)
        if rep.feasible:
            return q, rep

    zero = [0.0] * p.periods_per_day
    return zero, validate_dispatch(zero, soc_start_mwh, p)


def revenue(q_mw: Sequence[float], prices: Sequence[float]) -> float:
    """线上组日前收益 Σ p_t·Q_t·1h（手册 P24）。正数放电收入、负数充电支出。"""
    if len(q_mw) != len(prices):
        raise ValueError(f"长度不一致: q={len(q_mw)}, p={len(prices)}")
    return float(sum(float(q) * float(pr) for q, pr in zip(q_mw, prices)))


def realized_revenue(
    q_mw: Sequence[float],
    prices: Sequence[float],
    soc_start_mwh: float | None = None,
    params: StorageParams = DEFAULT_STORAGE,
) -> float:
    """"平台视角"收益：先校验，不可行则当日归零（收益 0）。"""
    rep = validate_dispatch(q_mw, soc_start_mwh, params)
    if not rep.feasible:
        return 0.0
    return revenue(q_mw, prices)
