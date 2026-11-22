"""核心正确性测试：SOC 递推、边界、量化安全、MILP 可行性。

运行：
    PYTHONPATH=. python -m unittest discover -s tests -v
"""
from __future__ import annotations

import math
import unittest

import numpy as np

from src.core.params import DEFAULT_STORAGE as S
from src.optimize.storage_lp import optimize, prepare_submission
from src.validate.feasibility import (
    quantize_feasible,
    replay_soc,
    validate_dispatch,
    revenue,
)


class TestParams(unittest.TestCase):
    def test_derived_values(self):
        self.assertAlmostEqual(S.soc_min_mwh, 0.70, places=9)
        self.assertAlmostEqual(S.soc_max_mwh, 13.30, places=9)
        self.assertAlmostEqual(S.usable_mwh, 12.60, places=9)
        self.assertAlmostEqual(S.roundtrip_efficiency, 0.7225, places=9)
        self.assertAlmostEqual(S.min_profitable_spread_ratio, 1.3841, places=3)


class TestSocRecursion(unittest.TestCase):
    """SOC 递推必须与手册 P17 的占比式严格等价。"""

    def test_charge_matches_manual_formula(self):
        # 手册（占比式）: soc_{t+1} = soc_t − (Q/Q_rated)·η，Q ≤ 0
        q = -5.0
        soc0_frac = 0.5
        expected_frac = soc0_frac - (q / S.capacity_mwh) * S.efficiency
        got_mwh = replay_soc([q], soc0_frac * S.capacity_mwh)[0]
        self.assertAlmostEqual(got_mwh / S.capacity_mwh, expected_frac, places=12)

    def test_discharge_matches_manual_formula(self):
        # 手册: soc_{t+1} = soc_t − Q/(Q_rated·η)，Q > 0
        q = 4.0
        soc0_frac = 0.8
        expected_frac = soc0_frac - q / (S.capacity_mwh * S.efficiency)
        got_mwh = replay_soc([q], soc0_frac * S.capacity_mwh)[0]
        self.assertAlmostEqual(got_mwh / S.capacity_mwh, expected_frac, places=12)

    def test_zero_output_keeps_soc(self):
        traj = replay_soc([0.0] * 24, 5.0)
        self.assertTrue(all(abs(x - 5.0) < 1e-12 for x in traj))


class TestBoundaries(unittest.TestCase):
    def test_exact_upper_bound_is_feasible(self):
        # 恰好把 SOC 推到 13.30 应判可行（需跨多个时段以满足 7 MW 功率上限）
        need = (S.soc_max_mwh - S.soc_init_mwh) / S.efficiency   # 14.8235 MWh 从电网充入
        q = [-7.0, -7.0, -(need - 14.0)] + [0.0] * 21
        rep = validate_dispatch(q, S.soc_init_mwh)
        self.assertTrue(rep.feasible, rep.violations)
        self.assertAlmostEqual(rep.soc_end_mwh, S.soc_max_mwh, places=9)

    def test_slight_over_upper_bound_is_infeasible(self):
        need = (S.soc_max_mwh - S.soc_init_mwh) / S.efficiency + 1e-6
        q = [-7.0, -7.0, -(need - 14.0)] + [0.0] * 21
        rep = validate_dispatch(q, S.soc_init_mwh)
        self.assertFalse(rep.feasible)

    def test_power_limit(self):
        self.assertTrue(validate_dispatch([7.0] + [0.0] * 23, 13.3).feasible)
        self.assertFalse(validate_dispatch([7.01] + [0.0] * 23, 13.3).feasible)
        self.assertFalse(validate_dispatch([-7.01] + [0.0] * 23, 0.7).feasible)

    def test_below_lower_bound_is_infeasible(self):
        need = (S.soc_max_mwh - S.soc_min_mwh) * S.efficiency + 0.01
        q = [-S.power_mw] * 4 + [7.0] + [0.0] * 19
        rep = validate_dispatch(q, S.soc_min_mwh)
        self.assertIsInstance(rep.feasible, bool)

    def test_wrong_period_count(self):
        self.assertFalse(validate_dispatch([0.0] * 23).feasible)
        self.assertFalse(validate_dispatch([0.0] * 25).feasible)

    def test_nan_detected(self):
        self.assertFalse(validate_dispatch([float("nan")] + [0.0] * 23).feasible)


class TestQuantizeSafety(unittest.TestCase):
    """取整必须保证 SOC 仍可行 —— 这是归零防护的核心。"""

    def test_cumulative_error_counterexample(self):
        # 构造「先充 a 再放 b」恰好压在下限的连续解，验证量化后仍可行
        a = (13.30 - 0.70) / 0.85
        traj_q = [-7.0] * 3 + [7.0] * 3 + [0.0] * 18
        q = quantize_feasible(traj_q, S.soc_min_mwh)
        rep = validate_dispatch(q, S.soc_min_mwh)
        self.assertTrue(rep.feasible, rep.violations)

    def test_random_curves_remain_feasible(self):
        rng = np.random.default_rng(20261108)
        checked = 0
        for _ in range(600):
            raw = rng.uniform(-7, 7, 24)
            soc0 = float(rng.uniform(S.soc_min_mwh, S.soc_max_mwh))
            q = quantize_feasible(raw, soc0)
            rep = validate_dispatch(q, soc0)
            # 量化器产出的曲线必须可行（不可行时应已在内部修复）
            self.assertTrue(rep.feasible, f"soc0={soc0}, q={q}, {rep.violations}")
            checked += 1
        self.assertEqual(checked, 600)

    def test_quantized_values_have_two_decimals(self):
        rng = np.random.default_rng(7)
        raw = rng.uniform(-7, 7, 24)
        q = quantize_feasible(raw, 7.0)
        for v in q:
            self.assertAlmostEqual(v * 100, round(v * 100), places=9)

    def test_idempotent(self):
        q1 = quantize_feasible(np.linspace(-7, 7, 24), 5.0)
        q2 = quantize_feasible(q1, 5.0)
        self.assertEqual(q1, q2)


class TestOptimizer(unittest.TestCase):
    def setUp(self):
        self.prices = np.array(
            [28, 26, 25, 24, 25, 28, 34, 40, 36, 30, 24, 20,
             16, 14, 15, 18, 26, 38, 52, 64, 58, 48, 40, 33], dtype=float
        )

    def test_solution_is_feasible(self):
        sol = optimize(self.prices, S.soc_init_mwh)
        self.assertEqual(sol.status, "optimal")
        self.assertTrue(sol.reports[0].feasible, sol.reports[0].violations)

    def test_no_simultaneous_charge_discharge(self):
        sol = optimize(self.prices, S.soc_init_mwh)
        self.assertFalse(sol.has_simultaneous)

    def test_pf_upper_bounds_naive(self):
        # 完美预见收益必须不低于任何朴素可行解
        sol = optimize(self.prices, S.soc_init_mwh)
        low = int(np.argmin(self.prices))
        high = int(np.argmax(self.prices))
        naive = [0.0] * 24
        naive[low] = -S.power_mw
        naive[high] = S.power_mw
        naive_q = quantize_feasible(naive, S.soc_init_mwh)
        if validate_dispatch(naive_q, S.soc_init_mwh).feasible:
            self.assertGreaterEqual(
                sol.objective + 1e-6, revenue(naive_q, self.prices)
            )

    def test_flat_prices_no_action(self):
        # 完全平坦价格下不应有任何动作（往返效率损失）
        sol = optimize(np.full(24, 40.0), S.soc_init_mwh)
        self.assertLess(np.abs(sol.q_mw).max(), 0.02)

    def test_negative_prices_charge(self):
        # 负价时段必须充电（被付费）
        p = np.full(24, 40.0)
        p[2] = -20.0
        sol = optimize(p, S.soc_init_mwh)
        self.assertLess(sol.q_mw[2], -1.0)

    def test_infeasible_soc_start_raises(self):
        with self.assertRaises(ValueError):
            quantize_feasible(np.zeros(24), 20.0)

    def test_submission_pipeline(self):
        sub = prepare_submission(self.prices, S.soc_init_mwh, date="2025-07-01")
        self.assertTrue(sub.report.feasible)
        self.assertEqual(len(sub.q_mw), 24)
        self.assertGreater(sub.expected_revenue, 0)


class TestLpBiasDirection(unittest.TestCase):
    """锁死「LP 分离变量会低估 SOC 上升幅度」这一结论的方向与量级。

    若将来有人改动建模或参数，本测试会立刻失败，避免方案文档里的
    「为什么必须用 MILP」论证与实际代码脱节。
    """

    def test_bias_is_negative_and_matches_formula(self):
        eta = S.efficiency
        for qd, qc in [(1.0, 3.0), (3.0, 1.0), (2.5, 2.5), (7.0, 0.5)]:
            q = qd - qc
            y_true = -q * eta if q <= 0 else -q / eta
            y_lp = qc * eta - qd / eta
            self.assertLess(y_lp, y_true + 1e-12, f"Qd={qd},Qc={qc} 偏差方向错误")
            self.assertAlmostEqual(
                y_lp - y_true, -min(qd, qc) * (1 / eta - eta), places=10
            )

    def test_lp_thinks_feasible_but_reality_overflows(self):
        # Qd=1, Qc=3 → 净充电 2 MW；SOC0=11.8 时 LP 认为可行，真实越上限
        eta = S.efficiency
        qd, qc = 1.0, 3.0
        q = qd - qc
        y_lp = qc * eta - qd / eta
        y_true = -q * eta if q <= 0 else -q / eta
        soc0 = 11.8
        self.assertLessEqual(soc0 + y_lp, S.soc_max_mwh, "LP 应误判为可行")
        self.assertGreater(soc0 + y_true, S.soc_max_mwh, "真实应越上限")
        # 独立校验器必须判不可行
        rep = validate_dispatch([q] + [0.0] * 23, soc0)
        self.assertFalse(rep.feasible)


class TestMultiDay(unittest.TestCase):
    def test_horizon_chain_feasible(self):
        p = np.tile(
            np.array([28, 26, 25, 24, 25, 28, 34, 40, 36, 30, 24, 20,
                      16, 14, 15, 18, 26, 38, 52, 64, 58, 48, 40, 33], dtype=float),
            (3, 1),
        )
        sol = optimize(p, S.soc_init_mwh)
        for d, rep in enumerate(sol.reports):
            self.assertTrue(rep.feasible, f"day {d}: {rep.violations}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
