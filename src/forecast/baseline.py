"""日前价格预测基线。

为什么基线优先于复杂模型：
同业实测（CAISO 2025 全年、4h 电池、SP15 hub）显示，用「昨日价格」这类
朴素 persistence 策略即可拿到完美预见上界的 **95.1%** 收益。原因是鸭形曲线
的形态日间高度重复：对储能套利而言，**价格曲线的相对形状比绝对水平重要得多**
（决定充放排序），绝对水平只在判断负电价与保本价差时起作用。

因此本模块提供的是「形状稳健」的基线族，而非 ML 模型。这也是相对评分机制下
的正确取舍：目标是稳定跑赢中位数，不是追求绝对最优。

所有预测器只使用目标日**之前**的数据，严格无穿越。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class PriceForecaster:
    """24 时段日前价格预测器。

    method 取值：
      - persistence  : 直接用前一日价格（同业最强朴素基准）
      - median7      : 最近 7 天同时段中位数（抗单日异常）
      - median14     : 最近 14 天同时段中位数
      - dow_median   : 同星期几最近 4 周中位数（捕捉周内差异）
      - blend        : 0.4·persistence + 0.6·median7（默认）
      - shape_level  : 7 天中位数形状 × 近 3 日均值水平
    """

    method: str = "blend"
    lookback: int = 7
    min_history: int = 1
    fallback_price: float = 35.0
    """历史不足时使用的默认价格（CAISO SP15 长期均值量级）。"""

    def _hist(self, history: pd.DataFrame, n: int) -> pd.DataFrame:
        return history.iloc[-n:] if len(history) >= n else history

    def predict(self, history: pd.DataFrame) -> np.ndarray:
        """返回 24 维预测价格。

        Args:
            history: 目标日之前的历史价格矩阵，index 为可排序日期，columns 为 1..24。
        """
        if history is None or len(history) < self.min_history:
            return np.full(24, self.fallback_price, dtype=float)

        h = history.copy()
        h.columns = [int(c) for c in h.columns]

        if self.method == "persistence":
            return h.iloc[-1].to_numpy(dtype=float)

        if self.method == "median7":
            return self._hist(h, 7).median(axis=0).to_numpy(dtype=float)

        if self.method == "median14":
            return self._hist(h, 14).median(axis=0).to_numpy(dtype=float)

        if self.method == "dow_median":
            try:
                target_dow = (pd.Timestamp(h.index[-1]) + pd.Timedelta(days=1)).dayofweek
            except Exception:  # noqa: BLE001
                return self._hist(h, 7).median(axis=0).to_numpy(dtype=float)
            same = h[[pd.Timestamp(d).dayofweek == target_dow for d in h.index]]
            if len(same) == 0:
                return self._hist(h, 7).median(axis=0).to_numpy(dtype=float)
            return same.iloc[-4:].median(axis=0).to_numpy(dtype=float)

        if self.method == "shape_level":
            base = self._hist(h, 7).median(axis=0).to_numpy(dtype=float)
            level = float(np.mean(self._hist(h, 3).to_numpy(dtype=float)))
            m = float(np.mean(base))
            if abs(m) < 1e-9:
                return np.full(24, level, dtype=float)
            # 保留形状（含负价相对深度），把水平锚定到近期均值
            return base * (level / m)

        if self.method == "blend":
            pers = h.iloc[-1].to_numpy(dtype=float)
            med = self._hist(h, 7).median(axis=0).to_numpy(dtype=float)
            return 0.4 * pers + 0.6 * med

        raise ValueError(f"未知预测方法: {self.method}")


DEFAULT_FORECASTER = PriceForecaster(method="blend")
