#!/usr/bin/env python3
"""线上组交易决策 · 收益上界与预测价值校验。

用途：每次改动策略后，快速回答三个问题
  1. 完美信息上界是多少（策略天花板）
  2. 朴素预测能拿到上界的百分之几（预测的边际价值）
  3. 加"价差阈值开关"是否划算（默认否，见输出）

储能参数按参赛手册：14 MWh / 7 MW / 单程效率 85% / SOC 5%-95% / 首日 5%。
结算口径：仅日前价格，收益 = Σ p_t · Q_t，放电为正、充电为负。

    python3 arb_sanity.py --since 2025-11-01
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from scipy.optimize import linprog

Q_R, P, ETA, LO, HI, N = 14.0, 7.0, 0.85, 0.05, 0.95, 24
CSV = "data/raw/sp15_dam_lmp.csv"


def solve(price: np.ndarray, soc0: float):
    """24 时段 LP。变量 c[24] 充电、d[24] 放电、s[24] 时段末 SOC。"""
    c, d = slice(0, N), slice(N, 2 * N)
    obj = np.zeros(3 * N)
    obj[c] = price
    obj[d] = -price

    A, b = [], []
    for t in range(N):
        row = np.zeros(3 * N)
        row[2 * N + t] = 1.0
        if t:
            row[2 * N + t - 1] = -1.0
        row[c.start + t] = -ETA / Q_R
        row[d.start + t] = 1.0 / (Q_R * ETA)
        A.append(row)
        b.append(soc0 if t == 0 else 0.0)

    r = linprog(
        obj,
        A_eq=np.array(A),
        b_eq=np.array(b),
        bounds=[(0, P)] * N + [(0, P)] * N + [(LO, HI)] * N,
        method="highs",
    )
    if not r.success:
        return None, None
    return r.x[d] - r.x[c], r.x[2 * N :]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", default="2025-11-01")
    ap.add_argument("--csv", default=CSV)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    m = (
        df.pivot_table(index="date", columns="hour", values="lmp", aggfunc="last")
        .reindex(columns=range(1, N + 1))
        .dropna()
    )
    m = m[m.index >= args.since]

    rows = []
    soc = {"perfect": LO, "y1": LO, "y2": LO}
    for d in m.index:
        i = m.index.get_loc(d)
        pt = m.iloc[i].to_numpy()
        p1 = m.iloc[i - 1].to_numpy() if i >= 1 else pt
        p2 = m.iloc[i - 2].to_numpy() if i >= 2 else p1
        out = {"date": d, "spread": pt.max() - pt.min()}
        for k, ph in (("perfect", pt), ("y1", p1), ("y2", 0.5 * p1 + 0.5 * p2)):
            q, se = solve(ph, soc[k])
            out[k] = 0.0 if q is None else float(pt @ q)
            if se is not None:
                soc[k] = float(se[-1])
        rows.append(out)

    r = pd.DataFrame(rows)
    print(f"样本 {len(r)} 天  {r.date.iloc[0]} ~ {r.date.iloc[-1]}")
    print(f"完美信息日均 ${r.perfect.mean():.1f}")
    for k, name in (("y1", "昨日同时段"), ("y2", "两日均值")):
        print(f"{name}日均 ${r[k].mean():.1f}")

    ok = r[r.perfect > 1.0].copy()
    ok["cap"] = ok.y2 / ok.perfect * 100
    print(f"\n捕获率（剔除完美上界<$1 的噪声日，剩 {len(ok)} 天）")
    print(f"  均值 {ok.cap.mean():.1f}%  中位 {ok.cap.median():.1f}%")
    print(
        f"  P10 {ok.cap.quantile(.10):.1f}%  P25 {ok.cap.quantile(.25):.1f}%"
        f"  P75 {ok.cap.quantile(.75):.1f}%  P90 {ok.cap.quantile(.90):.1f}%"
    )
    print(f"  负收益天数 {(r.y2 < 0).sum()}")

    print("\n按价差分档（中位捕获率）")
    for name, sub in (
        (">80 尖峰日", ok[ok.spread > 80]),
        ("40-80", ok[(ok.spread >= 40) & (ok.spread <= 80)]),
        ("<40 平淡日", ok[ok.spread < 40]),
    ):
        if len(sub):
            print(f"  {name:12s} n={len(sub):3d}  {sub.cap.median():5.1f}%")

    print("\n价差阈值开关（低于阈值空仓）")
    print(f"  {'阈值':>4} {'日均收益':>10} {'捕获率均值':>10}")
    for th in (0, 10, 15, 20, 25, 30):
        soc_v, tot = LO, 0.0
        for d in m.index:
            i = m.index.get_loc(d)
            pt = m.iloc[i].to_numpy()
            py = m.iloc[i - 1].to_numpy() if i >= 1 else pt
            q, se = solve(py, soc_v)
            if q is None:
                continue
            if py.max() - py.min() < th:
                q = np.zeros(N)
            else:
                soc_v = float(se[-1])
            tot += float(pt @ q)
        print(f"  {th:4d} ${tot / len(m.index):9.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
