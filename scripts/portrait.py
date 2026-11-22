"""市场画像统计：把 SP15 DAM 价格矩阵压成一组可引用的结构性指标。

输出全部为控制台文本，供报告取数。所有窗口在计算前显式声明，避免混用。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MAT = ROOT / "data" / "raw" / "sp15_dam_matrix.csv"

ETA = 0.85
MIN_SPREAD = 1.0 / ETA**2  # 保本所需的最低 卖出/买入 价比


def load() -> pd.DataFrame:
    df = pd.read_csv(MAT, index_col=0, parse_dates=True).sort_index()
    df.columns = [int(c) for c in df.columns]
    before = len(df)
    df = df.dropna(how="any")
    df = df[(df > -1000).all(axis=1)]
    print(f"[load] rows={before} -> usable={len(df)}  "
          f"window={df.index.min().date()} ~ {df.index.max().date()}")
    return df


def sec(t: str) -> None:
    print("\n" + "=" * 78)
    print(t)
    print("=" * 78)


def main() -> None:
    df = load()
    n = len(df)

    # ---------- 1. 整体水位 ----------
    sec("1. 价格水平与离散度（$/MWh，日前 SP15）")
    flat = df.to_numpy().ravel()
    print(f"样本          : {n} 天 × 24 时段 = {flat.size:,} 个时段价")
    print(f"均值 / 中位   : {flat.mean():.2f} / {np.median(flat):.2f}")
    print(f"标准差        : {flat.std(ddof=1):.2f}")
    qs = [0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100]
    for q in qs:
        print(f"  P{q:<3d}         : {np.percentile(flat, q):9.2f}")
    print(f"负价时段占比  : {(flat < 0).mean() * 100:.2f}%")
    print(f"零价以下(<=0) : {(flat <= 0).mean() * 100:.2f}%")
    print(f">100 的时段   : {(flat > 100).mean() * 100:.2f}%")
    print(f">200 的时段   : {(flat > 200).mean() * 100:.2f}%")

    # ---------- 2. 逐年漂移 ----------
    sec("2. 逐年结构漂移（检验 regime 是否稳定）")
    y = df.groupby(df.index.year)
    rows = []
    for yr, g in y:
        f = g.to_numpy().ravel()
        rows.append({
            "年": yr, "天数": len(g), "均价": f.mean(), "中位": np.median(f),
            "标准差": f.std(ddof=1), "负价%": (f < 0).mean() * 100,
            "P95": np.percentile(f, 95), "日极差": (g.max(axis=1) - g.min(axis=1)).mean(),
        })
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f"{v:8.2f}"))

    # ---------- 3. 日内形态 ----------
    sec("3. 日内形态：按季节的时段均价（$/MWh）")
    def season(idx: pd.DatetimeIndex) -> pd.Series:
        m = idx.month
        s = pd.Series("冬(12-2)", index=idx)
        s[m.isin([3, 4, 5])] = "春(3-5)"
        s[m.isin([6, 7, 8])] = "夏(6-8)"
        s[m.isin([9, 10, 11])] = "秋(9-11)"
        s[m == 12] = "冬(12-2)"
        return s
    seas = season(df.index).reindex(df.index)
    lg = df.stack().rename_axis(["date", "hour"]).reset_index(name="lmp")
    lg["season"] = lg["date"].map(seas)
    prof = lg.pivot_table(index="hour", columns="season", values="lmp", aggfunc="mean")
    prof = prof[["冬(12-2)", "春(3-5)", "夏(6-8)", "秋(9-11)"]]
    prof.index.name = "时段"
    print(prof.round(1).to_string())
    print("\n谷段/峰段定位（各季最低与最高时段）：")
    for c in prof.columns:
        print(f"  {c}: 最低 时段{prof[c].idxmin():>2d} = {prof[c].min():7.2f} | "
              f"最高 时段{prof[c].idxmax():>2d} = {prof[c].max():7.2f} | "
              f"峰谷差 {prof[c].max() - prof[c].min():7.2f}")

    # ---------- 4. 负价结构 ----------
    sec("4. 负价结构（决定「充电是否被付费」）")
    neg = (df < 0)
    print("按年：")
    for yr, g in df.groupby(df.index.year):
        nn = (g < 0).to_numpy().sum()
        print(f"  {yr}: 负价时段 {nn:5d} / {g.size:6d} = {nn / g.size * 100:5.2f}%  "
              f"负价日 {int((g < 0).any(axis=1).sum()):3d} 天")
    print("\n按时段（全样本合计）：")
    byh = neg.sum()
    top = byh.sort_values(ascending=False).head(8)
    for h, c in top.items():
        print(f"  时段 {h:>2d}: {int(c):4d} 次 ({c / n * 100:5.2f}% 的日)")
    print(f"\n负价日均价（仅统计出现负价的时段）: {df.where(df < 0).stack().mean():.2f}")

    # ---------- 5. 日价差分布 ----------
    sec("5. 日价差分布（套利的原料）")
    def tb(k: int) -> pd.Series:
        arr = np.sort(df.to_numpy(), axis=1)
        return arr[:, -k:].mean(axis=1) - arr[:, :k].mean(axis=1)
    d = pd.DataFrame({
        "极差": df.max(axis=1) - df.min(axis=1),
        "TB1": tb(1), "TB2": tb(2), "TB3": tb(3), "TB4": tb(4),
        "日内std": df.std(axis=1),
    })
    print(d.describe(percentiles=[.05, .1, .25, .5, .75, .9, .95])
          .round(2).to_string())
    print(f"\n保本门槛 = 1/η² = {MIN_SPREAD:.4f}（价差倍数，非 $）")
    print(f"TB2 高于 0 的天数        : {(d.TB2 > 0).sum()} / {n} = {(d.TB2 > 0).mean()*100:.1f}%")
    print(f"TB2 低于 $10 的天数      : {(d.TB2 < 10).sum()} = {(d.TB2 < 10).mean()*100:.1f}%")
    print(f"TB2 低于 $20 的天数      : {(d.TB2 < 20).sum()} = {(d.TB2 < 20).mean()*100:.1f}%")

    # ---------- 6. 可预测性 ----------
    sec("6. 可预测性（决定预测方法的天花板）")
    # 逐日、跨 24 小时的 Pearson 相关 = 形态相关（决定能否排对充放顺序）
    corr_shape = df.corrwith(df.shift(1), axis=1)
    print(f"D 与 D-1 形态相关（逐日跨24时段均值）  : {corr_shape.mean():.4f}")
    print(f"  10 分位 / 90 分位                    : {corr_shape.quantile(.1):.4f} / {corr_shape.quantile(.9):.4f}")
    print("  按年                                : " + ", ".join(
        f"{yr}={corr_shape[corr_shape.index.year == yr].mean():.3f}"
        for yr in sorted(set(df.index.year))))

    # 逐时段、跨天的相关 = 水平持续性
    corr_lev = df.corrwith(df.shift(1), axis=0)
    print(f"D 与 D-1 水平相关（逐时段跨天均值）    : {corr_lev.mean():.4f}")

    med7 = df.rolling(7).median().shift(1)
    print(f"D 与近 7 日中位数 形态相关            : {df.corrwith(med7, axis=1).mean():.4f}")

    # 绝对误差（预测的实际代价）
    err = (df - df.shift(1)).abs().dropna()
    print(f"\n|D 价 − D-1 价| 全样本 MAE           : ${err.to_numpy().mean():.2f}/MWh")
    print(f"  占全样本均价的比例                  : {err.to_numpy().mean() / df.to_numpy().mean() * 100:.1f}%")
    print(f"  分年 MAE                            : " + ", ".join(
        f"{yr}=${err[err.index.year == yr].to_numpy().mean():.2f}" for yr in sorted(set(df.index.year))))

    # 昨日价差对今日价差的预测力（真正决定套利空间）
    def tb_all(k: int) -> pd.Series:
        a = np.sort(df.to_numpy(), axis=1)
        return pd.Series(a[:, -k:].mean(axis=1) - a[:, :k].mean(axis=1), index=df.index)
    t2 = tb_all(2)
    print(f"\n昨日 TB2 → 今日 TB2 相关             : {t2.corr(t2.shift(1)):.4f}")
    print(f"  含义：昨日价差只能解释今日价差方差的 {t2.corr(t2.shift(1)) ** 2 * 100:.0f}%")

    # ---------- 7. 套利收益的理论上界 ----------
    sec("7. 收益换算（14 MWh / 7 MW / η=0.85，7 MW = 7,000 kW）")
    print("说明：$\/kW-yr = 年收益 ÷ 额定功率(kW)，是与同业对标的标准口径。")

    # ---------- 8. 实际回测收益 ----------
    sec("8. 实际回测收益（data/out/backtest/daily.csv，2025 全年 364 天）")
    bt = pd.read_csv(ROOT / "data" / "out" / "backtest" / "daily.csv", parse_dates=["date"])
    piv = bt.pivot_table(index="date", columns="strategy", values="revenue")
    pf = piv["PF(完美预见)"]
    KW = 7000.0
    print(f"完美预见  : 总 ${pf.sum():>9,.0f}  日均 ${pf.mean():>6.0f}  "
          f"${pf.sum() / KW:6.1f}/kW-yr  最差日 ${pf.min():>7.0f}")
    for s in ["opt-blend", "opt-median7", "opt-persistence", "naive-TB3"]:
        if s in piv:
            r = piv[s]
            print(f"{s:<10s}: 总 ${r.sum():>9,.0f}  日均 ${r.mean():>6.0f}  "
                  f"${r.sum() / KW:6.1f}/kW-yr  最差日 ${r.min():>7.0f}  "
                  f"捕获 {(r.sum() / pf.sum()) * 100:5.2f}%")

    # ---------- 9. 比赛期窗口 ----------
    sec("9. 比赛期窗口（11-23 ~ 12-06）")
    mask = ((df.index.month == 11) & (df.index.day >= 23)) | ((df.index.month == 12) & (df.index.day <= 6))
    win = df[mask]
    print(f"样本天数 {len(win)}（两年各 14 天）")
    print(f"均价 ${win.to_numpy().ravel().mean():.2f}  "
          f"负价占比 {(win.to_numpy().ravel() < 0).mean() * 100:.2f}%")
    tbw = t2[mask]
    print(f"TB2 均值 ${tbw.mean():.1f}  中位 ${np.median(tbw):.1f}  "
          f"P10 ${np.percentile(tbw, 10):.1f}  P90 ${np.percentile(tbw, 90):.1f}")
    pw = piv.loc[piv.index.isin(win.index)]
    for s in ["PF(完美预见)", "opt-blend"]:
        if s in pw:
            print(f"{s:<15s}: 日均 ${pw[s].mean():6.0f}  14天合计 ${pw[s].sum():7,.0f}")

    # ---------- 10. 2026 regime 与月度结构 ----------
    sec("10. 月度结构（各年均价 / 负价占比 / TB2），标出比赛期位置")
    lg2 = df.stack().rename_axis(["date", "hour"]).reset_index(name="lmp")
    lg2["ym"] = lg2["date"].dt.to_period("M")
    g = lg2.groupby("ym")["lmp"]
    mstat = pd.DataFrame({"均价": g.mean(), "负价%": g.apply(lambda s: (s < 0).mean() * 100)})
    mstat["TB2"] = t2.groupby(t2.index.to_period("M")).mean()
    mstat["月份"] = [p.month for p in mstat.index]
    for yr in sorted(set(df.index.year)):
        sub = mstat[[p.year == yr for p in mstat.index]].copy()
        sub.index = [f"{p.month}月" for p in sub.index]
        print(f"\n{yr}:")
        print(sub.round(1).to_string())

    sec("11. 2026 年能否沿用 2025 的收益预期？（同期对比）")
    for tag, a, b in [
        ("2025-10-01~11-06", "2025-10-01", "2025-11-06"),
        ("2026-10-01~11-06", "2026-10-01", "2026-11-06"),
    ]:
        sub = df.loc[a:b]
        if len(sub) == 0:
            continue
        f = sub.to_numpy().ravel()
        sb = tb_all(2).loc[a:b]
        print(f"{tag}: {len(sub)} 天  均价 ${f.mean():5.2f}  负价 {((f < 0).mean() * 100):4.1f}%  "
              f"TB2 ${sb.mean():5.1f}  日极差 ${(sub.max(axis=1) - sub.min(axis=1)).mean():5.1f}")

    # ---------- 12. 14 日窗口的收益分布 ----------
    sec("12. 14 天比赛窗口的收益分布（滚动窗口自助，opt-blend）")
    r = piv["opt-blend"].sort_index()
    pfr = pf.sort_index()
    for yr in sorted(set(r.index.year)):
        rr = r[r.index.year == yr].reset_index(drop=True)
        pp = pfr[pfr.index.year == yr].reset_index(drop=True)
        if len(rr) < 60:
            continue
        roll = rr.rolling(14).sum().dropna()
        rollpf = pp.rolling(14).sum().dropna()
        print(f"\n{yr}（{len(roll)} 个重叠窗口）：14 天合计收益")
        print(f"  均值 ${roll.mean():>7,.0f}  中位 ${roll.median():>7,.0f}  "
              f"P10 ${np.percentile(roll, 10):>7,.0f}  P90 ${np.percentile(roll, 90):>7,.0f}")
        print(f"  最好 ${roll.max():>7,.0f}  最差 ${roll.min():>7,.0f}  "
              f"（对应 PF 上界日均 ${rollpf.mean() / 14:.0f}）")

    sec("13. 误差的收益代价（把预测误差折算成钱）")
    print("路径：预测误差 → 充放时段错配 → 少赚的价差")
    print(f"  PF 日均 ${pf.mean():.0f}  →  blend 日均 ${piv['opt-blend'].mean():.0f}  "
          f"差 ${pf.mean() - piv['opt-blend'].mean():.0f}/日 = 捕获 {(1 - piv['opt-blend'].mean() / pf.mean()) * 100:.1f}% 的遗漏")
    print(f"  MAE ${err.to_numpy().mean():.2f}/MWh × 每日吞吐量 ≈ 代价量级")
    # 每日实际吞吐
    q = bt[bt.strategy == "opt-blend"].set_index("date")
    print(f"  blend 日均充电 {q['charge'].mean():.1f} MWh / 放电 {q['discharge'].mean():.1f} MWh "
          f"（满容量 12.6 MWh，利用率 {q['discharge'].mean() / 12.6 * 100:.0f}%）")


if __name__ == "__main__":
    main()
