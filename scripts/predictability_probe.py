"""可预测性诊断：新能源渗透上升后，到底是"整体变差"还是"状态依赖"。

输出四组证据：
  1. 逐年可比指标（形态相关 / 水平相关 / 持久性 MAE / 结构模型 R²）
  2. 按当日价差 TB2 分位分组的状态依赖检验
  3. 按"是否有负价 / 是否有尖峰"分组
  4. 价格落在平段（|p|<2）的比例，作为供给曲线平坦区的代理

数据：CAISO SP15 日前 LMP，1,038 天 × 24 时段。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MAT = ROOT / "data" / "raw" / "sp15_dam_matrix.csv"


def load() -> pd.DataFrame:
    df = pd.read_csv(MAT, index_col=0, parse_dates=True)
    df.columns = [int(c) for c in df.columns]
    df = df.sort_index().dropna()
    return df


def shape_model_r2(df: pd.DataFrame, year: int) -> tuple[float, float]:
    """以 (月, 时段) 哑变量均值作预测，留一年外评估。
    返回 (样本外 R², 样本外 MAE)。"""
    cols = ["month", "hour"]
    long = df.stack().rename_axis(["date", "hour"]).reset_index(name="lmp")
    long["month"] = long["date"].dt.month
    train = long[long["date"].dt.year != year]
    test = long[long["date"].dt.year == year]
    if len(test) == 0:
        return np.nan, np.nan
    mu = train.groupby(cols)["lmp"].mean()
    pred = test.set_index(cols).index.map(mu)
    pred = pd.Series(pred, index=test.index).astype(float)
    pred = pred.fillna(train["lmp"].mean())
    y = test["lmp"]
    sse = ((y - pred) ** 2).sum()
    sst = ((y - y.mean()) ** 2).sum()
    return 1 - sse / sst, float((y - pred).abs().mean())


def main() -> None:
    df = load()
    prev = df.shift(1).dropna()
    cur = df.loc[prev.index]

    # ---- 1. 逐年指标
    print("=" * 96)
    print("1. 逐年可比指标（预测器：昨日曲线 D-1）")
    print("=" * 96)
    rows = []
    for yr in sorted(set(df.index.year)):
        d = df[df.index.year == yr]
        p = prev[prev.index.year == yr]
        c = cur[cur.index.year == yr]
        if len(d) < 30:
            continue
        shape_corr = np.mean([np.corrcoef(c.loc[i], p.loc[i])[0, 1]
                              for i in c.index if i in p.index and p.loc[i].std() > 1e-6])
        flat_c = c.to_numpy().ravel()
        flat_p = p.to_numpy().ravel()
        mae = np.abs(flat_c - flat_p).mean()
        level_corr = np.corrcoef(flat_c, flat_p)[0, 1]
        r2, mae_shape = shape_model_r2(df, yr)
        rows.append({
            "年": yr, "天": len(d),
            "均价": d.to_numpy().mean(), "标准差": d.to_numpy().std(ddof=1),
            "形态相关": shape_corr, "水平相关": level_corr,
            "MAE": mae, "MAE/均价": mae / d.to_numpy().mean() * 100,
            "结构R2": r2, "结构MAE": mae_shape,
            "负价%": (d.to_numpy() < 0).mean() * 100,
            "|p|<2占比%": (np.abs(d.to_numpy()) < 2).mean() * 100,
            "P99": np.percentile(d.to_numpy(), 99),
            "日极差": (d.max(axis=1) - d.min(axis=1)).mean(),
        })
    print(pd.DataFrame(rows).round(3).to_string(index=False))

    # ---- 2. TB2 分位分组
    arr = np.sort(df.to_numpy(), axis=1)
    tb2 = pd.Series(arr[:, -2:].mean(axis=1) - arr[:, :2].mean(axis=1), index=df.index)
    mae_d = pd.Series(np.abs(cur.to_numpy() - prev.to_numpy()).mean(axis=1), index=prev.index)
    print()
    print("=" * 96)
    print("2. 状态依赖：按当日 TB2 四分位分组（MAE 为 D 与 D-1 的逐时绝对误差均值）")
    print("=" * 96)
    q = pd.qcut(tb2.loc[mae_d.index], 4, labels=["Q1 最窄", "Q2", "Q3", "Q4 最宽"])
    g = pd.DataFrame({"TB2": tb2.loc[mae_d.index], "MAE": mae_d, "q": q}).groupby("q", observed=True)
    out = g.agg(天数=("MAE", "size"), TB2均值=("TB2", "mean"),
                MAE均值=("MAE", "mean"), MAE中位=("MAE", "median"))
    out["MAE/TB2%"] = out["MAE均值"] / out["TB2均值"] * 100
    print(out.round(2).to_string())

    # ---- 3. 负价日 / 尖峰日
    print()
    print("=" * 96)
    print("3. 分组：是否有负价 / 是否有尖峰（>=100）")
    print("=" * 96)
    has_neg = (df < 0).any(axis=1).loc[mae_d.index]
    has_spike = (df >= 100).any(axis=1).loc[mae_d.index]
    rows = []
    for name, mask in [("有负价", has_neg), ("无负价", ~has_neg),
                       ("有尖峰>=100", has_spike), ("无尖峰", ~has_spike)]:
        sub = mae_d[mask]
        t = tb2.loc[sub.index]
        rows.append({"组": name, "天数": len(sub),
                     "占比%": len(sub) / len(mae_d) * 100,
                     "MAE均值": sub.mean(), "MAE中位": sub.median(),
                     "TB2均值": t.mean()})
    print(pd.DataFrame(rows).round(2).to_string(index=False))

    # ---- 4. 形状跨年稳定性
    print()
    print("=" * 96)
    print("4. 形状的跨年稳定性：第 N 年的 (月,时段) 均值曲线 与 第 N+1 年的相关")
    print("=" * 96)
    prof = {}
    for yr in sorted(set(df.index.year)):
        d = df[df.index.year == yr]
        if len(d) < 300:
            continue
        long = d.stack().rename_axis(["date", "hour"]).reset_index(name="lmp")
        long["month"] = long["date"].dt.month
        prof[yr] = long.groupby(["month", "hour"])["lmp"].mean()
    yrs = sorted(prof)
    for a, b in zip(yrs, yrs[1:]):
        x, y = prof[a].align(prof[b], join="inner")
        print(f"  {a} vs {b}: 相关 {np.corrcoef(x, y)[0,1]:.4f}   "
              f"平均绝对差 ${(x-y).abs().mean():.2f}   "
              f"形状差（去均值后）${((x-x.mean())-(y-y.mean())).abs().mean():.2f}")

    # ---- 5. 供给侧形态代理
    print()
    print("=" * 96)
    print("5. 供给曲线形态代理：低价区（|p|<2 与 p<0）与高价区的占比变化")
    print("=" * 96)
    for yr in sorted(set(df.index.year)):
        d = df[df.index.year == yr].to_numpy().ravel()
        if len(d) < 500:
            continue
        print(f"  {yr}: p<0 {(d<0).mean()*100:5.2f}%  |p|<2 {(np.abs(d)<2).mean()*100:5.2f}%  "
              f"p<10 {(d<10).mean()*100:5.2f}%  p>100 {(d>100).mean()*100:5.2f}%  "
              f"p>200 {(d>200).mean()*100:5.2f}%")


if __name__ == "__main__":
    main()
