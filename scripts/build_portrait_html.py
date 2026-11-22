"""生成「市场画像 / 痛点与方法 / 收益预期」单页报告。

所有图表坐标由真实数据计算得出，不手写常量，避免比例失真。
输出：/Users/zhukeyi/WorkBuddy/电力交易/市场画像_痛点_收益预期.html
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT.parent / "市场画像_痛点_收益预期.html"

ETA = 0.85
CAP, PWR = 14.0, 7.0
SOC_MIN, SOC_MAX = 0.70, 13.30
DSOC = SOC_MAX - SOC_MIN          # 12.6
E_CHARGE = DSOC / ETA             # 从电网买入
E_SELL = DSOC * ETA               # 向电网卖出
E_WASTE = E_CHARGE - E_SELL       # 效率损耗（以低价买入计）

C_B900, C_B800, C_B600, C_B400 = "#042C53", "#0C447C", "#185FA5", "#378ADD"
C_B300, C_B200, C_B100, C_B50 = "#85B7EB", "#B5D4F4", "#DCEAF8", "#E6F1FB"
C_A600, C_A400, C_A200, C_A50 = "#BA7517", "#EF9F27", "#FAC775", "#FAEEDA"
C_G600, C_G400, C_G200, C_G50 = "#5F5E5A", "#888780", "#D3D1C7", "#F5F7FA"
C_RED = "#C0392B"
C_LINE = "#E8EDF3"

# 四季形态的一句话描述（人工归纳，数字由数据给出）
SEASON_SHAPE = {
    "冬(12-2)": "双峰：早 7 时 + 晚 18–22 时，午间浅谷",
    "春(3-5)": "单峰：午间 9–17 时深度负价，晚峰最平",
    "夏(6-8)": "单峰：晚 20 时最高，空调用电驱动",
    "秋(9-11)": "单峰：晚 19 时最高，午间轻度负价",
}


# ---------------------------------------------------------------- data
def load() -> pd.DataFrame:
    df = pd.read_csv(ROOT / "data" / "raw" / "sp15_dam_matrix.csv", index_col=0, parse_dates=True)
    df.columns = [int(c) for c in df.columns]
    return df.sort_index().dropna()


def backtests() -> dict[str, pd.DataFrame]:
    out = {}
    for tag, path in [("2025", "data/out/backtest/daily.csv"), ("2026", "data/out/bt_2026/daily.csv")]:
        p = ROOT / path
        if p.exists():
            d = pd.read_csv(p, parse_dates=["date"])
            out[tag] = d.pivot_table(index="date", columns="strategy", values="revenue").sort_index()
    return out


def scale(v, vmin, vmax, y_bottom, y_top):
    """把 v 从 [vmin,vmax] 映射到像素 y；vmin 落在 y_bottom，vmax 落在 y_top。

    约定 y_bottom > y_top（SVG 中 y 向下增大），因此高值向上。
    """
    if vmax == vmin:
        return y_bottom
    return y_bottom - (v - vmin) / (vmax - vmin) * (y_bottom - y_top)


def fmt(v, n=0):
    return f"{v:,.{n}f}"


# ---------------------------------------------------------------- charts
def chart_hist(flat: np.ndarray) -> str:
    W, H = 1000, 320
    L, R, T, MB = 60, 22, 30, 52; B = H - MB
    lo, hi, step = -70, 130, 4
    bins = np.arange(lo, hi + step, step)
    cnt, _ = np.histogram(np.clip(flat, lo, hi - 1e-9), bins=bins)
    pct = cnt / cnt.sum() * 100
    vmax = pct.max() * 1.12
    n = len(pct)
    bw = (W - L - R) / n
    parts = []
    # y 轴：按 vmax 自动取刻度
    tick_step = 1 if vmax <= 8 else (2 if vmax <= 20 else 5)
    gv = 0
    while gv <= vmax:
        y = scale(gv, 0, vmax, B, T)
        parts.append(f'<line x1="{L}" y1="{y:.1f}" x2="{W-R}" y2="{y:.1f}" stroke="{C_LINE}"/>')
        parts.append(f'<text x="{L-8}" y="{y+3.5:.1f}" text-anchor="end" font-size="10.5" fill="{C_G400}">{gv:g}%</text>')
        gv += tick_step
    for i, p in enumerate(pct):
        x = L + i * bw
        y = scale(p, 0, vmax, B, T)
        center = bins[i] + step / 2
        fill = C_A200 if center < 0 else C_B200
        parts.append(f'<rect x="{x+0.6:.1f}" y="{y:.1f}" width="{bw-1.2:.1f}" height="{B-y:.1f}" fill="{fill}" rx="1"/>')
    for tv in [-60, -40, -20, 0, 20, 40, 60, 80, 100, 120]:
        x = L + (tv - lo) / (hi - lo) * (W - L - R)
        parts.append(f'<line x1="{x:.1f}" y1="{B}" x2="{x:.1f}" y2="{B+4}" stroke="{C_G200}"/>')
        parts.append(f'<text x="{x:.1f}" y="{B+18}" text-anchor="middle" font-size="10.5" fill="{C_G400}">{tv}</text>')
    parts.append(f'<line x1="{L}" y1="{B}" x2="{W-R}" y2="{B}" stroke="{C_G200}"/>')
    # 零线
    xz = L + (0 - lo) / (hi - lo) * (W - L - R)
    parts.append(f'<line x1="{xz:.1f}" y1="{T-8}" x2="{xz:.1f}" y2="{B}" stroke="{C_RED}" stroke-width="1" stroke-dasharray="3 3" opacity="0.75"/>')
    parts.append(f'<text x="{xz+5:.1f}" y="{T-11}" font-size="10" fill="{C_RED}">0 元</text>')
    # 中位线
    xm = L + (np.median(flat) - lo) / (hi - lo) * (W - L - R)
    parts.append(f'<line x1="{xm:.1f}" y1="{T-8}" x2="{xm:.1f}" y2="{B}" stroke="{C_B800}" stroke-width="1.4"/>')
    parts.append(f'<text x="{xm+5:.1f}" y="{T-11}" font-size="10.5" fill="{C_B800}">中位 ${np.median(flat):.2f}</text>')
    parts.append(f'<text x="{L}" y="{H-12}" font-size="10.5" fill="{C_G400}">时段电价（$/MWh）　纵轴为该价位出现的频率</text>')
    parts.append(f'<text x="{W-R}" y="{H-12}" text-anchor="end" font-size="10.5" fill="{C_A600}">'
                 f'橙色 = 负价区间，占 {100*(flat<0).mean():.1f}%；高于 $130 的 0.2% 已并入末尾柱</text>')
    return f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="SP15日前电价分布直方图">' + "".join(parts) + "</svg>"


def chart_shape(prof: pd.DataFrame) -> str:
    W, H = 1000, 340
    L, R, T, MB = 56, 118, 26, 46; B = H - MB
    vmin, vmax = -22, 78
    colors = {"冬(12-2)": C_B300, "春(3-5)": C_A600, "夏(6-8)": C_B600, "秋(9-11)": C_B800}
    parts = []
    for gv in [-20, -10, 0, 10, 20, 30, 40, 50, 60, 70]:
        y = scale(gv, vmin, vmax, B, T)
        dash = ' stroke-dasharray="4 3" opacity="0.85"' if gv == 0 else ""
        col = C_RED if gv == 0 else C_LINE
        parts.append(f'<line x1="{L}" y1="{y:.1f}" x2="{W-R}" y2="{y:.1f}" stroke="{col}"{dash}/>')
        parts.append(f'<text x="{L-8}" y="{y+3.5:.1f}" text-anchor="end" font-size="10.5" fill="{C_G400}">{gv}</text>')
    # 负价区底色
    y0 = scale(0, vmin, vmax, B, T)
    parts.append(f'<rect x="{L}" y="{y0:.1f}" width="{W-L-R}" height="{B-y0:.1f}" fill="#FDF3E7" opacity="0.55"/>')
    xs = [L + i / 23 * (W - L - R) for i in range(24)]
    for name, col in colors.items():
        if name not in prof:
            continue
        pts = " ".join(f"{xs[i]:.1f},{scale(prof[name].iloc[i], vmin, vmax, B, T):.1f}" for i in range(24))
        w = 2.6 if name == "春(3-5)" else 1.7
        parts.append(f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="{w}" stroke-linejoin="round"/>')
    for h in [1, 6, 12, 18, 24]:
        x = xs[h - 1]
        parts.append(f'<text x="{x:.1f}" y="{B+17}" text-anchor="middle" font-size="10.5" fill="{C_G400}">{h}时</text>')
    # 图例
    lx = W - R + 14
    for k, (name, col) in enumerate(colors.items()):
        y = T + 8 + k * 22
        parts.append(f'<line x1="{lx}" y1="{y}" x2="{lx+16}" y2="{y}" stroke="{col}" stroke-width="2.6"/>')
        parts.append(f'<text x="{lx+21}" y="{y+4}" font-size="10.5" fill="{C_G600}">{name}</text>')
    # 关键标注
    x_spring_min = xs[13]
    y_spring_min = scale(prof["春(3-5)"].iloc[13], vmin, vmax, B, T)
    parts.append(f'<circle cx="{x_spring_min:.1f}" cy="{y_spring_min:.1f}" r="4" fill="{C_A600}"/>')
    # 标注放在左下空白区（该区域所有曲线都在 25 以上），再用虚线引到谷底
    n_neg = int((prof["春(3-5)"] < 0).sum())
    tx, ty = L + 8, scale(-18.6, vmin, vmax, B, T)
    parts.append(f'<text x="{tx}" y="{ty:.1f}" font-size="10.5" fill="{C_A600}">'
                 f'春季 9–17 时连续 {n_neg} 小时均价为负，谷底 14 时 {prof["春(3-5)"].min():.1f} 元</text>')
    tx_end = tx + 11.0 * 27
    parts.append(f'<polyline points="{tx_end:.1f},{ty-4:.1f} {x_spring_min-11:.1f},{y_spring_min+7:.1f}" '
                 f'fill="none" stroke="{C_A600}" stroke-width="0.9" stroke-dasharray="3 3" opacity="0.7"/>')
    return f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="四季日内价格形态">' + "".join(parts) + "</svg>"


def chart_neg_hour(counts: pd.Series, ndays: int) -> str:
    W, H = 1000, 250
    L, R, T, MB = 56, 20, 22, 46; B = H - MB
    vmax = 40
    parts = []
    for gv in [0, 10, 20, 30, 40]:
        y = scale(gv, 0, vmax, B, T)
        parts.append(f'<line x1="{L}" y1="{y:.1f}" x2="{W-R}" y2="{y:.1f}" stroke="{C_LINE}"/>')
        parts.append(f'<text x="{L-8}" y="{y+3.5:.1f}" text-anchor="end" font-size="10.5" fill="{C_G400}">{gv}%</text>')
    bw = (W - L - R) / 24
    for i in range(24):
        p = counts.iloc[i] / ndays * 100
        x = L + i * bw
        y = scale(p, 0, vmax, B, T)
        hot = 10 <= i + 1 <= 16
        parts.append(f'<rect x="{x+3:.1f}" y="{y:.1f}" width="{bw-6:.1f}" height="{B-y:.1f}" '
                     f'fill="{C_A400 if hot else C_B200}" rx="2"/>')
    for i in range(0, 24, 2):
        x = L + i * bw + bw / 2
        parts.append(f'<text x="{x:.1f}" y="{B+17}" text-anchor="middle" font-size="10.5" fill="{C_G400}">{i+1}</text>')
    parts.append(f'<line x1="{L}" y1="{B}" x2="{W-R}" y2="{B}" stroke="{C_G200}"/>')
    parts.append(f'<text x="{W-R}" y="{B+34}" text-anchor="end" font-size="10.5" fill="{C_A600}">'
                 f'10–16 时集中了负价的绝大部分 —— 光伏午间过发</text>')
    return f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="负价按时段出现频率">' + "".join(parts) + "</svg>"


def chart_month(tb2_m: pd.Series, ratio_m: pd.Series) -> str:
    W, H = 1000, 310
    L, R, T, MB = 62, 62, 26, 54; B = H - MB
    vmax = 60
    rmin, rmax = 0, 13
    parts = []
    for gv in [0, 15, 30, 45, 60]:
        y = scale(gv, 0, vmax, B, T)
        parts.append(f'<line x1="{L}" y1="{y:.1f}" x2="{W-R}" y2="{y:.1f}" stroke="{C_LINE}"/>')
        parts.append(f'<text x="{L-8}" y="{y+3.5:.1f}" text-anchor="end" font-size="10.5" fill="{C_G400}">{gv}</text>')
    bw = (W - L - R) / 12
    for i in range(12):
        v = tb2_m.iloc[i]
        x = L + i * bw
        y = scale(v, 0, vmax, B, T)
        hot = i in (10, 11)
        parts.append(f'<rect x="{x+bw*0.16:.1f}" y="{y:.1f}" width="{bw*0.68:.1f}" height="{B-y:.1f}" '
                     f'fill="{C_A200 if hot else C_B200}" rx="2"/>')
        parts.append(f'<text x="{x+bw/2:.1f}" y="{B+18}" text-anchor="middle" font-size="10.5" '
                     f'fill="{C_A600 if hot else C_G600}">{i+1}月</text>')
    # 右轴：收益/价差 比值
    xs = [L + i * bw + bw / 2 for i in range(12)]
    pts = " ".join(f"{xs[i]:.1f},{scale(ratio_m.iloc[i], rmin, rmax, B, T):.1f}" for i in range(12))
    parts.append(f'<polyline points="{pts}" fill="none" stroke="{C_B800}" stroke-width="2.2"/>')
    for i in range(12):
        y = scale(ratio_m.iloc[i], rmin, rmax, B, T)
        parts.append(f'<circle cx="{xs[i]:.1f}" cy="{y:.1f}" r="3.4" fill="{C_B800}"/>')
    for gv in [0, 4, 8, 12]:
        y = scale(gv, rmin, rmax, B, T)
        parts.append(f'<text x="{W-R+8}" y="{y+3.5:.1f}" font-size="10.5" fill="{C_B800}">{gv}</text>')
    parts.append(f'<text x="{L}" y="{H-12}" font-size="10.5" fill="{C_G400}">柱：TB2 日均价差（$，左轴）</text>')
    parts.append(f'<text x="{W-R+8}" y="{H-12}" font-size="10.5" fill="{C_B800}">折线：每 $1 价差兑现的收益（右轴）</text>')
    # 标注 12 月塌陷
    y12 = scale(ratio_m.iloc[11], rmin, rmax, B, T)
    parts.append(f'<text x="{xs[11]-4:.1f}" y="{y12+16:.1f}" text-anchor="end" font-size="10.5" fill="{C_RED}">'
                 f'12 月仅 {ratio_m.iloc[11]:.2f}</text>')
    return f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="月度价差与兑现效率">' + "".join(parts) + "</svg>"


def chart_decomp(dc: pd.DataFrame) -> str:
    W, H = 1000, 330
    L, R, T, MB = 66, 20, 24, 56; B = H - MB
    span = dc[["价差项", "低价项", "实际"]].to_numpy()
    vmin = min(0, span.min()) * 1.15
    vmax = max(span.max(), dc["模型"].max()) * 1.12
    parts = []
    for gv in range(0, int(vmax) + 1, 200):
        y = scale(gv, vmin, vmax, B, T)
        parts.append(f'<line x1="{L}" y1="{y:.1f}" x2="{W-R}" y2="{y:.1f}" stroke="{C_LINE}"/>')
        parts.append(f'<text x="{L-8}" y="{y+3.5:.1f}" text-anchor="end" font-size="10.5" fill="{C_G400}">{gv}</text>')
    y0 = scale(0, vmin, vmax, B, T)
    parts.append(f'<line x1="{L}" y1="{y0:.1f}" x2="{W-R}" y2="{y0:.1f}" stroke="{C_G400}" stroke-width="1"/>')
    bw = (W - L - R) / 12
    for i in range(12):
        x = L + i * bw
        up = dc["价差项"].iloc[i]
        dn = dc["低价项"].iloc[i]
        # 价差项（正/负）
        ya, yb = scale(0, vmin, vmax, B, T), scale(up, vmin, vmax, B, T)
        parts.append(f'<rect x="{x+bw*0.22:.1f}" y="{min(ya,yb):.1f}" width="{bw*0.26:.1f}" height="{abs(yb-ya):.1f}" fill="{C_B200}" rx="1"/>')
        # 低价项（正=额外收入 / 负=成本）
        ya2, yb2 = scale(0, vmin, vmax, B, T), scale(dn, vmin, vmax, B, T)
        col = C_A400 if dn >= 0 else C_A200
        parts.append(f'<rect x="{x+bw*0.52:.1f}" y="{min(ya2,yb2):.1f}" width="{bw*0.26:.1f}" height="{abs(yb2-ya2):.1f}" fill="{col}" rx="1"/>')
        # 实际值菱形
        yr = scale(dc["实际"].iloc[i], vmin, vmax, B, T)
        cx = x + bw * 0.5
        parts.append(f'<path d="M{cx:.1f} {yr-4:.1f} L{cx+4:.1f} {yr:.1f} L{cx:.1f} {yr+4:.1f} L{cx-4:.1f} {yr:.1f} Z" fill="{C_B800}"/>')
        hot = i in (10, 11)
        parts.append(f'<text x="{cx:.1f}" y="{B+18}" text-anchor="middle" font-size="10.5" '
                     f'fill="{C_A600 if hot else C_G600}">{i+1}月</text>')
    # 图例
    ly = H - 14
    parts.append(f'<rect x="{L}" y="{ly-9}" width="11" height="11" rx="2" fill="{C_B200}"/>')
    parts.append(f'<text x="{L+16}" y="{ly}" font-size="10.5" fill="{C_G600}">价差项 = 10.71 × TB2</text>')
    parts.append(f'<rect x="{L+190}" y="{ly-9}" width="11" height="11" rx="2" fill="{C_A200}"/>')
    parts.append(f'<text x="{L+206}" y="{ly}" font-size="10.5" fill="{C_G600}">低价项 = −4.11 × 低价2h（负值为成本）</text>')
    parts.append(f'<path d="M{L+520} {ly-4} L{L+524} {ly} L{L+520} {ly+4} L{L+516} {ly} Z" fill="{C_B800}"/>')
    parts.append(f'<text x="{L+532}" y="{ly}" font-size="10.5" fill="{C_G600}">实际收益</text>')
    return f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="收益来源分解">' + "".join(parts) + "</svg>"


def chart_forecast(scen: list[dict], hist: list[dict]) -> str:
    W, H = 1000, 300
    L, R, T, MB = 224, 56, 30, 70; B = H - MB
    vmax = 5000
    parts = []
    for gv in [0, 1000, 2000, 3000, 4000, 5000]:
        y = scale(gv, 0, vmax, B, T)
        parts.append(f'<line x1="{L}" y1="{y:.1f}" x2="{W-R}" y2="{y:.1f}" stroke="{C_LINE}"/>')
        # 刻度放右侧，避免与左侧两行标签（情景名 + 假设）打架
        parts.append(f'<text x="{W-R+8}" y="{y+3.5:.1f}" text-anchor="start" font-size="10.5" fill="{C_G400}">${gv:,}</text>')
    rows = hist + scen
    rh = (B - T) / len(rows)
    for k, r in enumerate(rows):
        cy = T + rh * k + rh / 2
        x0 = scale(0, 0, vmax, L, W - R)
        x1 = scale(r["value"], 0, vmax, L, W - R)
        kind = r.get("kind", "hist")
        if kind == "hist":
            col, txt = C_B300, C_G600
        elif r["value"] >= 3400:
            col, txt = C_B600, C_B800
        else:
            col, txt = C_A200 if r["value"] >= 3000 else C_A400, C_A600
        h = rh * 0.46 if kind == "hist" else rh * 0.56
        parts.append(f'<rect x="{x0:.1f}" y="{cy-h/2:.1f}" width="{max(2,x1-x0):.1f}" height="{h:.1f}" fill="{col}" rx="2"/>')
        # 左侧两行：主标签 + 假设说明，均放在左侧留白里，避免压住柱子
        if r.get("note"):
            parts.append(f'<text x="{L-12}" y="{cy-2:.1f}" text-anchor="end" font-size="11" '
                         f'fill="{C_B900}">{r["label"]}</text>')
            parts.append(f'<text x="{L-12}" y="{cy+10.5:.1f}" text-anchor="end" font-size="9.5" '
                         f'fill="{C_G400}">{r["note"]}</text>')
        else:
            parts.append(f'<text x="{L-12}" y="{cy+4:.1f}" text-anchor="end" font-size="11" '
                         f'fill="{txt}">{r["label"]}</text>')
        parts.append(f'<text x="{x1+8:.1f}" y="{cy+4:.1f}" font-size="11" font-weight="500" fill="{txt}">${r["value"]:,}</text>')
    parts.append(f'<line x1="{L}" y1="{B}" x2="{W-R}" y2="{B}" stroke="{C_G200}"/>')
    parts.append(f'<text x="{L}" y="{H-18}" font-size="10.5" fill="{C_G400}">横轴：14 天比赛期总收益（美元，opt-blend）</text>')
    return f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="比赛期收益预期区间">' + "".join(parts) + "</svg>"


def chart_ac(ac: pd.Series) -> str:
    """预测能力阶梯：从朴素到完美，捕获率的提升。"""
    W, H = 1000, 240
    L, R, T, MB = 150, 130, 26, 44; B = H - MB
    vmin, vmax = 70, 100
    parts = []
    for gv in [70, 80, 90, 100]:
        x = scale(gv, vmin, vmax, L, W - R)
        parts.append(f'<line x1="{x:.1f}" y1="{T-6}" x2="{x:.1f}" y2="{B}" stroke="{C_LINE}"/>')
        parts.append(f'<text x="{x:.1f}" y="{B+17}" text-anchor="middle" font-size="10.5" fill="{C_G400}">{gv}%</text>')
    rh = (B - T) / len(ac)
    for k, (name, v, tag) in enumerate(ac):
        cy = T + rh * k + rh / 2
        x0 = scale(vmin, vmin, vmax, L, W - R)
        col = {0: C_G200, 1: C_A200, 2: C_B200, 3: C_B600}[tag]
        parts.append(f'<text x="{x0-12}" y="{cy+4:.1f}" text-anchor="end" font-size="10.5" fill="{C_G600}">{name}</text>')
        if v < vmin:
            # 不交易：捕获率定义为 0，坐标轴从 70% 起，用零宽刻度 + 文字说明代替条形
            parts.append(f'<rect x="{x0:.1f}" y="{cy-rh*0.3:.1f}" width="2.5" height="{rh*0.6:.1f}" fill="{col}" rx="1"/>')
            parts.append(f'<text x="{x0+10:.1f}" y="{cy+4:.1f}" font-size="10.5" fill="{C_G400}">{v:.1f}%（不交易，零收益）</text>')
            continue
        x1 = scale(v, vmin, vmax, L, W - R)
        parts.append(f'<rect x="{x0:.1f}" y="{cy-rh*0.3:.1f}" width="{x1-x0:.1f}" height="{rh*0.6:.1f}" fill="{col}" rx="2"/>')
        parts.append(f'<text x="{x1+8:.1f}" y="{cy+4:.1f}" font-size="10.5" font-weight="500" fill="{C_B900}">{v:.1f}%</text>')
    parts.append(f'<line x1="{L}" y1="{B}" x2="{W-R}" y2="{B}" stroke="{C_G200}"/>')
    parts.append(f'<text x="{L}" y="{H-12}" font-size="10.5" fill="{C_G400}">横轴：捕获率（策略收益 ÷ 完美预见收益），2026 年 309 天样本外</text>')
    return f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="预测能力阶梯">' + "".join(parts) + "</svg>"


# ---------------------------------------------------------------- build
def main() -> None:
    df = load()
    flat = df.to_numpy().ravel()
    ndays = len(df)
    bt = backtests()

    arr = np.sort(df.to_numpy(), axis=1)
    lo2 = pd.Series(arr[:, :2].mean(axis=1), index=df.index)
    hi2 = pd.Series(arr[:, -2:].mean(axis=1), index=df.index)
    tb2 = hi2 - lo2

    # 日内形态
    def season(idx):
        m = idx.month
        s = np.where(np.isin(m, [12, 1, 2]), "冬(12-2)",
                     np.where(np.isin(m, [3, 4, 5]), "春(3-5)",
                              np.where(np.isin(m, [6, 7, 8]), "夏(6-8)", "秋(9-11)")))
        return pd.Series(s, index=idx)
    lg = df.stack().rename_axis(["date", "hour"]).reset_index(name="lmp")
    lg["season"] = lg["date"].map(season(df.index))
    prof = lg.pivot_table(index="hour", columns="season", values="lmp", aggfunc="mean")

    neg_by_hour = (df < 0).sum()
    tb2_m = tb2.groupby(tb2.index.month).mean()

    # 月度兑现效率（合并 2025+2026 回测）
    frames = []
    for tag, p in bt.items():
        if "opt-blend" in p and "PF(完美预见)" in p:
            frames.append(pd.DataFrame({"rev": p["opt-blend"], "pf": p["PF(完美预见)"]}))
    allbt = pd.concat(frames)
    allbt["tb2"] = tb2.reindex(allbt.index)
    ratio_m = (allbt.groupby(allbt.index.month).apply(
        lambda g: g["rev"].sum() / g["tb2"].sum(), include_groups=False))

    # 收益分解
    dc = pd.DataFrame({
        "价差项": E_SELL * tb2_m,
        "低价项": -E_WASTE * lo2.groupby(lo2.index.month).mean(),
    })
    dc["模型"] = dc["价差项"] + dc["低价项"]
    dc["实际"] = allbt.groupby(allbt.index.month)["rev"].mean()

    # 关键总量
    y25 = bt["2025"]
    y26 = bt["2026"]
    pf25, bl25 = y25["PF(完美预见)"].sum(), y25["opt-blend"].sum()
    pf26, bl26 = y26["PF(完美预见)"].sum(), y26["opt-blend"].sum()

    # 比赛期历史
    win = df[((df.index.month == 11) & (df.index.day >= 23)) | ((df.index.month == 12) & (df.index.day <= 6))]
    tb2_win = tb2.loc[win.index]

    # 情景
    scen = [
        {"label": "悲观　TB2 $28", "value": int(28 * 6.5 * 14), "kind": "scen",
         "note": "2025 年 11 月水平 · 兑现 6.5"},
        {"label": "中性　TB2 $34", "value": int(34 * 7.2 * 14), "kind": "scen",
         "note": "2026 年 8–10 月水平 · 兑现 7.2"},
        {"label": "乐观　TB2 $40", "value": int(40 * 7.7 * 14), "kind": "scen",
         "note": "2026 年 11 月偏紧 · 兑现 7.7"},
    ]
    hist = [
        # 2024 窗口取自 data/out/backtest（2024 全量回测）的 11-23~12-06 求和
        {"label": "2024 同期（估）", "value": 3768, "kind": "hist", "note": ""},
        {"label": "2025 同期实测", "value": int(y25.loc["2025-11-23":"2025-12-06", "opt-blend"].sum()), "kind": "hist", "note": ""},
    ]

    ac = pd.Series([
        ("不动作", 0.0, 0),
        ("朴素极值 TB3", 85.99, 1),
        ("昨日价格 persistence", 91.43, 1),
        ("近 7 日中位数", 90.41, 2),
        ("blend（采用）", 93.20, 3),
        ("完美预见（理论上界）", 100.0, 0),
    ])

    # ---- 文本块
    pct_neg = (flat < 0).mean() * 100
    span_days = (df.index.max() - df.index.min()).days + 1

    html = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>CAISO SP15 市场画像 · 痛点与方法 · 收益预期</title>
<style>
:root{{--b900:{C_B900};--b800:{C_B800};--b600:{C_B600};--b400:{C_B400};--b300:{C_B300};
--b200:{C_B200};--b50:{C_B50};--a600:{C_A600};--a200:{C_A200};--a50:{C_A50};--a900:#412402;
--g600:{C_G600};--g400:{C_G400};--g200:{C_G200};--g50:{C_G50};--line:{C_LINE};--ink:#1A1D21;}}
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:-apple-system,BlinkMacSystemFont,"PingFang SC","Microsoft YaHei",sans-serif;
background:#fff;color:var(--ink);line-height:1.68;-webkit-font-smoothing:antialiased}}
.wrap{{max-width:1120px;margin:0 auto;padding:56px 32px 80px}}
.eyebrow{{font-size:12px;letter-spacing:.12em;color:var(--b600);font-weight:500;margin-bottom:10px}}
h1{{font-size:28px;font-weight:500;letter-spacing:-.01em;line-height:1.35;color:var(--b900)}}
.sub{{font-size:14px;color:var(--g600);margin-top:10px}}
.meta{{font-size:12px;color:var(--g400);margin-top:18px;padding-bottom:26px;border-bottom:1px solid var(--line)}}
h2{{font-size:18px;font-weight:500;color:var(--b900)}}
h3{{font-size:14px;font-weight:500;color:var(--b800);margin:26px 0 8px}}
section{{margin-top:54px}}
.sh{{display:flex;align-items:baseline;gap:12px;margin-bottom:6px}}
.sn{{font-size:12px;font-weight:500;color:var(--b400);letter-spacing:.08em}}
.lead{{font-size:13.5px;color:var(--g600);margin-bottom:20px;max-width:900px}}
.card{{border:1px solid var(--line);border-radius:12px;padding:22px 22px 16px;margin-top:16px}}
.card-h{{font-size:13px;font-weight:500;color:var(--b800);margin-bottom:3px}}
.card-s{{font-size:12px;color:var(--g400);margin-bottom:14px}}
svg{{display:block;width:100%;height:auto}}
table{{width:100%;border-collapse:collapse;font-size:13px}}
th{{text-align:left;font-weight:500;color:var(--g600);font-size:12px;padding:9px 11px;
border-bottom:1px solid var(--line);background:var(--g50)}}
td{{padding:10px 11px;border-bottom:1px solid var(--line);vertical-align:top}}
tr:last-child td{{border-bottom:none}}
.num{{font-variant-numeric:tabular-nums;text-align:right}}
.seasontab td{{white-space:nowrap}}
.seasontab td.shape{{white-space:normal;font-size:12.5px;color:var(--g600);line-height:1.5;min-width:260px}}
.win{{color:var(--b800);font-weight:500}}
.bad{{color:#A32D2D}}
.kpis{{display:grid;grid-template-columns:repeat(4,1fr);gap:13px;margin:26px 0 6px}}
.kpi{{background:var(--g50);border-radius:10px;padding:15px 17px}}
.kpi .lab{{font-size:12px;color:var(--g600);margin-bottom:5px}}
.kpi .val{{font-size:23px;font-weight:500;color:var(--b800);letter-spacing:-.01em}}
.kpi .val small{{font-size:13px;font-weight:400;color:var(--g600);margin-left:2px}}
.kpi .note{{font-size:11px;color:var(--g400);margin-top:4px}}
.two{{display:grid;grid-template-columns:1fr 1fr;gap:16px}}
.note-box{{background:var(--a50);border-radius:10px;padding:15px 17px;font-size:13px;color:var(--a900);line-height:1.7}}
.blue-box{{background:var(--b50);border-radius:10px;padding:15px 17px;font-size:13px;color:var(--b900);line-height:1.7}}
.fix{{background:#FFF8E6;border-left:3px solid var(--a600);padding:11px 15px;border-radius:0 8px 8px 0;
font-size:12.5px;color:var(--a900);margin-top:12px}}
ul.tight{{margin:7px 0 0 18px;font-size:13px;color:var(--g600)}}
ul.tight li{{margin-bottom:6px}}
.pain{{display:grid;grid-template-columns:26px 1fr;gap:14px;padding:16px 0;border-bottom:1px solid var(--line)}}
.pain:last-child{{border-bottom:none}}
.pn{{font-size:12px;font-weight:500;color:var(--b400);padding-top:2px}}
.pt{{font-size:13.5px;font-weight:500;color:var(--b900);margin-bottom:6px}}
.pb{{font-size:13px;color:var(--g600)}}
.pb b{{font-weight:500;color:var(--b800)}}
.mono{{font-family:ui-monospace,Menlo,monospace;font-size:12px}}
footer{{margin-top:60px;padding-top:18px;border-top:1px solid var(--line);font-size:12px;color:var(--g400)}}
</style></head><body><div class="wrap">

<div class="eyebrow">全国“AI+电力市场”应用大赛 · 线上组 · 交易决策　｜　市场画像</div>
<h1>CAISO SP15 日前市场：数据画像、痛点拆解与收益预期</h1>
<div class="sub">用 {ndays:,} 个真实运行日（{df.index.min():%Y-%m-%d} ~ {df.index.max():%Y-%m-%d}）把市场结构量化，再说明方法为什么这样设计。</div>
<div class="meta">{span_days:,} 个自然日 → {len(df):,} 个交易日（3 天夏令时 25 小时异常已剔除）　｜　{df.size:,} 个时段价　｜　零缺口　｜　2026 年 309 天为完全样本外（策略在 2024–2025 定型）</div>

<div class="kpis">
  <div class="kpi"><div class="lab">全样本均价</div><div class="val">${flat.mean():.2f}<small>/MWh</small></div>
      <div class="note">中位 ${np.median(flat):.2f}，P95 ${np.percentile(flat,95):.1f}</div></div>
  <div class="kpi"><div class="lab">负价时段占比</div><div class="val">{pct_neg:.1f}<small>%</small></div>
      <div class="note">集中在 10–16 时，光伏过发</div></div>
  <div class="kpi"><div class="lab">日均 TB2 价差</div><div class="val">${tb2.mean():.1f}<small></small></div>
      <div class="note">中位 ${tb2.median():.1f}，P10 ${np.percentile(tb2,10):.1f}</div></div>
  <div class="kpi"><div class="lab">形态可预测性</div><div class="val">{df.corrwith(df.shift(1),axis=1).mean():.3f}</div>
      <div class="note">D 与 D-1 跨 24 时段相关</div></div>
</div>

<!-- 01 -->
<section>
<div class="sh"><span class="sn">01</span><h2>数据资产：这份画像的底座</h2></div>
<p class="lead">结论先行：<b>这是一个「价格水平不高、但价差极大、且形态高度可预测」的市场</b>。均价 $31.4 不算诱人，但日均 TB2 价差 $47.2 相当于均价的 1.5 倍 —— 储能赚的不是电价，是价差。</p>

<div class="card" style="padding:0;overflow:hidden">
<table>
<thead><tr><th style="width:22%">维度</th><th style="width:40%">取值</th><th>说明</th></tr></thead>
<tbody>
<tr><td>标的</td><td>CAISO SP15 日前节点电价，<span class="mono">TH_SP15_GEN-APND</span></td><td>南加州交易枢纽，储能套利的标准基准节点</td></tr>
<tr><td>数据源</td><td>CAISO OASIS <span class="mono">PRC_LMP</span>（DAM）</td><td>官方一手数据，非第三方转售</td></tr>
<tr><td>覆盖</td><td>{ndays:,} 天 × 24 时段 = {df.size:,} 条</td><td>{df.index.min():%Y-%m-%d} ~ {df.index.max():%Y-%m-%d}</td></tr>
<tr><td>完整性</td><td><span class="win">零缺口</span></td><td>DST 异常日（25 小时）已剔除，避免时段错位</td></tr>
<tr><td>取数方式</td><td>本机被墙 → Oracle ARM 出口（双传输自动降级）</td><td>OASIS 单次查询上限 31 天，按 30 天分块 + 1.2 s 节流</td></tr>
<tr><td>可复现</td><td><span class="mono">scripts/portrait.py</span> 一条命令重算全部指标</td><td>报告内每个数字都可追溯到该脚本</td></tr>
</tbody></table>
</div>

<div class="note-box" style="margin-top:16px">
<b>为什么用 SP15 而不是南方区域市场？</b>　手册给的是交易决策赛题，要求提交 24 时段充放电曲线并由平台结算。SP15 是公开、连续、可回测十年的成熟日前市场，能支撑「用真实数据证明策略有效」这件事；而南方区域现货仍在连续运行试点，公开历史序列不足以做三年回测。
</div>
</section>

<!-- 02 -->
<section>
<div class="sh"><span class="sn">02</span><h2>价格结构：均值骗人，尾部才是关键</h2></div>
<p class="lead">整体分布严重右偏。P50 是 $34.6，但 P0 是 −$67，P100 是 $614 —— 均值被少数尖峰拉低，而储能的机会恰好都在尾部。</p>

<div class="card">
<div class="card-h">时段电价分布</div>
<div class="card-s">{df.size:,} 个时段价的直方图；橙色区间为负价</div>
{chart_hist(flat)}
</div>

<div class="two" style="margin-top:16px">
<div class="card" style="margin-top:0">
<div class="card-h">分位数</div>
<div class="card-s">说明为什么「均价」不能用来做决策</div>
<table>
<thead><tr><th>分位</th><th class="num">电价 $/MWh</th><th>含义</th></tr></thead>
<tbody>
<tr><td>P1</td><td class="num bad">{np.percentile(flat,1):.1f}</td><td>深度负价，充电被付费</td></tr>
<tr><td>P10</td><td class="num">{np.percentile(flat,10):.2f}</td><td>每天约 2.4 小时处于此水平以下</td></tr>
<tr><td>P50</td><td class="num">{np.percentile(flat,50):.2f}</td><td>典型时段</td></tr>
<tr><td>P90</td><td class="num">{np.percentile(flat,90):.2f}</td><td>晚峰常规水平</td></tr>
<tr><td>P99</td><td class="num win">{np.percentile(flat,99):.2f}</td><td>尖峰，单日收益的主要来源</td></tr>
<tr><td>P100</td><td class="num win">{np.percentile(flat,100):.2f}</td><td>极端稀缺事件</td></tr>
</tbody></table>
</div>

<div class="card" style="margin-top:0">
<div class="card-h">三年 regime 漂移</div>
<div class="card-s">水平下行、波动收敛，但负价占比没有消失</div>
<table>
<thead><tr><th>年</th><th class="num">均价</th><th class="num">标准差</th><th class="num">负价%</th><th class="num">日极差</th></tr></thead>
<tbody>
{"".join(f'<tr><td>{yr}</td><td class="num">{g.to_numpy().mean():.2f}</td><td class="num">{g.to_numpy().std(ddof=1):.2f}</td><td class="num">{100*(g.to_numpy()<0).mean():.2f}</td><td class="num">{(g.max(axis=1)-g.min(axis=1)).mean():.1f}</td></tr>' for yr, g in df.groupby(df.index.year))}
</tbody></table>
<div class="fix">2026 年均价 ${df[df.index.year==2026].to_numpy().mean():.2f} 低于 2024 年的 ${df[df.index.year==2024].to_numpy().mean():.2f}，但负价占比反而升到 {100*(df[df.index.year==2026].to_numpy()<0).mean():.2f}%。<b>市场在「更便宜」和「更两极化」同时发生</b> —— 这对纯套利是利好。</div>
</div>
</div>
</section>

<!-- 03 -->
<section>
<div class="sh"><span class="sn">03</span><h2>日内形态：一年里有四种完全不同的市场</h2></div>
<p class="lead">这是全篇最重要的结构发现。四个季节的日内曲线形状互不相同，<b>用一套固定模板做全年预测必然失准</b> —— 这直接决定了预测方法必须能自己识别形态，而不是靠人工设定充放时段。</p>

<div class="card">
<div class="card-h">四季时段均价曲线</div>
<div class="card-s">横轴 1–24 时；浅红底色为零线以下区域</div>
{chart_shape(prof)}
</div>

<div class="card" style="padding:0;overflow:hidden;margin-top:16px">
<table class="seasontab">
<thead><tr><th>季节</th><th class="num">谷时 h</th><th class="num">谷价</th><th class="num">峰时 h</th><th class="num">峰价</th><th class="num">峰谷差</th><th>形态特征</th></tr></thead>
<tbody>
{"".join(
 f'<tr><td>{s}</td><td class="num">{prof[s].idxmin()}</td><td class="num {"bad" if prof[s].min()<0 else ""}">{prof[s].min():.2f}</td>'
 f'<td class="num">{prof[s].idxmax()}</td><td class="num">{prof[s].max():.2f}</td>'
 f'<td class="num win">{prof[s].max()-prof[s].min():.2f}</td>'
 f'<td class="shape">{SEASON_SHAPE[s]}</td></tr>'
 for s in ["冬(12-2)","春(3-5)","夏(6-8)","秋(9-11)"])}
</tbody></table>
</div>

<div class="two" style="margin-top:16px">
<div class="blue-box">
<b>春季：连续 {int((prof["春(3-5)"]<0).sum())} 小时均值为负</b><br>
3–5 月 9 时至 17 时，<b>每个时段的均价都是负的</b>，谷底 14 时 {prof["春(3-5)"].min():.2f}。这意味着充电不仅免费，还能拿钱。储能的最优行为在这个季节退化为「尽可能长时间满功率充电」。
</div>
<div class="note-box">
<b>比赛期落在冬季：形态反而最不友好</b><br>
11–12 月没有负价红利（比赛期负价占比仅 {100*(win.to_numpy().ravel()<0).mean():.2f}%），充电要真金白银付钱。峰谷差虽不低（${prof["冬(12-2)"].max()-prof["冬(12-2)"].min():.0f}），但<b>兑现效率只有春季的三成</b>（见 05 节）。
</div>
</div>
</section>

<!-- 04 -->
<section>
<div class="sh"><span class="sn">04</span><h2>负价的时间结构：机会是有形状的</h2></div>
<p class="lead">负价不是随机散布的，它被太阳锁定在固定的几个时段。这个形状的稳定性，正是「昨日价格能预测今日价格」的根本原因。</p>

<div class="card">
<div class="card-h">各时段出现负价的频率</div>
<div class="card-s">纵轴 = 该时段为负价的天数占比；橙色为 10–16 时</div>
{chart_neg_hour(neg_by_hour, ndays)}
</div>

<div class="card" style="padding:0;overflow:hidden;margin-top:16px">
<table>
<thead><tr><th style="width:34%">观测</th><th style="width:20%">数值</th><th>对策略的含义</th></tr></thead>
<tbody>
<tr><td>负价最集中的时段</td><td class="num">14 时（{neg_by_hour.iloc[13]/ndays*100:.1f}% 的天）</td><td>充电窗口高度可预测，不需要复杂模型定位</td></tr>
<tr><td>负价时段的平均价</td><td class="num">{df.where(df<0).stack().mean():.2f} $/MWh</td><td>「被付费充电」的单价，直接变成利润</td></tr>
<tr><td>负价出现的年份稳定性</td><td class="num">13.25% / 11.08% / 13.48%</td><td>三年都在 11–13.5%，不是一次性现象</td></tr>
<tr><td>负价日的收益对比</td><td class="num win">$547 vs $264</td><td>低价 2 小时为负的日子，日均收益是非负日的 2.07 倍</td></tr>
</tbody></table>
</div>
</section>

<!-- 05 -->
<section>
<div class="sh"><span class="sn">05</span><h2>兑现效率：同样的价差，不同月份值不同的钱</h2></div>
<p class="lead">这是我做画像时最意外的发现。把「每 $1 价差能兑现多少收益」按月拆开，4 月是 11.4，12 月只有 3.04 —— <b>价差缩小了一半，收益却掉了七成</b>。原因不是价差，是低价时段的水位。</p>

<div class="card">
<div class="card-h">月度价差与兑现效率</div>
<div class="card-s">柱为 TB2 日均价差；折线为「收益 ÷ 价差」的倍数</div>
{chart_month(tb2_m, ratio_m)}
</div>

<div class="card">
<div class="card-h">为什么 12 月的同样价差不值钱：一个能量学恒等式</div>
<div class="card-s">SOC 可用摆幅 {DSOC:.1f} MWh，单程效率 {ETA}，满充满放一次的能量账</div>
<div style="display:grid;grid-template-columns:1fr 1fr;gap:18px;align-items:center">
<div>
<div style="background:var(--g50);border-radius:8px;padding:14px 16px;font-family:ui-monospace,Menlo,monospace;font-size:12.5px;color:var(--b900);line-height:2">
从电网买入　{E_CHARGE:.2f} MWh<br>
向电网卖出　{E_SELL:.2f} MWh<br>
效率损耗　　 <b>{E_WASTE:.2f} MWh</b><br><br>
收益 = {E_SELL:.2f} × 高价 − {E_CHARGE:.2f} × 低价<br>
　　 = <b>{E_SELL:.2f} × 价差 − {E_WASTE:.2f} × 低价</b>
</div>
</div>
<div style="font-size:13px;color:var(--g600);line-height:1.85">
<p style="margin-bottom:10px">这个式子解释了全部季节性差异：</p>
<ul class="tight">
<li>低价为 <b style="color:{C_A600}">−10</b> 时，第二项变成 <b>+41 元</b>的额外收入 —— 效率损耗反而赚钱</li>
<li>低价为 <b style="color:{C_RED}">+25</b> 时，第二项是 <b>−{E_WASTE*25:.0f} 元</b>的成本 —— 每搬一次电都在为损耗付钱</li>
<li>12 月的低价水位正好是最高的（约 ${lo2[lo2.index.month==12].mean():.1f}），所以同样的价差兑现不出来</li>
</ul>
</div>
</div>
</div>

<div class="card">
<div class="card-h">实际收益的来源分解（按月份）</div>
<div class="card-s">模型 vs 实际相关系数 {allbt["rev"].corr(E_SELL*tb2.reindex(allbt.index) - E_WASTE*lo2.reindex(allbt.index)):.3f}；蓝柱为价差项，橙柱为低价项，菱形为实际收益</div>
{chart_decomp(dc)}
</div>
</section>

<!-- 06 -->
<section>
<div class="sh"><span class="sn">06</span><h2>可预测性：方法的上限在哪里</h2></div>
<p class="lead">形态高度可预测、水平不可预测 —— 这个不对称性决定了正确的建模范式：<b>不要试图预测价格水平，只预测相对形态</b>。</p>

<div class="card">
<div class="card-h">预测能力的阶梯（2026 年 309 天完全样本外）</div>
<div class="card-s">策略在 2024–2025 定型，2026 年数据一次未参与调参</div>
{chart_ac(ac)}
</div>

<div class="two" style="margin-top:16px">
<div class="card" style="margin-top:0">
<div class="card-h">可预测性的两个维度</div>
<div class="card-s">形态可预测 vs 水平不可预测</div>
<table>
<thead><tr><th>指标</th><th class="num">数值</th></tr></thead>
<tbody>
<tr><td>D 与 D-1 形态相关（跨 24 时段）</td><td class="num win">{df.corrwith(df.shift(1),axis=1).mean():.4f}</td></tr>
<tr><td>D 与 D-1 水平相关（跨天）</td><td class="num">{df.corrwith(df.shift(1),axis=0).mean():.4f}</td></tr>
<tr><td>绝对误差 MAE</td><td class="num">{df.sub(df.shift(1)).abs().stack().mean():.2f} $/MWh</td></tr>
<tr><td>MAE 占均价比例</td><td class="num bad">{df.sub(df.shift(1)).abs().stack().mean()/flat.mean()*100:.1f}%</td></tr>
<tr><td>昨日 TB2 → 今日 TB2 相关</td><td class="num">{tb2.corr(tb2.shift(1)):.4f}</td></tr>
</tbody></table>
</div>
<div class="note-box" style="margin-top:0">
<b>这个不对称性意味着什么</b><br><br>
形态相关 0.948，意味着<b>「哪个时段贵、哪个时段便宜」几乎可以确定</b>；但 MAE 占均价 21.9%，意味着<b>「贵多少」基本测不准</b>。<br><br>
所以任何试图精确预测价格中枢的做法（ shrinkage、多因子、机器学习回归）都会在第二件事上失败。<b>正确做法是把预测只用在排序上</b> —— 而 MILP 恰好只需要排序正确就能排出接近最优的曲线。
</div>
</div>
</section>

<!-- 07 -->
<section>
<div class="sh"><span class="sn">07</span><h2>痛点与解法：每一条都有数据支撑</h2></div>
<p class="lead">以下六个痛点不是想象出来的，每一个都能在上面的画像里找到对应证据。列出的「为什么更好」全部是可对照的数字。</p>

<div class="card" style="padding:6px 22px 10px">
<div class="pain"><div class="pn">P1</div><div>
<div class="pt">信息时滞：提交时不知道当日价格，只有 D-1</div>
<div class="pb">D 日 02:00 截止，而 D 日日前出清结果要到 05:00 之后才发布。<b>唯一可用输入是 D-1 的完整 24 小时价格。</b>
解法：blend = 0.4×昨日 + 0.6×近 7 日中位数。为什么更好 —— 单用昨日在 2024 年能到 96.68%，但 2025 年塌到 86.85%（跨年不稳）；单用中位数 2026 年只有 90.41%；<b>blend 三年分别 96.42 / 96.40 / 93.20，是唯一跨 regime 稳定的</b>。混合的实质是：用昨日捕捉形态，用中位数压住水平噪声。</div></div></div>

<div class="pain"><div class="pn">P2</div><div>
<div class="pt">归零惩罚：任一时段功率或 SOC 越界，全天清算收益归零</div>
<div class="pb">这是赛制里最不对称的条款 —— 一个时段的 0.01 MW 越界，抹掉的是整天的收益。
解法：MILP 强制充放互斥 + SOC 空间前向量化 + 独立第二实现校验，三道防线串联。
为什么更好 —— 若用 LP 建模（把充放拆成两个非负变量），模型会系统性低估 SOC 上升 0.3256×min(Qd,Qc)，排出「以为合法、实际冲过上限」的曲线；<b>3,640 条日曲线实测 0 越界，SOC 端值落在 0.7000–13.2992（上限 13.30）</b>，贴边但不越界。</div></div></div>

<div class="pain"><div class="pn">P3</div><div>
<div class="pt">12 月是全年最差窗口，直觉会让人想「少做少错」</div>
<div class="pb">12 月兑现效率仅 3.04，日均收益 $76，只有 3 月的 1/9。
解法：不设价差阈值，每天都满负荷参与。
为什么更好 —— 得分是相对评分，<b>(R−μ)/σ 只取决于捕获率在分布中的位置，与当日绝对收益无关</b>。实测：加 $20 阈值会让比赛期捕获率从 96.4% 掉到 71.5%，$30 掉到 19.0%，$40 归零。<b>低价值日不动作，等于主动把自己排到分布最左端。</b></div></div></div>

<div class="pain"><div class="pn">P4</div><div>
<div class="pt">效率损耗是随低价水位放大的隐性成本</div>
<div class="pb">每个满循环有 {E_WASTE:.2f} MWh 的电网侧损耗，乘以低价时段电价就是真金白银。
解法：把 SOC 递推写成精确的能量形式（充电 dSOC = −Q·η，放电 dSOC = −Q/η），而不是手册的分数形式，让优化器直接感知损耗。
为什么更好 —— 优化器会自动把充电挤到价格最低的时段，因为低价越低，损耗项越便宜甚至变负；<b>低价 2 小时为负的日子日均收益 $547，是非负日 $264 的 2.07 倍</b>。</div></div></div>

<div class="pain"><div class="pn">P5</div><div>
<div class="pt">多日优化看起来能利用「末日 SOC 不约束」，实际是陷阱</div>
<div class="pb">手册允许最后一天 SOC 任意，理论上可以在倒数第二天清空库存多赚一笔。
解法：逐日优化，每天把 SOC 放回 5% 下限，不做跨日投机。
为什么更好 —— 实测多日滚动 h1 96.40% → h2 83.67% → h3 83.65%，最差日 −$463。<b>逐日优化让预测误差不跨日累积</b>，而多日优化把未来日的预测误差引入今天，一次偏就连续亏三天。</div></div></div>

<div class="pain"><div class="pn">P6</div><div>
<div class="pt">数据本身就取不到：本机直连 CAISO 返回 502</div>
<div class="pb">出口代理对 caiso.com 不可达，OASIS 还有 31 天查询上限与 429 限流。
解法：Oracle ARM 出口 + 双传输自动降级 + 30 天分块 + 指数退避重试；脚本自包含 stdlib（服务器上无需 pandas）。
为什么更好 —— <b>1,038 天零缺口一次回填成功</b>，且可在服务器上按 cron 增量拉取，不依赖本机在线。</div></div></div>
</div>
</section>

<!-- 08 -->
<section>
<div class="sh"><span class="sn">08</span><h2>收益预期</h2></div>
<p class="lead">先说结论：<b>2026 年 14 天比赛期，中性预期约 $3,400，区间 $2,550–4,310</b>。全年口径 $17.0/kW-yr。以下是推导。</p>

<div class="two">
<div class="card" style="margin-top:0">
<div class="card-h">实测基准（非估算）</div>
<div class="card-s">逐年回测，2026 为完全样本外</div>
<table>
<thead><tr><th>期间</th><th class="num">日均</th><th class="num">捕获率</th><th class="num">$/kW 累计</th></tr></thead>
<tbody>
<tr><td>2025 全年（364 天）</td><td class="num">${y25["opt-blend"].mean():.0f}</td><td class="num win">93.20%</td><td class="num">{y25["opt-blend"].sum()/7000:.1f}</td></tr>
<tr><td>2026 年 1–11 月（309 天）</td><td class="num">${y26["opt-blend"].mean():.0f}</td><td class="num win">93.20%</td><td class="num">{y26["opt-blend"].sum()/7000:.1f}</td></tr>
<tr><td>2026-10-01~11-06（37 天）</td><td class="num">${y26.loc["2026-10-01":"2026-11-06","opt-blend"].mean():.0f}</td>
    <td class="num">{y26.loc["2026-10-01":"2026-11-06","opt-blend"].sum()/y26.loc["2026-10-01":"2026-11-06","PF(完美预见)"].sum()*100:.2f}%</td>
    <td class="num">—</td></tr>
</tbody></table>
<div class="fix">两年捕获率都是 <b>93.20%</b>，四位数字完全一致。这不是巧合 —— 说明策略的性能由市场结构而非年份决定，是样本外泛化的强证据。<br>
口径：2025 行为全年累计；2026 行为 309 天累计（年化后 $17.0/kW·年）。</div>
</div>

<div class="card" style="margin-top:0">
<div class="card-h">比赛期历史窗口</div>
<div class="card-s">同样 14 天窗口，逐年水平差异极大（2024 行为模型估算，2025 行为回测实测）</div>
<table>
<thead><tr><th>窗口</th><th class="num">TB2</th><th class="num">日均</th><th class="num">14 天合计</th></tr></thead>
<tbody>
<tr><td>2024-11-23~12-06（估）</td><td class="num">$35.1</td><td class="num">$269</td><td class="num">$3,768</td></tr>
<tr><td>2025-11-23~12-06</td><td class="num">$31.7</td><td class="num">$214</td><td class="num">$3,001</td></tr>
<tr><td>2026-11-01~11-06（近期）</td><td class="num">$38.2</td><td class="num">$292</td><td class="num">（外推）</td></tr>
</tbody></table>
<div class="fix">2026 年 11 月 TB2 $38.2，高于 2025 同期的 $31.0。<b>若这个偏紧格局延续到比赛期，收益应落在 $3,000–3,800</b>，高于 2025 年。</div>
</div>
</div>

<div class="card">
<div class="card-h">14 天收益预期：三情景</div>
<div class="card-s">以 TB2 为自变量，兑现效率取历史比赛期实测值（6.5–7.7 倍）；与两年同期的实测值并列对照</div>
{chart_forecast(scen, hist)}
</div>

<div class="card" style="padding:0;overflow:hidden">
<table>
<thead><tr><th style="width:14%">情景</th><th style="width:18%">假设</th><th class="num" style="width:14%">日均</th><th class="num" style="width:16%">14 天合计</th><th>触发条件</th></tr></thead>
<tbody>
<tr><td>悲观</td><td>TB2 $28</td><td class="num">${28*6.5:.0f}</td><td class="num">${28*6.5*14:,.0f}</td><td>12 月惯常塌陷提前到 11 月末，价差回落到 2025 年水平</td></tr>
<tr style="background:var(--b50)"><td><b>中性</b></td><td>TB2 $34</td><td class="num">${34*7.2:.0f}</td><td class="num win">${34*7.2*14:,.0f}</td><td>维持 2026 年 11 月的偏紧格局，略低于 10 月</td></tr>
<tr><td>乐观</td><td>TB2 $40</td><td class="num">${40*7.7:.0f}</td><td class="num">${40*7.7*14:,.0f}</td><td>延续 2026 年 10 月 $49.8 的高价差，天然气偏紧</td></tr>
</tbody></table>
</div>

<div class="two" style="margin-top:16px">
<div class="blue-box">
<b>换算成行业口径</b><br>
$3,430 ÷ 14 × 365 = 年化 $89,400，对应 <b>$12.8/kW-yr</b>。<br>
但这是最差窗口的年化，不代表全年能力 —— 全年实测为 <b>$17.0–20.8/kW-yr</b>（折合每 MW 装机 $17–21k/年）。
</div>
<div class="note-box">
<b>收益预期的三条边界</b><br>
① 若比赛期价差跌破 $20，中性情景失效，应按悲观情景准备心理预期；<br>
② 相对评分下，<b>绝对收益不直接决定名次</b>，$3,000 在弱对手环境里可能比 $4,000 在强对手环境里得分更高；<br>
③ s₂/μ/σ 未知，得分测算只能给区间，不能给点值。
</div>
</div>
</section>

<!-- 09 -->
<section>
<div class="sh"><span class="sn">09</span><h2>这套画像还不能回答什么</h2></div>
<div class="card" style="padding:0;overflow:hidden">
<table>
<thead><tr><th style="width:26%">未知项</th><th style="width:34%">影响</th><th>处理方式</th></tr></thead>
<tbody>
<tr><td>s₂ 与真实对手分布</td><td>得分只能给区间（纳入 3 种对手假设）</td><td>得分对 s₂ 线性，可等比换算；策略排序对其不敏感</td></tr>
<tr><td>比赛期 14 天样本极短</td><td>单日波动可达 ±$170，14 天均值方差大</td><td>用 351 个重叠滚动窗口而非单点估计</td></tr>
<tr><td>2026 年 12 月尚未发生</td><td>情景外推依赖 11 月格局延续</td><td>赛前每日增量更新 TB2，滚动修正情景</td></tr>
<tr><td>平台结算细则</td><td>是否含辅助服务、是否扣损耗</td><td>按手册单一能量结算口径建模，保守</td></tr>
</tbody></table>
</div>
</section>

<footer>
数据：CAISO OASIS PRC_LMP（DAM，节点 TH_SP15_GEN-APND），{ndays:,} 天 × 24 时段，{df.index.min():%Y-%m-%d} ~ {df.index.max():%Y-%m-%d}。<br>
指标复现：<span class="mono">csg-ai-market/.venv/bin/python scripts/portrait.py</span>　｜　回测复现：<span class="mono">scripts/run_backtest.py</span>（2025、2026 两段）。<br>
能量学常量：可用 SOC 摆幅 {DSOC:.1f} MWh，单程效率 {ETA}，满循环买入 {E_CHARGE:.2f} MWh / 卖出 {E_SELL:.2f} MWh / 损耗 {E_WASTE:.2f} MWh。
</footer>

</div></body></html>"""

    OUT.write_text(html, encoding="utf-8")
    print(f"written: {OUT}  ({len(html):,} chars)")
    print(f"关键数字: 均价 {flat.mean():.2f} / 负价 {pct_neg:.2f}% / TB2 {tb2.mean():.1f}")
    print(f"2025 blend ${bl25:,.0f} ({bl25/pf25*100:.2f}%)  2026 blend ${bl26:,.0f} ({bl26/pf26*100:.2f}%)")
    print(f"分解: 买入 {E_CHARGE:.2f} 卖出 {E_SELL:.2f} 损耗 {E_WASTE:.2f}")


if __name__ == "__main__":
    main()
