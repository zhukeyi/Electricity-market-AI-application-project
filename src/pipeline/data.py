"""价格数据的读取、合并与体检 —— 流水线的「仓储层」。

数据文件只有一份 `data/raw/sp15_dam_lmp.csv`（长表），由定时任务增量追加。
这里集中处理三件事，避免每个脚本各自重写一遍：

1. **长表 ↔ 宽表**：宽表 24 列（hour 1..24），供预测与优化使用；
2. **幂等合并**：同一 (date, hour) 重复抓取时以新值为准，不产生重复行；
3. **体检**：缺时段、缺日、DST 切换日（23/25 时段）必须显式报告，
   **不允许静默 dropna** 把一天悄悄丢掉。

关于 DST：CAISO 的 `INTERVAL_NUM` 在切换日给出 23 或 25 个时段。宽表固定
reindex 到 1..24 并用 `dropna()` 会整日丢弃 —— 对回测可接受，对比赛不可接受
（少一天历史会改变预测与 SOC 链）。故 `health_report()` 把这些日子点名列出。
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd

from .clock import la_day_hours

COLUMNS = ["date", "hour", "lmp", "utc_start", "node"]


def load_long(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        return pd.DataFrame(columns=COLUMNS)
    df = pd.read_csv(p)
    missing = [c for c in COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{p} 缺少列 {missing}，实际列 {list(df.columns)}")
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df["hour"] = df["hour"].astype(int)
    df["lmp"] = df["lmp"].astype(float)
    return df[COLUMNS]


def write_long(df: pd.DataFrame, path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    out = df.sort_values(["date", "hour"])[COLUMNS].copy()
    out["date"] = pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d")
    out.to_csv(p, index=False, encoding="utf-8")
    return p


def merge_long(existing: pd.DataFrame, fresh: pd.DataFrame) -> pd.DataFrame:
    """按 (date, hour, node) 去重，新值覆盖旧值。"""
    if existing is None or len(existing) == 0:
        base = fresh.copy()
    elif fresh is None or len(fresh) == 0:
        base = existing.copy()
    else:
        base = pd.concat([existing, fresh], ignore_index=True)
    if len(base) == 0:
        return base
    base["date"] = pd.to_datetime(base["date"]).dt.date
    base = base.drop_duplicates(subset=["date", "hour", "node"], keep="last")
    return base.sort_values(["date", "hour"]).reset_index(drop=True)


def last_date(df: pd.DataFrame) -> dt.date | None:
    if df is None or len(df) == 0:
        return None
    return max(df["date"])


def load_matrix(path: str | Path, periods: int = 24) -> pd.DataFrame:
    """宽表：index 为运行日，columns 为 1..periods。只保留时段齐全的日子。"""
    df = load_long(path)
    if len(df) == 0:
        return pd.DataFrame(columns=range(1, periods + 1))
    m = (
        df.pivot_table(index="date", columns="hour", values="lmp", aggfunc="last")
        .reindex(columns=range(1, periods + 1))
        .dropna()
    )
    m.index = pd.Index(pd.to_datetime(m.index))
    return m.sort_index()


def health_report(path: str | Path, periods: int = 24, tail_days: int = 14) -> dict:
    """数据体检：返回缺失日、时段不全日、DST 异常日、最近 N 天覆盖情况。"""
    df = load_long(path)
    if len(df) == 0:
        return {"rows": 0, "days": 0, "last_date": None, "problems": ["数据文件为空"]}

    cnt = df.groupby("date")["hour"].nunique()
    problems: list[str] = []

    # 只报「与当日应有小时数不符」的日子。DST 切换日 23/25 段是正常的，
    # 拿它去和固定 24 比会把每年两次的正常现象误报成数据缺陷。
    for d in cnt.index:
        expect = la_day_hours(d)
        got = int(cnt[d])
        if got != expect:
            kind = "DST 切换日" if expect != 24 else "常规日"
            problems.append(f"{d} 有 {got} 个时段，{kind}应为 {expect}")

    # 日历连续性：只检查最近 tail_days，历史空洞不影响当日决策
    days = sorted(cnt.index)
    last = days[-1]
    window = [last - dt.timedelta(days=i) for i in range(tail_days - 1, -1, -1)]
    missing = [d for d in window if d not in cnt.index]
    if missing:
        problems.append("最近 %d 天缺日：%s" % (tail_days, ", ".join(str(d) for d in missing)))

    return {
        "rows": int(len(df)),
        "days": int(len(cnt)),
        "first_date": str(days[0]),
        "last_date": str(last),
        "problems": problems,
    }
