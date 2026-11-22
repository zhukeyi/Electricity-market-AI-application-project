#!/usr/bin/env python3
"""全链路彩排 —— 用历史数据把「预测 → 优化 → 落盘 → 自检 → 结算」跑一遍。

与回测的区别：回测只算收益，彩排要**证明整条提交链路没有机械错误**。
因此每天都会真的生成一个 CSV、真的跑一遍字节校验、真的把 SOC 链推下去，
再用当日实际价格结算，最后与「完美预见」对比得出捕获率。

    python scripts/rehearse.py --days 14
    python scripts/rehearse.py --days 30 --end 2026-11-13 --method persistence

判读要点：**捕获率的中位数才是有意义的指标**。均值会被少数完美预见收益
接近零的平淡日放大，看起来很难看但并不反映真实水平（见 arb_sanity.py）。
"""
from __future__ import annotations

import argparse
import statistics
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.core.params import DEFAULT_STORAGE  # noqa: E402
from src.forecast.baseline import PriceForecaster  # noqa: E402
from src.optimize.storage_lp import prepare_submission  # noqa: E402
from src.pipeline.data import load_matrix  # noqa: E402
from src.pipeline.runner import DEFAULT_CSV  # noqa: E402
from src.submit.writer import check_submission, write_submission  # noqa: E402
from src.validate.feasibility import apply_deadband, replay_soc, revenue  # noqa: E402


def pearson(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.std() < 1e-9 or b.std() < 1e-9:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="提交链路彩排")
    ap.add_argument("--days", type=int, default=14, help="回放最近 N 天")
    ap.add_argument("--end", default=None, help="截止日（含），缺省为数据最后一天")
    ap.add_argument("--method", default="blend")
    ap.add_argument("--csv", default=str(DEFAULT_CSV))
    ap.add_argument("--team", default="彩排队")
    ap.add_argument("--keep", default=None, help="把生成的文件留到此目录（缺省用临时目录）")
    args = ap.parse_args(argv)

    m = load_matrix(Path(args.csv))
    if len(m) < 30:
        print("历史不足 30 天，无法彩排", file=sys.stderr)
        return 2

    end = pd.Timestamp(args.end) if args.end else m.index[-1]
    days = [d for d in m.index if d <= end][-args.days:]
    if not days:
        print("窗口内无数据", file=sys.stderr)
        return 2

    outdir = Path(args.keep) if args.keep else Path(tempfile.mkdtemp())
    forecaster = PriceForecaster(method=args.method)

    soc = DEFAULT_STORAGE.soc_init_mwh
    rows = []
    bad_files = 0

    print(f"彩排 {len(days)} 天（{days[0].date()} ~ {days[-1].date()}）｜预测 {args.method}")
    print(f"{'日期':<12}{'相关':>7}{'实收$':>10}{'完美$':>10}{'捕获率':>9}{'末SOC':>8}  说明")

    for d in days:
        hist = m[m.index < d]
        if len(hist) < 8:
            continue

        fc = forecaster.predict(hist)
        actual = m.loc[d].to_numpy(dtype=float)

        day = prepare_submission(fc, soc, DEFAULT_STORAGE, date=str(d.date()))
        q, rep = apply_deadband(day.q_mw, soc, DEFAULT_STORAGE)
        if not rep.feasible:
            q, rep = prepare_submission(fc, soc, DEFAULT_STORAGE, date=str(d.date())).q_mw, rep

        # 真实生成一次文件，确认字节层面也没问题
        f = outdir / f"{d.date()}.csv"
        write_submission([(str(d.date()), q)], f, team=args.team, verify=False)
        chk = check_submission(f, soc_start_mwh=soc)
        if not chk.ok:
            bad_files += 1
            print(f"  ! {d.date()} 文件自检未通过：{chk.summary()}")

        got = revenue(q, actual)
        perfect = revenue(prepare_submission(actual, soc, DEFAULT_STORAGE).q_mw, actual)
        capture = got / perfect if perfect > 0.5 else float("nan")

        soc = replay_soc(q, soc)[-1]
        soc = min(max(soc, DEFAULT_STORAGE.soc_min_mwh), DEFAULT_STORAGE.soc_max_mwh)

        rows.append({"date": d.date(), "corr": pearson(fc, actual), "got": got,
                     "perfect": perfect, "capture": capture})
        flag = "" if perfect > 0.5 else "（完美收益过小，捕获率不具意义）"
        print(f"{str(d.date()):<12}{pearson(fc, actual):>7.3f}{got:>10.2f}{perfect:>10.2f}"
              f"{capture * 100:>8.1f}%{soc:>8.3f}  {flag}")

    if not rows:
        print("无可评估日", file=sys.stderr)
        return 2

    df = pd.DataFrame(rows)
    valid = df[df["perfect"] > 0.5]
    print()
    print(f"合计   实收 ${df['got'].sum():.2f} / 完美 ${df['perfect'].sum():.2f}"
          f" → 总体捕获 {df['got'].sum() / df['perfect'].sum() * 100:.1f}%")
    print(f"日均   实收 ${df['got'].mean():.2f} / 完美 ${df['perfect'].mean():.2f}")
    if len(valid):
        print(f"捕获率 中位 {valid['capture'].median() * 100:.1f}%"
              f" / 均值 {valid['capture'].mean() * 100:.1f}%"
              f" / 最低 {valid['capture'].min() * 100:.1f}%"
              f"（{len(valid)} 天）")
    print(f"预测相关 中位 {df['corr'].median():.3f} / 均值 {df['corr'].mean():.3f}")
    print(f"格式自检 {len(df) - bad_files}/{len(df)} 通过")
    if bad_files:
        print("!! 存在格式不合格文件，比赛前必须修掉", file=sys.stderr)
    print(f"\n生成文件目录：{outdir}")
    return 0 if bad_files == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
