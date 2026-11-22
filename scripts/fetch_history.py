#!/usr/bin/env python3
"""抓取 CAISO SP15 日前 LMP 历史数据并缓存为 CSV。

用法（本地运行，走 SSH 代理到 Oracle 出口）：
    PYTHONPATH=. python scripts/fetch_history.py --start 2023-01-01 --end 2026-11-07

输出：
    data/raw/sp15_dam_lmp.csv      长表（date, hour, lmp, utc_start, node）
    data/raw/sp15_dam_matrix.csv   宽表（date × 24），已剔除 DST 异常日
"""
from __future__ import annotations

import argparse
import datetime as dt
import logging
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.caiso.client import OasisClient  # noqa: E402
from src.caiso.parser import to_price_matrix  # noqa: E402
from src.core.params import DEFAULT_MARKET  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("fetch")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2023-01-01")
    ap.add_argument("--end", default="2026-11-07")
    ap.add_argument("--chunk-days", type=int, default=30)
    ap.add_argument("--transport", default="ssh", choices=["ssh", "direct"])
    ap.add_argument("--node", default=DEFAULT_MARKET.hub_node)
    args = ap.parse_args()

    start = dt.date.fromisoformat(args.start)
    end = dt.date.fromisoformat(args.end)

    client = OasisClient(transport=args.transport)
    df = client.fetch_dam_lmp(start, end, node=args.node, chunk_days=args.chunk_days)

    raw_dir = ROOT / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    raw_path = raw_dir / "sp15_dam_lmp.csv"
    df.to_csv(raw_path, index=False)
    log.info("长表已写入 %s（%d 行，%d 个运行日）", raw_path, len(df), df["date"].nunique())

    matrix, dropped = to_price_matrix(df, drop_incomplete=True)
    mpath = raw_dir / "sp15_dam_matrix.csv"
    matrix.to_csv(mpath)
    log.info("宽表已写入 %s（%d 天 × %d 时段）", mpath, len(matrix), matrix.shape[1])
    if dropped:
        log.warning("剔除 %d 个时段数≠24 的运行日：%s", len(dropped), dropped[:10])

    log.info("日期范围 %s ~ %s", matrix.index.min(), matrix.index.max())
    log.info("价格统计: mean=%.2f median=%.2f min=%.2f max=%.2f",
             matrix.stack().mean(), matrix.stack().median(),
             matrix.stack().min(), matrix.stack().max())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
