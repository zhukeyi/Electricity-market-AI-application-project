#!/usr/bin/env python3
"""Oracle 端自包含抓取脚本（仅用标准库，无需 pandas/numpy）。

在 Oracle ARM 服务器上直接运行，规避本地对 CAISO 的网络限制：

    python3 oasis_pull.py --start 2024-01-01 --end 2026-11-07 --out /tmp/sp15.csv

设计要点：
- 仅依赖 urllib/zipfile/xml.etree/csv，Oracle 上零安装；
- 30 天分块（OASIS 上限 31 天，且按太平洋日历日判定，30 天安全）；
- 每次请求间隔 1.2s，规避 OASIS 限流；
- 失败重试 5 次，指数退避；
- 故障以 XML 形式装在 200-OK 的 ZIP 内返回，需解析后判定，不能只看 HTTP 状态。
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import io
import sys
import time
import urllib.error
import urllib.request
import zipfile
import xml.etree.ElementTree as ET

BASE = "https://oasis.caiso.com/oasisapi/SingleZip"
UA = "Mozilla/5.0 (compatible; csg-ai-market/0.1)"


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def fetch(url: str, retries: int = 5, timeout: int = 60) -> bytes:
    last = None
    for i in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                data = r.read()
            if not data or data[:2] != b"PK":
                raise RuntimeError(f"非 zip 响应（{len(data)} 字节）: {data[:200]!r}")
            return data
        except Exception as exc:  # noqa: BLE001
            last = exc
            wait = 3.0 * (2 ** (i - 1))
            print(f"  重试 {i}/{retries}: {exc} → {wait:.0f}s", file=sys.stderr, flush=True)
            if i < retries:
                time.sleep(wait)
    raise RuntimeError(f"失败: {last}")


def parse_zip(data: bytes, node: str):
    out = []
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        names = [n for n in z.namelist() if n.lower().endswith(".xml")]
        if not names:
            raise RuntimeError(f"zip 内无 xml: {z.namelist()}")
        root = ET.fromstring(z.read(names[0]))

    for el in root.iter():
        if _local(el.tag).upper() == "ERROR":
            raise RuntimeError(f"OASIS 报错: {(el.text or '').strip()[:300]}")

    for item in root.iter():
        if _local(item.tag) != "REPORT_ITEM":
            continue
        for d in item:
            if _local(d.tag) != "REPORT_DATA":
                continue
            rec = {_local(f.tag): (f.text or "").strip() for f in d}
            if rec.get("DATA_ITEM") != "LMP_PRC":
                continue
            if node and rec.get("RESOURCE_NAME") != node:
                continue
            try:
                val = float(rec.get("VALUE", ""))
            except ValueError:
                continue
            out.append((
                rec.get("OPR_DATE", ""),
                int(rec.get("INTERVAL_NUM", 0)),
                val,
                rec.get("INTERVAL_START_GMT", ""),
                rec.get("RESOURCE_NAME", ""),
            ))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2024-01-01")
    ap.add_argument("--end", default="2026-11-07")
    ap.add_argument("--node", default="TH_SP15_GEN-APND")
    ap.add_argument("--out", default="/tmp/sp15_dam.csv")
    ap.add_argument("--chunk", type=int, default=30)
    ap.add_argument("--sleep", type=float, default=1.2)
    args = ap.parse_args()

    start = dt.date.fromisoformat(args.start)
    end = dt.date.fromisoformat(args.end)
    rows = []
    cur = start
    n_chunk = 0

    while cur <= end:
        ce = min(cur + dt.timedelta(days=args.chunk - 1), end)
        s = dt.datetime.combine(cur, dt.time(0, 0)) - dt.timedelta(hours=8)
        e = dt.datetime.combine(ce, dt.time(0, 0)) + dt.timedelta(hours=16)
        url = (
            f"{BASE}?queryname=PRC_LMP"
            f"&startdatetime={s:%Y%m%dT%H:%M}-0000"
            f"&enddatetime={e:%Y%m%dT%H:%M}-0000"
            f"&version=1&market_run_id=DAM&node={args.node}"
        )
        n_chunk += 1
        print(f"[{n_chunk}] {cur} ~ {ce}", file=sys.stderr, flush=True)
        got = parse_zip(fetch(url), args.node)
        rows.extend(got)
        print(f"    → {len(got)} 行", file=sys.stderr, flush=True)
        cur = ce + dt.timedelta(days=1)
        time.sleep(args.sleep)

    # 去重
    seen = {}
    for r in rows:
        seen[(r[0], r[1])] = r
    rows = sorted(seen.values())

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["date", "hour", "lmp", "utc_start", "node"])
        w.writerows(rows)

    days = len({r[0] for r in rows})
    print(f"完成: {len(rows)} 行 / {days} 个运行日 → {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
