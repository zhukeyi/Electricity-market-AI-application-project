"""OASIS XML 解析器。

OASIS 的 `SingleZip` 返回 `<OPR_DATE>_<OPR_DATE>_PRC_LMP_DAM_<ts>_v1.xml`，
结构为 `OASISReport/MessagePayload/RTO/REPORT_ITEM/REPORT_DATA/*`，字段：

    DATA_ITEM            LMP_PRC
    RESOURCE_NAME        TH_SP15_GEN-APND
    OPR_DATE             运行日（本地 PPT 日期，YYYY-MM-DD）
    INTERVAL_NUM         当日时段序号，1..24（夏令时切换日为 23 或 25）
    INTERVAL_START_GMT   时段起点 UTC
    INTERVAL_END_GMT     时段终点 UTC
    VALUE                价格，$/MWh

比赛中运行日固定为 24 时段，故 DST 切换日（23/25 时段）需显式标记，
由上层决定剔除或重采样，**不得静默按 24 硬对齐**。
"""
from __future__ import annotations

import io
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import dataclass

import pandas as pd

from .client import FetchError


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


@dataclass
class ParseResult:
    df: pd.DataFrame
    market_run_id: str = ""
    node: str = ""
    warnings: list[str] = None

    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []


def parse_prc_lmp_xml(xml_bytes: bytes, node: str | None = None) -> ParseResult:
    """解析 PRC_LMP XML，返回长表 DataFrame。"""
    warnings: list[str] = []
    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError as exc:
        head = xml_bytes[:500].decode("utf-8", "replace")
        raise FetchError(f"XML 解析失败: {exc}; 前 500 字节: {head}") from exc

    # OASIS 在参数错误时返回 <ERROR> 而非报表
    for el in root.iter():
        if _local(el.tag).upper() == "ERROR":
            raise FetchError(f"OASIS 返回错误: {(el.text or '').strip()[:300]}")

    rows: list[dict] = []
    mkt = ""
    for hdr in root.iter():
        if _local(hdr.tag) == "REPORT_HEADER":
            for ch in hdr:
                if _local(ch.tag) == "MKT_TYPE":
                    mkt = (ch.text or "").strip()

    for item in root.iter():
        if _local(item.tag) != "REPORT_ITEM":
            continue
        for data in item:
            if _local(data.tag) != "REPORT_DATA":
                continue
            rec: dict = {}
            for f in data:
                rec[_local(f.tag)] = (f.text or "").strip()
            if rec.get("DATA_ITEM") != "LMP_PRC":
                continue
            try:
                val = float(rec.get("VALUE", "nan"))
            except ValueError:
                warnings.append(f"无法解析价格: {rec.get('VALUE')}")
                continue
            rows.append({
                "date": rec.get("OPR_DATE"),
                "hour": int(rec.get("INTERVAL_NUM", 0)),
                "lmp": val,
                "utc_start": rec.get("INTERVAL_START_GMT"),
                "node": rec.get("RESOURCE_NAME", node or ""),
            })

    df = pd.DataFrame(rows)
    if df.empty:
        warnings.append("解析结果为空表")
        return ParseResult(df, mkt, node or "", warnings)

    if node:
        before = len(df)
        df = df[df["node"] == node]
        if len(df) < before:
            warnings.append(f"按节点 {node} 过滤：{before} → {len(df)} 行")

    # DST 切换日时段数异常检测
    counts = df.groupby("date")["hour"].count()
    odd = counts[counts != 24]
    if len(odd):
        preview = ", ".join(f"{d}({c})" for d, c in odd.head(6).items())
        warnings.append(f"{len(odd)} 个运行日时段数≠24（DST 切换日）: {preview}")

    return ParseResult(df.reset_index(drop=True), mkt, node or "", warnings)


def parse_prc_lmp_zip(data: bytes, node: str | None = None) -> pd.DataFrame:
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        names = [n for n in z.namelist() if n.lower().endswith(".xml")]
        if not names:
            raise FetchError(f"zip 内无 XML：{z.namelist()}")
        res = parse_prc_lmp_xml(z.read(names[0]), node=node)
    if not res.df.empty:
        res.df.attrs["market_run_id"] = res.market_run_id
        res.df.attrs["warnings"] = res.warnings
    return res.df


def to_price_matrix(df: pd.DataFrame, drop_incomplete: bool = True):
    """把长表转成 (日期 × 24) 价格矩阵。

    Args:
        drop_incomplete: True 时剔除时段数 ≠24 的日期（DST 日），避免污染训练集。

    Returns:
        (matrix: pd.DataFrame[index=date, columns=1..24], dropped: list[str])
    """
    if df.empty:
        return pd.DataFrame(columns=range(1, 25)), []

    pivot = df.pivot_table(index="date", columns="hour", values="lmp", aggfunc="first")
    full = [h for h in range(1, 25)]
    bad = [d for d, row in pivot.iterrows()
           if set(full) - set(pivot.columns[row.notna()])]
    dropped = sorted({str(d) for d in pivot.index if d in set(bad)})

    if drop_incomplete:
        pivot = pivot.drop(index=[d for d in pivot.index if d in set(bad)], errors="ignore")

    pivot = pivot.reindex(columns=full)
    pivot.index = pivot.index.astype(str)
    pivot.index.name = "date"
    pivot.columns = [int(c) for c in pivot.columns]
    return pivot.sort_index(), dropped
