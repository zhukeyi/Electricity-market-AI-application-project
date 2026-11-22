"""提交文件读写 —— 按官方 **V3.0（2026-11-19 发布版）** 模板生成与复核。

模板来源：`（2026_11_23_xxxx队_线上组_交易决策）AI+电力市场应用大赛结果提交.csv.xlsx`

V3.0 相对旧版（4 列）的**破坏性变更**：数据列由 4 列扩为 **6 列**，
新增 `OPR_HR`（1-based 时段号）与两个 GMT 时刻列；旧版的 `operate_date` /
`skd`（`H:00:00`）列名与取值方式全部作废。标题行的团队成员分隔符由 `、` 改为 `\\`。

| 列 | 内容 | 规格 |
|---|---|---|
| A 序号 | 1..24 | 整数，逐行连续 |
| B OPR_DT | 加州运行日 | `yyyy/m/d`，**月日不补零**（Excel 数字格式 `yyyy/m/d`） |
| C OPR_HR | 时段号 | **1..24**，非 0-based |
| D INTERVALSTARTTIME_GMT | 时段 UTC 起点 | `YYYY-MM-DDTHH:MM:SS-00:00`（偏移字面量恒为 `-00:00`） |
| E INTERVALENDTIME_GMT | 时段 UTC 终点 | 同上，等于下一行起点 |
| F storage_power_mw | 充放电功率 | 正放负充，两位小数，±7 MW |

字节规格沿用旧模板实测结论（CRLF 行尾、表头第二列前 BOM、说明行内部裸 LF），
因新版仅提供 xlsx，**导出 CSV 后的真实字节尚未经平台验证**，见 `TEMPLATE_NOTES`。

⚠ 官方示例曲线本身不可行：示例在 HR17–19 填 `+10 MW`，既超过 Pmax=7 MW，
又使 SOC 跌至 −157%（下限 5%）。照抄示例 = 当日全天收益归零。
本模块的校验器必须能拒绝该示例，相关回归用例见 `tests/test_submit.py`。
"""
from __future__ import annotations

import csv
import io
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, Sequence
from zoneinfo import ZoneInfo

from ..core.params import DEFAULT_STORAGE, StorageParams
from ..validate.feasibility import validate_dispatch

LA = ZoneInfo("America/Los_Angeles")
UTC = timezone.utc

# --- 模板字节常量（照抄，勿美化） ---
BOM = "﻿"
POWER_COL = "storage_power_mw"
COLUMNS = [
    "序号",
    "OPR_DT",
    "OPR_HR",
    "INTERVALSTARTTIME_GMT",
    "INTERVALENDTIME_GMT",
    POWER_COL,
]
HEADER_LINE = f"序号,{BOM}OPR_DT,OPR_HR,INTERVALSTARTTIME_GMT,INTERVALENDTIME_GMT,{POWER_COL}"

FOOTER_BODY = (
    "OPR_DT、OPR_HR：加州冬令时运行日期、时段；\n"
    "INTERVALSTARTTIME_GMT、INTERVALENDTIME_GMT:格林尼治标准时间GMT开始时刻、"
    "格林尼治标准时间GMT结束时刻\n"
    f"{POWER_COL}：储能自调度申报曲线，单位：MW，代表各时段储能充放电决策"
    "（正数放电、负数充电、0 为不动作）；保留小数点后两位。\n"
)
FOOTER_LINE = f'"{FOOTER_BODY}",,,,,'

TITLE_ORG = "线上组"
TITLE_SUFFIX = "——交易决策结果"

TEMPLATE_NOTES = (
    "V3.0 仅提供 xlsx，导出 CSV 后的 BOM/CRLF 字节规格未经平台回执验证，首日须人工核对；",
    "官方示例曲线 HR17-19 为 +10MW，超 Pmax=7MW 且 SOC 穿底，**严禁照抄**；",
    "GMT 偏移按模板字面量 `-00:00` 输出（非 `+00:00`/`Z`）；",
    "赛期为冬令时，UTC = 加州本地 +8h；若遇 DST 切换日须按 zoneinfo 实算，不得硬编码 +8。",
)

_DATE_RE = re.compile(r"^(\d{4})[-/](\d{1,2})[-/](\d{1,2})$")


@dataclass
class SubmitIssue:
    level: str          # "error" | "warn"
    where: str
    message: str

    def __str__(self) -> str:
        return f"[{self.level.upper()}] {self.where}: {self.message}"


@dataclass
class SubmitCheckReport:
    path: str
    rows: int = 0
    days: int = 0
    issues: list[SubmitIssue] = field(default_factory=list)
    per_day: dict[str, dict] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not any(i.level == "error" for i in self.issues)

    def summary(self) -> str:
        head = (
            f"{self.path}\n  {self.rows} 行 / {self.days} 天 → "
            f"{'通过' if self.ok else '不通过'}"
        )
        if not self.issues:
            return head + "\n  无问题"
        return head + "\n" + "\n".join("  " + str(i) for i in self.issues)


def _iso_date(date: str) -> str:
    m = _DATE_RE.match(date.strip())
    if not m:
        raise ValueError(f"无法识别的日期格式: {date!r}（应为 YYYY-MM-DD 或 YYYY/M/D）")
    y, mo, d = m.groups()
    return f"{int(y):04d}-{int(mo):02d}-{int(d):02d}"


def _fmt_date(date: str) -> str:
    """任意日期 → 模板 `yyyy/m/d`（**月日不补零**，与 Excel 数字格式一致）。"""
    y, m, d = _iso_date(date).split("-")
    return f"{y}/{int(m)}/{int(d)}"


def _fmt_value(x: float, ndigits: int = 2) -> str:
    v = round(float(x), ndigits)
    if abs(v) < 0.5 * 10 ** (-ndigits):
        v = 0.0
    return f"{v:.{ndigits}f}"


def _fmt_gmt(ts: datetime) -> str:
    """UTC 时刻 → 模板字面量 `YYYY-MM-DDTHH:MM:SS-00:00`。"""
    return ts.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%S") + "-00:00"


def interval_gmt(date: str, hour: int) -> tuple[str, str]:
    """加州运行日 + 1-based 时段号 → (GMT 起点, GMT 终点)。

    用 zoneinfo 实算 UTC 偏移，DST 切换日（23/25 时段）同样成立。
    """
    if not 1 <= hour <= 24:
        raise ValueError(f"OPR_HR 越界（应 1..24）: {hour}")
    day = datetime.strptime(_iso_date(date), "%Y-%m-%d")
    start = datetime(day.year, day.month, day.day, tzinfo=LA) + timedelta(hours=hour - 1)
    return _fmt_gmt(start), _fmt_gmt(start + timedelta(hours=1))


def _title(team: str, members: Sequence[str]) -> str:
    who = "\\".join(members) if members else "xxx\\xxx"
    return f"（{TITLE_ORG}）{team} (团队成员{who}){TITLE_SUFFIX}"


def write_submission(
    days: Iterable,
    out_path: str | Path,
    *,
    team: str = "xxxx队",
    members: Sequence[str] = (),
    ndigits: int = 2,
    params: StorageParams = DEFAULT_STORAGE,
    verify: bool = True,
) -> Path:
    """把若干 `SubmissionDay`（或 `(date, q_mw)` 二元组）写成 V3.0 提交 CSV。

    Args:
        days: 元素需有 `.date` 与 `.q_mw`，或为 `(date, q_mw)` 元组；`q_mw` 长度 24。
        out_path: 建议 `（{队名}_线上组_交易决策）AI+电力市场应用大赛结果提交.csv`。
        verify: 写入后立即回读复核（**不要关闭**）。
    """
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = [_title(team, members) + ",,,,,"]
    lines.append(HEADER_LINE)

    seq = 0
    for item in days:
        date, q = (item[0], item[1]) if isinstance(item, tuple) else (item.date, item.q_mw)
        q = [float(x) for x in q]
        if len(q) != params.periods_per_day:
            raise ValueError(f"{date}: 时段数 {len(q)} ≠ {params.periods_per_day}")
        for h, val in enumerate(q, start=1):
            seq += 1
            g0, g1 = interval_gmt(date, h)
            lines.append(
                f"{seq},{_fmt_date(date)},{h},{g0},{g1},{_fmt_value(val, ndigits)}"
            )

    lines.append(FOOTER_LINE)

    # 手工拼 CRLF：footer 内部含裸 LF，不能被整体替换
    text = "\r\n".join(lines[:-1]) + "\r\n" + lines[-1] + "\r\n"
    out.write_text(text, encoding="utf-8", newline="")

    if verify:
        rep = check_submission(out, params=params)
        if not rep.ok:
            raise ValueError(f"生成的文件未通过自检:\n{rep.summary()}")
    return out


def read_submission(path: str | Path, params: StorageParams = DEFAULT_STORAGE, *, allow_legacy: bool = False):
    """读回提交文件 → `(title, [(iso_date, q_mw)], footer)`。

    同时做 GMT 列与 OPR_DT/OPR_HR 的**交叉一致性**校验（防时区写错）。

    `allow_legacy=True` 时额外接受 V2 的 4 列旧格式（`operate_date`/`skd`），
    **仅供内部 SOC 链重放历史文件使用**。提交前自检必须保持默认的严格模式，
    否则旧格式文件会被误当合格产物交上去。
    """
    raw = Path(path).read_text(encoding="utf-8")
    rows = [r for r in csv.reader(io.StringIO(raw, newline="")) if r and any(c.strip() for c in r)]
    if len(rows) < 3:
        raise ValueError(f"文件行数不足: {len(rows)}")

    title = rows[0][0]
    header = [c.lstrip(BOM).strip() for c in rows[1]]
    legacy = header[:4] == ["序号", "operate_date", "skd", POWER_COL]
    if header[:6] != COLUMNS and not (allow_legacy and legacy):
        raise ValueError(f"表头不符（V3.0 应为 6 列 {COLUMNS}）: {header}")

    data: list[tuple[str, list[float]]] = []
    cur_date: str | None = None
    cur_q: list[float] = []
    footer = ""

    for r in rows[2:]:
        if not r[0].strip().isdigit():
            footer = r[0]
            continue
        date = _iso_date(r[1])
        if legacy:
            # 旧格式：A 序号 / B operate_date / C skd(`H:00:00`, 0-based) / D 功率
            if len(r) < 4:
                raise ValueError(f"旧格式数据行不足 4 列: {r}")
            hour = int(r[2].split(":")[0]) + 1
            value = r[3]
        else:
            if len(r) < 6:
                raise ValueError(f"数据行不足 6 列: {r}")
            hour = int(r[2])
            exp0, exp1 = interval_gmt(date, hour)
            if r[3].strip() != exp0 or r[4].strip() != exp1:
                raise ValueError(
                    f"{date} HR{hour} GMT 列与运行日不一致: "
                    f"文件=({r[3]}, {r[4]}) 应为=({exp0}, {exp1})"
                )
            value = r[5]
        if cur_date is not None and date != cur_date:
            data.append((cur_date, cur_q))
            cur_q = []
        if cur_date is not None and hour != len(cur_q) + 1:
            raise ValueError(f"{date} OPR_HR 不连续: 第 {len(cur_q)+1} 行实为 HR{hour}")
        cur_date = date
        cur_q.append(float(value or 0.0))

    if cur_date is not None:
        data.append((cur_date, cur_q))

    return title, data, footer


def check_submission(
    path: str | Path,
    *,
    params: StorageParams = DEFAULT_STORAGE,
    soc_start_mwh: float | None = DEFAULT_STORAGE.soc_init_mwh,
) -> SubmitCheckReport:
    """结构与物理双重校验。

    结构项：表头六列、序号连续、OPR_HR 连续、GMT 列与运行日一致、精度、值域、行尾。
    物理项：逐日重放 SOC（跨日链式），确认不触发"全天归零"。
    """
    p = Path(path)
    rep = SubmitCheckReport(path=str(p))
    if not p.exists():
        rep.issues.append(SubmitIssue("error", "file", "文件不存在"))
        return rep

    raw_bytes = p.read_bytes()
    if raw_bytes[:3] == b"\xef\xbb\xbf":
        rep.issues.append(
            SubmitIssue("warn", "编码", "文件头出现 BOM，模板中 BOM 位于第二列前而非文件头")
        )
    if b"\r\n" not in raw_bytes:
        rep.issues.append(SubmitIssue("error", "行尾", "未检测到 CRLF，模板为 CRLF"))

    try:
        title, data, footer = read_submission(p, params)
    except Exception as exc:  # noqa: BLE001
        rep.issues.append(SubmitIssue("error", "解析", str(exc)))
        return rep

    if not footer:
        rep.issues.append(SubmitIssue("error", "尾部", "缺少说明行"))
    if TITLE_ORG not in title:
        rep.issues.append(SubmitIssue("warn", "标题", f"标题未含「{TITLE_ORG}」: {title[:40]}"))

    first_two = raw_bytes.split(b"\r\n")[:2]
    if len(first_two) == 2 and b"\xef\xbb\xbf" not in first_two[1]:
        rep.issues.append(SubmitIssue("warn", "表头", "第二列前未见 BOM，与旧模板不一致"))

    rep.days = len(data)
    rep.rows = sum(len(q) for _, q in data)
    if not data:
        rep.issues.append(SubmitIssue("error", "数据", "无数据行"))
        return rep

    seqs = []
    for line in raw_bytes.decode("utf-8", "replace").split("\r\n")[2:]:
        c0 = line.split(",")[0].strip()
        if c0.isdigit():
            seqs.append(int(c0))
    if seqs != list(range(1, len(seqs) + 1)):
        rep.issues.append(SubmitIssue("error", "序号", f"序号不连续（共 {len(seqs)} 行）"))

    soc = params.soc_init_mwh if soc_start_mwh is None else float(soc_start_mwh)
    for date, q in data:
        if len(q) != params.periods_per_day:
            rep.issues.append(
                SubmitIssue("error", date, f"时段数 {len(q)} ≠ {params.periods_per_day}")
            )
            continue
        for h, v in enumerate(q, start=1):
            if abs(v) > params.power_mw + 1e-9:
                rep.issues.append(
                    SubmitIssue("error", f"{date} HR{h}", f"功率 {v:.4f} MW 越界 ±{params.power_mw}")
                )
            txt = f"{v:.6f}".rstrip("0")
            if "." in txt and len(txt.split(".")[1]) > 2 and abs(round(v, 2) - v) > 1e-9:
                rep.issues.append(SubmitIssue("warn", f"{date} HR{h}", f"小数位超过 2 位: {v}"))

        frep = validate_dispatch(q, soc, params)
        rep.per_day[date] = {
            "feasible": frep.feasible,
            "soc_min": frep.min_soc_mwh,
            "soc_max": frep.max_soc_mwh,
            "soc_end": frep.soc_end_mwh,
        }
        if not frep.feasible:
            for v in frep.violations[:3]:
                rep.issues.append(SubmitIssue("error", date, f"物理不可行（当日归零）: {v}"))
        soc = frep.soc_end_mwh

    return rep


def describe_template(path: str | Path) -> dict:
    """读取模板并输出结构画像，用于版本间回归比对（不依赖人工记忆）。"""
    raw = Path(path).read_bytes()
    lines = raw.decode("utf-8").split("\r\n")
    data_lines = [ln for ln in lines if ln[:1].isdigit()]
    return {
        "bytes": len(raw),
        "crlf": b"\r\n" in raw,
        "header_bom_pos": raw.find(b"\xef\xbb\xbf"),
        "title": lines[0] if lines else "",
        "header": lines[1] if len(lines) > 1 else "",
        "data_rows": len(data_lines),
        "hr_first": data_lines[0].split(",")[2] if data_lines else "",
        "hr_last": data_lines[-1].split(",")[2] if data_lines else "",
        "gmt_first": data_lines[0].split(",")[3] if data_lines else "",
        "gmt_last": data_lines[-1].split(",")[4] if data_lines else "",
    }
