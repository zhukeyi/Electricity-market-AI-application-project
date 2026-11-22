"""每日提交流水线的编排层。

一条命令完成：时序判定 → 增量抓数 → 预测 → 优化 → 死区清理 → 写盘 → 自检 → 记账 → 通知。
设计目标是**无人值守可跑 14 天**，因此有三个硬性要求：

1. **幂等**：同一运行日重复执行结果一致。已生成的文件不覆盖（除非 `--force`），
   但一定重新自检 —— 文件在、不代表文件对。
2. **可诊断**：每一步都有耗时与结论，失败点直接定位到步骤名；
   全量记录落在 `data/out/runs/<日期>.json` 与 `log.jsonl`。
3. **降级不静默**：抓数失败时若数据落后不超过 `max_stale_days` 天（默认 3），
   标注告警后继续；落后更多则**直接失败**，绝不拿过期数据硬算一条曲线交上去。

失败等级：
- `ok=False` 且 `fatal=True`：当天无法提交，需人工介入；
- `ok=False` 且 `fatal=False`：已产出文件但存在告警（如模板待核实项）。
"""
from __future__ import annotations

import datetime as dt
import fcntl
import json
import logging
import socket
import time
import traceback
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd

from ..caiso.client import OasisClient
from ..core.params import DEFAULT_MARKET, DEFAULT_STORAGE
from ..forecast.baseline import PriceForecaster
from ..optimize.storage_lp import SubmissionDay, prepare_submission
from ..submit.writer import TEMPLATE_NOTES, check_submission, read_submission, write_submission
from ..validate.feasibility import apply_deadband, replay_soc, revenue
from . import notify
from .clock import Schedule, schedule as make_schedule, trade_date_for
from .data import health_report, last_date, load_long, load_matrix, merge_long, write_long

log = logging.getLogger("pipeline")

DEFAULT_CSV = Path("data/raw/sp15_dam_lmp.csv")
DEFAULT_OUTDIR = Path("data/out/submission")
DEFAULT_RUNDIR = Path("data/out/runs")


# --------------------------------------------------------------------------- 配置


@dataclass
class Config:
    trade_date: dt.date | None = None
    team: str = "xxxx队"
    members: tuple[str, ...] = ()
    method: str = "blend"

    csv_path: Path = DEFAULT_CSV
    outdir: Path = DEFAULT_OUTDIR
    rundir: Path = DEFAULT_RUNDIR

    do_fetch: bool = True
    transport: str = "auto"  # auto | direct | ssh
    history_start: dt.date = dt.date(2024, 1, 1)
    max_stale_days: int = 3
    """抓数失败时允许的最大数据陈旧天数。

    为什么不设成 0：比赛期要无人值守跑 14 天，OASIS 或 SSH 偶发抽风不该让当天申报作废。
    缺 1~3 天时 blend 预测（昨日 + 7 天中位）仍有意义，劣化可接受；
    超过 3 天说明链路已经坏掉，此时硬算一条曲线反而是更糟的选择，故直接失败。
    """

    dry_run: bool = False
    force: bool = False
    allow_late: bool = False
    soc_start: float | None = None
    use_actual: bool = False
    webhook: str | None = None

    node: str = DEFAULT_MARKET.hub_node

    def submission_path(self, d: dt.date) -> Path:
        name = f"（{self.team}_线上组_交易决策）AI+电力市场应用大赛结果提交.csv"
        return Path(self.outdir) / str(d) / name


# --------------------------------------------------------------------------- 结果


@dataclass
class Step:
    name: str
    ok: bool
    seconds: float = 0.0
    skipped: bool = False
    detail: str = ""
    warn: bool = False

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "ok": self.ok,
            "skipped": self.skipped,
            "warn": self.warn,
            "seconds": round(self.seconds, 3),
            "detail": self.detail,
        }

    def line(self) -> str:
        if self.skipped:
            mark = "–"
        elif not self.ok:
            mark = "✗"
        elif self.warn:
            mark = "!"
        else:
            mark = "✓"
        return f"  {mark} {self.name:<9} {self.seconds:6.2f}s  {self.detail}"


@dataclass
class RunReport:
    trade_date: str
    schedule_text: str
    steps: list[Step] = field(default_factory=list)
    submission_path: str | None = None
    expected_revenue: float | None = None
    fatal: bool = False
    warnings: list[str] = field(default_factory=list)
    context: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        """是否产出了可提交的成果。`warn` 步骤算通过 —— 告警已在 warnings 里单独记录。"""
        return not self.fatal and all(s.ok or s.warn for s in self.steps)

    @property
    def clean(self) -> bool:
        """是否全程无告警。"""
        return self.ok and not self.warnings

    def add(self, step: Step) -> Step:
        self.steps.append(step)
        if step.warn and step.detail:
            self.warnings.append(f"{step.name}: {step.detail}")
        return step

    def summary(self) -> str:
        head = "完成" if self.clean else ("完成（有告警）" if self.ok else "失败")
        lines = [f"[{head}] 运行日 {self.trade_date}", f"  {self.schedule_text}"]
        lines += [s.line() for s in self.steps]
        if self.expected_revenue is not None:
            lines.append(f"  预期收益 ${self.expected_revenue:.2f}")
        if self.submission_path:
            lines.append(f"  文件 {self.submission_path}")
        if self.warnings:
            lines.append("  告警：" + "；".join(self.warnings))
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "trade_date": self.trade_date,
            "ok": self.ok,
            "fatal": self.fatal,
            "schedule": self.schedule_text,
            "expected_revenue": self.expected_revenue,
            "submission_path": self.submission_path,
            "steps": [s.to_dict() for s in self.steps],
            "warnings": self.warnings,
            **self.context,
        }


class PipelineError(RuntimeError):
    pass


# --------------------------------------------------------------------------- 工具


@contextmanager
def _locked(path: Path) -> Iterator[None]:
    """非阻塞文件锁，防止 cron 与手工执行撞车。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    fh = open(path, "w")
    try:
        try:
            fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise PipelineError(f"另一个实例正在运行（锁 {path}）") from exc
        yield
    finally:
        try:
            fcntl.flock(fh, fcntl.LOCK_UN)
        finally:
            fh.close()


def _probe_direct(host: str = "oasis.caiso.com", port: int = 443, timeout: float = 6.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def resolve_transport(prefer: str, probe=_probe_direct) -> tuple[str, str]:
    """auto 时按网络可达性选择：能直连则直连，否则走 SSH 隧道到 Oracle。"""
    if prefer in ("direct", "ssh"):
        return prefer, f"指定 {prefer}"
    if probe():
        return "direct", "探测到 oasis.caiso.com 可直连"
    return "ssh", "本机不可直连 CAISO，改用 Oracle 中转"


def history_submissions(
    outdir: Path, trade_date: dt.date, team: str | None = None, max_lookback: int = 20
) -> list[Path]:
    """取出 `trade_date` 之前的**全部**历史提交文件，按运行日升序、每日仅一份。

    同一日期目录可能残留多份文件（改队名、重跑 `--force` 都会留下），
    若不收敛到一份，SOC 链会被重复重放。选取规则：优先队名匹配，其次 mtime 最新。
    """
    out: list[Path] = []
    for i in range(1, max_lookback + 1):
        d = trade_date - dt.timedelta(days=i)
        hits = list(Path(outdir).glob(f"{d}/*.csv"))
        if not hits:
            continue
        picked = hits[0]
        if len(hits) > 1:
            same = [h for h in hits if team and team in h.name]
            pool = same or hits
            picked = max(pool, key=lambda h: h.stat().st_mtime)
            log.warning("%s 目录有 %d 份提交文件，取 %s", d, len(hits), picked.name)
        out.append(picked)
    return sorted(out, key=lambda h: h.parent.name)


def chain_soc(prev_files, soc_init: float) -> float:
    """从**首日**起按运行日顺序链式重放全部历史提交文件，得到今日初始 SOC。

    ⚠ 必须链式重放整个历史，不能只重放前一日文件：每日提交的是**单日**文件，
    其曲线是相对于"前一日真实末态"求解的。若把前一日曲线直接叠加到首日 SOC0 上，
    等于把历史净充放全部丢弃，初始 SOC 逐日漂移 —— 实测 2026-11-19 因此被算成
    0.698294 MWh（真实值 0.710176），跌破 5% 下限导致当日直接 abort。

    历史文件可能是 V2 旧格式（4 列），重放只需日期与功率，故允许 legacy 读取；
    这与提交前自检的严格模式是两回事，不要混用。
    """
    if isinstance(prev_files, (str, Path)):
        prev_files = [prev_files]
    soc = soc_init
    for f in prev_files:
        _, data, _ = read_submission(f, allow_legacy=True)
        for _, q in data:
            soc = replay_soc(q, soc)[-1]
    return soc


def clamp_soc(soc: float, params=DEFAULT_STORAGE, tol: float = 0.02) -> tuple[float, str]:
    """把重放出的初始 SOC 收敛到合法区间，吸收两位小数申报带来的累积舍入。

    申报值只保留两位小数，逐日重放的末态与平台内部真值可能有毫瓦时级差异。
    容差内直接钳制并说明；超出容差则说明上游数据/历史文件有问题，原样返回，
    交由下游 `prepare_submission` 报错（宁可当日报警，不可静默生成错误曲线）。
    """
    lo, hi = params.soc_min_mwh, params.soc_max_mwh
    if lo <= soc <= hi:
        return soc, ""
    if lo - tol <= soc <= hi + tol:
        fixed = min(max(soc, lo), hi)
        return fixed, f"初始 SOC {soc:.6f} 越界但在 {tol} MWh 容差内，已钳制为 {fixed:.6f}"
    return soc, f"初始 SOC {soc:.6f} 超出 [{lo}, {hi}] 且超过容差 {tol}，需人工核查历史提交"


def find_previous_submission(outdir: Path, trade_date: dt.date, max_lookback: int = 5) -> Path | None:
    """向前找最近的提交文件。跳日时需要回溯，不能只看 D-1。"""
    for i in range(1, max_lookback + 1):
        d = trade_date - dt.timedelta(days=i)
        hits = sorted(Path(outdir).glob(f"{d}/*.csv"))
        if hits:
            return hits[0]
    return None


# --------------------------------------------------------------------------- 主体


class Pipeline:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.rep: RunReport | None = None
        self.t0 = time.time()

    # -- 步骤计时装饰 ------------------------------------------------------
    @contextmanager
    def _step(self, name: str) -> Iterator[Step]:
        st = Step(name=name, ok=False)
        t = time.time()
        try:
            yield st
        except Exception as exc:  # noqa: BLE001
            st.ok = False
            st.detail = f"{type(exc).__name__}: {exc}"
            st.seconds = time.time() - t
            log.error("%s 失败: %s", name, st.detail)
            log.debug("%s", traceback.format_exc())
            self.rep.add(st)  # type: ignore[union-attr]
            raise
        else:
            st.seconds = time.time() - t
            self.rep.add(st)  # type: ignore[union-attr]

    # -- 主流程 ------------------------------------------------------------
    def run(self) -> RunReport:
        cfg = self.cfg
        sch = make_schedule(cfg.trade_date)
        self.rep = RunReport(trade_date=str(sch.trade_date), schedule_text=sch.summary())
        log.info("运行日 %s｜%s", sch.trade_date, sch.summary())

        Path(cfg.rundir).mkdir(parents=True, exist_ok=True)
        with _locked(Path(cfg.rundir) / ".lock"):
            try:
                self._preflight(sch)
                self._fetch(sch)
                day, prices, soc0 = self._decide(sch)
                self._write(day)
                self._verify(sch, soc0)
            except PipelineError as exc:
                self.rep.fatal = True
                if not any(s.name == "preflight" and not s.ok for s in self.rep.steps):
                    self.rep.add(Step("abort", False, detail=str(exc)))
                log.error("流水线中止: %s", exc)
            except Exception as exc:  # noqa: BLE001
                self.rep.fatal = True
                self.rep.add(Step("abort", False, detail=f"{type(exc).__name__}: {exc}"))
                log.error("流水线异常: %s", exc, exc_info=True)

            self._record(sch)
            self._notify()

        self.rep.context["elapsed_s"] = round(time.time() - self.t0, 2)
        return self.rep

    # -- 1 前置检查 --------------------------------------------------------
    def _preflight(self, sch: Schedule) -> None:
        cfg = self.cfg
        with self._step("preflight") as st:
            notes: list[str] = []

            if sch.is_late and not cfg.allow_late:
                st.ok = False
                st.detail = (
                    f"已过截止 {abs(sch.minutes_left):.0f} 分钟（截止 BJ {sch.deadline_bj:%m-%d %H:%M}）"
                )
                raise PipelineError(st.detail)

            if sch.la_hours != 24:
                st.ok = False
                st.detail = (
                    f"运行日 {sch.trade_date} 在 LA 有 {sch.la_hours} 小时（DST 切换），"
                    "与 24 时段模型不符，需人工处理"
                )
                raise PipelineError(st.detail)

            if sch.rolled:
                notes.append(f"自动前滚 {sch.rolled} 天（原运行日已过截止）")
                st.warn = True
            else:
                natural = trade_date_for(sch.now_utc)
                if sch.trade_date != natural:
                    # daily.sh 会把运行日算好再传进来，此处用来兜住"人为指定了别的日子"
                    diff = (sch.trade_date - natural).days
                    notes.append(f"运行日非默认（默认 {natural}，本次 {sch.trade_date}，差 {diff:+d} 天）")
                    st.warn = True

            lo, hi = DEFAULT_MARKET.contest_start, DEFAULT_MARKET.contest_end
            if not (lo <= str(sch.trade_date) <= hi):
                notes.append(f"运行日在赛期 {lo}~{hi} 之外")

            if cfg.team.strip() in ("", "xxxx队"):
                # 队名会直接进入文件名，交错了平台可能直接拒收，必须点名提醒
                notes.append("队名仍为占位符，提交前务必改成真实队名")
                st.warn = True

            existing = cfg.submission_path(sch.trade_date)
            if existing.exists() and not cfg.force and not cfg.dry_run:
                notes.append("文件已存在，跳过重写")

            st.ok = True
            st.detail = "；".join(notes) if notes else f"时序正常，{sch.la_hours}h"

    # -- 2 增量抓数 --------------------------------------------------------
    def _fetch(self, sch: Schedule) -> None:
        cfg = self.cfg
        need_end = sch.trade_date - dt.timedelta(days=1)

        if not cfg.do_fetch:
            with self._step("fetch") as st:
                st.ok, st.skipped = True, True
                st.detail = "已按参数跳过"
            return

        with self._step("fetch") as st:
            df = load_long(cfg.csv_path)
            have = last_date(df)

            if have is not None and have >= need_end:
                st.ok, st.skipped = True, True
                st.detail = f"本地已覆盖至 {have}，无需抓取（需 {need_end}）"
                return

            start = (have - dt.timedelta(days=1)) if have else cfg.history_start
            if start > need_end:
                start = need_end
            transport, why = resolve_transport(cfg.transport)
            log.info("抓取 %s ~ %s（%s：%s）", start, need_end, transport, why)

            try:
                client = OasisClient(transport=transport)
                fresh = client.fetch_dam_lmp(start, need_end, node=cfg.node)
            except Exception as exc:  # noqa: BLE001
                st.ok = False
                st.warn = True
                stale = (need_end - have).days if have else 9999
                st.detail = f"抓取失败（{transport}，数据落后 {stale} 天）：{exc}"
                if stale > cfg.max_stale_days:
                    raise PipelineError(
                        f"抓取失败且本地数据仅到 {have}（落后 {stale} 天 > 容忍 {cfg.max_stale_days} 天），"
                        f"不足以为 {sch.trade_date} 预测"
                    ) from exc
                log.warning("抓取失败但数据仅落后 %s 天，降级继续：%s", stale, exc)
                return

            merged = merge_long(df, fresh)
            write_long(merged, cfg.csv_path)
            added = len(merged) - len(df)
            now_last = last_date(merged)
            st.ok = True
            st.detail = f"{transport} +{added} 行 → 至 {now_last}"
            if now_last < need_end:
                st.warn = True
                st.detail += f"（仍缺至 {need_end}，CAISO 可能尚未发布）"

    # -- 3 预测 + 优化 -----------------------------------------------------
    def _decide(self, sch: Schedule) -> tuple[SubmissionDay, np.ndarray, float]:
        cfg = self.cfg
        with self._step("forecast") as st:
            m = load_matrix(cfg.csv_path)
            if len(m) == 0:
                st.ok = False
                st.detail = "无可用历史矩阵"
                raise PipelineError(st.detail)

            target = pd.Timestamp(sch.trade_date)
            hist = m[m.index < target]
            if len(hist) == 0:
                st.ok = False
                st.detail = f"{sch.trade_date} 之前无历史数据"
                raise PipelineError(st.detail)

            if cfg.use_actual:
                if target not in m.index:
                    st.ok = False
                    st.detail = f"--use-actual 要求 {sch.trade_date} 已在历史中"
                    raise PipelineError(st.detail)
                prices = m.loc[target].to_numpy(dtype=float)
                src = "实际价格（彩排）"
            else:
                prices = PriceForecaster(method=cfg.method).predict(hist)
                src = f"预测·{cfg.method}"

            st.ok = True
            st.detail = (
                f"{src}｜{len(hist)} 天历史｜"
                f"价 {prices.min():.1f}~{prices.max():.1f} 均值 {prices.mean():.1f}"
            )

        with self._step("optimize") as st:
            if cfg.soc_start is not None:
                soc0 = float(cfg.soc_start)
                how = "--soc-start 指定"
            elif str(sch.trade_date) == DEFAULT_MARKET.contest_start:
                # 赛期首日的 SOC0 由手册规定为 5%，**与赛前任何预演产物无关**。
                # 若不强制，11-22 晚跑首日时会把预演目录里的历史净充放链进来。
                soc0 = DEFAULT_STORAGE.soc_init_mwh
                how = f"赛期首日 {DEFAULT_MARKET.contest_start}，按手册 SOC0=5%，忽略赛前预演历史"
            else:
                hist = history_submissions(Path(cfg.outdir), sch.trade_date, team=cfg.team)
                if str(sch.trade_date) > DEFAULT_MARKET.contest_start:
                    # 赛期内的 SOC 链只由赛期文件构成；赛前预演日之间则照常链式，
                    # 否则多日彩排无法验证跨日耦合。
                    hist = [f for f in hist if f.parent.name >= DEFAULT_MARKET.contest_start]
                if hist:
                    soc0 = chain_soc(hist, DEFAULT_STORAGE.soc_init_mwh)
                    how = f"由 {hist[0].parent.name}~{hist[-1].parent.name} 共 {len(hist)} 日链式推出"
                    soc0, clamp_note = clamp_soc(soc0)
                    if clamp_note:
                        st.warn = True
                        how += f"｜{clamp_note}"
                else:
                    soc0 = DEFAULT_STORAGE.soc_init_mwh
                    how = "首日，取手册 SOC0=5%"

            day = prepare_submission(prices, soc0, DEFAULT_STORAGE, date=str(sch.trade_date))
            q_clean, rep_clean = apply_deadband(day.q_mw, soc0, DEFAULT_STORAGE)
            trimmed = 0
            if q_clean != day.q_mw:
                trimmed = sum(1 for a, b in zip(q_clean, day.q_mw) if a != b)
                day = SubmissionDay(
                    date=day.date,
                    q_mw=q_clean,
                    prices=day.prices,
                    expected_revenue=revenue(q_clean, day.prices),
                    report=rep_clean,
                    soc_start_mwh=rep_clean.soc_start_mwh,
                    note=f"{day.note}; deadband",
                )

            st.ok = True
            st.detail = (
                f"SOC {soc0:.3f}→{day.report.soc_end_mwh:.3f}（{how}）｜"
                f"充 {day.report.charge_mwh:.2f}/放 {day.report.discharge_mwh:.2f}｜"
                f"${day.expected_revenue:.2f}"
                + (f"｜死区归零 {trimmed} 段" if trimmed else "")
            )
            if not day.report.feasible:
                st.ok = False
                st.detail = "；".join(day.report.violations[:3])
                raise PipelineError(f"优化结果不可行：{st.detail}")

            n_act = sum(1 for q in day.q_mw if abs(q) > 1e-9)
            st.detail += f"｜{n_act}/24 段有动作"

        return day, prices, soc0

    # -- 4 写盘 ------------------------------------------------------------
    def _write(self, day: SubmissionDay) -> None:
        cfg = self.cfg
        out = cfg.submission_path(dt.date.fromisoformat(day.date))

        with self._step("write") as st:
            if cfg.dry_run:
                st.ok, st.skipped = True, True
                st.detail = "[dry-run] 未写文件"
            elif out.exists() and not cfg.force:
                # 已存在则复用。注意：**不能在这里 return**，否则下面的路径登记不会执行，
                # 报告里就会丢掉文件名与收益，下游（通知/记账）拿到空值。
                st.ok, st.skipped = True, True
                st.detail = f"已存在，复用 {out.name}"
            else:
                write_submission([day], out, team=cfg.team, members=cfg.members, verify=False)
                st.ok = True
                st.detail = f"{out}（{out.stat().st_size} 字节）"

        if not cfg.dry_run and out.exists():
            self.rep.submission_path = str(out)
            self.rep.expected_revenue = day.expected_revenue

    # -- 5 自检 ------------------------------------------------------------
    def _verify(self, sch: Schedule, soc0: float) -> None:
        cfg = self.cfg
        out = cfg.submission_path(sch.trade_date)

        with self._step("verify") as st:
            if cfg.dry_run:
                st.ok, st.skipped = True, True
                st.detail = "[dry-run] 无文件可校验"
                return
            if not out.exists():
                st.ok = False
                st.detail = f"找不到 {out}"
                raise PipelineError(st.detail)

            rep = check_submission(out, soc_start_mwh=soc0)
            st.ok = rep.ok
            st.detail = f"{rep.rows} 行 / {rep.days} 天｜{'通过' if rep.ok else rep.summary()}"
            if not rep.ok:
                raise PipelineError("提交文件自检未通过")
            if TEMPLATE_NOTES:
                st.warn = True
                st.detail += f"｜模板待核实 {len(TEMPLATE_NOTES)} 项"

    # -- 6 记账 ------------------------------------------------------------
    def _record(self, sch: Schedule) -> None:
        cfg = self.cfg
        rundir = Path(cfg.rundir)
        rundir.mkdir(parents=True, exist_ok=True)
        try:
            df = load_long(cfg.csv_path)
            self.rep.context["data"] = health_report(cfg.csv_path)
            self.rep.context["data_last"] = str(last_date(df)) if len(df) else None
        except Exception as exc:  # noqa: BLE001
            self.rep.context["data"] = {"error": str(exc)}

        payload = self.rep.to_dict()
        payload["finished_at"] = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")

        (rundir / f"{sch.trade_date}.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        with open(rundir / "log.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    # -- 7 通知 ------------------------------------------------------------
    def _notify(self) -> None:
        if not self.cfg.webhook:
            return
        ok, why = notify.send(self.rep.summary(), self.cfg.webhook)
        self.rep.context["notify"] = {"ok": ok, "detail": why}
        if not ok:
            log.warning("通知未送达：%s", why)


def run(cfg: Config) -> RunReport:
    return Pipeline(cfg).run()
