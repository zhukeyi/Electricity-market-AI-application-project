"""提交时序计算 —— 整条流水线最容易出错的地方，单独成模块并强制无副作用。

本模块只做一件事：回答「现在应该为哪个加州运行日生成申报」，以及「离截止还有多久」。
所有函数均为纯函数或只读时钟，便于用固定时刻做回归测试。

三条时间轴同时存在，混淆任何两条都会导致提交错日：

====================  ==========================  ==============================
时间轴                 时区                        用途
====================  ==========================  ==============================
北京 BJ                Asia/Shanghai (UTC+8)       主办方截止时间（手册 P25）
加州 LA                America/Los_Angeles         运行日归属、CAISO 价格时段
UTC                    UTC                         服务器 cron、OASIS 查询参数
====================  ==========================  ==============================

核心约束：运行日 D 的申报须在 **BJ D 日 02:00** 前提交，该时刻恰好等于
**LA D-1 日 10:00** —— CAISO 日前市场的报价截止时刻。所以：

1. 「要提交哪个运行日」必须按 **LA 日历**推算，不能按 BJ 日历。BJ D 日 23:00
   时 LA 尚在 D-1 日下午，若按 BJ 日历取「今天 +1」会在跨过 BJ 零点后错位一天。
2. 出清结果要到 LA D-1 日 13:00 才发布（CAISO Tariff 6.5.2.2），晚于报价截止三小时。
   因此 D 日价格**只能预测，不能读取** —— 这不是设计选择，是硬约束。
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from zoneinfo import ZoneInfo

LA = ZoneInfo("America/Los_Angeles")
BJ = ZoneInfo("Asia/Shanghai")
UTC = dt.timezone.utc


def utc_now() -> dt.datetime:
    return dt.datetime.now(UTC)


def to_la(moment: dt.datetime) -> dt.datetime:
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=UTC)
    return moment.astimezone(LA)


def to_bj(moment: dt.datetime) -> dt.datetime:
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=UTC)
    return moment.astimezone(BJ)


def la_today(now: dt.datetime | None = None) -> dt.date:
    return to_la(now or utc_now()).date()


def trade_date_for(now: dt.datetime | None = None, lead_days: int = 1) -> dt.date:
    """返回「LA 今天 + lead_days」这一天，**不做可提交性判断**。

    这是纯粹的日历换算。在 BJ 23:00 触发时它正好给出第二天要提交的运行日，
    但在 BJ 白天手工触发时会给出一个截止已过的日子 —— 要拿可提交的日期，
    请用 `next_submittable()`。
    """
    return la_today(now) + dt.timedelta(days=lead_days)


def next_submittable(now: dt.datetime | None = None, max_roll: int = 2) -> tuple[dt.date, int]:
    """返回仍可提交的最近运行日，以及相对「LA 明天」前滚的天数。

    为什么需要前滚：BJ 上午（= LA 前一日傍晚）手工触发时，「LA 明天」的截止
    （BJ D 02:00 = LA D-1 10:00）已经过去十几小时。此时硬按「LA 明天」算，
    preflight 会判定逾时而整条链路空转 —— 正确行为是顺延到下一个还没关闸的日子。

    返回 `roll=0` 表示无需前滚（定时任务的正常情形）。
    """
    n = now or utc_now()
    base = la_today(n) + dt.timedelta(days=1)
    for roll in range(max_roll + 1):
        cand = base + dt.timedelta(days=roll)
        if bj_deadline(cand) > (n if n.tzinfo else n.replace(tzinfo=UTC)):
            return cand, roll
    return base + dt.timedelta(days=max_roll), max_roll


def bj_deadline(trade_date: dt.date) -> dt.datetime:
    """运行日 D 的申报截止：BJ D 日 02:00。"""
    return dt.datetime.combine(trade_date, dt.time(2, 0), tzinfo=BJ)


def la_deadline(trade_date: dt.date) -> dt.datetime:
    """同一时刻的加州表示：LA D-1 日 10:00（CAISO DAM 报价截止）。"""
    return bj_deadline(trade_date).astimezone(LA)


def la_day_hours(day: dt.date) -> int:
    """LA 当日实际小时数：夏令时切换日为 23 或 25，平日 24。

    CAISO 在切换日的 `INTERVAL_NUM` 会给出 23 或 25 个时段。手册固定 24 时段，
    因此这类日子必须显式识别后处理，**不得静默按 24 硬对齐**（会错位一小时）。
    """
    start = dt.datetime.combine(day, dt.time(0, 0), tzinfo=LA)
    end = dt.datetime.combine(day + dt.timedelta(days=1), dt.time(0, 0), tzinfo=LA)
    return int(round((end.astimezone(UTC) - start.astimezone(UTC)).total_seconds() / 3600))


@dataclass(frozen=True)
class Schedule:
    """一次提交任务的时序快照。"""

    trade_date: dt.date
    now_utc: dt.datetime
    now_bj: dt.datetime
    now_la: dt.datetime
    deadline_bj: dt.datetime
    deadline_la: dt.datetime
    la_hours: int
    rolled: int = 0
    """相对「LA 明天」前滚的天数。0 = 定时任务的正常情形。"""

    @property
    def minutes_left(self) -> float:
        """距截止的剩余分钟数；为负表示已过截止时刻。"""
        return (self.deadline_bj - self.now_bj).total_seconds() / 60.0

    @property
    def is_late(self) -> bool:
        return self.minutes_left < 0

    @property
    def la_next_hours(self) -> int:
        """运行日本身的小时数（D 日），用于判断是否需要 23/25 时段处理。"""
        return la_day_hours(self.trade_date)

    def summary(self) -> str:
        flag = "已逾时" if self.is_late else "可提交"
        left = self.minutes_left
        left_s = f"{left / 60:.1f} 小时" if left >= 60 else f"{left:.0f} 分钟"
        roll = f"｜已前滚 {self.rolled} 天" if self.rolled else ""
        return (
            f"运行日 {self.trade_date}（LA）｜{flag}，剩余 {left_s}{roll}｜"
            f"截止 BJ {self.deadline_bj:%Y-%m-%d %H:%M} = LA {self.deadline_la:%Y-%m-%d %H:%M}"
        )


def schedule(trade_date: dt.date | None = None, now: dt.datetime | None = None) -> Schedule:
    """构造时序快照。不指定 `trade_date` 时自动取「仍可提交的最近运行日」。"""
    n = now or utc_now()
    rolled = 0
    if trade_date is None:
        d, rolled = next_submittable(n)
    else:
        d = trade_date
    return Schedule(
        trade_date=d,
        now_utc=n if n.tzinfo else n.replace(tzinfo=UTC),
        now_bj=to_bj(n),
        now_la=to_la(n),
        deadline_bj=bj_deadline(d),
        deadline_la=la_deadline(d),
        la_hours=la_day_hours(d),
        rolled=rolled,
    )
