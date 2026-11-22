"""流水线测试 —— 时序、幂等、降级、以及「不许悄悄出错」。

重点不是覆盖率，而是把四个**会直接导致交白卷**的场景钉死：

1. 运行日算错（跨 BJ 零点后偏移一天）
2. 已过截止时间仍照常提交
3. 抓数失败却拿残缺数据硬算
4. 已生成的文件被重复运行覆盖

另外覆盖 DST 切换日的 23/25 小时识别 —— 本赛期（2026-11-23~12-06）恰在
夏令时结束后，理论上都是 24h，但代码不能依赖这个巧合。
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd

from src.pipeline import clock, envfile, runner
from src.pipeline.data import health_report, last_date, load_long, load_matrix, merge_long, write_long
from src.core.params import DEFAULT_STORAGE
from src.pipeline.runner import (
    Config,
    chain_soc,
    clamp_soc,
    find_previous_submission,
    history_submissions,
    resolve_transport,
    run,
)

# 流水线会把失败原因写 ERROR 日志；测试里预期失败，无需污染输出
logging.getLogger("pipeline").addHandler(logging.NullHandler())
logging.getLogger("pipeline").propagate = False


def _bj(y, m, d, hh, mm=0) -> dt.datetime:
    return dt.datetime(y, m, d, hh, mm, tzinfo=clock.BJ)


def _utc(y, m, d, hh, mm=0) -> dt.datetime:
    return dt.datetime(y, m, d, hh, mm, tzinfo=dt.timezone.utc)


# --------------------------------------------------------------------------- 时序


class TestClock(unittest.TestCase):
    def test_trade_date_is_next_la_day_not_next_bj_day(self):
        """BJ 23:00 触发时 LA 是同日下午 → 运行日应为 LA 明天。"""
        now = _bj(2026, 11, 14, 23)  # = UTC 15:00 = LA 11-14 07:00 PST
        self.assertEqual(clock.trade_date_for(now), dt.date(2026, 11, 15))

    def test_trade_date_stable_across_bj_midnight(self):
        """BJ 01:00 仍是同一个运行日 —— 这里最容易写成「BJ 今天+1」而错位。"""
        self.assertEqual(clock.trade_date_for(_bj(2026, 11, 14, 23)), dt.date(2026, 11, 15))
        self.assertEqual(clock.trade_date_for(_bj(2026, 11, 15, 1, 30)), dt.date(2026, 11, 15))

    def test_deadline_equals_caiso_bid_close(self):
        """BJ D 02:00 必须等于 LA D-1 10:00，否则对 CAISO 报价截止的理解就是错的。"""
        sch = clock.schedule(dt.date(2026, 11, 23), now=_bj(2026, 11, 22, 23))
        self.assertEqual(sch.deadline_bj, _bj(2026, 11, 23, 2))
        self.assertEqual(
            (sch.deadline_la.strftime("%Y-%m-%d %H:%M"), sch.deadline_la.hour),
            ("2026-11-22 10:00", 10),
        )

    def test_minutes_left_and_lateness(self):
        d = dt.date(2026, 11, 23)
        self.assertAlmostEqual(clock.schedule(d, now=_bj(2026, 11, 22, 23)).minutes_left, 180.0)
        self.assertFalse(clock.schedule(d, now=_bj(2026, 11, 22, 23)).is_late)
        self.assertTrue(clock.schedule(d, now=_bj(2026, 11, 23, 3)).is_late)


class TestRollForward(unittest.TestCase):
    """「LA 明天」在 BJ 白天手工触发时已过截止，必须顺延到今天还没关闸的日子。

    这条规则是实测暴露出来的：在 BJ 上午激活定时器后手工触发一次，
    「LA 明天」的截止（BJ D 02:00）早已过去十几个小时，整条链路会空转。
    """

    def test_no_roll_at_scheduled_hour(self):
        d, roll = clock.next_submittable(_bj(2026, 11, 14, 23))
        self.assertEqual((d, roll), (dt.date(2026, 11, 15), 0))

    def test_rolls_one_day_in_bj_morning(self):
        # BJ 11-14 10:50 = LA 11-13 18:50 → LA 明天(11-14) 的截止已过
        d, roll = clock.next_submittable(_bj(2026, 11, 14, 10, 50))
        self.assertEqual((d, roll), (dt.date(2026, 11, 15), 1))

    def test_rolled_date_is_always_submittable(self):
        for now in (_bj(2026, 11, 14, h) for h in range(0, 24, 3)):
            d, _ = clock.next_submittable(now)
            sch = clock.schedule(d, now=now)
            self.assertFalse(sch.is_late, f"{now} → {d} 竟已逾时")

    def test_schedule_reports_roll(self):
        sch = clock.schedule(now=_bj(2026, 11, 14, 10, 50))
        self.assertEqual(sch.rolled, 1)
        self.assertIn("已前滚 1 天", sch.summary())

    def test_dst_day_hours(self):
        """2026 美国夏令时：3-08 起（23h），11-01 止（25h）。"""
        self.assertEqual(clock.la_day_hours(dt.date(2026, 11, 1)), 25)
        self.assertEqual(clock.la_day_hours(dt.date(2026, 3, 8)), 23)
        # 赛期全程应为 24h —— 若这条失败说明赛期内撞上了切换日，必须人工处理
        for day in (dt.date(2026, 11, 23), dt.date(2026, 12, 6)):
            self.assertEqual(clock.la_day_hours(day), 24)


class TestTransport(unittest.TestCase):
    def test_auto_falls_back_to_ssh(self):
        self.assertEqual(resolve_transport("auto", probe=lambda: False)[0], "ssh")

    def test_auto_uses_direct_when_reachable(self):
        self.assertEqual(resolve_transport("auto", probe=lambda: True)[0], "direct")

    def test_explicit_wins(self):
        self.assertEqual(resolve_transport("direct", probe=lambda: False), ("direct", "指定 direct"))


# --------------------------------------------------------------------------- 数据


def _long(day: dt.date, prices: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": [day] * len(prices),
            "hour": list(range(1, len(prices) + 1)),
            "lmp": prices,
            "utc_start": [""] * len(prices),
            "node": ["TH_SP15_GEN-APND"] * len(prices),
        }
    )


class TestData(unittest.TestCase):
    def test_merge_dedupes_keeping_latest(self):
        a = _long(dt.date(2026, 11, 1), [10.0] * 24)
        b = _long(dt.date(2026, 11, 1), [99.0] * 24)
        m = merge_long(a, b)
        self.assertEqual(len(m), 24)
        self.assertTrue((m["lmp"] == 99.0).all())

    def test_matrix_requires_full_day(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "x.csv"
            write_long(pd.concat([_long(dt.date(2026, 11, 1), [10.0] * 24),
                                  _long(dt.date(2026, 11, 2), [10.0] * 23)]), p)
            m = load_matrix(p)
            self.assertEqual(len(m), 1, "只有 23 段的那天不应进入宽表")

    def test_health_flags_short_day(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "x.csv"
            write_long(_long(dt.date(2026, 11, 2), [10.0] * 23), p)
            rep = health_report(p)
            self.assertTrue(any("23 个时段" in x for x in rep["problems"]), rep["problems"])


# --------------------------------------------------------------------------- 流水线


class _Harness:
    """在临时目录里铺一份 40 天历史，供流水线端到端跑。"""

    def __init__(self, days: int = 40, end: dt.date = dt.date(2026, 11, 20)):
        self.td = tempfile.mkdtemp()
        self.root = Path(self.td)
        self.csv = self.root / "data" / "sp15.csv"
        rows = []
        for i in range(days):
            d = end - dt.timedelta(days=days - 1 - i)
            # 造一条有鸭形特征的曲线：午间低、晚峰高
            px = [30 + 20 * ((h - 12) ** 2) / 144 for h in range(24)]
            rows.append(_long(d, px))
        write_long(pd.concat(rows, ignore_index=True), self.csv)

    def cfg(self, **kw) -> Config:
        base = dict(
            csv_path=self.csv,
            outdir=self.root / "out",
            rundir=self.root / "runs",
            team="测试队",
            members=("张三",),
            do_fetch=False,
            dry_run=False,
        )
        base.update(kw)
        return Config(**base)


class TestPipelineRun(unittest.TestCase):
    def test_happy_path(self):
        h = _Harness()
        rep = run(h.cfg(trade_date=dt.date(2026, 11, 23)))
        self.assertTrue(rep.ok, rep.summary())
        self.assertIsNotNone(rep.submission_path)
        self.assertTrue(Path(rep.submission_path).exists())
        self.assertGreater(rep.expected_revenue or 0, 0)
        names = [s.name for s in rep.steps]
        self.assertEqual(names, ["preflight", "fetch", "forecast", "optimize", "write", "verify"])

    def test_second_run_is_idempotent(self):
        h = _Harness()
        first = run(h.cfg(trade_date=dt.date(2026, 11, 23)))
        second = run(h.cfg(trade_date=dt.date(2026, 11, 23)))
        self.assertTrue(second.ok, second.summary())
        write_step = next(s for s in second.steps if s.name == "write")
        self.assertTrue(write_step.skipped, "文件已存在时不应重写")
        self.assertEqual(first.expected_revenue, second.expected_revenue)

    def test_force_rewrites(self):
        h = _Harness()
        run(h.cfg(trade_date=dt.date(2026, 11, 23)))
        rep = run(h.cfg(trade_date=dt.date(2026, 11, 23), force=True))
        write_step = next(s for s in rep.steps if s.name == "write")
        self.assertFalse(write_step.skipped)

    def test_late_submission_is_refused(self):
        h = _Harness()
        rep = run(h.cfg(trade_date=dt.date(2026, 11, 1)))  # 早已过期
        self.assertTrue(rep.fatal)
        self.assertIsNone(rep.submission_path)
        self.assertTrue(any(s.name == "preflight" and not s.ok for s in rep.steps))

    def test_allow_late_override(self):
        h = _Harness()
        # 2026-10-25 已过截止但非 DST 日，用于验证 --allow-late 能放行
        rep = run(h.cfg(trade_date=dt.date(2026, 10, 25), allow_late=True))
        self.assertFalse(rep.fatal, rep.summary())

    def test_dst_day_is_refused_even_when_allowed_late(self):
        """2026-11-01 是 25 小时日，24 时段模型无法表达，必须拒绝而非硬凑。"""
        h = _Harness()
        rep = run(h.cfg(trade_date=dt.date(2026, 11, 1), allow_late=True))
        self.assertTrue(rep.fatal)
        self.assertTrue(any("25 小时" in s.detail for s in rep.steps), rep.summary())

    def test_fetch_skipped_when_disabled(self):
        h = _Harness()
        rep = run(h.cfg(trade_date=dt.date(2026, 11, 23)))
        fetch = next(s for s in rep.steps if s.name == "fetch")
        self.assertTrue(fetch.skipped)
        self.assertEqual(fetch.detail, "已按参数跳过")

    def test_fetch_failure_degrades_when_slightly_stale(self):
        """数据仅落后 2 天（容忍 3 天）→ 标黄继续，不阻断当场申报。"""
        h = _Harness()  # 数据到 2026-11-20，运行日 11-23 需 11-22
        with mock.patch.object(runner, "OasisClient", side_effect=RuntimeError("OASIS 502")):
            rep = run(h.cfg(trade_date=dt.date(2026, 11, 23), do_fetch=True))
        self.assertTrue(rep.ok, rep.summary())
        fetch = next(s for s in rep.steps if s.name == "fetch")
        self.assertTrue(fetch.warn)
        self.assertIn("落后 2 天", fetch.detail)
        self.assertTrue(Path(rep.submission_path).exists())

    def test_fetch_failure_is_fatal_when_badly_stale(self):
        """数据落后 19 天 → 必须失败，不能拿过期数据交卷。"""
        h = _Harness()
        with mock.patch.object(runner, "OasisClient", side_effect=RuntimeError("OASIS 502")):
            rep = run(h.cfg(trade_date=dt.date(2026, 12, 10), allow_late=True, do_fetch=True))
        self.assertTrue(rep.fatal)
        self.assertIsNone(rep.submission_path)

    def test_insufficient_history_is_fatal(self):
        h = _Harness()
        rep = run(h.cfg(trade_date=dt.date(2026, 8, 1), allow_late=True))
        self.assertTrue(rep.fatal)

    def test_dry_run_writes_nothing(self):
        h = _Harness()
        rep = run(h.cfg(trade_date=dt.date(2026, 11, 23), dry_run=True))
        self.assertTrue(rep.ok, rep.summary())
        self.assertIsNone(rep.submission_path)
        self.assertFalse((h.root / "out").exists() and any((h.root / "out").rglob("*.csv")))

    def test_run_record_written(self):
        h = _Harness()
        rep = run(h.cfg(trade_date=dt.date(2026, 11, 23)))
        j = h.root / "runs" / "2026-11-23.json"
        self.assertTrue(j.exists())
        payload = json.loads(j.read_text(encoding="utf-8"))
        self.assertEqual(payload["trade_date"], "2026-11-23")
        self.assertTrue(payload["ok"])
        self.assertIn("data", payload)


class TestEnvFile(unittest.TestCase):
    def test_parse_ignores_comments_and_quotes(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / ".env"
            p.write_text(
                "# 注释\n\nCSG_TEAM=某队\nCSG_MEMBERS=\"张三,李四\"\nBAD_LINE\nCSG_METHOD=median7\n",
                encoding="utf-8",
            )
            d = envfile.parse_env_file(p)
            self.assertEqual(d["CSG_TEAM"], "某队")
            self.assertEqual(d["CSG_MEMBERS"], "张三,李四")
            self.assertEqual(d["CSG_METHOD"], "median7")
            self.assertNotIn("BAD_LINE", d)

    def test_load_does_not_override_real_env(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / ".env"
            p.write_text("CSG_TEAM=文件里的队\n", encoding="utf-8")
            with mock.patch.dict("os.environ", {"CSG_TEAM": "命令行覆盖的队"}, clear=False):
                envfile.load(p)
                self.assertEqual(os.environ["CSG_TEAM"], "命令行覆盖的队")

    def test_missing_file_is_silent(self):
        self.assertEqual(envfile.parse_env_file("/nonexistent/.pipeline.env"), {})


class TestPlaceholderGuard(unittest.TestCase):
    def test_placeholder_team_is_flagged(self):
        """队名写错会让 14 天的文件全部作废，必须升级为告警而非普通提示。"""
        h = _Harness()
        rep = run(h.cfg(trade_date=dt.date(2026, 11, 23), team="xxxx队"))
        pre = next(s for s in rep.steps if s.name == "preflight")
        self.assertTrue(pre.warn)
        self.assertIn("占位符", pre.detail)


class TestSocChain(unittest.TestCase):
    def test_find_previous_skips_missing_days(self):
        h = _Harness()
        run(h.cfg(trade_date=dt.date(2026, 11, 20), allow_late=True))
        prev = find_previous_submission(h.root / "out", dt.date(2026, 11, 23))
        self.assertIsNotNone(prev)
        self.assertEqual(prev.parent.name, "2026-11-20")

    def test_chain_soc_replays_to_end_state(self):
        h = _Harness()
        rep = run(h.cfg(trade_date=dt.date(2026, 11, 20), allow_late=True))
        end = chain_soc(Path(rep.submission_path), 0.7)
        self.assertGreaterEqual(end, 0.7 - 1e-9)
        self.assertLessEqual(end, 13.3 + 1e-9)


class TestSocChainMultiDay(unittest.TestCase):
    """2026-11-19 Oracle 真实事故的回归：初始 SOC 必须链式重放全部历史。

    事故经过：每日提交的是**单日**文件，而旧 `chain_soc` 只重放前一日文件、
    却把首日 SOC0=0.7 当作起点，等于丢掉了此前所有日的净充放。
    11-17 曲线净变化 −0.001706 MWh，于是 11-19 初始 SOC 被算成 0.698294
    （真实 0.710176），跌破下限 → optimize 直接 abort。若发生在赛期即为当日白卷。
    """

    def test_chain_uses_full_history_not_just_previous_day(self):
        h = _Harness()
        run(h.cfg(trade_date=dt.date(2026, 11, 20), allow_late=True))
        run(h.cfg(trade_date=dt.date(2026, 11, 21), allow_late=True))
        out = h.root / "out"
        hist = history_submissions(out, dt.date(2026, 11, 22))
        self.assertEqual(len(hist), 2)

        # 真值：首日 0.7 → 逐日链式
        truth = chain_soc(hist, DEFAULT_STORAGE.soc_init_mwh)
        # 旧算法的错误值：仅重放前一日、起点仍取 0.7
        wrong = chain_soc(hist[-1], DEFAULT_STORAGE.soc_init_mwh)
        self.assertNotAlmostEqual(truth, wrong, places=6, msg="测试前提失效：两日净变化恰抵消")

        rep = run(h.cfg(trade_date=dt.date(2026, 11, 22), allow_late=True))
        self.assertTrue(rep.ok, rep.fatal)
        # 第三日实际采用的初始 SOC 必须等于真值
        opt = next(s for s in rep.steps if s.name == "optimize")
        self.assertIn(f"SOC {truth:.3f}", opt.detail, opt.detail)

    def test_third_day_does_not_abort(self):
        """连跑 4 天，任何一天都不得因初始 SOC 越界而 abort。"""
        h = _Harness()
        for i in range(4):
            rep = run(h.cfg(trade_date=dt.date(2026, 11, 20) + dt.timedelta(days=i), allow_late=True))
            self.assertTrue(rep.ok, f"第 {i+1} 天失败: {rep.fatal}")

    def test_same_day_duplicate_files_are_deduplicated(self):
        """改队名/重跑会留下同日的多份文件，SOC 链不得重复重放。"""
        h = _Harness()
        rep = run(h.cfg(trade_date=dt.date(2026, 11, 20), allow_late=True))
        src = Path(rep.submission_path)
        dup = src.with_name(src.name.replace("xxxx队", "别的队"))
        dup.write_bytes(src.read_bytes())
        hist = history_submissions(h.root / "out", dt.date(2026, 11, 21))
        self.assertEqual(len(hist), 1, "同一运行日只能取一份文件")

    def test_gap_day_means_soc_unchanged(self):
        """历史断档 = 当日未申报。平台对失败日按不充不放处理，SOC 不变，重放口径一致。"""
        h = _Harness()
        run(h.cfg(trade_date=dt.date(2026, 11, 20), allow_late=True))
        rep22 = run(h.cfg(trade_date=dt.date(2026, 11, 22), allow_late=True))  # 跳过 11-21
        self.assertTrue(rep22.ok, rep22.fatal)
        hist = history_submissions(h.root / "out", dt.date(2026, 11, 22))
        self.assertEqual(len(hist), 1)


class TestClampSoc(unittest.TestCase):
    def test_inside_range_untouched(self):
        self.assertEqual(clamp_soc(0.710176), (0.710176, ""))

    def test_marginally_below_floor_is_clamped(self):
        val, note = clamp_soc(0.698294)
        self.assertAlmostEqual(val, DEFAULT_STORAGE.soc_min_mwh)
        self.assertIn("钳制", note)

    def test_far_out_of_range_is_not_hidden(self):
        """超容差必须原样返回并提示人工核查，不能静默钳制掩盖上游错误。"""
        val, note = clamp_soc(0.5)
        self.assertEqual(val, 0.5)
        self.assertIn("人工核查", note)


class TestSeasonStartSoc(unittest.TestCase):
    """赛期首日 SOC0 必须是手册规定的 5%，与赛前预演产物无关。"""

    def test_first_contest_day_ignores_preseason_history(self):
        h = _Harness(end=dt.date(2026, 11, 22))
        run(h.cfg(trade_date=dt.date(2026, 11, 20), allow_late=True))  # 留下赛前预演文件
        rep = run(h.cfg(trade_date=dt.date(2026, 11, 23), allow_late=True))
        self.assertTrue(rep.ok, rep.fatal)
        opt = next(s for s in rep.steps if s.name == "optimize")
        self.assertIn("赛期首日", opt.detail, opt.detail)
        self.assertIn("SOC 0.700", opt.detail, opt.detail)

    def test_second_contest_day_chains_from_first(self):
        """赛期第二日必须链自首日的真实末态，而不是回到 5%。"""
        h = _Harness(end=dt.date(2026, 11, 23))
        first = run(h.cfg(trade_date=dt.date(2026, 11, 23), allow_late=True))
        end_soc = chain_soc(Path(first.submission_path), DEFAULT_STORAGE.soc_init_mwh)
        rep = run(h.cfg(trade_date=dt.date(2026, 11, 24), allow_late=True))
        self.assertTrue(rep.ok, rep.fatal)
        opt = next(s for s in rep.steps if s.name == "optimize")
        self.assertNotIn("赛期首日", opt.detail)
        self.assertIn(f"SOC {end_soc:.3f}", opt.detail, opt.detail)


if __name__ == "__main__":
    unittest.main()
