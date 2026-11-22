"""提交文件格式测试 —— 锁定 **V3.0（2026-11-19 发布版）** 模板的字节级规格。

这些测试的目的不是"跑通"，而是**防止有人好心美化格式后导致平台拒收**。
每一条断言都对应模板里的一处字节特征，改动前请先看 `src/submit/writer.py` 的模块 docstring。
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.core.params import DEFAULT_STORAGE
from src.submit.writer import (
    FOOTER_LINE,
    HEADER_LINE,
    check_submission,
    describe_template,
    interval_gmt,
    read_submission,
    write_submission,
)
from src.validate.feasibility import validate_dispatch

# 官方 V3.0 模板逐行内容（OPR_DT = 2026-11-18），用于逐字节比对。
OFFICIAL_SAMPLE = {
    12: -7.0, 13: -7.0, 14: -0.82,
    17: 10.0, 18: 10.0, 19: 10.0,
    22: -7.0, 23: -7.0,
}


def _curve(charge_hours=(11, 12), discharge_hours=(17,), tail_mw=3.0, power=7.0):
    """物理可行示例：充 2h(7MW) → 入库 11.9 MWh → 18 点放 7MW、19 点放 3MW。

    SOC：0.70 → 12.60 → 4.365 → 0.836，全程在 [0.70, 13.30] 内。
    索引为 0-based（`q[11]` 即 OPR_HR=12）。
    """
    q = [0.0] * 24
    for h in charge_hours:
        q[h] = -power
    for h in discharge_hours:
        q[h] = power
    if tail_mw:
        q[18] = tail_mw
    return q


def _official_q() -> list[float]:
    q = [0.0] * 24
    for hr, v in OFFICIAL_SAMPLE.items():
        q[hr - 1] = v
    return q


class TestTemplateBytes(unittest.TestCase):
    def _gen(self, date="2026-11-24", q=None, **kw) -> bytes:
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "s.csv"
            write_submission([(date, q or _curve())], out, verify=False, **kw)
            return out.read_bytes()

    def test_header_has_inline_bom_not_file_bom(self):
        """BOM 位于第二列前，文件头不得有 BOM —— 模板最容易被误解的一处。"""
        raw = self._gen()
        self.assertNotEqual(raw[:3], b"\xef\xbb\xbf", "文件头不应有 BOM")
        self.assertIn(b",\xef\xbb\xbfOPR_DT", raw, "表头第二列前应有 BOM")

    def test_header_is_six_columns(self):
        """V3.0 破坏性变更：4 列 → 6 列，旧列名 operate_date/skd 全部作废。"""
        raw = self._gen().decode("utf-8")
        self.assertIn("OPR_HR", raw)
        self.assertIn("INTERVALSTARTTIME_GMT", raw)
        self.assertIn("INTERVALENDTIME_GMT", raw)
        self.assertNotIn("operate_date", raw)
        self.assertNotIn(",skd,", raw)
        self.assertEqual(HEADER_LINE.count(","), 5)

    def test_crlf_line_endings(self):
        raw = self._gen()
        self.assertIn(b"\r\n", raw)
        # 裸 LF 只允许出现在尾部说明行内部（V3.0 说明行 3 行 → 3 处 LF）
        self.assertEqual(raw.count(b"\n") - raw.count(b"\r\n"), 3)

    def test_opr_hr_is_one_based(self):
        """OPR_HR 为 1..24，**不是** 0-based，也不是旧版的 `H:00:00`。"""
        text = self._gen().decode("utf-8")
        self.assertIn("\r\n1,2026/11/24,1,", text)
        self.assertIn("\r\n24,2026/11/24,24,", text)
        self.assertNotIn(":00:00,", text.split("INTERVALENDTIME_GMT")[1].split("\r\n")[1])

    def test_date_slash_format_no_zero_padding(self):
        """模板数字格式为 `yyyy/m/d`：月日**不补零**（2026/12/6 而非 2026/12/06）。"""
        text = self._gen(date="2026-12-06").decode("utf-8")
        self.assertIn(",2026/12/6,", text)
        self.assertNotIn(",2026/12/06,", text)

    def test_two_decimals(self):
        q = _curve()
        q[5] = -0.82
        text = self._gen(q=q).decode("utf-8")
        self.assertIn(",-0.82\r\n", text)
        self.assertIn(",0.00\r\n", text)
        self.assertNotIn(",-0.8\r\n", text)

    def test_footer_present_and_quoted(self):
        raw = self._gen()
        self.assertTrue(raw.rstrip(b"\r\n").endswith(b'",,,,,'), "说明行后补 5 个逗号凑齐 6 列")
        self.assertIn("storage_power_mw".encode(), raw)
        self.assertIn("GMT".encode(), raw)

    def test_title_uses_backslash_between_members(self):
        """V3.0 标题：全角括号包组别、半角括号包成员、成员间用反斜杠。"""
        raw = self._gen(team="南网队", members=("张三", "李四")).decode("utf-8")
        self.assertTrue(raw.startswith("（线上组）南网队 (团队成员张三\\李四)——交易决策结果"))


class TestGmtColumns(unittest.TestCase):
    def test_matches_official_sample_exactly(self):
        """与官方 xlsx 的 D/E 列逐行字面比对（含 `-00:00` 这一非常规偏移写法）。"""
        self.assertEqual(
            interval_gmt("2026-11-18", 1),
            ("2026-11-18T08:00:00-00:00", "2026-11-18T09:00:00-00:00"),
        )
        self.assertEqual(
            interval_gmt("2026-11-18", 24),
            ("2026-11-19T07:00:00-00:00", "2026-11-19T08:00:00-00:00"),
        )

    def test_gmt_date_rolls_to_next_day_but_opr_dt_does_not(self):
        """HR17 起 GMT 已进入次日，而 B 列运行日恒为同一天 —— 不能按 GMT 日期分组。"""
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "s.csv"
            write_submission([("2026-11-18", _official_q())], out, verify=False)
            # 必须读字节：read_text 的通用换行会把 CRLF 折叠成 LF
            lines = out.read_bytes().decode("utf-8").split("\r\n")
            row17 = lines[2 + 16].split(",")
            self.assertEqual(row17[1], "2026/11/18")
            self.assertEqual(row17[2], "17")
            self.assertTrue(row17[3].startswith("2026-11-19T00:00:00"))

    def test_dst_spring_forward_uses_real_offset(self):
        """2026-03-08 美国夏令时开始（UTC-7），不得硬编码 +8。"""
        start, _ = interval_gmt("2026-03-08", 1)
        self.assertEqual(start, "2026-03-08T08:00:00-00:00")  # 切换前仍 PST
        start4, _ = interval_gmt("2026-03-08", 4)
        self.assertEqual(start4, "2026-03-08T10:00:00-00:00")  # 本地 03:00 已跳为 PDT → UTC 10:00

    def test_hour_out_of_range_rejected(self):
        with self.assertRaises(ValueError):
            interval_gmt("2026-11-18", 0)
        with self.assertRaises(ValueError):
            interval_gmt("2026-11-18", 25)


class TestRoundTrip(unittest.TestCase):
    def test_read_back_matches_written(self):
        q = _curve()
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "s.csv"
            write_submission([("2026-11-24", q)], out, team="甲队", members=("张三", "李四"))
            title, data, footer = read_submission(out)
            self.assertIn("甲队", title)
            self.assertIn("张三\\李四", title)
            self.assertEqual(len(data), 1)
            self.assertEqual(data[0][0], "2026-11-24", "读回统一为 ISO 日期")
            self.assertEqual(len(data[0][1]), 24)
            for a, b in zip(data[0][1], q):
                self.assertAlmostEqual(a, b, places=6)
            self.assertIn("storage_power_mw", footer)

    def test_multi_day_sequence_continuous(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "s.csv"
            write_submission(
                [("2026-11-23", _curve()), ("2026-11-24", _curve())], out, verify=False
            )
            rep = check_submission(out, soc_start_mwh=DEFAULT_STORAGE.soc_init_mwh)
            self.assertEqual(rep.days, 2)
            self.assertEqual(rep.rows, 48)
            self.assertTrue(rep.ok, rep.summary())

    def test_gmt_column_mismatch_is_caught(self):
        """手工把 GMT 列改错一小时，回读必须报错（防时区/夏令时写错）。"""
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "s.csv"
            write_submission([("2026-11-24", _curve())], out, verify=False)
            raw = out.read_text(encoding="utf-8").replace(
                "2026-11-24T08:00:00-00:00", "2026-11-24T07:00:00-00:00", 1
            )
            out.write_text(raw, encoding="utf-8", newline="")
            rep = check_submission(out)
            self.assertFalse(rep.ok)
            self.assertTrue(any("GMT" in i.message for i in rep.issues), rep.summary())


class TestLegacyReadback(unittest.TestCase):
    """历史提交文件是 V2 旧格式（Oracle 上 11-15~11-17 的产物）。

    内部 SOC 链重放必须能读它们，否则新代码一上线就因读不到历史而中断；
    但**提交前自检绝不放松**，旧格式一律拒收。
    """

    LEGACY = (
        "（线上组）T——交易决策结果,,,\r\n"
        "序号,\ufeffoperate_date,skd,storage_power_mw\r\n"
        + "".join(
            f"{i},2026/11/17,{i-1}:00:00,{-7.0 if i in (12,13) else 0.0:.2f}\r\n"
            for i in range(1, 25)
        )
        + '"说明\n",,,\r\n'
    )

    def _write(self) -> Path:
        p = Path(tempfile.mkdtemp()) / "legacy.csv"
        p.write_text(self.LEGACY, encoding="utf-8", newline="")
        return p

    def test_legacy_rejected_by_default(self):
        with self.assertRaises(ValueError):
            read_submission(self._write())

    def test_legacy_readable_for_internal_replay(self):
        _, data, _ = read_submission(self._write(), allow_legacy=True)
        self.assertEqual(len(data), 1)
        self.assertEqual(data[0][0], "2026-11-17")
        self.assertEqual(len(data[0][1]), 24)
        self.assertEqual(data[0][1][11], -7.0, "skd 是 0-based，须映射到正确的时段索引")

    def test_mixed_legacy_and_v3_chain(self):
        """新旧格式混排在历史目录里时，链式重放必须连续、不中断。"""
        from src.pipeline.runner import chain_soc

        legacy = self._write()
        with tempfile.TemporaryDirectory() as td:
            new = Path(td) / "new.csv"
            write_submission([("2026-11-18", _curve())], new, verify=False)
            both = chain_soc([legacy, new], DEFAULT_STORAGE.soc_init_mwh)
            only_new = chain_soc([new], DEFAULT_STORAGE.soc_init_mwh)
            self.assertNotAlmostEqual(both, only_new, places=6)
            self.assertGreaterEqual(both, DEFAULT_STORAGE.soc_min_mwh)


class TestCheckCatchesErrors(unittest.TestCase):
    def _write_raw(self, text: str) -> Path:
        p = Path(tempfile.mkdtemp()) / "bad.csv"
        p.write_text(text, encoding="utf-8", newline="")
        return p

    def _lines_for(self, date: str, q: list[float]) -> list[str]:
        lines = ["（线上组）T (团队成员a\\b)——交易决策结果,,,,,", HEADER_LINE]
        for i, v in enumerate(q, 1):
            g0, g1 = interval_gmt(date, i)
            lines.append(f"{i},2026/11/24,{i},{g0},{g1},{v:.2f}")
        lines.append(FOOTER_LINE)
        return lines

    def test_detects_missing_periods(self):
        """writer 有 24 段守卫，故手工构造 20 段文件绕过它，确认 check 也能抓到。"""
        p = self._write_raw("\r\n".join(self._lines_for("2026-11-24", _curve()[:20])) + "\r\n")
        rep = check_submission(p)
        self.assertFalse(rep.ok)
        self.assertTrue(any("时段数" in i.message for i in rep.issues), rep.summary())

    def test_detects_power_violation_without_writer_guard(self):
        q = _curve()
        q[3] = 9.5
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "s.csv"
            write_submission([("2026-11-24", q)], out, verify=False)
            rep = check_submission(out)
            self.assertFalse(rep.ok)
            self.assertTrue(any("越界" in i.message for i in rep.issues), rep.summary())

    def test_detects_infeasible_soc(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "s.csv"
            write_submission([("2026-11-24", [7.0] * 24)], out, verify=False)
            rep = check_submission(out)
            self.assertFalse(rep.ok)
            self.assertFalse(rep.per_day["2026-11-24"]["feasible"])

    def test_detects_lf_only_line_endings(self):
        p = self._write_raw("标题,,,,,\n" + HEADER_LINE + "\n")
        rep = check_submission(p)
        self.assertTrue(any("CRLF" in i.message for i in rep.issues), rep.summary())

    def test_old_four_column_file_is_rejected(self):
        """旧版 4 列文件必须被拒，防止赛前误用历史产物提交。"""
        p = self._write_raw(
            "（线上组）T——交易决策结果,,,\r\n序号,\ufeffoperate_date,skd,storage_power_mw\r\n"
            "1,2026/11/24,0:00:00,0.00\r\n"
        )
        rep = check_submission(p)
        self.assertFalse(rep.ok)
        self.assertTrue(any("表头" in i.message for i in rep.issues), rep.summary())


class TestOfficialSampleCurve(unittest.TestCase):
    """V3.0 官方示例曲线的两面性 —— 既是参数的证据，也是陷阱。

    充电段 HR12/13/14 = -7 / -7 / -0.82：前两格入库 11.9 MWh，SOC 推到 12.60；
    第三格若仍 -7 会冲到 18.55 超过上限 13.30，故作者精确取 -0.82 停在 13.297
    （余量 0.003 MWh）。这证明资产参数 14 MWh / 7 MW / η=0.85 与符号约定与官方一致。

    但放电段 HR17/18/19 = **+10 MW**：既超过 Pmax=7 MW，又使 SOC 跌至 −157%。
    结论：示例曲线整体**不可行**，照抄 = 当日全天收益归零。
    """

    def test_charge_segment_alone_lands_3wh_below_ceiling(self):
        q = [0.0] * 24
        for hr in (12, 13, 14):
            q[hr - 1] = OFFICIAL_SAMPLE[hr]
        rep = validate_dispatch(q, DEFAULT_STORAGE.soc_init_mwh, DEFAULT_STORAGE)
        self.assertTrue(rep.feasible, rep.violations)
        self.assertAlmostEqual(rep.max_soc_mwh, 13.2970, places=4)
        ceiling = DEFAULT_STORAGE.capacity_mwh * DEFAULT_STORAGE.soc_max_frac
        self.assertAlmostEqual(ceiling - rep.max_soc_mwh, 0.0030, places=4)

    def test_naive_minus7_would_overflow(self):
        """反证：HR14 若照抄 -7 必越上界，说明 -0.82 是对着上限反解的，非笔误。"""
        q = [0.0] * 24
        for hr in (12, 13, 14):
            q[hr - 1] = OFFICIAL_SAMPLE[hr]
        q[13] = -7.0
        rep = validate_dispatch(q, DEFAULT_STORAGE.soc_init_mwh, DEFAULT_STORAGE)
        self.assertFalse(rep.feasible)
        self.assertTrue(any("超过上限" in v for v in rep.violations), rep.violations)

    def test_official_sample_is_infeasible_and_must_be_rejected(self):
        """完整示例（含 +10MW 放电段）超功率且 SOC 穿底，校验器必须拒绝。"""
        rep = validate_dispatch(_official_q(), DEFAULT_STORAGE.soc_init_mwh, DEFAULT_STORAGE)
        self.assertFalse(rep.feasible, "官方示例曲线不可行，照抄即当日归零")

    def test_official_sample_file_fails_check(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "official.csv"
            write_submission([("2026-11-18", _official_q())], out, verify=False)
            rep = check_submission(out)
            self.assertFalse(rep.ok)
            self.assertTrue(any("越界" in i.message for i in rep.issues), rep.summary())
            self.assertTrue(any("物理不可行" in i.message for i in rep.issues), rep.summary())

    def test_official_sample_power_is_clamped_not_copied(self):
        """把放电段钳到 Pmax=7 后仍不可行（能量不足），证明不能靠"修功率"挽救示例。"""
        q = [min(v, DEFAULT_STORAGE.power_mw) for v in _official_q()]
        rep = validate_dispatch(q, DEFAULT_STORAGE.soc_init_mwh, DEFAULT_STORAGE)
        self.assertFalse(rep.feasible)


class TestTemplateProbe(unittest.TestCase):
    def test_probe_on_generated_file(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "s.csv"
            write_submission([("2026-11-24", _curve())], out, verify=False)
            d = describe_template(out)
            self.assertTrue(d["crlf"])
            self.assertEqual(d["data_rows"], 24)
            self.assertEqual(d["hr_first"], "1")
            self.assertEqual(d["hr_last"], "24")
            self.assertEqual(d["gmt_first"], "2026-11-24T08:00:00-00:00")
            self.assertEqual(d["gmt_last"], "2026-11-25T08:00:00-00:00")
            self.assertGreater(d["header_bom_pos"], 0)
            self.assertEqual(d["header"], HEADER_LINE)


if __name__ == "__main__":
    unittest.main()
