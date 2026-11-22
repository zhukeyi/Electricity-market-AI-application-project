"""CAISO OASIS 数据客户端。

背景（为什么代码里有 SSH 传输层）：
本地网络出口无法访问 `oasis.caiso.com`（curl 返回 `CONNECT tunnel failed 502`），
而 Oracle ARM 服务器（217.142.246.70）可直连并返回 200。因此客户端抽象出
`transport`：

- `direct`：直接 HTTPS 请求，在 Oracle 上运行或网络可达时使用；
- `ssh`   ：通过 SSH 在 Oracle 上执行 curl，base64 回传 zip。本地开发/单次补数用；
- `auto`  ：按 `mode` 自动选择（默认 ssh）。

CAISO OASIS 接口约定（实测确认）：
- 端点 `https://oasis.caiso.com/oasisapi/SingleZip`
- 必填 `queryname=PRC_LMP`、`startdatetime`/`enddatetime`（UTC，格式 `YYYYMMDDTHH:MM-0000`）、
  `version=1`、`market_run_id=DAM`、`node=TH_SP15_GEN-APND`
- 返回一个 zip，内含单个 XML（`<date>_<date>_PRC_LMP_DAM_<ts>_v1.xml`）
- 单次查询跨度建议 ≤ 31 天，超长区间 CAISO 会截断或报错
"""
from __future__ import annotations

import base64
import datetime as dt
import io
import logging
import subprocess
import time
import urllib.request
import zipfile
from dataclasses import dataclass
from typing import Optional

log = logging.getLogger(__name__)

UA = "Mozilla/5.0 (compatible; csg-ai-market/0.1)"
"""OASIS 对无 UA 的请求偶发拒绝，统一带上。"""

OASIS_BASE = "https://oasis.caiso.com/oasisapi/SingleZip"

DEFAULT_SSH_HOST = "ubuntu@217.142.246.70"
DEFAULT_SSH_KEY = "~/.ssh/id_rsa_oracle"


class FetchError(RuntimeError):
    pass


@dataclass
class OasisClient:
    transport: str = "auto"
    """direct | ssh | auto"""

    ssh_host: str = DEFAULT_SSH_HOST
    ssh_key: str = DEFAULT_SSH_KEY
    ssh_connect_timeout: int = 12
    http_timeout: int = 60
    max_retries: int = 4
    backoff_s: float = 2.0
    sleep_between: float = 0.4
    """OASIS 对高频请求会限流，批量抓取时保持间隔。"""

    def __post_init__(self) -> None:
        if self.transport == "auto":
            self.transport = "ssh"

    # ---------------- HTTP / SSH 传输层 ----------------

    def _get_direct(self, url: str) -> bytes:
        """直连抓取。仅用标准库 urllib —— 服务器上少装一个依赖就少一处运维风险。"""
        req = urllib.request.Request(url, headers={"User-Agent": UA})
        with urllib.request.urlopen(req, timeout=self.http_timeout) as resp:
            return resp.read()

    def _get_ssh(self, url: str) -> bytes:
        """在远端执行 curl，base64 回传，避免二进制经 SSH 的换行转换。"""
        remote = f"curl -sS --max-time {self.http_timeout} {url!r} | base64 -w0"
        cmd = [
            "ssh", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=no",
            "-o", f"ConnectTimeout={self.ssh_connect_timeout}",
            "-i", self.ssh_key, self.ssh_host, remote,
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=self.http_timeout + 30)
        if proc.returncode != 0:
            raise FetchError(f"SSH 传输失败 rc={proc.returncode}: {proc.stderr.strip()[:400]}")
        payload = proc.stdout.strip()
        if not payload:
            raise FetchError("SSH 传输返回空内容")
        try:
            return base64.b64decode(payload)
        except Exception as exc:  # noqa: BLE001
            raise FetchError(f"base64 解码失败: {exc}") from exc

    def get(self, url: str) -> bytes:
        last: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                data = self._get_direct(url) if self.transport == "direct" else self._get_ssh(url)
                if not data:
                    raise FetchError("响应为空")
                if data[:2] != b"PK":
                    head = data[:400].decode("utf-8", "replace")
                    raise FetchError(f"响应不是 zip（前 400 字节）: {head}")
                return data
            except Exception as exc:  # noqa: BLE001
                last = exc
                wait = self.backoff_s * (2 ** (attempt - 1))
                log.warning("OASIS 请求失败（第 %d/%d 次）：%s，%.1fs 后重试",
                            attempt, self.max_retries, exc, wait)
                if attempt < self.max_retries:
                    time.sleep(wait)
        raise FetchError(f"重试 {self.max_retries} 次仍失败: {last}")

    # ---------------- OASIS 查询 ----------------

    @staticmethod
    def _fmt_utc(d: dt.datetime) -> str:
        if d.tzinfo is not None:
            d = d.astimezone(dt.timezone.utc).replace(tzinfo=None)
        return d.strftime("%Y%m%dT%H:%M-0000")

    def query_prc_lmp(
        self,
        start_utc: dt.datetime,
        end_utc: dt.datetime,
        node: str = "TH_SP15_GEN-APND",
        market_run_id: str = "DAM",
    ) -> bytes:
        url = (
            f"{OASIS_BASE}?queryname=PRC_LMP"
            f"&startdatetime={self._fmt_utc(start_utc)}"
            f"&enddatetime={self._fmt_utc(end_utc)}"
            f"&version=1&market_run_id={market_run_id}&node={node}"
        )
        return self.get(url)

    def fetch_dam_lmp(
        self,
        start_date: dt.date,
        end_date: dt.date,
        node: str = "TH_SP15_GEN-APND",
        chunk_days: int = 30,
    ):
        """按 ≤31 天分块抓取，返回合并后的 DataFrame（列：date/hour/lmp/utc）。"""
        from .parser import parse_prc_lmp_zip

        frames = []
        cur = start_date
        while cur <= end_date:
            chunk_end = min(cur + dt.timedelta(days=chunk_days - 1), end_date)
            # 运行日 D 的日前价格在本地 D-1 日 10:00 前出清；用 UTC 覆盖全天即可
            s = dt.datetime.combine(cur, dt.time(0, 0)) - dt.timedelta(hours=8)
            e = dt.datetime.combine(chunk_end, dt.time(0, 0)) + dt.timedelta(hours=16)
            log.info("抓取 %s ~ %s", cur, chunk_end)
            raw = self.query_prc_lmp(s, e, node=node)
            df = parse_prc_lmp_zip(raw, node=node)
            frames.append(df)
            cur = chunk_end + dt.timedelta(days=1)
            time.sleep(self.sleep_between)

        import pandas as pd

        if not frames:
            return pd.DataFrame(columns=["date", "hour", "lmp", "utc_start", "node"])
        out = pd.concat(frames, ignore_index=True)
        out = out.drop_duplicates(subset=["date", "hour", "node"]).sort_values(["date", "hour"])
        return out.reset_index(drop=True)


def raw_to_zip_member(data: bytes) -> tuple[str, bytes]:
    """从 OASIS zip 中取出唯一的 XML 成员。"""
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        names = z.namelist()
        if not names:
            raise FetchError("zip 内无文件")
        name = names[0]
        return name, z.read(name)
