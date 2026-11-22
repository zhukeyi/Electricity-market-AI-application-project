"""提交完成后的通知出口。

只支持一种形态：向一个 HTTP 端点 POST JSON。企业微信/钉钉群机器人的 webhook
都吃这个形状，无需额外依赖。未配置时静默跳过，**不因为通知失败而让整条链路失败**
—— 通知是尽力而为，文件落地才是硬指标。
"""
from __future__ import annotations

import json
import urllib.request

WECOM_HOST = "qyapi.weixin.qq.com"


def build_payload(text: str, url: str) -> dict:
    if WECOM_HOST in url:
        return {"msgtype": "markdown", "markdown": {"content": text}}
    return {"text": text}


def send(text: str, url: str | None, timeout: int = 8) -> tuple[bool, str]:
    """返回 (是否成功, 说明)。永不抛出。"""
    if not url:
        return False, "未配置 webhook，跳过"
    try:
        body = json.dumps(build_payload(text, url), ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(
            url, data=body, headers={"Content-Type": "application/json"}, method="POST"
        )
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return 200 <= r.status < 300, f"HTTP {r.status}"
    except Exception as exc:  # noqa: BLE001
        return False, f"{type(exc).__name__}: {exc}"
