"""读取 `.pipeline.env`（KEY=VALUE 格式）。

存在的理由：队名会进入提交文件名，一旦配错，14 天的文件全是废的。
systemd 通过 `EnvironmentFile` 能读，但手工执行 `python scripts/pipeline.py`
时就读不到 —— 两种入口行为不一致是最难查的那类问题。所以在 CLI 层统一补上。

规则：**不覆盖已有的环境变量**。即 `CSG_TEAM=X python scripts/pipeline.py`
的优先级永远高于配置文件，便于临时试跑。
"""
from __future__ import annotations

import os
from pathlib import Path

DEFAULT_KEYS = ("CSG_TEAM", "CSG_MEMBERS", "CSG_METHOD", "CSG_WEBHOOK")


def parse_env_file(path: str | Path) -> dict[str, str]:
    p = Path(path)
    if not p.exists():
        return {}
    out: dict[str, str] = {}
    for raw in p.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k, v = k.strip(), v.strip().strip('"').strip("'")
        if k:
            out[k] = v
    return out


def load(path: str | Path, keys: tuple[str, ...] = DEFAULT_KEYS) -> dict[str, str]:
    """把文件里的值注入 os.environ（仅当同名环境变量尚未设置），返回实际生效的配置。"""
    values = parse_env_file(path)
    applied: dict[str, str] = {}
    for k, v in values.items():
        if k not in keys:
            continue
        if not os.environ.get(k):
            os.environ[k] = v
        applied[k] = os.environ[k]
    return applied
