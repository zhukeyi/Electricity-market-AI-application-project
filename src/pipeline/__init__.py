"""流水线公共出口。

    from src.pipeline import Config, run

    rep = run(Config(team="某队", members=("张三",)))
    print(rep.summary())
"""
from .clock import (
    Schedule,
    bj_deadline,
    la_day_hours,
    next_submittable,
    schedule,
    trade_date_for,
    utc_now,
)
from .runner import Config, Pipeline, PipelineError, RunReport, Step, run

__all__ = [
    "Config",
    "Pipeline",
    "PipelineError",
    "RunReport",
    "Schedule",
    "Step",
    "bj_deadline",
    "la_day_hours",
    "next_submittable",
    "run",
    "schedule",
    "trade_date_for",
    "utc_now",
]
