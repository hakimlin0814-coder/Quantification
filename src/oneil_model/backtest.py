"""Backtest authorization gate corresponding to 09 §21 and §29."""

from __future__ import annotations

from .config import OneilConfigurationError, OneilResearchConfig


class OneilBacktestEngine:
    """Explicit code gate: Q0 v1 has no authorised portfolio backtest yet."""

    def __init__(self, config: OneilResearchConfig) -> None:
        self.config = config

    def run(self) -> None:
        raise OneilConfigurationError(
            "组合回测尚未开放：D06、D07、D12、D13和同日触发顺序仍未确认；"
            "不能复用现有回测器的隐藏默认值"
        )
