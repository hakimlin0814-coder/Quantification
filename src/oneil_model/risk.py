"""Risk primitives and unresolved gates from 09 §19 and decisions D06/D07/D13."""

from __future__ import annotations


def original_stop_band(entry_price: float) -> tuple[float, float]:
    """Return the book's 7%-8% initial-loss band without choosing one endpoint."""
    if entry_price <= 0:
        raise ValueError("entry_price must be positive")
    return round(entry_price * 0.92, 6), round(entry_price * 0.93, 6)


def unresolved_portfolio_rules() -> list[str]:
    return ["D06 初始仓位与加仓", "D07 止损成本基准", "D12 成本与容量", "D13 组合暴露"]
