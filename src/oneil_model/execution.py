"""A-share tradability and T+1 guards from 05 §11/§15 and 09 §18."""

from __future__ import annotations

from typing import Any

import pandas as pd


def tradability_preview(row: pd.Series, limit_row: pd.Series | None = None) -> dict[str, Any]:
    """Describe observable tradability without assuming a future fill (09 §18)."""
    if bool(row.get("suspended_latest", False)):
        return {"status": "SUSPENDED", "buyable": False, "source": "A", "note": "观察日无行情"}
    close = pd.to_numeric(pd.Series([row.get("close")]), errors="coerce").iloc[0]
    high = pd.to_numeric(pd.Series([row.get("high")]), errors="coerce").iloc[0]
    low = pd.to_numeric(pd.Series([row.get("low")]), errors="coerce").iloc[0]
    up_limit = None
    limit_type = ""
    if limit_row is not None:
        up_limit = pd.to_numeric(pd.Series([limit_row.get("up_limit")]), errors="coerce").iloc[0]
        limit_type = str(limit_row.get("limit", limit_row.get("limit_type", ""))).strip().upper()
    one_price_up = bool(
        pd.notna(close)
        and pd.notna(high)
        and pd.notna(low)
        and abs(float(high) - float(low)) < 1e-9
        and (
            (pd.notna(up_limit) and abs(float(close) - float(up_limit)) < 0.005)
            or limit_type in {"U", "UP", "涨停"}
        )
    )
    if one_price_up:
        return {"status": "ONE_PRICE_LIMIT_UP", "buyable": False, "source": "A", "note": "一字涨停不假定买入成交"}
    return {
        "status": "NEXT_SESSION_UNKNOWN",
        "buyable": None,
        "source": "A",
        "note": "Q0筛选只生成观察候选；下一交易日需重新检查停牌、涨跌停和成交量",
    }


def enforce_t_plus_one(acquired_trade_date: str, requested_sell_date: str) -> bool:
    """Return whether shares acquired on one trade date may be sold on another."""
    return str(requested_sell_date) > str(acquired_trade_date)
