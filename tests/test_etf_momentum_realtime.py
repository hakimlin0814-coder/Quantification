from datetime import datetime

import pandas as pd

from quant_review.strategies.etf_momentum import EtfMomentumStrategy


TRADE_DATES = ["20260724", "20260727", "20260728"]


def test_snapshot_mode_uses_live_quotes_before_close():
    mode, target = EtfMomentumStrategy._snapshot_mode(
        datetime(2026, 7, 28, 10, 30), TRADE_DATES
    )
    assert (mode, target) == ("live", "20260728")


def test_snapshot_mode_uses_today_close_after_15():
    mode, target = EtfMomentumStrategy._snapshot_mode(
        datetime(2026, 7, 28, 15, 1), TRADE_DATES
    )
    assert (mode, target) == ("close", "20260728")


def test_snapshot_mode_uses_last_close_on_market_holiday():
    mode, target = EtfMomentumStrategy._snapshot_mode(
        datetime(2026, 7, 29, 10, 30), TRADE_DATES
    )
    assert (mode, target) == ("history", "20260728")


def test_snapshot_mode_uses_last_close_before_open():
    mode, target = EtfMomentumStrategy._snapshot_mode(
        datetime(2026, 7, 28, 8, 30), TRADE_DATES
    )
    assert (mode, target) == ("history", "20260727")


def test_stale_quotes_are_rejected():
    quotes = pd.DataFrame(
        {
            "ts_code": ["510300.SH"],
            "quote_at": ["2026-07-27 15:00:00"],
            "close": [4.2],
        }
    )
    result, quote_at = EtfMomentumStrategy._quotes_for_trade_date(
        quotes, "20260728"
    )
    assert result.empty
    assert quote_at == ""


def test_current_quotes_keep_actual_market_timestamp():
    quotes = pd.DataFrame(
        {
            "ts_code": ["510300.SH", "159919.SZ"],
            "quote_at": ["2026-07-28 10:30:01", "2026-07-28 10:30:03"],
            "close": [4.2, 4.1],
        }
    )
    result, quote_at = EtfMomentumStrategy._quotes_for_trade_date(
        quotes, "20260728"
    )
    assert len(result) == 2
    assert quote_at == "2026-07-28 10:30:03"
