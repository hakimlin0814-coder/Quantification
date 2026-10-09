from datetime import datetime
from types import SimpleNamespace
import unittest

import pandas as pd

from quant_review.engine import TushareReviewEngine
from quant_review.strategies.concept_radar import ConceptRadarStrategy
from quant_review.strategies.strong_concept_momentum import StrongConceptMomentumStrategy


class StrongConceptMomentumTests(unittest.TestCase):
    trade_dates = ["20260821", "20260824", "20260825"]

    def test_snapshot_mode_uses_live_quotes_after_open(self):
        mode, target = ConceptRadarStrategy._snapshot_mode(
            self.trade_dates, datetime(2026, 8, 25, 10, 0)
        )
        self.assertEqual((mode, target), ("live", "20260825"))

    def test_snapshot_mode_uses_1130_snapshot_during_lunch(self):
        mode, target = ConceptRadarStrategy._snapshot_mode(
            self.trade_dates, datetime(2026, 8, 25, 12, 0)
        )
        self.assertEqual((mode, target), ("lunch", "20260825"))

    def test_snapshot_mode_uses_close_after_15(self):
        mode, target = ConceptRadarStrategy._snapshot_mode(
            self.trade_dates, datetime(2026, 8, 25, 15, 1)
        )
        self.assertEqual((mode, target), ("close", "20260825"))

    def test_snapshot_mode_uses_previous_close_on_weekend(self):
        mode, target = ConceptRadarStrategy._snapshot_mode(
            self.trade_dates, datetime(2026, 8, 29, 10, 0)
        )
        self.assertEqual((mode, target), ("history", "20260825"))

    def test_normalize_rt_k_uses_close_as_current_price(self):
        raw = pd.DataFrame(
            [{
                "ts_code": "000001.SZ",
                "open": 10.0,
                "high": 10.5,
                "low": 9.8,
                "close": 10.2,
                "pre_close": 10.0,
                "vol": 1000,
                "amount": 2000000,
            }]
        )
        normalized = TushareReviewEngine._normalize_realtime(
            None, raw, pd.DataFrame(), "20260828"
        )
        self.assertEqual(len(normalized), 1)
        self.assertEqual(float(normalized.iloc[0]["close"]), 10.2)
        self.assertEqual(float(normalized.iloc[0]["pre_close"]), 10.0)

    def test_realtime_missing_ohlc_never_becomes_negative_100_percent(self):
        raw = pd.DataFrame(
            [{"code": "000001", "price": 10.2, "close": 10.0}]
        )
        normalized = TushareReviewEngine._normalize_realtime(
            None, raw, pd.DataFrame(), "20260828"
        )
        self.assertEqual(1, len(normalized))
        row = normalized.iloc[0]
        self.assertEqual(10.2, float(row["open"]))
        self.assertEqual(10.2, float(row["high"]))
        self.assertEqual(10.2, float(row["low"]))
        self.assertAlmostEqual(2.0, float(row["pct_chg"]))

    def test_realtime_explicit_stale_date_is_rejected(self):
        stale = pd.DataFrame(
            [{"code": "000001", "price": 10.2, "date": "20260827"}]
        )
        current = pd.DataFrame(
            [{"code": "000001", "price": 10.2, "date": "20260828"}]
        )
        self.assertFalse(TushareReviewEngine._realtime_frame_is_current(stale, "20260828"))
        self.assertTrue(TushareReviewEngine._realtime_frame_is_current(current, "20260828"))

    def test_last_close_snapshot_fills_missing_current_day(self):
        daily = pd.DataFrame(
            [
                {"ts_code": "000001.SZ", "trade_date": "20260827", "close": 10.0},
                {"ts_code": "000002.SZ", "trade_date": "20260826", "close": 20.0},
            ]
        )
        snapshot = TushareReviewEngine._apply_last_close_snapshot(
            daily, "20260828"
        )
        current = snapshot[snapshot["trade_date"] == "20260828"]
        self.assertEqual(len(current), 2)
        self.assertEqual(set(current["ts_code"]), {"000001.SZ", "000002.SZ"})

    def test_rank_value_is_shared_by_selected_mode(self):
        strategy = StrongConceptMomentumStrategy.__new__(StrongConceptMomentumStrategy)
        strategy.config = SimpleNamespace(
            strong_concept_stock_rank_mode="count",
            strong_concept_stock_top_count=12,
            strong_concept_stock_top_percent=10,
        )
        self.assertEqual(strategy._stock_rank_value(), 12)
        self.assertEqual(strategy._stock_rank_label(), "股票前12名")

        strategy.config.strong_concept_stock_rank_mode = "percentile"
        strategy.config.strong_concept_stock_top_percent = 8.5
        self.assertEqual(strategy._stock_rank_value(), 8.5)
        self.assertEqual(strategy._stock_rank_label(), "股票前8.5%")

    def test_industry_rank_level_defaults_to_sw_l3(self):
        strategy = StrongConceptMomentumStrategy.__new__(StrongConceptMomentumStrategy)
        strategy.config = SimpleNamespace()
        self.assertEqual(strategy._industry_rank_level(), "sw_l3")
        self.assertEqual(strategy._industry_rank_level_label(), "申万三级")

    def test_industry_rank_level_accepts_sw_l1_and_sw_l2(self):
        strategy = StrongConceptMomentumStrategy.__new__(StrongConceptMomentumStrategy)
        strategy.config = SimpleNamespace(strong_concept_industry_rank_level="sw_l2")
        self.assertEqual(strategy._industry_rank_level_label(), "申万二级")
        strategy.config.strong_concept_industry_rank_level = "sw_l1"
        self.assertEqual(strategy._industry_rank_level_label(), "申万一级")


if __name__ == "__main__":
    unittest.main()
