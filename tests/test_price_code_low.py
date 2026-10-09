from __future__ import annotations

import unittest

import pandas as pd

from quant_review.server import price_code_volume_range
from quant_review.strategies.price_code_low import PriceCodeLowStrategy


class PriceCodeDetectorTests(unittest.TestCase):
    def categories(self, price: float) -> set[str]:
        return {
            item["category"]
            for item in PriceCodeLowStrategy.detect_price_codes(price)
        }

    def details(self, price: float) -> str:
        return "；".join(
            item["detail"]
            for item in PriceCodeLowStrategy.detect_price_codes(price)
        )

    def test_detects_repeated_tail(self) -> None:
        categories = self.categories(7.77)
        self.assertIn("叠数", categories)
        self.assertIn("对称回文", categories)

    def test_detects_repeated_block_and_palindrome(self) -> None:
        self.assertIn("重复片段", self.categories(23.23))
        self.assertIn("对称回文", self.categories(1233.21))

    def test_detects_whole_sequences(self) -> None:
        self.assertIn("顺子", self.categories(1234.56))
        self.assertIn("顺子", self.categories(7.65))

    def test_removed_broad_categories_do_not_match(self) -> None:
        for price in [3.58, 9.45, 10.15, 20.30, 46.48, 24.98, 2.50]:
            self.assertEqual([], PriceCodeLowStrategy.detect_price_codes(price))

    def test_does_not_match_local_palindrome_or_sequence(self) -> None:
        for price in [23.63, 10.96, 11.31, 13.51]:
            self.assertEqual([], PriceCodeLowStrategy.detect_price_codes(price))

    def test_four_digit_integer_is_not_truncated(self) -> None:
        formatted, digits = PriceCodeLowStrategy._price_parts(1234.43)
        self.assertEqual("1234.43", formatted)
        self.assertEqual("123443", digits)
        self.assertIn("对称回文", self.categories(1234.43))

    def test_plain_price_can_have_no_code(self) -> None:
        self.assertEqual([], PriceCodeLowStrategy.detect_price_codes(14.62))

    def test_live_share_volume_is_converted_to_hands(self) -> None:
        quote = pd.Series({"vol": 1000000, "amount": 10000, "close": 10})
        self.assertEqual(
            10000,
            PriceCodeLowStrategy._live_volume_in_hands(quote),
        )

    def test_live_hand_volume_is_kept(self) -> None:
        quote = pd.Series({"vol": 10000, "amount": 10000, "close": 10})
        self.assertEqual(
            10000,
            PriceCodeLowStrategy._live_volume_in_hands(quote),
        )

    def test_rt_k_normalization_preserves_current_close(self) -> None:
        frame = pd.DataFrame(
            [
                {
                    "ts_code": "000001.SZ",
                    "pre_close": 11.32,
                    "high": 11.39,
                    "open": 11.32,
                    "low": 11.25,
                    "close": 11.30,
                    "vol": 1156222,
                    "amount": 1308134000,
                }
            ]
        )
        normalized = PriceCodeLowStrategy._normalize_rt_k(frame, "20260611")
        self.assertEqual(11.30, normalized.iloc[0]["close"])
        self.assertEqual(11.32, normalized.iloc[0]["pre_close"])
        self.assertEqual(1308134, normalized.iloc[0]["amount"])

    def test_stock_universe_filters_st_chinext_and_star(self) -> None:
        basic = pd.DataFrame(
            [
                {"ts_code": "000001.SZ", "symbol": "000001", "name": "平安银行"},
                {"ts_code": "000002.SZ", "symbol": "000002", "name": "*ST测试"},
                {"ts_code": "300001.SZ", "symbol": "300001", "name": "创业测试"},
                {"ts_code": "301001.SZ", "symbol": "301001", "name": "创业测试二"},
                {"ts_code": "688001.SH", "symbol": "688001", "name": "科创测试"},
                {"ts_code": "689001.SH", "symbol": "689001", "name": "科创测试二"},
                {"ts_code": "920001.BJ", "symbol": "920001", "name": "北交测试"},
            ]
        )
        filtered, counts = PriceCodeLowStrategy._filter_stock_universe(
            basic,
            exclude_st=True,
            exclude_chinext=True,
            exclude_star=True,
        )
        self.assertEqual(
            ["000001.SZ", "920001.BJ"],
            filtered["ts_code"].tolist(),
        )
        self.assertEqual({"st": 1, "chinext": 2, "star": 2}, counts)

    def test_stock_universe_keeps_disabled_filter_groups(self) -> None:
        basic = pd.DataFrame(
            [
                {"ts_code": "000002.SZ", "symbol": "000002", "name": "ST测试"},
                {"ts_code": "301001.SZ", "symbol": "301001", "name": "创业测试"},
                {"ts_code": "689001.SH", "symbol": "689001", "name": "科创测试"},
            ]
        )
        filtered, _ = PriceCodeLowStrategy._filter_stock_universe(
            basic,
            exclude_st=False,
            exclude_chinext=False,
            exclude_star=False,
        )
        self.assertEqual(3, len(filtered))

    def test_industry_market_cap_snapshot_ranks_each_sw_l2_group(self) -> None:
        basic = pd.DataFrame(
            [
                {"ts_code": "000001.SZ"},
                {"ts_code": "000002.SZ"},
                {"ts_code": "000003.SZ"},
                {"ts_code": "600001.SH"},
            ]
        )
        members = pd.DataFrame(
            [
                {"ts_code": "000001.SZ", "rank_industry": "行业A", "rank_industry_code": "A"},
                {"ts_code": "000002.SZ", "rank_industry": "行业A", "rank_industry_code": "A"},
                {"ts_code": "000003.SZ", "rank_industry": "行业A", "rank_industry_code": "A"},
                {"ts_code": "600001.SH", "rank_industry": "行业B", "rank_industry_code": "B"},
            ]
        )
        daily_basic = pd.DataFrame(
            [
                {"ts_code": "000001.SZ", "total_mv": 1000000},
                {"ts_code": "000002.SZ", "total_mv": 3000000},
                {"ts_code": "000003.SZ", "total_mv": 2000000},
                {"ts_code": "600001.SH", "total_mv": 500000},
            ]
        )
        ranked = PriceCodeLowStrategy._industry_market_cap_snapshot(
            basic,
            daily_basic,
            members,
            trade_date="20260120",
        )
        by_code = ranked.set_index("ts_code").to_dict("index")
        self.assertEqual(1, by_code["000002.SZ"]["industry_mv_rank"])
        self.assertEqual(2, by_code["000003.SZ"]["industry_mv_rank"])
        self.assertEqual(3, by_code["000001.SZ"]["industry_mv_rank"])
        self.assertEqual(1, by_code["600001.SH"]["industry_mv_rank"])
        self.assertEqual(300.0, by_code["000002.SZ"]["market_cap_yi"])
        self.assertEqual("20260120", by_code["000002.SZ"]["trade_date"])

    def test_build_rows_requires_code_and_complete_new_low_window(self) -> None:
        strategy = PriceCodeLowStrategy.__new__(PriceCodeLowStrategy)
        basic = pd.DataFrame(
            [
                {
                    "ts_code": "000001.SZ",
                    "symbol": "000001",
                    "name": "测试股票",
                    "industry": "测试行业",
                    "stock_type": "深市主板",
                }
            ]
        )
        daily = pd.DataFrame(
            [
                {
                    "ts_code": "000001.SZ",
                    "trade_date": f"202601{day:02d}",
                    "low": 10.10,
                    "vol": 10000,
                }
                for day in range(1, 20)
            ]
        )
        realtime = pd.DataFrame(
            [
                {
                    "ts_code": "000001.SZ",
                    "trade_date": "20260120",
                    "open": 10.20,
                    "high": 10.30,
                    "low": 9.99,
                    "close": 10.00,
                    "pre_close": 10.20,
                    "pct_chg": -1.96,
                    "vol": 5000,
                    "amount": 20000,
                }
            ]
        )

        rows = strategy._build_rows(
            basic=basic,
            daily=daily,
            realtime=realtime,
            trade_date="20260120",
            low_days=20,
            volume_days=5,
        )

        self.assertEqual(1, len(rows))
        self.assertEqual(9.99, rows[0]["low"])
        self.assertEqual("缩量 50.00%", rows[0]["volume_state"])
        self.assertEqual(0.5, rows[0]["volume_ratio"])

    def test_backtest_reports_next_trading_day_close(self) -> None:
        strategy = PriceCodeLowStrategy.__new__(PriceCodeLowStrategy)
        basic = pd.DataFrame(
            [
                {
                    "ts_code": "000001.SZ",
                    "symbol": "000001",
                    "name": "测试股票",
                    "industry": "测试行业",
                    "stock_type": "深市主板",
                }
            ]
        )
        daily_rows = [
            {
                "ts_code": "000001.SZ",
                "trade_date": f"202601{day:02d}",
                "low": 10.10,
                "close": 10.20,
                "vol": 10000,
            }
            for day in range(1, 20)
        ]
        daily_rows.extend(
            [
                {
                    "ts_code": "000001.SZ",
                    "trade_date": "20260120",
                    "low": 9.99,
                    "close": 10.00,
                    "vol": 5000,
                },
                {
                    "ts_code": "000001.SZ",
                    "trade_date": "20260121",
                    "low": 10.20,
                    "close": 10.50,
                    "vol": 15000,
                },
            ]
        )

        rows = strategy._build_backtest_rows(
            basic=basic,
            daily=pd.DataFrame(daily_rows),
            start_date="20260120",
            end_date="20260120",
            low_days=20,
            volume_days=5,
            next_trade_dates={"20260120": "20260121"},
        )

        self.assertEqual(1, len(rows))
        self.assertEqual("20260121", rows[0]["next_date"])
        self.assertEqual(10.50, rows[0]["next_close"])
        self.assertEqual(5.0, rows[0]["next_return_pct"])
        self.assertEqual("上涨", rows[0]["next_result"])
        self.assertEqual("缩量 50.00%", rows[0]["volume_state"])

    def test_backtest_filters_by_historical_industry_market_cap_rank(self) -> None:
        strategy = PriceCodeLowStrategy.__new__(PriceCodeLowStrategy)
        basic = pd.DataFrame(
            [
                {
                    "ts_code": "000001.SZ",
                    "symbol": "000001",
                    "name": "测试股票",
                    "industry": "测试行业",
                    "stock_type": "深市主板",
                }
            ]
        )
        daily_rows = [
            {
                "ts_code": "000001.SZ",
                "trade_date": f"202601{day:02d}",
                "low": 10.10,
                "close": 10.20,
                "vol": 10000,
            }
            for day in range(1, 20)
        ]
        daily_rows.extend(
            [
                {
                    "ts_code": "000001.SZ",
                    "trade_date": "20260120",
                    "low": 9.99,
                    "close": 10.00,
                    "vol": 5000,
                },
                {
                    "ts_code": "000001.SZ",
                    "trade_date": "20260121",
                    "low": 10.20,
                    "close": 10.50,
                    "vol": 15000,
                },
            ]
        )
        rank_history = pd.DataFrame(
            [
                {
                    "trade_date": "20260120",
                    "ts_code": "000001.SZ",
                    "rank_industry": "行业A",
                    "rank_industry_code": "A",
                    "market_cap_yi": 200,
                    "industry_mv_rank": 2,
                }
            ]
        )
        common = {
            "basic": basic,
            "daily": pd.DataFrame(daily_rows),
            "start_date": "20260120",
            "end_date": "20260120",
            "low_days": 20,
            "volume_days": 5,
            "next_trade_dates": {"20260120": "20260121"},
            "industry_rank_history": rank_history,
        }
        self.assertEqual(
            [],
            strategy._build_backtest_rows(
                **common,
                industry_mv_rank_top=1,
            ),
        )
        rows = strategy._build_backtest_rows(
            **common,
            industry_mv_rank_top=2,
        )
        self.assertEqual(1, len(rows))
        self.assertEqual("行业A", rows[0]["rank_industry"])
        self.assertEqual(200.0, rows[0]["market_cap_yi"])
        self.assertEqual(2, rows[0]["industry_mv_rank"])

    def test_price_volume_direction_must_match(self) -> None:
        match = PriceCodeLowStrategy._price_volume_match
        self.assertTrue(match(2.0, 20.0, 1.2))
        self.assertTrue(match(-2.0, -20.0, 0.8))
        self.assertFalse(match(2.0, -20.0, 0.8))
        self.assertFalse(match(-2.0, 20.0, 1.2))
        self.assertFalse(match(0.0, 20.0, 1.2))
        self.assertFalse(match(2.0, 2.0, 1.02))

    def test_volume_filter_can_keep_only_expand_or_shrink(self) -> None:
        match = PriceCodeLowStrategy._volume_filter_match
        self.assertTrue(match("all", 20.0))
        self.assertTrue(match("all", -20.0))
        self.assertTrue(match("expand", 20.0))
        self.assertFalse(match("expand", -20.0))
        self.assertTrue(match("shrink", -20.0))
        self.assertFalse(match("shrink", 20.0))

    def test_custom_volume_threshold_applies_to_direction_and_labels(self) -> None:
        match = PriceCodeLowStrategy._price_volume_match
        self.assertFalse(match(2.0, 20.0, 1.2, 30.0))
        self.assertTrue(match(2.0, 30.0, 1.3, 30.0))
        self.assertFalse(match(-2.0, -20.0, 0.8, 30.0))
        self.assertTrue(match(-2.0, -35.0, 0.65, 30.0))
        self.assertEqual(
            "基本持平 +20.00%",
            PriceCodeLowStrategy._volume_state(20.0, 1.2, 30.0),
        )
        self.assertEqual(
            "缩量 35.00%",
            PriceCodeLowStrategy._volume_state(-35.0, 0.65, 30.0),
        )

    def test_volume_range_has_minimum_and_maximum(self) -> None:
        self.assertEqual(
            (10.0, 30.0),
            PriceCodeLowStrategy._volume_range(10, 30),
        )
        match = PriceCodeLowStrategy._price_volume_match
        self.assertTrue(match(2.0, 20.0, 1.2, 10.0, 30.0))
        self.assertFalse(match(2.0, 40.0, 1.4, 10.0, 30.0))
        self.assertTrue(match(-2.0, -30.0, 0.7, 10.0, 30.0))
        with self.assertRaises(RuntimeError):
            PriceCodeLowStrategy._volume_range(30, 10)

    def test_legacy_volume_threshold_payload_remains_supported(self) -> None:
        self.assertEqual(
            (20.0, 10000.0),
            price_code_volume_range({"price_code_volume_threshold_pct": 20}),
        )
        self.assertEqual(
            (10.0, 30.0),
            price_code_volume_range(
                {
                    "price_code_volume_threshold_pct": 20,
                    "price_code_volume_min_pct": 10,
                    "price_code_volume_max_pct": 30,
                }
            ),
        )

    def test_build_rows_respects_custom_shrink_threshold(self) -> None:
        strategy = PriceCodeLowStrategy.__new__(PriceCodeLowStrategy)
        basic = pd.DataFrame(
            [
                {
                    "ts_code": "000001.SZ",
                    "symbol": "000001",
                    "name": "测试股票",
                    "industry": "测试行业",
                    "stock_type": "深市主板",
                }
            ]
        )
        daily = pd.DataFrame(
            [
                {
                    "ts_code": "000001.SZ",
                    "trade_date": f"202601{day:02d}",
                    "low": 10.10,
                    "vol": 10000,
                }
                for day in range(1, 20)
            ]
        )
        realtime = pd.DataFrame(
            [
                {
                    "ts_code": "000001.SZ",
                    "trade_date": "20260120",
                    "low": 9.99,
                    "close": 10.00,
                    "pre_close": 10.20,
                    "pct_chg": -1.96,
                    "vol": 5000,
                    "amount": 20000,
                }
            ]
        )
        rows = strategy._build_rows(
            basic=basic,
            daily=daily,
            realtime=realtime,
            trade_date="20260120",
            low_days=20,
            volume_days=5,
            volume_threshold_pct=60,
        )
        self.assertEqual([], rows)

    def test_build_rows_can_disable_price_code_requirement(self) -> None:
        strategy = PriceCodeLowStrategy.__new__(PriceCodeLowStrategy)
        basic = pd.DataFrame(
            [
                {
                    "ts_code": "000001.SZ",
                    "symbol": "000001",
                    "name": "测试股票",
                    "industry": "测试行业",
                    "stock_type": "深市主板",
                }
            ]
        )
        daily = pd.DataFrame(
            [
                {
                    "ts_code": "000001.SZ",
                    "trade_date": f"202601{day:02d}",
                    "low": 10.10,
                    "vol": 10000,
                }
                for day in range(1, 20)
            ]
        )
        realtime = pd.DataFrame(
            [
                {
                    "ts_code": "000001.SZ",
                    "trade_date": "20260120",
                    "low": 9.98,
                    "close": 10.00,
                    "pre_close": 10.20,
                    "pct_chg": -1.96,
                    "vol": 8000,
                    "amount": 20000,
                }
            ]
        )
        rows = strategy._build_rows(
            basic=basic,
            daily=daily,
            realtime=realtime,
            trade_date="20260120",
            low_days=20,
            volume_days=5,
            volume_min_pct=10,
            volume_max_pct=30,
            use_patterns=False,
        )
        self.assertEqual(1, len(rows))
        self.assertEqual(0, rows[0]["code_count"])
        self.assertEqual(-20.0, rows[0]["volume_change_pct"])

    def test_selected_codes_are_normalized(self) -> None:
        self.assertEqual(
            {"000001", "600000", "300001"},
            PriceCodeLowStrategy._selected_codes(
                ["000001", "600000.SH", "300001.SZ", "", None, "12"]
            ),
        )

    def test_scan_thresholds_and_parameter_rows(self) -> None:
        thresholds = PriceCodeLowStrategy._scan_thresholds(20, 50, 30)
        self.assertEqual([20, 50], thresholds)
        self.assertEqual(
            [10, 20, 30, 40, 50],
            PriceCodeLowStrategy._scan_low_days(10, 50, 10),
        )
        rows = PriceCodeLowStrategy._parameter_scan_rows(
            [
                {
                    "ts_code": "000001.SZ",
                    "signal_pct_chg": 2,
                    "volume_change_pct": 30,
                    "next_return_pct": 5,
                    "code_categories": "叠数",
                },
                {
                    "ts_code": "000002.SZ",
                    "signal_pct_chg": 3,
                    "volume_change_pct": 60,
                    "next_return_pct": -1,
                    "code_categories": "重复片段",
                },
                {
                    "ts_code": "000003.SZ",
                    "signal_pct_chg": -2,
                    "volume_change_pct": -25,
                    "next_return_pct": 2,
                    "code_categories": "对称回文",
                },
                {
                    "ts_code": "000004.SZ",
                    "signal_pct_chg": -3,
                    "volume_change_pct": -70,
                    "next_return_pct": -3,
                    "code_categories": "顺子",
                },
            ],
            thresholds=thresholds,
            direction="both",
            min_signals=2,
            low_days=30,
        )
        self.assertEqual(4, len(rows))
        by_parameter = {
            (row["direction"], row["threshold_pct"]): row
            for row in rows
        }
        self.assertEqual(2, by_parameter[("expand", 20.0)]["signal_count"])
        self.assertEqual(1, by_parameter[("expand", 50.0)]["signal_count"])
        self.assertEqual(2, by_parameter[("shrink", 20.0)]["signal_count"])
        self.assertEqual(1, by_parameter[("shrink", 50.0)]["signal_count"])
        self.assertTrue(by_parameter[("expand", 20.0)]["qualified"])
        self.assertFalse(by_parameter[("expand", 50.0)]["qualified"])
        self.assertTrue(by_parameter[("expand", 20.0)]["positive_expectation"])
        self.assertFalse(by_parameter[("shrink", 20.0)]["positive_expectation"])
        self.assertTrue(all(row["low_days"] == 30 for row in rows))
        self.assertEqual(("expand", 20.0), (rows[0]["direction"], rows[0]["threshold_pct"]))

    def test_parameter_scan_expands_industry_rank_top_one_to_five(self) -> None:
        rows = PriceCodeLowStrategy._parameter_scan_rows(
            [
                {
                    "ts_code": f"00000{rank}.SZ",
                    "signal_pct_chg": 2,
                    "volume_change_pct": 30,
                    "next_return_pct": rank,
                    "code_categories": "叠数",
                    "industry_mv_rank": rank,
                }
                for rank in range(1, 6)
            ],
            thresholds=[20],
            direction="expand",
            min_signals=1,
            low_days=20,
            industry_mv_rank_tops=[1, 2, 3, 4, 5],
        )
        self.assertEqual(5, len(rows))
        signal_counts = {
            row["industry_mv_rank_top"]: row["signal_count"]
            for row in rows
        }
        self.assertEqual({1: 1, 2: 2, 3: 3, 4: 4, 5: 5}, signal_counts)

    def test_scan_industry_ranks_uses_inclusive_step(self) -> None:
        self.assertEqual(
            [1, 2, 3, 4, 5],
            PriceCodeLowStrategy._scan_industry_ranks(1, 5, 1),
        )
        self.assertEqual(
            [1, 3, 5],
            PriceCodeLowStrategy._scan_industry_ranks(1, 5, 2),
        )
        with self.assertRaises(RuntimeError):
            PriceCodeLowStrategy._scan_industry_ranks(5, 1, 1)

    def test_scan_thresholds_reject_invalid_range(self) -> None:
        with self.assertRaises(RuntimeError):
            PriceCodeLowStrategy._scan_thresholds(50, 20, 10)
        with self.assertRaises(RuntimeError):
            PriceCodeLowStrategy._scan_low_days(50, 10, 10)

    def test_backtest_summary_uses_signal_events(self) -> None:
        summary = PriceCodeLowStrategy._backtest_summary(
            [
                {
                    "ts_code": "000001.SZ",
                    "next_return_pct": -5,
                    "code_categories": "叠数",
                },
                {
                    "ts_code": "000001.SZ",
                    "next_return_pct": 2,
                    "code_categories": "叠数、重复片段",
                },
                {
                    "ts_code": "000002.SZ",
                    "next_return_pct": 1,
                    "code_categories": "重复片段",
                },
            ]
        )
        self.assertEqual(3, summary["signal_count"])
        self.assertEqual(2, summary["stock_count"])
        self.assertEqual(66.67, summary["win_rate"])
        self.assertEqual("重复片段", summary["best_category"]["category"])
        self.assertEqual(100.0, summary["best_category"]["win_rate"])
        self.assertEqual(2, len(summary["category_stats"]))


if __name__ == "__main__":
    unittest.main()
