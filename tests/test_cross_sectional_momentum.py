import unittest
from unittest.mock import patch

import pandas as pd

from quant_review.config import ReviewConfig
from quant_review.strategies.cross_sectional_momentum import CrossSectionalMomentumStrategy


class FakeCrossMomentumEngine:
    def __init__(self):
        self.trade_dates = [f"202608{day:02d}" for day in range(17, 32)]
        self.codes = ["000001.SZ", "000002.SZ", "600000.SH"]
        self.config = ReviewConfig(
            end_date=self.trade_dates[-1],
            lookback_days=20,
            cross_momentum_top_count=3,
            cross_momentum_low_threshold=-6,
            cross_momentum_recommend_count=1,
            cross_momentum_realtime_enabled=False,
        )
        self.logs = []
        self._last_realtime_snapshot = {}
        self.basic = pd.DataFrame(
            [
                {
                    "ts_code": code,
                    "symbol": code[:6],
                    "name": f"测试{code[:6]}",
                    "industry": "银行",
                    "market": "主板",
                    "stock_type": "深市主板" if code.startswith("000") else "沪市主板",
                    "list_date": "20200101",
                }
                for code in self.codes
            ]
        )
        daily_rows = []
        self.limits = {}
        for index, trade_date in enumerate(self.trade_dates):
            limit_codes = ["000001.SZ"] if trade_date == self.trade_dates[-4] else ["000002.SZ"]
            self.limits[trade_date] = pd.DataFrame({"ts_code": limit_codes, "limit_type": ["U"] * len(limit_codes)})
            for code in self.codes:
                close = 10 + index * 0.1 + (0 if code == "000001.SZ" else -index * 0.02)
                pre_close = close / 1.01
                low = close * 0.99
                pct = 1.0
                if trade_date == self.trade_dates[-3] and code == "000001.SZ":
                    pct = -1.0
                if trade_date == self.trade_dates[-2] and code == "000001.SZ":
                    low = pre_close * 0.90
                    pct = -2.0
                daily_rows.append(
                    {
                        "ts_code": code,
                        "trade_date": trade_date,
                        "open": close * 1.001,
                        "high": close * 1.01,
                        "low": low,
                        "close": close,
                        "pre_close": pre_close,
                        "pct_chg": pct,
                        "vol": 1000,
                        "amount": 100000,
                    }
                )
        self.daily = pd.DataFrame(daily_rows)
        self.daily_basic = pd.DataFrame(
            {
                "ts_code": self.codes,
                "turnover_rate": [2, 3, 4],
                "turnover_rate_f": [2, 3, 4],
                "volume_ratio": [1.2, 1.1, 1.0],
                "total_mv": [100000, 90000, 80000],
                "circ_mv": [80000, 70000, 60000],
            }
        )

    def log(self, message):
        self.logs.append(message)

    def _trade_dates(self):
        end_date = str(self.config.end_date)
        if end_date in self.trade_dates:
            return [date for date in self.trade_dates if date <= end_date]
        return self.trade_dates

    def _stock_basic(self):
        return self.basic

    def _daily_window(self, trade_dates):
        return self.daily

    def _daily_basic(self, trade_date):
        result = self.daily_basic.copy()
        if trade_date == self.trade_dates[-4]:
            result["turnover_rate_f"] = [2, 3, 4]
        elif trade_date == self.trade_dates[-3]:
            result["turnover_rate_f"] = [3, 3, 4]
        return result

    def _limit_list(self, trade_date):
        return self.limits.get(trade_date, pd.DataFrame())


class CrossSectionalMomentumStrategyTests(unittest.TestCase):
    def test_dynamic_offsets_and_recommendation(self):
        engine = FakeCrossMomentumEngine()
        result = CrossSectionalMomentumStrategy(engine).run()

        self.assertTrue(result["ok"])
        self.assertEqual(1, len(result["rows"]))
        row = result["rows"][0]
        self.assertEqual("000001", row["code"])
        self.assertGreater(row["open_price"], 0)
        self.assertGreater(row["close_price"], 0)
        self.assertEqual("https://stockpage.10jqka.com.cn/000001/", row["ths_url"])
        self.assertTrue(row["recommended"])
        self.assertEqual(1, len(result["recommended_rows"]))

    def test_realtime_is_off_by_default(self):
        engine = FakeCrossMomentumEngine()
        self.assertFalse(engine.config.cross_momentum_realtime_enabled)
        result = CrossSectionalMomentumStrategy(engine).run()
        self.assertEqual("日线收盘", result["rows"][0]["data_state"])

    def test_rows_include_t1_snapshot_and_backtest_export_fields(self):
        engine = FakeCrossMomentumEngine()
        result = CrossSectionalMomentumStrategy(engine).run()
        row = result["rows"][0]
        self.assertEqual(engine.trade_dates[-2], row["t1_trade_date"])
        self.assertGreater(row["t1_amount_yi"], 0)
        self.assertGreater(row["t1_turnover_rate"], 0)
        self.assertGreater(row["t1_market_cap_yi"], 0)
        self.assertGreater(row["t1_circ_mv_yi"], 0)

        engine.config.cross_momentum_backtest_days = 2
        backtest = CrossSectionalMomentumStrategy(engine).run_backtest()
        self.assertTrue(backtest["ok"])
        self.assertEqual(2, backtest["actual_days"])
        self.assertTrue(backtest["rows"])
        history = backtest["rows"][0]
        self.assertIn("tminus5_pct_chg", history)
        self.assertIn("tminus1_turnover_rate", history)
        self.assertIn("tminus1_turnover_rate_f", history)
        self.assertIn("tminus1_market_cap_yi", history)
        self.assertIn("tminus1_circ_mv_yi", history)

    def test_dynamic_low_condition_keeps_original_row_alignment(self):
        engine = FakeCrossMomentumEngine()
        engine.config.cross_momentum_conditions = [
            {
                "enabled": True,
                "object": "stock",
                "time_type": "offset",
                "offset": -1,
                "indicator": "low_pct",
                "operator": "lt",
                "threshold": -6,
            }
        ]
        strategy = CrossSectionalMomentumStrategy(engine)
        strategy._condition_diagnostics = []
        features = pd.DataFrame(
            {"ts_code": ["000001.SZ", "000560.SZ"], "close": [1.0, 3.15]},
            index=[4, 19],
        )
        daily = pd.DataFrame(
            [
                {"ts_code": "000001.SZ", "trade_date": "20260830", "low": 1.0, "pre_close": 1.0},
                {"ts_code": "000560.SZ", "trade_date": "20260830", "low": 2.86, "pre_close": 3.18},
            ]
        )
        result, mask = strategy._apply_dynamic_conditions(features, daily, ["20260830", "20260904"])
        self.assertAlmostEqual(-10.0629, result.loc[19, "condition_1_value"], places=3)
        self.assertTrue(mask.loc[19])

    def test_dynamic_sum_and_real_turnover_ratio_conditions(self):
        engine = FakeCrossMomentumEngine()
        engine.config.cross_momentum_conditions = [
            {"enabled": True, "object": "stock", "calculation": "sum", "range_start_offset": -5, "range_end_offset": -4, "indicator": "pct_chg", "operator": "le", "threshold": 3},
            {"enabled": True, "object": "stock", "calculation": "ratio", "offset": -2, "compare_offset": -3, "indicator": "turnover_rate_f", "operator": "ge", "threshold": 1.3},
        ]
        result = CrossSectionalMomentumStrategy(engine).run()
        self.assertEqual(1, len(result["rows"]))
        diagnostics = result["condition_diagnostics"]
        self.assertIn("累计", diagnostics[0]["condition"])
        self.assertIn("实换手率比值", diagnostics[1]["condition"])

    def test_concept_top_matches_returns_by_concept_code(self):
        engine = FakeCrossMomentumEngine()
        engine.config.cross_momentum_concept_top_count = 1
        strategy = CrossSectionalMomentumStrategy(engine)

        class FakeConceptRadar:
            def __init__(self, _engine):
                pass

            @staticmethod
            def _ths_concepts():
                return pd.DataFrame(
                    [
                        {"ts_code": "C1.TI", "name": "强势概念"},
                        {"ts_code": "C2.TI", "name": "弱势概念"},
                    ]
                )

            @staticmethod
            def _ths_daily_window(_trade_dates, _concept_codes):
                return pd.DataFrame(
                    [
                        {"ts_code": "C1.TI", "trade_date": "20260827", "close": 100},
                        {"ts_code": "C1.TI", "trade_date": "20260828", "close": 110},
                        {"ts_code": "C1.TI", "trade_date": "20260829", "close": 120},
                        {"ts_code": "C2.TI", "trade_date": "20260827", "close": 100},
                        {"ts_code": "C2.TI", "trade_date": "20260828", "close": 99},
                        {"ts_code": "C2.TI", "trade_date": "20260829", "close": 98},
                    ]
                )

            @staticmethod
            def _ths_members(_top):
                return pd.DataFrame(
                    [
                        {"ts_code": "000001.SZ", "concept_code": "C1.TI", "concept_name": "强势概念"},
                        {"ts_code": "000002.SZ", "concept_code": "C2.TI", "concept_name": "弱势概念"},
                    ]
                )

        features = pd.DataFrame(
            [
                {"ts_code": "000001.SZ", "close": 10.0},
                {"ts_code": "000002.SZ", "close": 10.0},
            ]
        )
        with patch("quant_review.strategies.cross_sectional_momentum.ConceptRadarStrategy", FakeConceptRadar):
            result, top = strategy._attach_concept_top(features, ["20260827", "20260828", "20260829"], 2)

        self.assertEqual("C1.TI", top[0]["concept_code"])
        self.assertAlmostEqual(20.0, top[0]["concept_return"])
        self.assertTrue(result.loc[result["ts_code"] == "000001.SZ", "concept_pass"].iloc[0])


if __name__ == "__main__":
    unittest.main()
