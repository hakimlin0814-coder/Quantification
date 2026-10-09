from __future__ import annotations

import unittest
from types import SimpleNamespace

import pandas as pd

from quant_review.config import ReviewConfig
from quant_review.strategies.canslim_tao import CanslimTaoStrategy
from quant_review.strategies.industry_mainline import (
    MAINLINE_CONFIRMED,
    MAINLINE_NONE,
    score_industry_mainlines,
)


class IndustryMainlineTests(unittest.TestCase):
    def test_review_config_defaults_to_sw_level_2(self) -> None:
        self.assertEqual(ReviewConfig().canslim_mainline_industry_level, "sw_l2")

    def setUp(self) -> None:
        self.trade_dates = [f"D{index:03d}" for index in range(131)]
        basic_rows = []
        member_rows = []
        daily_rows = []
        industries = [
            ("TOP", "801010.SI", 1.009),
            ("MID", "801020.SI", 1.001),
            ("LOW", "801030.SI", 0.996),
        ]
        serial = 0
        for industry, industry_code, growth in industries:
            for member in range(5):
                serial += 1
                ts_code = f"{serial:06d}.SZ"
                basic_rows.append({"ts_code": ts_code, "industry": f"基础{industry}"})
                member_rows.append(
                    {
                        "ts_code": ts_code,
                        "rank_industry": industry,
                        "rank_industry_code": industry_code,
                    }
                )
                for day, trade_date in enumerate(self.trade_dates):
                    close = (10 + member * 0.05) * (growth**day)
                    daily_rows.append(
                        {
                            "ts_code": ts_code,
                            "trade_date": trade_date,
                            "close": close,
                        }
                    )
        self.basic = pd.DataFrame(basic_rows)
        self.members = pd.DataFrame(member_rows)
        self.daily = pd.DataFrame(daily_rows)

    def test_scores_strong_persistent_industry_as_confirmed(self) -> None:
        industries, stocks = score_industry_mainlines(
            self.basic,
            self.daily,
            self.trade_dates,
            self.members,
        )

        top = industries.set_index("mainline_industry").loc["TOP"]
        low = industries.set_index("mainline_industry").loc["LOW"]
        self.assertEqual(top["mainline_state"], MAINLINE_CONFIRMED)
        self.assertEqual(low["mainline_state"], MAINLINE_NONE)
        self.assertEqual(int(top["industry_rank"]), 1)
        self.assertGreater(float(top["industry_rps_combo"]), float(low["industry_rps_combo"]))
        self.assertGreater(float(top["industry_breadth_score"]), float(low["industry_breadth_score"]))
        self.assertEqual(len(stocks), 15)
        self.assertTrue((stocks["mainline_industry_source"] == "申万行业").all())

    def test_falls_back_to_basic_industry_mapping(self) -> None:
        industries, stocks = score_industry_mainlines(
            self.basic,
            self.daily,
            self.trade_dates,
            industry_members=pd.DataFrame(),
        )

        self.assertFalse(industries.empty)
        self.assertTrue((stocks["mainline_industry_source"] == "基础行业").all())
        self.assertIn("基础TOP", set(stocks["mainline_industry"]))


class CanslimMainlineFilterTests(unittest.TestCase):
    def _strategy(self, mode: str, minimum: float = 0) -> CanslimTaoStrategy:
        config = SimpleNamespace(
            canslim_mainline_enabled=True,
            canslim_mainline_filter=mode,
            canslim_mainline_min_score=minimum,
        )
        engine = SimpleNamespace(config=config)
        return CanslimTaoStrategy(engine)

    def test_candidate_filter_keeps_candidate_and_confirmed(self) -> None:
        data = pd.DataFrame(
            {
                "mainline_state": ["主线确认", "主线候选", "非主线"],
                "mainline_score": [82, 67, 48],
            }
        )
        result = self._strategy("candidate")._apply_mainline_filter(data)
        self.assertEqual(result["mainline_state"].tolist(), ["主线确认", "主线候选"])

    def test_minimum_score_applies_without_state_filter(self) -> None:
        data = pd.DataFrame(
            {
                "mainline_state": ["主线确认", "主线候选", "非主线"],
                "mainline_score": [82, 67, 48],
            }
        )
        result = self._strategy("all", minimum=65)._apply_mainline_filter(data)
        self.assertEqual(result["mainline_score"].tolist(), [82, 67])


class CanslimMainlineWeightTests(unittest.TestCase):
    class ScoreStrategy(CanslimTaoStrategy):
        def _new_factor_score(self, row: pd.Series) -> float:
            return 80.0

        def _supply_demand_score(self, row: pd.Series) -> float:
            return 80.0

        def _sponsorship_proxy_score(self, row: pd.Series) -> float:
            return 80.0

        def _risk_penalty(self, row: pd.Series) -> float:
            return 0.0

        def _setup_label(self, row: pd.Series) -> str:
            return "测试"

        def _tao_pass(self, row: pd.Series) -> bool:
            return False

        def _canslim_pass(self, row: pd.Series) -> bool:
            return False

    def test_final_score_blends_base_and_mainline_score(self) -> None:
        config = SimpleNamespace(
            canslim_mainline_enabled=True,
            canslim_mainline_weight=0.25,
        )
        strategy = self.ScoreStrategy(SimpleNamespace(config=config))
        data = pd.DataFrame(
            {
                "c_score": [80.0],
                "a_score": [80.0],
                "rps_combo": [80.0],
                "mainline_score": [100.0],
            }
        )

        result = strategy._add_scores(data, {"score": 80.0})
        self.assertAlmostEqual(float(result.iloc[0]["base_score"]), 80.0)
        self.assertAlmostEqual(float(result.iloc[0]["score"]), 85.0)


if __name__ == "__main__":
    unittest.main()
