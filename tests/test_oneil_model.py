from __future__ import annotations

import unittest
import json
import threading
from http.server import ThreadingHTTPServer
from urllib.request import Request, urlopen

import pandas as pd

from src.oneil_model.backtest import OneilBacktestEngine
from src.oneil_model.config import OneilConfigurationError, OneilResearchConfig
from src.oneil_model.data import OneilDataGateway
from src.oneil_model.execution import enforce_t_plus_one, tradability_preview
from src.oneil_model.factors import (
    compute_financial_factors,
    compute_price_factors,
    filter_financials_as_of,
)
from src.oneil_model.signals import evaluate_q0_rows
from quant_review.server import ReviewHandler


class OneilConfigTests(unittest.TestCase):
    def test_baseline_exposes_rule_provenance(self) -> None:
        config = OneilResearchConfig()
        specs = {item.rule_id: item for item in config.rule_specs()}
        self.assertEqual("O", specs["C_EPS_YOY"].source.value)
        self.assertEqual("P", specs["N_NEAR_HIGH"].source.value)
        self.assertEqual("A", specs["UNIVERSE_ST"].source.value)
        self.assertEqual("U", specs["PATTERN"].source.value)
        self.assertEqual(16, len(config.config_hash))

    def test_unconfirmed_modules_cannot_be_silently_enabled(self) -> None:
        with self.assertRaises(OneilConfigurationError):
            OneilResearchConfig(pattern_mode="cup_handle").validate()
        with self.assertRaises(OneilConfigurationError):
            OneilResearchConfig(market_gate_mode="entry_filter").validate()

    def test_backtest_is_explicitly_blocked(self) -> None:
        with self.assertRaises(OneilConfigurationError):
            OneilBacktestEngine(OneilResearchConfig()).run()


class PointInTimeFinancialTests(unittest.TestCase):
    def setUp(self) -> None:
        self.rows = pd.DataFrame(
            [
                {"ts_code": "000001.SZ", "end_date": "20241231", "ann_date": "20250301", "eps": 1.0},
                {"ts_code": "000001.SZ", "end_date": "20241231", "ann_date": "20250501", "eps": 2.0},
                {"ts_code": "000001.SZ", "end_date": "20250331", "ann_date": "20250420", "q_eps": 0.4, "basic_eps_yoy": 30, "q_sales_yoy": 28},
                {"ts_code": "000002.SZ", "end_date": "20250331", "ann_date": "", "q_eps": 0.5, "basic_eps_yoy": 40, "q_sales_yoy": 35},
            ]
        )

    def test_future_revision_is_not_backfilled(self) -> None:
        visible = filter_financials_as_of(self.rows, "20250430")
        annual = visible[(visible["ts_code"] == "000001.SZ") & (visible["end_date"] == "20241231")]
        self.assertEqual(1.0, float(annual.iloc[0]["eps"]))

    def test_missing_announcement_date_is_excluded(self) -> None:
        visible = filter_financials_as_of(self.rows, "20250430")
        self.assertNotIn("000002.SZ", visible["ts_code"].tolist())

    def test_akshare_revision_risk_cannot_be_financial_complete(self) -> None:
        annual = []
        for year, eps in [(2021, 1.0), (2022, 1.3), (2023, 1.7), (2024, 2.2)]:
            annual.append(
                {
                    "ts_code": "000001.SZ",
                    "end_date": f"{year}1231",
                    "ann_date": f"{year + 1}0301",
                    "eps": eps,
                    "roe": 20,
                    "data_source": "AkShare",
                    "point_in_time_risk": True,
                }
            )
        annual.append(
            {
                "ts_code": "000001.SZ",
                "end_date": "20250331",
                "ann_date": "20250420",
                "q_eps": 0.5,
                "basic_eps_yoy": 35,
                "q_sales_yoy": 30,
                "data_source": "AkShare",
                "point_in_time_risk": True,
            }
        )
        result = compute_financial_factors(pd.DataFrame(annual), "20250430", OneilResearchConfig())
        self.assertFalse(bool(result.iloc[0]["financial_complete"]))
        self.assertTrue(bool(result.iloc[0]["financial_point_in_time_risk"]))


class PriceFactorTests(unittest.TestCase):
    def test_price_factors_use_prior_volume_and_cross_sectional_rank(self) -> None:
        rows = []
        dates = [f"2024{index + 1:04d}" for index in range(260)]
        for stock_index, code in enumerate(["000001.SZ", "000002.SZ", "000003.SZ"]):
            growth = [0.25, 0.15, 0.05][stock_index]
            for day, trade_date in enumerate(dates):
                pct = growth
                close = 10 * ((1 + growth / 100) ** day)
                rows.append(
                    {
                        "ts_code": code,
                        "trade_date": trade_date,
                        "open": close,
                        "high": close,
                        "low": close,
                        "close": close,
                        "pre_close": close / (1 + growth / 100),
                        "pct_chg": growth,
                        "vol": 200 if day == 259 and code == "000001.SZ" else 100,
                        "amount": 10000,
                    }
                )
        result = compute_price_factors(pd.DataFrame(rows), OneilResearchConfig()).set_index("ts_code")
        leader = result.loc["000001.SZ"]
        self.assertAlmostEqual(2.0, float(leader["breakout_rvol"]), places=5)
        self.assertTrue(bool(leader["new_high_breakout"]))
        self.assertEqual(100.0, float(leader["rs_percentile"]))
        self.assertTrue(bool(leader["price_complete"]))


class SignalAndExecutionTests(unittest.TestCase):
    def complete_row(self) -> dict[str, object]:
        return {
            "ts_code": "000001.SZ",
            "eps_yoy_q": 30.0,
            "sales_yoy_q": 30.0,
            "annual_eps_up": True,
            "annual_eps_cagr": 30.0,
            "annual_roe": 20.0,
            "near_high_ratio": 0.95,
            "new_high_breakout": True,
            "breakout_rvol": 1.6,
            "rs_percentile": 90.0,
            "financial_complete": True,
            "price_complete": True,
            "universe_point_in_time_risk": False,
        }

    def test_q0_pass_is_not_complete_oneil_signal(self) -> None:
        result = evaluate_q0_rows(pd.DataFrame([self.complete_row()]), OneilResearchConfig()).iloc[0]
        self.assertEqual("PASS", result["status"])
        self.assertTrue(bool(result["q0_observation_pass"]))
        self.assertEqual("INCOMPLETE", result["complete_oneil_status"])

    def test_missing_value_never_passes(self) -> None:
        row = self.complete_row()
        row["annual_roe"] = None
        result = evaluate_q0_rows(pd.DataFrame([row]), OneilResearchConfig()).iloc[0]
        self.assertEqual("INCOMPLETE", result["status"])
        self.assertFalse(bool(result["q0_observation_pass"]))

    def test_one_price_limit_up_is_not_buyable(self) -> None:
        row = pd.Series({"close": 11, "high": 11, "low": 11, "suspended_latest": False})
        limit_row = pd.Series({"limit": "U"})
        result = tradability_preview(row, limit_row)
        self.assertEqual("ONE_PRICE_LIMIT_UP", result["status"])
        self.assertFalse(result["buyable"])

    def test_t_plus_one(self) -> None:
        self.assertFalse(enforce_t_plus_one("20250102", "20250102"))
        self.assertTrue(enforce_t_plus_one("20250102", "20250103"))


class AkshareNormalizerTests(unittest.TestCase):
    def test_akshare_financial_is_marked_revision_risk(self) -> None:
        frame = pd.DataFrame(
            [
                {
                    "股票代码": "1",
                    "每股收益": 1.2,
                    "营业总收入-同比增长": 30,
                    "净资产收益率": 20,
                    "最新公告日期": "2025-04-20",
                }
            ]
        )
        result = OneilDataGateway._normalize_akshare_financial(frame, "20250331")
        self.assertEqual("000001.SZ", result.iloc[0]["ts_code"])
        self.assertTrue(bool(result.iloc[0]["point_in_time_risk"]))


class OneilHttpRouteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), ReviewHandler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.base = f"http://127.0.0.1:{cls.server.server_address[1]}"

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=2)

    def test_page_route_exists(self) -> None:
        with urlopen(f"{self.base}/oneil-model", timeout=5) as response:
            body = response.read().decode("utf-8")
        self.assertIn("欧奈尔模型", body)
        self.assertIn("Q0研究观察池", body)

    def test_config_validation_endpoint_exposes_sources(self) -> None:
        request = Request(
            f"{self.base}/api/oneil_model/config/validate",
            data=json.dumps({"oneil_research_level": "Q0"}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request, timeout=5) as response:
            data = json.loads(response.read().decode("utf-8"))
        self.assertTrue(data["ok"])
        sources = {item["source"] for item in data["config"]["rules"]}
        self.assertEqual({"O", "P", "A", "U"}, sources)


if __name__ == "__main__":
    unittest.main()
