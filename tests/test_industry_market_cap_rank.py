from __future__ import annotations

import unittest

import pandas as pd

from quant_review.strategies.industry_market_cap_rank import (
    rank_industry_market_caps,
    rank_stock_market_caps,
    rank_stock_prices,
)


class IndustryMarketCapRankTests(unittest.TestCase):
    def test_ranks_industries_by_total_market_cap(self) -> None:
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
                {"ts_code": "000001.SZ", "rank_industry": "银行", "rank_industry_code": "801780.SI"},
                {"ts_code": "000002.SZ", "rank_industry": "房地产", "rank_industry_code": "801180.SI"},
                {"ts_code": "000003.SZ", "rank_industry": "银行", "rank_industry_code": "801780.SI"},
                {"ts_code": "600001.SH", "rank_industry": "房地产", "rank_industry_code": "801180.SI"},
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

        ranked = rank_industry_market_caps(basic, daily_basic, members)

        first = ranked.iloc[0]
        second = ranked.iloc[1]
        self.assertEqual(1, int(first["rank"]))
        self.assertEqual("房地产", first["rank_industry"])
        self.assertEqual(350.0, float(first["total_market_cap_yi"]))
        self.assertEqual(2, int(first["member_count"]))
        self.assertEqual(2, int(first["priced_member_count"]))
        self.assertEqual(2, int(second["rank"]))
        self.assertEqual("银行", second["rank_industry"])
        self.assertEqual(300.0, float(second["total_market_cap_yi"]))

    def test_ignores_zero_or_missing_market_cap_in_totals(self) -> None:
        basic = pd.DataFrame(
            [
                {"ts_code": "000001.SZ"},
                {"ts_code": "000002.SZ"},
                {"ts_code": "000003.SZ"},
            ]
        )
        members = pd.DataFrame(
            [
                {"ts_code": "000001.SZ", "rank_industry": "行业A", "rank_industry_code": "A"},
                {"ts_code": "000002.SZ", "rank_industry": "行业A", "rank_industry_code": "A"},
                {"ts_code": "000003.SZ", "rank_industry": "行业B", "rank_industry_code": "B"},
            ]
        )
        daily_basic = pd.DataFrame(
            [
                {"ts_code": "000001.SZ", "total_mv": 1200000},
                {"ts_code": "000002.SZ", "total_mv": 0},
            ]
        )

        ranked = rank_industry_market_caps(basic, daily_basic, members)

        self.assertEqual(1, len(ranked))
        row = ranked.iloc[0]
        self.assertEqual("行业A", row["rank_industry"])
        self.assertEqual(120.0, float(row["total_market_cap_yi"]))
        self.assertEqual(2, int(row["member_count"]))
        self.assertEqual(1, int(row["priced_member_count"]))

    def test_ranks_stocks_by_total_market_cap(self) -> None:
        basic = pd.DataFrame(
            [
                {"ts_code": "000001.SZ", "symbol": "000001", "name": "平安银行", "industry": "银行", "stock_type": "深市主板"},
                {"ts_code": "000002.SZ", "symbol": "000002", "name": "万科A", "industry": "房地产", "stock_type": "深市主板"},
                {"ts_code": "000003.SZ", "symbol": "000003", "name": "零市值", "industry": "其他", "stock_type": "深市主板"},
            ]
        )
        daily_basic = pd.DataFrame(
            [
                {"ts_code": "000001.SZ", "total_mv": 1000000, "close": 10.5, "pct_chg": 1.2, "trade_date": "20260706"},
                {"ts_code": "000002.SZ", "total_mv": 3000000, "close": 20.1, "pct_chg": -0.5, "trade_date": "20260706"},
                {"ts_code": "000003.SZ", "total_mv": 0, "close": 3.2, "pct_chg": 0, "trade_date": "20260706"},
            ]
        )

        ranked = rank_stock_market_caps(basic, daily_basic)

        self.assertEqual(2, len(ranked))
        self.assertEqual("000002", ranked.iloc[0]["code"])
        self.assertEqual(300.0, float(ranked.iloc[0]["total_market_cap_yi"]))
        self.assertEqual("000001", ranked.iloc[1]["code"])

    def test_ranks_stocks_by_price(self) -> None:
        basic = pd.DataFrame(
            [
                {"ts_code": "000001.SZ", "symbol": "000001", "name": "低价股", "industry": "银行", "stock_type": "深市主板"},
                {"ts_code": "000002.SZ", "symbol": "000002", "name": "高价股", "industry": "消费", "stock_type": "深市主板"},
            ]
        )
        price_data = pd.DataFrame(
            [
                {"ts_code": "000001.SZ", "close": 12.3, "pct_chg": 1.1, "trade_date": "20260706"},
                {"ts_code": "000002.SZ", "close": 88.8, "pct_chg": -2.0, "trade_date": "20260706"},
            ]
        )
        daily_basic = pd.DataFrame(
            [
                {"ts_code": "000001.SZ", "total_mv": 1000000},
                {"ts_code": "000002.SZ", "total_mv": 2000000},
            ]
        )

        ranked = rank_stock_prices(basic, price_data, daily_basic)

        self.assertEqual(2, len(ranked))
        self.assertEqual("000002", ranked.iloc[0]["code"])
        self.assertEqual(88.8, float(ranked.iloc[0]["close"]))
        self.assertEqual(200.0, float(ranked.iloc[0]["total_market_cap_yi"]))


if __name__ == "__main__":
    unittest.main()
