import unittest

import pandas as pd

from quant_review.engine import TushareReviewEngine


class TradeDateShiftTests(unittest.TestCase):
    def setUp(self):
        self.engine = object.__new__(TushareReviewEngine)
        calendar = pd.DataFrame(
            {
                "cal_date": [
                    "20260903",
                    "20260904",
                    "20260907",
                    "20260908",
                ]
            }
        )
        self.engine._cached = lambda _name, _fetcher: calendar

    def test_previous_trade_date_skips_weekend(self):
        self.assertEqual("20260904", self.engine.shift_trade_date("20260906", -1))

    def test_next_trade_date_skips_weekend(self):
        self.assertEqual("20260907", self.engine.shift_trade_date("20260904", 1))


if __name__ == "__main__":
    unittest.main()
