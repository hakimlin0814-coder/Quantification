from __future__ import annotations

import unittest

from quant_review.strategies.realtime_ma import RealtimeMaStrategy


class RealtimeMaPullbackTests(unittest.TestCase):
    def test_ma10_pullback_requires_price_cross_and_ma5_above_ma10(self) -> None:
        self.assertTrue(RealtimeMaStrategy._pullback_ma10_ok(9.8, 10.2, 10.1, 10.0))
        self.assertFalse(RealtimeMaStrategy._pullback_ma10_ok(10.1, 10.2, 10.3, 10.0))
        self.assertFalse(RealtimeMaStrategy._pullback_ma10_ok(9.8, 10.2, 9.9, 10.0))

    def test_ma20_pullback_requires_price_cross_ma20_and_ma10_above(self) -> None:
        self.assertTrue(RealtimeMaStrategy._pullback_ma20_ok(9.8, 10.2, 10.0, 9.9))
        self.assertFalse(RealtimeMaStrategy._pullback_ma20_ok(9.95, 10.2, 10.0, 9.9))
        self.assertFalse(RealtimeMaStrategy._pullback_ma20_ok(9.8, 10.2, 9.8, 9.9))


if __name__ == "__main__":
    unittest.main()
