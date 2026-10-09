from __future__ import annotations

import math
from typing import Any

import pandas as pd

from quant_review.strategies.canslim_tao import CanslimTaoStrategy


class CanslimEarlyStrategy(CanslimTaoStrategy):
    """CANSLIM + RPS early-strength variant with stale-momentum filters.

    This strategy intentionally keeps the original CANSLIM + Tao/RPS engine
    intact, then adds a second layer for stage, price-staleness, and growth
    deceleration. It is meant to find earlier strong candidates, not mature
    high-RPS names that have already become crowded.
    """

    def run(self) -> dict[str, Any]:
        self.log("启动 CANSLIM+RPS 早期防钝化策略。")
        result = super().run()
        result["strategy"] = "canslim_early"
        return result

    def _price_features(self, daily: pd.DataFrame, trade_dates: list[str]) -> pd.DataFrame:
        features = super()._price_features(daily, trade_dates)
        close = daily.pivot_table(index="ts_code", columns="trade_date", values="close", aggfunc="last").reindex(columns=trade_dates)
        high = daily.pivot_table(index="ts_code", columns="trade_date", values="high", aggfunc="last").reindex(columns=trade_dates)
        low = daily.pivot_table(index="ts_code", columns="trade_date", values="low", aggfunc="last").reindex(columns=trade_dates)
        open_ = daily.pivot_table(index="ts_code", columns="trade_date", values="open", aggfunc="last").reindex(columns=trade_dates)
        amount = daily.pivot_table(index="ts_code", columns="trade_date", values="amount", aggfunc="last").reindex(columns=trade_dates)

        extra = pd.DataFrame(index=close.index)
        extra["ts_code"] = extra.index.astype(str)
        for window in [10, 20, 40, 60]:
            extra[f"ret{window}_early"] = self._window_return(close, window)
        current_features = features.set_index("ts_code")
        current_rps20 = extra["ts_code"].map(current_features["rps20"]) if "rps20" in current_features.columns else extra["ret20_early"].rank(pct=True) * 100
        extra["rps20_prev10"] = self._window_return(close, 20, end_offset=10).rank(pct=True) * 100
        extra["rps20_delta10"] = current_rps20 - extra["rps20_prev10"]
        current_rps50 = extra["ts_code"].map(current_features["rps50"]) if "rps50" in current_features.columns else pd.NA
        extra["rps50_prev20"] = self._window_return(close, 50, end_offset=20).rank(pct=True) * 100
        extra["rps50_delta20"] = current_rps50 - extra["rps50_prev20"]

        recent_high = high.iloc[:, -min(20, len(trade_dates)) :].max(axis=1)
        recent_low = low.iloc[:, -min(20, len(trade_dates)) :].min(axis=1)
        range20 = (recent_high / recent_low - 1).replace([math.inf, -math.inf], pd.NA) * 100
        extra["high_efficiency_20"] = (extra["ret20_early"] / range20.replace(0, pd.NA) * 100).clip(-100, 100)

        amount5 = amount.iloc[:, -min(5, len(trade_dates)) :].mean(axis=1)
        amount20 = amount.iloc[:, -min(20, len(trade_dates)) :].mean(axis=1)
        amount_expansion = (amount5 / amount20).replace([math.inf, -math.inf], pd.NA)
        extra["volume_price_efficiency"] = (extra["ret20_early"].clip(lower=0) / amount_expansion.replace(0, pd.NA)).fillna(0)

        take = min(10, len(trade_dates))
        recent_open = open_.iloc[:, -take:]
        recent_close = close.iloc[:, -take:]
        upper = high.iloc[:, -take:] - recent_open.where(recent_open > recent_close, recent_close)
        upper_ratio = upper / close.iloc[:, -take:].replace(0, pd.NA)
        extra["upper_shadow_days_10"] = (upper_ratio >= 0.035).sum(axis=1)

        return features.merge(extra.reset_index(drop=True), on="ts_code", how="left")

    def _window_return(self, close: pd.DataFrame, window: int, end_offset: int = 0) -> pd.Series:
        if close.empty or len(close.columns) <= end_offset:
            return pd.Series(pd.NA, index=close.index, dtype="float64")
        end_pos = len(close.columns) - 1 - end_offset
        base_pos = max(0, end_pos - window)
        end_price = close.iloc[:, end_pos]
        base_price = close.iloc[:, base_pos]
        return ((end_price / base_price - 1) * 100).replace([math.inf, -math.inf], pd.NA)

    def _fundamental_scores(self, trade_date: str) -> pd.DataFrame:
        result = super()._fundamental_scores(trade_date)
        if result.empty or "ts_code" not in result.columns:
            return result
        try:
            growth = self._growth_history(trade_date, set(result["ts_code"].astype(str)))
        except Exception as exc:
            self.log(f"增速趋势读取失败，按未知处理: {exc}")
            growth = pd.DataFrame()
        if growth.empty:
            result["growth_accel_score"] = 50.0
            result["growth_trend"] = "未知"
            result["growth_deceleration"] = False
            return result
        return result.merge(growth, on="ts_code", how="left").assign(
            growth_accel_score=lambda df: pd.to_numeric(df["growth_accel_score"], errors="coerce").fillna(50.0),
            growth_trend=lambda df: df["growth_trend"].fillna("未知"),
            growth_deceleration=lambda df: df["growth_deceleration"].fillna(False),
        )

    def _growth_history(self, trade_date: str, ts_codes: set[str]) -> pd.DataFrame:
        frames = []
        for period in self._report_periods(trade_date):
            frame = self.engine._fina_indicator(period)
            if frame.empty or "ts_code" not in frame.columns:
                continue
            use = frame.copy()
            use["ts_code"] = use["ts_code"].astype(str)
            use = use[use["ts_code"].isin(ts_codes)]
            if use.empty:
                continue
            if "end_date" not in use.columns:
                use["end_date"] = period
            use["end_date"] = use["end_date"].astype(str)
            for col in ["basic_eps_yoy", "dt_eps_yoy", "q_netprofit_yoy", "q_profit_yoy", "dt_netprofit_yoy", "netprofit_yoy", "q_sales_yoy", "or_yoy"]:
                if col not in use.columns:
                    use[col] = pd.NA
                use[col] = pd.to_numeric(use[col], errors="coerce")
            use["eps_growth"] = use["dt_eps_yoy"].combine_first(use["basic_eps_yoy"])
            use["profit_growth"] = use["q_netprofit_yoy"].combine_first(use["q_profit_yoy"]).combine_first(use["dt_netprofit_yoy"]).combine_first(use["netprofit_yoy"])
            use["sales_growth"] = use["q_sales_yoy"].combine_first(use["or_yoy"])
            frames.append(use[["ts_code", "end_date", "eps_growth", "profit_growth", "sales_growth"]])
        if not frames:
            return pd.DataFrame()

        data = pd.concat(frames, ignore_index=True).dropna(subset=["end_date"])
        data = data.sort_values(["ts_code", "end_date"])
        rows = []
        for ts_code, group in data.groupby("ts_code"):
            last_two = group.tail(2)
            latest = last_two.iloc[-1]
            prev = last_two.iloc[-2] if len(last_two) >= 2 else None
            deltas = []
            decelerated = False
            for col in ["eps_growth", "profit_growth", "sales_growth"]:
                latest_value = self._num(latest.get(col), None)
                prev_value = self._num(prev.get(col), None) if prev is not None else None
                if latest_value is None or prev_value is None:
                    continue
                delta = latest_value - prev_value
                deltas.append(delta)
                if prev_value >= 30 and delta <= -20:
                    decelerated = True
            if not deltas:
                score = 50.0
                trend = "未知"
            else:
                avg_delta = sum(deltas) / len(deltas)
                score = self._growth_accel_score(avg_delta)
                if avg_delta >= 8:
                    trend = "加速"
                elif avg_delta >= -8:
                    trend = "稳定"
                elif avg_delta >= -20:
                    trend = "小幅降速"
                else:
                    trend = "明显降速"
            rows.append(
                {
                    "ts_code": ts_code,
                    "growth_accel_score": score,
                    "growth_trend": trend,
                    "growth_deceleration": decelerated,
                }
            )
        return pd.DataFrame(rows)

    @staticmethod
    def _growth_accel_score(delta: float) -> float:
        if delta >= 15:
            return 100.0
        if delta >= 0:
            return 70.0 + delta / 15 * 30
        if delta >= -20:
            return 45.0 + (delta + 20) / 20 * 25
        return max(0.0, 45.0 + (delta + 20) * 1.5)

    def _add_scores(self, data: pd.DataFrame, market: dict[str, Any]) -> pd.DataFrame:
        data = super()._add_scores(data, market)
        data["pre_early_score"] = data["score"]
        data["stale_score"] = data.apply(self._stale_score, axis=1)
        data["freshness_score"] = data.apply(self._freshness_score, axis=1)
        if "growth_accel_score" not in data.columns:
            data["growth_accel_score"] = 50.0
        data["score"] = (
            data["pre_early_score"] * 0.72
            + data["freshness_score"] * 0.22
            + pd.to_numeric(data["growth_accel_score"], errors="coerce").fillna(50.0) * 0.10
            - data["stale_score"] * 0.22
        ).clip(0, 100)
        data["stage"] = data.apply(self._stage_label, axis=1)
        data["stale_label"] = data["stale_score"].apply(self._stale_label)
        data["stale_reason"] = data.apply(self._stale_reason, axis=1)
        return data

    def _apply_channel_mode(self, data: pd.DataFrame) -> pd.DataFrame:
        result = super()._apply_channel_mode(data)
        if result.empty:
            return result
        return self._apply_early_filters(result)

    def _apply_early_filters(self, data: pd.DataFrame) -> pd.DataFrame:
        result = data.copy()
        result = result[pd.to_numeric(result["stale_score"], errors="coerce").fillna(100) <= float(getattr(self.config, "canslim_early_max_stale_score", 58))]
        result = result[pd.to_numeric(result["rps20"], errors="coerce").fillna(0) >= float(getattr(self.config, "canslim_early_min_rps20", 68))]
        result = result[pd.to_numeric(result["ret20_early"], errors="coerce").fillna(999) <= float(getattr(self.config, "canslim_early_max_ret20", 32))]
        result = result[pd.to_numeric(result["ret60_early"], errors="coerce").fillna(999) <= float(getattr(self.config, "canslim_early_max_ret60", 80))]
        result = result[pd.to_numeric(result["distance_ma20"], errors="coerce").fillna(999) <= float(getattr(self.config, "canslim_early_max_distance_ma20", 18))]
        result = result[pd.to_numeric(result["distance_ma50"], errors="coerce").fillna(999) <= float(getattr(self.config, "canslim_early_max_distance_ma50", 38))]
        result = result[pd.to_numeric(result["upper_shadow_days_10"], errors="coerce").fillna(99) <= int(getattr(self.config, "canslim_early_max_upper_shadow_days10", 4))]
        min_growth = float(getattr(self.config, "canslim_early_min_growth_accel_score", 35))
        result = result[pd.to_numeric(result.get("growth_accel_score", 50), errors="coerce").fillna(50) >= min_growth]
        return result

    def _stale_score(self, row: pd.Series) -> float:
        score = 0.0
        if self._num(row.get("ret20_early"), 0) > float(getattr(self.config, "canslim_early_max_ret20", 32)):
            score += 24
        if self._num(row.get("ret60_early"), 0) > float(getattr(self.config, "canslim_early_max_ret60", 80)):
            score += 18
        if self._num(row.get("distance_ma20"), 0) > float(getattr(self.config, "canslim_early_max_distance_ma20", 18)):
            score += 22
        if self._num(row.get("distance_ma50"), 0) > float(getattr(self.config, "canslim_early_max_distance_ma50", 38)):
            score += 12
        if self._num(row.get("rps20_delta10"), 0) < -5:
            score += 14
        extra_shadow = self._num(row.get("upper_shadow_days_10"), 0) - int(getattr(self.config, "canslim_early_max_upper_shadow_days10", 4))
        if extra_shadow > 0:
            score += min(16.0, extra_shadow * 5)
        if self._num(row.get("volume_price_efficiency"), 0) < 5 and self._num(row.get("amount_ratio"), 1) >= 1.5:
            score += 12
        if bool(row.get("growth_deceleration", False)):
            score += 16
        return min(100.0, score)

    def _freshness_score(self, row: pd.Series) -> float:
        score = 35.0
        score += self._clip(self._num(row.get("rps20"), 0), 60, 92) * 0.22
        score += self._clip(self._num(row.get("rps20_delta10"), 0), -5, 25) * 0.24
        score += self._clip(self._num(row.get("high_efficiency_20"), 0), 20, 85) * 0.16
        score += self._clip(18 - abs(self._num(row.get("distance_ma20"), 0)), 0, 18) * 0.16
        if bool(row.get("breakout_20")):
            score += 8
        if bool(row.get("pocket_pivot")):
            score += 8
        if bool(row.get("new_high_5")) and self._num(row.get("ret20_early"), 0) <= float(getattr(self.config, "canslim_early_max_ret20", 32)):
            score += 6
        return min(100.0, score)

    def _stage_label(self, row: pd.Series) -> str:
        stale = self._num(row.get("stale_score"), 0)
        if stale >= 65:
            return "高位钝化"
        if self._num(row.get("rps20_delta10"), 0) >= float(getattr(self.config, "canslim_early_min_rps20_delta10", 8)) and self._num(row.get("distance_ma20"), 0) <= 12:
            return "早期加速"
        if bool(row.get("breakout_20")) or bool(row.get("pocket_pivot")):
            return "启动确认"
        if self._num(row.get("high_position_250"), 0) >= 0.9:
            return "主升观察"
        return "观察"

    @staticmethod
    def _stale_label(value: Any) -> str:
        try:
            score = float(value)
        except (TypeError, ValueError):
            return "未知"
        if score >= 65:
            return "明显"
        if score >= 40:
            return "轻微"
        return "否"

    def _stale_reason(self, row: pd.Series) -> str:
        parts = []
        if self._num(row.get("ret20_early"), 0) > float(getattr(self.config, "canslim_early_max_ret20", 32)):
            parts.append("20日涨幅过大")
        if self._num(row.get("distance_ma20"), 0) > float(getattr(self.config, "canslim_early_max_distance_ma20", 18)):
            parts.append("远离20日线")
        if self._num(row.get("rps20_delta10"), 0) < -5:
            parts.append("短期RPS回落")
        if self._num(row.get("upper_shadow_days_10"), 0) > int(getattr(self.config, "canslim_early_max_upper_shadow_days10", 4)):
            parts.append("上影偏多")
        if bool(row.get("growth_deceleration", False)):
            parts.append("增速降档")
        return "；".join(parts[:4]) or "未见明显钝化"

    def _probe_checks(self, row: pd.Series) -> list[dict[str, Any]]:
        checks = super()._probe_checks(row)
        checks.extend(
            [
                self._check("早期RPS20", row.get("rps20"), ">=", float(getattr(self.config, "canslim_early_min_rps20", 68)), self._num(row.get("rps20"), 0) >= float(getattr(self.config, "canslim_early_min_rps20", 68))),
                self._check("20日涨幅不过热", row.get("ret20_early"), "<=", float(getattr(self.config, "canslim_early_max_ret20", 32)), self._num(row.get("ret20_early"), 999) <= float(getattr(self.config, "canslim_early_max_ret20", 32)), "%"),
                self._check("距20日线不过远", row.get("distance_ma20"), "<=", float(getattr(self.config, "canslim_early_max_distance_ma20", 18)), self._num(row.get("distance_ma20"), 999) <= float(getattr(self.config, "canslim_early_max_distance_ma20", 18)), "%"),
                self._check("钝化分", row.get("stale_score"), "<=", float(getattr(self.config, "canslim_early_max_stale_score", 58)), self._num(row.get("stale_score"), 100) <= float(getattr(self.config, "canslim_early_max_stale_score", 58))),
                self._check("增速趋势", row.get("growth_trend", "未知"), "avoid", "明显降速", str(row.get("growth_trend", "")) != "明显降速"),
            ]
        )
        return checks

    def _to_row(self, row: pd.Series) -> dict[str, Any]:
        result = super()._to_row(row)
        result.update(
            {
                "stage": row.get("stage", ""),
                "stale_label": row.get("stale_label", ""),
                "stale_score": round(self._num(row.get("stale_score"), 0), 1),
                "stale_reason": row.get("stale_reason", ""),
                "growth_trend": row.get("growth_trend", ""),
                "growth_accel_score": round(self._num(row.get("growth_accel_score"), 50), 1),
                "base_score": round(self._num(row.get("base_score"), 0), 2),
                "freshness_score": round(self._num(row.get("freshness_score"), 0), 1),
                "rps20": round(self._num(row.get("rps20"), 0), 1),
                "rps20_delta10": round(self._num(row.get("rps20_delta10"), 0), 1),
                "ret20_early": round(self._num(row.get("ret20_early"), 0), 2),
                "ret60_early": round(self._num(row.get("ret60_early"), 0), 2),
                "high_efficiency_20": round(self._num(row.get("high_efficiency_20"), 0), 1),
                "volume_price_efficiency": round(self._num(row.get("volume_price_efficiency"), 0), 1),
                "upper_shadow_days_10": int(self._num(row.get("upper_shadow_days_10"), 0)),
            }
        )
        return result
