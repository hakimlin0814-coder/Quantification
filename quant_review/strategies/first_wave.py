from __future__ import annotations

import math
from typing import Any

import pandas as pd


class FirstWaveStrategy:
    """First-wave launch strategy for A-share observation.

    The strategy deliberately avoids mature high-RPS names. It looks for
    compressed bases, the first clear volume breakout, and the first controlled
    pullback after that breakout. Rows are observation candidates, not automatic
    buy instructions.
    """

    MAIN_BOARD_PREFIXES = ("000", "001", "002", "003", "600", "601", "603", "605")
    STAR_PREFIXES = ("688", "689")
    CHINEXT_PREFIXES = ("300", "301")
    BJ_PREFIXES = ("8", "4", "92")

    def __init__(self, engine: Any) -> None:
        self.engine = engine
        self.config = engine.config

    def log(self, message: str) -> None:
        self.engine.log(message)

    def run(self) -> dict[str, Any]:
        old_lookback = int(self.config.lookback_days)
        self.config.lookback_days = max(old_lookback, 170)
        try:
            self.log("读取交易日历，第一波策略需要约170个交易日用于平台和RPS判断...")
            trade_dates = self.engine._trade_dates()
        finally:
            self.config.lookback_days = old_lookback

        trade_date = trade_dates[-1]
        self.config.end_date = trade_date
        self.log(f"使用交易日: {trade_date}")

        self.log("读取股票基础信息...")
        basic = self.engine._stock_basic()

        self.log("读取日线行情并构建复权近似序列...")
        daily = self.engine._daily_window(trade_dates)
        intraday_note = ""
        if self.config.review_mode == "intraday":
            self.log("盘中模式：读取实时行情并覆盖当天临时日K...")
            daily = self.engine._apply_intraday_snapshot(daily, basic, trade_date)
            intraday_note = "盘中临时日K会随行情变化，只适合作为试探信号。"

        self.log("读取每日指标...")
        daily_basic = self.engine._daily_basic(trade_date)
        if self.config.review_mode == "intraday" and self.engine._daily_basic_is_empty(daily_basic) and len(trade_dates) >= 2:
            prev_date = trade_dates[-2]
            self.log(f"盘中模式：当日每日指标未更新，回退到 {prev_date}。")
            daily_basic = self.engine._daily_basic(prev_date)

        self.log("计算第一波启动特征...")
        features = self._features(daily, basic, daily_basic, trade_dates)

        self.log("识别启动日和回踩确认候选...")
        rows = self._rows(features, trade_dates)
        market = self._market_snapshot(features, trade_dates[-1], rows, intraday_note)

        self.log(f"第一波启动策略完成，共 {len(rows)} 只候选。")
        return {
            "ok": True,
            "trade_date": trade_date,
            "market": market,
            "rows": rows,
            "logs": self.engine.logs,
        }

    def _features(
        self,
        daily: pd.DataFrame,
        basic: pd.DataFrame,
        daily_basic: pd.DataFrame,
        trade_dates: list[str],
    ) -> pd.DataFrame:
        data = daily.copy()
        data["ts_code"] = data["ts_code"].astype(str)
        data["trade_date"] = data["trade_date"].astype(str)
        for col in ["open", "high", "low", "close", "pre_close", "pct_chg", "vol", "amount"]:
            if col not in data.columns:
                data[col] = 0
            data[col] = pd.to_numeric(data[col], errors="coerce").fillna(0)
        data = data[data["close"] > 0].sort_values(["ts_code", "trade_date"]).reset_index(drop=True)

        basic_cols = ["ts_code", "symbol", "name", "industry", "market", "stock_type", "list_date"]
        basic_use = basic[[col for col in basic_cols if col in basic.columns]].copy()
        basic_use["ts_code"] = basic_use["ts_code"].astype(str)
        data = data.merge(basic_use, on="ts_code", how="left")
        data["symbol"] = data["symbol"].fillna(data["ts_code"].str.slice(0, 6)).astype(str).str.zfill(6)
        data["name"] = data["name"].fillna("")
        data["industry"] = data["industry"].fillna("未分类").replace("", "未分类")
        data["stock_type"] = data["stock_type"].fillna("")
        data["list_date"] = data["list_date"].fillna("19000101").astype(str)

        db = daily_basic.copy()
        if "ts_code" not in db.columns:
            db["ts_code"] = pd.Series(dtype=str)
        if "ts_code" in db.columns:
            db["ts_code"] = db["ts_code"].astype(str)
        for col in ["turnover_rate", "turnover_rate_f", "volume_ratio", "total_mv", "circ_mv"]:
            if col not in db.columns:
                db[col] = 0
            db[col] = pd.to_numeric(db[col], errors="coerce").fillna(0)
        db = db[["ts_code", "turnover_rate", "turnover_rate_f", "volume_ratio", "total_mv", "circ_mv"]]
        data = data.merge(db, on="ts_code", how="left")
        for col in ["turnover_rate", "turnover_rate_f", "volume_ratio", "total_mv", "circ_mv"]:
            data[col] = pd.to_numeric(data[col], errors="coerce").fillna(0)

        ret_factor = (1 + data["pct_chg"] / 100).replace([math.inf, -math.inf], 1).fillna(1)
        cum_ret = ret_factor.groupby(data["ts_code"]).cumprod()
        first_close = data.groupby("ts_code")["close"].transform("first")
        first_cum = cum_ret.groupby(data["ts_code"]).transform("first")
        first_cum = first_cum.mask(first_cum == 0, 1).fillna(1)
        data["adj_close"] = first_close * cum_ret / first_cum
        price_factor = (data["adj_close"] / data["close"].mask(data["close"] == 0, 1)).replace([math.inf, -math.inf], 1).fillna(1)
        for col in ["open", "high", "low", "close"]:
            data[f"adj_{col}"] = data[col] * price_factor

        data["limit_rate"] = [self._limit_rate(symbol, name) for symbol, name in zip(data["symbol"], data["name"])]
        data["amount_yi"] = data["amount"] / 100000
        data["market_cap_yi"] = data["total_mv"] / 10000
        data["circ_mv_yi"] = data["circ_mv"] / 10000
        data["list_age_days"] = (
            pd.to_datetime(data["trade_date"], format="%Y%m%d", errors="coerce")
            - pd.to_datetime(data["list_date"], format="%Y%m%d", errors="coerce")
        ).dt.days.fillna(9999)
        data["is_main_board"] = data["symbol"].str.startswith(self.MAIN_BOARD_PREFIXES)

        data["eligible_base"] = self._eligible_base(data)

        grouped = data.groupby("ts_code", group_keys=False)
        for window in [5, 10, 20, 60]:
            min_periods = max(5, int(window * 0.8))
            data[f"ma{window}"] = grouped["adj_close"].transform(lambda series, window=window, min_periods=min_periods: series.rolling(window, min_periods=min_periods).mean())
        data["ma20_prev5"] = data.groupby("ts_code")["ma20"].shift(5)
        data["high40_prev"] = grouped["adj_high"].transform(lambda series: series.shift(1).rolling(40, min_periods=30).max())
        data["high20"] = grouped["adj_high"].transform(lambda series: series.rolling(20, min_periods=20).max())
        data["low20"] = grouped["adj_low"].transform(lambda series: series.rolling(20, min_periods=20).min())
        data["high60"] = grouped["adj_high"].transform(lambda series: series.rolling(60, min_periods=50).max())
        data["low60"] = grouped["adj_low"].transform(lambda series: series.rolling(60, min_periods=50).min())
        data["low120"] = grouped["adj_low"].transform(lambda series: series.rolling(120, min_periods=90).min())
        data["avg_amount20_prev"] = grouped["amount"].transform(lambda series: series.shift(1).rolling(20, min_periods=15).mean())
        data["max_pct_prior30"] = grouped["pct_chg"].transform(lambda series: series.shift(1).rolling(30, min_periods=20).max())

        data["pre_ret20"] = grouped["adj_close"].shift(1) / grouped["adj_close"].shift(21) - 1
        data["pre_ret60"] = grouped["adj_close"].shift(1) / grouped["adj_close"].shift(61) - 1
        for window in [5, 20, 60, 120]:
            data[f"ret{window}"] = data["adj_close"] / grouped["adj_close"].shift(window) - 1

        data["range20"] = data["high20"] / data["low20"] - 1
        data["range60"] = data["high60"] / data["low60"] - 1
        data["since_low120"] = data["adj_close"] / data["low120"] - 1
        data["amount_ratio"] = data["amount"] / data["avg_amount20_prev"]
        data["break_strength"] = data["adj_close"] / data["high40_prev"] - 1
        data["close_near_high"] = data["adj_close"] / data["adj_high"]
        data["ma20_slope5"] = data["ma20"] / data["ma20_prev5"] - 1

        for source, target in [("ret20", "rps20"), ("ret60", "rps60"), ("ret120", "rps120")]:
            rank_source = data[source].where(data["eligible_base"])
            data[target] = rank_source.groupby(data["trade_date"]).rank(pct=True) * 100
        data["rps20_delta10"] = data["rps20"] - data.groupby("ts_code")["rps20"].shift(10)

        data["above_ma20"] = data["eligible_base"] & (data["adj_close"] > data["ma20"])
        data["above_ma60"] = data["eligible_base"] & (data["adj_close"] > data["ma60"])
        market = data[data["eligible_base"]].groupby("trade_date").agg(
            breadth20=("above_ma20", "mean"),
            breadth60=("above_ma60", "mean"),
            up_ratio=("pct_chg", lambda values: (values > 0).mean()),
            median_pct=("pct_chg", "median"),
        ).reset_index()
        data = data.merge(market, on="trade_date", how="left")

        industry = data[data["eligible_base"]].groupby(["trade_date", "industry"], as_index=False).agg(
            industry_ret5=("ret5", "median"),
            industry_amt_ratio=("amount_ratio", "median"),
            industry_count=("ts_code", "count"),
        )
        industry["industry_rps5"] = industry.groupby("trade_date")["industry_ret5"].rank(pct=True) * 100
        industry["industry_amt_rps"] = industry.groupby("trade_date")["industry_amt_ratio"].rank(pct=True) * 100
        data = data.merge(
            industry[["trade_date", "industry", "industry_rps5", "industry_amt_rps", "industry_count"]],
            on=["trade_date", "industry"],
            how="left",
        )

        data["first_wave_score"] = self._score(data).fillna(0)
        data["first_wave_launch"] = self._launch_mask(data)
        return data.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)

    def _eligible_base(self, data: pd.DataFrame) -> pd.Series:
        symbol = data["symbol"].astype(str)
        name = data["name"].astype(str)
        mask = (data["list_age_days"] >= 180) & ~symbol.str.startswith(self.BJ_PREFIXES)
        if bool(getattr(self.config, "first_wave_exclude_st", True)):
            mask &= ~name.str.upper().str.contains("ST", regex=False)
        if bool(getattr(self.config, "first_wave_exclude_star", True)):
            mask &= ~symbol.str.startswith(self.STAR_PREFIXES)
        if bool(getattr(self.config, "first_wave_exclude_chinext", True)):
            mask &= ~symbol.str.startswith(self.CHINEXT_PREFIXES)
        if str(getattr(self.config, "first_wave_stock_scope", "main_board") or "main_board") == "main_board":
            mask &= symbol.str.startswith(self.MAIN_BOARD_PREFIXES)
        return mask

    def _score(self, data: pd.DataFrame) -> pd.Series:
        clip = lambda series: series.clip(lower=0, upper=1)
        max_range20 = max(float(getattr(self.config, "first_wave_max_range20", 30)) / 100, 0.01)
        max_pre_ret20 = max(float(getattr(self.config, "first_wave_max_pre_ret20", 16)) / 100, 0.01)
        max_since_low = max(float(getattr(self.config, "first_wave_max_since_low120", 55)) / 100, 0.01)
        return (
            18 * clip((data["rps20"] - 60) / 36)
            + 14 * clip(data["rps20_delta10"] / 35)
            + 16 * clip((data["amount_ratio"] - 1.3) / 3.2)
            + 14 * clip(data["break_strength"] / 0.08)
            + 10 * clip((max_range20 - data["range20"]) / max(max_range20 * 0.65, 0.01))
            + 8 * clip((max_pre_ret20 - data["pre_ret20"]) / max(max_pre_ret20, 0.01))
            + 10 * clip(data["industry_rps5"] / 100)
            + 6 * clip((data["breadth20"] - 0.42) / 0.28)
            + 4 * clip((max_since_low - data["since_low120"]) / max(max_since_low * 0.82, 0.01))
        )

    def _launch_mask(self, data: pd.DataFrame) -> pd.Series:
        cfg = self.config
        min_amount_yi = float(getattr(cfg, "first_wave_min_amount_yi", 1.5))
        min_amount_ratio = float(getattr(cfg, "first_wave_min_amount_ratio", 1.8))
        max_amount_ratio = float(getattr(cfg, "first_wave_max_amount_ratio", 5.5))
        max_amount_ok = pd.Series(True, index=data.index)
        if max_amount_ratio > 0:
            max_amount_ok = data["amount_ratio"] <= max_amount_ratio
        return (
            data["eligible_base"]
            & (data["first_wave_score"] >= float(getattr(cfg, "first_wave_min_score", 70)))
            & (data["amount_yi"] >= min_amount_yi)
            & (data["amount_ratio"] >= min_amount_ratio)
            & max_amount_ok
            & (data["adj_close"] > data["high40_prev"] * (1 + float(getattr(cfg, "first_wave_breakout_pct", 0.5)) / 100))
            & (data["pct_chg"] >= float(getattr(cfg, "first_wave_min_launch_pct", 5)))
            & (data["pct_chg"] <= data["limit_rate"] * 100 + 2)
            & (data["close_near_high"] >= float(getattr(cfg, "first_wave_close_near_high", 0.97)))
            & (data["adj_close"] > data["ma20"])
            & (data["adj_close"] > data["ma60"])
            & (data["ma20"] >= data["ma60"] * 0.97)
            & (data["ma20_slope5"] > 0)
            & (data["range20"] <= float(getattr(cfg, "first_wave_max_range20", 30)) / 100)
            & (data["range60"] <= float(getattr(cfg, "first_wave_max_range60", 75)) / 100)
            & (data["pre_ret20"] <= float(getattr(cfg, "first_wave_max_pre_ret20", 16)) / 100)
            & (data["pre_ret60"] <= float(getattr(cfg, "first_wave_max_pre_ret60", 32)) / 100)
            & (data["since_low120"] <= float(getattr(cfg, "first_wave_max_since_low120", 55)) / 100)
            & (data["max_pct_prior30"] < data["limit_rate"] * 100 * 0.86)
            & (data["rps20"] >= float(getattr(cfg, "first_wave_min_rps20", 68)))
            & (data["rps20"] <= float(getattr(cfg, "first_wave_max_rps20", 93)))
            & (data["rps20_delta10"] >= float(getattr(cfg, "first_wave_min_rps20_delta10", 15)))
            & (data["rps120"] <= float(getattr(cfg, "first_wave_max_rps120", 96)))
            & (data["breadth20"] >= float(getattr(cfg, "first_wave_min_breadth20", 50)) / 100)
            & (data["breadth60"] >= float(getattr(cfg, "first_wave_min_breadth60", 40)) / 100)
            & (data["up_ratio"] >= float(getattr(cfg, "first_wave_min_up_ratio", 45)) / 100)
            & (data["industry_rps5"] >= float(getattr(cfg, "first_wave_min_industry_rps5", 60)))
            & (data["industry_amt_rps"] >= float(getattr(cfg, "first_wave_min_industry_amt_rps", 45)))
        ).fillna(False)

    def _rows(self, features: pd.DataFrame, trade_dates: list[str]) -> list[dict[str, Any]]:
        trade_date = trade_dates[-1]
        mode = str(getattr(self.config, "first_wave_entry_mode", "both") or "both")
        if mode not in {"both", "launch", "pullback"}:
            mode = "both"
        rows: list[dict[str, Any]] = []
        latest = features[features["trade_date"] == trade_date].copy()
        if mode in {"both", "launch"}:
            launch_rows = latest[latest["first_wave_launch"]].copy()
            for _, row in launch_rows.iterrows():
                rows.append(self._row_dict(row, "启动日", "低缺口跟随"))

        if mode in {"both", "pullback"}:
            pullback_rows = self._pullback_rows(features, trade_dates)
            existing = {row["ts_code"] for row in rows}
            for _, row in pullback_rows.iterrows():
                if str(row["ts_code"]) in existing:
                    continue
                rows.append(self._row_dict(row, "回踩确认", "回踩确认"))

        rows = sorted(rows, key=lambda item: item["score"], reverse=True)
        top_count = int(getattr(self.config, "first_wave_top_count", 80) or 80)
        return rows[: max(1, top_count)]

    def _pullback_rows(self, features: pd.DataFrame, trade_dates: list[str]) -> pd.DataFrame:
        trade_date = trade_dates[-1]
        date_to_index = {date: index for index, date in enumerate(trade_dates)}
        current_index = date_to_index[trade_date]
        wait_days = max(1, int(getattr(self.config, "first_wave_pullback_wait_days", 5)))
        latest = features[(features["trade_date"] == trade_date) & features["eligible_base"]].copy()
        if latest.empty:
            return latest

        frames_by_code = {
            ts_code: frame.set_index("trade_date")
            for ts_code, frame in features[features["trade_date"].isin(trade_dates[max(0, current_index - wait_days - 1) : current_index + 1])].groupby("ts_code")
        }
        accepted = []
        for _, current in latest.iterrows():
            frame = frames_by_code.get(str(current["ts_code"]))
            if frame is None:
                continue
            launch_dates = []
            for offset in range(wait_days, 0, -1):
                launch_index = current_index - offset
                if launch_index < 0:
                    continue
                launch_date = trade_dates[launch_index]
                if launch_date not in frame.index:
                    continue
                launch = frame.loc[launch_date]
                if isinstance(launch, pd.DataFrame):
                    launch = launch.iloc[-1]
                if bool(launch.get("first_wave_launch", False)):
                    launch_dates.append(launch_date)
            if not launch_dates:
                continue
            launch_date = launch_dates[-1]
            launch = frame.loc[launch_date]
            if isinstance(launch, pd.DataFrame):
                launch = launch.iloc[-1]
            if not self._pullback_ok(current, launch):
                continue
            if not self._first_pullback_confirmation(frame, launch_date, trade_date, launch, trade_dates):
                continue
            enriched = current.copy()
            enriched["launch_date"] = launch_date
            enriched["launch_close"] = launch.get("close", 0)
            enriched["launch_pct_chg"] = launch.get("pct_chg", 0)
            enriched["launch_amount_yi"] = launch.get("amount_yi", 0)
            accepted.append(enriched)
        if not accepted:
            return latest.iloc[0:0].copy()
        return pd.DataFrame(accepted)

    def mark_pullbacks(self, features: pd.DataFrame, trade_dates: list[str]) -> pd.DataFrame:
        """Mark the first valid pullback after each launch across a date range."""
        data = features.copy()
        data["first_wave_pullback"] = False
        data["launch_date"] = ""
        data["launch_close"] = 0.0
        data["launch_pct_chg"] = 0.0
        data["launch_amount_yi"] = 0.0
        date_to_index = {date: index for index, date in enumerate(trade_dates)}
        wait_days = max(1, int(getattr(self.config, "first_wave_pullback_wait_days", 5)))

        for _, frame in data.groupby("ts_code", sort=False):
            launch: pd.Series | None = None
            launch_date = ""
            launch_index = -1
            confirmed = False
            for index, row in frame.iterrows():
                row_date = str(row.get("trade_date", ""))
                row_index = date_to_index.get(row_date, -1)
                if row_index < 0:
                    continue
                if bool(row.get("first_wave_launch", False)):
                    launch = row
                    launch_date = row_date
                    launch_index = row_index
                    confirmed = False
                    continue
                if launch is None:
                    continue
                distance = row_index - launch_index
                if distance < 1:
                    continue
                if distance > wait_days:
                    launch = None
                    launch_date = ""
                    launch_index = -1
                    confirmed = False
                    continue
                if confirmed:
                    continue
                if self._pullback_ok(row, launch):
                    data.at[index, "first_wave_pullback"] = True
                    data.at[index, "launch_date"] = launch_date
                    data.at[index, "launch_close"] = launch.get("close", 0)
                    data.at[index, "launch_pct_chg"] = launch.get("pct_chg", 0)
                    data.at[index, "launch_amount_yi"] = launch.get("amount_yi", 0)
                    confirmed = True
        return data

    def _first_pullback_confirmation(
        self,
        frame: pd.DataFrame,
        launch_date: str,
        trade_date: str,
        launch: pd.Series,
        trade_dates: list[str],
    ) -> bool:
        start = trade_dates.index(launch_date) + 1
        end = trade_dates.index(trade_date)
        for prior_date in trade_dates[start:end]:
            if prior_date not in frame.index:
                continue
            prior = frame.loc[prior_date]
            if isinstance(prior, pd.DataFrame):
                prior = prior.iloc[-1]
            if self._pullback_ok(prior, launch):
                return False
        return True

    def _pullback_ok(self, current: pd.Series, launch: pd.Series) -> bool:
        max_above = float(getattr(self.config, "first_wave_pullback_max_above", 3.5)) / 100
        max_below = float(getattr(self.config, "first_wave_pullback_max_below", 3.5)) / 100
        launch_close = self._num(launch.get("adj_close"))
        launch_low = self._num(launch.get("adj_low"))
        current_close = self._num(current.get("adj_close"))
        current_low = self._num(current.get("adj_low"))
        current_ma5 = self._num(current.get("ma5"))
        if launch_close <= 0 or current_close <= 0:
            return False
        limit_close = self._num(current.get("pre_close")) * (1 + self._num(current.get("limit_rate")))
        not_limit_like = self._num(current.get("close")) < limit_close * 0.985 if limit_close > 0 else True
        return bool(
            current_low <= launch_close * (1 + max_above)
            and current_close >= launch_close * (1 - max_below)
            and current_close >= max(current_ma5, launch_low)
            and self._num(current.get("amount")) <= self._num(launch.get("amount")) * 1.15
            and self._num(current.get("pct_chg")) > -5.5
            and current_close > self._num(current.get("ma20"))
            and self._num(current.get("breadth20")) >= float(getattr(self.config, "first_wave_min_breadth20", 50)) / 100
            and not_limit_like
        )

    def _row_dict(self, row: pd.Series, setup: str, entry_mode: str) -> dict[str, Any]:
        max_gap = float(getattr(self.config, "first_wave_max_buy_gap_pct", 3))
        min_gap = float(getattr(self.config, "first_wave_min_buy_gap_pct", -3))
        reason = self._reason(row, setup)
        return {
            "ts_code": str(row.get("ts_code", "")),
            "code": str(row.get("symbol", str(row.get("ts_code", ""))[:6])).zfill(6),
            "name": str(row.get("name", "")),
            "stock_type": str(row.get("stock_type", "")),
            "industry": str(row.get("industry", "")),
            "setup": setup,
            "entry_mode": entry_mode,
            "score": round(self._num(row.get("first_wave_score")), 2),
            "close": round(self._num(row.get("close")), 2),
            "pct_chg": round(self._num(row.get("pct_chg")), 2),
            "amount_yi": round(self._num(row.get("amount_yi")), 2),
            "amount_ratio": round(self._num(row.get("amount_ratio")), 2),
            "turnover_rate": round(self._num(row.get("turnover_rate")), 2),
            "turnover_rate_f": round(self._num(row.get("turnover_rate_f")), 2),
            "volume_ratio": round(self._num(row.get("volume_ratio")), 2),
            "market_cap_yi": round(self._num(row.get("market_cap_yi")), 2),
            "circ_mv_yi": round(self._num(row.get("circ_mv_yi")), 2),
            "rps20": round(self._num(row.get("rps20")), 2),
            "rps20_delta10": round(self._num(row.get("rps20_delta10")), 2),
            "rps60": round(self._num(row.get("rps60")), 2),
            "rps120": round(self._num(row.get("rps120")), 2),
            "range20": round(self._num(row.get("range20")) * 100, 2),
            "range60": round(self._num(row.get("range60")) * 100, 2),
            "pre_ret20": round(self._num(row.get("pre_ret20")) * 100, 2),
            "pre_ret60": round(self._num(row.get("pre_ret60")) * 100, 2),
            "since_low120": round(self._num(row.get("since_low120")) * 100, 2),
            "break_strength": round(self._num(row.get("break_strength")) * 100, 2),
            "industry_rps5": round(self._num(row.get("industry_rps5")), 2),
            "industry_amt_rps": round(self._num(row.get("industry_amt_rps")), 2),
            "breadth20": round(self._num(row.get("breadth20")) * 100, 2),
            "breadth60": round(self._num(row.get("breadth60")) * 100, 2),
            "up_ratio": round(self._num(row.get("up_ratio")) * 100, 2),
            "launch_date": str(row.get("launch_date", row.get("trade_date", ""))),
            "launch_pct_chg": round(self._num(row.get("launch_pct_chg", row.get("pct_chg", 0))), 2),
            "launch_amount_yi": round(self._num(row.get("launch_amount_yi", row.get("amount_yi", 0))), 2),
            "trigger_close": round(self._num(row.get("close")), 3),
            "trigger_close_raw": round(self._num(row.get("close")), 3),
            "max_buy_gap_pct": max_gap,
            "min_buy_gap_pct": min_gap,
            "buy_plan": f"次日开盘相对触发收盘价在 {min_gap:g}% 到 +{max_gap:g}% 内才考虑；高开过多放弃。",
            "invalid_condition": "跌回启动阳线低点或跌破MA10；盘中信号需等收盘确认。",
            "risk_flags": self._risk_flags(row),
            "reason": reason,
        }

    def _reason(self, row: pd.Series, setup: str) -> str:
        pieces = [
            setup,
            f"RPS20 {self._num(row.get('rps20')):.1f}",
            f"10日抬升 {self._num(row.get('rps20_delta10')):.1f}",
            f"量能 {self._num(row.get('amount_ratio')):.1f}倍",
            f"20日振幅 {self._num(row.get('range20')) * 100:.1f}%",
            f"行业强度 {self._num(row.get('industry_rps5')):.1f}",
        ]
        return "；".join(pieces)

    def _risk_flags(self, row: pd.Series) -> str:
        flags = []
        if self._num(row.get("amount_ratio")) > 4.5:
            flags.append("量能偏猛")
        if self._num(row.get("pct_chg")) >= self._num(row.get("limit_rate")) * 100 * 0.92:
            flags.append("接近涨停")
        if self._num(row.get("turnover_rate_f")) > 15:
            flags.append("实换手偏高")
        if self._num(row.get("breadth20")) < 0.55:
            flags.append("市场广度一般")
        return "；".join(flags) if flags else "正常"

    def _market_snapshot(self, features: pd.DataFrame, trade_date: str, rows: list[dict[str, Any]], intraday_note: str) -> dict[str, Any]:
        latest = features[features["trade_date"] == trade_date]
        if latest.empty:
            return {}
        market_row = latest.iloc[-1]
        launch_count = int(latest["first_wave_launch"].fillna(False).sum())
        pullback_count = sum(1 for row in rows if row.get("setup") == "回踩确认")
        note = "市场广度满足第一波策略最低要求。"
        if self._num(market_row.get("breadth20")) < float(getattr(self.config, "first_wave_min_breadth20", 50)) / 100:
            note = "市场广度不足，第一波信号更适合只观察。"
        if intraday_note:
            note = f"{note} {intraday_note}"
        return {
            "trade_date": trade_date,
            "review_mode": self.config.review_mode,
            "breadth20": round(self._num(market_row.get("breadth20")) * 100, 2),
            "breadth60": round(self._num(market_row.get("breadth60")) * 100, 2),
            "up_ratio": round(self._num(market_row.get("up_ratio")) * 100, 2),
            "median_pct": round(self._num(market_row.get("median_pct")), 2),
            "launch_count": launch_count,
            "candidate_count": len(rows),
            "pullback_count": pullback_count,
            "note": note,
        }

    @staticmethod
    def _limit_rate(symbol: str, name: str) -> float:
        symbol = str(symbol)
        if "ST" in str(name).upper():
            return 0.05
        if symbol.startswith(("300", "301", "688", "689")):
            return 0.20
        if symbol.startswith(("8", "4", "92")):
            return 0.30
        return 0.10

    @staticmethod
    def _num(value: Any) -> float:
        try:
            if pd.isna(value):
                return 0.0
            return float(value)
        except (TypeError, ValueError):
            return 0.0
