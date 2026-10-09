from __future__ import annotations

import math
import json
from datetime import date, datetime
from typing import Any

import pandas as pd
import requests


class MainlineReviewStrategy:
    """主线复盘策略：市场温度、板块强度、个股角色与风控提示。"""

    MARKET_INDEXES = [
        ("000001.SH", "上证指数"),
        ("399001.SZ", "深证成指"),
        ("399006.SZ", "创业板指"),
    ]
    INDEX_PERIODS = {
        "daily": "日线",
        "120min": "120分钟",
        "60min": "60分钟",
        "30min": "30分钟",
        "15min": "15分钟",
        "5min": "5分钟",
    }
    INTRADAY_LOOKBACK_CAP = {
        "120min": 220,
        "60min": 140,
        "30min": 90,
        "15min": 60,
        "5min": 20,
    }
    EASTMONEY_INDEX_MARKET = {
        "000001.SH": "1",
        "399001.SZ": "0",
        "399006.SZ": "0",
    }
    SINA_INDEX_SYMBOLS = {
        "000001.SH": "sh000001",
        "399001.SZ": "sz399001",
        "399006.SZ": "sz399006",
    }

    def __init__(self, engine: Any) -> None:
        self.engine = engine
        self.config = engine.config

    def __getattr__(self, name: str) -> Any:
        return getattr(self.engine, name)

    def run(self) -> dict[str, Any]:
        """一次完整复盘的主流程。

        执行顺序：
        1. 找到最近交易日。
        2. 读取股票基础信息。
        3. 读取历史日线。
        4. 如果是盘中模式，用实时行情覆盖当天临时日 K。
        5. 读取市值/换手率、涨跌停、大盘资金流。
        6. 计算市场温度。
        7. 计算一级概念或行业板块强度。
        8. 在候选板块中筛选个股。
        """
        self.log("读取交易日历...")
        trade_dates = self._trade_dates()
        end_date = trade_dates[-1]
        self.config.end_date = end_date
        self.log(f"使用交易日: {end_date}")

        self.log("读取股票基础信息...")
        basic = self._stock_basic()

        self.log("读取近端日线行情...")
        daily = self._daily_window(trade_dates)
        if self.config.review_mode == "intraday":
            self.log("盘中模式：读取实时行情并合成临时日K...")
            daily = self._apply_intraday_snapshot(daily, basic, trade_dates[-1])

        self.log("读取每日指标...")
        daily_basic = self._daily_basic(end_date)
        if self.config.review_mode == "intraday" and self._daily_basic_is_empty(daily_basic) and len(trade_dates) >= 2:
            prev_date = trade_dates[-2]
            self.log(f"盘中模式：当日每日指标未更新，回退到 {prev_date}。")
            daily_basic = self._daily_basic(prev_date)

        self.log("读取涨跌停数据...")
        limit_list = self._limit_list(end_date)

        self.log("读取大盘资金流...")
        market_flow = self._market_flow(end_date)

        self.log("计算市场温度...")
        market = self._market_state(daily, daily_basic, limit_list, trade_dates, market_flow)

        self.log("计算一级概念/板块强度...")
        boards = self._score_boards(basic, daily, daily_basic, limit_list, trade_dates)

        self.log("计算个股候选...")
        stocks = self._score_stocks(basic, daily, daily_basic, limit_list, boards)

        self.log("复盘完成。")
        return {
            "ok": True,
            "trade_date": end_date,
            "market": market,
            "boards": boards.head(self.config.top_board_count).fillna("").to_dict("records"),
            "stocks": stocks.head(self.config.top_stock_count).fillna("").to_dict("records"),
            "logs": self.logs,
        }

    def market_overview(self) -> dict[str, Any]:
        """只计算首页需要的大盘温度信息。"""
        self.log("读取交易日历...")
        trade_dates = self._trade_dates()
        end_date = trade_dates[-1]
        self.config.end_date = end_date
        self.log(f"使用交易日: {end_date}")

        self.log("读取近端日线行情...")
        daily = self._daily_window(trade_dates)
        if self.config.review_mode == "intraday":
            self.log("盘中模式：读取股票基础信息和实时行情...")
            basic = self._stock_basic()
            daily = self._apply_intraday_snapshot(daily, basic, trade_dates[-1])

        self.log("读取每日指标...")
        daily_basic = self._daily_basic(end_date)
        if self.config.review_mode == "intraday" and self._daily_basic_is_empty(daily_basic) and len(trade_dates) >= 2:
            prev_date = trade_dates[-2]
            self.log(f"盘中模式：当日每日指标未更新，回退到 {prev_date}。")
            daily_basic = self._daily_basic(prev_date)

        self.log("读取涨跌停数据...")
        limit_list = self._limit_list(end_date)

        self.log("读取大盘资金流...")
        market_flow = self._market_flow(end_date)

        self.log("计算市场温度...")
        market = self._market_state(daily, daily_basic, limit_list, trade_dates, market_flow)

        period = self._index_period()
        period_label = self.INDEX_PERIODS[period]
        self.log(f"读取指数{period_label}K线并分析艾略特波浪...")
        index_charts = self._market_index_charts(trade_dates)

        self.log("大盘概览完成。")
        return {
            "ok": True,
            "trade_date": end_date,
            "market": market,
            "index_period": period,
            "index_period_label": period_label,
            "index_charts": index_charts,
            "logs": self.logs,
        }

    def _market_index_charts(self, trade_dates: list[str]) -> list[dict[str, Any]]:
        """Return K-line chart data and rule-based Elliott wave notes for main indexes.

        The wave label is deliberately heuristic. It is useful for describing
        market structure, but it should sit behind breadth, liquidity, and trend
        scores when making trading decisions.
        """
        period = self._index_period()
        period_label = self.INDEX_PERIODS[period]
        charts: list[dict[str, Any]] = []
        for code, name in self.MARKET_INDEXES:
            frame = self._index_window(code, name, trade_dates, period)
            if frame.empty:
                continue
            frame = self._add_index_indicators(frame)
            wave = self._elliott_wave(frame)
            last = frame.iloc[-1]
            charts.append(
                {
                    "code": code,
                    "name": name,
                    "period": period,
                    "period_label": period_label,
                    "trade_date": str(last["trade_date"]),
                    "close": round(float(last["close"]), 2),
                    "pct_chg": round(float(last.get("pct_chg", 0) or 0), 2),
                    "trend": self._index_trend_label(frame),
                    "kline": self._index_kline_rows(frame),
                    "wave": wave,
                }
            )
        if not charts:
            self.log(f"指数{period_label}K线为空：请检查数据源权限、网络或缓存日期。")
        return charts

    def _index_period(self) -> str:
        raw = str(getattr(self.config, "index_period", "daily") or "daily").strip().lower()
        aliases = {
            "d": "daily",
            "day": "daily",
            "1d": "daily",
            "日线": "daily",
            "120": "120min",
            "120m": "120min",
            "120min": "120min",
            "60": "60min",
            "60m": "60min",
            "60min": "60min",
            "30": "30min",
            "30m": "30min",
            "30min": "30min",
            "15": "15min",
            "15m": "15min",
            "15min": "15min",
            "5": "5min",
            "5m": "5min",
            "5min": "5min",
        }
        return aliases.get(raw, "daily")

    def _index_window(self, code: str, name: str, trade_dates: list[str], period: str) -> pd.DataFrame:
        if period == "daily":
            return self._index_daily_window(code, name, trade_dates)
        return self._index_intraday_window(code, name, trade_dates, period)

    def _index_daily_window(self, code: str, name: str, trade_dates: list[str]) -> pd.DataFrame:
        start_date = trade_dates[0]
        end_date = trade_dates[-1]
        try:
            frame = self._cached(
                f"index_daily_{code}_{start_date}_{end_date}",
                lambda: self.pro.index_daily(ts_code=code, start_date=start_date, end_date=end_date),
            )
        except Exception as exc:
            self.log(f"{name} K线读取失败: {exc}")
            return pd.DataFrame()
        if frame.empty or "trade_date" not in frame.columns:
            self.log(f"{name} K线为空。")
            return pd.DataFrame()
        frame = frame.copy()
        frame["trade_date"] = frame["trade_date"].astype(str)
        frame = frame[frame["trade_date"].isin(set(trade_dates))]
        for col in ["open", "high", "low", "close", "pre_close", "change", "pct_chg", "vol", "amount"]:
            if col not in frame.columns:
                frame[col] = 0
            frame[col] = pd.to_numeric(frame[col], errors="coerce").fillna(0)
        frame = frame.sort_values("trade_date")
        result = frame[frame["close"] > 0].reset_index(drop=True)
        result.attrs["index_period"] = "daily"
        result.attrs["amount_unit"] = "tushare_index"
        result.attrs["row_limit"] = 320
        return result

    def _index_intraday_window(self, code: str, name: str, trade_dates: list[str], period: str) -> pd.DataFrame:
        base_period = "60min" if period == "120min" else period
        cap = self.INTRADAY_LOOKBACK_CAP.get(period, 60)
        use_dates = trade_dates[-min(len(trade_dates), cap) :]
        start_date = use_dates[0]
        end_date = use_dates[-1]
        if len(use_dates) < len(trade_dates):
            self.log(f"{name} {self.INDEX_PERIODS[period]}：分钟线回看限制为最近 {len(use_dates)} 个交易日，避免接口过慢。")
        try:
            frame = self._cached_or_fetch_index_minutes(code, base_period, start_date, end_date)
        except Exception as exc:
            self.log(f"{name} {self.INDEX_PERIODS[period]}K线读取失败: {exc}")
            return pd.DataFrame()
        if frame.empty:
            self.log(f"{name} {self.INDEX_PERIODS[period]}K线为空。")
            return pd.DataFrame()
        frame = self._normalize_index_minute_frame(frame)
        if frame.empty:
            self.log(f"{name} {self.INDEX_PERIODS[period]}K线字段异常。")
            return pd.DataFrame()
        frame = frame[frame["session_date"].isin(set(use_dates))].copy()
        if period == "120min":
            frame = self._aggregate_120min(frame)
            frame = self._normalize_index_minute_frame(frame)
        frame = frame.sort_values("trade_date").reset_index(drop=True)
        frame.attrs["index_period"] = period
        frame.attrs["amount_unit"] = "yuan"
        frame.attrs["row_limit"] = self._index_row_limit(period)
        return frame[frame["close"] > 0].reset_index(drop=True)

    def _cached_or_fetch_index_minutes(self, code: str, period: str, start_date: str, end_date: str) -> pd.DataFrame:
        cache_name = f"index_min_{period}_{code}_{start_date}_{end_date}"
        if self._should_refresh_intraday_cache(end_date):
            return self._fetch_index_minutes(code, period, start_date, end_date)
        return self._cached(
            cache_name,
            lambda: self._fetch_index_minutes(code, period, start_date, end_date),
        )

    def _should_refresh_intraday_cache(self, end_date: str) -> bool:
        if self.config.review_mode != "intraday":
            return False
        return end_date >= date.today().strftime("%Y%m%d")

    def _fetch_index_minutes(self, code: str, period: str, start_date: str, end_date: str) -> pd.DataFrame:
        try:
            frame = self._fetch_sina_index_minutes(code, period, start_date, end_date)
            if not frame.empty:
                frame["minute_source"] = "sina"
                return frame
        except Exception as exc:
            self.log(f"{code} 新浪{self.INDEX_PERIODS.get(period, period)}分钟源失败，尝试东财: {exc}")
        frame = self._fetch_eastmoney_index_minutes(code, period, start_date, end_date)
        if not frame.empty:
            frame["minute_source"] = "eastmoney"
        return frame

    def _fetch_eastmoney_index_minutes(self, code: str, period: str, start_date: str, end_date: str) -> pd.DataFrame:
        symbol = code.split(".")[0]
        market = self.EASTMONEY_INDEX_MARKET.get(code, "1" if code.endswith(".SH") else "0")
        klt = period.replace("min", "")
        url = "https://push2his.eastmoney.com/api/qt/stock/kline/get"
        params = {
            "secid": f"{market}.{symbol}",
            "ut": "7eea3edcaed734bea9cbfc24409ed989",
            "fields1": "f1,f2,f3,f4,f5,f6",
            "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61",
            "klt": klt,
            "fqt": "1",
            "beg": start_date,
            "end": end_date,
        }
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()
        data = response.json().get("data") or {}
        rows = data.get("klines") or []
        if not rows:
            return pd.DataFrame()
        frame = pd.DataFrame([item.split(",") for item in rows])
        frame = frame.iloc[:, :11].copy()
        frame.columns = ["trade_time", "open", "close", "high", "low", "vol", "amount", "amplitude", "pct_chg", "change", "turnover"]
        frame["trade_time"] = pd.to_datetime(frame["trade_time"], errors="coerce")
        frame = frame.dropna(subset=["trade_time"]).copy()
        for col in ["open", "close", "high", "low", "vol", "amount", "pct_chg", "change"]:
            frame[col] = pd.to_numeric(frame[col], errors="coerce").fillna(0)
        frame["session_date"] = frame["trade_time"].dt.strftime("%Y%m%d")
        frame["trade_date"] = frame["trade_time"].dt.strftime("%Y%m%d %H:%M")
        frame["pre_close"] = frame["close"].shift(1).fillna(frame["open"])
        return frame[["trade_date", "session_date", "open", "high", "low", "close", "pre_close", "change", "pct_chg", "vol", "amount"]]

    def _fetch_sina_index_minutes(self, code: str, period: str, start_date: str, end_date: str) -> pd.DataFrame:
        symbol = self.SINA_INDEX_SYMBOLS.get(code)
        if not symbol:
            return pd.DataFrame()
        scale = period.replace("min", "")
        url = "https://quotes.sina.cn/cn/api/jsonp_v2.php/=/CN_MarketDataService.getKLineData"
        params = {"symbol": symbol, "scale": scale, "ma": "no", "datalen": "1970"}
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()
        text = response.text
        start = text.find("=(")
        end = text.rfind(");")
        if start < 0 or end <= start:
            return pd.DataFrame()
        records = json.loads(text[start + 2 : end])
        frame = pd.DataFrame(records)
        if frame.empty or "day" not in frame.columns:
            return pd.DataFrame()
        frame = frame.rename(columns={"day": "trade_time", "volume": "vol"})
        frame["trade_time"] = pd.to_datetime(frame["trade_time"], errors="coerce")
        frame = frame.dropna(subset=["trade_time"]).copy()
        for col in ["open", "high", "low", "close", "vol", "amount"]:
            if col not in frame.columns:
                frame[col] = 0
            frame[col] = pd.to_numeric(frame[col], errors="coerce").fillna(0)
        frame["session_date"] = frame["trade_time"].dt.strftime("%Y%m%d")
        frame = frame[(frame["session_date"] >= start_date) & (frame["session_date"] <= end_date)].copy()
        if frame.empty:
            return pd.DataFrame()
        frame = frame.sort_values("trade_time").reset_index(drop=True)
        frame["trade_date"] = frame["trade_time"].dt.strftime("%Y%m%d %H:%M")
        frame["pre_close"] = frame["close"].shift(1).fillna(frame["open"])
        frame["change"] = frame["close"] - frame["pre_close"]
        frame["pct_chg"] = (frame["close"] / frame["pre_close"].replace(0, pd.NA) - 1).fillna(0) * 100
        return frame[["trade_date", "session_date", "open", "high", "low", "close", "pre_close", "change", "pct_chg", "vol", "amount"]]

    @staticmethod
    def _normalize_index_minute_frame(frame: pd.DataFrame) -> pd.DataFrame:
        required = ["trade_date", "session_date", "open", "high", "low", "close", "pre_close", "change", "pct_chg", "vol", "amount"]
        if frame is None or frame.empty:
            return pd.DataFrame(columns=required)
        data = frame.copy()
        for col in required:
            if col not in data.columns:
                data[col] = 0 if col not in {"trade_date", "session_date"} else ""
        data["trade_date"] = data["trade_date"].astype(str)
        data["session_date"] = data["session_date"].astype(str)
        for col in ["open", "high", "low", "close", "pre_close", "change", "pct_chg", "vol", "amount"]:
            data[col] = pd.to_numeric(data[col], errors="coerce").fillna(0)
        return data[required]

    @staticmethod
    def _aggregate_120min(frame: pd.DataFrame) -> pd.DataFrame:
        if frame.empty:
            return frame
        data = frame.sort_values("trade_date").copy()
        data["slot"] = data.groupby("session_date").cumcount() // 2
        grouped = data.groupby(["session_date", "slot"], as_index=False)
        result = grouped.agg(
            trade_date=("trade_date", "last"),
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
            vol=("vol", "sum"),
            amount=("amount", "sum"),
        )
        result["pre_close"] = result["close"].shift(1).fillna(result["open"])
        result["change"] = result["close"] - result["pre_close"]
        result["pct_chg"] = (result["close"] / result["pre_close"].replace(0, pd.NA) - 1).fillna(0) * 100
        return result[["trade_date", "session_date", "open", "high", "low", "close", "pre_close", "change", "pct_chg", "vol", "amount"]]

    @staticmethod
    def _add_index_indicators(frame: pd.DataFrame) -> pd.DataFrame:
        result = frame.copy()
        for window in [5, 10, 20, 60]:
            result[f"ma{window}"] = result["close"].rolling(window, min_periods=window).mean()
        amount_unit = result.attrs.get("amount_unit", "tushare_index")
        result["amount_yi"] = result["amount"] / (100000000 if amount_unit == "yuan" else 100000)
        result.attrs.update(frame.attrs)
        return result

    def _index_kline_rows(self, frame: pd.DataFrame) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for idx, row in frame.tail(self._index_row_limit(str(frame.attrs.get("index_period", "daily")))).iterrows():
            rows.append(
                {
                    "idx": int(idx),
                    "date": str(row["trade_date"]),
                    "open": round(float(row["open"]), 2),
                    "high": round(float(row["high"]), 2),
                    "low": round(float(row["low"]), 2),
                    "close": round(float(row["close"]), 2),
                    "pct_chg": round(float(row.get("pct_chg", 0) or 0), 2),
                    "amount_yi": round(float(row.get("amount_yi", 0) or 0), 1),
                    "ma5": self._round_or_none(row.get("ma5")),
                    "ma10": self._round_or_none(row.get("ma10")),
                    "ma20": self._round_or_none(row.get("ma20")),
                    "ma60": self._round_or_none(row.get("ma60")),
                }
            )
        return rows

    def _index_row_limit(self, period: str) -> int:
        return {
            "daily": 320,
            "120min": 520,
            "60min": 640,
            "30min": 820,
            "15min": 1040,
            "5min": 1200,
        }.get(period, 320)

    def _elliott_wave(self, frame: pd.DataFrame) -> dict[str, Any]:
        pivots = self._zigzag_pivots(frame)
        trend = self._index_trend_label(frame)
        latest_close = float(frame.iloc[-1]["close"])
        if len(pivots) < 3:
            channel, channels = self._channel_bundle(frame, pivots, trend)
            return {
                "stage": "波段不足",
                "bias": "中性",
                "confidence": "低",
                "summary": "有效转折点不足，暂时只参考均线和市场温度。",
                "support": "",
                "resistance": "",
                "invalidation": "等待至少3个有效转折点后再推演。",
                "points": [],
                "channel": channel,
                "channels": channels,
                "method": "按收盘价ZigZag识别主要转折，过滤过小波动。",
            }

        if trend == "下行":
            result = self._down_wave_summary(frame, pivots, latest_close)
            result["channel"], result["channels"] = self._channel_bundle(frame, pivots, trend)
            return result
        if trend == "震荡":
            result = self._range_wave_summary(frame, pivots, latest_close)
            result["channel"], result["channels"] = self._channel_bundle(frame, pivots, trend)
            return result
        result = self._up_wave_summary(frame, pivots, latest_close)
        result["channel"], result["channels"] = self._channel_bundle(frame, pivots, trend)
        return result

    def _up_wave_summary(self, frame: pd.DataFrame, pivots: list[dict[str, Any]], latest_close: float) -> dict[str, Any]:
        anchor = self._last_major_pivot(pivots, "L", prefer_lowest=True)
        seq = pivots[anchor:] if anchor is not None else pivots[-6:]
        up_legs = max(0, sum(1 for item in seq if item["kind"] == "H"))
        down_legs = max(0, sum(1 for item in seq if item["kind"] == "L") - 1)
        last_kind = seq[-1]["kind"]
        last_high = self._last_pivot_price(seq, "H")
        last_low = self._last_pivot_price(seq, "L")

        if last_kind == "H":
            if up_legs >= 3:
                stage = "疑似第5浪上行/延伸"
                bias = "偏多但防追高"
                summary = "指数仍在上升结构内，但已进入较后段推进，突破前高要看量能和市场广度能否继续扩散。"
            elif up_legs == 2:
                stage = "疑似第3浪推进"
                bias = "偏多"
                summary = "上升结构处在主推阶段的概率较高，回踩不破前一转折低点时，市场风险偏好通常更友好。"
            else:
                stage = "疑似第1浪启动"
                bias = "试探偏多"
                summary = "上升结构刚形成，仍需要后续回踩确认第二浪是否温和。"
        else:
            if down_legs >= 2:
                stage = "疑似第4浪整理"
                bias = "中性偏多"
                summary = "指数处在上升后的整理段，关键是回调不放量跌破前一主要低点。"
            else:
                stage = "疑似第2浪回调"
                bias = "等待确认"
                summary = "启动后进入第一次回踩，若缩量企稳并重新站回短期均线，才更像健康二浪。"

        confidence = self._wave_confidence(seq, frame)
        point_seq = seq if len(seq) >= 3 else pivots[-6:]
        points = self._wave_points(point_seq, ["起点", "1", "2", "3", "4", "5"])
        support = self._round_or_none(last_low)
        resistance = self._round_or_none(last_high)
        invalidation = f"跌破上一主要低点 {support}，上升浪推演降级。" if support else "跌破最近主要低点则降级。"
        if resistance and latest_close < float(resistance):
            invalidation = f"不能重新突破 {resistance}，且跌破 {support}，上升结构转弱。" if support else invalidation
        return {
            "stage": stage,
            "bias": bias,
            "confidence": confidence,
            "summary": summary,
            "support": support or "",
            "resistance": resistance or "",
            "invalidation": invalidation,
            "points": points,
            "method": "上涨结构按 低-高-低-高-低-高 的有效转折推演，标签为规则化参考。",
        }

    def _down_wave_summary(self, frame: pd.DataFrame, pivots: list[dict[str, Any]], latest_close: float) -> dict[str, Any]:
        anchor = self._last_major_pivot(pivots, "H", prefer_lowest=False)
        seq = pivots[anchor:] if anchor is not None else pivots[-5:]
        down_legs = max(0, sum(1 for item in seq if item["kind"] == "L"))
        last_kind = seq[-1]["kind"]
        last_high = self._last_pivot_price(seq, "H")
        last_low = self._last_pivot_price(seq, "L")

        if last_kind == "L":
            stage = "疑似C浪下行" if down_legs >= 2 else "疑似A浪下行"
            bias = "防守"
            summary = "指数处在调整结构中，强势策略应降低追高权重，优先等待止跌转折和市场温度修复。"
        else:
            stage = "疑似B浪反弹"
            bias = "中性防守"
            summary = "下跌后的反弹仍需观察能否放量越过前一主要高点，否则容易只是修复浪。"

        confidence = self._wave_confidence(seq, frame)
        point_seq = seq if len(seq) >= 3 else pivots[-6:]
        points = self._wave_points(point_seq, ["起点", "A", "B", "C", "反弹"])
        support = self._round_or_none(last_low)
        resistance = self._round_or_none(last_high)
        invalidation = f"放量突破上一主要高点 {resistance}，调整浪推演失效。" if resistance else "重新站回主要均线并突破前高则降级调整浪。"
        if support and latest_close < float(support):
            invalidation = f"跌破 {support} 后继续按调整浪处理，等待新的转折低点。"
        return {
            "stage": stage,
            "bias": bias,
            "confidence": confidence,
            "summary": summary,
            "support": support or "",
            "resistance": resistance or "",
            "invalidation": invalidation,
            "points": points,
            "method": "调整结构按 高-低-高-低 的有效转折推演，标签为规则化参考。",
        }

    def _range_wave_summary(self, frame: pd.DataFrame, pivots: list[dict[str, Any]], latest_close: float) -> dict[str, Any]:
        seq = pivots[-6:]
        last_high = self._last_pivot_price(seq, "H")
        last_low = self._last_pivot_price(seq, "L")
        support = self._round_or_none(last_low)
        resistance = self._round_or_none(last_high)
        return {
            "stage": "震荡浪/方向未确认",
            "bias": "中性",
            "confidence": "中",
            "summary": "均线和波段没有形成清晰单边结构，先按箱体震荡处理，等突破压力或跌破支撑再重新计浪。",
            "support": support or "",
            "resistance": resistance or "",
            "invalidation": f"突破 {resistance} 或跌破 {support} 后重新确认浪型。" if support and resistance else "等待有效突破后重新确认浪型。",
            "points": self._wave_points(seq, ["低", "高", "低", "高", "低", "高"]),
            "method": "震荡阶段不强行数浪，只标出最近主要转折和箱体边界。",
        }

    def _channel_bundle(self, frame: pd.DataFrame, pivots: list[dict[str, Any]], trend: str) -> tuple[dict[str, Any], dict[str, Any]]:
        major = self._trend_channel(frame, pivots, trend)
        short = self._short_regression_channel(frame, pivots)
        selected = short if short.get("available") else major
        return selected, {"short": short, "major": major}

    def _short_regression_channel(self, frame: pd.DataFrame, pivots: list[dict[str, Any]], window: int = 60) -> dict[str, Any]:
        """Short-term regression channel for the current visible swing.

        The major channel is based on ZigZag turning points. This short channel
        deliberately uses a more recent regression line so a strong rebound after
        the last low is not mislabeled as only a falling large-scale structure.
        """
        if len(frame) < 25:
            return self._empty_channel("短期样本不足，暂不画短期通道。")
        pivot_lows = [item for item in pivots if item["kind"] == "L"]
        start_idx = max(0, len(frame) - window)
        period = str(frame.attrs.get("index_period", "daily"))
        unit = "个交易日" if period == "daily" else f"根{self.INDEX_PERIODS.get(period, '')}K线"
        basis_prefix = f"最近{min(window, len(frame))}{unit}"
        if pivot_lows:
            candidate_start = int(pivot_lows[-1]["idx"])
            span = len(frame) - candidate_start
            if 22 <= span <= 90:
                start_idx = candidate_start
                basis_prefix = f"自最近主要低点 {pivot_lows[-1]['date']} 起"
        take = len(frame) - start_idx
        if take < 22:
            take = min(window, len(frame))
            start_idx = len(frame) - take
        end_idx = len(frame) - 1
        close = frame["close"].astype(float).tail(take).tolist()
        x_mean = (take - 1) / 2
        y_mean = sum(close) / take
        denom = sum((idx - x_mean) ** 2 for idx in range(take))
        if denom <= 0:
            return self._empty_channel("短期通道样本异常。")
        slope = sum((idx - x_mean) * (price - y_mean) for idx, price in enumerate(close)) / denom
        intercept = y_mean - slope * x_mean
        fitted = [slope * idx + intercept for idx in range(take)]
        residuals = [price - fitted[idx] for idx, price in enumerate(close)]
        lower_offset = min(residuals)
        upper_offset = max(residuals)
        latest_close = float(frame.iloc[-1]["close"])
        if upper_offset - lower_offset <= latest_close * 0.006:
            band = max(latest_close * 0.01, abs(slope) * take * 0.30)
            lower_offset = -band
            upper_offset = band
        global_intercept = intercept - slope * start_idx
        if slope > latest_close * 0.00012:
            channel_type = "短期上升通道"
        elif slope < -latest_close * 0.00012:
            channel_type = "短期下降通道"
        else:
            channel_type = "短期震荡通道"
        return self._channel_payload(
            frame,
            channel_type,
            start_idx,
            end_idx,
            slope,
            global_intercept + lower_offset,
            global_intercept + (lower_offset + upper_offset) / 2,
            global_intercept + upper_offset,
            latest_close,
            f"{basis_prefix}的收盘价线性回归；中轨为短期趋势线，上下轨包络主要偏离。",
        )

    def _trend_channel(self, frame: pd.DataFrame, pivots: list[dict[str, Any]], trend: str) -> dict[str, Any]:
        latest_idx = int(len(frame) - 1)
        latest_close = float(frame.iloc[-1]["close"])
        if latest_idx <= 0:
            return self._empty_channel("历史数据不足，暂不画趋势通道。")

        if trend == "上行":
            lows = [item for item in pivots if item["kind"] == "L"]
            highs = [item for item in pivots if item["kind"] == "H"]
            if len(lows) < 2 or not highs:
                return self._empty_channel("上升通道需要至少两个有效低点和一个高点。")
            p1, p2 = lows[-2], lows[-1]
            if p1["idx"] == p2["idx"]:
                return self._empty_channel("低点间距不足，暂不画上升通道。")
            slope = (float(p2["price"]) - float(p1["price"])) / (int(p2["idx"]) - int(p1["idx"]))
            if slope <= 0:
                falling = self._falling_channel_from_highs(frame, pivots, latest_idx, latest_close)
                if falling:
                    return falling
                lower_intercept = float(p1["price"]) - slope * int(p1["idx"])
                relevant_highs = [item for item in highs if int(item["idx"]) >= int(p1["idx"])]
                width = max(float(item["price"]) - (slope * int(item["idx"]) + lower_intercept) for item in relevant_highs)
                if width <= latest_close * 0.008:
                    return self._empty_channel("均线偏强，但主要低点仍在下移，通道宽度不足。")
                return self._channel_payload(
                    frame,
                    "下降整理通道",
                    int(p1["idx"]),
                    latest_idx,
                    slope,
                    lower_intercept,
                    lower_intercept + width / 2,
                    lower_intercept + width,
                    latest_close,
                    f"低点 {p1['date']} / {p2['date']} 仍在下移，上轨按主要高点平移形成下降整理通道。",
                )
            lower_intercept = float(p1["price"]) - slope * int(p1["idx"])
            relevant_highs = [item for item in highs if int(item["idx"]) >= int(p1["idx"])]
            width = max(float(item["price"]) - (slope * int(item["idx"]) + lower_intercept) for item in relevant_highs)
            if width <= latest_close * 0.008:
                return self._empty_channel("上升通道宽度过窄，暂不作为有效边界。")
            upper_intercept = lower_intercept + width
            mid_intercept = lower_intercept + width / 2
            return self._channel_payload(
                frame,
                "上升通道",
                int(p1["idx"]),
                latest_idx,
                slope,
                lower_intercept,
                mid_intercept,
                upper_intercept,
                latest_close,
                f"下轨取最近两个主要低点 {p1['date']} / {p2['date']}，上轨平移至最近主要高点。",
            )

        if trend == "下行":
            highs = [item for item in pivots if item["kind"] == "H"]
            lows = [item for item in pivots if item["kind"] == "L"]
            if len(highs) < 2 or not lows:
                return self._empty_channel("下降通道需要至少两个有效高点和一个低点。")
            p1, p2 = highs[-2], highs[-1]
            if p1["idx"] == p2["idx"]:
                return self._empty_channel("高点间距不足，暂不画下降通道。")
            slope = (float(p2["price"]) - float(p1["price"])) / (int(p2["idx"]) - int(p1["idx"]))
            if slope >= 0:
                return self._rising_channel_from_lows(frame, pivots, latest_idx, latest_close) or self._empty_channel("均线偏弱，但主要高点没有继续下移，暂不标记为下降通道。")
            upper_intercept = float(p1["price"]) - slope * int(p1["idx"])
            relevant_lows = [item for item in lows if int(item["idx"]) >= int(p1["idx"])]
            offset = min(float(item["price"]) - (slope * int(item["idx"]) + upper_intercept) for item in relevant_lows)
            if abs(offset) <= latest_close * 0.008:
                return self._empty_channel("下降通道宽度过窄，暂不作为有效边界。")
            lower_intercept = upper_intercept + offset
            mid_intercept = upper_intercept + offset / 2
            return self._channel_payload(
                frame,
                "下降通道",
                int(p1["idx"]),
                latest_idx,
                slope,
                lower_intercept,
                mid_intercept,
                upper_intercept,
                latest_close,
                f"上轨取最近两个主要高点 {p1['date']} / {p2['date']}，下轨平移至最近主要低点。",
            )

        seq = pivots[-6:]
        highs = [float(item["price"]) for item in seq if item["kind"] == "H"]
        lows = [float(item["price"]) for item in seq if item["kind"] == "L"]
        if not highs or not lows:
            return self._empty_channel("震荡阶段有效高低点不足，暂不画箱体。")
        start_idx = min(int(item["idx"]) for item in seq)
        lower = min(lows)
        upper = max(highs)
        if upper <= lower:
            return self._empty_channel("震荡箱体宽度不足。")
        return self._channel_payload(
            frame,
            "震荡箱体",
            start_idx,
            latest_idx,
            0.0,
            lower,
            (lower + upper) / 2,
            upper,
            latest_close,
            "震荡阶段不强行画趋势斜率，使用最近主要高低点构成箱体。",
        )

    def _rising_channel_from_lows(
        self,
        frame: pd.DataFrame,
        pivots: list[dict[str, Any]],
        latest_idx: int,
        latest_close: float,
    ) -> dict[str, Any] | None:
        lows = [item for item in pivots if item["kind"] == "L"]
        highs = [item for item in pivots if item["kind"] == "H"]
        if len(lows) < 2 or not highs:
            return None
        p1, p2 = lows[-2], lows[-1]
        if p1["idx"] == p2["idx"]:
            return None
        slope = (float(p2["price"]) - float(p1["price"])) / (int(p2["idx"]) - int(p1["idx"]))
        if slope <= 0:
            return None
        lower_intercept = float(p1["price"]) - slope * int(p1["idx"])
        relevant_highs = [item for item in highs if int(item["idx"]) >= int(p1["idx"])]
        if not relevant_highs:
            return None
        width = max(float(item["price"]) - (slope * int(item["idx"]) + lower_intercept) for item in relevant_highs)
        if width <= latest_close * 0.008:
            return None
        return self._channel_payload(
            frame,
            "上升通道",
            int(p1["idx"]),
            latest_idx,
            slope,
            lower_intercept,
            lower_intercept + width / 2,
            lower_intercept + width,
            latest_close,
            f"下轨取最近两个主要低点 {p1['date']} / {p2['date']}，上轨平移至最近主要高点。",
        )

    def _falling_channel_from_highs(
        self,
        frame: pd.DataFrame,
        pivots: list[dict[str, Any]],
        latest_idx: int,
        latest_close: float,
    ) -> dict[str, Any] | None:
        highs = [item for item in pivots if item["kind"] == "H"]
        lows = [item for item in pivots if item["kind"] == "L"]
        if len(highs) < 2 or not lows:
            return None
        p1, p2 = highs[-2], highs[-1]
        if p1["idx"] == p2["idx"]:
            return None
        slope = (float(p2["price"]) - float(p1["price"])) / (int(p2["idx"]) - int(p1["idx"]))
        if slope >= 0:
            return None
        upper_intercept = float(p1["price"]) - slope * int(p1["idx"])
        relevant_lows = [item for item in lows if int(item["idx"]) >= int(p1["idx"])]
        if not relevant_lows:
            return None
        offset = min(float(item["price"]) - (slope * int(item["idx"]) + upper_intercept) for item in relevant_lows)
        if abs(offset) <= latest_close * 0.008:
            return None
        return self._channel_payload(
            frame,
            "下降通道",
            int(p1["idx"]),
            latest_idx,
            slope,
            upper_intercept + offset,
            upper_intercept + offset / 2,
            upper_intercept,
            latest_close,
            f"上轨取最近两个主要高点 {p1['date']} / {p2['date']}，下轨平移至最近主要低点。",
        )

    def _channel_payload(
        self,
        frame: pd.DataFrame,
        channel_type: str,
        start_idx: int,
        end_idx: int,
        slope: float,
        lower_intercept: float,
        mid_intercept: float,
        upper_intercept: float,
        latest_close: float,
        basis: str,
    ) -> dict[str, Any]:
        lower_now = slope * end_idx + lower_intercept
        mid_now = slope * end_idx + mid_intercept
        upper_now = slope * end_idx + upper_intercept
        width = upper_now - lower_now
        position = (latest_close - lower_now) / width if width else 0.5
        position = max(0.0, min(1.0, position))
        zone = self._channel_zone(position)
        if "上升" in channel_type:
            invalidation = f"有效跌破下轨 {lower_now:.2f}，上升通道降级。"
            summary = f"当前处于{zone}，下轨看支撑，上轨看压力；通道仍以趋势边界为主，不作精确预测。"
        elif "下降" in channel_type:
            invalidation = f"有效突破上轨 {upper_now:.2f}，下降通道解除或进入震荡修复。"
            summary = f"当前处于{zone}，上轨是反弹压力，下轨是惯性下探边界。"
        else:
            invalidation = f"突破箱体上沿 {upper_now:.2f} 或跌破箱体下沿 {lower_now:.2f} 后重新定方向。"
            summary = f"当前处于{zone}，先按箱体处理，不强行给单边趋势。"
        return {
            "available": True,
            "type": channel_type,
            "basis": basis,
            "zone": zone,
            "position_pct": round(position * 100, 1),
            "support": round(lower_now, 2),
            "mid": round(mid_now, 2),
            "resistance": round(upper_now, 2),
            "summary": summary,
            "invalidation": invalidation,
            "lines": {
                "lower": self._channel_line(frame, start_idx, end_idx, slope, lower_intercept),
                "mid": self._channel_line(frame, start_idx, end_idx, slope, mid_intercept),
                "upper": self._channel_line(frame, start_idx, end_idx, slope, upper_intercept),
            },
        }

    def _channel_line(self, frame: pd.DataFrame, start_idx: int, end_idx: int, slope: float, intercept: float) -> list[dict[str, Any]]:
        max_idx = len(frame) - 1
        start = max(0, min(max_idx, int(start_idx)))
        end = max(0, min(max_idx, int(end_idx)))
        return [
            {
                "idx": start,
                "date": str(frame.iloc[start]["trade_date"]),
                "price": round(float(slope * start + intercept), 2),
            },
            {
                "idx": end,
                "date": str(frame.iloc[end]["trade_date"]),
                "price": round(float(slope * end + intercept), 2),
            },
        ]

    @staticmethod
    def _channel_zone(position: float) -> str:
        if position >= 0.82:
            return "通道上轨压力区"
        if position >= 0.58:
            return "通道上半区"
        if position >= 0.42:
            return "通道中轨附近"
        if position >= 0.18:
            return "通道下半区"
        return "通道下轨支撑区"

    @staticmethod
    def _empty_channel(summary: str) -> dict[str, Any]:
        return {
            "available": False,
            "type": "未形成",
            "basis": "",
            "zone": "",
            "position_pct": "",
            "support": "",
            "mid": "",
            "resistance": "",
            "summary": summary,
            "invalidation": "",
            "lines": {},
        }

    def _zigzag_pivots(self, frame: pd.DataFrame) -> list[dict[str, Any]]:
        closes = frame["close"].astype(float).tolist()
        dates = frame["trade_date"].astype(str).tolist()
        if len(closes) < 8:
            return []
        threshold = self._zigzag_threshold(frame)
        pivots: list[dict[str, Any]] = []
        last_pivot_price = closes[0]
        last_pivot_idx = 0
        extreme_price = closes[0]
        extreme_idx = 0
        direction = 0

        for idx, price in enumerate(closes[1:], start=1):
            if direction == 0:
                move = price / last_pivot_price - 1
                if abs(move) >= threshold:
                    direction = 1 if move > 0 else -1
                    start_kind = "L" if direction == 1 else "H"
                    pivots.append(self._pivot(dates, last_pivot_idx, last_pivot_price, start_kind))
                    extreme_price = price
                    extreme_idx = idx
                continue

            if direction == 1:
                if price >= extreme_price:
                    extreme_price = price
                    extreme_idx = idx
                elif price / extreme_price - 1 <= -threshold:
                    pivots.append(self._pivot(dates, extreme_idx, extreme_price, "H"))
                    direction = -1
                    extreme_price = price
                    extreme_idx = idx
            else:
                if price <= extreme_price:
                    extreme_price = price
                    extreme_idx = idx
                elif price / extreme_price - 1 >= threshold:
                    pivots.append(self._pivot(dates, extreme_idx, extreme_price, "L"))
                    direction = 1
                    extreme_price = price
                    extreme_idx = idx

        if direction != 0:
            end_kind = "H" if direction == 1 else "L"
            end = self._pivot(dates, extreme_idx, extreme_price, end_kind, provisional=extreme_idx == len(closes) - 1)
            if not pivots or pivots[-1]["idx"] != end["idx"]:
                pivots.append(end)
        if len(pivots) < 2:
            end_kind = "H" if closes[-1] >= closes[0] else "L"
            pivots = [
                self._pivot(dates, 0, closes[0], "L" if end_kind == "H" else "H"),
                self._pivot(dates, len(closes) - 1, closes[-1], end_kind, provisional=True),
            ]
        return pivots[-10:]

    @staticmethod
    def _pivot(dates: list[str], idx: int, price: float, kind: str, provisional: bool = False) -> dict[str, Any]:
        return {
            "idx": int(idx),
            "date": str(dates[idx]),
            "price": round(float(price), 2),
            "kind": kind,
            "provisional": bool(provisional),
        }

    def _zigzag_threshold(self, frame: pd.DataFrame) -> float:
        pct = pd.to_numeric(frame.get("pct_chg", pd.Series(dtype=float)), errors="coerce").abs().tail(40)
        median_move = float(pct.median() / 100) if len(pct.dropna()) else 0.01
        period = str(frame.attrs.get("index_period", "daily"))
        floors = {
            "daily": 0.018,
            "120min": 0.006,
            "60min": 0.005,
            "30min": 0.0035,
            "15min": 0.0025,
            "5min": 0.0015,
        }
        caps = {
            "daily": 0.045,
            "120min": 0.025,
            "60min": 0.020,
            "30min": 0.016,
            "15min": 0.012,
            "5min": 0.008,
        }
        multiplier = 3.0 if period == "daily" else 4.0
        return max(floors.get(period, 0.018), min(caps.get(period, 0.045), median_move * multiplier))

    def _last_major_pivot(self, pivots: list[dict[str, Any]], kind: str, prefer_lowest: bool) -> int | None:
        candidates = [(idx, item) for idx, item in enumerate(pivots[:-1]) if item["kind"] == kind]
        if not candidates:
            return None
        window = candidates[-5:]
        key = (lambda pair: pair[1]["price"]) if prefer_lowest else (lambda pair: -pair[1]["price"])
        return min(window, key=key)[0]

    @staticmethod
    def _last_pivot_price(seq: list[dict[str, Any]], kind: str) -> float | None:
        for item in reversed(seq):
            if item["kind"] == kind:
                return float(item["price"])
        return None

    def _wave_confidence(self, seq: list[dict[str, Any]], frame: pd.DataFrame) -> str:
        score = 0
        if len(seq) >= 4:
            score += 1
        if len(seq) >= 6:
            score += 1
        kinds = [item["kind"] for item in seq]
        if all(kinds[idx] != kinds[idx - 1] for idx in range(1, len(kinds))):
            score += 1
        trend = self._index_trend_label(frame)
        close = float(frame.iloc[-1]["close"])
        ma20 = self._num_or_none(frame.iloc[-1].get("ma20"))
        ma60 = self._num_or_none(frame.iloc[-1].get("ma60"))
        if trend == "上行" and ma20 is not None and ma60 is not None and close > ma20 > ma60:
            score += 1
        if trend == "下行" and ma20 is not None and ma60 is not None and close < ma20 < ma60:
            score += 1
        return "高" if score >= 4 else "中" if score >= 2 else "低"

    @staticmethod
    def _wave_points(seq: list[dict[str, Any]], labels: list[str]) -> list[dict[str, Any]]:
        points = []
        for idx, item in enumerate(seq[: len(labels)]):
            point = dict(item)
            point["label"] = labels[idx]
            points.append(point)
        return points

    def _index_trend_label(self, frame: pd.DataFrame) -> str:
        last = frame.iloc[-1]
        close = float(last["close"])
        ma20 = self._num_or_none(last.get("ma20"))
        ma60 = self._num_or_none(last.get("ma60"))
        if ma20 is None:
            return "震荡"
        if ma60 is None:
            return "上行" if close > ma20 else "下行" if close < ma20 else "震荡"
        if close > ma20 and ma20 >= ma60:
            return "上行"
        if close < ma20 and ma20 <= ma60:
            return "下行"
        return "震荡"

    @staticmethod
    def _num_or_none(value: Any) -> float | None:
        if value is None or pd.isna(value):
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _round_or_none(value: Any) -> float | None:
        if value is None or pd.isna(value):
            return None
        try:
            return round(float(value), 2)
        except (TypeError, ValueError):
            return None

    def _market_state(
        self,
        daily: pd.DataFrame,
        daily_basic: pd.DataFrame,
        limit_list: pd.DataFrame,
        trade_dates: list[str],
        market_flow: pd.DataFrame,
    ) -> dict[str, Any]:
        """判断整体市场状态。

        这个函数不是用来预测明天涨跌，而是判断当前环境是否适合做
        “主线概念 + 强势个股 + 突破/回踩”的策略。

        输出结果用于控制仓位和交易方式：
        - 主升扩散：可以积极观察主线和中军。
        - 强修复/可做：可以试错，但要等确认。
        - 震荡可做：轻仓做核心，不追后排。
        - 分歧/弱势：只做观察池，尾盘确认优先。
        - 退潮/冰点：防守，不做突破追高。
        """
        latest_date = trade_dates[-1]
        latest = daily[daily["trade_date"] == latest_date].copy()
        up_ratio = float((latest["pct_chg"] > 0).mean()) if len(latest) else 0
        strong_up_ratio = float((latest["pct_chg"] >= 3).mean()) if len(latest) else 0
        big_down_ratio = float((latest["pct_chg"] <= -5).mean()) if len(latest) else 0
        amount_yi = float(latest["amount"].sum() / 100000)
        limit_up = self._limit_count(limit_list, latest, "up")
        limit_down = self._limit_count(limit_list, latest, "down")
        broken_limit = self._limit_count(limit_list, latest, "broken")
        max_limit_times = self._max_limit_times(limit_list)
        amount_ratio = self._market_amount_ratio(daily, trade_dates)
        trend_score = self._index_trend_proxy(daily, trade_dates)
        flow_score, market_net_yi = self._market_flow_score(market_flow)

        # 赚钱效应：看上涨家数和强涨股票占比。
        # 如果上涨股票多、涨幅超过 3% 的股票也多，说明市场容易赚钱。
        profit_score = (
            self._clip(up_ratio, 0.35, 0.75) * 0.65
            + self._clip(strong_up_ratio, 0.05, 0.22) * 0.35
        )

        # 短线情绪：看涨停数量和连板高度。
        # A 股主线行情通常会通过涨停和连板扩散来体现资金热度。
        emotion_score = (
            self._clip(limit_up, 25, 120) * 0.70
            + self._clip(max_limit_times, 1, 6) * 0.30
        )

        # 亏钱效应：看跌停、大跌和炸板。
        # 即使有强板块，只要亏钱效应很重，突破失败率也会明显上升。
        risk_score = (
            self._clip(limit_down, 5, 60) * 0.55
            + self._clip(big_down_ratio, 0.02, 0.14) * 0.30
            + self._clip(broken_limit, 5, 55) * 0.15
        )

        # 流动性：看全市场成交额和成交额是否放大。
        # 没量的突破更容易冲高回落，有量的主线更容易持续。
        liquidity_score = (
            self._clip(amount_yi, 7000, 14000) * 0.55
            + self._clip(amount_ratio, 0.85, 1.35) * 0.45
        )

        # 市场综合温度。
        # 风险分是扣分项，因为亏钱效应会降低所有强势信号的质量。
        hot_score = (
            profit_score * 0.30
            + emotion_score * 0.25
            + liquidity_score * 0.18
            + trend_score * 0.17
            + flow_score * 0.10
            - risk_score * 0.25
        )
        if risk_score >= 68 and profit_score < 45:
            state = "退潮"
            position = "0%-15%"
            playbook = "防守为主，只观察新主线修复，不做追高突破。"
        elif hot_score >= 70 and profit_score >= 65 and emotion_score >= 55:
            state = "主升扩散"
            position = "50%-80%"
            playbook = "可积极观察主线突破和首次回踩，优先核心概念和中军。"
        elif hot_score >= 55 and risk_score < 55:
            state = "强修复/可做"
            position = "30%-60%"
            playbook = "允许试错，等待板块分歧后修复确认，少做后排。"
        elif hot_score >= 45:
            state = "震荡可做"
            position = "20%-50%"
            playbook = "轻仓做核心回踩，降低追突破权重。"
        elif hot_score >= 25:
            state = "分歧/弱势"
            position = "0%-20%"
            playbook = "只做观察池，不在早盘追，尾盘确认优先。"
        else:
            state = "冰点/空仓"
            position = "0%"
            playbook = "空仓等待，只记录抗跌方向和下一轮修复的首批领涨线索。"

        return {
            "state": state,
            "position": position,
            "score": round(max(0, min(100, hot_score)), 1),
            "profit_score": round(profit_score, 1),
            "emotion_score": round(emotion_score, 1),
            "risk_score": round(risk_score, 1),
            "liquidity_score": round(liquidity_score, 1),
            "trend_score": round(trend_score, 1),
            "flow_score": round(flow_score, 1),
            "up_ratio": round(up_ratio, 3),
            "strong_up_ratio": round(strong_up_ratio, 3),
            "big_down_ratio": round(big_down_ratio, 3),
            "limit_up": int(limit_up),
            "limit_down": int(limit_down),
            "broken_limit": int(broken_limit),
            "max_limit_times": int(max_limit_times),
            "amount_yi": round(amount_yi, 1),
            "amount_ratio": round(amount_ratio, 2),
            "market_net_yi": round(market_net_yi, 1),
            "playbook": playbook,
            "trade_date": latest_date,
            "review_mode": "盘中预复盘" if self.config.review_mode == "intraday" else "盘后复盘",
        }

    def _score_boards(
        self,
        basic: pd.DataFrame,
        daily: pd.DataFrame,
        daily_basic: pd.DataFrame,
        limit_list: pd.DataFrame,
        trade_dates: list[str],
    ) -> pd.DataFrame:
        """计算板块强度。

        默认优先使用东财一级概念：
        1. 先读取概念资金流排名靠前的板块。
        2. 再读取每个概念的成分股。
        3. 用成分股表现计算趋势、扩散、情绪、资金、持续性和风险。

        如果概念接口失败，回退到 stock_basic.industry 的行业聚合版本，
        保证系统至少还能给出一个可用的复盘结果。
        """
        if self.config.board_source in {"concept", "auto"}:
            try:
                boards = self._score_concept_boards(basic, daily, limit_list, trade_dates)
                if not boards.empty:
                    return boards
            except Exception as exc:
                self.log(f"一级概念板块读取失败，回退到行业聚合: {exc}")
        return self._score_industry_boards(basic, daily, limit_list, trade_dates)

    def _score_concept_boards(
        self,
        basic: pd.DataFrame,
        daily: pd.DataFrame,
        limit_list: pd.DataFrame,
        trade_dates: list[str],
    ) -> pd.DataFrame:
        """计算东财一级概念板块分数。

        板块分数拆成六部分：
        1. 趋势分：5/10/20 日涨幅和近端概念走势。
        2. 扩散分：上涨比例、强涨比例、站上 10 日线比例。
        3. 情绪分：涨停比例、连板高度、炸板情况。
        4. 资金分：成交额放大、成交额规模、主力净流入。
        5. 持续分：近 5 日排名稳定性、上涨天数、近 5 日表现。
        6. 风险扣分：过热、长上影、大跌比例。

        这样做是为了避免只看涨幅，把“一日游”和“少数龙头硬拉”
        从真正的主线板块里区分出来。
        """
        latest_date = trade_dates[-1]
        latest = daily[daily["trade_date"] == latest_date].copy()
        returns = self._returns_table(daily, trade_dates)
        latest = latest.merge(returns, on="ts_code", how="left")
        basic_cols = ["ts_code", "symbol", "name", "industry", "stock_type"]
        latest = latest.merge(basic[[col for col in basic_cols if col in basic.columns]], on="ts_code", how="left")
        latest["amount_yi"] = latest["amount"] / 100000

        flow = self._board_flow_window(trade_dates, "概念")
        if flow.empty:
            raise RuntimeError("概念资金流为空")
        latest_flow = flow[flow["trade_date"] == latest_date].copy()
        if latest_flow.empty:
            raise RuntimeError("当日概念资金流为空")
        latest_flow["pct_change"] = pd.to_numeric(latest_flow["pct_change"], errors="coerce").fillna(0)
        latest_flow["net_amount"] = pd.to_numeric(latest_flow["net_amount"], errors="coerce").fillna(0)
        latest_flow["rank"] = pd.to_numeric(latest_flow.get("rank", 9999), errors="coerce").fillna(9999)
        latest_flow = latest_flow.sort_values(["rank", "pct_change", "net_amount"], ascending=[True, False, False])
        latest_flow = latest_flow.head(self.config.board_scan_limit)

        limit_codes = self._limit_up_codes(limit_list, latest)
        broken_codes = self._broken_limit_codes(limit_list)
        rows = []
        for _, board_row in latest_flow.iterrows():
            board_code = str(board_row.get("ts_code", ""))
            board_name = str(board_row.get("name", ""))
            members = self._dc_members(latest_date, board_code)
            if members.empty or "con_code" not in members.columns:
                continue
            member_codes = set(members["con_code"].astype(str))
            group = latest[latest["ts_code"].isin(member_codes)].copy()
            if len(group) < self.config.min_member_count:
                continue

            metrics = self._board_member_metrics(group, daily, trade_dates, limit_codes, broken_codes)
            flow_metrics = self._board_flow_metrics(flow, board_code, trade_dates)
            overheated = bool(
                metrics["ret5"] > self.config.max_board_ret5
                or (metrics["ret10"] > self.config.max_board_ret10 and metrics["limit_ratio"] >= 0.08)
                or (metrics["amount_ratio"] > 2.2 and metrics["pct_change"] < 1)
            )
            # 趋势分：看成分股 5/10/20 日平均强度，以及概念资金流口径近 3 日表现。
            # 这对应“这个板块是不是持续走强”。
            trend_score = (
                self._clip(metrics["ret5"], -3, 10) * 0.35
                + self._clip(metrics["ret10"], -5, 18) * 0.35
                + self._clip(metrics["ret20"], -8, 28) * 0.15
                + self._clip(flow_metrics["flow_ret3"], -3, 8) * 0.15
            )

            # 扩散分：看上涨比例、强涨比例、站上 10 日线比例。
            # 这对应“板块是不是内部扩散，而不是一两只股票硬拉”。
            breadth_score = (
                self._clip(metrics["up_ratio"], 0.40, 0.85) * 0.45
                + self._clip(metrics["strong_ratio"], 0.04, 0.24) * 0.25
                + self._clip(metrics["above_ma10_ratio"], 0.35, 0.80) * 0.30
            )

            # 情绪分：看涨停参与度、连板高度和炸板比例。
            # 主线题材如果被短线资金认可，通常会有涨停和连板扩散。
            emotion_score = (
                self._clip(metrics["limit_ratio"], 0.00, 0.08) * 0.65
                + self._clip(metrics["max_limit_times"], 1, 5) * 0.20
                - self._clip(metrics["broken_ratio"], 0.02, 0.18) * 0.15
            )

            # 资金分：看成交额放大、成交额规模、主力净流入和净流入占比。
            # 只有涨幅没有资金，持续性通常会差。
            fund_score = (
                self._clip(metrics["amount_ratio"], 0.80, 1.80) * 0.35
                + self._clip(metrics["amount_yi"], 40, 700) * 0.20
                + self._clip(flow_metrics["net_amount_yi"], -5, 35) * 0.30
                + self._clip(flow_metrics["net_amount_rate"], -3, 6) * 0.15
            )

            # 持续分：看近 5 日是否持续靠前，避免把一日游当成主线。
            persistence_score = (
                self._clip(flow_metrics["rank_strength"], 0, 100) * 0.40
                + self._clip(flow_metrics["positive_days_5"], 1, 5) * 0.30
                + self._clip(flow_metrics["flow_ret5"], -5, 12) * 0.30
            )

            # 风险扣分：短期涨太多、长上影多、大跌股多都会扣分。
            # 这主要是防止追到板块高潮或分歧末端。
            risk_penalty = (
                self._clip(metrics["upper_shadow_ratio"], 0.08, 0.25) * 0.35
                + self._clip(metrics["big_down_ratio"], 0.03, 0.16) * 0.35
                + (30 if overheated else 0)
            )
            score = (
                trend_score * 0.24
                + breadth_score * 0.22
                + emotion_score * 0.17
                + fund_score * 0.19
                + persistence_score * 0.18
                - risk_penalty * 0.20
            )
            candidate = score >= self.config.min_board_score and metrics["up_ratio"] >= self.config.min_board_up_ratio and not overheated
            rows.append(
                {
                    "board": board_name,
                    "board_code": board_code,
                    "board_type": "一级概念",
                    "lifecycle": self._lifecycle_v2(metrics, flow_metrics, overheated),
                    "board_score": round(max(0, score), 2),
                    "trend_score": round(trend_score, 1),
                    "breadth_score": round(breadth_score, 1),
                    "emotion_score": round(emotion_score, 1),
                    "fund_score": round(fund_score, 1),
                    "persistence_score": round(persistence_score, 1),
                    "risk_penalty": round(risk_penalty, 1),
                    "ret5": round(metrics["ret5"], 2),
                    "ret10": round(metrics["ret10"], 2),
                    "ret20": round(metrics["ret20"], 2),
                    "pct_change": round(metrics["pct_change"], 2),
                    "up_ratio": round(metrics["up_ratio"], 3),
                    "strong_ratio": round(metrics["strong_ratio"], 3),
                    "above_ma10_ratio": round(metrics["above_ma10_ratio"], 3),
                    "limit_count": int(metrics["limit_count"]),
                    "broken_count": int(metrics["broken_count"]),
                    "amount_yi": round(metrics["amount_yi"], 1),
                    "amount_ratio": round(metrics["amount_ratio"], 2),
                    "net_amount_yi": round(flow_metrics["net_amount_yi"], 1),
                    "member_count": len(group),
                    "overheated": overheated,
                    "is_candidate": candidate,
                }
            )
        if not rows:
            raise RuntimeError("概念板块成分不足，无法评分")
        return pd.DataFrame(rows).sort_values("board_score", ascending=False)

    def _score_industry_boards(
        self,
        basic: pd.DataFrame,
        daily: pd.DataFrame,
        limit_list: pd.DataFrame,
        trade_dates: list[str],
    ) -> pd.DataFrame:
        """行业兜底版板块评分。

        当一级概念接口不可用时，使用 stock_basic.industry 分组。
        它不如概念板块贴近题材主线，但可以作为稳定备用方案。
        """
        latest_date = trade_dates[-1]
        latest = daily[daily["trade_date"] == latest_date].merge(basic[["ts_code", "industry"]], on="ts_code", how="left")
        latest["industry"] = latest["industry"].fillna("未分类")
        returns = self._returns_table(daily, trade_dates)
        latest = latest.merge(returns, on="ts_code", how="left")
        latest["amount_yi"] = latest["amount"] / 100000

        limit_codes = self._limit_up_codes(limit_list, latest)
        rows = []
        for industry, group in latest.groupby("industry"):
            if len(group) < 5:
                continue
            ret5 = group["ret5"].mean()
            ret10 = group["ret10"].mean()
            ret20 = group["ret20"].mean()
            up_ratio = float((group["pct_chg"] > 0).mean())
            limit_count = int(group["ts_code"].isin(limit_codes).sum())
            amount_yi = float(group["amount_yi"].sum())
            amount_ratio = self._board_amount_ratio(daily, basic, industry, trade_dates)
            overheated = bool(ret5 > 14 or (ret10 > 24 and limit_count >= max(5, len(group) * 0.08)))
            score = (
                self._clip(ret5, -3, 10) * 0.22
                + self._clip(ret10, -5, 18) * 0.22
                + self._clip(ret20, -8, 28) * 0.10
                + self._clip(up_ratio, 0.40, 0.85) * 0.18
                + self._clip(limit_count / max(len(group), 1), 0.00, 0.08) * 0.13
                + self._clip(amount_ratio, 0.80, 1.80) * 0.10
                + self._clip(amount_yi, 80, 900) * 0.05
            )
            candidate = score >= self.config.min_board_score and up_ratio >= self.config.min_board_up_ratio and not overheated
            rows.append(
                {
                    "board": industry,
                    "board_code": industry,
                    "board_type": "行业",
                    "lifecycle": self._lifecycle(ret5, ret10, up_ratio, limit_count, overheated),
                    "board_score": round(score, 2),
                    "trend_score": "",
                    "breadth_score": "",
                    "emotion_score": "",
                    "fund_score": "",
                    "persistence_score": "",
                    "risk_penalty": "",
                    "ret5": round(ret5, 2),
                    "ret10": round(ret10, 2),
                    "ret20": round(ret20, 2),
                    "pct_change": round(float(group["pct_chg"].mean()), 2),
                    "up_ratio": round(up_ratio, 3),
                    "strong_ratio": round(float((group["pct_chg"] >= 3).mean()), 3),
                    "above_ma10_ratio": "",
                    "limit_count": limit_count,
                    "broken_count": "",
                    "amount_yi": round(amount_yi, 1),
                    "amount_ratio": round(amount_ratio, 2),
                    "net_amount_yi": "",
                    "member_count": len(group),
                    "overheated": overheated,
                    "is_candidate": candidate,
                }
            )
        if not rows:
            self.log("没有个股达到最低分条件，可以调低最低个股分或扩大展示板块数。")
            return pd.DataFrame()
        return pd.DataFrame(rows).sort_values("board_score", ascending=False)

    def _board_flow_window(self, trade_dates: list[str], content_type: str) -> pd.DataFrame:
        """读取最近一段时间的东财板块资金流。

        content_type 通常是“概念”或“行业”。
        这里取最近 20 个交易日，是为了判断板块是不是持续靠前，
        而不是只在当天突然冲高。
        """
        frames = []
        for td in trade_dates[-20:]:
            try:
                frame = self._cached(
                    f"moneyflow_ind_dc_{content_type}_{td}",
                    lambda td=td: self.pro.moneyflow_ind_dc(trade_date=td, content_type=content_type),
                )
                if not frame.empty:
                    frames.append(frame)
            except Exception as exc:
                self.log(f"{td} {content_type}资金流读取失败: {exc}")
        if not frames:
            return pd.DataFrame()
        data = pd.concat(frames, ignore_index=True)
        for col in ["trade_date", "ts_code", "name"]:
            if col in data.columns:
                data[col] = data[col].astype(str)
        for col in ["pct_change", "close", "net_amount", "net_amount_rate", "rank"]:
            if col not in data.columns:
                data[col] = 0
            data[col] = pd.to_numeric(data[col], errors="coerce").fillna(0)
        return data

    def _dc_members(self, trade_date: str, board_code: str) -> pd.DataFrame:
        """读取一个东财概念/板块的成分股。

        dc_member 返回的是板块代码对应的成分股列表。
        后续板块强度和个股候选都基于这些成分股计算。
        """
        try:
            df = self._cached(
                f"dc_member_{trade_date}_{board_code}",
                lambda: self.pro.dc_member(trade_date=trade_date, ts_code=board_code),
            )
        except Exception as exc:
            self.log(f"{board_code} 成分股读取失败: {exc}")
            return pd.DataFrame()
        if "con_code" in df.columns:
            df["con_code"] = df["con_code"].astype(str)
        return df

    def _board_member_metrics(
        self,
        group: pd.DataFrame,
        daily: pd.DataFrame,
        trade_dates: list[str],
        limit_codes: set[str],
        broken_codes: set[str],
    ) -> dict[str, float]:
        """根据板块成分股计算板块内部指标。

        这里不直接相信板块指数，而是看成分股内部状态：
        - 平均 5/10/20 日涨幅：趋势强不强。
        - 上涨比例和强涨比例：扩散好不好。
        - 涨停/炸板比例：短线情绪是否健康。
        - 站上 10 日线比例：趋势结构是否成片。
        - 长上影和大跌比例：是否已经开始分歧或退潮。
        """
        ret5 = float(group["ret5"].mean())
        ret10 = float(group["ret10"].mean())
        ret20 = float(group["ret20"].mean())
        pct_change = float(group["pct_chg"].mean())
        up_ratio = float((group["pct_chg"] > 0).mean())
        strong_ratio = float((group["pct_chg"] >= 3).mean())
        big_down_ratio = float((group["pct_chg"] <= -5).mean())
        limit_count = int(group["ts_code"].isin(limit_codes).sum())
        broken_count = int(group["ts_code"].isin(broken_codes).sum())
        limit_ratio = limit_count / max(len(group), 1)
        broken_ratio = broken_count / max(len(group), 1)
        amount_yi = float(group["amount_yi"].sum())
        member_daily = daily[daily["ts_code"].isin(set(group["ts_code"]))]
        by_date = member_daily.groupby("trade_date")["amount"].sum().reindex(trade_dates).fillna(0)
        base = by_date.tail(20).mean()
        amount_ratio = float(by_date.tail(3).mean() / base) if base else 1.0
        above_ma10_ratio = self._above_ma_ratio(member_daily, group["ts_code"].tolist(), 10)
        upper_shadow_ratio = float(
            ((group["high"] - group[["open", "close"]].max(axis=1)) / group["close"].replace(0, pd.NA) > 0.04).fillna(False).mean()
        )
        max_limit_times = 1
        return {
            "ret5": ret5,
            "ret10": ret10,
            "ret20": ret20,
            "pct_change": pct_change,
            "up_ratio": up_ratio,
            "strong_ratio": strong_ratio,
            "big_down_ratio": big_down_ratio,
            "limit_count": limit_count,
            "broken_count": broken_count,
            "limit_ratio": limit_ratio,
            "broken_ratio": broken_ratio,
            "amount_yi": amount_yi,
            "amount_ratio": amount_ratio,
            "above_ma10_ratio": above_ma10_ratio,
            "upper_shadow_ratio": upper_shadow_ratio,
            "max_limit_times": max_limit_times,
        }

    def _board_flow_metrics(self, flow: pd.DataFrame, board_code: str, trade_dates: list[str]) -> dict[str, float]:
        """汇总单个板块的资金流趋势和排名稳定性。

        资金流不是唯一依据，但可以帮助识别：
        - 是否有主力资金持续进入。
        - 是否近几天排名稳定靠前。
        - 是否只是当天的一日游。
        """
        data = flow[flow["ts_code"] == board_code].sort_values("trade_date").copy()
        if data.empty:
            return {"flow_ret3": 0, "flow_ret5": 0, "net_amount_yi": 0, "net_amount_rate": 0, "rank_strength": 0, "positive_days_5": 0}
        latest = data.iloc[-1]
        pct = pd.to_numeric(data["pct_change"], errors="coerce").fillna(0)
        rank = pd.to_numeric(data.get("rank", 9999), errors="coerce").fillna(9999)
        return {
            "flow_ret3": float(pct.tail(3).sum()),
            "flow_ret5": float(pct.tail(5).sum()),
            "net_amount_yi": float(pd.to_numeric(pd.Series([latest.get("net_amount", 0)]), errors="coerce").fillna(0).iloc[0] / 100000000),
            "net_amount_rate": float(pd.to_numeric(pd.Series([latest.get("net_amount_rate", 0)]), errors="coerce").fillna(0).iloc[0]),
            "rank_strength": float(max(0, 100 - rank.tail(5).mean())),
            "positive_days_5": float((pct.tail(5) > 0).sum()),
        }

    def _above_ma_ratio(self, daily: pd.DataFrame, codes: list[str], window: int) -> float:
        """计算板块内站上某条均线的成分股比例。

        例如 window=10 时，表示有多少成分股收盘价站上 10 日线。
        这个指标用来衡量板块趋势是不是成片，而不是只有龙头强。
        """
        if daily.empty:
            return 0.0
        ok = 0
        total = 0
        for code in codes:
            hist = daily[daily["ts_code"] == code].sort_values("trade_date")
            if len(hist) < window:
                continue
            total += 1
            if float(hist.iloc[-1]["close"]) > float(hist["close"].tail(window).mean()):
                ok += 1
        return ok / total if total else 0.0

    def _score_stocks(
        self,
        basic: pd.DataFrame,
        daily: pd.DataFrame,
        daily_basic: pd.DataFrame,
        limit_list: pd.DataFrame,
        boards: pd.DataFrame,
    ) -> pd.DataFrame:
        """在候选板块中筛选个股。

        逻辑：
        1. 只在候选概念/行业里找股票。
        2. 如果是一级概念，就按 dc_member 的成分股筛。
        3. 对每个板块内部计算强度排名、成交额排名、市值排名。
        4. 这些排名用于识别情绪龙头、趋势龙头、中军趋势和后排风险。

        这样做比全市场直接选股更贴近你的策略：
        先有主线，再找主线里的核心票。
        """
        if boards.empty:
            return pd.DataFrame()
        candidate_boards = boards[boards["is_candidate"]].head(self.config.top_board_count)
        if candidate_boards.empty:
            candidate_boards = boards.head(max(3, self.config.top_board_count // 2))

        latest_date = daily["trade_date"].max()
        latest = daily[daily["trade_date"] == latest_date].copy()
        basic_cols = ["ts_code", "symbol", "name", "industry", "stock_type"]
        latest = latest.merge(basic[[col for col in basic_cols if col in basic.columns]], on="ts_code", how="left")
        latest = latest.merge(daily_basic, on="ts_code", how="left")
        returns = self._returns_table(daily, sorted(daily["trade_date"].unique().tolist()))
        latest = latest.merge(returns, on="ts_code", how="left")
        latest["amount_yi"] = latest["amount"] / 100000
        latest["market_cap_yi"] = pd.to_numeric(latest.get("total_mv", 0), errors="coerce").fillna(0) / 10000

        limit_codes = self._limit_up_codes(limit_list, latest)
        broken_codes = self._broken_limit_codes(limit_list)
        board_map = boards.set_index("board").to_dict("index")
        rows = []
        for _, board_row in candidate_boards.iterrows():
            board = str(board_row["board"])
            board_code = str(board_row.get("board_code", board))
            board_type = str(board_row.get("board_type", "行业"))
            if board_type == "一级概念":
                member_df = self._dc_members(latest_date, board_code)
                if member_df.empty or "con_code" not in member_df.columns:
                    continue
                member_codes = set(member_df["con_code"].astype(str))
                members = latest[latest["ts_code"].isin(member_codes)].copy()
            else:
                members = latest[latest["industry"] == board].copy()
            if members.empty:
                continue
            members["board"] = board
            members["board_code"] = board_code
            members["board_type"] = board_type
            members["ret10_rank_pct"] = members["ret10"].rank(ascending=False, pct=True)
            members["amount_rank_pct"] = members["amount_yi"].rank(ascending=False, pct=True)
            members["mv_rank_pct"] = members["market_cap_yi"].rank(ascending=False, pct=True)
            members["member_count"] = len(members)
            members = members.sort_values(["market_cap_yi", "amount_yi"], ascending=False)
            take_n = max(5, int(len(members) * self.config.top_stock_percent))
            for _, row in members.head(take_n).iterrows():
                try:
                    hist = daily[daily["ts_code"] == row["ts_code"]].sort_values("trade_date")
                    scored = self._score_one_stock(
                        row,
                        hist,
                        board_map.get(board, {}),
                        row["ts_code"] in limit_codes,
                        row["ts_code"] in broken_codes,
                    )
                    if scored["stock_score"] >= self.config.min_stock_score:
                        rows.append(scored)
                except Exception as exc:
                    self.log(f"跳过 {row.get('name', row.get('ts_code'))}: {exc}")
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows).sort_values("stock_score", ascending=False)

    def _score_one_stock(
        self,
        row: pd.Series,
        hist: pd.DataFrame,
        board: dict[str, Any],
        is_limit_up: bool,
        is_broken_limit: bool,
    ) -> dict[str, Any]:
        """给单只股票打分，并判断它在板块里的角色。

        主要判断：
        - 趋势结构：是否 5/10/20 日均线多头排列。
        - 买点结构：是否放量突破，或第一次回踩 10 日线企稳。
        - 相对强度：是否强于所属板块。
        - 板块内排名：涨幅、成交额、市值是否靠前。
        - 风险扣分：离均线过远、长上影、炸板、后排补涨。
        """
        close = hist["close"].astype(float)
        high = hist["high"].astype(float)
        low = hist["low"].astype(float)
        amount = hist["amount"].astype(float)
        if len(close) < 22:
            raise RuntimeError("历史数据不足")

        ma5 = close.tail(5).mean()
        ma10 = close.tail(10).mean()
        ma20 = close.tail(20).mean()
        last = hist.iloc[-1]
        prev_high20 = high.iloc[:-1].tail(20).max()
        distance_ma10 = (last["close"] / ma10 - 1) * 100 if ma10 else 999
        amount_ratio = amount.tail(3).mean() / amount.tail(20).mean() if amount.tail(20).mean() else 1
        close_position = (last["close"] - last["low"]) / max(last["high"] - last["low"], 0.01)
        upper_shadow = (last["high"] - max(last["open"], last["close"])) / last["close"] * 100
        trend_ok = bool(last["close"] > ma5 > ma10 > ma20)
        breakout = bool(last["close"] > prev_high20 and amount_ratio >= 1.15 and close_position >= 0.65)
        pullback = bool(last["low"] <= ma10 * 1.015 and last["close"] > ma10 and amount.iloc[-1] < amount.tail(10).mean() * 1.15)
        relative_board = float(row.get("ret10", 0)) - float(board.get("ret10", 0))
        ret_rank_pct = float(row.get("ret10_rank_pct", 1) or 1)
        amount_rank_pct = float(row.get("amount_rank_pct", 1) or 1)
        mv_rank_pct = float(row.get("mv_rank_pct", 1) or 1)
        # 板块内相对排名。数值越小越靠前，例如 0.10 表示板块前 10%。
        # 龙头/中军不能只看涨幅，还要看成交额和市值容量。
        core_rank_score = (
            self._clip(1 - ret_rank_pct, 0.50, 0.98) * 0.35
            + self._clip(1 - amount_rank_pct, 0.50, 0.98) * 0.35
            + self._clip(1 - mv_rank_pct, 0.50, 0.98) * 0.30
        )
        role = self._stock_role(row, is_limit_up, is_broken_limit, trend_ok, breakout, pullback, relative_board)

        signal = "突破" if breakout else "回踩" if pullback else "趋势观察"
        raw_score = (
            float(board.get("board_score", 0)) * 0.25
            + self._clip(float(row.get("ret5", 0)), -3, 12) * 0.15
            + self._clip(float(row.get("ret10", 0)), -5, 20) * 0.15
            + self._clip(float(row.get("ret20", 0)), -8, 35) * 0.08
            + self._clip(float(row.get("amount_yi", 0)), self.config.min_amount_yi, 25) * 0.10
            + self._clip(float(row.get("market_cap_yi", 0)), 50, 1500) * 0.08
            + self._clip(relative_board, -5, 8) * 0.06
            + core_rank_score * 0.08
            + (7 if trend_ok else 0)
            + (7 if breakout else 0)
            + (6 if pullback else 0)
            + (3 if is_limit_up else 0)
        )
        penalty = 0.0
        if distance_ma10 > self.config.max_distance_ma10:
            penalty += min(18, distance_ma10 - self.config.max_distance_ma10)
        if upper_shadow > 4:
            penalty += min(10, upper_shadow)
        if float(row.get("ret5", 0)) > 25:
            penalty += 10
        if is_broken_limit:
            penalty += 8
        if role == "后排风险":
            penalty += 8

        stop_hint = self._stop_hint(signal, ma10, prev_high20, low.tail(5).min())
        return {
            "board": row.get("board", row.get("industry", "")),
            "board_type": row.get("board_type", "行业"),
            "role": role,
            "code": row.get("symbol", ""),
            "name": row.get("name", ""),
            "stock_type": row.get("stock_type", ""),
            "stock_score": round(max(0, raw_score - penalty), 2),
            "signal": signal,
            "close": round(float(last["close"]), 2),
            "pct_chg": round(float(last["pct_chg"]), 2),
            "ret5": round(float(row.get("ret5", 0)), 2),
            "ret10": round(float(row.get("ret10", 0)), 2),
            "ret20": round(float(row.get("ret20", 0)), 2),
            "amount_yi": round(float(row.get("amount_yi", 0)), 2),
            "market_cap_yi": round(float(row.get("market_cap_yi", 0)), 2),
            "distance_ma10": round(distance_ma10, 2),
            "amount_ratio": round(amount_ratio, 2),
            "upper_shadow": round(upper_shadow, 2),
            "relative_board": round(relative_board, 2),
            "ret10_rank_pct": round(ret_rank_pct, 3),
            "amount_rank_pct": round(amount_rank_pct, 3),
            "mv_rank_pct": round(mv_rank_pct, 3),
            "is_limit_up": is_limit_up,
            "is_broken_limit": is_broken_limit,
            "stop_hint": stop_hint,
        }

    def _returns_table(self, daily: pd.DataFrame, trade_dates: list[str]) -> pd.DataFrame:
        """为每只股票计算 5/10/20 日涨幅。

        这些涨幅既用于板块聚合，也用于个股相对强弱判断。
        """
        pivot = daily.pivot(index="ts_code", columns="trade_date", values="close")
        latest = trade_dates[-1]
        rows = pd.DataFrame({"ts_code": pivot.index})
        for days in [5, 10, 20]:
            if len(trade_dates) > days:
                base = trade_dates[-days - 1]
                rows[f"ret{days}"] = ((pivot[latest] / pivot[base] - 1) * 100).values
            else:
                rows[f"ret{days}"] = 0
        return rows

    def _board_amount_ratio(self, daily: pd.DataFrame, basic: pd.DataFrame, industry: str, trade_dates: list[str]) -> float:
        """行业兜底版的量能比：近 3 日成交额 / 近 20 日成交额。"""
        codes = basic[basic["industry"] == industry]["ts_code"]
        data = daily[daily["ts_code"].isin(codes)]
        by_date = data.groupby("trade_date")["amount"].sum().reindex(trade_dates).fillna(0)
        base = by_date.tail(20).mean()
        return float(by_date.tail(3).mean() / base) if base else 1.0

    def _limit_up_codes(self, limit_list: pd.DataFrame, latest: pd.DataFrame) -> set[str]:
        """获取涨停股票代码。

        优先使用 Tushare limit_list_d。
        如果接口字段不完整，则退化为 pct_chg >= 9.8 的近似判断。
        """
        if not limit_list.empty and "ts_code" in limit_list.columns:
            if "limit_type" in limit_list.columns:
                mask = limit_list["limit_type"].astype(str).str.upper().str.contains("U")
                return set(limit_list[mask]["ts_code"].astype(str))
            if "limit" in limit_list.columns:
                mask = limit_list["limit"].astype(str).str.contains("涨|U|up", case=False, regex=True)
                return set(limit_list[mask]["ts_code"].astype(str))
        return set(latest[latest["pct_chg"] >= 9.8]["ts_code"].astype(str))

    def _broken_limit_codes(self, limit_list: pd.DataFrame) -> set[str]:
        """获取炸板股票代码。

        如果 limit_list_d 暴露了炸板标记，就用它识别。
        如果接口没有该字段，则返回空集合，不影响整体复盘。
        """
        if limit_list.empty or "ts_code" not in limit_list.columns:
            return set()
        if "limit" in limit_list.columns:
            mask = limit_list["limit"].astype(str).str.upper().str.contains("Z")
            return set(limit_list[mask]["ts_code"].astype(str))
        if "limit_type" in limit_list.columns:
            mask = limit_list["limit_type"].astype(str).str.upper().str.contains("Z")
            return set(limit_list[mask]["ts_code"].astype(str))
        return set()

    def _limit_count(self, limit_list: pd.DataFrame, latest: pd.DataFrame, direction: str) -> int:
        """统计涨停、跌停或炸板数量。"""
        if direction == "up":
            return len(self._limit_up_codes(limit_list, latest))
        if direction == "broken":
            if not limit_list.empty and "limit" in limit_list.columns:
                return int(limit_list["limit"].astype(str).str.upper().str.contains("Z").sum())
            if not limit_list.empty and "limit_type" in limit_list.columns:
                return int(limit_list["limit_type"].astype(str).str.upper().str.contains("Z").sum())
            return 0
        if not limit_list.empty and "ts_code" in limit_list.columns:
            if "limit_type" in limit_list.columns:
                mask = limit_list["limit_type"].astype(str).str.upper().str.contains("D")
                return int(mask.sum())
            if "limit" in limit_list.columns:
                mask = limit_list["limit"].astype(str).str.contains("跌|D|down", case=False, regex=True)
                return int(mask.sum())
        return int((latest["pct_chg"] <= -9.8).sum())

    def _max_limit_times(self, limit_list: pd.DataFrame) -> int:
        """从 limit_times 估算市场连板高度。"""
        if limit_list.empty or "limit_times" not in limit_list.columns:
            return 1
        values = pd.to_numeric(limit_list["limit_times"], errors="coerce").fillna(0)
        return int(values.max()) if len(values) else 1

    def _market_amount_ratio(self, daily: pd.DataFrame, trade_dates: list[str]) -> float:
        """全市场量能比：近 3 日成交额 / 近 20 日成交额。"""
        by_date = daily.groupby("trade_date")["amount"].sum().reindex(trade_dates).fillna(0)
        base = by_date.tail(20).mean()
        return float(by_date.tail(3).mean() / base) if base else 1.0

    def _index_trend_proxy(self, daily: pd.DataFrame, trade_dates: list[str]) -> float:
        """用全市场平均涨跌幅近似指数趋势。

        这里没有直接拉上证/创业板指数，而是用全市场股票平均表现做趋势代理。
        它的作用是判断市场背景是顺风、震荡还是逆风。
        """
        by_date = daily.groupby("trade_date")["pct_chg"].mean().reindex(trade_dates).fillna(0)
        if len(by_date) < 20:
            return 50.0
        short = by_date.tail(5).sum()
        mid = by_date.tail(10).sum()
        long = by_date.tail(20).sum()
        return (
            self._clip(short, -2, 4) * 0.45
            + self._clip(mid, -3, 7) * 0.35
            + self._clip(long, -5, 10) * 0.20
        )

    def _market_flow_score(self, market_flow: pd.DataFrame) -> tuple[float, float]:
        """把大盘净流入换算成 0-100 分，并返回净流入金额，单位亿元。"""
        if market_flow.empty or "net_amount" not in market_flow.columns:
            return 50.0, 0.0
        net = pd.to_numeric(market_flow["net_amount"], errors="coerce").fillna(0)
        net_yi = float(net.sum() / 100000000)
        return self._clip(net_yi, -500, 500), net_yi

    @staticmethod
    def _clip(value: float, low: float, high: float) -> float:
        """把不同量纲的指标线性压缩到 0-100 分。

        例子：
        _clip(上涨比例, 0.35, 0.75)
        表示上涨比例低于 35% 得 0 分，高于 75% 得 100 分，中间线性插值。
        """
        if value is None or math.isnan(float(value)) or high <= low:
            return 0.0
        return max(0.0, min(100.0, (float(value) - low) / (high - low) * 100))

    @staticmethod
    def _lifecycle(ret5: float, ret10: float, up_ratio: float, limit_count: int, overheated: bool) -> str:
        """行业兜底版生命周期判断。"""
        if overheated:
            return "高潮/过热"
        if ret5 > 4 and ret10 > 8 and up_ratio > 0.62:
            return "主升"
        if ret5 > 2 and up_ratio > 0.58:
            return "启动"
        if ret10 > 5 and up_ratio < 0.48:
            return "分歧"
        if ret5 < -2:
            return "退潮"
        return "观察"

    @staticmethod
    def _lifecycle_v2(metrics: dict[str, float], flow_metrics: dict[str, float], overheated: bool) -> str:
        """一级概念生命周期判断。

        生命周期用于区分：
        - 启动：刚开始走强，可加入观察池。
        - 主升：扩散好、趋势强，是你的策略最舒服的阶段。
        - 分歧：前面有强度，但当天扩散变差、炸板增加。
        - 修复：分歧后重新转强，适合尾盘确认。
        - 高潮/过热：短期涨太多，不适合新开仓追。
        - 退潮：走弱，要降低仓位或退出。
        - 二波观察：前主线缩量整理后可能再起。
        """
        if overheated:
            return "高潮/过热"
        if metrics["ret5"] < -2 or (metrics["up_ratio"] < 0.38 and metrics["big_down_ratio"] > 0.08):
            return "退潮"
        if metrics["ret10"] > 5 and metrics["up_ratio"] < 0.48 and metrics["broken_ratio"] > 0.04:
            return "分歧"
        if metrics["ret5"] > 3 and metrics["ret10"] > 7 and metrics["up_ratio"] > 0.62 and metrics["above_ma10_ratio"] > 0.58:
            return "主升"
        if metrics["ret5"] > 1.5 and metrics["up_ratio"] > 0.58 and flow_metrics["positive_days_5"] <= 2:
            return "启动"
        if metrics["up_ratio"] > 0.58 and metrics["ret5"] > 0 and flow_metrics["flow_ret3"] > 0:
            return "修复"
        if metrics["ret10"] > 6 and metrics["amount_ratio"] < 0.90 and metrics["above_ma10_ratio"] > 0.50:
            return "二波观察"
        return "观察"

    @staticmethod
    def _stock_type(row: pd.Series, is_limit_up: bool) -> str:
        """旧版个股类型标签。

        现在页面主要看 role 字段，这个函数保留只是为了兼容旧逻辑。
        """
        mv = float(row.get("market_cap_yi", 0) or 0)
        if mv >= 500:
            return "中军趋势"
        if is_limit_up:
            return "弹性进攻"
        return "趋势候选"

    @staticmethod
    def _stock_role(
        row: pd.Series,
        is_limit_up: bool,
        is_broken_limit: bool,
        trend_ok: bool,
        breakout: bool,
        pullback: bool,
        relative_board: float,
    ) -> str:
        """判断个股在板块里的角色。

        这里故意使用“板块内排名”，而不是全市场排名。
        因为你的策略是先找主线，再找主线里的核心。

        角色含义：
        - 情绪龙头：涨停、强度靠前、成交额靠前，代表短线情绪。
        - 趋势龙头：趋势多头、强度靠前、成交额靠前，代表持续趋势。
        - 中军趋势：市值和成交额靠前，走势稳定，最适合你的偏趋势交易。
        - 核心跟随：形态不错，但带动性弱于龙头/中军。
        - 低位补涨：可能有弹性，但容易是后排，不优先。
        - 后排风险：炸板或高位补涨，容易在分歧日先掉队。
        """
        ret_rank = float(row.get("ret10_rank_pct", 1) or 1)
        amount_rank = float(row.get("amount_rank_pct", 1) or 1)
        mv_rank = float(row.get("mv_rank_pct", 1) or 1)
        mv = float(row.get("market_cap_yi", 0) or 0)
        ret5 = float(row.get("ret5", 0) or 0)
        amount_yi = float(row.get("amount_yi", 0) or 0)
        if is_broken_limit or (ret5 > 22 and amount_rank > 0.45 and mv_rank > 0.45):
            return "后排风险"
        if is_limit_up and ret_rank <= 0.20 and amount_rank <= 0.35:
            return "情绪龙头"
        if trend_ok and ret_rank <= 0.25 and amount_rank <= 0.35 and relative_board > 0:
            return "趋势龙头"
        if mv_rank <= 0.30 and amount_rank <= 0.30 and trend_ok and mv >= 120:
            return "中军趋势"
        if (breakout or pullback) and ret_rank <= 0.55 and amount_rank <= 0.55:
            return "核心跟随"
        if ret5 > 12 and ret_rank > 0.55:
            return "低位补涨"
        return "趋势候选"

    @staticmethod
    def _stop_hint(signal: str, ma10: float, breakout_line: float, pullback_low: float) -> str:
        """生成结构化风控提示。

        突破票看突破位，回踩票看 10 日线和回踩低点。
        这只是复盘参考，实际交易还要结合板块是否退潮。
        """
        if signal == "突破":
            return f"收盘跌回突破位 {breakout_line:.2f} 下方"
        if signal == "回踩":
            return f"收盘跌破10日线 {ma10:.2f} 或回踩低点 {pullback_low:.2f}"
        return f"观察10日线 {ma10:.2f}，不追高"


