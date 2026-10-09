from __future__ import annotations

import math
import re
from typing import Any

import pandas as pd
import requests


class RealtimeMaStrategy:
    """Batch realtime MA checker.

    Users paste stock codes or names, then the strategy resolves them to A-share
    ts_codes, fetches rt_k snapshots, and evaluates MA5/MA10/MA20 pullbacks.
    """

    WINDOWS = [5, 10, 20, 30, 60]

    def __init__(self, engine: Any) -> None:
        self.engine = engine
        self.config = engine.config

    def run(self) -> dict[str, Any]:
        raw_input = str(getattr(self.config, "realtime_ma_input", "") or "").strip()
        if not raw_input:
            return {
                "ok": True,
                "rows": [],
                "unmatched": [],
                "count": 0,
                "matched_count": 0,
                "qualified_count": 0,
                "pullback_ma10_count": 0,
                "pullback_ma20_count": 0,
                "logs": self.engine.logs,
            }

        self.engine.log("读取股票基础信息，识别输入列表...")
        basic = self.engine._stock_basic()
        matches, unmatched = self._resolve_inputs(basic, raw_input)
        if not matches:
            self.engine.log("没有识别出有效股票。")
            return {
                "ok": True,
                "rows": [],
                "unmatched": unmatched,
                "count": 0,
                "matched_count": 0,
                "qualified_count": 0,
                "pullback_ma10_count": 0,
                "pullback_ma20_count": 0,
                "logs": self.engine.logs,
            }

        old_lookback = int(getattr(self.config, "lookback_days", 70) or 70)
        self.config.lookback_days = max(old_lookback, 80)
        try:
            trade_dates = self.engine._trade_dates()
        finally:
            self.config.lookback_days = old_lookback
        trade_date = trade_dates[-1]

        self.engine.log(f"读取最近 {len(trade_dates)} 个交易日的历史日线，用于计算均线...")
        daily = self.engine._daily_window(trade_dates)
        codes = [item["ts_code"] for item in matches]
        daily = daily[daily["ts_code"].isin(set(codes))].copy()

        self.engine.log(f"使用 rt_k 刷新 {len(codes)} 只股票实时日K...")
        realtime = self._rt_k(codes)
        rows = self._build_rows(matches, daily, realtime, trade_date)
        qualified_count = sum(1 for row in rows if row["qualified"] == "是")
        pullback_ma10_count = sum(1 for row in rows if row["pullback_ma10"] == "是")
        pullback_ma20_count = sum(1 for row in rows if row["pullback_ma20"] == "是")
        self.engine.log(
            f"实时MA检查完成：识别 {len(rows)} 只，"
            f"回踩MA5 {qualified_count} 只，回踩MA10 {pullback_ma10_count} 只，"
            f"回踩MA20条件 {pullback_ma20_count} 只。"
        )

        return {
            "ok": True,
            "trade_date": trade_date,
            "rows": rows,
            "unmatched": unmatched,
            "count": len(rows),
            "matched_count": len(rows),
            "qualified_count": qualified_count,
            "pullback_ma10_count": pullback_ma10_count,
            "pullback_ma20_count": pullback_ma20_count,
            "logs": self.engine.logs,
        }

    def _resolve_inputs(self, basic: pd.DataFrame, raw_input: str) -> tuple[list[dict[str, Any]], list[str]]:
        by_ts = {str(row["ts_code"]).upper(): row for _, row in basic.iterrows()}
        by_symbol = {str(row["symbol"]).zfill(6): row for _, row in basic.iterrows()}
        name_groups: dict[str, list[pd.Series]] = {}
        for _, row in basic.iterrows():
            name = str(row.get("name", "") or "").strip()
            if name:
                name_groups.setdefault(name, []).append(row)

        results: list[dict[str, Any]] = []
        unmatched: list[str] = []
        seen: set[str] = set()

        for token in self._tokens(raw_input):
            row, note = self._resolve_one(token, by_ts, by_symbol, name_groups, basic)
            if row is None:
                unmatched.append(f"{token}: {note}")
                continue
            ts_code = str(row.get("ts_code", "")).upper()
            if not ts_code or ts_code in seen:
                continue
            seen.add(ts_code)
            results.append(
                {
                    "input": token,
                    "ts_code": ts_code,
                    "code": str(row.get("symbol", "") or ts_code[:6]).zfill(6),
                    "name": str(row.get("name", "") or ""),
                    "industry": str(row.get("industry", "") or ""),
                }
            )
        return results, unmatched

    def _resolve_one(
        self,
        token: str,
        by_ts: dict[str, pd.Series],
        by_symbol: dict[str, pd.Series],
        name_groups: dict[str, list[pd.Series]],
        basic: pd.DataFrame,
    ) -> tuple[pd.Series | None, str]:
        code_match = re.search(r"(?<!\d)(\d{6})(?:\.(SH|SZ|BJ))?", token.upper())
        if code_match:
            symbol = code_match.group(1)
            suffix = code_match.group(2)
            if suffix:
                row = by_ts.get(f"{symbol}.{suffix}")
                if row is not None:
                    return row, "代码匹配"
            row = by_symbol.get(symbol)
            if row is not None:
                return row, "代码匹配"
            return None, "代码不存在"

        candidates = name_groups.get(token)
        if candidates:
            return candidates[0], "名称精确匹配"

        clean_name = re.sub(r"[A-Za-z0-9._()（）【】\[\]{}<>《》-]+", "", token).strip()
        if clean_name and clean_name != token:
            candidates = name_groups.get(clean_name)
            if candidates:
                return candidates[0], "名称精确匹配"

        name_query = clean_name or token
        if name_query:
            contains = basic[basic["name"].astype(str).str.contains(re.escape(name_query), na=False)]
            if len(contains) == 1:
                return contains.iloc[0], "名称片段匹配"
            if len(contains) > 1:
                names = "、".join(contains["name"].astype(str).head(5).tolist())
                return None, f"名称不唯一，匹配到 {len(contains)} 只：{names}"

        return None, "未匹配"

    @staticmethod
    def _tokens(raw_input: str) -> list[str]:
        parts = re.split(r"[\s,，;；、|/]+", raw_input.replace("\r", "\n"))
        result: list[str] = []
        seen: set[str] = set()
        for part in parts:
            token = part.strip()
            if not token or token in seen:
                continue
            seen.add(token)
            result.append(token)
        return result

    def _rt_k(self, codes: list[str]) -> pd.DataFrame:
        rt_data = self._normalize_rt_frame(self._rt_k_one_by_one(codes), "rt_k")
        valid_codes = set()
        if not rt_data.empty:
            valid_codes = set(rt_data.loc[rt_data["close"] > 0, "ts_code"].astype(str))

        missing_codes = [code for code in codes if code not in valid_codes]
        fallback = pd.DataFrame()
        if missing_codes:
            self.engine.log(f"rt_k 未返回 {len(missing_codes)} 只股票，使用东财实时行情兜底。")
            fallback = self._eastmoney_spot(missing_codes)

        frames = []
        if not rt_data.empty:
            frames.append(rt_data[rt_data["close"] > 0].copy())
        if not fallback.empty:
            frames.append(fallback)
        if not frames:
            return pd.DataFrame(columns=["ts_code", "pre_close", "open", "high", "low", "close", "vol", "amount", "data_source"])
        return pd.concat(frames, ignore_index=True).drop_duplicates("ts_code", keep="first")

    def _rt_k_one_by_one(self, codes: list[str]) -> pd.DataFrame:
        frames: list[pd.DataFrame] = []
        for code in codes:
            try:
                frame = self.engine.pro.rt_k(ts_code=code)
            except Exception as exc:
                self.engine.log(f"{code} rt_k 读取失败: {exc}")
                continue
            if frame is not None and not frame.empty:
                frames.append(frame)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def _eastmoney_spot(self, codes: list[str]) -> pd.DataFrame:
        frames: list[pd.DataFrame] = []
        for chunk in self._chunks(codes, 80):
            secids = ",".join(self._eastmoney_secid(code) for code in chunk)
            try:
                response = requests.get(
                    "https://push2.eastmoney.com/api/qt/ulist.np/get",
                    params={
                        "fltt": "2",
                        "invt": "2",
                        "fields": "f12,f14,f2,f17,f15,f16,f18,f5,f6",
                        "secids": secids,
                    },
                    timeout=10,
                )
                response.raise_for_status()
                items = (response.json().get("data") or {}).get("diff") or []
            except Exception as exc:
                self.engine.log(f"东财实时行情兜底失败: {exc}")
                continue
            rows = []
            for item in items:
                symbol = str(item.get("f12", "")).zfill(6)
                ts_code = self._ts_code_from_symbol(symbol)
                rows.append(
                    {
                        "ts_code": ts_code,
                        "name": item.get("f14", ""),
                        "close": item.get("f2", 0),
                        "open": item.get("f17", 0),
                        "high": item.get("f15", 0),
                        "low": item.get("f16", 0),
                        "pre_close": item.get("f18", 0),
                        "vol": item.get("f5", 0),
                        "amount": item.get("f6", 0),
                    }
                )
            if rows:
                frames.append(pd.DataFrame(rows))
        if not frames:
            return pd.DataFrame()
        return self._normalize_rt_frame(pd.concat(frames, ignore_index=True), "东财实时")

    @classmethod
    def _eastmoney_secid(cls, ts_code: str) -> str:
        symbol = str(ts_code).split(".")[0].zfill(6)
        market = "1" if str(ts_code).upper().endswith(".SH") or symbol.startswith(("5", "6", "9")) else "0"
        return f"{market}.{symbol}"

    @staticmethod
    def _ts_code_from_symbol(symbol: str) -> str:
        if symbol.startswith(("8", "4", "92")):
            suffix = "BJ"
        else:
            suffix = "SH" if symbol.startswith(("5", "6", "9")) else "SZ"
        return f"{symbol}.{suffix}"

    @staticmethod
    def _normalize_rt_frame(frame: pd.DataFrame, source: str) -> pd.DataFrame:
        if frame is None or frame.empty:
            return pd.DataFrame(columns=["ts_code", "pre_close", "open", "high", "low", "close", "vol", "amount", "data_source"])
        data = frame.copy()
        data["ts_code"] = data["ts_code"].astype(str).str.upper()
        for col in ["pre_close", "open", "high", "low", "close", "vol", "amount"]:
            if col not in data.columns:
                data[col] = 0
            data[col] = pd.to_numeric(data[col], errors="coerce").fillna(0)
        data["data_source"] = source
        return data[["ts_code", "pre_close", "open", "high", "low", "close", "vol", "amount", "data_source"]]

    @staticmethod
    def _chunks(items: list[str], size: int) -> list[list[str]]:
        return [items[idx : idx + size] for idx in range(0, len(items), size)]

    def _build_rows(
        self,
        matches: list[dict[str, Any]],
        daily: pd.DataFrame,
        realtime: pd.DataFrame,
        trade_date: str,
    ) -> list[dict[str, Any]]:
        rt_by_code = realtime.set_index("ts_code") if not realtime.empty else pd.DataFrame()
        hist_by_code = {code: frame.sort_values("trade_date") for code, frame in daily.groupby("ts_code")}
        rows: list[dict[str, Any]] = []

        for item in matches:
            ts_code = str(item["ts_code"]).upper()
            live = rt_by_code.loc[ts_code] if not rt_by_code.empty and ts_code in rt_by_code.index else pd.Series(dtype=object)
            hist = hist_by_code.get(ts_code, pd.DataFrame()).copy()
            if not hist.empty:
                hist["close"] = pd.to_numeric(hist["close"], errors="coerce")
                hist = hist[hist["trade_date"].astype(str) != trade_date]

            close = self._num(live.get("close"))
            open_price = self._num(live.get("open")) or close
            high = self._num(live.get("high")) or max(open_price, close)
            low = self._num(live.get("low")) or min(open_price, close)
            pre_close = self._num(live.get("pre_close")) or self._last_close(hist)
            data_source = str(live.get("data_source", "") or "")
            if close <= 0:
                hist_row = self._last_hist_row(hist)
                if hist_row is not None:
                    close = self._num(hist_row.get("close"))
                    open_price = self._num(hist_row.get("open")) or close
                    high = self._num(hist_row.get("high")) or max(open_price, close)
                    low = self._num(hist_row.get("low")) or min(open_price, close)
                    pre_close = self._num(hist_row.get("pre_close")) or pre_close
                    data_source = "历史日线"
                else:
                    data_source = "实时缺失"

            closes = hist["close"].dropna().astype(float).tolist()
            if close > 0:
                closes.append(close)
            ma_values = {f"ma{window}": self._ma(closes, window) for window in self.WINDOWS}
            ma5 = ma_values["ma5"]
            ma10 = ma_values["ma10"]
            ma20 = ma_values["ma20"]
            ma_stack_ok = self._ma_stack_ok(ma_values)
            price_condition_ok = bool(low > 0 and high > 0 and close > 0 and ma5 is not None and low <= ma5 <= close <= high)
            qualified = price_condition_ok and ma_stack_ok
            pullback_ma10 = self._pullback_ma10_ok(low, close, ma5, ma10)
            pullback_ma20 = self._pullback_ma20_ok(low, close, ma10, ma20)
            pct_chg = (close / pre_close - 1) * 100 if close > 0 and pre_close > 0 else 0.0

            rows.append(
                {
                    "code": item["code"],
                    "ts_code": ts_code,
                    "name": item["name"],
                    "industry": item.get("industry", ""),
                    "data_source": data_source or "实时缺失",
                    "latest_price": self._round(close),
                    "pct_chg": self._round(pct_chg),
                    "qualified": "是" if qualified else "否",
                    "pullback_ma10": "是" if pullback_ma10 else "否",
                    "pullback_ma20": "是" if pullback_ma20 else "否",
                    "ma_stack_ok": "是" if ma_stack_ok else "否",
                    "pre_close": self._round(pre_close),
                    "open": self._round(open_price),
                    "high": self._round(high),
                    "low": self._round(low),
                    "ma5": self._round(ma_values["ma5"]),
                    "ma10": self._round(ma_values["ma10"]),
                    "ma20": self._round(ma_values["ma20"]),
                    "ma30": self._round(ma_values["ma30"]),
                    "ma60": self._round(ma_values["ma60"]),
                    "below_ma5": self._below_ma(close, ma_values["ma5"]),
                    "below_ma10": self._below_ma(close, ma_values["ma10"]),
                    "below_ma20": self._below_ma(close, ma_values["ma20"]),
                    "below_ma30": self._below_ma(close, ma_values["ma30"]),
                    "below_ma60": self._below_ma(close, ma_values["ma60"]),
                }
            )
        return rows

    @staticmethod
    def _below_ma(close: float, ma_value: float | None) -> str:
        if close <= 0 or ma_value is None:
            return ""
        return "是" if close < ma_value else "否"

    @staticmethod
    def _pullback_ma10_ok(
        low: float,
        close: float,
        ma5: float | None,
        ma10: float | None,
    ) -> bool:
        return bool(
            low > 0
            and close > 0
            and ma5 is not None
            and ma10 is not None
            and low <= ma10 <= close
            and ma5 >= ma10
        )

    @staticmethod
    def _pullback_ma20_ok(
        low: float,
        close: float,
        ma10: float | None,
        ma20: float | None,
    ) -> bool:
        return bool(
            low > 0
            and close > 0
            and ma10 is not None
            and ma20 is not None
            and low <= ma20 <= close
            and ma10 >= ma20
        )

    @staticmethod
    def _ma_stack_ok(ma_values: dict[str, float | None]) -> bool:
        values = [ma_values.get(f"ma{window}") for window in RealtimeMaStrategy.WINDOWS]
        if any(value is None for value in values):
            return False
        return bool(values[0] > values[1] > values[2] > values[3] > values[4])

    @staticmethod
    def _ma(closes: list[float], window: int) -> float | None:
        if len(closes) < window:
            return None
        values = closes[-window:]
        return sum(values) / window

    @staticmethod
    def _last_close(hist: pd.DataFrame) -> float:
        if hist.empty or "close" not in hist.columns:
            return 0.0
        values = pd.to_numeric(hist["close"], errors="coerce").dropna()
        return float(values.iloc[-1]) if not values.empty else 0.0

    @staticmethod
    def _last_hist_row(hist: pd.DataFrame) -> pd.Series | None:
        if hist.empty:
            return None
        return hist.sort_values("trade_date").iloc[-1]

    @staticmethod
    def _num(value: Any) -> float:
        try:
            result = float(value)
        except (TypeError, ValueError):
            return 0.0
        if math.isnan(result) or math.isinf(result):
            return 0.0
        return result

    @staticmethod
    def _round(value: float | None) -> float | str:
        if value is None:
            return ""
        try:
            result = float(value)
        except (TypeError, ValueError):
            return ""
        if math.isnan(result) or math.isinf(result):
            return ""
        return round(result, 2)
